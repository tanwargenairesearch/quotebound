#include "draft.h"

#include <simdjson.h>

#include <algorithm>
#include <fstream>
#include <stdexcept>

#include "json.h"

namespace {

uint64_t context_hash(const int* ctx, int n) {
  uint64_t h = 1469598103934665603ull;
  for (int i = 0; i < n; ++i) h = (h ^ uint64_t(uint32_t(ctx[i]))) * 1099511628211ull;
  return h ^ uint64_t(n);
}

}  // namespace

void NgramTable::add(const std::vector<int>& seq) {
  for (size_t t = 1; t < seq.size(); ++t)
    for (int n = 1; n <= order_ && size_t(n) <= t; ++n) ++next_[n - 1][context_hash(seq.data() + t - n, n)][seq[t]];
}

void NgramTable::finish() {
  best_.assign(next_.size(), {});
  for (size_t n = 0; n < next_.size(); ++n)
    for (const auto& [h, counts] : next_[n]) {
      int tok = -1, c = 0;
      for (const auto& [id, k] : counts)
        if (k > c || (k == c && id < tok)) tok = id, c = k;
      best_[n][h] = tok;
    }
}

std::pair<int, int> NgramTable::next(const int* ctx, int n) const {
  for (int k = std::min(n, order_); k >= 1; --k) {
    const auto& m = best_[k - 1];
    if (auto it = m.find(context_hash(ctx + n - k, k)); it != m.end()) return {it->second, k};
  }
  return {-1, 0};
}

size_t NgramTable::contexts() const {
  size_t n = 0;
  for (const auto& m : best_) n += m.size();
  return n;
}

std::unordered_map<std::string, NgramTable> training_tables(const Tokenizer& t, const std::string& path, int order) {
  std::unordered_map<std::string, NgramTable> tables;
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot read " + path);
  simdjson::dom::parser parser;
  std::string line;
  for (int n = 1; std::getline(f, line); ++n) {
    if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
    const std::string at = path + ":" + std::to_string(n);
    std::vector<Turn> turns;
    try {
      for (simdjson::dom::element m : get<simdjson::dom::array>(parser.parse(line), "messages", at))
        turns.push_back({std::string(get<std::string_view>(m, "role", at)),
                         std::string(get<std::string_view>(m, "content", at))});
    } catch (const simdjson::simdjson_error& e) {
      throw std::runtime_error(at + ": " + e.what());
    }
    const std::string system = !turns.empty() && turns[0].role == "system" ? turns[0].content : "";
    const Rendered r = render_chat(t, turns);
    NgramTable& table = tables.try_emplace(system, order).first->second;
    for (size_t i = 0; i < r.ids.size();) {  // each assistant turn: a run of trained tokens, its <|im_end|> last
      if (!r.loss[i]) {
        ++i;
        continue;
      }
      size_t j = i;
      while (j < r.ids.size() && r.loss[j]) ++j;
      table.add(std::vector<int>(r.ids.begin() + std::ptrdiff_t(i), r.ids.begin() + std::ptrdiff_t(j)));
      i = j;
    }
  }
  for (auto& [s, table] : tables) table.finish();
  return tables;
}

std::pair<int, int> prompt_lookup(const std::vector<int>& ctx, int max_n, int min_n) {
  const int T = int(ctx.size());
  for (int n = std::min(max_n, T - 1); n >= min_n; --n) {
    const int* tail = ctx.data() + T - n;
    for (int s = T - n - 1; s >= 0; --s) {  // the latest start whose n tokens equal the tail, a token after them
      int k = 0;
      while (k < n && ctx[s + k] == tail[k]) ++k;
      if (k == n) return {ctx[s + n], n};
    }
  }
  return {-1, 0};
}

int Drafter::guess(const std::vector<int>& ctx, const std::vector<int>& allow) const {
  if (allow.size() == 1) return allow[0];
  auto ok = [&](int id) { return id >= 0 && std::binary_search(allow.begin(), allow.end(), id); };
  if (table)
    if (const int g = table->next(ctx.data(), int(ctx.size())).first; ok(g)) return g;
  if (const int p = prompt_lookup(ctx, lookup_max, 1).first; ok(p)) return p;
  return -1;
}
