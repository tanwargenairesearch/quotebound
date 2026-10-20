#include "data.h"

#include <algorithm>
#include <fstream>
#include <random>
#include <stdexcept>

#include "json.h"
#include "provenance.h"

namespace {

constexpr int kBucket = 128;

void shuffle(std::vector<int>& v, std::mt19937_64& rng) {
  for (size_t i = v.size(); i > 1; --i) std::swap(v[i - 1], v[rng() % i]);
}

void shuffle(std::vector<std::vector<int>>& v, std::mt19937_64& rng) {
  for (size_t i = v.size(); i > 1; --i) std::swap(v[i - 1], v[rng() % i]);
}

// Consecutive runs of `order` whose (count x longest) fits the budget.
std::vector<std::vector<int>> pack(const std::vector<Example>& ex, const std::vector<int>& order, int batch_tokens) {
  std::vector<std::vector<int>> batches;
  size_t longest = 0;
  for (int i : order) {
    const size_t n = ex[i].ids.size(), l = std::max(longest, n);
    if (batches.empty() || (batches.back().size() + 1) * l > size_t(batch_tokens)) {
      batches.emplace_back();
      longest = n;
    } else {
      longest = l;
    }
    batches.back().push_back(i);
  }
  return batches;
}

}  // namespace

Dataset load_examples(const std::string& path, const Tokenizer& t, int max_len) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot read " + path);
  Dataset d;
  simdjson::dom::parser parser;
  std::string line;
  for (int n = 1; std::getline(f, line); ++n) {
    if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
    const std::string where = path + ":" + std::to_string(n);
    std::vector<Turn> turns;
    try {
      simdjson::dom::element row = parser.parse(line);
      for (simdjson::dom::element m : get<simdjson::dom::array>(row, "messages", where))
        turns.push_back({std::string(get<std::string_view>(m, "role", where + " message")),
                         std::string(get<std::string_view>(m, "content", where + " message"))});
      std::string id = "line " + std::to_string(n);
      if (has_key(row, "id")) id = std::string(get<std::string_view>(row, "id", where));
      Rendered r = render_chat(t, turns);
      // The last answer's first trained token; everything before it is that answer's prompt.
      size_t last = r.loss.size();
      while (last > 0 && !r.loss[last - 1]) --last;
      while (last > 0 && r.loss[last - 1]) --last;
      if (int(r.ids.size()) > max_len) {
        d.too_long.push_back(id);
      } else if (std::find(r.loss.begin() + 1, r.loss.end(), 1) == r.loss.end()) {
        d.no_answer.push_back(id);
      } else {
        const std::string& first = turns[0].content;
        d.system_prompts[turns[0].role == "system" ? sha256_hex(first.data(), first.size()) : "none"] += 1;
        d.examples.push_back({std::move(r.ids), std::move(r.loss), id, int(last)});
      }
    } catch (const simdjson::simdjson_error& e) {
      throw std::runtime_error(where + ": " + e.what());
    } catch (const std::runtime_error& e) {
      const std::string what = e.what();
      throw std::runtime_error(what.starts_with(path) ? what : where + ": " + what);
    }
  }
  return d;
}

std::string examples_sha256(const std::vector<Example>& ex) {
  std::string bytes;
  for (const Example& e : ex) {
    const uint32_t n = uint32_t(e.ids.size());
    bytes.append(reinterpret_cast<const char*>(&n), sizeof n);
    bytes.append(reinterpret_cast<const char*>(e.ids.data()), e.ids.size() * sizeof(int));
    bytes.append(reinterpret_cast<const char*>(e.loss.data()), e.loss.size());
  }
  return sha256_hex(bytes.data(), bytes.size());
}

std::vector<int> seeded_subset(size_t n, size_t max, uint64_t seed) {
  std::mt19937_64 rng(seed ^ 0xD1B54A32D192ED03ULL);
  std::vector<int> order(n);
  for (size_t i = 0; i < n; ++i) order[i] = int(i);
  shuffle(order, rng);
  order.resize(std::min(n, max));
  return order;
}

std::vector<std::vector<int>> plan_epoch(const std::vector<Example>& ex, int batch_tokens, uint64_t seed, int epoch) {
  std::mt19937_64 rng(seed * 0x9E3779B97F4A7C15ULL + uint64_t(epoch));
  std::vector<int> order(ex.size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = int(i);
  shuffle(order, rng);
  std::vector<std::vector<int>> batches;
  for (size_t at = 0; at < order.size(); at += kBucket) {
    std::vector<int> bucket(order.begin() + at, order.begin() + std::min(order.size(), at + kBucket));
    std::stable_sort(bucket.begin(), bucket.end(),
                     [&](int a, int b) { return ex[a].ids.size() < ex[b].ids.size(); });
    for (auto& b : pack(ex, bucket, batch_tokens)) batches.push_back(std::move(b));
  }
  shuffle(batches, rng);
  return batches;
}

std::vector<std::vector<int>> plan_eval(const std::vector<Example>& ex, int batch_tokens) {
  std::vector<int> order(ex.size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = int(i);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return ex[a].ids.size() < ex[b].ids.size(); });
  return pack(ex, order, batch_tokens);
}

Batch make_batch(const std::vector<Example>& ex, const std::vector<int>& rows) {
  size_t L = 0;
  for (int i : rows) L = std::max(L, ex[i].ids.size());
  std::vector<int32_t> tokens(rows.size() * L, 0), positions, targets;
  int real = 0;
  for (size_t r = 0; r < rows.size(); ++r) {
    const Example& e = ex[rows[r]];
    std::copy(e.ids.begin(), e.ids.end(), tokens.begin() + r * L);
    real += int(e.ids.size());
    for (size_t t = 0; t + 1 < e.ids.size(); ++t)
      if (e.loss[t + 1]) positions.push_back(int32_t(r * L + t)), targets.push_back(e.ids[t + 1]);
  }
  const int B = int(rows.size()), N = int(positions.size());
  return {mx::array(tokens.data(), {B, int(L)}, mx::int32), mx::array(positions.data(), {N}, mx::int32),
          mx::array(targets.data(), {N}, mx::int32), real};
}
