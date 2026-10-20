#include "batch.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <optional>
#include <stdexcept>

#include "files.h"
#include "generate.h"
#include "json.h"
#include "unicode.h"

std::vector<BatchRow> read_batch(std::istream& in, const std::string& where) {
  std::vector<BatchRow> rows;
  std::set<std::string> seen;
  simdjson::dom::parser parser;
  std::string line;
  for (int n = 1; std::getline(in, line); ++n) {
    if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
    const std::string at = where + ":" + std::to_string(n);
    BatchRow r;
    try {
      simdjson::dom::element row = parser.parse(line);
      r.id = std::string(get<std::string_view>(row, "id", at));
      for (simdjson::dom::element m : get<simdjson::dom::array>(row, "messages", at))
        r.turns.push_back({std::string(get<std::string_view>(m, "role", at + " message")),
                           std::string(get<std::string_view>(m, "content", at + " message"))});
      simdjson::dom::element schema;
      if (!row["schema"].get(schema)) r.schema = simdjson::minify(schema);
    } catch (const simdjson::simdjson_error& e) {
      throw std::runtime_error(at + ": " + e.what());
    }
    if (r.id.empty()) throw std::runtime_error(at + ": the id is empty");
    if (r.turns.empty() || r.turns.back().role != "user")
      throw std::runtime_error(at + ": the messages do not end in a user turn");
    if (!seen.insert(r.id).second) throw std::runtime_error(at + ": id " + json_quote(r.id) + " is repeated");
    rows.push_back(std::move(r));
  }
  return rows;
}

namespace {

std::string valid_utf8(std::string_view s) {
  std::vector<char32_t> cps;
  std::vector<uint32_t> offs;
  utf8_decode(s, cps, offs);
  std::string out;
  out.reserve(s.size());
  for (char32_t c : cps) utf8_append(out, c == 0xFFFFFFFF ? char32_t(0xFFFD) : c);
  return out;
}

}  // namespace

std::string overrides_json(const std::vector<Override>& o) {
  std::string spans, steps;
  for (const Override& x : o) {
    spans += (spans.empty() ? "[" : ",[") + std::to_string(x.start) + "," + std::to_string(x.end) + "]";
    steps += (steps.empty() ? "" : ",") + std::to_string(x.step);
  }
  return ",\"overridden\":[" + spans + "],\"overridden_steps\":[" + steps + "]";
}

std::string tokens_json(const std::vector<TokenRecord>& r) {
  std::string out = ",\"tokens\":[";
  auto num = [](float x) {
    if (!std::isfinite(x)) return std::string("null");
    char b[32];
    std::snprintf(b, sizeof b, "%.6g", double(x));
    return std::string(b);
  };
  for (size_t i = 0; i < r.size(); ++i)
    out += (i ? ",[" : "[") + std::to_string(r[i].start) + "," + std::to_string(r[i].end) + "," + num(r[i].logprob) +
           "," + num(r[i].margin) + "]";
  return out + "]";
}

std::string batch_line(const BatchResult& r) {
  char nums[256];
  std::snprintf(nums, sizeof nums,
                "\"prompt_tokens\":%zu,\"gen_tokens\":%zu,\"prefill_ms\":%.1f,\"decode_ms\":%.1f,\"total_ms\":%.1f,"
                "\"truncated\":%s}\n",
                r.prompt_tokens, r.gen_tokens, r.prefill_ms, r.decode_ms, r.total_ms, r.truncated ? "true" : "false");
  std::string tail = nums;
  if (r.mask_ms >= 0) {
    char more[96];
    std::snprintf(more, sizeof more, ",\"mask_ms\":%.1f,\"budget_cut\":%s", r.mask_ms, r.budget_cut ? "true" : "false");
    tail = tail.substr(0, tail.size() - 2) + more + overrides_json(r.overridden) + "}\n";
  }
  if (!r.tokens.empty()) tail = tail.substr(0, tail.size() - 2) + tokens_json(r.tokens) + "}\n";
  if (r.passes >= 0)
    tail = tail.substr(0, tail.size() - 2) + ",\"passes\":" + std::to_string(r.passes) + ",\"drafted\":" +
           std::to_string(r.drafted) + "}\n";
  return "{\"id\":" + json_quote(valid_utf8(r.id)) + ",\"text\":" + json_quote(valid_utf8(r.text)) + "," + tail;
}

std::set<std::string> resume_ids(const std::string& path, const std::vector<BatchRow>& rows) {
  const std::string bytes = read_file(path);
  const size_t whole = bytes.rfind('\n') == std::string::npos ? 0 : bytes.rfind('\n') + 1;
  if (whole < bytes.size()) {
    if (::truncate(path.c_str(), off_t(whole)) != 0) throw std::runtime_error("cannot cut the torn last line of " + path);
    sync_path(path);
  }
  std::set<std::string> in;
  for (const BatchRow& r : rows) in.insert(r.id);
  std::set<std::string> done;
  simdjson::dom::parser parser;
  size_t start = 0;
  for (int n = 1; start < whole; ++n) {
    const size_t end = bytes.find('\n', start);
    const std::string line = bytes.substr(start, end - start);
    start = end + 1;
    if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
    const std::string at = path + ":" + std::to_string(n);
    std::string id;
    try {
      id = std::string(get<std::string_view>(parser.parse(line), "id", at));
    } catch (const simdjson::simdjson_error& e) {
      throw std::runtime_error(at + ": " + e.what());
    }
    if (!in.count(id)) throw std::runtime_error(at + ": id " + json_quote(id) + " is not a row of this input");
    done.insert(id);
  }
  return done;
}

LineFile::LineFile(const std::string& path) : path_(path) {
  const bool existed = std::filesystem::exists(path);
  fd_ = ::open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
  if (fd_ < 0) throw std::runtime_error("cannot open " + path + " to append to it");
  if (!existed) {  // the new file's directory entry, durable before any line is
    const std::filesystem::path dir = std::filesystem::path(path).parent_path();
    sync_path(dir.empty() ? "." : dir.string());
  }
}

LineFile::~LineFile() {
  if (fd_ >= 0) ::close(fd_);
}

void LineFile::append(const std::string& line) {
  for (size_t at = 0; at < line.size();) {
    const ssize_t n = ::write(fd_, line.data() + at, line.size() - at);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) throw std::runtime_error("cannot write " + path_);
    at += size_t(n);
  }
  if (::fcntl(fd_, F_FULLFSYNC) != 0 && ::fsync(fd_) != 0) throw std::runtime_error("cannot sync " + path_);
}

std::vector<int> stop_tokens(const Model& m, const Tokenizer& t) {
  std::vector<int> stop = m.cfg.eos;
  stop.push_back(token_id(t, "<|im_end|>"));
  stop.push_back(token_id(t, "<|endoftext|>"));
  return stop;
}

BatchStats run_batch(const Model& m, const Tokenizer& t, const std::vector<BatchRow>& rows,
                     const std::vector<std::vector<int>>& prompts, const std::set<std::string>& done, int max_tokens,
                     const std::function<void(const BatchResult&)>& emit, const BatchConstraints* constraints,
                     bool confidence, const BatchDrafts* drafts) {
  using clock = std::chrono::steady_clock;
  const std::vector<int> stop = stop_tokens(m, t);
  BatchStats stats;
  for (size_t i = 0; i < rows.size(); ++i) {
    if (done.count(rows[i].id)) continue;
    const auto t0 = clock::now();
    const Schema* schema = nullptr;
    if (constraints) schema = rows[i].schema.empty() ? constraints->fallback : &constraints->by_text.at(rows[i].schema);
    std::optional<QuoteText> quotes;
    std::optional<Constraint> c;
    if (schema) {
      if (constraints->quote_user) quotes = quote_text(rows[i].turns.back().content);
      c.emplace(*schema, t, *constraints->trie, stop, quotes ? &*quotes : nullptr);
    }
    std::optional<Draft> draft;
    if (drafts && c) {
      const std::vector<Turn>& turns = rows[i].turns;
      const auto it = drafts->tables.find(turns[0].role == "system" ? turns[0].content : "");
      draft = Draft{Drafter{it == drafts->tables.end() ? nullptr : &it->second}, drafts->k};
    }
    Generation g;
    try {
      g = greedy(m, prompts[i], max_tokens, stop, {}, c ? &*c : nullptr, confidence, draft ? &*draft : nullptr);
    } catch (const std::exception& e) {
      throw std::runtime_error("row " + json_quote(rows[i].id) + ": " + e.what());
    }
    const bool stopped = !g.ids.empty() && std::find(stop.begin(), stop.end(), g.ids.back()) != stop.end();
    BatchResult r{rows[i].id, generation_text(t, g.ids, stopped), prompts[i].size(), g.ids.size(), g.prefill_ms,
                  g.decode_ms, 0, !stopped};
    if (c) r.mask_ms = g.mask_ms, r.budget_cut = c->budget_cut, r.overridden = g.overridden;
    if (confidence) r.tokens = token_records(t, g, stopped);
    if (draft) {
      r.passes = g.passes, r.drafted = g.drafted;
      stats.tokens += long(g.ids.size()), stats.passes += g.passes, stats.drafted += g.drafted;
    }
    r.total_ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    emit(r);
  }
  return stats;
}
