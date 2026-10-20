// Schema-constrained decoding (constrain.h), with a checkpoint's tokenizer (not its weights) and the tiny Qwen3.5 of
// tiny.h widened to its vocabulary:
// - schemas with a keyword the decoder does not read are refused, naming it;
// - the example schema (testdata/example_schema.json, a booking request) and a follow-up questions reply's schema:
//   valid answers accepted byte by byte, and each kind of break refused (a key the schema has not, a value outside an
//   enum, a quote that is not the message's words standing alone), with JSON's escapes in quotes;
// - every token of valid answers, as the tokenizer splits them, is allowed at its step, and the masked argmax then
//   picks it when it is the logits' argmax: a valid unconstrained answer comes out unchanged;
// - random weights, under a small schema and under the example schema, always give an answer that parses, fits the
//   schema (checked here independently of the decoder), quotes only the message, and stops within the token cap, the
//   cap forcing the close;
// - the steps the decoder overrode (its token not the model's argmax) are exactly those a teacher-forced pass over the
//   answer finds, with the bytes each such token wrote;
// - the host's time per step, beside the decode step it overlaps;
// - token confidence: on, the same tokens and overrides as off, under a schema and without one; each token's recorded
//   log-probability and margin those of a teacher-forced pass (through the cache step by step to 1e-5; the whole
//   answer in one forward with no cache to kWholeTol: float32 and bfloat16 round the two paths differently), its
//   bytes tiling the text, an override's span the token's.
#include <simdjson.h>

#include <chrono>
#include <cmath>
#include <fstream>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <string>

#include "../src/batch.h"
#include "../src/constrain.h"
#include "../src/exact.h"
#include "../src/generate.h"
#include "../src/json.h"
#include "../src/schema.h"
#include "../src/unicode.h"
#include "check.h"
#include "tiny.h"

static Schema schema_of(const std::string& json) {
  simdjson::dom::parser p;
  return compile_schema(p.parse(json), "test schema");
}

static void refused_schema(const std::string& json, const std::string& want) {
  try {
    schema_of(json);
  } catch (const std::runtime_error& e) {
    CHECK(std::string(e.what()).find(want) != std::string::npos, "refusal \"%s\" does not say \"%s\"", e.what(),
          want.c_str());
    return;
  }
  CHECK(false, "a schema with %s compiled", want.c_str());
}

static std::vector<int> stops(const Tokenizer& tok) {
  return {token_id(tok, "<|im_end|>"), token_id(tok, "<|endoftext|>")};
}

// The answer through the constraint one byte at a time (each byte's own token): accepted throughout, and complete.
static bool accepts(const Schema& s, const Tokenizer& tok, const VocabTrie& trie, const std::string& text,
                    const std::string& json) {
  const QuoteText q = quote_text(text);
  Constraint c(s, tok, trie, stops(tok), &q);
  try {
    for (unsigned char b : json) c.accept(tok.byte_id[b]);
  } catch (const std::runtime_error&) {
    return false;
  }
  return c.complete();
}

// --- an independent reading of what the decoder must produce ---------------------------------------------------

static bool py_space(char32_t c) { return c == ' ' || (c >= 9 && c <= 13) || c == 0xA0 || c == 0x2028 || c == 0x3000; }
static bool word(char32_t c) { return c < 128 && (std::isalnum(int(c)) || c == '_'); }
static bool ascii_digit(char32_t c) { return c >= '0' && c <= '9'; }

// The decoder's rule for a quote (constrain.h, QuoteText) transcribed apart from it, with no byte limit: the quote,
// stripped, occurs in the text (ASCII letter case aside) somewhere not inside a word, and a figure not inside a wider
// one.
static bool stands(const std::string& text, const std::string& quote) {
  std::vector<char32_t> s, q;
  std::vector<uint32_t> o;
  utf8_decode(text, s, o);
  utf8_decode(quote, q, o);
  while (!q.empty() && py_space(q.front())) q.erase(q.begin());
  while (!q.empty() && py_space(q.back())) q.pop_back();
  if (q.empty()) return false;
  auto low = [](char32_t c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; };
  const size_t n = s.size();
  for (size_t i = 0; i + q.size() <= n; ++i) {
    bool same = true;
    for (size_t k = 0; k < q.size() && same; ++k) same = low(s[i + k]) == low(q[k]);
    if (!same) continue;
    const size_t j = i + q.size();
    const bool left = !(i > 0 && ((word(s[i - 1]) && word(s[i])) ||
                                  (ascii_digit(s[i]) && (s[i - 1] == ',' || s[i - 1] == '.' || s[i - 1] == U'£'))));
    const bool right = !(j < n && ((word(s[j]) && word(s[j - 1])) ||
                                   (ascii_digit(s[j - 1]) && (s[j] == '%' || ((s[j] == ',' || s[j] == '.') &&
                                                                              j + 1 < n && ascii_digit(s[j + 1]))))));
    if (left && right) return true;
  }
  return false;
}

static size_t chars(std::string_view s) {
  std::vector<char32_t> cps;
  std::vector<uint32_t> o;
  utf8_decode(s, cps, o);
  return cps.size();
}

// No white space outside strings: the compact form.
static bool compact(const std::string& json) {
  bool in = false, esc = false;
  for (char c : json) {
    if (in) {
      if (esc) esc = false;
      else if (c == '\\') esc = true;
      else if (c == '"') in = false;
    } else if (c == '"') {
      in = true;
    } else if (c == ' ' || c == '\n' || c == '\t' || c == '\r') {
      return false;
    }
  }
  return true;
}

// The small test schema, checked by hand: what is wrong with `doc`, or "".
static const std::string kSmall = R"({
  "type": "object", "additionalProperties": false, "required": ["kind", "items"],
  "properties": {
    "kind": {"enum": ["a", "b", "c"]},
    "note": {"type": "string", "maxLength": 12, "pattern": "^[^0-9]+$"},
    "items": {"type": "array", "minItems": 1, "maxItems": 3, "items": {"$ref": "#/$defs/item"}},
    "flag": {"type": "boolean"},
    "quote": {"type": "string", "minLength": 1, "maxLength": 40}
  },
  "$defs": {
    "item": {"type": "object", "additionalProperties": false, "required": ["n"], "properties": {
      "n": {"type": "integer", "minimum": 0},
      "x": {"type": "number"},
      "q": {"$ref": "#/$defs/quote"},
      "tags": {"type": "array", "items": {"enum": ["p", "q", "r"]}, "uniqueItems": true, "maxItems": 3}}},
    "quote": {"type": "string", "minLength": 1, "maxLength": 30}
  }
})";

static std::string small_problem(simdjson::dom::element doc, const std::string& raw, const std::string& text) {
  using namespace simdjson::dom;
  object o;
  if (doc.get(o)) return "not an object";
  if (!compact(raw)) return "not compact";
  std::set<std::string> keys;
  for (auto [k, v] : o) {
    if (!keys.insert(std::string(k)).second) return "a repeated key";
    std::string_view sv;
    if (k == "kind") {
      if (v.get(sv) || (sv != "a" && sv != "b" && sv != "c")) return "kind";
    } else if (k == "note") {
      if (v.get(sv) || chars(sv) > 12 || sv.find_first_of("0123456789") != std::string_view::npos) return "note";
    } else if (k == "flag") {
      bool b;
      if (v.get(b)) return "flag";
    } else if (k == "quote") {
      if (v.get(sv) || chars(sv) < 1 || chars(sv) > 40 || !stands(text, std::string(sv))) return "quote";
    } else if (k == "items") {
      array a;
      if (v.get(a) || a.size() < 1 || a.size() > 3) return "items";
      for (element it : a) {
        object io;
        if (it.get(io)) return "an item";
        std::set<std::string> ik;
        for (auto [kk, vv] : io) {
          if (!ik.insert(std::string(kk)).second) return "a repeated item key";
          if (kk == "n") {
            if (!vv.is_int64() && !vv.is_uint64()) return "n";
            if (vv.is_int64() && int64_t(vv) < 0) return "n below 0";
          } else if (kk == "x") {
            if (!vv.is_number()) return "x";
          } else if (kk == "q") {
            if (vv.get(sv) || chars(sv) < 1 || chars(sv) > 30 || !stands(text, std::string(sv))) return "q";
          } else if (kk == "tags") {
            array t;
            if (vv.get(t) || t.size() > 3) return "tags";
            std::set<std::string> seen;
            for (element tg : t)
              if (tg.get(sv) || (sv != "p" && sv != "q" && sv != "r") || !seen.insert(std::string(sv)).second)
                return "a tag";
          } else {
            return "item key " + std::string(kk);
          }
        }
        if (!ik.count("n")) return "an item without n";
      }
    } else {
      return "key " + std::string(k);
    }
  }
  if (!keys.count("kind") || !keys.count("items")) return "a required key missing";
  return "";
}

// Every quote of an example frame: at a quote key, the declined words, the venues.
static void example_quotes(simdjson::dom::element e, std::string_view key, std::vector<std::string>& out) {
  using namespace simdjson::dom;
  std::string_view sv;
  if (!e.get(sv)) {
    if (key == "quote" || key == "declined" || key == "venues") out.push_back(std::string(sv));
    return;
  }
  object o;
  if (!e.get(o)) {
    for (auto [k, v] : o) example_quotes(v, k, out);
    return;
  }
  array a;
  if (!e.get(a))
    for (element v : a) example_quotes(v, key, out);
}

// --- the tests ------------------------------------------------------------------------------------------------------

static void refusals() {
  refused_schema(R"({"type":"object","properties":{"a":{"type":"integer","maximum":3}},"additionalProperties":false})",
                 "keyword \"maximum\"");
  refused_schema(R"({"type":"object","properties":{"a":{"type":"string"}}})", "additionalProperties: false");
  refused_schema(R"({"type":"string","pattern":"^a+b$"})", "pattern");
  refused_schema(R"({"$ref":"other.json#/x"})", "$ref");
  refused_schema(R"({"type":["string","null"]})", "type must be one name");
  refused_schema(R"({"type":"object","properties":{},"additionalProperties":false,"patternProperties":{}})",
                 "keyword \"patternProperties\"");
  refused_schema(R"({"type":"number","minimum":1})", "minimum other than 0");
  refused_schema(R"({"type":"array","items":{"type":"object","properties":{},"additionalProperties":false},
                     "uniqueItems":true})",
                 "uniqueItems");
  std::printf("schema refusals: 8 schemas with a keyword or form the decoder does not read, each refused by name\n");
}

static const std::string kM1 =
    "Party of 12 for a birthday dinner on Friday, 3 hours from 7pm, set menu. Budget €45 a head with a 10% deposit. "
    "Outdoor table if dry. Can Café Lumen take us?";
static const std::string kF1 =
    R"({"intents":["book"],"venues":["Café Lumen"],"figures":{"guests":{"value":12,"quote":"12"},"hours":)"
    R"({"value":3,"quote":"3 hours"},"budget":{"value":45,"quote":"€45 a head"},"deposit_pct":{"value":10,)"
    R"("quote":"10%"}},"choices":{"party":{"value":"group","quote":"Party of 12"},"occasion":{"value":"birthday",)"
    R"("quote":"birthday"},"meal":{"value":"dinner","quote":"dinner"},"seating":{"value":"outdoor","quote":)"
    R"("Outdoor table"}}})";
static const std::string kM2 =
    "Group booking:\nI'm 38 (vegetarian, \"no nuts\"), my partner is 41 and eats anything. Two children aged 4 and 7. "
    "Book us into Harbour Hall and pay with my card please";
static const std::string kF2 =
    R"({"intents":["book"],"declined":"pay with my card","venues":["Harbour Hall"],"figures":{"children":{"value":2,)"
    R"("quote":"Two children"}},"choices":{"party":{"value":"group","quote":"Group booking"}},"attendees":[{"age":)"
    R"({"value":38,"quote":"38"},"diet":{"value":"vegetarian","quote":"vegetarian"},"allergy":{"value":"nuts",)"
    R"("quote":"\"no nuts\""}},{"age":{"value":41,"quote":"41"},"diet":{"value":"any","quote":"eats anything"}}],)"
    R"("child_ages":[{"value":4,"quote":"4"},{"value":7,"quote":"7"}]})";
static const std::string kM3 = "Is a 5% deposit enough to hold a table for 9 on Sunday? Lunch, about €31 each.";

// A follow-up questions reply: these targets, in this order, each question words alone (no digits, %, or currency).
static const std::string kFollowUp =
    R"({"type":"array","items":false,"minItems":2,"prefixItems":[)"
    R"({"type":"object","additionalProperties":false,"required":["target","question"],"properties":)"
    R"({"target":{"const":"attendees.0.diet"},"question":{"type":"string","minLength":8,)"
    R"("maxLength":160,"pattern":"^[^0-9%$€¥]+$"}}},)"
    R"({"type":"object","additionalProperties":false,"required":["target","question"],"properties":)"
    R"({"target":{"const":"date"},"question":{"type":"string","minLength":8,)"
    R"("maxLength":160,"pattern":"^[^0-9%$€¥]+$"}}}]})";

static std::string with(std::string s, const std::string& from, const std::string& to) {
  const size_t at = s.find(from);
  CHECK(at != std::string::npos, "fixture has no %s", from.c_str());
  return s.replace(at, from.size(), to);
}

static void example(const Tokenizer& tok, const VocabTrie& trie, const Schema& ex) {
  CHECK(accepts(ex, tok, trie, kM1, kF1), "a valid frame refused");
  CHECK(accepts(ex, tok, trie, kM2, kF2), "a valid frame with escaped quotes refused");
  struct Case {
    const char* what;
    std::string msg, json;
    bool ok;
  };
  const std::vector<Case> cases = {
      {"a quote in other letter case", kM1, with(kF1, "\"Outdoor table\"", "\"outdoor TABLE\""), true},
      {"a quote across a line break, as \\n", kM2, R"({"intents":["unknown"],"declined":"Group booking:\nI'm"})", true},
      {"a per-person budget", kM1, with(kF1, "\"€45 a head\"}", "\"€45 a head\",\"per\":\"person\"}"), true},
      {"a decimal and an exponent", kM1,
       with(with(kF1, "\"value\":45,", "\"value\":45.5,"), "\"value\":12,", "\"value\":1.2e1,"), true},
      {"general alone", kM1, R"({"intents":["general"]})", true},
      {"two intents", kM1, with(kF1, "[\"book\"]", "[\"book\",\"price\"]"), true},
      {"a catering block with one key", kM1,
       with(kF1, "\"venues\":", "\"catering\":{\"menu\":{\"value\":\"set\",\"quote\":\"set menu\"}},\"venues\":"),
       true},
      {"an attendee field the schema has not", kM2, with(kF2, "\"age\":{\"value\":41,", "\"height\":{\"value\":41,"),
       false},
      {"a figure key the schema has not", kM1, with(kF1, "\"guests\"", "\"people\""), false},
      {"a choice the schema has not", kM1, with(kF1, "\"seating\":", "\"outdoor\":"), false},
      {"a value outside the enum", kM1, with(kF1, "\"value\":\"outdoor\"", "\"value\":\"outside\""), false},
      {"a quote not in the message", kM1, with(kF1, "\"€45 a head\"", "\"€50 a head\""), false},
      {"a quote inside a word", kM1, with(kF1, "\"Outdoor table\"", "\"utdoor table\""), false},
      {"a figure inside a wider one", kM1, with(kF1, "\"10%\"", "\"10\""), false},
      {"a quote with a space at its end", kM1, with(kF1, "\"quote\":\"birthday\"", "\"quote\":\"birthday \""), false},
      {"a repeated key", kM1, with(kF1, "{\"intents\":[\"book\"],", "{\"intents\":[\"book\"],\"intents\":[\"book\"],"),
       false},
      {"a repeated venue", kM2, with(kF2, "[\"Harbour Hall\"]", "[\"Harbour Hall\",\"Harbour Hall\"]"), false},
      {"a repeated intent", kM1, with(kF1, "[\"book\"]", "[\"book\",\"book\"]"), false},
      {"general with another intent", kM1, with(kF1, "[\"book\"]", "[\"general\",\"book\"]"), false},
      {"no intents", kM1, R"({"venues":["Café Lumen"]})", false},
      {"white space", kM1, with(kF1, "{\"intents\":", "{\"intents\": "), false},
      {"a negative value", kM1, with(kF1, "\"value\":45,", "\"value\":-45,"), false},
      {"a leading zero", kM1, with(kF1, "\"value\":45,", "\"value\":045,"), false},
      {"a per outside its enum", kM1, with(kF1, "\"€45 a head\"}", "\"€45 a head\",\"per\":\"day\"}"), false},
      {"an empty catering block", kM1, with(kF1, "\"venues\":", "\"catering\":{},\"venues\":"), false},
      {"a byte after the frame", kM1, kF1 + " ", false},
      {"a non-canonical escape", kM2, with(kF2, "\"\\\"no nuts\\\"\"", "\"\\u0022no nuts\\u0022\""), false},
  };
  size_t ok = 0;
  for (const Case& k : cases) {
    CHECK(accepts(ex, tok, trie, k.msg, k.json) == k.ok, "%s: %s", k.what, k.ok ? "refused" : "accepted");
    ok += k.ok;
  }
  std::printf("example schema: 2 frames and %zu variants (%zu accepted, %zu refused), each as it should be (quotes as "
              "stretches of the message standing alone, letter case aside)\n",
              cases.size(), ok, cases.size() - ok);

  const Schema fs = schema_of(kFollowUp);
  const std::string good = R"([{"target":"attendees.0.diet","question":"What does the first guest eat?"},)"
                           R"({"target":"date","question":"Which day is the booking for?"}])";
  CHECK(accepts(fs, tok, trie, "", good), "a valid follow-up reply refused");
  const std::vector<std::pair<const char*, std::string>> bad = {
      {"the targets in another order", R"([{"target":"date","question":"Which day is the booking for?"},)"
                                        R"({"target":"attendees.0.diet","question":"What diet?..."}])"},
      {"a third question", good.substr(0, good.size() - 1) + R"(,{"target":"date","question":"Anything else here?"}])"},
      {"a figure in a question", with(good, "Which day is the booking for?", "Is the booking for the 5th?")},
      {"€ in a question", with(good, "Which day is the booking for?", "How many € is the booking?")},
      {"a question under 8 characters", with(good, "Which day is the booking for?", "Date?")},
      {"one question of two", good.substr(0, good.find(",{\"target\":\"date\"")) + "]"},
  };
  for (const auto& [what, json] : bad) CHECK(!accepts(fs, tok, trie, "", json), "follow-up: %s accepted", what);
  std::printf("follow-up questions: one valid reply accepted; %zu breaks refused (order, count, figures, currency, "
              "length)\n",
              bad.size());
}

// Each token of each answer, as encode() splits it, is allowed at its step; with that token the argmax of random
// logits, the masked argmax picks it; the stop is allowed after the last.
static void tokens_allowed(const Tokenizer& tok, const VocabTrie& trie, const Schema& ex) {
  const std::vector<std::pair<std::string, std::string>> answers = {{kM1, kF1}, {kM2, kF2}};
  int steps = 0;
  double ms = 0, worst = 0;
  size_t allowed = 0;
  mx::random::seed(3);
  for (const auto& [msg, json] : answers) {
    const QuoteText q = quote_text(msg);
    Constraint c(ex, tok, trie, stops(tok), &q);
    const std::vector<int> ids = encode(tok, json);
    for (size_t i = 0; i <= ids.size(); ++i) {
      const auto t0 = std::chrono::steady_clock::now();
      const std::vector<int> allow = c.allowed(512 - int(i));
      const double t = ms_since(t0);
      ms += t, worst = std::max(worst, t), ++steps, allowed += allow.size();
      const int want = i < ids.size() ? ids[i] : stops(tok)[0];
      CHECK(std::binary_search(allow.begin(), allow.end(), want), "token %zu (%s) of a valid answer not allowed", i,
            json_quote(i < ids.size() ? tok.id_bytes[want] : "<stop>").c_str());
      mx::array logits = mx::random::normal({int(tok.id_bytes.size())}) +
                         mx::where(mx::equal(mx::arange(int(tok.id_bytes.size())), mx::array(want)),
                                   mx::array(100.0f), mx::array(0.0f));
      CHECK(pick_allowed(logits, allow) == want, "the masked argmax did not pick the argmax token %d", want);
      if (i < ids.size()) c.accept(ids[i]);
    }
  }
  std::printf("valid answers token by token: all %d steps allow the answer's token and pick it when it is the "
              "argmax (stop included); allowed() %.3f ms a step on average, %.2f ms at most, %.1f ids allowed\n",
              steps, ms / steps, worst, double(allowed) / steps);
}

// The FreeTable against the walk it stands in for: two constraints in lockstep, one reading free strings' bodies from
// the table and one walking the trie at every step, along random paths (each step's token drawn from the allowed set,
// so strings run into multi-byte characters, escapes and maxLength, and caps from the shortest answer's to 200 tokens
// over it make the budget cut): the same allowed ids and the same budget_cut at every step. The follow-up reply's
// schema (its free strings carry a class) and a small schema with a classless free string and a unique array of
// strings (the close's hash).
static void free_tables(const Tokenizer& tok, const VocabTrie& trie) {
  const std::string free = R"({"type":"object","additionalProperties":false,"required":["note","tags"],"properties":)"
                           R"({"note":{"type":"string","maxLength":40},"tags":{"type":"array","uniqueItems":true,)"
                           R"("items":{"type":"string","minLength":2,"maxLength":12},"maxItems":4}}})";
  int steps = 0, table_steps = 0, cuts = 0, paths = 0;
  double ms_table = 0, ms_walk = 0;
  std::mt19937_64 rng(11);
  for (const std::string& json : {kFollowUp, free}) {
    const Schema sc = schema_of(json);
    const int least = Constraint(sc, tok, trie, stops(tok), nullptr).min_close() + 1;
    for (int over : {0, 3, 10, 24, 48, 80, 120, 200}) {
      const int cap = least + over;
      for (int rep = 0; rep < 2; ++rep, ++paths) {
        Constraint a(sc, tok, trie, stops(tok), nullptr), b(sc, tok, trie, stops(tok), nullptr);
        b.free_tables = false;
        for (int n = 0; n < cap; ++n) {
          auto t0 = std::chrono::steady_clock::now();
          const std::vector<int> x = a.allowed(cap - n);
          ms_table += ms_since(t0);
          t0 = std::chrono::steady_clock::now();
          const std::vector<int> y = b.allowed(cap - n);
          ms_walk += ms_since(t0);
          ++steps;
          CHECK(x == y, "path %d step %d: the table allows %zu ids, the walk %zu", paths, n, x.size(), y.size());
          CHECK(a.budget_cut == b.budget_cut, "path %d step %d: budget_cut differs", paths, n);
          CHECK(!x.empty(), "path %d step %d: nothing allowed", paths, n);
          // half the time any allowed id; else prefer one that is not a single-byte ASCII letter, to reach odd states
          int id = x[rng() % x.size()];
          if (rng() % 2) {
            for (int tries = 0; tries < 8; ++tries) {
              const int c = x[rng() % x.size()];
              if (size_t(c) < tok.id_bytes.size() && (tok.id_bytes[c].size() > 1 || uint8_t(tok.id_bytes[c][0]) >= 0x80)) {
                id = c;
                break;
              }
            }
          }
          const std::vector<int> stop_ids = stops(tok);
          const bool stop = std::find(stop_ids.begin(), stop_ids.end(), id) != stop_ids.end();
          a.accept(id), b.accept(id);
          if (stop) break;
        }
        cuts += a.budget_cut;
      }
    }
  }
  (void)table_steps;
  std::printf("free-string tables: %d random paths, %d steps, the table's allowed ids and budget cut equal the walk's "
              "at every step (%d paths cut for the budget); %.3f ms a step against the walk's %.3f\n",
              paths, steps, cuts, ms_table / steps, ms_walk / steps);
}

// Random weights under a schema: every answer parses, fits it, quotes the message, and stops within the cap.
// The overrides recomputed apart from the decoder: the answer's tokens teacher-forced through the cache one step at a
// time, each step's argmax over the whole vocabulary against the token taken, and the bytes each token wrote.
static std::vector<Override> overrides_of(const Model& m, const Tokenizer& tok, const std::vector<int>& prompt,
                                          const std::vector<int>& ids, const std::vector<int>& stop) {
  KVCache cache;
  mx::array logits = prefill(m, prompt, cache);
  std::vector<Override> out;
  int pos = 0;
  for (size_t n = 0; n < ids.size(); ++n) {
    const int top = int(mx::argmax(mx::reshape(logits, {-1}), 0).item<uint32_t>());
    const bool is_stop = std::find(stop.begin(), stop.end(), ids[n]) != stop.end();
    const int len = is_stop ? 0 : int(tok.id_bytes[ids[n]].size());
    if (top != ids[n]) out.push_back({int(n), pos, pos + len});
    pos += len;
    if (n + 1 < ids.size()) logits = forward(m, mx::array(&ids[n], {1, 1}, mx::int32), &cache, true);
  }
  return out;
}

// Constrained greedy written plainly: each step's token chosen over its whole allowed set from the logits read back,
// then one forward on it. greedy() under a constraint must equal it token for token, however it schedules its steps
// (the step it queues on the argmax before the constraint has chosen, dropped when the argmax is refused).
static std::vector<int> plain_constrained(const Model& m, const std::vector<int>& prompt, int cap,
                                          const std::vector<int>& stop, Constraint& c) {
  KVCache cache;
  mx::array logits = prefill(m, prompt, cache);
  std::vector<int> ids;
  for (int n = 0; n < cap; ++n) {
    const std::vector<int> allow = c.allowed(cap - n);
    const mx::array l = mx::reshape(logits, {-1});
    mx::eval(l);
    const float* p = l.data<float>();
    int best = allow[0];
    for (int id : allow)
      if (p[id] > p[best]) best = id;  // allow is sorted: the lowest id on a tie
    ids.push_back(best);
    c.accept(best);
    if (std::find(stop.begin(), stop.end(), best) != stop.end()) break;
    logits = forward(m, mx::array(&best, {1, 1}, mx::int32), &cache, true);
  }
  return ids;
}

static void random_weights(const Tokenizer& tok, const VocabTrie& trie, const Schema& ex) {
  Model m = tiny(mx::bfloat16, true, 3, int(tok.id_bytes.size()));
  const std::vector<int> stop = stops(tok);
  const Schema small = schema_of(kSmall);
  const std::vector<std::string> msgs = {kM1, kM2, kM3};
  simdjson::dom::parser parser;
  int answers = 0, cut = 0;
  size_t overrides = 0, empty_spans = 0;
  double mask_ms = 0, decode_ms = 0, free_ms = 0;
  size_t steps = 0, free_steps = 0;
  for (const std::string& msg : msgs) {
    const std::vector<int> prompt = chat_ids(tok, {{"system", "Reply with JSON."}, {"user", msg}});
    for (int cap : {200, 64, 40}) {
      for (const Schema* s : {&small, &ex}) {
        const QuoteText q = quote_text(msg);
        Constraint c(*s, tok, trie, stop, &q);
        if (c.min_close() + 1 > cap) continue;
        const Generation g = greedy(m, prompt, cap, stop, {}, &c);
        Constraint plain(*s, tok, trie, stop, &q);
        CHECK(g.ids == plain_constrained(m, prompt, cap, stop, plain), "cap %d: greedy() under the constraint is "
              "not the plain constrained greedy", cap);
        const bool stopped = std::find(stop.begin(), stop.end(), g.ids.back()) != stop.end();
        const std::string text = generation_text(tok, g.ids, stopped);
        CHECK(stopped && int(g.ids.size()) <= cap, "cap %d: no stop within the cap: %s", cap, text.c_str());
        simdjson::dom::element doc;
        CHECK(!parser.parse(text).get(doc), "cap %d: not JSON: %s", cap, text.c_str());
        if (s == &small) {
          const std::string p = small_problem(doc, text, msg);
          CHECK(p.empty(), "cap %d: %s: %s", cap, p.c_str(), text.c_str());
        } else {
          CHECK(compact(text), "not compact: %s", text.c_str());
          std::vector<std::string> quotes;
          example_quotes(doc, "", quotes);
          for (const std::string& qt : quotes) CHECK(stands(msg, qt), "quote %s does not stand", qt.c_str());
        }
        const std::vector<Override> want = overrides_of(m, tok, prompt, g.ids, stop);
        CHECK(g.overridden == want, "cap %d: %zu steps recorded as overridden, %zu by the teacher-forced pass", cap,
              g.overridden.size(), want.size());
        for (const Override& o : g.overridden) {
          CHECK(o.start <= o.end && size_t(o.end) <= text.size(), "an override's span is outside the answer");
          empty_spans += o.start == o.end;
        }
        overrides += g.overridden.size();
        ++answers, cut += c.budget_cut;
        mask_ms += g.mask_ms, decode_ms += g.decode_ms, steps += g.ids.size();
        if (cap == 200 && s == &small) {
          const Generation free = greedy(m, prompt, int(g.ids.size()), stop);
          free_ms += free.decode_ms, free_steps += free.ids.size();
        }
      }
    }
  }
  // A free string: most of a random model's tokens fit one, so most steps are the model's own and only some are not.
  const Schema free_text = schema_of(R"({"type":"string","maxLength":200})");
  size_t free_own = 0, free_all = 0;
  for (const std::string& msg : msgs) {
    const std::vector<int> prompt = chat_ids(tok, {{"system", "Reply with a JSON string."}, {"user", msg}});
    Constraint c(free_text, tok, trie, stop, nullptr);
    const Generation g = greedy(m, prompt, 64, stop, {}, &c);
    CHECK(g.overridden == overrides_of(m, tok, prompt, g.ids, stop), "free string: the overrides differ from the "
          "teacher-forced pass's");
    Constraint plain(free_text, tok, trie, stop, nullptr);
    CHECK(g.ids == plain_constrained(m, prompt, 64, stop, plain), "free string: greedy() is not the plain "
          "constrained greedy");
    free_own += g.ids.size() - g.overridden.size(), free_all += g.ids.size();
  }
  CHECK(free_own > 0 && free_own < free_all, "free string: %zu of %zu steps the model's own", free_own, free_all);
  std::printf("random weights: %d answers (3 messages, caps 200/64/40, the small schema and the example) all parse, "
              "fit the schema, quote only the message, stop within the cap and equal the plain constrained greedy; %d "
              "closed early for the cap. Overrides: %zu of %zu steps (%zu a stop for a model that wanted more), and "
              "under a free string %zu of %zu, each exactly a step a teacher-forced pass finds the argmax not taken, "
              "with the bytes it wrote. Per step: %.3f ms choosing tokens, %.2f ms decoding constrained, %.2f ms "
              "unconstrained\n",
              answers, cut, overrides, steps, empty_spans, free_all - free_own, free_all, mask_ms / steps,
              decode_ms / steps, free_ms / free_steps);
}

// Exact speculative decoding (generate.h's Draft) on random weights: under drafters that are mostly right (the plain
// answer's own n-grams), mostly wrong (another message's answer's) and prompt lookup alone, at several k, every answer
// is the plain constrained decoder's token for token, its overrides and budget cut included.
static void speculative_drafts(const Tokenizer& tok, const VocabTrie& trie, const Schema& ex) {
  const std::vector<int> stop = stops(tok);
  const Schema small = schema_of(kSmall);
  const std::vector<std::string> msgs = {kM1, kM2, kM3};
  long answers = 0, cuts = 0;
  long tokens[3] = {}, passes[3] = {}, drafted[3] = {};
  for (mx::Dtype dt : {mx::float32, mx::bfloat16}) {
    Model m = tiny(dt, true, 3, int(tok.id_bytes.size()));
    std::vector<std::vector<int>> prompts;
    for (const std::string& msg : msgs) prompts.push_back(chat_ids(tok, {{"system", "Reply with JSON."}, {"user", msg}}));
    for (int cap : {200, 40})
      for (const Schema* s : {&small, &ex}) {
        std::vector<Generation> plain;
        std::vector<bool> cut;
        for (size_t i = 0; i < msgs.size(); ++i) {
          const QuoteText q = quote_text(msgs[i]);
          Constraint c(*s, tok, trie, stop, &q);
          if (c.min_close() + 1 > cap) break;
          plain.push_back(greedy(m, prompts[i], cap, stop, {}, &c));
          cut.push_back(c.budget_cut);
        }
        if (plain.size() < msgs.size()) continue;
        for (size_t i = 0; i < msgs.size(); ++i) {
          NgramTable own(4), other(4);
          own.add(plain[i].ids), own.finish();
          other.add(plain[(i + 1) % msgs.size()].ids), other.finish();
          const NgramTable* tables[3] = {&own, &other, nullptr};
          for (int d = 0; d < 3; ++d)
            for (int k : {1, 3, 8, kMaxRows - 1}) {
              const QuoteText q = quote_text(msgs[i]);
              Constraint c(*s, tok, trie, stop, &q);
              const Draft draft{Drafter{tables[d]}, k};
              const Generation g = greedy(m, prompts[i], cap, stop, {}, &c, false, &draft);
              CHECK(g.ids == plain[i].ids, "drafter %d, k %d, cap %d: the answer is not the plain decoder's", d, k,
                    cap);
              CHECK(g.overridden == plain[i].overridden && c.budget_cut == cut[i], "drafter %d, k %d, cap %d: the "
                    "overrides or the budget cut differ from the plain decoder's", d, k, cap);
              CHECK(g.passes + 1 >= int(g.ids.size()) - g.drafted && g.drafted >= 0, "passes %d, drafted %d, %zu "
                    "tokens", g.passes, g.drafted, g.ids.size());
              ++answers, cuts += cut[i];
              if (k == 8) tokens[d] += long(g.ids.size()), passes[d] += g.passes, drafted[d] += g.drafted;
            }
        }
      }
  }
  std::printf("speculative drafts: %ld answers (float32 and bfloat16, 3 drafters, k 1/3/8/%d, %ld cut for the "
              "budget) equal the plain constrained decoder's, overrides and budget cut included. At "
              "k 8, tokens a pass: own n-grams %.2f (%ld drafts), another answer's %.2f (%ld), prompt lookup %.2f (%ld)\n",
              answers, kMaxRows - 1, cuts, double(tokens[0]) / double(passes[0] + 1), drafted[0],
              double(tokens[1]) / double(passes[1] + 1), drafted[1], double(tokens[2]) / double(passes[2] + 1),
              drafted[2]);
}

// The largest difference allowed between the recorded confidence and a whole-answer teacher-forced forward, by dtype: the
// cached decode and the one-pass forward run different kernels (single-query attention, the DeltaNet step against its
// chunked sequence), which round differently, most in bfloat16. Measured worst on these models: see the printed line.
static float whole_tol(mx::Dtype dt) { return dt == mx::float32 ? 1e-3f : 0.15f; }

static void confidence(const Tokenizer& tok, const VocabTrie& trie, const Schema& ex) {
  const std::vector<int> stop = stops(tok);
  const Schema small = schema_of(kSmall);
  const std::vector<std::string> msgs = {kM1, kM2};
  for (mx::Dtype dt : {mx::float32, mx::bfloat16}) {
    Model m = tiny(dt, true, 3, int(tok.id_bytes.size()));
    double worst_cached = 0, worst_whole_lp = 0, worst_whole_mg = 0;
    size_t tokens = 0, answers = 0, overrides = 0;
    for (const std::string& msg : msgs) {
      const std::vector<int> prompt = chat_ids(tok, {{"system", "Reply with JSON."}, {"user", msg}});
      for (const Schema* s : {static_cast<const Schema*>(nullptr), &small, &ex}) {
        const QuoteText q = quote_text(msg);
        std::optional<Constraint> c0, c1;
        if (s) c0.emplace(*s, tok, trie, stop, &q), c1.emplace(*s, tok, trie, stop, &q);
        const int cap = 96;
        const Generation off = greedy(m, prompt, cap, stop, {}, c0 ? &*c0 : nullptr);
        const Generation on = greedy(m, prompt, cap, stop, {}, c1 ? &*c1 : nullptr, true);
        CHECK(on.ids == off.ids && on.overridden == off.overridden, "%s: confidence on changed the tokens taken",
              s ? "constrained" : "free");
        CHECK(off.confidence.empty() && on.confidence.size() == on.ids.size(), "%zu confidences for %zu tokens",
              on.confidence.size(), on.ids.size());
        const auto cached = teacher_forced(m, prompt, on.ids, false), whole = teacher_forced(m, prompt, on.ids, true);
        for (size_t k = 0; k < on.ids.size(); ++k) {
          const TokenConfidence& c = on.confidence[k];
          CHECK(c.logprob <= 1e-5f && c.margin >= 0, "token %zu: logprob %g, margin %g", k, c.logprob, c.margin);
          const double dc = std::max(std::abs(c.logprob - cached[k].first), std::abs(c.margin - cached[k].second));
          CHECK(dc <= 1e-5, "token %zu: recorded (%g, %g), cached teacher-forced (%g, %g)", k, c.logprob, c.margin,
                cached[k].first, cached[k].second);
          const double dl = std::abs(c.logprob - whole[k].first), dm = std::abs(c.margin - whole[k].second);
          CHECK(dl <= whole_tol(dt) && dm <= whole_tol(dt), "token %zu: recorded (%g, %g), whole teacher-forced "
                "(%g, %g)", k, c.logprob, c.margin, whole[k].first, whole[k].second);
          worst_cached = std::max(worst_cached, dc);
          worst_whole_lp = std::max(worst_whole_lp, dl), worst_whole_mg = std::max(worst_whole_mg, dm);
        }
        const bool stopped = std::find(stop.begin(), stop.end(), on.ids.back()) != stop.end();
        const std::string text = generation_text(tok, on.ids, stopped);
        const std::vector<TokenRecord> rec = token_records(tok, on, stopped);
        CHECK(rec.size() == on.ids.size() && rec.back().end == int(text.size()), "the token spans do not cover the "
              "text (%zu records, last end %d, text %zu bytes)", rec.size(), rec.back().end, text.size());
        for (size_t k = 0; k < rec.size(); ++k)
          CHECK(rec[k].start == (k ? rec[k - 1].end : 0) && rec[k].end >= rec[k].start &&
                    rec[k].logprob == on.confidence[k].logprob && rec[k].margin == on.confidence[k].margin,
                "token %zu: span [%d, %d) does not follow the last", k, rec[k].start, rec[k].end);
        for (const Override& o : on.overridden)
          CHECK(rec[o.step].start == o.start && rec[o.step].end == o.end, "override at step %d: span [%d, %d), "
                "the token's [%d, %d)", o.step, o.start, o.end, rec[o.step].start, rec[o.step].end);
        tokens += on.ids.size(), ++answers, overrides += on.overridden.size();
      }
    }
    std::printf("token confidence (%s): %zu answers (free, the small schema, the example), %zu tokens, %zu "
                "overridden: the same tokens and overrides as off; against teacher-forced passes, through the cache at "
                "most %.2g, the whole answer at once at most %.2g in logprob and %.2g in margin (allowed %.2g); spans "
                "tile the text, an override's span its token's\n",
                dt == mx::float32 ? "float32" : "bfloat16", answers, tokens, overrides, worst_cached, worst_whole_lp,
                worst_whole_mg, whole_tol(dt));
  }
}

int main(int argc, char** argv) {
  CHECK(argc == 3, "usage: test_constrain MODEL_DIR EXAMPLE_SCHEMA (testdata/example_schema.json; the model's "
        "tokenizer is read, not its weights)");
  const Tokenizer tok = load_tokenizer(argv[1]);
  const auto t0 = std::chrono::steady_clock::now();
  const VocabTrie trie = build_trie(tok);
  std::printf("vocabulary trie: %zu tokens, %zu nodes, built in %.0f ms\n", trie.ids.size(), trie.byte.size(),
              ms_since(t0));
  const Schema ex = load_schema(argv[2]);
  refusals();
  example(tok, trie, ex);
  tokens_allowed(tok, trie, ex);
  free_tables(tok, trie);
  random_weights(tok, trie, ex);
  confidence(tok, trie, ex);
  speculative_drafts(tok, trie, ex);
  return 0;
}
