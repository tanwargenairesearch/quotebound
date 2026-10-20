// generate --batch on the tiny Qwen3.5 of tiny.h (its vocabulary widened to a checkpoint's tokenizer): rows read and
// refused; each row's prompt is the trainer's (the loader's prompt for the same conversation with its answer), thinking
// off; the answers, in order, equal greedy() run alone on each prompt and are valid JSON lines; a stop token ends an
// answer untruncated; a run cut short (a torn last line) and resumed equals the run straight through, answering
// nothing twice; and under schemas (a fallback with quotes from the user turn, a row's own), each answer is greedy()'s
// alone under the same constraint and fits its schema. Token confidence, free and constrained: the same answers and
// lines as without it but for "tokens", which tile the text and match a teacher-forced pass; without it, the lines
// are today's, byte for byte.
#include <simdjson.h>

#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>

#include "../src/batch.h"
#include "../src/constrain.h"
#include "../src/data.h"
#include "../src/files.h"
#include "../src/generate.h"
#include "../src/json.h"
#include "../src/schema.h"
#include "check.h"
#include "tiny.h"

namespace fs = std::filesystem;

static std::string temp(const std::string& name) { return (fs::temp_directory_path() / ("quotebound_" + name)).string(); }

static std::string row_json(const std::string& id, const std::vector<Turn>& turns) {
  std::string s = id.empty() ? "{" : "{\"id\": " + json_quote(id) + ", ";
  s += "\"messages\": [";
  for (size_t i = 0; i < turns.size(); ++i)
    s += (i ? ", " : "") + std::string("{\"role\": ") + json_quote(turns[i].role) + ", \"content\": " +
         json_quote(turns[i].content) + "}";
  return s + "]}\n";
}

static void refused(const std::string& jsonl, const std::string& want) {
  std::istringstream in(jsonl);
  try {
    read_batch(in, "in.jsonl");
  } catch (const std::runtime_error& e) {
    CHECK(std::string(e.what()).find(want) != std::string::npos, "refusal \"%s\" does not say \"%s\"", e.what(),
          want.c_str());
    return;
  }
  CHECK(false, "read_batch took a batch it must refuse (%s)", want.c_str());
}

static const std::string kSystem =
    "Read the guest's message to the booking desk and reply with its booking frame as compact JSON. Quote the "
    "guest's own words exactly; never compute a value.";
static const std::vector<std::string> kUsers = {
    "Hi, we are 34 people, €2,500 in all, a hall at €350 an hour with €35 a head for food. Which venues?",
    "Group booking:\nI'm 38 (vegetarian, €52 each), my partner's 41 and vegan — café regulars. Two kids aged 4 and 7.",
    "Does Harbour Hall take group bookings for a grandparent's party?",
    "Book Harbour Hall for this one please",
};

static void reading() {
  std::istringstream ok(row_json("a", {{"system", kSystem}, {"user", kUsers[0]}}) + "\n" +
                        row_json("b", {{"user", kUsers[1]}}));
  const std::vector<BatchRow> rows = read_batch(ok, "in.jsonl");
  CHECK(rows.size() == 2 && rows[0].id == "a" && rows[1].id == "b" && rows[0].turns.size() == 2 &&
            rows[0].turns[1].content == kUsers[0],
        "two rows read wrong");
  refused(row_json("", {{"user", "x"}}), "in.jsonl:1: missing key \"id\"");
  refused(row_json("a", {{"user", "x"}}) + row_json("a", {{"user", "y"}}), "in.jsonl:2: id \"a\" is repeated");
  refused(row_json("a", {{"user", "x"}, {"assistant", "y"}}), "in.jsonl:1: the messages do not end in a user turn");
  refused(row_json("a", {}), "in.jsonl:1: the messages do not end in a user turn");
  refused("{\"id\": \"a\", \"messages\": [", "in.jsonl:1:");
  refused("{\"id\": \"\", \"messages\": [{\"role\": \"user\", \"content\": \"x\"}]}", "the id is empty");
  std::printf("batch rows: read, and refused naming the line: no id, a repeated id, no user turn last, bad JSON\n");
}

// The prompt a batch row renders to is exactly the trainer's: load_examples' prompt for the same conversation with an
// answer, which ends in the empty think block (thinking off).
static void prompts_are_the_trainers(const Tokenizer& tok) {
  std::string lines;
  std::vector<std::vector<int>> prompts;
  for (size_t i = 0; i < kUsers.size(); ++i) {
    const std::vector<Turn> asked{{"system", kSystem}, {"user", kUsers[i]}};
    std::vector<Turn> trained = asked;
    trained.push_back({"assistant", "{\"intents\":[\"book\"],\"figures\":{\"guests\":\"34\"}}"});
    lines += row_json("t" + std::to_string(i), trained);
    prompts.push_back(chat_ids(tok, asked));
    const std::string text = decode(tok, prompts.back());
    const std::string tail = "<|im_start|>assistant\n<think>\n\n</think>\n\n";
    CHECK(text.size() > tail.size() && text.compare(text.size() - tail.size(), tail.size(), tail) == 0,
          "row %zu: the prompt does not end in the empty think block", i);
  }
  const std::string path = temp("batch_train.jsonl");
  write_file(path, lines);
  const Dataset d = load_examples(path, tok, 1 << 20);
  fs::remove(path);
  CHECK(d.examples.size() == kUsers.size(), "the loader kept %zu of %zu rows", d.examples.size(), kUsers.size());
  for (size_t i = 0; i < kUsers.size(); ++i) {
    const Example& e = d.examples[i];
    CHECK(std::vector<int>(e.ids.begin(), e.ids.begin() + e.prompt_len) == prompts[i],
          "row %zu: the batch prompt is not the trainer's", i);
  }
  std::printf("batch prompts: %zu rows render to the trainer's prompt token for token, ending in the empty think "
              "block\n",
              kUsers.size());
}

static std::vector<std::string> file_lines(const std::string& path) {
  std::vector<std::string> out;
  std::istringstream in(read_file(path));
  for (std::string l; std::getline(in, l);) out.push_back(l);
  return out;
}

static void runs(const Tokenizer& tok) {
  Model m = tiny(mx::bfloat16, true, 3, int(tok.id_bytes.size()));
  std::vector<BatchRow> rows;
  std::vector<std::vector<int>> prompts;
  for (size_t i = 0; i < kUsers.size(); ++i) {
    rows.push_back({"r" + std::to_string(i), {{"system", kSystem}, {"user", kUsers[i]}}, ""});
    prompts.push_back(chat_ids(tok, rows.back().turns));
  }
  const int cap = 12;
  std::vector<BatchResult> straight;
  run_batch(m, tok, rows, prompts, {}, cap, [&](const BatchResult& r) { straight.push_back(r); });
  CHECK(straight.size() == rows.size(), "%zu answers for %zu rows", straight.size(), rows.size());
  const std::vector<int> stop = stop_tokens(m, tok);
  simdjson::dom::parser parser;
  for (size_t i = 0; i < rows.size(); ++i) {
    const BatchResult& r = straight[i];
    const Generation alone = greedy(m, prompts[i], cap, stop);
    const bool stopped = std::find(stop.begin(), stop.end(), alone.ids.back()) != stop.end();
    CHECK(r.id == rows[i].id && r.prompt_tokens == prompts[i].size() && r.gen_tokens == alone.ids.size() &&
              r.text == generation_text(tok, alone.ids, stopped) && r.truncated == !stopped,
          "row %zu: the batch answer is not greedy()'s alone", i);
    CHECK(r.gen_tokens <= size_t(cap) && r.total_ms >= r.prefill_ms + r.decode_ms - 1e-6, "row %zu: counts", i);
    const std::string line = batch_line(r);
    simdjson::dom::element e;
    CHECK(!parser.parse(line).get(e), "row %zu: the line is not valid JSON (UTF-8)", i);
    CHECK(std::string_view(e["id"]) == r.id && uint64_t(e["gen_tokens"]) == r.gen_tokens &&
              bool(e["truncated"]) == r.truncated && line.back() == '\n',
          "row %zu: the line does not carry the result", i);
  }

  // A stop token ends an answer: it is counted, left out of the text, and the answer is not truncated.
  Model stops = tiny(mx::bfloat16, true, 3, int(tok.id_bytes.size()));
  stops.cfg.eos = {greedy(stops, prompts[0], 1, {}).ids[0]};
  std::vector<BatchResult> one;
  run_batch(stops, tok, {rows[0]}, {prompts[0]}, {}, cap, [&](const BatchResult& r) { one.push_back(r); });
  CHECK(one.size() == 1 && one[0].gen_tokens == 1 && one[0].text.empty() && !one[0].truncated,
        "a first-token stop gave %zu tokens, text \"%s\", truncated %d", one[0].gen_tokens, one[0].text.c_str(),
        int(one[0].truncated));

  // Cut after two rows with a torn third line, resumed: the file is the straight run's lines, nothing twice.
  const std::string out = temp("batch_out.jsonl");
  fs::remove(out);
  {
    LineFile f(out);
    f.append(batch_line(straight[0]));
    f.append(batch_line(straight[1]));
  }
  std::ofstream(out, std::ios::app) << "{\"id\":\"r2\",\"te";
  const std::set<std::string> done = resume_ids(out, rows);
  CHECK(done == std::set<std::string>({"r0", "r1"}), "resume found %zu ids", done.size());
  CHECK(read_file(out) == batch_line(straight[0]) + batch_line(straight[1]), "the torn line is still in the file");
  {
    LineFile f(out);
    size_t n = 0;
    run_batch(m, tok, rows, prompts, done, cap, [&](const BatchResult& r) { f.append(batch_line(r)), ++n; });
    CHECK(n == 2, "the resumed run answered %zu rows, not the 2 left", n);
  }
  const std::vector<std::string> got = file_lines(out);
  CHECK(got.size() == rows.size(), "%zu lines after the resume", got.size());
  for (size_t i = 0; i < rows.size(); ++i) {
    simdjson::dom::element e = parser.parse(got[i]);
    CHECK(std::string_view(e["id"]) == rows[i].id && std::string_view(e["text"]) == straight[i].text,
          "line %zu after the resume differs from the straight run", i);
  }
  // An output holding an id the input does not, or a whole line that is not a result, is refused.
  for (const std::string& bad : {std::string("{\"id\":\"zz\",\"text\":\"\"}\n"), std::string("not json\n")}) {
    write_file(out, bad);
    bool threw = false;
    try {
      resume_ids(out, rows);
    } catch (const std::runtime_error&) {
      threw = true;
    }
    CHECK(threw, "resume took an output it must refuse: %s", bad.c_str());
  }
  fs::remove(out);
  std::printf("batch runs (tiny Qwen3.5, vocabulary %d): %zu answers equal greedy() alone, valid JSON lines; a stop "
              "token ends an answer untruncated; cut with a torn line and resumed, the file equals the straight run\n",
              m.cfg.vocab, rows.size());
}

// Rows without a schema take the fallback (a quote from the user turn and a count); a row with its own schema (a
// follow-up questions reply's shape: these targets, in order) takes that. Read from JSONL as the reader writes it.
static void constrained_runs(const Tokenizer& tok) {
  static const char* kFallback =
      R"({"type":"object","additionalProperties":false,"required":["quote","n"],"properties":{)"
      R"("quote":{"type":"string","minLength":1},"n":{"type":"integer","minimum":0}}})";
  static const char* kOwn =
      R"({"type":"array","items":false,"minItems":2,"prefixItems":[)"
      R"({"type":"object","additionalProperties":false,"required":["target","question"],"properties":{"target":)"
      R"({"const":"date"},"question":{"type":"string","minLength":8,"maxLength":40,"pattern":"^[^0-9%$€¥]+$"}}},)"
      R"({"type":"object","additionalProperties":false,"required":["target","question"],"properties":{"target":)"
      R"({"const":"guests"},"question":{"type":"string","minLength":8,"maxLength":40,"pattern":"^[^0-9%$€¥]+$"}}}]})";
  std::string lines;
  for (size_t i = 0; i < kUsers.size(); ++i) {
    std::string l = row_json("c" + std::to_string(i), {{"system", kSystem}, {"user", kUsers[i]}});
    if (i == 1) l = l.substr(0, l.size() - 2) + ", \"schema\": " + kOwn + "}\n";
    lines += l;
  }
  std::istringstream in(lines);
  const std::vector<BatchRow> rows = read_batch(in, "in.jsonl");
  CHECK(rows[0].schema.empty() && !rows[1].schema.empty(), "the row's own schema was not read");
  simdjson::dom::parser sp, op;
  const Schema fallback = compile_schema(sp.parse(std::string(kFallback)), "fallback");
  BatchConstraints bc;
  const VocabTrie trie = build_trie(tok);
  bc.fallback = &fallback, bc.trie = &trie, bc.quote_user = true;
  bc.by_text.emplace(rows[1].schema, compile_schema(op.parse(rows[1].schema), "own"));
  Model m = tiny(mx::bfloat16, true, 3, int(tok.id_bytes.size()));
  std::vector<std::vector<int>> prompts;
  for (const BatchRow& r : rows) prompts.push_back(chat_ids(tok, r.turns));
  const int cap = 160;
  std::vector<BatchResult> got;
  run_batch(m, tok, rows, prompts, {}, cap, [&](const BatchResult& r) { got.push_back(r); }, &bc);
  CHECK(got.size() == rows.size(), "%zu answers for %zu rows", got.size(), rows.size());
  const std::vector<int> stop = stop_tokens(m, tok);
  simdjson::dom::parser parser;
  size_t overrides = 0;
  for (size_t i = 0; i < rows.size(); ++i) {
    const Schema& s = rows[i].schema.empty() ? fallback : bc.by_text.at(rows[i].schema);
    const QuoteText q = quote_text(rows[i].turns.back().content);
    Constraint c(s, tok, trie, stop, &q);
    const Generation alone = greedy(m, prompts[i], cap, stop, {}, &c);
    const bool stopped = std::find(stop.begin(), stop.end(), alone.ids.back()) != stop.end();
    CHECK(got[i].text == generation_text(tok, alone.ids, stopped) && got[i].gen_tokens == alone.ids.size() &&
              got[i].mask_ms >= 0 && got[i].overridden == alone.overridden && stopped && !got[i].truncated,
          "row %zu: the constrained batch answer is not greedy()'s alone, stopped", i);
    simdjson::dom::element e;
    CHECK(!parser.parse(got[i].text).get(e), "row %zu: \"%s\" is not JSON", i, got[i].text.c_str());
    if (i == 1) {
      simdjson::dom::array a = e;
      CHECK(a.size() == 2 && std::string_view(a.at(0)["target"]) == "date" &&
                std::string_view(a.at(1)["target"]) == "guests",
            "row 1 does not fit its own schema: %s", got[i].text.c_str());
    } else {
      std::string quote{std::string_view(e["quote"])}, user = rows[i].turns.back().content;
      for (std::string* x : {&quote, &user})
        for (char& ch : *x) ch = char(std::tolower(static_cast<unsigned char>(ch)));
      CHECK(user.find(quote) != std::string::npos, "row %zu: quote \"%s\" is not in the message", i, quote.c_str());
    }
    simdjson::dom::element line;
    CHECK(!parser.parse(batch_line(got[i])).get(line) && double(line["mask_ms"]) >= 0 &&
              bool(line["budget_cut"]) == got[i].budget_cut,
          "row %zu: the line does not carry mask_ms and budget_cut", i);
    simdjson::dom::array spans = line["overridden"], steps = line["overridden_steps"];
    CHECK(spans.size() == got[i].overridden.size() && steps.size() == got[i].overridden.size(),
          "row %zu: the line carries %zu override spans for %zu overrides", i, spans.size(), got[i].overridden.size());
    size_t k = 0;
    for (simdjson::dom::element sp : spans) {
      const Override& o = got[i].overridden[k];
      CHECK(int64_t(sp.at(0)) == o.start && int64_t(sp.at(1)) == o.end && int64_t(steps.at(k)) == o.step,
            "row %zu: override %zu is written wrong", i, k);
      ++k;
    }
    overrides += got[i].overridden.size();
  }
  // Drafts (a table from the answers themselves, so most drafts hold, and none: prompt lookup alone): the same
  // answers, overrides and budget cuts.
  BatchDrafts own;
  {
    NgramTable& t = own.tables.try_emplace(kSystem, 4).first->second;
    for (size_t i = 0; i < rows.size(); ++i) {
      const Schema& s = rows[i].schema.empty() ? fallback : bc.by_text.at(rows[i].schema);
      const QuoteText q = quote_text(rows[i].turns.back().content);
      Constraint c(s, tok, trie, stop, &q);
      t.add(greedy(m, prompts[i], cap, stop, {}, &c).ids);
    }
    t.finish();
  }
  BatchDrafts lookup;
  long passes = 0, drafted = 0, tokens = 0;
  for (int arm = 0; arm < 2; ++arm) {
    std::vector<BatchResult> again;
    const BatchStats st = run_batch(m, tok, rows, prompts, {}, cap, [&](const BatchResult& r) { again.push_back(r); },
                                    &bc, false, arm == 0 ? &own : &lookup);
    for (size_t i = 0; i < rows.size(); ++i) {
      CHECK(again[i].text == got[i].text && again[i].gen_tokens == got[i].gen_tokens &&
                again[i].overridden == got[i].overridden && again[i].budget_cut == got[i].budget_cut,
            "arm %d row %zu: the answer differs from the plain batch's", arm, i);
      CHECK(again[i].passes >= 0, "arm %d row %zu: passes %d", arm, i, again[i].passes);
      simdjson::dom::element line;
      CHECK(!parser.parse(batch_line(again[i])).get(line) && int64_t(line["passes"]) == again[i].passes &&
                int64_t(line["drafted"]) == again[i].drafted,
            "arm %d row %zu: the line does not carry passes and drafted", arm, i);
    }
    if (arm == 0) passes = st.passes, drafted = st.drafted, tokens = st.tokens;
  }
  std::printf("constrained batch: %zu rows (fallback schema quoting the user turn; one row's own follow-up "
              "schema) equal greedy() alone under the same constraint, overrides included (%zu, each written as its "
              "span and step), stop, and fit their schemas; the same with drafts (the answers' own n-grams: %ld "
              "tokens in %ld passes after the prefill, %ld drafted)\n",
              rows.size(), overrides, tokens, passes, drafted);
}

// Lines without token confidence, as they were before it: pinned byte for byte.
static void lines_unchanged() {
  BatchResult free{"a", "x", 3, 2, 1.0, 2.0, 3.5, false};
  CHECK(batch_line(free) == "{\"id\":\"a\",\"text\":\"x\",\"prompt_tokens\":3,\"gen_tokens\":2,\"prefill_ms\":1.0,"
                            "\"decode_ms\":2.0,\"total_ms\":3.5,\"truncated\":false}\n",
        "an unconstrained line changed: %s", batch_line(free).c_str());
  BatchResult c = free;
  c.mask_ms = 0.5, c.overridden = {{1, 0, 1}};
  CHECK(batch_line(c) == "{\"id\":\"a\",\"text\":\"x\",\"prompt_tokens\":3,\"gen_tokens\":2,\"prefill_ms\":1.0,"
                         "\"decode_ms\":2.0,\"total_ms\":3.5,\"truncated\":false,\"mask_ms\":0.5,\"budget_cut\":false,"
                         "\"overridden\":[[0,1]],\"overridden_steps\":[1]}\n",
        "a constrained line changed: %s", batch_line(c).c_str());
  const std::string before = batch_line(c);
  c.tokens = {{0, 1, -0.5f, 2.25f}, {1, 1, -1e-7f, 30.f}};
  CHECK(batch_line(c) == before.substr(0, before.size() - 2) + ",\"tokens\":[[0,1,-0.5,2.25],[1,1,-1e-07,30]]}\n",
        "the tokens are written wrong: %s", batch_line(c).c_str());
  std::printf("batch lines: without token confidence, free and constrained, byte for byte as before; with it, "
              "\"tokens\" last\n");
}

// Free rows and constrained rows (the fallback schema, quoting the user turn), run with token confidence off and on.
static void confidence_runs(const Tokenizer& tok) {
  static const char* kFallback =
      R"({"type":"object","additionalProperties":false,"required":["quote","n"],"properties":{)"
      R"("quote":{"type":"string","minLength":1},"n":{"type":"integer","minimum":0}}})";
  simdjson::dom::parser sp;
  const Schema fallback = compile_schema(sp.parse(std::string(kFallback)), "fallback");
  const VocabTrie trie = build_trie(tok);
  BatchConstraints bc;
  bc.fallback = &fallback, bc.trie = &trie, bc.quote_user = true;
  Model m = tiny(mx::bfloat16, true, 3, int(tok.id_bytes.size()));
  std::vector<BatchRow> rows;
  std::vector<std::vector<int>> prompts;
  for (size_t i = 0; i < kUsers.size(); ++i) {
    rows.push_back({"k" + std::to_string(i), {{"system", kSystem}, {"user", kUsers[i]}}, ""});
    prompts.push_back(chat_ids(tok, rows.back().turns));
  }
  simdjson::dom::parser parser;
  size_t tokens = 0;
  double worst = 0;
  const BatchConstraints* const constrained = &bc;
  for (const BatchConstraints* cons : {static_cast<const BatchConstraints*>(nullptr), constrained}) {
    const int cap = cons ? 120 : 24;
    std::vector<BatchResult> off, on;
    run_batch(m, tok, rows, prompts, {}, cap, [&](const BatchResult& r) { off.push_back(r); }, cons);
    run_batch(m, tok, rows, prompts, {}, cap, [&](const BatchResult& r) { on.push_back(r); }, cons, true);
    CHECK(off.size() == rows.size() && on.size() == rows.size(), "answers missing");
    for (size_t i = 0; i < rows.size(); ++i) {
      CHECK(on[i].text == off[i].text && on[i].gen_tokens == off[i].gen_tokens && on[i].overridden == off[i].overridden &&
                on[i].truncated == off[i].truncated && on[i].budget_cut == off[i].budget_cut,
            "row %zu (%s): confidence on changed the answer", i, cons ? "constrained" : "free");
      const std::string lo = batch_line(off[i]), ln = batch_line(on[i]);
      CHECK(off[i].tokens.empty() && lo.find("\"tokens\"") == std::string::npos, "row %zu: tokens with it off", i);
      simdjson::dom::element e;
      CHECK(!parser.parse(ln).get(e), "row %zu: the line is not JSON", i);
      CHECK(std::string_view(e["text"]) == on[i].text, "row %zu: the text differs", i);
      simdjson::dom::array ts = e["tokens"];
      CHECK(ts.size() == on[i].gen_tokens, "row %zu: %zu tokens written for %zu generated", i, ts.size(),
            on[i].gen_tokens);
      // the decode alone, with confidence, and a teacher-forced pass over the whole answer
      const std::vector<int> stop = stop_tokens(m, tok);
      std::optional<QuoteText> q;
      std::optional<Constraint> c;
      if (cons) q = quote_text(rows[i].turns.back().content), c.emplace(fallback, tok, trie, stop, &*q);
      const Generation alone = greedy(m, prompts[i], cap, stop, {}, c ? &*c : nullptr, true);
      const auto want = teacher_forced(m, prompts[i], alone.ids, true);
      int64_t at = 0;
      size_t k = 0;
      for (simdjson::dom::element t : ts) {
        simdjson::dom::array x = t;
        const int64_t s = x.at(0), end = x.at(1);
        const double lp = x.at(2), mg = x.at(3);
        CHECK(s == at && end >= s && size_t(end) <= on[i].text.size(), "row %zu token %zu: span [%lld, %lld) does not "
              "follow %lld", i, k, (long long)s, (long long)end, (long long)at);
        CHECK(std::abs(lp - alone.confidence[k].logprob) <= 1e-5 * std::max(1.0, std::abs(lp)) &&
                  std::abs(mg - alone.confidence[k].margin) <= 1e-5 * std::max(1.0, std::abs(mg)),
              "row %zu token %zu: written (%g, %g), greedy() alone (%g, %g)", i, k, lp, mg,
              alone.confidence[k].logprob, alone.confidence[k].margin);
        CHECK(std::abs(lp - want[k].first) <= 0.15 && std::abs(mg - want[k].second) <= 0.15,
              "row %zu token %zu: written (%g, %g), teacher-forced (%g, %g)", i, k, lp, mg, want[k].first,
              want[k].second);
        worst = std::max({worst, std::abs(lp - want[k].first), std::abs(mg - want[k].second)});
        at = end, ++k;
      }
      CHECK(size_t(at) == on[i].text.size(), "row %zu: the spans end at %lld, the text at %zu", i, (long long)at,
            on[i].text.size());
      tokens += k;
    }
  }
  std::printf("batch token confidence (tiny Qwen3.5, bfloat16): %zu rows free and %zu constrained, %zu tokens: the "
              "same answers and lines as off but for \"tokens\", whose spans tile the text and whose numbers are "
              "greedy()'s alone, within %.2g of a whole-answer teacher-forced pass (allowed 0.15)\n",
              rows.size(), rows.size(), tokens, worst);
}

int main(int argc, char** argv) {
  CHECK(argc == 2, "usage: test_batch MODEL_DIR (its tokenizer is read, not its weights)");
  const Tokenizer tok = load_tokenizer(argv[1]);
  reading();
  prompts_are_the_trainers(tok);
  runs(tok);
  constrained_runs(tok);
  lines_unchanged();
  confidence_runs(tok);
  return 0;
}
