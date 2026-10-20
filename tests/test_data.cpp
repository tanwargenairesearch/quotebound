// The training-data loader on a real tokenizer (drops, refusals, ids, prompts, system prompts, digests), batch plans,
// the seeded dev subset, and the deny-list refusal.
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <set>
#include <string>

#include "../src/data.h"
#include "../src/files.h"
#include "../src/json.h"
#include "../src/provenance.h"
#include "check.h"

namespace fs = std::filesystem;

static std::string temp(const std::string& name) { return (fs::temp_directory_path() / ("quotebound_" + name)).string(); }

static std::string row(const std::vector<Turn>& turns, const std::string& id = "") {
  std::string s = "{";
  if (!id.empty()) s += "\"id\": " + json_quote(id) + ", ";
  s += "\"messages\": [";
  for (size_t i = 0; i < turns.size(); ++i)
    s += (i ? ", " : "") + std::string("{\"role\": ") + json_quote(turns[i].role) + ", \"content\": " +
         json_quote(turns[i].content) + "}";
  return s + "]}\n";
}

// Rows kept, dropped (too long, no answer) and skipped (blank), and each refusal naming the file and line.
static void loader(const Tokenizer& tok) {
  const std::string sys = "Pull the figures out of the case note.";
  const std::vector<Turn> ok{{"system", sys}, {"user", "Client earns 52k."}, {"assistant", "{\"income\":52000}"}};
  const std::vector<Turn> plain{{"user", "Hello"}, {"assistant", "Hi."}};
  std::string long_note;
  for (int i = 0; i < 40; ++i) long_note += "The applicant earns a basic salary and some overtime. ";
  const std::string path = temp("loader.jsonl");
  write_file(path, row(ok, "first") + row(plain) + "\n" + row({{"user", long_note}, {"assistant", "{}"}}, "long") +
                       row({{"system", sys}, {"user", "No answer here"}}, "unanswered"));
  const Dataset d = load_examples(path, tok, 120);
  CHECK(d.examples.size() == 2 && d.examples[0].id == "first" && d.examples[1].id == "line 2",
        "kept %zu rows, ids %s / %s", d.examples.size(), d.examples.empty() ? "-" : d.examples[0].id.c_str(),
        d.examples.size() < 2 ? "-" : d.examples[1].id.c_str());
  CHECK(d.too_long == std::vector<std::string>{"long"} && d.no_answer == std::vector<std::string>{"unanswered"},
        "drops: %zu too long, %zu without an answer", d.too_long.size(), d.no_answer.size());
  const std::string sys_sha = sha256_hex(sys.data(), sys.size());
  CHECK(d.system_prompts.size() == 2 && d.system_prompts.at(sys_sha) == 1 && d.system_prompts.at("none") == 1,
        "system prompts counted wrong");
  for (size_t i = 0; i < 2; ++i) {
    const std::vector<Turn>& turns = i == 0 ? ok : plain;
    const Example& e = d.examples[i];
    CHECK(std::vector<int>(e.ids.begin(), e.ids.begin() + e.prompt_len) ==
              chat_ids(tok, std::vector<Turn>(turns.begin(), turns.end() - 1)),
          "row %zu: prompt_len does not cut at the prompt", i);
  }

  const std::vector<std::pair<std::string, std::string>> refused{
      {row({{"user", "a"}, {"tool", "x"}}), "unsupported role"},
      {row({{"user", "a"}, {"system", "b"}, {"assistant", "c"}}), "system turn must come first"},
      {row({{"system", "a"}, {"assistant", "b"}}), "no user turn"},
      {row({{"user", "a"}, {"assistant", "x</think>y"}}), "</think>"},
      {"{\"messages\": [}\n", ""},
      {"{\"turns\": []}\n", "missing key \"messages\""},
      {"{\"id\": 7, \"messages\": []}\n", "\"id\" has the wrong type"}};
  int n = 0;
  for (const auto& [line, why] : refused) {
    write_file(path, row(plain) + line);
    std::string what;
    try {
      load_examples(path, tok, 1000);
    } catch (const std::runtime_error& e) {
      what = e.what();
    }
    CHECK(what.starts_with(path + ":2") && what.find(why) != std::string::npos, "row %d: refusal \"%s\"", n,
          what.c_str());
    ++n;
  }
  fs::remove(path);

  std::vector<Example> a = d.examples, b = d.examples;
  b[1].loss[b[1].loss.size() - 2] ^= 1;
  CHECK(examples_sha256(a) == examples_sha256(d.examples) && examples_sha256(a) != examples_sha256(b),
        "the digest does not follow the loss mask");
  std::printf("loader: 2 rows kept (ids from \"id\" or the line), 1 too long and 1 without an answer dropped by id, "
              "a blank line skipped, system prompts counted by sha256, prompt_len at chat_ids of the turns before the "
              "answer; %d malformed rows refused naming file:line; the digest follows the loss mask\n",
              n);
}

// Every example exactly once an epoch, every batch within the token budget, the same plan for the same (seed, epoch)
// and a different one for the next epoch; plan_eval covers each example once.
static void plans() {
  std::mt19937_64 rng(3);
  std::vector<Example> ex(300);
  for (Example& e : ex) e.ids.assign(5 + rng() % 56, 1);
  const int budget = 256;
  for (int epoch = 0; epoch < 3; ++epoch) {
    const auto plan = plan_epoch(ex, budget, 9, epoch);
    std::vector<int> seen(ex.size(), 0);
    for (const auto& batch : plan) {
      size_t longest = 0;
      for (int i : batch) seen[i] += 1, longest = std::max(longest, ex[i].ids.size());
      CHECK(batch.size() * longest <= size_t(budget), "a batch of %zu x %zu exceeds %d", batch.size(), longest,
            budget);
    }
    CHECK(std::all_of(seen.begin(), seen.end(), [](int s) { return s == 1; }), "epoch %d: not every example once",
          epoch);
    CHECK(plan == plan_epoch(ex, budget, 9, epoch), "the plan is not deterministic");
    CHECK(plan != plan_epoch(ex, budget, 9, epoch + 1), "two epochs share a plan");
  }
  std::vector<int> seen(ex.size(), 0);
  for (const auto& batch : plan_eval(ex, budget))
    for (int i : batch) seen[i] += 1;
  CHECK(std::all_of(seen.begin(), seen.end(), [](int s) { return s == 1; }), "plan_eval misses or repeats a row");

  const std::vector<int> sub = seeded_subset(500, 200, 4);
  const std::set<int> distinct(sub.begin(), sub.end());
  CHECK(sub.size() == 200 && distinct.size() == 200 && *distinct.rbegin() < 500 && sub == seeded_subset(500, 200, 4) &&
            seeded_subset(50, 200, 4).size() == 50 && sub != seeded_subset(500, 200, 5),
        "seeded_subset");
  std::printf("plans: 300 examples, 3 epochs, budget %d: each example once an epoch, every batch within budget, "
              "deterministic per (seed, epoch), new each epoch; plan_eval once each; the dev subset is 200 distinct "
              "of 500, fixed by its seed\n",
              budget);
}

// Paths under a denied directory are refused, however they are spelled; neighbours are not. Then the environment's
// list, "~/" expansion, and every directory this build was compiled to deny.
static void denied() {
  auto refused = [](const std::string& p, const std::vector<std::string>& dirs) {
    try {
      refuse_denied("--train", p, dirs);
    } catch (const std::runtime_error&) {
      return true;
    }
    return false;
  };
  const std::vector<std::string> held = {temp("held-out")};
  CHECK(refused(temp("held-out") + "/never-read.jsonl", held), "a file under a denied directory");
  CHECK(refused(temp("held-out"), held), "the denied directory itself");
  CHECK(refused(temp("no-such-dir") + "/../" + fs::path(temp("held-out")).filename().string() + "/x", held),
        "a path through ..");
  CHECK(!refused(temp("held-out-other") + "/x.jsonl", held) && !refused(temp("x.jsonl"), held) &&
            !refused(temp("held-out") + "/x", {}),
        "a neighbour of a denied directory, or an empty list");
  const std::string home = std::getenv("HOME");
  CHECK(split_deny_list("~/a:/b::c") == std::vector<std::string>({home + "/a", "/b", "c"}), "the list's parsing");
  ::setenv("QUOTEBOUND_DENY", (temp("env-1") + ":" + temp("env-2")).c_str(), 1);
  CHECK(refused(temp("env-2") + "/x", deny_dirs()), "a directory from QUOTEBOUND_DENY");
  ::unsetenv("QUOTEBOUND_DENY");
  const std::vector<std::string> built = split_deny_list(QUOTEBOUND_DEFAULT_DENY);
  for (const std::string& d : built) CHECK(refused(d + "/never-read.jsonl", deny_dirs()), "the build's DENY: %s", d.c_str());
  std::printf("denied: paths under a denied directory refused (a file, the directory, through ..), neighbours "
              "accepted; QUOTEBOUND_DENY and \"~/\" read; this build denies %zu director%s%s%s\n",
              built.size(), built.size() == 1 ? "y" : "ies", built.empty() ? "" : ": ",
              built.empty() ? "" : QUOTEBOUND_DEFAULT_DENY);
}

int main(int argc, char** argv) {
  CHECK(argc == 2, "usage: test_data MODEL_DIR (its tokenizer only)");
  const Tokenizer tok = load_tokenizer(argv[1]);
  loader(tok);
  plans();
  denied();
  return 0;
}
