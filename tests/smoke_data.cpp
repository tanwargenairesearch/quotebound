// Synthetic chat data for the training smoke run and the throughput measurements, generated here from a seed.
//   smoke_data task DIR [--seed S] [--train N] [--dev N]
//     DIR/train.jsonl and DIR/dev.jsonl: a templated case note in, its figures out as one line of JSON.
//   smoke_data long FILE --model DIR --tokens L --count N [--seed S]
//     N conversations of at most L tokens under DIR's tokenizer, within one note of L: many notes in, one JSON line
//     per note out.
#include <cstdio>
#include <cstring>
#include <fstream>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "../src/json.h"
#include "../src/tokenizer.h"

static const char* kSystem = "Pull the figures out of the case note.";

struct Note {
  std::string text, json;
};

static std::string with_commas(long v) {
  std::string s = std::to_string(v);
  for (int i = int(s.size()) - 3; i > 0; i -= 3) s.insert(size_t(i), ",");
  return s;
}

// A money amount in one of the ways adviser notes write it.
static std::string money(long v, std::mt19937_64& rng) {
  char buf[64];
  switch (rng() % 4) {
    case 0:
      return "£" + with_commas(v);
    case 1:
      return "£" + std::to_string(v);
    case 2:
      if (v % 100 == 0) {
        std::snprintf(buf, sizeof buf, "%gk", v / 1000.0);
        return buf;
      }
      return "£" + with_commas(v);
    default:
      return with_commas(v) + " pounds";
  }
}

static Note note(std::mt19937_64& rng) {
  auto pick = [&](long lo, long hi, long step) { return lo + long(rng() % uint64_t((hi - lo) / step + 1)) * step; };
  const long age = pick(21, 68, 1), income = pick(18000, 240000, 100), value = pick(90000, 1200000, 5000);
  const long loan = (value * long(50 + rng() % 46) / 100) / 1000 * 1000, term = pick(5, 40, 1);
  // Each fact is left out one time in eight; the answer then has null.
  std::vector<bool> has(5);
  for (auto&& h : has) h = rng() % 8 != 0;
  std::vector<std::string> parts;
  const char* who[] = {"The applicant", "My client", "The borrower", "Client"};
  if (has[0]) {
    const char* forms[] = {"is %ld", "is aged %ld", "is a %ld-year-old"};
    char buf[64];
    std::snprintf(buf, sizeof buf, forms[rng() % 3], age);
    parts.push_back(buf);
  }
  if (has[1]) {
    const char* forms[] = {"earns %s a year", "has a basic salary of %s", "has an income of %s per annum"};
    char buf[128];
    std::snprintf(buf, sizeof buf, forms[rng() % 3], money(income, rng).c_str());
    parts.push_back(buf);
  }
  if (has[2]) parts.push_back((rng() % 2 ? "wants to borrow " : "needs a loan of ") + money(loan, rng));
  if (has[3])
    parts.push_back((rng() % 2 ? "is buying a home worth " : "has found a property valued at ") + money(value, rng));
  if (has[4])
    parts.push_back((rng() % 2 ? "wants to repay over " : "wants a term of ") + std::to_string(term) + " years");
  for (size_t i = parts.size(); i > 1; --i) std::swap(parts[i - 1], parts[rng() % i]);
  std::string text = who[rng() % 4];
  for (size_t i = 0; i < parts.size(); ++i)
    text += (i == 0 ? " " : i + 1 == parts.size() ? " and " : ", ") + parts[i];
  if (parts.empty()) text += " has not given any figures yet";
  text += ".";
  const char* asides[] = {"", " Self-employed for three years.", " First-time buyer.", " Two dependants.",
                          " Wants to fix for five years."};
  text += asides[rng() % 5];
  auto field = [&](const char* k, bool h, long v) {
    return std::string("\"") + k + "\":" + (h ? std::to_string(v) : "null");
  };
  std::string json = "{" + field("age", has[0], age) + "," + field("income", has[1], income) + "," +
                     field("loan", has[2], loan) + "," + field("value", has[3], value) + "," +
                     field("term", has[4], term) + "}";
  return {text, json};
}

static std::string line(const std::string& user, const std::string& answer) {
  return "{\"messages\": [{\"role\": \"system\", \"content\": " + json_quote(kSystem) +
         "}, {\"role\": \"user\", \"content\": " + json_quote(user) + "}, {\"role\": \"assistant\", \"content\": " +
         json_quote(answer) + "}]}\n";
}

static int usage() {
  std::fprintf(stderr,
               "usage: smoke_data task DIR [--seed S] [--train N] [--dev N]\n"
               "       smoke_data long FILE --model DIR --tokens L --count N [--seed S]\n");
  return 2;
}

int main(int argc, char** argv) {
  if (argc < 3) return usage();
  const std::string mode = argv[1], out = argv[2];
  long seed = 0, train = 2000, dev = 100, tokens = 0, count = 0;
  std::string model;
  for (int i = 3; i + 1 < argc; i += 2) {
    const std::string k = argv[i];
    if (k == "--model") {
      model = argv[i + 1];
      continue;
    }
    long* v = k == "--seed" ? &seed : k == "--train" ? &train : k == "--dev" ? &dev : k == "--tokens" ? &tokens
            : k == "--count" ? &count : nullptr;
    if (!v) return usage();
    *v = std::atol(argv[i + 1]);
  }
  std::mt19937_64 rng{uint64_t(seed)};
  if (mode == "task") {
    for (auto [name, n] : {std::pair{"train", train}, std::pair{"dev", dev}}) {
      std::ofstream f(out + "/" + name + ".jsonl");
      if (!f) return std::fprintf(stderr, "cannot write %s/%s.jsonl\n", out.c_str(), name), 1;
      for (long i = 0; i < n; ++i) {
        Note x = note(rng);
        f << line(x.text, x.json);
      }
    }
    return 0;
  }
  if (mode != "long" || model.empty() || tokens < 64 || count < 1) return usage();
  const Tokenizer tok = load_tokenizer(model);
  std::ofstream f(out);
  if (!f) return std::fprintf(stderr, "cannot write %s\n", out.c_str()), 1;
  for (long c = 0; c < count; ++c) {
    std::string user, answer;
    for (int k = 1;; ++k) {
      Note x = note(rng);
      std::string u = user + std::to_string(k) + ". " + x.text + "\n", a = answer + x.json + "\n";
      const size_t n = render_chat(tok, {{"system", kSystem}, {"user", u}, {"assistant", a}}).ids.size();
      if (n > size_t(tokens)) break;
      user = u, answer = a;
      if (n + 32 >= size_t(tokens)) break;
    }
    f << line(user, answer);
  }
  return 0;
}
