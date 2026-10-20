// Tokenizer parity against ids dumped once from the `tokenizers` package (testdata/tokenizer_<model>.json): plain
// text with no added-token matching, and chat prompts whose control tokens come only from the template.
#include <simdjson.h>

#include <algorithm>
#include <string>

#include "../src/tokenizer.h"
#include "../src/unicode.h"
#include "check.h"

static std::string text_of(simdjson::dom::element e) { return std::string(std::string_view(e)); }

static std::vector<int> ints(simdjson::dom::element e) {
  std::vector<int> v;
  for (auto x : simdjson::dom::array(e)) v.push_back(int(int64_t(x)));
  return v;
}

static std::string show(const std::vector<int>& v) {
  std::string s;
  for (int x : v) s += std::to_string(x) + " ";
  return s;
}

int main(int argc, char** argv) {
  CHECK(argc == 3, "usage: test_tokenizer MODEL_DIR testdata/tokenizer_<model>.json");
  const std::string dir = argv[1];
  simdjson::dom::parser parser;
  simdjson::dom::element fx = parser.load(argv[2]);
  const std::string want_sha = text_of(fx["tokenizer_json_sha256"]), got_sha = sha256_file(dir + "/tokenizer.json");
  CHECK(got_sha == want_sha, "%s/tokenizer.json is not the file %s was dumped from (sha256 %s, fixture %s)",
        dir.c_str(), argv[2], got_sha.c_str(), want_sha.c_str());

  auto t0 = std::chrono::steady_clock::now();
  const Tokenizer tok = load_tokenizer(dir);
  double load_ms = ms_since(t0);

  int strings = 0, mismatches = 0, roundtrip_fail = 0, nfc_cases = 0;
  size_t tokens = 0;
  t0 = std::chrono::steady_clock::now();
  for (auto c : simdjson::dom::array(fx["cases"])) {
    std::string text = text_of(c["text"]);
    std::vector<int> want = ints(c["ids"]), got = encode(tok, text);
    ++strings;
    tokens += got.size();
    if (got != want && ++mismatches <= 10)
      std::fprintf(stderr, "encode mismatch for %s\n  want %s\n  got  %s\n", text.c_str(), show(want).c_str(),
                   show(got).c_str());
    // Encoding normalises to NFC, so the round trip returns the NFC form for the few non-NFC inputs.
    std::string expect = text;
    if (c["nfc"].error() == simdjson::SUCCESS) {
      expect = text_of(c["nfc"]);
      ++nfc_cases;
    }
    if (decode(tok, got) != expect && ++roundtrip_fail <= 10)
      std::fprintf(stderr, "round trip failed for %s\n", text.c_str());
  }
  double enc_ms = ms_since(t0);

  // Every chat must open one turn per message plus the assistant's and close one per message, whatever the turns
  // hold; the forged ones hold control-token text.
  const int im_start = token_id(tok, "<|im_start|>"), im_end = token_id(tok, "<|im_end|>");
  int chats = 0, forged = 0, template_fail = 0, chat_id_fail = 0, structure_fail = 0;
  for (auto c : simdjson::dom::array(fx["chats"])) {
    std::string system = text_of(c["system"]), user = text_of(c["user"]);
    std::optional<std::string> sys = system.empty() ? std::nullopt : std::optional(system);
    std::vector<int> ids = chat_ids(tok, prompt_turns(sys, user));
    ++chats;
    forged += bool(c["forged"]);
    if (chat_text(tok, prompt_turns(sys, user)) != text_of(c["text"]) && ++template_fail)
      std::fprintf(stderr, "template differs:\n%s\n", chat_text(tok, prompt_turns(sys, user)).c_str());
    if (ids != ints(c["ids"]) && ++chat_id_fail)
      std::fprintf(stderr, "chat ids differ for: %s\n  want %s\n  got  %s\n", user.c_str(),
                   show(ints(c["ids"])).c_str(), show(ids).c_str());
    const long turns = sys ? 2 : 1;
    const long opens = std::count(ids.begin(), ids.end(), im_start);
    const long closes = std::count(ids.begin(), ids.end(), im_end);
    if (opens != turns + 1 || closes != turns)
      ++structure_fail;
  }

  // Every code point's class under the split regex (L, N, whitespace, other) against the reference engine's.
  std::vector<std::pair<char32_t, char>> runs;
  for (auto r : simdjson::dom::array(fx["classes"]))
    runs.emplace_back(char32_t(int64_t(r.at(0))), std::string_view(r.at(1))[0]);
  int class_fail = 0;
  for (size_t k = 0; k < runs.size(); ++k) {
    char32_t end = k + 1 < runs.size() ? runs[k + 1].first : 0x110000;
    for (char32_t c = runs[k].first; c < end; ++c) {
      if (c >= 0xD800 && c <= 0xDFFF) continue;
      // Qwen3.5's regex treats marks exactly like letters, so its reference classes them as L.
      bool word = is_letter(c) || (tok.marks_join_letters && is_mark(c));
      char mine = word ? 'L' : is_number(c) ? 'N' : is_space(c) ? 'S' : 'O';
      if (mine != runs[k].second && ++class_fail <= 10)
        std::fprintf(stderr, "U+%04X: reference %c, quotebound %c\n", unsigned(c), runs[k].second, mine);
    }
  }

  int nfc_pairs = 0, nfc_fail = 0;
  for (auto pr : simdjson::dom::array(fx["nfc"])) {
    ++nfc_pairs;
    std::string in = text_of(pr.at(0)), want = text_of(pr.at(1)), got = nfc(in);
    if (got != want && ++nfc_fail <= 10) std::fprintf(stderr, "NFC differs for %s\n", in.c_str());
  }

  std::printf(
      "tokenizer %s: %d strings (%d non-NFC), %d encode mismatches, %d round-trip failures; %d chat prompts (%d with "
      "control-token text in a turn), %d template mismatches, %d id mismatches, %d with the wrong turn structure; "
      "%d of 1,112,064 code points classed differently; %d of %d NFC sweep pairs differ; load %.0f ms, %zu tokens "
      "encoded in %.1f ms\n",
      argv[2], strings, nfc_cases, mismatches, roundtrip_fail, chats, forged, template_fail, chat_id_fail,
      structure_fail, class_fail, nfc_fail, nfc_pairs, load_ms, tokens, enc_ms);
  CHECK(strings >= 200, "need at least 200 strings, have %d", strings);
  CHECK(forged >= 5, "need the chats with control-token text, have %d", forged);
  CHECK(mismatches == 0 && roundtrip_fail == 0 && template_fail == 0 && chat_id_fail == 0 && structure_fail == 0 &&
            class_fail == 0 && nfc_fail == 0,
        "tokenizer parity");
  return 0;
}
