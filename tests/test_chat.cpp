// Whole conversations for training against the chat template as transformers renders them
// (testdata/chats_<model>.json): the text and ids, the loss mask on exactly each answer and its <|im_end|>, and the
// ids before the last answer equal to chat_ids of the turns before it, also as the loader cuts them.
#include <simdjson.h>

#include <filesystem>
#include <string>

#include "../src/data.h"
#include "../src/files.h"
#include "../src/json.h"
#include "../src/tokenizer.h"
#include "check.h"

static std::string text_of(simdjson::dom::element e) { return std::string(std::string_view(e)); }

int main(int argc, char** argv) {
  CHECK(argc == 3, "usage: test_chat MODEL_DIR testdata/chats_<model>.json");
  const std::string dir = argv[1];
  simdjson::dom::parser parser;
  simdjson::dom::element fx = parser.load(argv[2]);
  const std::string want_sha = text_of(fx["tokenizer_json_sha256"]), got_sha = sha256_file(dir + "/tokenizer.json");
  CHECK(got_sha == want_sha, "%s/tokenizer.json is not the file %s was dumped from", dir.c_str(), argv[2]);
  const Tokenizer tok = load_tokenizer(dir);

  int chats = 0, answers = 0;
  std::string lines;
  std::vector<std::vector<int>> expected_prompts;
  for (auto c : simdjson::dom::array(fx["chats"])) {
    std::vector<Turn> turns;
    for (auto t : simdjson::dom::array(c["turns"])) turns.push_back({text_of(t.at(0)), text_of(t.at(1))});
    const Rendered r = render_chat(tok, turns);
    std::vector<int> want;
    for (auto x : simdjson::dom::array(c["ids"])) want.push_back(int(int64_t(x)));
    CHECK(decode(tok, r.ids) == text_of(c["text"]), "chat %d: text differs:\n%s", chats, decode(tok, r.ids).c_str());
    CHECK(r.ids == want, "chat %d: ids differ from transformers' encoding of the rendered text", chats);

    // Runs of loss tokens, each decoded: exactly one per answer, the answer as the template wrote it.
    std::vector<std::string> spans;
    for (size_t i = 0; i < r.ids.size(); ++i) {
      if (!r.loss[i]) continue;
      if (i == 0 || !r.loss[i - 1]) spans.emplace_back();
      spans.back() += decode(tok, {r.ids[i]});
    }
    std::vector<std::string> trained;
    for (auto a : simdjson::dom::array(c["trained"])) trained.push_back(text_of(a));
    CHECK(spans == trained, "chat %d: %zu loss spans for %zu answers, or a span is not the answer", chats,
          spans.size(), trained.size());
    answers += int(trained.size());

    // Everything before the last answer is chat_ids of the turns before it: the prompt generation starts from.
    const std::vector<Turn> before(turns.begin(), turns.end() - 1);
    size_t last = r.ids.size();
    while (last > 0 && !r.loss[last - 1]) --last;
    while (last > 0 && r.loss[last - 1]) --last;
    CHECK(std::vector<int>(r.ids.begin(), r.ids.begin() + last) == chat_ids(tok, before),
          "chat %d: the tokens before the last answer are not chat_ids of the turns before it", chats);
    lines += "{\"id\": \"chat " + std::to_string(chats) + "\", \"messages\": [";
    for (size_t i = 0; i < turns.size(); ++i)
      lines += (i ? ", " : "") + std::string("{\"role\": ") + json_quote(turns[i].role) +
               ", \"content\": " + json_quote(turns[i].content) + "}";
    lines += "]}\n";
    expected_prompts.push_back(chat_ids(tok, before));
    ++chats;
  }
  // The loader's prompt_len, which dev generations start from, is that same prompt.
  const std::string jsonl = (std::filesystem::temp_directory_path() / "quotebound_test_chats.jsonl").string();
  write_file(jsonl, lines);
  const Dataset d = load_examples(jsonl, tok, 1 << 20);
  CHECK(d.examples.size() == expected_prompts.size(), "the loader kept %zu of %zu conversations", d.examples.size(),
        expected_prompts.size());
  for (size_t i = 0; i < d.examples.size(); ++i) {
    const Example& e = d.examples[i];
    CHECK(std::vector<int>(e.ids.begin(), e.ids.begin() + e.prompt_len) == expected_prompts[i],
          "conversation %zu: the loader's prompt is not chat_ids of the turns before the last answer", i);
  }
  std::filesystem::remove(jsonl);
  std::printf("chat %s: %d conversations render to transformers' text and ids; %d loss spans, each exactly an answer "
              "and its <|im_end|>; in each, the ids before the last answer are chat_ids of the turns before it, and "
              "the loader's prompt_len cuts there\n",
              argv[2], chats, answers);
  CHECK(chats >= 6, "fixture too small");
  return 0;
}
