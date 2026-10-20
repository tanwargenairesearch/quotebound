#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

enum class ChatTemplate { Qwen3, Qwen35 };

// Byte-level BPE as declared by a Qwen tokenizer.json: NFC, the split regex, the byte-to-unicode map and ranked
// merges. Read-only once loaded, so one Tokenizer can serve several threads.
struct Tokenizer {
  std::vector<std::string> id_bytes;  // decode table: id -> raw bytes (added tokens: their text)
  std::unordered_map<std::string, int> added;  // added-token text -> id
  std::unordered_map<uint64_t, std::pair<int, int>> merges;  // (left id, right id) -> (rank, merged id)
  int byte_id[256] = {};
  bool marks_join_letters = false;  // which of the two split regexes tokenizer.json declares
  ChatTemplate chat = ChatTemplate::Qwen3;
  uint64_t template_fnv1a = 0;  // of the chat template's text, which picked `chat`
};

// Reads tokenizer.json, and the chat template from chat_template.jinja when present (transformers prefers it),
// otherwise from tokenizer_config.json.
Tokenizer load_tokenizer(const std::string& model_dir);

// Plain text to ids. Added-token strings such as "<|im_end|>" or "<think>" are not matched: they come out as
// ordinary BPE, so text can never forge a control token. (Hugging Face's encode matches them by default.)
std::vector<int> encode(const Tokenizer& t, std::string_view text);
// Throws on an id the tokenizer has no entry for.
std::string decode(const Tokenizer& t, const std::vector<int>& ids);
int token_id(const Tokenizer& t, std::string_view added_token);

// A conversation: an optional leading system turn, then user and assistant turns.
struct Turn {
  std::string role, content;
};

// A conversation ending in a user turn, then the generation prompt, as the model's chat template renders them with
// enable_thinking=False. chat_text is the rendered text; chat_ids emits the template's own control tokens as ids and
// encodes the text between them, turn content included, with encode(). Throws on the turns render_chat refuses, or a
// last turn that is not the user's.
std::string chat_text(const Tokenizer& t, const std::vector<Turn>& turns);
std::vector<int> chat_ids(const Tokenizer& t, const std::vector<Turn>& turns);
// {system, user}, or {user} without a system prompt.
std::vector<Turn> prompt_turns(const std::optional<std::string>& system, const std::string& user);

struct Rendered {
  std::vector<int> ids;
  std::vector<uint8_t> loss;  // per id: 1 for each assistant turn's content tokens and its <|im_end|>
};
// The whole conversation as the chat template renders it with no generation prompt, its control tokens as ids. An
// assistant turn after the last user turn opens with the empty think block, exactly the prompt chat_ids ends with.
// Throws on a role other than system/user/assistant, a system turn that is not first, no user turn, or assistant
// content holding "</think>" (the template would split it into reasoning).
Rendered render_chat(const Tokenizer& t, const std::vector<Turn>& turns);
