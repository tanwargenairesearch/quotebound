#pragma once
// `generate --batch`: one model load, many conversations, one JSON line out per conversation.
#include <functional>
#include <istream>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "constrain.h"
#include "draft.h"
#include "generate.h"
#include "model.h"
#include "schema.h"
#include "tokenizer.h"

// One input row: {"id": "...", "messages": [{"role", "content"}, ...], "schema"?: {...}}, the messages ending in a
// user turn; a row's own JSON Schema constrains its answer in place of --schema.
struct BatchRow {
  std::string id;
  std::vector<Turn> turns;
  std::string schema;  // the row's schema as minified JSON, empty for none
};

// JSONL rows, blank lines skipped. Throws naming `where` and the line on malformed JSON, a missing, empty or repeated
// id, or messages that do not end in a user turn.
std::vector<BatchRow> read_batch(std::istream& in, const std::string& where);

struct BatchResult {
  std::string id, text;
  size_t prompt_tokens = 0, gen_tokens = 0;  // gen_tokens counts the stop token when one ended the answer
  double prefill_ms = 0, decode_ms = 0, total_ms = 0;
  bool truncated = false;  // the answer hit the token cap: no stop token ended it
  double mask_ms = -1;     // constrained rows: the host's time choosing allowed tokens; -1 unconstrained
  bool budget_cut = false; // constrained rows: some token was left out only so the answer could close in budget
  std::vector<Override> overridden{};  // constrained rows: the steps whose token was not the model's argmax
  std::vector<TokenRecord> tokens{};   // with token confidence: every generated token, its bytes and confidence
  int passes = -1, drafted = 0;        // with drafts: the row's model passes after the prefill, and drafts run
};
// {"id", "text", "prompt_tokens", "gen_tokens", "prefill_ms", "decode_ms", "total_ms", "truncated"} (and for a
// constrained row "mask_ms", "budget_cut", "overridden": [[start, end], ...] byte spans of text the decoder chose,
// and "overridden_steps": [step, ...] beside them; with token confidence "tokens", tokens_json's; with drafts "passes"
// and "drafted") and a newline; text's
// bytes that are not UTF-8 written as U+FFFD, so every line is valid JSON.
std::string batch_line(const BatchResult& r);
// ,"overridden":[[start,end],...],"overridden_steps":[step,...]
std::string overrides_json(const std::vector<Override>& o);
// ,"tokens":[[start,end,logprob,margin],...]: each generated token's bytes of the text and its confidence (generate.h),
// the numbers to 6 significant digits, a non-finite one as null.
std::string tokens_json(const std::vector<TokenRecord>& r);

// The ids an output file already answers, one per whole line. A last line cut short by a crash (no newline) is cut
// off the file first. Throws on a whole line that is not a result, or an id that `rows` does not hold.
std::set<std::string> resume_ids(const std::string& path, const std::vector<BatchRow>& rows);

// Appends whole lines to a file, each flushed to stable storage (F_FULLFSYNC) before append returns, so the file is
// always the lines written so far, at most the last one torn by a crash.
class LineFile {
 public:
  explicit LineFile(const std::string& path);
  ~LineFile();
  LineFile(const LineFile&) = delete;
  LineFile& operator=(const LineFile&) = delete;
  void append(const std::string& line);

 private:
  std::string path_;
  int fd_ = -1;
};

// The stop tokens: the model's EOS, <|im_end|> and <|endoftext|>, as `generate` stops.
std::vector<int> stop_tokens(const Model& m, const Tokenizer& t);

// What constrains the rows: a row's own schema (compiled by its text), else `fallback`, else nothing; with
// quote_user, the strings at quote keys copy the row's last user turn.
struct BatchConstraints {
  const Schema* fallback = nullptr;
  std::map<std::string, Schema> by_text;
  const VocabTrie* trie = nullptr;
  bool quote_user = false;
};

// Drafts for the constrained rows (generate.h's Draft): the training answers' tables by system prompt, a row drafting
// from its own system turn's (prompt lookup only when the set has none for it).
struct BatchDrafts {
  std::unordered_map<std::string, NgramTable> tables;
  int k = 8;
};

// With drafts, what run_batch's rows took together: the tokens generated, passes run and drafts verified.
struct BatchStats {
  long tokens = 0, passes = 0, drafted = 0;
};

// Greedy decoding of every row not in `done`, in order, from prompts[i] (chat_ids of row i's turns, thinking off as
// the trainer renders it), at most max_tokens each: `emit` gets each result as it finishes. total_ms is the row's
// decoding and its text, from prompt in to the last token out. With confidence, each result carries its tokens. With
// drafts, the constrained rows decode by verified drafts, to the same answers.
BatchStats run_batch(const Model& m, const Tokenizer& t, const std::vector<BatchRow>& rows,
                     const std::vector<std::vector<int>>& prompts, const std::set<std::string>& done, int max_tokens,
                     const std::function<void(const BatchResult&)>& emit, const BatchConstraints* constraints = nullptr,
                     bool confidence = false, const BatchDrafts* drafts = nullptr);
