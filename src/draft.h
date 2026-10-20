#pragma once
// Free drafts for exact speculative decoding: guesses at the next tokens that cost no model call. Two sources, both
// read from text the decoder already holds:
// - prompt lookup: the answer's quotes are copied from the user's message, so the tokens that followed the latest
//   earlier occurrence of the context's last n tokens (prompt and answer so far) are a guess at what follows now;
// - an n-gram table of training answers: the most frequent next token after the last n tokens, over the assistant
//   turns of a training set (the frames' keys, punctuation and enum values), one table per system prompt.
// A guess is only ever verified against the model's own greedy choice (generate.h), so a wrong one costs a little
// verify work, never a different answer.
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "tokenizer.h"

// Next-token counts after every context of 1..order tokens in a set of token sequences; next() is the most frequent
// next token after the longest context the table holds (the lowest id on a tie), with that context's length.
class NgramTable {
 public:
  explicit NgramTable(int order = 4) : order_(order), next_(size_t(order)) {}
  void add(const std::vector<int>& seq);
  // Fills `best` from the counts; call once after the last add.
  void finish();
  // ctx[0, n): the token after it, and the length of the context it was found under (0, -1 for none).
  std::pair<int, int> next(const int* ctx, int n) const;
  int order() const { return order_; }
  size_t contexts() const;

 private:
  int order_;
  std::vector<std::unordered_map<uint64_t, std::unordered_map<int, int>>> next_;  // per length - 1
  std::vector<std::unordered_map<uint64_t, int>> best_;
};

// The tables of a training set (JSONL rows {"messages": [system?, user, assistant...]}): every assistant turn's
// content, encoded as the trainer renders it, added to the table of the row's system prompt (empty for none).
std::unordered_map<std::string, NgramTable> training_tables(const Tokenizer& t, const std::string& path, int order = 4);

// The latest earlier occurrence of ctx's last n tokens (for n from max_n down to min_n): the token that followed it
// and n; {-1, 0} when none.
std::pair<int, int> prompt_lookup(const std::vector<int>& ctx, int max_n = 3, int min_n = 1);

// One draft token after ctx among `allow` (the constraint's allowed ids there, sorted): the only one when one is allowed,
// else the table's guess when allowed, else the prompt lookup's when allowed; -1 for none. Measured on a JSON
// extraction task's dev replies (39,182 tokens): 78.9% of tokens guessed, against 77.8% for the table alone and 32.2% for the
// lookup alone.
struct Drafter {
  const NgramTable* table = nullptr;  // may be null: prompt lookup only
  int lookup_max = 3;
  int guess(const std::vector<int>& ctx, const std::vector<int>& allow) const;
};
