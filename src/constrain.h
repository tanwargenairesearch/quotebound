#pragma once
// Schema-constrained greedy decoding: at each step, only the tokens whose bytes keep the answer a prefix of compact
// JSON (json.dumps(v, ensure_ascii=False, separators=(",", ":"))) the schema accepts, and that can still close
// within the token budget. The schema becomes a byte-level pushdown automaton; the vocabulary is a trie of token
// bytes, walked from the automaton's state with every subtree it refuses pruned.
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "schema.h"
#include "tokenizer.h"

// What a free string's body allows from a clean state (no character, escape or \u in progress), for one character
// class, worked out once over the whole trie: `allowed` then reads it instead of walking the trie at such steps.
// - body: the tokens whose every byte stays inside the string (no unescaped quote), sorted by id, with what each
//   leaves: the characters it completes and the bytes still owed by one in progress. Whether one fits maxLength and
//   the budget depends only on those and the string's state, so they are checked per step, all at once when no
//   token can fail them.
// - closes: each trie node reached by a body-valid prefix then an unescaped quote, the string's close: the walk
//   resumes there through the whole automaton, as it would.
struct FreeTable {
  std::vector<int32_t> ids;
  std::vector<int16_t> complete;
  std::vector<uint8_t> pending;
  int max_started = 0, max_pending = 0;  // over the body tokens: characters begun, bytes owed
  int min_char_bytes = 1;                // the class's cheapest character
  struct Close {
    int32_t node, level, chars;  // the quote's trie node, its depth, the characters its prefix completes
    std::string prefix;          // the bytes before the quote
  };
  std::vector<Close> closes;
};

// Every ordinary token's bytes as a trie (added tokens such as <|im_end|> left out: they are not text).
struct VocabTrie {
  std::vector<int32_t> first_child, next_sibling;  // per node, -1 for none; node 0 is the root
  std::vector<uint8_t> byte;                       // the byte on the edge into the node
  std::vector<int32_t> tok_begin, tok_end;         // the tokens whose bytes end at the node: ids[begin, end)
  std::vector<int32_t> ids;
  int max_depth = 0;                               // the longest token, in bytes
  // FreeTables by character class, built on first use and shared by every answer decoded over this trie.
  struct Tables {
    std::mutex mu;
    std::map<std::string, std::shared_ptr<const FreeTable>> by_class;
  };
  std::shared_ptr<Tables> free = std::make_shared<Tables>();
};
VocabTrie build_trie(const Tokenizer& t);

// The text quotes are copied from, as json.dumps writes it inside a string. A quote is the JSON text of a stretch of
// it that stands as whole words (ASCII letter case aside: not inside a word, and a number not inside a wider
// one), with no white space at either end.
struct QuoteText {
  std::string e;                 // the text's JSON string bytes, ASCII letters of the text in lower case
  std::vector<int32_t> sa;       // byte offsets of e where a quote may start, sorted by e's bytes from there
  std::vector<int32_t> next_end; // per offset of e (and e.size()): the first offset >= it where a quote may end
  std::vector<uint8_t> end_ok;   // per offset of e (and e.size()): a quote may end there
  std::vector<int32_t> char_at;  // per offset (and e.size()): the index of the character it belongs to
  std::vector<int32_t> byte_of;  // per character index (and the count): its offset in e
};
QuoteText quote_text(std::string_view text);

// One answer's automaton state: a few configurations (anyOf alternatives) each a stack of open values.
struct Frame {
  uint8_t kind = 0, phase = 0, need = 0, lead = 0, u = 0;
  int16_t shape = -1, key = -1, lo = 0, hi = 0, pos = 0, count = 0;
  int32_t chars = 0, qlo = 0, qhi = 0, qlen = 0;
  char32_t cp = 0;
  uint64_t seen = 0, used = 0, hash = 0;
};
constexpr int kMaxDepth = 12;
struct Stack {
  int depth = 0;
  bool done = false;
  int8_t nhash = 0;              // the open unique string array's items so far
  uint64_t hashes[8] = {};
  Frame f[kMaxDepth];
};

// A step whose token the constraint changed from the model's own argmax: the step, and the answer's bytes [start, end)
// the token it took wrote (empty for a stop). What is written there is the decoder's choice, not the model's.
struct Override {
  int step = 0, start = 0, end = 0;
  bool operator==(const Override&) const = default;
};

class Constraint {
 public:
  // stop: the ids that end an answer, allowed only once the JSON is complete. quotes: the text quote strings are
  // copied from, or null for none (quote strings are then free strings).
  Constraint(const Schema& s, const Tokenizer& tok, const VocabTrie& trie, std::vector<int> stop,
             const QuoteText* quotes);
  // The ids that may come next with `remaining` tokens left, this one included, sorted: tokens that keep the bytes a
  // prefix of a valid answer that can still be closed and stopped within the budget, and the stop ids once the
  // answer is complete. Empty only if the schema admits no answer within the budget.
  std::vector<int> allowed(int remaining);
  // Advances by one token; throws if its bytes leave the schema.
  void accept(int id);
  bool complete() const;
  // The fewest bytes that complete the answer from here (0 when complete).
  int min_close() const;
  // The answer's bytes so far.
  size_t bytes() const { return bytes_; }
  bool budget_cut = false;  // some step left a token out only for the budget
  // Read free strings' bodies from the trie's FreeTable (the same tokens, fewer bytes walked); off, every step walks
  // the trie, the reference the table is tested against.
  bool free_tables = true;

 private:
  const Schema& s_;
  const Tokenizer& tok_;
  const VocabTrie& trie_;
  std::vector<int> stop_;
  const QuoteText* q_;
  std::vector<int> mv_node_, mv_shape_;  // the fewest bytes of a whole value of each node and shape
  std::vector<Stack> state_;
  std::vector<std::vector<Stack>> bufs_;
  std::vector<int> out_;
  int remaining_ = 0;
  size_t bytes_ = 0;

  int mv_of_node(int n);
  int mv_of_shape(int sh);
  bool push_value(const Stack& c, int node, std::vector<Stack>& out) const;
  void feed(Stack c, uint8_t b, std::vector<Stack>& out) const;
  bool pop_value(Stack& c, int lit) const;
  bool quote_alive(const Frame& f, const Shape& sh) const;
  int quote_cost(const Frame& f, const Shape& sh) const;
  bool quote_closable(const Frame& f) const;
  int close_cost(const Shape& sh, uint64_t seen, bool first) const;
  int frame_cost(const Stack& c, int i) const;
  int cost(const Stack& c) const;
  bool lazy_complete(const Stack& c) const;
  int min_cost(const std::vector<Stack>& st) const;
  bool free_string_top(const std::vector<Stack>& st) const;
  void walk(int node, int level);
  void walk_free(int node, int level, const Stack& base, Frame fs, int rest);
  void take_tokens(int node, int cost);
  const FreeTable& free_table(const Shape& sh);
  bool from_table(const Stack& base, int rest);
  std::vector<int> body_;      // the FreeTable's body tokens a step allows, when it read one
  bool body_all_ = false;      // ... or all of them (table_->ids)
  const FreeTable* table_ = nullptr;
  int stop_in_table_ = -1;     // -1 unknown; 1 when a stop id is an ordinary token the table holds
};
