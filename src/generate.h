#pragma once
#include <functional>
#include <vector>

#include "constrain.h"
#include "draft.h"
#include "model.h"
#include "tokenizer.h"

// One generated token under the model's own distribution over the whole vocabulary, before any mask: the token's
// log-probability (log-softmax of the full logits, at the token taken) and the margin, the top logit less the second.
struct TokenConfidence {
  float logprob = 0, margin = 0;
  bool operator==(const TokenConfidence&) const = default;
};

// Exact speculative decoding, under a constraint and without confidence: after each token taken, up to k more drafted
// at no model cost (draft.h: the one token the constraint allows, else the drafter's guess when the constraint allows
// it) and verified in one pass of exact rows (exact.h). The tokens taken, the overrides and the budget cut are the
// plain decoder's, token for token.
struct Draft {
  Drafter drafter;
  int k = 8;  // drafts a pass at most, 1..kMaxRows - 1
};

struct Generation {
  std::vector<int> ids;      // generated tokens, the stop token included if one was hit
  double prefill_ms = 0;     // prompt in, first token out
  double decode_ms = 0;      // first token out to last token out
  double mask_ms = 0;        // constrained: the host's time choosing the allowed tokens, summed over the steps
  std::vector<Override> overridden;  // constrained: each step whose token was not the model's argmax, in order
  std::vector<TokenConfidence> confidence;  // with confidence on: one per id, in order
  int passes = 0, drafted = 0;  // with a draft: model passes after the prefill, and the drafts they ran
};

// Runs the prompt into the cache `chunk` tokens at a time, evaluating the cache after each chunk so no chunk's
// activations outlive it, and returns the last position's float32 logits [1, 1, vocab], not yet evaluated.
mx::array prefill(const Model& m, const std::vector<int>& prompt, KVCache& cache, int chunk = 2048);

// Greedy decoding: prefill, then one token per step. Throws if a prompt id is outside the model's vocabulary or
// prompt + max_tokens exceeds max_position_embeddings. With a constraint, each step takes the highest logit among
// the tokens it allows (constrain.h), and the answer ends only when the constraint is complete: its stop ids should
// be `stop`. The allowed set is worked out on the host while the GPU runs the step's forward. Every step whose
// allowed token is not the argmax over the whole vocabulary is recorded in `overridden`. With confidence, each step's
// TokenConfidence is recorded too: queued beside the step and read at the end, so the decoder never waits for it, and
// the tokens taken are the same as without it.
// With a draft and a constraint, the steps are verified drafts (Draft); a draft with confidence is refused, and without
// a constraint it is not used.
Generation greedy(const Model& m, const std::vector<int>& prompt, int max_tokens, const std::vector<int>& stop,
                  const std::function<void(int)>& on_token = {}, Constraint* constraint = nullptr,
                  bool confidence = false, const Draft* draft = nullptr);

// The allowed id with the highest logit (the lowest id on a tie, as argmax over all of them takes): logits is one
// step's [.., vocab], allow sorted and not empty.
int pick_allowed(const mx::array& logits, const std::vector<int>& allow);

// A greedy answer as text: the stop token that ended it dropped (stopped), an id past the tokenizer's table written as
// "<id N>" (the vocabulary is padded past it, and an undertrained model can emit one), and an answer cut at its token
// cap without the UTF-8 sequence it ends inside.
std::string generation_text(const Tokenizer& t, std::vector<int> ids, bool stopped);

// A generated token as the output records it: the bytes [start, end) of generation_text's text it wrote (empty for the
// stop token that ended the answer; cut to the text where the text drops a split UTF-8 sequence), and its confidence.
struct TokenRecord {
  int start = 0, end = 0;
  float logprob = 0, margin = 0;
};
// One record per id of g that has a confidence (all of them when greedy ran with confidence on).
std::vector<TokenRecord> token_records(const Tokenizer& t, const Generation& g, bool stopped);
