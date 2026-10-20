#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <mlx/mlx.h>

#include "tokenizer.h"

namespace mx = mlx::core;

struct Example {
  std::vector<int> ids;
  std::vector<uint8_t> loss;  // per id, from render_chat
  std::string id;             // the row's "id", else "line N"
  int prompt_len = 0;         // ids before the last answer: the prompt generation starts from
};

struct Dataset {
  std::vector<Example> examples;
  std::vector<std::string> too_long;   // ids left out: more than max_len tokens
  std::vector<std::string> no_answer;  // ids left out: no token to train on
  std::map<std::string, int> system_prompts;  // sha256 of each kept row's system turn ("none" without one) -> rows
};

// JSONL, one {"messages": [{"role": ..., "content": ...}, ...], "id": optional string} per line, rendered by
// render_chat. Throws naming the file and line on anything else.
Dataset load_examples(const std::string& path, const Tokenizer& t, int max_len);

// sha256 over every example's ids and loss mask, in order: what the rows render to, not where they were read from.
std::string examples_sha256(const std::vector<Example>& ex);

// The first min(n, max) of 0..n-1 shuffled by seed: a fixed subset, in an order whose prefixes are subsets too.
std::vector<int> seeded_subset(size_t n, size_t max, uint64_t seed);

// One epoch's batches as example indices: the examples shuffled by (seed, epoch), sorted by length within buckets
// of 128, packed while (count x longest) stays within batch_tokens, then the batches shuffled. Deterministic: the
// shuffles are Fisher-Yates over std::mt19937_64, whose output the standard fixes.
std::vector<std::vector<int>> plan_epoch(const std::vector<Example>& ex, int batch_tokens, uint64_t seed, int epoch);
// For evaluation: every example once, by length, no shuffle.
std::vector<std::vector<int>> plan_eval(const std::vector<Example>& ex, int batch_tokens);

struct Batch {
  mx::array tokens;     // [B, L] int32, right-padded; padding only ever follows a row's last trained position
  mx::array positions;  // [N] int32, flat (row * L + t): positions whose next token is trained on
  mx::array targets;    // [N] int32, those next tokens
  int real_tokens = 0;  // tokens before padding
};
Batch make_batch(const std::vector<Example>& ex, const std::vector<int>& rows);
