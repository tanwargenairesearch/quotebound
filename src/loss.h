#pragma once
#include "data.h"
#include "model.h"

// Summed cross-entropy, in float32, over a batch's trained positions: hidden(), then logits_at() on 256 positions at
// a time, each chunk under mx::checkpoint so its logits (the vocabulary is 248,320 wide) are recomputed in the
// backward rather than kept. checkpoint is passed to hidden() and decides the chunks' checkpoints too.
mx::array loss_sum(const Model& m, const Batch& b, bool checkpoint = true);
