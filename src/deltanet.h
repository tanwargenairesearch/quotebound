#pragma once
#include <utility>

#include "model.h"

// Qwen3.5's Gated DeltaNet token mixer, step for step as mlx-lm's qwen3_5.GatedDeltaNet runs it in eval mode,
// except for the recurrence primitive (below). With train, the recurrence's gradient is gated_delta_trainable's.
// With rows, as RowStates says: the conv and the recurrence one token at a time, each token's conv window and state
// kept at rows->conv[layer], rows->state[layer].
mx::array deltanet(const DeltaNet& w, const Config& c, const mx::array& x, LayerCache* cache, bool train = false,
                   RowStates* rows = nullptr, int layer = 0);

// The gated delta rule over T steps, as the model runs it: mx::fast::gated_delta_update with every input in
// float32. q, k [B, T, Hk, Dk]; v [B, T, Hv, Dv]; g (float32), beta [B, T, Hv]; state [B, Hv, Dv, Dk] float32.
// Returns y [B, T, Hv, Dv] in q's dtype and the new state. With train, the same forward with
// gated_delta_trainable's backward.
std::pair<mx::array, mx::array> gated_delta(const mx::array& q, const mx::array& k, const mx::array& v,
                                            const mx::array& g, const mx::array& beta, const mx::array& state,
                                            bool train = false);
