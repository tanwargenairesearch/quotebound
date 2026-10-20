#pragma once
#include <utility>
#include <vector>

#include <mlx/mlx.h>

namespace mx = mlx::core;

// The gated delta rule with a backward of its own. The forward is the model's single
// mx::fast::gated_delta_update call in float32, unchanged. The backward first runs the same op chunk by chunk to
// get the state at every chunk start, then a Metal kernel recomputes each chunk's states from its start and runs
// the reverse recurrence: memory O(T/C states + T rows), not the plain-ops vjp's O(T states).
// q, k [B, T, Hk, Dk]; v [B, T, Hv, Dv]; g, beta [B, T, Hv]; state [B, Hv, Dv, Dk]; all float32.
std::pair<mx::array, mx::array> gated_delta_trainable(const mx::array& q, const mx::array& k, const mx::array& v,
                                                      const mx::array& g, const mx::array& beta,
                                                      const mx::array& state);

// The backward alone: cotangents of (q, k, v, g, beta, state) given those of y and the final state.
std::vector<mx::array> gated_delta_vjp(const std::vector<mx::array>& primals, const mx::array& dy,
                                       const mx::array& dstate);
