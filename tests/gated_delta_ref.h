#pragma once
// Holds mlx-lm's gated_delta Metal kernel unchanged (Copyright © 2023 Apple Inc., MIT License; see NOTICE).
// Two test oracles for the gated delta recurrence that src/deltanet.cpp runs through mx::fast::gated_delta_update:
// mlx-lm 0.31.3's own Metal kernel (the one its Qwen3.5 runs) and the recurrence written as plain MLX ops.
#include <mlx/mlx.h>

#include <utility>

namespace mx = mlx::core;

// mlx-lm's gated_delta.py kernel for a per-head scalar gate and no mask, unchanged. One simdgroup of 32
// threads owns one (batch, value head, value row): each thread keeps Dk/32 state entries in registers across
// all T steps, and simd_sum does the two reductions over Dk.
inline const char* kGatedDeltaSource = R"(
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }

        // g: [B, T, Hv]
        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
          if (true) {
            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] * g_[hv_idx];
              kv_mem += state[i] * k_[s_idx];
            }
            kv_mem = simd_sum(kv_mem);

            auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] + k_[s_idx] * delta;
              out += state[i] * q_[s_idx];
            }
            out = simd_sum(out);
            if (thread_index_in_simdgroup == 0) {
              y[dv_idx] = static_cast<InT>(out);
            }
          } else {
            y[dv_idx] = static_cast<InT>(0);
          }
          // Increment data pointers to next time step
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<StT>(state[i]);
        }
    )";

// q, k [B, T, Hk, Dk]; v [B, T, Hv, Dv]; g, beta [B, T, Hv]; state [B, Hv, Dv, Dk] float32.
// Returns y [B, T, Hv, Dv] in q's dtype and the new state.
inline std::pair<mx::array, mx::array> gated_delta_kernel(const mx::array& q, const mx::array& k, const mx::array& v,
                                                          const mx::array& g, const mx::array& beta,
                                                          const mx::array& state) {
  static auto kernel = mx::fast::metal_kernel("gated_delta_step", {"q", "k", "v", "g", "beta", "state_in", "T"},
                                              {"y", "state_out"}, kGatedDeltaSource);
  const int B = k.shape(0), T = k.shape(1), Hk = k.shape(2), Dk = k.shape(3), Hv = v.shape(2), Dv = v.shape(3);
  auto out = kernel({q, k, v, g, beta, state, mx::array(T)}, {{B, T, Hv, Dv}, state.shape()},
                    {q.dtype(), state.dtype()}, {32, Dv, B * Hv}, {32, 4, 1},
                    {{"InT", q.dtype()}, {"StT", state.dtype()}, {"Dk", Dk}, {"Dv", Dv}, {"Hk", Hk}, {"Hv", Hv}},
                    std::nullopt, false, {});
  return {out[0], out[1]};
}

inline std::pair<mx::array, mx::array> gated_delta_ops(const mx::array& q, const mx::array& k, const mx::array& v,
                                                       const mx::array& g, const mx::array& beta, mx::array state) {
  const int B = q.shape(0), T = q.shape(1), Hk = q.shape(2), Hv = v.shape(2);
  mx::array qr = Hv > Hk ? mx::repeat(q, Hv / Hk, 2) : q, kr = Hv > Hk ? mx::repeat(k, Hv / Hk, 2) : k;
  auto at = [&](const mx::array& x, int t) {
    mx::Shape lo(x.ndim(), 0), hi = x.shape();
    lo[1] = t, hi[1] = t + 1;
    return mx::squeeze(mx::slice(x, lo, hi), 1);
  };
  std::vector<mx::array> ys;
  for (int t = 0; t < T; ++t) {
    mx::array qt = at(qr, t), kt = mx::expand_dims(at(kr, t), 2);  // [B,H,Dk], [B,H,1,Dk]
    mx::array gt = mx::reshape(at(g, t), {B, Hv, 1, 1}), bt = mx::expand_dims(at(beta, t), 2);
    state = state * gt;
    mx::array kv_mem = mx::sum(state * kt, -1);                              // [B,H,Dv]
    mx::array delta = (at(v, t) - kv_mem) * bt;                              // [B,H,Dv]
    state = state + kt * mx::expand_dims(delta, 3);
    ys.push_back(mx::astype(mx::sum(state * mx::expand_dims(qt, 2), -1), q.dtype()));  // [B,H,Dv]
  }
  return {mx::stack(ys, 1), state};
}

