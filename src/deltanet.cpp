// Follows mlx-lm's Qwen3 / Qwen3.5 implementation for token-for-token parity (Copyright © 2023 Apple Inc., MIT
// License; see NOTICE).
#include "deltanet.h"

#include <cmath>
#include <optional>

#include "delta_grad.h"
#include "exact.h"

namespace {

using Fn = std::function<std::vector<mx::array>(const std::vector<mx::array>&)>;

// The three small graphs mlx-lm compiles (shapeless), compiled the same way so they fuse the same way.
const Fn& silu() {
  static const Fn f = mx::compile(
      [](const std::vector<mx::array>& in) { return std::vector<mx::array>{in[0] * mx::sigmoid(in[0])}; }, true);
  return f;
}

// g = exp(-exp(A_log) * softplus(a + dt_bias)), softplus(x) = logaddexp(x, 0).
const Fn& compute_g() {
  static const Fn f = mx::compile(
      [](const std::vector<mx::array>& in) {
        const mx::array &a_log = in[0], &a = in[1], &dt_bias = in[2];
        mx::array x = a + dt_bias;
        mx::array softplus = mx::logaddexp(x, mx::array(0.0f, x.dtype()));
        return std::vector<mx::array>{mx::exp(-mx::exp(mx::astype(a_log, mx::float32)) * softplus)};
      },
      true);
  return f;
}

// Qwen3NextRMSNormGated's tail: (silu(z) * normed) in float32, back to the input dtype.
const Fn& precise_swiglu() {
  static const Fn f = mx::compile(
      [](const std::vector<mx::array>& in) {
        const mx::array &h = in[0], &z = in[1], &x = in[2];
        mx::array zf = mx::astype(z, mx::float32);
        return std::vector<mx::array>{mx::astype((zf * mx::sigmoid(zf)) * mx::astype(x, mx::float32), h.dtype())};
      },
      true);
  return f;
}

}  // namespace

std::pair<mx::array, mx::array> gated_delta(const mx::array& q, const mx::array& k, const mx::array& v,
                                            const mx::array& g, const mx::array& beta, const mx::array& state,
                                            bool train) {
  // Fed bf16, the fast op also rounds g to bf16: with slow decay (g near 1) its state drifts by percents over a few
  // hundred steps (test_deltanet prints it). In float32 it agrees with mlx-lm's kernel to float32 rounding.
  auto f32 = [](const mx::array& x) { return mx::astype(x, mx::float32); };
  if (train) {
    auto [y, s] = gated_delta_trainable(f32(q), f32(k), f32(v), f32(g), f32(beta), state);
    return {mx::astype(y, q.dtype()), s};
  }
  auto r = mx::fast::gated_delta_update(f32(q), f32(k), f32(v), f32(g), f32(beta), state);
  return {mx::astype(r[0], q.dtype()), r[1]};
}

mx::array deltanet(const DeltaNet& w, const Config& c, const mx::array& x, LayerCache* cache, bool train,
                   RowStates* rows, int layer) {
  const int B = x.shape(0), S = x.shape(1), K = c.conv_kernel;
  const int Hk = c.lin_k_heads, Hv = c.lin_v_heads, Dk = c.lin_k_dim, Dv = c.lin_v_dim;
  const int key_dim = Hk * Dk, value_dim = Hv * Dv, conv_dim = 2 * key_dim + value_dim;
  auto lin = [&](const mx::array& in, const Proj& p) { return rows ? linear_rows(in, p) : linear(in, p); };

  mx::array qkv = lin(x, w.in_qkv);
  mx::array z = mx::reshape(lin(x, w.in_z), {B, S, Hv, Dv});
  mx::array b = lin(x, w.in_b), a = lin(x, w.in_a);

  // Depthwise causal conv over time: the previous K-1 inputs (zeros at the start) lead the new ones.
  mx::array conv_state = cache && cache->conv ? *cache->conv : mx::zeros({B, K - 1, conv_dim}, x.dtype());
  mx::array conv_in = mx::concatenate({conv_state, qkv}, 1);
  if (cache) cache->conv = mx::contiguous(mx::slice(conv_in, {0, S, 0}, {B, S + K - 1, conv_dim}));
  std::optional<mx::array> conv_rows;
  if (rows) {  // each token's window of K inputs on its own, as a decode step convolves it
    std::vector<mx::array> outs;
    for (int j = 0; j < S; ++j) {
      const mx::array window = mx::slice(conv_in, {0, j, 0}, {B, j + K, conv_dim});
      outs.push_back(silu()({mx::conv1d(window, w.conv, 1, 0, 1, conv_dim)})[0]);
      rows->conv[layer].push_back(mx::contiguous(mx::slice(conv_in, {0, j + 1, 0}, {B, j + K, conv_dim})));
    }
    conv_rows = S == 1 ? outs[0] : mx::concatenate(outs, 1);
  }
  mx::array conv_out = conv_rows ? *conv_rows : silu()({mx::conv1d(conv_in, w.conv, 1, 0, 1, conv_dim)})[0];

  auto parts = mx::split(conv_out, mx::Shape{key_dim, 2 * key_dim}, -1);
  mx::array q = mx::reshape(parts[0], {B, S, Hk, Dk});
  mx::array k = mx::reshape(parts[1], {B, S, Hk, Dk});
  mx::array v = mx::reshape(parts[2], {B, S, Hv, Dv});
  // Scalars take the activation dtype, as a Python float does in mlx-lm: 1/Dk on q, 1/sqrt(Dk) on k.
  const double inv = std::pow(double(Dk), -0.5);
  q = mx::array(float(inv * inv), q.dtype()) * mx::fast::rms_norm(q, std::nullopt, 1e-6f);
  k = mx::array(float(inv), k.dtype()) * mx::fast::rms_norm(k, std::nullopt, 1e-6f);

  mx::array beta = mx::sigmoid(b);
  mx::array g = compute_g()({w.a_log, a, w.dt_bias})[0];
  mx::array state = cache && cache->state ? *cache->state : mx::zeros({B, Hv, Dv, Dk}, mx::float32);
  std::optional<mx::array> y_rows;
  if (rows) {  // one step at a time, each step's state kept
    std::vector<mx::array> ys;
    for (int j = 0; j < S; ++j) {
      auto at = [&](const mx::array& t) {
        mx::Shape lo(t.ndim(), 0), hi = t.shape();
        lo[1] = j, hi[1] = j + 1;
        return mx::slice(t, lo, hi);
      };
      auto [yj, sj] = gated_delta(at(q), at(k), at(v), at(g), at(beta), state, false);
      ys.push_back(yj);
      rows->state[layer].push_back(sj);
      state = sj;
    }
    y_rows = S == 1 ? ys[0] : mx::concatenate(ys, 1);
  }
  auto [y, new_state] = y_rows ? std::pair<mx::array, mx::array>{*y_rows, state}
                               : gated_delta(q, k, v, g, beta, state, train);
  if (cache) cache->state = new_state;

  mx::array o = precise_swiglu()({y, z, mx::fast::rms_norm(y, w.norm, c.eps)})[0];
  return lin(mx::reshape(o, {B, S, value_dim}), w.out);
}
