// The gated delta recurrence three ways on random inputs: the model's path (mx::fast::gated_delta_update fed
// float32), mlx-lm's Metal kernel and plain MLX ops. Then the fast op's gradient, and gated_delta_trainable's
// (chunk checkpoints and a Metal backward kernel), against autodiff through the plain ops. Shapes are mostly Qwen3.5's
// (16 key heads, 32 value heads, Dk = Dv = 128): with fewer than 16 key heads the fast op falls back to plain ops and
// its kernel would go untested.
#include <cmath>

#include "../src/delta_grad.h"
#include "../src/deltanet.h"
#include "check.h"
#include "gated_delta_ref.h"

static float max_abs(const mx::array& a) { return mx::max(mx::abs(mx::astype(a, mx::float32))).item<float>(); }
static float max_diff(const mx::array& a, const mx::array& b) {
  return max_abs(mx::astype(a, mx::float32) - mx::astype(b, mx::float32));
}

static mx::array steps(const mx::array& x, int a, int b) {
  mx::Shape lo(x.ndim(), 0), hi = x.shape();
  lo[1] = a, hi[1] = b;
  return mx::slice(x, lo, hi);
}

// g uniform in [g_lo, 1): g_lo = 0.99 is slow decay, where the state carries hundreds of steps of history.
static void agree(mx::Dtype dt, int T, float g_lo) {
  const int B = 2, Hk = 16, Hv = 32, Dk = 128, Dv = 128;
  mx::random::seed(T + (dt == mx::float32 ? 0 : 1000));
  // q and k as the model feeds them: unit RMS, scaled by 1/Dk and 1/sqrt(Dk).
  auto unit = [&](mx::Shape s, float scale) {
    return mx::astype(mx::fast::rms_norm(mx::random::normal(s), std::nullopt, 1e-6f) * scale, dt);
  };
  mx::array q = unit({B, T, Hk, Dk}, 1.f / Dk), k = unit({B, T, Hk, Dk}, 1.f / std::sqrt(float(Dk)));
  mx::array v = mx::astype(mx::random::normal({B, T, Hv, Dv}), dt);
  mx::array g = mx::random::uniform(g_lo, 1.f, {B, T, Hv});
  mx::array beta = mx::astype(mx::sigmoid(mx::random::normal({B, T, Hv})), dt);
  mx::array state = mx::random::normal({B, Hv, Dv, Dk}, 0.f, 0.1f);

  auto [yf, sf] = gated_delta(q, k, v, g, beta, state);
  auto [yk, sk] = gated_delta_kernel(q, k, v, g, beta, state);
  auto [yo, so] = gated_delta_ops(q, k, v, g, beta, state);
  const float y_scale = max_abs(yo), s_scale = max_abs(so);
  // Float32 differs only in summation order; bf16 outputs may also round to the neighbouring bf16 value.
  const float y_tol = dt == mx::float32 ? 1e-5f * y_scale : std::ldexp(1.0f, std::ilogb(y_scale) - 7);
  const float s_tol = 1e-5f * s_scale;
  const float fy = max_diff(yf, yo), fs = max_diff(sf, so), ky = max_diff(yk, yo), ks = max_diff(sk, so);

  // The model's decode carries the state between calls: 20 + rest steps against one call.
  auto [y1, s1] = gated_delta(steps(q, 0, 20), steps(k, 0, 20), steps(v, 0, 20), steps(g, 0, 20),
                              steps(beta, 0, 20), state);
  auto [y2, s2] = gated_delta(steps(q, 20, T), steps(k, 20, T), steps(v, 20, T), steps(g, 20, T),
                              steps(beta, 20, T), s1);
  const float split_y = max_diff(mx::concatenate({y1, y2}, 1), yo), split_s = max_diff(s2, so);

  // For the record, not used: the fast op fed the activations' own dtype, which rounds g to it too.
  std::string raw;
  if (dt != mx::float32)
    raw = "; fed bf16 instead (not used): max |dstate| " +
          std::to_string(max_diff(mx::fast::gated_delta_update(q, k, v, mx::astype(g, dt), beta, state)[1], so));

  std::printf("gated delta %s T=%d g in [%.2f, 1): vs ops, fast op max |dy| %.3g, |dstate| %.3g; mlx-lm kernel %.3g, "
              "%.3g; 20+%d split %.3g, %.3g (tolerances %.3g, %.3g; max|y| %.3f, max|state| %.2f)%s\n",
              dt == mx::float32 ? "float32" : "bfloat16", T, g_lo, fy, fs, ky, ks, T - 20, split_y, split_s, y_tol,
              s_tol, y_scale, s_scale, raw.c_str());
  CHECK(fy <= y_tol && fs <= s_tol, "fast op and ops recurrence differ");
  CHECK(ky <= y_tol && ks <= s_tol, "mlx-lm kernel and ops recurrence differ");
  CHECK(split_y <= y_tol && split_s <= s_tol, "carrying the state across calls changed the result");
}

// d/d(q, k, v, g, beta, state) of sum(y * cy) + sum(state_out * cs), float32.
static void gradient() {
  const int B = 1, T = 19, Hk = 16, Hv = 32, Dk = 128, Dv = 128;
  mx::random::seed(3);
  auto unit = [&](mx::Shape s, float scale) {
    return mx::fast::rms_norm(mx::random::normal(s), std::nullopt, 1e-6f) * scale;
  };
  std::vector<mx::array> in{unit({B, T, Hk, Dk}, 1.f / Dk), unit({B, T, Hk, Dk}, 1.f / std::sqrt(float(Dk))),
                            mx::random::normal({B, T, Hv, Dv}), mx::random::uniform(0.5f, 1.f, {B, T, Hv}),
                            mx::random::uniform(0.f, 1.f, {B, T, Hv}), mx::random::normal({B, Hv, Dv, Dk}, 0.f, 0.1f)};
  mx::array cy = mx::random::normal({B, T, Hv, Dv}), cs = mx::random::normal({B, Hv, Dv, Dk});
  auto loss = [&](bool fast) {
    return [&, fast](const std::vector<mx::array>& x) {
      auto [y, s] = fast ? gated_delta(x[0], x[1], x[2], x[3], x[4], x[5])
                         : gated_delta_ops(x[0], x[1], x[2], x[3], x[4], x[5]);
      return std::vector<mx::array>{mx::sum(y * cy) + mx::sum(s * cs)};
    };
  };
  auto [lf, gf] = mx::vjp(loss(true), in, {mx::array(1.0f)});
  auto [lo, go] = mx::vjp(loss(false), in, {mx::array(1.0f)});
  const char* names[] = {"q", "k", "v", "g", "beta", "state"};
  std::string line;
  float worst = 0;
  for (int i = 0; i < 6; ++i) {
    float d = max_diff(gf[i], go[i]), scale = max_abs(go[i]);
    worst = std::max(worst, d / scale);
    char buf[96];
    std::snprintf(buf, sizeof buf, "%s%s %.3g of %.3g", i ? ", " : "", names[i], d, scale);
    line += buf;
  }
  std::printf("gated delta gradient, fast op vs autodiff through ops (float32): max |d grad| %s\n", line.c_str());
  CHECK(worst <= 1e-5f, "gradients differ (worst %.3g of the gradient's max)", worst);
}

// gated_delta_trainable's gradient against autodiff through the plain ops, float32, with a nonzero initial state and
// cotangents on y and the final state. Chunks are 64 steps: T below, at and across a chunk, and a last chunk of one.
static void trainable_gradient(int B, int T, int Hk, int Hv, int Dk, int Dv, float g_lo) {
  mx::random::seed(100 + T);
  auto unit = [&](mx::Shape s, float scale) {
    return mx::fast::rms_norm(mx::random::normal(s), std::nullopt, 1e-6f) * scale;
  };
  std::vector<mx::array> in{unit({B, T, Hk, Dk}, 1.f / Dk), unit({B, T, Hk, Dk}, 1.f / std::sqrt(float(Dk))),
                            mx::random::normal({B, T, Hv, Dv}), mx::random::uniform(g_lo, 1.f, {B, T, Hv}),
                            mx::random::uniform(0.f, 1.f, {B, T, Hv}), mx::random::normal({B, Hv, Dv, Dk}, 0.f, 0.1f)};
  mx::array cy = mx::random::normal({B, T, Hv, Dv}), cs = mx::random::normal({B, Hv, Dv, Dk});
  auto loss = [&](bool trainable) {
    return [&, trainable](const std::vector<mx::array>& x) {
      auto [y, s] = trainable ? gated_delta_trainable(x[0], x[1], x[2], x[3], x[4], x[5])
                              : gated_delta_ops(x[0], x[1], x[2], x[3], x[4], x[5]);
      return std::vector<mx::array>{mx::sum(y * cy) + mx::sum(s * cs)};
    };
  };
  auto [lt, gt] = mx::vjp(loss(true), in, {mx::array(1.0f)});
  auto [lo, go] = mx::vjp(loss(false), in, {mx::array(1.0f)});
  const char* names[] = {"q", "k", "v", "g", "beta", "state"};
  std::string line;
  float worst = 0;
  for (int i = 0; i < 6; ++i) {
    float d = max_diff(gt[i], go[i]), scale = max_abs(go[i]);
    worst = std::max(worst, d / scale);
    char buf[96];
    std::snprintf(buf, sizeof buf, "%s%s %.2g of %.3g", i ? ", " : "", names[i], d, scale);
    line += buf;
  }
  std::printf("trainable gradient B=%d T=%d Hk=%d Hv=%d Dk=%d Dv=%d g in [%.2f, 1) vs autodiff through ops: "
              "max |d grad| %s (worst %.2g of the gradient's max)\n",
              B, T, Hk, Hv, Dk, Dv, g_lo, line.c_str(), worst);
  CHECK(worst <= 1e-5f, "gradients differ (worst %.3g of the gradient's max)", worst);
}

int main() {
  for (mx::Dtype dt : {mx::float32, mx::bfloat16}) {
    agree(dt, 37, 0.5f);
    agree(dt, 300, 0.99f);
  }
  gradient();
  trainable_gradient(1, 19, 16, 32, 128, 128, 0.5f);
  trainable_gradient(1, 64, 16, 32, 128, 128, 0.9f);
  trainable_gradient(2, 150, 16, 32, 128, 128, 0.99f);
  trainable_gradient(1, 129, 16, 32, 128, 128, 0.9f);
  trainable_gradient(1, 70, 2, 8, 32, 32, 0.9f);
  trainable_gradient(2, 33, 4, 4, 64, 64, 0.99f);
  // Fewer threads than the row reduction's 2 Dk columns (4 rows of 32 lanes for 256 columns).
  trainable_gradient(1, 70, 2, 4, 128, 4, 0.9f);
  // 32 rows of Dk 256 would need 64 KB of threadgroup memory, past Metal's 32 KB.
  trainable_gradient(1, 40, 2, 4, 256, 64, 0.9f);
  return 0;
}
