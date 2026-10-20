// KV cache and causal mask on the small random-weight models of tiny.h, in float32 and bfloat16; and quantized weights
// on them (quantize_weights).
#include <algorithm>
#include <cmath>
#include <string>
#include <utility>

#include "../src/generate.h"
#include "../src/lora.h"
#include "../src/model.h"
#include "check.h"
#include "tiny.h"

static mx::array rows(const mx::array& x, int a, int b) { return mx::slice(x, {0, a, 0}, {1, b, x.shape(2)}); }
static float max_diff(const mx::array& a, const mx::array& b) { return mx::max(mx::abs(a - b)).item<float>(); }

// tol_ulps = 0: absolute tolerance tol. Otherwise tol bf16 ulps of the largest |logit| (bf16 keeps 8 significant
// bits, so the spacing at x is 2^(floor(log2 x) - 7)).
static void run(bool q35, mx::Dtype dt, float tol, bool tol_ulps) {
  std::string name = std::string(q35 ? "qwen3.5" : "qwen3") + (dt == mx::float32 ? " float32" : " bfloat16");
  const int T = 300, J = 137;  // T crosses the cache's 256-row growth step
  Model m = tiny(dt, q35);
  mx::array toks = mx::random::randint(0, m.cfg.vocab, {1, T}, mx::int32);
  mx::array full = forward(m, toks);
  mx::eval(full);
  float scale = mx::max(mx::abs(full)).item<float>();
  if (tol_ulps) tol *= std::ldexp(1.0f, std::ilogb(scale) - 7);

  // One token at a time from an empty cache, compared with the full forward at every position.
  KVCache c1;
  float inc = 0;
  int top1_diff = 0;
  for (int t = 0; t < T; ++t) {
    mx::array lt = forward(m, mx::slice(toks, {0, t}, {1, t + 1}), &c1);
    mx::array ft = rows(full, t, t + 1);
    inc = std::max(inc, max_diff(lt, ft));
    top1_diff += mx::argmax(lt, -1).item<uint32_t>() != mx::argmax(ft, -1).item<uint32_t>();
  }

  // Prefill 250, a 7-token chunk on top of the cache (causal with fewer queries than keys), then single tokens.
  KVCache c2;
  float chunk = max_diff(forward(m, mx::slice(toks, {0, 0}, {1, 250}), &c2), rows(full, 0, 250));
  chunk = std::max(chunk, max_diff(forward(m, mx::slice(toks, {0, 250}, {1, 257}), &c2), rows(full, 250, 257)));
  for (int t = 257; t < T; ++t)
    chunk = std::max(chunk, max_diff(forward(m, mx::slice(toks, {0, t}, {1, t + 1}), &c2), rows(full, t, t + 1)));
  KVCache c3;
  float last = max_diff(forward(m, toks, &c3, true), rows(full, T - 1, T));
  // greedy's prefill in 64-token chunks, the cache evaluated between them.
  const int32_t* tp = toks.data<int32_t>();
  KVCache c4;
  float chunked_prefill = max_diff(prefill(m, std::vector<int>(tp, tp + T), c4, 64), rows(full, T - 1, T));

  // Changing token J must leave every earlier position's logits untouched and move the later ones.
  mx::array changed = mx::remainder(mx::slice(toks, {0, J}, {1, J + 1}) + 1, mx::array(m.cfg.vocab));
  mx::array toks2 = mx::slice_update(toks, changed, {0, J}, {1, J + 1});
  mx::array full2 = forward(m, toks2);
  float before = max_diff(rows(full2, 0, J), rows(full, 0, J));
  float at = max_diff(rows(full2, J, J + 1), rows(full, J, J + 1));

  std::printf(
      "tiny %s (max|logit| %.2f, tolerance %g): incremental vs full max diff %.3g over %d positions "
      "(top-1 differs at %d); chunked %.3g; last-only %.3g; prefill in 64-token chunks %.3g; causal: positions < %d "
      "moved %.3g, position %d moved %.3g\n",
      name.c_str(), scale, tol, inc, T, top1_diff, chunk, last, chunked_prefill, J, before, J, at);
  CHECK(inc <= tol && chunk <= tol && last <= tol && chunked_prefill <= tol,
        "cache path differs from the full forward (%s)", name.c_str());
  CHECK(before == 0.0f, "a later token changed earlier logits (%s)", name.c_str());
  CHECK(at > 1e-3f, "changing a token did not change its own logits (%s)", name.c_str());
}

// 8-bit weights on tiny Qwen3.5 (bf16; groups of 32, the tiny models' 96-wide inputs not dividing into 64): the
// projections quantized are all but in_proj_b and in_proj_a, which keep their bf16 weights, plus the head; every
// quantized projection and the head run their own dequantized weight (within 4 bf16 ulps of the largest output: the
// kernel dequantizes in float32 where a dequantized bf16 weight is rounded); the whole model's logits sit nearer the
// bf16 model holding those dequantized weights than the unquantized model (so forward reads the quantized weights);
// and a projection with an adapter is refused. How far the logits move from the unquantized model's is printed, not
// judged: a random model amplifies any perturbation, so it says nothing of a trained one.
static void quantized() {
  const Model base = tiny(mx::bfloat16, true, 4);
  Model q = tiny(mx::bfloat16, true, 4);
  const int n = quantize_weights(q, 8, 32);
  CHECK(q.head_q, "the head was not quantized");
  Model dqm = tiny(mx::bfloat16, true, 4);  // the bf16 model holding the dequantized weights
  auto dequant = [](const Quant& w) { return mx::dequantize(w.w, w.scales, w.biases, w.group, w.bits, "affine"); };
  auto within = [](const mx::array& got, const mx::array& want) {
    const float d = mx::max(mx::abs(mx::astype(got, mx::float32) - mx::astype(want, mx::float32))).item<float>();
    const float y = mx::max(mx::abs(mx::astype(want, mx::float32))).item<float>();
    return std::pair<float, float>{d, 4 * std::ldexp(1.0f, std::ilogb(y) - 7)};
  };
  int want = 0, kept = 0;
  float worst = 0;
  for (size_t i = 0; i < q.layers.size(); ++i) {
    auto qp = projections(q.layers[i]);
    auto bp = projections(base.layers[i]);
    auto dp = projections(dqm.layers[i]);
    for (size_t j = 0; j < qp.size(); ++j) {
      const std::string& name = qp[j].first;
      const Proj& p = *qp[j].second;
      const bool gate = name == "linear_attn.in_proj_b" || name == "linear_attn.in_proj_a";
      CHECK(bool(p.q) != gate, "layer %zu %s: quantized %d", i, name.c_str(), int(bool(p.q)));
      if (gate) {
        CHECK(mx::array_equal(p.w, bp[j].second->w).item<bool>(), "layer %zu %s: the bf16 weight changed", i,
              name.c_str());
        ++kept;
        continue;
      }
      ++want;
      dp[j].second->w = dequant(*p.q);
      const mx::array x = mx::astype(mx::random::normal({1, 5, dp[j].second->w.shape(1)}), mx::bfloat16);
      const auto [d, tol] = within(linear(x, p), linear(x, dp[j].second->w));
      CHECK(d <= tol, "layer %zu %s: quantized vs its dequantized weight %g (allowed %g)", i, name.c_str(), d, tol);
      worst = std::max(worst, d / tol);
    }
  }
  CHECK(n == want, "%d projections quantized, want %d", n, want);
  {
    const mx::array h = mx::astype(mx::random::normal({1, 5, q.cfg.hidden}), mx::bfloat16);
    dqm.lm_head = dequant(*q.head_q);
    const auto [d, tol] = within(linear(h, Proj{dqm.lm_head, std::nullopt, q.head_q}), linear(h, dqm.lm_head));
    CHECK(d <= tol, "the head: quantized vs its dequantized weight %g (allowed %g)", d, tol);
    worst = std::max(worst, d / tol);
  }
  dqm.cfg.tied = false;  // the head is its own (dequantized) matrix now; the lookup keeps the bf16 embedding
  const mx::array toks = mx::random::randint(0, base.cfg.vocab, {1, 64}, mx::int32);
  const mx::array lb = forward(base, toks), lq = forward(q, toks), ld = forward(dqm, toks);
  const float dd = max_diff(lq, ld), d = max_diff(lq, lb), spread = mx::max(mx::abs(lb)).item<float>();
  CHECK(d > 0 && dd < d, "8-bit logits: %g from the dequantized model's, %g from the unquantized model's", dd, d);
  Model a = tiny(mx::bfloat16, true, 3);
  a.layers[0].up.lora = Lora{mx::zeros({2, a.cfg.hidden}), mx::zeros({a.cfg.intermediate, 2}), 1.0f};
  bool refused = false;
  try {
    quantize_weights(a, 8, 32);
  } catch (const std::exception&) {
    refused = true;
  }
  CHECK(refused, "quantize_weights took a projection with an adapter");
  std::printf("8-bit weights (tiny Qwen3.5, groups of 32): %d projections and the head quantized, %d gate projections "
              "kept bf16; each runs its dequantized weight (worst %.2f of the allowed 4 ulps); the model's logits %g "
              "from the dequantized model's, %g from the unquantized model's (largest |logit| %g; printed only); a "
              "projection with an adapter is refused\n",
              n, kept, worst, dd, d, spread);
}

int main() {
  quantized();
  for (bool q35 : {false, true}) {
    // float32 isolates the cache logic from rounding: the paths differ only in accumulation order.
    run(q35, mx::float32, 1e-4f, false);
    // In bf16 the single-token kernels round differently from the full ones; 4 ulps of the largest logit allows
    // that and still catches a wrong position or a missing cache row, which move logits by whole units.
    run(q35, mx::bfloat16, 4, true);
  }
  return 0;
}
