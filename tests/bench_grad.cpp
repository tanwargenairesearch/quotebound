// Time and peak memory of the gated delta recurrence's backward: the fast op's own vjp (autodiff through its plain-ops
// fallback, every step's state kept) against gated_delta_trainable's. First the recurrence alone, then one whole
// Qwen3.5-4B Gated DeltaNet layer (random bf16 weights) through block(), both at B = 1.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

#include "../src/delta_grad.h"
#include "../src/model.h"
#include "check.h"

struct Measure {
  double ms;
  double peak_gb;  // above what was allocated before the call
};

// Median of 3 timed runs after one warm-up; the peak is the largest over the runs.
template <typename F>
static Measure measure(F&& run) {
  mx::eval(run());
  std::vector<double> times;
  double peak = 0;
  for (int i = 0; i < 3; ++i) {
    mx::synchronize();  // the last run's buffers are released when its command buffers complete
    mx::clear_cache();
    const size_t before = mx::get_active_memory();
    mx::reset_peak_memory();
    auto t0 = std::chrono::steady_clock::now();
    mx::eval(run());
    times.push_back(ms_since(t0));
    peak = std::max(peak, double(mx::get_peak_memory() - before) / 1e9);
  }
  std::sort(times.begin(), times.end());
  return {times[1], peak};
}

static void recurrence(int T, bool with_ops) {
  const int B = 1, Hk = 16, Hv = 32, Dk = 128, Dv = 128;
  mx::random::seed(T);
  auto unit = [&](mx::Shape s, float scale) {
    return mx::fast::rms_norm(mx::random::normal(s), std::nullopt, 1e-6f) * scale;
  };
  std::vector<mx::array> in{unit({B, T, Hk, Dk}, 1.f / Dk), unit({B, T, Hk, Dk}, 1.f / std::sqrt(float(Dk))),
                            mx::random::normal({B, T, Hv, Dv}), mx::random::uniform(0.9f, 1.f, {B, T, Hv}),
                            mx::random::uniform(0.f, 1.f, {B, T, Hv}), mx::zeros({B, Hv, Dv, Dk})};
  mx::array cy = mx::random::normal({B, T, Hv, Dv});
  mx::eval(in), mx::eval(cy);
  auto grad = [&](bool trainable) {
    return [&, trainable] {
      auto loss = [&](const std::vector<mx::array>& x) {
        auto y = trainable ? gated_delta_trainable(x[0], x[1], x[2], x[3], x[4], x[5]).first
                           : mx::fast::gated_delta_update(x[0], x[1], x[2], x[3], x[4], x[5])[0];
        return std::vector<mx::array>{mx::sum(y * cy)};
      };
      return mx::vjp(loss, in, {mx::array(1.0f)}).second;
    };
  };
  Measure fwd = measure([&] { return mx::fast::gated_delta_update(in[0], in[1], in[2], in[3], in[4], in[5]); });
  Measure mine = measure(grad(true));
  std::printf("recurrence T=%5d: forward %7.2f ms; trainable vjp %8.2f ms, peak %6.3f GB", T, fwd.ms, mine.ms,
              mine.peak_gb);
  if (with_ops) {
    Measure ops = measure(grad(false));
    std::printf("; ops vjp %8.2f ms, peak %6.3f GB", ops.ms, ops.peak_gb);
  }
  std::printf("\n");
}

static Layer deltanet_layer(const Config& c) {
  mx::random::seed(11);
  auto lin = [&](int out, int in) {
    return Proj{mx::astype(mx::random::normal({out, in}, 0.f, 1.f / std::sqrt(float(in))), mx::bfloat16),
                std::nullopt};
  };
  auto bf = [](mx::array a) { return mx::astype(a, mx::bfloat16); };
  const int d = c.hidden, f = c.intermediate, kd = c.lin_k_heads * c.lin_k_dim, vd = c.lin_v_heads * c.lin_v_dim;
  Layer l{bf(mx::random::normal({d}, 1.f, 0.1f)), bf(mx::random::normal({d}, 1.f, 0.1f)), lin(f, d), lin(f, d),
          lin(d, f), std::nullopt, std::nullopt};
  l.delta = DeltaNet{lin(2 * kd + vd, d),
                     lin(vd, d),
                     lin(c.lin_v_heads, d),
                     lin(c.lin_v_heads, d),
                     bf(mx::random::normal({2 * kd + vd, c.conv_kernel, 1}, 0.f, 0.5f)),
                     bf(mx::random::normal({c.lin_v_heads}, 1.f, 0.1f)),
                     mx::log(mx::random::uniform(1.f, 16.f, {c.lin_v_heads})),
                     mx::random::normal({c.lin_v_dim}, 1.f, 0.1f),
                     lin(d, vd)};
  std::vector<mx::array> all{l.in_norm, l.post_norm, l.delta->conv, l.delta->dt_bias, l.delta->a_log, l.delta->norm};
  for (const auto& [name, p] : projections(l)) all.push_back(p->w);
  mx::eval(all);
  return l;
}

static void layer(int T, bool with_ops) {
  Config c;  // Qwen3.5-4B's text config, one Gated DeltaNet layer of it
  c.type = "qwen3_5", c.hidden = 2560, c.intermediate = 9216, c.eps = 1e-6f;
  c.lin_k_heads = 16, c.lin_v_heads = 32, c.lin_k_dim = 128, c.lin_v_dim = 128, c.conv_kernel = 4;
  static const Layer w = deltanet_layer(c);
  mx::random::seed(T);
  mx::array x = mx::astype(mx::random::normal({1, T, c.hidden}), mx::bfloat16);
  mx::array cot = mx::astype(mx::random::normal({1, T, c.hidden}), mx::bfloat16);
  mx::eval(x, cot);
  auto grad = [&](bool trainable) {
    return [&, trainable] {
      auto f = [&](const std::vector<mx::array>& in) {
        mx::array out = block(w, c, in[0], 0, nullptr, trainable);
        return std::vector<mx::array>{mx::sum(mx::astype(out * cot, mx::float32))};
      };
      return mx::vjp(f, {x}, {mx::array(1.0f)}).second;
    };
  };
  Measure fwd = measure([&] { return std::vector<mx::array>{block(w, c, x, 0, nullptr, false)}; });
  Measure mine = measure(grad(true));
  std::printf("4B DeltaNet layer T=%5d: forward %7.2f ms; backward with trainable vjp %8.2f ms, peak %6.3f GB", T,
              fwd.ms, mine.ms, mine.peak_gb);
  if (with_ops) {
    Measure ops = measure(grad(false));
    std::printf("; with ops vjp %8.2f ms, peak %6.3f GB", ops.ms, ops.peak_gb);
  }
  std::printf("\n");
}

int main(int argc, char** argv) {
  // --no-ops skips the ops vjp, which at 2,048 tokens needs several GB.
  const bool ops = !(argc > 1 && std::strcmp(argv[1], "--no-ops") == 0);
  for (int T : {128, 512, 1024, 2048}) recurrence(T, ops);
  for (int T : {128, 512, 1024, 2048}) layer(T, ops);
  return 0;
}
