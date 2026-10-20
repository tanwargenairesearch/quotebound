#include "adamw.h"

#include <algorithm>
#include <cmath>
#include <optional>

AdamWState adamw_init(const std::vector<mx::array>& params) {
  AdamWState s;
  for (const mx::array& p : params) {
    s.m.push_back(mx::zeros(p.shape(), mx::float32));
    s.v.push_back(mx::zeros(p.shape(), mx::float32));
  }
  return s;
}

mx::array global_norm(const std::vector<mx::array>& grads) {
  mx::array sq = mx::array(0.0f);
  for (const mx::array& g : grads) sq = sq + mx::sum(mx::square(g));
  return mx::sqrt(sq);
}

std::vector<mx::array> adamw_update(const std::vector<mx::array>& params, const std::vector<mx::array>& grads,
                                    const mx::array& norm, AdamWState& s, const AdamWConfig& c, float lr) {
  // torch.nn.utils.clip_grad_norm_: scale by clip / (norm + 1e-6) when that is below 1.
  std::optional<mx::array> coef;
  if (c.clip > 0) coef = mx::minimum(mx::array(1.0f), mx::array(c.clip) / (norm + mx::array(1e-6f)));

  s.step += 1;
  const float bc1 = float(1.0 - std::pow(double(c.beta1), s.step));
  const float bc2_sqrt = float(std::sqrt(1.0 - std::pow(double(c.beta2), s.step)));
  const mx::array b1(c.beta1), b2(c.beta2), one_b1(1.0f - c.beta1), one_b2(1.0f - c.beta2);
  std::vector<mx::array> out;
  for (size_t i = 0; i < params.size(); ++i) {
    mx::array g = coef ? grads[i] * *coef : grads[i];
    s.m[i] = b1 * s.m[i] + one_b1 * g;
    s.v[i] = b2 * s.v[i] + one_b2 * mx::square(g);
    mx::array p = params[i] * mx::array(1.0f - lr * c.weight_decay);
    mx::array denom = mx::sqrt(s.v[i]) / mx::array(bc2_sqrt) + mx::array(c.eps);
    out.push_back(p - mx::array(lr / bc1) * (s.m[i] / denom));
  }
  return out;
}

float lr_at(int step, int total, int warmup, float peak) {
  if (step < warmup) return peak * float(step + 1) / float(warmup);
  if (total <= warmup) return peak;
  const double progress = double(step - warmup) / double(total - warmup);
  return float(0.5 * peak * (1.0 + std::cos(M_PI * std::min(progress, 1.0))));
}
