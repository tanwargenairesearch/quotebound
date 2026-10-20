#include "loss.h"

#include <algorithm>
#include <memory>

namespace {
constexpr int kChunk = 256;
}

mx::array loss_sum(const Model& m, const Batch& b, bool checkpoint) {
  const int N = b.positions.shape(0);
  const mx::array h = hidden(m, b.tokens, checkpoint);
  // The backward calls a checkpointed function again after the caller's Model may be gone: it holds its own.
  auto model = std::make_shared<const Model>(m);
  mx::array total = mx::array(0.0f);
  for (int s = 0; s < N; s += kChunk) {
    const int e = std::min(N, s + kChunk);
    auto ce = [model, at = mx::slice(b.positions, {s}, {e}), target = mx::slice(b.targets, {s}, {e})](
                  const std::vector<mx::array>& x) {
      mx::array logits = logits_at(*model, x[0], at);
      mx::array picked = mx::squeeze(mx::take_along_axis(logits, mx::expand_dims(target, -1), -1), -1);
      return std::vector<mx::array>{mx::sum(mx::logsumexp(logits, -1) - picked)};
    };
    total = total + (checkpoint ? mx::checkpoint(ce)({h}) : ce({h}))[0];
  }
  return total;
}
