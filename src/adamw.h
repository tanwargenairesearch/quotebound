#pragma once
#include <vector>

#include <mlx/mlx.h>

namespace mx = mlx::core;

struct AdamWConfig {
  float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
  float weight_decay = 0.0f;  // decoupled: the parameter shrinks by lr * weight_decay each step
  float clip = 1.0f;          // global gradient norm cap; 0 turns clipping off
};

struct AdamWState {
  std::vector<mx::array> m, v;  // float32, one pair per parameter
  int step = 0;                 // updates taken
};

AdamWState adamw_init(const std::vector<mx::array>& params);

// sqrt of the sum of every gradient entry squared; not evaluated.
mx::array global_norm(const std::vector<mx::array>& grads);

// Clips the gradients to a global norm of clip (norm is global_norm(grads)), then takes one AdamW step at lr, as
// torch.optim.AdamW does it (bias-corrected moments, weight decay applied to the parameter before the moment step).
// Returns the new parameters and updates s; nothing is evaluated.
std::vector<mx::array> adamw_update(const std::vector<mx::array>& params, const std::vector<mx::array>& grads,
                                    const mx::array& norm, AdamWState& s, const AdamWConfig& c, float lr);

// Linear warmup from peak / warmup to peak over the first `warmup` steps, then cosine decay to zero at `total`.
// step counts from 0.
float lr_at(int step, int total, int warmup, float peak);
