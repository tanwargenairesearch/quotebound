#pragma once
#include <cstdio>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "adamw.h"
#include "data.h"
#include "lora.h"
#include "model.h"
#include "provenance.h"
#include "tokenizer.h"

struct TrainArgs {
  std::string model_dir, train_path, dev_path, out_dir;
  LoraConfig lora;  // its seed is `seed`
  float lr = 1e-4f;
  int steps = 0, epochs = 0;  // the run's length: exactly one of them is set
  int batch_tokens = 4096, max_len = 2048;
  uint64_t seed = 0;
  int eval_every = 50, save_every = 100;
  int warmup = -1;    // steps; -1 is 5% of the run
  int dev_max = 200;  // dev rows scored at each evaluation: a seeded subset when the file has more
  int gen_dev = 16;   // of those, rows whose greedy answers are written at each evaluation
  int keep = 5;       // step-N/ adapter directories kept, newest first; 0 keeps them all
  int patience = 0;   // stop after this many evaluations without a lower dev loss; 0 never stops early
  AdamWConfig adamw;
};

// Throws, naming the flag, on arguments no run can use.
void check_args(const TrainArgs& a);

struct Trainer {
  TrainArgs args;
  BaseModel base;
  const Tokenizer* tok = nullptr;  // decodes the dev generations; without one gen_dev must be 0
  Model model;                     // with adapters
  std::vector<Example> train, dev;  // dev: the scored subset, in its seeded order
  Dataset train_meta, dev_meta;     // the files' drops and system prompts (examples moved out)
  std::string train_sha256, dev_sha256;  // of every kept row of each file
  size_t dev_rows = 0;                   // kept rows of the dev file, before the subset
  AdamWState opt;
  int step = 0, total_steps = 0, warmup = 0;
  int epoch = 0;
  size_t cursor = 0;                   // next batch of `plan`
  std::vector<std::vector<int>> plan;  // this epoch's batches
  double best_dev = std::numeric_limits<double>::infinity();
  int best_step = -1;
  int stale_evals = 0;  // evaluations since best_dev last fell
};

// A fresh run: adapters added to m from args.lora, the optimizer at zero, epoch 0's batches. Throws on an empty dev
// set or any dev row the loader left out (each is named): the dev loss must cover the whole file it names.
Trainer make_trainer(Model m, Dataset train, Dataset dev, const TrainArgs& a, const BaseModel& base,
                     const Tokenizer* tok);

// The training closure: the mean loss over a batch's trained tokens, its gradient with respect to every adapter
// array (adapter_arrays order) and the gradients' global norm, all evaluated. The loss and gradients are evaluated
// alone, then the norm, then (apply_gradients) the update: anything that reads every gradient, evaluated in one graph
// with them, lets MLX's breadth-first order run several layers' recomputations at once. At 4B with 1,024 tokens the
// norm in the gradients' graph peaked at 17.6 GB against 10.5 GB, and at 4,096 tokens ran out of memory.
struct LossAndGrads {
  mx::array loss, norm;
  std::vector<mx::array> grads;
};
LossAndGrads adapter_loss_and_grads(const Model& m, const Batch& b);

// One AdamW step from evaluated gradients at lr. Throws before anything changes when the loss or the norm is not
// finite, so a NaN never reaches the adapters, the moments or a checkpoint.
void apply_gradients(Trainer& t, const LossAndGrads& g, float lr);

struct StepStats {
  float loss = 0, lr = 0, grad_norm = 0;
  int trained = 0, tokens = 0;  // positions with a loss; tokens before padding
  double ms = 0;
  size_t peak_bytes = 0;  // mx::get_peak_memory over this step
};
StepStats train_step(Trainer& t);

// Summed cross-entropy and the number of trained tokens over a set, every example once.
std::pair<double, long> eval_loss(const Model& m, const std::vector<Example>& ex, int batch_tokens);

// Greedy answers to the first gen_dev dev rows' prompts, as {"id", "output", "truncated"} JSON lines. Each answer may
// run to twice its reference's length plus 32 tokens.
std::vector<std::string> dev_generations(const Trainer& t);

// The run's record: its arguments, base checkpoint, data digests, drops, dev subset and system prompts, a JSON object.
std::string run_record(const Trainer& t);

// out_dir/state.safetensors holds the adapters, AdamW's moments and step, the epoch's batch plan and the position in
// it, in one file so a crash cannot leave them out of step; the previous one is kept as state.prev.safetensors.
// out_dir/adapters.safetensors is a copy for generate.
void save_checkpoint(const Trainer& t);
// Resumes t from out_dir/state.safetensors, or from state.prev.safetensors when that will not read. Throws if the
// saved run's arguments, data or base differ from t's, or if the saved plan does not fit t's training set.
void load_checkpoint(Trainer& t);
// With resume, load_checkpoint; without it, refuses an out_dir that already holds a run's state.
void start_or_resume(Trainer& t, bool resume);

// The run: an evaluation before the first step, every eval_every steps and at the end. Each writes the dev loss,
// out_dir/generations/step-N.jsonl and out_dir/step-N/ (the newest `keep` kept), and the best adapters by dev loss to
// out_dir/best. A checkpoint every save_every steps and at the end; an early stop after `patience` evaluations
// without a lower dev loss. Every log line goes to each of `log`.
void run_training(Trainer& t, const std::vector<std::FILE*>& log);
