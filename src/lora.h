#pragma once
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "model.h"
#include "provenance.h"

struct LoraConfig {
  int rank = 16;
  float alpha = 32;
  std::vector<std::string> targets;  // projection names after the last '.', e.g. "q_proj", "in_proj_qkv"
  uint64_t seed = 0;
};

// Full attention q/k/v/o, Gated DeltaNet in_proj_qkv/in_proj_z/out_proj, MLP gate/up/down.
const std::vector<std::string>& default_targets();

// Puts a fresh adapter in every targeted projection of every layer: a uniform in +-1/sqrt(in) from a key split off
// the seed per projection, b zeros, both float32, scale alpha / rank. Throws on a target that is no projection name,
// or when nothing in this model is targeted.
void add_lora(Model& m, const LoraConfig& c);

// Every adapter array in a fixed order (layer by layer, a then b), their names ("model.layers.3.mlp.up_proj.lora_a"),
// and the inverse of the first.
std::vector<mx::array> adapter_arrays(const Model& m);
std::vector<std::string> adapter_names(const Model& m);
void set_adapter_arrays(Model& m, const std::vector<mx::array>& arrays);

// What adapters were trained from and how far, recorded beside them.
struct AdapterInfo {
  LoraConfig lora;
  BaseModel base;
  int step = 0;
  double dev_loss = std::numeric_limits<double>::quiet_NaN();  // NaN: not measured at this step
  std::string run = "{}";  // the run's record (arguments, data digests, system prompts) as a JSON object
};

// dir/adapters.safetensors, then dir/adapter_config.json holding `info` and the adapters file's sha256. Each is
// written to a temporary name, synced and renamed.
void save_adapters(const std::string& dir, const Model& m, const AdapterInfo& info);
// Reads both into m's slots. Throws on adapters for another architecture or of the wrong shape, on a base
// checkpoint or chat template other than `base`, and on an adapters file whose sha256 is not the recorded one
// (a crash between the two writes).
AdapterInfo load_adapters(const std::string& dir, Model& m, const BaseModel& base);

// Folds every adapter into its frozen weight and drops it: W' = W + scale (B A), the sum in float32, rounded once to W's
// dtype. Inference then reads one matrix per projection. Not the training path's arithmetic (that adds the low-rank
// product to the activations), so the logits move by rounding; returns the number of projections merged.
int merge_lora(Model& m);
