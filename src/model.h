#pragma once
#include <mlx/mlx.h>

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace mx = mlx::core;

// Qwen3 and Qwen3.5 text decoders, read from config.json. Qwen3 is the case with every layer full attention,
// no output gate and rope over the whole head.
struct Config {
  std::string type;  // config.json model_type: qwen3 or qwen3_5
  int hidden = 0, layers = 0, heads = 0, kv_heads = 0, head_dim = 0, intermediate = 0, vocab = 0;
  float eps = 0, rope_theta = 0;
  int rope_dims = 0;         // leading dims of each head that rope rotates
  int max_positions = 0;     // max_position_embeddings
  bool attn_gate = false;    // attention output times sigmoid of the second half of q_proj
  bool tied = false;         // lm_head is the embedding matrix
  std::vector<bool> linear;  // per layer: Gated DeltaNet instead of full attention
  int lin_k_heads = 0, lin_v_heads = 0, lin_k_dim = 0, lin_v_dim = 0, conv_kernel = 0;
  std::vector<int> eos;
};

// A low-rank adapter on a frozen projection: y = x W^T + scale (x A^T) B^T, with a [rank, in] and b [out, rank]
// in float32 and scale = alpha / rank.
struct Lora {
  mx::array a, b;
  float scale = 1;
};

// A weight quantized by MLX's affine quantization along its input axis: each `group` inputs of a row share a scale and
// a bias, the values stored in `bits` bits.
struct Quant {
  mx::array w, scales, biases;
  int group = 64, bits = 8;
};

// A frozen bf16 projection with a slot for an adapter; quantized (quantize_weights), `q` holds the weight it runs.
struct Proj {
  mx::array w;
  std::optional<Lora> lora;
  std::optional<Quant> q;
};

struct Attention {
  Proj q, k, v, o;
  mx::array q_norm, k_norm;
};

struct DeltaNet {
  Proj in_qkv, in_z, in_b, in_a;
  mx::array conv, dt_bias, a_log, norm;
  Proj out;
};

struct Layer {
  mx::array in_norm, post_norm;
  Proj gate, up, down;
  std::optional<Attention> attn;
  std::optional<DeltaNet> delta;
};

struct Model {
  Config cfg;
  mx::array embed, norm, lm_head;
  std::vector<Layer> layers;
  std::optional<Quant> head_q;  // quantized: the output head (the embedding stays bf16 for the lookup)
};

// Full attention keeps keys/values [B, kv_heads, capacity, head_dim], grown in 256-row steps like mlx-lm's.
// Gated DeltaNet keeps its last conv_kernel-1 inputs and the float32 recurrent state [B, Hv, Dv, Dk].
struct LayerCache {
  std::optional<mx::array> k, v, conv, state;
};
struct KVCache {
  std::vector<LayerCache> layers;
  int offset = 0;
};
// Every array the cache holds, to evaluate with a step so the cache does not keep that step's inputs alive.
std::vector<mx::array> cache_arrays(const KVCache& cache);

// A pass over several tokens run as that many decode steps (exact.h): projections through gemv_rows, attention and the
// Gated DeltaNet conv and recurrence one token at a time. Each DeltaNet layer's conv window and recurrent state after
// each token are kept here (empty for attention layers, which the cache's offset rolls back).
struct RowStates {
  std::vector<std::vector<mx::array>> conv, state;  // [layer][row]
};

Config load_config(const std::string& path);
// Takes Hugging Face weight names under "model." (Qwen3.5's "model.language_model." already mapped there);
// throws naming the tensor on a missing, extra or mis-shaped one.
Model make_model(const Config& cfg, std::unordered_map<std::string, mx::array> w);
// Also checks every tensor's dtype: bf16, except Qwen3.5's A_log and gated-norm weight, stored float32.
Model load_model(const std::string& dir);
// Every weight, adapters included.
std::vector<mx::array> parameters(const Model& m);

// A layer's projections in a fixed order, named as in the checkpoint without ".weight" ("self_attn.q_proj").
std::vector<std::pair<std::string, Proj*>> projections(Layer& l);
std::vector<std::pair<std::string, const Proj*>> projections(const Layer& l);
// A layer's adapter arrays (a then b, projection by projection), and the inverse: the slots that hold an adapter
// take the next arrays in that order.
std::vector<mx::array> adapter_arrays(const Layer& l);
void set_adapter_arrays(Layer& l, const mx::array* arrays);

// tokens [B, L] -> float32 logits [B, L, vocab] ([B, 1, vocab] with last_only). Without a cache the
// positions start at 0 and nothing is kept.
// With rows (and a cache), the pass is RowStates' one: each position bit-identical to a decode step of its own.
mx::array forward(const Model& m, const mx::array& tokens, KVCache* cache = nullptr, bool last_only = false,
                  RowStates* rows = nullptr);

// One decoder layer: h [B, L, hidden] -> h after its token mixer and MLP. `train` runs the recurrence through
// gated_delta_trainable, whose forward is the same op and whose backward does not keep every step's state.
mx::array block(const Layer& w, const Config& c, const mx::array& h, int offset, LayerCache* cache, bool train,
                RowStates* rows = nullptr, int layer = 0);

// For training: tokens [B, L] -> the final normed hidden states [B, L, hidden], no cache. With checkpoint, each
// layer keeps only its input for the backward and recomputes the rest; its adapters are its only other inputs, so
// they are what a gradient flows to.
mx::array hidden(const Model& m, const mx::array& tokens, bool checkpoint = true);
// Float32 logits [N, vocab] of the N flat positions (row-major over [B, L]) of hidden's output: lm_head never runs
// on positions that are not asked for.
mx::array logits_at(const Model& m, const mx::array& h, const mx::array& positions);

mx::array linear(const mx::array& x, const mx::array& w);
mx::array linear(const mx::array& x, const Proj& p);

// Quantizes the weights decoding reads (MLX affine, `bits` bits, `group` inputs a scale and a bias): every
// projection but the Gated DeltaNet's in_proj_b and in_proj_a ([value heads, hidden]: the recurrence's write strength
// and decay, a few kB a layer), and the output head as a quantized copy of the (tied) embedding, whose lookup reads its
// bf16 rows. Norms, the conv, A_log and dt_bias stay as they are; the recurrent state stays float32. The bf16
// projection weights are dropped. Adapters must be merged first (merge_lora). Returns the projections quantized.
int quantize_weights(Model& m, int bits, int group);
