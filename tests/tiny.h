#pragma once
// Small random-weight models for the tests, in float32 or bfloat16: a 2-layer Qwen3, and a Qwen3.5 of 3 layers by
// default (two Gated DeltaNet layers, then one gated full-attention layer, tied embeddings; deeper, every fourth layer
// and the last are full attention, as in the checkpoints). The DeltaNet layers have Qwen3.5's 16 key and 32 value
// heads: with fewer, mx::fast::gated_delta_update falls back to plain ops instead of its kernel. `vocab` widens the
// vocabulary to a real tokenizer's, for tests that feed it rendered chats.
#include <cmath>
#include <string>
#include <unordered_map>

#include "../src/model.h"

inline Model tiny(mx::Dtype dt, bool q35, int q35_layers = 3, int vocab = 101) {
  Config c;
  c.type = q35 ? "qwen3_5" : "qwen3";
  c.hidden = 96, c.intermediate = 192, c.vocab = vocab, c.eps = 1e-6f, c.eos = {0}, c.max_positions = 4096;
  if (!q35) {
    c.layers = 2, c.heads = 4, c.kv_heads = 2, c.head_dim = 128, c.rope_dims = 128, c.rope_theta = 1e6f;
    c.linear = {false, false};
  } else {
    c.layers = q35_layers, c.heads = 2, c.kv_heads = 1, c.head_dim = 256, c.rope_dims = 64, c.rope_theta = 1e7f;
    c.attn_gate = true, c.tied = true;
    for (int l = 0; l < c.layers; ++l) c.linear.push_back(l % 4 != 3 && l != c.layers - 1);
    c.lin_k_heads = 16, c.lin_v_heads = 32, c.lin_k_dim = 128, c.lin_v_dim = 128, c.conv_kernel = 4;
  }
  mx::random::seed(7);
  auto rnd = [&](mx::Shape s, float mean, float sd) { return mx::astype(mx::random::normal(s, mean, sd), dt); };
  auto lin = [&](int out, int in) { return rnd({out, in}, 0.f, 1.f / std::sqrt(float(in))); };
  auto norm = [&](int n) { return rnd({n}, 1.f, 0.1f); };
  const int q = c.heads * c.head_dim, kv = c.kv_heads * c.head_dim;
  const int kd = c.lin_k_heads * c.lin_k_dim, vd = c.lin_v_heads * c.lin_v_dim, conv = 2 * kd + vd;
  std::unordered_map<std::string, mx::array> w;
  // Tied, the embedding is also the output projection, so it gets that projection's scale.
  float embed_sd = c.tied ? 1.f / std::sqrt(float(c.hidden)) : 1.f;
  w.emplace("model.embed_tokens.weight", rnd({c.vocab, c.hidden}, 0.f, embed_sd));
  w.emplace("model.norm.weight", norm(c.hidden));
  if (!c.tied) w.emplace("lm_head.weight", lin(c.vocab, c.hidden));
  for (int l = 0; l < c.layers; ++l) {
    std::string p = "model.layers." + std::to_string(l) + ".";
    w.emplace(p + "input_layernorm.weight", norm(c.hidden));
    w.emplace(p + "post_attention_layernorm.weight", norm(c.hidden));
    w.emplace(p + "mlp.gate_proj.weight", lin(c.intermediate, c.hidden));
    w.emplace(p + "mlp.up_proj.weight", lin(c.intermediate, c.hidden));
    w.emplace(p + "mlp.down_proj.weight", lin(c.hidden, c.intermediate));
    if (c.linear[l]) {
      std::string a = p + "linear_attn.";
      w.emplace(a + "in_proj_qkv.weight", lin(conv, c.hidden));
      w.emplace(a + "in_proj_z.weight", lin(vd, c.hidden));
      w.emplace(a + "in_proj_b.weight", lin(c.lin_v_heads, c.hidden));
      w.emplace(a + "in_proj_a.weight", lin(c.lin_v_heads, c.hidden));
      w.emplace(a + "conv1d.weight", rnd({conv, c.conv_kernel, 1}, 0.f, 0.5f));
      w.emplace(a + "dt_bias", rnd({c.lin_v_heads}, 1.f, 0.1f));
      // As the checkpoints store them: A_log and the gated norm in float32.
      w.emplace(a + "A_log", mx::log(mx::random::uniform(1.f, 16.f, {c.lin_v_heads})));
      w.emplace(a + "norm.weight", mx::random::normal({c.lin_v_dim}, 1.f, 0.1f));
      w.emplace(a + "out_proj.weight", lin(c.hidden, vd));
    } else {
      std::string a = p + "self_attn.";
      w.emplace(a + "q_proj.weight", lin(q * (c.attn_gate ? 2 : 1), c.hidden));
      w.emplace(a + "k_proj.weight", lin(kv, c.hidden));
      w.emplace(a + "v_proj.weight", lin(kv, c.hidden));
      w.emplace(a + "o_proj.weight", lin(c.hidden, q));
      w.emplace(a + "q_norm.weight", norm(c.head_dim));
      w.emplace(a + "k_norm.weight", norm(c.head_dim));
    }
  }
  Model m = make_model(c, std::move(w));
  mx::eval(parameters(m));
  return m;
}

// The oracle for greedy()'s token confidence: the answer teacher-forced. Each id's log-probability (log-softmax over the
// whole vocabulary of the logits at the position before it) and that position's top logit less its second, as
// {logprob, margin} per id. whole: prompt and answer in one forward with no cache; else through the cache one token at
// a time after the prompt's prefill, the decoder's own arithmetic.
#include <utility>
#include <vector>

#include "../src/generate.h"

inline std::vector<std::pair<float, float>> teacher_forced(const Model& m, const std::vector<int>& prompt,
                                                           const std::vector<int>& ids, bool whole) {
  const int P = int(prompt.size()), n = int(ids.size());
  std::vector<mx::array> rows;  // the logits each id was chosen from, [vocab] each
  if (whole) {
    std::vector<int> seq = prompt;
    seq.insert(seq.end(), ids.begin(), ids.end() - 1);
    const mx::array logits = forward(m, mx::array(seq.data(), {1, int(seq.size())}, mx::int32));
    const int V = logits.shape(-1);
    rows.push_back(mx::slice(mx::reshape(logits, {int(seq.size()), V}), {P - 1, 0}, {P - 1 + n, V}));
  } else {
    KVCache cache;
    mx::array logits = prefill(m, prompt, cache);
    for (int k = 0; k < n; ++k) {
      rows.push_back(mx::reshape(logits, {1, -1}));
      mx::eval(rows.back());
      if (k + 1 < n) logits = forward(m, mx::array(&ids[k], {1, 1}, mx::int32), &cache, true);
    }
  }
  const mx::array all = rows.size() == 1 ? rows[0] : mx::concatenate(rows, 0);  // [n, vocab]
  const mx::array idx(ids.data(), {n, 1}, mx::int32);
  const mx::array lp = mx::reshape(mx::take_along_axis(all, idx, -1), {-1}) - mx::logsumexp(all, -1);
  const mx::array two = mx::topk(all, 2, -1);
  const mx::array margin = mx::max(two, -1) - mx::min(two, -1);
  mx::eval({lp, margin});
  std::vector<std::pair<float, float>> out;
  for (int k = 0; k < n; ++k) out.push_back({lp.data<float>()[k], margin.data<float>()[k]});
  return out;
}
