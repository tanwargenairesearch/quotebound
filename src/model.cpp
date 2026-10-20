// Follows mlx-lm's Qwen3 / Qwen3.5 implementation for token-for-token parity (Copyright © 2023 Apple Inc., MIT
// License; see NOTICE).
#include "model.h"

#include <cmath>
#include <set>
#include <sstream>
#include <stdexcept>

#include "deltanet.h"
#include "exact.h"
#include "json.h"

namespace {

void need(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error(what);
}

bool ends_with(const std::string& s, std::string_view tail) {
  return s.size() >= tail.size() && s.compare(s.size() - tail.size(), tail.size(), tail) == 0;
}

template <typename T>
std::string str(const T& x) {
  std::ostringstream os;
  os << x;
  return os.str();
}

mx::array swiglu(const mx::array& gate, const mx::array& up) {
  // Compiled like mlx-lm's swiglu so the fused kernel, and its bf16 rounding, are the same.
  static auto fn = mx::compile(
      [](const std::vector<mx::array>& in) {
        return std::vector<mx::array>{mx::multiply(mx::multiply(in[0], mx::sigmoid(in[0])), in[1])};
      },
      /*shapeless=*/true);
  return fn({gate, up})[0];
}

// Mirrors mlx-lm's KVCache.update_and_fetch: grow by whole 256-row steps, write the new rows in place.
std::pair<mx::array, mx::array> cache_update(LayerCache& c, int prev, const mx::array& k, const mx::array& v) {
  const int step = 256, B = k.shape(0), H = k.shape(1), L = k.shape(2), D = k.shape(3);
  if (!c.k || prev + L > c.k->shape(2)) {
    mx::Shape grow{B, H, ((step + L - 1) / step) * step, D};
    mx::array nk = mx::zeros(grow, k.dtype()), nv = mx::zeros(grow, v.dtype());
    if (!c.k) {
      c.k = nk, c.v = nv;
    } else {
      c.k = mx::concatenate({mx::slice(*c.k, {0, 0, 0, 0}, {B, H, prev, D}), nk}, 2);
      c.v = mx::concatenate({mx::slice(*c.v, {0, 0, 0, 0}, {B, H, prev, D}), nv}, 2);
    }
  }
  c.k = mx::slice_update(*c.k, k, {0, 0, prev, 0}, {B, H, prev + L, D});
  c.v = mx::slice_update(*c.v, v, {0, 0, prev, 0}, {B, H, prev + L, D});
  return {mx::slice(*c.k, {0, 0, 0, 0}, {B, H, prev + L, D}), mx::slice(*c.v, {0, 0, 0, 0}, {B, H, prev + L, D})};
}

mx::array attention(const Attention& w, const Config& c, const mx::array& x, int offset, LayerCache* cache,
                    bool rows) {
  const int B = x.shape(0), L = x.shape(1), hd = c.head_dim;
  auto lin = [&](const mx::array& in, const Proj& p) { return rows ? linear_rows(in, p) : linear(in, p); };
  mx::array q = lin(x, w.q);
  std::optional<mx::array> gate;
  if (c.attn_gate) {  // q_proj holds [query | gate] per head
    auto halves = mx::split(mx::reshape(q, {B, L, c.heads, 2 * hd}), 2, -1);
    q = halves[0];
    gate = mx::reshape(halves[1], {B, L, c.heads * hd});
  } else {
    q = mx::reshape(q, {B, L, c.heads, hd});
  }
  mx::array k = mx::reshape(lin(x, w.k), {B, L, c.kv_heads, hd});
  mx::array v = mx::reshape(lin(x, w.v), {B, L, c.kv_heads, hd});
  q = mx::transpose(mx::fast::rms_norm(q, w.q_norm, c.eps), {0, 2, 1, 3});
  k = mx::transpose(mx::fast::rms_norm(k, w.k_norm, c.eps), {0, 2, 1, 3});
  v = mx::transpose(v, {0, 2, 1, 3});
  // traditional=false is Hugging Face's rotate-half layout; dims < head_dim leaves the tail of each head as is.
  q = mx::fast::rope(q, c.rope_dims, false, c.rope_theta, 1.0f, offset);
  k = mx::fast::rope(k, c.rope_dims, false, c.rope_theta, 1.0f, offset);
  if (cache) std::tie(k, v) = cache_update(*cache, offset, k, v);
  // A single new token attends to everything cached, so it needs no mask.
  const float scale = float(std::pow(double(hd), -0.5));
  std::optional<mx::array> a_rows;
  if (rows) {  // each token on its own, over the keys up to it, as its decode step attends
    std::vector<mx::array> outs;
    const int kv = k.shape(2);
    for (int j = 0; j < L; ++j) {
      const int n = kv - L + j + 1;
      outs.push_back(mx::fast::scaled_dot_product_attention(
          mx::slice(q, {0, 0, j, 0}, {B, c.heads, j + 1, hd}), mx::slice(k, {0, 0, 0, 0}, {B, c.kv_heads, n, hd}),
          mx::slice(v, {0, 0, 0, 0}, {B, c.kv_heads, n, hd}), scale, ""));
    }
    a_rows = L == 1 ? outs[0] : mx::concatenate(outs, 2);
  }
  mx::array a = a_rows ? *a_rows : mx::fast::scaled_dot_product_attention(q, k, v, scale, L > 1 ? "causal" : "");
  a = mx::reshape(mx::transpose(a, {0, 2, 1, 3}), {B, L, c.heads * hd});
  if (gate) a = a * mx::sigmoid(*gate);
  return lin(a, w.o);
}

}  // namespace

mx::array linear(const mx::array& x, const mx::array& w) { return mx::matmul(x, mx::transpose(w)); }

static mx::array linear(const mx::array& x, const Quant& q) {
  return mx::quantized_matmul(x, q.w, q.scales, q.biases, true, q.group, q.bits);
}

mx::array linear(const mx::array& x, const Proj& p) {
  mx::array y = p.q ? linear(x, *p.q) : linear(x, p.w);
  if (!p.lora) return y;
  // As mlx-lm's LoRALinear: the low-rank path in float32, added back in the activations' dtype.
  mx::array z = mx::matmul(mx::matmul(x, mx::transpose(p.lora->a)), mx::transpose(p.lora->b));
  return y + mx::astype(z * mx::array(p.lora->scale, mx::float32), y.dtype());
}


Config load_config(const std::string& path) {
  using simdjson::dom::array, simdjson::dom::element;
  using sv = std::string_view;
  simdjson::dom::parser parser;
  element top = load_json(parser, path);
  sv type = get<sv>(top, "model_type", path);
  need(type == "qwen3" || type == "qwen3_5", path + ": model_type " + std::string(type) +
                                                 " is not implemented (only qwen3 and qwen3_5)");
  const bool q35 = type == "qwen3_5";
  const std::string where = q35 ? path + " text_config" : path;
  element j = q35 ? get<element>(top, "text_config", path) : top;
  auto flag = [&](const char* k) { return has_key(j, k) && get<bool>(j, k, where); };
  auto refuse = [&](bool bad, const std::string& what) { need(!bad, where + ": " + what + " is not implemented"); };
  refuse(flag("attention_bias"), "attention_bias");
  refuse(flag("use_sliding_window"), "use_sliding_window");
  refuse(get<sv>(j, "hidden_act", where) != "silu", "a hidden_act other than silu");
  refuse(has_key(j, "num_experts") && get<int64_t>(j, "num_experts", where) != 0, "mixture-of-experts");

  auto i = [&](const char* k) { return int(get<int64_t>(j, k, where)); };
  Config c;
  c.type = std::string(type);
  c.hidden = i("hidden_size");
  c.layers = i("num_hidden_layers");
  c.heads = i("num_attention_heads");
  c.kv_heads = i("num_key_value_heads");
  c.head_dim = i("head_dim");
  c.intermediate = i("intermediate_size");
  c.vocab = i("vocab_size");
  c.max_positions = i("max_position_embeddings");
  c.eps = float(get<double>(j, "rms_norm_eps", where));
  // mlx-lm reads tie_word_embeddings from the text config only (default false); refuse a top level that disagrees.
  c.tied = flag("tie_word_embeddings");
  need(!has_key(top, "tie_word_embeddings") || get<bool>(top, "tie_word_embeddings", path) == c.tied,
       path + ": tie_word_embeddings differs between text_config and the top level");
  element eos = get<element>(j, "eos_token_id", where);
  if (eos.is_array())
    for (element e : array(eos)) c.eos.push_back(int(int64_t(e)));
  else
    c.eos.push_back(int(get<int64_t>(j, "eos_token_id", where)));

  if (!q35) {
    refuse(has_key(j, "rope_scaling"), "rope_scaling");
    c.rope_theta = float(get<double>(j, "rope_theta", where));
    c.rope_dims = c.head_dim;
    c.linear.assign(c.layers, false);
  } else {
    // Text-only positions are the same on mrope's three axes, so its sections reduce to plain rope.
    element rp = get<element>(j, "rope_parameters", where);
    const std::string wr = where + " rope_parameters";
    refuse(get<sv>(rp, "rope_type", wr) != "default", "a rope_type other than default");
    c.rope_theta = float(get<double>(rp, "rope_theta", wr));
    c.rope_dims = int(c.head_dim * get<double>(rp, "partial_rotary_factor", wr));
    c.attn_gate = flag("attn_output_gate");
    refuse(has_key(j, "mlp_only_layers") && get<array>(j, "mlp_only_layers", where).size() != 0, "mlp_only_layers");
    const int interval = i("full_attention_interval");
    for (element t : get<array>(j, "layer_types", where)) {
      sv s = t;
      need(s == "linear_attention" || s == "full_attention", where + ": unknown layer type " + std::string(s));
      c.linear.push_back(s == "linear_attention");
    }
    need(int(c.linear.size()) == c.layers, where + ": layer_types has " + std::to_string(c.linear.size()) +
                                               " entries for " + std::to_string(c.layers) + " layers");
    for (int l = 0; l < c.layers; ++l)  // mlx-lm places the layers by the interval; both must agree
      need(c.linear[l] == ((l + 1) % interval != 0),
           where + ": layer_types disagrees with full_attention_interval at layer " + std::to_string(l));
    c.lin_k_heads = i("linear_num_key_heads");
    c.lin_v_heads = i("linear_num_value_heads");
    c.lin_k_dim = i("linear_key_head_dim");
    c.lin_v_dim = i("linear_value_head_dim");
    c.conv_kernel = i("linear_conv_kernel_dim");
    need(c.lin_v_heads % c.lin_k_heads == 0, where + ": linear value heads must be a multiple of key heads");
  }
  need(c.heads % c.kv_heads == 0, where + ": num_attention_heads must be a multiple of num_key_value_heads");
  return c;
}

Model make_model(const Config& c, std::unordered_map<std::string, mx::array> w) {
  auto take = [&](const std::string& name, mx::Shape shape) {
    auto it = w.find(name);
    need(it != w.end(), "missing weight " + name);
    need(it->second.shape() == shape, name + ": shape " + str(it->second.shape()) + ", want " + str(shape));
    mx::array a = it->second;
    w.erase(it);
    return a;
  };
  const int d = c.hidden, f = c.intermediate, hd = c.head_dim;
  const int q_out = c.heads * hd * (c.attn_gate ? 2 : 1), kv = c.kv_heads * hd;
  const int kd = c.lin_k_heads * c.lin_k_dim, vd = c.lin_v_heads * c.lin_v_dim, conv = 2 * kd + vd;
  auto proj = [&](const std::string& name, mx::Shape shape) { return Proj{take(name, shape), std::nullopt, std::nullopt}; };
  std::vector<Layer> layers;
  for (int l = 0; l < c.layers; ++l) {
    std::string p = "model.layers." + std::to_string(l) + ".";
    Layer layer{take(p + "input_layernorm.weight", {d}),
                take(p + "post_attention_layernorm.weight", {d}),
                proj(p + "mlp.gate_proj.weight", {f, d}),
                proj(p + "mlp.up_proj.weight", {f, d}),
                proj(p + "mlp.down_proj.weight", {d, f}),
                std::nullopt,
                std::nullopt};
    if (c.linear[l]) {
      std::string a = p + "linear_attn.";
      layer.delta = DeltaNet{proj(a + "in_proj_qkv.weight", {conv, d}),
                             proj(a + "in_proj_z.weight", {vd, d}),
                             proj(a + "in_proj_b.weight", {c.lin_v_heads, d}),
                             proj(a + "in_proj_a.weight", {c.lin_v_heads, d}),
                             take(a + "conv1d.weight", {conv, c.conv_kernel, 1}),
                             take(a + "dt_bias", {c.lin_v_heads}),
                             take(a + "A_log", {c.lin_v_heads}),
                             take(a + "norm.weight", {c.lin_v_dim}),
                             proj(a + "out_proj.weight", {d, vd})};
    } else {
      std::string a = p + "self_attn.";
      layer.attn = Attention{proj(a + "q_proj.weight", {q_out, d}), proj(a + "k_proj.weight", {kv, d}),
                             proj(a + "v_proj.weight", {kv, d}), proj(a + "o_proj.weight", {d, c.heads * hd}),
                             take(a + "q_norm.weight", {hd}), take(a + "k_norm.weight", {hd})};
    }
    layers.push_back(std::move(layer));
  }
  mx::array embed = take("model.embed_tokens.weight", {c.vocab, d}), norm = take("model.norm.weight", {d});
  mx::array lm_head = c.tied ? embed : take("lm_head.weight", {c.vocab, d});
  Model m{c, embed, norm, lm_head, std::move(layers), std::nullopt};
  need(w.empty(), "unexpected weight " + (w.empty() ? std::string() : w.begin()->first));
  return m;
}

Model load_model(const std::string& dir) {
  Config cfg = load_config(dir + "/config.json");
  const bool q35 = cfg.type == "qwen3_5";
  const std::string index_path = dir + "/model.safetensors.index.json";
  simdjson::dom::parser parser;
  simdjson::dom::element index = load_json(parser, index_path);
  std::set<std::string> names, files;
  bool has_mtp = false;
  for (auto [name, file] : get<simdjson::dom::object>(index, "weight_map", index_path)) {
    std::string n(name);
    has_mtp = has_mtp || n.starts_with("mtp.");
    if (q35 && (n.starts_with("mtp.") || n.starts_with("model.visual."))) continue;  // text decoder only
    names.insert(n);
    files.emplace(std::string_view(file));
  }
  const std::string text_prefix = "model.language_model.";
  std::unordered_map<std::string, mx::array> w;
  for (const auto& f : files) {
    std::unordered_map<std::string, mx::array> shard;
    try {
      shard = mx::load_safetensors(dir + "/" + f).first;
    } catch (const std::exception& e) {
      throw std::runtime_error(dir + "/" + f + ": " + e.what());
    }
    for (auto& [name, a] : shard) {
      if (!names.count(name)) continue;  // vision tower and multi-token-prediction head: never read
      std::string n = name.starts_with(text_prefix) ? "model." + name.substr(text_prefix.size()) : name;
      w.insert_or_assign(n, a);
      names.erase(name);
    }
  }
  need(names.empty(), index_path + " lists " + (names.empty() ? std::string() : *names.begin()) +
                          ", which its shard does not hold");
  // bf16 throughout, except Qwen3.5's A_log and gated-norm weight, stored float32 and left so as mlx-lm does.
  for (const auto& [n, a] : w) {
    bool f32 = ends_with(n, ".linear_attn.A_log") || ends_with(n, ".linear_attn.norm.weight");
    mx::Dtype want = f32 ? mx::float32 : mx::bfloat16;
    need(a.dtype() == want, dir + ": " + n + " is " + str(a.dtype()) + ", want " + str(want));
  }
  if (q35) {
    // mlx-lm's sanitize for Hugging Face checkpoints: conv1d [C, 1, K] -> MLX's [C, K, 1], and the RMSNorm weights
    // stored as offsets from 1 get the 1 added back (in their own dtype). Its gated norm is stored as is.
    bool raw_conv = false;
    for (auto& [n, a] : w) raw_conv = raw_conv || (ends_with(n, "conv1d.weight") && a.shape(-1) != 1);
    for (auto& [n, a] : w) {
      if (ends_with(n, "conv1d.weight") && a.shape(-1) != 1) a = mx::moveaxis(a, 2, 1);
      bool shifted = ends_with(n, ".input_layernorm.weight") || ends_with(n, ".post_attention_layernorm.weight") ||
                     n == "model.norm.weight" || ends_with(n, ".q_norm.weight") || ends_with(n, ".k_norm.weight");
      if ((has_mtp || raw_conv) && shifted && a.ndim() == 1) a = a + mx::array(1.0f, a.dtype());
    }
  }
  try {
    Model m = make_model(cfg, std::move(w));
    mx::eval(parameters(m));
    return m;
  } catch (const std::runtime_error& e) {
    throw std::runtime_error(dir + ": " + e.what());
  }
}

std::vector<mx::array> parameters(const Model& m) {
  std::vector<mx::array> p{m.embed, m.norm};
  if (!m.cfg.tied) p.push_back(m.lm_head);
  for (const Layer& l : m.layers) {
    p.insert(p.end(), {l.in_norm, l.post_norm});
    if (l.attn) p.insert(p.end(), {l.attn->q_norm, l.attn->k_norm});
    if (l.delta) p.insert(p.end(), {l.delta->conv, l.delta->dt_bias, l.delta->a_log, l.delta->norm});
    for (const auto& [name, proj] : projections(l)) {
      p.push_back(proj->w);
      if (proj->lora) p.insert(p.end(), {proj->lora->a, proj->lora->b});
      if (proj->q) p.insert(p.end(), {proj->q->w, proj->q->scales, proj->q->biases});
    }
  }
  if (m.head_q) p.insert(p.end(), {m.head_q->w, m.head_q->scales, m.head_q->biases});
  return p;
}

std::vector<std::pair<std::string, Proj*>> projections(Layer& l) {
  std::vector<std::pair<std::string, Proj*>> p;
  if (l.attn)
    p = {{"self_attn.q_proj", &l.attn->q}, {"self_attn.k_proj", &l.attn->k}, {"self_attn.v_proj", &l.attn->v},
         {"self_attn.o_proj", &l.attn->o}};
  if (l.delta)
    p = {{"linear_attn.in_proj_qkv", &l.delta->in_qkv}, {"linear_attn.in_proj_z", &l.delta->in_z},
         {"linear_attn.in_proj_b", &l.delta->in_b},     {"linear_attn.in_proj_a", &l.delta->in_a},
         {"linear_attn.out_proj", &l.delta->out}};
  p.insert(p.end(), {{"mlp.gate_proj", &l.gate}, {"mlp.up_proj", &l.up}, {"mlp.down_proj", &l.down}});
  return p;
}

std::vector<std::pair<std::string, const Proj*>> projections(const Layer& l) {
  std::vector<std::pair<std::string, const Proj*>> out;
  for (const auto& [name, p] : projections(const_cast<Layer&>(l))) out.emplace_back(name, p);
  return out;
}

std::vector<mx::array> adapter_arrays(const Layer& l) {
  std::vector<mx::array> out;
  for (const auto& [name, p] : projections(l))
    if (p->lora) out.insert(out.end(), {p->lora->a, p->lora->b});
  return out;
}

void set_adapter_arrays(Layer& l, const mx::array* arrays) {
  for (const auto& [name, p] : projections(l))
    if (p->lora) p->lora->a = *arrays++, p->lora->b = *arrays++;
}

std::vector<mx::array> cache_arrays(const KVCache& cache) {
  std::vector<mx::array> out;
  for (const LayerCache& c : cache.layers)
    for (const auto* a : {&c.k, &c.v, &c.conv, &c.state})
      if (*a) out.push_back(**a);
  return out;
}

mx::array block(const Layer& w, const Config& c, const mx::array& h, int offset, LayerCache* cache, bool train,
                RowStates* rows, int layer) {
  auto lin = [&](const mx::array& in, const Proj& p) { return rows ? linear_rows(in, p) : linear(in, p); };
  mx::array x = mx::fast::rms_norm(h, w.in_norm, c.eps);
  mx::array r = h + (w.delta ? deltanet(*w.delta, c, x, cache, train, rows, layer)
                             : attention(*w.attn, c, x, offset, cache, rows != nullptr));
  x = mx::fast::rms_norm(r, w.post_norm, c.eps);
  return r + lin(swiglu(lin(x, w.gate), lin(x, w.up)), w.down);
}

mx::array forward(const Model& m, const mx::array& tokens, KVCache* cache, bool last_only, RowStates* rows) {
  if (rows && !cache) throw std::runtime_error("forward: a pass of decode steps needs a cache");
  const Config& c = m.cfg;
  const int B = tokens.shape(0), L = tokens.shape(1);
  const int offset = cache ? cache->offset : 0;
  if (cache) cache->layers.resize(m.layers.size());

  mx::array h = mx::take(m.embed, tokens, 0);
  for (size_t i = 0; i < m.layers.size(); ++i)
    h = block(m.layers[i], c, h, offset, cache ? &cache->layers[i] : nullptr, false, rows, int(i));
  if (cache) cache->offset += L;
  if (last_only) h = mx::slice(h, {0, L - 1, 0}, {B, L, c.hidden});
  h = mx::fast::rms_norm(h, m.norm, c.eps);
  if (rows && m.head_q) throw std::runtime_error("forward: exact decode steps run bf16 weights, not quantized ones");
  return mx::astype(rows      ? linear_rows(h, Proj{m.lm_head, std::nullopt, std::nullopt})
                    : m.head_q ? linear(h, *m.head_q)
                               : linear(h, m.lm_head),
                    mx::float32);
}

mx::array hidden(const Model& m, const mx::array& tokens, bool checkpoint) {
  const Config& c = m.cfg;
  mx::array h = mx::take(m.embed, tokens, 0);
  for (const Layer& w : m.layers) {
    if (!checkpoint) {
      h = block(w, c, h, 0, nullptr, true);
      continue;
    }
    std::vector<mx::array> in{h};
    for (const mx::array& a : adapter_arrays(w)) in.push_back(a);
    auto fn = [w, c](const std::vector<mx::array>& x) {
      Layer l = w;
      set_adapter_arrays(l, x.data() + 1);
      return std::vector<mx::array>{block(l, c, x[0], 0, nullptr, true)};
    };
    h = mx::checkpoint(fn)(in)[0];
  }
  return mx::fast::rms_norm(h, m.norm, c.eps);
}

mx::array logits_at(const Model& m, const mx::array& h, const mx::array& positions) {
  mx::array rows = mx::take(mx::reshape(h, {-1, m.cfg.hidden}), positions, 0);
  return mx::astype(m.head_q ? linear(rows, *m.head_q) : linear(rows, m.lm_head), mx::float32);
}

int quantize_weights(Model& m, int bits, int group) {
  auto quant = [&](const mx::array& w) {
    const std::vector<mx::array> q = mx::quantize(w, group, bits, "affine");
    Quant out{q[0], q[1], q[2], group, bits};
    mx::eval({out.w, out.scales, out.biases});
    return out;
  };
  int n = 0;
  for (Layer& l : m.layers)
    for (auto& [name, p] : projections(l)) {
      if (p->lora) throw std::runtime_error("quantize: " + name + " has an adapter: merge it first (--merge-lora)");
      if (name == "linear_attn.in_proj_b" || name == "linear_attn.in_proj_a") continue;
      p->q = quant(p->w);
      p->w = mx::zeros({0}, p->w.dtype());
      ++n;
    }
  m.head_q = quant(m.lm_head);
  if (!m.cfg.tied) m.lm_head = mx::zeros({0}, m.lm_head.dtype());
  return n;
}
