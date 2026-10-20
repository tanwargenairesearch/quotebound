#include "lora.h"

#include <cmath>
#include <filesystem>
#include <set>
#include <stdexcept>

#include "files.h"
#include "json.h"

namespace {

std::string short_name(const std::string& name) { return name.substr(name.rfind('.') + 1); }

std::string layer_prefix(size_t l) { return "model.layers." + std::to_string(l) + "."; }

}  // namespace

const std::vector<std::string>& default_targets() {
  static const std::vector<std::string> t{"q_proj",   "k_proj",    "v_proj",    "o_proj",  "in_proj_qkv",
                                          "in_proj_z", "out_proj", "gate_proj", "up_proj", "down_proj"};
  return t;
}

void add_lora(Model& m, const LoraConfig& c) {
  if (c.rank < 1) throw std::runtime_error("LoRA rank must be at least 1");
  // Any projection name of either architecture is a valid target; a model without it (Qwen3 has no in_proj_qkv)
  // simply has nothing there to adapt.
  static const std::set<std::string> known{"q_proj",    "k_proj",    "v_proj",   "o_proj",
                                           "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a",
                                           "out_proj",  "gate_proj", "up_proj",  "down_proj"};
  const std::set<std::string> targets(c.targets.begin(), c.targets.end());
  for (const std::string& t : targets)
    if (!known.count(t)) throw std::runtime_error("LoRA target " + t + " is not a projection name");
  std::vector<Proj*> slots;
  for (Layer& l : m.layers)
    for (auto& [name, p] : projections(l))
      if (targets.count(short_name(name))) slots.push_back(p);
  if (slots.empty()) throw std::runtime_error("no projection of this model is a LoRA target");
  // One key per targeted projection, split off the seed in model order.
  mx::array keys = mx::random::split(mx::random::key(c.seed), int(slots.size()));
  for (size_t i = 0; i < slots.size(); ++i) {
    const int out = slots[i]->w.shape(0), in = slots[i]->w.shape(1);
    const float bound = 1.0f / std::sqrt(float(in));
    mx::array a = mx::random::uniform(-bound, bound, {c.rank, in}, mx::float32, mx::take(keys, int(i), 0));
    slots[i]->lora = Lora{a, mx::zeros({out, c.rank}, mx::float32), c.alpha / float(c.rank)};
  }
  mx::eval(adapter_arrays(m));
}

std::vector<mx::array> adapter_arrays(const Model& m) {
  std::vector<mx::array> out;
  for (const Layer& l : m.layers)
    for (const mx::array& a : adapter_arrays(l)) out.push_back(a);
  return out;
}

std::vector<std::string> adapter_names(const Model& m) {
  std::vector<std::string> out;
  for (size_t l = 0; l < m.layers.size(); ++l)
    for (const auto& [name, p] : projections(m.layers[l]))
      if (p->lora) out.insert(out.end(), {layer_prefix(l) + name + ".lora_a", layer_prefix(l) + name + ".lora_b"});
  return out;
}

void set_adapter_arrays(Model& m, const std::vector<mx::array>& arrays) {
  size_t at = 0;
  for (Layer& l : m.layers) {
    const size_t n = adapter_arrays(l).size();
    if (at + n > arrays.size()) throw std::runtime_error("set_adapter_arrays: too few arrays");
    set_adapter_arrays(l, arrays.data() + at);
    at += n;
  }
  if (at != arrays.size()) throw std::runtime_error("set_adapter_arrays: too many arrays");
}

void save_adapters(const std::string& dir, const Model& m, const AdapterInfo& info) {
  std::filesystem::create_directories(dir);
  std::unordered_map<std::string, mx::array> named;
  const std::vector<mx::array> arrays = adapter_arrays(m);
  const std::vector<std::string> names = adapter_names(m);
  for (size_t i = 0; i < arrays.size(); ++i) named.insert_or_assign(names[i], arrays[i]);
  // mx::save_safetensors appends ".safetensors" to a name without it, so the temporary name keeps the suffix.
  const std::string tmp = dir + "/adapters.tmp.safetensors";
  mx::save_safetensors(tmp, named);
  const std::string sha = sha256_file(tmp);
  commit_file(tmp, dir + "/adapters.safetensors");
  const LoraConfig& c = info.lora;
  std::string targets;
  for (const std::string& t : c.targets) targets += (targets.empty() ? "" : ", ") + json_quote(t);
  char loss[32] = "null", head[256];
  if (std::isfinite(info.dev_loss)) std::snprintf(loss, sizeof loss, "%.9g", info.dev_loss);
  std::snprintf(head, sizeof head, "{\"rank\": %d, \"alpha\": %.9g, \"seed\": %llu, \"step\": %d, \"dev_loss\": %s, ",
                c.rank, double(c.alpha), (unsigned long long)c.seed, info.step, loss);
  write_file(dir + "/adapter_config.json",
             std::string(head) + "\"targets\": [" + targets + "], \"model_type\": " + json_quote(m.cfg.type) +
                 ", \"layers\": " + std::to_string(m.cfg.layers) + ", \"hidden\": " +
                 std::to_string(m.cfg.hidden) + ", \"adapters_sha256\": " + json_quote(sha) +
                 ",\n \"base\": " + base_json(info.base) + ",\n \"run\": " + info.run + "}\n");
}

AdapterInfo load_adapters(const std::string& dir, Model& m, const BaseModel& base) {
  const std::string path = dir + "/adapter_config.json";
  simdjson::dom::parser parser;
  simdjson::dom::element j = load_json(parser, path);
  AdapterInfo info;
  LoraConfig& c = info.lora;
  c.rank = int(get<int64_t>(j, "rank", path));
  c.alpha = float(get<double>(j, "alpha", path));
  c.seed = uint64_t(get<int64_t>(j, "seed", path));
  for (simdjson::dom::element t : get<simdjson::dom::array>(j, "targets", path))
    c.targets.emplace_back(std::string_view(t));
  info.step = int(get<int64_t>(j, "step", path));
  if (has_key(j, "dev_loss")) info.dev_loss = get<double>(j, "dev_loss", path);
  info.base = parse_base(get<simdjson::dom::element>(j, "base", path), path + " base");
  info.run = simdjson::minify(get<simdjson::dom::element>(j, "run", path));
  if (get<std::string_view>(j, "model_type", path) != m.cfg.type || get<int64_t>(j, "layers", path) != m.cfg.layers ||
      get<int64_t>(j, "hidden", path) != m.cfg.hidden)
    throw std::runtime_error(path + ": the adapters are for another model (model_type, layers or hidden differ)");
  if (info.base.chat_template != base.chat_template)
    throw std::runtime_error(path + ": the adapters were trained under chat template " + info.base.chat_template +
                             "; this model's is " + base.chat_template);
  if (!(info.base == base))
    throw std::runtime_error(path + ": the adapters were trained on another base checkpoint (config.json, the "
                             "safetensors index or a shard's name or size differs)");
  const std::string file = dir + "/adapters.safetensors";
  if (sha256_file(file) != get<std::string_view>(j, "adapters_sha256", path))
    throw std::runtime_error(file + " is not the file " + path + " was written with (sha256 differs)");
  add_lora(m, c);
  std::unordered_map<std::string, mx::array> saved = mx::load_safetensors(file).first;
  std::vector<mx::array> arrays = adapter_arrays(m);
  const std::vector<std::string> names = adapter_names(m);
  for (size_t i = 0; i < names.size(); ++i) {
    auto it = saved.find(names[i]);
    if (it == saved.end()) throw std::runtime_error(file + ": missing " + names[i]);
    if (it->second.shape() != arrays[i].shape() || it->second.dtype() != mx::float32)
      throw std::runtime_error(file + ": " + names[i] + " has the wrong shape or dtype");
    arrays[i] = it->second;
    saved.erase(it);
  }
  if (!saved.empty()) throw std::runtime_error(file + ": unexpected " + saved.begin()->first);
  set_adapter_arrays(m, arrays);
  mx::eval(arrays);
  return info;
}

int merge_lora(Model& m) {
  int merged = 0;
  for (Layer& l : m.layers)
    for (auto& [name, p] : projections(l)) {
      if (!p->lora) continue;
      const Lora& a = *p->lora;
      const mx::array delta = mx::matmul(a.b, a.a) * mx::array(a.scale, mx::float32);  // [out, in], float32
      p->w = mx::astype(mx::astype(p->w, mx::float32) + delta, p->w.dtype());
      p->lora.reset();
      mx::eval(p->w);  // one projection's float32 temporaries at a time
      ++merged;
    }
  return merged;
}
