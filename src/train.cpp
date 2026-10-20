#include "train.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <filesystem>
#include <numeric>
#include <stdexcept>

#include "files.h"
#include "generate.h"
#include "json.h"
#include "loss.h"

namespace fs = std::filesystem;

namespace {

using clock_type = std::chrono::steady_clock;

double ms_since(clock_type::time_point t0) {
  return std::chrono::duration<double, std::milli>(clock_type::now() - t0).count();
}

std::string format(const char* f, ...) {
  va_list ap, again;
  va_start(ap, f);
  va_copy(again, ap);
  std::string s(size_t(std::vsnprintf(nullptr, 0, f, ap)), '\0');
  va_end(ap);
  std::vsnprintf(s.data(), s.size() + 1, f, again);
  va_end(again);
  return s;
}

void emit(const std::vector<std::FILE*>& log, const std::string& line) {
  for (std::FILE* f : log) {
    std::fputs(line.c_str(), f);
    std::fputc('\n', f);
    std::fflush(f);
  }
}

std::string json_list(const std::vector<std::string>& v) {
  std::string out = "[";
  for (const std::string& s : v) out += (out.size() > 1 ? ", " : "") + json_quote(s);
  return out + "]";
}

std::string json_counts(const std::map<std::string, int>& m) {
  std::string out = "{";
  for (const auto& [k, n] : m) out += (out.size() > 1 ? ", " : "") + json_quote(k) + ": " + std::to_string(n);
  return out + "}";
}

// Up to 20 ids, then how many more.
std::string some(const std::vector<std::string>& ids) {
  std::string out;
  for (size_t i = 0; i < ids.size() && i < 20; ++i) out += (i ? ", " : "") + ids[i];
  return ids.size() > 20 ? out + " and " + std::to_string(ids.size() - 20) + " more" : out;
}

// The arguments that decide the run's arithmetic. Paths, eval_every, save_every, gen_dev, keep and patience may
// change on a resume; what the paths hold is checked by content instead.
std::string arithmetic_args(const TrainArgs& a) {
  std::string targets;
  for (const std::string& t : a.lora.targets) targets += t + ",";
  return format("rank=%d alpha=%.9g targets=%s lr=%.9g steps=%d epochs=%d batch_tokens=%d max_len=%d seed=%llu "
                "warmup=%d dev_max=%d beta1=%.9g beta2=%.9g eps=%.9g weight_decay=%.9g clip=%.9g",
                a.lora.rank, double(a.lora.alpha), targets.c_str(), double(a.lr), a.steps, a.epochs, a.batch_tokens,
                a.max_len, (unsigned long long)a.seed, a.warmup, a.dev_max, double(a.adamw.beta1),
                double(a.adamw.beta2), double(a.adamw.eps), double(a.adamw.weight_decay), double(a.adamw.clip));
}

// A resume must match this: the arguments above, the rendered training and dev rows, and the base checkpoint.
std::string run_key(const Trainer& t) {
  return arithmetic_args(t.args) + " train_sha256=" + t.train_sha256 + " dev_sha256=" + t.dev_sha256 +
         " base=" + base_json(t.base);
}

std::string args_json(const TrainArgs& a) {
  return format("{\"model\": %s, \"train\": %s, \"dev\": %s, \"out\": %s, \"rank\": %d, \"alpha\": %.9g, "
                "\"targets\": %s, \"lr\": %.9g, \"steps\": %d, \"epochs\": %d, \"batch_tokens\": %d, \"max_len\": %d, "
                "\"seed\": %llu, \"eval_every\": %d, \"save_every\": %d, \"warmup\": %d, \"dev_max\": %d, "
                "\"gen_dev\": %d, \"keep\": %d, \"patience\": %d, \"beta1\": %.9g, \"beta2\": %.9g, \"eps\": %.9g, "
                "\"weight_decay\": %.9g, \"clip\": %.9g}",
                json_quote(a.model_dir).c_str(), json_quote(a.train_path).c_str(), json_quote(a.dev_path).c_str(),
                json_quote(a.out_dir).c_str(), a.lora.rank, double(a.lora.alpha), json_list(a.lora.targets).c_str(),
                double(a.lr), a.steps, a.epochs, a.batch_tokens, a.max_len, (unsigned long long)a.seed, a.eval_every,
                a.save_every, a.warmup, a.dev_max, a.gen_dev, a.keep, a.patience, double(a.adamw.beta1),
                double(a.adamw.beta2), double(a.adamw.eps), double(a.adamw.weight_decay), double(a.adamw.clip));
}

void next_batch_plan(Trainer& t) {
  if (t.cursor < t.plan.size()) return;
  t.epoch += 1;
  t.plan = plan_epoch(t.train, t.args.batch_tokens, t.args.seed, t.epoch);
  t.cursor = 0;
}

AdapterInfo adapter_info(const Trainer& t, double dev_loss) {
  return {t.args.lora, t.base, t.step, dev_loss, run_record(t)};
}

// The newest `keep` step-N directories stay.
void prune_steps(const fs::path& out, int keep) {
  if (keep <= 0) return;
  std::vector<long> steps;
  for (const fs::directory_entry& e : fs::directory_iterator(out)) {
    const std::string name = e.path().filename().string();
    if (e.is_directory() && name.starts_with("step-") && name.size() > 5 &&
        name.find_first_not_of("0123456789", 5) == std::string::npos)
      steps.push_back(std::stol(name.substr(5)));
  }
  std::sort(steps.rbegin(), steps.rend());
  for (size_t i = size_t(keep); i < steps.size(); ++i) fs::remove_all(out / ("step-" + std::to_string(steps[i])));
}

// A state file that cannot be read back, as against one that reads but belongs to another run.
struct Unreadable : std::runtime_error {
  using std::runtime_error::runtime_error;
};

void read_state(Trainer& t, const std::string& file) {
  std::unordered_map<std::string, mx::array> arrays;
  std::unordered_map<std::string, std::string> meta;
  auto field = [&](const std::string& k) {
    auto it = meta.find(k);
    if (it == meta.end()) throw Unreadable(file + ": no " + k);
    return it->second;
  };
  auto array = [&](const std::string& k) {
    auto it = arrays.find(k);
    if (it == arrays.end()) throw Unreadable(file + ": no " + k);
    return it->second;
  };
  const std::vector<std::string> names = adapter_names(t.model);
  std::vector<mx::array> params, m, v;
  try {
    std::tie(arrays, meta) = mx::load_safetensors(file);
    for (const std::string& n : names) {
      params.push_back(array("p." + n));
      m.push_back(array("m." + n));
      v.push_back(array("v." + n));
    }
    std::vector<mx::array> all = params;
    all.insert(all.end(), m.begin(), m.end());
    all.insert(all.end(), v.begin(), v.end());
    all.push_back(array("plan")), all.push_back(array("plan_sizes"));
    mx::eval(all);
  } catch (const Unreadable&) {
    throw;
  } catch (const std::exception& e) {
    throw Unreadable(file + ": " + e.what());
  }
  if (field("run") != run_key(t))
    throw std::runtime_error(file + " is from a run with other arguments, data or base:\n  saved: " + field("run") +
                             "\n  now:   " + run_key(t));
  const std::vector<mx::array> now = adapter_arrays(t.model);
  for (size_t i = 0; i < names.size(); ++i)
    for (const mx::array* a : {&params[i], &m[i], &v[i]})
      if (a->shape() != now[i].shape() || a->dtype() != mx::float32)
        throw std::runtime_error(file + ": " + names[i] + " has the wrong shape or dtype");
  const int step = std::stoi(field("step")), total = std::stoi(field("total_steps"));
  const int opt_step = std::stoi(field("opt_step"));
  const size_t cursor = std::stoul(field("cursor"));
  // The saved plan indexes the training set by position: it must fit this one exactly.
  mx::array flat = mx::astype(array("plan"), mx::int32), sizes = mx::astype(array("plan_sizes"), mx::int32);
  mx::eval(flat, sizes);
  std::vector<std::vector<int>> plan;
  size_t at = 0;
  for (size_t i = 0; i < sizes.size(); ++i) {
    const int n = sizes.data<int32_t>()[i];
    if (n < 1 || at + size_t(n) > flat.size())
      throw std::runtime_error(file + ": the saved batch plan's sizes do not add up to its " +
                               std::to_string(flat.size()) + " indices");
    plan.emplace_back(flat.data<int32_t>() + at, flat.data<int32_t>() + at + n);
    at += size_t(n);
  }
  if (at != flat.size())
    throw std::runtime_error(file + ": the saved batch plan's sizes do not add up to its " +
                             std::to_string(flat.size()) + " indices");
  for (const std::vector<int>& b : plan)
    for (int i : b)
      if (i < 0 || size_t(i) >= t.train.size())
        throw std::runtime_error(file + ": the saved batch plan names example " + std::to_string(i) +
                                 " of a training set of " + std::to_string(t.train.size()));
  if (cursor > plan.size() || step > total || total != t.total_steps || opt_step != step)
    throw std::runtime_error(file + ": step " + std::to_string(step) + " of " + std::to_string(total) +
                             ", batch " + std::to_string(cursor) + " of " + std::to_string(plan.size()) +
                             ", optimizer step " + std::to_string(opt_step) + " do not fit this run of " +
                             std::to_string(t.total_steps) + " steps");
  set_adapter_arrays(t.model, params);
  t.opt.m = m, t.opt.v = v, t.opt.step = opt_step;
  t.step = step;
  t.epoch = std::stoi(field("epoch"));
  t.cursor = cursor;
  t.plan = std::move(plan);
  t.best_dev = std::strtod(field("best_dev").c_str(), nullptr);
  t.best_step = std::stoi(field("best_step"));
  t.stale_evals = std::stoi(field("stale_evals"));
}

}  // namespace

void check_args(const TrainArgs& a) {
  if ((a.steps > 0) == (a.epochs > 0)) throw std::runtime_error("give exactly one of --steps and --epochs");
  if (a.max_len > a.batch_tokens)
    throw std::runtime_error("--max-len " + std::to_string(a.max_len) + " exceeds --batch-tokens " +
                             std::to_string(a.batch_tokens) + ": a row could never fit a batch");
  if (a.gen_dev > a.dev_max)
    throw std::runtime_error("--gen-dev " + std::to_string(a.gen_dev) + " exceeds --dev-max " +
                             std::to_string(a.dev_max));
}

Trainer make_trainer(Model m, Dataset train, Dataset dev, const TrainArgs& a, const BaseModel& base,
                     const Tokenizer* tok) {
  check_args(a);
  if (train.examples.empty()) throw std::runtime_error("no training examples");
  if (!dev.too_long.empty())
    throw std::runtime_error("dev rows longer than --max-len " + std::to_string(a.max_len) +
                             " (the dev loss must cover every row): " + some(dev.too_long));
  if (!dev.no_answer.empty())
    throw std::runtime_error("dev rows with nothing to score (no assistant answer): " + some(dev.no_answer));
  if (dev.examples.empty()) throw std::runtime_error("the dev set is empty");
  if (a.gen_dev > 0 && !tok) throw std::runtime_error("dev generations need a tokenizer");
  Trainer t{.args = a, .base = base, .tok = tok, .model = std::move(m)};
  t.args.lora.seed = a.seed;
  t.train_sha256 = examples_sha256(train.examples);
  t.dev_sha256 = examples_sha256(dev.examples);
  t.dev_rows = dev.examples.size();
  t.train = std::move(train.examples);
  for (int i : seeded_subset(dev.examples.size(), size_t(a.dev_max), a.seed)) t.dev.push_back(dev.examples[i]);
  train.examples.clear(), dev.examples.clear();
  t.train_meta = std::move(train), t.dev_meta = std::move(dev);
  add_lora(t.model, t.args.lora);
  t.opt = adamw_init(adapter_arrays(t.model));
  // An epoch's batch count depends on its shuffle, so --epochs counts each epoch's own.
  t.total_steps = a.steps;
  for (int e = 0; e < a.epochs; ++e) t.total_steps += int(plan_epoch(t.train, a.batch_tokens, a.seed, e).size());
  t.warmup = a.warmup >= 0 ? a.warmup : t.total_steps / 20;
  t.plan = plan_epoch(t.train, a.batch_tokens, a.seed, 0);
  return t;
}

LossAndGrads adapter_loss_and_grads(const Model& m, const Batch& b) {
  const float n = float(b.positions.shape(0));
  auto mean_loss = [&](const std::vector<mx::array>& p) {
    Model mm = m;
    set_adapter_arrays(mm, p);
    return std::vector<mx::array>{loss_sum(mm, b) / mx::array(n)};
  };
  const std::vector<mx::array> params = adapter_arrays(m);
  std::vector<int> argnums(params.size());
  std::iota(argnums.begin(), argnums.end(), 0);
  auto [loss, grads] = mx::value_and_grad(mean_loss, argnums)(params);
  std::vector<mx::array> out = grads;
  out.push_back(loss[0]);
  mx::eval(out);
  mx::array norm = global_norm(grads);
  mx::eval(norm);
  return {loss[0], norm, grads};
}

void apply_gradients(Trainer& t, const LossAndGrads& g, float lr) {
  const float loss = g.loss.item<float>(), norm = g.norm.item<float>();
  if (!std::isfinite(loss) || !std::isfinite(norm))
    throw std::runtime_error(format("step %d: loss %g, gradient norm %g: not finite; the adapters and optimizer "
                                    "stay at step %d",
                                    t.step + 1, double(loss), double(norm), t.step));
  const std::vector<mx::array> next = adamw_update(adapter_arrays(t.model), g.grads, g.norm, t.opt, t.args.adamw, lr);
  set_adapter_arrays(t.model, next);
  std::vector<mx::array> out = next;
  out.insert(out.end(), t.opt.m.begin(), t.opt.m.end());
  out.insert(out.end(), t.opt.v.begin(), t.opt.v.end());
  mx::eval(out);
}

StepStats train_step(Trainer& t) {
  auto t0 = clock_type::now();
  mx::reset_peak_memory();
  next_batch_plan(t);
  const Batch b = make_batch(t.train, t.plan[t.cursor]);
  const LossAndGrads g = adapter_loss_and_grads(t.model, b);
  const float lr = lr_at(t.step, t.total_steps, t.warmup, t.args.lr);
  apply_gradients(t, g, lr);
  t.cursor += 1;
  t.step += 1;
  return {g.loss.item<float>(), lr,          g.norm.item<float>(), int(b.positions.shape(0)), b.real_tokens,
          ms_since(t0),         mx::get_peak_memory()};
}

std::pair<double, long> eval_loss(const Model& m, const std::vector<Example>& ex, int batch_tokens) {
  double sum = 0;
  long n = 0;
  for (const std::vector<int>& rows : plan_eval(ex, batch_tokens)) {
    const Batch b = make_batch(ex, rows);
    mx::array s = loss_sum(m, b, false);
    mx::eval(s);
    sum += double(s.item<float>());
    n += b.positions.shape(0);
  }
  return {sum, n};
}

std::vector<std::string> dev_generations(const Trainer& t) {
  std::vector<std::string> out;
  if (t.args.gen_dev <= 0) return out;
  std::vector<int> stop = t.model.cfg.eos;
  stop.push_back(token_id(*t.tok, "<|im_end|>"));
  stop.push_back(token_id(*t.tok, "<|endoftext|>"));
  for (size_t i = 0; i < t.dev.size() && i < size_t(t.args.gen_dev); ++i) {
    const Example& e = t.dev[i];
    const std::vector<int> prompt(e.ids.begin(), e.ids.begin() + e.prompt_len);
    const Generation g = greedy(t.model, prompt, 2 * int(e.ids.size() - size_t(e.prompt_len)) + 32, stop);
    const bool stopped = !g.ids.empty() && std::find(stop.begin(), stop.end(), g.ids.back()) != stop.end();
    const std::string text = generation_text(*t.tok, g.ids, stopped);
    out.push_back("{\"id\": " + json_quote(e.id) + ", \"output\": " + json_quote(text) +
                  ", \"truncated\": " + (stopped ? "false" : "true") + "}");
  }
  return out;
}

std::string run_record(const Trainer& t) {
  std::vector<std::string> scored, generated;
  for (size_t i = 0; i < t.dev.size(); ++i) {
    scored.push_back(t.dev[i].id);
    if (i < size_t(std::max(t.args.gen_dev, 0))) generated.push_back(t.dev[i].id);
  }
  return format("{\"mlx\": \"%s\", \"args\": %s,\n \"base\": %s,\n \"train\": {\"sha256\": \"%s\", \"examples\": %zu, "
                "\"too_long\": %s, \"no_answer\": %s, \"system_prompts_sha256\": %s},\n \"dev\": {\"sha256\": \"%s\", "
                "\"examples\": %zu, \"scored\": %s, \"generated\": %s, \"system_prompts_sha256\": %s}}",
                mx::version(), args_json(t.args).c_str(), base_json(t.base).c_str(), t.train_sha256.c_str(),
                t.train.size(), json_list(t.train_meta.too_long).c_str(), json_list(t.train_meta.no_answer).c_str(),
                json_counts(t.train_meta.system_prompts).c_str(), t.dev_sha256.c_str(),
                t.dev_rows, json_list(scored).c_str(), json_list(generated).c_str(),
                json_counts(t.dev_meta.system_prompts).c_str());
}

void save_checkpoint(const Trainer& t) {
  const std::string dir = t.args.out_dir;
  fs::create_directories(dir);
  const std::vector<mx::array> params = adapter_arrays(t.model);
  const std::vector<std::string> names = adapter_names(t.model);
  std::unordered_map<std::string, mx::array> arrays;
  for (size_t i = 0; i < params.size(); ++i) {
    arrays.insert_or_assign("p." + names[i], params[i]);
    arrays.insert_or_assign("m." + names[i], t.opt.m[i]);
    arrays.insert_or_assign("v." + names[i], t.opt.v[i]);
  }
  std::vector<int32_t> flat, sizes;
  for (const auto& batch : t.plan) {
    sizes.push_back(int32_t(batch.size()));
    flat.insert(flat.end(), batch.begin(), batch.end());
  }
  arrays.insert_or_assign("plan", mx::array(flat.data(), {int(flat.size())}, mx::int32));
  arrays.insert_or_assign("plan_sizes", mx::array(sizes.data(), {int(sizes.size())}, mx::int32));
  std::unordered_map<std::string, std::string> meta{
      {"step", std::to_string(t.step)},         {"epoch", std::to_string(t.epoch)},
      {"cursor", std::to_string(t.cursor)},     {"total_steps", std::to_string(t.total_steps)},
      {"warmup", std::to_string(t.warmup)},     {"opt_step", std::to_string(t.opt.step)},
      {"best_dev", format("%.17g", t.best_dev)}, {"best_step", std::to_string(t.best_step)},
      {"stale_evals", std::to_string(t.stale_evals)}, {"run", run_key(t)}};
  const std::string tmp = dir + "/state.tmp.safetensors", state = dir + "/state.safetensors";
  mx::save_safetensors(tmp, arrays, meta);
  // The previous state stays as state.prev: a crash between the renames, or a state that will not read, falls
  // back to it.
  if (fs::exists(state)) fs::rename(state, dir + "/state.prev.safetensors");
  commit_file(tmp, state);
  save_adapters(dir, t.model, adapter_info(t, std::numeric_limits<double>::quiet_NaN()));
}

void load_checkpoint(Trainer& t) {
  const std::string state = t.args.out_dir + "/state.safetensors", prev = t.args.out_dir + "/state.prev.safetensors";
  try {
    read_state(t, state);
  } catch (const Unreadable& e) {
    if (!fs::exists(prev)) throw;
    std::fprintf(stderr, "quotebound: %s; resuming from %s instead\n", e.what(), prev.c_str());
    read_state(t, prev);
  }
}

void start_or_resume(Trainer& t, bool resume) {
  const std::string dir = t.args.out_dir;
  const bool held = fs::exists(dir + "/state.safetensors") || fs::exists(dir + "/state.prev.safetensors");
  if (resume && !held) throw std::runtime_error("--resume: " + dir + " holds no state.safetensors");
  if (!resume && held)
    throw std::runtime_error(dir + " already holds a run (state.safetensors): pass --resume to continue it, or "
                             "choose another --out");
  if (resume) load_checkpoint(t);
}

void run_training(Trainer& t, const std::vector<std::FILE*>& log) {
  const fs::path out = t.args.out_dir;
  fs::create_directories(out / "generations");
  write_file((out / "run.json").string(), run_record(t) + "\n");
  auto stopping = [&] { return t.args.patience > 0 && t.stale_evals >= t.args.patience; };
  auto evaluate = [&] {
    auto t0 = clock_type::now();
    auto [sum, n] = eval_loss(t.model, t.dev, t.args.batch_tokens);
    const double loss = sum / double(n), eval_ms = ms_since(t0);
    t0 = clock_type::now();
    std::string lines;
    const std::vector<std::string> gens = dev_generations(t);
    for (const std::string& g : gens) lines += g + "\n";
    write_file((out / "generations" / ("step-" + std::to_string(t.step) + ".jsonl")).string(), lines);
    const double gen_ms = ms_since(t0);
    const bool best = loss < t.best_dev;
    if (best) {
      t.best_dev = loss, t.best_step = t.step, t.stale_evals = 0;
      save_adapters((out / "best").string(), t.model, adapter_info(t, loss));
    } else {
      t.stale_evals += 1;
    }
    save_adapters((out / ("step-" + std::to_string(t.step))).string(), t.model, adapter_info(t, loss));
    prune_steps(out, t.args.keep);
    emit(log, format("{\"step\":%d,\"dev_loss\":%.6f,\"dev_tokens\":%ld,\"best\":%s,\"eval_ms\":%.0f,"
                     "\"generated\":%zu,\"gen_ms\":%.0f}",
                     t.step, loss, n, best ? "true" : "false", eval_ms, gens.size(), gen_ms));
  };
  if (t.step == 0) evaluate();
  while (t.step < t.total_steps && !stopping()) {
    const StepStats s = train_step(t);
    emit(log, format("{\"step\":%d,\"epoch\":%d,\"loss\":%.6f,\"lr\":%.6g,\"grad_norm\":%.4f,\"trained_tokens\":%d,"
                     "\"tokens\":%d,\"tok_s\":%.1f,\"step_ms\":%.1f,\"peak_memory_gb\":%.3f}",
                     t.step, t.epoch, s.loss, s.lr, s.grad_norm, s.trained, s.tokens, s.tokens / (s.ms / 1000.0),
                     s.ms, s.peak_bytes / 1e9));
    const bool last = t.step == t.total_steps;
    if (t.step % t.args.eval_every == 0 || last) evaluate();
    if (t.step % t.args.save_every == 0 || last || stopping()) save_checkpoint(t);
  }
  if (stopping())
    emit(log, format("{\"step\":%d,\"stopped\":\"patience\",\"evaluations_without_improvement\":%d,"
                     "\"best_step\":%d,\"best_dev_loss\":%.6f}",
                     t.step, t.stale_evals, t.best_step, t.best_dev));
}
