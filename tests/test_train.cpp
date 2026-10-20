// LoRA training on the tiny models of tiny.h: the adapters' seams, gradients, loss mask, optimizer, a non-finite
// step, memorisation and generation from the trained adapter, resume and its refusals, save/load and its refusals,
// generation with an adapter, and the run loop (evaluations, generations, best, step directories, patience).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <numeric>
#include <random>
#include <string>

#include "../src/adamw.h"
#include "../src/data.h"
#include "../src/files.h"
#include "../src/generate.h"
#include "../src/json.h"
#include "../src/lora.h"
#include "../src/loss.h"
#include "../src/model.h"
#include "../src/train.h"
#include "check.h"
#include "tiny.h"

static float max_abs(const mx::array& a) { return mx::max(mx::abs(mx::astype(a, mx::float32))).item<float>(); }
static float max_diff(const mx::array& a, const mx::array& b) {
  return max_abs(mx::astype(a, mx::float32) - mx::astype(b, mx::float32));
}
static bool same(const mx::array& a, const mx::array& b) { return mx::array_equal(a, b).item<bool>(); }

// The tiny models have no checkpoint directory; their runs record this stand-in.
static const BaseModel kTinyBase{"tiny-config", "tiny-index", "tiny-template", {{"tiny.safetensors", 1}}};

static bool throws_with(const std::function<void()>& f, const std::string& part, std::string* what = nullptr) {
  try {
    f();
  } catch (const std::runtime_error& e) {
    if (what) *what = e.what();
    return std::string(e.what()).find(part) != std::string::npos;
  }
  return false;
}

static std::string scratch(const std::string& name) {
  std::string dir = std::filesystem::temp_directory_path() / ("quotebound_test_" + name);
  std::filesystem::remove_all(dir);
  return dir;
}

// Gives every adapter a nonzero b so the low-rank path, and the gradient through a, are live.
static void perturb_b(Model& m, uint64_t seed) {
  std::vector<mx::array> p = adapter_arrays(m);
  mx::random::seed(seed);
  for (size_t i = 1; i < p.size(); i += 2) p[i] = mx::random::normal(p[i].shape(), mx::float32, 0.f, 0.05f);
  set_adapter_arrays(m, p);
  mx::eval(p);
}

// Conversations as the renderer lays them out: a prompt (no loss), an answer and its end token (loss), a trailing
// token (no loss), on a vocabulary of `vocab`.
static std::vector<Example> examples(int n, int vocab, uint64_t seed, int prompt = 12, int answer = 8) {
  std::mt19937_64 rng(seed);
  std::vector<Example> out;
  for (int i = 0; i < n; ++i) {
    Example e;
    const int p = prompt + int(rng() % 5), a = answer + int(rng() % 3);
    for (int t = 0; t < p + a + 2; ++t) {
      e.ids.push_back(1 + int(rng() % (vocab - 1)));
      e.loss.push_back(t >= p && t <= p + a);  // the answer and its end token
    }
    e.id = "ex " + std::to_string(i), e.prompt_len = p;
    out.push_back(std::move(e));
  }
  return out;
}

static std::vector<int> all_rows(size_t n) {
  std::vector<int> r(n);
  std::iota(r.begin(), r.end(), 0);
  return r;
}

// Zero-init adapters (b = 0) leave the logits bit-identical: through forward() with and without them, and through the
// training seams (hidden + logits_at, the recurrence's trainable path, layer checkpoints).
static void zero_adapters_are_identity() {
  for (bool q35 : {false, true}) {
    Model base = tiny(mx::bfloat16, q35);
    mx::array toks = mx::random::randint(0, base.cfg.vocab, {2, 83}, mx::int32);
    mx::array want = forward(base, toks);
    Model m = base;
    add_lora(m, {4, 8, default_targets(), 1});
    mx::array with = forward(m, toks);
    mx::array positions = mx::arange(2 * 83, mx::int32);
    mx::array seams = mx::reshape(logits_at(m, hidden(m, toks), positions), {2, 83, m.cfg.vocab});
    mx::eval(want, with, seams);
    const bool a = same(want, with), b = same(want, seams);
    std::printf("zero adapters (%s bf16, %zu adapter arrays): forward bit-identical %s; hidden + logits_at "
                "bit-identical %s\n",
                q35 ? "qwen3.5" : "qwen3", adapter_arrays(m).size(), a ? "yes" : "NO", b ? "yes" : "NO");
    CHECK(a && b, "zero-init adapters changed the logits (max diff %g, %g)", max_diff(want, with),
          max_diff(want, seams));
  }
}

// d(loss)/d(adapters) from value_and_grad through the training path against central differences, float32: along the
// gradient's own direction (where the derivative is its norm), 3 random unit directions, and the 8 coordinates with
// the largest gradient, each to 1% of its value, through adapter_loss_and_grads (train_step's closure: the mean loss).
// The step moves the mean loss (about 5) by about 0.003 (eps = min(0.05, 0.003 / |slope|)): large against its float32
// rounding, small enough that curvature stays below it. A first run with 0.1 failed the gradient direction by 0.8% and
// a coordinate by 4.9%: that step was 30x the parameter step of the earlier summed-loss version of this test.
static void finite_differences() {
  Model m = tiny(mx::float32, true);
  add_lora(m, {2, 4, default_targets(), 3});
  perturb_b(m, 4);
  const std::vector<Example> ex = examples(3, m.cfg.vocab, 5);
  const Batch b = make_batch(ex, all_rows(ex.size()));
  const std::vector<mx::array> p0 = adapter_arrays(m);
  // The training closure itself, for the gradient and for every loss the differences take.
  auto loss_at = [&](const std::vector<mx::array>& p) {
    Model mm = m;
    set_adapter_arrays(mm, p);
    return adapter_loss_and_grads(mm, b).loss.item<float>();
  };
  const LossAndGrads lg = adapter_loss_and_grads(m, b);
  const std::vector<mx::array>& g = lg.grads;
  // d/dt loss(p0 + t d) at 0 for a direction d over all adapters, by the gradient and by central differences.
  auto along = [&](const std::vector<mx::array>& d) {
    mx::array sq(0.0f), dot(0.0f);
    for (size_t i = 0; i < d.size(); ++i) {
      sq = sq + mx::sum(mx::square(d[i]));
      dot = dot + mx::sum(g[i] * d[i]);
    }
    const float inv = 1.0f / std::sqrt(sq.item<float>()), an = dot.item<float>() * inv;
    const float eps = std::min(0.05f, 0.003f / std::abs(an));
    std::vector<mx::array> plus, minus;
    for (size_t i = 0; i < d.size(); ++i) {
      plus.push_back(p0[i] + d[i] * (eps * inv));
      minus.push_back(p0[i] - d[i] * (eps * inv));
    }
    return std::make_pair(an, (loss_at(plus) - loss_at(minus)) / (2 * eps));
  };
  float worst = 0;
  std::string line;
  mx::random::seed(6);
  for (int k = 0; k < 4; ++k) {
    std::vector<mx::array> d;
    for (size_t i = 0; i < p0.size(); ++i) d.push_back(k == 0 ? g[i] : mx::random::normal(p0[i].shape()));
    auto [an, fd] = along(d);
    worst = std::max(worst, std::abs(fd - an) / std::abs(an));
    char buf[96];
    std::snprintf(buf, sizeof buf, "%s%s %.5g vs %.5g", k ? ", " : "", k ? "random" : "gradient direction", an, fd);
    line += buf;
  }
  struct Coord {
    float g;
    size_t array, index;
  };
  std::vector<Coord> coords;
  for (size_t i = 0; i < g.size(); ++i) {
    mx::array gi = mx::reshape(g[i], {-1});
    mx::eval(gi);
    for (size_t j = 0; j < size_t(gi.size()); ++j) coords.push_back({gi.data<float>()[j], i, j});
  }
  std::partial_sort(coords.begin(), coords.begin() + 8, coords.end(),
                    [](const Coord& a, const Coord& b) { return std::abs(a.g) > std::abs(b.g); });
  float worst_coord = 0;
  for (int k = 0; k < 8; ++k) {
    std::vector<mx::array> d;
    for (size_t i = 0; i < p0.size(); ++i) {
      std::vector<float> one(p0[i].size(), 0.0f);
      if (i == coords[k].array) one[coords[k].index] = 1.0f;
      d.push_back(mx::reshape(mx::array(one.data(), {int(one.size())}), p0[i].shape()));
    }
    auto [an, fd] = along(d);
    worst_coord = std::max(worst_coord, std::abs(fd - an) / std::abs(an));
  }
  std::printf("LoRA gradient vs central differences (float32, mean loss %.4f): %s; worst relative error %.2g over the "
              "4 directions, %.2g over the 8 largest coordinates\n",
              lg.loss.item<float>(), line.c_str(), worst, worst_coord);
  CHECK(worst <= 1e-2f && worst_coord <= 1e-2f, "gradient and finite differences differ");
}

// The loss is the cross-entropy of exactly the trained positions (against a direct computation from the full
// logits), and padding or tokens after the last trained position change neither the loss nor any gradient.
static void loss_mask() {
  Model m = tiny(mx::float32, true);
  add_lora(m, {2, 4, default_targets(), 7});
  perturb_b(m, 8);
  std::vector<Example> ex = examples(3, m.cfg.vocab, 9);
  const Batch b = make_batch(ex, all_rows(ex.size()));
  mx::array got = loss_sum(m, b);

  // Direct: log-softmax of forward()'s logits, summed where the next token is trained on.
  double want = 0;
  int trained = 0;
  for (const Example& e : ex) {
    mx::array lp = forward(m, mx::array(e.ids.data(), {1, int(e.ids.size())}, mx::int32));
    lp = lp - mx::logsumexp(lp, -1, true);
    mx::eval(lp);
    for (size_t t = 0; t + 1 < e.ids.size(); ++t)
      if (e.loss[t + 1]) want -= lp.data<float>()[t * m.cfg.vocab + e.ids[t + 1]], ++trained;
  }
  const float rel = std::abs(got.item<float>() - float(want)) / float(want);

  // Change the token after each answer's end token and the padding value: nothing trained on can see them.
  auto grads = [&](const Batch& bb) {
    LossAndGrads lg = adapter_loss_and_grads(m, bb);
    lg.grads.push_back(lg.loss);
    return lg.grads;
  };
  std::vector<Example> changed = ex;
  for (Example& e : changed) e.ids.back() = (e.ids.back() + 17) % m.cfg.vocab;
  Batch b2 = make_batch(changed, all_rows(changed.size()));
  b2.tokens = mx::where(mx::equal(b2.tokens, mx::array(0)), mx::array(53), b2.tokens);  // the padding
  const std::vector<mx::array> g1 = grads(b), g2 = grads(b2);
  bool identical = true;
  for (size_t i = 0; i < g1.size(); ++i) identical = identical && same(g1[i], g2[i]);

  // And a trained token does move it.
  std::vector<Example> moved = ex;
  for (size_t t = 0; t < moved[0].ids.size(); ++t)
    if (moved[0].loss[t]) moved[0].ids[t] = (moved[0].ids[t] + 1) % m.cfg.vocab;
  const Batch bm = make_batch(moved, all_rows(moved.size()));
  const float shift = std::abs(loss_sum(m, bm).item<float>() - got.item<float>());

  std::printf("loss mask: summed loss %.6f over %d trained positions vs %.6f from forward()'s logits (relative %.2g); "
              "padding and tokens after the answers changed: loss and %zu gradients bit-identical %s; changing an "
              "answer moves the loss by %.3g\n",
              got.item<float>(), trained, want, rel, g1.size() - 1, identical ? "yes" : "NO", shift);
  CHECK(trained == int(b.positions.shape(0)), "trained positions %d, batch has %d", trained, b.positions.shape(0));
  CHECK(rel <= 1e-5f, "loss differs from the direct computation");
  CHECK(identical, "tokens outside the trained spans changed the loss or a gradient");
  CHECK(shift > 1e-3f, "changing trained tokens did not change the loss");
}

// adamw_update against the same arithmetic in double, by hand, over 3 steps: weight decay on, the first step's
// gradients clipped, the learning rate from lr_at with one warmup step.
static void adamw_reference() {
  const AdamWConfig c{0.9f, 0.999f, 1e-8f, 0.1f, 1.0f};
  std::vector<std::vector<float>> p0 = {{0.5f, -1.0f, 0.25f, 2.0f, 0.0f, -0.75f}, {0.1f, 0.2f, -0.3f}};
  std::vector<std::vector<std::vector<float>>> grads = {
      {{3.0f, -2.0f, 1.0f, 0.5f, -4.0f, 2.5f}, {1.0f, -1.0f, 2.0f}},       // norm 6.5: clipped to 1
      {{0.1f, 0.05f, -0.2f, 0.0f, 0.3f, -0.1f}, {0.02f, 0.4f, -0.05f}},    // norm 0.58: not clipped
      {{-0.3f, 0.2f, 0.1f, -0.05f, 0.0f, 0.15f}, {0.2f, -0.1f, 0.3f}}};
  const int total = 3, warmup = 1;
  const float peak = 0.01f;

  std::vector<mx::array> params;
  for (auto& p : p0) params.push_back(mx::array(p.data(), {int(p.size())}));
  AdamWState s = adamw_init(params);
  std::vector<std::vector<double>> p(p0.size()), m(p0.size()), v(p0.size());
  for (size_t i = 0; i < p0.size(); ++i) {
    p[i].assign(p0[i].begin(), p0[i].end());
    m[i].assign(p0[i].size(), 0.0), v[i].assign(p0[i].size(), 0.0);
  }
  float worst = 0, norm_err = 0;
  for (int step = 0; step < 3; ++step) {
    const float lr = lr_at(step, total, warmup, peak);
    std::vector<mx::array> g;
    double sq = 0;
    for (auto& gi : grads[step]) {
      g.push_back(mx::array(gi.data(), {int(gi.size())}));
      for (float x : gi) sq += double(x) * x;
    }
    const mx::array norm = global_norm(g);
    const std::vector<mx::array> next = adamw_update(params, g, norm, s, c, lr);
    mx::eval(next), mx::eval(norm);
    params = next;
    // By hand.
    const double gn = std::sqrt(sq), coef = std::min(1.0, c.clip / (gn + 1e-6));
    const int t = step + 1;
    const double bc1 = 1 - std::pow(0.9, t), bc2 = 1 - std::pow(0.999, t);
    for (size_t i = 0; i < p.size(); ++i)
      for (size_t j = 0; j < p[i].size(); ++j) {
        const double gj = grads[step][i][j] * coef;
        m[i][j] = 0.9 * m[i][j] + 0.1 * gj;
        v[i][j] = 0.999 * v[i][j] + 0.001 * gj * gj;
        p[i][j] *= 1 - double(lr) * 0.1;
        p[i][j] -= double(lr) / bc1 * m[i][j] / (std::sqrt(v[i][j]) / std::sqrt(bc2) + 1e-8);
      }
    norm_err = std::max(norm_err, float(std::abs(norm.item<float>() - gn) / gn));
    for (size_t i = 0; i < p.size(); ++i) {
      const float* mine = params[i].data<float>();
      for (size_t j = 0; j < p[i].size(); ++j) worst = std::max(worst, float(std::abs(mine[j] - p[i][j])));
    }
  }
  std::printf("AdamW vs a hand-computed double reference over 3 steps (clipped first step, weight decay 0.1, lr "
              "%.4g/%.4g/%.4g): max |dparam| %.2g, gradient norm relative error %.2g\n",
              lr_at(0, total, warmup, peak), lr_at(1, total, warmup, peak), lr_at(2, total, warmup, peak), worst,
              norm_err);
  CHECK(worst <= 1e-6f && norm_err <= 1e-6f, "AdamW differs from the reference");
}

static TrainArgs tiny_args(int steps, float lr) {
  TrainArgs a;
  a.lora = {8, 16, default_targets(), 0};
  a.lr = lr, a.steps = steps, a.batch_tokens = 64, a.max_len = 32, a.seed = 11, a.warmup = 5, a.gen_dev = 0;
  return a;
}

static Trainer tiny_trainer(Model m, const std::vector<Example>& train, const std::vector<Example>& dev,
                            const TrainArgs& a, const Tokenizer* tok = nullptr) {
  return make_trainer(std::move(m), Dataset{.examples = train}, Dataset{.examples = dev}, a, kTinyBase, tok);
}

// Adapters, both moments and the optimizer's step, as one comparable list.
static std::vector<mx::array> state_of(const Trainer& t) {
  std::vector<mx::array> s = adapter_arrays(t.model);
  s.insert(s.end(), t.opt.m.begin(), t.opt.m.end());
  s.insert(s.end(), t.opt.v.begin(), t.opt.v.end());
  s.push_back(mx::array(t.opt.step));
  s.push_back(mx::array(t.step));
  return s;
}

static bool same_state(const std::vector<mx::array>& a, const std::vector<mx::array>& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (!same(a[i], b[i])) return false;
  return true;
}

// A non-finite loss or gradient norm stops the step before AdamW runs: an inf injected into one gradient entry, and
// an inf in a frozen weight (so the loss itself is inf), each leave the adapters, moments and steps bit-identical.
static void non_finite() {
  const std::vector<Example> ex = examples(6, 101, 41);
  Trainer t = tiny_trainer(tiny(mx::bfloat16, true), ex, ex, tiny_args(6, 1e-2f));
  train_step(t);  // moments nonzero
  const std::vector<mx::array> before = state_of(t);
  LossAndGrads g = adapter_loss_and_grads(t.model, make_batch(t.train, t.plan[t.cursor]));
  mx::array flat = mx::reshape(g.grads[3], {-1});
  g.grads[3] = mx::reshape(mx::where(mx::equal(mx::arange(int(flat.size())), mx::array(0)),
                                     mx::array(std::numeric_limits<float>::infinity()), flat),
                           g.grads[3].shape());
  g.norm = global_norm(g.grads);
  mx::eval(g.grads[3], g.norm);
  std::string what;
  const bool injected = throws_with([&] { apply_gradients(t, g, 1e-2f); }, "not finite", &what);
  const bool kept = same_state(before, state_of(t));

  Trainer poisoned = tiny_trainer(tiny(mx::bfloat16, true), ex, ex, tiny_args(6, 1e-2f));
  train_step(poisoned);
  const std::vector<mx::array> before2 = state_of(poisoned);
  poisoned.model.norm = mx::full(poisoned.model.norm.shape(), std::numeric_limits<float>::infinity(), mx::bfloat16);
  const bool frozen = throws_with([&] { train_step(poisoned); }, "not finite");
  const bool kept2 = same_state(before2, state_of(poisoned));
  std::printf("non-finite step: an inf gradient entry refused (\"%s\"), state bit-identical %s; an inf frozen weight "
              "refused %s, state bit-identical %s\n",
              what.c_str(), kept ? "yes" : "NO", frozen ? "yes" : "NO", kept2 ? "yes" : "NO");
  CHECK(injected && kept && frozen && kept2, "a non-finite step reached the adapters or optimizer");
}

// adapter_loss_and_grads peaks no higher than the loss and gradients evaluated alone: its norm adds no memory. With
// the norm in the gradients' graph, MLX's order ran several layers' recomputations at once; that takes depth, so
// this model has 8 layers. The 4B's peak rose 67% that way, and 4,096 tokens ran out of memory.
static void step_peak() {
  Model m = tiny(mx::bfloat16, true, 8);
  add_lora(m, {8, 16, default_targets(), 0});
  perturb_b(m, 43);
  std::mt19937_64 rng(43);
  Example e;
  for (int t = 0; t < 512; ++t) e.ids.push_back(int(rng() % 101)), e.loss.push_back(t >= 300);
  const Batch b = make_batch({e}, {0});
  const std::vector<mx::array> params = adapter_arrays(m);
  std::vector<int> argnums(params.size());
  std::iota(argnums.begin(), argnums.end(), 0);
  auto alone = [&] {
    auto [loss, grads] = mx::value_and_grad(
        [&](const std::vector<mx::array>& p) {
          Model mm = m;
          set_adapter_arrays(mm, p);
          return std::vector<mx::array>{loss_sum(mm, b) / mx::array(float(b.positions.shape(0)))};
        },
        argnums)(params);
    grads.push_back(loss[0]);
    mx::eval(grads);
  };
  // Above the memory in use before, on a second run: the first compiles kernels.
  auto peak = [](const std::function<void()>& f) {
    f();
    mx::synchronize();
    mx::clear_cache();
    const size_t before = mx::get_active_memory();
    mx::reset_peak_memory();
    f();
    mx::synchronize();
    return double(mx::get_peak_memory() - before);
  };
  const double grads = peak(alone), step = peak([&] { adapter_loss_and_grads(m, b); });
  std::printf("step peak (tiny qwen3.5, 8 layers, 512 tokens): %.1f MB above the start, the gradients alone %.1f MB\n",
              step / 1e6, grads / 1e6);
  CHECK(step <= grads * 1.01, "adapter_loss_and_grads peaks above the gradients alone");
}

// The bf16 tiny Qwen3.5 learns 4 answers by heart. Its final norm is scaled by 4 first: as built, the norm and the
// tied embedding cap a logit's margin near 10, about what a loss of 0.01 over 100 rivals needs, and LoRA cannot move
// either. Then, with that trained adapter in bf16, what is exact is asserted exactly: the training path's logits
// (hidden + logits_at) equal forward()'s bit for bit, and greedy decoding with the cache from each prompt gives back
// the memorised answer. Cache and full forward are not asserted equal: they sum in another order.
static void memorise() {
  const std::vector<Example> ex = examples(4, 101, 12);
  TrainArgs a = tiny_args(150, 1e-2f);
  a.batch_tokens = 128;  // all four in one batch
  Model m = tiny(mx::bfloat16, true);
  m.norm = m.norm * mx::array(4.0f, mx::bfloat16);
  Trainer t = tiny_trainer(std::move(m), ex, ex, a);
  const auto [s0, n0] = eval_loss(t.model, ex, a.batch_tokens);
  std::vector<float> curve;
  while (t.step < t.total_steps) curve.push_back(train_step(t).loss);
  const auto [s1, n1] = eval_loss(t.model, ex, a.batch_tokens);
  std::printf("memorise 4 examples (%ld answer tokens), tiny qwen3.5 bf16, rank 8, lr 1e-2, %d steps: loss %.3g "
              "before, step 25 %.3g, step 50 %.3g, step 100 %.3g, after %.3g\n",
              n1, t.step, s0 / n0, curve[24], curve[49], curve[99], s1 / n1);
  CHECK(s1 / n1 < 0.01, "did not memorise: loss %.4f", s1 / n1);

  int recalled = 0, seams_exact = 0;
  for (const Example& e : ex) {
    mx::array toks = mx::array(e.ids.data(), {1, int(e.ids.size())}, mx::int32);
    mx::array full = forward(t.model, toks);
    mx::array seams = mx::reshape(logits_at(t.model, hidden(t.model, toks), mx::arange(int(e.ids.size()), mx::int32)),
                                  full.shape());
    seams_exact += same(full, seams);
    const int answer = int(std::count(e.loss.begin(), e.loss.end(), 1));
    const std::vector<int> prompt(e.ids.begin(), e.ids.begin() + e.prompt_len);
    const Generation g = greedy(t.model, prompt, answer, {});
    recalled += g.ids == std::vector<int>(e.ids.begin() + e.prompt_len, e.ids.begin() + e.prompt_len + answer);
  }
  std::printf("trained adapter, bf16: training-path logits bit-identical to forward() for %d/4 rows; greedy with the "
              "cache recalls %d/4 answers exactly\n",
              seams_exact, recalled);
  CHECK(seams_exact == 4 && recalled == 4, "the trained adapter's serving path differs from its training path");
}

// N steps straight equal N/2 steps, a checkpoint written and read by a fresh trainer, then N/2 more: adapters and
// AdamW's moments bit for bit. 10 examples in batches of 2 are 5 batches an epoch, so the checkpoint at step 7 is
// mid-epoch and the resumed run replays the saved plan before drawing the next. A second checkpoint at step 5 is
// state.prev: with state.safetensors truncated, the resume falls back to it and still ends bit-identical. Refused:
// other arguments, other training rows, a plan that does not fit, and an out_dir with a run in it without --resume.
static void resume() {
  const std::vector<Example> ex = examples(10, 101, 13), dev = examples(3, 101, 14);
  TrainArgs a = tiny_args(14, 5e-3f);
  a.out_dir = scratch("resume");
  Trainer full = tiny_trainer(tiny(mx::bfloat16, true), ex, dev, a);
  while (full.step < full.total_steps) train_step(full);

  Trainer first = tiny_trainer(tiny(mx::bfloat16, true), ex, dev, a);
  while (first.step < 5) train_step(first);
  save_checkpoint(first);
  while (first.step < 7) train_step(first);
  save_checkpoint(first);
  const size_t cursor = first.cursor, batches = first.plan.size();
  auto finish = [&](Trainer& t) {
    while (t.step < t.total_steps) train_step(t);
    return t.epoch == full.epoch && t.cursor == full.cursor && same_state(state_of(full), state_of(t));
  };
  Trainer second = tiny_trainer(tiny(mx::bfloat16, true), ex, dev, a);
  start_or_resume(second, true);
  const bool identical = second.step == 7 && finish(second);

  // A torn or corrupt state.safetensors: the resume reads state.prev (step 5) and still ends bit-identical.
  const std::string state = a.out_dir + "/state.safetensors";
  std::filesystem::resize_file(state, 1024);
  Trainer third = tiny_trainer(tiny(mx::bfloat16, true), ex, dev, a);
  load_checkpoint(third);
  const bool fell_back = third.step == 5 && finish(third);

  std::printf("resume: %d steps (%zu batches an epoch, ended in epoch %d) vs 7 + checkpoint (batch %zu of %zu) + 7: "
              "adapters and moments bit-identical %s; from state.prev after a truncated state, step 5 + 9: %s\n",
              full.step, plan_epoch(ex, a.batch_tokens, a.seed, 0).size(), full.epoch, cursor, batches,
              identical ? "yes" : "NO", fell_back ? "yes" : "NO");
  CHECK(cursor > 0 && cursor < batches, "the checkpoint is not mid-epoch");
  CHECK(full.epoch >= 2, "the run did not cross two epochs");
  CHECK(identical && fell_back, "resumed run differs");

  // Refusals, each with its message. Rewrite a good state first.
  save_checkpoint(first);
  std::string what;
  TrainArgs other = a;
  other.lr = 1e-3f;
  Trainer wrong = tiny_trainer(tiny(mx::bfloat16, true), ex, dev, other);
  const bool args_refused = throws_with([&] { load_checkpoint(wrong); }, "is from a run with other arguments", &what);
  std::vector<Example> edited = ex;
  edited[4].ids[2] = (edited[4].ids[2] + 1) % 101;
  Trainer data = tiny_trainer(tiny(mx::bfloat16, true), edited, dev, a);
  const bool data_refused = throws_with([&] { load_checkpoint(data); }, "train_sha256=");
  Trainer fresh = tiny_trainer(tiny(mx::bfloat16, true), ex, dev, a);
  const bool held = throws_with([&] { start_or_resume(fresh, false); }, "already holds a run");
  // The plan as saved, then pointing past the training set, then with sizes that do not add up.
  auto [arrays, meta] = mx::load_safetensors(state);
  auto rewrite_plan = [&, arrays = arrays, meta = meta](std::vector<int32_t> plan, std::vector<int32_t> sizes) mutable {
    arrays.insert_or_assign("plan", mx::array(plan.data(), {int(plan.size())}, mx::int32));
    arrays.insert_or_assign("plan_sizes", mx::array(sizes.data(), {int(sizes.size())}, mx::int32));
    mx::save_safetensors(state, arrays, meta);
    std::filesystem::remove(a.out_dir + "/state.prev.safetensors");
    Trainer t = tiny_trainer(tiny(mx::bfloat16, true), ex, dev, a);
    std::string why;
    throws_with([&] { load_checkpoint(t); }, "", &why);
    return why;
  };
  const std::string past = rewrite_plan({0, 1, 2, 30, 4, 5, 6, 7, 8, 9}, {2, 2, 2, 2, 2});
  const std::string sizes = rewrite_plan({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {2, 2, 2, 2, 3});
  std::printf("resume refusals: other lr (\"%s...\"), an edited training row, a plan naming example 30 (\"%s\"), "
              "plan sizes that do not add up (\"%s\"), and a run in --out without --resume\n",
              what.substr(what.find("is from"), 41).c_str(), past.substr(past.find(": ") + 2).c_str(),
              sizes.substr(sizes.find(": ") + 2).c_str());
  CHECK(args_refused && data_refused && held, "a resume was not refused");
  CHECK(past.find("names example 30") != std::string::npos && sizes.find("do not add up") != std::string::npos,
        "a plan that does not fit was not refused");
  std::filesystem::remove_all(a.out_dir);
}

// save_adapters then load_adapters into a fresh model: the same arrays bit for bit, the same logits, and the record
// back. Refused: another base checkpoint, another chat template, and an adapters file the config was not written with.
static void round_trip() {
  Model m = tiny(mx::bfloat16, true);
  LoraConfig c{4, 8, {"q_proj", "o_proj", "in_proj_qkv", "out_proj", "down_proj"}, 21};
  add_lora(m, c);
  perturb_b(m, 22);
  const std::string dir = scratch("round_trip");
  save_adapters(dir, m, {c, kTinyBase, 40, 0.125, "{\"note\": \"x\"}"});
  Model back = tiny(mx::bfloat16, true);
  const AdapterInfo info = load_adapters(dir, back, kTinyBase);
  const std::vector<mx::array> a = adapter_arrays(m), b = adapter_arrays(back);
  bool identical = a.size() == b.size() && info.lora.rank == c.rank && info.lora.alpha == c.alpha &&
                   info.lora.targets == c.targets && info.step == 40 && info.dev_loss == 0.125 &&
                   info.run == "{\"note\":\"x\"}" && info.base == kTinyBase;
  for (size_t i = 0; identical && i < a.size(); ++i) identical = same(a[i], b[i]);
  mx::array toks = mx::random::randint(0, m.cfg.vocab, {1, 40}, mx::int32);
  const bool logits = same(forward(m, toks), forward(back, toks));

  BaseModel other_base = kTinyBase, other_template = kTinyBase;
  other_base.shards[0].second = 2;
  other_template.chat_template = "other";
  Model x = tiny(mx::bfloat16, true);
  const bool base_refused = throws_with([&] { load_adapters(dir, x, other_base); }, "another base checkpoint");
  const bool template_refused = throws_with([&] { load_adapters(dir, x, other_template); }, "chat template");
  save_adapters(dir + "/other", m, {c, kTinyBase, 41, 0.5, "{}"});
  std::filesystem::copy_file(dir + "/other/adapter_config.json", dir + "/adapter_config.json",
                             std::filesystem::copy_options::overwrite_existing);
  perturb_b(m, 23);
  save_adapters(dir + "/other", m, {c, kTinyBase, 41, 0.5, "{}"});
  std::filesystem::copy_file(dir + "/other/adapters.safetensors", dir + "/adapters.safetensors",
                             std::filesystem::copy_options::overwrite_existing);
  // The config records the first "other" adapters and the file holds the second: a torn pair.
  const bool torn = throws_with([&] { load_adapters(dir, x, kTinyBase); }, "sha256 differs");
  std::printf("adapter round trip: %zu arrays bit-identical %s, config and record the same, logits bit-identical %s; "
              "refused: another base %s, another chat template %s, a torn adapters/config pair %s\n",
              a.size(), identical ? "yes" : "NO", logits ? "yes" : "NO", base_refused ? "yes" : "NO",
              template_refused ? "yes" : "NO", torn ? "yes" : "NO");
  CHECK(identical && logits, "adapters changed through save and load");
  CHECK(base_refused && template_refused && torn, "a mismatched adapter was loaded");
  std::filesystem::remove_all(dir);
}

// generate --adapter's path (load_adapters, then greedy with the cache) against the full forward with the adapter,
// float32: each generated token is the argmax of the full forward over prompt + tokens so far, and the training path's
// logits equal forward()'s bit for bit. The adapter must change the output for the check to mean anything.
static void generate_with_adapter() {
  Model m = tiny(mx::float32, true);
  LoraConfig c{4, 8, default_targets(), 31};
  add_lora(m, c);
  perturb_b(m, 32);
  const std::string dir = scratch("generate");
  save_adapters(dir, m, {c, kTinyBase});
  Model g = tiny(mx::float32, true);
  load_adapters(dir, g, kTinyBase);
  std::mt19937_64 rng(33);
  std::vector<int> prompt;
  for (int i = 0; i < 30; ++i) prompt.push_back(1 + int(rng() % 100));
  const Generation out = greedy(g, prompt, 24, {});
  std::vector<int> all = prompt;
  all.insert(all.end(), out.ids.begin(), out.ids.end());
  const int P = int(prompt.size()), n = int(out.ids.size());
  mx::array toks = mx::array(all.data(), {1, int(all.size())}, mx::int32);
  mx::array full = forward(g, toks);
  mx::array trainpath = mx::reshape(logits_at(g, hidden(g, toks), mx::arange(int(all.size()), mx::int32)),
                                    full.shape());
  mx::array picks = mx::argmax(mx::slice(full, {0, P - 1, 0}, {1, P - 1 + n, g.cfg.vocab}), -1);
  mx::eval(picks);
  int agree = 0;
  for (int i = 0; i < n; ++i) agree += picks.data<uint32_t>()[i] == uint32_t(out.ids[i]);
  const Generation base = greedy(tiny(mx::float32, true), prompt, 24, {});
  const bool seam = same(full, trainpath);
  std::printf("generate with an adapter (tiny qwen3.5 float32): %d/%d tokens are the full forward's argmax; the "
              "training path's logits bit-identical to forward's %s; without the adapter %s\n",
              agree, n, seam ? "yes" : "NO", base.ids == out.ids ? "THE SAME TOKENS" : "different tokens");
  CHECK(agree == n && seam, "generation with the adapter differs from the forward with it");
  CHECK(base.ids != out.ids, "the adapter did not change generation");
  std::filesystem::remove_all(dir);
}

// merge_lora: every adapter folded into its weight (none left), the merged forward equal to the adapter forward to
// float32 rounding (float32) and within bf16 rounding (bf16); the merged weight is W + scale B A exactly as float32 sums
// it, rounded once.
static void merge_adapters() {
  for (mx::Dtype dt : {mx::float32, mx::bfloat16}) {
    Model m = tiny(dt, true);
    add_lora(m, {4, 8, default_targets(), 61});
    perturb_b(m, 62);
    const size_t slots = adapter_arrays(m).size() / 2;
    std::mt19937_64 rng(63);
    std::vector<int> prompt;
    for (int i = 0; i < 40; ++i) prompt.push_back(1 + int(rng() % 100));
    const mx::array toks(prompt.data(), {1, int(prompt.size())}, mx::int32);
    const mx::array with = forward(m, toks), base = forward(tiny(dt, true), toks);
    const Proj p = m.layers[0].gate;
    const mx::array want = mx::astype(
        mx::astype(p.w, mx::float32) + mx::matmul(p.lora->b, p.lora->a) * mx::array(p.lora->scale, mx::float32), dt);
    const int merged = merge_lora(m);
    const mx::array folded = forward(m, toks);
    const float d = max_diff(folded, with), moved = max_diff(with, base), scale = max_abs(with);
    std::printf("merge_lora (tiny qwen3.5 %s): %d of %zu projections merged, %zu adapters left; merged vs adapter "
                "logits max |d| %.3g (the adapter moves them %.3g; max |logit| %.3g); gate_proj weight as summed: %s\n",
                dt == mx::float32 ? "float32" : "bf16", merged, slots, adapter_arrays(m).size(), d, moved, scale,
                same(m.layers[0].gate.w, want) ? "yes" : "NO");
    CHECK(merged == int(slots) && adapter_arrays(m).empty(), "adapters left after the merge");
    CHECK(same(m.layers[0].gate.w, want), "the merged weight is not W + scale B A rounded once");
    // float32: rounding only. bf16: the merged weight is rounded once where the adapter path rounds x W and its
    // low-rank term apart, and a random model amplifies that through its layers (measured 0.22 of a largest logit
    // of 3.3), so bf16 is held only to keeping the adapter's effect: a difference under a tenth of what the adapter
    // moves. Whether that rounding changes real answers is the dev bench's question, not this test's.
    if (dt == mx::float32) CHECK(d <= 1e-4f, "merged logits are not the adapter's");
    CHECK(moved > 10 * d, "merged vs adapter logits differ by %g, the adapter moves them %g", d, moved);
  }
}

// make_trainer refuses a dev set it cannot score whole: empty, or with rows the loader left out.
static void dev_refusals() {
  const std::vector<Example> ex = examples(4, 101, 51);
  const TrainArgs a = tiny_args(4, 1e-3f);
  auto make = [&](Dataset dev) {
    make_trainer(tiny(mx::bfloat16, true), Dataset{.examples = ex}, std::move(dev), a, kTinyBase, nullptr);
  };
  std::string too_long, unanswered;
  const bool empty = throws_with([&] { make(Dataset{}); }, "the dev set is empty");
  const bool long_rows =
      throws_with([&] { make(Dataset{.examples = ex, .too_long = {"row-a", "row-b"}}); }, "row-a, row-b", &too_long);
  const bool no_answer =
      throws_with([&] { make(Dataset{.examples = ex, .no_answer = {"row-c"}}); }, "row-c", &unanswered);
  std::printf("dev refusals: empty %s; rows left out as too long (\"%s\") and without an answer (\"%s\")\n",
              empty ? "yes" : "NO", too_long.c_str(), unanswered.c_str());
  CHECK(empty && long_rows && no_answer, "a dev set that cannot be scored whole was accepted");
}

static std::vector<std::string> lines_of(const std::string& path) {
  std::vector<std::string> out;
  std::string all = read_file(path);
  for (size_t at = 0, nl; (nl = all.find('\n', at)) != std::string::npos; at = nl + 1)
    out.push_back(all.substr(at, nl - at));
  return out;
}

static double number_after(const std::string& line, const std::string& key) {
  const size_t at = line.find("\"" + key + "\":");
  CHECK(at != std::string::npos, "no %s in %s", key.c_str(), line.c_str());
  return std::strtod(line.c_str() + at + key.size() + 3, nullptr);
}

// run_training on the tiny model: an evaluation at step 0 and every 2 steps; best/ is the argmin of the logged dev
// losses with its step and loss recorded; generations/step-N.jsonl holds gen_dev answers each time; only the newest
// `keep` step-N/ directories stay; run.json is written. Then with lr 0 the dev loss never falls, and patience 2 stops
// the run after two more evaluations.
static void run_loop(const Tokenizer& tok) {
  const std::vector<Example> ex = examples(8, 101, 61), dev = examples(5, 101, 62);
  TrainArgs a = tiny_args(6, 1e-2f);
  a.out_dir = scratch("run_loop");
  a.eval_every = 2, a.save_every = 3, a.gen_dev = 2, a.keep = 2, a.dev_max = 4;
  Trainer t = tiny_trainer(tiny(mx::bfloat16, true), ex, dev, a, &tok);
  const std::string log = a.out_dir + "/log.jsonl";
  std::filesystem::create_directories(a.out_dir);
  std::FILE* f = std::fopen(log.c_str(), "w");
  run_training(t, {f});
  std::fclose(f);
  std::vector<int> eval_steps;
  std::vector<double> losses;
  for (const std::string& l : lines_of(log))
    if (l.find("dev_loss") != std::string::npos)
      eval_steps.push_back(int(number_after(l, "step"))), losses.push_back(number_after(l, "dev_loss"));
  const size_t argmin = size_t(std::min_element(losses.begin(), losses.end()) - losses.begin());
  simdjson::dom::parser parser;
  simdjson::dom::element best = load_json(parser, a.out_dir + "/best/adapter_config.json");
  const int best_step = int(int64_t(best["step"]));
  const double best_loss = double(best["dev_loss"]);
  bool generations = true;
  for (int s : eval_steps) {
    const std::vector<std::string> g = lines_of(a.out_dir + "/generations/step-" + std::to_string(s) + ".jsonl");
    generations = generations && g.size() == 2 && g[0].starts_with("{\"id\": \"ex ");
  }
  std::vector<std::string> step_dirs;
  for (const auto& e : std::filesystem::directory_iterator(a.out_dir))
    if (e.path().filename().string().starts_with("step-")) step_dirs.push_back(e.path().filename().string());
  std::sort(step_dirs.begin(), step_dirs.end());
  const bool files = std::filesystem::exists(a.out_dir + "/run.json") &&
                     std::filesystem::exists(a.out_dir + "/state.safetensors") &&
                     std::filesystem::exists(a.out_dir + "/state.prev.safetensors");
  std::printf("run loop: evaluations at steps");
  for (int s : eval_steps) std::printf(" %d", s);
  std::printf("; best/ is step %d (dev loss %.6f, the logged minimum at step %d); 2 generations per evaluation %s; "
              "step directories kept: %s %s; run.json and state + state.prev %s\n",
              best_step, best_loss, eval_steps[argmin], generations ? "yes" : "NO", step_dirs.at(0).c_str(),
              step_dirs.size() > 1 ? step_dirs[1].c_str() : "", files ? "yes" : "NO");
  CHECK((eval_steps == std::vector<int>{0, 2, 4, 6}), "evaluations at the wrong steps");
  CHECK(best_step == eval_steps[argmin] && std::abs(best_loss - losses[argmin]) < 1e-6, "best/ is not the argmin");
  CHECK(generations && files, "generations, run.json or the checkpoints missing");
  CHECK((step_dirs == std::vector<std::string>{"step-4", "step-6"}), "--keep 2 kept the wrong step directories");

  TrainArgs p = a;
  p.out_dir = scratch("patience");
  p.lr = 0, p.steps = 20, p.patience = 2;
  Trainer still = tiny_trainer(tiny(mx::bfloat16, true), ex, dev, p, &tok);
  std::filesystem::create_directories(p.out_dir);
  f = std::fopen((p.out_dir + "/log.jsonl").c_str(), "w");
  run_training(still, {f});
  std::fclose(f);
  const std::string stop = lines_of(p.out_dir + "/log.jsonl").back();
  std::printf("patience 2 at lr 0: stopped at step %d of 20: %s\n", still.step, stop.c_str());
  CHECK(still.step == 4 && stop.find("\"stopped\":\"patience\"") != std::string::npos, "patience did not stop the run");
  std::filesystem::remove_all(a.out_dir);
  std::filesystem::remove_all(p.out_dir);
}

int main(int argc, char** argv) {
  CHECK(argc == 2, "usage: test_train MODEL_DIR (its tokenizer decodes the run loop's generations)");
  const Tokenizer tok = load_tokenizer(argv[1]);
  zero_adapters_are_identity();
  finite_differences();
  loss_mask();
  adamw_reference();
  non_finite();
  step_peak();
  memorise();
  resume();
  round_trip();
  generate_with_adapter();
  merge_adapters();
  dev_refusals();
  run_loop(tok);
  return 0;
}
