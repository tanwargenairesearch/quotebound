// Follows mlx-lm's Qwen3 / Qwen3.5 implementation for token-for-token parity (Copyright © 2023 Apple Inc., MIT
// License; see NOTICE).
#include "generate.h"

#include "exact.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

mx::array prefill(const Model& m, const std::vector<int>& prompt, KVCache& cache, int chunk) {
  const int T = int(prompt.size());
  int done = 0;
  for (; T - done > chunk; done += chunk) {
    forward(m, mx::array(prompt.data() + done, {1, chunk}, mx::int32), &cache, true);  // logits never evaluated
    mx::eval(cache_arrays(cache));
  }
  return forward(m, mx::array(prompt.data() + done, {1, T - done}, mx::int32), &cache, true);
}

int pick_allowed(const mx::array& logits, const std::vector<int>& allow) {
  if (allow.size() == 1) return allow[0];
  const mx::array idx(allow.data(), {int(allow.size())}, mx::int32);
  return allow[mx::argmax(mx::take(mx::reshape(logits, {-1}), idx), 0).item<uint32_t>()];
}

namespace {

// [top logit, second logit, log-sum-exp] over the whole vocabulary of one step's logits [.., vocab], `top` their argmax:
// the second is the largest logit at any other id, so a tie for the top gives a margin of 0. Float32, as the logits.
mx::array logit_stats(const mx::array& logits, const mx::array& top) {
  const mx::array flat = mx::reshape(logits, {-1});
  const mx::array others = mx::where(mx::equal(mx::arange(int(flat.size())), mx::astype(mx::reshape(top, {}), mx::int32)),
                                     mx::array(-std::numeric_limits<float>::infinity()), flat);
  return mx::stack({mx::max(flat), mx::max(others), mx::logsumexp(flat)});
}

// Greedy under a constraint. The masked argmax is the model's own argmax whenever the constraint allows it: the lowest
// id among the top logits is then also the lowest among the allowed ones, so the allowed set only has to be searched
// for it, and the logits are gathered over the allowed ids (pick_allowed) only when it is refused. So each step queues
// the next one on that argmax before reading anything; when the constraint takes it, as it nearly always does, the GPU
// never waits on the host, and when it does not, the queued step is dropped and the step run on the token taken. Not
// where the answer may stop: the argmax is then likely the stop, and a step queued past it would only delay the next
// answer. The allowed set is still worked out every step, while the GPU runs, so budget_cut and mask_ms are as
// before. With confidence, each step's logit_stats are queued with its argmax and the taken token's logit after it is
// chosen, all read at the end.
Generation constrained(const Model& m, const std::vector<int>& prompt, int max_tokens, const std::vector<int>& stop,
                       const std::function<void(int)>& on_token, Constraint& c, bool confidence) {
  using clock = std::chrono::steady_clock;
  auto ms = [](clock::time_point a, clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  Generation g;
  KVCache cache;
  auto t0 = clock::now(), t1 = t0;
  // A step's logits and argmax, queued; with confidence its logit_stats too.
  struct Queued {
    mx::array logits, top;
    std::vector<mx::array> conf;
  };
  auto queue = [&](const mx::array& logits, std::vector<mx::array> also) {
    Queued q{logits, mx::argmax(mx::reshape(logits, {-1}), 0), {}};
    also.insert(also.end(), {q.logits, q.top});
    if (confidence) q.conf.push_back(logit_stats(q.logits, q.top)), also.push_back(q.conf[0]);
    mx::async_eval(also);
    return q;
  };
  Queued cur = queue(prefill(m, prompt, cache), cache_arrays(cache));
  struct Step {
    int tok, start, end, want;
    std::vector<mx::array> conf;  // with confidence: the step's logit_stats, then the taken token's logit
  };
  std::vector<Step> steps;
  for (int n = 0; n < max_tokens; ++n) {
    std::optional<KVCache> ahead_cache;
    std::optional<Queued> ahead;
    if (n + 1 < max_tokens && !c.complete()) {
      // The queued step takes the attention layers' key and value buffers, so it writes its row in place rather than
      // copying them; a step run instead on another token writes that same row over it. The DeltaNet layers' conv and
      // state are new arrays each step, so `cache` keeps the ones to go back to.
      ahead_cache = cache;
      for (LayerCache& l : cache.layers) l.k.reset(), l.v.reset();
      ahead = queue(forward(m, mx::reshape(mx::astype(cur.top, mx::int32), {1, 1}), &*ahead_cache, true), {});
    }
    const auto a = clock::now();
    const std::vector<int> allow = c.allowed(max_tokens - n);
    g.mask_ms += ms(a, clock::now());
    if (allow.empty()) throw std::runtime_error("constrained decoding: no token can continue the answer");
    const int want = int(cur.top.item<uint32_t>());
    const int tok = std::binary_search(allow.begin(), allow.end(), want) ? want : pick_allowed(cur.logits, allow);
    if (n == 0) {
      t1 = clock::now();
      g.prefill_ms = ms(t0, t1);
    }
    g.ids.push_back(tok);
    if (on_token) on_token(tok);
    const int start = int(c.bytes());
    c.accept(tok);
    if (confidence) {
      cur.conf.push_back(mx::take(mx::reshape(cur.logits, {-1}), tok));
      mx::async_eval(cur.conf.back());
    }
    steps.push_back({tok, start, int(c.bytes()), want, std::move(cur.conf)});
    if (std::find(stop.begin(), stop.end(), tok) != stop.end() || n + 1 == max_tokens) break;
    if (ahead && tok == want) {
      cache = std::move(*ahead_cache);
      cur = std::move(*ahead);
      continue;
    }
    if (ahead) {
      for (size_t l = 0; l < cache.layers.size(); ++l)
        cache.layers[l].k = std::move(ahead_cache->layers[l].k), cache.layers[l].v = std::move(ahead_cache->layers[l].v);
      ahead.reset();
      ahead_cache.reset();
    }
    cur = queue(forward(m, mx::array(&tok, {1, 1}, mx::int32), &cache, true), {});
  }
  g.decode_ms = ms(t1, clock::now());
  for (size_t n = 0; n < steps.size(); ++n)
    if (steps[n].want != steps[n].tok) g.overridden.push_back({int(n), steps[n].start, steps[n].end});
  if (confidence) {
    std::vector<mx::array> all;
    for (const Step& s : steps) all.insert(all.end(), s.conf.begin(), s.conf.end());
    mx::eval(all);
    for (const Step& s : steps) {
      const float* st = s.conf[0].data<float>();
      g.confidence.push_back({s.conf[1].item<float>() - st[2], st[0] - st[1]});
    }
  }
  return g;
}

// Exact speculative decoding under a constraint (Draft, generate.h). Each pass runs the token just taken and the drafts
// after it through forward_rows, whose row j is bit for bit the step a plain decoder would run there; the rows are
// then read as plain steps are: row j's token is chosen from its logits among the allowed ids, and while it equals the
// draft that followed, the next row is read too. The first row whose token is not the draft (or the last row) ends
// the pass: its token is the next pass's first, and the cache is cut back to the rows kept (keep_rows). The allowed
// sets the drafting worked out are the ones a plain step works out at a row it reaches (the same answer so far, the
// same budget), so they are reused, and a budget cut is recorded only for a row reached. A pass with no draft is a
// plain step (forward).
Generation speculative(const Model& m, const std::vector<int>& prompt, int max_tokens, const std::vector<int>& stop,
                       const std::function<void(int)>& on_token, Constraint& c, const Draft& d) {
  using clock = std::chrono::steady_clock;
  auto ms = [](clock::time_point a, clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  auto is_stop = [&](int id) { return std::find(stop.begin(), stop.end(), id) != stop.end(); };
  const int vocab = m.cfg.vocab;
  Generation g;
  KVCache cache;
  const auto t0 = clock::now();
  auto t1 = t0;
  mx::array logits = mx::reshape(prefill(m, prompt, cache), {1, vocab});  // [rows, vocab]
  mx::array tops = mx::argmax(logits, 1);
  {
    std::vector<mx::array> first = cache_arrays(cache);
    first.insert(first.end(), {logits, tops});
    mx::async_eval(first);
  }
  std::vector<int> ctx = prompt;               // the prompt and the answer so far: what the drafter reads
  std::vector<int> fed;                        // the drafts the pass ran after its first token
  std::vector<std::vector<int>> allow_at;      // the allowed sets the drafting worked out, row by row
  std::vector<char> cut_at;                    // ... and whether each left a token out for the budget
  RowStates rs;
  struct Step {
    int tok, start, end, want;
  };
  std::vector<Step> steps;
  for (int n = 0, rows = 1;;) {
    mx::eval(tops);
    const std::vector<uint32_t> want(tops.data<uint32_t>(), tops.data<uint32_t>() + rows);
    int kept = 0;
    bool done = false;
    for (int j = 0; j < rows; ++j) {
      std::vector<int> allow;
      if (j < int(allow_at.size())) {
        allow = std::move(allow_at[j]);
        if (cut_at[j]) c.budget_cut = true;
      } else {
        const auto a = clock::now();
        allow = c.allowed(max_tokens - n);
        g.mask_ms += ms(a, clock::now());
      }
      if (allow.empty()) throw std::runtime_error("constrained decoding: no token can continue the answer");
      const int w = int(want[j]);
      const int tok = std::binary_search(allow.begin(), allow.end(), w)
                          ? w
                          : pick_allowed(mx::slice(logits, {j, 0}, {j + 1, vocab}), allow);
      if (n == 0) {
        t1 = clock::now();
        g.prefill_ms = ms(t0, t1);
      }
      g.ids.push_back(tok);
      if (on_token) on_token(tok);
      const int start = int(c.bytes());
      c.accept(tok);
      steps.push_back({tok, start, int(c.bytes()), w});
      ++n;
      if (is_stop(tok) || n == max_tokens) {
        done = true;
        break;
      }
      ctx.push_back(tok);
      kept = j + 1;
      if (j + 1 == rows || tok != fed[j]) break;
    }
    if (done) break;
    if (rows > 1) keep_rows(cache, rs, rows, kept);
    // Draft after the token just taken, on a copy of the constraint: at most d.k, and none the cap would not let
    // the answer take.
    const int x = g.ids.back();
    fed.clear(), allow_at.clear(), cut_at.clear();
    {
      Constraint cd = c;
      const int most = std::min(d.k, max_tokens - n - 1);
      const auto a = clock::now();
      for (int i = 0; i < most; ++i) {
        cd.budget_cut = false;
        allow_at.push_back(cd.allowed(max_tokens - n - i));
        cut_at.push_back(cd.budget_cut);
        const int guess = d.drafter.guess(ctx, allow_at.back());
        if (guess < 0 || is_stop(guess)) break;
        cd.accept(guess);
        fed.push_back(guess);
        ctx.push_back(guess);
      }
      g.mask_ms += ms(a, clock::now());
      ctx.resize(ctx.size() - fed.size());
    }
    rows = 1 + int(fed.size());
    ++g.passes, g.drafted += int(fed.size());
    if (rows == 1) {
      logits = mx::reshape(forward(m, mx::array(&x, {1, 1}, mx::int32), &cache, true), {1, vocab});
    } else {
      std::vector<int> in{x};
      in.insert(in.end(), fed.begin(), fed.end());
      logits = forward_rows(m, in, cache, rs);
    }
    tops = mx::argmax(logits, 1);
    mx::async_eval({logits, tops});
  }
  g.decode_ms = ms(t1, clock::now());
  for (size_t n = 0; n < steps.size(); ++n)
    if (steps[n].want != steps[n].tok) g.overridden.push_back({int(n), steps[n].start, steps[n].end});
  return g;
}

}  // namespace

Generation greedy(const Model& m, const std::vector<int>& prompt, int max_tokens, const std::vector<int>& stop,
                  const std::function<void(int)>& on_token, Constraint* constraint, bool confidence,
                  const Draft* draft) {
  using clock = std::chrono::steady_clock;
  auto ms = [](clock::time_point a, clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  Generation g;
  if (max_tokens <= 0 || prompt.empty()) return g;
  for (int id : prompt)
    if (id < 0 || id >= m.cfg.vocab)
      throw std::runtime_error("prompt token id " + std::to_string(id) + " is outside the model's vocabulary of " +
                               std::to_string(m.cfg.vocab));
  if (prompt.size() + size_t(max_tokens) > size_t(m.cfg.max_positions))
    throw std::runtime_error("prompt (" + std::to_string(prompt.size()) + " tokens) + max tokens (" +
                             std::to_string(max_tokens) + ") exceeds max_position_embeddings (" +
                             std::to_string(m.cfg.max_positions) + ")");
  if (constraint) {
    if (constraint->min_close() + 1 > max_tokens)
      throw std::runtime_error("--max-tokens " + std::to_string(max_tokens) + " cannot hold the shortest answer the "
                               "schema allows (" + std::to_string(constraint->min_close()) + " bytes and a stop)");
    if (draft) {
      if (confidence) throw std::runtime_error("drafts do not record token confidence: run without one of them");
      if (draft->k < 1 || draft->k >= kMaxRows)
        throw std::runtime_error("drafts: k " + std::to_string(draft->k) + " outside 1.." + std::to_string(kMaxRows - 1));
      return speculative(m, prompt, max_tokens, stop, on_token, *constraint, *draft);
    }
    return constrained(m, prompt, max_tokens, stop, on_token, *constraint, confidence);
  }
  KVCache cache;
  auto t0 = clock::now(), t1 = t0;
  // With confidence, each step's logit_stats are queued with its token and read at the end: the token taken is the
  // argmax, so its logit is the top one.
  std::vector<mx::array> stats;
  mx::array y = [&] {
    const mx::array logits = prefill(m, prompt, cache);
    mx::array top = mx::argmax(logits, -1);
    if (confidence) stats.push_back(logit_stats(logits, top));
    return top;
  }();
  std::vector<mx::array> first = cache_arrays(cache);  // with the first token, so the cache drops the last chunk
  first.push_back(y);
  if (confidence) first.push_back(stats.back());
  mx::async_eval(first);
  for (int n = 0;; ++n) {
    // Queue step n+1 before reading token n so the GPU never waits on the host, as mlx-lm does.
    std::optional<mx::array> next;
    if (n + 1 < max_tokens) {
      if (confidence) {
        const mx::array logits = forward(m, y, &cache, true);
        next = mx::argmax(logits, -1);
        stats.push_back(logit_stats(logits, *next));
        mx::async_eval({*next, stats.back()});
      } else {
        next = mx::argmax(forward(m, y, &cache, true), -1);
        mx::async_eval(*next);
      }
    }
    int tok = int(y.item<uint32_t>());
    if (n == 0) {
      t1 = clock::now();
      g.prefill_ms = ms(t0, t1);
    }
    g.ids.push_back(tok);
    if (on_token) on_token(tok);
    if (!next || std::find(stop.begin(), stop.end(), tok) != stop.end()) break;
    y = *next;
  }
  g.decode_ms = ms(t1, clock::now());
  if (confidence) {
    stats.erase(stats.begin() + std::ptrdiff_t(g.ids.size()), stats.end());  // a step queued past a stop token
    mx::eval(stats);
    for (const mx::array& s : stats) {
      const float* st = s.data<float>();
      g.confidence.push_back({st[0] - st[2], st[0] - st[1]});
    }
  }
  return g;
}

namespace {

// A greedy answer cut at its token cap can end inside a UTF-8 sequence; JSON readers refuse that.
void drop_partial_utf8(std::string& s) {
  size_t start = s.size();
  for (size_t back = 1; back <= 4 && back <= s.size(); ++back) {
    const unsigned char c = s[s.size() - back];
    if ((c & 0xC0) != 0x80) {
      start = s.size() - back;
      break;
    }
  }
  if (start == s.size()) return;
  const unsigned char lead = s[start];
  const size_t need = lead >= 0xF0 ? 4 : lead >= 0xE0 ? 3 : lead >= 0xC0 ? 2 : 1;
  if (s.size() - start < need) s.resize(start);
}

}  // namespace

std::string generation_text(const Tokenizer& t, std::vector<int> ids, bool stopped) {
  if (stopped && !ids.empty()) ids.pop_back();
  std::string text;
  for (int id : ids) text += size_t(id) < t.id_bytes.size() ? t.id_bytes[id] : "<id " + std::to_string(id) + ">";
  if (!stopped) drop_partial_utf8(text);
  return text;
}

std::vector<TokenRecord> token_records(const Tokenizer& t, const Generation& g, bool stopped) {
  const size_t size = generation_text(t, g.ids, stopped).size();
  std::vector<TokenRecord> out;
  size_t at = 0;
  for (size_t n = 0; n < g.ids.size() && n < g.confidence.size(); ++n) {
    const int id = g.ids[n];
    const size_t len = stopped && n + 1 == g.ids.size() ? 0
                       : size_t(id) < t.id_bytes.size() ? t.id_bytes[id].size()
                                                        : ("<id " + std::to_string(id) + ">").size();
    out.push_back({int(std::min(at, size)), int(std::min(at + len, size)), g.confidence[n].logprob,
                   g.confidence[n].margin});
    at += len;
  }
  return out;
}
