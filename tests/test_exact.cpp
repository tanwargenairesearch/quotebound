// Exact rows (src/exact.h), no real weights: gemv_rows against mx::matmul one row at a time, bit for bit, on every
// projection shape of Qwen3.5-4B (random values; its adapters' float32 shapes too) and on the tiny models', at each row
// count; then forward_rows on tiny Qwen3.5 models (float32 and bfloat16, with and without adapters) against decode
// steps one token at a time after a prompt, the logits and, after keep_rows to each length, the cache.
#include <cstdio>
#include <set>
#include <vector>

#include "../src/exact.h"
#include "../src/generate.h"
#include "../src/lora.h"
#include "check.h"
#include "tiny.h"

static bool same(const mx::array& a, const mx::array& b) {
  return a.shape() == b.shape() && a.dtype() == b.dtype() && mx::array_equal(a, b).item<bool>();
}

static int rows_against_matmul(const mx::array& w, const std::vector<int>& counts, const char* what) {
  const int N = w.shape(0), K = w.shape(1);
  int n = 0;
  for (int V : counts) {
    const mx::array x = mx::astype(mx::random::normal({V, K}), w.dtype());
    const mx::array rows = gemv_rows(x, w);
    std::vector<mx::array> one;
    for (int v = 0; v < V; ++v)
      one.push_back(mx::reshape(mx::matmul(mx::reshape(mx::slice(x, {v, 0}, {v + 1, K}), {1, 1, K}), mx::transpose(w)),
                                {1, N}));
    CHECK(same(rows, mx::concatenate(one, 0)), "%s [%d, %d] %s, %d rows: gemv_rows is not matmul row by row", what, N,
          K, w.dtype() == mx::float32 ? "float32" : "bfloat16", V);
    ++n;
  }
  return n;
}

static void gemv_shapes() {
  mx::random::seed(3);
  const std::vector<int> counts{1, 2, 3, 4, 5, 8, 9, 15, 16};
  // Qwen3.5-4B: hidden 2560, intermediate 9216, attention q (gated) 8192 / kv 1024 / o 4096 wide, DeltaNet qkv 8192,
  // z 4096, b and a 32, out 4096; a vocabulary-wide stand-in of 8200 rows (no multiple of a threadgroup's 32);
  // rank-16 adapters in float32 on 2560-, 4096- and 9216-wide inputs.
  const std::vector<std::pair<int, int>> bf16{{8192, 2560}, {1024, 2560}, {2560, 4096}, {9216, 2560}, {2560, 9216},
                                              {4096, 2560}, {32, 2560},   {8200, 2560}};
  const std::vector<std::pair<int, int>> f32{{16, 2560}, {16, 4096}, {16, 9216}, {8192, 16}, {2560, 16}, {32, 16}};
  int n = 0;
  for (auto [N, K] : bf16) n += rows_against_matmul(mx::astype(mx::random::normal({N, K}, 0.f, 0.02f), mx::bfloat16), counts, "bf16");
  for (auto [N, K] : f32) n += rows_against_matmul(mx::random::normal({N, K}, 0.f, 0.02f), counts, "f32");
  // The tiny models' shapes: inputs 96, 192, 512 and 4096 wide.
  for (auto [N, K] : std::vector<std::pair<int, int>>{{192, 96}, {96, 192}, {1024, 96}, {96, 512}, {96, 4096}, {32, 96},
                                                      {8192, 96}, {101, 96}})
    n += rows_against_matmul(mx::random::normal({N, K}), counts, "tiny f32") +
         rows_against_matmul(mx::astype(mx::random::normal({N, K}), mx::bfloat16), counts, "tiny bf16");
  std::printf("gemv_rows: %d (shape, dtype, rows) cases bit-identical to matmul one row at a time\n", n);
}

// Adapters on every projection of a tiny model, their B non-zero so they move the logits.
static void attach_adapters(Model& m) {
  mx::random::seed(11);
  for (Layer& l : m.layers)
    for (auto& [name, p] : projections(l)) {
      const int out = p->w.shape(0), in = p->w.shape(1), r = 4;
      p->lora = Lora{mx::random::normal({r, in}, 0.f, 0.1f), mx::random::normal({out, r}, 0.f, 0.1f), 2.0f};
    }
}

static bool same_cache(const KVCache& a, const KVCache& b) {
  if (a.offset != b.offset) return false;
  for (size_t l = 0; l < a.layers.size(); ++l) {
    const LayerCache &x = a.layers[l], &y = b.layers[l];
    if (x.k) {
      auto head = [&](const mx::array& z) { return mx::slice(z, {0, 0, 0, 0}, {z.shape(0), z.shape(1), a.offset, z.shape(3)}); };
      if (!same(head(*x.k), head(*y.k)) || !same(head(*x.v), head(*y.v))) return false;
    } else if (!same(*x.conv, *y.conv) || !same(*x.state, *y.state)) {
      return false;
    }
  }
  return true;
}

static void rows_against_steps() {
  int passes = 0, caches = 0;
  for (mx::Dtype dt : {mx::float32, mx::bfloat16})
    for (bool adapters : {false, true}) {
      Model m = tiny(dt, true, 4);
      if (adapters) attach_adapters(m);
      std::vector<int> prompt, next;
      for (int i = 0; i < 37; ++i) prompt.push_back((i * 29 + 5) % m.cfg.vocab);
      for (int i = 0; i < 16; ++i) next.push_back((i * 13 + 7) % m.cfg.vocab);
      for (int V : {1, 2, 3, 6, 9, 16}) {
        KVCache steps;
        mx::eval(prefill(m, prompt, steps));
        KVCache rows_cache = steps;
        std::vector<mx::array> want;
        std::vector<KVCache> after;
        for (int i = 0; i < V; ++i) {
          const mx::array lg = forward(m, mx::array(&next[i], {1, 1}, mx::int32), &steps, true);
          mx::eval(lg), mx::eval(cache_arrays(steps));
          want.push_back(mx::reshape(lg, {1, -1}));
          after.push_back(steps);
        }
        RowStates rs;
        const mx::array got = forward_rows(m, std::vector<int>(next.begin(), next.begin() + V), rows_cache, rs);
        CHECK(same(got, mx::concatenate(want, 0)), "%s%s, %d rows: forward_rows' logits are not the decode steps'",
              dt == mx::float32 ? "float32" : "bfloat16", adapters ? " with adapters" : "", V);
        ++passes;
        for (int keep = 1; keep <= V; ++keep) {
          KVCache c = rows_cache;
          keep_rows(c, rs, V, keep);
          CHECK(same_cache(c, after[keep - 1]), "%d rows kept of %d: the cache is not the decode steps'", keep, V);
          ++caches;
          // Decoding on from the cut cache: the next step as after the steps themselves.
          if (keep < V) {
            KVCache d = after[keep - 1];
            const mx::array a = forward(m, mx::array(&next[keep], {1, 1}, mx::int32), &c, true);
            const mx::array b = forward(m, mx::array(&next[keep], {1, 1}, mx::int32), &d, true);
            CHECK(same(a, b), "a step after keeping %d rows of %d is not the step after the decode steps", keep, V);
          }
        }
      }
    }
  std::printf("forward_rows: %d passes (float32 and bfloat16, adapters on and off, 1 to 16 rows) bit-identical to "
              "decode steps; %d caches cut by keep_rows equal the steps' and decode on alike\n", passes, caches);
}

int main() {
  gemv_shapes();
  rows_against_steps();
  return 0;
}
