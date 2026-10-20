// The kernel below is modified from MLX's GEMVKernel (Copyright © 2023 Apple Inc., MIT License; see NOTICE).
#include "exact.h"

#include <stdexcept>
#include <string>

namespace {

// MLX's GEMVKernel (mlx/backend/metal/kernels/gemv.h, the non-transposed matrix: out = mat · vec) with V vectors in
// place of one. Each vector's sums are the gemv's own: its lanes split K the same way, each lane adds its products in
// the same order into its own accumulators, then the same simdgroup shuffles and threadgroup reduction. The matrix
// tile a lane loads is applied to every vector before the next is read.
constexpr const char* kSource = R"(
  constexpr int threadsM = BM * SM;
  constexpr int threadsN = BN * SN;
  constexpr int blockM = threadsM * TM;
  constexpr int blockN = threadsN * TN;
  threadgroup float tgp_memory[BN > 1 ? V * BN * (blockM + TM) : 1];
  const uint simd_gid = simdgroup_index_in_threadgroup;
  const uint simd_lid = thread_index_in_simdgroup;
  const uint3 tid = threadgroup_position_in_grid;
  const int in_vec_size = K;
  const int out_vec_size = N;
  const int matrix_ld = K;

  float result[V][TM];
  for (int v = 0; v < V; v++)
    for (int tm = 0; tm < TM; tm++) result[v][tm] = 0;
  T inter[TN];
  float v_coeff[V][TN];

  const int thrM = SN != 32 ? simd_lid / SN : 0;
  const int thrN = SN != 32 ? simd_lid % SN : int(simd_lid);
  const int sgN = BN != 1 ? (simd_gid % BN) : 0;
  const int simdM = BN != 1 ? SM * (simd_gid / BN) : int(SM * simd_gid);
  const int simdN = BN != 1 ? SN * (simd_gid % BN) : 0;
  int bm = (simdM + thrM) * TM;
  int bn = (simdN + thrN) * TN;
  int out_row = tid.x * blockM + bm;
  if (out_row >= out_vec_size) return;
  out_row = out_row + TM <= out_vec_size ? out_row : out_vec_size - TM;
  const device T* m = mat + size_t(out_row) * matrix_ld;

  const int n_iter = in_vec_size / blockN;
  const int last_iter = blockN * n_iter;
  const int leftover = in_vec_size - last_iter;
  for (int i = 0; i < n_iter; ++i) {
    for (int v = 0; v < V; v++)
      for (int tn = 0; tn < TN; tn++) v_coeff[v][tn] = static_cast<float>(vec[v * in_vec_size + bn + tn]);
    int mat_offset = 0;
    for (int tm = 0; tm < TM; tm++) {
      for (int tn = 0; tn < TN; tn++) inter[tn] = m[mat_offset + bn + tn];
      for (int v = 0; v < V; v++)
        for (int tn = 0; tn < TN; tn++) result[v][tm] += inter[tn] * v_coeff[v][tn];
      mat_offset += matrix_ld;
    }
    bn += blockN;
  }
  if (leftover > 0) {
    for (int v = 0; v < V; v++) {
      if (bn + TN <= in_vec_size) {
        for (int tn = 0; tn < TN; tn++) v_coeff[v][tn] = static_cast<float>(vec[v * in_vec_size + bn + tn]);
      } else {
        for (int tn = 0; tn < TN; tn++)
          v_coeff[v][tn] = bn + tn < in_vec_size ? static_cast<float>(vec[v * in_vec_size + bn + tn]) : float(0);
      }
    }
    for (int tm = 0; tm < TM; tm++) {
      const device T* src = &m[tm * matrix_ld];
      if (bn + TN <= in_vec_size) {
        for (int tn = 0; tn < TN; tn++) inter[tn] = src[bn + tn];
      } else {
        for (int tn = 0; tn < TN; tn++) inter[tn] = bn + tn < in_vec_size ? src[bn + tn] : T(0);
      }
      for (int v = 0; v < V; v++)
        for (int tn = 0; tn < TN; tn++) result[v][tm] += inter[tn] * v_coeff[v][tn];
    }
  }
  for (int v = 0; v < V; v++)
    for (int tm = 0; tm < TM; tm++)
      for (ushort sn = (SN / 2); sn >= 1; sn >>= 1) result[v][tm] += simd_shuffle_down(result[v][tm], sn);
  if (BN > 1) {
    threadgroup float* tgp_results = tgp_memory + sgN * (blockM + TM) + bm;
    if (thrN == 0) {
      for (int v = 0; v < V; v++)
        for (int tm = 0; tm < TM; tm++) tgp_results[v * BN * (blockM + TM) + tm] = result[v][tm];
      threadgroup_barrier(mem_flags::mem_none);
      if (sgN == 0) {
        for (int sgn = 1; sgn < BN; sgn++)
          for (int v = 0; v < V; v++)
            for (int tm = 0; tm < TM; tm++)
              result[v][tm] += tgp_results[v * BN * (blockM + TM) + sgn * (blockM + TM) + tm];
      }
    }
  }
  if (simdN == 0 && thrN == 0) {
    for (int v = 0; v < V; v++)
      for (int tm = 0; tm < TM; tm++) out[size_t(v) * out_vec_size + out_row + tm] = static_cast<T>(result[v][tm]);
  }
)";

// The gemv tiling MLX 0.32 dispatches for a [N, K] matrix (matmul.cpp's gemv, the non-transposed case); only the
// lanes over K (SN, TN, BN, SM) decide a row's sums, the rest how rows group into threadgroups.
struct Tiling {
  int bm, bn, sm, sn, tm, tn;
};
Tiling tiling(int K, int N) {
  Tiling t{N >= 4096 ? 8 : 4, 1, 1, 32, 4, 4};
  if (K <= 64) t.bm = 1, t.sm = 8, t.sn = 4;
  else if (K >= 16 * N) t.bm = 1, t.bn = 8;
  t.tm = N < t.tm ? 1 : t.tm;
  return t;
}

}  // namespace

mx::array gemv_rows(const mx::array& x, const mx::array& w) {
  if (x.ndim() != 2 || w.ndim() != 2 || x.shape(1) != w.shape(1) || x.dtype() != w.dtype())
    throw std::runtime_error("gemv_rows: want x [V, K] and w [N, K] of one dtype");
  const int V = x.shape(0), K = w.shape(1), N = w.shape(0);
  if (V < 1 || V > kMaxRows) throw std::runtime_error("gemv_rows: " + std::to_string(V) + " rows");
  static auto kernel = mx::fast::metal_kernel("gemv_rows", {"mat", "vec", "K", "N"}, {"out"}, kSource);
  const Tiling t = tiling(K, N);
  const int per_tgp = t.bm * t.sm * t.tm, n_tgp = (N + per_tgp - 1) / per_tgp;
  return kernel({w, x, mx::array(K), mx::array(N)}, {{V, N}}, {w.dtype()}, {32 * n_tgp, t.bn, t.bm},
                {32, t.bn, t.bm},
                {{"T", w.dtype()}, {"V", V}, {"BM", t.bm}, {"BN", t.bn}, {"SM", t.sm}, {"SN", t.sn}, {"TM", t.tm},
                 {"TN", t.tn}},
                std::nullopt, false, {})[0];
}

mx::array linear_rows(const mx::array& x, const Proj& p) {
  if (p.q) throw std::runtime_error("linear_rows: exact decode steps run bf16 weights, not quantized ones");
  const int V = x.shape(1), K = x.shape(2);
  const mx::array rows = mx::reshape(x, {V, K});
  mx::array y = mx::reshape(gemv_rows(rows, p.w), {1, V, p.w.shape(0)});
  if (!p.lora) return y;
  // As linear(): x promoted to float32 for x A^T, then (x A^T) B^T, scaled, rounded to y's dtype and added.
  const mx::array xa = gemv_rows(mx::astype(rows, mx::float32), p.lora->a);
  const mx::array z = mx::reshape(gemv_rows(xa, p.lora->b), {1, V, p.w.shape(0)});
  return y + mx::astype(z * mx::array(p.lora->scale, mx::float32), y.dtype());
}

mx::array forward_rows(const Model& m, const std::vector<int>& tokens, KVCache& cache, RowStates& rows) {
  rows.conv.assign(m.layers.size(), {});
  rows.state.assign(m.layers.size(), {});
  return mx::reshape(forward(m, mx::array(tokens.data(), {1, int(tokens.size())}, mx::int32), &cache, false, &rows),
                     {int(tokens.size()), m.cfg.vocab});
}

void keep_rows(KVCache& cache, const RowStates& rows, int pass_rows, int n) {
  if (n < 1 || n > pass_rows) throw std::runtime_error("keep_rows: " + std::to_string(n) + " of " +
                                                       std::to_string(pass_rows));
  cache.offset -= pass_rows - n;
  for (size_t l = 0; l < cache.layers.size(); ++l)
    if (l < rows.state.size() && !rows.state[l].empty()) {
      cache.layers[l].conv = rows.conv[l][n - 1];
      cache.layers[l].state = rows.state[l][n - 1];
    }
}
