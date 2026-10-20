#pragma once
// Several decode steps in one pass, each bit for bit the single-token step: what exact speculative decoding verifies
// drafts with. A one-token step's projections run through MLX's gemv (M = 1); a pass over V tokens through mx::matmul
// would run its gemv_wide or GEMM kernels, which sum in another order and so round differently. Here every row is
// summed exactly as MLX's gemv sums one vector (the same lanes, the same order, the same reductions), but the matrix is
// read once for all V rows.
#include <vector>

#include "model.h"

// x [V, K] (bf16 or float32, as w) times w [N, K] transposed: [V, N], row v bit-identical to
// mx::matmul(x[v] as [1, 1, K], transpose(w)). V from 1 to kMaxRows.
constexpr int kMaxRows = 16;
mx::array gemv_rows(const mx::array& x, const mx::array& w);

// linear(x, p) for x [1, V, in], each row bit-identical to linear() of that row alone (its adapter's low-rank path
// included, in float32 as linear() runs it).
mx::array linear_rows(const mx::array& x, const Proj& p);

// tokens [V] after the cache: float32 logits [V, vocab], row j bit-identical to forward() of tokens[j] alone after
// tokens[0, j) (one decode step each). The cache's attention keys and values gain all V rows and its offset moves by
// V; the DeltaNet layers' conv and state are left at the last row, and `rows` gets every row's, so a caller that
// keeps only the first a + 1 tokens sets the cache back with keep_rows.
mx::array forward_rows(const Model& m, const std::vector<int>& tokens, KVCache& cache, RowStates& rows);
// The cache as forward_rows left it, cut back to its first n rows of that pass (n >= 1).
void keep_rows(KVCache& cache, const RowStates& rows, int pass_rows, int n);
