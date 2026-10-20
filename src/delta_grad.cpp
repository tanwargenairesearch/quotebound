#include "delta_grad.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace {

// Steps between saved states. The backward holds a chunk's sub-chunk starts in registers (C / SUB of them) and one
// sub-chunk's states (SUB + 1), so C = SUB^2 keeps both small.
constexpr int kChunk = 64;

// Per value head, with S the state [Dv, Dk], step t is
//   S~ = g S_prev;  u = S~ k;  delta = beta (v - u);  S = S~ + delta k^T;  y = S q.
// Rows of S evolve independently, so as in the forward kernel one simdgroup owns one row and each lane Dk/32 of its
// columns. Going back over step t with dS the cotangent of S:
//   dS += dy q^T;  dq = S^T dy;  ddelta = dS k;  dk = dS^T delta;  dv = beta ddelta;  dbeta = sum(ddelta (v - u));
//   du = -beta ddelta;  dS~ = dS + du k^T;  dk += S~^T du;  dg = sum(dS~ * S_prev);  dS_prev = g dS~.
// dv, and dg and dbeta per row, are row-local. dq and dk sum over rows: the NSG rows of a threadgroup are summed in
// threadgroup memory in a fixed order, and the Dv / NSG row blocks and the value heads sharing a key head are
// summed outside, so the result does not depend on scheduling. Within a chunk the states are recomputed from the
// chunk's start, the sub-chunk starts kept in registers and one sub-chunk's states at a time.
const char* kBackwardSource = R"(
  constexpr int NPT = Dk / 32;
  constexpr int SUB = 8;
  constexpr int NSUB = C / SUB;
  constexpr int NB = Dv / NSG;
  threadgroup float red[NSG][2 * Dk];

  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const uint tid = thread_index_in_threadgroup;
  const int blk = threadgroup_position_in_grid.y;
  const int n = threadgroup_position_in_grid.z;
  const int b = n / Hv, hv = n % Hv, hk = hv / (Hv / Hk);
  const int row = blk * NSG + sg;
  const int NC = (T + C - 1) / C;

  const size_t qk0 = size_t(b) * T * Hk * Dk + hk * Dk + lane * NPT;
  const size_t vr0 = size_t(b) * T * Hv * Dv + hv * Dv + row;
  const size_t gb0 = size_t(b) * T * Hv + hv;
  const size_t part0 = (size_t(b) * T * Hv + hv) * NB * Dk + blk * Dk;
  const size_t s0 = (size_t(n) * Dv + row) * Dk + lane * NPT;

  float ds[NPT];
  for (int i = 0; i < NPT; ++i) ds[i] = dstate[s0 + i];

  for (int c = NC - 1; c >= 0; --c) {
    const int t0 = c * C, t1 = min(T, t0 + C);
    const size_t ck0 = ((size_t(b) * NC + c) * Hv + hv) * Dv * Dk + row * Dk + lane * NPT;
    float s[NPT];
    for (int i = 0; i < NPT; ++i) s[i] = ckpt[ck0 + i];

    float sub[NSUB][NPT];
    for (int m = 0; m < NSUB; ++m) {
      for (int i = 0; i < NPT; ++i) sub[m][i] = s[i];
      for (int tt = 0; tt < SUB; ++tt) {
        const int t = t0 + m * SUB + tt;
        if (t < t1) {
          const float gt = g[gb0 + t * Hv], bt = beta[gb0 + t * Hv], vt = v[vr0 + size_t(t) * Hv * Dv];
          const size_t kq = qk0 + size_t(t) * Hk * Dk;
          float u = 0.0f;
          for (int i = 0; i < NPT; ++i) {
            s[i] = s[i] * gt;
            u += s[i] * k[kq + i];
          }
          u = simd_sum(u);
          const float d = (vt - u) * bt;
          for (int i = 0; i < NPT; ++i) s[i] = s[i] + k[kq + i] * d;
        }
      }
    }

    for (int m = NSUB - 1; m >= 0; --m) {
      const int ts = t0 + m * SUB;
      if (ts >= t1) continue;
      float st[SUB + 1][NPT], del[SUB], uu[SUB];
      for (int i = 0; i < NPT; ++i) st[0][i] = sub[m][i];
      for (int tt = 0; tt < SUB; ++tt) {
        const int t = ts + tt;
        if (t < t1) {
          const float gt = g[gb0 + t * Hv], bt = beta[gb0 + t * Hv], vt = v[vr0 + size_t(t) * Hv * Dv];
          const size_t kq = qk0 + size_t(t) * Hk * Dk;
          float u = 0.0f;
          for (int i = 0; i < NPT; ++i) {
            st[tt + 1][i] = st[tt][i] * gt;
            u += st[tt + 1][i] * k[kq + i];
          }
          u = simd_sum(u);
          const float d = (vt - u) * bt;
          for (int i = 0; i < NPT; ++i) st[tt + 1][i] = st[tt + 1][i] + k[kq + i] * d;
          del[tt] = d;
          uu[tt] = u;
        }
      }

      for (int tt = SUB - 1; tt >= 0; --tt) {
        const int t = ts + tt;
        if (t >= t1) continue;  // the same for every thread of the threadgroup, so the barriers below are safe
        const float gt = g[gb0 + t * Hv], bt = beta[gb0 + t * Hv];
        const size_t vr = vr0 + size_t(t) * Hv * Dv, kq = qk0 + size_t(t) * Hk * Dk;
        const float vt = v[vr], dyt = dy[vr];
        float pq[NPT], pk[NPT];
        float dd = 0.0f;
        for (int i = 0; i < NPT; ++i) {
          ds[i] += dyt * q[kq + i];
          pq[i] = dyt * st[tt + 1][i];
          dd += ds[i] * k[kq + i];
        }
        dd = simd_sum(dd);
        const float du = -bt * dd;
        float dg = 0.0f;
        for (int i = 0; i < NPT; ++i) {
          pk[i] = ds[i] * del[tt] + du * (st[tt][i] * gt);
          ds[i] += du * k[kq + i];
          dg += ds[i] * st[tt][i];
          ds[i] *= gt;
        }
        dg = simd_sum(dg);
        if (lane == 0) {
          dv[vr] = bt * dd;
          dbeta_row[vr] = dd * (vt - uu[tt]);
          dg_row[vr] = dg;
        }
        for (int i = 0; i < NPT; ++i) {
          red[sg][lane * NPT + i] = pq[i];
          red[sg][Dk + lane * NPT + i] = pk[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // 2 Dk columns, 32 NSG threads: with few rows (small Dv) a thread sums several columns.
        const size_t at = part0 + size_t(t) * Hv * NB * Dk;
        for (uint col = tid; col < 2 * Dk; col += 32 * NSG) {
          float acc = 0.0f;
          for (int r = 0; r < NSG; ++r) acc += red[r][col];
          if (col < Dk)
            dq_part[at + col] = acc;
          else
            dk_part[at + col - Dk] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
  }
  for (int i = 0; i < NPT; ++i) dstate0[s0 + i] = ds[i];
)";

mx::array steps(const mx::array& x, int a, int b) {
  mx::Shape lo(x.ndim(), 0), hi = x.shape();
  lo[1] = a, hi[1] = b;
  return mx::slice(x, lo, hi);
}

}  // namespace

std::vector<mx::array> gated_delta_vjp(const std::vector<mx::array>& in, const mx::array& dy,
                                       const mx::array& dstate) {
  const mx::array &q = in[0], &k = in[1], &v = in[2], &g = in[3], &beta = in[4], &state = in[5];
  const int B = q.shape(0), T = q.shape(1), Hk = q.shape(2), Dk = q.shape(3), Hv = v.shape(2), Dv = v.shape(3);
  // Rows per threadgroup: at most 32 (32 simdgroups of 32 lanes is Metal's 1,024-thread limit), and few enough that
  // red[NSG][2 Dk] floats fit the 32 KB of threadgroup memory every Apple GPU has.
  constexpr int kThreadgroupBytes = 32768;
  int nsg = std::min(32, Dv);
  while (nsg > 1 && nsg * 2 * Dk * 4 > kThreadgroupBytes) nsg /= 2;
  if (Dk % 32 || Dv % nsg || Hv % Hk || nsg * 2 * Dk * 4 > kThreadgroupBytes)
    throw std::runtime_error("gated delta backward: needs Dk a multiple of 32 and at most " +
                             std::to_string(kThreadgroupBytes / 8) + ", Dv a multiple of " + std::to_string(nsg) +
                             " and value heads a multiple of key heads (Dk " + std::to_string(Dk) + ", Dv " +
                             std::to_string(Dv) + ", Hk " + std::to_string(Hk) + ", Hv " + std::to_string(Hv) + ")");
  for (const mx::array& x : in)
    if (x.dtype() != mx::float32) throw std::runtime_error("gated delta backward: inputs must be float32");

  // Chunk starts from the forward op itself, so the recomputed states begin from the forward's own bits.
  std::vector<mx::array> starts{state};
  mx::array s = state;
  for (int t = 0; t + kChunk < T; t += kChunk) {
    const int e = t + kChunk;
    s = mx::fast::gated_delta_update(steps(q, t, e), steps(k, t, e), steps(v, t, e), steps(g, t, e),
                                     steps(beta, t, e), s)[1];
    starts.push_back(s);
  }
  mx::array ckpt = mx::stack(starts, 1);  // [B, NC, Hv, Dv, Dk]

  static auto kernel = mx::fast::metal_kernel(
      "gated_delta_backward", {"q", "k", "v", "g", "beta", "ckpt", "dy", "dstate", "T"},
      {"dq_part", "dk_part", "dv", "dg_row", "dbeta_row", "dstate0"}, kBackwardSource);
  const int nb = Dv / nsg;
  auto f32 = [](const mx::array& x) { return mx::astype(x, mx::float32); };
  auto out = kernel({q, k, v, g, beta, ckpt, f32(dy), f32(dstate), mx::array(T)},
                    {{B, T, Hv, nb, Dk}, {B, T, Hv, nb, Dk}, {B, T, Hv, Dv}, {B, T, Hv, Dv}, {B, T, Hv, Dv},
                     {B, Hv, Dv, Dk}},
                    std::vector<mx::Dtype>(6, mx::float32), {32 * nsg, nb, B * Hv}, {32 * nsg, 1, 1},
                    {{"Dk", Dk}, {"Dv", Dv}, {"Hk", Hk}, {"Hv", Hv}, {"C", kChunk}, {"NSG", nsg}}, std::nullopt,
                    false, {});
  // Row blocks, then the value heads that share each key head.
  auto per_key_head = [&](const mx::array& part) {
    return mx::sum(mx::reshape(mx::sum(part, 3), {B, T, Hk, Hv / Hk, Dk}), 3);
  };
  return {per_key_head(out[0]), per_key_head(out[1]), out[2], mx::sum(out[3], -1), mx::sum(out[4], -1), out[5]};
}

std::pair<mx::array, mx::array> gated_delta_trainable(const mx::array& q, const mx::array& k, const mx::array& v,
                                                      const mx::array& g, const mx::array& beta,
                                                      const mx::array& state) {
  static auto fn = mx::custom_vjp(
      [](const std::vector<mx::array>& in) {
        return mx::fast::gated_delta_update(in[0], in[1], in[2], in[3], in[4], in[5]);
      },
      [](const std::vector<mx::array>& primals, const std::vector<mx::array>& cotangents,
         const std::vector<mx::array>&) { return gated_delta_vjp(primals, cotangents[0], cotangents[1]); });
  auto r = fn({q, k, v, g, beta, state});
  return {r[0], r[1]};
}
