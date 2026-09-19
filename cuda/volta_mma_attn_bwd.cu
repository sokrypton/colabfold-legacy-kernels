// Backward pass for volta_mma_attn.cu (sm_75+), with mma.sync.m16n8k8.
//
// The forward is flash attention: it never materialises the [Sq, Sk] softmax,
// so the backward cannot read it back and has to recompute it. What it does
// NOT recompute is the row statistic -- VoltaMmaFwd returns `lse` (log2 domain,
// m + log2(l)) alongside the output, which is enough to rebuild P in one pass
// instead of two.
//
// Inputs   q,k,v   [N,H,Sq|Sk,D] f16
//          bias    [H,Sq,Sk] f16        (shared across the batch)
//          kmask   [N,Sk] u8
//          dout    [N,H,Sq,D] f16
//          lse     [N,H,Sq] f32         from VoltaMmaFwd
//          delta   [N,H,Sq] f32         rowsum(dout * out); one line of XLA,
//                                       so it is not worth a kernel here
// Outputs  dq,dk,dv [same as q,k,v] f16
//          dbias   [H,Sq,Sk] f32        summed over the batch with atomics
//
// dBIAS IS THE REASON THIS EXISTS. ColabFold's AF2 path never needs it -- its
// pair bias is not on the gradient path -- but AlphaFold 3 reaches the pair
// representation THROUGH the attention bias, so an attention backward without
// dBias is useless to it.
//
// Two kernels, split the standard way so that only dbias needs atomics:
//   dq kernel:  one block owns 16 query rows per warp and loops over all keys.
//   dkdv kernel: one block owns 16 key rows per warp and loops over all
//                queries, working on the transposed problem.
//
// Fragment layout for m16n8k8 (CUTLASS, and the same in the forward):
//   A[m][k]: lane holds m = lr, lr+8   and k = lc, lc+1
//   B[k][n]: lane holds k = lc, lc+1   and n = lr
//   C[m][n]: lane holds m = lr, lr+8   and n = lc, lc+1
// where lr = lane>>2 and lc = (lane&3)*2. A and C share a layout, so a
// computed tile becomes the next GEMM's A operand with only a cast.

#include <cuda_fp16.h>
#include <cstdint>
#include <string>

#include "cutlass/cutlass.h"
#include "cutlass/arch/mma.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

#define WARP 32
#define MMA_M 16
#define MMA_N 8
#define MMA_K 8
#define LOG2E 1.4426950408889634f

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 750)
#error "volta_mma_attn_bwd.cu requires sm_75+."
#endif

#define MAX_DEVICES 16
static int bwd_shared_limit(int device) {
    static int cache[MAX_DEVICES] = {};
    if (device < 0 || device >= MAX_DEVICES) {
        return 0;
    }
    if (cache[device] == 0) {
        cudaDeviceGetAttribute(&cache[device], cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    }
    return cache[device];
}

static bool bwd_needs_optin(int device, const void* fn) {
    static const void* cache[MAX_DEVICES] = {};
    if (device < 0 || device >= MAX_DEVICES) {
        return true;
    }
    if (cache[device] == fn) {
        return false;
    }
    cache[device] = fn;
    return true;
}

using MmaOp =
    cutlass::arch::Mma<cutlass::gemm::GemmShape<MMA_M, MMA_N, MMA_K>, WARP, cutlass::half_t,
                       cutlass::layout::RowMajor, cutlass::half_t, cutlass::layout::ColumnMajor,
                       float, cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAdd>;

using FragA = cutlass::Array<cutlass::half_t, 4>;
using FragB = cutlass::Array<cutlass::half_t, 2>;
using FragC = cutlass::Array<float, 4>;

// Stage a [rows, D] tile of a [N,H,S,D] tensor, zero-padded past the end.
__device__ inline void load_tile(__half* dst, const __half* src, long long base, int row0, int rows,
                                 int S, int D, int tid, int nthreads) {
    for (int i = tid; i < rows * D; i += nthreads) {
        int r = i / D, c = i - r * D;
        int g = row0 + r;
        dst[i] = (g < S) ? src[base + (long long)g * D + c] : __float2half(0.f);
    }
}

// -----------------------------------------------------------------------------
// dQ (and dBias): 16 query rows per warp, looping over every key.
// -----------------------------------------------------------------------------
template <int D, int BQ, int BK>
__global__ __launch_bounds__(BQ / MMA_M * WARP) void volta_mma_bwd_dq_kernel(
    const __half* __restrict__ q, const __half* __restrict__ k, const __half* __restrict__ v,
    const __half* __restrict__ bias, const uint8_t* __restrict__ kmask,
    const __half* __restrict__ dout, const float* __restrict__ lse,
    const float* __restrict__ delta, __half* __restrict__ dq, float* __restrict__ dbias, int N,
    int H, int Sq, int Sk, float sm_scale) {
    constexpr int NWARP = BQ / MMA_M;
    constexpr int NK = BK / MMA_N;
    constexpr int ND = D / MMA_N;
    constexpr int KSTEP = D / MMA_K;

    const int lane = threadIdx.x & (WARP - 1);
    const int warp = threadIdx.x / WARP;
    const int tid = threadIdx.x;
    const int nthreads = NWARP * WARP;

    const int qtile = blockIdx.x, h = blockIdx.y, n = blockIdx.z;
    const int q0 = qtile * BQ;
    if (q0 >= Sq) {
        return;
    }
    const int lr = lane >> 2;
    const int lc = (lane & 3) * 2;

    extern __shared__ char smem[];
    __half* Qs = reinterpret_cast<__half*>(smem);
    __half* Os = Qs + BQ * D;   // dout tile
    __half* Ks = Os + BQ * D;
    __half* Vs = Ks + BK * D;
    __half* Bs = Vs + BK * D;   // [BQ][BK]

    const long long qkv = (long long)(n * H + h) * Sq * D;
    const long long kv = (long long)(n * H + h) * Sk * D;
    const long long bh = (long long)h * Sq * Sk;
    const long long rowbase = (long long)(n * H + h) * Sq;

    load_tile(Qs, q, qkv, q0, BQ, Sq, D, tid, nthreads);
    load_tile(Os, dout, qkv, q0, BQ, Sq, D, tid, nthreads);
    __syncthreads();

    FragA fq[KSTEP], fo[KSTEP];
#pragma unroll
    for (int ks = 0; ks < KSTEP; ++ks) {
        const int qrow = warp * MMA_M + lr;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            fq[ks][0 + j] = reinterpret_cast<cutlass::half_t&>(Qs[qrow * D + ks * MMA_K + lc + j]);
            fq[ks][2 + j] =
                reinterpret_cast<cutlass::half_t&>(Qs[(qrow + 8) * D + ks * MMA_K + lc + j]);
            fo[ks][0 + j] = reinterpret_cast<cutlass::half_t&>(Os[qrow * D + ks * MMA_K + lc + j]);
            fo[ks][2 + j] =
                reinterpret_cast<cutlass::half_t&>(Os[(qrow + 8) * D + ks * MMA_K + lc + j]);
        }
    }

    // Per-lane row statistics: the lane's two query rows.
    float lse_r[2], del_r[2];
#pragma unroll
    for (int half_i = 0; half_i < 2; ++half_i) {
        const int gq = q0 + warp * MMA_M + lr + half_i * 8;
        lse_r[half_i] = (gq < Sq) ? lse[rowbase + gq] : 0.f;
        del_r[half_i] = (gq < Sq) ? delta[rowbase + gq] : 0.f;
    }

    FragC acc[ND];
#pragma unroll
    for (int d = 0; d < ND; ++d) {
        acc[d].clear();
    }

    MmaOp mma_op;
    const float qk_scale = sm_scale * LOG2E;

    for (int k0 = 0; k0 < Sk; k0 += BK) {
        __syncthreads();
        for (int i = tid; i < BK * D; i += nthreads) {
            int r = i / D, c = i - r * D;
            int gk = k0 + r;
            bool ok = (gk < Sk);
            Ks[i] = ok ? k[kv + (long long)gk * D + c] : __float2half(0.f);
            Vs[i] = ok ? v[kv + (long long)gk * D + c] : __float2half(0.f);
        }
        for (int i = tid; i < BQ * BK; i += nthreads) {
            int r = i / BK, c = i - r * BK;
            int gq = q0 + r, gk = k0 + c;
            Bs[i] = (gq < Sq && gk < Sk) ? bias[bh + (long long)gq * Sk + gk] : __float2half(0.f);
        }
        __syncthreads();

        // S = Q @ K^T and dP = dO @ V^T, the same GEMM shape twice.
        FragC s[NK], dp[NK];
#pragma unroll
        for (int nt = 0; nt < NK; ++nt) {
            s[nt].clear();
            dp[nt].clear();
#pragma unroll
            for (int ks = 0; ks < KSTEP; ++ks) {
                FragB fk, fv;
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    fk[j] = reinterpret_cast<cutlass::half_t&>(
                        Ks[(nt * MMA_N + lr) * D + ks * MMA_K + lc + j]);
                    fv[j] = reinterpret_cast<cutlass::half_t&>(
                        Vs[(nt * MMA_N + lr) * D + ks * MMA_K + lc + j]);
                }
                mma_op(s[nt], fq[ks], fk, s[nt]);
                mma_op(dp[nt], fo[ks], fv, dp[nt]);
            }
        }

        // P from the forward's statistic; dS = P * (dP - delta).
        FragA ds[NK];
#pragma unroll
        for (int nt = 0; nt < NK; ++nt) {
#pragma unroll
            for (int half_i = 0; half_i < 2; ++half_i) {
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int idx = half_i * 2 + j;
                    const int r_loc = warp * MMA_M + lr + half_i * 8;
                    const int gq = q0 + r_loc;
                    const int gk = k0 + nt * MMA_N + lc + j;
                    float p = 0.f;
                    if (gq < Sq && gk < Sk && kmask[(long long)n * Sk + gk] != 0) {
                        const float l2 = s[nt][idx] * qk_scale +
                                         __half2float(Bs[r_loc * BK + nt * MMA_N + lc + j]) * LOG2E;
                        p = exp2f(l2 - lse_r[half_i]);
                    }
                    const float g = p * (dp[nt][idx] - del_r[half_i]);
                    ds[nt][idx] = cutlass::half_t(g);
                    // dBias sees the gradient of the pre-softmax logit itself,
                    // unscaled: the bias is added AFTER the q.k scaling.
                    if (dbias != nullptr && gq < Sq && gk < Sk && g != 0.f) {
                        atomicAdd(&dbias[bh + (long long)gq * Sk + gk], g);
                    }
                }
            }
        }

        // dQ += dS @ K   (dS is in C layout, which is also the A layout)
#pragma unroll
        for (int d = 0; d < ND; ++d) {
#pragma unroll
            for (int nt = 0; nt < NK; ++nt) {
                FragB fk;
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    fk[j] = reinterpret_cast<cutlass::half_t&>(
                        Ks[(nt * MMA_N + lc + j) * D + d * MMA_N + lr]);
                }
                mma_op(acc[d], ds[nt], fk, acc[d]);
            }
        }
    }

#pragma unroll
    for (int d = 0; d < ND; ++d) {
#pragma unroll
        for (int half_i = 0; half_i < 2; ++half_i) {
            const int gq = q0 + warp * MMA_M + lr + half_i * 8;
            if (gq >= Sq) {
                continue;
            }
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int col = d * MMA_N + lc + j;
                dq[qkv + (long long)gq * D + col] =
                    __float2half(acc[d][half_i * 2 + j] * sm_scale);
            }
        }
    }
}

// -----------------------------------------------------------------------------
// dK and dV: 16 key rows per warp, looping over every query. Everything is the
// transposed problem, so the key index is the GEMM's m and the query its n.
// -----------------------------------------------------------------------------
template <int D, int BK, int BQ>
__global__ __launch_bounds__(BK / MMA_M * WARP) void volta_mma_bwd_dkdv_kernel(
    const __half* __restrict__ q, const __half* __restrict__ k, const __half* __restrict__ v,
    const __half* __restrict__ bias, const uint8_t* __restrict__ kmask,
    const __half* __restrict__ dout, const float* __restrict__ lse,
    const float* __restrict__ delta, __half* __restrict__ dk, __half* __restrict__ dv, int N,
    int H, int Sq, int Sk, float sm_scale) {
    constexpr int NWARP = BK / MMA_M;
    constexpr int NQ = BQ / MMA_N;
    constexpr int ND = D / MMA_N;
    constexpr int KSTEP = D / MMA_K;

    const int lane = threadIdx.x & (WARP - 1);
    const int warp = threadIdx.x / WARP;
    const int tid = threadIdx.x;
    const int nthreads = NWARP * WARP;

    const int ktile = blockIdx.x, h = blockIdx.y, n = blockIdx.z;
    const int k0 = ktile * BK;
    if (k0 >= Sk) {
        return;
    }
    const int lr = lane >> 2;
    const int lc = (lane & 3) * 2;

    extern __shared__ char smem[];
    __half* Ks = reinterpret_cast<__half*>(smem);
    __half* Vs = Ks + BK * D;
    __half* Qs = Vs + BK * D;
    __half* Os = Qs + BQ * D;      // dout tile
    __half* Bs = Os + BQ * D;      // [BK][BQ], transposed
    float* Ls = reinterpret_cast<float*>(Bs + BK * BQ);   // lse   [BQ]
    float* Ds = Ls + BQ;                                  // delta [BQ]

    const long long qkv = (long long)(n * H + h) * Sq * D;
    const long long kv = (long long)(n * H + h) * Sk * D;
    const long long bh = (long long)h * Sq * Sk;
    const long long rowbase = (long long)(n * H + h) * Sq;

    load_tile(Ks, k, kv, k0, BK, Sk, D, tid, nthreads);
    load_tile(Vs, v, kv, k0, BK, Sk, D, tid, nthreads);
    __syncthreads();

    FragA fk[KSTEP], fv[KSTEP];
#pragma unroll
    for (int ks = 0; ks < KSTEP; ++ks) {
        const int krow = warp * MMA_M + lr;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            fk[ks][0 + j] = reinterpret_cast<cutlass::half_t&>(Ks[krow * D + ks * MMA_K + lc + j]);
            fk[ks][2 + j] =
                reinterpret_cast<cutlass::half_t&>(Ks[(krow + 8) * D + ks * MMA_K + lc + j]);
            fv[ks][0 + j] = reinterpret_cast<cutlass::half_t&>(Vs[krow * D + ks * MMA_K + lc + j]);
            fv[ks][2 + j] =
                reinterpret_cast<cutlass::half_t&>(Vs[(krow + 8) * D + ks * MMA_K + lc + j]);
        }
    }

    // A key row that the mask kills contributes nothing to dK or dV.
    bool live[2];
#pragma unroll
    for (int half_i = 0; half_i < 2; ++half_i) {
        const int gk = k0 + warp * MMA_M + lr + half_i * 8;
        live[half_i] = (gk < Sk) && (kmask[(long long)n * Sk + gk] != 0);
    }

    FragC acc_dk[ND], acc_dv[ND];
#pragma unroll
    for (int d = 0; d < ND; ++d) {
        acc_dk[d].clear();
        acc_dv[d].clear();
    }

    MmaOp mma_op;
    const float qk_scale = sm_scale * LOG2E;

    for (int qq = 0; qq < Sq; qq += BQ) {
        __syncthreads();
        load_tile(Qs, q, qkv, qq, BQ, Sq, D, tid, nthreads);
        load_tile(Os, dout, qkv, qq, BQ, Sq, D, tid, nthreads);
        for (int i = tid; i < BK * BQ; i += nthreads) {   // bias tile, transposed
            int r = i / BQ, c = i - r * BQ;
            int gk = k0 + r, gq = qq + c;
            Bs[i] = (gq < Sq && gk < Sk) ? bias[bh + (long long)gq * Sk + gk] : __float2half(0.f);
        }
        for (int i = tid; i < BQ; i += nthreads) {
            int gq = qq + i;
            Ls[i] = (gq < Sq) ? lse[rowbase + gq] : 0.f;
            Ds[i] = (gq < Sq) ? delta[rowbase + gq] : 0.f;
        }
        __syncthreads();

        // S^T = K @ Q^T and dP^T = V @ dO^T
        FragC st[NQ], dpt[NQ];
#pragma unroll
        for (int nt = 0; nt < NQ; ++nt) {
            st[nt].clear();
            dpt[nt].clear();
#pragma unroll
            for (int ks = 0; ks < KSTEP; ++ks) {
                FragB fqb, fob;
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    fqb[j] = reinterpret_cast<cutlass::half_t&>(
                        Qs[(nt * MMA_N + lr) * D + ks * MMA_K + lc + j]);
                    fob[j] = reinterpret_cast<cutlass::half_t&>(
                        Os[(nt * MMA_N + lr) * D + ks * MMA_K + lc + j]);
                }
                mma_op(st[nt], fk[ks], fqb, st[nt]);
                mma_op(dpt[nt], fv[ks], fob, dpt[nt]);
            }
        }

        FragA pt[NQ], dst[NQ];
#pragma unroll
        for (int nt = 0; nt < NQ; ++nt) {
#pragma unroll
            for (int half_i = 0; half_i < 2; ++half_i) {
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int idx = half_i * 2 + j;
                    const int k_loc = warp * MMA_M + lr + half_i * 8;
                    const int q_loc = nt * MMA_N + lc + j;
                    const int gq = qq + q_loc;
                    float p = 0.f;
                    if (live[half_i] && gq < Sq) {
                        const float l2 = st[nt][idx] * qk_scale +
                                         __half2float(Bs[k_loc * BQ + q_loc]) * LOG2E;
                        p = exp2f(l2 - Ls[q_loc]);
                    }
                    pt[nt][idx] = cutlass::half_t(p);
                    dst[nt][idx] = cutlass::half_t(p * (dpt[nt][idx] - Ds[q_loc]));
                }
            }
        }

        // dV += P^T @ dO ; dK += dS^T @ Q
#pragma unroll
        for (int d = 0; d < ND; ++d) {
#pragma unroll
            for (int nt = 0; nt < NQ; ++nt) {
                FragB fo, fq2;
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    fo[j] = reinterpret_cast<cutlass::half_t&>(
                        Os[(nt * MMA_N + lc + j) * D + d * MMA_N + lr]);
                    fq2[j] = reinterpret_cast<cutlass::half_t&>(
                        Qs[(nt * MMA_N + lc + j) * D + d * MMA_N + lr]);
                }
                mma_op(acc_dv[d], pt[nt], fo, acc_dv[d]);
                mma_op(acc_dk[d], dst[nt], fq2, acc_dk[d]);
            }
        }
    }

#pragma unroll
    for (int d = 0; d < ND; ++d) {
#pragma unroll
        for (int half_i = 0; half_i < 2; ++half_i) {
            const int gk = k0 + warp * MMA_M + lr + half_i * 8;
            if (gk >= Sk) {
                continue;
            }
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int col = d * MMA_N + lc + j;
                const int idx = half_i * 2 + j;
                dk[kv + (long long)gk * D + col] = __float2half(acc_dk[d][idx] * sm_scale);
                dv[kv + (long long)gk * D + col] = __float2half(acc_dv[d][idx]);
            }
        }
    }
}

__global__ void zero_f32(float* p, long long n) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x) {
        p[i] = 0.f;
    }
}

template <int D, int BQ, int BK>
static ffi::Error launch_bwd(cudaStream_t stream, int device, const __half* q, const __half* k,
                             const __half* v, const __half* bias, const uint8_t* kmask,
                             const __half* dout, const float* lse, const float* delta, __half* dq,
                             __half* dk, __half* dv, float* dbias, int N, int H, int Sq, int Sk,
                             float scale) {
    const size_t smem_dq = (size_t)(2 * BQ * D + 2 * BK * D + BQ * BK) * sizeof(__half);
    const size_t smem_dkdv =
        (size_t)(2 * BK * D + 2 * BQ * D + BK * BQ) * sizeof(__half) + 2 * BQ * sizeof(float);
    const int max_smem = bwd_shared_limit(device);
    const size_t need = smem_dq > smem_dkdv ? smem_dq : smem_dkdv;
    if ((int)need > max_smem) {
        return ffi::Error::InvalidArgument("volta_mma_bwd: needs " + std::to_string(need / 1024) +
                                           " KB shared, device allows " +
                                           std::to_string(max_smem / 1024) + " KB");
    }

    if (dbias != nullptr) {
        const long long nb = (long long)H * Sq * Sk;
        zero_f32<<<256, 256, 0, stream>>>(dbias, nb);
    }

    auto kern_dq = volta_mma_bwd_dq_kernel<D, BQ, BK>;
    if (smem_dq > 48 * 1024 && bwd_needs_optin(device, (const void*)kern_dq)) {
        cudaFuncSetAttribute(kern_dq, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_dq);
    }
    dim3 grid_dq((Sq + BQ - 1) / BQ, H, N);
    kern_dq<<<grid_dq, (BQ / MMA_M) * WARP, smem_dq, stream>>>(
        q, k, v, bias, kmask, dout, lse, delta, dq, dbias, N, H, Sq, Sk, scale);

    auto kern_dkdv = volta_mma_bwd_dkdv_kernel<D, BK, BQ>;
    if (smem_dkdv > 48 * 1024 && bwd_needs_optin(device, (const void*)kern_dkdv)) {
        cudaFuncSetAttribute(kern_dkdv, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem_dkdv);
    }
    dim3 grid_dkdv((Sk + BK - 1) / BK, H, N);
    kern_dkdv<<<grid_dkdv, (BK / MMA_M) * WARP, smem_dkdv, stream>>>(
        q, k, v, bias, kmask, dout, lse, delta, dk, dv, N, H, Sq, Sk, scale);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error::Internal(std::string("volta_mma_bwd launch: ") +
                                    cudaGetErrorString(err));
    }
    return ffi::Error::Success();
}

ffi::Error VoltaMmaBwdImpl(cudaStream_t stream, int32_t device, ffi::Buffer<ffi::DataType::F16> q,
                           ffi::Buffer<ffi::DataType::F16> k, ffi::Buffer<ffi::DataType::F16> v,
                           ffi::Buffer<ffi::DataType::F16> bias,
                           ffi::Buffer<ffi::DataType::U8> kmask,
                           ffi::Buffer<ffi::DataType::F16> dout,
                           ffi::Buffer<ffi::DataType::F32> lse,
                           ffi::Buffer<ffi::DataType::F32> delta,
                           ffi::Result<ffi::Buffer<ffi::DataType::F16>> dq,
                           ffi::Result<ffi::Buffer<ffi::DataType::F16>> dk,
                           ffi::Result<ffi::Buffer<ffi::DataType::F16>> dv,
                           ffi::Result<ffi::Buffer<ffi::DataType::F32>> dbias, float scale,
                           int64_t block_q, int64_t block_k) {
    auto d = q.dimensions();
    if (d.size() != 4) {
        return ffi::Error::InvalidArgument("q must be [N,H,S,D]");
    }
    const int N = (int)d[0], H = (int)d[1], Sq = (int)d[2], D = (int)d[3];
    const int Sk = (int)k.dimensions()[2];
    const __half* qp = reinterpret_cast<const __half*>(q.typed_data());
    const __half* kp = reinterpret_cast<const __half*>(k.typed_data());
    const __half* vp = reinterpret_cast<const __half*>(v.typed_data());
    const __half* bp = reinterpret_cast<const __half*>(bias.typed_data());
    const uint8_t* mp = kmask.typed_data();
    const __half* dop = reinterpret_cast<const __half*>(dout.typed_data());
    const float* lp = lse.typed_data();
    const float* dlp = delta.typed_data();
    __half* dqp = reinterpret_cast<__half*>(dq->typed_data());
    __half* dkp = reinterpret_cast<__half*>(dk->typed_data());
    __half* dvp = reinterpret_cast<__half*>(dv->typed_data());
    float* dbp = reinterpret_cast<float*>(dbias->typed_data());

#define DISPATCH_BWD(DD, BQ, BK)                                                                   \
    if (D == (DD) && block_q == (BQ) && block_k == (BK))                                           \
        return launch_bwd<DD, BQ, BK>(stream, device, qp, kp, vp, bp, mp, dop, lp, dlp, dqp, dkp,  \
                                      dvp, dbp, N, H, Sq, Sk, scale);
    DISPATCH_BWD(8, 64, 64)
    DISPATCH_BWD(8, 64, 32)
    DISPATCH_BWD(8, 32, 32)
    DISPATCH_BWD(16, 64, 64)
    DISPATCH_BWD(16, 64, 32)
    DISPATCH_BWD(16, 32, 32)
    DISPATCH_BWD(32, 64, 64)
    DISPATCH_BWD(32, 64, 32)
    DISPATCH_BWD(32, 32, 32)
    DISPATCH_BWD(64, 64, 64)
    DISPATCH_BWD(64, 64, 32)
    DISPATCH_BWD(64, 32, 32)
#undef DISPATCH_BWD
    return ffi::Error::InvalidArgument("volta_mma_bwd: unsupported (D, bq, bk)");
}

// NOT kCmdBufferCompatible: this handler starts three kernels, and the first
// one zeroes the dbias accumulator.
XLA_FFI_DEFINE_HANDLER_SYMBOL(VoltaMmaBwd, VoltaMmaBwdImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Ctx<ffi::DeviceOrdinal>()
                                  .Arg<ffi::Buffer<ffi::DataType::F16>>()
                                  .Arg<ffi::Buffer<ffi::DataType::F16>>()
                                  .Arg<ffi::Buffer<ffi::DataType::F16>>()
                                  .Arg<ffi::Buffer<ffi::DataType::F16>>()
                                  .Arg<ffi::Buffer<ffi::DataType::U8>>()
                                  .Arg<ffi::Buffer<ffi::DataType::F16>>()
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()
                                  .Ret<ffi::Buffer<ffi::DataType::F16>>()
                                  .Ret<ffi::Buffer<ffi::DataType::F16>>()
                                  .Ret<ffi::Buffer<ffi::DataType::F16>>()
                                  .Ret<ffi::Buffer<ffi::DataType::F32>>()
                                  .Attr<float>("scale")
                                  .Attr<int64_t>("block_q")
                                  .Attr<int64_t>("block_k"));
