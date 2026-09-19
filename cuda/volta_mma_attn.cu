// Flash attention for Turing and later (sm_75+), with CUTLASS mma.sync.m16n8k8.
// The accumulator and the operand A have the same fragment layout.
// Thus the softmax stays in registers. Volta cannot do this.
// Inputs: q,k,v [N,H,S,D] f16, bias [H,Sq,Sk] f16, kmask [N,Sk] u8.

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
#define NEG_F16 (-1.0e4f)
#define LOG2E 1.4426950408889634f

// CUTLASS gives no error for sm_70, but it builds a kernel with no tensor cores.
// Thus stop the build here.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 750)
#error "volta_mma_attn.cu requires sm_75+. Volta (sm_70) needs an m8n8k4 port."
#endif

#define MAX_DEVICES 16
static int device_shared_limit(int device) {
    static int cache[MAX_DEVICES] = {};
    if (device < 0 || device >= MAX_DEVICES) {
        return 0;
    }
    if (cache[device] == 0) {
        cudaDeviceGetAttribute(&cache[device], cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    }
    return cache[device];
}

// Tell if the kernel must get the opt-in shared memory attribute on this device.
static bool needs_smem_optin(int device, const void* fn) {
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

// BQ must be a multiple of 16 (one warp owns MMA_M=16 query rows).
// BK and D must be multiples of 8.
template <int D, int BQ, int BK>
__global__ __launch_bounds__(BQ / MMA_M * WARP) void volta_mma_kernel(
    const __half* __restrict__ q, const __half* __restrict__ k, const __half* __restrict__ v,
    const __half* __restrict__ bias, const uint8_t* __restrict__ kmask, __half* __restrict__ out,
    float* __restrict__ lse, int N, int H, int Sq, int Sk, float sm_scale) {
    constexpr int NWARP = BQ / MMA_M;
    constexpr int NK = BK / MMA_N;   // key sub-tiles per BK tile
    constexpr int ND = D / MMA_N;    // output dim sub-tiles
    constexpr int KSTEP = D / MMA_K; // contraction steps for Q@K^T

    const int lane = threadIdx.x & (WARP - 1);
    const int warp = threadIdx.x / WARP;
    const int tid = threadIdx.x;
    const int nthreads = NWARP * WARP;

    const int qtile = blockIdx.x, h = blockIdx.y, n = blockIdx.z;
    const int q0 = qtile * BQ;
    if (q0 >= Sq) {
        return;
    }

    // lane -> (row pair, column pair) for every fragment we touch
    const int lr = lane >> 2;                 // 0..7   : row within the 16-row tile
    const int lc = (lane & 3) * 2;            // 0,2,4,6: first of the lane's column pair
    const int row_a = q0 + warp * MMA_M + lr; // this lane's first query row
    const int row_b = row_a + 8;              // ... and its second

    extern __shared__ char smem[];
    __half* Qs = reinterpret_cast<__half*>(smem); // [BQ][D]
    __half* Ks = Qs + BQ * D;                     // [BK][D]
    __half* Vs = Ks + BK * D;                     // [BK][D]
    __half* Bs = Vs + BK * D;                     // [BQ][BK] bias tile

    const long long qkv = (long long)(n * H + h) * Sq * D;
    const long long kv = (long long)(n * H + h) * Sk * D;
    const long long bh = (long long)h * Sq * Sk;

    for (int i = tid; i < BQ * D; i += nthreads) {
        int r = i / D, c = i - r * D;
        int gq = q0 + r;
        Qs[i] = (gq < Sq) ? q[qkv + (long long)gq * D + c] : __float2half(0.f);
    }
    __syncthreads();

    // Q operand fragments: fixed for this warp, so hoist them out of the k-loop.
    FragA fq[KSTEP];
#pragma unroll
    for (int ks = 0; ks < KSTEP; ++ks) {
        const int qrow = warp * MMA_M + lr;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            fq[ks][0 + j] = reinterpret_cast<cutlass::half_t&>(Qs[(qrow)*D + ks * MMA_K + lc + j]);
            fq[ks][2 + j] =
                reinterpret_cast<cutlass::half_t&>(Qs[(qrow + 8) * D + ks * MMA_K + lc + j]);
        }
    }

    // Running online-softmax state and output accumulator -- all in registers.
    float m_run[2] = {-INFINITY, -INFINITY};
    float l_run[2] = {0.f, 0.f};
    FragC acc[ND];
#pragma unroll
    for (int d = 0; d < ND; ++d) {
        acc[d].clear();
    }

    MmaOp mma_op;
    const float qk_scale = sm_scale * LOG2E; // fold log2(e) so we can use exp2

    for (int k0 = 0; k0 < Sk; k0 += BK) {
        __syncthreads();
        for (int i = tid; i < BK * D; i += nthreads) {
            int r = i / D, c = i - r * D;
            int gk = k0 + r;
            bool ok = (gk < Sk);
            Ks[i] = ok ? k[kv + (long long)gk * D + c] : __float2half(0.f);
            Vs[i] = ok ? v[kv + (long long)gk * D + c] : __float2half(0.f);
        }
        for (int i = tid; i < BQ * BK; i += nthreads) { // coalesced bias staging
            int r = i / BK, c = i - r * BK;
            int gq = q0 + r, gk = k0 + c;
            Bs[i] = (gq < Sq && gk < Sk) ? bias[bh + (long long)gq * Sk + gk] : __float2half(0.f);
        }
        __syncthreads();

        // S = Q @ K^T, then softmax, entirely in registers
        FragC s[NK];
#pragma unroll
        for (int nt = 0; nt < NK; ++nt) {
            s[nt].clear();
#pragma unroll
            for (int ks = 0; ks < KSTEP; ++ks) {
                // B is column-major (k x n): lane t holds k=(t%4)*2+{0,1}, n=t/4.
                // Here k indexes head-dim and n indexes key, so B[k][n] = K[key][dim].
                FragB fb;
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    fb[j] = reinterpret_cast<cutlass::half_t&>(
                        Ks[(nt * MMA_N + lr) * D + ks * MMA_K + lc + j]);
                }
                mma_op(s[nt], fq[ks], fb, s[nt]);
            }
        }

        // scale + bias + mask; then row max over this BK tile (2 rows per lane)
        float m_tile[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int nt = 0; nt < NK; ++nt) {
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int col = k0 + nt * MMA_N + lc + j;
#pragma unroll
                for (int half_i = 0; half_i < 2; ++half_i) {
                    const int r_loc = warp * MMA_M + lr + half_i * 8;
                    const int g_q = q0 + r_loc;
                    float val;
                    if (g_q >= Sq || col >= Sk || kmask[(long long)n * Sk + col] == 0) {
                        val = NEG_F16 * LOG2E;
                    } else {
                        val = s[nt][half_i * 2 + j] * qk_scale +
                              __half2float(Bs[r_loc * BK + nt * MMA_N + lc + j]) * LOG2E;
                    }
                    s[nt][half_i * 2 + j] = val;
                    m_tile[half_i] = fmaxf(m_tile[half_i], val);
                }
            }
        }
// the 8 columns of a row live in lanes {4r..4r+3}
#pragma unroll
        for (int half_i = 0; half_i < 2; ++half_i) {
            m_tile[half_i] = fmaxf(m_tile[half_i], __shfl_xor_sync(0xffffffffu, m_tile[half_i], 1));
            m_tile[half_i] = fmaxf(m_tile[half_i], __shfl_xor_sync(0xffffffffu, m_tile[half_i], 2));
        }

        float m_new[2], alpha[2], l_tile[2] = {0.f, 0.f};
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            m_new[i] = fmaxf(m_run[i], m_tile[i]);
            alpha[i] = exp2f(m_run[i] - m_new[i]); // 0 on the first tile
        }

        // P = exp2(S - m_new), converted in place to the A operand of P@V
        FragA p[NK];
#pragma unroll
        for (int nt = 0; nt < NK; ++nt) {
#pragma unroll
            for (int half_i = 0; half_i < 2; ++half_i) {
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    float e = exp2f(s[nt][half_i * 2 + j] - m_new[half_i]);
                    l_tile[half_i] += e;
                    p[nt][half_i * 2 + j] = cutlass::half_t(e);
                }
            }
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            l_tile[i] += __shfl_xor_sync(0xffffffffu, l_tile[i], 1);
            l_tile[i] += __shfl_xor_sync(0xffffffffu, l_tile[i], 2);
            l_run[i] = l_run[i] * alpha[i] + l_tile[i];
            m_run[i] = m_new[i];
        }

// O = O*alpha + P @ V
#pragma unroll
        for (int d = 0; d < ND; ++d) {
#pragma unroll
            for (int half_i = 0; half_i < 2; ++half_i)
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    acc[d][half_i * 2 + j] *= alpha[half_i];
                }
#pragma unroll
            for (int nt = 0; nt < NK; ++nt) {
                // B[k][n] = V[key][dim]; lane holds k=(t%4)*2+{0,1}, n=t/4
                FragB fv;
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    fv[j] = reinterpret_cast<cutlass::half_t&>(
                        Vs[(nt * MMA_N + lc + j) * D + d * MMA_N + lr]);
                }
                mma_op(acc[d], p[nt], fv, acc[d]);
            }
        }
    }

    // softmax statistic, for the backward pass. Every lane of a row group
    // holds the same reduced l_run/m_run, so one lane per group writes.
    if (lse != nullptr && (lane & 3) == 0) {
#pragma unroll
        for (int half_i = 0; half_i < 2; ++half_i) {
            const int gq = (half_i == 0) ? row_a : row_b;
            if (gq < Sq) {
                const float l = l_run[half_i];
                lse[(long long)(n * H + h) * Sq + gq] =
                    (l > 0.f) ? (m_run[half_i] + log2f(l)) : -INFINITY;
            }
        }
    }

// normalise and store
#pragma unroll
    for (int d = 0; d < ND; ++d) {
#pragma unroll
        for (int half_i = 0; half_i < 2; ++half_i) {
            const int gq = (half_i == 0) ? row_a : row_b;
            if (gq >= Sq) {
                continue;
            }
            const float inv = 1.0f / fmaxf(l_run[half_i], 1e-30f);
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int col = d * MMA_N + lc + j;
                out[qkv + (long long)gq * D + col] = __float2half(acc[d][half_i * 2 + j] * inv);
            }
        }
    }
}

//
template <int D, int BQ, int BK>
static ffi::Error launch(cudaStream_t stream, int device, const __half* q, const __half* k,
                         const __half* v, const __half* bias, const uint8_t* kmask, __half* out,
                         float* lse, int N, int H, int Sq, int Sk, float scale) {
    constexpr int NWARP = BQ / MMA_M;
    const size_t smem = (size_t)(BQ * D + 2 * BK * D + BQ * BK) * sizeof(__half);
    auto kern = volta_mma_kernel<D, BQ, BK>;

    // Read the device limit one time only.
    // The handler must do no host work, or XLA cannot put it in a CUDA graph.
    const int max_smem = device_shared_limit(device);
    if ((int)smem > max_smem) {
        return ffi::Error::InvalidArgument("volta_mma: needs " + std::to_string(smem / 1024) +
                                           " KB shared, device allows " +
                                           std::to_string(max_smem / 1024) + " KB");
    }
    if (smem > 48 * 1024 && needs_smem_optin(device, (const void*)kern)) {
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    }
    dim3 grid((Sq + BQ - 1) / BQ, H, N);
    kern<<<grid, NWARP * WARP, smem, stream>>>(q, k, v, bias, kmask, out, lse, N, H, Sq, Sk,
                                               scale);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error::Internal(std::string("volta_mma launch: ") + cudaGetErrorString(err));
    }
    return ffi::Error::Success();
}

static ffi::Error volta_mma_common(cudaStream_t stream, int32_t device,
                                   ffi::Buffer<ffi::DataType::F16> q,
                                   ffi::Buffer<ffi::DataType::F16> k,
                                   ffi::Buffer<ffi::DataType::F16> v,
                                   ffi::Buffer<ffi::DataType::F16> bias,
                                   ffi::Buffer<ffi::DataType::U8> kmask,
                                   ffi::Result<ffi::Buffer<ffi::DataType::F16>> out, float* lse,
                                   float scale, int64_t block_q, int64_t block_k) {
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
    __half* op = reinterpret_cast<__half*>(out->typed_data());

#define DISPATCH(DD, BQ, BK)                                                                       \
    if (D == (DD) && block_q == (BQ) && block_k == (BK))                                           \
        return launch<DD, BQ, BK>(stream, device, qp, kp, vp, bp, mp, op, lse, N, H, Sq, Sk,      \
                                  scale);
    DISPATCH(8, 64, 64)
    DISPATCH(8, 64, 32)
    DISPATCH(8, 32, 32)
    DISPATCH(8, 128, 64) DISPATCH(8, 32, 64) DISPATCH(32, 64, 64) DISPATCH(32, 64, 32)
        DISPATCH(32, 32, 64) DISPATCH(32, 32, 32) DISPATCH(32, 128, 64) DISPATCH(32, 16, 64)
            DISPATCH(16, 64, 64) DISPATCH(16, 32, 32) DISPATCH(16, 64, 32) DISPATCH(64, 64, 64)
                DISPATCH(64, 32, 32)
#undef DISPATCH
                    return ffi::Error::InvalidArgument("volta_mma: unsupported (D, bq, bk)");
}

ffi::Error VoltaMmaImpl(cudaStream_t stream, int32_t device, ffi::Buffer<ffi::DataType::F16> q,
                        ffi::Buffer<ffi::DataType::F16> k, ffi::Buffer<ffi::DataType::F16> v,
                        ffi::Buffer<ffi::DataType::F16> bias, ffi::Buffer<ffi::DataType::U8> kmask,
                        ffi::Result<ffi::Buffer<ffi::DataType::F16>> out, float scale,
                        int64_t block_q, int64_t block_k) {
    return volta_mma_common(stream, device, q, k, v, bias, kmask, out, nullptr, scale, block_q,
                            block_k);
}

// Same kernel, one more result: the softmax statistic the backward needs.
// A separate symbol rather than a second Ret on VoltaMma, so a wheel with this
// in it still drives every caller written against the original ABI.
ffi::Error VoltaMmaFwdImpl(cudaStream_t stream, int32_t device, ffi::Buffer<ffi::DataType::F16> q,
                           ffi::Buffer<ffi::DataType::F16> k, ffi::Buffer<ffi::DataType::F16> v,
                           ffi::Buffer<ffi::DataType::F16> bias,
                           ffi::Buffer<ffi::DataType::U8> kmask,
                           ffi::Result<ffi::Buffer<ffi::DataType::F16>> out,
                           ffi::Result<ffi::Buffer<ffi::DataType::F32>> lse, float scale,
                           int64_t block_q, int64_t block_k) {
    return volta_mma_common(stream, device, q, k, v, bias, kmask, out, lse->typed_data(), scale,
                            block_q, block_k);
}

// kCmdBufferCompatible lets XLA put this call in a CUDA graph.
// This is correct because the handler only starts one kernel on the XLA stream.
XLA_FFI_DEFINE_HANDLER_SYMBOL(VoltaMma, VoltaMmaImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Ctx<ffi::DeviceOrdinal>()
                                  .Arg<ffi::Buffer<ffi::DataType::F16>>()
                                  .Arg<ffi::Buffer<ffi::DataType::F16>>()
                                  .Arg<ffi::Buffer<ffi::DataType::F16>>()
                                  .Arg<ffi::Buffer<ffi::DataType::F16>>()
                                  .Arg<ffi::Buffer<ffi::DataType::U8>>()
                                  .Ret<ffi::Buffer<ffi::DataType::F16>>()
                                  .Attr<float>("scale")
                                  .Attr<int64_t>("block_q")
                                  .Attr<int64_t>("block_k"),
                              {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(VoltaMmaFwd, VoltaMmaFwdImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Ctx<ffi::DeviceOrdinal>()
                                  .Arg<ffi::Buffer<ffi::DataType::F16>>()
                                  .Arg<ffi::Buffer<ffi::DataType::F16>>()
                                  .Arg<ffi::Buffer<ffi::DataType::F16>>()
                                  .Arg<ffi::Buffer<ffi::DataType::F16>>()
                                  .Arg<ffi::Buffer<ffi::DataType::U8>>()
                                  .Ret<ffi::Buffer<ffi::DataType::F16>>()
                                  .Ret<ffi::Buffer<ffi::DataType::F32>>()
                                  .Attr<float>("scale")
                                  .Attr<int64_t>("block_q")
                                  .Attr<int64_t>("block_k"),
                              {ffi::Traits::kCmdBufferCompatible});
