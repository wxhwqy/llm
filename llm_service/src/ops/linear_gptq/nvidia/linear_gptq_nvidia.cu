#include "linear_gptq_nvidia.cuh"
#include "../../nvidia_common.cuh"
#include "../../../device/nvidia/nvidia_resource.cuh"
#include <cublas_v2.h>

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t status = (call);                                        \
        if (status != CUBLAS_STATUS_SUCCESS) {                                 \
            throw std::runtime_error(                                          \
                std::string("cuBLAS error: ") + std::to_string(status));       \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// GPTQ INT4 Fused Dequant + GEMV kernel (M=1, decode path).
//
// Each block computes one output element: out[n] = dot(dequant(W[n,:]), x[:])
//
// qweight layout: [in_features/pack, out_features] INT32 (row-major)
//   Each INT32 packs `pack` (=8 for 4-bit) consecutive input-dimension values
//   for a single output column.
//
// scales layout: [num_groups, out_features] BF16
//   Each group covers `group_size` input elements.
//
// qzeros layout: [num_groups, out_features/pack] INT32
//   Each INT32 packs `pack` zero-points for consecutive output columns.
//   If nullptr, symmetric quantization with zero=8 (midpoint for 4-bit).
// ---------------------------------------------------------------------------

constexpr int GPTQ_GEMV_BLOCK = 256;

__global__ void fused_gptq_gemv_kernel(
    __nv_bfloat16 *__restrict__ out,
    const int32_t *__restrict__ qweight,     // [qw_rows, out_features]
    const __nv_bfloat16 *__restrict__ scales, // [num_groups, out_features]
    const int32_t *__restrict__ qzeros,       // [num_groups, out_features/pack] or nullptr
    const __nv_bfloat16 *__restrict__ in_vec, // [in_features]
    size_t qw_rows,       // in_features / pack
    size_t out_features,
    int bits,
    int pack,             // 32 / bits
    int mask,             // (1 << bits) - 1
    int group_size)
{
    size_t n = blockIdx.x;  // output column index
    if (n >= out_features) return;

    // Pre-compute qzeros column info for this output
    size_t zp_col = n / pack;
    int zp_shift = (n % pack) * bits;

    float sum = 0.0f;

    // Each thread handles a subset of qw_rows
    for (size_t ig = threadIdx.x; ig < qw_rows; ig += GPTQ_GEMV_BLOCK) {
        int32_t packed = qweight[ig * out_features + n];
        size_t base_row = ig * pack;
        size_t group_idx = base_row / group_size;

        float scale = __bfloat162float(scales[group_idx * out_features + n]);

        int zero = 8;  // symmetric default
        if (qzeros) {
            int32_t zp_packed = qzeros[group_idx * (out_features / pack) + zp_col];
            zero = (zp_packed >> zp_shift) & mask;
        }

        // Unpack pack (8) int4 values and compute dot product
        #pragma unroll
        for (int k = 0; k < 8; k++) {  // Unrolled for pack=8 (4-bit)
            if (k < pack) {
                int int4_val = (packed >> (k * bits)) & mask;
                float w_val = (float)(int4_val - zero) * scale;
                sum += w_val * __bfloat162float(in_vec[base_row + k]);
            }
        }
    }

    // Warp-level reduction
    for (int off = warpSize / 2; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, off);

    // Block-level reduction via shared memory
    __shared__ float warp_buf[GPTQ_GEMV_BLOCK / 32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    if (lane == 0) warp_buf[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        constexpr int nwarps = GPTQ_GEMV_BLOCK / 32;
        sum = (lane < nwarps) ? warp_buf[lane] : 0.0f;
        for (int off = warpSize / 2; off > 0; off >>= 1)
            sum += __shfl_down_sync(0xFFFFFFFF, sum, off);
        if (lane == 0)
            out[n] = __float2bfloat16(sum);
    }
}

// ---------------------------------------------------------------------------
// Full dequant kernel: INT4 -> BF16 weight matrix (for M>1 prefill path).
//
// Writes to a [in_features, out_features] BF16 buffer (row-major, transposed
// vs qweight layout) so it can be used with cuBLAS as weight^T.
//
// Actually, to match the existing linear convention (weight is [N, K] and
// cuBLAS computes out = in @ W^T), we produce the dequantized weight in
// [out_features, in_features] layout (same as the logical weight matrix).
// ---------------------------------------------------------------------------

__global__ void dequant_gptq_full_kernel(
    __nv_bfloat16 *__restrict__ dequant_out,   // [out_features, in_features]
    const int32_t *__restrict__ qweight,         // [qw_rows, out_features]
    const __nv_bfloat16 *__restrict__ scales,    // [num_groups, out_features]
    const int32_t *__restrict__ qzeros,          // [num_groups, out_features/pack] or nullptr
    size_t qw_rows,
    size_t in_features,
    size_t out_features,
    int bits,
    int pack,
    int mask,
    int group_size)
{
    // Each thread dequantizes one packed INT32 → pack BF16 values.
    // Total work items: qw_rows * out_features
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= qw_rows * out_features) return;

    size_t ig = idx / out_features;   // packed row index
    size_t n = idx % out_features;    // output column

    int32_t packed = qweight[ig * out_features + n];
    size_t base_row = ig * pack;
    size_t group_idx = base_row / group_size;

    float scale = __bfloat162float(scales[group_idx * out_features + n]);

    int zero = 8;
    if (qzeros) {
        size_t zp_col = n / pack;
        int32_t zp_packed = qzeros[group_idx * (out_features / pack) + zp_col];
        zero = (zp_packed >> ((n % pack) * bits)) & mask;
    }

    // Write pack values: dequant_out[n, base_row+k] for k in [0, pack)
    // Layout: [out_features, in_features] row-major
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        if (k < pack) {
            int int4_val = (packed >> (k * bits)) & mask;
            float w_val = (float)(int4_val - zero) * scale;
            dequant_out[n * in_features + base_row + k] = __float2bfloat16(w_val);
        }
    }
}

namespace llaisys::ops::nvidia {

void linear_gptq(std::byte *output, const std::byte *input,
                 const std::byte *qweight, const std::byte *scales,
                 const std::byte *qzeros,
                 size_t M, size_t in_features, size_t out_features,
                 int bits, int group_size) {

    int pack = 32 / bits;
    int mask = (1 << bits) - 1;
    size_t qw_rows = in_features / pack;

    // M=1 fast path: fused dequant + GEMV, no temp buffer
    if (M == 1) {
        fused_gptq_gemv_kernel<<<(int)out_features, GPTQ_GEMV_BLOCK>>>(
            (__nv_bfloat16 *)output,
            (const int32_t *)qweight,
            (const __nv_bfloat16 *)scales,
            qzeros ? (const int32_t *)qzeros : nullptr,
            (const __nv_bfloat16 *)input,
            qw_rows, out_features, bits, pack, mask, group_size);
        CUDA_KERNEL_CHECK();
        return;
    }

    // M>1: full dequant to BF16 + cuBLAS GEMM
    auto &res = llaisys::device::nvidia::Resource::get();
    auto handle = res.cublasHandle();

    // Allocate temp buffer for dequantized weight [out_features, in_features] BF16
    size_t dequant_elems = out_features * in_features;
    void *dequant_buf = res.getTileBuffer(dequant_elems * sizeof(__nv_bfloat16));

    // Dequantize: each thread handles one packed INT32 → 8 BF16 values
    size_t total_packed = qw_rows * out_features;
    int threads = 256;
    int blocks = ceil_div((int)total_packed, threads);

    dequant_gptq_full_kernel<<<blocks, threads>>>(
        (__nv_bfloat16 *)dequant_buf,
        (const int32_t *)qweight,
        (const __nv_bfloat16 *)scales,
        qzeros ? (const int32_t *)qzeros : nullptr,
        qw_rows, in_features, out_features,
        bits, pack, mask, group_size);
    CUDA_KERNEL_CHECK();

    // cuBLAS GEMM: out[M,N] = input[M,K] @ dequant_weight[N,K]^T
    // Column-major: C(N,M) = A(N,K) * B(K,M)
    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                               (int)out_features, (int)M, (int)in_features,
                               &alpha,
                               dequant_buf, CUDA_R_16BF, (int)in_features,
                               input, CUDA_R_16BF, (int)in_features,
                               &beta,
                               output, CUDA_R_16BF, (int)out_features,
                               CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

} // namespace llaisys::ops::nvidia
