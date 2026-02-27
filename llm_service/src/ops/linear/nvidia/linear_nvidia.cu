#include "linear_nvidia.cuh"
#include "../../nvidia_common.cuh"
#include "../../../device/nvidia/nvidia_resource.cuh"
#include <cuda_fp8.h>
#include <cublas_v2.h>

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t status = (call);                                        \
        if (status != CUBLAS_STATUS_SUCCESS) {                                 \
            throw std::runtime_error(                                          \
                std::string("cuBLAS error: ") + std::to_string(status));       \
        }                                                                      \
    } while (0)

template <typename T>
__global__ void add_bias_kernel(T *out, const T *bias, size_t M, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = idx / N;
    size_t col = idx % N;
    if (row < M && col < N) {
        out[idx] = from_float<T>(to_float(out[idx]) + to_float(bias[col]));
    }
}

namespace llaisys::ops::nvidia {

void linear(std::byte *out, const std::byte *in, const std::byte *weight,
            const std::byte *bias, llaisysDataType_t dtype,
            size_t M, size_t K, size_t N) {
    auto handle = llaisys::device::nvidia::Resource::get().cublasHandle();

    // out = in @ weight^T -> cuBLAS column-major: C^T = W * in^T
    // cuBLAS: C(N,M) = alpha * A(N,K) * B(K,M) + beta * C(N,M)
    // where A=weight, B=in (in column-major view)

    if (dtype == LLAISYS_DTYPE_F32) {
        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                  (int)N, (int)M, (int)K,
                                  &alpha,
                                  (const float *)weight, (int)K,
                                  (const float *)in, (int)K,
                                  &beta,
                                  (float *)out, (int)N));
    } else if (dtype == LLAISYS_DTYPE_F16) {
        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                   (int)N, (int)M, (int)K,
                                   &alpha,
                                   weight, CUDA_R_16F, (int)K,
                                   in, CUDA_R_16F, (int)K,
                                   &beta,
                                   out, CUDA_R_16F, (int)N,
                                   CUBLAS_COMPUTE_32F,
                                   CUBLAS_GEMM_DEFAULT));
    } else if (dtype == LLAISYS_DTYPE_BF16) {
        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                   (int)N, (int)M, (int)K,
                                   &alpha,
                                   weight, CUDA_R_16BF, (int)K,
                                   in, CUDA_R_16BF, (int)K,
                                   &beta,
                                   out, CUDA_R_16BF, (int)N,
                                   CUBLAS_COMPUTE_32F,
                                   CUBLAS_GEMM_DEFAULT));
    } else {
        throw std::runtime_error("Unsupported dtype for CUDA linear");
    }

    if (bias) {
        int threads = 256;
        int blocks = ceil_div((int)(M * N), threads);
        switch (dtype) {
        case LLAISYS_DTYPE_F32:
            add_bias_kernel<<<blocks, threads>>>((float *)out, (const float *)bias, M, N);
            break;
        case LLAISYS_DTYPE_F16:
            add_bias_kernel<<<blocks, threads>>>((__half *)out, (const __half *)bias, M, N);
            break;
        case LLAISYS_DTYPE_BF16:
            add_bias_kernel<<<blocks, threads>>>((__nv_bfloat16 *)out, (const __nv_bfloat16 *)bias, M, N);
            break;
        default:
            break;
        }
        CUDA_KERNEL_CHECK();
    }
}

// ---------------------------------------------------------------------------
// Tiled FP8 dequant + GEMM
// Dequantizes block_h rows of weight at a time → tiny tile buffer → cuBLAS
// ---------------------------------------------------------------------------

// Kernel: dequantize tile_rows × K FP8 weights to BF16 using block-wise scale.
// scale_inv already points to the first scale block row for this tile.
// Since tile_rows == block_h, all rows in the tile share the same scale row,
// so the scale index is only column-dependent: c / block_w.
__global__ void dequant_tile_kernel(
    __nv_bfloat16 *out_tile, const __nv_fp8_e4m3 *in_tile,
    const float *scale_inv,
    size_t tile_rows, size_t K, size_t block_w) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tile_rows * K) return;
    size_t c = idx % K;
    float fp8_val = float(in_tile[idx]);
    float scale = scale_inv[c / block_w];  // all rows in tile use same scale row
    out_tile[idx] = __float2bfloat16(fp8_val * scale);
}

void linear_fp8(std::byte *out, const std::byte *in,
                const std::byte *weight_fp8, const std::byte *scale_inv,
                llaisysDataType_t compute_dtype,
                size_t M, size_t K, size_t N,
                size_t block_h, size_t block_w) {
    auto &res = llaisys::device::nvidia::Resource::get();
    auto handle = res.cublasHandle();

    size_t tile_n = block_h;  // one tile = one row of scale blocks
    size_t scale_cols = (K + block_w - 1) / block_w;

    // Tile buffer: tile_n × K × 2 bytes (BF16)
    // e.g. 128 × 12800 × 2 = ~3.3 MB for Qwen3-32B TP=2 down_proj
    void *tile_buf = res.getTileBuffer(tile_n * K * sizeof(__nv_bfloat16));

    // compute dtype is always BF16 (Qwen3 uses BF16 compute)
    float alpha = 1.0f, beta = 0.0f;
    int threads = 256;

    const __nv_fp8_e4m3 *w_fp8 = (const __nv_fp8_e4m3 *)weight_fp8;
    const float *s_inv = (const float *)scale_inv;

    for (size_t n_start = 0; n_start < N; n_start += tile_n) {
        size_t actual_tile = (n_start + tile_n <= N) ? tile_n : (N - n_start);
        size_t tile_elems = actual_tile * K;

        // Dequantize this tile: w_fp8[n_start:n_start+actual_tile, :]
        int dq_blocks = ceil_div((int)tile_elems, threads);
        size_t scale_row_start = n_start / block_h;
        dequant_tile_kernel<<<dq_blocks, threads>>>(
            (__nv_bfloat16 *)tile_buf,
            w_fp8 + n_start * K,
            s_inv + scale_row_start * scale_cols,
            actual_tile, K, block_w);
        CUDA_KERNEL_CHECK();

        // GEMM: out[M, n_start:n_start+actual_tile] = in[M,K] @ tile^T[K, actual_tile]
        // cuBLAS col-major: C[actual_tile, M] = tile[actual_tile,K] @ in^T[K,M]
        // out pointer offset by n_start (column offset), ldc = N (full row stride)
        __nv_bfloat16 *out_ptr = (__nv_bfloat16 *)out + n_start;

        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                   (int)actual_tile, (int)M, (int)K,
                                   &alpha,
                                   tile_buf, CUDA_R_16BF, (int)K,
                                   in, CUDA_R_16BF, (int)K,
                                   &beta,
                                   out_ptr, CUDA_R_16BF, (int)N,
                                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }
}

} // namespace llaisys::ops::nvidia
