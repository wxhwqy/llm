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
// Full-matrix FP8 dequant + single cuBLAS GEMM
// Dequantizes the entire weight [N,K] to BF16 in one kernel, then one GEMM.
// Trades ~N*K*2 bytes of temp buffer for eliminating N/block_h kernel launches.
// ---------------------------------------------------------------------------

__global__ void dequant_full_kernel(
    __nv_bfloat16 *out, const __nv_fp8_e4m3 *in, const float *scale_inv,
    size_t N, size_t K, size_t block_h, size_t block_w, size_t scale_cols) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * K) return;
    size_t row = idx / K;
    size_t col = idx % K;
    float fp8_val = float(in[idx]);
    float scale = scale_inv[(row / block_h) * scale_cols + (col / block_w)];
    out[idx] = __float2bfloat16(fp8_val * scale);
}

void linear_fp8(std::byte *out, const std::byte *in,
                const std::byte *weight_fp8, const std::byte *scale_inv,
                llaisysDataType_t compute_dtype,
                size_t M, size_t K, size_t N,
                size_t block_h, size_t block_w) {
    auto &res = llaisys::device::nvidia::Resource::get();
    auto handle = res.cublasHandle();

    size_t scale_cols = (K + block_w - 1) / block_w;
    size_t total_elems = N * K;

    // Dequant buffer: N × K × 2 bytes (BF16). Lazily grows via getTileBuffer.
    void *dequant_buf = res.getTileBuffer(total_elems * sizeof(__nv_bfloat16));

    int threads = 256;
    int dq_blocks = ceil_div((int)total_elems, threads);
    dequant_full_kernel<<<dq_blocks, threads>>>(
        (__nv_bfloat16 *)dequant_buf,
        (const __nv_fp8_e4m3 *)weight_fp8,
        (const float *)scale_inv,
        N, K, block_h, block_w, scale_cols);
    CUDA_KERNEL_CHECK();

    // Single GEMM: out[M,N] = in[M,K] @ dequant_buf[N,K]^T
    // cuBLAS col-major: C(N,M) = A(N,K) * B(K,M), A=weight, B=in
    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                               (int)N, (int)M, (int)K,
                               &alpha,
                               dequant_buf, CUDA_R_16BF, (int)K,
                               in, CUDA_R_16BF, (int)K,
                               &beta,
                               out, CUDA_R_16BF, (int)N,
                               CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

} // namespace llaisys::ops::nvidia
