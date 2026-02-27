#include "dequantize_fp8_nvidia.cuh"
#include "../../nvidia_common.cuh"
#include <cuda_fp8.h>

__global__ void dequantize_fp8_kernel(
    __nv_bfloat16 *out, const __nv_fp8_e4m3 *in, const float *scale_inv,
    size_t M, size_t K, size_t block_h, size_t block_w, size_t scale_cols) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * K) return;

    size_t row = idx / K;
    size_t col = idx % K;
    size_t scale_row = row / block_h;
    size_t scale_col = col / block_w;

    float fp8_val = float(in[idx]);
    float scale = scale_inv[scale_row * scale_cols + scale_col];
    out[idx] = __float2bfloat16(fp8_val * scale);
}

namespace llaisys::ops::nvidia {
void dequantize_fp8(std::byte *out_bf16, const std::byte *in_fp8,
                    const std::byte *scale_inv,
                    size_t M, size_t K, size_t block_h, size_t block_w) {
    size_t total = M * K;
    size_t scale_cols = (K + block_w - 1) / block_w;
    int threads = 256;
    int blocks = ceil_div((int)total, threads);

    dequantize_fp8_kernel<<<blocks, threads>>>(
        (__nv_bfloat16 *)out_bf16,
        (const __nv_fp8_e4m3 *)in_fp8,
        (const float *)scale_inv,
        M, K, block_h, block_w, scale_cols);
    CUDA_KERNEL_CHECK();
}
} // namespace llaisys::ops::nvidia
