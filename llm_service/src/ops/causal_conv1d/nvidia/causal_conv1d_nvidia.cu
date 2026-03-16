#include "causal_conv1d_nvidia.cuh"
#include "../../nvidia_common.cuh"

// Prefill: causal 1D conv + SiLU
// in layout: [seq_len, d_inner] (row-major, each row is one timestep)
// weight layout: [d_inner, kernel_size] (each channel has its own 1D kernel)
// For position t, channel c: out[t,c] = silu(sum_{k=0}^{K-1} weight[c,k] * in[t-K+1+k, c] + bias[c])
// where in[t', c] = 0 if t' < 0 (causal padding)
template <typename T>
__global__ void causal_conv1d_kernel(T *out, const T *in, const T *weight, const T *bias,
                                      size_t seq_len, size_t d_inner, size_t kernel_size) {
    size_t total = seq_len * d_inner;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    size_t c = idx % d_inner;
    size_t t = idx / d_inner;

    float acc = 0.0f;
    for (size_t k = 0; k < kernel_size; k++) {
        int t_in = (int)t - (int)kernel_size + 1 + (int)k;
        if (t_in >= 0) {
            acc += to_float(weight[c * kernel_size + k]) * to_float(in[t_in * d_inner + c]);
        }
    }
    if (bias) acc += to_float(bias[c]);

    // SiLU activation
    float silu = acc / (1.0f + expf(-acc));
    out[idx] = from_float<T>(silu);
}

// Decode step: shift conv_state left, append new input, compute dot + SiLU
// conv_state: [d_inner, kernel_size], in_col: [d_inner], out_col: [d_inner]
template <typename T>
__global__ void causal_conv1d_step_kernel(T *out_col, T *conv_state, const T *in_col,
                                           const T *weight, const T *bias,
                                           size_t d_inner, size_t kernel_size) {
    size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= d_inner) return;

    // Shift state left by 1
    for (size_t k = 0; k < kernel_size - 1; k++) {
        conv_state[c * kernel_size + k] = conv_state[c * kernel_size + k + 1];
    }
    // Append new input
    conv_state[c * kernel_size + kernel_size - 1] = in_col[c];

    // Compute convolution
    float acc = 0.0f;
    for (size_t k = 0; k < kernel_size; k++) {
        acc += to_float(weight[c * kernel_size + k]) * to_float(conv_state[c * kernel_size + k]);
    }
    if (bias) acc += to_float(bias[c]);

    float silu = acc / (1.0f + expf(-acc));
    out_col[c] = from_float<T>(silu);
}

namespace llaisys::ops::nvidia {
void causal_conv1d(std::byte *out, const std::byte *in, const std::byte *weight,
                    const std::byte *bias, llaisysDataType_t dtype,
                    size_t seq_len, size_t d_inner, size_t kernel_size) {
    int threads = 256;
    int blocks = ceil_div((int)(seq_len * d_inner), threads);

    switch (dtype) {
    case LLAISYS_DTYPE_BF16:
        causal_conv1d_kernel<<<blocks, threads>>>(
            (__nv_bfloat16 *)out, (const __nv_bfloat16 *)in, (const __nv_bfloat16 *)weight,
            bias ? (const __nv_bfloat16 *)bias : nullptr, seq_len, d_inner, kernel_size);
        break;
    case LLAISYS_DTYPE_F16:
        causal_conv1d_kernel<<<blocks, threads>>>(
            (__half *)out, (const __half *)in, (const __half *)weight,
            bias ? (const __half *)bias : nullptr, seq_len, d_inner, kernel_size);
        break;
    case LLAISYS_DTYPE_F32:
        causal_conv1d_kernel<<<blocks, threads>>>(
            (float *)out, (const float *)in, (const float *)weight,
            bias ? (const float *)bias : nullptr, seq_len, d_inner, kernel_size);
        break;
    default:
        throw std::runtime_error("Unsupported dtype for CUDA causal_conv1d");
    }
    CUDA_KERNEL_CHECK();
}

void causal_conv1d_step(std::byte *out_col, std::byte *conv_state, const std::byte *in_col,
                         const std::byte *weight, const std::byte *bias,
                         llaisysDataType_t dtype, size_t d_inner, size_t kernel_size) {
    int threads = 256;
    int blocks = ceil_div((int)d_inner, threads);

    switch (dtype) {
    case LLAISYS_DTYPE_BF16:
        causal_conv1d_step_kernel<<<blocks, threads>>>(
            (__nv_bfloat16 *)out_col, (__nv_bfloat16 *)conv_state, (const __nv_bfloat16 *)in_col,
            (const __nv_bfloat16 *)weight, bias ? (const __nv_bfloat16 *)bias : nullptr,
            d_inner, kernel_size);
        break;
    case LLAISYS_DTYPE_F16:
        causal_conv1d_step_kernel<<<blocks, threads>>>(
            (__half *)out_col, (__half *)conv_state, (const __half *)in_col,
            (const __half *)weight, bias ? (const __half *)bias : nullptr,
            d_inner, kernel_size);
        break;
    case LLAISYS_DTYPE_F32:
        causal_conv1d_step_kernel<<<blocks, threads>>>(
            (float *)out_col, (float *)conv_state, (const float *)in_col,
            (const float *)weight, bias ? (const float *)bias : nullptr,
            d_inner, kernel_size);
        break;
    default:
        throw std::runtime_error("Unsupported dtype for CUDA causal_conv1d_step");
    }
    CUDA_KERNEL_CHECK();
}
} // namespace llaisys::ops::nvidia
