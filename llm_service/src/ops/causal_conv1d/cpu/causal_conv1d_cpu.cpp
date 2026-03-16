#include "causal_conv1d_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

template <typename T>
static void causal_conv1d_scalar(T *out, const T *in, const T *weight, const T *bias,
                                  size_t seq_len, size_t d_inner, size_t kernel_size) {
    for (size_t t = 0; t < seq_len; t++) {
        for (size_t c = 0; c < d_inner; c++) {
            float acc = 0.0f;
            for (size_t k = 0; k < kernel_size; k++) {
                int t_in = (int)t - (int)kernel_size + 1 + (int)k;
                if (t_in >= 0) {
                    float w, x;
                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        w = llaisys::utils::cast<float>(weight[c * kernel_size + k]);
                        x = llaisys::utils::cast<float>(in[t_in * d_inner + c]);
                    } else {
                        w = static_cast<float>(weight[c * kernel_size + k]);
                        x = static_cast<float>(in[t_in * d_inner + c]);
                    }
                    acc += w * x;
                }
            }
            if (bias) {
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    acc += llaisys::utils::cast<float>(bias[c]);
                } else {
                    acc += static_cast<float>(bias[c]);
                }
            }
            float silu = acc / (1.0f + std::exp(-acc));
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                out[t * d_inner + c] = llaisys::utils::cast<T>(silu);
            } else {
                out[t * d_inner + c] = static_cast<T>(silu);
            }
        }
    }
}

template <typename T>
static void causal_conv1d_step_scalar(T *out_col, T *conv_state, const T *in_col,
                                       const T *weight, const T *bias,
                                       size_t d_inner, size_t kernel_size) {
    for (size_t c = 0; c < d_inner; c++) {
        for (size_t k = 0; k < kernel_size - 1; k++) {
            conv_state[c * kernel_size + k] = conv_state[c * kernel_size + k + 1];
        }
        conv_state[c * kernel_size + kernel_size - 1] = in_col[c];

        float acc = 0.0f;
        for (size_t k = 0; k < kernel_size; k++) {
            float w, s;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                w = llaisys::utils::cast<float>(weight[c * kernel_size + k]);
                s = llaisys::utils::cast<float>(conv_state[c * kernel_size + k]);
            } else {
                w = static_cast<float>(weight[c * kernel_size + k]);
                s = static_cast<float>(conv_state[c * kernel_size + k]);
            }
            acc += w * s;
        }
        if (bias) {
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                acc += llaisys::utils::cast<float>(bias[c]);
            } else {
                acc += static_cast<float>(bias[c]);
            }
        }
        float silu = acc / (1.0f + std::exp(-acc));
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            out_col[c] = llaisys::utils::cast<T>(silu);
        } else {
            out_col[c] = static_cast<T>(silu);
        }
    }
}

namespace llaisys::ops::cpu {
void causal_conv1d(std::byte *out_ptr, const std::byte *in_ptr, const std::byte *weight_ptr,
                    const std::byte *bias_ptr, llaisysDataType_t dtype,
                    size_t seq_len, size_t d_inner, size_t kernel_size) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return causal_conv1d_scalar((float *)out_ptr, (const float *)in_ptr,
                                    (const float *)weight_ptr, bias_ptr ? (const float *)bias_ptr : nullptr,
                                    seq_len, d_inner, kernel_size);
    case LLAISYS_DTYPE_BF16:
        return causal_conv1d_scalar((llaisys::bf16_t *)out_ptr, (const llaisys::bf16_t *)in_ptr,
                                    (const llaisys::bf16_t *)weight_ptr,
                                    bias_ptr ? (const llaisys::bf16_t *)bias_ptr : nullptr,
                                    seq_len, d_inner, kernel_size);
    case LLAISYS_DTYPE_F16:
        return causal_conv1d_scalar((llaisys::fp16_t *)out_ptr, (const llaisys::fp16_t *)in_ptr,
                                    (const llaisys::fp16_t *)weight_ptr,
                                    bias_ptr ? (const llaisys::fp16_t *)bias_ptr : nullptr,
                                    seq_len, d_inner, kernel_size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void causal_conv1d_step(std::byte *out_col_ptr, std::byte *conv_state_ptr, const std::byte *in_col_ptr,
                         const std::byte *weight_ptr, const std::byte *bias_ptr,
                         llaisysDataType_t dtype, size_t d_inner, size_t kernel_size) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return causal_conv1d_step_scalar((float *)out_col_ptr, (float *)conv_state_ptr,
                                          (const float *)in_col_ptr, (const float *)weight_ptr,
                                          bias_ptr ? (const float *)bias_ptr : nullptr, d_inner, kernel_size);
    case LLAISYS_DTYPE_BF16:
        return causal_conv1d_step_scalar((llaisys::bf16_t *)out_col_ptr, (llaisys::bf16_t *)conv_state_ptr,
                                          (const llaisys::bf16_t *)in_col_ptr, (const llaisys::bf16_t *)weight_ptr,
                                          bias_ptr ? (const llaisys::bf16_t *)bias_ptr : nullptr, d_inner, kernel_size);
    case LLAISYS_DTYPE_F16:
        return causal_conv1d_step_scalar((llaisys::fp16_t *)out_col_ptr, (llaisys::fp16_t *)conv_state_ptr,
                                          (const llaisys::fp16_t *)in_col_ptr, (const llaisys::fp16_t *)weight_ptr,
                                          bias_ptr ? (const llaisys::fp16_t *)bias_ptr : nullptr, d_inner, kernel_size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
