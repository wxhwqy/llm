#include "gated_rms_norm_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

template <typename T>
static void gated_rms_norm_scalar(T *out, const T *x, const T *z, const T *weight,
                                    float eps, size_t M, size_t N) {
    for (size_t row = 0; row < M; row++) {
        const T *x_row = x + row * N;
        const T *z_row = z + row * N;
        T *out_row = out + row * N;

        float sum_sq = 0.0f;
        for (size_t i = 0; i < N; i++) {
            float v;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                v = llaisys::utils::cast<float>(x_row[i]);
            } else {
                v = static_cast<float>(x_row[i]);
            }
            sum_sq += v * v;
        }
        float rms = 1.0f / std::sqrt(sum_sq / (float)N + eps);

        for (size_t i = 0; i < N; i++) {
            float x_val, z_val, w_val;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                x_val = llaisys::utils::cast<float>(x_row[i]);
                z_val = llaisys::utils::cast<float>(z_row[i]);
                w_val = llaisys::utils::cast<float>(weight[i]);
            } else {
                x_val = static_cast<float>(x_row[i]);
                z_val = static_cast<float>(z_row[i]);
                w_val = static_cast<float>(weight[i]);
            }
            float normed = x_val * rms * w_val;
            float silu_z = z_val / (1.0f + std::exp(-z_val));
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                out_row[i] = llaisys::utils::cast<T>(normed * silu_z);
            } else {
                out_row[i] = static_cast<T>(normed * silu_z);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void gated_rms_norm(std::byte *out_ptr, const std::byte *x_ptr, const std::byte *z_ptr,
                     const std::byte *weight_ptr, float eps, llaisysDataType_t dtype,
                     size_t M, size_t N) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return gated_rms_norm_scalar(reinterpret_cast<float *>(out_ptr),
                                      reinterpret_cast<const float *>(x_ptr),
                                      reinterpret_cast<const float *>(z_ptr),
                                      reinterpret_cast<const float *>(weight_ptr), eps, M, N);
    case LLAISYS_DTYPE_BF16:
        return gated_rms_norm_scalar(reinterpret_cast<llaisys::bf16_t *>(out_ptr),
                                      reinterpret_cast<const llaisys::bf16_t *>(x_ptr),
                                      reinterpret_cast<const llaisys::bf16_t *>(z_ptr),
                                      reinterpret_cast<const llaisys::bf16_t *>(weight_ptr), eps, M, N);
    case LLAISYS_DTYPE_F16:
        return gated_rms_norm_scalar(reinterpret_cast<llaisys::fp16_t *>(out_ptr),
                                      reinterpret_cast<const llaisys::fp16_t *>(x_ptr),
                                      reinterpret_cast<const llaisys::fp16_t *>(z_ptr),
                                      reinterpret_cast<const llaisys::fp16_t *>(weight_ptr), eps, M, N);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
