#include "../../cpu_simd_utils.hpp"

#include "argmax_cpu.hpp"
#include "../../../utils.hpp"

#include <cmath>
#include <limits>

#ifdef __AVX2__
static void argmax_f32_avx2(int64_t *max_idx, float *max_val, const float *vals, size_t numel) {
    using namespace llaisys::ops::cpu;

    float best_val = -std::numeric_limits<float>::infinity();
    int64_t best_idx = 0;

    size_t i = 0;
    if (numel >= 8) {
        __m256 vmax = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
        // First pass: find the global max value using SIMD
        for (; i + 8 <= numel; i += 8) {
            __m256 v = _mm256_loadu_ps(vals + i);
            vmax = _mm256_max_ps(vmax, v);
        }
        best_val = hmax_f32x8(vmax);
        // Handle remainder
        for (; i < numel; i++) {
            if (vals[i] > best_val) {
                best_val = vals[i];
            }
        }
        // Second pass: find the index of the max value (first occurrence)
        for (size_t j = 0; j < numel; j++) {
            if (vals[j] == best_val) {
                best_idx = static_cast<int64_t>(j);
                break;
            }
        }
    } else {
        for (; i < numel; i++) {
            if (vals[i] > best_val) {
                best_val = vals[i];
                best_idx = static_cast<int64_t>(i);
            }
        }
    }

    max_idx[0] = best_idx;
    max_val[0] = best_val;
}
#endif // __AVX2__

template <typename T>
static void argmax_scalar(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    int64_t best_idx = 0;
    float best_val = -std::numeric_limits<float>::infinity();

    for (size_t i = 0; i < numel; i++) {
        float val;
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            val = llaisys::utils::cast<float>(vals[i]);
        } else {
            val = static_cast<float>(vals[i]);
        }
        if (val > best_val) {
            best_val = val;
            best_idx = static_cast<int64_t>(i);
        }
    }

    max_idx[0] = best_idx;
    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        max_val[0] = llaisys::utils::cast<T>(best_val);
    } else {
        max_val[0] = static_cast<T>(best_val);
    }
}

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx_ptr, std::byte *max_val_ptr, const std::byte *vals_ptr,
            llaisysDataType_t dtype, size_t numel) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
#ifdef __AVX2__
        return argmax_f32_avx2(reinterpret_cast<int64_t *>(max_idx_ptr),
                               reinterpret_cast<float *>(max_val_ptr),
                               reinterpret_cast<const float *>(vals_ptr), numel);
#else
        return argmax_scalar(reinterpret_cast<int64_t *>(max_idx_ptr),
                             reinterpret_cast<float *>(max_val_ptr),
                             reinterpret_cast<const float *>(vals_ptr), numel);
#endif
    case LLAISYS_DTYPE_BF16:
        return argmax_scalar(reinterpret_cast<int64_t *>(max_idx_ptr),
                             reinterpret_cast<bf16_t *>(max_val_ptr),
                             reinterpret_cast<const bf16_t *>(vals_ptr), numel);
    case LLAISYS_DTYPE_F16:
        return argmax_scalar(reinterpret_cast<int64_t *>(max_idx_ptr),
                             reinterpret_cast<fp16_t *>(max_val_ptr),
                             reinterpret_cast<const fp16_t *>(vals_ptr), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
