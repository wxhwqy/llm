#include "../../cpu_simd_utils.hpp"

#include "rms_norm_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

#ifdef __AVX2__
static void rms_norm_f32_avx2(float *out, const float *in, const float *weight,
                               float eps, size_t M, size_t N) {
    using namespace llaisys::ops::cpu;

    for (size_t m = 0; m < M; m++) {
        const float *row_in = in + m * N;
        float *row_out = out + m * N;

        // Pass 1: compute sum of squares with FMA
        __m256 vsum_sq = _mm256_setzero_ps();
        size_t n = 0;
        for (; n + 8 <= N; n += 8) {
            __m256 v = _mm256_loadu_ps(row_in + n);
#ifdef __FMA__
            vsum_sq = _mm256_fmadd_ps(v, v, vsum_sq);
#else
            vsum_sq = _mm256_add_ps(vsum_sq, _mm256_mul_ps(v, v));
#endif
        }
        float sum_sq = hsum_f32x8(vsum_sq);
        for (; n < N; n++) {
            sum_sq += row_in[n] * row_in[n];
        }

        float rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(N) + eps);
        __m256 vrms = _mm256_set1_ps(rms);

        // Pass 2: normalize and scale
        n = 0;
        for (; n + 8 <= N; n += 8) {
            __m256 v = _mm256_loadu_ps(row_in + n);
            __m256 w = _mm256_loadu_ps(weight + n);
            __m256 result = _mm256_mul_ps(_mm256_mul_ps(v, vrms), w);
            _mm256_storeu_ps(row_out + n, result);
        }
        for (; n < N; n++) {
            row_out[n] = weight[n] * row_in[n] * rms;
        }
    }
}

static void rms_norm_bf16_avx2(llaisys::bf16_t *out, const llaisys::bf16_t *in,
                                const llaisys::bf16_t *weight, float eps, size_t M, size_t N) {
    using namespace llaisys::ops::cpu;

    for (size_t m = 0; m < M; m++) {
        const llaisys::bf16_t *row_in = in + m * N;
        llaisys::bf16_t *row_out = out + m * N;

        // Pass 1: sum of squares
        __m256 vsum_sq = _mm256_setzero_ps();
        size_t n = 0;
        for (; n + 8 <= N; n += 8) {
            __m256 v = bf16x8_to_f32x8(row_in + n);
#ifdef __FMA__
            vsum_sq = _mm256_fmadd_ps(v, v, vsum_sq);
#else
            vsum_sq = _mm256_add_ps(vsum_sq, _mm256_mul_ps(v, v));
#endif
        }
        float sum_sq = hsum_f32x8(vsum_sq);
        for (; n < N; n++) {
            float val = llaisys::utils::cast<float>(row_in[n]);
            sum_sq += val * val;
        }

        float rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(N) + eps);
        __m256 vrms = _mm256_set1_ps(rms);

        // Pass 2: normalize and scale
        n = 0;
        for (; n + 8 <= N; n += 8) {
            __m256 v = bf16x8_to_f32x8(row_in + n);
            __m256 w = bf16x8_to_f32x8(weight + n);
            __m256 result = _mm256_mul_ps(_mm256_mul_ps(v, vrms), w);
            f32x8_store_bf16(row_out + n, result);
        }
        for (; n < N; n++) {
            float in_val = llaisys::utils::cast<float>(row_in[n]);
            float w_val = llaisys::utils::cast<float>(weight[n]);
            row_out[n] = llaisys::utils::cast<llaisys::bf16_t>(w_val * in_val * rms);
        }
    }
}
#endif // __AVX2__

template <typename T>
static void rms_norm_scalar(T *out, const T *in, const T *weight,
                            float eps, size_t M, size_t N) {
    for (size_t m = 0; m < M; m++) {
        float sum_sq = 0.0f;
        for (size_t n = 0; n < N; n++) {
            float val;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                val = llaisys::utils::cast<float>(in[m * N + n]);
            } else {
                val = static_cast<float>(in[m * N + n]);
            }
            sum_sq += val * val;
        }
        float rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(N) + eps);

        for (size_t n = 0; n < N; n++) {
            float in_val, w_val;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                in_val = llaisys::utils::cast<float>(in[m * N + n]);
                w_val = llaisys::utils::cast<float>(weight[n]);
            } else {
                in_val = static_cast<float>(in[m * N + n]);
                w_val = static_cast<float>(weight[n]);
            }
            float result = w_val * in_val * rms;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                out[m * N + n] = llaisys::utils::cast<T>(result);
            } else {
                out[m * N + n] = static_cast<T>(result);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out_ptr, const std::byte *in_ptr, const std::byte *weight_ptr,
              float eps, llaisysDataType_t dtype, size_t M, size_t N) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
#ifdef __AVX2__
        return rms_norm_f32_avx2(reinterpret_cast<float *>(out_ptr),
                                  reinterpret_cast<const float *>(in_ptr),
                                  reinterpret_cast<const float *>(weight_ptr), eps, M, N);
#else
        return rms_norm_scalar(reinterpret_cast<float *>(out_ptr),
                               reinterpret_cast<const float *>(in_ptr),
                               reinterpret_cast<const float *>(weight_ptr), eps, M, N);
#endif
    case LLAISYS_DTYPE_BF16:
#ifdef __AVX2__
        return rms_norm_bf16_avx2(reinterpret_cast<bf16_t *>(out_ptr),
                                   reinterpret_cast<const bf16_t *>(in_ptr),
                                   reinterpret_cast<const bf16_t *>(weight_ptr), eps, M, N);
#else
        return rms_norm_scalar(reinterpret_cast<bf16_t *>(out_ptr),
                               reinterpret_cast<const bf16_t *>(in_ptr),
                               reinterpret_cast<const bf16_t *>(weight_ptr), eps, M, N);
#endif
    case LLAISYS_DTYPE_F16:
        return rms_norm_scalar(reinterpret_cast<fp16_t *>(out_ptr),
                               reinterpret_cast<const fp16_t *>(in_ptr),
                               reinterpret_cast<const fp16_t *>(weight_ptr), eps, M, N);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
