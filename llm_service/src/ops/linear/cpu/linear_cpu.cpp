#include "../../cpu_simd_utils.hpp"

#include "linear_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

#ifdef __AVX2__
// GEMM: out[M,N] = in[M,K] * weight[N,K]^T + bias[N]
// Weight layout: [N, K] (row-major, each row is one output neuron)
static void linear_f32_avx2(float *out, const float *in, const float *weight,
                             const float *bias, size_t M, size_t K, size_t N) {
    using namespace llaisys::ops::cpu;

    for (size_t m = 0; m < M; m++) {
        const float *in_row = in + m * K;
        float *out_row = out + m * N;

        for (size_t n = 0; n < N; n++) {
            const float *w_row = weight + n * K;
            __m256 vsum = _mm256_setzero_ps();

            size_t k = 0;
            for (; k + 8 <= K; k += 8) {
                __m256 vi = _mm256_loadu_ps(in_row + k);
                __m256 vw = _mm256_loadu_ps(w_row + k);
#ifdef __FMA__
                vsum = _mm256_fmadd_ps(vi, vw, vsum);
#else
                vsum = _mm256_add_ps(vsum, _mm256_mul_ps(vi, vw));
#endif
            }
            float sum = hsum_f32x8(vsum);
            for (; k < K; k++) {
                sum += in_row[k] * w_row[k];
            }

            if (bias) {
                sum += bias[n];
            }
            out_row[n] = sum;
        }
    }
}

static void linear_bf16_avx2(llaisys::bf16_t *out, const llaisys::bf16_t *in,
                              const llaisys::bf16_t *weight, const llaisys::bf16_t *bias,
                              size_t M, size_t K, size_t N) {
    using namespace llaisys::ops::cpu;

    for (size_t m = 0; m < M; m++) {
        const llaisys::bf16_t *in_row = in + m * K;
        llaisys::bf16_t *out_row = out + m * N;

        for (size_t n = 0; n < N; n++) {
            const llaisys::bf16_t *w_row = weight + n * K;
            __m256 vsum = _mm256_setzero_ps();

            size_t k = 0;
            for (; k + 8 <= K; k += 8) {
                __m256 vi = bf16x8_to_f32x8(in_row + k);
                __m256 vw = bf16x8_to_f32x8(w_row + k);
#ifdef __FMA__
                vsum = _mm256_fmadd_ps(vi, vw, vsum);
#else
                vsum = _mm256_add_ps(vsum, _mm256_mul_ps(vi, vw));
#endif
            }
            float sum = hsum_f32x8(vsum);
            for (; k < K; k++) {
                float iv = llaisys::utils::cast<float>(in_row[k]);
                float wv = llaisys::utils::cast<float>(w_row[k]);
                sum += iv * wv;
            }

            if (bias) {
                sum += llaisys::utils::cast<float>(bias[n]);
            }
            out_row[n] = llaisys::utils::cast<llaisys::bf16_t>(sum);
        }
    }
}
#endif // __AVX2__

template <typename T>
static void linear_scalar(T *out, const T *in, const T *weight, const T *bias,
                          size_t M, size_t K, size_t N) {
    for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; k++) {
                float in_val, w_val;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    in_val = llaisys::utils::cast<float>(in[m * K + k]);
                    w_val = llaisys::utils::cast<float>(weight[n * K + k]);
                } else {
                    in_val = static_cast<float>(in[m * K + k]);
                    w_val = static_cast<float>(weight[n * K + k]);
                }
                sum += in_val * w_val;
            }
            if (bias) {
                float b_val;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    b_val = llaisys::utils::cast<float>(bias[n]);
                } else {
                    b_val = static_cast<float>(bias[n]);
                }
                sum += b_val;
            }
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                out[m * N + n] = llaisys::utils::cast<T>(sum);
            } else {
                out[m * N + n] = static_cast<T>(sum);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out_ptr, const std::byte *in_ptr, const std::byte *weight_ptr,
            const std::byte *bias_ptr, llaisysDataType_t dtype,
            size_t M, size_t K, size_t N) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
#ifdef __AVX2__
        return linear_f32_avx2(reinterpret_cast<float *>(out_ptr),
                               reinterpret_cast<const float *>(in_ptr),
                               reinterpret_cast<const float *>(weight_ptr),
                               bias_ptr ? reinterpret_cast<const float *>(bias_ptr) : nullptr,
                               M, K, N);
#else
        return linear_scalar(reinterpret_cast<float *>(out_ptr),
                             reinterpret_cast<const float *>(in_ptr),
                             reinterpret_cast<const float *>(weight_ptr),
                             bias_ptr ? reinterpret_cast<const float *>(bias_ptr) : nullptr,
                             M, K, N);
#endif
    case LLAISYS_DTYPE_BF16:
#ifdef __AVX2__
        return linear_bf16_avx2(reinterpret_cast<bf16_t *>(out_ptr),
                                reinterpret_cast<const bf16_t *>(in_ptr),
                                reinterpret_cast<const bf16_t *>(weight_ptr),
                                bias_ptr ? reinterpret_cast<const bf16_t *>(bias_ptr) : nullptr,
                                M, K, N);
#else
        return linear_scalar(reinterpret_cast<bf16_t *>(out_ptr),
                             reinterpret_cast<const bf16_t *>(in_ptr),
                             reinterpret_cast<const bf16_t *>(weight_ptr),
                             bias_ptr ? reinterpret_cast<const bf16_t *>(bias_ptr) : nullptr,
                             M, K, N);
#endif
    case LLAISYS_DTYPE_F16:
        return linear_scalar(reinterpret_cast<fp16_t *>(out_ptr),
                             reinterpret_cast<const fp16_t *>(in_ptr),
                             reinterpret_cast<const fp16_t *>(weight_ptr),
                             bias_ptr ? reinterpret_cast<const fp16_t *>(bias_ptr) : nullptr,
                             M, K, N);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
