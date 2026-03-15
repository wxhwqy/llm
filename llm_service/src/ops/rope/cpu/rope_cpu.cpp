#include "../../cpu_simd_utils.hpp"

#include "rope_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>
#include <vector>

#ifdef __AVX2__
// RoPE applies rotation to pairs (x[j], x[j+half_dim]):
//   out[j]          = x[j] * cos(freq) - x[j+half_dim] * sin(freq)
//   out[j+half_dim] = x[j+half_dim] * cos(freq) + x[j] * sin(freq)
//
// For each head, we precompute cos/sin arrays for half_dim elements,
// then apply the rotation 8 elements at a time using AVX2.

static void rope_f32_avx2(float *out, const float *in, const int64_t *pos_ids,
                           float theta, size_t seq_len, size_t n_head, size_t head_dim) {
    using namespace llaisys::ops::cpu;
    size_t half_dim = head_dim / 2;

    // Precompute frequency factors: freq_j = 1/theta^(2j/head_dim)
    std::vector<float> inv_freq(half_dim);
    for (size_t j = 0; j < half_dim; j++) {
        inv_freq[j] = 1.0f / std::pow(theta, 2.0f * static_cast<float>(j) / static_cast<float>(head_dim));
    }

    for (size_t s = 0; s < seq_len; s++) {
        float pos = static_cast<float>(pos_ids[s]);

        // Precompute cos/sin for this position
        std::vector<float> cos_vals(half_dim);
        std::vector<float> sin_vals(half_dim);
        for (size_t j = 0; j < half_dim; j++) {
            float freq = pos * inv_freq[j];
            cos_vals[j] = std::cos(freq);
            sin_vals[j] = std::sin(freq);
        }

        for (size_t h = 0; h < n_head; h++) {
            size_t base = s * n_head * head_dim + h * head_dim;
            const float *in_a = in + base;            // first half
            const float *in_b = in + base + half_dim; // second half
            float *out_a = out + base;
            float *out_b = out + base + half_dim;

            size_t j = 0;
            for (; j + 8 <= half_dim; j += 8) {
                __m256 va = _mm256_loadu_ps(in_a + j);
                __m256 vb = _mm256_loadu_ps(in_b + j);
                __m256 vc = _mm256_loadu_ps(cos_vals.data() + j);
                __m256 vs = _mm256_loadu_ps(sin_vals.data() + j);

                // out_a = a * cos - b * sin
                // out_b = b * cos + a * sin
#ifdef __FMA__
                __m256 ra = _mm256_fmsub_ps(va, vc, _mm256_mul_ps(vb, vs));
                __m256 rb = _mm256_fmadd_ps(va, vs, _mm256_mul_ps(vb, vc));
#else
                __m256 ra = _mm256_sub_ps(_mm256_mul_ps(va, vc), _mm256_mul_ps(vb, vs));
                __m256 rb = _mm256_add_ps(_mm256_mul_ps(vb, vc), _mm256_mul_ps(va, vs));
#endif
                _mm256_storeu_ps(out_a + j, ra);
                _mm256_storeu_ps(out_b + j, rb);
            }
            for (; j < half_dim; j++) {
                float a_val = in_a[j];
                float b_val = in_b[j];
                out_a[j] = a_val * cos_vals[j] - b_val * sin_vals[j];
                out_b[j] = b_val * cos_vals[j] + a_val * sin_vals[j];
            }
        }
    }
}

static void rope_bf16_avx2(llaisys::bf16_t *out, const llaisys::bf16_t *in,
                            const int64_t *pos_ids, float theta,
                            size_t seq_len, size_t n_head, size_t head_dim) {
    using namespace llaisys::ops::cpu;
    size_t half_dim = head_dim / 2;

    std::vector<float> inv_freq(half_dim);
    for (size_t j = 0; j < half_dim; j++) {
        inv_freq[j] = 1.0f / std::pow(theta, 2.0f * static_cast<float>(j) / static_cast<float>(head_dim));
    }

    for (size_t s = 0; s < seq_len; s++) {
        float pos = static_cast<float>(pos_ids[s]);

        std::vector<float> cos_vals(half_dim);
        std::vector<float> sin_vals(half_dim);
        for (size_t j = 0; j < half_dim; j++) {
            float freq = pos * inv_freq[j];
            cos_vals[j] = std::cos(freq);
            sin_vals[j] = std::sin(freq);
        }

        for (size_t h = 0; h < n_head; h++) {
            size_t base = s * n_head * head_dim + h * head_dim;
            const llaisys::bf16_t *in_a = in + base;
            const llaisys::bf16_t *in_b = in + base + half_dim;
            llaisys::bf16_t *out_a = out + base;
            llaisys::bf16_t *out_b = out + base + half_dim;

            size_t j = 0;
            for (; j + 8 <= half_dim; j += 8) {
                __m256 va = bf16x8_to_f32x8(in_a + j);
                __m256 vb = bf16x8_to_f32x8(in_b + j);
                __m256 vc = _mm256_loadu_ps(cos_vals.data() + j);
                __m256 vs = _mm256_loadu_ps(sin_vals.data() + j);

#ifdef __FMA__
                __m256 ra = _mm256_fmsub_ps(va, vc, _mm256_mul_ps(vb, vs));
                __m256 rb = _mm256_fmadd_ps(va, vs, _mm256_mul_ps(vb, vc));
#else
                __m256 ra = _mm256_sub_ps(_mm256_mul_ps(va, vc), _mm256_mul_ps(vb, vs));
                __m256 rb = _mm256_add_ps(_mm256_mul_ps(vb, vc), _mm256_mul_ps(va, vs));
#endif
                f32x8_store_bf16(out_a + j, ra);
                f32x8_store_bf16(out_b + j, rb);
            }
            for (; j < half_dim; j++) {
                float a_val = llaisys::utils::cast<float>(in_a[j]);
                float b_val = llaisys::utils::cast<float>(in_b[j]);
                out_a[j] = llaisys::utils::cast<llaisys::bf16_t>(a_val * cos_vals[j] - b_val * sin_vals[j]);
                out_b[j] = llaisys::utils::cast<llaisys::bf16_t>(b_val * cos_vals[j] + a_val * sin_vals[j]);
            }
        }
    }
}
#endif // __AVX2__

template <typename T>
static void rope_scalar(T *out, const T *in, const int64_t *pos_ids,
                        float theta, size_t seq_len, size_t n_head, size_t head_dim) {
    size_t half_dim = head_dim / 2;

    for (size_t s = 0; s < seq_len; s++) {
        float pos = static_cast<float>(pos_ids[s]);
        for (size_t h = 0; h < n_head; h++) {
            for (size_t j = 0; j < half_dim; j++) {
                float freq = pos / std::pow(theta, 2.0f * static_cast<float>(j) / static_cast<float>(head_dim));
                float cos_val = std::cos(freq);
                float sin_val = std::sin(freq);

                size_t idx_a = s * n_head * head_dim + h * head_dim + j;
                size_t idx_b = s * n_head * head_dim + h * head_dim + j + half_dim;

                float a_val, b_val;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    a_val = llaisys::utils::cast<float>(in[idx_a]);
                    b_val = llaisys::utils::cast<float>(in[idx_b]);
                } else {
                    a_val = static_cast<float>(in[idx_a]);
                    b_val = static_cast<float>(in[idx_b]);
                }

                float a_out = a_val * cos_val - b_val * sin_val;
                float b_out = b_val * cos_val + a_val * sin_val;

                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    out[idx_a] = llaisys::utils::cast<T>(a_out);
                    out[idx_b] = llaisys::utils::cast<T>(b_out);
                } else {
                    out[idx_a] = static_cast<T>(a_out);
                    out[idx_b] = static_cast<T>(b_out);
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out_ptr, const std::byte *in_ptr, const std::byte *pos_ids_ptr,
          float theta, llaisysDataType_t dtype,
          size_t seq_len, size_t n_head, size_t head_dim) {
    const int64_t *pos_ids = reinterpret_cast<const int64_t *>(pos_ids_ptr);

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
#ifdef __AVX2__
        return rope_f32_avx2(reinterpret_cast<float *>(out_ptr),
                             reinterpret_cast<const float *>(in_ptr),
                             pos_ids, theta, seq_len, n_head, head_dim);
#else
        return rope_scalar(reinterpret_cast<float *>(out_ptr),
                           reinterpret_cast<const float *>(in_ptr),
                           pos_ids, theta, seq_len, n_head, head_dim);
#endif
    case LLAISYS_DTYPE_BF16:
#ifdef __AVX2__
        return rope_bf16_avx2(reinterpret_cast<bf16_t *>(out_ptr),
                              reinterpret_cast<const bf16_t *>(in_ptr),
                              pos_ids, theta, seq_len, n_head, head_dim);
#else
        return rope_scalar(reinterpret_cast<bf16_t *>(out_ptr),
                           reinterpret_cast<const bf16_t *>(in_ptr),
                           pos_ids, theta, seq_len, n_head, head_dim);
#endif
    case LLAISYS_DTYPE_F16:
        return rope_scalar(reinterpret_cast<fp16_t *>(out_ptr),
                           reinterpret_cast<const fp16_t *>(in_ptr),
                           pos_ids, theta, seq_len, n_head, head_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
