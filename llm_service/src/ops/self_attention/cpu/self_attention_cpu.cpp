#include "../../cpu_simd_utils.hpp"

#include "self_attention_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>
#include <limits>
#include <vector>

#ifdef __AVX2__
// AVX2 dot product of two float vectors
static inline float dot_f32_avx2(const float *a, const float *b, size_t len) {
    using namespace llaisys::ops::cpu;
    __m256 vsum = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= len; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
#ifdef __FMA__
        vsum = _mm256_fmadd_ps(va, vb, vsum);
#else
        vsum = _mm256_add_ps(vsum, _mm256_mul_ps(va, vb));
#endif
    }
    float sum = llaisys::ops::cpu::hsum_f32x8(vsum);
    for (; i < len; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

// AVX2 weighted sum: out[d] += weight * v[d]
static inline void weighted_add_f32_avx2(float *out, const float *v, float weight, size_t len) {
    __m256 vw = _mm256_set1_ps(weight);
    size_t i = 0;
    for (; i + 8 <= len; i += 8) {
        __m256 vout = _mm256_loadu_ps(out + i);
        __m256 vv = _mm256_loadu_ps(v + i);
#ifdef __FMA__
        vout = _mm256_fmadd_ps(vv, vw, vout);
#else
        vout = _mm256_add_ps(vout, _mm256_mul_ps(vv, vw));
#endif
        _mm256_storeu_ps(out + i, vout);
    }
    for (; i < len; i++) {
        out[i] += weight * v[i];
    }
}

static void self_attention_f32_avx2(float *attn_val, const float *q, const float *k,
                                     const float *v, float scale,
                                     size_t qlen, size_t kvlen, size_t nhead, size_t nkvhead, size_t hd) {
    using namespace llaisys::ops::cpu;
    size_t heads_per_kv = nhead / nkvhead;

    for (size_t h = 0; h < nhead; h++) {
        size_t kv_h = h / heads_per_kv;

        for (size_t qi = 0; qi < qlen; qi++) {
            const float *q_vec = q + qi * nhead * hd + h * hd;

            // Compute attention scores with AVX2 dot product
            std::vector<float> scores(kvlen);
            float max_score = -std::numeric_limits<float>::infinity();

            size_t causal_limit = qi + (kvlen - qlen);
            for (size_t ki = 0; ki <= causal_limit && ki < kvlen; ki++) {
                const float *k_vec = k + ki * nkvhead * hd + kv_h * hd;
                scores[ki] = dot_f32_avx2(q_vec, k_vec, hd) * scale;
                if (scores[ki] > max_score) {
                    max_score = scores[ki];
                }
            }
            for (size_t ki = causal_limit + 1; ki < kvlen; ki++) {
                scores[ki] = -std::numeric_limits<float>::infinity();
            }

            // Softmax with AVX2
            __m256 vsum_exp = _mm256_setzero_ps();
            __m256 vmax = _mm256_set1_ps(max_score);
            __m256 neg_inf_half = _mm256_set1_ps(-std::numeric_limits<float>::infinity() / 2);

            size_t ki = 0;
            for (; ki + 8 <= kvlen; ki += 8) {
                __m256 vs = _mm256_loadu_ps(scores.data() + ki);
                // Mask: scores > -inf/2
                __m256 mask = _mm256_cmp_ps(vs, neg_inf_half, _CMP_GT_OQ);
                __m256 shifted = _mm256_sub_ps(vs, vmax);
                // For exp, we just store shifted values and compute exp below
                // (approximate exp is available but for softmax accuracy, use scalar)
                _mm256_storeu_ps(scores.data() + ki, shifted);
            }
            for (; ki < kvlen; ki++) {
                if (scores[ki] > -std::numeric_limits<float>::infinity() / 2) {
                    scores[ki] -= max_score;
                }
            }

            // Compute exp and sum (scalar for precision)
            float sum_exp = 0.0f;
            for (size_t ki2 = 0; ki2 < kvlen; ki2++) {
                if (scores[ki2] > -88.0f) { // exp underflow threshold
                    scores[ki2] = std::exp(scores[ki2]);
                    sum_exp += scores[ki2];
                } else {
                    scores[ki2] = 0.0f;
                }
            }

            float inv_sum = 1.0f / sum_exp;
            for (size_t ki2 = 0; ki2 < kvlen; ki2++) {
                scores[ki2] *= inv_sum;
            }

            // Weighted sum of V with AVX2
            float *out_vec = attn_val + qi * nhead * hd + h * hd;
            // Zero output
            size_t d = 0;
            for (; d + 8 <= hd; d += 8) {
                _mm256_storeu_ps(out_vec + d, _mm256_setzero_ps());
            }
            for (; d < hd; d++) {
                out_vec[d] = 0.0f;
            }

            for (size_t ki2 = 0; ki2 < kvlen; ki2++) {
                if (scores[ki2] > 0.0f) {
                    const float *v_vec = v + ki2 * nkvhead * hd + kv_h * hd;
                    weighted_add_f32_avx2(out_vec, v_vec, scores[ki2], hd);
                }
            }
        }
    }
}
#endif // __AVX2__

template <typename T>
static void self_attention_scalar(T *attn_val, const T *q, const T *k, const T *v,
                                  float scale, size_t qlen, size_t kvlen,
                                  size_t nhead, size_t nkvhead, size_t hd) {
    size_t heads_per_kv = nhead / nkvhead;

    for (size_t h = 0; h < nhead; h++) {
        size_t kv_h = h / heads_per_kv;

        for (size_t qi = 0; qi < qlen; qi++) {
            std::vector<float> scores(kvlen);
            float max_score = -std::numeric_limits<float>::infinity();

            for (size_t ki = 0; ki < kvlen; ki++) {
                size_t causal_limit = qi + (kvlen - qlen);
                if (ki > causal_limit) {
                    scores[ki] = -std::numeric_limits<float>::infinity();
                } else {
                    float dot = 0.0f;
                    for (size_t d = 0; d < hd; d++) {
                        float q_val, k_val;
                        size_t q_idx = qi * nhead * hd + h * hd + d;
                        size_t k_idx = ki * nkvhead * hd + kv_h * hd + d;
                        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                            q_val = llaisys::utils::cast<float>(q[q_idx]);
                            k_val = llaisys::utils::cast<float>(k[k_idx]);
                        } else {
                            q_val = static_cast<float>(q[q_idx]);
                            k_val = static_cast<float>(k[k_idx]);
                        }
                        dot += q_val * k_val;
                    }
                    scores[ki] = dot * scale;
                }
                if (scores[ki] > max_score) {
                    max_score = scores[ki];
                }
            }

            float sum_exp = 0.0f;
            for (size_t ki = 0; ki < kvlen; ki++) {
                if (scores[ki] > -std::numeric_limits<float>::infinity() / 2) {
                    scores[ki] = std::exp(scores[ki] - max_score);
                    sum_exp += scores[ki];
                } else {
                    scores[ki] = 0.0f;
                }
            }
            for (size_t ki = 0; ki < kvlen; ki++) {
                scores[ki] /= sum_exp;
            }

            for (size_t d = 0; d < hd; d++) {
                float out_val = 0.0f;
                for (size_t ki = 0; ki < kvlen; ki++) {
                    if (scores[ki] > 0.0f) {
                        float v_val;
                        size_t v_idx = ki * nkvhead * hd + kv_h * hd + d;
                        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                            v_val = llaisys::utils::cast<float>(v[v_idx]);
                        } else {
                            v_val = static_cast<float>(v[v_idx]);
                        }
                        out_val += scores[ki] * v_val;
                    }
                }
                size_t out_idx = qi * nhead * hd + h * hd + d;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    attn_val[out_idx] = llaisys::utils::cast<T>(out_val);
                } else {
                    attn_val[out_idx] = static_cast<T>(out_val);
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val_ptr, const std::byte *q_ptr, const std::byte *k_ptr,
                    const std::byte *v_ptr, float scale, llaisysDataType_t dtype,
                    size_t qlen, size_t kvlen, size_t nhead, size_t nkvhead, size_t hd) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
#ifdef __AVX2__
        return self_attention_f32_avx2(reinterpret_cast<float *>(attn_val_ptr),
                                        reinterpret_cast<const float *>(q_ptr),
                                        reinterpret_cast<const float *>(k_ptr),
                                        reinterpret_cast<const float *>(v_ptr),
                                        scale, qlen, kvlen, nhead, nkvhead, hd);
#else
        return self_attention_scalar(reinterpret_cast<float *>(attn_val_ptr),
                                     reinterpret_cast<const float *>(q_ptr),
                                     reinterpret_cast<const float *>(k_ptr),
                                     reinterpret_cast<const float *>(v_ptr),
                                     scale, qlen, kvlen, nhead, nkvhead, hd);
#endif
    case LLAISYS_DTYPE_BF16:
        return self_attention_scalar(reinterpret_cast<bf16_t *>(attn_val_ptr),
                                     reinterpret_cast<const bf16_t *>(q_ptr),
                                     reinterpret_cast<const bf16_t *>(k_ptr),
                                     reinterpret_cast<const bf16_t *>(v_ptr),
                                     scale, qlen, kvlen, nhead, nkvhead, hd);
    case LLAISYS_DTYPE_F16:
        return self_attention_scalar(reinterpret_cast<fp16_t *>(attn_val_ptr),
                                     reinterpret_cast<const fp16_t *>(q_ptr),
                                     reinterpret_cast<const fp16_t *>(k_ptr),
                                     reinterpret_cast<const fp16_t *>(v_ptr),
                                     scale, qlen, kvlen, nhead, nkvhead, hd);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
