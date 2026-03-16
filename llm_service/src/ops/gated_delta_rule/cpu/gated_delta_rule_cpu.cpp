#include "gated_delta_rule_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>
#include <vector>

// Helper to convert any type to float
template <typename T>
static inline float to_f(T v) {
    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        return llaisys::utils::cast<float>(v);
    } else {
        return static_cast<float>(v);
    }
}

template <typename T>
static inline T from_f(float v) {
    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        return llaisys::utils::cast<T>(v);
    } else {
        return static_cast<T>(v);
    }
}

template <typename T>
static void recurrent_impl(T *out, float *state,
                            const T *q, const T *k, const T *v,
                            const float *g, const float *beta,
                            size_t n_heads, size_t dk, size_t dv) {
    for (size_t h = 0; h < n_heads; h++) {
        float decay = g[h];
        float b = beta[h];
        float *S = state + h * dv * dk;

        // Proper delta rule (HF convention):
        // Step 1: Decay state: S = decay * S
        // Step 2: Retrieve from decayed state: retrieved = S @ k
        // Step 3: delta = beta * (v - retrieved)
        // Step 4: Update: S = S + delta ⊗ k
        for (size_t di = 0; di < dv; di++) {
            for (size_t dj = 0; dj < dk; dj++) {
                S[di * dk + dj] *= decay;
            }
        }
        for (size_t di = 0; di < dv; di++) {
            float retrieved = 0.0f;
            for (size_t dj = 0; dj < dk; dj++) {
                retrieved += S[di * dk + dj] * to_f(k[h * dk + dj]);
            }
            float delta = b * (to_f(v[h * dv + di]) - retrieved);
            for (size_t dj = 0; dj < dk; dj++) {
                S[di * dk + dj] += delta * to_f(k[h * dk + dj]);
            }
        }

        // Output: o = S @ q
        for (size_t di = 0; di < dv; di++) {
            float acc = 0.0f;
            for (size_t dj = 0; dj < dk; dj++) {
                acc += S[di * dk + dj] * to_f(q[h * dk + dj]);
            }
            out[h * dv + di] = from_f<T>(acc);
        }
    }
}

template <typename T>
static void chunk_impl(T *out, float *final_state,
                        const T *q, const T *k, const T *v,
                        const float *g, const float *beta,
                        size_t seq_len, size_t n_heads, size_t dk, size_t dv) {
    // Zero init state
    for (size_t i = 0; i < n_heads * dv * dk; i++) final_state[i] = 0.0f;

    for (size_t t = 0; t < seq_len; t++) {
        for (size_t h = 0; h < n_heads; h++) {
            float decay = g[t * n_heads + h];
            float b = beta[t * n_heads + h];
            float *S = final_state + h * dv * dk;

            const T *q_t = q + t * n_heads * dk + h * dk;
            const T *k_t = k + t * n_heads * dk + h * dk;
            const T *v_t = v + t * n_heads * dv + h * dv;

            // Proper delta rule (HF convention):
            // Step 1: Decay state, Step 2: Retrieve from decayed, Step 3: Update
            for (size_t di = 0; di < dv; di++) {
                for (size_t dj = 0; dj < dk; dj++) {
                    S[di * dk + dj] *= decay;
                }
            }
            for (size_t di = 0; di < dv; di++) {
                float retrieved = 0.0f;
                for (size_t dj = 0; dj < dk; dj++) {
                    retrieved += S[di * dk + dj] * to_f(k_t[dj]);
                }
                float delta = b * (to_f(v_t[di]) - retrieved);
                for (size_t dj = 0; dj < dk; dj++) {
                    S[di * dk + dj] += delta * to_f(k_t[dj]);
                }
            }

            // Output: o = S @ q
            for (size_t di = 0; di < dv; di++) {
                float acc = 0.0f;
                for (size_t dj = 0; dj < dk; dj++) {
                    acc += S[di * dk + dj] * to_f(q_t[dj]);
                }
                out[t * n_heads * dv + h * dv + di] = from_f<T>(acc);
            }
        }
    }
}

namespace llaisys::ops::cpu {

void gated_delta_rule_recurrent(
    std::byte *out, std::byte *state,
    const std::byte *q, const std::byte *k, const std::byte *v,
    const std::byte *g, const std::byte *beta,
    llaisysDataType_t dtype, size_t n_heads, size_t dk, size_t dv) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return recurrent_impl((float *)out, (float *)state, (const float *)q, (const float *)k,
                               (const float *)v, (const float *)g, (const float *)beta, n_heads, dk, dv);
    case LLAISYS_DTYPE_BF16:
        return recurrent_impl((llaisys::bf16_t *)out, (float *)state, (const llaisys::bf16_t *)q,
                               (const llaisys::bf16_t *)k, (const llaisys::bf16_t *)v,
                               (const float *)g, (const float *)beta, n_heads, dk, dv);
    case LLAISYS_DTYPE_F16:
        return recurrent_impl((llaisys::fp16_t *)out, (float *)state, (const llaisys::fp16_t *)q,
                               (const llaisys::fp16_t *)k, (const llaisys::fp16_t *)v,
                               (const float *)g, (const float *)beta, n_heads, dk, dv);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void gated_delta_rule_chunk(
    std::byte *out, std::byte *final_state,
    const std::byte *q, const std::byte *k, const std::byte *v,
    const std::byte *g, const std::byte *beta,
    llaisysDataType_t dtype, size_t seq_len, size_t n_heads, size_t dk, size_t dv) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return chunk_impl((float *)out, (float *)final_state, (const float *)q, (const float *)k,
                           (const float *)v, (const float *)g, (const float *)beta, seq_len, n_heads, dk, dv);
    case LLAISYS_DTYPE_BF16:
        return chunk_impl((llaisys::bf16_t *)out, (float *)final_state, (const llaisys::bf16_t *)q,
                           (const llaisys::bf16_t *)k, (const llaisys::bf16_t *)v,
                           (const float *)g, (const float *)beta, seq_len, n_heads, dk, dv);
    case LLAISYS_DTYPE_F16:
        return chunk_impl((llaisys::fp16_t *)out, (float *)final_state, (const llaisys::fp16_t *)q,
                           (const llaisys::fp16_t *)k, (const llaisys::fp16_t *)v,
                           (const float *)g, (const float *)beta, seq_len, n_heads, dk, dv);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu
