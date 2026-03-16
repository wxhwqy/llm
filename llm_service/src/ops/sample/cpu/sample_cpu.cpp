#include "../../cpu_simd_utils.hpp"

#include "sample_cpu.hpp"
#include "../../../utils.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>

#ifdef __AVX2__
static void apply_temperature_f32_avx2(float *out, const float *in, float inv_temp, size_t n) {
    __m256 vtemp = _mm256_set1_ps(inv_temp);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(in + i);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(v, vtemp));
    }
    for (; i < n; i++) {
        out[i] = in[i] * inv_temp;
    }
}
#endif

// Softmax in-place
static void softmax_f32(float *data, size_t n) {
    // Find max
    float max_val = -std::numeric_limits<float>::infinity();
#ifdef __AVX2__
    {
        using namespace llaisys::ops::cpu;
        __m256 vmax = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 v = _mm256_loadu_ps(data + i);
            vmax = _mm256_max_ps(vmax, v);
        }
        max_val = hmax_f32x8(vmax);
        for (; i < n; i++) {
            if (data[i] > max_val) max_val = data[i];
        }
    }
#else
    for (size_t i = 0; i < n; i++) {
        if (data[i] > max_val) max_val = data[i];
    }
#endif

    // exp and sum
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        data[i] = std::exp(data[i] - max_val);
        sum += data[i];
    }

    // normalize
    float inv_sum = 1.0f / sum;
#ifdef __AVX2__
    {
        __m256 vinv = _mm256_set1_ps(inv_sum);
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 v = _mm256_loadu_ps(data + i);
            _mm256_storeu_ps(data + i, _mm256_mul_ps(v, vinv));
        }
        for (; i < n; i++) {
            data[i] *= inv_sum;
        }
    }
#else
    for (size_t i = 0; i < n; i++) {
        data[i] *= inv_sum;
    }
#endif
}

template <typename T>
static void sample_impl(int64_t *output_idx, const T *logits, float *workspace,
                         size_t vocab_size, float temperature, int top_k, float top_p,
                         uint64_t seed,
                         const int64_t *penalty_tokens, size_t n_penalty_tokens,
                         float repetition_penalty) {
    // Convert logits to float workspace with temperature scaling
    float inv_temp = 1.0f / temperature;

    if constexpr (std::is_same_v<T, float>) {
#ifdef __AVX2__
        apply_temperature_f32_avx2(workspace, logits, inv_temp, vocab_size);
#else
        for (size_t i = 0; i < vocab_size; i++) {
            workspace[i] = logits[i] * inv_temp;
        }
#endif
    } else {
        for (size_t i = 0; i < vocab_size; i++) {
            float val;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                val = llaisys::utils::cast<float>(logits[i]);
            } else {
                val = static_cast<float>(logits[i]);
            }
            workspace[i] = val * inv_temp;
        }
    }

    // Apply repetition penalty: penalize tokens that appeared in history
    if (repetition_penalty != 1.0f && n_penalty_tokens > 0) {
        for (size_t i = 0; i < n_penalty_tokens; i++) {
            int64_t tid = penalty_tokens[i];
            if (tid >= 0 && static_cast<size_t>(tid) < vocab_size) {
                float &logit = workspace[tid];
                // If logit > 0, divide by penalty; if logit < 0, multiply by penalty
                if (logit > 0.0f) {
                    logit /= repetition_penalty;
                } else {
                    logit *= repetition_penalty;
                }
            }
        }
    }

    // Apply softmax
    softmax_f32(workspace, vocab_size);

    // Create index array sorted by probability (descending)
    std::vector<int64_t> indices(vocab_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(),
                      indices.begin() + std::min(static_cast<size_t>(top_k > 0 ? top_k : vocab_size), vocab_size),
                      indices.end(),
                      [&](int64_t a, int64_t b) { return workspace[a] > workspace[b]; });

    // Apply top-k
    size_t effective_k = (top_k > 0 && static_cast<size_t>(top_k) < vocab_size)
                             ? static_cast<size_t>(top_k) : vocab_size;

    // Apply top-p (nucleus sampling)
    float cumulative = 0.0f;
    size_t effective_n = effective_k;
    for (size_t i = 0; i < effective_k; i++) {
        cumulative += workspace[indices[i]];
        if (cumulative >= top_p) {
            effective_n = i + 1;
            break;
        }
    }

    // Re-normalize the selected tokens
    float selected_sum = 0.0f;
    for (size_t i = 0; i < effective_n; i++) {
        selected_sum += workspace[indices[i]];
    }

    // Sample from the distribution
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, selected_sum);
    float r = dist(rng);

    float running = 0.0f;
    for (size_t i = 0; i < effective_n; i++) {
        running += workspace[indices[i]];
        if (running >= r) {
            output_idx[0] = indices[i];
            return;
        }
    }
    // Fallback: return the most probable token
    output_idx[0] = indices[0];
}

namespace llaisys::ops::cpu {
void sample(std::byte *output_idx_ptr, const std::byte *logits_ptr, std::byte *workspace_ptr,
            llaisysDataType_t dtype, size_t vocab_size,
            float temperature, int top_k, float top_p, uint64_t seed,
            const int64_t *penalty_tokens, size_t n_penalty_tokens,
            float repetition_penalty) {
    int64_t *output_idx = reinterpret_cast<int64_t *>(output_idx_ptr);
    float *workspace = reinterpret_cast<float *>(workspace_ptr);

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return sample_impl(output_idx, reinterpret_cast<const float *>(logits_ptr),
                           workspace, vocab_size, temperature, top_k, top_p, seed,
                           penalty_tokens, n_penalty_tokens, repetition_penalty);
    case LLAISYS_DTYPE_BF16:
        return sample_impl(output_idx, reinterpret_cast<const bf16_t *>(logits_ptr),
                           workspace, vocab_size, temperature, top_k, top_p, seed,
                           penalty_tokens, n_penalty_tokens, repetition_penalty);
    case LLAISYS_DTYPE_F16:
        return sample_impl(output_idx, reinterpret_cast<const fp16_t *>(logits_ptr),
                           workspace, vocab_size, temperature, top_k, top_p, seed,
                           penalty_tokens, n_penalty_tokens, repetition_penalty);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
