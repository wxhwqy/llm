#include "sample_nvidia.cuh"
#include "../../nvidia_common.cuh"
#include <float.h>

#define BLOCK_SIZE 256

__device__ uint64_t xorshift64(uint64_t x) {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return x;
}

__device__ float u64_to_float01(uint64_t x) {
    return (x >> 11) * (1.0f / 9007199254740992.0f);
}

__device__ float block_reduce_max(float val, float *shared) {
    int tid = threadIdx.x;
    shared[tid] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] = fmaxf(shared[tid], shared[tid + s]);
        __syncthreads();
    }
    return shared[0];
}

__device__ float block_reduce_sum(float val, float *shared) {
    int tid = threadIdx.x;
    shared[tid] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    return shared[0];
}

__device__ int block_reduce_sum_int(int val, int *shared) {
    int tid = threadIdx.x;
    shared[tid] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    return shared[0];
}

template <typename T>
__global__ void sample_kernel(
    int64_t *output_idx,
    const T *logits,
    float *probs,
    size_t vocab_size,
    float temperature,
    int top_k,
    float top_p,
    uint64_t seed,
    const int64_t *penalty_tokens,
    size_t n_penalty_tokens,
    float repetition_penalty
) {
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    __shared__ float s_f[BLOCK_SIZE];
    __shared__ int   s_i[BLOCK_SIZE];

    // Phase 1: temperature scaling + stable softmax (find max)
    float inv_temp = 1.0f / temperature;
    float local_max = -FLT_MAX;
    for (size_t i = tid; i < vocab_size; i += nthreads) {
        float v = to_float(logits[i]) * inv_temp;
        probs[i] = v;
        local_max = fmaxf(local_max, v);
    }
    float global_max = block_reduce_max(local_max, s_f);

    // Apply repetition penalty after temperature scaling, before softmax
    if (repetition_penalty != 1.0f && n_penalty_tokens > 0) {
        for (size_t i = tid; i < n_penalty_tokens; i += nthreads) {
            int64_t token_id = penalty_tokens[i];
            if (token_id >= 0 && (size_t)token_id < vocab_size) {
                float &logit = probs[token_id];
                if (logit > 0.0f) {
                    logit /= repetition_penalty;
                } else {
                    logit *= repetition_penalty;
                }
            }
        }
        __syncthreads();
        // Recompute max after penalty
        local_max = -FLT_MAX;
        for (size_t i = tid; i < vocab_size; i += nthreads) {
            local_max = fmaxf(local_max, probs[i]);
        }
        global_max = block_reduce_max(local_max, s_f);
    }

    // Phase 2: exp + sum
    float local_sum = 0.0f;
    for (size_t i = tid; i < vocab_size; i += nthreads) {
        float e = expf(probs[i] - global_max);
        probs[i] = e;
        local_sum += e;
    }
    float global_sum = block_reduce_sum(local_sum, s_f);

    // Phase 3: normalize to probabilities
    float inv_sum = 1.0f / global_sum;
    for (size_t i = tid; i < vocab_size; i += nthreads) {
        probs[i] *= inv_sum;
    }
    __syncthreads();

    // Phase 4: Top-K filtering via binary search on probability threshold
    if (top_k > 0 && (size_t)top_k < vocab_size) {
        float local_max_p = 0.0f;
        for (size_t i = tid; i < vocab_size; i += nthreads) {
            local_max_p = fmaxf(local_max_p, probs[i]);
        }
        float hi = block_reduce_max(local_max_p, s_f);
        float lo = 0.0f;

        for (int iter = 0; iter < 40; iter++) {
            float mid = (lo + hi) * 0.5f;
            int local_count = 0;
            for (size_t i = tid; i < vocab_size; i += nthreads) {
                if (probs[i] > mid) local_count++;
            }
            int total_count = block_reduce_sum_int(local_count, s_i);
            if (total_count > top_k) lo = mid;
            else hi = mid;
        }

        float threshold = lo;
        for (size_t i = tid; i < vocab_size; i += nthreads) {
            if (probs[i] < threshold) probs[i] = 0.0f;
        }
        __syncthreads();
    }

    // Phase 5: Top-P (nucleus) filtering via binary search on cumulative sum
    if (top_p < 1.0f && top_p > 0.0f) {
        float local_max_p = 0.0f;
        for (size_t i = tid; i < vocab_size; i += nthreads) {
            local_max_p = fmaxf(local_max_p, probs[i]);
        }
        float hi = block_reduce_max(local_max_p, s_f);
        float lo = 0.0f;

        for (int iter = 0; iter < 40; iter++) {
            float mid = (lo + hi) * 0.5f;
            float local_sum_above = 0.0f;
            for (size_t i = tid; i < vocab_size; i += nthreads) {
                if (probs[i] >= mid) local_sum_above += probs[i];
            }
            float total_sum_above = block_reduce_sum(local_sum_above, s_f);
            if (total_sum_above > top_p) lo = mid;
            else hi = mid;
        }

        float threshold = hi;
        for (size_t i = tid; i < vocab_size; i += nthreads) {
            if (probs[i] < threshold) probs[i] = 0.0f;
        }
        __syncthreads();
    }

    // Phase 6: renormalize
    local_sum = 0.0f;
    for (size_t i = tid; i < vocab_size; i += nthreads) {
        local_sum += probs[i];
    }
    float final_sum = block_reduce_sum(local_sum, s_f);

    // Phase 7: sample from CDF (thread 0 only)
    if (tid == 0) {
        uint64_t rng = xorshift64(seed ^ 0x9E3779B97F4A7C15ULL);
        rng = xorshift64(rng);
        float r = u64_to_float01(rng) * final_sum;

        float cumsum = 0.0f;
        for (size_t i = 0; i < vocab_size; i++) {
            cumsum += probs[i];
            if (cumsum >= r) {
                output_idx[0] = (int64_t)i;
                return;
            }
        }
        output_idx[0] = (int64_t)(vocab_size - 1);
    }
}

namespace llaisys::ops::nvidia {
void sample(std::byte *output_idx, const std::byte *logits, std::byte *workspace,
            llaisysDataType_t dtype, size_t vocab_size,
            float temperature, int top_k, float top_p, uint64_t seed,
            const int64_t *penalty_tokens_host, size_t n_penalty_tokens,
            float repetition_penalty) {
    float *probs = reinterpret_cast<float *>(workspace);

    // Copy penalty tokens to device if needed
    int64_t *d_penalty = nullptr;
    if (repetition_penalty != 1.0f && n_penalty_tokens > 0 && penalty_tokens_host) {
        cudaMalloc(&d_penalty, n_penalty_tokens * sizeof(int64_t));
        cudaMemcpy(d_penalty, penalty_tokens_host,
                   n_penalty_tokens * sizeof(int64_t), cudaMemcpyHostToDevice);
    }

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        sample_kernel<<<1, BLOCK_SIZE>>>(
            (int64_t *)output_idx, (const float *)logits, probs,
            vocab_size, temperature, top_k, top_p, seed,
            d_penalty, n_penalty_tokens, repetition_penalty);
        break;
    case LLAISYS_DTYPE_F16:
        sample_kernel<<<1, BLOCK_SIZE>>>(
            (int64_t *)output_idx, (const __half *)logits, probs,
            vocab_size, temperature, top_k, top_p, seed,
            d_penalty, n_penalty_tokens, repetition_penalty);
        break;
    case LLAISYS_DTYPE_BF16:
        sample_kernel<<<1, BLOCK_SIZE>>>(
            (int64_t *)output_idx, (const __nv_bfloat16 *)logits, probs,
            vocab_size, temperature, top_k, top_p, seed,
            d_penalty, n_penalty_tokens, repetition_penalty);
        break;
    default:
        if (d_penalty) cudaFree(d_penalty);
        throw std::runtime_error("Unsupported dtype for CUDA sample");
    }
    CUDA_KERNEL_CHECK();
    if (d_penalty) cudaFree(d_penalty);
}
} // namespace llaisys::ops::nvidia
