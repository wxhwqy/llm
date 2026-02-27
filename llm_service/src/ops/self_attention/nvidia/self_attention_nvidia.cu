#include "self_attention_nvidia.cuh"
#include "../../nvidia_common.cuh"
#include <float.h>

template <typename T>
__global__ void self_attention_kernel(T *attn_val, const T *q, const T *k, const T *v,
                                       float scale, size_t qlen, size_t kvlen,
                                       size_t nhead, size_t nkvhead, size_t hd) {
    size_t h = blockIdx.x;
    size_t qi = blockIdx.y;
    if (h >= nhead || qi >= qlen) return;

    size_t kv_h = h / (nhead / nkvhead);

    extern __shared__ float smem[];
    float *scores = smem;

    for (size_t ki = threadIdx.x; ki < kvlen; ki += blockDim.x) {
        size_t causal_limit = qi + (kvlen - qlen);
        if (ki > causal_limit) {
            scores[ki] = -FLT_MAX;
        } else {
            float dot = 0.0f;
            for (size_t d = 0; d < hd; d++) {
                float q_val = to_float(q[qi * nhead * hd + h * hd + d]);
                float k_val = to_float(k[ki * nkvhead * hd + kv_h * hd + d]);
                dot += q_val * k_val;
            }
            scores[ki] = dot * scale;
        }
    }
    __syncthreads();

    // Find max (single thread for simplicity with small kvlen during decode)
    __shared__ float max_score;
    if (threadIdx.x == 0) {
        max_score = -FLT_MAX;
        for (size_t ki = 0; ki < kvlen; ki++) {
            if (scores[ki] > max_score) max_score = scores[ki];
        }
    }
    __syncthreads();

    // Softmax: exp and sum
    __shared__ float sum_exp;
    if (threadIdx.x == 0) {
        sum_exp = 0.0f;
        for (size_t ki = 0; ki < kvlen; ki++) {
            if (scores[ki] > -FLT_MAX / 2.0f) {
                scores[ki] = expf(scores[ki] - max_score);
                sum_exp += scores[ki];
            } else {
                scores[ki] = 0.0f;
            }
        }
        for (size_t ki = 0; ki < kvlen; ki++) {
            scores[ki] /= sum_exp;
        }
    }
    __syncthreads();

    // Weighted sum over V
    for (size_t d = threadIdx.x; d < hd; d += blockDim.x) {
        float out_val = 0.0f;
        for (size_t ki = 0; ki < kvlen; ki++) {
            if (scores[ki] > 0.0f) {
                out_val += scores[ki] * to_float(v[ki * nkvhead * hd + kv_h * hd + d]);
            }
        }
        attn_val[qi * nhead * hd + h * hd + d] = from_float<T>(out_val);
    }
}

namespace llaisys::ops::nvidia {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k,
                    const std::byte *v, float scale, llaisysDataType_t dtype,
                    size_t qlen, size_t kvlen, size_t nhead, size_t nkvhead, size_t hd) {
    dim3 grid((unsigned)nhead, (unsigned)qlen);
    int threads = (int)hd;
    if (threads < 32) threads = 32;
    if (threads > 1024) threads = 1024;
    size_t shared_size = kvlen * sizeof(float);

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        self_attention_kernel<<<grid, threads, shared_size>>>(
            (float *)attn_val, (const float *)q, (const float *)k, (const float *)v,
            scale, qlen, kvlen, nhead, nkvhead, hd);
        break;
    case LLAISYS_DTYPE_F16:
        self_attention_kernel<<<grid, threads, shared_size>>>(
            (__half *)attn_val, (const __half *)q, (const __half *)k, (const __half *)v,
            scale, qlen, kvlen, nhead, nkvhead, hd);
        break;
    case LLAISYS_DTYPE_BF16:
        self_attention_kernel<<<grid, threads, shared_size>>>(
            (__nv_bfloat16 *)attn_val, (const __nv_bfloat16 *)q, (const __nv_bfloat16 *)k,
            (const __nv_bfloat16 *)v, scale, qlen, kvlen, nhead, nkvhead, hd);
        break;
    default:
        throw std::runtime_error("Unsupported dtype for CUDA self_attention");
    }
    CUDA_KERNEL_CHECK();
}
} // namespace llaisys::ops::nvidia
