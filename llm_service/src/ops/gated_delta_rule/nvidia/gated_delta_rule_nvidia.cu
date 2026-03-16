#include "gated_delta_rule_nvidia.cuh"
#include "../../nvidia_common.cuh"

// Recurrent decode: one block per head
// state [n_heads, dv, dk] F32
// q [n_heads, dk] compute_dtype, k [n_heads, dk], v [n_heads, dv]
// g [n_heads] F32, beta [n_heads] F32
// out [n_heads, dv] compute_dtype
template <typename T>
__global__ void gated_delta_rule_recurrent_kernel(
    T *out, float *state,
    const T *q, const T *k, const T *v,
    const float *g, const float *beta,
    size_t n_heads, size_t dk, size_t dv) {
    size_t h = blockIdx.x;
    if (h >= n_heads) return;

    float decay = g[h];
    float b = beta[h];
    float *S = state + h * dv * dk;  // [dv, dk]
    const T *q_h = q + h * dk;
    const T *k_h = k + h * dk;
    const T *v_h = v + h * dv;

    // Step 1: Decay state: S = decay * S
    for (size_t i = threadIdx.x; i < dv * dk; i += blockDim.x) {
        S[i] *= decay;
    }
    __syncthreads();

    // Step 2-4: For each dv dimension, retrieve, compute delta, update
    // Must be sequential over di since retrieval needs consistent S
    for (size_t di = 0; di < dv; di++) {
        // Step 2: retrieved = sum_j S[di, j] * k[j]
        // Parallel reduction over dk
        __shared__ float s_partial[256];
        float local_sum = 0.0f;
        for (size_t dj = threadIdx.x; dj < dk; dj += blockDim.x) {
            local_sum += S[di * dk + dj] * to_float(k_h[dj]);
        }
        s_partial[threadIdx.x] = local_sum;
        __syncthreads();

        // Block reduction
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < (unsigned)stride) {
                s_partial[threadIdx.x] += s_partial[threadIdx.x + stride];
            }
            __syncthreads();
        }
        float retrieved = s_partial[0];

        // Step 3-4: delta = beta * (v[di] - retrieved), S[di,:] += delta * k[:]
        float delta = b * (to_float(v_h[di]) - retrieved);
        for (size_t dj = threadIdx.x; dj < dk; dj += blockDim.x) {
            S[di * dk + dj] += delta * to_float(k_h[dj]);
        }
        __syncthreads();
    }

    // Output: o = S @ q  → out[h, di] = sum_j S[di, j] * q[j]
    for (size_t di = threadIdx.x; di < dv; di += blockDim.x) {
        float acc = 0.0f;
        for (size_t dj = 0; dj < dk; dj++) {
            acc += S[di * dk + dj] * to_float(q_h[dj]);
        }
        out[h * dv + di] = from_float<T>(acc);
    }
}

// Prefill: sequential over timesteps, one block per head
// Uses proper delta rule: decay → retrieve → delta → update per timestep.
template <typename T>
__global__ void gated_delta_rule_chunk_kernel(
    T *out, float *final_state,
    const T *q, const T *k, const T *v,
    const float *g, const float *beta,
    size_t seq_len, size_t n_heads, size_t dk, size_t dv) {
    size_t h = blockIdx.x;
    if (h >= n_heads) return;

    float *S = final_state + h * dv * dk;  // [dv, dk]

    // Zero-initialize state
    for (size_t i = threadIdx.x; i < dv * dk; i += blockDim.x) {
        S[i] = 0.0f;
    }
    __syncthreads();

    for (size_t t = 0; t < seq_len; t++) {
        float decay = g[t * n_heads + h];
        float b = beta[t * n_heads + h];
        const T *q_t = q + t * n_heads * dk + h * dk;
        const T *k_t = k + t * n_heads * dk + h * dk;
        const T *v_t = v + t * n_heads * dv + h * dv;

        // Step 1: Decay state: S = decay * S
        for (size_t i = threadIdx.x; i < dv * dk; i += blockDim.x) {
            S[i] *= decay;
        }
        __syncthreads();

        // Step 2-4: For each dv dimension, retrieve, compute delta, update
        for (size_t di = 0; di < dv; di++) {
            // Step 2: retrieved = sum_j S[di, j] * k[j]
            __shared__ float s_partial[256];
            float local_sum = 0.0f;
            for (size_t dj = threadIdx.x; dj < dk; dj += blockDim.x) {
                local_sum += S[di * dk + dj] * to_float(k_t[dj]);
            }
            s_partial[threadIdx.x] = local_sum;
            __syncthreads();

            // Block reduction
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (threadIdx.x < (unsigned)stride) {
                    s_partial[threadIdx.x] += s_partial[threadIdx.x + stride];
                }
                __syncthreads();
            }
            float retrieved = s_partial[0];

            // Step 3-4: delta = beta * (v[di] - retrieved), S[di,:] += delta * k[:]
            float delta = b * (to_float(v_t[di]) - retrieved);
            for (size_t dj = threadIdx.x; dj < dk; dj += blockDim.x) {
                S[di * dk + dj] += delta * to_float(k_t[dj]);
            }
            __syncthreads();
        }

        // Output: o_t = S @ q_t
        for (size_t di = threadIdx.x; di < dv; di += blockDim.x) {
            float acc = 0.0f;
            for (size_t dj = 0; dj < dk; dj++) {
                acc += S[di * dk + dj] * to_float(q_t[dj]);
            }
            out[t * n_heads * dv + h * dv + di] = from_float<T>(acc);
        }
        __syncthreads();
    }
}

namespace llaisys::ops::nvidia {

void gated_delta_rule_recurrent(
    std::byte *out, std::byte *state,
    const std::byte *q, const std::byte *k, const std::byte *v,
    const std::byte *g, const std::byte *beta,
    llaisysDataType_t dtype, size_t n_heads, size_t dk, size_t dv) {
    int threads = 256;

    switch (dtype) {
    case LLAISYS_DTYPE_BF16:
        gated_delta_rule_recurrent_kernel<<<(unsigned)n_heads, threads>>>(
            (__nv_bfloat16 *)out, (float *)state,
            (const __nv_bfloat16 *)q, (const __nv_bfloat16 *)k, (const __nv_bfloat16 *)v,
            (const float *)g, (const float *)beta, n_heads, dk, dv);
        break;
    case LLAISYS_DTYPE_F16:
        gated_delta_rule_recurrent_kernel<<<(unsigned)n_heads, threads>>>(
            (__half *)out, (float *)state,
            (const __half *)q, (const __half *)k, (const __half *)v,
            (const float *)g, (const float *)beta, n_heads, dk, dv);
        break;
    case LLAISYS_DTYPE_F32:
        gated_delta_rule_recurrent_kernel<<<(unsigned)n_heads, threads>>>(
            (float *)out, (float *)state,
            (const float *)q, (const float *)k, (const float *)v,
            (const float *)g, (const float *)beta, n_heads, dk, dv);
        break;
    default:
        throw std::runtime_error("Unsupported dtype for gated_delta_rule_recurrent");
    }
    CUDA_KERNEL_CHECK();
}

void gated_delta_rule_chunk(
    std::byte *out, std::byte *final_state,
    const std::byte *q, const std::byte *k, const std::byte *v,
    const std::byte *g, const std::byte *beta,
    llaisysDataType_t dtype, size_t seq_len, size_t n_heads, size_t dk, size_t dv) {
    int threads = 256;

    switch (dtype) {
    case LLAISYS_DTYPE_BF16:
        gated_delta_rule_chunk_kernel<<<(unsigned)n_heads, threads>>>(
            (__nv_bfloat16 *)out, (float *)final_state,
            (const __nv_bfloat16 *)q, (const __nv_bfloat16 *)k, (const __nv_bfloat16 *)v,
            (const float *)g, (const float *)beta, seq_len, n_heads, dk, dv);
        break;
    case LLAISYS_DTYPE_F16:
        gated_delta_rule_chunk_kernel<<<(unsigned)n_heads, threads>>>(
            (__half *)out, (float *)final_state,
            (const __half *)q, (const __half *)k, (const __half *)v,
            (const float *)g, (const float *)beta, seq_len, n_heads, dk, dv);
        break;
    case LLAISYS_DTYPE_F32:
        gated_delta_rule_chunk_kernel<<<(unsigned)n_heads, threads>>>(
            (float *)out, (float *)final_state,
            (const float *)q, (const float *)k, (const float *)v,
            (const float *)g, (const float *)beta, seq_len, n_heads, dk, dv);
        break;
    default:
        throw std::runtime_error("Unsupported dtype for gated_delta_rule_chunk");
    }
    CUDA_KERNEL_CHECK();
}

} // namespace llaisys::ops::nvidia
