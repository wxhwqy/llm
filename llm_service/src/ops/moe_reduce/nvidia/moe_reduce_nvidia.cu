#include "moe_reduce_nvidia.cuh"
#include "../../nvidia_common.cuh"

// ---------------------------------------------------------------------------
// moe_accumulate: accum[F32] += weight * expert_out[BF16]
// ---------------------------------------------------------------------------
__global__ void moe_accumulate_kernel(
    float *__restrict__ accum,
    const __nv_bfloat16 *__restrict__ expert_out,
    float weight,
    size_t total_elems)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elems) return;
    accum[idx] += weight * __bfloat162float(expert_out[idx]);
}

__global__ void moe_accumulate_single_token_kernel(
    float *__restrict__ accum,          // offset to token row
    const __nv_bfloat16 *__restrict__ expert_out, // offset to token row
    float weight,
    size_t hidden)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hidden) return;
    accum[idx] += weight * __bfloat162float(expert_out[idx]);
}

// ---------------------------------------------------------------------------
// moe_combine: hidden[BF16] = residual[BF16] + accum[F32] + shared_out[BF16]
// ---------------------------------------------------------------------------
__global__ void moe_combine_kernel(
    __nv_bfloat16 *__restrict__ hidden,
    const __nv_bfloat16 *__restrict__ residual,
    const float *__restrict__ accum,
    const __nv_bfloat16 *__restrict__ shared_out,
    size_t total_elems)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elems) return;
    float r = __bfloat162float(residual[idx]);
    float a = accum[idx];
    float s = __bfloat162float(shared_out[idx]);
    hidden[idx] = __float2bfloat16(r + a + s);
}

// ---------------------------------------------------------------------------
// moe_shared_gate: shared_out *= sigmoid(gate_weight . normed[t])
//
// Each block handles one token. Threads collaborate to compute the dot
// product (reduction), then each thread applies the gate to its elements.
// ---------------------------------------------------------------------------
constexpr int GATE_BLOCK = 256;

__global__ void moe_shared_gate_kernel(
    __nv_bfloat16 *__restrict__ shared_out,    // [seq_len, hidden_size]
    const __nv_bfloat16 *__restrict__ normed,  // [seq_len, hidden_size]
    const __nv_bfloat16 *__restrict__ gate_w,  // [hidden_size]
    size_t hidden_size)
{
    size_t t = blockIdx.x;  // token index

    const __nv_bfloat16 *norm_row = normed + t * hidden_size;
    __nv_bfloat16 *sout_row = shared_out + t * hidden_size;

    // Phase 1: compute dot product gate_weight . normed[t]
    float dot = 0.0f;
    for (size_t d = threadIdx.x; d < hidden_size; d += GATE_BLOCK) {
        dot += __bfloat162float(gate_w[d]) * __bfloat162float(norm_row[d]);
    }

    // Warp reduction
    for (int off = warpSize / 2; off > 0; off >>= 1)
        dot += __shfl_down_sync(0xFFFFFFFF, dot, off);

    __shared__ float warp_buf[GATE_BLOCK / 32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) warp_buf[warp] = dot;
    __syncthreads();

    // Final reduction in warp 0
    if (warp == 0) {
        constexpr int nwarps = GATE_BLOCK / 32;
        dot = (lane < nwarps) ? warp_buf[lane] : 0.0f;
        for (int off = warpSize / 2; off > 0; off >>= 1)
            dot += __shfl_down_sync(0xFFFFFFFF, dot, off);
    }

    // Broadcast gate value
    __shared__ float gate_val;
    if (threadIdx.x == 0) {
        gate_val = 1.0f / (1.0f + expf(-dot));
    }
    __syncthreads();

    // Phase 2: scale shared_out
    float g = gate_val;
    for (size_t d = threadIdx.x; d < hidden_size; d += GATE_BLOCK) {
        float v = __bfloat162float(sout_row[d]) * g;
        sout_row[d] = __float2bfloat16(v);
    }
}

namespace llaisys::ops::nvidia {

void moe_accumulate(std::byte *accum, const std::byte *expert_out,
                    float weight, int token_idx,
                    size_t seq_len, size_t hidden) {
    int threads = 256;

    if (token_idx >= 0) {
        int blocks = ceil_div((int)hidden, threads);
        moe_accumulate_single_token_kernel<<<blocks, threads>>>(
            (float*)accum + token_idx * hidden,
            (const __nv_bfloat16*)expert_out + token_idx * hidden,
            weight, hidden);
    } else {
        size_t total = seq_len * hidden;
        int blocks = ceil_div((int)total, threads);
        moe_accumulate_kernel<<<blocks, threads>>>(
            (float*)accum,
            (const __nv_bfloat16*)expert_out,
            weight, total);
    }
    CUDA_KERNEL_CHECK();
}

void moe_combine(std::byte *hidden, const std::byte *residual,
                 const std::byte *accum, const std::byte *shared_out,
                 size_t seq_len, size_t hidden_size) {
    size_t total = seq_len * hidden_size;
    int threads = 256;
    int blocks = ceil_div((int)total, threads);
    moe_combine_kernel<<<blocks, threads>>>(
        (__nv_bfloat16*)hidden,
        (const __nv_bfloat16*)residual,
        (const float*)accum,
        (const __nv_bfloat16*)shared_out,
        total);
    CUDA_KERNEL_CHECK();
}

void moe_shared_gate(std::byte *shared_out, const std::byte *normed,
                     const std::byte *gate_weight,
                     size_t seq_len, size_t hidden_size) {
    moe_shared_gate_kernel<<<(int)seq_len, GATE_BLOCK>>>(
        (__nv_bfloat16*)shared_out,
        (const __nv_bfloat16*)normed,
        (const __nv_bfloat16*)gate_weight,
        hidden_size);
    CUDA_KERNEL_CHECK();
}

} // namespace llaisys::ops::nvidia
