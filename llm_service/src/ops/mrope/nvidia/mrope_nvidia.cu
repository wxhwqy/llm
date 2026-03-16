#include "mrope_nvidia.cuh"
#include "../../nvidia_common.cuh"

// M-RoPE with interleaved layout.
// For each pair (2i, 2i+1) in the rotary region:
//   Determine which section this pair belongs to (section 0,1,2 based on cumulative sections)
//   Use pos_ids[section_idx * seq_len + s] as position
//   Compute freq = pos / theta^(2*local_i / rotary_dim)
//   Apply rotation: out[2i] = x[2i]*cos - x[2i+1]*sin, out[2i+1] = x[2i+1]*cos + x[2i]*sin
// Dimensions beyond rotary_dim pass through unchanged.
//
// sections_cumsum is precomputed on host and stored in constant memory.
__constant__ int d_sections_cumsum[4]; // [0, s0, s0+s1, s0+s1+s2]

template <typename T>
__global__ void mrope_kernel(T *out, const T *in, const int64_t *pos_ids,
                              float theta, size_t rotary_dim,
                              size_t seq_len, size_t n_head, size_t head_dim) {
    // Total threads over (seq_len, n_head, rotary_dim/2)
    size_t n_pairs = rotary_dim / 2;
    size_t total = seq_len * n_head * n_pairs;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    size_t pair_idx = idx % n_pairs;          // which pair (0..n_pairs-1)
    size_t h = (idx / n_pairs) % n_head;
    size_t s = idx / (n_pairs * n_head);

    // Determine which section this pair belongs to
    int section_idx = 0;
    if ((int)pair_idx >= d_sections_cumsum[1]) section_idx = 1;
    if ((int)pair_idx >= d_sections_cumsum[2]) section_idx = 2;

    // Local index within the section for frequency computation
    int local_i = (int)pair_idx - d_sections_cumsum[section_idx];

    // Position for this section dimension
    float pos = (float)pos_ids[section_idx * seq_len + s];
    float freq = pos / powf(theta, 2.0f * (float)local_i / (float)rotary_dim);
    float cos_v = cosf(freq);
    float sin_v = sinf(freq);

    // Interleaved layout: pair at (2*pair_idx, 2*pair_idx+1)
    size_t base = s * n_head * head_dim + h * head_dim;
    size_t idx_a = base + 2 * pair_idx;
    size_t idx_b = base + 2 * pair_idx + 1;

    float a_val = to_float(in[idx_a]);
    float b_val = to_float(in[idx_b]);

    out[idx_a] = from_float<T>(a_val * cos_v - b_val * sin_v);
    out[idx_b] = from_float<T>(b_val * cos_v + a_val * sin_v);
}

// Copy non-rotary dimensions (pass-through)
template <typename T>
__global__ void mrope_passthrough_kernel(T *out, const T *in,
                                          size_t rotary_dim,
                                          size_t seq_len, size_t n_head, size_t head_dim) {
    size_t pass_dim = head_dim - rotary_dim;
    size_t total = seq_len * n_head * pass_dim;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    size_t d = idx % pass_dim;
    size_t h = (idx / pass_dim) % n_head;
    size_t s = idx / (pass_dim * n_head);

    size_t offset = s * n_head * head_dim + h * head_dim + rotary_dim + d;
    out[offset] = in[offset];
}

namespace llaisys::ops::nvidia {
void mrope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
           float theta, const int *sections, size_t rotary_dim,
           llaisysDataType_t dtype,
           size_t seq_len, size_t n_head, size_t head_dim) {
    // Precompute cumulative sections and upload to constant memory
    int cumsum[4] = {0, sections[0], sections[0] + sections[1],
                     sections[0] + sections[1] + sections[2]};
    cudaMemcpyToSymbol(d_sections_cumsum, cumsum, sizeof(cumsum));

    int threads = 256;

    // Rotary part
    size_t n_pairs = rotary_dim / 2;
    size_t total_rot = seq_len * n_head * n_pairs;
    int blocks_rot = ceil_div((int)total_rot, threads);

    // Passthrough part
    size_t pass_dim = head_dim - rotary_dim;
    size_t total_pass = seq_len * n_head * pass_dim;
    int blocks_pass = (total_pass > 0) ? ceil_div((int)total_pass, threads) : 0;

    switch (dtype) {
    case LLAISYS_DTYPE_BF16:
        mrope_kernel<<<blocks_rot, threads>>>(
            (__nv_bfloat16 *)out, (const __nv_bfloat16 *)in, (const int64_t *)pos_ids,
            theta, rotary_dim, seq_len, n_head, head_dim);
        if (blocks_pass > 0)
            mrope_passthrough_kernel<<<blocks_pass, threads>>>(
                (__nv_bfloat16 *)out, (const __nv_bfloat16 *)in,
                rotary_dim, seq_len, n_head, head_dim);
        break;
    case LLAISYS_DTYPE_F16:
        mrope_kernel<<<blocks_rot, threads>>>(
            (__half *)out, (const __half *)in, (const int64_t *)pos_ids,
            theta, rotary_dim, seq_len, n_head, head_dim);
        if (blocks_pass > 0)
            mrope_passthrough_kernel<<<blocks_pass, threads>>>(
                (__half *)out, (const __half *)in,
                rotary_dim, seq_len, n_head, head_dim);
        break;
    case LLAISYS_DTYPE_F32:
        mrope_kernel<<<blocks_rot, threads>>>(
            (float *)out, (const float *)in, (const int64_t *)pos_ids,
            theta, rotary_dim, seq_len, n_head, head_dim);
        if (blocks_pass > 0)
            mrope_passthrough_kernel<<<blocks_pass, threads>>>(
                (float *)out, (const float *)in,
                rotary_dim, seq_len, n_head, head_dim);
        break;
    default:
        throw std::runtime_error("Unsupported dtype for CUDA mrope");
    }
    CUDA_KERNEL_CHECK();
}
} // namespace llaisys::ops::nvidia
