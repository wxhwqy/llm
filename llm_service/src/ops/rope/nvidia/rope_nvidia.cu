#include "rope_nvidia.cuh"
#include "../../nvidia_common.cuh"

template <typename T>
__global__ void rope_kernel(T *out, const T *in, const int64_t *pos_ids,
                            float theta, size_t seq_len, size_t n_head, size_t head_dim) {
    size_t half_dim = head_dim / 2;
    size_t total = seq_len * n_head * half_dim;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    size_t j = idx % half_dim;
    size_t h = (idx / half_dim) % n_head;
    size_t s = idx / (half_dim * n_head);

    float pos = (float)pos_ids[s];
    float freq = pos / powf(theta, 2.0f * (float)j / (float)head_dim);
    float cos_v = cosf(freq);
    float sin_v = sinf(freq);

    size_t idx_a = s * n_head * head_dim + h * head_dim + j;
    size_t idx_b = idx_a + half_dim;

    float a_val = to_float(in[idx_a]);
    float b_val = to_float(in[idx_b]);

    out[idx_a] = from_float<T>(a_val * cos_v - b_val * sin_v);
    out[idx_b] = from_float<T>(b_val * cos_v + a_val * sin_v);
}

namespace llaisys::ops::nvidia {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          float theta, llaisysDataType_t dtype,
          size_t seq_len, size_t n_head, size_t head_dim) {
    size_t total = seq_len * n_head * (head_dim / 2);
    int threads = 256;
    int blocks = ceil_div((int)total, threads);

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        rope_kernel<<<blocks, threads>>>((float *)out, (const float *)in,
            (const int64_t *)pos_ids, theta, seq_len, n_head, head_dim);
        break;
    case LLAISYS_DTYPE_F16:
        rope_kernel<<<blocks, threads>>>((__half *)out, (const __half *)in,
            (const int64_t *)pos_ids, theta, seq_len, n_head, head_dim);
        break;
    case LLAISYS_DTYPE_BF16:
        rope_kernel<<<blocks, threads>>>((__nv_bfloat16 *)out, (const __nv_bfloat16 *)in,
            (const int64_t *)pos_ids, theta, seq_len, n_head, head_dim);
        break;
    default:
        throw std::runtime_error("Unsupported dtype for CUDA rope");
    }
    CUDA_KERNEL_CHECK();
}
} // namespace llaisys::ops::nvidia
