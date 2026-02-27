#include "embedding_nvidia.cuh"
#include "../../nvidia_common.cuh"

template <typename T>
__global__ void embedding_kernel(T *out, const int64_t *index, const T *weight,
                                  size_t seq_len, size_t hidden_size) {
    size_t row = blockIdx.x;
    size_t col = blockIdx.y * blockDim.x + threadIdx.x;
    if (row < seq_len && col < hidden_size) {
        int64_t idx = index[row];
        out[row * hidden_size + col] = weight[idx * hidden_size + col];
    }
}

namespace llaisys::ops::nvidia {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t dtype, size_t seq_len, size_t hidden_size) {
    int threads = 256;
    dim3 blocks((unsigned)seq_len, (unsigned)ceil_div((int)hidden_size, threads));

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        embedding_kernel<<<blocks, threads>>>((float *)out, (const int64_t *)index, (const float *)weight, seq_len, hidden_size);
        break;
    case LLAISYS_DTYPE_F16:
        embedding_kernel<<<blocks, threads>>>((__half *)out, (const int64_t *)index, (const __half *)weight, seq_len, hidden_size);
        break;
    case LLAISYS_DTYPE_BF16:
        embedding_kernel<<<blocks, threads>>>((__nv_bfloat16 *)out, (const int64_t *)index, (const __nv_bfloat16 *)weight, seq_len, hidden_size);
        break;
    default:
        throw std::runtime_error("Unsupported dtype for CUDA embedding");
    }
    CUDA_KERNEL_CHECK();
}
} // namespace llaisys::ops::nvidia
