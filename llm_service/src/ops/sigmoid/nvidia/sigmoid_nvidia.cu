#include "sigmoid_nvidia.cuh"
#include "../../nvidia_common.cuh"

template <typename T>
__global__ void sigmoid_kernel(T *out, const T *in, size_t numel) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    float v = to_float(in[idx]);
    out[idx] = from_float<T>(1.0f / (1.0f + expf(-v)));
}

namespace llaisys::ops::nvidia {
void sigmoid(std::byte *out, const std::byte *in,
            llaisysDataType_t dtype, size_t numel) {
    int threads = 256;
    int blocks = ceil_div((int)numel, threads);

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        sigmoid_kernel<<<blocks, threads>>>((float *)out, (const float *)in, numel);
        break;
    case LLAISYS_DTYPE_F16:
        sigmoid_kernel<<<blocks, threads>>>((__half *)out, (const __half *)in, numel);
        break;
    case LLAISYS_DTYPE_BF16:
        sigmoid_kernel<<<blocks, threads>>>((__nv_bfloat16 *)out, (const __nv_bfloat16 *)in, numel);
        break;
    default:
        throw std::runtime_error("Unsupported dtype for CUDA sigmoid");
    }
    CUDA_KERNEL_CHECK();
}
} // namespace llaisys::ops::nvidia
