#include "softplus_nvidia.cuh"
#include "../../nvidia_common.cuh"

template <typename T>
__global__ void softplus_kernel(T *out, const T *in, size_t numel) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    float v = to_float(in[idx]);
    // Numerically stable: for large v, softplus(v) ≈ v
    float result = (v > 20.0f) ? v : logf(1.0f + expf(v));
    out[idx] = from_float<T>(result);
}

namespace llaisys::ops::nvidia {
void softplus(std::byte *out, const std::byte *in,
             llaisysDataType_t dtype, size_t numel) {
    int threads = 256;
    int blocks = ceil_div((int)numel, threads);

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        softplus_kernel<<<blocks, threads>>>((float *)out, (const float *)in, numel);
        break;
    case LLAISYS_DTYPE_F16:
        softplus_kernel<<<blocks, threads>>>((__half *)out, (const __half *)in, numel);
        break;
    case LLAISYS_DTYPE_BF16:
        softplus_kernel<<<blocks, threads>>>((__nv_bfloat16 *)out, (const __nv_bfloat16 *)in, numel);
        break;
    default:
        throw std::runtime_error("Unsupported dtype for CUDA softplus");
    }
    CUDA_KERNEL_CHECK();
}
} // namespace llaisys::ops::nvidia
