#include "swiglu_nvidia.cuh"
#include "../../nvidia_common.cuh"

template <typename T>
__global__ void swiglu_kernel(T *out, const T *gate, const T *up, size_t numel) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    float g = to_float(gate[idx]);
    float u = to_float(up[idx]);
    float sigmoid = 1.0f / (1.0f + expf(-g));
    out[idx] = from_float<T>(u * g * sigmoid);
}

namespace llaisys::ops::nvidia {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t dtype, size_t numel) {
    int threads = 256;
    int blocks = ceil_div((int)numel, threads);

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        swiglu_kernel<<<blocks, threads>>>((float *)out, (const float *)gate, (const float *)up, numel);
        break;
    case LLAISYS_DTYPE_F16:
        swiglu_kernel<<<blocks, threads>>>((__half *)out, (const __half *)gate, (const __half *)up, numel);
        break;
    case LLAISYS_DTYPE_BF16:
        swiglu_kernel<<<blocks, threads>>>((__nv_bfloat16 *)out, (const __nv_bfloat16 *)gate, (const __nv_bfloat16 *)up, numel);
        break;
    default:
        throw std::runtime_error("Unsupported dtype for CUDA swiglu");
    }
    CUDA_KERNEL_CHECK();
}
} // namespace llaisys::ops::nvidia
