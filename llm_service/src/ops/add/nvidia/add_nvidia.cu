#include "add_nvidia.cuh"
#include "../../nvidia_common.cuh"

template <typename T>
__global__ void add_kernel(T *c, const T *a, const T *b, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        c[idx] = from_float<T>(to_float(a[idx]) + to_float(b[idx]));
    }
}

namespace llaisys::ops::nvidia {
void add(std::byte *c, const std::byte *a, const std::byte *b,
         llaisysDataType_t type, size_t numel) {
    int threads = 256;
    int blocks = ceil_div((int)numel, threads);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        add_kernel<<<blocks, threads>>>((float *)c, (const float *)a, (const float *)b, numel);
        break;
    case LLAISYS_DTYPE_F16:
        add_kernel<<<blocks, threads>>>((__half *)c, (const __half *)a, (const __half *)b, numel);
        break;
    case LLAISYS_DTYPE_BF16:
        add_kernel<<<blocks, threads>>>((__nv_bfloat16 *)c, (const __nv_bfloat16 *)a, (const __nv_bfloat16 *)b, numel);
        break;
    default:
        throw std::runtime_error("Unsupported dtype for CUDA add");
    }
    CUDA_KERNEL_CHECK();
}
} // namespace llaisys::ops::nvidia
