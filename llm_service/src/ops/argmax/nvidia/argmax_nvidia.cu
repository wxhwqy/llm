#include "argmax_nvidia.cuh"
#include "../../nvidia_common.cuh"
#include <float.h>

template <typename T>
__global__ void argmax_kernel(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    __shared__ float shared_vals[256];
    __shared__ int64_t shared_idxs[256];

    int tid = threadIdx.x;
    float best_val = -FLT_MAX;
    int64_t best_idx = 0;

    for (size_t i = tid; i < numel; i += blockDim.x) {
        float v = to_float(vals[i]);
        if (v > best_val) {
            best_val = v;
            best_idx = (int64_t)i;
        }
    }

    shared_vals[tid] = best_val;
    shared_idxs[tid] = best_idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && shared_vals[tid + s] > shared_vals[tid]) {
            shared_vals[tid] = shared_vals[tid + s];
            shared_idxs[tid] = shared_idxs[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        max_idx[0] = shared_idxs[0];
        max_val[0] = from_float<T>(shared_vals[0]);
    }
}

namespace llaisys::ops::nvidia {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals,
            llaisysDataType_t dtype, size_t numel) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        argmax_kernel<<<1, 256>>>((int64_t *)max_idx, (float *)max_val, (const float *)vals, numel);
        break;
    case LLAISYS_DTYPE_F16:
        argmax_kernel<<<1, 256>>>((int64_t *)max_idx, (__half *)max_val, (const __half *)vals, numel);
        break;
    case LLAISYS_DTYPE_BF16:
        argmax_kernel<<<1, 256>>>((int64_t *)max_idx, (__nv_bfloat16 *)max_val, (const __nv_bfloat16 *)vals, numel);
        break;
    default:
        throw std::runtime_error("Unsupported dtype for CUDA argmax");
    }
    CUDA_KERNEL_CHECK();
}
} // namespace llaisys::ops::nvidia
