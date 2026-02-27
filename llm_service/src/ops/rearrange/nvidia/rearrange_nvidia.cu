#include "rearrange_nvidia.cuh"
#include "../../nvidia_common.cuh"

#define MAX_DIMS 8

struct StridesInfo {
    size_t shape[MAX_DIMS];
    ptrdiff_t out_strides[MAX_DIMS];
    ptrdiff_t in_strides[MAX_DIMS];
    int ndim;
};

template <typename T>
__global__ void rearrange_kernel(T *out, const T *in, StridesInfo info, size_t numel) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    size_t remaining = idx;
    ptrdiff_t out_offset = 0;
    ptrdiff_t in_offset = 0;
    for (int d = 0; d < info.ndim; d++) {
        size_t coord = remaining / 1;
        size_t stride_prod = 1;
        for (int dd = d + 1; dd < info.ndim; dd++) {
            stride_prod *= info.shape[dd];
        }
        coord = remaining / stride_prod;
        remaining = remaining % stride_prod;
        out_offset += (ptrdiff_t)coord * info.out_strides[d];
        in_offset += (ptrdiff_t)coord * info.in_strides[d];
    }
    out[out_offset] = in[in_offset];
}

namespace llaisys::ops::nvidia {
void rearrange(std::byte *out, const std::byte *in,
               const std::vector<size_t> &shape,
               const std::vector<ptrdiff_t> &out_strides,
               const std::vector<ptrdiff_t> &in_strides,
               llaisysDataType_t dtype, size_t numel) {
    StridesInfo info;
    info.ndim = (int)shape.size();
    for (int i = 0; i < info.ndim && i < MAX_DIMS; i++) {
        info.shape[i] = shape[i];
        info.out_strides[i] = out_strides[i];
        info.in_strides[i] = in_strides[i];
    }

    int threads = 256;
    int blocks = ceil_div((int)numel, threads);

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        rearrange_kernel<<<blocks, threads>>>((float *)out, (const float *)in, info, numel);
        break;
    case LLAISYS_DTYPE_F16:
        rearrange_kernel<<<blocks, threads>>>((__half *)out, (const __half *)in, info, numel);
        break;
    case LLAISYS_DTYPE_BF16:
        rearrange_kernel<<<blocks, threads>>>((__nv_bfloat16 *)out, (const __nv_bfloat16 *)in, info, numel);
        break;
    case LLAISYS_DTYPE_I64:
        rearrange_kernel<<<blocks, threads>>>((int64_t *)out, (const int64_t *)in, info, numel);
        break;
    case LLAISYS_DTYPE_I32:
        rearrange_kernel<<<blocks, threads>>>((int32_t *)out, (const int32_t *)in, info, numel);
        break;
    default:
        throw std::runtime_error("Unsupported dtype for CUDA rearrange");
    }
    CUDA_KERNEL_CHECK();
}
} // namespace llaisys::ops::nvidia
