#include "rms_norm_nvidia.cuh"
#include "../../nvidia_common.cuh"

template <typename T>
__global__ void rms_norm_kernel(T *out, const T *in, const T *weight,
                                 float eps, size_t M, size_t N) {
    size_t row = blockIdx.x;
    if (row >= M) return;

    extern __shared__ float sdata[];

    const T *in_row = in + row * N;
    T *out_row = out + row * N;

    float sum_sq = 0.0f;
    for (size_t i = threadIdx.x; i < N; i += blockDim.x) {
        float v = to_float(in_row[i]);
        sum_sq += v * v;
    }

    sdata[threadIdx.x] = sum_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    float rms = rsqrtf(sdata[0] / (float)N + eps);

    for (size_t i = threadIdx.x; i < N; i += blockDim.x) {
        float in_val = to_float(in_row[i]);
        float w_val = to_float(weight[i]);
        out_row[i] = from_float<T>(w_val * in_val * rms);
    }
}

namespace llaisys::ops::nvidia {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              float eps, llaisysDataType_t dtype, size_t M, size_t N) {
    int threads = 256;
    if ((int)N < threads) threads = (int)N;
    // Round to next power of 2 for reduction
    int t = 1;
    while (t < threads) t <<= 1;
    threads = t;
    if (threads > 1024) threads = 1024;

    size_t shared_size = threads * sizeof(float);

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        rms_norm_kernel<<<(unsigned)M, threads, shared_size>>>((float *)out, (const float *)in, (const float *)weight, eps, M, N);
        break;
    case LLAISYS_DTYPE_F16:
        rms_norm_kernel<<<(unsigned)M, threads, shared_size>>>((__half *)out, (const __half *)in, (const __half *)weight, eps, M, N);
        break;
    case LLAISYS_DTYPE_BF16:
        rms_norm_kernel<<<(unsigned)M, threads, shared_size>>>((__nv_bfloat16 *)out, (const __nv_bfloat16 *)in, (const __nv_bfloat16 *)weight, eps, M, N);
        break;
    default:
        throw std::runtime_error("Unsupported dtype for CUDA rms_norm");
    }
    CUDA_KERNEL_CHECK();
}
} // namespace llaisys::ops::nvidia
