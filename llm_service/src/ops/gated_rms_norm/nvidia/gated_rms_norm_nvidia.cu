#include "gated_rms_norm_nvidia.cuh"
#include "../../nvidia_common.cuh"

// out[row][i] = rms_norm(x[row], weight, eps)[i] * silu(z[row][i])
// Fused into a single kernel: one block per row
template <typename T>
__global__ void gated_rms_norm_kernel(T *out, const T *x, const T *z, const T *weight,
                                       float eps, size_t M, size_t N) {
    size_t row = blockIdx.x;
    if (row >= M) return;

    extern __shared__ float sdata[];

    const T *x_row = x + row * N;
    const T *z_row = z + row * N;
    T *out_row = out + row * N;

    // Phase 1: compute sum of squares for RMS
    float sum_sq = 0.0f;
    for (size_t i = threadIdx.x; i < N; i += blockDim.x) {
        float v = to_float(x_row[i]);
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

    // Phase 2: out = (x * rms * weight) * silu(z)
    for (size_t i = threadIdx.x; i < N; i += blockDim.x) {
        float x_val = to_float(x_row[i]);
        float w_val = to_float(weight[i]);
        float z_val = to_float(z_row[i]);
        float normed = x_val * rms * w_val;
        float silu_z = z_val / (1.0f + expf(-z_val));
        out_row[i] = from_float<T>(normed * silu_z);
    }
}

namespace llaisys::ops::nvidia {
void gated_rms_norm(std::byte *out, const std::byte *x, const std::byte *z,
                     const std::byte *weight, float eps, llaisysDataType_t dtype,
                     size_t M, size_t N) {
    int threads = 256;
    if ((int)N < threads) threads = (int)N;
    int t = 1;
    while (t < threads) t <<= 1;
    threads = t;
    if (threads > 1024) threads = 1024;

    size_t shared_size = threads * sizeof(float);

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        gated_rms_norm_kernel<<<(unsigned)M, threads, shared_size>>>(
            (float *)out, (const float *)x, (const float *)z, (const float *)weight, eps, M, N);
        break;
    case LLAISYS_DTYPE_F16:
        gated_rms_norm_kernel<<<(unsigned)M, threads, shared_size>>>(
            (__half *)out, (const __half *)x, (const __half *)z, (const __half *)weight, eps, M, N);
        break;
    case LLAISYS_DTYPE_BF16:
        gated_rms_norm_kernel<<<(unsigned)M, threads, shared_size>>>(
            (__nv_bfloat16 *)out, (const __nv_bfloat16 *)x, (const __nv_bfloat16 *)z,
            (const __nv_bfloat16 *)weight, eps, M, N);
        break;
    default:
        throw std::runtime_error("Unsupported dtype for CUDA gated_rms_norm");
    }
    CUDA_KERNEL_CHECK();
}
} // namespace llaisys::ops::nvidia
