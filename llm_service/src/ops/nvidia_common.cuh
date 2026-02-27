#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            throw std::runtime_error(std::string("CUDA error at ") +           \
                                     __FILE__ + ":" +                          \
                                     std::to_string(__LINE__) + ": " +         \
                                     cudaGetErrorString(err));                 \
        }                                                                      \
    } while (0)

#define CUDA_KERNEL_CHECK()                                                    \
    do {                                                                       \
        cudaError_t err = cudaGetLastError();                                  \
        if (err != cudaSuccess) {                                              \
            throw std::runtime_error(std::string("CUDA kernel error at ") +    \
                                     __FILE__ + ":" +                          \
                                     std::to_string(__LINE__) + ": " +         \
                                     cudaGetErrorString(err));                 \
        }                                                                      \
    } while (0)

__device__ __forceinline__ float to_float(__half v) { return __half2float(v); }
__device__ __forceinline__ float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float to_float(float v) { return v; }

__device__ __forceinline__ __half from_float_h(float v) { return __float2half(v); }
__device__ __forceinline__ __nv_bfloat16 from_float_bf(float v) { return __float2bfloat16(v); }

template <typename T>
__device__ __forceinline__ T from_float(float v);

template <>
__device__ __forceinline__ float from_float<float>(float v) { return v; }
template <>
__device__ __forceinline__ __half from_float<__half>(float v) { return __float2half(v); }
template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }
