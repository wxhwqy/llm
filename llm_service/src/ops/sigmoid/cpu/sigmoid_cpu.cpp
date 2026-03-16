#include "sigmoid_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

template <typename T>
static void sigmoid_scalar(T *out, const T *in, size_t numel) {
    for (size_t i = 0; i < numel; i++) {
        float v;
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            v = llaisys::utils::cast<float>(in[i]);
        } else {
            v = static_cast<float>(in[i]);
        }
        float result = 1.0f / (1.0f + std::exp(-v));
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            out[i] = llaisys::utils::cast<T>(result);
        } else {
            out[i] = static_cast<T>(result);
        }
    }
}

namespace llaisys::ops::cpu {
void sigmoid(std::byte *out_ptr, const std::byte *in_ptr,
            llaisysDataType_t dtype, size_t numel) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return sigmoid_scalar(reinterpret_cast<float *>(out_ptr),
                             reinterpret_cast<const float *>(in_ptr), numel);
    case LLAISYS_DTYPE_BF16:
        return sigmoid_scalar(reinterpret_cast<llaisys::bf16_t *>(out_ptr),
                             reinterpret_cast<const llaisys::bf16_t *>(in_ptr), numel);
    case LLAISYS_DTYPE_F16:
        return sigmoid_scalar(reinterpret_cast<llaisys::fp16_t *>(out_ptr),
                             reinterpret_cast<const llaisys::fp16_t *>(in_ptr), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
