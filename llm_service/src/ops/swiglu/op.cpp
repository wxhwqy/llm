#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/swiglu_nvidia.cuh"
#endif

#include <cmath>

namespace llaisys::ops {

template <typename T>
void swiglu_cpu(std::byte *out_ptr, const std::byte *gate_ptr, const std::byte *up_ptr, size_t numel) {
    T *out = reinterpret_cast<T *>(out_ptr);
    const T *gate = reinterpret_cast<const T *>(gate_ptr);
    const T *up = reinterpret_cast<const T *>(up_ptr);

    for (size_t i = 0; i < numel; i++) {
        float gate_val, up_val;
        if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
            gate_val = utils::cast<float>(gate[i]);
            up_val = utils::cast<float>(up[i]);
        } else {
            gate_val = static_cast<float>(gate[i]);
            up_val = static_cast<float>(up[i]);
        }

        float sigmoid = 1.0f / (1.0f + std::exp(-gate_val));
        float silu = gate_val * sigmoid;
        float result = up_val * silu;

        if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
            out[i] = utils::cast<T>(result);
        } else {
            out[i] = static_cast<T>(result);
        }
    }
}

void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());
    CHECK_SAME_SHAPE(out->shape(), gate->shape(), up->shape());
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(),
           "不连续");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        switch (out->dtype()) {
        case LLAISYS_DTYPE_F32:
            return swiglu_cpu<float>(out->data(), gate->data(), up->data(), out->numel());
        case LLAISYS_DTYPE_F16:
            return swiglu_cpu<fp16_t>(out->data(), gate->data(), up->data(), out->numel());
        case LLAISYS_DTYPE_BF16:
            return swiglu_cpu<bf16_t>(out->data(), gate->data(), up->data(), out->numel());
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
        }
    }

#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
        return nvidia::swiglu(out->data(), gate->data(), up->data(), out->dtype(), out->numel());
    }
#endif

    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
