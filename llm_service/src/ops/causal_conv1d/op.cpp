#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/causal_conv1d_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/causal_conv1d_nvidia.cuh"
#endif

namespace llaisys::ops {

void causal_conv1d(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias,
                    size_t seq_len, size_t d_inner, size_t kernel_size) {
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::causal_conv1d(out->data(), in->data(), weight->data(),
                                   bias ? bias->data() : nullptr,
                                   out->dtype(), seq_len, d_inner, kernel_size);
    }
#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
        return nvidia::causal_conv1d(out->data(), in->data(), weight->data(),
                                      bias ? bias->data() : nullptr,
                                      out->dtype(), seq_len, d_inner, kernel_size);
    }
#endif
    EXCEPTION_UNSUPPORTED_DEVICE;
}

void causal_conv1d_step(tensor_t out_col, tensor_t conv_state, tensor_t in_col,
                         tensor_t weight, tensor_t bias,
                         size_t d_inner, size_t kernel_size) {
    if (out_col->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::causal_conv1d_step(out_col->data(), conv_state->data(), in_col->data(),
                                        weight->data(), bias ? bias->data() : nullptr,
                                        out_col->dtype(), d_inner, kernel_size);
    }
#ifdef ENABLE_NVIDIA_API
    if (out_col->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        llaisys::core::context().setDevice(out_col->deviceType(), out_col->deviceId());
        return nvidia::causal_conv1d_step(out_col->data(), conv_state->data(), in_col->data(),
                                           weight->data(), bias ? bias->data() : nullptr,
                                           out_col->dtype(), d_inner, kernel_size);
    }
#endif
    EXCEPTION_UNSUPPORTED_DEVICE;
}

} // namespace llaisys::ops
