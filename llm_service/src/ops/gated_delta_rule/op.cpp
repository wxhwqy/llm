#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/gated_delta_rule_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/gated_delta_rule_nvidia.cuh"
#endif

namespace llaisys::ops {

void gated_delta_rule_recurrent(
    tensor_t out, tensor_t state,
    tensor_t q, tensor_t k, tensor_t v,
    tensor_t g, tensor_t beta,
    size_t n_heads, size_t dk, size_t dv) {

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::gated_delta_rule_recurrent(
            out->data(), state->data(), q->data(), k->data(), v->data(),
            g->data(), beta->data(), out->dtype(), n_heads, dk, dv);
    }
#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
        return nvidia::gated_delta_rule_recurrent(
            out->data(), state->data(), q->data(), k->data(), v->data(),
            g->data(), beta->data(), out->dtype(), n_heads, dk, dv);
    }
#endif
    EXCEPTION_UNSUPPORTED_DEVICE;
}

void gated_delta_rule_chunk(
    tensor_t out, tensor_t final_state,
    tensor_t q, tensor_t k, tensor_t v,
    tensor_t g, tensor_t beta,
    size_t seq_len, size_t n_heads, size_t dk, size_t dv) {

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::gated_delta_rule_chunk(
            out->data(), final_state->data(), q->data(), k->data(), v->data(),
            g->data(), beta->data(), out->dtype(), seq_len, n_heads, dk, dv);
    }
#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
        return nvidia::gated_delta_rule_chunk(
            out->data(), final_state->data(), q->data(), k->data(), v->data(),
            g->data(), beta->data(), out->dtype(), seq_len, n_heads, dk, dv);
    }
#endif
    EXCEPTION_UNSUPPORTED_DEVICE;
}

} // namespace llaisys::ops
