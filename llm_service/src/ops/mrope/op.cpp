#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/mrope_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/mrope_nvidia.cuh"
#endif

namespace llaisys::ops {

void mrope(tensor_t out, tensor_t in, tensor_t pos_ids,
           float theta, const int *sections, size_t rotary_dim,
           size_t seq_len, size_t n_head, size_t head_dim) {
    CHECK_SAME_DEVICE(out, in);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::mrope(out->data(), in->data(), pos_ids->data(),
                          theta, sections, rotary_dim, out->dtype(),
                          seq_len, n_head, head_dim);
    }

#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
        return nvidia::mrope(out->data(), in->data(), pos_ids->data(),
                              theta, sections, rotary_dim, out->dtype(),
                              seq_len, n_head, head_dim);
    }
#endif

    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
