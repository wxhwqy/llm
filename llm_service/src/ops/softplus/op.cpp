#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/softplus_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/softplus_nvidia.cuh"
#endif

namespace llaisys::ops {

void softplus(tensor_t out, tensor_t in) {
    CHECK_SAME_DEVICE(out, in);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    ASSERT(out->isContiguous() && in->isContiguous(), "not contiguous");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::softplus(out->data(), in->data(), out->dtype(), out->numel());
    }

#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
        return nvidia::softplus(out->data(), in->data(), out->dtype(), out->numel());
    }
#endif

    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
