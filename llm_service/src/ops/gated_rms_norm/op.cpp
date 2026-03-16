#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/gated_rms_norm_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/gated_rms_norm_nvidia.cuh"
#endif

namespace llaisys::ops {

void gated_rms_norm(tensor_t out, tensor_t x, tensor_t z, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, x, z);
    CHECK_SAME_DTYPE(out->dtype(), x->dtype(), z->dtype());
    CHECK_SAME_SHAPE(out->shape(), x->shape(), z->shape());
    ASSERT(out->isContiguous() && x->isContiguous() && z->isContiguous() && weight->isContiguous(),
           "not contiguous");

    size_t M = out->shape()[0];
    for (size_t i = 1; i < out->ndim() - 1; i++) M *= out->shape()[i];
    size_t N = out->shape()[out->ndim() - 1];

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::gated_rms_norm(out->data(), x->data(), z->data(), weight->data(),
                                    eps, out->dtype(), M, N);
    }

#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
        return nvidia::gated_rms_norm(out->data(), x->data(), z->data(), weight->data(),
                                       eps, out->dtype(), M, N);
    }
#endif

    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
