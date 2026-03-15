#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/linear_nvidia.cuh"
#endif

namespace llaisys::ops {

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    if (bias) {
        CHECK_SAME_DEVICE(out, bias);
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
    }
    ASSERT(out->ndim() == 2 && in->ndim() == 2 && weight->ndim() == 2,
           "不是2维的");
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "不连续");
    if (bias) {
        ASSERT(bias->ndim() == 1 && bias->isContiguous(), "bias不是1维或者不连续");
    }

    size_t M = in->shape()[0];
    size_t K = in->shape()[1];
    size_t N = weight->shape()[0];

    ASSERT(weight->shape()[1] == K, "weight形状不对");
    ASSERT(out->shape()[0] == M && out->shape()[1] == N, "output形状不对");
    if (bias) {
        ASSERT(bias->shape()[0] == N, "bias形状不对");
    }

    std::byte *bias_ptr = bias ? bias->data() : nullptr;

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(), bias_ptr,
                           out->dtype(), M, K, N);
    }

#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
        return nvidia::linear(out->data(), in->data(), weight->data(), bias_ptr, out->dtype(), M, K, N);
    }
#endif

    EXCEPTION_UNSUPPORTED_DEVICE;
}

void linear_fp8(tensor_t out, tensor_t in,
                tensor_t weight_fp8, tensor_t scale_inv,
                size_t fp8_block_h, size_t fp8_block_w) {
#ifdef ENABLE_NVIDIA_API
    CHECK_SAME_DEVICE(out, in, weight_fp8, scale_inv);
    ASSERT(out->deviceType() == LLAISYS_DEVICE_NVIDIA, "linear_fp8 requires NVIDIA GPU");
    ASSERT(weight_fp8->dtype() == LLAISYS_DTYPE_F8, "weight_fp8 must be FP8");
    ASSERT(scale_inv->dtype() == LLAISYS_DTYPE_F32, "scale_inv must be F32");
    ASSERT(out->ndim() == 2 && in->ndim() == 2 && weight_fp8->ndim() == 2, "inputs must be 2D");
    ASSERT(out->isContiguous() && in->isContiguous() &&
           weight_fp8->isContiguous() && scale_inv->isContiguous(),
           "all tensors must be contiguous");

    size_t M = in->shape()[0];
    size_t K = in->shape()[1];
    size_t N = weight_fp8->shape()[0];
    ASSERT(weight_fp8->shape()[1] == K, "weight K dim mismatch");
    ASSERT(out->shape()[0] == M && out->shape()[1] == N, "out shape mismatch");

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    return nvidia::linear_fp8(out->data(), in->data(),
                               weight_fp8->data(), scale_inv->data(),
                               out->dtype(), M, K, N, fp8_block_h, fp8_block_w);
#else
    EXCEPTION_UNSUPPORTED_DEVICE;
#endif
}

} // namespace llaisys::ops
