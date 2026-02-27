#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/dequantize_fp8_nvidia.cuh"
#endif

namespace llaisys::ops {
void dequantize_fp8(tensor_t out_bf16, tensor_t in_fp8, tensor_t scale_inv,
                    size_t block_h, size_t block_w) {
    CHECK_SAME_DEVICE(out_bf16, in_fp8, scale_inv);
    ASSERT(in_fp8->dtype() == LLAISYS_DTYPE_F8, "input must be FP8");
    ASSERT(out_bf16->dtype() == LLAISYS_DTYPE_BF16, "output must be BF16");
    ASSERT(scale_inv->dtype() == LLAISYS_DTYPE_F32, "scale_inv must be F32");
    ASSERT(in_fp8->ndim() == 2, "input must be 2D");
    ASSERT(out_bf16->ndim() == 2, "output must be 2D");
    ASSERT(in_fp8->isContiguous() && out_bf16->isContiguous() && scale_inv->isContiguous(),
           "all tensors must be contiguous");

    size_t M = in_fp8->shape()[0];
    size_t K = in_fp8->shape()[1];
    CHECK_SAME_SHAPE(out_bf16->shape(), in_fp8->shape());

#ifdef ENABLE_NVIDIA_API
    if (in_fp8->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        llaisys::core::context().setDevice(in_fp8->deviceType(), in_fp8->deviceId());
        return nvidia::dequantize_fp8(out_bf16->data(), in_fp8->data(),
                                       scale_inv->data(), M, K, block_h, block_w);
    }
#endif

    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
