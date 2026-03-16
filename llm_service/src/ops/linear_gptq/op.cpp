#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_gptq_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/linear_gptq_nvidia.cuh"
#endif

namespace llaisys::ops {

void linear_gptq(tensor_t output, tensor_t input,
                 tensor_t qweight, tensor_t scales, tensor_t qzeros,
                 size_t in_features, size_t out_features,
                 int bits, int group_size) {
    CHECK_SAME_DEVICE(output, input, qweight, scales);
    if (qzeros) CHECK_SAME_DEVICE(output, qzeros);

    ASSERT(output->ndim() == 2 && input->ndim() == 2, "input/output must be 2D");
    ASSERT(qweight->ndim() == 2 && scales->ndim() == 2, "qweight/scales must be 2D");
    ASSERT(input->dtype() == LLAISYS_DTYPE_BF16, "input must be BF16");
    ASSERT(output->dtype() == LLAISYS_DTYPE_BF16, "output must be BF16");
    ASSERT(scales->dtype() == LLAISYS_DTYPE_BF16, "scales must be BF16");
    ASSERT(qweight->dtype() == LLAISYS_DTYPE_I32, "qweight must be INT32");

    size_t M = input->shape()[0];
    ASSERT(input->shape()[1] == in_features, "input K dim mismatch");
    ASSERT(output->shape()[0] == M && output->shape()[1] == out_features, "output shape mismatch");

    if (output->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear_gptq(
            output->data(), input->data(),
            qweight->data(), scales->data(),
            qzeros ? qzeros->data() : nullptr,
            M, in_features, out_features, bits, group_size);
    }

#ifdef ENABLE_NVIDIA_API
    if (output->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        llaisys::core::context().setDevice(output->deviceType(), output->deviceId());
        return nvidia::linear_gptq(
            output->data(), input->data(),
            qweight->data(), scales->data(),
            qzeros ? qzeros->data() : nullptr,
            M, in_features, out_features, bits, group_size);
    }
#endif

    EXCEPTION_UNSUPPORTED_DEVICE;
}

} // namespace llaisys::ops
