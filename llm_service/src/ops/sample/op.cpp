#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/sample_nvidia.cuh"
#endif

namespace llaisys::ops {

void sample(tensor_t output_idx, tensor_t logits, tensor_t workspace,
            float temperature, int top_k, float top_p, uint64_t seed) {
    ASSERT(output_idx->dtype() == LLAISYS_DTYPE_I64, "output_idx must be int64");
    ASSERT(workspace->dtype() == LLAISYS_DTYPE_F32, "workspace must be float32");
    ASSERT(logits->isContiguous(), "logits must be contiguous");
    ASSERT(workspace->numel() >= logits->numel(), "workspace too small");

    size_t vocab_size = logits->numel();

    if (temperature <= 0.0f) temperature = 1e-6f;
    if (top_k < 0) top_k = 0;
    if (top_p <= 0.0f) top_p = 1.0f;
    if (top_p > 1.0f) top_p = 1.0f;

#ifdef ENABLE_NVIDIA_API
    if (logits->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        llaisys::core::context().setDevice(logits->deviceType(), logits->deviceId());
        return nvidia::sample(output_idx->data(), logits->data(), workspace->data(),
                              logits->dtype(), vocab_size,
                              temperature, top_k, top_p, seed);
    }
#endif

    EXCEPTION_UNSUPPORTED_DEVICE;
}

} // namespace llaisys::ops
