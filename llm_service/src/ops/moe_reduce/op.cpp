#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include <cstring>

#include "cpu/moe_reduce_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/moe_reduce_nvidia.cuh"
#include <cuda_runtime.h>
#endif

namespace llaisys::ops {

void moe_accumulate(tensor_t accum, tensor_t expert_out,
                    float weight, int token_idx) {
    CHECK_SAME_DEVICE(accum, expert_out);
    ASSERT(accum->dtype() == LLAISYS_DTYPE_F32, "accum must be F32");
    ASSERT(expert_out->dtype() == LLAISYS_DTYPE_BF16, "expert_out must be BF16");

    size_t seq_len = accum->shape()[0];
    size_t hidden = accum->shape()[1];

    if (accum->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::moe_accumulate(accum->data(), expert_out->data(),
                                   weight, token_idx, seq_len, hidden);
    }
#ifdef ENABLE_NVIDIA_API
    if (accum->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        llaisys::core::context().setDevice(accum->deviceType(), accum->deviceId());
        return nvidia::moe_accumulate(accum->data(), expert_out->data(),
                                       weight, token_idx, seq_len, hidden);
    }
#endif
    EXCEPTION_UNSUPPORTED_DEVICE;
}

void moe_combine(tensor_t hidden, tensor_t residual,
                 tensor_t accum, tensor_t shared_out) {
    CHECK_SAME_DEVICE(hidden, residual, accum, shared_out);

    size_t seq_len = hidden->shape()[0];
    size_t hs = hidden->shape()[1];

    if (hidden->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::moe_combine(hidden->data(), residual->data(),
                                accum->data(), shared_out->data(),
                                seq_len, hs);
    }
#ifdef ENABLE_NVIDIA_API
    if (hidden->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        llaisys::core::context().setDevice(hidden->deviceType(), hidden->deviceId());
        return nvidia::moe_combine(hidden->data(), residual->data(),
                                    accum->data(), shared_out->data(),
                                    seq_len, hs);
    }
#endif
    EXCEPTION_UNSUPPORTED_DEVICE;
}

void moe_shared_gate(tensor_t shared_out, tensor_t normed,
                     tensor_t gate_weight) {
    CHECK_SAME_DEVICE(shared_out, normed, gate_weight);

    size_t seq_len = shared_out->shape()[0];
    size_t hs = shared_out->shape()[1];

    if (shared_out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::moe_shared_gate(shared_out->data(), normed->data(),
                                     gate_weight->data(), seq_len, hs);
    }
#ifdef ENABLE_NVIDIA_API
    if (shared_out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        llaisys::core::context().setDevice(shared_out->deviceType(), shared_out->deviceId());
        return nvidia::moe_shared_gate(shared_out->data(), normed->data(),
                                        gate_weight->data(), seq_len, hs);
    }
#endif
    EXCEPTION_UNSUPPORTED_DEVICE;
}

void moe_zero_accum(tensor_t accum) {
    ASSERT(accum->dtype() == LLAISYS_DTYPE_F32, "accum must be F32");
    size_t bytes = accum->numel() * sizeof(float);

    if (accum->deviceType() == LLAISYS_DEVICE_CPU) {
        std::memset(accum->data(), 0, bytes);
        return;
    }
#ifdef ENABLE_NVIDIA_API
    if (accum->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        llaisys::core::context().setDevice(accum->deviceType(), accum->deviceId());
        cudaMemsetAsync(accum->data(), 0, bytes);
        return;
    }
#endif
    EXCEPTION_UNSUPPORTED_DEVICE;
}

} // namespace llaisys::ops
