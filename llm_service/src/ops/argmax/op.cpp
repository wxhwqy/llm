#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/argmax_nvidia.cuh"
#endif

#include <cmath>
#include <limits>

namespace llaisys::ops {

template <typename T>
void argmax_cpu(std::byte *max_idx_ptr, std::byte *max_val_ptr, const std::byte *vals_ptr, size_t numel) {
    int64_t *max_idx = reinterpret_cast<int64_t *>(max_idx_ptr);
    T *max_val = reinterpret_cast<T *>(max_val_ptr);
    const T *vals = reinterpret_cast<const T *>(vals_ptr);

    int64_t best_idx = 0;
    float best_val = -std::numeric_limits<float>::infinity();

    for (size_t i = 0; i < numel; i++) {
        float val;
        if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
            val = utils::cast<float>(vals[i]);
        } else {
            val = static_cast<float>(vals[i]);
        }
        if (val > best_val) {
            best_val = val;
            best_idx = static_cast<int64_t>(i);
        }
    }

    max_idx[0] = best_idx;
    if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
        max_val[0] = utils::cast<T>(best_val);
    } else {
        max_val[0] = static_cast<T>(best_val);
    }
}

void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    ASSERT(max_idx->dtype() == LLAISYS_DTYPE_I64, "不是int64");
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());
    ASSERT(vals->isContiguous(), "不连续");

    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        switch (vals->dtype()) {
        case LLAISYS_DTYPE_F32:
            return argmax_cpu<float>(max_idx->data(), max_val->data(), vals->data(), vals->numel());
        case LLAISYS_DTYPE_F16:
            return argmax_cpu<fp16_t>(max_idx->data(), max_val->data(), vals->data(), vals->numel());
        case LLAISYS_DTYPE_BF16:
            return argmax_cpu<bf16_t>(max_idx->data(), max_val->data(), vals->data(), vals->numel());
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(vals->dtype());
        }
    }

#ifdef ENABLE_NVIDIA_API
    if (vals->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());
        return nvidia::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
    }
#endif

    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
