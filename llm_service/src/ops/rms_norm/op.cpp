#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/rms_norm_nvidia.cuh"
#endif

#include <cmath>

namespace llaisys::ops {

template <typename T>
void rms_norm_cpu(std::byte *out_ptr, const std::byte *in_ptr, const std::byte *weight_ptr,
                  float eps, size_t M, size_t N) {
    T *out = reinterpret_cast<T *>(out_ptr);
    const T *in = reinterpret_cast<const T *>(in_ptr);
    const T *weight = reinterpret_cast<const T *>(weight_ptr);

    for (size_t m = 0; m < M; m++) {
        float sum_sq = 0.0f;
        for (size_t n = 0; n < N; n++) {
            float val;
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                val = utils::cast<float>(in[m * N + n]);
            } else {
                val = static_cast<float>(in[m * N + n]);
            }
            sum_sq += val * val;
        }
        float rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(N) + eps);

        for (size_t n = 0; n < N; n++) {
            float in_val, w_val;
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                in_val = utils::cast<float>(in[m * N + n]);
                w_val = utils::cast<float>(weight[n]);
            } else {
                in_val = static_cast<float>(in[m * N + n]);
                w_val = static_cast<float>(weight[n]);
            }
            float result = w_val * in_val * rms;
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                out[m * N + n] = utils::cast<T>(result);
            } else {
                out[m * N + n] = static_cast<T>(result);
            }
        }
    }
}

void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    ASSERT(out->ndim() == 2 && in->ndim() == 2, "out和in不是2维的");
    ASSERT(weight->ndim() == 1, "weight不是1维的");
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "不连续");
    CHECK_SAME_SHAPE(out->shape(), in->shape());

    size_t M = in->shape()[0];
    size_t N = in->shape()[1];

    ASSERT(weight->shape()[0] == N, "weight形状不对");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        switch (out->dtype()) {
        case LLAISYS_DTYPE_F32:
            return rms_norm_cpu<float>(out->data(), in->data(), weight->data(), eps, M, N);
        case LLAISYS_DTYPE_F16:
            return rms_norm_cpu<fp16_t>(out->data(), in->data(), weight->data(), eps, M, N);
        case LLAISYS_DTYPE_BF16:
            return rms_norm_cpu<bf16_t>(out->data(), in->data(), weight->data(), eps, M, N);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
        }
    }

#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
        return nvidia::rms_norm(out->data(), in->data(), weight->data(), eps, out->dtype(), M, N);
    }
#endif

    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
