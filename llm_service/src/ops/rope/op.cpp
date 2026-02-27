#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/rope_nvidia.cuh"
#endif

#include <cmath>

namespace llaisys::ops {

template <typename T>
void rope_cpu(std::byte *out_ptr, const std::byte *in_ptr, const std::byte *pos_ids_ptr,
              float theta, size_t seq_len, size_t n_head, size_t head_dim) {
    T *out = reinterpret_cast<T *>(out_ptr);
    const T *in = reinterpret_cast<const T *>(in_ptr);
    const int64_t *pos_ids = reinterpret_cast<const int64_t *>(pos_ids_ptr);

    size_t half_dim = head_dim / 2;

    for (size_t s = 0; s < seq_len; s++) {
        float pos = static_cast<float>(pos_ids[s]);
        for (size_t h = 0; h < n_head; h++) {
            for (size_t j = 0; j < half_dim; j++) {
                float freq = pos / std::pow(theta, 2.0f * static_cast<float>(j) / static_cast<float>(head_dim));
                float cos_val = std::cos(freq);
                float sin_val = std::sin(freq);

                size_t idx_a = s * n_head * head_dim + h * head_dim + j;
                size_t idx_b = s * n_head * head_dim + h * head_dim + j + half_dim;

                float a_val, b_val;
                if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                    a_val = utils::cast<float>(in[idx_a]);
                    b_val = utils::cast<float>(in[idx_b]);
                } else {
                    a_val = static_cast<float>(in[idx_a]);
                    b_val = static_cast<float>(in[idx_b]);
                }

                float a_out = a_val * cos_val - b_val * sin_val;
                float b_out = b_val * cos_val + a_val * sin_val;

                if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                    out[idx_a] = utils::cast<T>(a_out);
                    out[idx_b] = utils::cast<T>(b_out);
                } else {
                    out[idx_a] = static_cast<T>(a_out);
                    out[idx_b] = static_cast<T>(b_out);
                }
            }
        }
    }
}

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "pos_ids不是int64");
    ASSERT(out->ndim() == 3 && in->ndim() == 3, "out和in不是3维的");
    ASSERT(pos_ids->ndim() == 1, "pos_ids不是1维的");
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
           "不连续");
    CHECK_SAME_SHAPE(out->shape(), in->shape());

    size_t seq_len = in->shape()[0];
    size_t n_head = in->shape()[1];
    size_t head_dim = in->shape()[2];

    ASSERT(pos_ids->shape()[0] == seq_len, "pos_ids长度不对");
    ASSERT(head_dim % 2 == 0, "head_dim不是偶数");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        switch (out->dtype()) {
        case LLAISYS_DTYPE_F32:
            return rope_cpu<float>(out->data(), in->data(), pos_ids->data(), theta, seq_len, n_head, head_dim);
        case LLAISYS_DTYPE_F16:
            return rope_cpu<fp16_t>(out->data(), in->data(), pos_ids->data(), theta, seq_len, n_head, head_dim);
        case LLAISYS_DTYPE_BF16:
            return rope_cpu<bf16_t>(out->data(), in->data(), pos_ids->data(), theta, seq_len, n_head, head_dim);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
        }
    }

#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
        return nvidia::rope(out->data(), in->data(), pos_ids->data(), theta, out->dtype(), seq_len, n_head, head_dim);
    }
#endif

    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
