#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/embedding_nvidia.cuh"
#endif

#include <cstring>

namespace llaisys::ops {

template <typename T>
void embedding_cpu(std::byte *out_ptr, const std::byte *index_ptr, const std::byte *weight_ptr,
                   size_t seq_len, size_t vocab_size, size_t hidden_size) {
    T *out = reinterpret_cast<T *>(out_ptr);
    const int64_t *index = reinterpret_cast<const int64_t *>(index_ptr);
    const T *weight = reinterpret_cast<const T *>(weight_ptr);

    for (size_t i = 0; i < seq_len; i++) {
        int64_t idx = index[i];
        const T *src = weight + idx * hidden_size;
        T *dst = out + i * hidden_size;
        std::memcpy(dst, src, hidden_size * sizeof(T));
    }
}

void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "不是int64");
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    ASSERT(index->ndim() == 1, "index不是1D");
    ASSERT(weight->ndim() == 2, "weight不是2D");
    ASSERT(out->ndim() == 2, "out不是2D");
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(),
           "不连续");

    size_t seq_len = index->shape()[0];
    size_t vocab_size = weight->shape()[0];
    size_t hidden_size = weight->shape()[1];

    ASSERT(out->shape()[0] == seq_len && out->shape()[1] == hidden_size,
           "out大小不对");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        switch (out->dtype()) {
        case LLAISYS_DTYPE_F32:
            return embedding_cpu<float>(out->data(), index->data(), weight->data(),
                                        seq_len, vocab_size, hidden_size);
        case LLAISYS_DTYPE_F16:
            return embedding_cpu<fp16_t>(out->data(), index->data(), weight->data(),
                                         seq_len, vocab_size, hidden_size);
        case LLAISYS_DTYPE_BF16:
            return embedding_cpu<bf16_t>(out->data(), index->data(), weight->data(),
                                         seq_len, vocab_size, hidden_size);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
        }
    }

#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
        return nvidia::embedding(out->data(), index->data(), weight->data(), out->dtype(), seq_len, hidden_size);
    }
#endif

    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
