#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>

template <typename T>
static void embedding_impl(T *out, const int64_t *index, const T *weight,
                           size_t seq_len, size_t hidden_size) {
    for (size_t i = 0; i < seq_len; i++) {
        int64_t idx = index[i];
        const T *src = weight + idx * hidden_size;
        T *dst = out + i * hidden_size;
        std::memcpy(dst, src, hidden_size * sizeof(T));
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out_ptr, const std::byte *index_ptr, const std::byte *weight_ptr,
               llaisysDataType_t dtype, size_t seq_len, size_t hidden_size) {
    const int64_t *index = reinterpret_cast<const int64_t *>(index_ptr);

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return embedding_impl(reinterpret_cast<float *>(out_ptr), index,
                              reinterpret_cast<const float *>(weight_ptr), seq_len, hidden_size);
    case LLAISYS_DTYPE_BF16:
        return embedding_impl(reinterpret_cast<bf16_t *>(out_ptr), index,
                              reinterpret_cast<const bf16_t *>(weight_ptr), seq_len, hidden_size);
    case LLAISYS_DTYPE_F16:
        return embedding_impl(reinterpret_cast<fp16_t *>(out_ptr), index,
                              reinterpret_cast<const fp16_t *>(weight_ptr), seq_len, hidden_size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
