#include "mrope_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

template <typename T>
static void mrope_scalar(T *out, const T *in, const int64_t *pos_ids,
                          float theta, const int *sections, size_t rotary_dim,
                          size_t seq_len, size_t n_head, size_t head_dim) {
    int cumsum[4] = {0, sections[0], sections[0] + sections[1],
                     sections[0] + sections[1] + sections[2]};
    size_t n_pairs = rotary_dim / 2;

    for (size_t s = 0; s < seq_len; s++) {
        for (size_t h = 0; h < n_head; h++) {
            size_t base = s * n_head * head_dim + h * head_dim;
            // Rotary part
            for (size_t p = 0; p < n_pairs; p++) {
                int section_idx = 0;
                if ((int)p >= cumsum[1]) section_idx = 1;
                if ((int)p >= cumsum[2]) section_idx = 2;
                int local_i = (int)p - cumsum[section_idx];
                float pos = (float)pos_ids[section_idx * seq_len + s];
                float freq = pos / std::pow(theta, 2.0f * (float)local_i / (float)rotary_dim);
                float cos_v = std::cos(freq);
                float sin_v = std::sin(freq);

                size_t idx_a = base + 2 * p;
                size_t idx_b = base + 2 * p + 1;
                float a, b;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    a = llaisys::utils::cast<float>(in[idx_a]);
                    b = llaisys::utils::cast<float>(in[idx_b]);
                    out[idx_a] = llaisys::utils::cast<T>(a * cos_v - b * sin_v);
                    out[idx_b] = llaisys::utils::cast<T>(b * cos_v + a * sin_v);
                } else {
                    a = static_cast<float>(in[idx_a]);
                    b = static_cast<float>(in[idx_b]);
                    out[idx_a] = static_cast<T>(a * cos_v - b * sin_v);
                    out[idx_b] = static_cast<T>(b * cos_v + a * sin_v);
                }
            }
            // Pass-through non-rotary dimensions
            for (size_t d = rotary_dim; d < head_dim; d++) {
                out[base + d] = in[base + d];
            }
        }
    }
}

namespace llaisys::ops::cpu {
void mrope(std::byte *out_ptr, const std::byte *in_ptr, const std::byte *pos_ids_ptr,
           float theta, const int *sections, size_t rotary_dim,
           llaisysDataType_t dtype,
           size_t seq_len, size_t n_head, size_t head_dim) {
    auto *pos = reinterpret_cast<const int64_t *>(pos_ids_ptr);
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return mrope_scalar(reinterpret_cast<float *>(out_ptr),
                            reinterpret_cast<const float *>(in_ptr), pos,
                            theta, sections, rotary_dim, seq_len, n_head, head_dim);
    case LLAISYS_DTYPE_BF16:
        return mrope_scalar(reinterpret_cast<llaisys::bf16_t *>(out_ptr),
                            reinterpret_cast<const llaisys::bf16_t *>(in_ptr), pos,
                            theta, sections, rotary_dim, seq_len, n_head, head_dim);
    case LLAISYS_DTYPE_F16:
        return mrope_scalar(reinterpret_cast<llaisys::fp16_t *>(out_ptr),
                            reinterpret_cast<const llaisys::fp16_t *>(in_ptr), pos,
                            theta, sections, rotary_dim, seq_len, n_head, head_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
