#include "linear_gptq_cpu.hpp"
#include <cstring>
#include <cstdint>

namespace llaisys::ops::cpu {

static inline float bf16_to_f(uint16_t raw) {
    uint32_t bits = static_cast<uint32_t>(raw) << 16;
    float f; std::memcpy(&f, &bits, sizeof(float)); return f;
}
static inline uint16_t f_to_bf16(float f) {
    uint32_t bits; std::memcpy(&bits, &f, sizeof(float));
    return static_cast<uint16_t>(bits >> 16);
}

void linear_gptq(std::byte *output, const std::byte *input,
                 const std::byte *qweight, const std::byte *scales,
                 const std::byte *qzeros,
                 size_t M, size_t in_features, size_t out_features,
                 int bits, int group_size) {
    int pack = 32 / bits;  // 8 for 4-bit
    int mask = (1 << bits) - 1;  // 0xF

    const uint16_t *inp = reinterpret_cast<const uint16_t*>(input);
    uint16_t *out = reinterpret_cast<uint16_t*>(output);
    const int32_t *qw = reinterpret_cast<const int32_t*>(qweight);
    const uint16_t *sc = reinterpret_cast<const uint16_t*>(scales);
    const int32_t *qz = qzeros ? reinterpret_cast<const int32_t*>(qzeros) : nullptr;

    size_t qw_rows = in_features / pack;

    for (size_t s = 0; s < M; s++) {
        const uint16_t *x = inp + s * in_features;
        uint16_t *y = out + s * out_features;

        for (size_t j = 0; j < out_features; j++) {
            float sum = 0.0f;

            for (size_t ig = 0; ig < qw_rows; ig++) {
                int32_t packed = qw[ig * out_features + j];
                size_t base_row = ig * pack;
                size_t group_idx = base_row / group_size;

                float scale = bf16_to_f(sc[group_idx * out_features + j]);

                int zero = 8;  // symmetric default
                if (qz) {
                    size_t zp_col = j / pack;
                    int32_t zp_packed = qz[group_idx * (out_features / pack) + zp_col];
                    zero = (zp_packed >> ((j % pack) * bits)) & mask;
                }

                for (int k = 0; k < pack; k++) {
                    int int4_val = (packed >> (k * bits)) & mask;
                    float w_val = (float)(int4_val - zero) * scale;
                    sum += bf16_to_f(x[base_row + k]) * w_val;
                }
            }
            y[j] = f_to_bf16(sum);
        }
    }
}

} // namespace llaisys::ops::cpu
