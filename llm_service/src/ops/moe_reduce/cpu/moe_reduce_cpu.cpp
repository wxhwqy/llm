#include "moe_reduce_cpu.hpp"
#include <cstring>
#include <cstdint>
#include <cmath>

namespace llaisys::ops::cpu {

static inline float bf16_to_f(uint16_t raw) {
    uint32_t bits = static_cast<uint32_t>(raw) << 16;
    float f; std::memcpy(&f, &bits, sizeof(float)); return f;
}
static inline uint16_t f_to_bf16(float f) {
    uint32_t bits; std::memcpy(&bits, &f, sizeof(float));
    return static_cast<uint16_t>(bits >> 16);
}

void moe_accumulate(std::byte *accum, const std::byte *expert_out,
                    float weight, int token_idx,
                    size_t seq_len, size_t hidden) {
    float *acc = reinterpret_cast<float*>(accum);
    const uint16_t *eout = reinterpret_cast<const uint16_t*>(expert_out);

    if (token_idx >= 0) {
        // Single token
        float *a = acc + token_idx * hidden;
        const uint16_t *e = eout + token_idx * hidden;
        for (size_t d = 0; d < hidden; d++)
            a[d] += weight * bf16_to_f(e[d]);
    } else {
        // All tokens
        for (size_t t = 0; t < seq_len; t++) {
            float *a = acc + t * hidden;
            const uint16_t *e = eout + t * hidden;
            for (size_t d = 0; d < hidden; d++)
                a[d] += weight * bf16_to_f(e[d]);
        }
    }
}

void moe_combine(std::byte *hidden, const std::byte *residual,
                 const std::byte *accum, const std::byte *shared_out,
                 size_t seq_len, size_t hidden_size) {
    uint16_t *h = reinterpret_cast<uint16_t*>(hidden);
    const uint16_t *r = reinterpret_cast<const uint16_t*>(residual);
    const float *a = reinterpret_cast<const float*>(accum);
    const uint16_t *s = reinterpret_cast<const uint16_t*>(shared_out);

    for (size_t i = 0; i < seq_len * hidden_size; i++) {
        float result = bf16_to_f(r[i]) + a[i] + bf16_to_f(s[i]);
        h[i] = f_to_bf16(result);
    }
}

void moe_shared_gate(std::byte *shared_out, const std::byte *normed,
                     const std::byte *gate_weight,
                     size_t seq_len, size_t hidden_size) {
    uint16_t *sout = reinterpret_cast<uint16_t*>(shared_out);
    const uint16_t *inp = reinterpret_cast<const uint16_t*>(normed);
    const uint16_t *gw = reinterpret_cast<const uint16_t*>(gate_weight);

    for (size_t t = 0; t < seq_len; t++) {
        // dot product: gate_weight . normed[t]
        float dot = 0.0f;
        for (size_t d = 0; d < hidden_size; d++)
            dot += bf16_to_f(gw[d]) * bf16_to_f(inp[t * hidden_size + d]);

        float gate_val = 1.0f / (1.0f + std::exp(-dot));

        // scale shared expert output
        for (size_t d = 0; d < hidden_size; d++) {
            float v = bf16_to_f(sout[t * hidden_size + d]) * gate_val;
            sout[t * hidden_size + d] = f_to_bf16(v);
        }
    }
}

} // namespace llaisys::ops::cpu
