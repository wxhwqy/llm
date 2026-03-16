#pragma once

#include <cstddef>

namespace llaisys::ops::nvidia {

void moe_accumulate(std::byte *accum, const std::byte *expert_out,
                    float weight, int token_idx,
                    size_t seq_len, size_t hidden);

void moe_combine(std::byte *hidden, const std::byte *residual,
                 const std::byte *accum, const std::byte *shared_out,
                 size_t seq_len, size_t hidden_size);

void moe_shared_gate(std::byte *shared_out, const std::byte *normed,
                     const std::byte *gate_weight,
                     size_t seq_len, size_t hidden_size);

} // namespace llaisys::ops::nvidia
