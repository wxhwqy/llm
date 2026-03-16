#pragma once
#include "llaisys.h"

#include <cstddef>
#include <cstdint>

namespace llaisys::ops::cpu {
void sample(std::byte *output_idx, const std::byte *logits, std::byte *workspace,
            llaisysDataType_t dtype, size_t vocab_size,
            float temperature, int top_k, float top_p, uint64_t seed,
            const int64_t *penalty_tokens = nullptr, size_t n_penalty_tokens = 0,
            float repetition_penalty = 1.0f);
}
