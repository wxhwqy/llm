#pragma once
#include "llaisys.h"
#include <cstddef>
#include <cstdint>

namespace llaisys::ops::nvidia {
void sample(std::byte *output_idx, const std::byte *logits, std::byte *workspace,
            llaisysDataType_t dtype, size_t vocab_size,
            float temperature, int top_k, float top_p, uint64_t seed);
}
