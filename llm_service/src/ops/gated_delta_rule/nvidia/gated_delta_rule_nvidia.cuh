#pragma once
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::nvidia {
void gated_delta_rule_recurrent(
    std::byte *out, std::byte *state,
    const std::byte *q, const std::byte *k, const std::byte *v,
    const std::byte *g, const std::byte *beta,
    llaisysDataType_t dtype, size_t n_heads, size_t dk, size_t dv);

void gated_delta_rule_chunk(
    std::byte *out, std::byte *final_state,
    const std::byte *q, const std::byte *k, const std::byte *v,
    const std::byte *g, const std::byte *beta,
    llaisysDataType_t dtype, size_t seq_len, size_t n_heads, size_t dk, size_t dv);
}
