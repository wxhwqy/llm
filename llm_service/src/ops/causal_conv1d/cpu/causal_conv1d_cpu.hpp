#pragma once
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::cpu {
void causal_conv1d(std::byte *out, const std::byte *in, const std::byte *weight,
                    const std::byte *bias, llaisysDataType_t dtype,
                    size_t seq_len, size_t d_inner, size_t kernel_size);

void causal_conv1d_step(std::byte *out_col, std::byte *conv_state, const std::byte *in_col,
                         const std::byte *weight, const std::byte *bias,
                         llaisysDataType_t dtype, size_t d_inner, size_t kernel_size);
}
