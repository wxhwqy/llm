#pragma once
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::nvidia {
void dequantize_fp8(std::byte *out_bf16, const std::byte *in_fp8,
                    const std::byte *scale_inv,
                    size_t M, size_t K, size_t block_h, size_t block_w);
}
