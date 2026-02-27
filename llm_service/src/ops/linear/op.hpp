#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias);

// Fused tiled FP8 dequant + GEMM. Uses a tiny shared tile buffer instead of
// materializing the full BF16 weight matrix.
void linear_fp8(tensor_t out, tensor_t in,
                tensor_t weight_fp8, tensor_t scale_inv,
                size_t fp8_block_h, size_t fp8_block_w);
}
