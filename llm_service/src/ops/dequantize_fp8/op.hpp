#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
void dequantize_fp8(tensor_t out_bf16, tensor_t in_fp8, tensor_t scale_inv,
                    size_t block_h, size_t block_w);
}
