#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {

// Fused GPTQ INT4 dequant + linear.
//   output = input @ dequant(qweight, scales, qzeros).T
//   input:   [M, in_features]  BF16
//   output:  [M, out_features] BF16
//   qweight: [in_features/pack, out_features] INT32   (pack = 32/bits)
//   scales:  [num_groups, out_features]       BF16
//   qzeros:  [num_groups, out_features/pack]  INT32
void linear_gptq(tensor_t output, tensor_t input,
                 tensor_t qweight, tensor_t scales, tensor_t qzeros,
                 size_t in_features, size_t out_features,
                 int bits, int group_size);

} // namespace llaisys::ops
