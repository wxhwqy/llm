#pragma once

#include "../../tensor/tensor.hpp"
#include <cstdint>

namespace llaisys::ops {
void sample(tensor_t output_idx, tensor_t logits, tensor_t workspace,
            float temperature, int top_k, float top_p, uint64_t seed);
}
