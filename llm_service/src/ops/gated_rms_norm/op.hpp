#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
// out = rms_norm(x, weight, eps) * silu(z)
void gated_rms_norm(tensor_t out, tensor_t x, tensor_t z, tensor_t weight, float eps);
}
