#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
// Prefill: full causal 1D convolution with SiLU
// in: [seq_len, d_inner], weight: [d_inner, kernel_size], bias: [d_inner] or nullptr
// out: [seq_len, d_inner]
void causal_conv1d(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias,
                    size_t seq_len, size_t d_inner, size_t kernel_size);

// Decode: incremental single-step update
// conv_state: [d_inner, kernel_size] — mutable, shifted left and new input appended
// in_col: [d_inner] — single input
// out_col: [d_inner] — single output
void causal_conv1d_step(tensor_t out_col, tensor_t conv_state, tensor_t in_col,
                         tensor_t weight, tensor_t bias,
                         size_t d_inner, size_t kernel_size);
}
