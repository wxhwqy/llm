#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
// M-RoPE: multi-dimensional rotary position embedding
// q/k: [seq_len, n_head, head_dim]
// pos_ids: [3, seq_len] (temporal, height, width positions)
// sections: [3] array of ints (e.g. [11, 11, 10])
// rotary_dim: number of dimensions to rotate (head_dim * partial_rotary_factor)
void mrope(tensor_t out, tensor_t in, tensor_t pos_ids,
           float theta, const int *sections, size_t rotary_dim,
           size_t seq_len, size_t n_head, size_t head_dim);
}
