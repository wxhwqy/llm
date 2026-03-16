#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {

// Decode: single-step recurrent update
// state [n_heads, dv, dk] F32 (in-place), q/k [n_heads, dk], v [n_heads, dv]
// g/beta [n_heads] F32, out [n_heads, dv]
void gated_delta_rule_recurrent(
    tensor_t out, tensor_t state,
    tensor_t q, tensor_t k, tensor_t v,
    tensor_t g, tensor_t beta,
    size_t n_heads, size_t dk, size_t dv);

// Prefill: sequential over seq_len, outputs all positions
// q/k [seq_len, n_heads, dk], v [seq_len, n_heads, dv]
// g/beta [seq_len, n_heads] F32, out [seq_len, n_heads, dv]
// final_state [n_heads, dv, dk] F32 (output)
void gated_delta_rule_chunk(
    tensor_t out, tensor_t final_state,
    tensor_t q, tensor_t k, tensor_t v,
    tensor_t g, tensor_t beta,
    size_t seq_len, size_t n_heads, size_t dk, size_t dv);
}
