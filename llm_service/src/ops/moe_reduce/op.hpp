#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {

// Weighted accumulate: accum += weight * expert_out
//   accum:      [seq_len, hidden_size] F32 (in-place accumulate)
//   expert_out: [seq_len, hidden_size] BF16
//   weight:     scalar float
//   token_idx:  which token (row) in accum/expert_out to process
//               If token_idx < 0, process all tokens.
void moe_accumulate(tensor_t accum, tensor_t expert_out,
                    float weight, int token_idx);

// Finalize MoE output: hidden = residual + accum + shared_out
//   hidden:     [seq_len, hidden_size] BF16 (output)
//   residual:   [seq_len, hidden_size] BF16
//   accum:      [seq_len, hidden_size] F32  (routed expert accumulator)
//   shared_out: [seq_len, hidden_size] BF16
void moe_combine(tensor_t hidden, tensor_t residual,
                 tensor_t accum, tensor_t shared_out);

// Shared expert gate: shared_out *= sigmoid(gate_weight @ normed)
//   shared_out: [seq_len, hidden_size] BF16 (in-place)
//   normed:     [seq_len, hidden_size] BF16
//   gate_weight:[1, hidden_size] BF16
void moe_shared_gate(tensor_t shared_out, tensor_t normed,
                     tensor_t gate_weight);

// Zero-fill an F32 accumulator tensor
void moe_zero_accum(tensor_t accum);

} // namespace llaisys::ops
