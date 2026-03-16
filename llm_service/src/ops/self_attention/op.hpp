#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale);
// Gated attention: out = self_attention(q,k,v) * sigmoid(gate)
// gate shape: [qlen, nhead, hd] (same as attn_val)
void self_attention_gated(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, tensor_t gate, float scale);
}
