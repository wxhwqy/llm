#pragma once
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::nvidia {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k,
                    const std::byte *v, float scale, llaisysDataType_t dtype,
                    size_t qlen, size_t kvlen, size_t nhead, size_t nkvhead, size_t hd);
void self_attention_gated(std::byte *attn_val, const std::byte *q, const std::byte *k,
                          const std::byte *v, const std::byte *gate, float scale,
                          llaisysDataType_t dtype,
                          size_t qlen, size_t kvlen, size_t nhead, size_t nkvhead, size_t hd);
}
