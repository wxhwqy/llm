#pragma once
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::nvidia {
void linear(std::byte *out, const std::byte *in, const std::byte *weight,
            const std::byte *bias, llaisysDataType_t dtype,
            size_t M, size_t K, size_t N);

// Tiled FP8 dequant + GEMM: out[M,N] = in[M,K] @ dequant(weight_fp8[N,K])^T
// Uses a tiny tile buffer (tile_n × K × 2 bytes) instead of a full N×K buffer.
void linear_fp8(std::byte *out, const std::byte *in,
                const std::byte *weight_fp8, const std::byte *scale_inv,
                llaisysDataType_t compute_dtype,
                size_t M, size_t K, size_t N,
                size_t block_h, size_t block_w);
}
