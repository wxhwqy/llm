#pragma once

// IMPORTANT: immintrin.h must be included BEFORE llaisys.h because
// llaisys.h defines __C as "extern C" which conflicts with intrinsic
// parameter names (e.g., __C in CRC32 intrinsics).
#ifdef __AVX2__
#include <immintrin.h>
#endif

#include "../utils/types.hpp"

#include <cstddef>
#include <cstring>

namespace llaisys::ops::cpu {

#ifdef __AVX2__

// ============================================================
// BF16 <-> F32 batch conversion using AVX2
// BF16 is the upper 16 bits of F32, so conversion is a shift.
// ============================================================

inline __m256 bf16x8_to_f32x8(const llaisys::bf16_t *src) {
    __m128i bf16_vals = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src));
    __m256i i32_vals = _mm256_cvtepu16_epi32(bf16_vals);
    __m256i f32_bits = _mm256_slli_epi32(i32_vals, 16);
    return _mm256_castsi256_ps(f32_bits);
}

// Store 8 F32 values as BF16 with round-to-nearest-even
inline void f32x8_store_bf16(llaisys::bf16_t *dst, __m256 f32_vals) {
    __m256i bits = _mm256_castps_si256(f32_vals);
    // Rounding bias: 0x7FFF + lsb(bit16)
    __m256i truncated = _mm256_srli_epi32(bits, 16);
    __m256i lsb = _mm256_and_si256(truncated, _mm256_set1_epi32(1));
    __m256i rounding_bias = _mm256_add_epi32(_mm256_set1_epi32(0x7FFF), lsb);
    __m256i rounded = _mm256_add_epi32(bits, rounding_bias);
    __m256i result = _mm256_srli_epi32(rounded, 16);
    // Pack 8x32-bit to 8x16-bit using packus across lanes then permute
    // _mm256_packus_epi32 packs within 128-bit lanes:
    //   [a0,a1,a2,a3, a4,a5,a6,a7] -> [a0,a1,a2,a3, a0,a1,a2,a3, a4,a5,a6,a7, a4,a5,a6,a7]
    // We pack with zeros and fix lane ordering
    __m256i packed = _mm256_packus_epi32(result, _mm256_setzero_si256());
    // After packus: [r0,r1,r2,r3, 0,0,0,0, r4,r5,r6,r7, 0,0,0,0]
    // Need to move lane1[0:63] to lane0[64:127]
    __m256i perm = _mm256_permute4x64_epi64(packed, 0xD8); // 0b11_01_10_00 -> [0,2,1,3]
    // Now: [r0,r1,r2,r3, r4,r5,r6,r7, 0,0,0,0, 0,0,0,0]
    _mm_storeu_si128(reinterpret_cast<__m128i *>(dst), _mm256_castsi256_si128(perm));
}

// ============================================================
// FP16 <-> F32 batch conversion using F16C instructions
// ============================================================

#ifdef __F16C__
inline __m256 fp16x8_to_f32x8(const llaisys::fp16_t *src) {
    __m128i fp16_vals = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src));
    return _mm256_cvtph_ps(fp16_vals);
}

inline void f32x8_store_fp16(llaisys::fp16_t *dst, __m256 f32_vals) {
    __m128i fp16_vals = _mm256_cvtps_ph(f32_vals, _MM_FROUND_TO_NEAREST_INT);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(dst), fp16_vals);
}
#endif // __F16C__

// ============================================================
// Horizontal sum of __m256 (8 floats -> 1 float)
// ============================================================
inline float hsum_f32x8(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 sum4 = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(sum4);
    __m128 sum2 = _mm_add_ps(sum4, shuf);
    shuf = _mm_movehl_ps(shuf, sum2);
    __m128 sum1 = _mm_add_ss(sum2, shuf);
    return _mm_cvtss_f32(sum1);
}

// ============================================================
// Horizontal max of __m256 (8 floats -> 1 float)
// ============================================================
inline float hmax_f32x8(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 max4 = _mm_max_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(max4);
    __m128 max2 = _mm_max_ps(max4, shuf);
    shuf = _mm_movehl_ps(shuf, max2);
    __m128 max1 = _mm_max_ss(max2, shuf);
    return _mm_cvtss_f32(max1);
}

#endif // __AVX2__

} // namespace llaisys::ops::cpu
