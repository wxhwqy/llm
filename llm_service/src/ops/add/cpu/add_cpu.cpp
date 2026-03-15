#include "../../cpu_simd_utils.hpp"

#include "add_cpu.hpp"
#include "../../../utils.hpp"

#include <cmath>

#ifdef __AVX2__
static void add_f32_avx2(float *c, const float *a, const float *b, size_t numel) {
    size_t i = 0;
    for (; i + 8 <= numel; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(c + i, vc);
    }
    for (; i < numel; i++) {
        c[i] = a[i] + b[i];
    }
}

static void add_bf16_avx2(llaisys::bf16_t *c, const llaisys::bf16_t *a,
                           const llaisys::bf16_t *b, size_t numel) {
    using namespace llaisys::ops::cpu;
    size_t i = 0;
    for (; i + 8 <= numel; i += 8) {
        __m256 va = bf16x8_to_f32x8(a + i);
        __m256 vb = bf16x8_to_f32x8(b + i);
        __m256 vc = _mm256_add_ps(va, vb);
        f32x8_store_bf16(c + i, vc);
    }
    for (; i < numel; i++) {
        float av = llaisys::utils::cast<float>(a[i]);
        float bv = llaisys::utils::cast<float>(b[i]);
        c[i] = llaisys::utils::cast<llaisys::bf16_t>(av + bv);
    }
}

#ifdef __F16C__
static void add_fp16_avx2(llaisys::fp16_t *c, const llaisys::fp16_t *a,
                           const llaisys::fp16_t *b, size_t numel) {
    using namespace llaisys::ops::cpu;
    size_t i = 0;
    for (; i + 8 <= numel; i += 8) {
        __m256 va = fp16x8_to_f32x8(a + i);
        __m256 vb = fp16x8_to_f32x8(b + i);
        __m256 vc = _mm256_add_ps(va, vb);
        f32x8_store_fp16(c + i, vc);
    }
    for (; i < numel; i++) {
        float av = llaisys::utils::cast<float>(a[i]);
        float bv = llaisys::utils::cast<float>(b[i]);
        c[i] = llaisys::utils::cast<llaisys::fp16_t>(av + bv);
    }
}
#endif // __F16C__
#endif // __AVX2__

template <typename T>
static void add_scalar(T *c, const T *a, const T *b, size_t numel) {
    for (size_t i = 0; i < numel; i++) {
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            c[i] = llaisys::utils::cast<T>(llaisys::utils::cast<float>(a[i]) + llaisys::utils::cast<float>(b[i]));
        } else {
            c[i] = a[i] + b[i];
        }
    }
}

namespace llaisys::ops::cpu {
void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
#ifdef __AVX2__
        return add_f32_avx2(reinterpret_cast<float *>(c), reinterpret_cast<const float *>(a),
                            reinterpret_cast<const float *>(b), numel);
#else
        return add_scalar(reinterpret_cast<float *>(c), reinterpret_cast<const float *>(a),
                          reinterpret_cast<const float *>(b), numel);
#endif
    case LLAISYS_DTYPE_BF16:
#ifdef __AVX2__
        return add_bf16_avx2(reinterpret_cast<bf16_t *>(c), reinterpret_cast<const bf16_t *>(a),
                             reinterpret_cast<const bf16_t *>(b), numel);
#else
        return add_scalar(reinterpret_cast<bf16_t *>(c), reinterpret_cast<const bf16_t *>(a),
                          reinterpret_cast<const bf16_t *>(b), numel);
#endif
    case LLAISYS_DTYPE_F16:
#if defined(__AVX2__) && defined(__F16C__)
        return add_fp16_avx2(reinterpret_cast<fp16_t *>(c), reinterpret_cast<const fp16_t *>(a),
                             reinterpret_cast<const fp16_t *>(b), numel);
#else
        return add_scalar(reinterpret_cast<fp16_t *>(c), reinterpret_cast<const fp16_t *>(a),
                          reinterpret_cast<const fp16_t *>(b), numel);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
