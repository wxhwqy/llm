#include "../../cpu_simd_utils.hpp"

#include "swiglu_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

#ifdef __AVX2__
// Fast approximate exp for AVX2 using polynomial approximation
// Accuracy: ~1e-4 relative error, sufficient for sigmoid in inference
static inline __m256 exp_approx_avx2(__m256 x) {
    // Clamp input to avoid overflow/underflow
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.0f));
    x = _mm256_min_ps(x, _mm256_set1_ps(88.0f));

    // exp(x) = 2^(x * log2(e))
    // x * log2(e)
    const __m256 log2e = _mm256_set1_ps(1.44269504089f);
    __m256 t = _mm256_mul_ps(x, log2e);

    // Round to nearest integer
    __m256 ti = _mm256_round_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 tf = _mm256_sub_ps(t, ti); // fractional part in [-0.5, 0.5]

    // Convert integer part to float exponent via bit manipulation
    __m256i ti_int = _mm256_cvtps_epi32(ti);
    __m256i exp_bits = _mm256_slli_epi32(_mm256_add_epi32(ti_int, _mm256_set1_epi32(127)), 23);
    __m256 pow2_int = _mm256_castsi256_ps(exp_bits);

    // Polynomial approximation for 2^tf where tf in [-0.5, 0.5]
    // Minimax polynomial coefficients
    const __m256 p0 = _mm256_set1_ps(1.0f);
    const __m256 p1 = _mm256_set1_ps(0.6931472f);
    const __m256 p2 = _mm256_set1_ps(0.2402265f);
    const __m256 p3 = _mm256_set1_ps(0.0554953f);
    const __m256 p4 = _mm256_set1_ps(0.0096838f);
    const __m256 p5 = _mm256_set1_ps(0.0013364f);

    // Horner's method: p0 + tf*(p1 + tf*(p2 + tf*(p3 + tf*(p4 + tf*p5))))
    __m256 poly = p5;
#ifdef __FMA__
    poly = _mm256_fmadd_ps(poly, tf, p4);
    poly = _mm256_fmadd_ps(poly, tf, p3);
    poly = _mm256_fmadd_ps(poly, tf, p2);
    poly = _mm256_fmadd_ps(poly, tf, p1);
    poly = _mm256_fmadd_ps(poly, tf, p0);
#else
    poly = _mm256_add_ps(_mm256_mul_ps(poly, tf), p4);
    poly = _mm256_add_ps(_mm256_mul_ps(poly, tf), p3);
    poly = _mm256_add_ps(_mm256_mul_ps(poly, tf), p2);
    poly = _mm256_add_ps(_mm256_mul_ps(poly, tf), p1);
    poly = _mm256_add_ps(_mm256_mul_ps(poly, tf), p0);
#endif

    return _mm256_mul_ps(pow2_int, poly);
}

static void swiglu_f32_avx2(float *out, const float *gate, const float *up, size_t numel) {
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 neg_one = _mm256_set1_ps(-1.0f);

    size_t i = 0;
    for (; i + 8 <= numel; i += 8) {
        __m256 vgate = _mm256_loadu_ps(gate + i);
        __m256 vup = _mm256_loadu_ps(up + i);

        // sigmoid(gate) = 1 / (1 + exp(-gate))
        __m256 neg_gate = _mm256_mul_ps(vgate, neg_one);
        __m256 exp_neg = exp_approx_avx2(neg_gate);
        __m256 sigmoid = _mm256_div_ps(one, _mm256_add_ps(one, exp_neg));

        // silu(gate) = gate * sigmoid(gate)
        // result = up * silu(gate)
        __m256 result = _mm256_mul_ps(vup, _mm256_mul_ps(vgate, sigmoid));
        _mm256_storeu_ps(out + i, result);
    }
    for (; i < numel; i++) {
        float g = gate[i];
        float u = up[i];
        float sig = 1.0f / (1.0f + std::exp(-g));
        out[i] = u * g * sig;
    }
}

static void swiglu_bf16_avx2(llaisys::bf16_t *out, const llaisys::bf16_t *gate,
                              const llaisys::bf16_t *up, size_t numel) {
    using namespace llaisys::ops::cpu;
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 neg_one = _mm256_set1_ps(-1.0f);

    size_t i = 0;
    for (; i + 8 <= numel; i += 8) {
        __m256 vgate = bf16x8_to_f32x8(gate + i);
        __m256 vup = bf16x8_to_f32x8(up + i);

        __m256 neg_gate = _mm256_mul_ps(vgate, neg_one);
        __m256 exp_neg = exp_approx_avx2(neg_gate);
        __m256 sigmoid = _mm256_div_ps(one, _mm256_add_ps(one, exp_neg));
        __m256 result = _mm256_mul_ps(vup, _mm256_mul_ps(vgate, sigmoid));
        f32x8_store_bf16(out + i, result);
    }
    for (; i < numel; i++) {
        float g = llaisys::utils::cast<float>(gate[i]);
        float u = llaisys::utils::cast<float>(up[i]);
        float sig = 1.0f / (1.0f + std::exp(-g));
        out[i] = llaisys::utils::cast<llaisys::bf16_t>(u * g * sig);
    }
}
#endif // __AVX2__

template <typename T>
static void swiglu_scalar(T *out, const T *gate, const T *up, size_t numel) {
    for (size_t i = 0; i < numel; i++) {
        float gate_val, up_val;
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            gate_val = llaisys::utils::cast<float>(gate[i]);
            up_val = llaisys::utils::cast<float>(up[i]);
        } else {
            gate_val = static_cast<float>(gate[i]);
            up_val = static_cast<float>(up[i]);
        }
        float sigmoid = 1.0f / (1.0f + std::exp(-gate_val));
        float silu = gate_val * sigmoid;
        float result = up_val * silu;
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            out[i] = llaisys::utils::cast<T>(result);
        } else {
            out[i] = static_cast<T>(result);
        }
    }
}

namespace llaisys::ops::cpu {
void swiglu(std::byte *out_ptr, const std::byte *gate_ptr, const std::byte *up_ptr,
            llaisysDataType_t dtype, size_t numel) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
#ifdef __AVX2__
        return swiglu_f32_avx2(reinterpret_cast<float *>(out_ptr),
                               reinterpret_cast<const float *>(gate_ptr),
                               reinterpret_cast<const float *>(up_ptr), numel);
#else
        return swiglu_scalar(reinterpret_cast<float *>(out_ptr),
                             reinterpret_cast<const float *>(gate_ptr),
                             reinterpret_cast<const float *>(up_ptr), numel);
#endif
    case LLAISYS_DTYPE_BF16:
#ifdef __AVX2__
        return swiglu_bf16_avx2(reinterpret_cast<bf16_t *>(out_ptr),
                                reinterpret_cast<const bf16_t *>(gate_ptr),
                                reinterpret_cast<const bf16_t *>(up_ptr), numel);
#else
        return swiglu_scalar(reinterpret_cast<bf16_t *>(out_ptr),
                             reinterpret_cast<const bf16_t *>(gate_ptr),
                             reinterpret_cast<const bf16_t *>(up_ptr), numel);
#endif
    case LLAISYS_DTYPE_F16:
        return swiglu_scalar(reinterpret_cast<fp16_t *>(out_ptr),
                             reinterpret_cast<const fp16_t *>(gate_ptr),
                             reinterpret_cast<const fp16_t *>(up_ptr), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
