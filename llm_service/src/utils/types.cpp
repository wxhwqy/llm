#include "types.hpp"

#include <cmath>
#include <cstring>
#include <limits>

namespace llaisys::utils {
float _f16_to_f32(fp16_t val) {
    uint16_t h = val._v;
    uint32_t sign = (h & 0x8000) << 16;
    int32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;

    uint32_t f32;
    if (exponent == 31) {
        if (mantissa != 0) {
            f32 = sign | 0x7F800000 | (mantissa << 13);
        } else {
            f32 = sign | 0x7F800000;
        }
    } else if (exponent == 0) {
        if (mantissa == 0) {
            f32 = sign;
        } else {
            exponent = -14;
            while ((mantissa & 0x400) == 0) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x3FF;
            f32 = sign | ((exponent + 127) << 23) | (mantissa << 13);
        }
    } else {
        f32 = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }

    float result;
    memcpy(&result, &f32, sizeof(result));
    return result;
}

fp16_t _f32_to_f16(float val) {
    uint32_t f32;
    memcpy(&f32, &val, sizeof(f32));               // Read the bits of the float32
    uint16_t sign = (f32 >> 16) & 0x8000;          // Extract the sign bit
    int32_t exponent = ((f32 >> 23) & 0xFF) - 127; // Extract and de-bias the exponent
    uint32_t mantissa = f32 & 0x7FFFFF;            // Extract the mantissa (fraction part)

    if (exponent >= 16) { // Special cases for Inf and NaN
        // NaN
        if (exponent == 128 && mantissa != 0) {
            return fp16_t{static_cast<uint16_t>(sign | 0x7E00)};
        }
        // Infinity
        return fp16_t{static_cast<uint16_t>(sign | 0x7C00)};
    } else if (exponent >= -14) { // Normalized case
        return fp16_t{(uint16_t)(sign | ((exponent + 15) << 10) | (mantissa >> 13))};
    } else if (exponent >= -24) {
        mantissa |= 0x800000; // Add implicit leading 1
        mantissa >>= (-14 - exponent);
        return fp16_t{(uint16_t)(sign | (mantissa >> 13))};
    } else {
        // Too small for subnormal: return signed zero
        return fp16_t{(uint16_t)sign};
    }
}

float _bf16_to_f32(bf16_t val) {
    uint32_t bits32 = static_cast<uint32_t>(val._v) << 16;

    float out;
    std::memcpy(&out, &bits32, sizeof(out));
    return out;
}

bf16_t _f32_to_bf16(float val) {
    uint32_t bits32;
    std::memcpy(&bits32, &val, sizeof(bits32));

    const uint32_t rounding_bias = 0x00007FFF + // 0111 1111 1111 1111
                                   ((bits32 >> 16) & 1);

    uint16_t bf16_bits = static_cast<uint16_t>((bits32 + rounding_bias) >> 16);

    return bf16_t{bf16_bits};
}
float _fp8e4m3_to_f32(fp8e4m3_t val) {
    uint8_t bits = val._v;
    uint32_t sign = (bits & 0x80) << 24;
    int32_t exponent = (bits >> 3) & 0x0F;
    uint32_t mantissa = bits & 0x07;

    if (exponent == 15 && mantissa == 7) {
        return sign ? -std::numeric_limits<float>::quiet_NaN()
                    : std::numeric_limits<float>::quiet_NaN();
    }

    uint32_t f32;
    if (exponent == 0) {
        if (mantissa == 0) {
            f32 = sign;
        } else {
            // E4M3 bias = 7, subnormal: value = (-1)^s * 2^(1-7) * (0.mantissa)
            float m = (float)mantissa / 8.0f;
            float result = std::ldexp(m, -6); // 2^(1-7) = 2^-6
            std::memcpy(&f32, &result, sizeof(f32));
            f32 = (f32 & 0x7FFFFFFF) | sign;
        }
    } else {
        // Normalized: value = (-1)^s * 2^(e-7) * (1 + mantissa/8)
        int32_t unbiased = exponent - 7;
        f32 = sign | ((unbiased + 127) << 23) | (mantissa << 20);
    }

    float result;
    std::memcpy(&result, &f32, sizeof(result));
    return result;
}

fp8e4m3_t _f32_to_fp8e4m3(float val) {
    uint32_t f32;
    std::memcpy(&f32, &val, sizeof(f32));
    uint8_t sign = (f32 >> 24) & 0x80;
    int32_t exponent = ((f32 >> 23) & 0xFF) - 127;
    uint32_t mantissa = f32 & 0x7FFFFF;

    if (exponent > 8) {
        // Clamp to max finite: s_1111_110 = +-448
        return fp8e4m3_t{static_cast<uint8_t>(sign | 0x7E)};
    } else if (exponent >= -6) {
        uint8_t e = (uint8_t)(exponent + 7);
        uint8_t m = (uint8_t)(mantissa >> 20);
        return fp8e4m3_t{static_cast<uint8_t>(sign | (e << 3) | m)};
    } else if (exponent >= -9) {
        // Subnormal
        uint32_t full_m = mantissa | 0x800000;
        full_m >>= (-6 - exponent);
        uint8_t m = (uint8_t)((full_m >> 20) & 0x07);
        return fp8e4m3_t{static_cast<uint8_t>(sign | m)};
    } else {
        return fp8e4m3_t{static_cast<uint8_t>(sign)};
    }
}
} // namespace llaisys::utils
