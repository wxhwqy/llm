#pragma once
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::nvidia {
void gated_rms_norm(std::byte *out, const std::byte *x, const std::byte *z,
                     const std::byte *weight, float eps, llaisysDataType_t dtype,
                     size_t M, size_t N);
}
