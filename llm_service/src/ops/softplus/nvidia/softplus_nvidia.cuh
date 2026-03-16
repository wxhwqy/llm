#pragma once
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::nvidia {
void softplus(std::byte *out, const std::byte *in,
             llaisysDataType_t dtype, size_t numel);
}
