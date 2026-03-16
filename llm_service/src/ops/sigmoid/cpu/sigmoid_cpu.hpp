#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void sigmoid(std::byte *out, const std::byte *in,
            llaisysDataType_t dtype, size_t numel);
}
