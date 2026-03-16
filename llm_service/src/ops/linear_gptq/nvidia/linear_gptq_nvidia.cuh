#pragma once

#include <cstddef>
#include <cstdint>

namespace llaisys::ops::nvidia {

void linear_gptq(std::byte *output, const std::byte *input,
                 const std::byte *qweight, const std::byte *scales,
                 const std::byte *qzeros,
                 size_t M, size_t in_features, size_t out_features,
                 int bits, int group_size);

} // namespace llaisys::ops::nvidia
