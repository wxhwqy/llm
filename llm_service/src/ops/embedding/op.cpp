#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/embedding_nvidia.cuh"
#endif

namespace llaisys::ops {

void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "不是int64");
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    ASSERT(index->ndim() == 1, "index不是1D");
    ASSERT(weight->ndim() == 2, "weight不是2D");
    ASSERT(out->ndim() == 2, "out不是2D");
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(),
           "不连续");

    size_t seq_len = index->shape()[0];
    size_t hidden_size = weight->shape()[1];

    ASSERT(out->shape()[0] == seq_len && out->shape()[1] == hidden_size,
           "out大小不对");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), weight->data(),
                              out->dtype(), seq_len, hidden_size);
    }

#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
        return nvidia::embedding(out->data(), index->data(), weight->data(), out->dtype(), seq_len, hidden_size);
    }
#endif

    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
