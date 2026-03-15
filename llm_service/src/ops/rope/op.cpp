#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/rope_nvidia.cuh"
#endif

namespace llaisys::ops {

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "pos_ids不是int64");
    ASSERT(out->ndim() == 3 && in->ndim() == 3, "out和in不是3维的");
    ASSERT(pos_ids->ndim() == 1, "pos_ids不是1维的");
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
           "不连续");
    CHECK_SAME_SHAPE(out->shape(), in->shape());

    size_t seq_len = in->shape()[0];
    size_t n_head = in->shape()[1];
    size_t head_dim = in->shape()[2];

    ASSERT(pos_ids->shape()[0] == seq_len, "pos_ids长度不对");
    ASSERT(head_dim % 2 == 0, "head_dim不是偶数");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), pos_ids->data(), theta,
                         out->dtype(), seq_len, n_head, head_dim);
    }

#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
        return nvidia::rope(out->data(), in->data(), pos_ids->data(), theta, out->dtype(), seq_len, n_head, head_dim);
    }
#endif

    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
