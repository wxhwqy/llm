#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/self_attention_nvidia.cuh"
#endif

namespace llaisys::ops {

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    ASSERT(attn_val->ndim() == 3 && q->ndim() == 3 && k->ndim() == 3 && v->ndim() == 3,
           "不是3维的");
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "不连续");

    size_t qlen = q->shape()[0];
    size_t nhead = q->shape()[1];
    size_t hd = q->shape()[2];
    size_t kvlen = k->shape()[0];
    size_t nkvhead = k->shape()[1];

    ASSERT(attn_val->shape()[0] == qlen && attn_val->shape()[1] == nhead && attn_val->shape()[2] == hd,
           "attn_val形状不对");
    ASSERT(k->shape()[2] == hd, "k形状不对");
    ASSERT(v->shape()[0] == kvlen && v->shape()[1] == nkvhead && v->shape()[2] == hd,
           "v形状不对");
    ASSERT(nhead % nkvhead == 0, "nhead不是nkvhead的倍数");

    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                   scale, attn_val->dtype(), qlen, kvlen, nhead, nkvhead, hd);
    }

#ifdef ENABLE_NVIDIA_API
    if (attn_val->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());
        return nvidia::self_attention(attn_val->data(), q->data(), k->data(), v->data(), scale, attn_val->dtype(), qlen, kvlen, nhead, nkvhead, hd);
    }
#endif

    EXCEPTION_UNSUPPORTED_DEVICE;
}

void self_attention_gated(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, tensor_t gate, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    ASSERT(gate->dtype() == attn_val->dtype(), "gate dtype mismatch");
    ASSERT(attn_val->ndim() == 3 && q->ndim() == 3 && k->ndim() == 3 && v->ndim() == 3 && gate->ndim() == 3,
           "不是3维的");
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous() && gate->isContiguous(),
           "不连续");

    size_t qlen = q->shape()[0];
    size_t nhead = q->shape()[1];
    size_t hd = q->shape()[2];
    size_t kvlen = k->shape()[0];
    size_t nkvhead = k->shape()[1];

    ASSERT(attn_val->shape()[0] == qlen && attn_val->shape()[1] == nhead && attn_val->shape()[2] == hd,
           "attn_val形状不对");
    ASSERT(gate->shape()[0] == qlen && gate->shape()[1] == nhead && gate->shape()[2] == hd,
           "gate形状不对");
    ASSERT(k->shape()[2] == hd, "k形状不对");
    ASSERT(v->shape()[0] == kvlen && v->shape()[1] == nkvhead && v->shape()[2] == hd,
           "v形状不对");
    ASSERT(nhead % nkvhead == 0, "nhead不是nkvhead的倍数");

    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention_gated(attn_val->data(), q->data(), k->data(), v->data(),
                                          gate->data(), scale, attn_val->dtype(),
                                          qlen, kvlen, nhead, nkvhead, hd);
    }

#ifdef ENABLE_NVIDIA_API
    if (attn_val->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());
        return nvidia::self_attention_gated(attn_val->data(), q->data(), k->data(), v->data(),
                                             gate->data(), scale, attn_val->dtype(),
                                             qlen, kvlen, nhead, nkvhead, hd);
    }
#endif

    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
