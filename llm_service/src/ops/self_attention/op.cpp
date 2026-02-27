#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/self_attention_nvidia.cuh"
#endif

#include <cmath>
#include <limits>
#include <vector>

namespace llaisys::ops {

template <typename T>
void self_attention_cpu(std::byte *attn_val_ptr, const std::byte *q_ptr, const std::byte *k_ptr,
                        const std::byte *v_ptr, float scale, size_t qlen, size_t kvlen,
                        size_t nhead, size_t nkvhead, size_t hd) {
    T *attn_val = reinterpret_cast<T *>(attn_val_ptr);
    const T *q = reinterpret_cast<const T *>(q_ptr);
    const T *k = reinterpret_cast<const T *>(k_ptr);
    const T *v = reinterpret_cast<const T *>(v_ptr);

    size_t heads_per_kv = nhead / nkvhead;

    for (size_t h = 0; h < nhead; h++) {
        size_t kv_h = h / heads_per_kv;  

        for (size_t qi = 0; qi < qlen; qi++) {
            std::vector<float> scores(kvlen);
            float max_score = -std::numeric_limits<float>::infinity();

            for (size_t ki = 0; ki < kvlen; ki++) {
                size_t causal_limit = qi + (kvlen - qlen);
                if (ki > causal_limit) {
                    scores[ki] = -std::numeric_limits<float>::infinity();
                } else {
                    float dot = 0.0f;
                    for (size_t d = 0; d < hd; d++) {
                        float q_val, k_val;
                        size_t q_idx = qi * nhead * hd + h * hd + d;
                        size_t k_idx = ki * nkvhead * hd + kv_h * hd + d;
                        if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                            q_val = utils::cast<float>(q[q_idx]);
                            k_val = utils::cast<float>(k[k_idx]);
                        } else {
                            q_val = static_cast<float>(q[q_idx]);
                            k_val = static_cast<float>(k[k_idx]);
                        }
                        dot += q_val * k_val;
                    }
                    scores[ki] = dot * scale;
                }
                if (scores[ki] > max_score) {
                    max_score = scores[ki];
                }
            }

            float sum_exp = 0.0f;
            for (size_t ki = 0; ki < kvlen; ki++) {
                if (scores[ki] > -std::numeric_limits<float>::infinity() / 2) {
                    scores[ki] = std::exp(scores[ki] - max_score);
                    sum_exp += scores[ki];
                } else {
                    scores[ki] = 0.0f;
                }
            }
            for (size_t ki = 0; ki < kvlen; ki++) {
                scores[ki] /= sum_exp;
            }

            for (size_t d = 0; d < hd; d++) {
                float out_val = 0.0f;
                for (size_t ki = 0; ki < kvlen; ki++) {
                    if (scores[ki] > 0.0f) {
                        float v_val;
                        size_t v_idx = ki * nkvhead * hd + kv_h * hd + d;
                        if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                            v_val = utils::cast<float>(v[v_idx]);
                        } else {
                            v_val = static_cast<float>(v[v_idx]);
                        }
                        out_val += scores[ki] * v_val;
                    }
                }
                size_t out_idx = qi * nhead * hd + h * hd + d;
                if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                    attn_val[out_idx] = utils::cast<T>(out_val);
                } else {
                    attn_val[out_idx] = static_cast<T>(out_val);
                }
            }
        }
    }
}

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
        switch (attn_val->dtype()) {
        case LLAISYS_DTYPE_F32:
            return self_attention_cpu<float>(attn_val->data(), q->data(), k->data(), v->data(),
                                             scale, qlen, kvlen, nhead, nkvhead, hd);
        case LLAISYS_DTYPE_F16:
            return self_attention_cpu<fp16_t>(attn_val->data(), q->data(), k->data(), v->data(),
                                              scale, qlen, kvlen, nhead, nkvhead, hd);
        case LLAISYS_DTYPE_BF16:
            return self_attention_cpu<bf16_t>(attn_val->data(), q->data(), k->data(), v->data(),
                                              scale, qlen, kvlen, nhead, nkvhead, hd);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(attn_val->dtype());
        }
    }

#ifdef ENABLE_NVIDIA_API
    if (attn_val->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());
        return nvidia::self_attention(attn_val->data(), q->data(), k->data(), v->data(), scale, attn_val->dtype(), qlen, kvlen, nhead, nkvhead, hd);
    }
#endif

    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
