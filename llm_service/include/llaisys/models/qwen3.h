#ifndef LLAISYS_MODELS_QWEN3_H
#define LLAISYS_MODELS_QWEN3_H

#include "../tensor.h"

__C {
    struct LlaisysQwen3Meta {
        llaisysDataType_t dtype;
        size_t nlayer, hs, nh, nkvh, dh, di, maxseq, voc;
        float epsilon, theta;
        int64_t end_token;
        uint8_t use_fp8;
        size_t fp8_block_h, fp8_block_w;
    };

    struct LlaisysQwen3FP8Linear {
        llaisysTensor_t weight_fp8;
        llaisysTensor_t scale_inv;
    };

    struct LlaisysQwen3Weights {
        llaisysTensor_t in_embed;
        llaisysTensor_t out_embed;
        llaisysTensor_t out_norm_w;

        llaisysTensor_t *attn_norm_w;
        llaisysTensor_t *mlp_norm_w;

        llaisysTensor_t *q_norm_w;
        llaisysTensor_t *k_norm_w;

        struct LlaisysQwen3FP8Linear *q_proj;
        struct LlaisysQwen3FP8Linear *k_proj;
        struct LlaisysQwen3FP8Linear *v_proj;
        struct LlaisysQwen3FP8Linear *o_proj;
        struct LlaisysQwen3FP8Linear *gate_proj;
        struct LlaisysQwen3FP8Linear *up_proj;
        struct LlaisysQwen3FP8Linear *down_proj;
    };

    struct LlaisysQwen3Model;

    __export struct LlaisysQwen3Model *llaisysQwen3ModelCreate(const struct LlaisysQwen3Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice);
    __export void llaisysQwen3ModelDestroy(struct LlaisysQwen3Model *model);
    __export struct LlaisysQwen3Weights *llaisysQwen3ModelWeights(struct LlaisysQwen3Model *model);
    __export int llaisysQwen3ModelTPSize(struct LlaisysQwen3Model *model);
    __export struct LlaisysQwen3Weights *llaisysQwen3ModelTPWeights(struct LlaisysQwen3Model *model, int dev_idx);
    __export int64_t llaisysQwen3ModelInfer(struct LlaisysQwen3Model *model, int64_t *token_ids, size_t ntoken);
    __export int64_t llaisysQwen3ModelInferSampled(struct LlaisysQwen3Model *model, int64_t *token_ids, size_t ntoken,
                                                    float temperature, int top_k, float top_p, uint64_t seed);
    __export void llaisysQwen3ModelReset(struct LlaisysQwen3Model *model);
    __export void llaisysQwen3ModelSetCacheLen(struct LlaisysQwen3Model *model, size_t cache_len);
    __export size_t llaisysQwen3ModelGetCacheLen(struct LlaisysQwen3Model *model);
    __export void llaisysQwen3ModelSetProfile(struct LlaisysQwen3Model *model, int enabled);
}

#endif
