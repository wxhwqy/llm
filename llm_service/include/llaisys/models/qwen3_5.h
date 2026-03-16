#ifndef LLAISYS_MODELS_QWEN3_5_H
#define LLAISYS_MODELS_QWEN3_5_H

#include "../tensor.h"

__C {
    struct LlaisysQwen3_5Meta {
        llaisysDataType_t dtype;
        size_t num_layers;
        size_t hidden_size;
        size_t intermediate_size;
        size_t vocab_size;
        size_t max_seq_len;

        // Full attention
        size_t num_attn_heads;
        size_t num_kv_heads;
        size_t attn_head_dim;

        // Linear attention (DeltaNet)
        size_t linear_num_key_heads;
        size_t linear_key_head_dim;
        size_t linear_num_value_heads;
        size_t linear_value_head_dim;
        size_t conv_kernel_size;

        float rms_norm_eps;
        float rope_theta;
        float partial_rotary_factor;
        int mrope_section[3];

        int64_t eos_token_id;

        // layer_types[i]: 0 = LINEAR_ATTENTION, 1 = FULL_ATTENTION
        uint8_t *layer_types;
    };

    // DeltaNet layer weights
    struct LlaisysQwen3_5DeltaNetWeights {
        llaisysTensor_t qkv_proj;      // fused Q+K+V projection
        llaisysTensor_t o_proj;
        llaisysTensor_t z_proj;        // output gate z
        llaisysTensor_t b_proj;        // beta (update gate)
        llaisysTensor_t a_proj;        // alpha (decay gate input)
        llaisysTensor_t A_log;         // F32
        llaisysTensor_t dt_bias;       // F32
        llaisysTensor_t conv_weight;
        llaisysTensor_t norm_weight;   // gated rms norm per head_v_dim
    };

    // Gated Full Attention layer weights
    struct LlaisysQwen3_5GatedAttnWeights {
        llaisysTensor_t q_proj;        // 2x dim (Q + gate)
        llaisysTensor_t k_proj;
        llaisysTensor_t v_proj;
        llaisysTensor_t o_proj;
        llaisysTensor_t q_norm;
        llaisysTensor_t k_norm;
    };

    struct LlaisysQwen3_5Weights {
        llaisysTensor_t in_embed;
        llaisysTensor_t out_embed;
        llaisysTensor_t out_norm_w;

        llaisysTensor_t *attn_norm_w;       // [num_layers]
        llaisysTensor_t *mlp_norm_w;        // [num_layers]
        llaisysTensor_t *mlp_gate_proj;     // [num_layers]
        llaisysTensor_t *mlp_up_proj;       // [num_layers]
        llaisysTensor_t *mlp_down_proj;     // [num_layers]

        struct LlaisysQwen3_5DeltaNetWeights *deltanet;      // [n_deltanet_layers]
        struct LlaisysQwen3_5GatedAttnWeights *gated_attn;   // [n_fullattn_layers]

        // layer_attn_idx[i] = index into deltanet or gated_attn array
        size_t *layer_attn_idx;
    };

    struct LlaisysQwen3_5Model;

    __export struct LlaisysQwen3_5Model *llaisysQwen3_5ModelCreate(
        const struct LlaisysQwen3_5Meta *meta, llaisysDeviceType_t device, int device_id);
    __export void llaisysQwen3_5ModelDestroy(struct LlaisysQwen3_5Model *model);
    __export struct LlaisysQwen3_5Weights *llaisysQwen3_5ModelWeights(struct LlaisysQwen3_5Model *model);
    __export int64_t llaisysQwen3_5ModelInfer(struct LlaisysQwen3_5Model *model, int64_t *token_ids, size_t ntoken);
    __export int64_t llaisysQwen3_5ModelInferSampled(struct LlaisysQwen3_5Model *model, int64_t *token_ids, size_t ntoken,
                                                      float temperature, int top_k, float top_p, uint64_t seed);
    __export void llaisysQwen3_5ModelReset(struct LlaisysQwen3_5Model *model);
    __export void llaisysQwen3_5ModelSetCacheLen(struct LlaisysQwen3_5Model *model, size_t cache_len);
    __export size_t llaisysQwen3_5ModelGetCacheLen(struct LlaisysQwen3_5Model *model);
    __export void llaisysQwen3_5ModelSetProfile(struct LlaisysQwen3_5Model *model, int enabled);
    __export void llaisysQwen3_5ModelSetRepetitionPenalty(struct LlaisysQwen3_5Model *model, float penalty);
}

#endif
