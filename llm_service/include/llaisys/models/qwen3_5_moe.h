#ifndef LLAISYS_MODELS_QWEN3_5_MOE_H
#define LLAISYS_MODELS_QWEN3_5_MOE_H

#include "../tensor.h"

__C {
    struct LlaisysQwen3_5MoeMeta {
        llaisysDataType_t dtype;
        size_t num_layers;
        size_t hidden_size;
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

        // MoE
        size_t num_experts;
        size_t num_experts_per_tok;
        size_t moe_intermediate_size;
        size_t shared_expert_intermediate_size;

        // GPTQ
        int gptq_bits;
        int gptq_group_size;

        float rms_norm_eps;
        float rope_theta;
        float partial_rotary_factor;
        int mrope_section[3];

        int64_t eos_token_id;

        // layer_types[i]: 0 = LINEAR_ATTENTION, 1 = FULL_ATTENTION
        uint8_t *layer_types;
    };

    // GPTQ quantized linear weight (packed INT4)
    struct LlaisysGPTQWeight {
        llaisysTensor_t qweight;   // INT32 [in/8, out]
        llaisysTensor_t scales;    // BF16 [groups, out]
        llaisysTensor_t qzeros;    // INT32 [groups, out/8]
    };

    // Single MoE expert (GPTQ quantized)
    struct LlaisysMoeExpert {
        struct LlaisysGPTQWeight gate_proj;
        struct LlaisysGPTQWeight up_proj;
        struct LlaisysGPTQWeight down_proj;
    };

    // Shared expert (BF16, not quantized)
    struct LlaisysMoeSharedExpert {
        llaisysTensor_t gate_proj;
        llaisysTensor_t up_proj;
        llaisysTensor_t down_proj;
    };

    // MoE block per layer
    struct LlaisysMoeBlock {
        llaisysTensor_t router;                          // [num_experts, hidden_size]
        llaisysTensor_t shared_expert_gate;              // [1, hidden_size]
        struct LlaisysMoeSharedExpert shared_expert;
        struct LlaisysMoeExpert *experts;                // [num_experts]
    };

    // Reuse DeltaNet and GatedAttn weight structs from qwen3_5.h
    struct LlaisysQwen3_5MoeDeltaNetWeights {
        llaisysTensor_t qkv_proj;
        llaisysTensor_t o_proj;
        llaisysTensor_t z_proj;
        llaisysTensor_t b_proj;
        llaisysTensor_t a_proj;
        llaisysTensor_t A_log;
        llaisysTensor_t dt_bias;
        llaisysTensor_t conv_weight;
        llaisysTensor_t norm_weight;
    };

    struct LlaisysQwen3_5MoeGatedAttnWeights {
        llaisysTensor_t q_proj;
        llaisysTensor_t k_proj;
        llaisysTensor_t v_proj;
        llaisysTensor_t o_proj;
        llaisysTensor_t q_norm;
        llaisysTensor_t k_norm;
    };

    struct LlaisysQwen3_5MoeWeights {
        llaisysTensor_t in_embed;
        llaisysTensor_t out_embed;
        llaisysTensor_t out_norm_w;

        llaisysTensor_t *attn_norm_w;       // [num_layers]
        llaisysTensor_t *mlp_norm_w;        // [num_layers]

        struct LlaisysMoeBlock *moe;        // [num_layers]

        struct LlaisysQwen3_5MoeDeltaNetWeights *deltanet;
        struct LlaisysQwen3_5MoeGatedAttnWeights *gated_attn;

        size_t *layer_attn_idx;
    };

    struct LlaisysQwen3_5MoeModel;

    __export struct LlaisysQwen3_5MoeModel *llaisysQwen3_5MoeModelCreate(
        const struct LlaisysQwen3_5MoeMeta *meta, llaisysDeviceType_t device, int device_id);
    __export void llaisysQwen3_5MoeModelDestroy(struct LlaisysQwen3_5MoeModel *model);
    __export struct LlaisysQwen3_5MoeWeights *llaisysQwen3_5MoeModelWeights(struct LlaisysQwen3_5MoeModel *model);
    __export int64_t llaisysQwen3_5MoeModelInfer(struct LlaisysQwen3_5MoeModel *model, int64_t *token_ids, size_t ntoken);
    __export int64_t llaisysQwen3_5MoeModelInferSampled(struct LlaisysQwen3_5MoeModel *model, int64_t *token_ids, size_t ntoken,
                                                         float temperature, int top_k, float top_p, uint64_t seed);
    __export void llaisysQwen3_5MoeModelReset(struct LlaisysQwen3_5MoeModel *model);
    __export void llaisysQwen3_5MoeModelSetCacheLen(struct LlaisysQwen3_5MoeModel *model, size_t cache_len);
    __export size_t llaisysQwen3_5MoeModelGetCacheLen(struct LlaisysQwen3_5MoeModel *model);
    __export void llaisysQwen3_5MoeModelSetProfile(struct LlaisysQwen3_5MoeModel *model, int enabled);
    __export void llaisysQwen3_5MoeModelSetRepetitionPenalty(struct LlaisysQwen3_5MoeModel *model, float penalty);
}

#endif
