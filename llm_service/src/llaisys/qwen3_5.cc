#include "llaisys/models/qwen3_5.h"

#include "llaisys_tensor.hpp"
#include "../models/qwen3_5.hpp"

__C {

struct LlaisysQwen3_5Model {
    llaisys::models::Qwen3_5Model *model;
    LlaisysQwen3_5Weights weights_c;
    size_t num_layers;
    size_t n_deltanet;
    size_t n_fullattn;
};

static LlaisysQwen3_5Weights wrapWeights(llaisys::models::Qwen3_5Weights &w,
                                           size_t nl, size_t n_dn, size_t n_fa) {
    LlaisysQwen3_5Weights wc;
    wc.in_embed = new LlaisysTensor{w.embed_tokens};
    wc.out_embed = new LlaisysTensor{w.lm_head};
    wc.out_norm_w = new LlaisysTensor{w.final_norm};

    wc.attn_norm_w = new llaisysTensor_t[nl];
    wc.mlp_norm_w = new llaisysTensor_t[nl];
    wc.mlp_gate_proj = new llaisysTensor_t[nl];
    wc.mlp_up_proj = new llaisysTensor_t[nl];
    wc.mlp_down_proj = new llaisysTensor_t[nl];
    wc.layer_attn_idx = new size_t[nl];

    for (size_t i = 0; i < nl; i++) {
        wc.attn_norm_w[i] = new LlaisysTensor{w.input_layernorm[i]};
        wc.mlp_norm_w[i] = new LlaisysTensor{w.post_attn_layernorm[i]};
        wc.mlp_gate_proj[i] = new LlaisysTensor{w.mlp_gate_proj[i]};
        wc.mlp_up_proj[i] = new LlaisysTensor{w.mlp_up_proj[i]};
        wc.mlp_down_proj[i] = new LlaisysTensor{w.mlp_down_proj[i]};
        wc.layer_attn_idx[i] = w.layer_attn_idx[i];
    }

    // DeltaNet weights
    wc.deltanet = new LlaisysQwen3_5DeltaNetWeights[n_dn];
    for (size_t i = 0; i < n_dn; i++) {
        auto &src = w.deltanet_weights[i];
        auto &dst = wc.deltanet[i];
        dst.qkv_proj = new LlaisysTensor{src.qkv_proj};
        dst.o_proj = new LlaisysTensor{src.o_proj};
        dst.z_proj = new LlaisysTensor{src.z_proj};
        dst.b_proj = new LlaisysTensor{src.b_proj};
        dst.a_proj = new LlaisysTensor{src.a_proj};
        dst.A_log = new LlaisysTensor{src.A_log};
        dst.dt_bias = new LlaisysTensor{src.dt_bias};
        dst.conv_weight = new LlaisysTensor{src.conv_weight};
        dst.norm_weight = new LlaisysTensor{src.norm_weight};
    }

    // Gated attention weights
    wc.gated_attn = new LlaisysQwen3_5GatedAttnWeights[n_fa];
    for (size_t i = 0; i < n_fa; i++) {
        auto &src = w.gated_attn_weights[i];
        auto &dst = wc.gated_attn[i];
        dst.q_proj = new LlaisysTensor{src.q_proj};
        dst.k_proj = new LlaisysTensor{src.k_proj};
        dst.v_proj = new LlaisysTensor{src.v_proj};
        dst.o_proj = new LlaisysTensor{src.o_proj};
        dst.q_norm = new LlaisysTensor{src.q_norm};
        dst.k_norm = new LlaisysTensor{src.k_norm};
    }

    return wc;
}

static void freeWeightsC(LlaisysQwen3_5Weights &wc, size_t nl, size_t n_dn, size_t n_fa) {
    delete wc.in_embed;
    delete wc.out_embed;
    delete wc.out_norm_w;

    for (size_t i = 0; i < nl; i++) {
        delete wc.attn_norm_w[i];
        delete wc.mlp_norm_w[i];
        delete wc.mlp_gate_proj[i];
        delete wc.mlp_up_proj[i];
        delete wc.mlp_down_proj[i];
    }
    delete[] wc.attn_norm_w;
    delete[] wc.mlp_norm_w;
    delete[] wc.mlp_gate_proj;
    delete[] wc.mlp_up_proj;
    delete[] wc.mlp_down_proj;
    delete[] wc.layer_attn_idx;

    for (size_t i = 0; i < n_dn; i++) {
        auto &d = wc.deltanet[i];
        delete d.qkv_proj; delete d.o_proj;
        delete d.z_proj; delete d.b_proj; delete d.a_proj;
        delete d.A_log; delete d.dt_bias;
        delete d.conv_weight;
        delete d.norm_weight;
    }
    delete[] wc.deltanet;

    for (size_t i = 0; i < n_fa; i++) {
        auto &a = wc.gated_attn[i];
        delete a.q_proj; delete a.k_proj; delete a.v_proj; delete a.o_proj;
        delete a.q_norm; delete a.k_norm;
    }
    delete[] wc.gated_attn;
}

static llaisys::models::Qwen3_5Config metaToConfig(const struct LlaisysQwen3_5Meta *meta) {
    llaisys::models::Qwen3_5Config c;
    c.dtype = meta->dtype;
    c.num_layers = meta->num_layers;
    c.hidden_size = meta->hidden_size;
    c.intermediate_size = meta->intermediate_size;
    c.vocab_size = meta->vocab_size;
    c.max_seq_len = meta->max_seq_len;

    c.num_attn_heads = meta->num_attn_heads;
    c.num_kv_heads = meta->num_kv_heads;
    c.attn_head_dim = meta->attn_head_dim;

    c.linear_num_key_heads = meta->linear_num_key_heads;
    c.linear_key_head_dim = meta->linear_key_head_dim;
    c.linear_num_value_heads = meta->linear_num_value_heads;
    c.linear_value_head_dim = meta->linear_value_head_dim;
    c.conv_kernel_size = meta->conv_kernel_size;

    c.rms_norm_eps = meta->rms_norm_eps;
    c.rope_theta = meta->rope_theta;
    c.partial_rotary_factor = meta->partial_rotary_factor;
    c.mrope_section[0] = meta->mrope_section[0];
    c.mrope_section[1] = meta->mrope_section[1];
    c.mrope_section[2] = meta->mrope_section[2];

    c.eos_token_id = meta->eos_token_id;

    // Build layer_types vector
    c.layer_types.resize(meta->num_layers);
    for (size_t i = 0; i < meta->num_layers; i++) {
        c.layer_types[i] = static_cast<llaisys::models::LayerType>(meta->layer_types[i]);
    }

    return c;
}

__export struct LlaisysQwen3_5Model *llaisysQwen3_5ModelCreate(
    const struct LlaisysQwen3_5Meta *meta,
    llaisysDeviceType_t device,
    int device_id) {

    auto config = metaToConfig(meta);
    auto m = new LlaisysQwen3_5Model();
    m->num_layers = config.num_layers;

    // Count layer types
    m->n_deltanet = 0;
    m->n_fullattn = 0;
    for (size_t i = 0; i < config.num_layers; i++) {
        if (config.layer_types[i] == llaisys::models::LayerType::LINEAR_ATTENTION)
            m->n_deltanet++;
        else
            m->n_fullattn++;
    }

    m->model = new llaisys::models::Qwen3_5Model(config, device, device_id);
    m->weights_c = wrapWeights(m->model->weights(), config.num_layers, m->n_deltanet, m->n_fullattn);
    return m;
}

__export void llaisysQwen3_5ModelDestroy(struct LlaisysQwen3_5Model *model) {
    if (!model) return;
    freeWeightsC(model->weights_c, model->num_layers, model->n_deltanet, model->n_fullattn);
    delete model->model;
    delete model;
}

__export struct LlaisysQwen3_5Weights *llaisysQwen3_5ModelWeights(struct LlaisysQwen3_5Model *model) {
    return &model->weights_c;
}

__export int64_t llaisysQwen3_5ModelInfer(struct LlaisysQwen3_5Model *model,
                                           int64_t *token_ids, size_t ntoken) {
    return model->model->infer(token_ids, ntoken);
}

__export int64_t llaisysQwen3_5ModelInferSampled(struct LlaisysQwen3_5Model *model,
                                                   int64_t *token_ids, size_t ntoken,
                                                   float temperature, int top_k,
                                                   float top_p, uint64_t seed) {
    return model->model->infer(token_ids, ntoken, temperature, top_k, top_p, seed);
}

__export void llaisysQwen3_5ModelReset(struct LlaisysQwen3_5Model *model) {
    model->model->resetCache();
}

__export void llaisysQwen3_5ModelSetCacheLen(struct LlaisysQwen3_5Model *model, size_t cache_len) {
    model->model->setCacheLen(cache_len);
}

__export size_t llaisysQwen3_5ModelGetCacheLen(struct LlaisysQwen3_5Model *model) {
    return model->model->cacheLen();
}

__export void llaisysQwen3_5ModelSetProfile(struct LlaisysQwen3_5Model *model, int enabled) {
    model->model->profiler().setEnabled(enabled != 0);
}

__export void llaisysQwen3_5ModelSetRepetitionPenalty(struct LlaisysQwen3_5Model *model, float penalty) {
    model->model->setRepetitionPenalty(penalty);
}

}
