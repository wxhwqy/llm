#include "llaisys/models/qwen3_5_moe.h"

#include "llaisys_tensor.hpp"
#include "../models/qwen3_5_moe.hpp"

__C {

struct LlaisysQwen3_5MoeModel {
    llaisys::models::Qwen3_5MoeModel *model;
    LlaisysQwen3_5MoeWeights weights_c;
    size_t num_layers;
    size_t n_deltanet;
    size_t n_fullattn;
    size_t num_experts;
};

static llaisysTensor_t wrapT(llaisys::tensor_t &t) { return new LlaisysTensor{t}; }

static LlaisysGPTQWeight wrapGPTQ(llaisys::models::GPTQWeight &g) {
    LlaisysGPTQWeight w;
    w.qweight = wrapT(g.qweight);
    w.scales = wrapT(g.scales);
    w.qzeros = wrapT(g.qzeros);
    return w;
}

static LlaisysQwen3_5MoeWeights wrapWeights(llaisys::models::Qwen3_5MoeWeights &w,
                                               size_t nl, size_t n_dn, size_t n_fa,
                                               size_t ne) {
    LlaisysQwen3_5MoeWeights wc;
    wc.in_embed = wrapT(w.embed_tokens);
    wc.out_embed = wrapT(w.lm_head);
    wc.out_norm_w = wrapT(w.final_norm);

    wc.attn_norm_w = new llaisysTensor_t[nl];
    wc.mlp_norm_w = new llaisysTensor_t[nl];
    wc.layer_attn_idx = new size_t[nl];

    for (size_t i = 0; i < nl; i++) {
        wc.attn_norm_w[i] = wrapT(w.input_layernorm[i]);
        wc.mlp_norm_w[i] = wrapT(w.post_attn_layernorm[i]);
        wc.layer_attn_idx[i] = w.layer_attn_idx[i];
    }

    // MoE blocks
    wc.moe = new LlaisysMoeBlock[nl];
    for (size_t i = 0; i < nl; i++) {
        auto &src = w.moe_blocks[i];
        auto &dst = wc.moe[i];
        dst.router = wrapT(src.router);
        dst.shared_expert_gate = wrapT(src.shared_expert_gate);
        dst.shared_expert.gate_proj = wrapT(src.shared_expert.gate_proj);
        dst.shared_expert.up_proj = wrapT(src.shared_expert.up_proj);
        dst.shared_expert.down_proj = wrapT(src.shared_expert.down_proj);

        dst.experts = new LlaisysMoeExpert[ne];
        for (size_t e = 0; e < ne; e++) {
            dst.experts[e].gate_proj = wrapGPTQ(src.experts[e].gate_proj);
            dst.experts[e].up_proj = wrapGPTQ(src.experts[e].up_proj);
            dst.experts[e].down_proj = wrapGPTQ(src.experts[e].down_proj);
        }
    }

    // DeltaNet weights
    wc.deltanet = new LlaisysQwen3_5MoeDeltaNetWeights[n_dn];
    for (size_t i = 0; i < n_dn; i++) {
        auto &src = w.deltanet_weights[i];
        auto &dst = wc.deltanet[i];
        dst.qkv_proj = wrapT(src.qkv_proj);
        dst.o_proj = wrapT(src.o_proj);
        dst.z_proj = wrapT(src.z_proj);
        dst.b_proj = wrapT(src.b_proj);
        dst.a_proj = wrapT(src.a_proj);
        dst.A_log = wrapT(src.A_log);
        dst.dt_bias = wrapT(src.dt_bias);
        dst.conv_weight = wrapT(src.conv_weight);
        dst.norm_weight = wrapT(src.norm_weight);
    }

    // Gated attention weights
    wc.gated_attn = new LlaisysQwen3_5MoeGatedAttnWeights[n_fa];
    for (size_t i = 0; i < n_fa; i++) {
        auto &src = w.gated_attn_weights[i];
        auto &dst = wc.gated_attn[i];
        dst.q_proj = wrapT(src.q_proj);
        dst.k_proj = wrapT(src.k_proj);
        dst.v_proj = wrapT(src.v_proj);
        dst.o_proj = wrapT(src.o_proj);
        dst.q_norm = wrapT(src.q_norm);
        dst.k_norm = wrapT(src.k_norm);
    }

    return wc;
}

static void freeWeightsC(LlaisysQwen3_5MoeWeights &wc, size_t nl, size_t n_dn, size_t n_fa, size_t ne) {
    delete wc.in_embed;
    delete wc.out_embed;
    delete wc.out_norm_w;

    for (size_t i = 0; i < nl; i++) {
        delete wc.attn_norm_w[i];
        delete wc.mlp_norm_w[i];
    }
    delete[] wc.attn_norm_w;
    delete[] wc.mlp_norm_w;
    delete[] wc.layer_attn_idx;

    for (size_t i = 0; i < nl; i++) {
        auto &m = wc.moe[i];
        delete m.router;
        delete m.shared_expert_gate;
        delete m.shared_expert.gate_proj;
        delete m.shared_expert.up_proj;
        delete m.shared_expert.down_proj;

        for (size_t e = 0; e < ne; e++) {
            auto &ex = m.experts[e];
            delete ex.gate_proj.qweight; delete ex.gate_proj.scales; delete ex.gate_proj.qzeros;
            delete ex.up_proj.qweight; delete ex.up_proj.scales; delete ex.up_proj.qzeros;
            delete ex.down_proj.qweight; delete ex.down_proj.scales; delete ex.down_proj.qzeros;
        }
        delete[] m.experts;
    }
    delete[] wc.moe;

    for (size_t i = 0; i < n_dn; i++) {
        auto &d = wc.deltanet[i];
        delete d.qkv_proj; delete d.o_proj;
        delete d.z_proj; delete d.b_proj; delete d.a_proj;
        delete d.A_log; delete d.dt_bias;
        delete d.conv_weight; delete d.norm_weight;
    }
    delete[] wc.deltanet;

    for (size_t i = 0; i < n_fa; i++) {
        auto &a = wc.gated_attn[i];
        delete a.q_proj; delete a.k_proj; delete a.v_proj; delete a.o_proj;
        delete a.q_norm; delete a.k_norm;
    }
    delete[] wc.gated_attn;
}

static llaisys::models::Qwen3_5MoeConfig metaToConfig(const struct LlaisysQwen3_5MoeMeta *meta) {
    llaisys::models::Qwen3_5MoeConfig c;
    c.dtype = meta->dtype;
    c.num_layers = meta->num_layers;
    c.hidden_size = meta->hidden_size;
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

    c.num_experts = meta->num_experts;
    c.num_experts_per_tok = meta->num_experts_per_tok;
    c.moe_intermediate_size = meta->moe_intermediate_size;
    c.shared_expert_intermediate_size = meta->shared_expert_intermediate_size;

    c.gptq_bits = meta->gptq_bits;
    c.gptq_group_size = meta->gptq_group_size;

    c.rms_norm_eps = meta->rms_norm_eps;
    c.rope_theta = meta->rope_theta;
    c.partial_rotary_factor = meta->partial_rotary_factor;
    c.mrope_section[0] = meta->mrope_section[0];
    c.mrope_section[1] = meta->mrope_section[1];
    c.mrope_section[2] = meta->mrope_section[2];

    c.eos_token_id = meta->eos_token_id;

    c.layer_types.resize(meta->num_layers);
    for (size_t i = 0; i < meta->num_layers; i++)
        c.layer_types[i] = static_cast<llaisys::models::MoeLayerType>(meta->layer_types[i]);

    return c;
}

__export struct LlaisysQwen3_5MoeModel *llaisysQwen3_5MoeModelCreate(
    const struct LlaisysQwen3_5MoeMeta *meta,
    llaisysDeviceType_t device,
    int device_id) {

    auto config = metaToConfig(meta);
    auto m = new LlaisysQwen3_5MoeModel();
    m->num_layers = config.num_layers;
    m->num_experts = config.num_experts;

    m->n_deltanet = 0;
    m->n_fullattn = 0;
    for (size_t i = 0; i < config.num_layers; i++) {
        if (config.layer_types[i] == llaisys::models::MoeLayerType::LINEAR_ATTENTION)
            m->n_deltanet++;
        else
            m->n_fullattn++;
    }

    m->model = new llaisys::models::Qwen3_5MoeModel(config, device, device_id);
    m->weights_c = wrapWeights(m->model->weights(), config.num_layers,
                                m->n_deltanet, m->n_fullattn, config.num_experts);
    return m;
}

__export void llaisysQwen3_5MoeModelDestroy(struct LlaisysQwen3_5MoeModel *model) {
    if (!model) return;
    freeWeightsC(model->weights_c, model->num_layers, model->n_deltanet,
                  model->n_fullattn, model->num_experts);
    delete model->model;
    delete model;
}

__export struct LlaisysQwen3_5MoeWeights *llaisysQwen3_5MoeModelWeights(struct LlaisysQwen3_5MoeModel *model) {
    return &model->weights_c;
}

__export int64_t llaisysQwen3_5MoeModelInfer(struct LlaisysQwen3_5MoeModel *model,
                                               int64_t *token_ids, size_t ntoken) {
    return model->model->infer(token_ids, ntoken);
}

__export int64_t llaisysQwen3_5MoeModelInferSampled(struct LlaisysQwen3_5MoeModel *model,
                                                       int64_t *token_ids, size_t ntoken,
                                                       float temperature, int top_k,
                                                       float top_p, uint64_t seed) {
    return model->model->infer(token_ids, ntoken, temperature, top_k, top_p, seed);
}

__export void llaisysQwen3_5MoeModelReset(struct LlaisysQwen3_5MoeModel *model) {
    model->model->resetCache();
}

__export void llaisysQwen3_5MoeModelSetCacheLen(struct LlaisysQwen3_5MoeModel *model, size_t cache_len) {
    model->model->setCacheLen(cache_len);
}

__export size_t llaisysQwen3_5MoeModelGetCacheLen(struct LlaisysQwen3_5MoeModel *model) {
    return model->model->cacheLen();
}

__export void llaisysQwen3_5MoeModelSetProfile(struct LlaisysQwen3_5MoeModel *model, int enabled) {
    model->model->profiler().setEnabled(enabled != 0);
}

__export void llaisysQwen3_5MoeModelSetRepetitionPenalty(struct LlaisysQwen3_5MoeModel *model, float penalty) {
    model->model->setRepetitionPenalty(penalty);
}

}
