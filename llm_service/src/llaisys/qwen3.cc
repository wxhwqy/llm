#include "llaisys/models/qwen3.h"

#include "llaisys_tensor.hpp"
#include "../models/qwen3.hpp"
#ifdef ENABLE_NVIDIA_API
#include "../models/qwen3_tp.hpp"
#endif

__C {

struct LlaisysQwen3Model {
    llaisys::models::Qwen3Model *model;
#ifdef ENABLE_NVIDIA_API
    llaisys::models::Qwen3ModelTP *tp_model;
#else
    void *tp_model;
#endif
    bool is_tp;
    int tp_size;
    LlaisysQwen3Weights weights_c;
    std::vector<LlaisysQwen3Weights> tp_weights_c;
    size_t num_layers;
};

static LlaisysQwen3FP8Linear wrapFP8Linear(const llaisys::models::Qwen3FP8Linear &src) {
    LlaisysQwen3FP8Linear dst;
    dst.weight_fp8 = new LlaisysTensor{src.weight_fp8};
    dst.scale_inv = src.scale_inv ? new LlaisysTensor{src.scale_inv} : nullptr;
    return dst;
}

static void freeFP8Linear(LlaisysQwen3FP8Linear &l) {
    delete l.weight_fp8;
    if (l.scale_inv) delete l.scale_inv;
}

static LlaisysQwen3Weights wrapWeightsSingle(llaisys::models::Qwen3Weights &w, size_t nl) {
    LlaisysQwen3Weights wc;
    wc.in_embed = new LlaisysTensor{w.embed_tokens};
    wc.out_embed = new LlaisysTensor{w.lm_head};
    wc.out_norm_w = new LlaisysTensor{w.final_norm};
    wc.attn_norm_w = new llaisysTensor_t[nl];
    wc.mlp_norm_w = new llaisysTensor_t[nl];
    wc.q_norm_w = new llaisysTensor_t[nl];
    wc.k_norm_w = new llaisysTensor_t[nl];
    wc.q_proj = new LlaisysQwen3FP8Linear[nl];
    wc.k_proj = new LlaisysQwen3FP8Linear[nl];
    wc.v_proj = new LlaisysQwen3FP8Linear[nl];
    wc.o_proj = new LlaisysQwen3FP8Linear[nl];
    wc.gate_proj = new LlaisysQwen3FP8Linear[nl];
    wc.up_proj = new LlaisysQwen3FP8Linear[nl];
    wc.down_proj = new LlaisysQwen3FP8Linear[nl];
    for (size_t i = 0; i < nl; i++) {
        wc.attn_norm_w[i] = new LlaisysTensor{w.input_layernorm[i]};
        wc.mlp_norm_w[i] = new LlaisysTensor{w.post_attn_layernorm[i]};
        wc.q_norm_w[i] = new LlaisysTensor{w.q_norm_weight[i]};
        wc.k_norm_w[i] = new LlaisysTensor{w.k_norm_weight[i]};
        wc.q_proj[i] = wrapFP8Linear(w.q_proj[i]);
        wc.k_proj[i] = wrapFP8Linear(w.k_proj[i]);
        wc.v_proj[i] = wrapFP8Linear(w.v_proj[i]);
        wc.o_proj[i] = wrapFP8Linear(w.o_proj[i]);
        wc.gate_proj[i] = wrapFP8Linear(w.gate_proj[i]);
        wc.up_proj[i] = wrapFP8Linear(w.up_proj[i]);
        wc.down_proj[i] = wrapFP8Linear(w.down_proj[i]);
    }
    return wc;
}

#ifdef ENABLE_NVIDIA_API
static LlaisysQwen3Weights wrapWeightsTP(llaisys::models::Qwen3TPDeviceState &d, size_t nl) {
    LlaisysQwen3Weights wc;
    wc.in_embed = new LlaisysTensor{d.embed_tokens};
    wc.out_embed = new LlaisysTensor{d.lm_head};
    wc.out_norm_w = new LlaisysTensor{d.final_norm};
    wc.attn_norm_w = new llaisysTensor_t[nl];
    wc.mlp_norm_w = new llaisysTensor_t[nl];
    wc.q_norm_w = new llaisysTensor_t[nl];
    wc.k_norm_w = new llaisysTensor_t[nl];
    wc.q_proj = new LlaisysQwen3FP8Linear[nl];
    wc.k_proj = new LlaisysQwen3FP8Linear[nl];
    wc.v_proj = new LlaisysQwen3FP8Linear[nl];
    wc.o_proj = new LlaisysQwen3FP8Linear[nl];
    wc.gate_proj = new LlaisysQwen3FP8Linear[nl];
    wc.up_proj = new LlaisysQwen3FP8Linear[nl];
    wc.down_proj = new LlaisysQwen3FP8Linear[nl];
    for (size_t i = 0; i < nl; i++) {
        wc.attn_norm_w[i] = new LlaisysTensor{d.input_layernorm[i]};
        wc.mlp_norm_w[i] = new LlaisysTensor{d.post_attn_layernorm[i]};
        wc.q_norm_w[i] = new LlaisysTensor{d.q_norm_weight[i]};
        wc.k_norm_w[i] = new LlaisysTensor{d.k_norm_weight[i]};
        wc.q_proj[i] = wrapFP8Linear(d.q_proj[i]);
        wc.k_proj[i] = wrapFP8Linear(d.k_proj[i]);
        wc.v_proj[i] = wrapFP8Linear(d.v_proj[i]);
        wc.o_proj[i] = wrapFP8Linear(d.o_proj[i]);
        wc.gate_proj[i] = wrapFP8Linear(d.gate_proj[i]);
        wc.up_proj[i] = wrapFP8Linear(d.up_proj[i]);
        wc.down_proj[i] = wrapFP8Linear(d.down_proj[i]);
    }
    return wc;
}
#endif

static void freeWeightsC(LlaisysQwen3Weights &wc, size_t nl) {
    delete wc.in_embed;
    delete wc.out_embed;
    delete wc.out_norm_w;
    for (size_t i = 0; i < nl; i++) {
        delete wc.attn_norm_w[i];
        delete wc.mlp_norm_w[i];
        delete wc.q_norm_w[i];
        delete wc.k_norm_w[i];
        freeFP8Linear(wc.q_proj[i]);
        freeFP8Linear(wc.k_proj[i]);
        freeFP8Linear(wc.v_proj[i]);
        freeFP8Linear(wc.o_proj[i]);
        freeFP8Linear(wc.gate_proj[i]);
        freeFP8Linear(wc.up_proj[i]);
        freeFP8Linear(wc.down_proj[i]);
    }
    delete[] wc.attn_norm_w;
    delete[] wc.mlp_norm_w;
    delete[] wc.q_norm_w;
    delete[] wc.k_norm_w;
    delete[] wc.q_proj;
    delete[] wc.k_proj;
    delete[] wc.v_proj;
    delete[] wc.o_proj;
    delete[] wc.gate_proj;
    delete[] wc.up_proj;
    delete[] wc.down_proj;
}

static llaisys::models::Qwen3Config metaToConfig(const struct LlaisysQwen3Meta *meta) {
    llaisys::models::Qwen3Config c;
    c.dtype = meta->dtype;
    c.num_layers = meta->nlayer;
    c.hidden_size = meta->hs;
    c.num_heads = meta->nh;
    c.num_kv_heads = meta->nkvh;
    c.head_dim = meta->dh;
    c.intermediate_size = meta->di;
    c.max_seq_len = meta->maxseq;
    c.vocab_size = meta->voc;
    c.rms_norm_eps = meta->epsilon;
    c.rope_theta = meta->theta;
    c.eos_token_id = meta->end_token;
    c.use_fp8 = meta->use_fp8 != 0;
    c.fp8_block_h = meta->fp8_block_h;
    c.fp8_block_w = meta->fp8_block_w;
    return c;
}

__export struct LlaisysQwen3Model *llaisysQwen3ModelCreate(
    const struct LlaisysQwen3Meta *meta,
    llaisysDeviceType_t device,
    int *device_ids,
    int ndevice) {

    auto config = metaToConfig(meta);
    auto m = new LlaisysQwen3Model();
    m->num_layers = config.num_layers;

#ifdef ENABLE_NVIDIA_API
    if (ndevice > 1 && device == LLAISYS_DEVICE_NVIDIA) {
        m->is_tp = true;
        m->tp_size = ndevice;
        m->model = nullptr;
        m->tp_model = new llaisys::models::Qwen3ModelTP(config, device, device_ids, ndevice);

        m->tp_weights_c.resize(ndevice);
        for (int i = 0; i < ndevice; i++) {
            m->tp_weights_c[i] = wrapWeightsTP(m->tp_model->devices()[i], config.num_layers);
        }
        return m;
    }
#endif

    m->is_tp = false;
    m->tp_size = 1;
    m->tp_model = nullptr;
    int dev_id = (ndevice > 0 && device_ids) ? device_ids[0] : 0;
    m->model = new llaisys::models::Qwen3Model(config, device, dev_id);
    m->weights_c = wrapWeightsSingle(m->model->weights(), config.num_layers);
    return m;
}

__export void llaisysQwen3ModelDestroy(struct LlaisysQwen3Model *model) {
    if (!model) return;
    size_t nl = model->num_layers;

    if (model->is_tp) {
        for (auto &wc : model->tp_weights_c) freeWeightsC(wc, nl);
#ifdef ENABLE_NVIDIA_API
        delete model->tp_model;
#endif
    } else {
        freeWeightsC(model->weights_c, nl);
        delete model->model;
    }
    delete model;
}

__export struct LlaisysQwen3Weights *llaisysQwen3ModelWeights(struct LlaisysQwen3Model *model) {
    if (model->is_tp) {
        return &model->tp_weights_c[0];
    }
    return &model->weights_c;
}

__export int llaisysQwen3ModelTPSize(struct LlaisysQwen3Model *model) {
    return model->tp_size;
}

__export struct LlaisysQwen3Weights *llaisysQwen3ModelTPWeights(struct LlaisysQwen3Model *model, int dev_idx) {
    if (model->is_tp && dev_idx < model->tp_size) {
        return &model->tp_weights_c[dev_idx];
    }
    return &model->weights_c;
}

__export int64_t llaisysQwen3ModelInfer(struct LlaisysQwen3Model *model, int64_t *token_ids, size_t ntoken) {
    if (model->is_tp) {
#ifdef ENABLE_NVIDIA_API
        return model->tp_model->infer(token_ids, ntoken);
#else
        return -1;
#endif
    }
    return model->model->infer(token_ids, ntoken);
}

__export int64_t llaisysQwen3ModelInferSampled(struct LlaisysQwen3Model *model, int64_t *token_ids, size_t ntoken,
                                                float temperature, int top_k, float top_p, uint64_t seed) {
    if (model->is_tp) {
#ifdef ENABLE_NVIDIA_API
        return model->tp_model->infer(token_ids, ntoken, temperature, top_k, top_p, seed);
#else
        return -1;
#endif
    }
    return model->model->infer(token_ids, ntoken, temperature, top_k, top_p, seed);
}

__export void llaisysQwen3ModelReset(struct LlaisysQwen3Model *model) {
    if (model->is_tp) {
#ifdef ENABLE_NVIDIA_API
        model->tp_model->resetCache();
#endif
    } else {
        model->model->resetCache();
    }
}

__export void llaisysQwen3ModelSetCacheLen(struct LlaisysQwen3Model *model, size_t cache_len) {
    if (model->is_tp) {
#ifdef ENABLE_NVIDIA_API
        model->tp_model->setCacheLen(cache_len);
#endif
    } else {
        model->model->setCacheLen(cache_len);
    }
}

__export size_t llaisysQwen3ModelGetCacheLen(struct LlaisysQwen3Model *model) {
    if (model->is_tp) {
#ifdef ENABLE_NVIDIA_API
        return model->tp_model->cacheLen();
#else
        return 0;
#endif
    }
    return model->model->cacheLen();
}

__export void llaisysQwen3ModelSetProfile(struct LlaisysQwen3Model *model, int enabled) {
    bool e = (enabled != 0);
    if (model->is_tp) {
#ifdef ENABLE_NVIDIA_API
        model->tp_model->profiler().setEnabled(e);
#endif
    } else {
        model->model->profiler().setEnabled(e);
    }
}

}
