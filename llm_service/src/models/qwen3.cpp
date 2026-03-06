#include "qwen3.hpp"

#include "../core/llaisys_core.hpp"
#include "../utils.hpp"

#include <cmath>
#include <cstring>
#include <iostream>
#include <algorithm>

namespace llaisys::models {

static Qwen3FP8Linear createFP8Linear(size_t out_dim, size_t in_dim,
                                        size_t block_h, size_t block_w,
                                        llaisysDeviceType_t dev, int dev_id) {
    Qwen3FP8Linear l;
    l.weight_fp8 = Tensor::create({out_dim, in_dim}, LLAISYS_DTYPE_F8, dev, dev_id);
    size_t sh = (out_dim + block_h - 1) / block_h;
    size_t sw = (in_dim + block_w - 1) / block_w;
    l.scale_inv = Tensor::create({sh, sw}, LLAISYS_DTYPE_F32, dev, dev_id);
    return l;
}

Qwen3Model::Qwen3Model(const Qwen3Config &config, llaisysDeviceType_t device_type, int device_id)
    : config_(config), device_type_(device_type), device_id_(device_id), cache_len_(0) {

    core::context().setDevice(device_type_, device_id_);

    size_t nl = config_.num_layers;
    size_t hs = config_.hidden_size;
    size_t nh = config_.num_heads;
    size_t nkvh = config_.num_kv_heads;
    size_t dh = config_.head_dim;
    size_t di = config_.intermediate_size;
    size_t voc = config_.vocab_size;
    size_t bh = config_.fp8_block_h;
    size_t bw = config_.fp8_block_w;

    weights_.embed_tokens = Tensor::create({voc, hs}, config_.dtype, device_type_, device_id_);
    weights_.lm_head = Tensor::create({voc, hs}, config_.dtype, device_type_, device_id_);
    weights_.final_norm = Tensor::create({hs}, config_.dtype, device_type_, device_id_);

    weights_.input_layernorm.resize(nl);
    weights_.post_attn_layernorm.resize(nl);
    weights_.q_norm_weight.resize(nl);
    weights_.k_norm_weight.resize(nl);
    weights_.q_proj.resize(nl);
    weights_.k_proj.resize(nl);
    weights_.v_proj.resize(nl);
    weights_.o_proj.resize(nl);
    weights_.gate_proj.resize(nl);
    weights_.up_proj.resize(nl);
    weights_.down_proj.resize(nl);

    for (size_t i = 0; i < nl; i++) {
        weights_.input_layernorm[i] = Tensor::create({hs}, config_.dtype, device_type_, device_id_);
        weights_.post_attn_layernorm[i] = Tensor::create({hs}, config_.dtype, device_type_, device_id_);
        weights_.q_norm_weight[i] = Tensor::create({dh}, config_.dtype, device_type_, device_id_);
        weights_.k_norm_weight[i] = Tensor::create({dh}, config_.dtype, device_type_, device_id_);

        if (config_.use_fp8) {
            weights_.q_proj[i] = createFP8Linear(nh * dh, hs, bh, bw, device_type_, device_id_);
            weights_.k_proj[i] = createFP8Linear(nkvh * dh, hs, bh, bw, device_type_, device_id_);
            weights_.v_proj[i] = createFP8Linear(nkvh * dh, hs, bh, bw, device_type_, device_id_);
            weights_.o_proj[i] = createFP8Linear(hs, nh * dh, bh, bw, device_type_, device_id_);
            weights_.gate_proj[i] = createFP8Linear(di, hs, bh, bw, device_type_, device_id_);
            weights_.up_proj[i] = createFP8Linear(di, hs, bh, bw, device_type_, device_id_);
            weights_.down_proj[i] = createFP8Linear(hs, di, bh, bw, device_type_, device_id_);
        } else {
            // Non-FP8: store as BF16 with null scale_inv
            weights_.q_proj[i] = {Tensor::create({nh * dh, hs}, config_.dtype, device_type_, device_id_), nullptr};
            weights_.k_proj[i] = {Tensor::create({nkvh * dh, hs}, config_.dtype, device_type_, device_id_), nullptr};
            weights_.v_proj[i] = {Tensor::create({nkvh * dh, hs}, config_.dtype, device_type_, device_id_), nullptr};
            weights_.o_proj[i] = {Tensor::create({hs, nh * dh}, config_.dtype, device_type_, device_id_), nullptr};
            weights_.gate_proj[i] = {Tensor::create({di, hs}, config_.dtype, device_type_, device_id_), nullptr};
            weights_.up_proj[i] = {Tensor::create({di, hs}, config_.dtype, device_type_, device_id_), nullptr};
            weights_.down_proj[i] = {Tensor::create({hs, di}, config_.dtype, device_type_, device_id_), nullptr};
        }
    }

    initKVCache();
    allocateBuffers(config_.max_seq_len);
}

void Qwen3Model::initKVCache() {
    size_t nl = config_.num_layers;
    size_t nkvh = config_.num_kv_heads;
    size_t dh = config_.head_dim;
    size_t maxseq = config_.max_seq_len;

    kv_cache_.resize(nl);
    for (size_t i = 0; i < nl; i++) {
        kv_cache_[i].resize(2);
        kv_cache_[i][0] = Tensor::create({maxseq, nkvh, dh}, config_.dtype, device_type_, device_id_);
        kv_cache_[i][1] = Tensor::create({maxseq, nkvh, dh}, config_.dtype, device_type_, device_id_);
    }
    cache_len_ = 0;
}

void Qwen3Model::allocateBuffers(size_t max_batch_seq) {
    size_t hs = config_.hidden_size;
    size_t nh = config_.num_heads;
    size_t nkvh = config_.num_kv_heads;
    size_t dh = config_.head_dim;
    size_t di = config_.intermediate_size;
    size_t voc = config_.vocab_size;

    hidden_states_ = Tensor::create({max_batch_seq, hs}, config_.dtype, device_type_, device_id_);
    residual_ = Tensor::create({max_batch_seq, hs}, config_.dtype, device_type_, device_id_);
    normed_ = Tensor::create({max_batch_seq, hs}, config_.dtype, device_type_, device_id_);

    q_out_ = Tensor::create({max_batch_seq, nh * dh}, config_.dtype, device_type_, device_id_);
    k_out_ = Tensor::create({max_batch_seq, nkvh * dh}, config_.dtype, device_type_, device_id_);
    v_out_ = Tensor::create({max_batch_seq, nkvh * dh}, config_.dtype, device_type_, device_id_);

    q_normed_ = Tensor::create({max_batch_seq, nh, dh}, config_.dtype, device_type_, device_id_);
    k_normed_ = Tensor::create({max_batch_seq, nkvh, dh}, config_.dtype, device_type_, device_id_);

    q_rope_ = Tensor::create({max_batch_seq, nh, dh}, config_.dtype, device_type_, device_id_);
    k_rope_ = Tensor::create({max_batch_seq, nkvh, dh}, config_.dtype, device_type_, device_id_);

    attn_out_ = Tensor::create({max_batch_seq, nh, dh}, config_.dtype, device_type_, device_id_);
    o_proj_out_ = Tensor::create({max_batch_seq, hs}, config_.dtype, device_type_, device_id_);

    gate_out_ = Tensor::create({max_batch_seq, di}, config_.dtype, device_type_, device_id_);
    up_out_ = Tensor::create({max_batch_seq, di}, config_.dtype, device_type_, device_id_);
    mlp_out_ = Tensor::create({max_batch_seq, hs}, config_.dtype, device_type_, device_id_);

    logits_ = Tensor::create({1, voc}, config_.dtype, device_type_, device_id_);
    max_idx_ = Tensor::create({1}, LLAISYS_DTYPE_I64, device_type_, device_id_);
    max_val_ = Tensor::create({1}, config_.dtype, device_type_, device_id_);
    sample_workspace_ = Tensor::create({voc}, LLAISYS_DTYPE_F32, device_type_, device_id_);

    // No large dequant buffer needed: linear_fp8 uses a tiny per-device tile
    // buffer in nvidia::Resource (block_h × max_K × 2 bytes, lazily allocated).
}

void Qwen3Model::resetCache() {
    cache_len_ = 0;
}

void Qwen3Model::linearFP8(tensor_t out, tensor_t in, const Qwen3FP8Linear &fp8,
                            size_t rows, size_t cols) {
    if (config_.use_fp8 && fp8.scale_inv) {
        ops::linear_fp8(out, in, fp8.weight_fp8, fp8.scale_inv,
                        config_.fp8_block_h, config_.fp8_block_w);
    } else {
        ops::linear(out, in, fp8.weight_fp8, nullptr);
    }
}

void Qwen3Model::forwardLayer(size_t layer_idx, size_t seq_len, size_t start_pos) {
    LLAISYS_NVTX_RANGE("forwardLayer");

    size_t hs = config_.hidden_size;
    size_t nh = config_.num_heads;
    size_t nkvh = config_.num_kv_heads;
    size_t dh = config_.head_dim;

    core::context().setDevice(device_type_, device_id_);
    auto api = core::context().runtime().api();

    auto hidden_view = hidden_states_->slice(0, 0, seq_len);
    auto residual_view = residual_->slice(0, 0, seq_len);
    auto normed_view = normed_->slice(0, 0, seq_len);

    api->memcpy_sync(residual_view->data(), hidden_view->data(),
                     seq_len * hs * hidden_view->elementSize(), LLAISYS_MEMCPY_D2D);

    {
        ScopedOpTimer _t(profiler_, "attn_norm", layer_idx);
        LLAISYS_NVTX_RANGE("attn_norm");
        ops::rms_norm(normed_view, hidden_view, weights_.input_layernorm[layer_idx], config_.rms_norm_eps);
    }

    auto q_view = q_out_->slice(0, 0, seq_len);
    auto k_view = k_out_->slice(0, 0, seq_len);
    auto v_view = v_out_->slice(0, 0, seq_len);

    {
        ScopedOpTimer _t(profiler_, "qkv_proj", layer_idx);
        LLAISYS_NVTX_RANGE("qkv_proj");
        linearFP8(q_view, normed_view, weights_.q_proj[layer_idx], nh * dh, hs);
        linearFP8(k_view, normed_view, weights_.k_proj[layer_idx], nkvh * dh, hs);
        linearFP8(v_view, normed_view, weights_.v_proj[layer_idx], nkvh * dh, hs);
    }

    auto q_reshaped = q_view->view({seq_len, nh, dh});
    auto k_reshaped = k_view->view({seq_len, nkvh, dh});
    auto v_reshaped = v_view->view({seq_len, nkvh, dh});

    auto q_normed_view = q_normed_->slice(0, 0, seq_len);
    auto k_normed_view = k_normed_->slice(0, 0, seq_len);

    {
        ScopedOpTimer _t(profiler_, "qk_norm", layer_idx);
        LLAISYS_NVTX_RANGE("qk_norm");
        auto q_flat = q_reshaped->view({seq_len * nh, dh});
        auto q_normed_flat = q_normed_view->view({seq_len * nh, dh});
        ops::rms_norm(q_normed_flat, q_flat, weights_.q_norm_weight[layer_idx], config_.rms_norm_eps);

        auto k_flat = k_reshaped->view({seq_len * nkvh, dh});
        auto k_normed_flat = k_normed_view->view({seq_len * nkvh, dh});
        ops::rms_norm(k_normed_flat, k_flat, weights_.k_norm_weight[layer_idx], config_.rms_norm_eps);
    }

    auto q_rope_view = q_rope_->slice(0, 0, seq_len);
    auto k_rope_view = k_rope_->slice(0, 0, seq_len);

    {
        ScopedOpTimer _t(profiler_, "rope", layer_idx);
        LLAISYS_NVTX_RANGE("rope");
        auto pos_ids = Tensor::create({seq_len}, LLAISYS_DTYPE_I64, device_type_, device_id_);
        std::vector<int64_t> pos_data(seq_len);
        for (size_t i = 0; i < seq_len; i++) {
            pos_data[i] = static_cast<int64_t>(start_pos + i);
        }
        pos_ids->load(pos_data.data());

        ops::rope(q_rope_view, q_normed_view, pos_ids, config_.rope_theta);
        ops::rope(k_rope_view, k_normed_view, pos_ids, config_.rope_theta);
    }

    {
        ScopedOpTimer _t(profiler_, "kv_cache_update", layer_idx);
        LLAISYS_NVTX_RANGE("kv_cache_update");
        size_t kv_bytes = seq_len * nkvh * dh * q_rope_view->elementSize();
        std::byte *k_cache_ptr = kv_cache_[layer_idx][0]->data() + start_pos * nkvh * dh * q_rope_view->elementSize();
        std::byte *v_cache_ptr = kv_cache_[layer_idx][1]->data() + start_pos * nkvh * dh * v_reshaped->elementSize();
        api->memcpy_sync(k_cache_ptr, k_rope_view->data(), kv_bytes, LLAISYS_MEMCPY_D2D);
        api->memcpy_sync(v_cache_ptr, v_reshaped->data(), kv_bytes, LLAISYS_MEMCPY_D2D);
    }

    {
        ScopedOpTimer _t(profiler_, "self_attention", layer_idx);
        LLAISYS_NVTX_RANGE("self_attention");
        size_t total_len = start_pos + seq_len;
        auto k_cache_view = kv_cache_[layer_idx][0]->slice(0, 0, total_len);
        auto v_cache_view = kv_cache_[layer_idx][1]->slice(0, 0, total_len);

        auto attn_view = attn_out_->slice(0, 0, seq_len);
        float scale = 1.0f / std::sqrt(static_cast<float>(dh));
        ops::self_attention(attn_view, q_rope_view, k_cache_view, v_cache_view, scale);
    }

    {
        ScopedOpTimer _t(profiler_, "o_proj", layer_idx);
        LLAISYS_NVTX_RANGE("o_proj");
        auto attn_view = attn_out_->slice(0, 0, seq_len);
        auto attn_flat = attn_view->view({seq_len, hs});
        auto o_proj_view = o_proj_out_->slice(0, 0, seq_len);
        linearFP8(o_proj_view, attn_flat, weights_.o_proj[layer_idx], hs, nh * dh);
    }

    ops::add(hidden_view, residual_view, o_proj_out_->slice(0, 0, seq_len));

    api->memcpy_sync(residual_view->data(), hidden_view->data(),
                     seq_len * hs * hidden_view->elementSize(), LLAISYS_MEMCPY_D2D);

    {
        ScopedOpTimer _t(profiler_, "mlp_norm", layer_idx);
        LLAISYS_NVTX_RANGE("mlp_norm");
        ops::rms_norm(normed_view, hidden_view, weights_.post_attn_layernorm[layer_idx], config_.rms_norm_eps);
    }

    auto gate_view = gate_out_->slice(0, 0, seq_len);
    auto up_view = up_out_->slice(0, 0, seq_len);
    auto mlp_view = mlp_out_->slice(0, 0, seq_len);

    {
        ScopedOpTimer _t(profiler_, "gate_up_proj", layer_idx);
        LLAISYS_NVTX_RANGE("gate_up_proj");
        linearFP8(gate_view, normed_view, weights_.gate_proj[layer_idx], config_.intermediate_size, hs);
        linearFP8(up_view, normed_view, weights_.up_proj[layer_idx], config_.intermediate_size, hs);
    }

    {
        ScopedOpTimer _t(profiler_, "swiglu", layer_idx);
        LLAISYS_NVTX_RANGE("swiglu");
        ops::swiglu(gate_view, gate_view, up_view);
    }

    {
        ScopedOpTimer _t(profiler_, "down_proj", layer_idx);
        LLAISYS_NVTX_RANGE("down_proj");
        linearFP8(mlp_view, gate_view, weights_.down_proj[layer_idx], hs, config_.intermediate_size);
    }

    ops::add(hidden_view, residual_view, mlp_view);
}

int64_t Qwen3Model::infer(const int64_t *token_ids, size_t num_tokens,
                          float temperature, int top_k, float top_p,
                          uint64_t seed) {
    bool is_prefill = (num_tokens > 1);
    LLAISYS_NVTX_RANGE(is_prefill ? "infer_prefill" : "infer_decode");
    profiler_.beginInfer(num_tokens, is_prefill);

    core::context().setDevice(device_type_, device_id_);

    size_t voc = config_.vocab_size;
    size_t start_pos = cache_len_;

    auto input_ids = Tensor::create({num_tokens}, LLAISYS_DTYPE_I64, device_type_, device_id_);
    input_ids->load(token_ids);

    auto hidden_view = hidden_states_->slice(0, 0, num_tokens);

    {
        ScopedOpTimer _t(profiler_, "embedding", 0);
        LLAISYS_NVTX_RANGE("embedding");
        ops::embedding(hidden_view, input_ids, weights_.embed_tokens);
    }

    for (size_t layer = 0; layer < config_.num_layers; layer++) {
        forwardLayer(layer, num_tokens, start_pos);
    }

    {
        ScopedOpTimer _t(profiler_, "final_norm", 0);
        LLAISYS_NVTX_RANGE("final_norm");
        auto last_hidden = hidden_view->slice(0, num_tokens - 1, num_tokens);
        auto normed_last = normed_->slice(0, 0, 1);
        ops::rms_norm(normed_last, last_hidden, weights_.final_norm, config_.rms_norm_eps);
    }

    {
        ScopedOpTimer _t(profiler_, "lm_head", 0);
        LLAISYS_NVTX_RANGE("lm_head");
        auto normed_last = normed_->slice(0, 0, 1);
        ops::linear(logits_, normed_last, weights_.lm_head, nullptr);
    }

    {
        ScopedOpTimer _t(profiler_, "sampling", 0);
        LLAISYS_NVTX_RANGE("sampling");
        auto logits_flat = logits_->view({voc});
        bool use_sampling = (temperature > 0.0f) && (top_k != 1);
        if (use_sampling) {
            ops::sample(max_idx_, logits_flat, sample_workspace_,
                        temperature, top_k, top_p, seed);
        } else {
            ops::argmax(max_idx_, max_val_, logits_flat);
        }
    }

    int64_t next_token;
    auto api = core::context().runtime().api();
    api->memcpy_sync(&next_token, max_idx_->data(), sizeof(int64_t), LLAISYS_MEMCPY_D2H);

    cache_len_ = start_pos + num_tokens;

    profiler_.endInfer();
    return next_token;
}

} // namespace llaisys::models
