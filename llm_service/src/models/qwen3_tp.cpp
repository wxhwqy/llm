#include "qwen3_tp.hpp"
#include "../core/llaisys_core.hpp"
#include "../utils.hpp"
#include <cmath>
#include <algorithm>

namespace llaisys::models {

static Qwen3FP8Linear createFP8LinearTP(size_t out_dim, size_t in_dim,
                                         size_t bh, size_t bw,
                                         llaisysDeviceType_t dev_type, int dev_id) {
    Qwen3FP8Linear l;
    l.weight_fp8 = Tensor::create({out_dim, in_dim}, LLAISYS_DTYPE_F8, dev_type, dev_id);
    size_t sh = (out_dim + bh - 1) / bh;
    size_t sw = (in_dim + bw - 1) / bw;
    l.scale_inv = Tensor::create({sh, sw}, LLAISYS_DTYPE_F32, dev_type, dev_id);
    return l;
}

Qwen3ModelTP::Qwen3ModelTP(const Qwen3Config &config, llaisysDeviceType_t device_type,
                           const int *device_ids, int ndevice)
    : config_(config), device_type_(device_type), tp_size_(ndevice), cache_len_(0) {

    device_ids_.assign(device_ids, device_ids + ndevice);
    nh_per_tp_ = config_.num_heads / tp_size_;
    nkvh_per_tp_ = config_.num_kv_heads / tp_size_;
    di_per_tp_ = config_.intermediate_size / tp_size_;

    nccl_.init(tp_size_, device_ids_.data());

    devs_.resize(tp_size_);
    for (int i = 0; i < tp_size_; i++) {
        devs_[i].device_id = device_ids_[i];
        allocateDeviceState(i);
    }
}

void Qwen3ModelTP::allocateDeviceState(int dev_idx) {
    auto &d = devs_[dev_idx];
    int dev_id = d.device_id;
    core::context().setDevice(device_type_, dev_id);

    size_t nl = config_.num_layers;
    size_t hs = config_.hidden_size;
    size_t dh = config_.head_dim;
    size_t voc = config_.vocab_size;
    size_t maxseq = config_.max_seq_len;
    size_t bh = config_.fp8_block_h;
    size_t bw = config_.fp8_block_w;
    auto dtype = config_.dtype;

    d.embed_tokens = Tensor::create({voc, hs}, dtype, device_type_, dev_id);
    d.lm_head = Tensor::create({voc, hs}, dtype, device_type_, dev_id);
    d.final_norm = Tensor::create({hs}, dtype, device_type_, dev_id);

    d.input_layernorm.resize(nl);
    d.post_attn_layernorm.resize(nl);
    d.q_norm_weight.resize(nl);
    d.k_norm_weight.resize(nl);
    d.q_proj.resize(nl);
    d.k_proj.resize(nl);
    d.v_proj.resize(nl);
    d.o_proj.resize(nl);
    d.gate_proj.resize(nl);
    d.up_proj.resize(nl);
    d.down_proj.resize(nl);

    for (size_t i = 0; i < nl; i++) {
        d.input_layernorm[i] = Tensor::create({hs}, dtype, device_type_, dev_id);
        d.post_attn_layernorm[i] = Tensor::create({hs}, dtype, device_type_, dev_id);
        d.q_norm_weight[i] = Tensor::create({dh}, dtype, device_type_, dev_id);
        d.k_norm_weight[i] = Tensor::create({dh}, dtype, device_type_, dev_id);

        if (config_.use_fp8) {
            d.q_proj[i] = createFP8LinearTP(nh_per_tp_ * dh, hs, bh, bw, device_type_, dev_id);
            d.k_proj[i] = createFP8LinearTP(nkvh_per_tp_ * dh, hs, bh, bw, device_type_, dev_id);
            d.v_proj[i] = createFP8LinearTP(nkvh_per_tp_ * dh, hs, bh, bw, device_type_, dev_id);
            d.o_proj[i] = createFP8LinearTP(hs, nh_per_tp_ * dh, bh, bw, device_type_, dev_id);
            d.gate_proj[i] = createFP8LinearTP(di_per_tp_, hs, bh, bw, device_type_, dev_id);
            d.up_proj[i] = createFP8LinearTP(di_per_tp_, hs, bh, bw, device_type_, dev_id);
            d.down_proj[i] = createFP8LinearTP(hs, di_per_tp_, bh, bw, device_type_, dev_id);
        } else {
            d.q_proj[i] = {Tensor::create({nh_per_tp_ * dh, hs}, dtype, device_type_, dev_id), nullptr};
            d.k_proj[i] = {Tensor::create({nkvh_per_tp_ * dh, hs}, dtype, device_type_, dev_id), nullptr};
            d.v_proj[i] = {Tensor::create({nkvh_per_tp_ * dh, hs}, dtype, device_type_, dev_id), nullptr};
            d.o_proj[i] = {Tensor::create({hs, nh_per_tp_ * dh}, dtype, device_type_, dev_id), nullptr};
            d.gate_proj[i] = {Tensor::create({di_per_tp_, hs}, dtype, device_type_, dev_id), nullptr};
            d.up_proj[i] = {Tensor::create({di_per_tp_, hs}, dtype, device_type_, dev_id), nullptr};
            d.down_proj[i] = {Tensor::create({hs, di_per_tp_}, dtype, device_type_, dev_id), nullptr};
        }
    }

    // KV cache
    d.kv_cache.resize(nl);
    for (size_t i = 0; i < nl; i++) {
        d.kv_cache[i].resize(2);
        d.kv_cache[i][0] = Tensor::create({maxseq, nkvh_per_tp_, dh}, dtype, device_type_, dev_id);
        d.kv_cache[i][1] = Tensor::create({maxseq, nkvh_per_tp_, dh}, dtype, device_type_, dev_id);
    }

    // Buffers
    d.hidden_states = Tensor::create({maxseq, hs}, dtype, device_type_, dev_id);
    d.residual = Tensor::create({maxseq, hs}, dtype, device_type_, dev_id);
    d.normed = Tensor::create({maxseq, hs}, dtype, device_type_, dev_id);
    d.q_out = Tensor::create({maxseq, nh_per_tp_ * dh}, dtype, device_type_, dev_id);
    d.k_out = Tensor::create({maxseq, nkvh_per_tp_ * dh}, dtype, device_type_, dev_id);
    d.v_out = Tensor::create({maxseq, nkvh_per_tp_ * dh}, dtype, device_type_, dev_id);
    d.q_normed = Tensor::create({maxseq, nh_per_tp_, dh}, dtype, device_type_, dev_id);
    d.k_normed = Tensor::create({maxseq, nkvh_per_tp_, dh}, dtype, device_type_, dev_id);
    d.q_rope = Tensor::create({maxseq, nh_per_tp_, dh}, dtype, device_type_, dev_id);
    d.k_rope = Tensor::create({maxseq, nkvh_per_tp_, dh}, dtype, device_type_, dev_id);
    d.attn_out = Tensor::create({maxseq, nh_per_tp_, dh}, dtype, device_type_, dev_id);
    d.o_proj_out = Tensor::create({maxseq, hs}, dtype, device_type_, dev_id);
    d.gate_out = Tensor::create({maxseq, di_per_tp_}, dtype, device_type_, dev_id);
    d.up_out = Tensor::create({maxseq, di_per_tp_}, dtype, device_type_, dev_id);
    d.mlp_out = Tensor::create({maxseq, hs}, dtype, device_type_, dev_id);

    if (dev_idx == 0) {
        d.logits = Tensor::create({1, voc}, dtype, device_type_, dev_id);
        d.max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, device_type_, dev_id);
        d.max_val = Tensor::create({1}, dtype, device_type_, dev_id);
        d.sample_workspace = Tensor::create({voc}, LLAISYS_DTYPE_F32, device_type_, dev_id);
    }

    // No large dequant buffer needed.
}

void Qwen3ModelTP::resetCache() {
    cache_len_ = 0;
}

void Qwen3ModelTP::linearFP8(int dev_idx, tensor_t out, tensor_t in,
                              const Qwen3FP8Linear &fp8, size_t rows, size_t cols) {
    (void)dev_idx; (void)rows; (void)cols;
    if (config_.use_fp8 && fp8.scale_inv) {
        ops::linear_fp8(out, in, fp8.weight_fp8, fp8.scale_inv,
                        config_.fp8_block_h, config_.fp8_block_w);
    } else {
        ops::linear(out, in, fp8.weight_fp8, nullptr);
    }
}

void Qwen3ModelTP::allReduceHidden(size_t seq_len) {
    size_t count = seq_len * config_.hidden_size;
    nccl_.groupStart();
    for (int i = 0; i < tp_size_; i++) {
        cudaSetDevice(device_ids_[i]);
        auto hidden_view = devs_[i].hidden_states->slice(0, 0, seq_len);
        nccl_.allReduceSumBf16(hidden_view->data(), count, i);
    }
    nccl_.groupEnd();
    nccl_.syncAll();
}

void Qwen3ModelTP::forwardLayer(size_t layer_idx, size_t seq_len, size_t start_pos) {
    size_t hs = config_.hidden_size;
    size_t dh = config_.head_dim;

    // Phase 1: Local ops on each device (QKV -> Attention -> O_proj)
    for (int di = 0; di < tp_size_; di++) {
        auto &d = devs_[di];
        core::context().setDevice(device_type_, d.device_id);
        auto api = core::context().runtime().api();

        auto hidden_view = d.hidden_states->slice(0, 0, seq_len);
        auto residual_view = d.residual->slice(0, 0, seq_len);
        auto normed_view = d.normed->slice(0, 0, seq_len);

        api->memcpy_sync(residual_view->data(), hidden_view->data(),
                         seq_len * hs * hidden_view->elementSize(), LLAISYS_MEMCPY_D2D);

        ops::rms_norm(normed_view, hidden_view, d.input_layernorm[layer_idx], config_.rms_norm_eps);

        // Column-parallel QKV
        auto q_view = d.q_out->slice(0, 0, seq_len);
        auto k_view = d.k_out->slice(0, 0, seq_len);
        auto v_view = d.v_out->slice(0, 0, seq_len);

        linearFP8(di, q_view, normed_view, d.q_proj[layer_idx], nh_per_tp_ * dh, hs);
        linearFP8(di, k_view, normed_view, d.k_proj[layer_idx], nkvh_per_tp_ * dh, hs);
        linearFP8(di, v_view, normed_view, d.v_proj[layer_idx], nkvh_per_tp_ * dh, hs);

        // QK-Norm
        auto q_reshaped = q_view->view({seq_len, nh_per_tp_, dh});
        auto k_reshaped = k_view->view({seq_len, nkvh_per_tp_, dh});
        auto v_reshaped = v_view->view({seq_len, nkvh_per_tp_, dh});

        auto q_normed_view = d.q_normed->slice(0, 0, seq_len);
        auto k_normed_view = d.k_normed->slice(0, 0, seq_len);

        auto q_flat = q_reshaped->view({seq_len * nh_per_tp_, dh});
        auto qn_flat = q_normed_view->view({seq_len * nh_per_tp_, dh});
        ops::rms_norm(qn_flat, q_flat, d.q_norm_weight[layer_idx], config_.rms_norm_eps);

        auto k_flat = k_reshaped->view({seq_len * nkvh_per_tp_, dh});
        auto kn_flat = k_normed_view->view({seq_len * nkvh_per_tp_, dh});
        ops::rms_norm(kn_flat, k_flat, d.k_norm_weight[layer_idx], config_.rms_norm_eps);

        // RoPE
        auto q_rope_view = d.q_rope->slice(0, 0, seq_len);
        auto k_rope_view = d.k_rope->slice(0, 0, seq_len);

        auto pos_ids = Tensor::create({seq_len}, LLAISYS_DTYPE_I64, device_type_, d.device_id);
        std::vector<int64_t> pos_data(seq_len);
        for (size_t j = 0; j < seq_len; j++) pos_data[j] = (int64_t)(start_pos + j);
        pos_ids->load(pos_data.data());

        ops::rope(q_rope_view, q_normed_view, pos_ids, config_.rope_theta);
        ops::rope(k_rope_view, k_normed_view, pos_ids, config_.rope_theta);

        // KV cache
        size_t kv_bytes = seq_len * nkvh_per_tp_ * dh * q_rope_view->elementSize();
        std::byte *kc = d.kv_cache[layer_idx][0]->data() + start_pos * nkvh_per_tp_ * dh * q_rope_view->elementSize();
        std::byte *vc = d.kv_cache[layer_idx][1]->data() + start_pos * nkvh_per_tp_ * dh * v_reshaped->elementSize();
        api->memcpy_sync(kc, k_rope_view->data(), kv_bytes, LLAISYS_MEMCPY_D2D);
        api->memcpy_sync(vc, v_reshaped->data(), kv_bytes, LLAISYS_MEMCPY_D2D);

        // Self-attention
        size_t total_len = start_pos + seq_len;
        auto k_cache_view = d.kv_cache[layer_idx][0]->slice(0, 0, total_len);
        auto v_cache_view = d.kv_cache[layer_idx][1]->slice(0, 0, total_len);

        auto attn_view = d.attn_out->slice(0, 0, seq_len);
        float scale = 1.0f / std::sqrt((float)dh);
        ops::self_attention(attn_view, q_rope_view, k_cache_view, v_cache_view, scale);

        // Row-parallel O_proj (output goes into o_proj_out which is [seq, hs])
        auto attn_flat = attn_view->view({seq_len, nh_per_tp_ * dh});
        auto o_proj_view = d.o_proj_out->slice(0, 0, seq_len);
        linearFP8(di, o_proj_view, attn_flat, d.o_proj[layer_idx], hs, nh_per_tp_ * dh);
    }

    // AllReduce o_proj_out -> write result into hidden_states (as partial sum)
    // First copy o_proj_out into hidden_states for in-place AllReduce
    for (int di = 0; di < tp_size_; di++) {
        auto &d = devs_[di];
        core::context().setDevice(device_type_, d.device_id);
        auto api = core::context().runtime().api();
        auto h = d.hidden_states->slice(0, 0, seq_len);
        auto o = d.o_proj_out->slice(0, 0, seq_len);
        api->memcpy_sync(h->data(), o->data(), seq_len * hs * h->elementSize(), LLAISYS_MEMCPY_D2D);
    }
    allReduceHidden(seq_len);

    // Residual add: hidden = residual + allreduced(o_proj)
    for (int di = 0; di < tp_size_; di++) {
        auto &d = devs_[di];
        core::context().setDevice(device_type_, d.device_id);
        auto h = d.hidden_states->slice(0, 0, seq_len);
        auto r = d.residual->slice(0, 0, seq_len);
        ops::add(h, r, h);
    }

    // Phase 2: MLP
    for (int di = 0; di < tp_size_; di++) {
        auto &d = devs_[di];
        core::context().setDevice(device_type_, d.device_id);
        auto api = core::context().runtime().api();

        auto hidden_view = d.hidden_states->slice(0, 0, seq_len);
        auto residual_view = d.residual->slice(0, 0, seq_len);
        auto normed_view = d.normed->slice(0, 0, seq_len);

        api->memcpy_sync(residual_view->data(), hidden_view->data(),
                         seq_len * hs * hidden_view->elementSize(), LLAISYS_MEMCPY_D2D);

        ops::rms_norm(normed_view, hidden_view, d.post_attn_layernorm[layer_idx], config_.rms_norm_eps);

        // Column-parallel gate/up
        auto gate_view = d.gate_out->slice(0, 0, seq_len);
        auto up_view = d.up_out->slice(0, 0, seq_len);
        auto mlp_view = d.mlp_out->slice(0, 0, seq_len);

        linearFP8(di, gate_view, normed_view, d.gate_proj[layer_idx], di_per_tp_, hs);
        linearFP8(di, up_view, normed_view, d.up_proj[layer_idx], di_per_tp_, hs);

        ops::swiglu(gate_view, gate_view, up_view);

        // Row-parallel down
        linearFP8(di, mlp_view, gate_view, d.down_proj[layer_idx], hs, di_per_tp_);
    }

    // AllReduce mlp_out -> hidden_states
    for (int di = 0; di < tp_size_; di++) {
        auto &d = devs_[di];
        core::context().setDevice(device_type_, d.device_id);
        auto api = core::context().runtime().api();
        auto h = d.hidden_states->slice(0, 0, seq_len);
        auto m = d.mlp_out->slice(0, 0, seq_len);
        api->memcpy_sync(h->data(), m->data(), seq_len * hs * h->elementSize(), LLAISYS_MEMCPY_D2D);
    }
    allReduceHidden(seq_len);

    // Residual add
    for (int di = 0; di < tp_size_; di++) {
        auto &d = devs_[di];
        core::context().setDevice(device_type_, d.device_id);
        auto h = d.hidden_states->slice(0, 0, seq_len);
        auto r = d.residual->slice(0, 0, seq_len);
        ops::add(h, r, h);
    }
}

int64_t Qwen3ModelTP::infer(const int64_t *token_ids, size_t num_tokens,
                            float temperature, int top_k, float top_p,
                            uint64_t seed) {
    size_t voc = config_.vocab_size;
    size_t start_pos = cache_len_;

    // Embedding on all devices (replicated)
    for (int di = 0; di < tp_size_; di++) {
        auto &d = devs_[di];
        core::context().setDevice(device_type_, d.device_id);

        auto input_ids = Tensor::create({num_tokens}, LLAISYS_DTYPE_I64, device_type_, d.device_id);
        input_ids->load(token_ids);
        auto h = d.hidden_states->slice(0, 0, num_tokens);
        ops::embedding(h, input_ids, d.embed_tokens);
    }

    for (size_t layer = 0; layer < config_.num_layers; layer++) {
        forwardLayer(layer, num_tokens, start_pos);
    }

    // Final norm + lm_head + sampling on device 0
    auto &d0 = devs_[0];
    core::context().setDevice(device_type_, d0.device_id);

    auto hidden_view = d0.hidden_states->slice(0, 0, num_tokens);
    auto last_hidden = hidden_view->slice(0, num_tokens - 1, num_tokens);
    auto normed_view = d0.normed->slice(0, 0, 1);
    ops::rms_norm(normed_view, last_hidden, d0.final_norm, config_.rms_norm_eps);

    ops::linear(d0.logits, normed_view, d0.lm_head, nullptr);

    auto logits_flat = d0.logits->view({voc});

    bool use_sampling = (temperature > 0.0f) && (top_k != 1);
    if (use_sampling) {
        ops::sample(d0.max_idx, logits_flat, d0.sample_workspace,
                    temperature, top_k, top_p, seed);
    } else {
        ops::argmax(d0.max_idx, d0.max_val, logits_flat);
    }

    int64_t next_token;
    auto api = core::context().runtime().api();
    api->memcpy_sync(&next_token, d0.max_idx->data(), sizeof(int64_t), LLAISYS_MEMCPY_D2H);

    cache_len_ = start_pos + num_tokens;
    return next_token;
}

} // namespace llaisys::models
