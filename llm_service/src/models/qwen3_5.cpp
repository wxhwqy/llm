#include "qwen3_5.hpp"

#include "../core/llaisys_core.hpp"
#include "../utils.hpp"

#include <cmath>
#include <cstring>
#include <iostream>
#include <algorithm>

namespace llaisys::models {

Qwen3_5Model::Qwen3_5Model(const Qwen3_5Config &config, llaisysDeviceType_t device_type, int device_id)
    : config_(config), device_type_(device_type), device_id_(device_id), cache_len_(0) {

    core::context().setDevice(device_type_, device_id_);

    size_t nl = config_.num_layers;
    size_t hs = config_.hidden_size;
    size_t voc = config_.vocab_size;
    size_t di = config_.intermediate_size;

    // Full attention params
    size_t nh = config_.num_attn_heads;
    size_t nkvh = config_.num_kv_heads;
    size_t ahd = config_.attn_head_dim;

    // DeltaNet params
    size_t n_kh = config_.linear_num_key_heads;
    size_t dk = config_.linear_key_head_dim;
    size_t n_vh = config_.linear_num_value_heads;
    size_t dv = config_.linear_value_head_dim;
    size_t d_qk = n_kh * dk;           // 2048
    size_t d_v = n_vh * dv;             // 4096
    size_t d_conv = d_qk * 2 + d_v;    // 8192 (Q+K+V through conv)

    // Embeddings
    weights_.embed_tokens = Tensor::create({voc, hs}, config_.dtype, device_type_, device_id_);
    weights_.lm_head = Tensor::create({voc, hs}, config_.dtype, device_type_, device_id_);
    weights_.final_norm = Tensor::create({hs}, config_.dtype, device_type_, device_id_);

    // Per-layer common weights
    weights_.input_layernorm.resize(nl);
    weights_.post_attn_layernorm.resize(nl);
    weights_.mlp_gate_proj.resize(nl);
    weights_.mlp_up_proj.resize(nl);
    weights_.mlp_down_proj.resize(nl);
    weights_.layer_attn_idx.resize(nl);

    // Count layer types
    size_t n_deltanet = 0, n_fullattn = 0;
    for (size_t i = 0; i < nl; i++) {
        if (config_.layer_types[i] == LayerType::LINEAR_ATTENTION) {
            weights_.layer_attn_idx[i] = n_deltanet++;
        } else {
            weights_.layer_attn_idx[i] = n_fullattn++;
        }
    }

    weights_.deltanet_weights.resize(n_deltanet);
    weights_.gated_attn_weights.resize(n_fullattn);

    for (size_t i = 0; i < nl; i++) {
        weights_.input_layernorm[i] = Tensor::create({hs}, config_.dtype, device_type_, device_id_);
        weights_.post_attn_layernorm[i] = Tensor::create({hs}, config_.dtype, device_type_, device_id_);
        weights_.mlp_gate_proj[i] = Tensor::create({di, hs}, config_.dtype, device_type_, device_id_);
        weights_.mlp_up_proj[i] = Tensor::create({di, hs}, config_.dtype, device_type_, device_id_);
        weights_.mlp_down_proj[i] = Tensor::create({hs, di}, config_.dtype, device_type_, device_id_);

        size_t attn_idx = weights_.layer_attn_idx[i];

        if (config_.layer_types[i] == LayerType::LINEAR_ATTENTION) {
            auto &w = weights_.deltanet_weights[attn_idx];
            w.qkv_proj = Tensor::create({d_conv, hs}, config_.dtype, device_type_, device_id_);
            w.o_proj = Tensor::create({hs, d_v}, config_.dtype, device_type_, device_id_);
            w.z_proj = Tensor::create({d_v, hs}, config_.dtype, device_type_, device_id_);
            w.b_proj = Tensor::create({n_vh, hs}, config_.dtype, device_type_, device_id_);
            w.a_proj = Tensor::create({n_vh, hs}, config_.dtype, device_type_, device_id_);
            w.A_log = Tensor::create({n_vh}, LLAISYS_DTYPE_F32, device_type_, device_id_);
            w.dt_bias = Tensor::create({n_vh}, LLAISYS_DTYPE_F32, device_type_, device_id_);
            w.conv_weight = Tensor::create({d_conv, config_.conv_kernel_size}, config_.dtype, device_type_, device_id_);
            w.norm_weight = Tensor::create({dv}, config_.dtype, device_type_, device_id_);
        } else {
            auto &w = weights_.gated_attn_weights[attn_idx];
            // Q proj outputs 2x (Q + gate for sigmoid gating)
            w.q_proj = Tensor::create({2 * nh * ahd, hs}, config_.dtype, device_type_, device_id_);
            w.k_proj = Tensor::create({nkvh * ahd, hs}, config_.dtype, device_type_, device_id_);
            w.v_proj = Tensor::create({nkvh * ahd, hs}, config_.dtype, device_type_, device_id_);
            w.o_proj = Tensor::create({hs, nh * ahd}, config_.dtype, device_type_, device_id_);
            w.q_norm = Tensor::create({ahd}, config_.dtype, device_type_, device_id_);
            w.k_norm = Tensor::create({ahd}, config_.dtype, device_type_, device_id_);
        }
    }

    initCaches();
    allocateBuffers(config_.max_seq_len);
}

void Qwen3_5Model::initCaches() {
    size_t nl = config_.num_layers;
    size_t nkvh = config_.num_kv_heads;
    size_t ahd = config_.attn_head_dim;
    size_t maxseq = config_.max_seq_len;

    size_t n_kh = config_.linear_num_key_heads;
    size_t dk = config_.linear_key_head_dim;
    size_t n_vh = config_.linear_num_value_heads;
    size_t dv = config_.linear_value_head_dim;
    size_t d_conv = n_kh * dk * 2 + n_vh * dv;  // 8192

    size_t n_deltanet = 0, n_fullattn = 0;
    for (size_t i = 0; i < nl; i++) {
        if (config_.layer_types[i] == LayerType::LINEAR_ATTENTION)
            n_deltanet++;
        else
            n_fullattn++;
    }

    // KV cache for full attention layers
    kv_cache_.resize(n_fullattn);
    for (size_t i = 0; i < n_fullattn; i++) {
        kv_cache_[i].resize(2);
        kv_cache_[i][0] = Tensor::create({maxseq, nkvh, ahd}, config_.dtype, device_type_, device_id_);
        kv_cache_[i][1] = Tensor::create({maxseq, nkvh, ahd}, config_.dtype, device_type_, device_id_);
    }

    // DeltaNet states
    conv_states_.resize(n_deltanet);
    recurrent_states_.resize(n_deltanet);
    for (size_t i = 0; i < n_deltanet; i++) {
        conv_states_[i] = Tensor::create({d_conv, config_.conv_kernel_size}, config_.dtype, device_type_, device_id_);
        // Recurrent state: [n_vh, dv, dk] F32 for numerical stability
        recurrent_states_[i] = Tensor::create({n_vh, dv, dk}, LLAISYS_DTYPE_F32, device_type_, device_id_);
    }

    cache_len_ = 0;
}

void Qwen3_5Model::allocateBuffers(size_t max_batch_seq) {
    size_t hs = config_.hidden_size;
    size_t di = config_.intermediate_size;
    size_t voc = config_.vocab_size;

    size_t nh = config_.num_attn_heads;
    size_t nkvh = config_.num_kv_heads;
    size_t ahd = config_.attn_head_dim;

    size_t n_kh = config_.linear_num_key_heads;
    size_t dk = config_.linear_key_head_dim;
    size_t n_vh = config_.linear_num_value_heads;
    size_t dv = config_.linear_value_head_dim;
    size_t d_qk = n_kh * dk;
    size_t d_v = n_vh * dv;
    size_t d_conv = d_qk * 2 + d_v;    // 8192 (Q+K+V through conv)

    hidden_states_ = Tensor::create({max_batch_seq, hs}, config_.dtype, device_type_, device_id_);
    residual_ = Tensor::create({max_batch_seq, hs}, config_.dtype, device_type_, device_id_);
    normed_ = Tensor::create({max_batch_seq, hs}, config_.dtype, device_type_, device_id_);

    // Full attention buffers
    fa_q_out_ = Tensor::create({max_batch_seq, 2 * nh * ahd}, config_.dtype, device_type_, device_id_);
    fa_k_out_ = Tensor::create({max_batch_seq, nkvh * ahd}, config_.dtype, device_type_, device_id_);
    fa_v_out_ = Tensor::create({max_batch_seq, nkvh * ahd}, config_.dtype, device_type_, device_id_);
    fa_q_normed_ = Tensor::create({max_batch_seq, nh, ahd}, config_.dtype, device_type_, device_id_);
    fa_k_normed_ = Tensor::create({max_batch_seq, nkvh, ahd}, config_.dtype, device_type_, device_id_);
    fa_q_rope_ = Tensor::create({max_batch_seq, nh, ahd}, config_.dtype, device_type_, device_id_);
    fa_k_rope_ = Tensor::create({max_batch_seq, nkvh, ahd}, config_.dtype, device_type_, device_id_);
    fa_attn_out_ = Tensor::create({max_batch_seq, nh, ahd}, config_.dtype, device_type_, device_id_);
    fa_o_proj_out_ = Tensor::create({max_batch_seq, hs}, config_.dtype, device_type_, device_id_);

    // DeltaNet buffers
    dn_qkv_out_ = Tensor::create({max_batch_seq, d_conv}, config_.dtype, device_type_, device_id_);
    dn_conv_out_ = Tensor::create({max_batch_seq, d_conv}, config_.dtype, device_type_, device_id_);
    dn_z_out_ = Tensor::create({max_batch_seq, d_v}, config_.dtype, device_type_, device_id_);
    dn_b_out_ = Tensor::create({max_batch_seq, n_vh}, config_.dtype, device_type_, device_id_);
    dn_a_out_ = Tensor::create({max_batch_seq, n_vh}, config_.dtype, device_type_, device_id_);
    dn_q_expanded_ = Tensor::create({max_batch_seq, n_vh, dk}, config_.dtype, device_type_, device_id_);
    dn_k_expanded_ = Tensor::create({max_batch_seq, n_vh, dk}, config_.dtype, device_type_, device_id_);
    dn_g_buf_ = Tensor::create({max_batch_seq, n_vh}, LLAISYS_DTYPE_F32, device_type_, device_id_);
    dn_beta_buf_ = Tensor::create({max_batch_seq, n_vh}, LLAISYS_DTYPE_F32, device_type_, device_id_);
    dn_attn_out_ = Tensor::create({max_batch_seq, d_v}, config_.dtype, device_type_, device_id_);
    dn_normed_out_ = Tensor::create({max_batch_seq, d_v}, config_.dtype, device_type_, device_id_);
    dn_o_proj_out_ = Tensor::create({max_batch_seq, hs}, config_.dtype, device_type_, device_id_);

    // MLP buffers
    gate_out_ = Tensor::create({max_batch_seq, di}, config_.dtype, device_type_, device_id_);
    up_out_ = Tensor::create({max_batch_seq, di}, config_.dtype, device_type_, device_id_);
    mlp_out_ = Tensor::create({max_batch_seq, hs}, config_.dtype, device_type_, device_id_);

    // Sampling
    logits_ = Tensor::create({1, voc}, config_.dtype, device_type_, device_id_);
    max_idx_ = Tensor::create({1}, LLAISYS_DTYPE_I64, device_type_, device_id_);
    max_val_ = Tensor::create({1}, config_.dtype, device_type_, device_id_);
    sample_workspace_ = Tensor::create({voc}, LLAISYS_DTYPE_F32, device_type_, device_id_);

    // Decode
    decode_pos_ids_ = Tensor::create({3, 1}, LLAISYS_DTYPE_I64, device_type_, device_id_);
    decode_input_id_ = Tensor::create({1}, LLAISYS_DTYPE_I64, device_type_, device_id_);
}

void Qwen3_5Model::resetCache() {
    cache_len_ = 0;
    token_history_.clear();
    // Zero out DeltaNet states using memcpy from a zero buffer
    for (auto &s : conv_states_) {
        size_t bytes = s->numel() * s->elementSize();
        std::vector<std::byte> zeros(bytes, std::byte{0});
        auto api = core::context().runtime().api();
        api->memcpy_sync(s->data(), zeros.data(), bytes, LLAISYS_MEMCPY_H2D);
    }
    for (auto &s : recurrent_states_) {
        size_t bytes = s->numel() * s->elementSize();
        std::vector<std::byte> zeros(bytes, std::byte{0});
        auto api = core::context().runtime().api();
        api->memcpy_sync(s->data(), zeros.data(), bytes, LLAISYS_MEMCPY_H2D);
    }
}

void Qwen3_5Model::setCacheLen(size_t len) {
    ASSERT(len <= config_.max_seq_len, "cache_len exceeds max_seq_len");
    cache_len_ = len;
    if (len < token_history_.size()) {
        token_history_.resize(len);
    }
}

void Qwen3_5Model::forwardDeltaNetLayer(size_t layer_idx, size_t attn_idx,
                                          size_t seq_len, size_t start_pos) {
    LLAISYS_NVTX_RANGE("forwardDeltaNet");

    size_t hs = config_.hidden_size;
    size_t n_kh = config_.linear_num_key_heads;
    size_t dk = config_.linear_key_head_dim;
    size_t n_vh = config_.linear_num_value_heads;
    size_t dv = config_.linear_value_head_dim;
    size_t d_qk = n_kh * dk;           // 2048
    size_t d_v = n_vh * dv;             // 4096
    size_t d_conv = d_qk * 2 + d_v;    // 8192
    size_t heads_per_kv = n_vh / n_kh;  // 2

    auto &w = weights_.deltanet_weights[attn_idx];
    core::context().setDevice(device_type_, device_id_);
    auto api = core::context().runtime().api();

    auto hidden_view = hidden_states_->slice(0, 0, seq_len);
    auto residual_view = residual_->slice(0, 0, seq_len);
    auto normed_view = normed_->slice(0, 0, seq_len);

    // Save residual
    api->memcpy_sync(residual_view->data(), hidden_view->data(),
                     seq_len * hs * hidden_view->elementSize(), LLAISYS_MEMCPY_D2D);

    // Pre-attention layernorm
    {
        ScopedOpTimer _t(profiler_, "attn_norm", layer_idx);
        ops::rms_norm(normed_view, hidden_view, weights_.input_layernorm[layer_idx], config_.rms_norm_eps);
    }

    // Projections: fused QKV, z (output gate), b (beta), a (alpha)
    auto qkv_view = dn_qkv_out_->slice(0, 0, seq_len);    // [seq, d_conv]
    auto z_view = dn_z_out_->slice(0, 0, seq_len);         // [seq, d_v]
    auto b_view = dn_b_out_->slice(0, 0, seq_len);         // [seq, n_vh]
    auto a_view = dn_a_out_->slice(0, 0, seq_len);         // [seq, n_vh]

    {
        ScopedOpTimer _t(profiler_, "dn_proj", layer_idx);
        ops::linear(qkv_view, normed_view, w.qkv_proj, nullptr);
        ops::linear(z_view, normed_view, w.z_proj, nullptr);
        ops::linear(b_view, normed_view, w.b_proj, nullptr);
        ops::linear(a_view, normed_view, w.a_proj, nullptr);
    }

    // Causal Conv1d + SiLU on fused QKV
    auto conv_out = dn_conv_out_->slice(0, 0, seq_len);
    {
        ScopedOpTimer _t(profiler_, "conv1d", layer_idx);
        bool is_decode = (seq_len == 1);
        if (is_decode) {
            auto in_col = qkv_view->view({d_conv});
            auto out_col = conv_out->view({d_conv});
            // conv_bias is nullptr (no bias in Qwen3.5 conv1d)
            ops::causal_conv1d_step(out_col, conv_states_[attn_idx], in_col,
                                     w.conv_weight, nullptr,
                                     d_conv, config_.conv_kernel_size);
        } else {
            ops::causal_conv1d(conv_out, qkv_view, w.conv_weight, nullptr,
                               seq_len, d_conv, config_.conv_kernel_size);
            // Initialize conv_state from prefill tail for decode
            // HF: conv_state = F.pad(mixed_qkv, (ks - seq_len, 0)) — stores last ks columns
            // conv_state layout: [d_conv, kernel_size]
            // qkv_view layout: [seq_len, d_conv] (row-major)
            // We need to copy the last ks timesteps, transposed
            {
                size_t ks = config_.conv_kernel_size;
                size_t elem_sz = qkv_view->elementSize();
                size_t start_t = (seq_len >= ks) ? (seq_len - ks) : 0;
                size_t valid = (seq_len >= ks) ? ks : seq_len;
                size_t pad = ks - valid;

                // Zero the entire conv state first
                size_t bytes = conv_states_[attn_idx]->numel() * elem_sz;
                std::vector<std::byte> zeros(bytes, std::byte{0});
                api->memcpy_sync(conv_states_[attn_idx]->data(), zeros.data(), bytes, LLAISYS_MEMCPY_H2D);

                // Copy last 'valid' timesteps transposed: state[ch][pad+i] = qkv[start_t+i][ch]
                for (size_t i = 0; i < valid; i++) {
                    for (size_t ch = 0; ch < d_conv; ch++) {
                        size_t src_off = (start_t + i) * d_conv + ch;
                        size_t dst_off = ch * ks + (pad + i);
                        api->memcpy_sync(
                            conv_states_[attn_idx]->data() + dst_off * elem_sz,
                            qkv_view->data() + src_off * elem_sz,
                            elem_sz, LLAISYS_MEMCPY_D2D);
                    }
                }
            }
        }
    }

    // Split conv output into Q [d_qk], K [d_qk], V [d_v]
    // Then repeat Q/K from n_kh to n_vh heads, and L2-normalize
    auto q_expanded = dn_q_expanded_->slice(0, 0, seq_len);  // [seq, n_vh, dk]
    auto k_expanded = dn_k_expanded_->slice(0, 0, seq_len);  // [seq, n_vh, dk]
    size_t elem_size = conv_out->elementSize();

    // Compute gates: g = exp(-exp(A_log) * softplus(a + dt_bias)), beta = sigmoid(b)
    // We pre-compute these as F32 on host, then load to device
    auto g_view = dn_g_buf_->slice(0, 0, seq_len);
    auto beta_view = dn_beta_buf_->slice(0, 0, seq_len);
    {
        ScopedOpTimer _t(profiler_, "dn_gates", layer_idx);

        // Read A_log, dt_bias from device (F32, small: n_vh each)
        std::vector<float> A_log_h(n_vh), dt_bias_h(n_vh);
        api->memcpy_sync(A_log_h.data(), w.A_log->data(), n_vh * sizeof(float), LLAISYS_MEMCPY_D2H);
        api->memcpy_sync(dt_bias_h.data(), w.dt_bias->data(), n_vh * sizeof(float), LLAISYS_MEMCPY_D2H);

        // Read a and b projections from device
        // a_view, b_view are in compute dtype (BF16), need to convert
        std::vector<float> a_h(seq_len * n_vh), b_h(seq_len * n_vh);
        if (config_.dtype == LLAISYS_DTYPE_BF16) {
            std::vector<uint16_t> raw(seq_len * n_vh);
            api->memcpy_sync(raw.data(), a_view->data(), seq_len * n_vh * sizeof(uint16_t), LLAISYS_MEMCPY_D2H);
            for (size_t i = 0; i < seq_len * n_vh; i++) {
                uint32_t bits = static_cast<uint32_t>(raw[i]) << 16;
                std::memcpy(&a_h[i], &bits, sizeof(float));
            }
            api->memcpy_sync(raw.data(), b_view->data(), seq_len * n_vh * sizeof(uint16_t), LLAISYS_MEMCPY_D2H);
            for (size_t i = 0; i < seq_len * n_vh; i++) {
                uint32_t bits = static_cast<uint32_t>(raw[i]) << 16;
                std::memcpy(&b_h[i], &bits, sizeof(float));
            }
        } else {
            api->memcpy_sync(a_h.data(), a_view->data(), seq_len * n_vh * sizeof(float), LLAISYS_MEMCPY_D2H);
            api->memcpy_sync(b_h.data(), b_view->data(), seq_len * n_vh * sizeof(float), LLAISYS_MEMCPY_D2H);
        }

        // Compute g and beta on host
        std::vector<float> g_h(seq_len * n_vh), beta_h(seq_len * n_vh);
        for (size_t t = 0; t < seq_len; t++) {
            for (size_t h = 0; h < n_vh; h++) {
                size_t idx = t * n_vh + h;
                float a_val = a_h[idx] + dt_bias_h[h];
                // softplus(x) = log(1 + exp(x))
                float sp = (a_val > 20.0f) ? a_val : std::log1p(std::exp(a_val));
                float log_decay = -std::exp(A_log_h[h]) * sp;
                g_h[idx] = std::exp(log_decay);  // actual multiplicative decay factor
                beta_h[idx] = 1.0f / (1.0f + std::exp(-b_h[idx]));  // sigmoid
            }
        }
        api->memcpy_sync(g_view->data(), g_h.data(), seq_len * n_vh * sizeof(float), LLAISYS_MEMCPY_H2D);
        api->memcpy_sync(beta_view->data(), beta_h.data(), seq_len * n_vh * sizeof(float), LLAISYS_MEMCPY_H2D);

    }

    // Split conv output → Q, K, V; expand Q/K to n_vh heads; L2-normalize Q/K
    // conv_out: [seq, d_conv] where d_conv = d_qk + d_qk + d_v
    // V is used directly reshaped: [seq, n_vh, dv]
    auto attn_view = dn_attn_out_->slice(0, 0, seq_len);
    {
        ScopedOpTimer _t(profiler_, "dn_split_expand", layer_idx);

        // We'll do this on host for correctness (data is small per token)
        // For GPU path, this would be a kernel

        // Read conv_out to host, split/expand/normalize, write back
        std::vector<std::byte> conv_host(seq_len * d_conv * elem_size);
        api->memcpy_sync(conv_host.data(), conv_out->data(),
                         seq_len * d_conv * elem_size, LLAISYS_MEMCPY_D2H);

        // Convert to float for processing
        auto bf16_to_f = [](const std::byte *p) -> float {
            uint16_t raw;
            std::memcpy(&raw, p, 2);
            uint32_t bits = static_cast<uint32_t>(raw) << 16;
            float f;
            std::memcpy(&f, &bits, sizeof(float));
            return f;
        };
        auto f_to_bf16 = [](float f, std::byte *p) {
            uint32_t bits;
            std::memcpy(&bits, &f, sizeof(float));
            uint16_t raw = static_cast<uint16_t>(bits >> 16);
            std::memcpy(p, &raw, 2);
        };

        bool is_bf16 = (config_.dtype == LLAISYS_DTYPE_BF16);
        size_t es = elem_size;

        // Prepare expanded Q/K host buffers
        std::vector<std::byte> q_exp_host(seq_len * n_vh * dk * es);
        std::vector<std::byte> k_exp_host(seq_len * n_vh * dk * es);

        for (size_t t = 0; t < seq_len; t++) {
            const std::byte *row = conv_host.data() + t * d_conv * es;
            const std::byte *q_src = row;                      // [d_qk] = [n_kh * dk]
            const std::byte *k_src = row + d_qk * es;          // [d_qk]

            for (size_t kh = 0; kh < n_kh; kh++) {
                // L2 norm of this Q head
                float q_norm_sq = 0.0f, k_norm_sq = 0.0f;
                for (size_t d = 0; d < dk; d++) {
                    float qv = is_bf16 ? bf16_to_f(q_src + (kh * dk + d) * es)
                                        : *reinterpret_cast<const float *>(q_src + (kh * dk + d) * es);
                    float kv = is_bf16 ? bf16_to_f(k_src + (kh * dk + d) * es)
                                        : *reinterpret_cast<const float *>(k_src + (kh * dk + d) * es);
                    q_norm_sq += qv * qv;
                    k_norm_sq += kv * kv;
                }
                float q_inv_norm = 1.0f / (std::sqrt(q_norm_sq) + 1e-6f);
                float k_inv_norm = 1.0f / (std::sqrt(k_norm_sq) + 1e-6f);

                // Scale factor: 1/sqrt(dk), applied after L2 norm (matches HF reference)
                float q_scale = q_inv_norm / std::sqrt(static_cast<float>(dk));

                // Repeat to heads_per_kv output heads and normalize
                for (size_t rep = 0; rep < heads_per_kv; rep++) {
                    size_t vh = kh * heads_per_kv + rep;
                    for (size_t d = 0; d < dk; d++) {
                        float qv = is_bf16 ? bf16_to_f(q_src + (kh * dk + d) * es)
                                            : *reinterpret_cast<const float *>(q_src + (kh * dk + d) * es);
                        float kv = is_bf16 ? bf16_to_f(k_src + (kh * dk + d) * es)
                                            : *reinterpret_cast<const float *>(k_src + (kh * dk + d) * es);
                        size_t out_off = (t * n_vh * dk + vh * dk + d) * es;
                        if (is_bf16) {
                            f_to_bf16(qv * q_scale, q_exp_host.data() + out_off);
                            f_to_bf16(kv * k_inv_norm, k_exp_host.data() + out_off);
                        } else {
                            *reinterpret_cast<float *>(q_exp_host.data() + out_off) = qv * q_scale;
                            *reinterpret_cast<float *>(k_exp_host.data() + out_off) = kv * k_inv_norm;
                        }
                    }
                }
            }
        }

        api->memcpy_sync(q_expanded->data(), q_exp_host.data(),
                         seq_len * n_vh * dk * es, LLAISYS_MEMCPY_H2D);
        api->memcpy_sync(k_expanded->data(), k_exp_host.data(),
                         seq_len * n_vh * dk * es, LLAISYS_MEMCPY_H2D);

        // V is at offset d_qk*2 in conv_out, already [seq, d_v] = [seq, n_vh*dv]
        // We can use it directly from conv_out by pointer offset, but for simplicity
        // copy V into attn_out buffer temporarily (it's the right shape)
        // Actually, we'll just use conv_out with an offset for V
    }

    // V is at conv_out + d_qk*2 per row. Create a view or just pass pointer.
    // We need [seq, n_vh, dv] — conv_out rows are [d_qk, d_qk, d_v]
    // Since d_v = n_vh * dv, V portion is contiguous per-row but interleaved with Q/K across rows.
    // For the delta rule, we need V as a separate contiguous tensor.
    // Copy V portion out:
    {
        // V starts at offset d_qk*2 in each row of conv_out
        size_t v_start = d_qk * 2;
        for (size_t t = 0; t < seq_len; t++) {
            api->memcpy_sync(
                attn_view->data() + t * d_v * elem_size,  // reuse attn_out as temp for V
                conv_out->data() + (t * d_conv + v_start) * elem_size,
                d_v * elem_size, LLAISYS_MEMCPY_D2D);
        }
    }
    // Now attn_view holds V [seq, d_v] = [seq, n_vh, dv]
    auto v_3d = attn_view->view({seq_len, n_vh, dv});

    // Allocate a separate buffer for attention output (can reuse normed_out temporarily)
    auto attn_out = dn_normed_out_->slice(0, 0, seq_len);  // reuse as temp [seq, d_v]
    auto attn_out_3d = attn_out->view({seq_len, n_vh, dv});

    // DeltaNet recurrence with n_vh heads
    {
        ScopedOpTimer _t(profiler_, "delta_rule", layer_idx);
        bool is_decode = (seq_len == 1);
        if (is_decode) {
            auto q_step = q_expanded->view({n_vh, dk});
            auto k_step = k_expanded->view({n_vh, dk});
            auto v_step = v_3d->view({n_vh, dv});
            auto out_step = attn_out_3d->view({n_vh, dv});
            auto g_step = g_view->view({n_vh});
            auto beta_step = beta_view->view({n_vh});

            ops::gated_delta_rule_recurrent(
                out_step, recurrent_states_[attn_idx],
                q_step, k_step, v_step,
                g_step, beta_step,
                n_vh, dk, dv);
        } else {
            ops::gated_delta_rule_chunk(
                attn_out_3d, recurrent_states_[attn_idx],
                q_expanded, k_expanded, v_3d,
                g_view, beta_view->view({seq_len, n_vh}),
                seq_len, n_vh, dk, dv);
        }
    }

    // Gated RMS Norm: reshape to [seq*n_vh, dv], apply norm with gate z
    // z_view: [seq, d_v] → reshape to [seq*n_vh, dv]
    // attn_out: [seq, d_v] → reshape to [seq*n_vh, dv]
    auto normed_out = dn_attn_out_->slice(0, 0, seq_len);  // reuse for normed output [seq, d_v]
    {
        ScopedOpTimer _t(profiler_, "gated_norm", layer_idx);
        auto attn_flat = attn_out->view({seq_len * n_vh, dv});
        auto z_flat = z_view->view({seq_len * n_vh, dv});
        auto normed_flat = normed_out->view({seq_len * n_vh, dv});
        ops::gated_rms_norm(normed_flat, attn_flat, z_flat, w.norm_weight, config_.rms_norm_eps);
    }

    // Output projection
    auto o_proj_view = dn_o_proj_out_->slice(0, 0, seq_len);
    {
        ScopedOpTimer _t(profiler_, "o_proj", layer_idx);
        ops::linear(o_proj_view, normed_out, w.o_proj, nullptr);
    }

    // Residual add
    ops::add(hidden_view, residual_view, o_proj_view);
}

void Qwen3_5Model::forwardGatedAttnLayer(size_t layer_idx, size_t attn_idx,
                                           size_t seq_len, size_t start_pos) {
    LLAISYS_NVTX_RANGE("forwardGatedAttn");

    size_t hs = config_.hidden_size;
    size_t nh = config_.num_attn_heads;
    size_t nkvh = config_.num_kv_heads;
    size_t ahd = config_.attn_head_dim;
    size_t rotary_dim = static_cast<size_t>(ahd * config_.partial_rotary_factor); // 64

    auto &w = weights_.gated_attn_weights[attn_idx];
    core::context().setDevice(device_type_, device_id_);
    auto api = core::context().runtime().api();

    auto hidden_view = hidden_states_->slice(0, 0, seq_len);
    auto residual_view = residual_->slice(0, 0, seq_len);
    auto normed_view = normed_->slice(0, 0, seq_len);

    // Save residual
    api->memcpy_sync(residual_view->data(), hidden_view->data(),
                     seq_len * hs * hidden_view->elementSize(), LLAISYS_MEMCPY_D2D);

    // Pre-attention layernorm
    {
        ScopedOpTimer _t(profiler_, "attn_norm", layer_idx);
        ops::rms_norm(normed_view, hidden_view, weights_.input_layernorm[layer_idx], config_.rms_norm_eps);
    }

    // QKV projection (Q is 2x for gate)
    auto q_full = fa_q_out_->slice(0, 0, seq_len);  // [seq, 2*nh*ahd]
    auto k_view = fa_k_out_->slice(0, 0, seq_len);
    auto v_view = fa_v_out_->slice(0, 0, seq_len);

    {
        ScopedOpTimer _t(profiler_, "qkv_proj", layer_idx);
        ops::linear(q_full, normed_view, w.q_proj, nullptr);
        ops::linear(k_view, normed_view, w.k_proj, nullptr);
        ops::linear(v_view, normed_view, w.v_proj, nullptr);
    }

    // Split Q output into Q and gate
    // HF: q_proj output [seq, 2*nh*ahd] → reshape [seq, nh, 2*ahd] → chunk(2, dim=-1)
    // So per-head layout: [Q_d0..Q_d(ahd-1), gate_d0..gate_d(ahd-1)]
    // Q → reuse fa_q_rope_ as temp (will be overwritten by RoPE later)
    // gate → reuse fa_o_proj_out_ (not needed until after attention; nh*ahd == hs)
    auto q_split = fa_q_rope_->slice(0, 0, seq_len);
    auto gate_buf = fa_o_proj_out_->slice(0, 0, seq_len);
    {
        size_t es = q_full->elementSize();
        // Deinterleave: for each position and head, copy ahd Q elements and ahd gate elements
        for (size_t s = 0; s < seq_len; s++) {
            for (size_t h = 0; h < nh; h++) {
                // In q_full: per head has 2*ahd elements: [Q(ahd), gate(ahd)]
                size_t src_base = (s * 2 * nh * ahd + h * 2 * ahd) * es;
                size_t q_dst = (s * nh * ahd + h * ahd) * es;
                size_t g_dst = (s * nh * ahd + h * ahd) * es;
                // Copy Q part
                api->memcpy_sync(
                    q_split->data() + q_dst,
                    q_full->data() + src_base,
                    ahd * es, LLAISYS_MEMCPY_D2D);
                // Copy gate part
                api->memcpy_sync(
                    gate_buf->data() + g_dst,
                    q_full->data() + src_base + ahd * es,
                    ahd * es, LLAISYS_MEMCPY_D2D);
            }
        }
    }
    auto q_reshaped = q_split->view({seq_len, nh, ahd});
    auto gate_reshaped = gate_buf->view({seq_len, nh, ahd});

    auto k_reshaped = k_view->view({seq_len, nkvh, ahd});
    auto v_reshaped = v_view->view({seq_len, nkvh, ahd});

    // QK-Norm
    auto q_normed_view = fa_q_normed_->slice(0, 0, seq_len);
    auto k_normed_view = fa_k_normed_->slice(0, 0, seq_len);
    {
        ScopedOpTimer _t(profiler_, "qk_norm", layer_idx);
        auto q_flat = q_reshaped->view({seq_len * nh, ahd});
        auto q_normed_flat = q_normed_view->view({seq_len * nh, ahd});
        ops::rms_norm(q_normed_flat, q_flat, w.q_norm, config_.rms_norm_eps);

        auto k_flat = k_reshaped->view({seq_len * nkvh, ahd});
        auto k_normed_flat = k_normed_view->view({seq_len * nkvh, ahd});
        ops::rms_norm(k_normed_flat, k_flat, w.k_norm, config_.rms_norm_eps);
    }

    // M-RoPE
    auto q_rope_view = fa_q_rope_->slice(0, 0, seq_len);
    auto k_rope_view = fa_k_rope_->slice(0, 0, seq_len);

    {
        ScopedOpTimer _t(profiler_, "mrope", layer_idx);

        tensor_t pos_ids;
        if (seq_len == 1) {
            // Decode: single position, all 3 dims same
            int64_t pos_val = static_cast<int64_t>(start_pos);
            int64_t pos_data[3] = {pos_val, pos_val, pos_val};
            decode_pos_ids_->load(pos_data);
            pos_ids = decode_pos_ids_;
        } else {
            // Prefill: sequential positions, all 3 dims same (text-only)
            pos_ids = Tensor::create({3, seq_len}, LLAISYS_DTYPE_I64, device_type_, device_id_);
            std::vector<int64_t> pos_data(3 * seq_len);
            for (size_t d = 0; d < 3; d++) {
                for (size_t i = 0; i < seq_len; i++) {
                    pos_data[d * seq_len + i] = static_cast<int64_t>(start_pos + i);
                }
            }
            pos_ids->load(pos_data.data());
        }

        ops::mrope(q_rope_view, q_normed_view, pos_ids,
                   config_.rope_theta, config_.mrope_section, rotary_dim,
                   seq_len, nh, ahd);
        ops::mrope(k_rope_view, k_normed_view, pos_ids,
                   config_.rope_theta, config_.mrope_section, rotary_dim,
                   seq_len, nkvh, ahd);
    }

    // Update KV cache
    {
        ScopedOpTimer _t(profiler_, "kv_cache_update", layer_idx);
        size_t kv_bytes = seq_len * nkvh * ahd * k_rope_view->elementSize();
        std::byte *k_cache_ptr = kv_cache_[attn_idx][0]->data() + start_pos * nkvh * ahd * k_rope_view->elementSize();
        std::byte *v_cache_ptr = kv_cache_[attn_idx][1]->data() + start_pos * nkvh * ahd * v_reshaped->elementSize();
        api->memcpy_sync(k_cache_ptr, k_rope_view->data(), kv_bytes, LLAISYS_MEMCPY_D2D);
        api->memcpy_sync(v_cache_ptr, v_reshaped->data(), kv_bytes, LLAISYS_MEMCPY_D2D);
    }

    // Gated self-attention
    {
        ScopedOpTimer _t(profiler_, "self_attention", layer_idx);
        size_t total_len = start_pos + seq_len;
        auto k_cache_view = kv_cache_[attn_idx][0]->slice(0, 0, total_len);
        auto v_cache_view = kv_cache_[attn_idx][1]->slice(0, 0, total_len);

        auto attn_view = fa_attn_out_->slice(0, 0, seq_len);
        float scale = 1.0f / std::sqrt(static_cast<float>(ahd));
        ops::self_attention_gated(attn_view, q_rope_view, k_cache_view, v_cache_view,
                                   gate_reshaped, scale);
    }

    // Output projection
    {
        ScopedOpTimer _t(profiler_, "o_proj", layer_idx);
        auto attn_view = fa_attn_out_->slice(0, 0, seq_len);
        auto attn_flat = attn_view->view({seq_len, nh * ahd});
        auto o_proj_view = fa_o_proj_out_->slice(0, 0, seq_len);
        ops::linear(o_proj_view, attn_flat, w.o_proj, nullptr);
    }

    // Residual add
    ops::add(hidden_view, residual_view, fa_o_proj_out_->slice(0, 0, seq_len));
}

void Qwen3_5Model::forwardMLP(size_t layer_idx, size_t seq_len) {
    LLAISYS_NVTX_RANGE("forwardMLP");

    size_t hs = config_.hidden_size;
    auto api = core::context().runtime().api();

    auto hidden_view = hidden_states_->slice(0, 0, seq_len);
    auto residual_view = residual_->slice(0, 0, seq_len);
    auto normed_view = normed_->slice(0, 0, seq_len);

    // Save residual
    api->memcpy_sync(residual_view->data(), hidden_view->data(),
                     seq_len * hs * hidden_view->elementSize(), LLAISYS_MEMCPY_D2D);

    // Post-attention layernorm
    {
        ScopedOpTimer _t(profiler_, "mlp_norm", layer_idx);
        ops::rms_norm(normed_view, hidden_view, weights_.post_attn_layernorm[layer_idx], config_.rms_norm_eps);
    }

    auto gate_view = gate_out_->slice(0, 0, seq_len);
    auto up_view = up_out_->slice(0, 0, seq_len);
    auto mlp_view = mlp_out_->slice(0, 0, seq_len);

    // Gate + Up projection
    {
        ScopedOpTimer _t(profiler_, "gate_up_proj", layer_idx);
        ops::linear(gate_view, normed_view, weights_.mlp_gate_proj[layer_idx], nullptr);
        ops::linear(up_view, normed_view, weights_.mlp_up_proj[layer_idx], nullptr);
    }

    // SwiGLU
    {
        ScopedOpTimer _t(profiler_, "swiglu", layer_idx);
        ops::swiglu(gate_view, gate_view, up_view);
    }

    // Down projection
    {
        ScopedOpTimer _t(profiler_, "down_proj", layer_idx);
        ops::linear(mlp_view, gate_view, weights_.mlp_down_proj[layer_idx], nullptr);
    }

    // Residual add
    ops::add(hidden_view, residual_view, mlp_view);
}

int64_t Qwen3_5Model::infer(const int64_t *token_ids, size_t num_tokens,
                              float temperature, int top_k, float top_p,
                              uint64_t seed) {
    bool is_prefill = (num_tokens > 1);
    LLAISYS_NVTX_RANGE(is_prefill ? "infer_prefill" : "infer_decode");
    profiler_.beginInfer(num_tokens, is_prefill);

    core::context().setDevice(device_type_, device_id_);

    size_t voc = config_.vocab_size;
    size_t start_pos = cache_len_;

    tensor_t input_ids;
    if (num_tokens == 1) {
        decode_input_id_->load(token_ids);
        input_ids = decode_input_id_;
    } else {
        input_ids = Tensor::create({num_tokens}, LLAISYS_DTYPE_I64, device_type_, device_id_);
        input_ids->load(token_ids);
    }

    auto hidden_view = hidden_states_->slice(0, 0, num_tokens);

    {
        ScopedOpTimer _t(profiler_, "embedding", 0);
        LLAISYS_NVTX_RANGE("embedding");
        ops::embedding(hidden_view, input_ids, weights_.embed_tokens);
    }

    for (size_t layer = 0; layer < config_.num_layers; layer++) {
        size_t attn_idx = weights_.layer_attn_idx[layer];

        if (config_.layer_types[layer] == LayerType::LINEAR_ATTENTION) {
            forwardDeltaNetLayer(layer, attn_idx, num_tokens, start_pos);
        } else {
            forwardGatedAttnLayer(layer, attn_idx, num_tokens, start_pos);
        }

        forwardMLP(layer, num_tokens);
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

    // Record input tokens in history
    for (size_t i = 0; i < num_tokens; i++) {
        token_history_.push_back(token_ids[i]);
    }

    {
        ScopedOpTimer _t(profiler_, "sampling", 0);
        LLAISYS_NVTX_RANGE("sampling");
        auto logits_flat = logits_->view({voc});
        bool use_sampling = (temperature > 0.0f) && (top_k != 1);
        if (use_sampling) {
            ops::sample(max_idx_, logits_flat, sample_workspace_,
                        temperature, top_k, top_p, seed,
                        token_history_.data(), token_history_.size(),
                        repetition_penalty_);
        } else {
            ops::argmax(max_idx_, max_val_, logits_flat);
        }
    }

    int64_t next_token;
    auto api = core::context().runtime().api();
    api->memcpy_sync(&next_token, max_idx_->data(), sizeof(int64_t), LLAISYS_MEMCPY_D2H);

    // Record the generated token
    token_history_.push_back(next_token);
    cache_len_ = start_pos + num_tokens;

    profiler_.endInfer();
    return next_token;
}

} // namespace llaisys::models
