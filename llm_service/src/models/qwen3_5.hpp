#pragma once

#include "../tensor/tensor.hpp"
#include "../ops/argmax/op.hpp"
#include "../ops/sample/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"
#include "../ops/add/op.hpp"
#include "../ops/sigmoid/op.hpp"
#include "../ops/softplus/op.hpp"
#include "../ops/gated_rms_norm/op.hpp"
#include "../ops/mrope/op.hpp"
#include "../ops/causal_conv1d/op.hpp"
#include "../ops/gated_delta_rule/op.hpp"

#include <vector>
#include <cstdint>
#include <unordered_set>

#include "../utils/profiler.hpp"

namespace llaisys::models {

enum class LayerType : uint8_t {
    LINEAR_ATTENTION = 0,   // DeltaNet
    FULL_ATTENTION   = 1,   // Gated Full Attention
};

struct Qwen3_5Config {
    llaisysDataType_t dtype;        // compute dtype (BF16)
    size_t num_layers;              // 32
    size_t hidden_size;             // 4096
    size_t intermediate_size;       // 12288 (Dense MLP)
    size_t vocab_size;              // 248320
    size_t max_seq_len;             // configurable

    // Full attention params
    size_t num_attn_heads;          // 16
    size_t num_kv_heads;            // 4
    size_t attn_head_dim;           // 256

    // Linear attention (DeltaNet) params
    size_t linear_num_key_heads;    // 16
    size_t linear_key_head_dim;     // 128
    size_t linear_num_value_heads;  // 32
    size_t linear_value_head_dim;   // 128
    size_t conv_kernel_size;        // 4

    // RoPE / M-RoPE
    float rms_norm_eps;             // 1e-6
    float rope_theta;               // 10000000
    float partial_rotary_factor;    // 0.25
    int mrope_section[3];           // {11, 11, 10}

    int64_t eos_token_id;           // 248044

    // Layer type pattern (32 entries)
    std::vector<LayerType> layer_types;
};

// DeltaNet layer weights
struct DeltaNetWeights {
    tensor_t qkv_proj;      // [d_qk*2 + d_v, hidden_size] fused Q+K+V projection
    tensor_t o_proj;         // [hidden_size, d_v]
    tensor_t z_proj;         // [d_v, hidden_size] (output gate z)
    tensor_t b_proj;         // [n_vh, hidden_size] (beta update gate)
    tensor_t a_proj;         // [n_vh, hidden_size] (alpha decay gate input)
    tensor_t A_log;          // [n_vh] learnable log decay (F32)
    tensor_t dt_bias;        // [n_vh] (F32)
    tensor_t conv_weight;    // [d_conv, kernel_size] where d_conv = d_qk*2 + d_v
    tensor_t norm_weight;    // [head_v_dim] gated rms norm per-head
};

// Gated Full Attention layer weights
struct GatedAttnWeights {
    tensor_t q_proj;        // [2 * num_attn_heads * attn_head_dim, hidden_size] (Q + gate)
    tensor_t k_proj;        // [num_kv_heads * attn_head_dim, hidden_size]
    tensor_t v_proj;        // [num_kv_heads * attn_head_dim, hidden_size]
    tensor_t o_proj;        // [hidden_size, num_attn_heads * attn_head_dim]
    tensor_t q_norm;        // [attn_head_dim]
    tensor_t k_norm;        // [attn_head_dim]
};

struct Qwen3_5Weights {
    tensor_t embed_tokens;      // [vocab_size, hidden_size]
    tensor_t lm_head;           // [vocab_size, hidden_size]
    tensor_t final_norm;        // [hidden_size]

    // Per-layer
    std::vector<tensor_t> input_layernorm;       // [num_layers][hidden_size]
    std::vector<tensor_t> post_attn_layernorm;   // [num_layers][hidden_size]

    // MLP weights (shared by all layer types)
    std::vector<tensor_t> mlp_gate_proj;     // [num_layers][intermediate_size, hidden_size]
    std::vector<tensor_t> mlp_up_proj;       // [num_layers][intermediate_size, hidden_size]
    std::vector<tensor_t> mlp_down_proj;     // [num_layers][hidden_size, intermediate_size]

    // Layer-specific attention weights
    std::vector<DeltaNetWeights> deltanet_weights;   // indexed by deltanet layer index
    std::vector<GatedAttnWeights> gated_attn_weights; // indexed by full attn layer index

    // Mapping: layer_idx → index into deltanet_weights or gated_attn_weights
    std::vector<size_t> layer_attn_idx;
};

class Qwen3_5Model {
private:
    Qwen3_5Config config_;
    Qwen3_5Weights weights_;
    llaisysDeviceType_t device_type_;
    int device_id_;

    // KV cache for full attention layers only
    // kv_cache_[attn_layer_idx] = {k_cache, v_cache}  [max_seq, nkvh, hd]
    std::vector<std::vector<tensor_t>> kv_cache_;
    size_t cache_len_;

    // DeltaNet states
    // conv_states_[deltanet_idx] = [d_conv, kernel_size]
    // recurrent_states_[deltanet_idx] = [n_heads, dv_per_head, dk] F32
    std::vector<tensor_t> conv_states_;
    std::vector<tensor_t> recurrent_states_;

    // Temporary buffers
    tensor_t hidden_states_;
    tensor_t residual_;
    tensor_t normed_;

    // Full attention buffers
    tensor_t fa_q_out_;         // includes gate: [seq, 2*nh*hd]
    tensor_t fa_k_out_;
    tensor_t fa_v_out_;
    tensor_t fa_q_normed_;
    tensor_t fa_k_normed_;
    tensor_t fa_q_rope_;
    tensor_t fa_k_rope_;
    tensor_t fa_attn_out_;
    tensor_t fa_o_proj_out_;

    // DeltaNet buffers
    tensor_t dn_qkv_out_;       // [seq, d_qk*2 + d_v] fused QKV output / conv input
    tensor_t dn_conv_out_;      // [seq, d_conv] conv output
    tensor_t dn_z_out_;         // [seq, d_v] output gate z
    tensor_t dn_b_out_;         // [seq, n_vh] beta logit
    tensor_t dn_a_out_;         // [seq, n_vh] alpha logit
    tensor_t dn_q_expanded_;    // [seq, n_vh, dk] Q repeated to n_vh heads
    tensor_t dn_k_expanded_;    // [seq, n_vh, dk] K repeated to n_vh heads
    tensor_t dn_g_buf_;         // [seq, n_vh] pre-computed decay (F32)
    tensor_t dn_beta_buf_;      // [seq, n_vh] pre-computed beta (F32)
    tensor_t dn_attn_out_;      // [seq, n_vh * dv]
    tensor_t dn_normed_out_;    // [seq, n_vh * dv]
    tensor_t dn_o_proj_out_;    // [seq, hidden_size]

    // MLP buffers
    tensor_t gate_out_, up_out_;
    tensor_t mlp_out_;

    // Sampling
    tensor_t logits_;
    tensor_t max_idx_, max_val_;
    tensor_t sample_workspace_;

    // Decode buffers
    tensor_t decode_pos_ids_;   // [3, 1] for M-RoPE
    tensor_t decode_input_id_;

    // Repetition penalty
    float repetition_penalty_ = 1.0f;
    std::vector<int64_t> token_history_;
    void applyRepetitionPenalty(tensor_t logits, size_t vocab_size);

    void allocateBuffers(size_t max_batch_seq);
    void initCaches();
    void forwardDeltaNetLayer(size_t layer_idx, size_t attn_idx, size_t seq_len, size_t start_pos);
    void forwardGatedAttnLayer(size_t layer_idx, size_t attn_idx, size_t seq_len, size_t start_pos);
    void forwardMLP(size_t layer_idx, size_t seq_len);

public:
    Qwen3_5Model(const Qwen3_5Config &config, llaisysDeviceType_t device_type, int device_id);
    ~Qwen3_5Model() = default;

    Qwen3_5Weights &weights() { return weights_; }
    const Qwen3_5Config &config() const { return config_; }

    InferProfiler &profiler() { return profiler_; }

    void resetCache();
    void setCacheLen(size_t len);
    size_t cacheLen() const { return cache_len_; }
    void setRepetitionPenalty(float penalty) { repetition_penalty_ = penalty; }

    int64_t infer(const int64_t *token_ids, size_t num_tokens,
                  float temperature = 0.0f, int top_k = 0, float top_p = 1.0f,
                  uint64_t seed = 0);

private:
    InferProfiler profiler_;
};

} // namespace llaisys::models
