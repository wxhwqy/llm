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
#include "../ops/linear_gptq/op.hpp"
#include "../ops/moe_reduce/op.hpp"

#include <vector>
#include <cstdint>
#include <unordered_set>

#include "../utils/profiler.hpp"

namespace llaisys::models {

// Reuse LayerType from qwen3_5.hpp (or redefine if needed)
enum class MoeLayerType : uint8_t {
    LINEAR_ATTENTION = 0,
    FULL_ATTENTION   = 1,
};

struct Qwen3_5MoeConfig {
    llaisysDataType_t dtype;
    size_t num_layers;
    size_t hidden_size;
    size_t vocab_size;
    size_t max_seq_len;

    // Full attention params
    size_t num_attn_heads;
    size_t num_kv_heads;
    size_t attn_head_dim;

    // Linear attention (DeltaNet) params
    size_t linear_num_key_heads;
    size_t linear_key_head_dim;
    size_t linear_num_value_heads;
    size_t linear_value_head_dim;
    size_t conv_kernel_size;

    // MoE params
    size_t num_experts;
    size_t num_experts_per_tok;
    size_t moe_intermediate_size;
    size_t shared_expert_intermediate_size;

    // GPTQ params
    int gptq_bits;       // 4
    int gptq_group_size; // 128

    // RoPE / M-RoPE
    float rms_norm_eps;
    float rope_theta;
    float partial_rotary_factor;
    int mrope_section[3];

    int64_t eos_token_id;

    std::vector<MoeLayerType> layer_types;
};

// GPTQ packed linear weights
struct GPTQWeight {
    tensor_t qweight;   // INT32 [in_features/8, out_features]
    tensor_t scales;     // BF16 [num_groups, out_features]
    tensor_t qzeros;     // INT32 [num_groups, out_features/8]
};

// Single MoE expert weights (GPTQ quantized)
struct MoeExpertWeights {
    GPTQWeight gate_proj;   // [hidden_size] -> [moe_intermediate_size]
    GPTQWeight up_proj;     // [hidden_size] -> [moe_intermediate_size]
    GPTQWeight down_proj;   // [moe_intermediate_size] -> [hidden_size]
};

// Shared expert weights (BF16)
struct MoeSharedExpertWeights {
    tensor_t gate_proj;     // [shared_intermediate, hidden_size]
    tensor_t up_proj;       // [shared_intermediate, hidden_size]
    tensor_t down_proj;     // [hidden_size, shared_intermediate]
};

// MoE block weights per layer
struct MoeBlockWeights {
    tensor_t router;                            // [num_experts, hidden_size] BF16
    tensor_t shared_expert_gate;                // [1, hidden_size] BF16
    MoeSharedExpertWeights shared_expert;
    std::vector<MoeExpertWeights> experts;      // [num_experts]
};

// DeltaNet layer weights (same as dense model)
struct MoeDeltaNetWeights {
    tensor_t qkv_proj;
    tensor_t o_proj;
    tensor_t z_proj;
    tensor_t b_proj;
    tensor_t a_proj;
    tensor_t A_log;
    tensor_t dt_bias;
    tensor_t conv_weight;
    tensor_t norm_weight;
};

// Gated Full Attention layer weights (same as dense model)
struct MoeGatedAttnWeights {
    tensor_t q_proj;
    tensor_t k_proj;
    tensor_t v_proj;
    tensor_t o_proj;
    tensor_t q_norm;
    tensor_t k_norm;
};

struct Qwen3_5MoeWeights {
    tensor_t embed_tokens;
    tensor_t lm_head;
    tensor_t final_norm;

    std::vector<tensor_t> input_layernorm;
    std::vector<tensor_t> post_attn_layernorm;

    std::vector<MoeBlockWeights> moe_blocks;

    std::vector<MoeDeltaNetWeights> deltanet_weights;
    std::vector<MoeGatedAttnWeights> gated_attn_weights;

    std::vector<size_t> layer_attn_idx;
};

class Qwen3_5MoeModel {
private:
    Qwen3_5MoeConfig config_;
    Qwen3_5MoeWeights weights_;
    llaisysDeviceType_t device_type_;
    int device_id_;

    // KV cache for full attention layers
    std::vector<std::vector<tensor_t>> kv_cache_;
    size_t cache_len_;

    // DeltaNet states
    std::vector<tensor_t> conv_states_;
    std::vector<tensor_t> recurrent_states_;

    // Temporary buffers
    tensor_t hidden_states_;
    tensor_t residual_;
    tensor_t normed_;

    // Full attention buffers
    tensor_t fa_q_out_;
    tensor_t fa_k_out_;
    tensor_t fa_v_out_;
    tensor_t fa_q_normed_;
    tensor_t fa_k_normed_;
    tensor_t fa_q_rope_;
    tensor_t fa_k_rope_;
    tensor_t fa_attn_out_;
    tensor_t fa_gate_buf_;     // [max_seq, nh * ahd] for gated attention gate values
    tensor_t fa_o_proj_out_;

    // DeltaNet buffers
    tensor_t dn_qkv_out_;
    tensor_t dn_conv_out_;
    tensor_t dn_z_out_;
    tensor_t dn_b_out_;
    tensor_t dn_a_out_;
    tensor_t dn_q_expanded_;
    tensor_t dn_k_expanded_;
    tensor_t dn_g_buf_;
    tensor_t dn_beta_buf_;
    tensor_t dn_attn_out_;
    tensor_t dn_normed_out_;
    tensor_t dn_o_proj_out_;

    // MoE buffers
    tensor_t moe_router_logits_;   // [max_seq, num_experts] F32
    tensor_t moe_gate_out_;        // [1, moe_intermediate_size] BF16
    tensor_t moe_up_out_;          // [1, moe_intermediate_size] BF16
    tensor_t moe_expert_out_;      // [1, hidden_size] BF16
    tensor_t moe_accum_;           // [max_seq, hidden_size] F32 (accumulator)
    tensor_t moe_shared_gate_;     // [max_seq, shared_intermediate] BF16
    tensor_t moe_shared_up_;       // [max_seq, shared_intermediate] BF16
    tensor_t moe_shared_out_;      // [max_seq, hidden_size] BF16
    tensor_t moe_dequant_buf_;     // temp buffer for dequantized weight [max_dim, max_dim]

    // Sampling
    tensor_t logits_;
    tensor_t max_idx_, max_val_;
    tensor_t sample_workspace_;

    // Decode buffers
    tensor_t decode_pos_ids_;
    tensor_t decode_input_id_;

    // Repetition penalty
    float repetition_penalty_ = 1.0f;
    std::vector<int64_t> token_history_;
    void applyRepetitionPenalty(tensor_t logits, size_t vocab_size);

    void allocateBuffers(size_t max_batch_seq);
    void initCaches();
    void forwardDeltaNetLayer(size_t layer_idx, size_t attn_idx, size_t seq_len, size_t start_pos);
    void forwardGatedAttnLayer(size_t layer_idx, size_t attn_idx, size_t seq_len, size_t start_pos);
    void forwardMoE(size_t layer_idx, size_t seq_len);

    // GPTQ helpers
    void gptqLinear(tensor_t output, tensor_t input,
                    const GPTQWeight &w,
                    size_t in_features, size_t out_features);

public:
    Qwen3_5MoeModel(const Qwen3_5MoeConfig &config, llaisysDeviceType_t device_type, int device_id);
    ~Qwen3_5MoeModel() = default;

    Qwen3_5MoeWeights &weights() { return weights_; }
    const Qwen3_5MoeConfig &config() const { return config_; }

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
