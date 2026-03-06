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
#include "../ops/dequantize_fp8/op.hpp"

#include <vector>
#include <cstdint>

#include "../utils/profiler.hpp"

namespace llaisys::models {

struct Qwen3Config {
    llaisysDataType_t dtype;        // compute dtype (BF16)
    size_t num_layers;
    size_t hidden_size;
    size_t num_heads;
    size_t num_kv_heads;
    size_t head_dim;
    size_t intermediate_size;
    size_t max_seq_len;
    size_t vocab_size;
    float rms_norm_eps;
    float rope_theta;
    int64_t eos_token_id;
    bool use_fp8;
    size_t fp8_block_h;
    size_t fp8_block_w;
};

struct Qwen3FP8Linear {
    tensor_t weight_fp8;    // [out, in] FP8
    tensor_t scale_inv;     // [ceil(out/block_h), ceil(in/block_w)] F32
};

struct Qwen3Weights {
    tensor_t embed_tokens;      // [vocab_size, hidden_size] BF16
    tensor_t lm_head;           // [vocab_size, hidden_size] BF16
    tensor_t final_norm;        // [hidden_size] BF16

    std::vector<tensor_t> input_layernorm;
    std::vector<tensor_t> post_attn_layernorm;

    // QK-Norm weights
    std::vector<tensor_t> q_norm_weight;    // [nlayer][head_dim]
    std::vector<tensor_t> k_norm_weight;    // [nlayer][head_dim]

    // FP8 linear weights (when use_fp8=true)
    std::vector<Qwen3FP8Linear> q_proj;
    std::vector<Qwen3FP8Linear> k_proj;
    std::vector<Qwen3FP8Linear> v_proj;
    std::vector<Qwen3FP8Linear> o_proj;
    std::vector<Qwen3FP8Linear> gate_proj;
    std::vector<Qwen3FP8Linear> up_proj;
    std::vector<Qwen3FP8Linear> down_proj;

    // BF16 linear weights (when use_fp8=false, for embed/lm_head)
    // embed_tokens and lm_head are always BF16
};

class Qwen3Model {
private:
    Qwen3Config config_;
    Qwen3Weights weights_;
    llaisysDeviceType_t device_type_;
    int device_id_;

    std::vector<std::vector<tensor_t>> kv_cache_;
    size_t cache_len_;

    // Temporary buffers
    tensor_t hidden_states_;
    tensor_t residual_;
    tensor_t normed_;
    tensor_t q_out_, k_out_, v_out_;
    tensor_t q_normed_, k_normed_;  // after QK-Norm
    tensor_t q_rope_, k_rope_;
    tensor_t attn_out_;
    tensor_t o_proj_out_;
    tensor_t gate_out_, up_out_;
    tensor_t mlp_out_;
    tensor_t logits_;
    tensor_t max_idx_, max_val_;
    tensor_t sample_workspace_;

    // Pre-allocated decode buffers (avoid cudaMalloc in hot path)
    tensor_t decode_pos_id_;
    tensor_t decode_input_id_;

    void allocateBuffers(size_t max_batch_seq);
    void initKVCache();
    void forwardLayer(size_t layer_idx, size_t seq_len, size_t start_pos);

    // Run linear using tiled FP8 dequant+GEMM (no large intermediate buffer).
    void linearFP8(tensor_t out, tensor_t in, const Qwen3FP8Linear &fp8,
                   size_t rows, size_t cols);

public:
    Qwen3Model(const Qwen3Config &config, llaisysDeviceType_t device_type, int device_id);
    ~Qwen3Model() = default;

    Qwen3Weights &weights() { return weights_; }
    const Qwen3Config &config() const { return config_; }

    InferProfiler &profiler() { return profiler_; }

    void resetCache();
    int64_t infer(const int64_t *token_ids, size_t num_tokens,
                  float temperature = 0.0f, int top_k = 0, float top_p = 1.0f,
                  uint64_t seed = 0);

private:
    InferProfiler profiler_;
};

} // namespace llaisys::models
