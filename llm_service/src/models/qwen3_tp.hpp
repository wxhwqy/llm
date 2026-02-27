#pragma once

#include "qwen3.hpp"
#include "../device/nvidia/nccl_comm.cuh"

#include <vector>
#include <cstdint>

namespace llaisys::models {

struct Qwen3TPDeviceState {
    int device_id;

    // Sharded weights per layer (column-parallel or row-parallel sized)
    std::vector<tensor_t> input_layernorm;      // replicated [hs]
    std::vector<tensor_t> post_attn_layernorm;  // replicated [hs]
    std::vector<tensor_t> q_norm_weight;        // replicated [dh]
    std::vector<tensor_t> k_norm_weight;        // replicated [dh]

    std::vector<Qwen3FP8Linear> q_proj;   // col-parallel [nh/tp*dh, hs]
    std::vector<Qwen3FP8Linear> k_proj;   // col-parallel [nkvh/tp*dh, hs]
    std::vector<Qwen3FP8Linear> v_proj;   // col-parallel
    std::vector<Qwen3FP8Linear> o_proj;   // row-parallel [hs, nh/tp*dh]
    std::vector<Qwen3FP8Linear> gate_proj;// col-parallel [di/tp, hs]
    std::vector<Qwen3FP8Linear> up_proj;  // col-parallel
    std::vector<Qwen3FP8Linear> down_proj;// row-parallel [hs, di/tp]

    // Global weights (replicated)
    tensor_t embed_tokens;
    tensor_t lm_head;
    tensor_t final_norm;

    // KV cache: [nlayer][2] -> [maxseq, nkvh/tp, dh]
    std::vector<std::vector<tensor_t>> kv_cache;

    // Buffers (sized for this device's shard)
    tensor_t hidden_states;    // [maxseq, hs]
    tensor_t residual;         // [maxseq, hs]
    tensor_t normed;           // [maxseq, hs]
    tensor_t q_out;            // [maxseq, nh/tp*dh]
    tensor_t k_out;            // [maxseq, nkvh/tp*dh]
    tensor_t v_out;            // [maxseq, nkvh/tp*dh]
    tensor_t q_normed;         // [maxseq, nh/tp, dh]
    tensor_t k_normed;         // [maxseq, nkvh/tp, dh]
    tensor_t q_rope;           // [maxseq, nh/tp, dh]
    tensor_t k_rope;           // [maxseq, nkvh/tp, dh]
    tensor_t attn_out;         // [maxseq, nh/tp, dh]
    tensor_t o_proj_out;       // [maxseq, hs]
    tensor_t gate_out;         // [maxseq, di/tp]
    tensor_t up_out;           // [maxseq, di/tp]
    tensor_t mlp_out;          // [maxseq, hs]
    // No large dequant buffer: linear_fp8 uses the tiny tile buffer in nvidia::Resource.
    tensor_t logits;           // [1, voc] only on device 0
    tensor_t max_idx;
    tensor_t max_val;
};

class Qwen3ModelTP {
private:
    Qwen3Config config_;
    llaisysDeviceType_t device_type_;
    int tp_size_;
    std::vector<int> device_ids_;

    llaisys::device::nvidia::NcclComm nccl_;
    std::vector<Qwen3TPDeviceState> devs_;
    size_t cache_len_;

    // TP-derived dimensions
    size_t nh_per_tp_;      // num_heads / tp
    size_t nkvh_per_tp_;    // num_kv_heads / tp
    size_t di_per_tp_;      // intermediate_size / tp

    void allocateDeviceState(int dev_idx);
    void forwardLayer(size_t layer_idx, size_t seq_len, size_t start_pos);
    void allReduceHidden(size_t seq_len);

    void linearFP8(int dev_idx, tensor_t out, tensor_t in,
                   const Qwen3FP8Linear &fp8, size_t rows, size_t cols);

public:
    Qwen3ModelTP(const Qwen3Config &config, llaisysDeviceType_t device_type,
                 const int *device_ids, int ndevice);
    ~Qwen3ModelTP() = default;

    std::vector<Qwen3TPDeviceState> &devices() { return devs_; }
    const Qwen3Config &config() const { return config_; }
    int tpSize() const { return tp_size_; }

    void resetCache();
    int64_t infer(const int64_t *token_ids, size_t num_tokens);
};

} // namespace llaisys::models
