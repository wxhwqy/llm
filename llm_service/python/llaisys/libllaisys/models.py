from ctypes import (
    Structure, POINTER, c_size_t, c_float, c_int64, c_int, c_uint8, c_uint64, c_void_p
)
from .tensor import llaisysTensor_t
from .llaisys_types import llaisysDataType_t, llaisysDeviceType_t


class LlaisysQwen3Meta(Structure):
    _fields_ = [
        ("dtype", llaisysDataType_t),
        ("nlayer", c_size_t),
        ("hs", c_size_t),
        ("nh", c_size_t),
        ("nkvh", c_size_t),
        ("dh", c_size_t),
        ("di", c_size_t),
        ("maxseq", c_size_t),
        ("voc", c_size_t),
        ("epsilon", c_float),
        ("theta", c_float),
        ("end_token", c_int64),
        ("use_fp8", c_uint8),
        ("fp8_block_h", c_size_t),
        ("fp8_block_w", c_size_t),
    ]


class LlaisysQwen3FP8Linear(Structure):
    _fields_ = [
        ("weight_fp8", llaisysTensor_t),
        ("scale_inv", llaisysTensor_t),
    ]


class LlaisysQwen3Weights(Structure):
    _fields_ = [
        ("in_embed", llaisysTensor_t),
        ("out_embed", llaisysTensor_t),
        ("out_norm_w", llaisysTensor_t),
        ("attn_norm_w", POINTER(llaisysTensor_t)),
        ("mlp_norm_w", POINTER(llaisysTensor_t)),
        ("q_norm_w", POINTER(llaisysTensor_t)),
        ("k_norm_w", POINTER(llaisysTensor_t)),
        ("q_proj", POINTER(LlaisysQwen3FP8Linear)),
        ("k_proj", POINTER(LlaisysQwen3FP8Linear)),
        ("v_proj", POINTER(LlaisysQwen3FP8Linear)),
        ("o_proj", POINTER(LlaisysQwen3FP8Linear)),
        ("gate_proj", POINTER(LlaisysQwen3FP8Linear)),
        ("up_proj", POINTER(LlaisysQwen3FP8Linear)),
        ("down_proj", POINTER(LlaisysQwen3FP8Linear)),
    ]


class LlaisysQwen3Model(Structure):
    pass


LlaisysQwen3Model_p = POINTER(LlaisysQwen3Model)


# ─── Qwen3.5 ──────────────────────────────────────────────

class LlaisysQwen3_5Meta(Structure):
    _fields_ = [
        ("dtype", llaisysDataType_t),
        ("num_layers", c_size_t),
        ("hidden_size", c_size_t),
        ("intermediate_size", c_size_t),
        ("vocab_size", c_size_t),
        ("max_seq_len", c_size_t),
        ("num_attn_heads", c_size_t),
        ("num_kv_heads", c_size_t),
        ("attn_head_dim", c_size_t),
        ("linear_num_key_heads", c_size_t),
        ("linear_key_head_dim", c_size_t),
        ("linear_num_value_heads", c_size_t),
        ("linear_value_head_dim", c_size_t),
        ("conv_kernel_size", c_size_t),
        ("rms_norm_eps", c_float),
        ("rope_theta", c_float),
        ("partial_rotary_factor", c_float),
        ("mrope_section", c_int * 3),
        ("eos_token_id", c_int64),
        ("layer_types", POINTER(c_uint8)),
    ]


class LlaisysQwen3_5DeltaNetWeights(Structure):
    _fields_ = [
        ("qkv_proj", llaisysTensor_t),
        ("o_proj", llaisysTensor_t),
        ("z_proj", llaisysTensor_t),
        ("b_proj", llaisysTensor_t),
        ("a_proj", llaisysTensor_t),
        ("A_log", llaisysTensor_t),
        ("dt_bias", llaisysTensor_t),
        ("conv_weight", llaisysTensor_t),
        ("norm_weight", llaisysTensor_t),
    ]


class LlaisysQwen3_5GatedAttnWeights(Structure):
    _fields_ = [
        ("q_proj", llaisysTensor_t),
        ("k_proj", llaisysTensor_t),
        ("v_proj", llaisysTensor_t),
        ("o_proj", llaisysTensor_t),
        ("q_norm", llaisysTensor_t),
        ("k_norm", llaisysTensor_t),
    ]


class LlaisysQwen3_5Weights(Structure):
    _fields_ = [
        ("in_embed", llaisysTensor_t),
        ("out_embed", llaisysTensor_t),
        ("out_norm_w", llaisysTensor_t),
        ("attn_norm_w", POINTER(llaisysTensor_t)),
        ("mlp_norm_w", POINTER(llaisysTensor_t)),
        ("mlp_gate_proj", POINTER(llaisysTensor_t)),
        ("mlp_up_proj", POINTER(llaisysTensor_t)),
        ("mlp_down_proj", POINTER(llaisysTensor_t)),
        ("deltanet", POINTER(LlaisysQwen3_5DeltaNetWeights)),
        ("gated_attn", POINTER(LlaisysQwen3_5GatedAttnWeights)),
        ("layer_attn_idx", POINTER(c_size_t)),
    ]


class LlaisysQwen3_5Model(Structure):
    pass


LlaisysQwen3_5Model_p = POINTER(LlaisysQwen3_5Model)


# ─── Qwen3.5 MoE ─────────────────────────────────────────

class LlaisysQwen3_5MoeMeta(Structure):
    _fields_ = [
        ("dtype", llaisysDataType_t),
        ("num_layers", c_size_t),
        ("hidden_size", c_size_t),
        ("vocab_size", c_size_t),
        ("max_seq_len", c_size_t),
        ("num_attn_heads", c_size_t),
        ("num_kv_heads", c_size_t),
        ("attn_head_dim", c_size_t),
        ("linear_num_key_heads", c_size_t),
        ("linear_key_head_dim", c_size_t),
        ("linear_num_value_heads", c_size_t),
        ("linear_value_head_dim", c_size_t),
        ("conv_kernel_size", c_size_t),
        ("num_experts", c_size_t),
        ("num_experts_per_tok", c_size_t),
        ("moe_intermediate_size", c_size_t),
        ("shared_expert_intermediate_size", c_size_t),
        ("gptq_bits", c_int),
        ("gptq_group_size", c_int),
        ("rms_norm_eps", c_float),
        ("rope_theta", c_float),
        ("partial_rotary_factor", c_float),
        ("mrope_section", c_int * 3),
        ("eos_token_id", c_int64),
        ("layer_types", POINTER(c_uint8)),
    ]


class LlaisysGPTQWeight(Structure):
    _fields_ = [
        ("qweight", llaisysTensor_t),
        ("scales", llaisysTensor_t),
        ("qzeros", llaisysTensor_t),
    ]


class LlaisysMoeExpert(Structure):
    _fields_ = [
        ("gate_proj", LlaisysGPTQWeight),
        ("up_proj", LlaisysGPTQWeight),
        ("down_proj", LlaisysGPTQWeight),
    ]


class LlaisysMoeSharedExpert(Structure):
    _fields_ = [
        ("gate_proj", llaisysTensor_t),
        ("up_proj", llaisysTensor_t),
        ("down_proj", llaisysTensor_t),
    ]


class LlaisysMoeBlock(Structure):
    _fields_ = [
        ("router", llaisysTensor_t),
        ("shared_expert_gate", llaisysTensor_t),
        ("shared_expert", LlaisysMoeSharedExpert),
        ("experts", POINTER(LlaisysMoeExpert)),
    ]


class LlaisysQwen3_5MoeDeltaNetWeights(Structure):
    _fields_ = [
        ("qkv_proj", llaisysTensor_t),
        ("o_proj", llaisysTensor_t),
        ("z_proj", llaisysTensor_t),
        ("b_proj", llaisysTensor_t),
        ("a_proj", llaisysTensor_t),
        ("A_log", llaisysTensor_t),
        ("dt_bias", llaisysTensor_t),
        ("conv_weight", llaisysTensor_t),
        ("norm_weight", llaisysTensor_t),
    ]


class LlaisysQwen3_5MoeGatedAttnWeights(Structure):
    _fields_ = [
        ("q_proj", llaisysTensor_t),
        ("k_proj", llaisysTensor_t),
        ("v_proj", llaisysTensor_t),
        ("o_proj", llaisysTensor_t),
        ("q_norm", llaisysTensor_t),
        ("k_norm", llaisysTensor_t),
    ]


class LlaisysQwen3_5MoeWeights(Structure):
    _fields_ = [
        ("in_embed", llaisysTensor_t),
        ("out_embed", llaisysTensor_t),
        ("out_norm_w", llaisysTensor_t),
        ("attn_norm_w", POINTER(llaisysTensor_t)),
        ("mlp_norm_w", POINTER(llaisysTensor_t)),
        ("moe", POINTER(LlaisysMoeBlock)),
        ("deltanet", POINTER(LlaisysQwen3_5MoeDeltaNetWeights)),
        ("gated_attn", POINTER(LlaisysQwen3_5MoeGatedAttnWeights)),
        ("layer_attn_idx", POINTER(c_size_t)),
    ]


class LlaisysQwen3_5MoeModel(Structure):
    pass


LlaisysQwen3_5MoeModel_p = POINTER(LlaisysQwen3_5MoeModel)


def load_models(lib):
    lib.llaisysQwen3ModelCreate.argtypes = [
        POINTER(LlaisysQwen3Meta), llaisysDeviceType_t, POINTER(c_int), c_int
    ]
    lib.llaisysQwen3ModelCreate.restype = LlaisysQwen3Model_p
    lib.llaisysQwen3ModelDestroy.argtypes = [LlaisysQwen3Model_p]
    lib.llaisysQwen3ModelDestroy.restype = None
    lib.llaisysQwen3ModelWeights.argtypes = [LlaisysQwen3Model_p]
    lib.llaisysQwen3ModelWeights.restype = POINTER(LlaisysQwen3Weights)
    lib.llaisysQwen3ModelInfer.argtypes = [LlaisysQwen3Model_p, POINTER(c_int64), c_size_t]
    lib.llaisysQwen3ModelInfer.restype = c_int64
    lib.llaisysQwen3ModelInferSampled.argtypes = [
        LlaisysQwen3Model_p, POINTER(c_int64), c_size_t,
        c_float, c_int, c_float, c_uint64
    ]
    lib.llaisysQwen3ModelInferSampled.restype = c_int64
    lib.llaisysQwen3ModelReset.argtypes = [LlaisysQwen3Model_p]
    lib.llaisysQwen3ModelReset.restype = None
    lib.llaisysQwen3ModelTPSize.argtypes = [LlaisysQwen3Model_p]
    lib.llaisysQwen3ModelTPSize.restype = c_int
    lib.llaisysQwen3ModelTPWeights.argtypes = [LlaisysQwen3Model_p, c_int]
    lib.llaisysQwen3ModelTPWeights.restype = POINTER(LlaisysQwen3Weights)
    lib.llaisysQwen3ModelSetCacheLen.argtypes = [LlaisysQwen3Model_p, c_size_t]
    lib.llaisysQwen3ModelSetCacheLen.restype = None
    lib.llaisysQwen3ModelGetCacheLen.argtypes = [LlaisysQwen3Model_p]
    lib.llaisysQwen3ModelGetCacheLen.restype = c_size_t
    lib.llaisysQwen3ModelSetProfile.argtypes = [LlaisysQwen3Model_p, c_int]
    lib.llaisysQwen3ModelSetProfile.restype = None
    lib.llaisysQwen3ModelSetRepetitionPenalty.argtypes = [LlaisysQwen3Model_p, c_float]
    lib.llaisysQwen3ModelSetRepetitionPenalty.restype = None

    # ─── Qwen3.5 ──────────────────────────────────────────────
    lib.llaisysQwen3_5ModelCreate.argtypes = [
        POINTER(LlaisysQwen3_5Meta), llaisysDeviceType_t, c_int
    ]
    lib.llaisysQwen3_5ModelCreate.restype = LlaisysQwen3_5Model_p
    lib.llaisysQwen3_5ModelDestroy.argtypes = [LlaisysQwen3_5Model_p]
    lib.llaisysQwen3_5ModelDestroy.restype = None
    lib.llaisysQwen3_5ModelWeights.argtypes = [LlaisysQwen3_5Model_p]
    lib.llaisysQwen3_5ModelWeights.restype = POINTER(LlaisysQwen3_5Weights)
    lib.llaisysQwen3_5ModelInfer.argtypes = [LlaisysQwen3_5Model_p, POINTER(c_int64), c_size_t]
    lib.llaisysQwen3_5ModelInfer.restype = c_int64
    lib.llaisysQwen3_5ModelInferSampled.argtypes = [
        LlaisysQwen3_5Model_p, POINTER(c_int64), c_size_t,
        c_float, c_int, c_float, c_uint64
    ]
    lib.llaisysQwen3_5ModelInferSampled.restype = c_int64
    lib.llaisysQwen3_5ModelReset.argtypes = [LlaisysQwen3_5Model_p]
    lib.llaisysQwen3_5ModelReset.restype = None
    lib.llaisysQwen3_5ModelSetCacheLen.argtypes = [LlaisysQwen3_5Model_p, c_size_t]
    lib.llaisysQwen3_5ModelSetCacheLen.restype = None
    lib.llaisysQwen3_5ModelGetCacheLen.argtypes = [LlaisysQwen3_5Model_p]
    lib.llaisysQwen3_5ModelGetCacheLen.restype = c_size_t
    lib.llaisysQwen3_5ModelSetProfile.argtypes = [LlaisysQwen3_5Model_p, c_int]
    lib.llaisysQwen3_5ModelSetProfile.restype = None
    lib.llaisysQwen3_5ModelSetRepetitionPenalty.argtypes = [LlaisysQwen3_5Model_p, c_float]
    lib.llaisysQwen3_5ModelSetRepetitionPenalty.restype = None

    # ─── Qwen3.5 MoE ─────────────────────────────────────────
    lib.llaisysQwen3_5MoeModelCreate.argtypes = [
        POINTER(LlaisysQwen3_5MoeMeta), llaisysDeviceType_t, c_int
    ]
    lib.llaisysQwen3_5MoeModelCreate.restype = LlaisysQwen3_5MoeModel_p
    lib.llaisysQwen3_5MoeModelDestroy.argtypes = [LlaisysQwen3_5MoeModel_p]
    lib.llaisysQwen3_5MoeModelDestroy.restype = None
    lib.llaisysQwen3_5MoeModelWeights.argtypes = [LlaisysQwen3_5MoeModel_p]
    lib.llaisysQwen3_5MoeModelWeights.restype = POINTER(LlaisysQwen3_5MoeWeights)
    lib.llaisysQwen3_5MoeModelInfer.argtypes = [LlaisysQwen3_5MoeModel_p, POINTER(c_int64), c_size_t]
    lib.llaisysQwen3_5MoeModelInfer.restype = c_int64
    lib.llaisysQwen3_5MoeModelInferSampled.argtypes = [
        LlaisysQwen3_5MoeModel_p, POINTER(c_int64), c_size_t,
        c_float, c_int, c_float, c_uint64
    ]
    lib.llaisysQwen3_5MoeModelInferSampled.restype = c_int64
    lib.llaisysQwen3_5MoeModelReset.argtypes = [LlaisysQwen3_5MoeModel_p]
    lib.llaisysQwen3_5MoeModelReset.restype = None
    lib.llaisysQwen3_5MoeModelSetCacheLen.argtypes = [LlaisysQwen3_5MoeModel_p, c_size_t]
    lib.llaisysQwen3_5MoeModelSetCacheLen.restype = None
    lib.llaisysQwen3_5MoeModelGetCacheLen.argtypes = [LlaisysQwen3_5MoeModel_p]
    lib.llaisysQwen3_5MoeModelGetCacheLen.restype = c_size_t
    lib.llaisysQwen3_5MoeModelSetProfile.argtypes = [LlaisysQwen3_5MoeModel_p, c_int]
    lib.llaisysQwen3_5MoeModelSetProfile.restype = None
    lib.llaisysQwen3_5MoeModelSetRepetitionPenalty.argtypes = [LlaisysQwen3_5MoeModel_p, c_float]
    lib.llaisysQwen3_5MoeModelSetRepetitionPenalty.restype = None
