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
