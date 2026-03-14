from typing import Sequence, List, Union
from ctypes import c_int64, c_int, c_uint8, POINTER, byref
import json

from ..libllaisys import LIB_LLAISYS, DeviceType, DataType
from ..libllaisys import LlaisysQwen3Meta, LlaisysQwen3Model_p

from pathlib import Path
import safetensors
import torch
import numpy as np

COLUMN_PARALLEL_PROJS = {"q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"}
ROW_PARALLEL_PROJS = {"o_proj", "down_proj"}
PROJ_MAP = {
    "self_attn.q_proj": "q_proj",
    "self_attn.k_proj": "k_proj",
    "self_attn.v_proj": "v_proj",
    "self_attn.o_proj": "o_proj",
    "mlp.gate_proj": "gate_proj",
    "mlp.up_proj": "up_proj",
    "mlp.down_proj": "down_proj",
}


class Qwen3:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU,
                 device_ids: Union[int, List[int]] = 0):
        model_path = Path(model_path)

        if isinstance(device_ids, int):
            device_ids = [device_ids]

        self.tp_size = len(device_ids)

        config_path = model_path / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)

        self.num_layers = config["num_hidden_layers"]
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.num_kv_heads = config["num_key_value_heads"]
        self.head_dim = config.get("head_dim", self.hidden_size // self.num_heads)
        self.intermediate_size = config["intermediate_size"]
        self.vocab_size = config["vocab_size"]
        self.rms_norm_eps = config["rms_norm_eps"]
        self.rope_theta = config.get("rope_theta", 1000000.0)
        self.eos_token_id = config["eos_token_id"]
        self.max_seq_len = min(config.get("max_position_embeddings", 131072), 8192)###############

        quant_config = config.get("quantization_config", {})
        self.use_fp8 = quant_config.get("quant_method") == "fp8"
        block_size = quant_config.get("weight_block_size", [128, 128])
        self.fp8_block_h = block_size[0]
        self.fp8_block_w = block_size[1]

        meta = LlaisysQwen3Meta()
        meta.dtype = DataType.BF16.value
        meta.nlayer = self.num_layers
        meta.hs = self.hidden_size
        meta.nh = self.num_heads
        meta.nkvh = self.num_kv_heads
        meta.dh = self.head_dim
        meta.di = self.intermediate_size
        meta.maxseq = self.max_seq_len
        meta.voc = self.vocab_size
        meta.epsilon = self.rms_norm_eps
        meta.theta = self.rope_theta
        meta.end_token = self.eos_token_id
        meta.use_fp8 = 1 if self.use_fp8 else 0
        meta.fp8_block_h = self.fp8_block_h
        meta.fp8_block_w = self.fp8_block_w

        c_device_ids = (c_int * self.tp_size)(*device_ids)
        self._model = LIB_LLAISYS.llaisysQwen3ModelCreate(
            byref(meta), device.value, c_device_ids, self.tp_size
        )

        if not self._model:
            raise RuntimeError("Failed to create Qwen3 model")

        actual_tp = LIB_LLAISYS.llaisysQwen3ModelTPSize(self._model)
        self.tp_size = actual_tp

        if self.tp_size > 1:
            self._tp_weights = []
            for i in range(self.tp_size):
                w = LIB_LLAISYS.llaisysQwen3ModelTPWeights(self._model, i)
                self._tp_weights.append(w)
        else:
            self._weights = LIB_LLAISYS.llaisysQwen3ModelWeights(self._model)

        # KV Cache reuse state
        self._prev_input_ids: list[int] | None = None
        self._prev_cache_len: int = 0

        self._load_weights(model_path)

    def _load_weights(self, model_path: Path):
        for file in sorted(model_path.glob("*.safetensors")):
            data = safetensors.safe_open(file, framework="pt", device="cpu")
            for name in data.keys():
                tensor = data.get_tensor(name)
                self._load_weight(name, tensor)

    def _load_fp8_tensor(self, c_tensor, tensor: torch.Tensor):
        arr = tensor.contiguous().view(torch.uint8).numpy()
        LIB_LLAISYS.tensorLoad(c_tensor, arr.ctypes.data)

    def _load_bf16_tensor(self, c_tensor, tensor: torch.Tensor):
        tensor = tensor.to(torch.bfloat16).contiguous()
        LIB_LLAISYS.tensorLoad(c_tensor, tensor.data_ptr())

    def _load_f32_tensor(self, c_tensor, tensor: torch.Tensor):
        tensor = tensor.to(torch.float32).contiguous()
        LIB_LLAISYS.tensorLoad(c_tensor, tensor.data_ptr())

    def _get_all_weights(self):
        if self.tp_size > 1:
            return [w.contents for w in self._tp_weights]
        return [self._weights.contents]

    def _load_replicated(self, tensor: torch.Tensor, get_target):
        for w in self._get_all_weights():
            self._load_bf16_tensor(get_target(w), tensor)

    def _load_weight(self, name: str, tensor: torch.Tensor):
        tp = self.tp_size

        if name == "model.embed_tokens.weight":
            self._load_replicated(tensor, lambda w: w.in_embed)
            return
        if name == "lm_head.weight":
            self._load_replicated(tensor, lambda w: w.out_embed)
            return
        if name == "model.norm.weight":
            self._load_replicated(tensor, lambda w: w.out_norm_w)
            return

        if ".layers." not in name:
            return

        parts = name.split(".")
        layer_idx = int(parts[2])

        if "input_layernorm.weight" in name:
            self._load_replicated(tensor, lambda w: w.attn_norm_w[layer_idx])
            return
        if "post_attention_layernorm.weight" in name:
            self._load_replicated(tensor, lambda w: w.mlp_norm_w[layer_idx])
            return
        if "self_attn.q_norm.weight" in name:
            self._load_replicated(tensor, lambda w: w.q_norm_w[layer_idx])
            return
        if "self_attn.k_norm.weight" in name:
            self._load_replicated(tensor, lambda w: w.k_norm_w[layer_idx])
            return

        for key, attr in PROJ_MAP.items():
            if key not in name:
                continue

            is_scale = name.endswith(".weight_scale_inv")
            is_weight = name.endswith(".weight") and not is_scale

            if not is_weight and not is_scale:
                break

            all_weights = self._get_all_weights()

            if tp == 1:
                fp8l = getattr(all_weights[0], attr)[layer_idx]
                if is_scale:
                    self._load_f32_tensor(fp8l.scale_inv, tensor)
                elif self.use_fp8 and tensor.dtype == torch.float8_e4m3fn:
                    self._load_fp8_tensor(fp8l.weight_fp8, tensor)
                else:
                    self._load_bf16_tensor(fp8l.weight_fp8, tensor)
            else:
                chunks = self._split_for_tp(tensor, attr, is_scale)
                for di, chunk in enumerate(chunks):
                    fp8l = getattr(all_weights[di], attr)[layer_idx]
                    if is_scale:
                        self._load_f32_tensor(fp8l.scale_inv, chunk)
                    elif self.use_fp8 and tensor.dtype == torch.float8_e4m3fn:
                        self._load_fp8_tensor(fp8l.weight_fp8, chunk)
                    else:
                        self._load_bf16_tensor(fp8l.weight_fp8, chunk)
            break

    def _split_for_tp(self, tensor: torch.Tensor, attr: str, is_scale: bool):
        tp = self.tp_size
        if attr in COLUMN_PARALLEL_PROJS:
            return tensor.chunk(tp, dim=0)
        elif attr in ROW_PARALLEL_PROJS:
            split_dim = 1 if not is_scale else 1
            return tensor.chunk(tp, dim=split_dim)
        return [tensor] * tp

    def __del__(self):
        if hasattr(self, "_model") and self._model:
            LIB_LLAISYS.llaisysQwen3ModelDestroy(self._model)

    def reset(self):
        LIB_LLAISYS.llaisysQwen3ModelReset(self._model)
        self._prev_input_ids: list[int] | None = None
        self._prev_cache_len: int = 0

    def set_cache_len(self, length: int):
        LIB_LLAISYS.llaisysQwen3ModelSetCacheLen(self._model, length)

    def get_cache_len(self) -> int:
        return LIB_LLAISYS.llaisysQwen3ModelGetCacheLen(self._model)

    def _compute_prefix_match(self, new_ids: list[int]) -> int:
        """Return the longest common prefix length between cached and new token IDs."""
        if self._prev_input_ids is None or self._prev_cache_len == 0:
            return 0
        old_ids = self._prev_input_ids
        max_match = min(len(old_ids), len(new_ids), self._prev_cache_len)
        prefix_len = 0
        for i in range(max_match):
            if old_ids[i] != new_ids[i]:
                break
            prefix_len = i + 1
        return prefix_len

    def set_profile(self, enabled: bool):
        LIB_LLAISYS.llaisysQwen3ModelSetProfile(self._model, 1 if enabled else 0)

    def _infer_one(self, input_array, n, temperature, top_k, top_p, seed):
        """Call the appropriate C infer function based on sampling params."""
        use_sampling = (temperature > 0.0) and (top_k != 1)
        if use_sampling:
            return LIB_LLAISYS.llaisysQwen3ModelInferSampled(
                self._model, input_array, n,
                temperature, top_k, top_p, seed,
            )
        return LIB_LLAISYS.llaisysQwen3ModelInfer(
            self._model, input_array, n
        )

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
        reuse_cache: bool = False,
    ):
        """Non-streaming generate: returns full token list."""
        return list(self.stream_generate(
            inputs, max_new_tokens=max_new_tokens,
            top_k=top_k, top_p=top_p, temperature=temperature,
            reuse_cache=reuse_cache,
        ))

    def stream_generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
        reuse_cache: bool = False,
    ):
        """Streaming generate: yields one token id at a time.

        Args:
            reuse_cache: When True, attempt to reuse the KV cache from the
                previous call by prefix-matching token IDs. Only the new
                (non-matching) suffix is fed through the prefill stage,
                which dramatically reduces TTFT for multi-turn conversations.
                Defaults to False for backward compatibility.
        """
        import random
        tokens = list(inputs)

        prefix_len = 0
        if reuse_cache:
            prefix_len = self._compute_prefix_match(tokens)

        # Ensure at least 1 token is fed to the prefill stage.
        # When the entire prompt matches the cached prefix (e.g. identical
        # re-request or regenerate), back off by 1 so ntoken > 0.
        if prefix_len >= len(tokens):
            prefix_len = max(len(tokens) - 1, 0)

        if prefix_len > 0:
            # Reuse mode: roll back cache_len to the matched prefix boundary,
            # then only prefill the new suffix tokens.
            self.set_cache_len(prefix_len)
            effective_input = tokens[prefix_len:]
        else:
            # Full reset: no reusable prefix (first turn or mismatch).
            LIB_LLAISYS.llaisysQwen3ModelReset(self._model)
            effective_input = tokens

        if max_new_tokens is None:
            max_new_tokens = 128
        max_new_tokens = min(max_new_tokens, self.max_seq_len - len(tokens))
        if max_new_tokens <= 0:
            return

        seed_base = random.getrandbits(64)

        # Prefill (full prompt if no reuse, or only new suffix)
        input_array = (c_int64 * len(effective_input))(*effective_input)
        next_token = self._infer_one(input_array, len(effective_input),
                                     temperature, top_k, top_p, seed_base)
        yield next_token

        # Decode
        generated = [next_token]
        for step in range(max_new_tokens - 1):
            if next_token == self.eos_token_id:
                break
            input_array = (c_int64 * 1)(next_token)
            next_token = self._infer_one(input_array, 1,
                                         temperature, top_k, top_p,
                                         seed_base + step + 1)
            yield next_token
            generated.append(next_token)

        # Save state for next call's prefix matching.
        # Include generated tokens because the next turn's prompt will contain
        # the assistant's reply (which is already in the KV cache).
        self._prev_input_ids = tokens + generated
        self._prev_cache_len = len(tokens) + len(generated)
