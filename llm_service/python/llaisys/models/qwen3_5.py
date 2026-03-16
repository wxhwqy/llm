"""Qwen3.5-9B model: hybrid DeltaNet + Gated Full Attention, Dense MLP."""

from typing import Sequence, List
from ctypes import c_int64, c_int, c_uint8, POINTER, byref
import json

from ..libllaisys import LIB_LLAISYS, DeviceType, DataType
from ..libllaisys import LlaisysQwen3_5Meta, LlaisysQwen3_5Model_p

from pathlib import Path
import safetensors
import torch


# Layer type constants (must match C++ LayerType enum)
LAYER_LINEAR_ATTENTION = 0
LAYER_FULL_ATTENTION = 1


class Qwen3_5:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU, device_id: int = 0):
        model_path = Path(model_path)

        config_path = model_path / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)

        # Handle nested config: VLM has text_config, pure text model has flat config
        text_cfg = config.get("text_config", config)

        self.num_layers = text_cfg["num_hidden_layers"]
        self.hidden_size = text_cfg["hidden_size"]
        self.intermediate_size = text_cfg["intermediate_size"]
        self.vocab_size = text_cfg["vocab_size"]
        self.rms_norm_eps = text_cfg["rms_norm_eps"]

        # RoPE params may be nested under rope_parameters or rope_scaling
        rope_params = text_cfg.get("rope_parameters", text_cfg.get("rope_scaling", {}))
        self.rope_theta = rope_params.get("rope_theta", text_cfg.get("rope_theta", 10000000.0))
        self.partial_rotary_factor = rope_params.get("partial_rotary_factor",
                                                      text_cfg.get("partial_rotary_factor", 0.25))
        self.mrope_section = rope_params.get("mrope_section",
                                              text_cfg.get("rope_scaling", {}).get("mrope_section", [11, 11, 10]))

        self.eos_token_id = text_cfg.get("eos_token_id", config.get("eos_token_id", 248044))
        if isinstance(self.eos_token_id, list):
            self.eos_token_id = self.eos_token_id[0]
        self.max_seq_len = min(text_cfg.get("max_position_embeddings", 262144), 16384)

        # Full attention params
        self.num_attn_heads = text_cfg["num_attention_heads"]
        self.num_kv_heads = text_cfg["num_key_value_heads"]
        self.attn_head_dim = text_cfg.get("head_dim", 256)

        # Linear attention (DeltaNet) params
        self.linear_num_key_heads = text_cfg.get("linear_num_key_heads", 16)
        self.linear_key_head_dim = text_cfg.get("linear_key_head_dim", 128)
        self.linear_num_value_heads = text_cfg.get("linear_num_value_heads", 32)
        self.linear_value_head_dim = text_cfg.get("linear_value_head_dim", 128)
        self.conv_kernel_size = text_cfg.get("linear_conv_kernel_dim", 4)

        # Layer types from config
        layer_type_names = text_cfg.get("layer_types", None)
        if layer_type_names is not None:
            self.layer_types = []
            for lt in layer_type_names:
                if lt == "full_attention":
                    self.layer_types.append(LAYER_FULL_ATTENTION)
                else:
                    self.layer_types.append(LAYER_LINEAR_ATTENTION)
        else:
            # Default pattern: 3 linear + 1 full, repeated
            self.layer_types = []
            for i in range(self.num_layers):
                if (i + 1) % 4 == 0:
                    self.layer_types.append(LAYER_FULL_ATTENTION)
                else:
                    self.layer_types.append(LAYER_LINEAR_ATTENTION)

        self.n_deltanet = sum(1 for lt in self.layer_types if lt == LAYER_LINEAR_ATTENTION)
        self.n_fullattn = sum(1 for lt in self.layer_types if lt == LAYER_FULL_ATTENTION)

        # Build layer_attn_idx mapping
        self.layer_attn_idx = []
        dn_count, fa_count = 0, 0
        for lt in self.layer_types:
            if lt == LAYER_LINEAR_ATTENTION:
                self.layer_attn_idx.append(dn_count)
                dn_count += 1
            else:
                self.layer_attn_idx.append(fa_count)
                fa_count += 1

        # Create C meta
        meta = LlaisysQwen3_5Meta()
        meta.dtype = DataType.BF16.value
        meta.num_layers = self.num_layers
        meta.hidden_size = self.hidden_size
        meta.intermediate_size = self.intermediate_size
        meta.vocab_size = self.vocab_size
        meta.max_seq_len = self.max_seq_len
        meta.num_attn_heads = self.num_attn_heads
        meta.num_kv_heads = self.num_kv_heads
        meta.attn_head_dim = self.attn_head_dim
        meta.linear_num_key_heads = self.linear_num_key_heads
        meta.linear_key_head_dim = self.linear_key_head_dim
        meta.linear_num_value_heads = self.linear_num_value_heads
        meta.linear_value_head_dim = self.linear_value_head_dim
        meta.conv_kernel_size = self.conv_kernel_size
        meta.rms_norm_eps = self.rms_norm_eps
        meta.rope_theta = self.rope_theta
        meta.partial_rotary_factor = self.partial_rotary_factor
        meta.mrope_section = (c_int * 3)(*self.mrope_section)
        meta.eos_token_id = self.eos_token_id

        c_layer_types = (c_uint8 * self.num_layers)(*self.layer_types)
        meta.layer_types = c_layer_types

        self._model = LIB_LLAISYS.llaisysQwen3_5ModelCreate(
            byref(meta), device.value, device_id
        )

        if not self._model:
            raise RuntimeError("Failed to create Qwen3.5 model")

        self._weights = LIB_LLAISYS.llaisysQwen3_5ModelWeights(self._model)

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

    def _load_bf16_tensor(self, c_tensor, tensor: torch.Tensor):
        tensor = tensor.to(torch.bfloat16).contiguous()
        LIB_LLAISYS.tensorLoad(c_tensor, tensor.data_ptr())

    def _load_f32_tensor(self, c_tensor, tensor: torch.Tensor):
        tensor = tensor.to(torch.float32).contiguous()
        LIB_LLAISYS.tensorLoad(c_tensor, tensor.data_ptr())

    def _load_weight(self, name: str, tensor: torch.Tensor):
        w = self._weights.contents

        # Handle VLM prefix: model.language_model.* → model.*
        # Also handle pure text model: model.* directly
        canonical = name
        if canonical.startswith("model.language_model."):
            canonical = "model." + canonical[len("model.language_model."):]

        # Skip vision/mtp weights
        if name.startswith("model.visual.") or name.startswith("mtp."):
            return

        # Global weights
        if canonical == "model.embed_tokens.weight":
            self._load_bf16_tensor(w.in_embed, tensor)
            return
        if name == "lm_head.weight":
            self._load_bf16_tensor(w.out_embed, tensor)
            return
        if canonical == "model.norm.weight":
            # Qwen3_5RMSNorm uses (1 + weight) parameterization
            self._load_bf16_tensor(w.out_norm_w, tensor.float() + 1.0)
            return

        if ".layers." not in canonical:
            return

        parts = canonical.split(".")
        layer_idx = int(parts[2])
        if layer_idx >= self.num_layers:
            return
        attn_idx = self.layer_attn_idx[layer_idx]
        is_deltanet = self.layer_types[layer_idx] == LAYER_LINEAR_ATTENTION

        # Per-layer norm weights
        if "input_layernorm.weight" in canonical:
            # Qwen3_5RMSNorm uses (1 + weight) parameterization
            self._load_bf16_tensor(w.attn_norm_w[layer_idx], tensor.float() + 1.0)
            return
        if "post_attention_layernorm.weight" in canonical:
            # Qwen3_5RMSNorm uses (1 + weight) parameterization
            self._load_bf16_tensor(w.mlp_norm_w[layer_idx], tensor.float() + 1.0)
            return

        # MLP weights (shared by all layer types)
        if "mlp.gate_proj.weight" in canonical:
            self._load_bf16_tensor(w.mlp_gate_proj[layer_idx], tensor)
            return
        if "mlp.up_proj.weight" in canonical:
            self._load_bf16_tensor(w.mlp_up_proj[layer_idx], tensor)
            return
        if "mlp.down_proj.weight" in canonical:
            self._load_bf16_tensor(w.mlp_down_proj[layer_idx], tensor)
            return

        # Attention weights depend on layer type
        if is_deltanet:
            dn = w.deltanet[attn_idx]
            # DeltaNet: weights are under "linear_attn.*"
            if "linear_attn.in_proj_qkv.weight" in canonical:
                self._load_bf16_tensor(dn.qkv_proj, tensor)
            elif "linear_attn.out_proj.weight" in canonical:
                self._load_bf16_tensor(dn.o_proj, tensor)
            elif "linear_attn.in_proj_z.weight" in canonical:
                self._load_bf16_tensor(dn.z_proj, tensor)
            elif "linear_attn.in_proj_b.weight" in canonical:
                self._load_bf16_tensor(dn.b_proj, tensor)
            elif "linear_attn.in_proj_a.weight" in canonical:
                self._load_bf16_tensor(dn.a_proj, tensor)
            elif "linear_attn.A_log" in canonical:
                self._load_f32_tensor(dn.A_log, tensor)
            elif "linear_attn.dt_bias" in canonical:
                self._load_f32_tensor(dn.dt_bias, tensor)
            elif "linear_attn.conv1d.weight" in canonical:
                # HF conv weight is [d_conv, 1, kernel_size], we need [d_conv, kernel_size]
                if tensor.ndim == 3:
                    tensor = tensor.squeeze(1)
                self._load_bf16_tensor(dn.conv_weight, tensor)
            elif "linear_attn.norm.weight" in canonical:
                self._load_bf16_tensor(dn.norm_weight, tensor)
        else:
            ga = w.gated_attn[attn_idx]
            # Gated Full Attention projections: under "self_attn.*"
            if "self_attn.q_proj.weight" in canonical:
                self._load_bf16_tensor(ga.q_proj, tensor)
            elif "self_attn.k_proj.weight" in canonical:
                self._load_bf16_tensor(ga.k_proj, tensor)
            elif "self_attn.v_proj.weight" in canonical:
                self._load_bf16_tensor(ga.v_proj, tensor)
            elif "self_attn.o_proj.weight" in canonical:
                self._load_bf16_tensor(ga.o_proj, tensor)
            elif "self_attn.q_norm.weight" in canonical:
                # Qwen3_5RMSNorm uses (1 + weight) parameterization
                self._load_bf16_tensor(ga.q_norm, tensor.float() + 1.0)
            elif "self_attn.k_norm.weight" in canonical:
                # Qwen3_5RMSNorm uses (1 + weight) parameterization
                self._load_bf16_tensor(ga.k_norm, tensor.float() + 1.0)

    def __del__(self):
        if hasattr(self, "_model") and self._model:
            LIB_LLAISYS.llaisysQwen3_5ModelDestroy(self._model)

    def reset(self):
        LIB_LLAISYS.llaisysQwen3_5ModelReset(self._model)
        self._prev_input_ids = None
        self._prev_cache_len = 0

    def set_cache_len(self, length: int):
        LIB_LLAISYS.llaisysQwen3_5ModelSetCacheLen(self._model, length)

    def get_cache_len(self) -> int:
        return LIB_LLAISYS.llaisysQwen3_5ModelGetCacheLen(self._model)

    def _compute_prefix_match(self, new_ids: list[int]) -> int:
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
        LIB_LLAISYS.llaisysQwen3_5ModelSetProfile(self._model, 1 if enabled else 0)

    def set_repetition_penalty(self, penalty: float):
        LIB_LLAISYS.llaisysQwen3_5ModelSetRepetitionPenalty(self._model, penalty)

    def _infer_one(self, input_array, n, temperature, top_k, top_p, seed):
        use_sampling = (temperature > 0.0) and (top_k != 1)
        if use_sampling:
            return LIB_LLAISYS.llaisysQwen3_5ModelInferSampled(
                self._model, input_array, n,
                temperature, top_k, top_p, seed,
            )
        return LIB_LLAISYS.llaisysQwen3_5ModelInfer(
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
        repetition_penalty: float = 1.2,
    ):
        return list(self.stream_generate(
            inputs, max_new_tokens=max_new_tokens,
            top_k=top_k, top_p=top_p, temperature=temperature,
            reuse_cache=reuse_cache, repetition_penalty=repetition_penalty,
        ))

    def stream_generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
        reuse_cache: bool = False,
        repetition_penalty: float = 1.2,
    ):
        import random
        self.set_repetition_penalty(repetition_penalty)
        tokens = list(inputs)

        prefix_len = 0
        if reuse_cache:
            prefix_len = self._compute_prefix_match(tokens)

        if prefix_len >= len(tokens):
            prefix_len = max(len(tokens) - 1, 0)

        if prefix_len > 0:
            self.set_cache_len(prefix_len)
            effective_input = tokens[prefix_len:]
        else:
            LIB_LLAISYS.llaisysQwen3_5ModelReset(self._model)
            effective_input = tokens

        if max_new_tokens is None:
            max_new_tokens = 128
        max_new_tokens = min(max_new_tokens, self.max_seq_len - len(tokens))
        if max_new_tokens <= 0:
            return

        seed_base = random.getrandbits(64)

        input_array = (c_int64 * len(effective_input))(*effective_input)
        next_token = self._infer_one(input_array, len(effective_input),
                                     temperature, top_k, top_p, seed_base)
        yield next_token

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

        self._prev_input_ids = tokens + generated
        self._prev_cache_len = len(tokens) + len(generated)
