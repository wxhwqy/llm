"""Qwen3.5-35B-A3B MoE: hybrid DeltaNet + Gated Full Attention, Sparse MoE, GPTQ INT4."""

from typing import Sequence
from ctypes import c_int64, c_int, c_uint8, POINTER, byref
import json
import re

from ..libllaisys import LIB_LLAISYS, DeviceType, DataType
from ..libllaisys import LlaisysQwen3_5MoeMeta, LlaisysQwen3_5MoeModel_p

from pathlib import Path
import safetensors
import torch

# Layer type constants (must match C++ MoeLayerType enum)
LAYER_LINEAR_ATTENTION = 0
LAYER_FULL_ATTENTION = 1


def dequant_gptq_to_bf16(qweight, scales, qzeros, bits=4, group_size=128):
    """Dequantize a GPTQ packed weight matrix to BF16.
    qweight: [in_features//8, out_features] int32
    scales: [num_groups, out_features] float16/bfloat16
    qzeros: [num_groups, out_features//8] int32
    Returns: [out_features, in_features] bfloat16 (transposed for nn.Linear convention)
    """
    pack = 32 // bits  # 8
    in_features = qweight.shape[0] * pack
    out_features = qweight.shape[1]

    # Unpack int4 values from int32
    # qweight[i, j] contains 8 int4 values for rows i*8..i*8+7 at column j
    weight_unpacked = torch.zeros(in_features, out_features, dtype=torch.float32)
    for k in range(pack):
        weight_unpacked[k::pack] = ((qweight >> (k * bits)) & ((1 << bits) - 1)).float()

    # Unpack zero points
    zeros_unpacked = torch.zeros(qzeros.shape[0], out_features, dtype=torch.float32)
    for k in range(pack):
        zeros_unpacked[:, k::pack] = ((qzeros >> (k * bits)) & ((1 << bits) - 1)).float()

    # Dequantize per group
    scales_f = scales.float()
    num_groups = scales.shape[0]
    for g in range(num_groups):
        start = g * group_size
        end = min(start + group_size, in_features)
        weight_unpacked[start:end] = (weight_unpacked[start:end] - zeros_unpacked[g:g+1]) * scales_f[g:g+1]

    # Transpose to [out_features, in_features] and convert to BF16
    return weight_unpacked.t().to(torch.bfloat16).contiguous()


class Qwen3_5Moe:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU, device_id: int = 0):
        model_path = Path(model_path)

        config_path = model_path / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)

        text_cfg = config.get("text_config", config)

        self.num_layers = text_cfg["num_hidden_layers"]
        self.hidden_size = text_cfg["hidden_size"]
        self.vocab_size = text_cfg["vocab_size"]
        self.rms_norm_eps = text_cfg["rms_norm_eps"]

        # MoE params
        self.num_experts = text_cfg["num_experts"]
        self.num_experts_per_tok = text_cfg["num_experts_per_tok"]
        self.moe_intermediate_size = text_cfg["moe_intermediate_size"]
        self.shared_expert_intermediate_size = text_cfg.get("shared_expert_intermediate_size",
                                                             text_cfg.get("moe_intermediate_size", 512))

        # GPTQ params
        quant_cfg = config.get("quantization_config", {})
        self.gptq_bits = quant_cfg.get("bits", 4)
        self.gptq_group_size = quant_cfg.get("group_size", 128)
        self.gptq_sym = quant_cfg.get("sym", True)

        # RoPE
        rope_params = text_cfg.get("rope_parameters", text_cfg.get("rope_scaling", {}))
        self.rope_theta = rope_params.get("rope_theta", text_cfg.get("rope_theta", 10000000.0))
        self.partial_rotary_factor = rope_params.get("partial_rotary_factor",
                                                      text_cfg.get("partial_rotary_factor", 0.25))
        self.mrope_section = rope_params.get("mrope_section",
                                              text_cfg.get("rope_scaling", {}).get("mrope_section", [11, 11, 10]))

        self.eos_token_id = text_cfg.get("eos_token_id", config.get("eos_token_id", 248044))
        if isinstance(self.eos_token_id, list):
            self.eos_token_id = self.eos_token_id[0]
        self.max_seq_len = min(text_cfg.get("max_position_embeddings", 262144), 8192)

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

        # Layer types
        layer_type_names = text_cfg.get("layer_types", None)
        if layer_type_names is not None:
            self.layer_types = []
            for lt in layer_type_names:
                if lt == "full_attention":
                    self.layer_types.append(LAYER_FULL_ATTENTION)
                else:
                    self.layer_types.append(LAYER_LINEAR_ATTENTION)
        else:
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
        meta = LlaisysQwen3_5MoeMeta()
        meta.dtype = DataType.BF16.value
        meta.num_layers = self.num_layers
        meta.hidden_size = self.hidden_size
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
        meta.num_experts = self.num_experts
        meta.num_experts_per_tok = self.num_experts_per_tok
        meta.moe_intermediate_size = self.moe_intermediate_size
        meta.shared_expert_intermediate_size = self.shared_expert_intermediate_size
        meta.gptq_bits = self.gptq_bits
        meta.gptq_group_size = self.gptq_group_size
        meta.rms_norm_eps = self.rms_norm_eps
        meta.rope_theta = self.rope_theta
        meta.partial_rotary_factor = self.partial_rotary_factor
        meta.mrope_section = (c_int * 3)(*self.mrope_section)
        meta.eos_token_id = self.eos_token_id

        c_layer_types = (c_uint8 * self.num_layers)(*self.layer_types)
        meta.layer_types = c_layer_types

        print(f"[Qwen3.5-MoE] Creating model: {self.num_layers} layers, "
              f"hs={self.hidden_size}, ne={self.num_experts}, topk={self.num_experts_per_tok}, "
              f"moe_dim={self.moe_intermediate_size}, gptq={self.gptq_bits}bit/gs{self.gptq_group_size}")

        self._model = LIB_LLAISYS.llaisysQwen3_5MoeModelCreate(
            byref(meta), device.value, device_id
        )

        if not self._model:
            raise RuntimeError("Failed to create Qwen3.5-MoE model")

        self._weights = LIB_LLAISYS.llaisysQwen3_5MoeModelWeights(self._model)

        # KV Cache reuse state
        self._prev_input_ids: list[int] | None = None
        self._prev_cache_len: int = 0

        # Pending GPTQ buffers: collect qweight/scales/qzeros before loading
        self._gptq_pending = {}

        self._load_weights(model_path)

    def _load_weights(self, model_path: Path):
        """Load weights from safetensors files."""
        for file in sorted(model_path.glob("*.safetensors")):
            if "model.safetensors.index" in str(file):
                continue
            print(f"  Loading {file.name}...")
            data = safetensors.safe_open(file, framework="pt", device="cpu")
            for name in data.keys():
                tensor = data.get_tensor(name)
                self._load_weight(name, tensor)

        # Process any remaining pending GPTQ weights
        self._flush_gptq_pending()

    def _load_bf16_tensor(self, c_tensor, tensor: torch.Tensor):
        tensor = tensor.to(torch.bfloat16).contiguous()
        LIB_LLAISYS.tensorLoad(c_tensor, tensor.data_ptr())

    def _load_f32_tensor(self, c_tensor, tensor: torch.Tensor):
        tensor = tensor.to(torch.float32).contiguous()
        LIB_LLAISYS.tensorLoad(c_tensor, tensor.data_ptr())

    def _load_i32_tensor(self, c_tensor, tensor: torch.Tensor):
        tensor = tensor.to(torch.int32).contiguous()
        LIB_LLAISYS.tensorLoad(c_tensor, tensor.data_ptr())

    def _flush_gptq_pending(self):
        """Process any accumulated GPTQ weight triplets."""
        pass  # All GPTQ weights are loaded directly as they come

    def _load_weight(self, name: str, tensor: torch.Tensor):
        w = self._weights.contents

        # Handle VLM prefix
        canonical = name
        if canonical.startswith("model.language_model."):
            canonical = "model." + canonical[len("model.language_model."):]

        # Skip vision/mtp weights
        if name.startswith("model.visual.") or name.startswith("mtp."):
            return

        # ── Global weights ──
        if canonical == "model.embed_tokens.weight":
            self._load_bf16_tensor(w.in_embed, tensor)
            return
        if name == "lm_head.weight":
            self._load_bf16_tensor(w.out_embed, tensor)
            return
        # lm_head GPTQ: dequantize at load time
        if name.startswith("lm_head.") and "qweight" in name:
            # Will be handled by the GPTQ dequant path below
            pass
        if canonical == "model.norm.weight":
            # Qwen3_5RMSNorm uses (1 + weight) parameterization
            self._load_bf16_tensor(w.out_norm_w, tensor.float() + 1.0)
            return

        if ".layers." not in canonical:
            # Handle lm_head GPTQ
            if name.startswith("lm_head."):
                self._handle_lm_head_gptq(name, tensor)
            return

        parts = canonical.split(".")
        layer_idx = int(parts[2])
        if layer_idx >= self.num_layers:
            return
        attn_idx = self.layer_attn_idx[layer_idx]
        is_deltanet = self.layer_types[layer_idx] == LAYER_LINEAR_ATTENTION

        # ── Per-layer norm weights ──
        if "input_layernorm.weight" in canonical:
            self._load_bf16_tensor(w.attn_norm_w[layer_idx], tensor.float() + 1.0)
            return
        if "post_attention_layernorm.weight" in canonical:
            self._load_bf16_tensor(w.mlp_norm_w[layer_idx], tensor.float() + 1.0)
            return

        # ── MoE weights ──
        if ".mlp." in canonical:
            self._load_moe_weight(canonical, layer_idx, tensor, w)
            return

        # ── Attention weights ──
        if is_deltanet:
            dn = w.deltanet[attn_idx]
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
                if tensor.ndim == 3:
                    tensor = tensor.squeeze(1)
                self._load_bf16_tensor(dn.conv_weight, tensor)
            elif "linear_attn.norm.weight" in canonical:
                self._load_bf16_tensor(dn.norm_weight, tensor)
        else:
            ga = w.gated_attn[attn_idx]
            if "self_attn.q_proj.weight" in canonical:
                self._load_bf16_tensor(ga.q_proj, tensor)
            elif "self_attn.k_proj.weight" in canonical:
                self._load_bf16_tensor(ga.k_proj, tensor)
            elif "self_attn.v_proj.weight" in canonical:
                self._load_bf16_tensor(ga.v_proj, tensor)
            elif "self_attn.o_proj.weight" in canonical:
                self._load_bf16_tensor(ga.o_proj, tensor)
            elif "self_attn.q_norm.weight" in canonical:
                self._load_bf16_tensor(ga.q_norm, tensor.float() + 1.0)
            elif "self_attn.k_norm.weight" in canonical:
                self._load_bf16_tensor(ga.k_norm, tensor.float() + 1.0)

    def _handle_lm_head_gptq(self, name: str, tensor: torch.Tensor):
        """Accumulate lm_head GPTQ components and dequantize when all parts arrive."""
        w = self._weights.contents
        key = "lm_head"
        if key not in self._gptq_pending:
            self._gptq_pending[key] = {}

        if name.endswith(".qweight"):
            self._gptq_pending[key]["qweight"] = tensor
        elif name.endswith(".scales"):
            self._gptq_pending[key]["scales"] = tensor
        elif name.endswith(".qzeros"):
            self._gptq_pending[key]["qzeros"] = tensor
        elif name.endswith(".g_idx"):
            pass  # Ignore g_idx (desc_act=false)

        p = self._gptq_pending[key]
        if "qweight" in p and "scales" in p and "qzeros" in p:
            print(f"  Dequantizing lm_head GPTQ -> BF16...")
            dequantized = dequant_gptq_to_bf16(
                p["qweight"], p["scales"], p["qzeros"],
                bits=self.gptq_bits, group_size=self.gptq_group_size)
            self._load_bf16_tensor(w.out_embed, dequantized)
            del self._gptq_pending[key]

    def _load_moe_weight(self, canonical: str, layer_idx: int, tensor: torch.Tensor, w):
        """Load MoE block weights: router, shared expert, and routed experts (GPTQ)."""
        moe_block = w.moe[layer_idx]

        # Router weight
        if ".mlp.gate.weight" in canonical:
            self._load_bf16_tensor(moe_block.router, tensor)
            return

        # Shared expert gate
        if ".mlp.shared_expert_gate.weight" in canonical:
            self._load_bf16_tensor(moe_block.shared_expert_gate, tensor)
            return

        # Shared expert (BF16, not quantized)
        if ".mlp.shared_expert." in canonical:
            se = moe_block.shared_expert
            if "gate_proj.weight" in canonical:
                self._load_bf16_tensor(se.gate_proj, tensor)
            elif "up_proj.weight" in canonical:
                self._load_bf16_tensor(se.up_proj, tensor)
            elif "down_proj.weight" in canonical:
                self._load_bf16_tensor(se.down_proj, tensor)
            return

        # Routed experts (GPTQ quantized)
        # Pattern: .mlp.experts.{idx}.{gate_proj|up_proj|down_proj}.{qweight|scales|qzeros}
        m = re.search(r'\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(qweight|scales|qzeros|g_idx)', canonical)
        if m:
            expert_idx = int(m.group(1))
            proj_name = m.group(2)
            weight_type = m.group(3)

            if weight_type == "g_idx":
                return  # Ignore

            if expert_idx >= self.num_experts:
                return

            expert = moe_block.experts[expert_idx]
            if proj_name == "gate_proj":
                gptq = expert.gate_proj
            elif proj_name == "up_proj":
                gptq = expert.up_proj
            else:
                gptq = expert.down_proj

            if weight_type == "qweight":
                self._load_i32_tensor(gptq.qweight, tensor)
            elif weight_type == "scales":
                self._load_bf16_tensor(gptq.scales, tensor)
            elif weight_type == "qzeros":
                self._load_i32_tensor(gptq.qzeros, tensor)
            return

        # Routed expert BF16 weights (non-quantized fallback)
        m2 = re.search(r'\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight', canonical)
        if m2:
            # This shouldn't happen for GPTQ model, but handle gracefully
            print(f"  Warning: BF16 expert weight found (expected GPTQ): {canonical}")
            return

    def __del__(self):
        if hasattr(self, "_model") and self._model:
            LIB_LLAISYS.llaisysQwen3_5MoeModelDestroy(self._model)

    def reset(self):
        LIB_LLAISYS.llaisysQwen3_5MoeModelReset(self._model)
        self._prev_input_ids = None
        self._prev_cache_len = 0

    def set_cache_len(self, length: int):
        LIB_LLAISYS.llaisysQwen3_5MoeModelSetCacheLen(self._model, length)

    def get_cache_len(self) -> int:
        return LIB_LLAISYS.llaisysQwen3_5MoeModelGetCacheLen(self._model)

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
        LIB_LLAISYS.llaisysQwen3_5MoeModelSetProfile(self._model, 1 if enabled else 0)

    def set_repetition_penalty(self, penalty: float):
        LIB_LLAISYS.llaisysQwen3_5MoeModelSetRepetitionPenalty(self._model, penalty)

    def _infer_one(self, input_array, n, temperature, top_k, top_p, seed):
        use_sampling = (temperature > 0.0) and (top_k != 1)
        if use_sampling:
            return LIB_LLAISYS.llaisysQwen3_5MoeModelInferSampled(
                self._model, input_array, n,
                temperature, top_k, top_p, seed,
            )
        return LIB_LLAISYS.llaisysQwen3_5MoeModelInfer(
            self._model, input_array, n
        )

    def generate(self, inputs: Sequence[int], max_new_tokens: int = None,
                 top_k: int = 1, top_p: float = 0.8, temperature: float = 0.8,
                 reuse_cache: bool = False, repetition_penalty: float = 1.2):
        return list(self.stream_generate(
            inputs, max_new_tokens=max_new_tokens,
            top_k=top_k, top_p=top_p, temperature=temperature,
            reuse_cache=reuse_cache, repetition_penalty=repetition_penalty,
        ))

    def stream_generate(self, inputs: Sequence[int], max_new_tokens: int = None,
                        top_k: int = 1, top_p: float = 0.8, temperature: float = 0.8,
                        reuse_cache: bool = False, repetition_penalty: float = 1.2):
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
            LIB_LLAISYS.llaisysQwen3_5MoeModelReset(self._model)
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
