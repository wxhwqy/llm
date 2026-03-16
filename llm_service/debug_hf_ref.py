"""Run reference delta rule computation using HF's torch_chunk_gated_delta_rule."""
import torch
import torch.nn.functional as F
import json, sys
from pathlib import Path
import safetensors.torch

def l2norm(x, dim=-1, eps=1e-6):
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm

def torch_recurrent_gated_delta_rule(
    query, key, value, g, beta, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=False
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]
    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale
    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim).to(value)
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None else initial_state.to(value)
    )
    for i in range(sequence_length):
        q_t = query[:, :, i]; k_t = key[:, :, i]; v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)
        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)
    if not output_final_state: last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state

MODEL_PATH = Path("models/qwen3_5_9b")
with open(MODEL_PATH / "config.json") as f:
    config = json.load(f)
text_cfg = config.get("text_config", config)

hs = text_cfg["hidden_size"]
n_kh, dk = text_cfg.get("linear_num_key_heads", 16), text_cfg.get("linear_key_head_dim", 128)
n_vh, dv = text_cfg.get("linear_num_value_heads", 32), text_cfg.get("linear_value_head_dim", 128)
d_qk, d_v = n_kh * dk, n_vh * dv
d_conv = d_qk * 2 + d_v
ks = text_cfg.get("linear_conv_kernel_dim", 4)
eps = text_cfg["rms_norm_eps"]

def load_w(name):
    for f in sorted(MODEL_PATH.glob("*.safetensors")):
        sd = safetensors.torch.load_file(str(f), device="cpu")
        if name in sd: return sd[name]
    raise KeyError(name)

prefix = "model.language_model.layers.0"
embed_w = load_w("model.language_model.embed_tokens.weight")
norm_w = load_w(f"{prefix}.input_layernorm.weight")
qkv_proj_w = load_w(f"{prefix}.linear_attn.in_proj_qkv.weight")
z_proj_w = load_w(f"{prefix}.linear_attn.in_proj_z.weight")
b_proj_w = load_w(f"{prefix}.linear_attn.in_proj_b.weight")
a_proj_w = load_w(f"{prefix}.linear_attn.in_proj_a.weight")
A_log = load_w(f"{prefix}.linear_attn.A_log")
dt_bias = load_w(f"{prefix}.linear_attn.dt_bias")
conv_w = load_w(f"{prefix}.linear_attn.conv1d.weight")
norm_gated_w = load_w(f"{prefix}.linear_attn.norm.weight")
o_proj_w = load_w(f"{prefix}.linear_attn.out_proj.weight")
lm_head_w = load_w("lm_head.weight")
final_norm_w = load_w("model.language_model.norm.weight")

from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file(str(MODEL_PATH / "tokenizer.json"))
prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nSay hello<|im_end|>\n<|im_start|>assistant\nHello"
input_ids = tokenizer.encode(prompt).ids
seq_len = len(input_ids)
print(f"Prompt: {prompt}")
print(f"Tokens ({seq_len}): {input_ids}")

# Forward pass in F32
hidden = embed_w[input_ids].float().unsqueeze(0)  # [1, seq, hs]

def rms_norm(x, w, eps):
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * w.float()

normed = rms_norm(hidden, norm_w, eps)
qkv = normed @ qkv_proj_w.float().T
z = normed @ z_proj_w.float().T
b = normed @ b_proj_w.float().T
a = normed @ a_proj_w.float().T

# Conv1d (channel-first, depthwise, causal)
qkv_cf = qkv.transpose(1, 2)  # [1, d_conv, seq]
padded = F.pad(qkv_cf, (ks - 1, 0))
conv_out = F.conv1d(padded, conv_w.float(), bias=None, groups=d_conv)
conv_out = F.silu(conv_out)
conv_out = conv_out.transpose(1, 2)  # [1, seq, d_conv]

q_raw, k_raw, v = conv_out.split([d_qk, d_qk, d_v], dim=-1)

beta = b.sigmoid()
g = -A_log.float().exp() * F.softplus(a.float() + dt_bias.float())

q_heads = q_raw.reshape(1, seq_len, n_kh, dk)
k_heads = k_raw.reshape(1, seq_len, n_kh, dk)
v_heads = v.reshape(1, seq_len, n_vh, dv)

heads_per_kv = n_vh // n_kh
q_expanded = q_heads.repeat_interleave(heads_per_kv, dim=2)
k_expanded = k_heads.repeat_interleave(heads_per_kv, dim=2)

# Run recurrent (simpler, easier to debug)
attn_out, state = torch_recurrent_gated_delta_rule(
    q_expanded, k_expanded, v_heads,
    g=g, beta=beta,
    initial_state=None, output_final_state=True,
    use_qk_l2norm_in_kernel=True,
)

print(f"\nDelta rule out:")
print(f"  shape: {attn_out.shape}")
print(f"  last token, head 0, first 5: {attn_out[0, -1, 0, :5].tolist()}")
print(f"  last token rms: {attn_out[0, -1].pow(2).mean().sqrt().item():.6f}")

# Gated RMS norm
attn_flat = attn_out.reshape(-1, dv)
z_flat = z.reshape(1, seq_len, n_vh, dv).reshape(-1, dv)
variance = attn_flat.pow(2).mean(-1, keepdim=True)
normed_attn = attn_flat * torch.rsqrt(variance + eps) * norm_gated_w.float()
normed_attn = normed_attn * F.silu(z_flat)
normed_attn = normed_attn.reshape(1, seq_len, -1)

print(f"Normed attn last rms: {normed_attn[0, -1].pow(2).mean().sqrt().item():.6f}")

o_out = normed_attn @ o_proj_w.float().T
result = hidden + o_out

print(f"\nAfter layer 0:")
print(f"  last token rms: {result[0, -1].pow(2).mean().sqrt().item():.6f}")
print(f"  first 5: {result[0, -1, :5].tolist()}")

# Compute logits from just layer 0 (for comparison with C++)
normed_final = rms_norm(result[:, -1:], final_norm_w, eps)
logits = normed_final.squeeze() @ lm_head_w.float().T
top5 = logits.topk(5)
print(f"\nLogits (1 layer only) top-5:")
for i in range(5):
    tid = top5.indices[i].item()
    val = top5.values[i].item()
    print(f"  {i}: token={tid} ({repr(tokenizer.decode([tid]))}) logit={val:.4f}")
print(f"Logits: min={logits.min().item():.4f}, max={logits.max().item():.4f}, mean={logits.mean().item():.4f}")

# Also check token 198 specifically
print(f"\nLogit[198]: {logits[198].item():.4f}")
print(f"Logit[220]: {logits[220].item():.4f}")
