"""Debug: compare C++ layer 0 output with pure Python reference."""
import numpy as np
import safetensors
import json
from pathlib import Path
import struct

MODEL_PATH = Path("models/qwen3_5_9b")

# Load config
with open(MODEL_PATH / "config.json") as f:
    config = json.load(f)
text_cfg = config.get("text_config", config)

hs = text_cfg["hidden_size"]  # 4096
n_kh = text_cfg.get("linear_num_key_heads", 16)
dk = text_cfg.get("linear_key_head_dim", 128)
n_vh = text_cfg.get("linear_num_value_heads", 32)
dv = text_cfg.get("linear_value_head_dim", 128)
d_qk = n_kh * dk  # 2048
d_v = n_vh * dv    # 4096
d_conv = d_qk * 2 + d_v  # 8192
ks = text_cfg.get("linear_conv_kernel_dim", 4)
eps = text_cfg["rms_norm_eps"]

print(f"Config: hs={hs}, n_kh={n_kh}, dk={dk}, n_vh={n_vh}, dv={dv}")
print(f"d_qk={d_qk}, d_v={d_v}, d_conv={d_conv}, ks={ks}")

# Load weights
def load_tensor(name):
    import torch
    for f in sorted(MODEL_PATH.glob("*.safetensors")):
        data = safetensors.safe_open(f, framework="pt", device="cpu")
        if name in data.keys():
            t = data.get_tensor(name)
            return t.float().numpy()
    raise KeyError(f"Weight {name} not found")

# Load needed weights for layer 0
prefix = "model.language_model.layers.0"
embed_w = load_tensor("model.language_model.embed_tokens.weight")
norm_w = load_tensor(f"{prefix}.input_layernorm.weight")
qkv_proj_w = load_tensor(f"{prefix}.linear_attn.in_proj_qkv.weight")
z_proj_w = load_tensor(f"{prefix}.linear_attn.in_proj_z.weight")
b_proj_w = load_tensor(f"{prefix}.linear_attn.in_proj_b.weight")
a_proj_w = load_tensor(f"{prefix}.linear_attn.in_proj_a.weight")
A_log = load_tensor(f"{prefix}.linear_attn.A_log")
dt_bias = load_tensor(f"{prefix}.linear_attn.dt_bias")
conv_w = load_tensor(f"{prefix}.linear_attn.conv1d.weight")
norm_gated_w = load_tensor(f"{prefix}.linear_attn.norm.weight")
o_proj_w = load_tensor(f"{prefix}.linear_attn.out_proj.weight")

print(f"\nWeight shapes:")
print(f"  embed: {embed_w.shape}")
print(f"  norm: {norm_w.shape}")
print(f"  qkv_proj: {qkv_proj_w.shape}")
print(f"  z_proj: {z_proj_w.shape}")
print(f"  b_proj: {b_proj_w.shape}")
print(f"  a_proj: {a_proj_w.shape}")
print(f"  A_log: {A_log.shape} {A_log.dtype}")
print(f"  dt_bias: {dt_bias.shape} {dt_bias.dtype}")
print(f"  conv_w: {conv_w.shape}")
print(f"  norm_gated: {norm_gated_w.shape}")
print(f"  o_proj: {o_proj_w.shape}")

# Convert to float32 for computation
def to_f32(x):
    return x.astype(np.float32) if x.dtype != np.float32 else x

embed_w = to_f32(embed_w)
norm_w = to_f32(norm_w)
qkv_proj_w = to_f32(qkv_proj_w)
z_proj_w = to_f32(z_proj_w)
b_proj_w = to_f32(b_proj_w)
a_proj_w = to_f32(a_proj_w)
A_log = to_f32(A_log)
dt_bias = to_f32(dt_bias)
conv_w_raw = conv_w
if conv_w.ndim == 3:
    conv_w = conv_w[:, 0, :]
conv_w = to_f32(conv_w)
norm_gated_w = to_f32(norm_gated_w)
o_proj_w = to_f32(o_proj_w)

# Use same input as test script
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file(str(MODEL_PATH / "tokenizer.json"))
prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello, who are you?<|im_end|>\n<|im_start|>assistant\n"
input_ids = tokenizer.encode(prompt).ids
seq_len = len(input_ids)
print(f"\nInput tokens ({seq_len}): {input_ids}")

# Step 1: Embedding
hidden = embed_w[input_ids]  # [seq, hs]
print(f"\nEmbedding: shape={hidden.shape}, min={hidden.min():.6f}, max={hidden.max():.6f}, rms={np.sqrt(np.mean(hidden[-1]**2)):.6f}")

# Step 2: RMS Norm
def rms_norm(x, w, eps):
    variance = np.mean(x ** 2, axis=-1, keepdims=True)
    return x * (1.0 / np.sqrt(variance + eps)) * w

normed = rms_norm(hidden, norm_w, eps)
print(f"Normed: shape={normed.shape}, min={normed.min():.6f}, max={normed.max():.6f}")
print(f"  normed[-1,:5] = {normed[-1,:5]}")

# Step 3: Projections (linear = x @ W.T)
qkv = normed @ qkv_proj_w.T  # [seq, d_conv]
z = normed @ z_proj_w.T      # [seq, d_v]
b = normed @ b_proj_w.T      # [seq, n_vh]
a = normed @ a_proj_w.T      # [seq, n_vh]

print(f"QKV proj: shape={qkv.shape}, min={qkv.min():.4f}, max={qkv.max():.4f}")
print(f"  qkv[-1,:5] = {qkv[-1,:5]}")

# Step 4: Causal Conv1d + SiLU
# Input: [seq, d_conv], Weight: [d_conv, ks], depthwise
def causal_conv1d_silu(x, w, seq_len, d_inner, kernel_size):
    out = np.zeros_like(x)
    for t in range(seq_len):
        for c in range(d_inner):
            acc = 0.0
            for k in range(kernel_size):
                t_in = t - kernel_size + 1 + k
                if t_in >= 0:
                    acc += w[c, k] * x[t_in, c]
            # SiLU
            out[t, c] = acc / (1.0 + np.exp(-acc))
    return out

conv_out = causal_conv1d_silu(qkv, conv_w, seq_len, d_conv, ks)
print(f"Conv out: shape={conv_out.shape}, min={conv_out.min():.4f}, max={conv_out.max():.4f}")
print(f"  conv_out[-1,:5] = {conv_out[-1,:5]}")

# Step 5: Split Q, K, V
q_raw = conv_out[:, :d_qk]         # [seq, d_qk] = [seq, n_kh*dk]
k_raw = conv_out[:, d_qk:d_qk*2]   # [seq, d_qk]
v = conv_out[:, d_qk*2:]            # [seq, d_v] = [seq, n_vh*dv]

# Step 6: Gates
softplus = lambda x: np.where(x > 20, x, np.log1p(np.exp(x)))
gate_g = np.exp(-np.exp(A_log) * softplus(a + dt_bias))  # [seq, n_vh]
beta = 1.0 / (1.0 + np.exp(-b))  # sigmoid  [seq, n_vh]

print(f"Gates: g min={gate_g.min():.6f}, max={gate_g.max():.6f}, mean={gate_g.mean():.6f}")
print(f"  beta min={beta.min():.6f}, max={beta.max():.6f}, mean={beta.mean():.6f}")

# Step 7: L2 normalize Q, K + repeat_interleave + scale
def l2norm(x, eps=1e-6):
    norm_sq = np.sum(x ** 2, axis=-1, keepdims=True)
    return x * (1.0 / np.sqrt(norm_sq + eps))

q_heads = q_raw.reshape(seq_len, n_kh, dk)  # [seq, n_kh, dk]
k_heads = k_raw.reshape(seq_len, n_kh, dk)
q_normed = l2norm(q_heads)
k_normed = l2norm(k_heads)

# repeat_interleave: each kh head expands to n_vh/n_kh = 2 heads
heads_per_kv = n_vh // n_kh
q_expanded = np.repeat(q_normed, heads_per_kv, axis=1)  # [seq, n_vh, dk]
k_expanded = np.repeat(k_normed, heads_per_kv, axis=1)  # [seq, n_vh, dk]

# Scale Q by 1/sqrt(dk)
scale = 1.0 / np.sqrt(dk)
q_expanded = q_expanded * scale

print(f"Q expanded: shape={q_expanded.shape}, rms={np.sqrt(np.mean(q_expanded[-1]**2)):.6f}")
print(f"K expanded: shape={k_expanded.shape}, rms={np.sqrt(np.mean(k_expanded[-1]**2)):.6f}")

# Step 8: Delta rule (chunk = sequential)
v_heads = v.reshape(seq_len, n_vh, dv)
state = np.zeros((n_vh, dv, dk), dtype=np.float32)

attn_out = np.zeros((seq_len, n_vh, dv), dtype=np.float32)
for t in range(seq_len):
    for h in range(n_vh):
        g = gate_g[t, h]
        b = beta[t, h]
        # retrieved = S[h] @ k[h]
        retrieved = state[h] @ k_expanded[t, h]  # [dv]
        delta = b * (v_heads[t, h] - retrieved)    # [dv]
        state[h] = g * state[h] + np.outer(delta, k_expanded[t, h])  # [dv, dk]
        attn_out[t, h] = state[h] @ q_expanded[t, h]  # [dv]

print(f"Delta rule out: shape={attn_out.shape}")
print(f"  attn_out[-1,0,:5] = {attn_out[-1,0,:5]}")
print(f"  rms={np.sqrt(np.mean(attn_out[-1]**2)):.6f}")

# Step 9: Gated RMS Norm
z_heads = z.reshape(seq_len, n_vh, dv)
def gated_rms_norm(x, z, w, eps):
    # x: [seq*n_vh, dv], z: [seq*n_vh, dv], w: [dv]
    variance = np.mean(x ** 2, axis=-1, keepdims=True)
    normed = x * (1.0 / np.sqrt(variance + eps)) * w
    silu_z = z * (1.0 / (1.0 + np.exp(-z)))  # SiLU
    return normed * silu_z

normed_out = gated_rms_norm(
    attn_out.reshape(-1, dv),
    z_heads.reshape(-1, dv),
    norm_gated_w,
    eps
).reshape(seq_len, n_vh * dv)

print(f"Normed out: shape={normed_out.shape}, rms={np.sqrt(np.mean(normed_out[-1]**2)):.6f}")
print(f"  normed_out[-1,:5] = {normed_out[-1,:5]}")

# Step 10: Output projection
o_out = normed_out @ o_proj_w.T  # [seq, hs]
print(f"O proj: shape={o_out.shape}, rms={np.sqrt(np.mean(o_out[-1]**2)):.6f}")
print(f"  o_out[-1,:5] = {o_out[-1,:5]}")

# Step 11: Residual add
result = hidden + o_out
print(f"\nAfter layer 0 (attention + residual):")
print(f"  shape={result.shape}")
print(f"  last token: min={result[-1].min():.6f}, max={result[-1].max():.6f}, rms={np.sqrt(np.mean(result[-1]**2)):.6f}")
print(f"  result[-1,:5] = {result[-1,:5]}")
