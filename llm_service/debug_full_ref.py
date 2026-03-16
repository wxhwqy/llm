"""Full 32-layer reference implementation to compare with C++."""
import torch
import torch.nn.functional as F
import json, sys, time, math
from pathlib import Path
import safetensors.torch

def l2norm(x, dim=-1, eps=1e-6):
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm

def rms_norm(x, w, eps):
    variance = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + eps) * w.float()).to(x.dtype)

def gated_rms_norm(x, z, w, eps):
    variance = x.float().pow(2).mean(-1, keepdim=True)
    normed = x.float() * torch.rsqrt(variance + eps) * w.float()
    return (normed * F.silu(z.float())).to(x.dtype)

def delta_rule_recurrent(q, k, v, g, beta, use_l2norm=True):
    """Simplified recurrent delta rule. q,k: [1,seq,n_vh,dk], v: [1,seq,n_vh,dv]"""
    if use_l2norm:
        q = l2norm(q.float(), dim=-1, eps=1e-6)
        k = l2norm(k.float(), dim=-1, eps=1e-6)
    q, k, v = q.float(), k.float(), v.float()
    B, S, H, dk = q.shape
    dv = v.shape[-1]
    scale = 1.0 / math.sqrt(dk)
    q = q * scale

    g = g.float()  # [1, S, H] — log decay
    beta = beta.float()  # [1, S, H]

    state = torch.zeros(B, H, dk, dv)
    out = torch.zeros(B, S, H, dv)

    for t in range(S):
        q_t = q[0, t]  # [H, dk]
        k_t = k[0, t]
        v_t = v[0, t]  # [H, dv]
        g_t = g[0, t].exp().unsqueeze(-1).unsqueeze(-1)  # [H, 1, 1]
        beta_t = beta[0, t].unsqueeze(-1)  # [H, 1]

        state = state * g_t
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)  # [1, H, dv]
        delta = (v_t - kv_mem[0]) * beta_t
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        out[0, t] = (state * q_t.unsqueeze(-1)).sum(dim=-2)[0]

    return out, state

def apply_rope(x, pos_ids, theta, rotary_dim, mrope_section):
    """Apply M-RoPE to x [B, S, nh, ahd]. For text-only, all dims same."""
    B, S, nh, ahd = x.shape
    # For text-only input, all 3 position dimensions are the same
    # so the interleaving doesn't matter — just use standard partial RoPE
    half_rotary = rotary_dim // 2
    # Compute frequencies
    inv_freq = 1.0 / (theta ** (torch.arange(0, half_rotary, dtype=torch.float32) * 2 / rotary_dim))
    # pos_ids: [S]
    freqs = torch.outer(pos_ids.float(), inv_freq)  # [S, half_rotary]
    cos = freqs.cos().unsqueeze(0).unsqueeze(2)  # [1, S, 1, half_rotary]
    sin = freqs.sin().unsqueeze(0).unsqueeze(2)

    x_rot = x[..., :rotary_dim].float()
    x_pass = x[..., rotary_dim:]

    x1 = x_rot[..., :half_rotary]
    x2 = x_rot[..., half_rotary:]
    rotated = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return torch.cat([rotated.to(x.dtype), x_pass], dim=-1)

MODEL_PATH = Path("models/qwen3_5_9b")
with open(MODEL_PATH / "config.json") as f:
    config = json.load(f)
text_cfg = config.get("text_config", config)

hs = text_cfg["hidden_size"]
di = text_cfg["intermediate_size"]
n_kh, dk = text_cfg.get("linear_num_key_heads", 16), text_cfg.get("linear_key_head_dim", 128)
n_vh, dv = text_cfg.get("linear_num_value_heads", 32), text_cfg.get("linear_value_head_dim", 128)
d_qk, d_v = n_kh * dk, n_vh * dv
d_conv = d_qk * 2 + d_v
ks = text_cfg.get("linear_conv_kernel_dim", 4)
eps = text_cfg["rms_norm_eps"]
nh = text_cfg["num_attention_heads"]  # 16
nkvh = text_cfg["num_key_value_heads"]  # 4
ahd = text_cfg.get("head_dim", 256)
theta = 10000000.0
rotary_dim = int(ahd * text_cfg.get("partial_rotary_factor", 0.25))
layer_types = text_cfg["layer_types"]
nl = text_cfg["num_hidden_layers"]

print(f"Config: {nl} layers, hs={hs}, di={di}")
print(f"DN: n_kh={n_kh}, dk={dk}, n_vh={n_vh}, dv={dv}")
print(f"FA: nh={nh}, nkvh={nkvh}, ahd={ahd}, rotary={rotary_dim}")

# Load all weights
print("Loading weights...")
t0 = time.time()
all_weights = {}
for f in sorted(MODEL_PATH.glob("*.safetensors")):
    sd = safetensors.torch.load_file(str(f), device="cpu")
    for k, v in sd.items():
        if k.startswith("model.language_model."):
            new_k = k[len("model.language_model."):]
            all_weights[new_k] = v
        elif k == "lm_head.weight":
            all_weights[k] = v
        elif not k.startswith("model.visual.") and not k.startswith("mtp."):
            all_weights[k] = v
print(f"Loaded {len(all_weights)} weights in {time.time()-t0:.1f}s")

def W(name):
    return all_weights[name].float()

from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file(str(MODEL_PATH / "tokenizer.json"))
prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nSay hello<|im_end|>\n<|im_start|>assistant\nHello"
input_ids = tokenizer.encode(prompt).ids
seq_len = len(input_ids)
print(f"\nPrompt: {prompt}")
print(f"Tokens ({seq_len}): {input_ids}")

# Forward pass
print("\nRunning forward pass...")
hidden = W("embed_tokens.weight")[input_ids].unsqueeze(0)  # [1, seq, hs]

for layer in range(nl):
    t0 = time.time()
    lt = layer_types[layer]
    prefix = f"layers.{layer}"
    residual = hidden.clone()

    # Pre-attention norm
    normed = rms_norm(hidden, W(f"{prefix}.input_layernorm.weight"), eps)

    if lt == "linear_attention":
        # DeltaNet layer
        qkv = normed @ W(f"{prefix}.linear_attn.in_proj_qkv.weight").T
        z = normed @ W(f"{prefix}.linear_attn.in_proj_z.weight").T
        b = normed @ W(f"{prefix}.linear_attn.in_proj_b.weight").T
        a = normed @ W(f"{prefix}.linear_attn.in_proj_a.weight").T

        # Conv1d
        cw = all_weights[f"{prefix}.linear_attn.conv1d.weight"].float()
        qkv_cf = qkv.transpose(1, 2)
        padded = F.pad(qkv_cf, (ks - 1, 0))
        conv_out = F.conv1d(padded, cw, bias=None, groups=d_conv)
        conv_out = F.silu(conv_out).transpose(1, 2)

        q_raw, k_raw, v = conv_out.split([d_qk, d_qk, d_v], dim=-1)

        beta = b.sigmoid()
        al = all_weights[f"{prefix}.linear_attn.A_log"].float()
        dtb = all_weights[f"{prefix}.linear_attn.dt_bias"].float()
        g = -al.exp() * F.softplus(a.float() + dtb)

        q_heads = q_raw.reshape(1, seq_len, n_kh, dk)
        k_heads = k_raw.reshape(1, seq_len, n_kh, dk)
        v_heads = v.reshape(1, seq_len, n_vh, dv)

        hpk = n_vh // n_kh
        q_exp = q_heads.repeat_interleave(hpk, dim=2)
        k_exp = k_heads.repeat_interleave(hpk, dim=2)

        attn_out, _ = delta_rule_recurrent(q_exp, k_exp, v_heads, g, beta)

        # Gated RMS norm
        nw = W(f"{prefix}.linear_attn.norm.weight")
        attn_flat = attn_out.reshape(-1, dv)
        z_flat = z.reshape(1, seq_len, n_vh, dv).reshape(-1, dv)
        normed_attn = gated_rms_norm(attn_flat, z_flat, nw, eps)
        normed_attn = normed_attn.reshape(1, seq_len, -1)

        o_out = normed_attn @ W(f"{prefix}.linear_attn.out_proj.weight").T
        hidden = residual + o_out
    else:
        # Full attention layer
        q_full = normed @ W(f"{prefix}.self_attn.q_proj.weight").T  # [1, seq, 2*nh*ahd]
        k_out = normed @ W(f"{prefix}.self_attn.k_proj.weight").T
        v_out = normed @ W(f"{prefix}.self_attn.v_proj.weight").T

        # Split Q and gate (HF: reshape to [1, seq, nh, 2*ahd] then chunk)
        q_gate = q_full.view(1, seq_len, nh, 2 * ahd)
        q_states, gate = q_gate.chunk(2, dim=-1)  # [1, seq, nh, ahd] each
        gate_flat = gate.reshape(1, seq_len, -1)  # [1, seq, nh*ahd]

        k_states = k_out.view(1, seq_len, nkvh, ahd)
        v_states = v_out.view(1, seq_len, nkvh, ahd)

        # QK-Norm
        q_normed = rms_norm(q_states.reshape(-1, ahd), W(f"{prefix}.self_attn.q_norm.weight"), eps).reshape(1, seq_len, nh, ahd)
        k_normed = rms_norm(k_states.reshape(-1, ahd), W(f"{prefix}.self_attn.k_norm.weight"), eps).reshape(1, seq_len, nkvh, ahd)

        # Apply RoPE (simplified for text-only: all dims same)
        pos_ids = torch.arange(seq_len)
        q_rope = apply_rope(q_normed, pos_ids, theta, rotary_dim, None)
        k_rope = apply_rope(k_normed, pos_ids, theta, rotary_dim, None)

        # Standard GQA attention
        # Expand K/V heads
        kv_groups = nh // nkvh
        k_exp = k_rope.repeat_interleave(kv_groups, dim=2)
        v_exp = v_states.repeat_interleave(kv_groups, dim=2)

        # Attention: [1, nh, seq, ahd]
        q_t = q_rope.transpose(1, 2).float()
        k_t = k_exp.transpose(1, 2).float()
        v_t = v_exp.transpose(1, 2).float()

        scale = 1.0 / math.sqrt(ahd)
        scores = (q_t @ k_t.transpose(-1, -2)) * scale
        # Causal mask
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
        scores = scores + mask
        attn = F.softmax(scores, dim=-1)
        attn_out = (attn @ v_t).transpose(1, 2)  # [1, seq, nh, ahd]

        # Gating
        attn_flat = attn_out.reshape(1, seq_len, -1).float()
        attn_flat = attn_flat * torch.sigmoid(gate_flat.float())

        o_out = attn_flat @ W(f"{prefix}.self_attn.o_proj.weight").T
        hidden = residual + o_out

    # MLP
    residual = hidden.clone()
    normed_mlp = rms_norm(hidden, W(f"{prefix}.post_attention_layernorm.weight"), eps)
    gate_proj = normed_mlp @ W(f"{prefix}.mlp.gate_proj.weight").T
    up_proj = normed_mlp @ W(f"{prefix}.mlp.up_proj.weight").T
    mlp_out = (F.silu(gate_proj) * up_proj) @ W(f"{prefix}.mlp.down_proj.weight").T
    hidden = residual + mlp_out

    elapsed = time.time() - t0
    hs_last = hidden[0, -1]
    print(f"  Layer {layer:2d} ({lt[:6]:6s}): rms={hs_last.pow(2).mean().sqrt().item():.4f} "
          f"min={hs_last.min().item():.4f} max={hs_last.max().item():.4f} [{elapsed:.1f}s]")

# Final norm + logits
normed_final = rms_norm(hidden[:, -1:], W("norm.weight"), eps)
logits = normed_final.squeeze() @ W("lm_head.weight").T

print(f"\nLogits: min={logits.min().item():.4f}, max={logits.max().item():.4f}, mean={logits.mean().item():.4f}")
top10 = logits.topk(10)
print("Top-10:")
for i in range(10):
    tid = top10.indices[i].item()
    val = top10.values[i].item()
    print(f"  {i}: token={tid} ({repr(tokenizer.decode([tid]))}) logit={val:.4f}")
print(f"\nLogit[198] ('\\ n'): {logits[198].item():.4f}")
print(f"Logit[220] (' '): {logits[220].item():.4f}")
