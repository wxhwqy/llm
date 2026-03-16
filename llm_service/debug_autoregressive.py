"""Autoregressive generation with full 32-layer reference implementation."""
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

def delta_rule_recurrent_with_state(q, k, v, g, beta, state, use_l2norm=True):
    """Recurrent delta rule with persistent state. q,k: [1,1,n_vh,dk], v: [1,1,n_vh,dv]"""
    if use_l2norm:
        q = l2norm(q.float(), dim=-1, eps=1e-6)
        k = l2norm(k.float(), dim=-1, eps=1e-6)
    q, k, v = q.float(), k.float(), v.float()
    B, S, H, dk = q.shape
    dv = v.shape[-1]
    scale = 1.0 / math.sqrt(dk)
    q = q * scale
    g = g.float()
    beta = beta.float()

    out = torch.zeros(B, S, H, dv)
    for t in range(S):
        q_t = q[0, t]
        k_t = k[0, t]
        v_t = v[0, t]
        g_t = g[0, t].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[0, t].unsqueeze(-1)

        state = state * g_t
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem[0]) * beta_t
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        out[0, t] = (state * q_t.unsqueeze(-1)).sum(dim=-2)[0]

    return out, state

def apply_rope(x, pos_ids, theta, rotary_dim, mrope_section):
    B, S, nh, ahd = x.shape
    half_rotary = rotary_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half_rotary, dtype=torch.float32) * 2 / rotary_dim))
    freqs = torch.outer(pos_ids.float(), inv_freq)
    cos = freqs.cos().unsqueeze(0).unsqueeze(2)
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
nh = text_cfg["num_attention_heads"]
nkvh = text_cfg["num_key_value_heads"]
ahd = text_cfg.get("head_dim", 256)
theta = 10000000.0
rotary_dim = int(ahd * text_cfg.get("partial_rotary_factor", 0.25))
layer_types = text_cfg["layer_types"]
nl = text_cfg["num_hidden_layers"]

print(f"Config: {nl} layers, hs={hs}")

# Load all weights
print("Loading weights...")
all_weights = {}
for f in sorted(MODEL_PATH.glob("*.safetensors")):
    sd = safetensors.torch.load_file(str(f), device="cpu")
    for k, v in sd.items():
        if k.startswith("model.language_model."):
            all_weights[k[len("model.language_model."):]] = v
        elif k == "lm_head.weight":
            all_weights[k] = v
        elif not k.startswith("model.visual.") and not k.startswith("mtp."):
            all_weights[k] = v
print(f"Loaded {len(all_weights)} weights")

def W(name):
    return all_weights[name].float()

from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file(str(MODEL_PATH / "tokenizer.json"))
prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nSay hello<|im_end|>\n<|im_start|>assistant\n"
input_ids = tokenizer.encode(prompt).ids
print(f"Prompt tokens ({len(input_ids)}): {input_ids}")

# Initialize persistent state for DeltaNet layers and KV caches for FA layers
dn_states = {}  # layer -> [1, n_vh, dk, dv] state tensor
dn_conv_states = {}  # layer -> [1, d_conv, ks] conv state
fa_kv_caches = {}  # layer -> (k_cache, v_cache) lists

for layer in range(nl):
    lt = layer_types[layer]
    if lt == "linear_attention":
        dn_states[layer] = torch.zeros(1, n_vh, dk, dv)
        dn_conv_states[layer] = torch.zeros(1, d_conv, ks)
    else:
        fa_kv_caches[layer] = ([], [])  # will append k, v tensors

def forward_one_pass(tokens, start_pos):
    """Forward pass for a sequence of tokens starting at position start_pos."""
    seq_len = len(tokens)
    hidden = W("embed_tokens.weight")[tokens].unsqueeze(0)  # [1, seq, hs]

    for layer in range(nl):
        lt = layer_types[layer]
        prefix = f"layers.{layer}"
        residual = hidden.clone()
        normed = rms_norm(hidden, W(f"{prefix}.input_layernorm.weight"), eps)

        if lt == "linear_attention":
            # Projections
            qkv = normed @ W(f"{prefix}.linear_attn.in_proj_qkv.weight").T
            z = normed @ W(f"{prefix}.linear_attn.in_proj_z.weight").T
            b = normed @ W(f"{prefix}.linear_attn.in_proj_b.weight").T
            a = normed @ W(f"{prefix}.linear_attn.in_proj_a.weight").T

            # Conv1d with state
            cw = all_weights[f"{prefix}.linear_attn.conv1d.weight"].float()

            # Update conv state and apply conv
            conv_state = dn_conv_states[layer]  # [1, d_conv, ks]

            if seq_len > 1:
                # Full sequence mode: pad and convolve
                qkv_cf = qkv.transpose(1, 2)  # [1, d_conv, seq]
                padded = F.pad(qkv_cf, (ks - 1, 0))
                conv_out = F.conv1d(padded, cw, bias=None, groups=d_conv)
                conv_out = F.silu(conv_out).transpose(1, 2)
                # Save last ks values as conv state
                for c in range(d_conv):
                    for ki in range(ks):
                        t_idx = seq_len - ks + ki
                        if t_idx >= 0:
                            conv_state[0, c, ki] = qkv[0, t_idx, c]
                        # else keep zero
                dn_conv_states[layer] = conv_state
            else:
                # Step mode: shift state, add new value, convolve
                conv_state = torch.cat([conv_state[:, :, 1:], qkv.transpose(1, 2)], dim=-1)
                dn_conv_states[layer] = conv_state
                # Depthwise conv: each channel independently
                conv_out = torch.zeros(1, 1, d_conv)
                for c in range(d_conv):
                    val = 0.0
                    for ki in range(ks):
                        val += conv_state[0, c, ki].item() * cw[c, 0, ki].item()
                    conv_out[0, 0, c] = val / (1.0 + math.exp(-val))  # SiLU

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

            attn_out, new_state = delta_rule_recurrent_with_state(
                q_exp, k_exp, v_heads, g, beta, dn_states[layer])
            dn_states[layer] = new_state

            nw = W(f"{prefix}.linear_attn.norm.weight")
            attn_flat = attn_out.reshape(-1, dv)
            z_flat = z.reshape(1, seq_len, n_vh, dv).reshape(-1, dv)
            normed_attn = gated_rms_norm(attn_flat, z_flat, nw, eps)
            normed_attn = normed_attn.reshape(1, seq_len, -1)

            o_out = normed_attn @ W(f"{prefix}.linear_attn.out_proj.weight").T
            hidden = residual + o_out
        else:
            # Full attention layer
            q_full = normed @ W(f"{prefix}.self_attn.q_proj.weight").T
            k_out = normed @ W(f"{prefix}.self_attn.k_proj.weight").T
            v_out = normed @ W(f"{prefix}.self_attn.v_proj.weight").T

            q_gate = q_full.view(1, seq_len, nh, 2 * ahd)
            q_states, gate = q_gate.chunk(2, dim=-1)
            gate_flat = gate.reshape(1, seq_len, -1)

            k_states = k_out.view(1, seq_len, nkvh, ahd)
            v_states = v_out.view(1, seq_len, nkvh, ahd)

            q_normed = rms_norm(q_states.reshape(-1, ahd), W(f"{prefix}.self_attn.q_norm.weight"), eps).reshape(1, seq_len, nh, ahd)
            k_normed = rms_norm(k_states.reshape(-1, ahd), W(f"{prefix}.self_attn.k_norm.weight"), eps).reshape(1, seq_len, nkvh, ahd)

            pos_ids = torch.arange(start_pos, start_pos + seq_len)
            q_rope = apply_rope(q_normed, pos_ids, theta, rotary_dim, None)
            k_rope = apply_rope(k_normed, pos_ids, theta, rotary_dim, None)

            # Update KV cache
            k_cache_list, v_cache_list = fa_kv_caches[layer]
            k_cache_list.append(k_rope)
            v_cache_list.append(v_states)
            k_full = torch.cat(k_cache_list, dim=1)  # [1, total_len, nkvh, ahd]
            v_full = torch.cat(v_cache_list, dim=1)

            kv_groups = nh // nkvh
            k_exp = k_full.repeat_interleave(kv_groups, dim=2)
            v_exp = v_full.repeat_interleave(kv_groups, dim=2)

            total_len = k_full.shape[1]
            q_t = q_rope.transpose(1, 2).float()
            k_t = k_exp.transpose(1, 2).float()
            v_t = v_exp.transpose(1, 2).float()

            scale = 1.0 / math.sqrt(ahd)
            scores = (q_t @ k_t.transpose(-1, -2)) * scale
            # Causal mask
            mask = torch.full((seq_len, total_len), float('-inf'))
            for i in range(seq_len):
                mask[i, :start_pos + i + 1] = 0.0
            scores = scores + mask.unsqueeze(0).unsqueeze(0)
            attn = F.softmax(scores, dim=-1)
            attn_out = (attn @ v_t).transpose(1, 2)

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

    # Final norm + logits (last token only)
    normed_final = rms_norm(hidden[:, -1:], W("norm.weight"), eps)
    logits = normed_final.squeeze() @ W("lm_head.weight").T
    return logits

# Generate tokens
print("\nGenerating...")
generated = []
pos = 0

# Prefill
logits = forward_one_pass(input_ids, 0)
pos = len(input_ids)
next_token = logits.argmax().item()
generated.append(next_token)
print(f"  Token 0: {next_token} ({repr(tokenizer.decode([next_token]))}) logit={logits[next_token].item():.4f}")

# Decode
for step in range(1, 16):
    logits = forward_one_pass([next_token], pos)
    pos += 1
    next_token = logits.argmax().item()
    generated.append(next_token)
    decoded = tokenizer.decode(generated)
    print(f"  Token {step}: {next_token} ({repr(tokenizer.decode([next_token]))}) logit={logits[next_token].item():.4f} | so far: {repr(decoded[:60])}")
    if next_token in (151643, 248046):  # EOS tokens
        break

print(f"\nFull output: {repr(tokenizer.decode(generated))}")
