# Qwen3.5 支持调研报告

## 1. 模型架构概览

**Qwen3.5-35B-A3B-FP8** 是一个 **多模态 MoE（Mixture of Experts）** 模型，架构名称 `Qwen3_5MoeForConditionalGeneration`。

### 核心参数

| 参数 | Qwen3 (现有) | Qwen3.5-35B-A3B | 差异 |
|------|-------------|-----------------|------|
| 架构 | Dense Transformer | Hybrid MoE (DeltaNet + Attention) | **全新** |
| hidden_size | 2048~5120 | 2048 | - |
| num_layers | 28~64 | 40 | - |
| num_attention_heads | 16~40 | 16 (full attn) / 16 K + 32 V (linear attn) | **不同** |
| num_kv_heads | 2~8 | 2 (full attn) | - |
| head_dim | 128 | 256 (full attn) / 128 (linear attn) | **不同** |
| intermediate_size | 8960~18944 | N/A (MoE替代) | **全新** |
| num_experts | 无 | **256** | **全新** |
| experts_per_token | 无 | **8 routed + 1 shared** | **全新** |
| moe_intermediate_size | 无 | 512 | **全新** |
| shared_expert_intermediate_size | 无 | 512 | **全新** |
| vocab_size | ~152k | 248,320 | 更大 |
| max_position_embeddings | 32k~128k | **262,144 (256K)** | 更大 |
| rope_theta | 1,000,000 | 10,000,000 | 不同 |
| RoPE 类型 | 标准 RoPE | **M-RoPE** (多维旋转，partial_rotary_factor=0.25) | **全新** |
| 注意力类型 | 全部 Full Attention | **混合：3×Linear + 1×Full（每4层循环）** | **全新** |
| MLP 类型 | Dense SwiGLU | **Sparse MoE + Shared Expert** | **全新** |
| 特殊机制 | QK-Norm | QK-Norm + **Attention Output Gate** + **Gated DeltaNet** | **全新** |
| 多模态 | 无 | **Vision Encoder（ViT）** | **全新** |
| MTP | 无 | **Multi-Token Prediction（1层）** | **全新** |

### 层结构模式（40层，10个循环）

```
每个循环 (×10):
  Layer 4n+0: Gated DeltaNet (Linear Attention) → MoE
  Layer 4n+1: Gated DeltaNet (Linear Attention) → MoE
  Layer 4n+2: Gated DeltaNet (Linear Attention) → MoE
  Layer 4n+3: Full Attention (Gated) → MoE
```

---

## 2. 新增架构组件详解

### 2.1 Gated DeltaNet（线性注意力，30/40层使用）

这是 Qwen3.5 最核心的新组件。它是一种**线性复杂度的注意力替代机制**，基于 Delta Rule 递推。

**前向传播流程：**

```
hidden_states [B, L, D]
    │
    ├─ in_proj_qkv → mixed_qkv [B, D_inner, L]
    ├─ in_proj_z   → z (门控向量)
    ├─ in_proj_b   → beta (更新门控)
    ├─ in_proj_a   → alpha (衰减因子)
    │
    ▼
Causal Conv1d (kernel_size=4, SiLU activation)
    │
    ▼
Split → Q [B, L, n_kh, dk], K [B, L, n_kh, dk], V [B, L, n_vh, dv]
    │
    ├─ beta = sigmoid(b)
    ├─ g = -exp(A_log) * softplus(a + dt_bias)   // 衰减率
    │
    ▼
Gated Delta Rule Recurrence:
  Prefill: chunk_gated_delta_rule(Q, K, V, g, beta)   // 分块并行
  Decode:  recurrent_gated_delta_rule(Q, K, V, g, beta, state)  // 逐步递推
    │
    ▼
Gated RMSNorm: output = norm(attn_out) * silu(z)
    │
    ▼
out_proj → output [B, L, D]
```

**需要的新算子：**

| 算子名称 | 功能 | 复杂度 |
|---------|------|--------|
| `causal_conv1d` | 因果一维卷积（kernel=4, SiLU），支持增量更新 | 中 |
| `chunk_gated_delta_rule` | 分块并行的门控 Delta Rule 递推（prefill） | **高** |
| `recurrent_gated_delta_rule` | 单步递推的门控 Delta Rule（decode） | 中 |
| `gated_rms_norm` | 门控 RMS 归一化：`norm(x) * silu(z)` | 低 |
| `softplus` | 激活函数：`log(1 + exp(x))` | 低 |

**DeltaNet 递推核心公式：**

```
# 对每个时间步 t:
S_t = g_t * S_{t-1} + beta_t * (v_t ⊗ k_t)    # 状态更新（外积）
o_t = S_t @ q_t                                  # 输出查询

其中：
  S ∈ R^{n_heads × d_v × d_k}   是递推状态矩阵
  g_t ∈ R^{n_heads}               是衰减门控
  beta_t ∈ R^{n_heads}            是更新门控
```

**Cache 结构：**
- `conv_states[layer]`: 卷积滑动窗口缓冲 `[B, D_inner, conv_kernel_size]`
- `recurrent_states[layer]`: 递推隐状态 `[B, n_heads, d_v, d_k]`

### 2.2 Mixture of Experts (MoE)

**所有40层的 MLP 都被 MoE 替代。**

```
hidden_states [B, L, D]
    │
    ├─── Router ──────────────────────────┐
    │    gate(hidden) → logits [B*L, 256] │
    │    softmax → top-8 experts selected  │
    │                                      │
    ├─── Routed Experts (×8 active) ──────┤
    │    For each selected expert:         │
    │      SwiGLU MLP (D→512→D)           │
    │    Weighted sum of expert outputs    │
    │                                      │
    ├─── Shared Expert (×1 always) ───────┤
    │    SwiGLU MLP (D→512→D)             │
    │    × sigmoid(shared_gate(hidden))    │
    │                                      │
    ▼                                      │
    output = routed_output + shared_output ◄┘
```

**需要的新算子：**

| 算子名称 | 功能 | 复杂度 |
|---------|------|--------|
| `moe_router_topk` | Softmax 路由 + Top-K 专家选择 | 中 |
| `moe_expert_dispatch` | Token → Expert 分发（scatter/gather） | **高** |
| `moe_expert_combine` | 专家输出加权聚合 | 中 |
| `sigmoid` | Sigmoid 激活（共享专家门控） | 低 |

**权重结构（每层）：**

```
experts.gate_proj.weight: [256, 512, 2048]   # 256个专家的gate权重
experts.up_proj.weight:   [256, 512, 2048]   # 256个专家的up权重
experts.down_proj.weight: [256, 2048, 512]   # 256个专家的down权重
router.weight:            [256, 2048]        # 路由器权重
shared_expert.gate_proj:  [512, 2048]        # 共享专家
shared_expert.up_proj:    [512, 2048]
shared_expert.down_proj:  [2048, 512]
shared_expert_gate.weight:[1, 2048]          # 共享专家门控
```

### 2.3 Gated Full Attention（10/40层使用）

在现有 Qwen3 Attention 基础上增加了 **Output Gate**：

```python
# 现有 Qwen3:
attn_output = O_proj(attention(Q, K, V))

# Qwen3.5:
Q, gate = chunk(Q_proj(x), 2)    # Q投影输出维度翻倍，拆分出gate
attn_output = attention(Q, K, V)
attn_output = attn_output * sigmoid(gate)   # 门控
attn_output = O_proj(attn_output)
```

**需要的新算子：**

| 算子名称 | 功能 | 复杂度 |
|---------|------|--------|
| `sigmoid_mul` | element-wise sigmoid 门控乘法 | 低 |

**注意：** Q_proj 的输出维度从 `num_heads * head_dim` 变为 `num_heads * head_dim * 2`（一半用作gate）。

### 2.4 M-RoPE（多维旋转位置编码）

与标准 RoPE 不同，M-RoPE 将位置编码分为3个维度段（时间、高度、宽度），支持多模态：

```
rope_parameters:
  mrope_section: [11, 11, 10]      # 频率维度分段
  partial_rotary_factor: 0.25      # 仅旋转25%的head_dim
  mrope_interleaved: true          # 交错排列
  rope_theta: 10,000,000

实际旋转维度 = head_dim * partial_rotary_factor = 256 * 0.25 = 64
频率分段: [11, 11, 10] 共32个频率对 → 64维

position_ids: [3, B, L]  → 三维位置（纯文本模式下三个维度相同）
```

**需要的新算子：**

| 算子名称 | 功能 | 复杂度 |
|---------|------|--------|
| `mrope` | 多维旋转位置编码（分段 + 部分旋转 + 交错） | 中 |

### 2.5 Multi-Token Prediction (MTP)

MTP 允许模型同时预测多个未来 token，用于推测解码加速：

```
mtp_num_hidden_layers: 1    # 1层MTP预测头
```

这是可选的推理加速特性，初期可不实现。

---

## 3. 新增算子完整清单

### 3.1 必须新增的算子（按优先级排序）

| # | 算子名称 | 输入/输出 | 说明 | 优先级 | 预估工作量 |
|---|---------|----------|------|--------|-----------|
| 1 | **`causal_conv1d`** | in: `[B, D, L]`, w: `[D, 1, K]` → out: `[B, D, L]` | 因果一维卷积 + SiLU，支持增量单步更新 | P0 | 3天 |
| 2 | **`chunk_gated_delta_rule`** | Q,K,V,g,beta → out, state | 分块并行门控 Delta Rule（prefill 核心） | P0 | 7-10天 |
| 3 | **`recurrent_gated_delta_rule`** | Q,K,V,g,beta,state → out, new_state | 单步递推门控 Delta Rule（decode 核心） | P0 | 5天 |
| 4 | **`moe_router_topk`** | hidden → expert_ids, weights | Softmax 路由 + Top-K 选择 | P0 | 2天 |
| 5 | **`moe_expert_dispatch`** | tokens, expert_ids → per_expert_tokens | Token 分发到专家（scatter） | P0 | 3天 |
| 6 | **`moe_expert_combine`** | expert_outputs, weights, expert_ids → output | 专家输出加权聚合（gather + weighted sum） | P0 | 2天 |
| 7 | **`gated_rms_norm`** | x, z, weight → `norm(x) * silu(z) * weight` | 门控归一化（DeltaNet 输出） | P0 | 1天 |
| 8 | **`mrope`** | Q, K, pos_ids_3d → Q_rotated, K_rotated | 多维 RoPE（分段、部分旋转、交错） | P0 | 2天 |
| 9 | **`sigmoid`** | x → sigmoid(x) | 共享专家门控 + 注意力输出门控 | P0 | 0.5天 |
| 10 | **`softplus`** | x → log(1+exp(x)) | DeltaNet 衰减因子计算 | P0 | 0.5天 |

### 3.2 可选/延后的算子

| # | 算子名称 | 说明 | 优先级 |
|---|---------|------|--------|
| 11 | `mtp_head` | Multi-Token Prediction 预测头 | P2 |
| 12 | `vision_patch_embed` | ViT 图像 Patch 嵌入 | P2 |
| 13 | `vision_attention` | ViT 自注意力（无因果掩码） | P2 |
| 14 | `gelu_tanh` | GELU(tanh) 激活（Vision Encoder） | P2 |
| 15 | `spatial_merge` | 视觉 token 空间合并 | P2 |

### 3.3 可复用的现有算子

| 现有算子 | 复用场景 | 是否需要修改 |
|---------|---------|------------|
| `embedding` | Token 嵌入（vocab 更大但接口不变） | 无需修改 |
| `linear` / `linear_fp8` | 所有线性投影、专家 MLP、LM Head | 无需修改 |
| `rms_norm` | input_layernorm, post_attn_layernorm, QK-Norm | 无需修改 |
| `self_attention` | Full Attention 层（10/40层） | **需要修改**：支持 partial RoPE + output gate |
| `add` | 残差连接 | 无需修改 |
| `swiglu` | 每个 Expert 的 MLP、共享 Expert 的 MLP | 无需修改 |
| `argmax` / `sample` | Token 采样 | 无需修改 |
| `rope` | 需要升级为 M-RoPE，或新写 `mrope` 算子替代 | **需要修改或替换** |

---

## 4. 需要修改的现有组件

### 4.1 模型层（C++）

```
需要新建:
  src/models/qwen3_5_moe.hpp    // Qwen3.5MoE 模型定义
  src/models/qwen3_5_moe.cpp    // 前向传播实现
  src/models/qwen3_5_moe_tp.hpp // TP 版本
  src/models/qwen3_5_moe_tp.cpp

关键新增数据结构:
  - Qwen3_5MoeConfig: 继承 Qwen3Config，增加 MoE/DeltaNet 参数
  - Qwen3_5MoeWeights: 包含 Expert 权重（3D tensor）、Router 权重、DeltaNet 投影
  - DeltaNetState: conv_state + recurrent_state 的 Cache 管理
```

### 4.2 Python 绑定层

```
需要新建:
  python/llaisys/models/qwen3_5_moe.py   // 权重加载、推理封装
  python/llaisys/libllaisys/models.py     // 新增 C API 绑定结构体

需要修改:
  python/llaisys/libllaisys/ops.py        // 注册新算子 FFI
```

### 4.3 C API 桥接层

```
需要新建:
  src/llaisys/qwen3_5_moe.cc   // C API 包装
  include/llaisys/models/qwen3_5_moe.h
```

### 4.4 FastAPI 服务层

```
需要修改:
  api/services/model_manager.py  // 识别 qwen3_5_moe 模型类型，加载对应类
  api/config.py                  // 新增模型配置项
  models.json                    // 新增 Qwen3.5 模型条目
```

---

## 5. Cache 架构对比

| 维度 | Qwen3 (现有) | Qwen3.5 |
|------|-------------|---------|
| Full Attention 层 | KV Cache `[max_seq, nkvh, dh]` | 相同（仅10/40层使用） |
| Linear Attention 层 | 无 | **Conv State** `[B, D_inner, 4]` + **Recurrent State** `[B, nH, dv, dk]` |
| 内存特征 | 随序列长度线性增长 | Conv+Recurrent 状态**固定大小**，KV Cache 仅10层 |
| 内存优势 | - | 约节省 75% 的 KV Cache 内存（30层不需要） |

**DeltaNet 的固定状态大小是 Qwen3.5 长上下文能力的关键优势。**

---

## 6. 推荐实施路径

### Phase 1: MoE 基础（2-3周）

1. 实现 `moe_router_topk` 算子
2. 实现 `moe_expert_dispatch` + `moe_expert_combine` 算子
3. 实现 `sigmoid` + `softplus` 基础算子
4. 构建 MoE MLP Block（可先用现有 `swiglu` + `linear` 组合 Expert 内部）
5. **验证**：单层 MoE 前向正确性

### Phase 2: Gated DeltaNet（3-4周）

1. 实现 `causal_conv1d` 算子（含增量更新模式）
2. 实现 `recurrent_gated_delta_rule`（decode 模式，较简单）
3. 实现 `chunk_gated_delta_rule`（prefill 模式，最复杂）
4. 实现 `gated_rms_norm` 算子
5. **验证**：单层 DeltaNet 前向正确性

### Phase 3: Full Attention 改造 + M-RoPE（1-2周）

1. 实现 `mrope` 或修改现有 `rope`（支持分段、部分旋转、交错）
2. 修改 `self_attention` 支持 output gate（sigmoid 门控）
3. 修改 Q_proj 支持双倍输出（Q + gate 拆分）
4. **验证**：单层 Gated Full Attention 正确性

### Phase 4: 模型集成（2-3周）

1. 实现 `Qwen3_5MoeModel` C++ 类（混合层调度）
2. 实现 DeltaNet Cache 管理（conv_state + recurrent_state 生命周期）
3. Python 权重加载（Expert 3D 权重映射）
4. C API 桥接
5. FastAPI 集成
6. **端到端验证**：完整模型推理 + 与 HuggingFace 参考实现对比

### Phase 5: 优化 + TP（2-3周）

1. Expert Parallelism（跨 GPU 分配专家）
2. DeltaNet CUDA kernel 优化（chunk 并行化）
3. MoE dispatch 优化（减少 scatter/gather 开销）
4. 内存池化（Expert 权重按需加载）

---

## 7. 技术风险与难点

| 风险 | 说明 | 缓解策略 |
|------|------|---------|
| **DeltaNet chunk 算子实现** | `chunk_gated_delta_rule` 是最复杂的新算子，涉及分块矩阵递推的并行化，CUDA 实现难度高 | 参考 `fla` (Flash Linear Attention) 库的 Triton 实现，先做 CPU 参考实现 |
| **MoE 负载均衡** | 256个专家 × Top-8 路由，token 分布不均会导致 GPU 利用率低 | 实现 padding-based 和 capacity-based 两种调度策略 |
| **Expert 权重内存** | 256 × 3 × [512, 2048] FP8 ≈ 0.8GB/层 × 40层 = 32GB 仅专家权重 | FP8 量化已大幅压缩；TP + Expert Parallelism 分摊 |
| **混合 Cache 管理** | 同时管理 KV Cache（10层）+ DeltaNet State（30层），生命周期不同 | 设计统一的 HybridCache 接口 |
| **M-RoPE 正确性** | 交错模式 + 部分旋转容易出错 | 逐维度单元测试 + 与 HF 参考输出对比 |
| **transformers 依赖** | Qwen3.5 需要 transformers main 分支（未正式发布） | 锁定特定 commit，提取 tokenizer 配置独立使用 |

---

## 8. 总结

支持 Qwen3.5 相比现有 Qwen3 是一次 **架构级升级**，核心挑战在于：

1. **10个全新算子**（其中 `chunk_gated_delta_rule` 复杂度最高）
2. **MoE 调度系统**（路由、分发、聚合的完整 pipeline）
3. **混合注意力架构**（30层 DeltaNet + 10层 Gated Attention 的异构调度）
4. **新型 Cache 机制**（Conv State + Recurrent State + KV Cache 共存）

预估总工作量：**10-15周**（1人全职），建议 2-3 人并行开发，可压缩到 **5-6周**。

现有代码中约 60% 的组件可直接复用（embedding、linear、rms_norm、swiglu、add、argmax、sample、基础设施层），重点投入在 DeltaNet 和 MoE 两个核心模块的开发上。
