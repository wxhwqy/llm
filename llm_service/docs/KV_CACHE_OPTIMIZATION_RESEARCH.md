# KV Cache 增强功能技术调研报告

**项目**: llm_service 推理框架  
**调研日期**: 2025年2月  
**版本**: v1.0

---

## 1. 概述

本报告针对 llm_service 推理框架中的 KV Cache 实现，调研业界主流的增强方案，分析其原理、收益与落地可行性，为后续技术选型与实现提供依据。

---

## 2. 当前实现分析

### 2.1 架构概览

llm_service 采用 C++ 核心 + Python 绑定的分层架构：

```
chat.py (Python)  →  llaisys.models.Qwen3  →  libllaisys (C API)  →  Qwen3Model / Qwen3ModelTP (C++)
```

### 2.2 现有 KV Cache 实现

| 维度 | 当前实现 | 位置 |
|------|----------|------|
| **存储结构** | 连续内存块 `[maxseq, nkvh, dh]` | `qwen3.cpp:94-95`, `qwen3_tp.cpp:97-98` |
| **分配策略** | 模型加载时预分配全量 | `initKVCache()` |
| **容量** | `max_seq_len`（当前限制为 8192） | `qwen3.py:51` |
| **精度** | BF16（与计算精度一致） | `config_.dtype` |
| **生命周期** | 每次 `generate()` 调用前 `resetCache()` | `qwen3.py:213`, `qwen3.cc:247` |
| **TP 支持** | 每 GPU 分片 `[maxseq, nkvh/tp, dh]` | `qwen3_tp.cpp:97-98` |

### 2.3 关键代码路径

```cpp
// 初始化：预分配 maxseq × nkvh × dh 的 K/V 张量
kv_cache_[i][0] = Tensor::create({maxseq, nkvh, dh}, config_.dtype, ...);
kv_cache_[i][1] = Tensor::create({maxseq, nkvh, dh}, config_.dtype, ...);

// 写入：按 start_pos 偏移写入新 token 的 K/V
std::byte *k_cache_ptr = kv_cache_[layer_idx][0]->data() + start_pos * nkvh * dh * elem_size;
api->memcpy_sync(k_cache_ptr, k_rope_view->data(), kv_bytes, LLAISYS_MEMCPY_D2D);

// 读取：slice 出 [0, total_len] 参与 attention
auto k_cache_view = kv_cache_[layer_idx][0]->slice(0, 0, total_len);
ops::self_attention(attn_view, q_rope_view, k_cache_view, v_cache_view, scale);
```

### 2.4 存在的问题

| 问题 | 影响 | 严重程度 |
|------|------|----------|
| **预分配全量** | 短序列场景下大量显存闲置（如 100 token 对话占用 8192 槽位） | 高 |
| **无跨请求复用** | 多轮对话每次重算历史，TTFT 无优化 | 高 |
| **无多请求共享** | 无法做 prefix caching，批量场景浪费 | 中 |
| **BF16 全精度** | KV 显存占用大，长序列易 OOM | 中 |
| **单请求设计** | 无法与 continuous batching 结合 | 中 |

---

## 3. 业界 KV Cache 增强技术

### 3.1 PagedAttention（分页注意力）

#### 3.1.1 原理

借鉴操作系统虚拟内存分页思想，将 KV Cache 划分为固定大小的「块」（block），按需分配，块可非连续存储。

- **传统方式**：为每个序列预分配 `[max_seq_len, nkvh, dh]` 的连续内存
- **PagedAttention**：将 KV 拆成多个 block（如 16 token/block），用 block 表记录物理位置

#### 3.1.2 收益

| 指标 | 典型提升 |
|------|----------|
| 显存利用率 | 碎片率从 60–80% 降至接近 0 |
| 吞吐量 | 相对 FasterTransformer/Orca 约 2–4× |
| 批大小 | 可支持更大 batch，利于 continuous batching |

#### 3.1.3 实现要点

1. **Block 管理**：维护空闲 block 池，按需分配/释放
2. **逻辑地址映射**：`(seq_id, logical_block_idx) → physical_block_ptr`
3. **Attention Kernel 改造**：从「连续 K/V 指针」改为「按 block 表间接访问」
4. **与 Flash Attention 集成**：vLLM 已实现 paged Flash Attention

#### 3.1.4 与本框架的适配

- **改动范围**：需修改 `initKVCache`、`forwardLayer` 中的 KV 读写逻辑，以及 `self_attention` 的调用方式
- **依赖**：若使用 paged Flash Attention，需引入或自研支持非连续 KV 的 kernel
- **复杂度**：高，涉及核心推理路径重构

---

### 3.2 KV Cache 量化

#### 3.2.1 原理

将 K/V 从 BF16/FP16 量化为 INT8 或 FP8，降低单元素存储与带宽占用。

| 精度 | 每元素字节 | 相对 BF16 压缩比 |
|------|------------|------------------|
| BF16 | 2 | 1× |
| FP8 | 1 | 2× |
| INT8 | 1 | 2× |

#### 3.2.2 量化方式

| 方式 | 说明 | 精度损失 |
|------|------|----------|
| Per-tensor | 整块 K/V 一个 scale | 较大 |
| Per-channel / Per-head | 每 head 或每通道独立 scale | 较小 |
| 校准 | 用少量数据估计 scale，减少误差 | 最小 |

#### 3.2.3 收益（参考数据）

- LLaMA-7B，batch=48：KV 从 63.5GB 降至 38.9GB（INT8 vs FP16）
- 典型场景下，准确率下降 < 2%

#### 3.2.4 实现要点

1. **量化**：在写入 KV cache 前对 K/V 做量化（scale + round）
2. **反量化**：在 attention 计算前或计算中反量化（或使用量化版 attention kernel）
3. **Scale 存储**：为每个量化块维护 scale，与 KV 一起存储

#### 3.2.5 与本框架的适配

- **改动范围**：在 `forwardLayer` 的 KV 写入路径增加量化，在 `self_attention` 前增加反量化或使用量化 kernel
- **依赖**：需有支持 INT8/FP8 的 attention 实现（当前 `self_attention_nvidia.cu` 为 BF16）
- **复杂度**：中高，需评估现有 CUDA kernel 的扩展性

---

### 3.3 Prefix Caching（前缀缓存）

#### 3.3.1 原理

当多个请求共享相同 prompt 前缀时，复用已计算的 KV Cache，只对新 token 做 prefill。

典型场景：
- 相同 system prompt 的多次对话
- 相同文档 + 不同问题的 RAG
- 多轮对话中历史轮次的 KV

#### 3.3.2 收益

- TTFT 可提升约 2×（当 >90% 前缀相同时）
- 减少重复 prefill 计算，降低延迟与算力消耗

#### 3.3.3 实现要点

1. **前缀匹配**：对 prompt 做 hash 或逐 token 比较，识别可复用前缀
2. **Cache 管理**：维护「前缀 → KV block 列表」的映射，支持 LRU 等淘汰策略
3. **Copy-on-Write**：多请求共享时，仅在写时复制，避免重复存储
4. **与 PagedAttention 结合**：共享的 prefix 以 block 为单位复用，天然兼容分页

#### 3.3.4 与本框架的适配

- **改动范围**：需在 Python 层或 C++ 层增加前缀匹配与 cache 查找逻辑；`generate` 需支持「从已有 KV 续写」而非每次 `reset`
- **依赖**：若采用分页 KV，prefix caching 与 PagedAttention 协同更自然
- **复杂度**：中，API 与调度逻辑改动较多

---

### 3.4 多轮对话 KV 复用（单请求场景）

#### 3.4.1 原理

在单次会话中，不调用 `resetCache()`，保留上一轮对话的 KV，仅对新一轮 user + assistant 输入做增量 prefill。

#### 3.4.2 收益

- 多轮对话时，每轮只需处理新增 token，TTFT 显著下降
- 实现难度低于跨请求 prefix caching

#### 3.4.3 实现要点

1. **API 设计**：`generate()` 支持 `keep_cache=True` 或显式的 `append_to_cache` 模式
2. **Python 层**：`chat.py` 中不再每次调用 `model.reset()`，改为累积 `input_ids`
3. **C++ 层**：`infer` 支持 `start_pos = cache_len_` 的增量写入，无需改动底层 KV 结构

#### 3.4.4 与本框架的适配

- **改动范围**：主要在 Python 绑定与 `chat.py`，C++ 层 `infer` 已支持 `start_pos`，改动较小
- **复杂度**：低，可作为优先落地的优化

---

## 4. 技术对比与选型建议

### 4.1 综合对比

| 技术 | 显存收益 | 延迟/吞吐收益 | 实现复杂度 | 依赖 |
|------|----------|---------------|------------|------|
| PagedAttention | 高 | 高 | 高 | 新 attention kernel |
| KV 量化 | 中高 | 中 | 中高 | 量化 attention |
| Prefix Caching | 中 | 高（TTFT） | 中 | 建议与 Paged 结合 |
| 多轮 KV 复用 | 无 | 高（多轮） | 低 | 无 |

### 4.2 推荐实施路径

```
Phase 1（短期，1–2 周）
├── 多轮对话 KV 复用
└── 评估：快速提升多轮对话体验，改动小

Phase 2（中期，1–2 月）
├── KV Cache 量化（FP8/INT8）
└── 评估：显存与带宽压力大时优先

Phase 3（长期，2–3 月）
├── PagedAttention
├── Prefix Caching（可依赖 Paged）
└── 评估：多请求、高并发场景必备
```

### 4.3 与现有架构的兼容性

| 组件 | 多轮复用 | KV 量化 | PagedAttention | Prefix Caching |
|------|----------|---------|----------------|----------------|
| `qwen3.cpp` infer | 小改 | 中改 | 大改 | 中改 |
| `self_attention` | 无 | 中改 | 大改 | 无 |
| `qwen3.cc` C API | 小改 | 小改 | 中改 | 中改 |
| `chat.py` | 中改 | 无 | 无 | 中改 |
| TP 支持 | 兼容 | 需验证 | 需验证 | 需验证 |

---

## 5. 参考实现与文献

### 5.1 开源项目

| 项目 | KV Cache 特性 | 参考价值 |
|------|---------------|----------|
| [vLLM](https://github.com/vllm-project/vllm) | PagedAttention、KV 量化、Prefix Caching | 架构与 kernel 设计 |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | Inflight batching、KV cache reuse | 工程实践 |
| [LMDeploy](https://github.com/InternLM/lmdeploy) | KV INT8 量化 | 量化实现 |
| [KVQuant](https://github.com/SqueezeAILab/KVQuant) | 非均匀量化、长上下文 | 量化算法 |

### 5.2 论文

- *Efficient Memory Management for Large Language Model Serving with PagedAttention* (SOSP 2023)
- *GPU-Accelerated INT8 Quantization for KV Cache Compression in Large Language Models* (arXiv 2026)

---

## 6. 风险与注意事项

1. **Attention Kernel**：当前 `self_attention_nvidia.cu` 使用 shared memory 存 scores，`kvlen` 大时有溢出风险（见 `CUDA_CODE_ANALYSIS.md`），长序列需考虑 Flash Attention 或分块
2. **TP 一致性**：多 GPU 下 KV 分片、block 分配需保证逻辑一致，避免死锁或数据错位
3. **精度回归**：KV 量化需做充分测试，关注长序列与敏感任务
4. **API 兼容**：新增 `keep_cache`、`prefix_cache` 等参数时，需保持向后兼容

---

## 7. 结论

llm_service 当前采用连续预分配、BF16、每次 reset 的 KV Cache 方案，在短序列、单请求、多轮对话等场景存在优化空间。建议按「多轮复用 → KV 量化 → PagedAttention + Prefix Caching」的顺序分阶段推进，在控制改动范围的前提下逐步提升显存利用率与推理性能。

---

*报告完成。如有疑问或需补充某技术细节，可进一步展开调研。*
