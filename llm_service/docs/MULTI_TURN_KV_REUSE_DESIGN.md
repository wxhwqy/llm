# 多轮对话 KV Cache 复用技术方案

## 1. 背景与目标

### 1.1 现状问题

当前推理引擎在每次 `stream_generate()` 调用时都会执行 `self.reset()`，将 `cache_len_` 清零并从头 prefill 整个 prompt。对于多轮对话场景，这意味着：

```
Turn 1: [system, user1]                    → prefill 200 tokens, decode 100 tokens
Turn 2: [system, user1, asst1, user2]      → prefill 400 tokens, decode 80 tokens  ← 200 tokens 重复计算
Turn 3: [system, user1, asst1, user2, asst2, user3] → prefill 680 tokens, decode 50 tokens  ← 480 tokens 重复计算
```

随着对话轮数增加，重复 prefill 的比例越来越大，TTFT（Time To First Token）线性增长。

### 1.2 优化目标

- **Turn 2+ 的 TTFT 降低 50-80%**：只 prefill 新增 token，跳过已缓存部分
- **零精度损失**：复用的 KV 与重新计算的结果 bit-exact 一致
- **向后兼容**：现有 API 行为不变，新功能通过可选参数启用
- **TP 兼容**：多卡 Tensor Parallel 模式下同样适用

### 1.3 适用范围

本方案针对 **同一会话内的多轮对话 KV 复用**（单 request 串行场景），不涉及跨 session 的 Prefix Caching 或 Continuous Batching。这是优化路线图中复杂度最低、收益最高的第一步。

---

## 2. 现有架构分析

### 2.1 KV Cache 数据结构

```cpp
// qwen3.hpp
std::vector<std::vector<tensor_t>> kv_cache_;  // [num_layers][2] → [max_seq_len, num_kv_heads, head_dim]
size_t cache_len_;                              // 当前已缓存的 token 数量
```

- 预分配 `[max_seq_len, num_kv_heads, head_dim]` 的连续显存（BF16）
- `cache_len_` 记录已写入的 token 位置，新 token 从 `start_pos = cache_len_` 处写入
- KV Cache 的写入和读取逻辑已经天然支持增量追加

### 2.2 当前推理流程

```
Python stream_generate()
  → self.reset()                    # cache_len_ = 0
  → _infer_one(all_tokens, N)       # prefill: start_pos=0, 写入 [0:N], cache_len_=N
  → _infer_one(new_token, 1)        # decode:  start_pos=N, 写入 [N:N+1], cache_len_=N+1
  → ...
```

### 2.3 请求数据流

```
Web Backend
  → 查询 DB 获取历史 messages
  → buildPrompt(): [system, history..., user_new]
  → POST /v1/chat/completions { messages: [...] }

LLM Service
  → tokenizer.apply_chat_template(messages) → text
  → tokenizer.encode(text) → input_ids
  → model.stream_generate(input_ids)    # 每次都从头 prefill
```

**关键发现**：C++ 层的 `infer()` 函数已经支持 `start_pos = cache_len_` 的增量写入，KV Cache 复用的核心障碍在 Python 层每次调用 `reset()`。

---

## 3. 技术方案设计

### 3.1 核心思路

利用 C++ 层已有的增量 KV Cache 机制，在 Python 层实现 **token 级别的前缀匹配**：
1. 保留上一轮的 `input_ids` 和对应的 `cache_len_`
2. 新一轮到来时，逐 token 比对新旧 `input_ids`，找到最长公共前缀 `prefix_len`
3. 如果 KV Cache 有效且 `prefix_len > 0`，跳过 `reset()`，只 prefill `input_ids[prefix_len:]`

### 3.2 前缀匹配策略

```python
def _compute_prefix_match(self, old_ids: list[int], new_ids: list[int], old_cache_len: int) -> int:
    """计算新旧 token 序列的最长公共前缀长度。

    返回值 prefix_len 满足：
    - old_ids[:prefix_len] == new_ids[:prefix_len]
    - prefix_len <= old_cache_len（不能超过实际已缓存的长度）
    """
    max_match = min(len(old_ids), len(new_ids), old_cache_len)
    prefix_len = 0
    for i in range(max_match):
        if old_ids[i] != new_ids[i]:
            break
        prefix_len = i + 1
    return prefix_len
```

**为什么用 token 级别比对而不是 message 级别**：
- `apply_chat_template` 会为每个 message 添加特殊 token（`<|im_start|>`, `<|im_end|>` 等）
- message 内容不变 ≠ token 序列不变（模板可能变化）
- token 级别比对是最精确、最安全的方式

### 3.3 缓存有效性判断

KV Cache 复用需要严格验证以下条件：

| 条件 | 说明 |
|------|------|
| 同一模型实例 | model 对象未被重新创建 |
| 未被其他请求使用 | 单并发场景下天然满足 |
| prefix_len > 阈值 | 前缀太短时复用收益不如直接 prefill（省去比对开销） |
| 未超过 max_seq_len | `prefix_len + new_tokens + max_new_tokens ≤ max_seq_len` |

### 3.4 修改架构图

```
                        ┌──────────────────────────┐
                        │   InferenceService       │
                        │                          │
  ChatCompletionReq ──► │  _tokenize(messages)     │
                        │       │                  │
                        │       ▼                  │
                        │  new_input_ids           │
                        │       │                  │
                        │  ┌────▼────────────┐     │
                        │  │ CacheState      │     │  ◄── NEW
                        │  │  .prev_ids      │     │
                        │  │  .cache_len     │     │
                        │  └────┬────────────┘     │
                        │       │                  │
                        │  prefix_match()          │  ◄── NEW
                        │       │                  │
                        │       ▼                  │
                        │  prefix_len > 0 ?        │
                        │   Y: skip reset,         │  ◄── CHANGED
                        │      prefill [prefix_len:]│
                        │   N: reset + full prefill │
                        │       │                  │
                        │       ▼                  │
                        │  stream_generate()       │
                        └──────────────────────────┘
```

---

## 4. 分层修改方案

### 4.1 C++ 层（变更最小）

C++ 的 `infer()` 已经通过 `cache_len_` 支持增量写入，**核心逻辑无需修改**。只需新增一个 C API 来支持外部设置 `cache_len_`：

```cpp
// src/llaisys/qwen3.cc — 新增
extern "C" void llaisysQwen3ModelSetCacheLen(LlaisysQwen3Model *handle, size_t cache_len) {
    auto *model = reinterpret_cast<Qwen3Model *>(handle);
    model->setCacheLen(cache_len);
}
```

```cpp
// src/models/qwen3.hpp — 新增
void setCacheLen(size_t len) {
    ASSERT(len <= config_.max_seq_len, "cache_len exceeds max_seq_len");
    cache_len_ = len;
}

size_t cacheLen() const { return cache_len_; }
```

```cpp
// src/models/qwen3_tp.hpp — TP 版本同样新增
void setCacheLen(size_t len) {
    ASSERT(len <= config_.max_seq_len, "cache_len exceeds max_seq_len");
    cache_len_ = len;
}
```

**为什么不直接在 C++ 层做前缀匹配**：
- C++ 层不持有 tokenizer，无法获知 token 序列
- 前缀匹配逻辑放在 Python 层更灵活，便于策略调整
- C++ 层只负责"被告知 cache_len 是多少"即可

### 4.2 C API / Python Bindings

```c
// include/llaisys/models/qwen3.h — 新增声明
void llaisysQwen3ModelSetCacheLen(struct LlaisysQwen3Model *model, size_t cache_len);
```

```python
# python/llaisys/libllaisys/models.py — 注册 ctypes 原型
LIB_LLAISYS.llaisysQwen3ModelSetCacheLen.argtypes = [c_void_p, c_size_t]
LIB_LLAISYS.llaisysQwen3ModelSetCacheLen.restype = None
```

### 4.3 Python Model 层

```python
# python/llaisys/models/qwen3.py

class Qwen3:
    def __init__(self, ...):
        ...
        # KV Cache 复用状态
        self._prev_input_ids: list[int] | None = None
        self._prev_cache_len: int = 0

    def reset(self):
        """完全重置 KV Cache 状态。"""
        LIB_LLAISYS.llaisysQwen3ModelReset(self._model)
        self._prev_input_ids = None
        self._prev_cache_len = 0

    def _compute_prefix_match(self, new_ids: list[int]) -> int:
        """计算可复用的 KV Cache 前缀长度。"""
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

    def stream_generate(self, inputs: Sequence[int], max_new_tokens: int = None,
                        top_k: int = 1, top_p: float = 0.8, temperature: float = 0.8,
                        reuse_cache: bool = False):
        """流式生成，支持 KV Cache 复用。

        Args:
            reuse_cache: 为 True 时尝试复用上一轮的 KV Cache，
                        仅 prefill 新增 token。默认 False 保持向后兼容。
        """
        import random
        tokens = list(inputs)

        if reuse_cache:
            prefix_len = self._compute_prefix_match(tokens)
        else:
            prefix_len = 0

        if prefix_len > 0:
            # 复用模式：回退 cache_len 到 prefix_len，只 prefill 新增部分
            LIB_LLAISYS.llaisysQwen3ModelSetCacheLen(self._model, prefix_len)
            effective_input = tokens[prefix_len:]
        else:
            # 非复用模式：完全重置
            LIB_LLAISYS.llaisysQwen3ModelReset(self._model)
            effective_input = tokens

        if max_new_tokens is None:
            max_new_tokens = 128
        max_new_tokens = min(max_new_tokens, self.max_seq_len - len(tokens))
        if max_new_tokens <= 0:
            return

        seed_base = random.getrandbits(64)

        # Prefill
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

        # 更新缓存状态：记录完整的 token 序列（prompt + generated）
        self._prev_input_ids = tokens + generated
        self._prev_cache_len = len(tokens) + len(generated)
```

**关键设计决策**：

1. **`_prev_input_ids` 包含生成的 token**：下一轮的 prompt 会包含 `assistant` 的回复，这些 token 也已经在 KV Cache 中，可以被复用。

2. **`prefix_len` 的回退语义**：当 `prefix_len < _prev_cache_len` 时，说明旧缓存的后半部分已失效（例如用户编辑了历史消息），此时设置 `cache_len = prefix_len`，覆盖写入新的 KV。

3. **向后兼容**：`reuse_cache=False`（默认值）时行为与现有完全一致。

### 4.4 FastAPI 服务层

```python
# api/services/inference.py

class InferenceService:
    def __init__(self):
        ...
        # 每个 session 的 cache 状态
        self._session_cache: dict[str, list[int]] = {}

    def _tokenize(self, loaded: LoadedModel, request: ChatCompletionRequest) -> tuple[list[int], int]:
        """tokenize 并返回 input_ids。"""
        messages = [m.model_dump() for m in request.messages]
        if request.no_think:
            ...
        text = loaded.tokenizer.apply_chat_template(
            conversation=messages, add_generation_prompt=True, tokenize=False
        )
        ids = loaded.tokenizer.encode(text)
        return ids, len(ids)

    async def chat_completion_stream(self, request: ChatCompletionRequest, request_id: str):
        loaded = await self._model_manager.get_model(request.model)
        input_ids, prompt_tokens = self._tokenize(loaded, request)

        # 判断是否启用 cache 复用
        session_id = getattr(request, 'session_id', None)
        reuse_cache = session_id is not None

        async for event in self._generate(
            loaded, input_ids, prompt_tokens, request, request_id,
            reuse_cache=reuse_cache,
        ):
            yield event
```

### 4.5 API Schema 扩展

```python
# api/schemas/chat.py — ChatCompletionRequest 新增可选字段

class ChatCompletionRequest(BaseModel):
    ...
    session_id: str | None = None  # 传入 session_id 时启用 KV Cache 复用
```

**设计考量**：
- `session_id` 作为 cache 复用的开关和索引 key
- 不传 `session_id` 时行为完全不变（向后兼容）
- Web Backend 在多轮对话时传入 `session.id` 即可启用

### 4.6 Web Backend 适配

```typescript
// web/src/server/services/llm-client.service.ts — 请求体新增 session_id

async function* streamChatCompletion(
  messages: ChatMessage[],
  modelId: string,
  sessionId?: string,  // 新增
) {
  const body = {
    model: modelId,
    messages,
    stream: true,
    session_id: sessionId,  // 新增
  };
  // ... POST /v1/chat/completions
}
```

---

## 5. 边界场景处理

### 5.1 缓存失效场景

| 场景 | 处理方式 |
|------|----------|
| 用户编辑了历史消息 | token 比对会发现 prefix 截断，从分叉点开始重新 prefill |
| 用户删除了消息 | 同上，prefix 会变短 |
| 切换了模型 | `model_manager` 加载新模型时 `_prev_input_ids` 被清空 |
| 修改了 system prompt | 第一个 token 就不匹配，退化为全量 prefill |
| 修改了采样参数 | 不影响 KV Cache（KV 只依赖 token 序列，与采样无关） |
| 服务重启 | 内存状态丢失，自动退化为全量 prefill |
| context window 裁剪了旧消息 | token 比对会在裁剪点发现不匹配，从该点重新 prefill |

### 5.2 并发安全

当前架构是 `max_concurrent=1` 的单请求串行模式，不存在并发问题。未来引入 Continuous Batching 时需要为每个 request 维护独立的 KV Cache slot，但那属于 Phase 3 的范畴。

### 5.3 显存溢出保护

```python
# stream_generate() 中的安全检查
total_needed = len(tokens) + max_new_tokens
if total_needed > self.max_seq_len:
    # 无法复用，必须裁剪
    # 由上层 context_manager 保证 prompt 不超限
    raise ValueError(f"Total sequence length {total_needed} exceeds max_seq_len {self.max_seq_len}")
```

---

## 6. 性能分析

### 6.1 理论收益

假设 Qwen3-32B FP8, TP=2, 双卡 RTX 4090：

| 场景 | 全量 Prefill | 复用后 Prefill | TTFT 降低 |
|------|-------------|---------------|-----------|
| Turn 2 (200 旧 + 50 新) | ~250 tokens × 2ms/token = 500ms | ~50 tokens × 2ms/token = 100ms | **80%** |
| Turn 5 (800 旧 + 50 新) | ~850 tokens × 2ms/token = 1700ms | ~50 tokens × 2ms/token = 100ms | **94%** |
| Turn 10 (2000 旧 + 50 新) | ~2050 tokens × 2ms/token = 4100ms | ~50 tokens × 2ms/token = 100ms | **98%** |

> 注：2ms/token 是 prefill 的粗略估计，实际取决于 batch size 和硬件，但比例关系成立。

### 6.2 额外开销

| 开销项 | 量级 | 说明 |
|--------|------|------|
| token 比对 | < 0.1ms | Python 循环对比，即使 8K tokens 也很快 |
| `_prev_input_ids` 内存 | < 100KB | 存储 token ID 列表 (int64 × max_seq_len) |
| `SetCacheLen` 调用 | < 0.01ms | 仅赋值一个 size_t |

开销可忽略不计。

### 6.3 Decode 速度无影响

KV Cache 复用只影响 prefill 阶段。Decode 阶段的行为完全不变（每次读取完整 cache 做 attention），因此 tokens/s 不受影响。

---

## 7. 测试计划

### 7.1 正确性验证

1. **Bit-exact 验证**：
   - 同一组 messages，分别用 `reuse_cache=False` 和 `reuse_cache=True` 生成
   - 在 `temperature=0, top_k=1` 下两者输出必须完全一致（deterministic sampling）
   - 验证所有层的 KV Cache 内容一致

2. **前缀匹配边界测试**：
   - 空前缀（首轮对话）→ 全量 prefill
   - 完全匹配（重复同一请求）→ 跳过全部 prefill，只做 decode
   - 部分匹配（新增 user 消息）→ 只 prefill 新增部分
   - 前缀为 0（编辑了第一条消息）→ 全量 prefill
   - `_prev_cache_len > new_ids_len`（新 prompt 比旧的短）→ 正确回退

3. **TP 模式验证**：
   - 所有 GPU 上的 `cache_len` 同步更新
   - 各 GPU 上 KV Cache 分片内容与非复用模式一致

### 7.2 性能基准

```python
# 测试脚本伪代码
model = Qwen3(...)
tokenizer = AutoTokenizer(...)

messages_history = []
for turn in range(10):
    messages_history.append({"role": "user", "content": f"Question {turn}"})

    input_ids = tokenizer.apply_chat_template(messages_history, ...)

    t0 = time.perf_counter()
    for token in model.stream_generate(input_ids, reuse_cache=True):
        t1 = time.perf_counter()  # TTFT
        break

    print(f"Turn {turn}: TTFT = {(t1-t0)*1000:.1f}ms, prompt_len={len(input_ids)}")

    messages_history.append({"role": "assistant", "content": "..."})
```

预期结果：Turn 2+ 的 TTFT 应显著低于 Turn 1。

### 7.3 API 兼容性测试

- 不传 `session_id` → 行为与现有完全一致
- 传入 `session_id` → 首次请求全量 prefill，后续请求增量 prefill
- 不同 `session_id` → 各自独立的 cache 状态

---

## 8. 实施步骤

### Phase 1: C++ 层（0.5 天）

1. `qwen3.hpp` / `qwen3_tp.hpp`：添加 `setCacheLen()` / `cacheLen()`
2. `qwen3.cc`：添加 `llaisysQwen3ModelSetCacheLen` C API
3. `qwen3.h`：添加声明
4. 编译验证

### Phase 2: Python Bindings（0.5 天）

1. `libllaisys/models.py`：注册 ctypes 原型
2. `models/qwen3.py`：实现 `_compute_prefix_match()`, 修改 `stream_generate()`
3. `chat.py`：修改交互式 chat 循环支持 `reuse_cache=True`
4. 单元测试

### Phase 3: FastAPI 服务层（0.5 天）

1. `schemas/chat.py`：添加 `session_id` 字段
2. `services/inference.py`：传递 `reuse_cache` 参数
3. API 测试

### Phase 4: Web Backend 适配（0.5 天）

1. `llm-client.service.ts`：请求体添加 `session_id`
2. `chat.service.ts`：传入 `session.id`
3. 端到端测试

### Phase 5: 性能验证与上线（0.5 天）

1. 多轮对话 TTFT 基准测试
2. 正确性回归测试
3. 长对话压力测试（接近 max_seq_len）

**预估总工期：2-3 天**

---

## 9. 后续演进

本方案完成后，为后续优化奠定基础：

| 后续优化 | 与本方案的关系 |
|----------|----------------|
| **Prefix Caching（跨 session）** | 将 `_prev_input_ids` 从 per-model 升级为全局 hash-indexed cache pool |
| **KV Cache 量化** | 正交优化，可独立实施。量化后的 KV 同样支持前缀匹配和复用 |
| **PagedAttention** | 需要重构 KV 存储结构，但前缀匹配的逻辑可复用 |
| **Continuous Batching** | 需要 per-request 的 KV Cache 管理，本方案中的 session 概念可扩展为 request slot |

---

## 10. 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| Token 比对错误导致 KV 不一致 | 低 | 高（输出错误） | 严格的 bit-exact 测试；生产环境可配置 fallback 开关 |
| `apply_chat_template` 非确定性 | 极低 | 高 | 使用固定 tokenizer 版本；比对的是 token ID 不是 text |
| 长期运行内存泄漏 | 低 | 中 | `_prev_input_ids` 有界（≤ max_seq_len）；session 结束时清理 |
| TP 模式下 cache_len 不同步 | 低 | 高 | `setCacheLen` 在所有 GPU 上同步调用（由 Python 层保证） |
