# 大模型推理服务 — 技术方案文档

> 版本：0.2 | 最后更新：2026-03-14
>
> 对应 PRD 版本：0.1-draft

---

## 1. 概述

### 1.1 文档范围

本文档覆盖 PRD 中**大模型服务 (llm_service/)** 层的技术实现方案。主要职责包括：

- 在自研 llaisys CUDA 推理引擎之上构建 OpenAI 兼容的 HTTP API 层
- 模型生命周期管理（加载、多模型注册、健康检查）
- 推理请求调度（请求队列、并发控制、取消）
- SSE 流式 Token 输出
- Token 用量统计与回传
- 后续引擎优化（KV-Cache 动态分配、Prefix Cache、Continuous Batching）的设计蓝图

### 1.2 系统定位

```
┌──────────────┐    HTTP/SSE    ┌──────────────┐   HTTP (OpenAI API)  ┌──────────────────────┐
│   Frontend   │ ◄────────────► │   Backend    │ ──────────────────►  │    LLM Service       │
│   (Next.js)  │                │ (Next.js API)│                      │    (本文档)           │
└──────────────┘                └──────────────┘                      │                      │
                                                                      │  ┌────────────────┐  │
                                                                      │  │  FastAPI 服务层  │  │
                                                                      │  └───────┬────────┘  │
                                                                      │          │           │
                                                                      │  ┌───────▼────────┐  │
                                                                      │  │ llaisys Python  │  │
                                                                      │  │   Bindings      │  │
                                                                      │  └───────┬────────┘  │
                                                                      │          │ ctypes    │
                                                                      │  ┌───────▼────────┐  │
                                                                      │  │ libllaisys.so   │  │
                                                                      │  │ (CUDA/C++)      │  │
                                                                      │  └────────────────┘  │
                                                                      └──────────────────────┘
```

### 1.3 已有能力盘点

基于对 llaisys 代码库的分析，当前引擎已具备以下能力：

| 能力 | 实现位置 | 说明 |
|------|---------|------|
| Qwen3 前向推理 | `src/models/qwen3.cpp` | Prefill + Decode 完整前向传播 |
| FP8 量化推理 | `src/ops/dequantize_fp8/` | Block-wise FP8 反量化，支持 `fp8_block_h × fp8_block_w` |
| Tensor Parallelism | `src/models/qwen3_tp.cpp` | 多卡 NCCL All-Reduce，Column/Row Parallel 切分 |
| Top-K / Top-P 采样 | `src/ops/sample/` | GPU 上的采样算子，支持 temperature 缩放 |
| KV-Cache 管理 | C++ 模型内部 | 预分配固定大小 (max_seq_len=8192)，`reset()` 全量清零 |
| 流式 Token 生成 | `Qwen3.stream_generate()` | Python Generator，逐 token yield |
| safetensors 加载 | `Qwen3._load_weights()` | 支持 TP 切分加载 |
| CLI 聊天 | `chat.py` | 基于 HuggingFace tokenizer 的交互式 CLI |

**C API 导出函数**（`include/llaisys/models/qwen3.h`）：

| 函数 | 说明 |
|------|------|
| `llaisysQwen3ModelCreate` | 创建模型实例，指定设备和 TP 配置 |
| `llaisysQwen3ModelDestroy` | 销毁模型实例 |
| `llaisysQwen3ModelWeights` | 获取权重指针（单卡） |
| `llaisysQwen3ModelTPWeights` | 获取 TP 权重指针（指定设备） |
| `llaisysQwen3ModelTPSize` | 获取实际 TP 大小 |
| `llaisysQwen3ModelInfer` | 贪婪推理（argmax） |
| `llaisysQwen3ModelInferSampled` | 带采样参数的推理 |
| `llaisysQwen3ModelReset` | 重置 KV-Cache（cache_len 归零） |
| `llaisysQwen3ModelSetCacheLen` | 设置 cache_len 到指定位置（用于前缀复用回滚） |
| `llaisysQwen3ModelGetCacheLen` | 获取当前 cache_len |

**当前限制**：

| 限制 | 说明 | 影响 |
|------|------|------|
| 单请求串行推理 | `stream_generate` 内部有状态（KV-Cache 位置指针），不支持并发 | Phase 1 仅支持单并发 |
| KV-Cache 固定预分配 | `max_seq_len` 写死 8192，全量预分配 | 显存浪费，无法动态扩缩 |
| 无 Batch 推理 | `Infer` 接口一次处理一个请求 | 吞吐受限 |
| 无跨会话 Prefix Cache | 不同会话间无法共享 System Prompt 的 KV-Cache | 相同 System Prompt 仍需各自 Prefill |
| ~~`reset()` 全量清零~~ | ~~会话间无法保留部分 KV-Cache~~ | **已解决**：`setCacheLen()` 支持回滚到任意前缀位置，同会话多轮对话可复用前缀 KV |

---

## 2. 技术选型

| 组件 | 选型 | 版本 | 理由 |
|------|------|------|------|
| HTTP 框架 | FastAPI | 0.115+ | 原生 async、SSE 支持、自动 OpenAPI 文档 |
| ASGI 服务器 | uvicorn | 0.34+ | 高性能 ASGI，配合 uvloop |
| Tokenizer | transformers (AutoTokenizer) | 4.48+ | 与 Qwen3 官方 tokenizer 完全兼容 |
| 数据校验 | Pydantic v2 | 2.x | FastAPI 内置，类型安全 |
| SSE 支持 | sse-starlette | 2.x | 标准 SSE 响应封装 |
| 进程管理 | 内置 asyncio | - | 配合线程池实现异步推理 |
| 配置管理 | pydantic-settings | 2.x | 从环境变量/配置文件加载 |
| 日志 | structlog | 24.x | 结构化日志，便于监控 |
| 指标 | prometheus-fastapi-instrumentator | - | Prometheus 指标暴露 |

### 2.1 依赖清单

```
# requirements.txt
fastapi>=0.115.0
uvicorn[standard]>=0.34.0
sse-starlette>=2.0.0
pydantic>=2.0
pydantic-settings>=2.0
transformers>=4.48.0
safetensors>=0.4.0
torch>=2.0.0
numpy>=1.26.0
structlog>=24.0.0
prometheus-fastapi-instrumentator>=7.0.0
```

---

## 3. 目录结构

```
llm_service/
├── api/                            # 新增：FastAPI HTTP 服务层
│   ├── main.py                     # FastAPI 应用入口 + lifespan
│   ├── config.py                   # 配置管理（pydantic-settings）
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── chat.py                 # POST /v1/chat/completions
│   │   ├── models.py              # GET /v1/models
│   │   └── health.py              # GET /health
│   ├── services/
│   │   ├── __init__.py
│   │   ├── model_manager.py       # 模型生命周期管理
│   │   ├── inference.py           # 推理调度（线程池 + 流式桥接）
│   │   ├── queue.py               # 请求队列（并发控制）
│   │   └── tokenizer_pool.py     # Tokenizer 管理
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── chat.py                # ChatCompletion 请求/响应 schema
│   │   ├── models.py             # Model 列表 schema
│   │   └── common.py             # 通用 schema（错误、健康检查）
│   └── middleware/
│       ├── __init__.py
│       ├── logging.py             # 请求日志中间件
│       └── error_handler.py       # 全局异常处理
│
├── python/llaisys/                 # 已有：Python 绑定
├── src/                            # 已有：C++/CUDA 源码
├── include/                        # 已有：C API 头文件
├── models/                         # 已有：模型权重目录
├── chat.py                         # 已有：CLI 聊天脚本
└── xmake.lua                       # 已有：构建配置
```

---

## 4. 核心架构

### 4.1 请求处理流水线

```
HTTP 请求 (POST /v1/chat/completions)
    │
    ▼
┌─────────────────────────────┐
│  [1] 请求解析与校验           │   FastAPI + Pydantic
│  - 解析 JSON Body            │
│  - 校验参数范围               │
│  - 解析 model ID             │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  [2] 请求排队                 │   QueueService
│  - 检查并发数                 │
│  - 入队等待 / 立即执行        │
│  - 队列满 → 503              │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  [3] Tokenizer 编码          │   TokenizerPool
│  - apply_chat_template       │
│  - encode → input_ids        │
│  - 计算 prompt_tokens        │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  [4] 推理执行                 │   InferenceService
│  - 在线程池中调用             │   (ThreadPoolExecutor)
│    model.stream_generate()   │
│  - 逐 token 产出             │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  [5] SSE 流式响应             │   StreamingResponse
│  - 封装为 OpenAI 格式 chunk   │
│  - 最后一个 chunk 附带 usage  │
│  - 发送 [DONE]               │
└─────────────────────────────┘
```

### 4.2 线程模型

```
┌─ uvicorn event loop (主线程) ─────────────────────────────────┐
│                                                                │
│   async def chat_completions():                                │
│       queue.acquire()                  # 异步信号量控制并发      │
│       input_ids = tokenize(messages)   # CPU 操作，快速         │
│       async for token in infer(ids):   # 从线程池桥接           │
│           yield sse_chunk(token)       # 异步 yield SSE        │
│       queue.release()                  # 释放并发槽位           │
│                                                                │
├─ ThreadPoolExecutor (推理线程) ────────────────────────────────┤
│                                                                │
│   def _run_inference(model, ids):                              │
│       for token_id in model.stream_generate(ids):              │
│           put_to_async_queue(token_id)  # 线程安全队列          │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

**设计决策**：llaisys 的推理调用（`stream_generate`）是同步阻塞的（涉及 CUDA kernel launch 和同步），不能直接在 asyncio event loop 中执行。通过 `ThreadPoolExecutor` 将推理逻辑调度到独立线程，再通过 `asyncio.Queue` 桥接回 async 上下文，实现非阻塞的流式输出。

---

## 5. 模块详细设计

### 5.1 配置管理

```python
# api/config.py

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # 服务配置
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1  # GPU 服务只能单进程

    # 模型配置
    model_configs: list[ModelConfig] = []
    default_model: str = "qwen3-32b"

    # 推理默认参数
    default_temperature: float = 0.6
    default_top_k: int = 20
    default_top_p: float = 0.95
    default_max_tokens: int = 2048

    # 队列配置
    max_concurrent: int = 1        # 最大并发推理数
    max_queue_size: int = 16       # 最大排队数
    queue_timeout_seconds: float = 120.0

    # 引擎配置
    device: str = "nvidia"         # nvidia | cpu
    device_ids: list[int] = [0, 1] # GPU 设备 ID 列表
    tp_size: int = 2               # Tensor Parallelism 度

    class Config:
        env_prefix = "LLM_"
        env_file = ".env"


class ModelConfig(BaseSettings):
    """单个模型的配置"""
    id: str                         # 对外暴露的模型 ID，如 "qwen3-8b"
    name: str                       # 人类可读名称
    path: str                       # 模型权重目录路径
    max_seq_len: int = 8192         # 最大序列长度
    device: str = "nvidia"
    device_ids: list[int] = [0, 1]
    tp_size: int = 2
```

**模型注册**通过配置文件 `models.json` 管理，支持动态添加新模型：

```json
[
  {
    "id": "qwen3-32b",
    "name": "Qwen3 32B (FP8)",
    "path": "models/qwen3_32b_fp8",
    "max_seq_len": 8192,
    "device": "nvidia",
    "device_ids": [0, 1],
    "tp_size": 2
  },
  {
    "id": "qwen3-8b",
    "name": "Qwen3 8B (FP8)",
    "path": "models/qwen3_8b_fp8",
    "max_seq_len": 8192,
    "device": "nvidia",
    "device_ids": [0],
    "tp_size": 1
  }
]
```

### 5.2 模型生命周期管理

```python
# api/services/model_manager.py

class ModelManager:
    """
    管理所有已加载模型的生命周期。

    由于 GPU 显存有限，通常同一时刻只加载一个模型。
    多模型场景下采用按需加载 + LRU 卸载策略。
    """

    def __init__(self, configs: list[ModelConfig]):
        self._configs: dict[str, ModelConfig] = {c.id: c for c in configs}
        self._loaded: dict[str, LoadedModel] = {}
        self._lock = asyncio.Lock()

    async def startup(self, default_model: str):
        """服务启动时预加载默认模型"""
        await self.get_model(default_model)

    async def get_model(self, model_id: str) -> LoadedModel:
        """获取已加载的模型，如未加载则触发加载"""
        async with self._lock:
            if model_id in self._loaded:
                return self._loaded[model_id]
            if model_id not in self._configs:
                raise ModelNotFoundError(model_id)
            loaded = await self._load_model(model_id)
            self._loaded[model_id] = loaded
            return loaded

    async def _load_model(self, model_id: str) -> LoadedModel:
        """在线程池中加载模型（加载过程涉及大量 I/O 和 GPU 内存分配）"""
        config = self._configs[model_id]
        loop = asyncio.get_event_loop()
        model, tokenizer = await loop.run_in_executor(
            None, self._load_model_sync, config
        )
        return LoadedModel(
            config=config,
            model=model,
            tokenizer=tokenizer,
        )

    def _load_model_sync(self, config: ModelConfig):
        """同步加载模型和 tokenizer"""
        from llaisys.libllaisys import DeviceType
        device = DeviceType.NVIDIA if config.device == "nvidia" else DeviceType.CPU
        device_ids = config.device_ids if config.tp_size > 1 else config.device_ids[0]

        model = llaisys.models.Qwen3(config.path, device, device_ids)
        tokenizer = AutoTokenizer.from_pretrained(config.path, local_files_only=True)
        return model, tokenizer

    def list_models(self) -> list[ModelInfo]:
        """返回所有已注册模型的信息"""
        return [
            ModelInfo(
                id=cfg.id,
                name=cfg.name,
                max_context_length=cfg.max_seq_len,
                status="loaded" if cfg.id in self._loaded else "available",
            )
            for cfg in self._configs.values()
        ]

    async def shutdown(self):
        """释放所有模型资源"""
        for loaded in self._loaded.values():
            del loaded.model
        self._loaded.clear()


@dataclass
class LoadedModel:
    config: ModelConfig
    model: Any            # llaisys.models.Qwen3 实例
    tokenizer: Any        # transformers.AutoTokenizer 实例
```

**Phase 1 简化**：只加载一个默认模型（`qwen3-32b`），不实现动态卸载。

### 5.3 请求队列

```python
# api/services/queue.py

class InferenceQueue:
    """
    基于 asyncio.Semaphore 的推理并发控制。

    控制同时执行的推理请求数，超出的请求排队等待。
    Phase 1: max_concurrent=1（单请求串行，受限于引擎 KV-Cache 设计）
    Phase 5: 配合 Continuous Batching 提升并发数
    """

    def __init__(self, max_concurrent: int, max_queue_size: int, timeout: float):
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._max_queue_size = max_queue_size
        self._timeout = timeout
        self._waiting = 0     # 当前排队等待数
        self._active = 0      # 当前活跃推理数

    @asynccontextmanager
    async def acquire(self):
        """
        获取推理槽位。

        如果所有槽位都被占用，请求进入等待状态。
        等待数超过 max_queue_size 时直接拒绝。
        等待超过 timeout 时超时失败。
        """
        if self._waiting >= self._max_queue_size:
            raise QueueFullError(
                queue_size=self._waiting,
                message="推理服务繁忙，请稍后重试"
            )

        self._waiting += 1
        try:
            acquired = await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self._timeout
            )
            self._waiting -= 1
            self._active += 1
            try:
                yield
            finally:
                self._active -= 1
                self._semaphore.release()
        except asyncio.TimeoutError:
            self._waiting -= 1
            raise QueueTimeoutError(
                wait_time=self._timeout,
                message=f"排队等待超时 ({self._timeout}s)"
            )

    @property
    def status(self) -> QueueStatus:
        return QueueStatus(
            active=self._active,
            waiting=self._waiting,
            max_concurrent=self._semaphore._value + self._active,
        )
```

### 5.4 推理服务

推理服务是最核心的模块，负责将同步的 `stream_generate` 调用桥接到 async SSE 流。

```python
# api/services/inference.py

class InferenceService:
    """
    推理调度服务。

    将 llaisys 的同步 stream_generate 桥接到 async 流式输出。
    通过 ThreadPoolExecutor 避免阻塞 event loop。
    """

    def __init__(self, model_manager: ModelManager, queue: InferenceQueue):
        self._model_manager = model_manager
        self._queue = queue
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._active_requests: dict[str, InferenceRequest] = {}

    async def stream_chat_completion(
        self,
        request: ChatCompletionRequest,
        request_id: str,
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """
        流式聊天补全。

        1. 获取模型和 tokenizer
        2. 编码输入
        3. 在推理队列中等待槽位
        4. 在线程池中执行推理
        5. 通过 asyncio.Queue 桥接流式输出
        """
        loaded = await self._model_manager.get_model(request.model)
        model = loaded.model
        tokenizer = loaded.tokenizer

        # Tokenize
        messages_dicts = [m.model_dump() for m in request.messages]
        input_text = tokenizer.apply_chat_template(
            conversation=messages_dicts,
            add_generation_prompt=True,
            tokenize=False,
        )
        input_ids = tokenizer.encode(input_text)
        prompt_tokens = len(input_ids)

        # 检查序列长度
        max_new = request.max_tokens or 2048
        if prompt_tokens + max_new > loaded.config.max_seq_len:
            max_new = loaded.config.max_seq_len - prompt_tokens
            if max_new <= 0:
                raise ContextLengthExceededError(
                    prompt_tokens=prompt_tokens,
                    max_seq_len=loaded.config.max_seq_len,
                )

        # 排队等待推理槽位
        async with self._queue.acquire():
            token_queue: asyncio.Queue[int | None] = asyncio.Queue()
            cancelled = threading.Event()

            # 注册活跃请求（用于取消）
            inf_req = InferenceRequest(
                id=request_id,
                cancelled=cancelled,
                token_queue=token_queue,
            )
            self._active_requests[request_id] = inf_req

            # 启动推理线程
            loop = asyncio.get_event_loop()
            future = loop.run_in_executor(
                self._executor,
                self._run_inference,
                model, tokenizer, input_ids, max_new,
                request.temperature, request.top_k, request.top_p,
                request.stop, token_queue, cancelled, loop,
            )

            # 流式输出
            completion_tokens = 0
            content_buffer = ""
            t0 = time.monotonic()
            ttft = None

            try:
                while True:
                    token_id = await token_queue.get()
                    if token_id is None:
                        break

                    completion_tokens += 1
                    if ttft is None:
                        ttft = time.monotonic() - t0

                    text = tokenizer.decode(
                        [token_id], skip_special_tokens=False
                    )

                    # 过滤特殊 token
                    if text in {"<|im_end|>", "<|endoftext|>"}:
                        break

                    content_buffer += text

                    yield ChatCompletionChunk(
                        id=request_id,
                        choices=[ChunkChoice(
                            index=0,
                            delta=ChunkDelta(content=text),
                            finish_reason=None,
                        )],
                    )

                # 最终 chunk（附带 usage 和 finish_reason）
                elapsed = time.monotonic() - t0
                yield ChatCompletionChunk(
                    id=request_id,
                    choices=[ChunkChoice(
                        index=0,
                        delta=ChunkDelta(),
                        finish_reason="stop",
                    )],
                    usage=UsageInfo(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                    ),
                )

            except asyncio.CancelledError:
                cancelled.set()
                raise
            finally:
                self._active_requests.pop(request_id, None)
                await asyncio.wrap_future(future)

    def _run_inference(
        self, model, tokenizer, input_ids, max_new_tokens,
        temperature, top_k, top_p, stop_sequences,
        token_queue, cancelled, loop,
    ):
        """在独立线程中运行同步推理"""
        try:
            for token_id in model.stream_generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            ):
                if cancelled.is_set():
                    break
                asyncio.run_coroutine_threadsafe(
                    token_queue.put(token_id), loop
                )
        finally:
            asyncio.run_coroutine_threadsafe(
                token_queue.put(None), loop  # sentinel
            )

    def cancel_request(self, request_id: str) -> bool:
        """取消指定请求"""
        req = self._active_requests.get(request_id)
        if req:
            req.cancelled.set()
            return True
        return False


@dataclass
class InferenceRequest:
    id: str
    cancelled: threading.Event
    token_queue: asyncio.Queue
```

**增量解码 vs 全量解码**：

当前实现采用逐 token 解码（`tokenizer.decode([token_id])`），这对于大多数情况足够。但部分 BPE token 可能跨越字符边界（如多字节 UTF-8 字符被拆成多个 token），会导致输出乱码。

**解决方案**（Phase 2 优化）：维护 token buffer，使用增量解码：

```python
class IncrementalDecoder:
    """处理 BPE token 跨字符边界的增量解码器"""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._token_buffer = []
        self._text_offset = 0

    def decode_token(self, token_id: int) -> str:
        self._token_buffer.append(token_id)
        full_text = self._tokenizer.decode(
            self._token_buffer, skip_special_tokens=False
        )
        new_text = full_text[self._text_offset:]
        # 只输出完整的 UTF-8 字符
        try:
            new_text.encode("utf-8")
            self._text_offset = len(full_text)
            return new_text
        except UnicodeEncodeError:
            return ""
```

### 5.5 Qwen3 Think 模式处理

Qwen3 支持"思考"模式（输出 `<think>...</think>` 标签包裹的推理过程）。API 层需要正确处理：

```python
class ThinkingFilter:
    """
    过滤或传递 Qwen3 的 <think> 内容。

    策略：
    - 默认：过滤 <think> 内容，只输出 </think> 之后的正文
    - 可选：通过请求参数 include_thinking=true 保留思考过程
    """

    def __init__(self, include_thinking: bool = False):
        self._include = include_thinking
        self._in_think = False
        self._think_done = False
        self._buffer = ""

    def process(self, text: str) -> str | None:
        self._buffer += text

        if not self._think_done:
            if "<think>" in self._buffer and "</think>" in self._buffer:
                parts = self._buffer.split("</think>", 1)
                self._think_done = True
                after = parts[1] if len(parts) > 1 else ""
                if self._include:
                    return self._buffer
                return after if after else None
            return None if not self._include else text

        return text
```

在 `/no_think` 模式下（用户消息末尾附加 `/no_think`），模型不会输出 `<think>` 标签，此过滤器自动透传。

### 5.6 SSE 路由实现

```python
# api/routes/chat.py

router = APIRouter()

@router.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    raw_request: Request,
    inference: InferenceService = Depends(get_inference_service),
):
    request_id = f"chatcmpl-{uuid4().hex[:12]}"

    if request.stream:
        return StreamingResponse(
            _stream_response(request, request_id, inference, raw_request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Nginx 反代时禁用缓冲
            },
        )
    else:
        return await _non_stream_response(request, request_id, inference)


async def _stream_response(
    request: ChatCompletionRequest,
    request_id: str,
    inference: InferenceService,
    raw_request: Request,
):
    """SSE 流式生成器"""
    try:
        async for chunk in inference.stream_chat_completion(request, request_id):
            # 检测客户端断开
            if await raw_request.is_disconnected():
                inference.cancel_request(request_id)
                break
            yield f"data: {chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
    except QueueFullError:
        error = {"error": {"message": "推理服务繁忙，请稍后重试", "type": "server_error"}}
        yield f"data: {json.dumps(error, ensure_ascii=False)}\n\n"
    except ContextLengthExceededError as e:
        error = {"error": {"message": f"输入过长 ({e.prompt_tokens} tokens)", "type": "invalid_request_error"}}
        yield f"data: {json.dumps(error, ensure_ascii=False)}\n\n"


async def _non_stream_response(
    request: ChatCompletionRequest,
    request_id: str,
    inference: InferenceService,
) -> ChatCompletionResponse:
    """非流式响应：收集所有 token 后一次性返回"""
    content = ""
    usage = None
    async for chunk in inference.stream_chat_completion(request, request_id):
        if chunk.choices[0].delta.content:
            content += chunk.choices[0].delta.content
        if chunk.usage:
            usage = chunk.usage

    return ChatCompletionResponse(
        id=request_id,
        object="chat.completion",
        choices=[CompletionChoice(
            index=0,
            message=ChatMessage(role="assistant", content=content),
            finish_reason="stop",
        )],
        usage=usage,
    )
```

### 5.7 模型列表与健康检查

```python
# api/routes/models.py

@router.get("/v1/models")
async def list_models(
    model_manager: ModelManager = Depends(get_model_manager),
):
    models = model_manager.list_models()
    return {
        "object": "list",
        "data": [
            {
                "id": m.id,
                "object": "model",
                "created": 0,
                "owned_by": "llaisys",
                "name": m.name,
                "max_context_length": m.max_context_length,
                "status": m.status,
            }
            for m in models
        ],
    }
```

```python
# api/routes/health.py

@router.get("/health")
async def health_check(
    model_manager: ModelManager = Depends(get_model_manager),
    queue: InferenceQueue = Depends(get_queue),
):
    models = model_manager.list_models()
    loaded_count = sum(1 for m in models if m.status == "loaded")
    q = queue.status

    return {
        "status": "healthy" if loaded_count > 0 else "degraded",
        "models": {
            "total": len(models),
            "loaded": loaded_count,
        },
        "queue": {
            "active": q.active,
            "waiting": q.waiting,
            "max_concurrent": q.max_concurrent,
        },
        "uptime_seconds": time.monotonic() - _start_time,
    }
```

---

## 6. 数据模型（Pydantic Schema）

### 6.1 请求模型

```python
# api/schemas/chat.py

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float = Field(default=0.6, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    top_k: int = Field(default=20, ge=0)
    max_tokens: int | None = Field(default=None, ge=1, le=32768)
    stop: list[str] | None = None

    # 扩展参数（非 OpenAI 标准）
    include_thinking: bool = Field(
        default=False,
        description="是否在输出中包含 Qwen3 的思考过程"
    )
```

### 6.2 响应模型

```python
class ChunkDelta(BaseModel):
    role: str | None = None
    content: str | None = None

class ChunkChoice(BaseModel):
    index: int
    delta: ChunkDelta
    finish_reason: str | None = None

class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str | None = None
    choices: list[ChunkChoice]
    usage: UsageInfo | None = None

class CompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str | None = None
    choices: list[CompletionChoice]
    usage: UsageInfo
```

---

## 7. 应用入口与生命周期

```python
# api/main.py

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务生命周期管理"""
    settings = get_settings()

    # 启动：加载模型
    logger.info("Starting LLM Service", models=len(settings.model_configs))
    model_manager = ModelManager(settings.model_configs)
    await model_manager.startup(settings.default_model)

    queue = InferenceQueue(
        max_concurrent=settings.max_concurrent,
        max_queue_size=settings.max_queue_size,
        timeout=settings.queue_timeout_seconds,
    )

    inference = InferenceService(model_manager, queue)

    # 注入到 app.state 供路由 Depends 使用
    app.state.model_manager = model_manager
    app.state.queue = queue
    app.state.inference = inference

    logger.info("LLM Service ready",
                model=settings.default_model,
                max_concurrent=settings.max_concurrent)
    yield

    # 关闭：释放资源
    logger.info("Shutting down LLM Service")
    await model_manager.shutdown()


app = FastAPI(
    title="LLM Service (llaisys)",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(chat_router)
app.include_router(models_router)
app.include_router(health_router)
```

**启动命令**：

```bash
cd llm_service
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

`workers=1` 是强制要求——GPU 模型实例不能跨进程共享，多进程会导致 CUDA context 冲突。

---

## 8. 可观测性

### 8.1 Prometheus 指标

```python
# 通过 prometheus-fastapi-instrumentator 自动暴露
# GET /metrics

# 自定义指标
inference_duration = Histogram(
    "llm_inference_duration_seconds",
    "推理请求总耗时",
    buckets=[0.5, 1, 2, 5, 10, 30, 60, 120],
)
ttft_duration = Histogram(
    "llm_ttft_seconds",
    "Time to First Token",
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
)
tokens_per_second = Histogram(
    "llm_tokens_per_second",
    "Decode 速度 (tokens/s)",
    buckets=[10, 20, 40, 60, 80, 100, 150],
)
queue_depth = Gauge(
    "llm_queue_depth",
    "当前排队等待数",
)
active_inferences = Gauge(
    "llm_active_inferences",
    "当前活跃推理数",
)
prompt_tokens_total = Counter(
    "llm_prompt_tokens_total",
    "累计 Prompt Token 数",
)
completion_tokens_total = Counter(
    "llm_completion_tokens_total",
    "累计 Completion Token 数",
)
```

### 8.2 结构化日志

每次推理请求完成后记录结构化日志：

```json
{
  "event": "inference_complete",
  "request_id": "chatcmpl-a1b2c3",
  "model": "qwen3-32b",
  "prompt_tokens": 512,
  "completion_tokens": 128,
  "ttft_ms": 180,
  "total_ms": 2340,
  "tokens_per_second": 54.7,
  "finish_reason": "stop",
  "cancelled": false
}
```

---

## 9. 错误处理

### 9.1 错误类型

| 错误 | HTTP 状态码 | 说明 |
|------|-----------|------|
| `ModelNotFoundError` | 404 | 请求的模型 ID 不存在 |
| `ContextLengthExceededError` | 400 | 输入超过模型上下文窗口 |
| `QueueFullError` | 503 | 推理队列已满 |
| `QueueTimeoutError` | 504 | 排队等待超时 |
| `InferenceError` | 500 | 推理过程中 CUDA 异常 |
| `ValidationError` | 422 | 请求参数校验失败（Pydantic 自动处理） |

### 9.2 全局异常处理

```python
# api/middleware/error_handler.py

@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.message,
                "type": exc.error_type,
                "code": exc.code,
            }
        },
    )

@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "type": "server_error",
            }
        },
    )
```

---

## 10. 性能指标目标

| 指标 | 目标（Qwen3-32B FP8, 双卡 TP=2） | 目标（Qwen3-8B FP8, 单卡） |
|------|----------------------------------|---------------------------|
| TTFT (prompt ≤ 1024 tokens) | < 500ms | < 200ms |
| TTFT (prompt ≤ 256 tokens) | < 150ms | < 80ms |
| Decode 速度（单请求） | ≥ 30 tokens/s | ≥ 60 tokens/s |
| 模型加载时间 | < 60s | < 30s |
| 显存占用 (模型权重) | ~34 GB (2×17GB) | ~8.5 GB |
| 显存占用 (KV-Cache, 8K) | ~2 GB | ~0.5 GB |

### 10.1 性能瓶颈分析

```
请求延迟组成：

[Tokenize]  ~5ms      ← CPU，可忽略
[Queue Wait] 0~120s   ← 取决于排队情况
[Prefill]   ~100-500ms ← GPU 计算密集，与 prompt 长度成正比
[Decode]    ~15-50ms/token ← GPU 访存密集，逐 token 串行
[Detokenize] ~0.1ms/token  ← CPU，可忽略
[HTTP/SSE]  ~1ms/chunk     ← 网络传输

瓶颈排序: Queue Wait >> Decode >> Prefill >> 其他
```

---

## 11. 引擎优化路线图

### 11.0 ✅ 已实现 — 多轮对话 KV-Cache 前缀复用

> 状态：**已上线** | 实现日期：2026-03-14

**解决的问题**：原实现中每次请求调用 `reset()` 全量清零 KV-Cache，多轮对话时历史 token 需要完整重新 Prefill，TTFT 随对话轮数线性增长。

**核心思路**：同一会话的多轮对话，prompt 存在大量公共前缀（System Prompt + 历史消息）。通过前缀匹配，只 Prefill 新增的 token，跳过已缓存的前缀部分。

**数据流**：

```
第 1 轮:  [System + User₁]                    → 完整 Prefill 200 tokens → 生成回复
第 2 轮:  [System + User₁ + Asst₁ + User₂]   → 前缀匹配 200+50=250 tokens → 只 Prefill 新增 30 tokens
第 3 轮:  [System + User₁ + Asst₁ + User₂ + Asst₂ + User₃] → 前缀匹配 330 tokens → 只 Prefill 新增 25 tokens
          ├─────── 缓存命中，跳过 ───────┤   ├── 实际计算 ──┤
```

**各层实现变更**：

| 层级 | 变更 | 文件 |
|------|------|------|
| C++ Model | 新增 `setCacheLen(size_t)` / `cacheLen()` 方法，支持将 cache_len 回滚到任意前缀位置 | `qwen3.hpp/cpp`, `qwen3_tp.hpp/cpp` |
| C API | 新增 `llaisysQwen3ModelSetCacheLen` / `llaisysQwen3ModelGetCacheLen` 导出函数 | `qwen3.h`, `qwen3.cc` |
| Python Bindings | 注册 ctypes 原型 | `libllaisys/models.py` |
| Python Model | `_compute_prefix_match()` 前缀匹配 + `stream_generate(reuse_cache=True)` 复用逻辑 | `models/qwen3.py` |
| FastAPI Schema | `ChatCompletionRequest` 新增 `session_id` 可选字段 | `schemas/chat.py` |
| FastAPI Inference | 当 `session_id` 非空时传递 `reuse_cache=True` 到引擎 | `services/inference.py` |
| Web Backend | `streamChatCompletion()` 透传 `session_id` 到 LLM 服务 | `llm-client.service.ts` |

**Python 层关键逻辑**（`Qwen3.stream_generate`）：

```python
# 1. 前缀匹配
prefix_len = self._compute_prefix_match(tokens)  # 逐 token 对比

# 2. 边界保护：至少保留 1 个 token 做 prefill（避免空输入）
if prefix_len >= len(tokens):
    prefix_len = max(len(tokens) - 1, 0)

# 3. 回滚 cache_len 到匹配位置，只 prefill 新增 suffix
if prefix_len > 0:
    self.set_cache_len(prefix_len)
    effective_input = tokens[prefix_len:]
else:
    self.reset()
    effective_input = tokens

# 4. 推理完成后，保存完整序列（含生成 tokens）供下次匹配
self._prev_input_ids = tokens + generated
self._prev_cache_len = len(tokens) + len(generated)
```

**API 使用方式**：

```bash
# 携带 session_id 即可启用 KV 复用
curl -X POST /v1/chat/completions -d '{
    "model": "qwen3-32b",
    "messages": [...],
    "session_id": "chat-session-abc123",  // ← 新增字段
    "stream": true
}'
```

**设计约束与限制**：

| 约束 | 说明 |
|------|------|
| 单会话模型 | 当前 `max_concurrent=1`，KV-Cache 状态属于模型实例，同时只有一个会话可复用 |
| 会话切换开销 | 不同 session_id 切换时前缀不匹配，退化为完整 Prefill（等价于原行为） |
| 不支持跨会话 | 同一 System Prompt 的不同会话仍需各自计算（需 PagedAttention + Prefix Cache 支持） |
| 生成内容变化 | 如果 temperature>0 导致上一轮回复不同，但 messages 中 assistant 内容匹配，KV 仍然正确（因为前缀匹配基于 input token IDs） |

**性能预期**：

| 指标 | 无复用 | 有复用（10 轮对话） | 说明 |
|------|--------|-------------------|------|
| TTFT | ~200ms/千 token | 第 2 轮起仅需 ~30ms | 只 Prefill 新增 token |
| 显存 | 不变 | 不变 | KV-Cache 仍为预分配，无额外开销 |
| 正确性 | 基线 | 等价 | 前缀 KV 与完整 Prefill 结果一致（确定性） |

---

### 11.1 P1 — KV-Cache 动态分配

**当前问题**：所有请求固定预分配 `max_seq_len=8192` 个位置的 KV-Cache，浪费显存。

**目标**：按实际序列长度按需分配，释放空闲的 KV-Cache。

**设计方案**：

```
┌────────────────────────────────────────────────┐
│                  KV-Cache Pool                   │
│                                                  │
│   ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐     │
│   │Block│ │Block│ │Block│ │Block│ │Block│ ... │
│   │  0  │ │  1  │ │  2  │ │  3  │ │  4  │     │
│   │128tk│ │128tk│ │128tk│ │128tk│ │128tk│     │
│   └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘     │
│      │       │       │       │       │         │
│      ▼       ▼       ▼       ▼       ▼         │
│   [Req A: Blocks 0,1,2]  [Req B: Blocks 3,4]  │
│   (384 tokens)            (256 tokens)          │
└────────────────────────────────────────────────┘
```

- Block 大小：128 tokens（可配置）
- Block 按需分配，序列增长时追加新 Block
- 请求结束后归还所有 Block 到空闲池
- 需要修改 C++ 层的 Self-Attention 算子以支持非连续 KV 存储

**C++ 层接口变更**：

```c
// 新增接口
struct LlaisysKVBlock;

LlaisysKVBlock* llaisysKVBlockPoolCreate(size_t block_size, size_t max_blocks, ...);
void llaisysKVBlockPoolDestroy(LlaisysKVBlock* pool);

// 修改推理接口，增加 KV-Block 管理参数
int64_t llaisysQwen3ModelInferPaged(
    struct LlaisysQwen3Model *model,
    int64_t *token_ids, size_t ntoken,
    int *block_table, size_t nblocks,  // 新增：Block 映射表
    float temperature, int top_k, float top_p, uint64_t seed
);
```

### 11.2 P1 — Prefix Cache

**目标**：对相同的 System Prompt 前缀，复用已计算的 KV-Cache，大幅降低 TTFT。

**设计方案**：

```
请求 A: [System Prompt (hash=abc123)] + [用户消息 A]
    ↓
Prefill System Prompt → 缓存 KV-Cache (key=abc123)
Prefill 用户消息 A → 全新计算
    ↓
请求 B: [System Prompt (hash=abc123)] + [用户消息 B]
    ↓
查找 Prefix Cache: hash=abc123 → 命中！
Copy-on-Write KV-Cache 前缀 → 只需 Prefill 用户消息 B
```

**实现要点**：

| 组件 | 说明 |
|------|------|
| Prefix Hash | 对 token 序列计算 SHA-256 前缀哈希 |
| Cache 存储 | 以 KV-Block 为单位存储（复用 11.1 的 Block Pool） |
| 匹配策略 | 最长前缀匹配——同一 System Prompt 的不同会话共享前缀 |
| 淘汰策略 | LRU（Least Recently Used），按时间戳淘汰 |
| 命中率指标 | 通过 Prometheus 暴露命中率 |

**Python 层接口**：

```python
class PrefixCache:
    def lookup(self, token_ids: list[int]) -> tuple[int, list[int]]:
        """
        查找最长匹配前缀。

        Returns:
            (matched_length, kv_block_ids)
            matched_length: 已缓存的前缀长度
            kv_block_ids: 对应的 KV-Cache Block ID 列表
        """
        ...

    def store(self, token_ids: list[int], kv_block_ids: list[int]):
        """存储前缀的 KV-Cache"""
        ...
```

### 11.3 P1 — Continuous Batching

**当前问题**：一次只能处理一个请求，GPU 利用率低（decode 阶段尤为明显，访存瓶颈）。

**目标**：多个请求的 decode 阶段合并为一个 batch 执行，提升吞吐。

**设计方案**：

```
时间 →
────────────────────────────────────────────────────
Step 1:  [Req A decode] [Req B decode]  ← 合并执行
Step 2:  [Req A decode] [Req B decode] [Req C prefill]  ← C 新加入
Step 3:  [Req A decode] [Req B done✓] [Req C decode]  ← B 完成退出
Step 4:  [Req A decode] [Req C decode] [Req D prefill]  ← D 插入
────────────────────────────────────────────────────

Scheduler 调度循环:

while True:
    1. 检查是否有新请求需要 Prefill
    2. 检查已完成 / 已取消的请求 → 移除并释放 KV-Cache
    3. 将所有活跃请求的当前 token 合并成 batch
    4. 执行一次 batched decode → 得到每个请求的下一个 token
    5. 分发 token 到各请求的输出队列
```

**C++ 层接口变更**：

```c
// Batched decode 接口
void llaisysQwen3ModelDecodeBatch(
    struct LlaisysQwen3Model *model,
    int64_t *token_ids,          // [batch_size] 当前 step 的 token
    int *seq_lengths,            // [batch_size] 每个序列的当前长度
    int **block_tables,          // [batch_size][max_blocks] 每个序列的 Block 映射
    int batch_size,
    int64_t *output_tokens       // [batch_size] 输出 next token
);

// Prefill 接口（单请求）
void llaisysQwen3ModelPrefill(
    struct LlaisysQwen3Model *model,
    int64_t *token_ids, size_t ntoken,
    int *block_table, size_t nblocks
);
```

**调度策略**：

| 策略 | 说明 |
|------|------|
| Prefill 优先 | 新请求到达时优先执行 Prefill，减少首 Token 延迟 |
| Prefill-Decode 分离 | Prefill 和 Decode 不在同一 batch 中混合（避免长 Prefill 阻塞 Decode） |
| 动态 Batch Size | 根据显存余量动态调整 batch 大小 |
| 抢占式调度 | KV-Cache 不足时，低优先级请求可被暂停（KV-Cache swap 到 CPU） |

### 11.4 P2 — 其他优化

| 优化项 | 说明 | 预期收益 |
|--------|------|---------|
| Chunked Prefill | 长 Prompt 分块 Prefill（如每 512 tokens 一块），每块之间可插入 Decode step | 降低长 Prompt 的 TTFT，减少 Decode 延迟抖动 |
| Speculative Decoding | 用 Qwen3-8B 作为 draft model 生成多个候选 token，用 Qwen3-32B 批量验证 | Decode 吞吐提升 2-3x（取决于接受率） |
| 量化 KV-Cache | KV-Cache 从 BF16 压缩到 FP8/INT8 | 显存减半，可支持更长上下文或更多并发 |
| FlashAttention 集成 | 使用 FlashAttention-2/3 替换当前的 Self-Attention 实现 | Prefill 加速 2-4x，显存效率提升 |

---

## 12. 分阶段实施计划

### Phase 1 — MVP OpenAI API（1-2 周）

| 任务 | 详情 | 依赖 |
|------|------|------|
| 1.1 项目初始化 | 创建 `api/` 目录，安装依赖，配置管理 | - |
| 1.2 Schema 定义 | Pydantic 模型：ChatCompletionRequest/Response/Chunk | - |
| 1.3 ModelManager | 模型加载、单模型管理、启动/关闭生命周期 | 1.1 |
| 1.4 InferenceService | 线程池推理 + asyncio.Queue 桥接 + 流式输出 | 1.3 |
| 1.5 InferenceQueue | asyncio.Semaphore 并发控制（max_concurrent=1） | 1.1 |
| 1.6 Chat 路由 | `POST /v1/chat/completions`（流式 + 非流式） | 1.2, 1.4, 1.5 |
| 1.7 Models 路由 | `GET /v1/models` | 1.3 |
| 1.8 Health 路由 | `GET /health` | 1.3, 1.5 |
| 1.9 错误处理 | 全局异常处理中间件 | 1.1 |
| 1.10 联调 | 与后端 Next.js 联调完整链路 | 全部 |

**Phase 1 交付物**：一个可运行的 FastAPI 服务，提供 OpenAI 兼容的 `/v1/chat/completions`（流式 SSE），可被后端 Next.js 调用。单请求模式。

### Phase 2 — 请求管理与稳定性（1 周）

| 任务 | 详情 |
|------|------|
| 2.1 请求取消 | 客户端断开检测 + cancel_request 实现 |
| 2.2 Think 模式 | ThinkingFilter + include_thinking 参数 |
| 2.3 增量解码 | IncrementalDecoder 处理 BPE 跨字符问题 |
| 2.4 多模型支持 | models.json 配置，ModelManager 多模型管理 |
| 2.5 Prometheus | 关键指标暴露 + `/metrics` 端点 |
| 2.6 结构化日志 | structlog 集成，推理请求完整日志 |
| 2.7 Stop 序列 | 支持自定义 stop 序列提前终止生成 |

### Phase 5 — 引擎优化（持续）

| 任务 | 详情 | 优先级 |
|------|------|--------|
| ~~5.0 多轮 KV 复用~~ | ~~同会话前缀匹配 + setCacheLen 回滚~~ | ✅ 已完成 |
| 5.1 KV-Cache Block Pool | C++ 层 Block 分配器 + Python 封装 | P1 |
| 5.2 PagedAttention | 修改 Self-Attention 算子支持非连续 KV | P1 |
| 5.3 Prefix Cache（跨会话） | 前缀哈希 + LRU 缓存 + 命中率监控（需 PagedAttention） | P1 |
| 5.4 Continuous Batching | Scheduler + Batched Decode + 动态插入 | P1 |
| 5.5 Prefill-Decode 分离 | 独立调度 Prefill 和 Decode | P1 |
| 5.6 Chunked Prefill | 长 Prompt 分块 + Decode 交错 | P2 |
| 5.7 Speculative Decoding | Draft-Verify 流水线 | P2 |
| 5.8 KV-Cache 量化 | FP8/INT8 KV 压缩 | P2 |

---

## 13. 部署方案

### 13.1 本地开发

```bash
# 1. 构建 libllaisys.so
cd llm_service
xmake f --nv-gpu=y -m release
xmake build llaisys

# 2. 安装 Python 依赖
pip install -r api/requirements.txt

# 3. 启动推理服务
LLM_DEFAULT_MODEL=qwen3-32b \
LLM_DEVICE=nvidia \
LLM_DEVICE_IDS="[0,1]" \
LLM_TP_SIZE=2 \
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 13.2 生产部署

```
┌────────────────────────────────────────────────────┐
│                  GPU 服务器                          │
│                                                      │
│   ┌──────────────────────────────────────┐          │
│   │     systemd / supervisor              │          │
│   │                                       │          │
│   │   uvicorn api.main:app               │          │
│   │     --host 0.0.0.0                   │  :8000   │
│   │     --port 8000                      │          │
│   │     --workers 1                      │          │
│   │     --timeout-keep-alive 300         │          │
│   └──────────────────────────────────────┘          │
│                                                      │
│   GPU 0: [Qwen3-32B TP shard 0]                    │
│   GPU 1: [Qwen3-32B TP shard 1]                    │
└────────────────────────────────────────────────────┘
```

**systemd unit 示例**：

```ini
[Unit]
Description=LLM Inference Service (llaisys)
After=network.target

[Service]
Type=simple
User=llm
WorkingDirectory=/home/llm/llm_service
Environment=LLM_DEFAULT_MODEL=qwen3-32b
Environment=LLM_DEVICE=nvidia
Environment=LLM_DEVICE_IDS=[0,1]
Environment=LLM_TP_SIZE=2
ExecStart=/home/llm/.venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1 --timeout-keep-alive 300
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### 13.3 与后端通信

后端 Next.js 通过环境变量 `LLM_SERVICE_URL` 指向推理服务：

```
LLM_SERVICE_URL=http://gpu-server:8000
```

通信协议完全兼容 OpenAI API，后端可无缝切换到其他兼容服务（如 vLLM、Ollama）用于测试。

---

## 14. 安全考虑

| 措施 | 说明 |
|------|------|
| 内网部署 | 推理服务仅在内网暴露，不对外网开放 |
| API Key（可选） | 通过 `Authorization: Bearer <key>` 验证调用方身份 |
| 请求大小限制 | FastAPI 限制请求 body 大小（默认 1MB），防止超大 Prompt 攻击 |
| 超时控制 | 单请求最大生成时间 300s，避免资源泄漏 |
| 输入过滤 | Token 数检查，拒绝超长输入 |
| 资源隔离 | 单进程运行，GPU 资源不与其他进程共享 |

---

## 15. 风险与应对

| 风险 | 影响 | 应对措施 |
|------|------|----------|
| CUDA OOM | 推理服务崩溃 | 严格的 max_seq_len 检查 + KV-Cache 预算控制；捕获 CUDA 异常后优雅降级 |
| 长生成阻塞 | 后续请求长时间排队 | 设置 max_tokens 上限 + 超时机制 + 客户端可取消 |
| 模型加载慢 | 服务启动时间长 | 异步加载 + 健康检查分级（degraded 状态允许排队） |
| Tokenizer 不一致 | Token 计数偏差 | 统一使用模型目录中的 tokenizer，确保 encode/decode 一致 |
| 热重载模型 | 切换模型时服务中断 | Phase 2+ 实现模型热切换：加载新模型完毕后原子切换，旧模型等待活跃请求完成后释放 |
| libllaisys.so 崩溃 | 整个服务不可恢复 | systemd 自动重启；Phase 5+ 考虑子进程隔离推理 |
