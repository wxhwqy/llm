# LLM Service API 文档

> 版本：0.1 | 最后更新：2026-03-03
>
> 服务地址：`http://<gpu-server>:8000`
>
> 兼容标准：OpenAI Chat Completions API

---

## 目录

1. [快速开始](#1-快速开始)
2. [POST /v1/chat/completions — 聊天补全](#2-post-v1chatcompletions)
3. [GET /v1/models — 模型列表](#3-get-v1models)
4. [GET /health — 健康检查](#4-get-health)
5. [错误码参考](#5-错误码参考)
6. [SSE 流式协议详解](#6-sse-流式协议详解)
7. [后端集成指南](#7-后端集成指南)

---

## 1. 快速开始

### 启动服务

```bash
cd llm_service
# 确保已构建 libllaisys.so 并安装 Python 依赖
pip install -r api/requirements.txt

# 启动（使用默认配置，加载 models.json 中的模型）
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLM_HOST` | `0.0.0.0` | 监听地址 |
| `LLM_PORT` | `8000` | 监听端口 |
| `LLM_DEFAULT_MODEL` | `qwen3-32b` | 启动时预加载的默认模型 |
| `LLM_MODELS_CONFIG` | `models.json` | 模型配置文件路径 |
| `LLM_MAX_CONCURRENT` | `1` | 最大并发推理数 |
| `LLM_MAX_QUEUE_SIZE` | `16` | 最大排队等待数 |
| `LLM_QUEUE_TIMEOUT` | `120` | 排队超时时间（秒） |
| `LLM_DEVICE` | `nvidia` | 设备类型（nvidia / cpu） |
| `LLM_DEVICE_IDS` | `[0,1]` | GPU 设备 ID 列表（JSON 数组） |
| `LLM_TP_SIZE` | `2` | Tensor Parallelism 并行度 |

### 快速测试

```bash
# 非流式请求
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-32b",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 100
  }'

# 流式请求
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-32b",
    "messages": [{"role": "user", "content": "你好"}],
    "stream": true,
    "max_tokens": 100
  }'
```

---

## 2. POST /v1/chat/completions

聊天补全接口，兼容 OpenAI `POST /v1/chat/completions` 格式。

### 请求

**URL**: `POST /v1/chat/completions`

**Content-Type**: `application/json`

**请求体**:

```json
{
  "model": "qwen3-32b",
  "messages": [
    {"role": "system", "content": "你是一个友善的助手。"},
    {"role": "user", "content": "介绍一下自己"}
  ],
  "stream": true,
  "temperature": 0.6,
  "top_p": 0.95,
  "top_k": 20,
  "max_tokens": 2048,
  "stop": ["<|endoftext|>"]
}
```

**参数说明**:

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `model` | string | **是** | — | 模型 ID，需与 `GET /v1/models` 返回的 id 匹配 |
| `messages` | array | **是** | — | 消息列表，每条消息包含 `role` 和 `content` |
| `messages[].role` | string | **是** | — | `"system"` / `"user"` / `"assistant"` |
| `messages[].content` | string | **是** | — | 消息文本内容 |
| `stream` | boolean | 否 | `false` | 是否使用 SSE 流式输出 |
| `temperature` | float | 否 | `0.6` | 采样温度，范围 `[0.0, 2.0]`。0 表示贪婪解码 |
| `top_p` | float | 否 | `0.95` | 核采样阈值，范围 `[0.0, 1.0]` |
| `top_k` | int | 否 | `20` | Top-K 采样。1 = 贪婪，0 = 不限制 |
| `max_tokens` | int \| null | 否 | `2048` | 最大生成 token 数，范围 `[1, 131072]` |
| `stop` | array \| null | 否 | `null` | 自定义停止序列（暂未实现，预留） |

### 非流式响应

**HTTP 200**

```json
{
  "id": "chatcmpl-a1b2c3d4e5f6a1b2c3d4e5f6",
  "object": "chat.completion",
  "created": 1709500000,
  "model": "qwen3-32b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "你好！我是 Qwen3，一个 AI 助手。很高兴认识你！"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 28,
    "completion_tokens": 20,
    "total_tokens": 48
  }
}
```

**字段说明**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 请求唯一标识，格式 `chatcmpl-{24位hex}` |
| `object` | string | 固定为 `"chat.completion"` |
| `created` | int | Unix 时间戳 |
| `model` | string | 实际使用的模型 ID |
| `choices[].message` | object | 生成的完整消息 |
| `choices[].finish_reason` | string | `"stop"` = 正常结束，`"length"` = 达到 max_tokens |
| `usage.prompt_tokens` | int | 输入消耗的 token 数 |
| `usage.completion_tokens` | int | 生成的 token 数 |
| `usage.total_tokens` | int | 总 token 数 |

### 流式响应 (stream=true)

**HTTP 200**, `Content-Type: text/event-stream`

SSE 流中的每个事件以 `data: ` 开头，以 `\n\n` 结尾：

```
data: {"id":"chatcmpl-a1b2c3d4e5f6a1b2c3d4e5f6","object":"chat.completion.chunk","created":1709500000,"model":"qwen3-32b","choices":[{"index":0,"delta":{"content":"你"},"finish_reason":null}],"usage":null}

data: {"id":"chatcmpl-a1b2c3d4e5f6a1b2c3d4e5f6","object":"chat.completion.chunk","created":1709500000,"model":"qwen3-32b","choices":[{"index":0,"delta":{"content":"好"},"finish_reason":null}],"usage":null}

data: {"id":"chatcmpl-a1b2c3d4e5f6a1b2c3d4e5f6","object":"chat.completion.chunk","created":1709500000,"model":"qwen3-32b","choices":[{"index":0,"delta":{"content":"！"},"finish_reason":null}],"usage":null}

data: {"id":"chatcmpl-a1b2c3d4e5f6a1b2c3d4e5f6","object":"chat.completion.chunk","created":1709500000,"model":"qwen3-32b","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":28,"completion_tokens":3,"total_tokens":31}}

data: [DONE]
```

**Chunk 字段说明**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `choices[].delta.content` | string \| null | 本次增量生成的文本片段，为 `null` 表示无新内容 |
| `choices[].finish_reason` | string \| null | 中间 chunk 为 `null`；最后一个 chunk 为 `"stop"` |
| `usage` | object \| null | 仅在最后一个 chunk（`finish_reason` 不为 `null`）中包含用量信息 |

**流结束标记**: `data: [DONE]\n\n` 表示 SSE 流结束。

---

## 3. GET /v1/models

获取所有已注册模型的列表。

### 请求

**URL**: `GET /v1/models`

### 响应

**HTTP 200**

```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen3-32b",
      "object": "model",
      "created": 0,
      "owned_by": "llaisys",
      "name": "Qwen3 32B (FP8)",
      "max_context_length": 8192,
      "status": "loaded"
    },
    {
      "id": "qwen3-8b",
      "object": "model",
      "created": 0,
      "owned_by": "llaisys",
      "name": "Qwen3 8B (FP8)",
      "max_context_length": 8192,
      "status": "available"
    }
  ]
}
```

**字段说明**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `data[].id` | string | 模型唯一标识，用于 `chat/completions` 的 `model` 参数 |
| `data[].name` | string | 模型人类可读名称，可用于前端下拉框显示 |
| `data[].max_context_length` | int | 最大上下文窗口长度 (tokens) |
| `data[].status` | string | `"loaded"` = 已加载到 GPU，`"available"` = 已注册但未加载 |

**后端使用建议**:

- 缓存此接口 60 秒，模型列表变化不频繁
- `status` 为 `"loaded"` 的模型可立即推理；`"available"` 的模型首次请求时需等待加载（约 30-60 秒）
- `max_context_length` 用于前端上下文用量进度条的计算

---

## 4. GET /health

服务健康检查，用于运维监控和后端启动探测。

### 请求

**URL**: `GET /health`

### 响应

**HTTP 200**

```json
{
  "status": "healthy",
  "models": {
    "total": 3,
    "loaded": 1
  },
  "queue": {
    "active": 0,
    "waiting": 2,
    "max_concurrent": 1
  },
  "uptime_seconds": 3600.5
}
```

**字段说明**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | string | `"healthy"` = 至少一个模型已加载；`"degraded"` = 无模型可用 |
| `models.total` | int | 注册的模型总数 |
| `models.loaded` | int | 当前已加载到 GPU 的模型数 |
| `queue.active` | int | 当前正在推理的请求数 |
| `queue.waiting` | int | 当前排队等待的请求数 |
| `queue.max_concurrent` | int | 最大并发推理数 |
| `uptime_seconds` | float | 服务运行时长（秒） |

**后端使用建议**:

- 用于服务启动时探测 LLM Service 是否就绪：轮询 `/health` 直到 `status == "healthy"`
- `queue.waiting` 可用于前端显示排队状态
- 连接失败时应优雅降级，不阻止后端启动

---

## 5. 错误码参考

所有错误响应使用统一格式：

```json
{
  "error": {
    "message": "错误描述信息",
    "type": "错误类型",
    "code": "错误码"
  }
}
```

### 错误码列表

| HTTP 状态码 | type | code | 触发场景 | 后端处理建议 |
|------------|------|------|---------|-------------|
| **400** | `invalid_request_error` | `context_length_exceeded` | Prompt token 数超过模型 `max_context_length` | 提示用户消息过长，触发上下文压缩后重试 |
| **404** | `invalid_request_error` | `model_not_found` | 请求的 model ID 不在注册列表中 | 检查 model ID 是否正确，从 `/v1/models` 获取可用列表 |
| **422** | `validation_error` | — | 请求体参数校验失败（Pydantic 自动处理） | 检查请求参数格式 |
| **503** | `server_error` | `queue_full` | 排队等待数已达 `max_queue_size` 上限 | 向用户显示"服务繁忙，请稍后重试" |
| **504** | `server_error` | `queue_timeout` | 排队等待超过 `queue_timeout_seconds` | 向用户显示"等待超时" |
| **500** | `server_error` | — | 未预期的服务端错误 | 记录日志，显示通用错误提示 |

### 流式响应中的错误

当 SSE 流已经开始后发生错误，错误会以内联 SSE 事件的形式发送：

```
data: {"error":{"message":"推理线程异常","type":"InferenceError"}}

data: [DONE]
```

后端收到包含 `error` 字段的 SSE data 时，应保留已接收的内容，向前端显示错误提示。

---

## 6. SSE 流式协议详解

### 解析伪代码（TypeScript / Node.js）

```typescript
async function streamChat(messages: ChatMessage[]): AsyncGenerator<string> {
  const response = await fetch("http://gpu-server:8000/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "qwen3-32b",
      messages,
      stream: true,
      temperature: 0.8,
      max_tokens: 2048,
    }),
    signal: abortController.signal,
  });

  if (!response.ok) {
    const err = await response.json();
    throw new Error(err.error.message);
  }

  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let usage = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const data = line.slice(6);

      if (data === "[DONE]") return;

      const chunk = JSON.parse(data);

      // 检查内联错误
      if (chunk.error) {
        throw new Error(chunk.error.message);
      }

      // 提取 delta content
      const content = chunk.choices?.[0]?.delta?.content;
      if (content) {
        yield content;
      }

      // 最后一个 chunk 包含 usage
      if (chunk.usage) {
        usage = chunk.usage;
      }
    }
  }
}
```

### 停止生成

客户端断开连接（`AbortController.abort()`）即可停止生成：

- 后端的 SSE 转发层检测到 `request.is_disconnected()` 后调用 `inference.cancel(request_id)`
- LLM Service 在推理线程中检测到 `cancelled` 标记后停止 token 生成
- 已生成的部分内容由后端负责保存

**无需额外的 stop API 调用**——直接断开 HTTP 连接即可。

### 客户端断开检测延迟

SSE 流的断开检测依赖于下一次写操作失败，因此可能有 1-2 个 token 的延迟。这在实际使用中可忽略。

---

## 7. 后端集成指南

### 7.1 环境变量配置

在后端 Next.js 的 `.env` 中配置：

```bash
LLM_SERVICE_URL=http://gpu-server:8000
```

### 7.2 调用示例（Next.js API Route）

```typescript
// 流式转发示例
export async function POST(request: NextRequest) {
  const body = await request.json();

  const llmResponse = await fetch(
    `${process.env.LLM_SERVICE_URL}/v1/chat/completions`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: body.modelId || "qwen3-32b",
        messages: body.messages,
        stream: true,
        temperature: 0.8,
        top_p: 0.95,
        max_tokens: 2048,
      }),
    }
  );

  if (!llmResponse.ok) {
    const error = await llmResponse.json();
    return NextResponse.json(error, { status: llmResponse.status });
  }

  // 透传 SSE 流
  return new Response(llmResponse.body, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "Connection": "keep-alive",
    },
  });
}
```

### 7.3 重要注意事项

| 事项 | 说明 |
|------|------|
| **workers=1** | LLM Service 必须单进程运行，GPU 模型不能跨进程共享 |
| **连接超时** | 建议后端连接超时设置为 5s，读超时不设限（流式） |
| **重试策略** | 非流式请求可重试 1 次（503/504）；流式请求不重试 |
| **Nginx 缓冲** | 如使用 Nginx 反代，需设置 `proxy_buffering off` 和 `X-Accel-Buffering: no` |
| **SSE Runtime** | Next.js 的 SSE 路由需使用 Node.js Runtime（`export const runtime = 'nodejs'`） |
| **兼容性** | 本 API 兼容 OpenAI 格式，后端可无缝切换到 vLLM / Ollama 等兼容服务进行测试 |

### 7.4 Token 用量提取

从流式响应的最后一个 chunk 中提取 `usage` 字段：

```typescript
// 最后一个 chunk (finish_reason 不为 null) 包含 usage
if (chunk.choices[0].finish_reason) {
  const usage = chunk.usage;
  // usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
  await recordTokenUsage(sessionId, usage);
}
```

**注意**：`usage` 仅在 `finish_reason` 不为 `null` 的 chunk 中存在。如果客户端提前断开（停止生成），则不会收到 usage 信息，后端应以 0 记录该次请求的 completion_tokens。

### 7.5 上下文长度管理

| 参数 | 来源 | 用途 |
|------|------|------|
| `max_context_length` | `GET /v1/models` 返回 | 上下文窗口总大小 |
| `prompt_tokens` | `usage.prompt_tokens` | 本次请求实际的输入 token 数 |
| `max_tokens` | 请求参数 | 限制本次生成的最大 token 数 |

后端在构建 Prompt 时应确保：

```
prompt_tokens + max_tokens ≤ max_context_length
```

如果 `prompt_tokens` 超过 `max_context_length`，LLM Service 会返回 400 错误。后端应在发送前进行预估并触发上下文压缩。

---

## 附录 A: models.json 配置格式

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
  }
]
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `id` | string | 是 | API 请求中使用的模型标识 |
| `name` | string | 是 | 人类可读名称 |
| `path` | string | 是 | 模型权重目录（相对于 llm_service/ 或绝对路径） |
| `max_seq_len` | int | 否 | 最大序列长度，默认 8192 |
| `device` | string | 否 | `"nvidia"` 或 `"cpu"`，默认 `"nvidia"` |
| `device_ids` | int[] | 否 | GPU 设备 ID 列表，默认 `[0, 1]` |
| `tp_size` | int | 否 | Tensor Parallelism 并行度，默认 2 |

## 附录 B: 自动 API 文档

服务启动后可访问以下自动生成的文档：

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`
