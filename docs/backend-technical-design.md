# 聊天机器人项目 — 后端技术方案文档

> 版本：0.7 | 最后更新：2026-03-14
>
> 对应 PRD 版本：0.3 | 对应 LLM Service API 版本：0.2

---

## 1. 文档范围与系统定位

本文档覆盖**用户后台 (server/)** 层——即 Next.js API Routes 中间层的技术方案。它向上为前端提供业务 API，向下对接大模型推理服务（LLM Service，详见 `docs/llm-service-api.md`）。

```
Frontend (Next.js)  ←─ HTTP/SSE ─→  Backend (本文档)  ←─ OpenAI API ─→  LLM Service (FastAPI :8000)
                                          │
                                    PostgreSQL + Redis
```

**核心职责**：认证授权、角色卡/世界书管理、Prompt 构建与上下文管理、SSE 流式代理、Token 计量、请求排队。

---

## 2. 技术选型

| 组件 | 选型 | 理由 |
|------|------|------|
| 框架 | Next.js 14+ (App Router) | 前后端同栈，API Routes 内置 |
| 语言 | TypeScript | 类型安全，配合 Prisma 自动生成类型 |
| ORM | Prisma | 类型安全、迁移管理、SQLite/PG 无缝切换 |
| 数据库 | SQLite (Phase 1-3) → PostgreSQL (Phase 4+) | 开发零配置，生产上 JSONB + 并发 |
| 缓存/队列 | Redis (Phase 4+) | 请求排队、速率限制、缓存 |
| 认证 | NextAuth.js + JWT + bcrypt | Phase 4 启用，之前硬编码默认用户 |
| 数据校验 | Zod | 运行时类型校验，与 TS 配合好 |
| PNG 解析 | png-chunks-extract + png-chunk-text | 提取 SillyTavern 角色卡 tEXt chunk |

---

## 3. 分层架构与目录结构

采用 **Route Handler → Service → Repository** 三层架构：

- **Route Handler**（`app/api/`）：参数校验、权限检查、调用 Service、返回响应
- **Service**（`server/services/`）：核心业务逻辑，不依赖 HTTP 层
- **Repository**（`server/db/repositories/`）：Prisma 查询封装

```
web/src/
├── app/api/                    # Route Handlers（按 PRD 6.2 API 表映射）
│   ├── auth/                   # login / register / logout / me
│   ├── characters/             # 列表、详情、标签（所有用户）
│   ├── admin/characters/       # 导入、创建、编辑、删除（🔒 管理员）
│   ├── worldbooks/             # CRUD + 词条 + 导入导出（按 scope 权限控制）
│   ├── chat/sessions/          # 会话 CRUD + 消息(SSE) + 重新生成 + 停止
│   ├── users/me/               # 用户信息 + Token 用量
│   └── models/                 # 代理 LLM Service 模型列表
├── server/
│   ├── services/               # 业务逻辑
│   │   ├── session.service.ts       # 会话 CRUD（创建、列表、更新、删除）
│   │   ├── message.service.ts       # 消息 CRUD（获取历史、编辑）
│   │   ├── chat-stream.service.ts   # 聊天流引擎：发消息、重新生成（组装 Prompt + 调用 SSE 引擎）
│   │   ├── sse-stream.service.ts    # 通用 SSE 流引擎：LLM 调用 → chunk 转发 → 回调
│   │   ├── prompt-builder.service.ts # Prompt 组装
│   │   ├── context-manager.service.ts # 上下文窗口管理 + 压缩
│   │   ├── worldbook-injector.service.ts # 世界书关键词匹配与注入
│   │   ├── character-import.service.ts   # PNG/JSON 角色卡解析
│   │   ├── llm-client.service.ts    # 封装 LLM Service HTTP 调用
│   │   └── ...                      # auth, worldbook, model, queue 等
│   ├── db/
│   │   ├── prisma.ts                # Prisma Client 单例
│   │   └── repositories/            # 数据访问层
│   │       ├── session.repo.ts      # ChatSession 查询 + 归属校验
│   │       └── message.repo.ts      # ChatMessage 查询 + 游标分页
│   ├── middleware/              # withAuth, withAdmin, rateLimit
│   ├── validators/              # Zod schemas
│   └── lib/                    # errors, config, redis, response 工具
├── types/                      # 全局类型定义
└── prisma/schema.prisma        # 数据库 Schema
```

---

## 4. 数据库设计

### 4.1 ER 关系概览

```
User ──1:N──► ChatSession ──1:N──► ChatMessage
  │                │
  │                ├──1:N──► TokenUsage
  │                │
  │                └──N:M──► WorldBook (SessionWorldBook, 个人世界书)
  │
  ├──1:N──► CharacterCard ──N:M──► WorldBook (CharacterWorldBook, 全局世界书)
  │
  └──1:N──► WorldBook (通过 userId 归属)
```

相比 PRD v0.1 的主要变化：

1. **WorldBook 增加 `scope` + `userId`**：区分全局世界书（管理员创建，所有人可见）和个人世界书（用户创建，仅自己可见）
2. **新增 SessionWorldBook 关联表**：用户可在会话中手动启用个人世界书
3. **CharacterCard 增加 `coverImage`**：大封面图 URL

### 4.2 核心表结构

> 以下为 Prisma Schema 的关键部分，省略通用注解。

**User**：id, username, email, passwordHash, role(USER/ADMIN), **status(ACTIVE/DISABLED)**, createdAt, updatedAt

**CharacterCard**：id, name, avatar?, coverImage?, description, personality, **preset**, scenario, systemPrompt, firstMessage, alternateGreetings[], exampleDialogue, creatorNotes, tags[], source(MANUAL/SILLYTAVERN_PNG/JSON_IMPORT), createdBy → User

> **v0.4 新增字段**：
> - `preset`（String, 默认空）：角色卡预设文本，在 Prompt 中位于最前面（system message 的第一段），用于注入全局性的角色扮演规则、输出格式要求等。与 `systemPrompt` 的区别：`preset` 是跨角色可复用的通用指令（如"请用中文回复"、"你是一个角色扮演 AI"），`systemPrompt` 是角色专属的提示词。

**WorldBook**：id, name, description, version, **scope(GLOBAL/PERSONAL)**, **userId → User**, totalTokenCount, createdAt, updatedAt

**WorldBookEntry**：id, worldBookId → WorldBook, keywords[], secondaryKeywords[], content, position(BEFORE_SYSTEM/AFTER_SYSTEM/BEFORE_USER), priority, enabled, tokenCount

**CharacterWorldBook**：characterId + worldBookId（角色卡关联全局世界书，N:M）

**SessionWorldBook**：sessionId + worldBookId（会话启用个人世界书，N:M）

**ChatSession**：id, userId → User, characterId → CharacterCard, modelId, title, usedTokens, maxTokens, createdAt, updatedAt

**ChatMessage**：id, sessionId → ChatSession, role(SYSTEM/USER/ASSISTANT), content, tokenCount, isCompressed, createdAt, editedAt?

**TokenUsage**：id, userId, sessionId, messageId, modelId, promptTokens, completionTokens, totalTokens, createdAt

**LlmProvider**（v0.6 新增）：id, name, baseUrl, apiKey, models(JSON), autoDiscover, enabled, priority, createdAt, updatedAt

### 4.3 关键索引

| 表 | 索引 | 用途 |
|---|---|---|
| CharacterCard | `[name]` | 模糊搜索 |
| WorldBook | `[scope, userId]` | 按可见范围过滤 |
| ChatSession | `[userId, updatedAt DESC]` | 会话列表排序 |
| ChatMessage | `[sessionId, createdAt]` | 游标分页 |
| TokenUsage | `[userId, createdAt]` | 用量聚合 |

### 4.4 SQLite → PostgreSQL 迁移

Phase 1-3 用 SQLite 开发（零配置），Phase 4 切 PostgreSQL。Prisma 自动处理 `String[]` 在 SQLite 中的 JSON 序列化、`enum` 的字符串退化等差异。切换时只需改 `provider` + `DATABASE_URL`，重新 `prisma migrate`。

---

## 5. 认证与授权

### 5.1 认证实现

采用 JWT + bcrypt 方案，不依赖 NextAuth.js（减少黑盒依赖）：

**注册流程**：
1. Zod 校验用户名（2-20字符）、邮箱格式、密码强度（≥8字符）
2. 检查用户名 / 邮箱唯一性，冲突返回 409
3. `bcrypt.hash(password, 12)` 生成密码哈希
4. 创建 User 记录（role 默认 `user`，status 默认 `active`）
5. 签发 JWT，写入 httpOnly Cookie

**登录流程**：
1. 按 email 查找用户，不存在返回 401
2. 检查 `status === "active"`，被禁用返回 403（"账号已被禁用"）
3. `bcrypt.compare(password, passwordHash)` 校验密码
4. 签发 JWT（payload: `{ userId, role }`，过期时间 7 天），写入 Cookie：
   `Set-Cookie: auth-token=xxx; HttpOnly; Secure; SameSite=Lax; Max-Age=604800; Path=/`

**JWT 校验（withAuth 中间件）**：
1. 从 `req.cookies.get("auth-token")` 读取 token
2. `jwt.verify(token, JWT_SECRET)` 解码
3. 按 userId 查询用户，不存在或 status 非 active 返回 401
4. 注入 user 到 handler 参数

**登出**：清除 `auth-token` Cookie 即可。

**密码修改**：
1. 校验旧密码（bcrypt.compare）
2. 哈希新密码并更新 DB

### 5.2 用户管理（管理员）

新增 `user.service.ts`，提供管理员用户管理功能：

| 功能 | 说明 |
|------|------|
| 用户列表 | 支持按用户名/邮箱搜索、按角色/状态筛选、分页 |
| 禁用/启用用户 | 设置 `status` 为 `disabled` / `active`。禁用后该用户的所有请求返回 403 |
| 修改用户角色 | 将普通用户提升为管理员或反之。不能修改自己的角色（防止唯一管理员降权） |
| 删除用户 | 级联删除该用户的所有会话、消息、token 记录。不能删除自己 |

**安全约束**：
- 管理员不能禁用/删除/降级自己
- 至少保留一个 active 的 admin 用户
- 删除用户是高危操作，需二次确认（前端实现）

### 5.3 目录结构

```
server/
├── services/
│   └── user.service.ts          # 用户 CRUD + 认证逻辑
├── db/repositories/
│   └── user.repo.ts             # User 查询封装
├── middleware/
│   └── auth.ts                  # withAuth / withAdmin（JWT 校验）
└── validators/
    └── user.ts                  # Zod schemas（login / register / update）

app/api/
├── auth/
│   ├── login/route.ts           # POST 登录
│   ├── register/route.ts        # POST 注册
│   ├── logout/route.ts          # POST 登出
│   └── me/route.ts              # GET 当前用户
├── users/me/
│   ├── route.ts                 # PUT 更新个人资料
│   ├── password/route.ts        # PUT 修改密码
│   └── usage/route.ts           # GET Token 用量（已实现）
└── admin/users/
    ├── route.ts                 # GET 用户列表
    └── [id]/route.ts            # PUT 更新用户 / DELETE 删除用户
```

### 5.4 权限控制

通过 `withAuth` / `withAdmin` 两个高阶函数包裹 Route Handler：

- `withAuth`：从 Cookie 解析 JWT，查 DB 校验用户存在且 active，注入 user 到 handler 参数
- `withAdmin`：在 `withAuth` 基础上检查 `user.role === "admin"`

### 5.5 权限矩阵

| 资源 | 操作 | 普通用户 | 管理员 |
|------|------|---------|--------|
| 角色卡 | 列表/搜索/详情 | ✅ | ✅ |
| 角色卡 | 导入/创建/编辑/删除 | ❌ | ✅ |
| 全局世界书 | 查看/导出 | ✅ | ✅ |
| 全局世界书 | 创建/编辑/删除 | ❌ | ✅ |
| 个人世界书 | 自己的 CRUD | ✅ | ✅ |
| 聊天会话 | 自己的 CRUD | ✅ | ✅ |
| 用户信息/用量 | 自己的 | ✅ | ✅ |
| 用户管理 | 列表/禁用/角色/删除 | ❌ | ✅ |
| LLM Provider | 列表/创建/编辑/删除/测试 | ❌ | ✅ |

世界书 API 的权限逻辑：查询时自动返回「自己的个人世界书 + 所有全局世界书」；写操作时检查 scope 和归属关系。

---

## 6. 核心业务逻辑

### 6.1 聊天消息处理（核心链路）

这是系统最关键的流程。v0.3 对聊天模块进行了模块化重构，拆分为四层：

```
Route Handler (app/api/chat/...)
    ↓ 调用
chat-stream.service.ts  — 编排层：Prompt 构建 + 世界书注入 + 上下文裁剪
    ↓ 调用
sse-stream.service.ts   — 通用 SSE 流引擎：LLM 调用 → chunk 转发 → 回调
    ↓ 调用
llm-client.service.ts   — HTTP 层：封装 LLM Service API 调用
```

**会话/消息 CRUD** 独立在 `session.service.ts` 和 `message.service.ts` 中，不再与流式逻辑混合。两者通过 Repository 层 (`session.repo.ts` / `message.repo.ts`) 访问数据库，Repository 封装了常用查询和归属校验（如 `findOwnedSession` 会同时校验 session 存在性和 userId 归属）。

用户发送消息后，经过以下步骤：

```
[1] 参数校验 + 权限验证（session 归属当前用户）  — Route Handler + Repository
 ↓
[2] 保存用户消息到 DB，计算 tokenCount           — chat-stream.service
 ↓
[3] Prompt 构建（详见 6.2）                       — chat-stream.service → prompt-builder
    收集：系统提示词 + 角色设定 + 世界书词条 + 历史消息 + 当前输入
 ↓
[4] 上下文窗口检查（详见 6.3）                    — chat-stream.service → context-manager
    超限 → 滑动窗口压缩 + 世界书优先级裁剪
 ↓
[5] 调用 LLM Service：POST /v1/chat/completions   — sse-stream → llm-client
    stream=true, session_id=当前会话 ID（详见 6.7）
 ↓
[6] SSE 流式转发（详见 6.5）                      — sse-stream.service
    读取上游 chunk → 透传前端 → 累计 content → 监听客户端断开
 ↓
[7] 生成完毕：保存 AI 回复 → 记录 TokenUsage → 更新 session  — chat-stream onComplete 回调
```

**停止生成**：前端断开 SSE 连接即可。后端监听 `req.signal` 的 abort 事件，调用 `AbortController.abort()` 取消上游请求。LLM Service 检测到连接断开后自动停止推理（参见 llm-service-api.md §6）。已生成的部分内容保存到 DB。**无需单独的 stop API**——直接复用 HTTP 连接断开机制。

> 注意：原方案中的 `POST /api/chat/sessions/[id]/stop` 可以移除，改为前端直接 `abortController.abort()` 断开 SSE 连接。

**重新生成**：删除最后一条 assistant 消息，用最后一条 user 消息重新触发上述完整流程，返回新的 SSE 流。

### 6.2 Prompt 构建

`PromptBuilderService` 将角色卡、世界书、历史消息组装为 LLM Service 要求的 `messages[]` 数组。构建顺序：

```
messages[0] = { role: "system", content: 拼接以下内容 }
  ├── 【v0.4 新增】角色卡 preset（预设，最前面）
  ├── [BEFORE_SYSTEM 世界书词条]
  ├── 角色卡 systemPrompt
  ├── 角色设定（description + scenario）
  ├── [AFTER_SYSTEM 世界书词条]
  └── 示例对话（如有）

messages[1] = { role: "assistant", content: 角色 firstMessage }  // 如有

messages[2..N-1] = 历史消息（按时间顺序）

messages[N] = { role: "user", content: [BEFORE_USER 世界书词条] + 当前输入 }
```

> **v0.4 变更**：
> - `preset` 位于 system message 的最前面，优先级最高。这确保预设中的全局规则（如输出语言、角色扮演规范）在所有其他内容之前被 LLM 读取。
> - `description`（原 `personality`）用于角色设定部分（给 AI 的角色定义），`personality` 改为仅供前端展示（给用户看的角色介绍）。详见 v0.3→v0.4 字段语义修正。

### 6.3 世界书注入

**词条来源**：角色卡关联的全局世界书 + 用户在会话中启用的个人世界书。两类世界书的词条统一处理。

**匹配算法**：
1. 构建扫描窗口：当前用户输入 + 最近 5 轮对话（可配置）
2. 遍历所有已启用词条，做关键词匹配（大小写不敏感的 `includes`）
3. 有 secondaryKeywords 时需主关键词和二级关键词同时命中
4. 命中词条按 priority 降序排序
5. Token 预算裁剪：从高优先级开始累加 tokenCount，超出预算（上下文窗口的 25%）则丢弃低优先级词条

### 6.4 上下文窗口管理

`ContextManagerService` 确保 Prompt 不超过模型上下文限制。模型的 `max_context_length` 从 LLM Service `GET /v1/models` 获取。

**Token 预算分配**（以 8192 为例）：

```
总上下文: 8192
  ├── max_tokens (预留给回复): 2048
  └── 可用预算: 6144
        ├── System Prompt + 角色设定: ~500 (固定)
        ├── 世界书词条: ≤1536 (25%)
        └── 历史消息: 剩余 (~4108)
```

**压缩策略**：
- **P0 滑动窗口**：总 Prompt Token 超过可用预算 80% 时，保留 System Prompt + firstMessage + 最近 N 条消息，中间旧消息在 Prompt 构建时跳过（DB 中保留）
- **P1 世界书裁剪**：上下文紧张时按优先级裁剪低优先级词条
- **P2 摘要压缩**：通过 LLM 对被裁剪消息生成摘要作为"记忆"

**Token 计数**：用 `js-tiktoken` 做预估，以 LLM Service 返回的 `usage` 做精确记录和校准。

### 6.5 SSE 流式转发（sse-stream.service）

v0.3 将 SSE 流的读取、解析、转发逻辑抽取为通用引擎 `sse-stream.service.ts`，通过回调机制与业务逻辑解耦。核心函数签名：

```typescript
createLLMStream(
  messages: { role: string; content: string }[],
  modelId: string,
  callbacks: StreamCallbacks,
  opts?: { abortSignal?: AbortSignal; sessionId?: string },
): ReadableStream<Uint8Array>
```

**StreamCallbacks** 生命周期回调：
- `onBeforeStream(send)` — LLM 流开始前，如发送 `user_message` 事件
- `onComplete(send, result)` — 生成完毕，保存 assistant 消息、记录 TokenUsage
- `onError(send, error)` — 错误处理

`chat-stream.service` 中的 `sendMessageStream` 和 `regenerateStream` 均通过此引擎实现，消除了原先两处 ~160 行的重复代码。后续新增"续写"、"分支对话"等功能时，只需提供不同的回调即可。

**opts.sessionId** 会透传给 `llm-client.service.ts` 的 `streamChatCompletion()`，作为 `session_id` 字段发送给 LLM Service（详见 6.7）。

关键实现要点：
- 使用 `ReadableStream` 构建 SSE 响应，Response Header 设置 `Content-Type: text/event-stream`
- 逐行解析上游 `data: {...}\n\n` 格式，检查 `delta.content` 累计文本，检查 `finish_reason` 和 `usage`
- 监听 `req.signal` abort 事件，断开时取消上游请求
- **内联错误处理**：LLM Service 可能在流中发送 `data: {"error": {...}}`，需检测并转发给前端
- 如果客户端提前断开（停止生成），不会收到 usage 信息，completion_tokens 以 0 记录
- SSE 路由必须使用 Node.js Runtime（`export const runtime = 'nodejs'`），不能用 Edge Runtime
- Nginx 反代需设置 `proxy_buffering off` + `X-Accel-Buffering: no`

### 6.6 角色卡导入

**SillyTavern PNG 导入**：
1. 用 `png-chunks-extract` 提取所有 chunk
2. 找 tEXt chunk（keyword="chara"）
3. Base64 解码 → JSON.parse → Character Card V2 对象
4. 字段映射：`first_mes` → `firstMessage`，`system_prompt` → `systemPrompt` 等
5. PNG 图片本体保存为 avatar 和 coverImage
6. 如存在 `character_book` 字段，同时创建全局 WorldBook 并自动关联

**JSON 导入**：通过检测 `spec` 字段判断是 SillyTavern 格式还是内部格式，做相应的映射。

### 6.7 LLM Service 客户端

`llm-client.service.ts` 封装与 LLM Service（`docs/llm-service-api.md`）的所有交互：

| 方法 | 对应 API | 说明 |
|------|---------|------|
| `streamChatCompletion()` | `POST /v1/chat/completions` (stream=true) | 流式推理，返回 ReadableStream |
| `chatCompletion()` | `POST /v1/chat/completions` (stream=false) | 非流式，用于摘要压缩等内部用途 |
| `listModels()` | `GET /v1/models` | 模型列表，60s 内存缓存 |
| `getHealthStatus()` | `GET /health` | 健康检查，返回队列状态信息 |

**v0.3 新增：session_id 透传**

`streamChatCompletion()` 新增可选参数 `sessionId`，传递时会在请求体中加入 `session_id` 字段：

```typescript
streamChatCompletion(
  messages: { role: string; content: string }[],
  modelId: string,
  signal?: AbortSignal,
  sessionId?: string,    // v0.3 新增
): Promise<Response>
```

请求体示例：
```json
{
  "model": "qwen3-32b",
  "messages": [...],
  "stream": true,
  "no_think": true,
  "temperature": 0.6,
  "top_p": 0.95,
  "max_tokens": 2048,
  "session_id": "ses_001"
}
```

`session_id` 由 `chat-stream.service` → `sse-stream.service` → `llm-client.service` 一路透传。LLM Service 可利用此字段做会话级别的状态管理（如 KV Cache 复用、日志关联等）。该字段为可选，不传时 LLM Service 行为与之前一致。

**容错策略**：
- 连接超时 5s，流式读无超时
- 非流式请求 503/504 可重试 1 次，流式不重试
- LLM Service 的错误码（context_length_exceeded / queue_full / queue_timeout / model_not_found）映射为后端对应的 HTTP 错误返回前端

### 6.8 多 Provider 支持（v0.6）

当前系统通过单一 `LLM_SERVICE_URL` 环境变量连接一个推理服务。v0.6 将支持管理员在后台配置多个 LLM Provider（如本地 vLLM、云端 OpenAI、DeepSeek 等），每个 Provider 可提供多个模型，用户在创建/切换会话时选择具体模型，系统自动路由到对应 Provider。

#### 6.8.1 核心概念

```
LlmProvider（提供商配置）
  ├── name: "本地 vLLM"
  ├── baseUrl: "http://localhost:8000"
  ├── apiKey: ""  (本地不需要)
  ├── apiStyle: "openai"
  └── models[] ──► 通过 /v1/models 自动发现，或手动配置

LlmProvider（提供商配置）
  ├── name: "DeepSeek"
  ├── baseUrl: "https://api.deepseek.com"
  ├── apiKey: "sk-xxx"  (加密存储)
  ├── apiStyle: "openai"
  └── models[] ──► 手动配置 ["deepseek-chat", "deepseek-reasoner"]
```

**apiStyle**：所有 Provider 统一使用 OpenAI-Compatible API 格式（`/v1/chat/completions`、`/v1/models`）。目前主流大模型平台（OpenAI、DeepSeek、智谱、Moonshot、本地 vLLM/Ollama）均兼容此格式，无需做格式适配。

#### 6.8.2 数据库设计

新增 `LlmProvider` 表：

```prisma
model LlmProvider {
  id          String   @id @default(cuid())
  name        String                           // 显示名称，如 "本地 vLLM"、"DeepSeek"
  baseUrl     String                           // API 基础地址
  apiKey      String   @default("")            // API Key（加密存储）
  models      String   @default("[]")          // JSON: 手动配置的模型列表 [{id, name, maxContextLength}]
  autoDiscover Boolean @default(true)          // 是否自动通过 /v1/models 发现模型
  enabled     Boolean  @default(true)          // 是否启用
  priority    Int      @default(0)             // 排序优先级（越大越靠前）
  createdAt   DateTime @default(now())
  updatedAt   DateTime @updatedAt
}
```

**字段说明**：

| 字段 | 说明 |
|------|------|
| `baseUrl` | Provider 的 API 地址。本地服务填 `http://localhost:8000`，云端填 `https://api.deepseek.com` 等 |
| `apiKey` | API 密钥。本地服务通常为空，云端服务必填。存储时使用 AES-256 加密，API 返回时脱敏为 `sk-***xxx` |
| `models` | JSON 数组，手动配置的模型列表。每项包含 `{id, name, maxContextLength}`。对于不支持 `/v1/models` 的 Provider 或需要手动指定模型的场景 |
| `autoDiscover` | 为 `true` 时，系统会定期调用 Provider 的 `GET /v1/models` 自动发现可用模型。自动发现的模型与手动配置的模型合并展示 |
| `priority` | 当多个 Provider 提供同名模型时，按 priority 选择（高优先）。也用于前端模型列表排序 |

#### 6.8.3 模型聚合与路由

**模型列表聚合**（`GET /api/models`）：

```
1. 遍历所有 enabled 的 LlmProvider
2. 对 autoDiscover=true 的 Provider，调用 GET {baseUrl}/v1/models（60s 缓存）
3. 合并自动发现的模型 + 手动配置的模型
4. 每个模型带上 providerId，按 priority 排序
5. 返回给前端：[{ id, name, maxContextLength, status, providerId, providerName }]
```

**聊天路由**（发消息/重新生成时）：

```
1. 从 ChatSession 获取 modelId
2. 在聚合的模型列表中查找 modelId → 找到对应 providerId
3. 从 DB 加载该 Provider 的 baseUrl 和 apiKey
4. 用该 Provider 的配置调用 streamChatCompletion()
```

**`llm-client.service.ts` 改造**：

现有的 `streamChatCompletion()` 直接使用 `config.llmServiceUrl`，改造后接受 Provider 配置：

```typescript
// 改造前
export async function streamChatCompletion(messages, modelId, signal?, sessionId?) {
  const res = await fetch(`${config.llmServiceUrl}/v1/chat/completions`, { ... });
}

// 改造后
export async function streamChatCompletion(
  messages, modelId, signal?, sessionId?,
  provider?: { baseUrl: string; apiKey: string }  // 新增
) {
  const baseUrl = provider?.baseUrl ?? config.llmServiceUrl;
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (provider?.apiKey) {
    headers["Authorization"] = `Bearer ${provider.apiKey}`;
  }
  const res = await fetch(`${baseUrl}/v1/chat/completions`, { headers, ... });
}
```

#### 6.8.4 兼容性与迁移

- **向后兼容**：如果数据库中没有任何 LlmProvider 记录，系统回退到 `LLM_SERVICE_URL` 环境变量（即当前行为），确保无缝升级
- **默认 Provider**：首次运行数据库 seed 时，如果 `LLM_SERVICE_URL` 有值，自动创建一个默认 Provider 记录
- **API Key 安全**：API 响应中 apiKey 始终脱敏，仅在服务端内部使用明文

#### 6.8.5 管理接口

新增管理员 API（详见前端 API 文档）：

| API | 方法 | 说明 |
|-----|------|------|
| `/api/admin/providers` | GET | 获取 Provider 列表 |
| `/api/admin/providers` | POST | 新增 Provider |
| `/api/admin/providers/:id` | PUT | 更新 Provider |
| `/api/admin/providers/:id` | DELETE | 删除 Provider |
| `/api/admin/providers/:id/test` | POST | 测试 Provider 连通性 |

#### 6.8.6 目录结构新增

```
server/
├── services/
│   └── llm-client.service.ts    # 改造：支持多 Provider 路由
├── db/repositories/
│   └── provider.repo.ts         # LlmProvider 查询封装
└── validators/
    └── provider.ts              # Zod schemas

app/api/
└── admin/providers/
    ├── route.ts                 # GET 列表 / POST 创建
    └── [id]/
        ├── route.ts             # PUT 更新 / DELETE 删除
        └── test/route.ts        # POST 测试连通性
```

### 6.9 对话摘要压缩（v0.7）

当上下文窗口接近满载时，通过 LLM 对旧对话生成摘要，替代原始消息注入 Prompt，在保留关键信息的同时释放 Token 空间。

#### 6.9.1 触发条件

在 `chat-stream.service.ts` 的 `onComplete` 回调中（即保存 assistant 消息之后），异步计算**下一轮 Prompt 预估 Token**：

```
nextRoundTokens = systemTokens + worldbookTokens + historyTokens（含刚保存的 assistant 消息）
availableBudget = session.maxTokens - config.defaultMaxTokens（预留给回复）

if nextRoundTokens >= availableBudget * 0.9:
    触发压缩
```

**计算说明**：
- `systemTokens`：角色卡 preset + systemPrompt + description + scenario + exampleDialogue + 世界书词条（即 `buildPrompt` 中 `role=system` 的消息）
- `historyTokens`：当前 DB 中该 session 所有 `role=user|assistant` 的消息 tokenCount 之和
- **不含用户下一轮输入**（无法预知），因此用 90% 而非 100% 作为阈值，留出余量

> 压缩是**异步后台任务**，不阻塞当前 SSE 响应。当前轮直接返回，压缩完成后下一轮自动生效。如果用户在压缩进行中发送了新消息，新请求会等待压缩完成后再用更新后的上下文发送（详见 6.9.10）。

#### 6.9.2 压缩范围与保留策略

```
全部对话消息（按时间升序）:
[msg_1, msg_2, ..., msg_N]

压缩区域（最早 70%）:     [msg_1, ..., msg_K]     → 送入 LLM 做摘要
保留区域（最近 30%）:     [msg_K+1, ..., msg_N]   → 原样保留

K = floor(N * 0.7)

特别说明：K 向下取整到 user/assistant 对的边界，确保不会把一对对话拆开。
```

**绝对不压缩的内容**（这些由 `buildPrompt` 在每轮独立构建，不属于对话历史）：
- 角色卡 preset（预设）
- 角色卡 systemPrompt（系统提示词）
- 角色卡 description + scenario（角色设定）
- 角色卡 exampleDialogue（示例对话）
- 世界书词条（每轮动态匹配注入）
- `role=system` 的所有消息

#### 6.9.3 摘要生成

使用 LLM 非流式调用（`chatCompletion`）生成摘要：

```typescript
// llm-client.service.ts 新增非流式方法
export async function chatCompletion(
  messages: { role: string; content: string }[],
  modelId: string,
  provider?: { baseUrl: string; apiKey: string },
): Promise<{ content: string; usage?: { prompt_tokens: number; completion_tokens: number; total_tokens: number } }>
```

**摘要 Prompt 构建**：

```
messages = [
  {
    role: "system",
    content: "你是一个对话摘要助手。请对以下角色扮演对话内容进行简洁摘要，保留关键情节、角色状态变化、重要约定和情感发展。摘要应以第三人称叙述，方便后续对话继续时作为上下文参考。"
  },
  {
    role: "user",
    content: "${existingSummary ? '[先前摘要]\n' + existingSummary + '\n\n' : ''}[对话内容]\n${对话文本}"
  }
]
```

**对话文本格式化**：将 `msg_1..msg_K` 格式化为：
```
用户: xxx
角色: xxx
用户: xxx
角色: xxx
...
```

**摘要使用的模型**：使用当前 session 绑定的 modelId 及其对应 Provider。后续可支持配置专用的摘要模型（如使用更快更便宜的模型）。

#### 6.9.4 数据存储

**方案：在 ChatSession 上新增 `contextSummary` 字段**

```prisma
model ChatSession {
  // ... 现有字段 ...
  contextSummary  String?   // 对话摘要文本（压缩后生成）
}
```

**为什么不存在 ChatMessage 表**：摘要是会话级别的状态，不是一条独立的消息。多次压缩只保留一份最新摘要，存在 session 上更简洁。

**被压缩的消息处理**：
- 将被压缩的消息的 `isCompressed` 标记为 `true`（利用已有字段）
- 在 `buildFinalMessages` 加载历史时，**跳过 `isCompressed=true` 的消息**
- 被压缩的消息仍然保留在 DB 中（用户可查看完整历史），仅在 Prompt 构建时排除

#### 6.9.5 Prompt 注入

在 `prompt-builder.service.ts` 或 `chat-stream.service.ts` 的 `buildFinalMessages` 中，如果 session 存在 `contextSummary`，将其作为历史消息的前缀注入：

```
最终 Prompt 结构:
messages[0]     = { role: "system", content: 角色卡 + 世界书 + ... }   ← 不变
messages[1]     = { role: "system", content: "[对话回顾]\n" + contextSummary }  ← 新增
messages[2..M]  = 未被压缩的历史消息（isCompressed=false）
messages[M+1]   = { role: "user", content: 当前输入 }
```

> 摘要用独立的 system message 注入，而非拼接到角色卡的 system message 中，便于 Token 计算和后续管理。

#### 6.9.6 多次压缩

随着对话继续，可能再次触发压缩。此时：

1. 从 DB 加载当前 `contextSummary`（先前的摘要）
2. 加载所有 `isCompressed=false` 的历史消息，取最早 70%
3. 将 **先前摘要 + 待压缩消息** 一并送入 LLM 生成新摘要
4. 用新摘要 **替换** `contextSummary`（始终只保留一份）
5. 将新一批消息标记为 `isCompressed=true`

```
第一次压缩:
  [msg1..msg7] → 摘要 A，保留 [msg8..msg10]

对话继续 → 触发第二次压缩:
  [摘要A + msg8..msg12] → 摘要 B，保留 [msg13..msg15]

DB 状态:
  session.contextSummary = 摘要B
  msg1..msg12.isCompressed = true
  msg13..msg15.isCompressed = false
```

#### 6.9.7 实现步骤

```
文件变更清单:

1. web/prisma/schema.prisma
   - ChatSession 新增 contextSummary String?

2. web/prisma/migrations/xxx_add_context_summary/
   - 数据库迁移

3. web/src/server/services/llm-client.service.ts
   - 新增 chatCompletion() 非流式方法（摘要压缩 + 未来内部用途）

4. web/src/server/services/compression.service.ts  (新建)
   - shouldCompress(session, systemTokens, historyTokens): boolean
   - startCompression(sessionId, modelId, provider): void  — 异步触发，Promise 存入 map
   - waitForCompression(sessionId): Promise<void>  — 等待压缩完成（无压缩时立即返回）
   - doCompress(sessionId, modelId, provider): Promise<void>  — 内部实现:
     - 加载未压缩消息，按 70/30 分割
     - 构建摘要 Prompt（含已有 contextSummary）
     - 调用 chatCompletion 生成摘要（带 30s 超时）
     - 更新 session.contextSummary
     - 批量标记消息 isCompressed=true
     - finally: 从 compressingMap 中移除

5. web/src/server/services/chat-stream.service.ts
   - sendMessageStream / regenerateStream 入口: await waitForCompression(sessionId)
   - onComplete 回调中：保存 assistant 消息后，调用 shouldCompress 检查
   - 触发时调用 startCompression（不阻塞当前 SSE 响应）

6. web/src/server/services/chat-stream.service.ts (buildFinalMessages)
   - 加载历史时过滤 isCompressed=true 的消息
   - 如 session.contextSummary 存在，注入为 system message

7. web/src/server/repositories/message.repository.ts
   - findMessagesBySession 新增 isCompressed 过滤选项
   - 新增 markMessagesCompressed(ids: string[]): 批量标记

8. web/src/server/repositories/session.repository.ts
   - findOwnedSessionWithCharacter 返回 contextSummary 字段
```

#### 6.9.8 配置项

| 配置 | 默认值 | 说明 |
|------|--------|------|
| `COMPRESS_TRIGGER_THRESHOLD` | `0.9` | 预估下一轮 token 占比超过此值触发压缩 |
| `COMPRESS_RATIO` | `0.7` | 压缩最早 70% 的对话消息 |

#### 6.9.9 注意事项

- **异步执行 + 请求门控**：压缩不阻塞当前轮 SSE 响应，但会阻塞下一轮请求直到压缩完成（详见 6.9.10）。压缩内部设置 30s 超时，超时后放弃压缩，阻塞的请求继续走滑动窗口兜底
- **并发保护**：同一 session 同时只能有一个压缩任务，用 `Map<sessionId, Promise>` 管理。`startCompression` 检查 map 中是否已有该 session 的任务，有则跳过
- **摘要质量**：Prompt 要求保留关键情节、角色状态、重要约定、情感发展。摘要过短会丢失信息，过长则失去压缩意义。经验值：压缩后摘要约为原文 15-25% 的 Token 量
- **firstMessage 处理**：角色卡的 firstMessage 作为第一条 assistant 消息存在于历史中，压缩时正常包含在内（其关键信息会被摘要保留）
- **Token 计数更新**：压缩完成后需要更新 session.usedTokens 的估算值

#### 6.9.10 压缩期间的请求阻塞

压缩是异步的，但用户可能在压缩尚未完成时就发送了下一条消息。如果不处理，新请求会用**压缩前的旧上下文**构建 Prompt，导致：
- 压缩白做（旧上下文仍然超限，触发滑动窗口丢弃）
- 压缩完成后 `isCompressed` 标记与已发送的 Prompt 不一致

**解决方案：基于 Promise 的请求门控**

```typescript
// compression.service.ts

// 每个 session 的压缩任务用 Promise 表示
const compressingMap = new Map<string, Promise<void>>();

// 触发压缩时，将 Promise 存入 map
export function startCompression(sessionId: string, ...args): void {
  const task = doCompress(sessionId, ...args)
    .finally(() => compressingMap.delete(sessionId));
  compressingMap.set(sessionId, task);
}

// 等待压缩完成（如果有正在进行的）
export async function waitForCompression(sessionId: string): Promise<void> {
  const task = compressingMap.get(sessionId);
  if (task) await task;
}
```

**在聊天请求入口处等待**：

```typescript
// chat-stream.service.ts → sendMessageStream()

export async function sendMessageStream(sessionId, userId, content, abortSignal?) {
  // ① 如果该 session 正在压缩，阻塞等待完成
  await waitForCompression(sessionId);

  // ② 压缩完成（或无压缩），继续正常流程
  const session = await sessionRepo.findOwnedSessionWithCharacter(sessionId, userId);
  // ... 保存用户消息、构建 Prompt、调用 LLM ...
}
```

**时序示意**：

```
用户轮次 N:
  ├── [1] 收到 assistant 回复，保存消息
  ├── [2] 检测到需压缩 → startCompression(sessionId)
  │       压缩任务开始（异步，不阻塞 SSE 返回）
  └── [3] SSE 响应完成，返回给前端 ✓

用户轮次 N+1（压缩仍在进行中）:
  ├── [4] 用户发送新消息 → sendMessageStream()
  ├── [5] await waitForCompression(sessionId)  ← 阻塞在此
  │       ...压缩任务完成: contextSummary 已写入, 消息已标记 isCompressed...
  ├── [6] 阻塞解除，继续执行
  ├── [7] 加载历史（此时 isCompressed=true 的已过滤，contextSummary 已注入）
  └── [8] 用压缩后的上下文构建 Prompt → 调用 LLM ✓
```

**关键细节**：

- `waitForCompression` 是一个简单的 `await`，不会轮询。如果没有正在进行的压缩，直接通过（`Map.get` 返回 `undefined`）
- 用户发送消息时保存 `user message` 到 DB 应该在 `waitForCompression` **之后**，这样保存的消息不会被当前正在进行的压缩所影响
- 压缩超时保护：`doCompress` 内部设置超时（如 30s），超时后 Promise reject，`finally` 清理 map，阻塞的请求继续执行（用未压缩的上下文兜底，走滑动窗口裁剪）
- `regenerateStream` 同理，入口处也需要 `await waitForCompression(sessionId)`

**对前端的影响**：

- 前端无需感知压缩过程。请求发出后如果被阻塞，表现为 SSE 连接建立后短暂等待才开始收到数据（类似 LLM 推理延迟）
- 如果需要更好的体验，可在 `waitForCompression` 之前先发一个 SSE 事件通知前端正在整理上下文：
  ```json
  { "type": "compressing", "message": "正在整理对话上下文..." }
  ```
  前端收到后可展示一个轻量提示，但这是**可选优化**，不实现也不影响功能正确性

---

## 7. API 设计要点

### 7.1 统一响应格式

所有非 SSE 接口统一使用 `{ success: true/false, data?, error?, meta? }` 格式。SSE 接口透传 OpenAI 格式的 chunk。

### 7.2 HTTP 状态码

| 状态码 | 使用场景 |
|--------|---------|
| 200/201 | 成功 |
| 400 | 参数校验失败 |
| 401/403 | 未认证 / 无权限 |
| 404 | 资源不存在 |
| 409 | 冲突（用户名重复等） |
| 422 | 业务逻辑错误（PNG 无角色卡数据等） |
| 429 | 速率限制 |
| 503 | LLM 服务不可用 / 队列满 |

### 7.3 API 变更历史

**v0.2（相比 PRD v0.1）：**

| API | 变更 |
|-----|------|
| `POST /api/auth/logout` | 新增，清除 Cookie |
| `GET /api/auth/me` | 新增，获取当前登录用户 |
| `GET /api/characters/tags` | 新增，获取所有标签用于筛选器 |
| `PUT /api/chat/sessions/[id]` | 新增，支持切换模型、绑定个人世界书、修改标题 |
| `GET /api/worldbooks` | 变更，自动返回「个人 + 全局」，支持 `?scope=global|personal` 筛选 |
| `POST /api/worldbooks` | 变更，普通用户只能创建 `scope=personal`；管理员可选 |
| `POST /api/chat/sessions/[id]/stop` | **移除**，改为前端直接断开 SSE 连接 |

**v0.3：**

| 变更项 | 说明 |
|--------|------|
| 后端 Service 层重构 | `chat.service.ts` 拆分为 `session.service` + `message.service` + `chat-stream.service` + `sse-stream.service` 四个模块 |
| 新增 Repository 层 | `session.repo.ts` + `message.repo.ts` 封装数据库查询和归属校验，Service 不再直接调用 Prisma |
| LLM 调用新增 `session_id` | `streamChatCompletion()` 向 LLM Service 传递 `session_id`（值为 ChatSession ID），用于会话级状态管理 |

**v0.4：**

| 变更项 | 说明 |
|--------|------|
| CharacterCard 新增 `preset` 字段 | 角色卡预设文本，在 Prompt 中位于 system message 最前面。用于注入跨角色的通用规则（输出语言、角色扮演规范等）。DB 层为 `String @default("")`，API 层为可选字段 |
| `description` / `personality` 字段语义修正 | `description` = 角色定义（给 AI 用，进 prompt）；`personality` = 角色介绍（给用户看，前端展示）。prompt-builder 改为读取 `description` 而非 `personality` |
| Prompt 构建顺序调整 | system message 内容顺序变为：preset → BEFORE_SYSTEM 世界书 → systemPrompt → 角色设定(description+scenario) → AFTER_SYSTEM 世界书 → 示例对话 |

**v0.5（本次）：**

| 变更项 | 说明 |
|--------|------|
| 用户管理模块 | 新增 `user.service.ts` + `user.repo.ts`，实现完整的用户认证（JWT + bcrypt）和管理员用户管理（列表/禁用/角色/删除） |
| User 模型新增 `status` 字段 | `String @default("active")`，值为 `active` / `disabled`。禁用用户所有请求返回 403 |
| auth 中间件重写 | 从硬编码默认用户改为真实 JWT 校验 + DB 查询，校验 status |
| 新增 API 路由 | `PUT /api/users/me`（更新资料）、`PUT /api/users/me/password`（修改密码）、`GET /api/admin/users`（用户列表）、`PUT /api/admin/users/:id`（管理用户）、`DELETE /api/admin/users/:id`（删除用户） |
| 数据库 seed | 预置一个管理员账号 `admin@example.com`（密码 `admin123456`），用于首次登录 |

**v0.6：**

| 变更项 | 说明 |
|--------|------|
| 多 Provider 支持 | 新增 `LlmProvider` 数据表和管理 API，支持管理员配置多个 LLM 推理服务（本地 vLLM、云端 OpenAI/DeepSeek 等）。模型列表聚合所有 Provider，聊天时自动路由到对应 Provider |
| 角色卡接口改为公开 | `GET /api/characters`、`GET /api/characters/:id`、`GET /api/characters/tags` 改为无需登录即可访问。未登录用户可浏览角色卡，登录校验延迟到发起对话、查看个人信息等操作 |
| 新增 Provider 管理 API | `GET/POST /api/admin/providers`、`PUT/DELETE /api/admin/providers/:id`、`POST /api/admin/providers/:id/test`（管理员权限） |

**v0.7（本次）：**

| 变更项 | 说明 |
|--------|------|
| 对话摘要压缩 | 上下文窗口接近满载（≥90%）时，LLM 自动对最早 70% 对话生成摘要，替代原始消息注入 Prompt。多次压缩时将先前摘要与新对话一并重新总结，始终只保留一份摘要 |
| ChatSession 新增 `contextSummary` 字段 | 存储 LLM 生成的对话摘要，用于下一轮 Prompt 构建时注入 |
| llm-client 新增 `chatCompletion()` | 非流式 LLM 调用方法，用于摘要压缩等内部用途 |
| 压缩期间请求门控 | 基于 `Map<sessionId, Promise>` 的并发控制，压缩进行中用户发送的新消息会阻塞等待压缩完成后再用更新后的上下文发送 |
| 新增配置项 | `SUMMARY_COMPRESS_TRIGGER`（0.9）、`SUMMARY_COMPRESS_RATIO`（0.7）、`SUMMARY_COMPRESS_TIMEOUT`（30s） |

### 7.4 分页策略

- 角色卡列表：offset 分页（`page` + `pageSize`），数据量有限
- 消息历史：游标分页（基于 `createdAt` + `id`），避免新消息插入导致分页错位
- 会话列表：按 `updatedAt DESC` 排序，offset 分页

---

## 8. 世界书权限模型

这是 PRD v0.3 引入的核心变更，独立说明设计思路。

### 8.1 两类世界书

| | 全局世界书 | 个人世界书 |
|---|---|---|
| 创建者 | 管理员 | 任意用户 |
| 可见范围 | 所有用户 | 仅创建者 |
| 写权限 | 仅管理员 | 仅创建者 |
| 关联方式 | 通过 CharacterWorldBook 关联到角色卡（自动生效） | 通过 SessionWorldBook 关联到会话（用户手动启用） |
| 聊天时注入 | 角色卡关联了哪些就注入哪些 | 用户在会话中启用了哪些就注入哪些 |

### 8.2 查询过滤逻辑

```
GET /api/worldbooks 的可见范围:
  WHERE scope = 'GLOBAL'                         // 所有全局世界书
     OR (scope = 'PERSONAL' AND userId = 当前用户)  // 自己的个人世界书
```

### 8.3 聊天时世界书收集

```
需要注入的世界书 = 
  角色卡关联的全局世界书 (CharacterWorldBook)
  + 会话启用的个人世界书 (SessionWorldBook)
```

收集后统一做关键词匹配、优先级排序、Token 预算裁剪。

---

## 9. 请求排队

### Phase 1-3：内存队列

用一个简单的 FIFO 队列控制并发推理数（默认 `maxConcurrent=1`）。超出 `maxQueueSize` 时返回 503。

由于 LLM Service 自身也有队列（见 llm-service-api.md §4 `/health` 的 queue 字段），后端的排队主要用于两个目的：
1. 提前知道排队情况并通知前端
2. 避免同时向 LLM Service 发送过多请求

### Phase 4+：Redis 队列

改用 Redis List 做 FIFO 队列 + 原子计数器控制并发。支持队列位置查询和 ETA 估算。

---

## 10. 错误处理

### 10.1 统一错误体系

定义 `AppError` 基类，派生 `ValidationError`(400)、`UnauthorizedError`(401)、`ForbiddenError`(403)、`NotFoundError`(404)、`ConflictError`(409)、`ServiceUnavailableError`(503) 等。

所有 Route Handler 通过统一的 `apiHandler` wrapper 捕获异常，AppError 返回结构化错误响应，未知异常返回 500。

### 10.2 LLM Service 异常映射

| LLM Service 错误 | 后端处理 |
|------------------|---------|
| 连接失败 | 503 "推理服务不可用" |
| 400 context_length_exceeded | 触发上下文压缩后重试一次，仍失败则 400 返回前端 |
| 503 queue_full | 503 "推理服务繁忙，请稍后重试" |
| 504 queue_timeout | 504 "等待超时" |
| 流中内联 error 事件 | 保留已接收内容，向前端转发错误事件后关闭流 |
| 客户端断开 (AbortError) | 保存已生成的部分内容，completion_tokens 记 0 |

---

## 11. 分阶段实施

### Phase 1 — MVP

- 数据模型：User（硬编码默认用户）+ ChatSession + ChatMessage
- API：创建会话 / 发送消息(SSE) / 消息历史 / 会话列表
- LLM 客户端：对接 `/v1/chat/completions` 流式
- 停止生成：前端断开连接即可
- 排队：内存队列，maxConcurrent=1

### Phase 2 — 角色卡与会话管理

- 数据模型扩展：CharacterCard（含 coverImage）+ CharacterWorldBook
- 角色卡 API：列表/搜索/详情（所有用户）+ 导入/编辑/删除（管理员）
- SillyTavern PNG 解析 + JSON 导入
- Prompt 构建 V1：System Prompt + 角色设定 + 历史消息
- 重新生成 + 编辑消息 + Token 计量

### Phase 3 — 世界书与上下文管理

- 数据模型扩展：WorldBook（含 scope/userId）+ WorldBookEntry + SessionWorldBook
- 世界书 API：CRUD + 词条管理 + 导入导出（按 scope 权限控制）
- 世界书关键词匹配 + 上下文注入（全局 + 个人统一处理）
- 上下文窗口管理：Token 预算分配 + 滑动窗口压缩 + 世界书裁剪
- 模型列表 API：代理 `/v1/models`，60s 缓存
- 会话更新 API：切换模型、绑定个人世界书

### Phase 4 — 多用户与生产化

- NextAuth.js + JWT + bcrypt 完整认证
- 用户数据隔离（查询加 userId 过滤）
- PostgreSQL 迁移 + Redis 集成
- 用量统计 API：按日/周/月聚合 TokenUsage
- 速率限制：基于 Redis 的 sliding window

---

## 12. 配置项

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `DATABASE_URL` | — | 数据库连接串 |
| `JWT_SECRET` | — | JWT 签名密钥（≥32字符） |
| `LLM_SERVICE_URL` | `http://localhost:8000` | LLM Service 地址 |
| `REDIS_URL` | — | Phase 4+ 启用 |
| `DEFAULT_MODEL_ID` | `qwen3-32b` | 新会话默认模型，与 LLM Service models.json 匹配 |
| `DEFAULT_TEMPERATURE` | `0.6` | 推理默认温度，与 LLM Service 默认值对齐 |
| `DEFAULT_MAX_TOKENS` | `2048` | 默认最大生成 token 数 |
| `MAX_CONCURRENT_INFERENCES` | `1` | 后端最大并发推理数 |
| `MAX_QUEUE_SIZE` | `10` | 最大排队数 |
| `WORLDBOOK_TOKEN_BUDGET_RATIO` | `0.25` | 世界书 Token 预算占比 |
| `CONTEXT_COMPRESS_THRESHOLD` | `0.8` | 滑动窗口裁剪的上下文使用率阈值 |
| `SUMMARY_COMPRESS_TRIGGER` | `0.9` | 触发 LLM 摘要压缩的上下文使用率阈值 |
| `SUMMARY_COMPRESS_RATIO` | `0.7` | 摘要压缩时处理最早对话的比例 |
| `WORLDBOOK_SCAN_ROUNDS` | `5` | 世界书关键词匹配的扫描轮数 |
| `UPLOAD_DIR` | `./uploads` | 文件上传目录 |
| `MAX_UPLOAD_SIZE_MB` | `10` | 最大上传文件大小 |

---

## 13. 安全与性能

### 安全

| 措施 | 说明 |
|------|------|
| 密码 | bcrypt, saltRounds=12 |
| JWT | httpOnly + secure + sameSite=lax Cookie |
| 输入校验 | 所有 API 入口用 Zod 严格校验 |
| SQL 注入 | Prisma 参数化查询自动防护 |
| 文件上传 | 限制大小 + 校验 MIME + 重命名防路径穿越 |
| 数据隔离 | 查询层面强制 userId 过滤 |
| 速率限制 | 登录 5次/分，聊天 20次/分，其他 100次/分 |

### 性能

| 策略 | 说明 |
|------|------|
| 列表查询 | select 指定字段，不返回大文本（systemPrompt 等） |
| 模型列表缓存 | 内存 60s TTL |
| 角色卡/世界书详情缓存 | 内存 LRU 300s，CUD 时主动失效 |
| SSE 连接管理 | 客户端断开时及时释放上游连接（AbortController） |
| Token 用量聚合 | 建索引 `[userId, createdAt]`，量大时可用物化视图 |

---

## 14. 可观测性

使用 `pino` 做结构化日志，记录关键指标：

| 指标 | 说明 |
|------|------|
| API 请求耗时 | method, path, userId, duration, status |
| LLM 推理指标 | TTFT, 总耗时, tokens/s, promptTokens, completionTokens |
| 队列状态 | 排队深度, 等待时间 |
| SSE 连接数 | 活跃连接数 |
| 上下文压缩 | 触发次数, 裁剪消息数 |
| 世界书注入 | 匹配词条数, 注入 token 数 |
