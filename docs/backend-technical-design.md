# 聊天机器人项目 — 后端技术方案文档

> 版本：0.2 | 最后更新：2026-03-03
>
> 对应 PRD 版本：0.3 | 对应 LLM Service API 版本：0.1

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
│   │   ├── chat.service.ts          # 聊天核心：发消息、重新生成、停止
│   │   ├── prompt-builder.service.ts # Prompt 组装
│   │   ├── context-manager.service.ts # 上下文窗口管理 + 压缩
│   │   ├── worldbook-injector.service.ts # 世界书关键词匹配与注入
│   │   ├── character-import.service.ts   # PNG/JSON 角色卡解析
│   │   ├── llm-client.service.ts    # 封装 LLM Service HTTP 调用
│   │   └── ...                      # auth, worldbook, model, queue 等
│   ├── db/
│   │   ├── prisma.ts                # Prisma Client 单例
│   │   └── repositories/            # 数据访问层
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

**User**：id, username, email, passwordHash, role(USER/ADMIN), createdAt, updatedAt

**CharacterCard**：id, name, avatar?, coverImage?, description, personality, scenario, systemPrompt, firstMessage, alternateGreetings[], exampleDialogue, creatorNotes, tags[], source(MANUAL/SILLYTAVERN_PNG/JSON_IMPORT), createdBy → User

**WorldBook**：id, name, description, version, **scope(GLOBAL/PERSONAL)**, **userId → User**, totalTokenCount, createdAt, updatedAt

**WorldBookEntry**：id, worldBookId → WorldBook, keywords[], secondaryKeywords[], content, position(BEFORE_SYSTEM/AFTER_SYSTEM/BEFORE_USER), priority, enabled, tokenCount

**CharacterWorldBook**：characterId + worldBookId（角色卡关联全局世界书，N:M）

**SessionWorldBook**：sessionId + worldBookId（会话启用个人世界书，N:M）

**ChatSession**：id, userId → User, characterId → CharacterCard, modelId, title, usedTokens, maxTokens, createdAt, updatedAt

**ChatMessage**：id, sessionId → ChatSession, role(SYSTEM/USER/ASSISTANT), content, tokenCount, isCompressed, createdAt, editedAt?

**TokenUsage**：id, userId, sessionId, messageId, modelId, promptTokens, completionTokens, totalTokens, createdAt

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

### 5.1 分阶段策略

- **Phase 1-3**：硬编码一个 `role=ADMIN` 的默认用户，数据库 seed 预置。所有请求自动以该身份执行，跳过登录
- **Phase 4**：接入 NextAuth.js (Credentials Provider)。注册时 `bcrypt.hash(password, 12)`，登录后签发 JWT 写入 httpOnly Cookie（7 天过期）

### 5.2 权限控制

通过 `withAuth` / `withAdmin` 两个高阶函数包裹 Route Handler：

- `withAuth`：从 Cookie 解析 JWT，校验有效性，注入 user 到 handler 参数
- `withAdmin`：在 `withAuth` 基础上检查 `user.role === ADMIN`

### 5.3 权限矩阵

| 资源 | 操作 | 普通用户 | 管理员 |
|------|------|---------|--------|
| 角色卡 | 列表/搜索/详情 | ✅ | ✅ |
| 角色卡 | 导入/创建/编辑/删除 | ❌ | ✅ |
| 全局世界书 | 查看/导出 | ✅ | ✅ |
| 全局世界书 | 创建/编辑/删除 | ❌ | ✅ |
| 个人世界书 | 自己的 CRUD | ✅ | ✅ |
| 聊天会话 | 自己的 CRUD | ✅ | ✅ |
| 用户信息/用量 | 自己的 | ✅ | ✅ |

世界书 API 的权限逻辑：查询时自动返回「自己的个人世界书 + 所有全局世界书」；写操作时检查 scope 和归属关系。

---

## 6. 核心业务逻辑

### 6.1 聊天消息处理（核心链路）

这是系统最关键的流程。用户发送消息后，经过以下步骤：

```
[1] 参数校验 + 权限验证（session 归属当前用户）
 ↓
[2] 保存用户消息到 DB，计算 tokenCount
 ↓
[3] Prompt 构建（详见 6.2）
    收集：系统提示词 + 角色设定 + 世界书词条 + 历史消息 + 当前输入
 ↓
[4] 上下文窗口检查（详见 6.3）
    超限 → 滑动窗口压缩 + 世界书优先级裁剪
 ↓
[5] 调用 LLM Service：POST /v1/chat/completions, stream=true
 ↓
[6] SSE 流式转发（详见 6.4）
    读取上游 chunk → 透传前端 → 累计 content → 监听客户端断开
 ↓
[7] 生成完毕：保存 AI 回复 → 从最后 chunk 提取 usage → 记录 TokenUsage → 更新 session
```

**停止生成**：前端断开 SSE 连接即可。后端监听 `req.signal` 的 abort 事件，调用 `AbortController.abort()` 取消上游请求。LLM Service 检测到连接断开后自动停止推理（参见 llm-service-api.md §6）。已生成的部分内容保存到 DB。**无需单独的 stop API**——直接复用 HTTP 连接断开机制。

> 注意：原方案中的 `POST /api/chat/sessions/[id]/stop` 可以移除，改为前端直接 `abortController.abort()` 断开 SSE 连接。

**重新生成**：删除最后一条 assistant 消息，用最后一条 user 消息重新触发上述完整流程，返回新的 SSE 流。

### 6.2 Prompt 构建

`PromptBuilderService` 将角色卡、世界书、历史消息组装为 LLM Service 要求的 `messages[]` 数组。构建顺序：

```
messages[0] = { role: "system", content: 拼接以下内容 }
  ├── [BEFORE_SYSTEM 世界书词条]
  ├── 角色卡 systemPrompt
  ├── 角色设定（personality + scenario）
  ├── [AFTER_SYSTEM 世界书词条]
  └── 示例对话（如有）

messages[1] = { role: "assistant", content: 角色 firstMessage }  // 如有

messages[2..N-1] = 历史消息（按时间顺序）

messages[N] = { role: "user", content: [BEFORE_USER 世界书词条] + 当前输入 }
```

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

### 6.5 SSE 流式转发

核心思路：后端作为 SSE 代理，从 LLM Service 读取 chunk，透传给前端，同时累计完整内容。

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

`LLMClientService` 封装与 LLM Service（`docs/llm-service-api.md`）的所有交互：

| 方法 | 对应 API | 说明 |
|------|---------|------|
| `streamChatCompletion()` | `POST /v1/chat/completions` (stream=true) | 流式推理，返回 ReadableStream |
| `chatCompletion()` | `POST /v1/chat/completions` (stream=false) | 非流式，用于摘要压缩等内部用途 |
| `listModels()` | `GET /v1/models` | 模型列表，60s 内存缓存 |
| `healthCheck()` | `GET /health` | 启动时探测，连接失败不阻止启动 |

**容错策略**：
- 连接超时 5s，流式读无超时
- 非流式请求 503/504 可重试 1 次，流式不重试
- LLM Service 的错误码（context_length_exceeded / queue_full / queue_timeout / model_not_found）映射为后端对应的 HTTP 错误返回前端

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

### 7.3 相比 PRD v0.1 新增/变更的 API

| API | 变更 |
|-----|------|
| `POST /api/auth/logout` | 新增，清除 Cookie |
| `GET /api/auth/me` | 新增，获取当前登录用户 |
| `GET /api/characters/tags` | 新增，获取所有标签用于筛选器 |
| `PUT /api/chat/sessions/[id]` | 新增，支持切换模型、绑定个人世界书、修改标题 |
| `GET /api/worldbooks` | 变更，自动返回「个人 + 全局」，支持 `?scope=global|personal` 筛选 |
| `POST /api/worldbooks` | 变更，普通用户只能创建 `scope=personal`；管理员可选 |
| `POST /api/chat/sessions/[id]/stop` | **移除**，改为前端直接断开 SSE 连接 |

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
| `CONTEXT_COMPRESS_THRESHOLD` | `0.8` | 触发压缩的上下文使用率阈值 |
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
