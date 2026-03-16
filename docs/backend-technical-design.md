# 聊天机器人项目 — 后端技术方案文档

> 版本：0.8 | 最后更新：2026-03-16
>
> 对应 PRD 版本：0.3 | 对应 LLM Service API 版本：0.2

---

## 1. 系统定位

本文档覆盖 **Next.js API Routes 中间层** 的技术方案。向上为前端提供业务 API，向下对接大模型推理服务。

```
Frontend (Next.js)  ←─ HTTP/SSE ─→  Backend (本文档)  ←─ OpenAI API ─→  LLM Service / 云端 Provider
                                          │
                                    SQLite → PostgreSQL
```

**核心职责**：认证授权、角色卡/世界书管理、Prompt 构建与上下文管理、SSE 流式代理、Token 计量、多 Provider 路由。

---

## 2. 技术选型

| 组件 | 选型 | 理由 |
|------|------|------|
| 框架 | Next.js 14+ (App Router) | 前后端同栈，API Routes 内置 |
| 语言 | TypeScript | 类型安全，配合 Prisma 自动生成类型 |
| ORM | Prisma | 类型安全、迁移管理、SQLite/PG 无缝切换 |
| 数据库 | SQLite → PostgreSQL | 开发零配置，生产上 JSONB + 并发 |
| 认证 | JWT + bcrypt | httpOnly Cookie，7 天过期 |
| 数据校验 | Zod | 运行时类型校验 |

---

## 3. 分层架构

采用 **Route Handler → Service → Repository** 三层架构：

```
web/src/
├── app/api/              # Route Handlers
│   ├── auth/             # login / register / logout / me
│   ├── characters/       # 列表、详情、标签（公开）
│   ├── admin/            # 角色卡管理、Provider 管理、用户管理（🔒管理员）
│   ├── worldbooks/       # CRUD + 词条 + 导入导出
│   ├── chat/sessions/    # 会话 CRUD + 消息(SSE) + 重新生成
│   ├── users/me/         # 用户信息 + Token 用量
│   └── models/           # 聚合所有 Provider 的模型列表
├── server/
│   ├── services/         # 核心业务逻辑
│   ├── repositories/     # Prisma 查询封装
│   ├── middleware/        # withAuth, withAdmin
│   ├── validators/        # Zod schemas
│   └── lib/              # errors, config, response 工具
└── prisma/schema.prisma
```

---

## 4. 数据库设计

### 4.1 ER 关系

```
User ──1:N──► ChatSession ──1:N──► ChatMessage
  │                │
  │                ├──1:N──► TokenUsage
  │                └──N:M──► WorldBook (SessionWorldBook)
  │
  ├──1:N──► CharacterCard ──N:M──► WorldBook (CharacterWorldBook)
  └──1:N──► WorldBook

LlmProvider（独立表，管理员配置）
```

### 4.2 核心表

| 表 | 关键字段 |
|----|---------|
| **User** | username, email, passwordHash, role(user/admin), status(active/disabled) |
| **CharacterCard** | name, avatar, coverImage, description, personality, preset, scenario, systemPrompt, firstMessage, alternateGreetings[], exampleDialogue, tags[], source |
| **WorldBook** | name, scope(global/personal), userId, totalTokenCount |
| **WorldBookEntry** | keywords[], secondaryKeywords[], content, position, priority, enabled, tokenCount |
| **ChatSession** | userId, characterId, modelId, title, usedTokens, maxTokens, **temperature?**, **topP?**, **topK?**, **contextSummary?** |
| **ChatMessage** | sessionId, role, content, tokenCount, **isCompressed** |
| **TokenUsage** | userId, sessionId, messageId, modelId, promptTokens, completionTokens |
| **LlmProvider** | name, baseUrl, apiKey, models(JSON), autoDiscover, enabled, priority |

### 4.3 关键索引

| 表 | 索引 | 用途 |
|---|---|---|
| ChatSession | `[userId, updatedAt]` | 会话列表 |
| ChatMessage | `[sessionId, createdAt]` | 游标分页 |
| TokenUsage | `[userId, createdAt]` | 用量聚合 |

---

## 5. 认证与授权

### 5.1 认证

- **注册**：Zod 校验 → 唯一性检查 → bcrypt(12) 哈希 → 签发 JWT Cookie
- **登录**：查用户 → 检查 status → bcrypt 校验 → 签发 JWT（7天）→ httpOnly Cookie
- **中间件**：`withAuth` 解析 JWT + 查 DB；`withAdmin` 额外检查 role
- **seed 默认账号**：`admin@example.com` / `admin123456`

### 5.2 权限矩阵

| 资源 | 操作 | 普通用户 | 管理员 |
|------|------|---------|--------|
| 角色卡 | 列表/搜索/详情 | ✅（公开） | ✅ |
| 角色卡 | 导入/创建/编辑/删除 | ❌ | ✅ |
| 全局世界书 | 查看/导出 | ✅ | ✅ |
| 全局世界书 | 创建/编辑/删除 | ❌ | ✅ |
| 个人世界书 | 自己的 CRUD | ✅ | ✅ |
| 聊天会话 | 自己的 CRUD | ✅ | ✅ |
| 用户管理 | 列表/禁用/角色/删除 | ❌ | ✅ |
| LLM Provider | CRUD + 测试 | ❌ | ✅ |

---

## 6. 核心业务逻辑

### 6.1 聊天消息处理

```
Route Handler → chat-stream.service → sse-stream.service → llm-client.service
```

流程：校验权限 → 保存用户消息 → 构建 Prompt（角色卡+世界书+历史） → 上下文裁剪 → 调用 LLM（SSE 流式） → 转发 chunk → 保存 assistant 消息 + TokenUsage → 检查是否需要摘要压缩

**停止生成**：前端断开 SSE 连接，后端 AbortController 取消上游请求，已生成内容保存。

**重新生成**：删除最后 assistant 消息，用最后 user 消息重走完整流程。

### 6.2 Prompt 构建

```
messages[0] = { role: "system", content: preset + 世界书(BEFORE_SYSTEM) + systemPrompt + 角色设定 + 世界书(AFTER_SYSTEM) + 示例对话 }
messages[1] = { role: "assistant", content: firstMessage }
messages[2..N-1] = 历史消息
messages[N] = { role: "user", content: 世界书(BEFORE_USER) + 当前输入 }
```

如存在 `contextSummary`，在 system 消息之后注入 `{ role: "system", content: "[对话回顾]\n" + summary }`。

### 6.3 世界书注入

- **来源**：角色卡关联的全局世界书 + 会话启用的个人世界书
- **匹配**：扫描当前输入 + 最近 5 轮对话，关键词匹配（支持二级关键词）
- **裁剪**：按 priority 排序，累计 tokenCount 超出预算（上下文 25%）则丢弃

### 6.4 上下文窗口管理

Token 预算 = maxContextLength - maxTokens(回复预留)

**三级压缩策略**：
1. **滑动窗口**（80% 阈值）：跳过旧消息，保留 system + firstMessage + 最近 N 条
2. **世界书裁剪**：按优先级丢弃低优先级词条
3. **LLM 摘要压缩**（90% 阈值，详见 6.9）：对最早 70% 对话生成摘要

### 6.5 SSE 流式引擎

`createLLMStream(messages, modelId, callbacks, opts)` — 通用 SSE 引擎：
- 回调：`onBeforeStream`、`onComplete`、`onError`
- 透传 `session_id` 给 LLM Service（KV Cache 复用）
- 监听 abort 信号，客户端断开时取消上游请求

### 6.6 角色卡导入

**PNG 导入**：解析 tEXt chunk(keyword="chara") → Base64 解码 → 字段映射 → 保存图片 → 创建角色卡

**character_book 世界书导入**：如 PNG/JSON 包含 `character_book` 字段，自动创建全局 WorldBook + Entries 并通过 CharacterWorldBook 关联。ST 的 entries 支持对象和数组两种格式。

字段映射：`keys`→`keywords`，`secondary_keys`→`secondaryKeywords`，`insertion_order`→`priority`，`position`(0→before_system, 1→after_system)

### 6.7 LLM Service 客户端

| 方法 | 说明 |
|------|------|
| `streamChatCompletion()` | 流式推理，支持 provider 路由 |
| `chatCompletion()` | 非流式，用于摘要压缩等内部用途 |
| `listModels()` | 模型列表，60s 缓存 |
| `getHealthStatus()` | 队列状态 |

所有方法自动处理 `baseUrl` 的 `/v1` 后缀归一化（`normalizeBaseUrl`），兼容 `http://localhost:8000` 和 `https://openrouter.ai/api/v1` 两种风格。

### 6.8 多 Provider 支持

管理员可配置多个 LLM Provider（本地 vLLM、OpenRouter、DeepSeek 等），统一使用 OpenAI-Compatible API 格式。

- **模型聚合**：`GET /api/models` 遍历所有 enabled 的 Provider，合并自动发现 + 手动配置的模型
- **聊天路由**：`resolveProviderForModel(modelId)` 查找对应 Provider 的 baseUrl + apiKey
- **兼容**：无 Provider 记录时回退到 `LLM_SERVICE_URL` 环境变量
- **管理 API**：`/api/admin/providers` 的 GET/POST/PUT/DELETE/test

### 6.9 采样参数

用户可以在会话级别自定义 LLM 采样参数。未设置时使用全局默认值。

| 参数 | 字段 | 范围 | 默认值 | 说明 |
|------|------|------|--------|------|
| 温度 | `temperature` | `[0.0, 2.0]` | 环境变量 `DEFAULT_TEMPERATURE`(0.6) | 控制随机性 |
| Top-P | `topP` | `[0.0, 1.0]` | 0.95 | 核采样阈值 |
| Top-K | `topK` | `[0, ∞)` | 不传（由 LLM Service 决定） | 候选 token 数量 |

**存储**：`ChatSession.temperature`/`topP`/`topK`，nullable，null 表示使用默认值。

**设置 API**：`PUT /api/chat/sessions/:id`

```json
{ "temperature": 0.8, "topP": 0.9, "topK": 50 }
```

**响应**：会话对象中返回 `samplingParams: { temperature, topP, topK }`，null 表示使用默认值。

**传递链路**：Session DB → `chat-stream.service`（读取会话参数） → `sse-stream.service` → `llm-client.service`（注入到 OpenAI API 请求体）

### 6.10 对话摘要压缩

上下文接近满载时，LLM 自动对旧对话生成摘要，替代原始消息注入 Prompt。

**触发**：assistant 回复后，预估下一轮 token ≥ 可用预算 90% → 异步触发压缩（不阻塞当前响应）

**压缩过程**：
1. 取未压缩消息的最早 70%（对齐到 user/assistant 对边界）
2. 连同已有 contextSummary 送 LLM 生成新摘要
3. 更新 `session.contextSummary`，标记消息 `isCompressed=true`
4. 始终只保留一份摘要，多次压缩时滚动合并

**请求门控**：压缩进行中用户发新消息 → `await waitForCompression(sessionId)` 阻塞等待 → 用压缩后的上下文发送

**不压缩**：角色卡内容（preset/systemPrompt/description/scenario/exampleDialogue）、世界书词条、system 消息 — 这些每轮独立构建。

---

## 7. API 设计

### 7.1 响应格式

非 SSE：`{ success, data?, error?, meta? }`。SSE：透传 OpenAI chunk 格式。

### 7.2 分页

- 角色卡/会话列表：offset 分页（`page` + `pageSize`）
- 消息历史：游标分页（基于 `createdAt`）

### 7.3 变更历史

| 版本 | 主要变更 |
|------|---------|
| v0.2 | 新增 logout/me/tags API，世界书 scope 权限，移除 stop API |
| v0.3 | Service 层重构（chat-stream + sse-stream），Repository 层，session_id 透传 |
| v0.4 | CharacterCard 新增 preset 字段，description/personality 语义修正 |
| v0.5 | JWT+bcrypt 认证，用户管理（CRUD+禁用），admin 路由 |
| v0.6 | 多 Provider 支持，角色卡接口公开化，Provider 管理 API |
| v0.7 | 对话摘要压缩（LLM 生成摘要+请求门控），character_book 世界书导入 |
| v0.8 | 会话级采样参数（temperature/topP/topK），用户可自定义 |

---

## 8. 世界书权限

| | 全局世界书 | 个人世界书 |
|---|---|---|
| 创建者 | 管理员 | 任意用户 |
| 可见范围 | 所有用户 | 仅创建者 |
| 关联方式 | CharacterWorldBook → 角色卡 | SessionWorldBook → 会话 |
| 聊天注入 | 角色卡关联即生效 | 用户在会话中手动启用 |

---

## 9. 配置项

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `LLM_SERVICE_URL` | `http://localhost:8000` | 默认 LLM Service 地址（无 Provider 时使用） |
| `JWT_SECRET` | dev-secret | JWT 签名密钥（≥32字符） |
| `DEFAULT_MODEL_ID` | `qwen3-32b` | 新会话默认模型 |
| `DEFAULT_TEMPERATURE` | `0.6` | 推理温度 |
| `DEFAULT_MAX_TOKENS` | `2048` | 最大生成 token |
| `WORLDBOOK_TOKEN_BUDGET_RATIO` | `0.25` | 世界书预算占比 |
| `CONTEXT_COMPRESS_THRESHOLD` | `0.8` | 滑动窗口裁剪阈值 |
| `SUMMARY_COMPRESS_TRIGGER` | `0.9` | LLM 摘要压缩触发阈值 |
| `SUMMARY_COMPRESS_RATIO` | `0.7` | 压缩最早对话比例 |
| `SUMMARY_COMPRESS_TIMEOUT` | `30000` | 压缩超时（ms） |

---

## 10. 安全与性能

**安全**：bcrypt(12) 密码哈希、httpOnly JWT Cookie、Zod 输入校验、Prisma 参数化防注入、文件上传限制、userId 数据隔离

**性能**：列表 select 指定字段、模型列表 60s 缓存、SSE 连接 abort 管理、Token 用量索引优化
