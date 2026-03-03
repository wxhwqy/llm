# 前端 API 接口文档

> 版本：0.2 | 最后更新：2026-03-03
>
> 本文档定义前端与后台（Next.js API Routes）之间的 HTTP API 接口规范。后台再通过 OpenAI-Compatible API 与大模型服务通信（详见 `docs/llm-service-api.md`）。

---

### v0.1 → v0.2 变更摘要

| # | 变更 | 原因 |
|---|------|------|
| 1 | 角色卡列表/详情/创建/更新均增加 `coverImage` 字段 | PRD v0.3 要求列表页为大封面图布局，缺此字段无法展示 |
| 2 | 移除 `POST /api/chat/sessions/:id/stop` 接口 | LLM Service 通过连接断开检测停止推理，前端 `AbortController.abort()` 即可，无需额外 API |
| 3 | 移除 `GET /api/users/me`（与 `GET /api/auth/me` 重复） | 两个接口返回完全相同的数据，保留 `/api/auth/me` |
| 4 | 消息历史改为游标分页 | 聊天消息可能在浏览历史时被插入，offset 分页会出现重复/跳过 |
| 5 | 模型 `maxTokens` → `maxContextLength` | 避免与生成参数 `max_tokens` 混淆，与 LLM Service 字段名对齐 |
| 6 | Cookie `SameSite=Strict` → `SameSite=Lax` | Strict 会导致从外部链接跳转时丢失登录态 |
| 7 | SSE 流增加 `type: "error"` 事件 | 覆盖流式生成中途出错的场景 |

---

## 1. 通用约定

### 1.1 Base URL

```
开发环境: http://localhost:3000/api
生产环境: https://<domain>/api
```

### 1.2 认证

所有需要认证的接口通过 `httpOnly Cookie` 携带 JWT Token（Phase 4 实现）。MVP 阶段使用硬编码默认用户，无需认证。

需要管理员权限的接口以 `/api/admin/` 开头，服务端 Middleware 校验 `user.role === 'admin'`。

### 1.3 请求/响应格式

- **Content-Type**: `application/json`（文件上传除外，使用 `multipart/form-data`）
- **日期格式**: ISO 8601（`2026-03-03T14:30:00.000Z`）
- **分页**: 列表接口使用 offset 分页（`?page=1&pageSize=20`），消息历史使用游标分页（`?cursor=xxx&limit=50`）

### 1.4 统一响应结构

**成功响应**：

```json
{
  "data": { ... },
  "pagination": {               // 仅 offset 分页的列表接口
    "page": 1,
    "pageSize": 20,
    "total": 156,
    "totalPages": 8
  }
}
```

**错误响应**：

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "角色名称不能为空",
    "details": [                // 可选，表单校验时返回字段级错误
      { "field": "name", "message": "不能为空" }
    ]
  }
}
```

### 1.5 HTTP 状态码

| 状态码 | 含义 | 前端处理 |
|--------|------|----------|
| 200 | 成功 | 正常处理 |
| 201 | 创建成功 | 正常处理 |
| 400 | 请求参数错误 | 显示 error.message |
| 401 | 未认证 | 重定向到 `/login` |
| 403 | 权限不足 | Toast "权限不足" |
| 404 | 资源不存在 | 显示 404 或 Toast |
| 409 | 资源冲突 | 显示 error.message |
| 422 | 业务逻辑错误 | 显示 error.message（如 PNG 文件无角色卡数据） |
| 429 | 请求过于频繁 | Toast "请稍后再试" |
| 503 | 服务繁忙（推理队列满） | 显示排队提示 + 重试按钮 |
| 500 | 服务器内部错误 | Toast "服务器错误" + 重试按钮 |

---

## 2. 认证 (`/api/auth`)

### 2.1 登录

```
POST /api/auth/login
```

**请求体**：

```json
{
  "email": "admin@example.com",
  "password": "password123"
}
```

**响应** `200`：

```json
{
  "data": {
    "user": {
      "id": "usr_abc123",
      "username": "Admin",
      "email": "admin@example.com",
      "role": "admin",
      "createdAt": "2026-02-01T00:00:00.000Z"
    }
  }
}
```

JWT Token 通过 `Set-Cookie: auth-token=xxx; HttpOnly; Secure; SameSite=Lax` 返回。

### 2.2 注册

```
POST /api/auth/register
```

**请求体**：

```json
{
  "username": "新用户",
  "email": "user@example.com",
  "password": "password123"
}
```

**响应** `201`：结构同登录。

### 2.3 登出

```
POST /api/auth/logout
```

**响应** `200`：清除 Cookie。

### 2.4 获取当前用户

```
GET /api/auth/me
```

**响应** `200`：

```json
{
  "data": {
    "id": "usr_abc123",
    "username": "Admin",
    "email": "admin@example.com",
    "role": "admin",
    "createdAt": "2026-02-01T00:00:00.000Z"
  }
}
```

---

## 3. 角色卡 (`/api/characters`)

### 3.1 获取角色卡列表

```
GET /api/characters?search=魔法&tag=奇幻&tag=校园&page=1&pageSize=20
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `search` | string | 否 | 按名称/描述模糊搜索 |
| `tag` | string | 否 | 按标签筛选，可多选（多个 `tag` 参数取并集） |
| `page` | number | 否 | 页码，默认 1 |
| `pageSize` | number | 否 | 每页数量，默认 20 |
| `sort` | string | 否 | 排序字段，默认 `createdAt` |
| `order` | string | 否 | `asc` / `desc`，默认 `desc` |

**响应** `200`：

```json
{
  "data": [
    {
      "id": "chr_abc123",
      "name": "艾莉丝",
      "avatar": "/uploads/avatars/chr_abc123.png",
      "coverImage": "/uploads/covers/chr_abc123.png",
      "description": "来自星辰学院的天才魔法少女...",
      "tags": ["奇幻", "魔法", "校园"],
      "source": "sillytavern_png",
      "createdAt": "2026-02-15T00:00:00.000Z",
      "updatedAt": "2026-02-15T00:00:00.000Z"
    }
  ],
  "pagination": {
    "page": 1,
    "pageSize": 20,
    "total": 6,
    "totalPages": 1
  }
}
```

> 列表接口返回摘要字段，不包含 `systemPrompt`、`firstMessage`、`exampleDialogue` 等大文本字段。`coverImage` 为 `null` 时前端使用渐变占位图。

### 3.2 获取角色卡详情

```
GET /api/characters/:id
```

**响应** `200`：

```json
{
  "data": {
    "id": "chr_abc123",
    "name": "艾莉丝",
    "avatar": "/uploads/avatars/chr_abc123.png",
    "coverImage": "/uploads/covers/chr_abc123.png",
    "description": "来自星辰学院的天才魔法少女...",
    "personality": "活泼开朗、好奇心强、善良正义、偶尔冒失",
    "scenario": "你在星辰学院的图书馆偶遇了正在研究禁忌魔法的艾莉丝",
    "systemPrompt": "你是艾莉丝，一个来自星辰学院的魔法少女...",
    "firstMessage": "啊！你、你看到了什么？...",
    "alternateGreetings": [],
    "exampleDialogue": "用户：你在研究什么？\n艾莉丝：嘿嘿~...",
    "creatorNotes": "适合轻松的奇幻冒险对话",
    "worldBookIds": ["wb_001"],
    "tags": ["奇幻", "魔法", "校园"],
    "source": "sillytavern_png",
    "createdBy": "usr_admin",
    "createdAt": "2026-02-15T00:00:00.000Z",
    "updatedAt": "2026-02-15T00:00:00.000Z"
  }
}
```

> 详情页中 `systemPrompt`、`exampleDialogue`、`creatorNotes` 仅对管理员有意义（编辑页使用）。前端详情页只展示 `personality`、`scenario`、`firstMessage`。

### 3.3 创建角色卡 🔒

```
POST /api/admin/characters
Content-Type: multipart/form-data
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `name` | string | 是 | 角色名称 |
| `description` | string | 是 | 角色简述 |
| `personality` | string | 是 | 性格描述 |
| `scenario` | string | 是 | 场景设定 |
| `systemPrompt` | string | 是 | 系统提示词 |
| `firstMessage` | string | 是 | 开场白 |
| `alternateGreetings` | string (JSON) | 否 | 替代开场白，JSON 数组字符串 |
| `exampleDialogue` | string | 否 | 示例对话 |
| `creatorNotes` | string | 否 | 创作者备注 |
| `worldBookIds` | string (JSON) | 否 | 关联世界书 ID，JSON 数组字符串 |
| `tags` | string (JSON) | 否 | 标签，JSON 数组字符串 |
| `coverImageFile` | File | 否 | 封面图片文件 |

> 使用 `multipart/form-data` 以支持封面图片上传。数组类型字段以 JSON 字符串传递。

**响应** `201`：返回完整角色卡对象（结构同 3.2）。

### 3.4 导入角色卡 🔒

```
POST /api/admin/characters/import
Content-Type: multipart/form-data
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `file` | File | `.png`（SillyTavern 格式）或 `.json` 文件 |

**响应** `201`：返回完整角色卡对象。

> PNG 导入时，图片本体同时作为 `avatar` 和 `coverImage`。如果角色卡内嵌世界书（`character_book`），也会自动创建为全局世界书并关联。

**错误** `422`：

```json
{
  "error": {
    "code": "INVALID_CHARACTER_FILE",
    "message": "PNG 文件不包含角色卡数据"
  }
}
```

### 3.5 更新角色卡 🔒

```
PUT /api/admin/characters/:id
Content-Type: multipart/form-data
```

与创建相同，所有字段可选（仅发送需要更新的字段）。如需更换封面图，传 `coverImageFile`。

**响应** `200`：返回更新后的完整角色卡对象。

### 3.6 删除角色卡 🔒

```
DELETE /api/admin/characters/:id
```

**响应** `200`：

```json
{
  "data": { "deleted": true }
}
```

### 3.7 导出角色卡 🔒

```
GET /api/admin/characters/:id/export
```

**响应** `200`：`Content-Type: application/json`，`Content-Disposition: attachment; filename="角色名.json"` 的文件下载。

### 3.8 获取所有标签

```
GET /api/characters/tags
```

**响应** `200`：

```json
{
  "data": ["奇幻", "魔法", "校园", "赛博朋克", "科幻", "日常"]
}
```

---

## 4. 世界书 (`/api/worldbooks`)

### 4.1 获取世界书列表

```
GET /api/worldbooks?scope=personal&page=1&pageSize=50
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `scope` | string | 否 | `global` / `personal`，不传则返回所有可见世界书 |
| `page` | number | 否 | 页码，默认 1 |
| `pageSize` | number | 否 | 每页数量，默认 50 |

> 可见范围：当前用户的个人世界书 + 所有全局世界书。

**响应** `200`：

```json
{
  "data": [
    {
      "id": "wb_001",
      "name": "星辰学院设定集",
      "description": "包含星辰学院的历史...",
      "scope": "global",
      "userId": "usr_admin",
      "entryCount": 5,
      "totalTokenCount": 4280,
      "characterCount": 2,
      "createdAt": "2026-02-10T00:00:00.000Z",
      "updatedAt": "2026-02-10T00:00:00.000Z"
    },
    {
      "id": "wb_004",
      "name": "我的角色扮演偏好",
      "description": "个人定制的对话风格偏好...",
      "scope": "personal",
      "userId": "usr_abc123",
      "entryCount": 2,
      "totalTokenCount": 520,
      "characterCount": 0,
      "createdAt": "2026-03-01T00:00:00.000Z",
      "updatedAt": "2026-03-01T00:00:00.000Z"
    }
  ],
  "pagination": { "page": 1, "pageSize": 50, "total": 5, "totalPages": 1 }
}
```

> 列表不返回 `entries` 数组，仅返回 `entryCount`。

### 4.2 获取世界书详情

```
GET /api/worldbooks/:id
```

> 权限：仅能访问自己的个人世界书或全局世界书。

**响应** `200`：

```json
{
  "data": {
    "id": "wb_001",
    "name": "星辰学院设定集",
    "description": "包含星辰学院的历史...",
    "scope": "global",
    "userId": "usr_admin",
    "totalTokenCount": 4280,
    "characterCount": 2,
    "entries": [
      {
        "id": "entry_001",
        "keywords": ["星辰学院", "学院"],
        "secondaryKeywords": [],
        "content": "星辰学院是大陆上最负盛名的魔法学府...",
        "position": "after_system",
        "priority": 10,
        "enabled": true,
        "tokenCount": 120
      }
    ],
    "createdAt": "2026-02-10T00:00:00.000Z",
    "updatedAt": "2026-02-10T00:00:00.000Z"
  }
}
```

### 4.3 创建世界书

```
POST /api/worldbooks
```

**请求体**：

```json
{
  "name": "新世界书",
  "description": "描述",
  "scope": "personal",
  "entries": []
}
```

> 普通用户只能创建 `scope: "personal"`，传 `"global"` 返回 403。管理员可选。

**响应** `201`：返回完整世界书对象（结构同 4.2，entries 为空数组）。

### 4.4 导入世界书

```
POST /api/worldbooks/import
Content-Type: multipart/form-data
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `file` | File | `.json` 文件（兼容 SillyTavern lorebook 格式） |
| `scope` | string | 可选，管理员可指定 `global`，普通用户忽略此参数（固定为 `personal`） |

**响应** `201`：返回完整世界书对象。

### 4.5 更新世界书

```
PUT /api/worldbooks/:id
```

**请求体**（部分更新）：

```json
{
  "name": "更新后的名称",
  "description": "更新后的描述"
}
```

> 权限：个人世界书仅创建者可改；全局世界书仅管理员可改。

**响应** `200`：返回更新后的完整世界书对象。

### 4.6 删除世界书

```
DELETE /api/worldbooks/:id
```

> 权限：个人世界书仅创建者可删；全局世界书仅管理员可删。

**响应** `200`：

```json
{
  "data": { "deleted": true }
}
```

### 4.7 导出世界书

```
GET /api/worldbooks/:id/export
```

> 可见即可导出（自己的个人世界书和所有全局世界书）。

**响应** `200`：文件下载，JSON 格式。

### 4.8 新增词条

```
POST /api/worldbooks/:id/entries
```

**请求体**：

```json
{
  "keywords": ["精灵族", "精灵"],
  "secondaryKeywords": [],
  "content": "精灵族是奇幻大陆上最古老的种族...",
  "position": "after_system",
  "priority": 10,
  "enabled": true
}
```

> 需有该世界书的写权限。`tokenCount` 由后端计算返回。

**响应** `201`：

```json
{
  "data": {
    "id": "entry_new",
    "keywords": ["精灵族", "精灵"],
    "secondaryKeywords": [],
    "content": "精灵族是奇幻大陆上最古老的种族...",
    "position": "after_system",
    "priority": 10,
    "enabled": true,
    "tokenCount": 85
  }
}
```

### 4.9 更新词条

```
PUT /api/worldbooks/:id/entries/:entryId
```

**请求体**：与新增相同，所有字段可选。

**响应** `200`：返回更新后的词条对象。

### 4.10 删除词条

```
DELETE /api/worldbooks/:id/entries/:entryId
```

**响应** `200`：

```json
{
  "data": { "deleted": true }
}
```

---

## 5. 聊天 (`/api/chat`)

### 5.1 获取会话列表

```
GET /api/chat/sessions?page=1&pageSize=50
```

**响应** `200`：

```json
{
  "data": [
    {
      "id": "ses_001",
      "characterId": "chr_abc123",
      "characterName": "艾莉丝",
      "characterAvatar": "/uploads/avatars/chr_abc123.png",
      "modelId": "qwen3-32b",
      "title": "图书馆的秘密研究",
      "lastMessage": "嘿嘿~那个魔法阵的核心其实是...",
      "personalWorldBookIds": ["wb_004"],
      "contextUsage": {
        "usedTokens": 3200,
        "maxTokens": 8192
      },
      "createdAt": "2026-03-03T06:00:00.000Z",
      "updatedAt": "2026-03-03T06:30:00.000Z"
    }
  ],
  "pagination": { "page": 1, "pageSize": 50, "total": 3, "totalPages": 1 }
}
```

> 按 `updatedAt` 降序排列。`lastMessage` 为最后一条消息的文本截取（前 100 字符）。

### 5.2 创建会话

```
POST /api/chat/sessions
```

**请求体**：

```json
{
  "characterId": "chr_abc123",
  "modelId": "qwen3-32b"
}
```

> `modelId` 可选，不传则使用服务端默认模型。

**响应** `201`：

```json
{
  "data": {
    "id": "ses_new",
    "characterId": "chr_abc123",
    "characterName": "艾莉丝",
    "characterAvatar": "/uploads/avatars/chr_abc123.png",
    "modelId": "qwen3-32b",
    "title": "新会话",
    "personalWorldBookIds": [],
    "contextUsage": { "usedTokens": 0, "maxTokens": 8192 },
    "messages": [
      {
        "id": "msg_001",
        "role": "assistant",
        "content": "啊！你、你看到了什么？...",
        "tokenCount": 85,
        "isCompressed": false,
        "createdAt": "2026-03-03T06:00:00.000Z"
      }
    ],
    "createdAt": "2026-03-03T06:00:00.000Z",
    "updatedAt": "2026-03-03T06:00:00.000Z"
  }
}
```

> 创建时自动注入角色卡的 `firstMessage` 作为第一条 assistant 消息。如角色卡无 firstMessage，`messages` 为空数组。

### 5.3 更新会话

```
PUT /api/chat/sessions/:id
```

**请求体**（部分更新）：

```json
{
  "modelId": "qwen3-8b",
  "personalWorldBookIds": ["wb_004", "wb_005"],
  "title": "自定义标题"
}
```

> 用于切换模型、更新个人世界书绑定、修改标题。`personalWorldBookIds` 为全量替换（传什么就绑什么，传空数组则解除所有绑定）。

**响应** `200`：返回更新后的会话对象（结构同 5.1 中的单个会话，不含 `messages`）。

### 5.4 删除会话

```
DELETE /api/chat/sessions/:id
```

**响应** `200`：

```json
{
  "data": { "deleted": true }
}
```

### 5.5 获取消息历史

```
GET /api/chat/sessions/:id/messages?limit=50
GET /api/chat/sessions/:id/messages?cursor=msg_002&limit=50
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `cursor` | string | 否 | 游标（消息 ID），不传则从最新消息开始 |
| `limit` | number | 否 | 每次加载数量，默认 50，最大 100 |

**响应** `200`：

```json
{
  "data": [
    {
      "id": "msg_001",
      "role": "assistant",
      "content": "啊！你、你看到了什么？...",
      "tokenCount": 85,
      "isCompressed": false,
      "createdAt": "2026-03-03T06:00:00.000Z",
      "editedAt": null
    },
    {
      "id": "msg_002",
      "role": "user",
      "content": "别紧张，我不会告诉别人的。",
      "tokenCount": 32,
      "isCompressed": false,
      "createdAt": "2026-03-03T06:05:00.000Z",
      "editedAt": null
    }
  ],
  "hasMore": true,
  "nextCursor": "msg_001"
}
```

> 消息按 `createdAt` 升序排列（旧→新）。
>
> **加载逻辑**：首次不传 `cursor`，返回最近 N 条。用户向上滚动加载更多时，传 `cursor=最早一条消息的id`，返回该消息之前的更早消息。`hasMore=false` 表示已到顶。

### 5.6 发送消息（流式）

```
POST /api/chat/sessions/:id/messages
```

**请求体**：

```json
{
  "content": "你在研究什么？"
}
```

**响应**：`Content-Type: text/event-stream`（SSE 流）

```
data: {"type":"user_message","message":{"id":"msg_003","role":"user","content":"你在研究什么？","tokenCount":12,"createdAt":"2026-03-03T06:10:00.000Z"}}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{"content":"真"},"index":0}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{"content":"、"},"index":0}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{"content":"真的吗"},"index":0}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{},"finish_reason":"stop","index":0}],"usage":{"prompt_tokens":256,"completion_tokens":180,"total_tokens":436}}

data: {"type":"message_complete","message":{"id":"msg_004","role":"assistant","content":"真、真的吗？...","tokenCount":180,"createdAt":"2026-03-03T06:10:05.000Z"},"contextUsage":{"usedTokens":3636,"maxTokens":8192}}

data: [DONE]
```

**SSE 事件类型**：

| 事件 | `type` 字段 | 说明 |
|------|------------|------|
| 用户消息已保存 | `user_message` | 包含服务端生成的 `id`、`tokenCount`。前端据此更新消息的临时 ID |
| 流式 chunk | *(无 type)* | OpenAI 格式，`delta.content` 为增量文本。中间 chunk 的 `finish_reason` 为 `null` |
| 生成完毕 | *(无 type)* | `finish_reason` 为 `"stop"` 或 `"length"`，附带 `usage` |
| AI 消息已保存 | `message_complete` | 完整 AI 消息对象 + 更新后的 `contextUsage` |
| 流中错误 | `error` | `{"type":"error","error":{"code":"...","message":"..."}}` |
| 流结束 | `[DONE]` | 字面量 `[DONE]`，非 JSON |

**错误处理**：
- HTTP 非 200 响应（如 503 队列满）：返回标准 JSON 错误，非 SSE
- 流式生成中途出错：发送 `type: "error"` 事件，然后 `[DONE]`。已生成的部分内容由后端保存

**停止生成**：前端调用 `abortController.abort()` 断开 SSE 连接即可。后端检测到连接断开后自动取消上游推理请求，并保存已生成的部分内容。

### 5.7 编辑消息

```
PUT /api/chat/sessions/:id/messages/:msgId
```

**请求体**：

```json
{
  "content": "编辑后的消息内容"
}
```

> 仅允许编辑 `role: "user"` 的消息。

**响应** `200`：

```json
{
  "data": {
    "id": "msg_002",
    "role": "user",
    "content": "编辑后的消息内容",
    "tokenCount": 15,
    "isCompressed": false,
    "createdAt": "2026-03-03T06:05:00.000Z",
    "editedAt": "2026-03-03T06:15:00.000Z"
  }
}
```

### 5.8 重新生成

```
POST /api/chat/sessions/:id/regenerate
```

**请求体**（可选）：

```json
{
  "modelId": "qwen3-8b"
}
```

> 删除最后一条 AI 消息，重新生成。可选指定不同模型（仅影响本次生成，不修改会话的 modelId）。

**响应**：SSE 流，格式同 5.6（但不包含 `user_message` 事件，直接开始 chunk 流）。

停止方式与 5.6 相同：`abortController.abort()` 断开连接。

---

## 6. Token 使用统计 (`/api/users/me/usage`)

### 6.1 获取 Token 使用统计

```
GET /api/users/me/usage?period=daily&from=2026-02-20&to=2026-03-03
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `period` | string | 否 | `daily` / `weekly` / `monthly`，默认 `daily` |
| `from` | string | 否 | 起始日期（YYYY-MM-DD），默认 30 天前 |
| `to` | string | 否 | 结束日期（YYYY-MM-DD），默认今天 |

**响应** `200`：

```json
{
  "data": {
    "summary": {
      "totalPromptTokens": 194900,
      "totalCompletionTokens": 129400,
      "totalTokens": 324300,
      "totalSessions": 3,
      "totalMessages": 12
    },
    "timeline": [
      {
        "date": "2026-02-20",
        "promptTokens": 12500,
        "completionTokens": 8300,
        "totalTokens": 20800
      },
      {
        "date": "2026-02-21",
        "promptTokens": 15200,
        "completionTokens": 10100,
        "totalTokens": 25300
      }
    ]
  }
}
```

> `timeline` 按日期升序排列。`period=weekly` 时 `date` 为周一日期，`period=monthly` 时为月初日期。

---

## 7. 模型 (`/api/models`)

### 7.1 获取可用模型列表

```
GET /api/models
```

**响应** `200`：

```json
{
  "data": [
    {
      "id": "qwen3-32b",
      "name": "Qwen3 32B (FP8)",
      "maxContextLength": 8192,
      "status": "online"
    },
    {
      "id": "qwen3-8b",
      "name": "Qwen3 8B (FP8)",
      "maxContextLength": 8192,
      "status": "offline"
    }
  ]
}
```

| 字段 | 说明 |
|------|------|
| `id` | 模型标识，用于 `chat/sessions` 的 `modelId` 参数 |
| `name` | 人类可读名称，来自 LLM Service `GET /v1/models` |
| `maxContextLength` | 最大上下文窗口 (tokens)，用于前端上下文进度条计算 |
| `status` | 见下表 |

| status | 含义 | 前端处理 |
|--------|------|---------|
| `online` | 已加载到 GPU，可立即推理 | 正常可选 |
| `offline` | 已注册但未加载，首次请求需等待 30-60s | 可选，但提示"首次使用需等待加载" |
| `busy` | 队列接近满载 | 可选，但提示"当前繁忙" |

> 后端缓存此接口 60 秒。`status` 由后端综合 LLM Service 的 `/v1/models`（loaded/available 状态）和 `/health`（队列信息）推算。

---

## 8. 前端调用示例

### 8.1 流式消息处理

```typescript
async function sendMessage(sessionId: string, content: string, signal?: AbortSignal) {
  const res = await fetch(`/api/chat/sessions/${sessionId}/messages`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ content }),
    signal,
  });

  if (!res.ok) throw new ApiError(res.status, await res.json());

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const data = line.slice(6);
      if (data === '[DONE]') return;

      const event = JSON.parse(data);

      if (event.type === 'user_message') {
        onUserMessageSaved(event.message);
      } else if (event.type === 'message_complete') {
        onMessageComplete(event.message, event.contextUsage);
      } else if (event.type === 'error') {
        onError(event.error);
      } else if (event.choices?.[0]?.delta?.content) {
        onToken(event.choices[0].delta.content);
      }
    }
  }
}
```

### 8.2 停止生成

```typescript
const abortController = new AbortController();

// 发送消息时传入 signal
sendMessage(sessionId, content, abortController.signal);

// 用户点击"停止"按钮
function stopGeneration() {
  abortController.abort();
}
```

### 8.3 游标分页加载消息历史

```typescript
let nextCursor: string | undefined;

async function loadMessages(sessionId: string) {
  const params = new URLSearchParams({ limit: '50' });
  if (nextCursor) params.set('cursor', nextCursor);

  const res = await fetch(`/api/chat/sessions/${sessionId}/messages?${params}`);
  const { data, hasMore, nextCursor: cursor } = await res.json();

  nextCursor = hasMore ? cursor : undefined;
  prependMessages(data); // 插入到消息列表顶部
}
```

---

## 9. 接口总览

| 接口 | 方法 | 说明 | 认证 |
|------|------|------|------|
| `/api/auth/login` | POST | 登录 | 否 |
| `/api/auth/register` | POST | 注册 | 否 |
| `/api/auth/logout` | POST | 登出 | 是 |
| `/api/auth/me` | GET | 获取当前用户 | 是 |
| `/api/characters` | GET | 角色卡列表/搜索 | 是 |
| `/api/characters/:id` | GET | 角色卡详情 | 是 |
| `/api/characters/tags` | GET | 所有标签 | 是 |
| `/api/admin/characters` | POST | 创建角色卡 | 🔒 管理员 |
| `/api/admin/characters/import` | POST | 导入角色卡 | 🔒 管理员 |
| `/api/admin/characters/:id` | PUT | 更新角色卡 | 🔒 管理员 |
| `/api/admin/characters/:id` | DELETE | 删除角色卡 | 🔒 管理员 |
| `/api/admin/characters/:id/export` | GET | 导出角色卡 | 🔒 管理员 |
| `/api/worldbooks` | GET | 世界书列表 | 是 |
| `/api/worldbooks` | POST | 创建世界书 | 是 |
| `/api/worldbooks/import` | POST | 导入世界书 | 是 |
| `/api/worldbooks/:id` | GET | 世界书详情 | 是 |
| `/api/worldbooks/:id` | PUT | 更新世界书 | 是 (写权限) |
| `/api/worldbooks/:id` | DELETE | 删除世界书 | 是 (写权限) |
| `/api/worldbooks/:id/export` | GET | 导出世界书 | 是 |
| `/api/worldbooks/:id/entries` | POST | 新增词条 | 是 (写权限) |
| `/api/worldbooks/:id/entries/:entryId` | PUT | 更新词条 | 是 (写权限) |
| `/api/worldbooks/:id/entries/:entryId` | DELETE | 删除词条 | 是 (写权限) |
| `/api/chat/sessions` | GET | 会话列表 | 是 |
| `/api/chat/sessions` | POST | 创建会话 | 是 |
| `/api/chat/sessions/:id` | PUT | 更新会话 | 是 |
| `/api/chat/sessions/:id` | DELETE | 删除会话 | 是 |
| `/api/chat/sessions/:id/messages` | GET | 消息历史（游标分页） | 是 |
| `/api/chat/sessions/:id/messages` | POST | 发送消息（SSE 流） | 是 |
| `/api/chat/sessions/:id/messages/:msgId` | PUT | 编辑消息 | 是 |
| `/api/chat/sessions/:id/regenerate` | POST | 重新生成（SSE 流） | 是 |
| `/api/users/me/usage` | GET | Token 使用统计 | 是 |
| `/api/models` | GET | 可用模型列表 | 是 |
