# 前端 API 接口文档

> 版本：0.6 | 最后更新：2026-03-14
>
> 本文档定义前端与后台（Next.js API Routes）之间的 HTTP API 接口规范。后台再通过 OpenAI-Compatible API 与大模型服务通信（详见 `docs/llm-service-api.md`）。

---

### v0.5 → v0.6 变更摘要

| # | 变更 | 影响范围 |
|---|------|----------|
| 1 | 角色卡列表/详情/标签接口改为公开（无需登录） | `GET /api/characters`、`GET /api/characters/:id`、`GET /api/characters/tags` 不再需要认证。前端首页无需登录即可显示角色卡列表。登录校验延迟到发起对话、查看"我的"等操作 |
| 2 | 新增 LLM Provider 管理 API（管理员） | 支持管理员在后台配置多个大模型推理服务（本地 vLLM、云端 OpenAI/DeepSeek 等）。新增 `GET/POST /api/admin/providers`、`PUT/DELETE /api/admin/providers/:id`、`POST /api/admin/providers/:id/test` |
| 3 | 模型列表接口响应新增字段 | `GET /api/models` 返回的模型对象新增 `providerId` 和 `providerName` 字段，标识模型来源 |

### v0.4 → v0.5 变更摘要

| # | 变更 | 影响范围 |
|---|------|----------|
| 1 | 认证接口真实实现 | 登录/注册/登出从桩代码改为真实 JWT + bcrypt 实现。前端请求/响应格式不变，但现在会真正校验密码和返回错误 |
| 2 | 新增 `PUT /api/users/me` | 用户修改自己的用户名/邮箱。前端个人资料页需新增编辑功能 |
| 3 | 新增 `PUT /api/users/me/password` | 用户修改密码（需验证旧密码）。前端个人资料页需新增修改密码表单 |
| 4 | 新增管理员用户管理 API | `GET /api/admin/users`（用户列表）、`PUT /api/admin/users/:id`（修改角色/状态）、`DELETE /api/admin/users/:id`（删除用户）。前端需新增管理员用户管理页面 |
| 5 | User 对象新增 `status` 字段 | 值为 `active` / `disabled`。被禁用的用户所有接口返回 403。影响 `GET /api/auth/me` 和管理员用户列表的响应结构 |
| 6 | 登录接口新增错误码 | `ACCOUNT_DISABLED`（账号被禁用，403）、`INVALID_CREDENTIALS`（密码错误，401） |

### v0.3 → v0.4 变更摘要

| # | 变更 | 影响范围 |
|---|------|----------|
| 1 | 角色卡新增 `preset` 字段 | 角色卡详情、创建、更新、导入接口均新增此字段。列表接口不返回（大文本字段）。前端编辑页需新增 preset 输入区域 |
| 2 | `description` / `personality` 字段语义明确化 | `description` = 角色定义（给 AI 用，进 prompt，编辑页维护）；`personality` = 角色介绍（给用户看，列表/详情页展示）。API 响应字段名不变，仅语义和用途调整 |
| 3 | Prompt 构建顺序调整 | preset 位于 system message 最前面，优先级最高。对前端透明 |

### v0.2 → v0.3 变更摘要

| # | 变更 | 原因 |
|---|------|------|
| 1 | 后端 Service 层重构（chat.service 拆分为 session / message / chat-stream / sse-stream 四模块，新增 Repository 层） | 原 chat.service.ts 527 行，SSE 流逻辑重复，无法扩展。前端 API 接口和响应格式不变 |
| 2 | 后端向 LLM Service 推理请求新增 `session_id` 字段（值为当前 ChatSession ID） | LLM Service 新增会话级状态管理能力（如 KV Cache 复用）。此变更对前端透明，无需修改前端代码 |

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

大部分接口通过 `httpOnly Cookie` 携带 JWT Token 进行认证。

**公开接口**（无需登录）：角色卡列表、角色卡详情、标签列表、登录、注册。未登录用户可浏览角色卡，但发起对话、查看个人信息等操作需要登录。

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
      "status": "active",
      "createdAt": "2026-02-01T00:00:00.000Z"
    }
  }
}
```

JWT Token 通过 `Set-Cookie: auth-token=xxx; HttpOnly; Secure; SameSite=Lax; Max-Age=604800; Path=/` 返回。

**错误响应**：

| 状态码 | code | 说明 |
|--------|------|------|
| 401 | `INVALID_CREDENTIALS` | 邮箱不存在或密码错误 |
| 403 | `ACCOUNT_DISABLED` | 账号已被管理员禁用 |

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

| 字段 | 校验规则 |
|------|----------|
| `username` | 2-20 字符 |
| `email` | 合法邮箱格式 |
| `password` | ≥8 字符 |

**响应** `201`：结构同登录（自动登录，Set-Cookie）。

**错误响应**：

| 状态码 | code | 说明 |
|--------|------|------|
| 409 | `CONFLICT` | 用户名或邮箱已被注册 |

### 2.3 登出

```
POST /api/auth/logout
```

**响应** `200`：清除 `auth-token` Cookie。

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
    "status": "active",
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
    "preset": "你是一个角色扮演 AI，请始终保持角色设定，用中文回复...",
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

**字段语义说明**：

| 字段 | 用途 | 前端使用场景 |
|------|------|-------------|
| `description` | 角色定义，给 AI 用（进 prompt 的角色设定段） | 管理员编辑页维护 |
| `personality` | 角色介绍，给用户看 | 角色列表页、详情页展示 |
| `preset` | 预设文本，进 prompt 最前面（v0.4 新增） | 管理员编辑页维护，位于编辑表单最顶部 |
| `systemPrompt` | 角色专属系统提示词 | 管理员编辑页维护 |
| `scenario` | 场景设定，进 prompt | 详情页展示 + 管理员编辑 |

> `preset`、`systemPrompt`、`description`、`exampleDialogue`、`creatorNotes` 仅对管理员有意义（编辑页使用）。前端详情页只展示 `personality`、`scenario`、`firstMessage`。

### 3.3 创建角色卡 🔒

```
POST /api/admin/characters
Content-Type: multipart/form-data
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `name` | string | 是 | 角色名称 |
| `description` | string | 是 | 角色定义（给 AI 用） |
| `personality` | string | 是 | 角色介绍（给用户看） |
| `preset` | string | 否 | 预设文本（进 prompt 最前面，v0.4 新增） |
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

### 6.2 更新个人资料

```
PUT /api/users/me
```

**请求体**（部分更新）：

```json
{
  "username": "新用户名",
  "email": "newemail@example.com"
}
```

| 字段 | 校验规则 |
|------|----------|
| `username` | 2-20 字符，唯一 |
| `email` | 合法邮箱格式，唯一 |

**响应** `200`：

```json
{
  "data": {
    "id": "usr_abc123",
    "username": "新用户名",
    "email": "newemail@example.com",
    "role": "user",
    "status": "active",
    "createdAt": "2026-02-01T00:00:00.000Z"
  }
}
```

**错误** `409`：用户名或邮箱已被占用。

### 6.3 修改密码

```
PUT /api/users/me/password
```

**请求体**：

```json
{
  "oldPassword": "当前密码",
  "newPassword": "新密码至少8位"
}
```

**响应** `200`：

```json
{
  "data": { "success": true }
}
```

**错误响应**：

| 状态码 | code | 说明 |
|--------|------|------|
| 401 | `INVALID_CREDENTIALS` | 旧密码错误 |
| 400 | `VALIDATION_ERROR` | 新密码不符合要求（≥8 字符） |

---

## 7. 用户管理 (`/api/admin/users`) 🔒

> 以下接口均需要管理员权限。

### 7.1 获取用户列表

```
GET /api/admin/users?search=test&role=user&status=active&page=1&pageSize=20
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `search` | string | 否 | 按用户名/邮箱模糊搜索 |
| `role` | string | 否 | 按角色筛选：`admin` / `user` |
| `status` | string | 否 | 按状态筛选：`active` / `disabled` |
| `page` | number | 否 | 页码，默认 1 |
| `pageSize` | number | 否 | 每页数量，默认 20 |

**响应** `200`：

```json
{
  "data": [
    {
      "id": "usr_abc123",
      "username": "Admin",
      "email": "admin@example.com",
      "role": "admin",
      "status": "active",
      "sessionCount": 15,
      "totalTokens": 324300,
      "createdAt": "2026-02-01T00:00:00.000Z",
      "updatedAt": "2026-02-01T00:00:00.000Z"
    },
    {
      "id": "usr_def456",
      "username": "测试用户",
      "email": "test@example.com",
      "role": "user",
      "status": "active",
      "sessionCount": 3,
      "totalTokens": 12500,
      "createdAt": "2026-03-10T00:00:00.000Z",
      "updatedAt": "2026-03-12T00:00:00.000Z"
    }
  ],
  "pagination": { "page": 1, "pageSize": 20, "total": 2, "totalPages": 1 }
}
```

> 列表中的 `sessionCount` 和 `totalTokens` 为聚合字段，方便管理员了解用户活跃度。

### 7.2 更新用户（角色/状态）

```
PUT /api/admin/users/:id
```

**请求体**（部分更新）：

```json
{
  "role": "admin",
  "status": "disabled"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `role` | string | `admin` / `user` |
| `status` | string | `active` / `disabled`，禁用后该用户所有请求返回 403 |

**响应** `200`：

```json
{
  "data": {
    "id": "usr_def456",
    "username": "测试用户",
    "email": "test@example.com",
    "role": "admin",
    "status": "disabled",
    "createdAt": "2026-03-10T00:00:00.000Z",
    "updatedAt": "2026-03-14T00:00:00.000Z"
  }
}
```

**错误响应**：

| 状态码 | code | 说明 |
|--------|------|------|
| 403 | `FORBIDDEN` | 不能修改自己的角色/状态 |
| 404 | `NOT_FOUND` | 用户不存在 |

### 7.3 删除用户

```
DELETE /api/admin/users/:id
```

> 级联删除该用户的所有会话、消息、token 记录、个人世界书。**此操作不可恢复**，前端应做二次确认弹窗。

**响应** `200`：

```json
{
  "data": { "deleted": true }
}
```

**错误响应**：

| 状态码 | code | 说明 |
|--------|------|------|
| 403 | `FORBIDDEN` | 不能删除自己 |
| 404 | `NOT_FOUND` | 用户不存在 |

---

## 8. 模型 (`/api/models`)

### 8.1 获取可用模型列表

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
      "status": "online",
      "providerId": "prov_001",
      "providerName": "本地 vLLM"
    },
    {
      "id": "deepseek-chat",
      "name": "DeepSeek Chat",
      "maxContextLength": 65536,
      "status": "online",
      "providerId": "prov_002",
      "providerName": "DeepSeek"
    }
  ]
}
```

| 字段 | 说明 |
|------|------|
| `id` | 模型标识，用于 `chat/sessions` 的 `modelId` 参数 |
| `name` | 人类可读名称，来自 Provider 的 `GET /v1/models` 或手动配置 |
| `maxContextLength` | 最大上下文窗口 (tokens)，用于前端上下文进度条计算 |
| `status` | 见下表 |
| `providerId` | 所属 Provider ID（v0.6 新增） |
| `providerName` | 所属 Provider 名称（v0.6 新增），前端可用于分组展示 |

| status | 含义 | 前端处理 |
|--------|------|---------|
| `online` | 已加载到 GPU，可立即推理 | 正常可选 |
| `offline` | 已注册但未加载，首次请求需等待 30-60s | 可选，但提示"首次使用需等待加载" |
| `busy` | 队列接近满载 | 可选，但提示"当前繁忙" |

> 后端缓存此接口 60 秒。`status` 由后端综合各 Provider 的 `/v1/models`（loaded/available 状态）和 `/health`（队列信息）推算。

---

## 9. LLM Provider 管理 (`/api/admin/providers`) 🔒

> 以下接口均需要管理员权限。用于配置连接多个大模型推理服务。

### 9.1 获取 Provider 列表

```
GET /api/admin/providers
```

**响应** `200`：

```json
{
  "data": [
    {
      "id": "prov_001",
      "name": "本地 vLLM",
      "baseUrl": "http://localhost:8000",
      "apiKey": "",
      "models": [],
      "autoDiscover": true,
      "enabled": true,
      "priority": 10,
      "createdAt": "2026-03-14T00:00:00.000Z",
      "updatedAt": "2026-03-14T00:00:00.000Z"
    },
    {
      "id": "prov_002",
      "name": "DeepSeek",
      "baseUrl": "https://api.deepseek.com",
      "apiKey": "sk-***eek",
      "models": [
        { "id": "deepseek-chat", "name": "DeepSeek Chat", "maxContextLength": 65536 },
        { "id": "deepseek-reasoner", "name": "DeepSeek Reasoner", "maxContextLength": 65536 }
      ],
      "autoDiscover": false,
      "enabled": true,
      "priority": 5,
      "createdAt": "2026-03-14T00:00:00.000Z",
      "updatedAt": "2026-03-14T00:00:00.000Z"
    }
  ]
}
```

> `apiKey` 在响应中始终脱敏（仅显示最后 3 位），如 `sk-***eek`。空字符串原样返回。

### 9.2 创建 Provider

```
POST /api/admin/providers
```

**请求体**：

```json
{
  "name": "DeepSeek",
  "baseUrl": "https://api.deepseek.com",
  "apiKey": "sk-xxxxxxxxxxxxxxxx",
  "models": [
    { "id": "deepseek-chat", "name": "DeepSeek Chat", "maxContextLength": 65536 }
  ],
  "autoDiscover": false,
  "enabled": true,
  "priority": 5
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `name` | string | 是 | Provider 显示名称 |
| `baseUrl` | string | 是 | API 基础地址（如 `http://localhost:8000`、`https://api.deepseek.com`） |
| `apiKey` | string | 否 | API 密钥，本地服务可不填 |
| `models` | array | 否 | 手动配置的模型列表，每项需包含 `id`、`name`、`maxContextLength` |
| `autoDiscover` | boolean | 否 | 是否自动通过 `/v1/models` 发现模型，默认 `true` |
| `enabled` | boolean | 否 | 是否启用，默认 `true` |
| `priority` | number | 否 | 排序优先级（越大越靠前），默认 `0` |

**响应** `201`：返回完整 Provider 对象（结构同 9.1 中的单项）。

### 9.3 更新 Provider

```
PUT /api/admin/providers/:id
```

**请求体**（部分更新）：与创建相同，所有字段可选。`apiKey` 传空字符串可清除密钥，不传则不修改。

**响应** `200`：返回更新后的完整 Provider 对象。

### 9.4 删除 Provider

```
DELETE /api/admin/providers/:id
```

> 删除 Provider 不会影响已有的聊天会话历史。但如果有会话正在使用该 Provider 的模型，后续发消息时会报错"模型不可用"。

**响应** `200`：

```json
{
  "data": { "deleted": true }
}
```

### 9.5 测试 Provider 连通性

```
POST /api/admin/providers/:id/test
```

> 后端尝试调用该 Provider 的 `GET /v1/models` 接口，验证 baseUrl 和 apiKey 是否正确。

**响应** `200`（连通成功）：

```json
{
  "data": {
    "success": true,
    "models": [
      { "id": "qwen3-32b", "name": "Qwen3 32B", "maxContextLength": 32768 }
    ],
    "latencyMs": 120
  }
}
```

**响应** `200`（连通失败）：

```json
{
  "data": {
    "success": false,
    "error": "连接超时"
  }
}
```

> 测试接口始终返回 200，通过 `success` 字段区分成功/失败，避免前端误判为服务器错误。

---

## 10. 前端调用示例

### 10.1 流式消息处理

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

### 10.2 停止生成

```typescript
const abortController = new AbortController();

// 发送消息时传入 signal
sendMessage(sessionId, content, abortController.signal);

// 用户点击"停止"按钮
function stopGeneration() {
  abortController.abort();
}
```

### 10.3 游标分页加载消息历史

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

## 11. 接口总览

| 接口 | 方法 | 说明 | 认证 |
|------|------|------|------|
| `/api/auth/login` | POST | 登录 | 否 |
| `/api/auth/register` | POST | 注册 | 否 |
| `/api/auth/logout` | POST | 登出 | 是 |
| `/api/auth/me` | GET | 获取当前用户 | 是 |
| `/api/characters` | GET | 角色卡列表/搜索 | **否** |
| `/api/characters/:id` | GET | 角色卡详情 | **否** |
| `/api/characters/tags` | GET | 所有标签 | **否** |
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
| `/api/users/me` | PUT | 更新个人资料 | 是 |
| `/api/users/me/password` | PUT | 修改密码 | 是 |
| `/api/users/me/usage` | GET | Token 使用统计 | 是 |
| `/api/admin/users` | GET | 用户列表 | 🔒 管理员 |
| `/api/admin/users/:id` | PUT | 更新用户（角色/状态） | 🔒 管理员 |
| `/api/admin/users/:id` | DELETE | 删除用户 | 🔒 管理员 |
| `/api/admin/providers` | GET | Provider 列表 | 🔒 管理员 |
| `/api/admin/providers` | POST | 创建 Provider | 🔒 管理员 |
| `/api/admin/providers/:id` | PUT | 更新 Provider | 🔒 管理员 |
| `/api/admin/providers/:id` | DELETE | 删除 Provider | 🔒 管理员 |
| `/api/admin/providers/:id/test` | POST | 测试 Provider 连通性 | 🔒 管理员 |
| `/api/models` | GET | 可用模型列表（聚合所有 Provider） | 是 |
