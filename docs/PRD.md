# 聊天机器人项目 — 产品需求文档 (PRD)

> 版本：0.3 | 最后更新：2026-03-03

---

## 1. 项目概述

### 1.1 项目愿景

构建一个自托管的 AI 角色扮演聊天平台，参考 SillyTavern 的核心体验（角色卡、世界书、流式对话），同时自主研发高性能 CUDA 推理引擎作为后端大模型服务，形成前端 → 后台 → 推理引擎的完整闭环。

### 1.2 核心价值


| 维度       | 描述                                                  |
| -------- | --------------------------------------------------- |
| **自主可控** | 推理引擎完全自研（llaisys），不依赖第三方推理框架                        |
| **高性能**  | FP8 量化、KV-Cache 管理、Prefix Cache、Decode Batching 等优化 |
| **角色扮演** | 世界书 + 角色卡系统，支持丰富的角色设定与世界观构建                         |
| **用户友好** | 现代化 Web UI，流式输出、上下文管理、Token 用量可视化                   |


### 1.3 系统三层架构

```
┌─────────────────────────────────────────────────────────────┐
│                     用户前端 (web/)                          │
│               Next.js + React + TailwindCSS                 │
│      角色卡页面 │ 聊天页面 │ 用户信息页面                      │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP / WebSocket (SSE)
┌──────────────────────▼──────────────────────────────────────┐
│                     用户后台 (server/)                        │
│               Next.js API Routes / tRPC                      │
│    用户管理 │ 角色卡/世界书 │ 聊天记录 │ Token 计量             │
│    上下文压缩 │ 限流/排队 │ 模型调度                           │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP (OpenAI-Compatible API)
┌──────────────────────▼──────────────────────────────────────┐
│                  大模型服务 (llm_service/)                    │
│               CUDA + C++ (llaisys)                           │
│    推理引擎 │ KV-Cache │ Prefix Cache │ Decode Batching       │
│    请求队列 │ 采样策略 │ 流式输出                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 用户角色


| 角色       | 描述                                                    |
| -------- | ----------------------------------------------------- |
| **普通用户** | 浏览和搜索角色卡，选择角色卡进行聊天，查看自己的 Token 使用情况。**不能上传/编辑/删除角色卡**。可以创建/编辑/删除自己的**个人世界书**（仅自己可见），也可以查看管理员创建的全局世界书 |
| **管理员**  | 上传/编辑/删除角色卡，管理**全局世界书**（所有用户可见，可关联到角色卡），管理用户，监控系统状态。拥有普通用户的全部权限 |


---

## 3. 用户前端 (Frontend)

> 技术栈：Next.js 14+ (App Router) / React / TailwindCSS / shadcn/ui
>
> **响应式设计**：所有页面均需适配桌面端、平板和手机。移动端聊天页面采用抽屉式会话列表（默认收起），角色卡页面采用单列卡片布局。

### 3.1 页面结构

```
/                        → 重定向到 /characters
/characters              → 角色卡列表页（大封面卡片网格）
/characters/[id]         → 角色卡详情页（封面+信息+开始对话）
/characters/[id]/edit    → 角色卡编辑页（🔒 仅管理员）
/chat                    → 聊天页面（含聊天列表 + 聊天窗口）
/chat/[sessionId]        → 具体聊天会话
/worldbooks              → 世界书管理页面
/worldbooks/[id]         → 世界书编辑页面
/profile                 → 用户信息页面
/login                   → 登录页面
```

### 3.2 P1 - 角色卡列表页 (`/characters`)

**功能描述**：展示所有可用的角色卡，用户可以搜索和浏览角色卡，点击后跳转到角色卡详情页。角色卡的上传/编辑/删除仅限管理员操作。

**卡片展示方式**：每张角色卡采用**大封面图 + 底部文案**的布局。封面图占据卡片 **~80%** 的面积，固定 3:4 竖向比例（超出部分裁切 `object-cover`）。封面下方显示角色名称、简短描述、标签，以及**右下角的相对时间**（如"2天前"、"1个月前"）。

**普通用户功能**：


| 功能点   | 详情                                   | 优先级 |
| ----- | ------------------------------------ | --- |
| 角色卡列表 | 大封面卡片网格布局（桌面 5 列，平板 3-4 列，手机 2 列）    | P1  |
| 搜索与筛选 | 按名称搜索，按标签筛选                          | P1  |
| 角色详情  | 点击卡片**跳转到独立的详情页** `/characters/[id]` | P1  |


**管理员额外功能**（仅管理员可见）：


| 功能点   | 详情                                                                               | 优先级 |
| ----- | -------------------------------------------------------------------------------- | --- |
| 导入角色卡 | **支持 SillyTavern .png 角色卡导入**（从 PNG 的 tEXt chunk 中解析嵌入的 JSON）；同时支持直接导入 JSON 文件   | P1  |
| 编辑角色卡 | 跳转到独立编辑页 `/characters/[id]/edit`，编辑各项字段（名称、描述、系统提示词、关联世界书等）。编辑入口隐藏在详情页角落，普通用户不可见 | P1  |
| 删除角色卡 | 在编辑页中删除，需二次确认                                                                    | P1  |
| 导出角色卡 | 导出为 JSON 文件（P2 支持导出为 SillyTavern 兼容的 .png 格式）                                    | P2  |


### 3.2.1 角色卡详情页 (`/characters/[id]`)

**功能描述**：点击角色卡后跳转到的独立页面，展示角色卡完整信息。

**页面布局**：

```
┌──────────────────────────────────────────────────────┐
│ ← 返回角色卡列表                            [编辑✏️] │  ← 编辑按钮极淡，仅管理员可用
├─────────────────┬────────────────────────────────────┤
│                 │  角色名称（大标题）                    │
│   封面图         │  来源 · 创建日期                      │
│   (3:4 比例)    │  [标签] [标签] [标签]                 │
│                 │  角色描述文字                          │
│                 │  关联世界书: xxx                       │
│                 │                                      │
│                 │  [========= 开始对话 =========]       │  ← 醒目大按钮
├─────────────────┴────────────────────────────────────┤
│ ┃ 角色介绍                                            │  ← 紫色左竖条标题
│   性格: ...                                           │
│   场景设定: ...                                        │
├──────────────────────────────────────────────────────┤
│ ┃ 开场白                                              │
│   "首条消息内容..."                                     │
└──────────────────────────────────────────────────────┘
```

**可见内容**：仅展示**角色介绍**（性格 + 场景设定）和**开场白**两个板块。系统提示词、示例对话、创作者备注等技术字段**不在详情页展示**，仅在管理员编辑页可见。

移动端布局：封面在上、信息在下，纵向排列。

**SillyTavern 角色卡兼容说明**：

SillyTavern 使用 Character Card V2 规范（基于 [CharacterAI Character Card Spec](https://github.com/malfoyslastname/character-card-spec-v2)），将角色数据以 JSON 形式嵌入 PNG 图片的 tEXt metadata chunk 中。导入流程：

```
上传 .png 文件 → 解析 PNG tEXt chunk（key: "chara"）→ Base64 解码 → JSON.parse → 映射到内部 CharacterCard 模型
```

**数据模型 — 角色卡 (CharacterCard)**：

```typescript
interface CharacterCard {
  id: string;
  name: string;
  avatar: string;           // 头像 URL（导入 ST 卡时从 PNG 本体提取）
  coverImage: string;       // 封面图 URL（角色卡列表和详情页展示，若无则用渐变占位）
  description: string;      // 角色简述（前端列表显示）
  personality: string;      // 性格描述
  scenario: string;         // 场景设定
  systemPrompt: string;     // 系统提示词
  firstMessage: string;     // 首条消息（角色的开场白）
  alternateGreetings: string[];  // 可选的替代开场白
  exampleDialogue: string;  // 示例对话
  creatorNotes: string;     // 创作者备注
  worldBookIds: string[];   // 关联的世界书 ID
  tags: string[];
  source: 'manual' | 'sillytavern_png' | 'json_import';
  createdBy: string;        // 上传者（管理员）ID
  createdAt: Date;
  updatedAt: Date;
}
```

### 3.3 P1 - 聊天页面 (`/chat`)

**功能描述**：核心交互页面。左侧为会话列表，右侧为聊天窗口。

#### 3.3.1 左侧 — 会话列表


| 功能点  | 详情                            |
| ---- | ----------------------------- |
| 会话列表 | 按最近活跃排序，显示角色头像、名称、最近一条消息预览、时间 |
| 新建会话 | 点击"+"按钮跳转到角色卡页面选择角色           |
| 删除会话 | 右键/滑动删除，需二次确认                 |


#### 3.3.2 右侧 — 聊天窗口


| 功能点     | 详情                                                           | 优先级 |
| ------- | ------------------------------------------------------------ | --- |
| 消息列表    | 气泡样式，区分用户消息与 AI 消息，支持 Markdown 渲染。**AI 头像紧贴左侧、用户头像紧贴右侧**，不居中 | P0  |
| 流式输出    | 实时显示 AI 生成文字，逐 Token 渲染                                      | P0  |
| 输入框     | 底部固定输入框，支持多行，Enter 发送，Shift+Enter 换行                         | P0  |
| 停止生成    | 生成过程中显示"停止"按钮，点击立即停止流式输出                                     | P0  |
| 重新生成    | AI 消息旁的操作按钮，点击后删除当前回复并重新生成                                   | P1  |
| 编辑消息    | 用户可点击自己的历史消息进行编辑，编辑后可选择重新生成                                  | P1  |
| 模型选择    | 聊天窗口顶部下拉框，可切换可用模型                                            | P1  |
| 上下文用量   | 进度条形式展示当前上下文窗口使用比例（已用/总量 Token）                              | P1  |
| 个人世界书选择 | 聊天窗口顶部或设置面板中，可为当前会话启用/禁用自己的个人世界书（多选）。全局世界书由角色卡自动关联，用户无需手动管理 | P1  |
| 上下文压缩提示 | 当上下文接近上限时，提示用户已进行自动压缩                                        | P2  |


**数据模型 — 聊天会话 (ChatSession)**：

```typescript
interface ChatSession {
  id: string;
  userId: string;
  characterId: string;
  modelId: string;
  title: string;               // 自动生成或用户自定义
  personalWorldBookIds: string[]; // 用户为此会话启用的个人世界书 ID 列表
  messages: ChatMessage[];
  contextUsage: {
    usedTokens: number;
    maxTokens: number;
  };
  createdAt: Date;
  updatedAt: Date;
}

interface ChatMessage {
  id: string;
  role: 'system' | 'user' | 'assistant';
  content: string;
  tokenCount: number;          // 该消息的 Token 数
  isCompressed: boolean;       // 是否为压缩后的摘要
  createdAt: Date;
  editedAt?: Date;
}
```

### 3.4 P1 - 世界书管理页面 (`/worldbooks`)

**功能描述**：前端提供完整的世界书创建、编辑、导入/导出界面。世界书以 JSON 文件为标准交换格式。世界书分为两类：

| 类型 | 创建者 | 可见范围 | 可编辑者 | 说明 |
|------|--------|----------|----------|------|
| **全局世界书** | 管理员 | 所有用户 | 仅管理员 | 关联到角色卡，聊天时自动注入 |
| **个人世界书** | 普通用户 | 仅创建者 | 仅创建者 | 用户自定义的世界观补充，聊天时可手动选择启用 |

#### 3.4.1 世界书列表页 (`/worldbooks`)

| 功能点 | 详情 | 优先级 |
|--------|------|--------|
| 世界书列表 | 统一列表展示所有可见的世界书（个人 + 全局），每条世界书显示名称、描述、词条数量，并用标签标识 **"全局"** 或 **"个人"**。支持按 scope 筛选 | P1 |
| 新建世界书 | 普通用户创建个人世界书；管理员可选择创建全局或个人世界书 | P1 |
| 导入世界书 | 上传 JSON 文件导入，兼容 SillyTavern lorebook 格式。普通用户导入为个人世界书 | P1 |
| 导出世界书 | 导出为 JSON 文件（自己的和全局的均可导出） | P1 |
| 删除世界书 | 需二次确认。普通用户只能删除自己的个人世界书；管理员可删除全局世界书（提示关联的角色卡将失去该世界书） | P1 |


#### 3.4.2 世界书编辑页 (`/worldbooks/[id]`)


| 功能点      | 详情                               | 优先级 |
| -------- | -------------------------------- | --- |
| 基本信息编辑   | 编辑世界书名称、描述                       | P1  |
| 词条列表     | 可折叠的词条列表，展示关键词、启用状态、优先级          | P1  |
| 词条编辑     | 内联编辑或弹窗编辑词条：关键词、内容、注入位置、优先级、启用开关 | P1  |
| 新增词条     | 添加新的世界书词条                        | P1  |
| 删除词条     | 删除单条词条                           | P1  |
| 批量操作     | 批量启用/禁用、批量删除                     | P2  |
| 词条搜索     | 按关键词或内容搜索词条                      | P2  |
| Token 统计 | 显示每条词条的 Token 数和世界书总 Token 数     | P1  |


**世界书 JSON 文件格式**：

```json
{
  "name": "奇幻大陆设定",
  "description": "包含奇幻大陆的地理、种族、魔法体系等设定",
  "version": "1.0",
  "entries": [
    {
      "keywords": ["精灵族", "精灵", "elf"],
      "content": "精灵族是奇幻大陆上最古老的种族之一，寿命可达千年...",
      "position": "after_system",
      "priority": 10,
      "enabled": true
    },
    {
      "keywords": ["魔法学院", "学院"],
      "content": "位于大陆中央的魔法学院是最高魔法学府...",
      "position": "after_system",
      "priority": 5,
      "enabled": true
    }
  ]
}
```

### 3.5 P1 - 用户信息页面 (`/profile`)

**功能描述**：展示当前登录用户的个人信息和使用统计数据。

| 功能点 | 详情 | 优先级 |
|--------|------|--------|
| 基本信息 | 头像、用户名、邮箱、角色（管理员/普通用户）、注册时间。只读展示 | P1 |
| Token 使用统计 | 统计卡片：总 Token 使用量、Prompt Tokens、Completion Tokens、总会话数。数值用面积图展示趋势 | P1 |
| 用量图表 | Recharts 面积图，支持按日/周/月切换时间维度，双曲线分别显示 Prompt 和 Completion Token | P1 |
| 会话统计 | 总会话数、总消息数 | P1 |

### 3.6 P2 - 登录页面 (`/login`)

**功能描述**：用户认证入口页面。MVP 阶段（Phase 1-3）跳过登录，使用硬编码默认用户；Phase 4 启用。

| 功能点 | 详情 | 优先级 |
|--------|------|--------|
| 登录表单 | 邮箱 + 密码输入，密码支持显示/隐藏切换 | P2 |
| 注册入口 | 底部"没有账号？注册"链接，跳转到注册页面（或内嵌注册表单） | P2 |
| 表单校验 | 前端校验邮箱格式、密码非空；后端校验失败时显示错误提示 | P2 |
| 登录成功 | 跳转到 `/characters`，JWT 存储在 httpOnly Cookie | P2 |

**页面布局**：居中卡片式表单，顶部 Logo + 标题"AI Chat · 角色扮演聊天平台"。

---

## 4. 用户后台 (Backend)

> 技术栈：Next.js API Routes / Prisma ORM / PostgreSQL / Redis

### 4.1 架构概览

```
Next.js API Routes
├── /api/auth/*              → 认证相关
├── /api/characters/*        → 角色卡查询（所有用户：列表、搜索、详情）
├── /api/admin/characters/*  → 角色卡管理（🔒 仅管理员：导入、编辑、删除）
├── /api/worldbooks/*        → 世界书 CRUD（按 scope + userId 自动过滤可见范围）
├── /api/chat/*              → 聊天相关（创建会话、发送消息、流式响应）
├── /api/users/*             → 用户管理
└── /api/models/*            → 模型列表与状态
```

### 4.2 核心模块

#### 4.2.1 角色卡与世界书管理


| 功能       | 详情                                                              |
| -------- | --------------------------------------------------------------- |
| 角色卡 CRUD | 创建/读取/更新/删除角色卡（**写操作仅限管理员**，普通用户只读）                             |
| 角色卡导入    | 解析 SillyTavern .png（tEXt chunk 嵌入 JSON）和 JSON 文件两种格式（**仅限管理员**） |
| 角色卡搜索    | 支持按名称模糊搜索、按标签筛选（所有用户可用）                                         |
| 世界书 CRUD | 世界书以 JSON 为存储格式，分为**全局**（管理员创建，所有用户可见）和**个人**（用户创建，仅自己可见）两类。每本包含多条词条(Entry)，每条有关键词触发条件 |
| 世界书权限    | 全局世界书仅管理员可写；个人世界书仅创建者可读写。查询 API 自动按用户身份过滤可见范围 |
| 世界书导入/导出 | 支持 JSON 文件上传/下载，兼容 SillyTavern lorebook 格式。普通用户导入为个人世界书，管理员可选择导入为全局或个人 |
| 世界书注入    | 聊天时注入来源包括：角色卡关联的全局世界书 + 用户在会话中手动启用的个人世界书。扫描用户输入及近期对话，匹配关键词后注入上下文 |


**数据模型 — 世界书 (WorldBook)**：

世界书在数据库中存储元数据，词条内容以 JSON 形式存储（PostgreSQL 的 JSONB 字段或 SQLite 的 TEXT 字段）。

```typescript
interface WorldBook {
  id: string;
  name: string;
  description: string;
  version: string;             // JSON 格式版本号
  scope: 'global' | 'personal'; // 全局（管理员创建，所有人可见）或个人（仅创建者可见）
  userId: string;              // 创建者 ID。全局世界书为管理员 ID，个人世界书为用户 ID
  entries: WorldBookEntry[];   // 存储为 JSONB / JSON TEXT
  totalTokenCount: number;     // 所有词条的 Token 总数
  createdAt: Date;
  updatedAt: Date;
}

interface WorldBookEntry {
  id: string;
  keywords: string[];          // 触发关键词
  secondaryKeywords?: string[];// 二级关键词（需同时命中主关键词+二级关键词才注入）
  content: string;             // 注入到上下文的内容
  position: 'before_system' | 'after_system' | 'before_user';
  priority: number;            // 优先级（数字越大越优先），上下文紧张时低优先级被裁剪
  enabled: boolean;
  tokenCount: number;
}
```

#### 4.2.2 聊天消息处理

**请求处理流程**：

```
用户发送消息
    │
    ▼
[1] 保存用户消息到数据库
    │
    ▼
[2] 构建 Prompt
    ├── 系统提示词（角色卡 systemPrompt）
    ├── 角色设定（personality + scenario）
    ├── 世界书词条（关键词匹配注入）
    │   ├── 全局世界书：角色卡关联的世界书（自动启用）
    │   └── 个人世界书：用户在会话中手动启用的世界书
    ├── 历史消息
    └── 当前用户输入
    │
    ▼
[3] 上下文窗口检查
    ├── 计算总 Token 数
    ├── 若超限 → 触发上下文压缩
    └── 返回上下文使用情况给前端
    │
    ▼
[4] 调用大模型服务（OpenAI-Compatible API）
    ├── POST /v1/chat/completions
    ├── stream: true
    └── 携带模型参数（temperature, top_p, max_tokens 等）
    │
    ▼
[5] 流式转发
    ├── 接收 SSE 数据流
    ├── 实时转发给前端
    └── 累计生成内容
    │
    ▼
[6] 生成完毕
    ├── 保存 AI 回复到数据库
    ├── 记录 Token 用量（prompt_tokens + completion_tokens）
    └── 更新会话上下文使用情况
```

#### 4.2.3 Token 计量与用量管理


| 功能       | 详情                                       |
| -------- | ---------------------------------------- |
| Token 记录 | 每次请求记录 prompt_tokens 和 completion_tokens |
| 用量统计     | 按用户、按日期聚合 Token 使用量                      |
| 用量限制（P2） | 可设置用户每日/每月 Token 上限                      |


**数据模型 — Token 用量**：

```typescript
interface TokenUsage {
  id: string;
  userId: string;
  sessionId: string;
  messageId: string;
  modelId: string;
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
  createdAt: Date;
}
```

#### 4.2.4 上下文压缩


| 策略    | 详情                                | 优先级 |
| ----- | --------------------------------- | --- |
| 滑动窗口  | 当历史消息超过上下文的 80% 时，保留最近 N 轮 + 系统提示 | P0  |
| 摘要压缩  | 将被裁剪的历史消息通过 LLM 生成摘要，作为"记忆"保留     | P2  |
| 世界书裁剪 | 上下文紧张时按优先级裁剪世界书词条                 | P1  |


#### 4.2.5 限流与排队


| 功能   | 详情                             |
| ---- | ------------------------------ |
| 请求排队 | 当大模型服务繁忙时（并发数达上限），新请求进入队列等待    |
| 队列状态 | 向前端返回排队位置和预计等待时间               |
| 超时处理 | 队列中等待超过阈值自动失败并通知用户             |
| 速度监控 | 记录每次推理的 tokens/s，用于估算等待时间和性能分析 |


### 4.3 数据库设计

```
┌────────────┐     ┌──────────────┐     ┌──────────────────┐
│   User     │     │ CharacterCard│     │    WorldBook      │
├────────────┤     ├──────────────┤     ├──────────────────┤
│ id         │     │ id           │     │ id               │
│ username   │     │ name         │     │ name             │
│ email      │     │ avatar       │     │ description      │
│ password   │     │ description  │     │ scope            │ ← 'global' | 'personal'
│ role       │     │ systemPrompt │     │ userId           │ ← 创建者(归属者)
│ createdAt  │     │ firstMessage │     │ entries[]        │
└─────┬──────┘     │ createdBy    │     │ createdAt        │
      │            │ worldBookIds │     └────┬────────┬────┘
      │            └──────────────┘         │        │
      │                                     │        │
      │ 1:N               N:M (角色卡↔全局世界书)│        │
      │            ┌───────────────────┐    │        │
      │            │ CharacterWorldBook│◄───┘        │
      │            ├───────────────────┤             │
      │            │ characterId       │             │
      │            │ worldBookId       │             │
      │            └───────────────────┘             │
      │                                              │
      │ 1:N               N:M (会话↔个人世界书)         │
      ▼            ┌───────────────────┐             │
┌──────────────┐   │ SessionWorldBook  │◄────────────┘
│ ChatSession  │   ├───────────────────┤
├──────────────┤   │ sessionId         │ ← 用户在会话中启用的个人世界书
│ id           │   │ worldBookId       │
│ userId       │   └───────────────────┘
│ characterId  │
│ modelId      │     ┌──────────────┐
│ title        │     │ ChatMessage  │
│ createdAt    │     ├──────────────┤
└──────┬───────┘     │ id           │
       │             │ sessionId    │
       │ 1:N         │ role         │
       ▼             │ content      │
┌──────────────┐     │ tokenCount   │
│ TokenUsage   │     │ isCompressed │
├──────────────┤     │ createdAt    │
│ id           │     └──────────────┘
│ userId       │
│ sessionId    │
│ promptTokens │
│ completionTk │
│ createdAt    │
└──────────────┘
```

---

## 5. 大模型服务 (LLM Service)

> 技术栈：CUDA + C++ (llaisys) / Python API 层

### 5.1 现有能力（llaisys 引擎）


| 能力                      | 状态                          |
| ----------------------- | --------------------------- |
| Qwen3 推理（8B/14B/32B）    | ✅ 已实现                       |
| FP8 量化                  | ✅ 已实现                       |
| Tensor Parallelism (多卡) | ✅ 已实现                       |
| Top-K / Top-P 采样        | ✅ 已实现                       |
| KV-Cache 基础管理           | ✅ 已实现（预分配，max_seq_len=8192） |
| 流式 Token 生成             | ✅ 已实现 (stream_generate)     |


### 5.2 需要新增的能力

#### 5.2.1 P0 — OpenAI-Compatible HTTP API

在 llaisys 之上包装一层 HTTP 服务，提供 OpenAI 格式的 API。

```
POST /v1/chat/completions
GET  /v1/models
GET  /health
```

**请求格式**（兼容 OpenAI）：

```json
{
  "model": "qwen3-8b",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "stream": true,
  "temperature": 0.8,
  "top_p": 0.95,
  "top_k": 40,
  "max_tokens": 2048,
  "stop": ["<|endoftext|>"]
}
```

**流式响应格式**（SSE）：

```
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{"content":"你"},"index":0}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{"content":"好"},"index":0}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{},"finish_reason":"stop","index":0}],"usage":{"prompt_tokens":128,"completion_tokens":64,"total_tokens":192}}

data: [DONE]
```

**技术方案**：使用 Python (FastAPI/uvicorn) 包装 llaisys Python 绑定，作为 HTTP 服务层。

#### 5.2.2 P0 — 请求队列管理


| 功能     | 详情                                  |
| ------ | ----------------------------------- |
| 等待队列   | 控制最大并发推理数，超出的请求进入 FIFO 队列           |
| 队列长度限制 | 队列满时返回 HTTP 503 Service Unavailable |
| 请求取消   | 支持客户端断开时取消正在排队/推理的请求                |


#### 5.2.3 P1 — KV-Cache 优化


| 功能                    | 详情                               |
| --------------------- | -------------------------------- |
| 动态分配                  | 按实际序列长度分配 KV-Cache，而非预分配固定大小     |
| 分页管理 (PagedAttention) | 将 KV-Cache 分成固定大小的 Block，按需分配和释放 |
| Cache 回收              | 会话结束或空闲超时后释放 KV-Cache            |


#### 5.2.4 P1 — Prefix/Prompt Cache


| 功能        | 详情                                    |
| --------- | ------------------------------------- |
| Prefix 缓存 | 对相同的 System Prompt 前缀，复用已计算的 KV-Cache |
| 命中策略      | 基于 Prefix 哈希匹配，LRU 淘汰                 |
| 命中率监控     | 暴露 Prefix Cache 命中率指标                 |


#### 5.2.5 P1 — Continuous Batching


| 功能              | 详情                                          |
| --------------- | ------------------------------------------- |
| Decode Batching | 多个请求的 decode 阶段合并为一个 batch 执行               |
| 动态插入            | 新请求可在当前 batch 的 decode 间隙插入                 |
| Prefill 分离      | Prefill 和 Decode 分离调度，避免长 Prefill 阻塞 Decode |


#### 5.2.6 P2 — 其他优化


| 功能                   | 详情                               |
| -------------------- | -------------------------------- |
| Speculative Decoding | 小模型草稿 + 大模型验证，提升吞吐               |
| Chunked Prefill      | 长 Prompt 分块 Prefill，减少首 Token 延迟 |
| 量化 KV-Cache          | 将 KV-Cache 从 BF16 压缩到 FP8/INT8   |


### 5.3 性能指标目标


| 指标                         | 目标（Qwen3-8B, 单卡）               |
| -------------------------- | ------------------------------ |
| Time to First Token (TTFT) | < 200ms (prompt ≤ 1024 tokens) |
| Decode 速度                  | ≥ 60 tokens/s (单请求)            |
| 并发吞吐                       | ≥ 8 并发请求                       |
| Prefix Cache 命中时 TTFT      | < 50ms                         |


---

## 6. API 接口设计

### 6.1 后台 → 大模型服务


| 接口                     | 方法   | 描述               |
| ---------------------- | ---- | ---------------- |
| `/v1/chat/completions` | POST | 发送聊天补全请求（流式/非流式） |
| `/v1/models`           | GET  | 获取可用模型列表         |
| `/health`              | GET  | 健康检查 + 队列状态      |


### 6.2 前端 → 后台


| 接口                                         | 方法     | 描述                                  |
| ------------------------------------------ | ------ | ----------------------------------- |
| `/api/auth/login`                          | POST   | 用户登录，返回 JWT（httpOnly Cookie）          |
| `/api/auth/register`                       | POST   | 用户注册                                |
| `/api/auth/logout`                         | POST   | 用户登出，清除 Cookie                      |
| `/api/auth/me`                             | GET    | 获取当前登录用户信息                          |
| `/api/characters`                          | GET    | 获取角色卡列表（支持 ?search=关键词&tag=标签 查询参数） |
| `/api/characters/[id]`                     | GET    | 获取角色卡详情                             |
| `/api/admin/characters`                    | POST   | 创建角色卡 🔒 管理员                        |
| `/api/admin/characters/import`             | POST   | 导入角色卡（支持 .png / .json）🔒 管理员        |
| `/api/admin/characters/[id]`               | PUT    | 编辑角色卡 🔒 管理员                        |
| `/api/admin/characters/[id]`               | DELETE | 删除角色卡 🔒 管理员                        |
| `/api/admin/characters/[id]/export`        | GET    | 导出角色卡为 JSON 🔒 管理员                  |
| `/api/characters/tags`                     | GET    | 获取所有角色卡标签（用于筛选器）                   |
| `/api/worldbooks`                          | GET    | 获取可见的世界书列表（自动返回：个人世界书 + 全局世界书，支持 ?scope=global\|personal 筛选） |
| `/api/worldbooks`                          | POST   | 创建世界书。普通用户只能创建 `scope=personal`；管理员可选 `global` 或 `personal` |
| `/api/worldbooks/import`                   | POST   | 导入世界书 JSON 文件。普通用户导入为个人世界书；管理员可指定 scope |
| `/api/worldbooks/[id]`                     | GET    | 获取世界书详情（含所有词条）。仅可访问自己的个人世界书或全局世界书 |
| `/api/worldbooks/[id]`                     | PUT    | 更新世界书。个人世界书仅创建者可改；全局世界书仅管理员可改 |
| `/api/worldbooks/[id]`                     | DELETE | 删除世界书。个人世界书仅创建者可删；全局世界书仅管理员可删 |
| `/api/worldbooks/[id]/export`              | GET    | 导出世界书为 JSON 文件（可见即可导出） |
| `/api/worldbooks/[id]/entries`             | POST   | 新增词条（需有世界书写权限） |
| `/api/worldbooks/[id]/entries/[entryId]`   | PUT    | 更新词条（需有世界书写权限） |
| `/api/worldbooks/[id]/entries/[entryId]`   | DELETE | 删除词条（需有世界书写权限） |
| `/api/chat/sessions`                       | GET    | 获取用户的会话列表                           |
| `/api/chat/sessions`                       | POST   | 创建新会话（绑定角色卡，自动注入首条消息）              |
| `/api/chat/sessions/[id]`                  | PUT    | 更新会话（切换模型、绑定个人世界书、修改标题）            |
| `/api/chat/sessions/[id]`                  | DELETE | 删除会话                                |
| `/api/chat/sessions/[id]/messages`         | GET    | 获取会话的消息历史                           |
| `/api/chat/sessions/[id]/messages`         | POST   | 发送消息（返回 SSE 流）                      |
| `/api/chat/sessions/[id]/messages/[msgId]` | PUT    | 编辑消息                                |
| `/api/chat/sessions/[id]/regenerate`       | POST   | 重新生成最后一条 AI 消息                      |
| `/api/chat/sessions/[id]/stop`             | POST   | 停止当前生成                              |
| `/api/users/me`                            | GET    | 获取当前用户信息                            |
| `/api/users/me/usage`                      | GET    | 获取 Token 使用统计                       |
| `/api/models`                              | GET    | 获取可用模型列表                            |


---

## 7. 非功能性需求


| 维度        | 要求                                               |
| --------- | ------------------------------------------------ |
| **安全性**   | 用户密码 bcrypt 加密存储；JWT 认证；API 请求签名验证               |
| **可用性**   | 大模型服务异常时，前端显示友好的错误提示和重试按钮                        |
| **可观测性**  | 推理服务暴露 metrics（请求数、延迟、Token 吞吐、队列长度、Cache 命中率）   |
| **可扩展性**  | 支持添加新模型（通过配置文件注册）；支持多卡/多机扩展                      |
| **数据持久化** | 聊天记录、用户数据持久化到 PostgreSQL；Redis 用于缓存和队列           |
| **主题切换**  | 支持亮色/暗色主题切换（Header 中提供切换按钮），**默认亮色主题**           |
| **响应式设计** | 全部页面适配桌面（≥1024px）、平板（768-1023px）、手机（<768px）三个断点  |
| **部署独立性** | 前端+后台（Next.js）与大模型服务（FastAPI）分开部署，通过 HTTP API 通信 |


---

## 8. 迭代开发计划

### Phase 1 — MVP Demo（2-3 周）

> 目标：跑通完整链路，单用户模式，能发送消息并看到流式响应


| 任务  | 模块    | 详情                                                                 |
| --- | ----- | ------------------------------------------------------------------ |
| 1.1 | 大模型服务 | 用 FastAPI 包装 llaisys，实现 `/v1/chat/completions`（流式）和 `/v1/models`   |
| 1.2 | 后台    | Next.js 项目初始化，Prisma + SQLite（先不用 PG），实现最小数据模型（硬编码默认用户，跳过登录）       |
| 1.3 | 后台    | 实现 `/api/chat/sessions` 和 `/api/chat/sessions/[id]/messages`（流式转发） |
| 1.4 | 前端    | 最简聊天页面：输入框 + 消息列表 + 流式显示（响应式布局）                                    |
| 1.5 | 前端    | 停止生成功能                                                             |
| 1.6 | 联调    | 前端 → 后台 → 大模型服务全链路跑通（分开部署，独立端口）                                    |


**Phase 1 交付物**：一个可以输入文字、看到 AI 流式回复、并能停止生成的最简聊天应用。单用户模式，无需登录。

### Phase 2 — 角色卡与会话管理（1-2 周）


| 任务  | 模块  | 详情                                                                     |
| --- | --- | ---------------------------------------------------------------------- |
| 2.1 | 后台  | 角色卡数据模型（含 coverImage 字段）+ 读取/搜索/标签 API（所有用户）+ 管理 API（仅管理员）              |
| 2.2 | 后台  | SillyTavern .png 角色卡解析（tEXt chunk → Base64 → JSON）+ JSON 导入 API（管理员接口） |
| 2.3 | 后台  | 创建会话时绑定角色卡，自动注入 System Prompt + 首条消息                                   |
| 2.4 | 前端  | 角色卡列表页（大封面卡片网格，桌面 5 列）+ 搜索/标签筛选 + 右下角相对时间                              |
| 2.5 | 前端  | 角色卡详情页（封面左+信息右布局，角色介绍+开场白，"开始对话"大按钮）                                   |
| 2.6 | 前端  | 角色卡编辑页（管理员独立页面，导入 PNG/JSON，编辑全部字段，删除确认）                                 |
| 2.7 | 前端  | 聊天页面左侧会话列表（移动端为可收起的抽屉）                                                 |
| 2.8 | 前端  | 重新生成 + 编辑消息功能                                                          |
| 2.9 | 后台  | Token 计量记录                                                             |


**Phase 2 交付物**：用户可以搜索/浏览角色卡（大封面卡片）、查看详情页、选择角色开始聊天、管理多个会话、编辑消息和重新生成。管理员可以在独立编辑页导入 SillyTavern .png 或 JSON 角色卡。

### Phase 3 — 世界书与上下文管理（1-2 周）


| 任务  | 模块  | 详情                                         |
| --- | --- | ------------------------------------------ |
| 3.1 | 后台  | 世界书数据模型（含 scope/userId 字段，JSONB 存储词条）+ CRUD API（按 scope + userId 自动过滤可见范围）+ 导入/导出 JSON |
| 3.2 | 前端  | 世界书列表页面：统一列表（全局+个人混排），scope 筛选标签栏（全部/全局/个人），scope 徽章 + 新建/导入/导出功能 |
| 3.3 | 前端  | 世界书编辑页面：词条的增删改查、启用/禁用开关、优先级设置、Token 计数     |
| 3.4 | 后台  | 世界书关键词匹配 + 上下文注入逻辑（全局世界书由角色卡自动关联 + 个人世界书由用户在会话中手动启用） |
| 3.5 | 前端  | 聊天页个人世界书选择：Header 中"世界书"按钮，点击弹出 Dialog，Switch 开关逐一启用/禁用个人世界书 |
| 3.6 | 后台  | 上下文窗口管理：Token 计数 + 滑动窗口压缩 + 世界书优先级裁剪       |
| 3.7 | 前端  | 上下文使用量进度条                                  |
| 3.8 | 前端  | 模型选择下拉框                                    |


**Phase 3 交付物**：完整的世界书系统（全局+个人两类，创建/编辑/导入/导出），聊天页中可选择启用个人世界书，上下文智能管理（含世界书注入和裁剪），用户可切换模型。

### Phase 4 — 多用户系统与打磨（1-2 周）


| 任务  | 模块  | 详情                                   |
| --- | --- | ------------------------------------ |
| 4.1 | 后台  | 用户注册/登录（NextAuth.js + JWT），替换硬编码默认用户 |
| 4.2 | 后台  | 用户数据隔离（每个用户只能看到自己的会话、角色卡关联等）         |
| 4.3 | 前端  | 登录/注册页面                              |
| 4.4 | 前端  | 用户信息页面 + Token 使用统计图表                |
| 4.5 | 后台  | 迁移到 PostgreSQL + Redis               |
| 4.6 | 全局  | UI/UX 打磨、错误处理完善、Loading 状态、响应式细节调优   |


### Phase 5 — 推理引擎优化（持续）


| 任务  | 模块    | 详情                             |
| --- | ----- | ------------------------------ |
| 5.1 | 大模型服务 | 请求队列 + 并发控制                    |
| 5.2 | 大模型服务 | Prefix Cache 实现                |
| 5.3 | 大模型服务 | Continuous Batching            |
| 5.4 | 大模型服务 | KV-Cache 动态分配 / PagedAttention |
| 5.5 | 后台    | 限流 + 排队等待 UI                   |
| 5.6 | 后台    | 上下文摘要压缩（LLM-based）             |


---

## 9. 技术选型汇总


| 组件       | 技术                            | 理由               |
| -------- | ----------------------------- | ---------------- |
| 前端框架     | Next.js 14 (App Router)       | SSR/SSG 支持，前后端同栈 |
| UI 组件库   | shadcn/ui + TailwindCSS       | 现代化设计，高度可定制      |
| 状态管理     | Zustand                       | 轻量，适合中等复杂度       |
| 后端框架     | Next.js API Routes            | 与前端同项目，降低部署复杂度   |
| 数据库      | PostgreSQL (生产) / SQLite (开发) | Prisma 支持无缝切换    |
| 缓存/队列    | Redis                         | 请求队列、Session 缓存  |
| ORM      | Prisma                        | 类型安全，迁移管理        |
| 认证       | NextAuth.js + JWT             | 成熟的 Next.js 认证方案 |
| 推理引擎     | llaisys (CUDA/C++)            | 自研，完全可控          |
| 推理 API 层 | FastAPI (Python)              | 异步支持好，SSE 支持原生   |
| 构建系统     | xmake (C++) / npm (Next.js)   | 已有基础设施           |


---

## 10. 目录结构规划

```
llm/
├── web/                           # Next.js 全栈应用（前端 + 后台）
│   ├── src/
│   │   ├── app/                   # App Router 页面
│   │   │   ├── (auth)/            # 认证相关页面
│   │   │   │   └── login/
│   │   │   ├── characters/        # 角色卡列表页
│   │   │   │   └── [id]/          # 角色卡详情页
│   │   │   │       └── edit/      # 角色卡编辑页（🔒 管理员）
│   │   │   ├── chat/              # 聊天页面
│   │   │   │   └── [sessionId]/
│   │   │   ├── worldbooks/        # 世界书管理页面
│   │   │   │   └── [id]/          # 世界书编辑页面
│   │   │   └── profile/           # 用户信息页面
│   │   ├── components/            # React 组件
│   │   │   ├── chat/              # 聊天相关组件
│   │   │   ├── characters/        # 角色卡相关组件
│   │   │   ├── worldbooks/        # 世界书相关组件
│   │   │   └── ui/                # 通用 UI 组件 (shadcn)
│   │   ├── lib/                   # 工具函数、API 客户端
│   │   ├── server/                # 服务端逻辑
│   │   │   ├── api/               # API Route handlers
│   │   │   ├── db/                # Prisma client & 数据访问层
│   │   │   └── services/          # 业务逻辑（聊天、世界书注入等）
│   │   └── stores/                # Zustand stores
│   ├── prisma/
│   │   └── schema.prisma          # 数据库 Schema
│   ├── package.json
│   └── next.config.js
│
├── llm_service/                   # 大模型推理服务（已有）
│   ├── api/                       # 新增：FastAPI HTTP 服务层
│   │   ├── main.py                # FastAPI 应用入口
│   │   ├── routes/                # 路由
│   │   │   ├── chat.py            # /v1/chat/completions
│   │   │   └── models.py          # /v1/models
│   │   ├── services/              # 业务逻辑
│   │   │   ├── inference.py       # 推理调度
│   │   │   └── queue.py           # 请求队列
│   │   └── schemas/               # Pydantic 数据模型
│   ├── python/                    # 已有：Python 绑定
│   ├── src/                       # 已有：C++/CUDA 源码
│   └── include/                   # 已有：C API 头文件
│
└── docs/
    ├── PRD.md                     # 产品需求文档（本文档）
    ├── frontend-tech-design.md    # 前端开发技术方案
    ├── frontend-api.md            # 前端 API 接口文档
    ├── llm-service-tech-design.md # 大模型服务技术方案
    └── llm-service-api.md         # 大模型服务 API 文档
```

---

## 11. 风险与应对


| 风险                | 影响     | 应对措施                               |
| ----------------- | ------ | ---------------------------------- |
| 推理引擎不稳定           | 服务中断   | Phase 1 先用简单的单请求模式验证，逐步增加并发        |
| 上下文窗口限制（当前 8192）  | 长对话被截断 | 先实现滑动窗口，后续扩展 max_seq_len           |
| 世界书注入导致 Prompt 过长 | 挤压对话空间 | 优先级裁剪 + Token 预算分配                 |
| 单机 GPU 资源有限       | 并发能力受限 | Continuous Batching 提升吞吐 + TP 多卡扩展 |
| 个人世界书滥用导致 Prompt 膨胀 | Token 预算超支 | 个人世界书有最大 Token 总量限制；聊天时对全局+个人世界书统一做优先级裁剪 |


---

## 12. 已确认的设计决策


| #   | 问题       | 决策                                                                |
| --- | -------- | ----------------------------------------------------------------- |
| 1   | 多用户支持    | **需要多用户支持**。MVP 阶段先做单用户模式（硬编码默认用户，跳过登录），Phase 4 再加完整用户系统          |
| 2   | 角色卡格式兼容性 | **兼容 SillyTavern 的 .png (embedded JSON) 角色卡格式**，支持从 ST 导出的角色卡直接导入 |
| 3   | 世界书编辑 UI | **前端提供完整的世界书创建/编辑界面**。世界书以 JSON 文件格式存储，支持导入/导出                    |
| 4   | 部署方案     | **三个服务分开部署**（前端+后台 / 大模型服务独立），各自独立进程、独立端口                         |
| 5   | 移动端适配    | **需要响应式设计**，支持手机和平板访问                                             |
| 6   | 角色卡详情交互  | **点击角色卡跳转到独立详情页**（非抽屉/弹窗），详情页上方为封面+标题的左右布局，下方展示角色介绍和开场白           |
| 7   | 详情页可见内容  | 详情页**不展示**系统提示词、示例对话、创作者备注等技术字段，这些仅在管理员编辑页可见                      |
| 8   | 主题默认值    | **默认亮色主题**，支持亮色/暗色手动切换                                            |
| 9   | 角色卡封面    | 卡片采用**大封面图 + 底部文案**布局，封面占卡片 ~80% 面积，3:4 竖向比例                      |


