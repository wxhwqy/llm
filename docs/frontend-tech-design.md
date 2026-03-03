# 前端开发技术方案文档

> 版本：0.2 | 最后更新：2026-03-03
>
> 基于：[PRD v0.1-draft](./PRD.md)

---

## 1. 技术栈总览

| 分类 | 技术选型 | 版本 | 选型理由 |
|------|----------|------|----------|
| 框架 | Next.js (App Router) | 14+ | SSR/SSG、前后端同栈、文件系统路由 |
| UI 库 | React | 18+ | 生态成熟，Concurrent Features 支持流式渲染 |
| 样式方案 | TailwindCSS | 3.x | 原子化 CSS，响应式断点开箱即用 |
| 组件库 | shadcn/ui | latest | 基于 Radix UI，可定制性强，无黑盒依赖 |
| 状态管理 | Zustand | 4.x | 轻量（~1KB），无 Provider 包裹，适合中等复杂度 |
| 数据请求 | TanStack Query (React Query) | 5.x | 缓存、重试、乐观更新、SSE 配合 |
| 表单处理 | React Hook Form + Zod | — | 类型安全校验，与 TypeScript 深度集成 |
| Markdown 渲染 | react-markdown + remark-gfm | — | 聊天消息的 Markdown 内容渲染 |
| 图表 | Recharts | 2.x | Token 用量统计图表，轻量且声明式 |
| 包管理 | pnpm | 8+ | 磁盘效率高，monorepo 友好 |
| 代码规范 | ESLint + Prettier | — | 统一代码风格 |
| 类型检查 | TypeScript | 5.x | 严格模式，端到端类型安全 |

---

## 2. 项目结构

```
web/
├── public/
│   └── favicon.ico
├── src/
│   ├── app/                          # Next.js App Router
│   │   ├── layout.tsx                # 根布局（全局 Provider、字体、主题）
│   │   ├── page.tsx                  # / → redirect 到 /characters
│   │   ├── (auth)/
│   │   │   ├── login/
│   │   │   │   └── page.tsx          # 登录页
│   │   │   └── layout.tsx            # 认证页面布局（居中卡片）
│   │   ├── (main)/                   # 需要登录的页面共用布局
│   │   │   ├── layout.tsx            # 主布局（顶部导航 + 侧边栏）
│   │   │   ├── characters/
│   │   │   │   ├── page.tsx          # 角色卡列表页
│   │   │   │   └── [id]/
│   │   │   │       ├── page.tsx      # 角色卡详情页
│   │   │   │       └── edit/
│   │   │   │           └── page.tsx  # 角色卡编辑页（🔒 管理员）
│   │   │   ├── chat/
│   │   │   │   ├── page.tsx          # 聊天页（默认：无选中会话）
│   │   │   │   └── [sessionId]/
│   │   │   │       └── page.tsx      # 具体会话页
│   │   │   ├── worldbooks/
│   │   │   │   ├── page.tsx          # 世界书列表页
│   │   │   │   └── [id]/
│   │   │   │       └── page.tsx      # 世界书编辑页
│   │   │   └── profile/
│   │   │       └── page.tsx          # 用户信息页
│   │   └── api/                      # API Route Handlers (后端)
│   │       ├── auth/
│   │       ├── characters/
│   │       ├── admin/
│   │       ├── chat/
│   │       ├── worldbooks/
│   │       ├── users/
│   │       └── models/
│   │
│   ├── components/
│   │   ├── ui/                       # shadcn/ui 基础组件
│   │   │   ├── button.tsx
│   │   │   ├── dialog.tsx
│   │   │   ├── dropdown-menu.tsx
│   │   │   ├── input.tsx
│   │   │   ├── sheet.tsx             # 抽屉组件（移动端会话列表）
│   │   │   ├── skeleton.tsx
│   │   │   ├── toast.tsx
│   │   │   └── ...
│   │   ├── layout/
│   │   │   ├── app-header.tsx        # 顶部导航栏
│   │   │   ├── app-sidebar.tsx       # 侧边导航
│   │   │   └── mobile-nav.tsx        # 移动端底部导航
│   │   ├── chat/
│   │   │   ├── chat-layout.tsx       # 聊天页整体布局（列表+窗口）
│   │   │   ├── session-list.tsx      # 会话列表
│   │   │   ├── session-item.tsx      # 单条会话项
│   │   │   ├── chat-window.tsx       # 聊天窗口容器
│   │   │   ├── message-list.tsx      # 消息列表（虚拟滚动）
│   │   │   ├── message-bubble.tsx    # 消息气泡
│   │   │   ├── message-actions.tsx   # 消息操作（编辑、重新生成）
│   │   │   ├── chat-input.tsx        # 输入框组件
│   │   │   ├── model-selector.tsx    # 模型选择下拉框
│   │   │   ├── context-usage-bar.tsx # 上下文用量进度条
│   │   │   └── streaming-indicator.tsx # 流式输出指示器
│   │   ├── characters/
│   │   │   ├── character-grid.tsx    # 角色卡网格布局
│   │   │   ├── character-card.tsx    # 单张角色卡（大封面+底部文案）
│   │   │   ├── cover-image.tsx       # 角色卡封面图组件
│   │   │   ├── character-search.tsx  # 搜索+标签筛选栏
│   │   │   ├── character-form.tsx    # 角色卡编辑表单（管理员）
│   │   │   └── character-import.tsx  # 角色卡导入（管理员）
│   │   ├── worldbooks/
│   │   │   ├── worldbook-list.tsx    # 世界书列表
│   │   │   ├── worldbook-editor.tsx  # 世界书编辑器
│   │   │   ├── entry-list.tsx        # 词条列表
│   │   │   ├── entry-editor.tsx      # 词条编辑（内联/弹窗）
│   │   │   └── worldbook-import.tsx  # 世界书导入
│   │   └── shared/
│   │       ├── confirm-dialog.tsx    # 通用确认弹窗
│   │       ├── empty-state.tsx       # 空状态占位
│   │       ├── loading-spinner.tsx   # 加载指示器
│   │       ├── error-boundary.tsx    # 错误边界
│   │       └── token-badge.tsx       # Token 数量标签
│   │
│   ├── hooks/
│   │   ├── use-chat-stream.ts        # SSE 流式消息处理
│   │   ├── use-auth.ts               # 认证状态
│   │   ├── use-media-query.ts        # 响应式断点检测
│   │   ├── use-debounce.ts           # 防抖（搜索输入）
│   │   └── use-infinite-scroll.ts    # 无限滚动（消息历史）
│   │
│   ├── lib/
│   │   ├── api-client.ts             # API 请求封装（fetch + 拦截器）
│   │   ├── sse-client.ts             # SSE 客户端封装
│   │   ├── png-parser.ts             # SillyTavern PNG tEXt chunk 解析
│   │   ├── markdown.ts               # Markdown 渲染配置
│   │   ├── token-counter.ts          # 客户端 Token 估算
│   │   ├── utils.ts                  # 工具函数（cn、formatDate 等）
│   │   └── constants.ts              # 常量（断点、默认值等）
│   │
│   ├── stores/
│   │   ├── chat-store.ts             # 聊天状态（当前会话、消息、流式状态）
│   │   ├── character-store.ts        # 角色卡筛选/搜索状态
│   │   ├── ui-store.ts               # UI 状态（侧边栏、移动端抽屉）
│   │   └── auth-store.ts             # 用户认证状态
│   │
│   └── types/
│       ├── character.ts              # CharacterCard 类型定义
│       ├── chat.ts                   # ChatSession / ChatMessage 类型
│       ├── worldbook.ts              # WorldBook / WorldBookEntry 类型
│       ├── user.ts                   # User / TokenUsage 类型
│       └── api.ts                    # API 请求/响应通用类型
│
├── prisma/
│   └── schema.prisma
├── tailwind.config.ts
├── next.config.js
├── tsconfig.json
├── package.json
└── .env.local
```

---

## 3. 路由设计

### 3.1 路由表

| 路由 | 组件 | 布局 | 权限 | 说明 |
|------|------|------|------|------|
| `/` | — | — | Public | `redirect('/characters')` |
| `/login` | `LoginPage` | AuthLayout | Public | 登录页（Phase 4） |
| `/characters` | `CharactersPage` | MainLayout | User | 角色卡列表（大封面卡片网格） |
| `/characters/[id]` | `CharacterDetailPage` | MainLayout | User | 角色卡详情页（封面+信息+开始对话） |
| `/characters/[id]/edit` | `CharacterEditPage` | MainLayout | Admin | 角色卡编辑页 |
| `/chat` | `ChatPage` | MainLayout | User | 聊天页（未选择会话） |
| `/chat/[sessionId]` | `ChatSessionPage` | MainLayout | User | 具体聊天会话 |
| `/worldbooks` | `WorldBooksPage` | MainLayout | User | 世界书列表（统一展示，标签区分全局/个人） |
| `/worldbooks/[id]` | `WorldBookEditPage` | MainLayout | User | 世界书编辑（个人世界书仅创建者可编辑；全局世界书仅管理员可编辑，普通用户只读） |
| `/profile` | `ProfilePage` | MainLayout | User | 个人信息 |

### 3.2 布局层级

```
RootLayout (font、ThemeProvider(next-themes)、toast container、query client)
├── AuthLayout (居中卡片布局)
│   └── /login
└── MainLayout (Header[含主题切换按钮] + 可选 Sidebar + 主内容区)
    ├── /characters                      ← 角色卡列表（大封面网格）
    ├── /characters/[id]                 ← 角色卡详情页（封面左+信息右）
    ├── /characters/[id]/edit            ← 角色卡编辑页（管理员）
    ├── /chat + /chat/[sessionId]        ← 聊天页有自己的二栏布局
    ├── /worldbooks + /worldbooks/[id]
    └── /profile
```

### 3.3 导航策略

- **桌面端**：顶部 Header 含 Logo + 导航链接（角色卡 / 聊天 / 世界书 / 个人信息）
- **移动端**：底部 Tab Bar（角色卡 / 聊天 / 更多），"更多" 展开世界书和个人信息入口
- 聊天页面的会话列表在桌面端为左侧固定面板，移动端为从左侧滑出的 Sheet（抽屉）

---

## 4. 核心模块设计

### 4.1 聊天模块（核心 P0）

聊天模块是最复杂的前端模块，涉及流式数据、实时渲染、状态同步等。

#### 4.1.1 流式消息处理（SSE）

采用 `EventSource` / `fetch + ReadableStream` 读取 SSE 流，核心封装在 `use-chat-stream` hook 中。

```typescript
// hooks/use-chat-stream.ts 核心逻辑

interface UseChatStreamOptions {
  sessionId: string;
  onToken: (token: string) => void;
  onComplete: (message: ChatMessage, usage: TokenUsage) => void;
  onError: (error: Error) => void;
}

interface UseChatStreamReturn {
  sendMessage: (content: string) => Promise<void>;
  stopGeneration: () => void;
  regenerate: (messageId: string) => Promise<void>;
  isStreaming: boolean;
}
```

**技术选型：`fetch` + `ReadableStream` 而非 `EventSource`**

理由：
- `EventSource` 仅支持 GET 请求，发送消息需要 POST
- `fetch` 的 `ReadableStream` 可携带自定义 Header（JWT Token）
- 可以通过 `AbortController` 精确取消请求（停止生成）

**流式渲染流程**：

```
用户点击发送
    │
    ▼
[1] 乐观更新：立即将用户消息添加到本地消息列表（临时 ID）
    │
    ▼
[2] 创建空的 assistant 消息占位（显示 loading 指示器）
    │
    ▼
[3] POST /api/chat/sessions/[id]/messages
    │  请求体：{ content: "用户输入" }
    │  响应：SSE 流
    │
    ▼
[4] 逐事件解析 SSE data（参见 frontend-api.md §5.6 SSE 事件类型）
    │  ├── type: "user_message" → 用临时 ID 替换为服务端 ID + tokenCount
    │  ├── delta.content 不为空 → 追加到 assistant 消息内容
    │  ├── finish_reason = "stop" → 标记完成，提取 usage 信息
    │  ├── type: "message_complete" → AI 消息已保存，更新 contextUsage
    │  ├── type: "error" → 保留已生成内容，显示错误提示 + 重试按钮
    │  └── [DONE] → 流结束
    │
    ▼
[5] 每次追加 content 时触发 React 重渲染
    │  使用 requestAnimationFrame 节流，避免逐字符重渲染
    │
    ▼
[6] 完成后更新 Zustand store 中的会话状态
    └── 更新 contextUsage、最新消息等
```

**SSE 解析器实现要点**：

```typescript
// lib/sse-client.ts

async function* parseSSEStream(
  response: Response
): AsyncGenerator<SSEEvent> {
  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = line.slice(6);
        if (data === '[DONE]') return;
        yield JSON.parse(data);
      }
    }
  }
}
```

#### 4.1.2 停止生成

通过 `AbortController` 断开 SSE 连接即可，**无需单独的 stop API**。后端检测到连接断开后自动取消上游推理请求，并保存已生成的部分内容。

```typescript
const abortController = useRef<AbortController | null>(null);

const stopGeneration = () => {
  abortController.current?.abort();
};
```

#### 4.1.3 消息列表性能优化

- **虚拟滚动**：当消息数量超过阈值（如 100 条）时启用虚拟滚动，仅渲染可视区域内的消息气泡
- **Markdown 缓存**：已完成的消息 Markdown 渲染结果通过 `React.memo` + `useMemo` 缓存，避免父组件重渲染导致的重复解析
- **流式消息特殊处理**：正在流式输出的消息不做 memo 缓存，每次 token 追加触发渲染；已完成的消息用 memo 包裹
- **自动滚动**：新消息到达时自动滚动到底部，用户手动上滑时暂停自动滚动，出现"回到底部"按钮

#### 4.1.4 聊天状态管理（Zustand Store）

```typescript
// stores/chat-store.ts

interface ChatStore {
  // 会话列表
  sessions: ChatSession[];
  currentSessionId: string | null;

  // 当前会话消息
  messages: ChatMessage[];

  // 流式状态
  isStreaming: boolean;
  streamingContent: string;   // 正在流式生成的内容（未完成）
  abortController: AbortController | null;

  // 上下文用量
  contextUsage: { usedTokens: number; maxTokens: number } | null;

  // Actions
  setCurrentSession: (id: string) => void;
  addMessage: (msg: ChatMessage) => void;
  appendStreamContent: (token: string) => void;
  finalizeStream: (msg: ChatMessage) => void;
  updateContextUsage: (usage: ContextUsage) => void;
  removeMessage: (id: string) => void;
  editMessage: (id: string, content: string) => void;
}
```

**选择 Zustand 而非 React Context 的原因**：
- 聊天流式输出频繁更新 `streamingContent`，Zustand 的 selector 机制可精确控制重渲染范围
- `message-list` 组件只订阅 `messages`，不会因 `streamingContent` 变化而重渲染整个列表
- `streaming-indicator` 组件只订阅 `isStreaming`
- 当前正在生成的消息气泡订阅 `streamingContent`

### 4.2 角色卡模块

#### 4.2.1 角色卡列表

- **卡片样式**：每张卡片采用**大封面图 + 底部文案**布局。封面占据卡片 ~80% 面积，固定 **3:4 竖向比例**（`aspect-[3/4]`），超出部分裁切（`object-cover`）。封面下方显示角色名称、简短描述、标签（左对齐），以及**相对时间**（右下角，如"2天前"，使用 `timeAgo()` 工具函数计算）
- **布局**：CSS Grid 响应式网格，容器最大宽度 1600px。桌面 5 列，平板 3-4 列，手机 2 列
- **搜索**：输入框使用 300ms debounce，发送 `GET /api/characters?search=xxx&tag=yyy`
- **标签筛选**：顶部 Tag 横向滚动条，支持多选
- **点击行为**：点击卡片跳转到 `/characters/[id]` 详情页（使用 `<Link>` 组件）
- **数据获取**：使用 TanStack Query，支持 `staleTime` 缓存

```typescript
const { data, isLoading } = useQuery({
  queryKey: ['characters', { search, tags }],
  queryFn: () => fetchCharacters({ search, tags }),
  staleTime: 30_000,
});
```

#### 4.2.2 角色卡详情页 (`/characters/[id]`)

独立页面，**不使用 Sheet/Drawer 弹窗**。

**布局结构**：
- **上方**：左右两栏（移动端变为上下）
  - 左侧：封面图（3:4 比例，320-360px 宽）
  - 右侧：角色名称（大标题）、来源+日期、标签、描述、关联世界书、**"开始对话"大按钮**（紫色全宽 h-12）
- **下方**：详情内容板块，每个板块用紫色左竖条标题分隔
  - **角色介绍**：性格 + 场景设定
  - **开场白**：角色的 firstMessage

**不展示的内容**：系统提示词、示例对话、创作者备注在详情页不可见，仅在管理员编辑页可见。

**编辑入口**：页面右上角极淡的铅笔图标（`text-muted-foreground/40`），仅管理员根据角色条件渲染可见，普通用户不感知。

#### 4.2.3 角色卡编辑页 (`/characters/[id]/edit`)

独立页面（非 Dialog 弹窗），仅管理员可访问。复用于"新建"（`/characters/new/edit`）和"编辑"两种场景。

**页面结构**：
- 顶部：返回按钮 + 标题 + 删除/保存按钮
- 封面预览区（宽幅渐变，hover 出现"更换封面"）
- 分段表单：基本信息 → 角色设定 → 对话内容 → 关联世界书 → 创作者备注
- 标签管理：回车添加，点击 X 删除

**PNG 导入**：前端解析 PNG tEXt chunk（`lib/png-parser.ts`），提取 Base64 编码的 JSON，预览后跳转到编辑页预填表单

```typescript
async function parseSillyTavernPng(file: File): Promise<{
  characterData: SillyTavernCharacterV2;
  avatarBlob: Blob;
}> {
  const buffer = await file.arrayBuffer();
  const chunks = parsePngChunks(buffer);
  const textChunk = chunks.find(c => c.keyword === 'chara');
  if (!textChunk) throw new Error('Not a SillyTavern character card');
  const json = JSON.parse(atob(textChunk.text));
  return { characterData: json, avatarBlob: new Blob([buffer], { type: 'image/png' }) };
}
```

### 4.3 世界书模块

世界书分为**全局世界书**（管理员创建，所有用户可见）和**个人世界书**（用户创建，仅自己可见）。

#### 4.3.1 世界书列表页

- **统一列表**：所有可见世界书在同一列表中展示（个人 + 全局混排），不分 Tab
- 表格/卡片混合布局，显示名称、描述、词条数量、总 Token 数
- 每条世界书名称旁用 **Badge 标签**标识类型：`全局`（蓝色）/ `个人`（绿色）
- 全局世界书额外显示关联的角色卡数
- 操作按钮根据权限条件渲染：个人世界书显示编辑/删除；全局世界书普通用户只显示导出，管理员显示编辑/删除
- 顶部可选筛选器：全部 / 仅全局 / 仅个人

```typescript
// 世界书列表查询（一次性拉取所有可见世界书）
const { data } = useQuery({
  queryKey: ['worldbooks', { scope: scopeFilter }],
  queryFn: () => fetchWorldBooks({ scope: scopeFilter }), // scope 为 undefined 时返回全部
  staleTime: 30_000,
});
```

#### 4.3.2 世界书编辑页

编辑页是一个复杂的表单页面，根据权限分为**可编辑模式**和**只读模式**：

- **可编辑**：个人世界书的创建者 或 全局世界书的管理员
- **只读**：普通用户查看全局世界书（隐藏编辑/删除按钮，所有字段 disabled）

```
┌─────────────────────────────────────────────────────┐
│ ← 返回    世界书名称（可编辑）  [个人/全局]  [保存] [导出] │
├─────────────────────────────────────────────────────┤
│ 描述（可编辑文本域）                                     │
│ 总词条数: 24  |  总 Token: 3,842                       │
├─────────────────────────────────────────────────────┤
│ [搜索词条...]                  [+ 新增词条] [批量操作]    │
├─────────────────────────────────────────────────────┤
│ ▼ 精灵族, 精灵, elf    [启用✓]  优先级:10               │
│   内容预览...           Token: 156                     │
│   [展开编辑]                                           │
├─────────────────────────────────────────────────────┤
│ ▼ 魔法学院, 学院         [启用✓]  优先级:5               │
│   ...                                                 │
└─────────────────────────────────────────────────────┘
```

- **词条编辑**：Accordion 折叠面板，展开后内联编辑关键词（Tag Input）、内容（Textarea）、注入位置（Select）、优先级（Number Input）、启用开关（Switch）
- **自动保存**：词条编辑后 2s debounce 自动保存，或手动点击保存
- **Token 计数**：内容变化时通过 `lib/token-counter.ts` 在客户端估算 Token 数，展示在每条词条旁
- **权限判断**：

```typescript
function useWorldBookPermission(worldBook: WorldBook) {
  const { user } = useAuth();
  const canEdit =
    worldBook.scope === 'personal' && worldBook.userId === user?.id
    || worldBook.scope === 'global' && user?.role === 'admin';
  const canDelete = canEdit;
  return { canEdit, canDelete };
}
```

#### 4.3.3 聊天页中的个人世界书选择

用户可以在聊天窗口中为当前会话启用自己的个人世界书，作为全局世界书（由角色卡自动关联）之外的补充。

- **入口**：聊天窗口顶部 Header 中的"世界书"按钮（书本图标 + 已启用数量角标 Badge），位于模型选择器与上下文用量之间
- **交互**：点击按钮弹出 **Dialog 对话框**（非 Popover），标题"个人世界书"，附带说明文字"全局世界书由角色卡自动关联，无需手动管理"
- **列表内容**：展示用户的所有个人世界书，每条显示名称、描述、词条数、Token 数，右侧为 **Switch 开关**。已启用的世界书卡片有蓝色边框高亮（`border-sky-500/30 bg-sky-500/5`）
- **保存**：Switch 切换后即时更新本地状态，同时调用 `PUT /api/chat/sessions/[id]` 更新 `personalWorldBookIds`

### 4.4 用户信息模块

- **基本信息**：用户名、邮箱、注册时间（只读展示）
- **Token 使用统计**：使用 Recharts 绘制折线图/柱状图
  - 时间维度切换：日 / 周 / 月
  - 分类展示：Prompt Tokens vs Completion Tokens
- **会话统计**：总会话数、总消息数

---

## 5. 响应式设计方案

### 5.1 断点定义

| 断点 | 宽度范围 | 对应设备 | TailwindCSS |
|------|----------|----------|-------------|
| `sm` | < 768px | 手机 | 默认 |
| `md` | 768px – 1023px | 平板 | `md:` |
| `lg` | ≥ 1024px | 桌面 | `lg:` |

### 5.2 各页面响应式策略

#### 聊天页面

| 元素 | 手机 (< 768px) | 平板 (768-1023px) | 桌面 (≥ 1024px) |
|------|----------------|-------------------|-----------------|
| 会话列表 | Sheet 抽屉，左上角按钮触发 | 左侧窄面板 (240px) | 左侧固定面板 (300px) |
| 聊天窗口 | 全屏宽 | 剩余空间 | 剩余空间 |
| 输入框 | 底部固定，单行自动扩展 | 底部固定 | 底部固定 |
| 消息气泡 | 最大宽度 75%，头像贴边 | 最大宽度 75%，头像贴边 | 最大宽度 65%，头像贴边 |

#### 角色卡列表页

| 元素 | 手机 (< 768px) | 平板 (768-1023px) | 桌面 (≥ 1024px) |
|------|----------------|-------------------|-----------------|
| 卡片网格 | 2 列 | 3-4 列 | 5 列（max-w-[1600px]） |
| 搜索栏 | 全宽 | 全宽 | 带标签筛选横排 |

#### 角色卡详情页

| 元素 | 手机 (< 768px) | 平板/桌面 (≥ 768px) |
|------|----------------|---------------------|
| 封面+信息 | 纵向排列（封面在上、信息在下） | 左右两栏（封面左 320-360px、信息右 flex-1） |
| "开始对话"按钮 | 全宽大按钮 | 全宽大按钮 |

#### 角色卡编辑页（管理员）

| 元素 | 手机 (< 768px) | 平板/桌面 (≥ 768px) |
|------|----------------|---------------------|
| 表单布局 | 单栏滚动 | 单栏滚动（max-w-3xl 居中） |

#### 世界书编辑页

| 元素 | 手机 | 平板/桌面 |
|------|------|-----------|
| 词条列表 | 全宽 Accordion | 全宽 Accordion |
| 词条编辑 | 全屏弹窗 | 内联展开编辑 |

### 5.3 移动端交互优化

- **触控目标**：所有可点击元素最小 44×44px
- **手势**：会话列表左滑删除（移动端），长按触发操作菜单
- **键盘适配**：输入框聚焦时自动调整视口，避免被虚拟键盘遮挡
- **安全区域**：底部导航和输入框使用 `env(safe-area-inset-bottom)` 适配刘海屏
- **overscroll 处理**：外层容器 `overflow-hidden` 防止整体滚动，Header 使用 `shrink-0` 而非 `sticky`。所有页面内容区滚动容器统一添加 `overscroll-none`（`overscroll-behavior: none`），防止滚动到顶部/底部后的橡皮筋回弹效果穿透到导航栏

---

## 6. 数据流与状态管理

### 6.1 状态分层

```
┌─────────────────────────────────────────────────┐
│  Server State（TanStack Query 管理）              │
│  - 角色卡列表、详情                                │
│  - 世界书列表、详情                                │
│  - 会话列表                                       │
│  - 用户信息、Token 统计                            │
│  - 模型列表                                       │
│  特点：缓存、后台刷新、重试、分页                     │
├─────────────────────────────────────────────────┤
│  Client State（Zustand 管理）                     │
│  - 当前聊天会话的消息列表和流式状态                   │
│  - UI 状态（侧边栏开关、移动端抽屉）                 │
│  - 搜索/筛选条件                                   │
│  特点：高频更新、跨组件共享、不需要持久化              │
├─────────────────────────────────────────────────┤
│  Local State（React useState/useReducer）         │
│  - 表单输入状态                                    │
│  - 组件内部 UI 切换（展开/折叠）                     │
│  特点：单组件内使用，不需要跨组件共享                  │
└─────────────────────────────────────────────────┘
```

### 6.2 数据获取策略

| 数据 | 获取方式 | 缓存策略 |
|------|----------|----------|
| 角色卡列表 | `useQuery` + `GET /api/characters` | `staleTime: 30s` |
| 会话列表 | `useQuery` + `GET /api/chat/sessions` | `staleTime: 10s`，发送消息后 `invalidate` |
| 消息历史 | `useInfiniteQuery`（游标分页）+ `GET /api/chat/sessions/[id]/messages?cursor=&limit=50` | 切换会话时获取最新 50 条，向上滚动时加载更多，缓存在 Zustand |
| 世界书详情 | `useQuery` + `GET /api/worldbooks/[id]` | `staleTime: 60s` |
| 模型列表 | `useQuery` + `GET /api/models` | `staleTime: 5min`，变化低频 |
| Token 统计 | `useQuery` + `GET /api/users/me/usage` | `staleTime: 60s` |

### 6.3 乐观更新场景

| 操作 | 乐观行为 | 失败回滚 |
|------|----------|----------|
| 发送消息 | 立即显示用户消息 | 移除消息，显示错误 Toast |
| 删除会话 | 立即从列表移除 | 还原列表，显示错误 Toast |
| 编辑消息 | 立即更新消息内容 | 还原原内容，显示错误 Toast |
| 启用/禁用词条 | 立即切换开关状态 | 还原状态 |

---

## 7. API 集成层

### 7.1 请求封装

```typescript
// lib/api-client.ts

class ApiClient {
  private baseUrl: string;

  async request<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const token = getAuthToken();
    const res = await fetch(`${this.baseUrl}${endpoint}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
        ...options?.headers,
      },
    });

    if (!res.ok) {
      const error = await res.json().catch(() => ({}));
      throw new ApiError(res.status, error.message || 'Request failed');
    }

    return res.json();
  }

  streamRequest(endpoint: string, body: unknown, signal?: AbortSignal) {
    const token = getAuthToken();
    return fetch(`${this.baseUrl}${endpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify(body),
      signal,
    });
  }
}

export const api = new ApiClient();
```

### 7.2 错误处理策略

| HTTP 状态码 | 前端行为 |
|-------------|----------|
| 401 | 清除 Token，重定向到 `/login` |
| 403 | Toast 提示"权限不足" |
| 404 | 显示 404 页面或 Toast |
| 429 | Toast 提示"请求过于频繁，请稍后再试" |
| 503 | Toast 提示"服务繁忙"，展示排队等待 UI |
| 500 | Toast 提示"服务器错误"，附带重试按钮 |
| 网络错误 | Toast 提示"网络连接失败"，自动重试（TanStack Query 内置） |

### 7.3 SSE 流错误处理

SSE 流中的错误分两类：HTTP 级别错误（非 200 响应）和流内错误事件（`type: "error"`）。

```typescript
try {
  for await (const event of parseSSEStream(response)) {
    if (event.type === 'error') {
      // 流内错误（如推理中途 OOM）
      // 保留已生成的部分内容，显示错误提示 + 重试按钮
      onStreamError(event.error);
      break;
    }
    // 正常处理 user_message / chunk / message_complete
  }
} catch (error) {
  if (error.name === 'AbortError') {
    // 用户主动停止（断开连接），正常结束
    // 后端自动保存已生成部分
  } else {
    // 网络错误或解析错误
    // 保留已生成的部分内容，显示错误提示 + 重试按钮
  }
}
```

---

## 8. 主题与样式方案

### 8.1 设计系统

基于 shadcn/ui 的 CSS Variables 主题系统：

```css
/* 支持 light/dark 两套主题 */
:root {
  --background: 0 0% 100%;
  --foreground: 240 10% 3.9%;
  --primary: 240 5.9% 10%;
  --primary-foreground: 0 0% 98%;
  --muted: 240 4.8% 95.9%;
  --accent: 240 4.8% 95.9%;
  /* ... */
}

.dark {
  --background: 240 10% 3.9%;
  --foreground: 0 0% 98%;
  /* ... */
}
```

- **默认使用亮色主题**，Header 中提供太阳/月亮图标按钮切换亮色/暗色
- 通过 `next-themes`（`ThemeProvider`）管理主题状态，`<html>` 标签添加 `suppressHydrationWarning`
- 品牌色使用 violet（紫色系），"开始对话"按钮等强调元素使用 `bg-violet-600`
- 聊天气泡使用区分色：用户消息（`bg-violet-600` 紫色背景白字）、AI 消息（`bg-muted` 灰色背景）

### 8.2 字体方案

```typescript
// app/layout.tsx
import { Inter } from 'next/font/google';

const inter = Inter({ subsets: ['latin'] });
// 中文回落到系统字体：-apple-system, "PingFang SC", "Microsoft YaHei"
```

### 8.3 动画与过渡

| 场景 | 动画 | 实现方式 |
|------|------|----------|
| 消息出现 | 从下方淡入滑出 | CSS `@keyframes` + `animation` |
| 流式文字 | 光标闪烁效果 | CSS `animation: blink` |
| 侧边栏展开/收起 | 水平滑动 | TailwindCSS `transition-transform` |
| Sheet 抽屉 | 从左/下滑入 + 遮罩 | Radix UI Sheet 内置 |
| 页面切换 | 无额外动画（保持快速） | — |
| 骨架屏 | 闪烁占位 | shadcn Skeleton 组件 |

---

## 9. 认证方案

### 9.1 分阶段实现

**Phase 1（MVP）**：无认证，硬编码默认用户

```typescript
// 所有 API 请求使用默认用户 ID
const DEFAULT_USER_ID = 'default-user';
```

**Phase 4**：完整认证流程

```
登录页面 → POST /api/auth/login → 返回 JWT
    │
    ▼
JWT 存储到 httpOnly Cookie（安全）
    │
    ▼
后续请求自动携带 Cookie
    │
    ▼
服务端 Middleware 验证 JWT，注入 user 信息
```

### 9.2 路由守卫

```typescript
// middleware.ts (Next.js Middleware)

export function middleware(request: NextRequest) {
  const token = request.cookies.get('auth-token');
  const isAuthPage = request.nextUrl.pathname.startsWith('/login');

  if (!token && !isAuthPage) {
    return NextResponse.redirect(new URL('/login', request.url));
  }

  if (token && isAuthPage) {
    return NextResponse.redirect(new URL('/characters', request.url));
  }
}

export const config = {
  matcher: ['/((?!api|_next/static|_next/image|favicon.ico).*)'],
};
```

### 9.3 权限控制

管理员功能入口隐藏设计：

- **角色卡详情页**：编辑入口为页面右上角极淡的铅笔图标（`text-muted-foreground/40`），仅管理员可见，普通用户不感知
- **角色卡列表页**：顶部的"导入"/"新建"按钮仅管理员可见
- **角色卡编辑页**：独立页面 `/characters/[id]/edit`，路由级别做权限校验

```typescript
// 角色卡详情页中的隐藏编辑入口
function AdminEditLink({ characterId }: { characterId: string }) {
  const { user } = useAuth();
  if (user?.role !== 'admin') return null;

  return (
    <Link
      href={`/characters/${characterId}/edit`}
      className="text-muted-foreground/40 hover:text-muted-foreground transition-colors"
      title="编辑"
    >
      <Pencil className="h-3.5 w-3.5" />
    </Link>
  );
}
```

---

## 10. 性能优化策略

### 10.1 首屏加载

| 策略 | 实现 |
|------|------|
| 代码分割 | Next.js App Router 自动按路由分割 |
| 动态导入 | 世界书编辑器、图表组件使用 `next/dynamic` 懒加载 |
| 图片优化 | 角色头像使用 `next/image`，自动 WebP 转换 + 响应式尺寸 |
| 字体优化 | `next/font` 自托管，消除 FOUT |
| 预取 | 导航链接使用 `<Link prefetch>` 预取目标页面 |

### 10.2 运行时性能

| 场景 | 优化手段 |
|------|----------|
| 流式渲染高频更新 | `requestAnimationFrame` 节流，批量更新 DOM |
| 长消息列表 | 虚拟滚动（`@tanstack/react-virtual`），超出 100 条时启用 |
| Markdown 解析 | `React.memo` 包裹已完成消息，避免重复解析 |
| 搜索输入 | 300ms debounce，避免频繁请求 |
| 大型世界书 | 词条列表虚拟化，仅展开的词条加载完整内容 |

### 10.3 Bundle 体积控制

| 策略 | 说明 |
|------|------|
| Tree Shaking | TailwindCSS purge 未使用类名；按需导入 Recharts 组件 |
| 动态导入 | `react-markdown`、`recharts` 等大依赖仅在使用页面加载 |
| 分析工具 | 使用 `@next/bundle-analyzer` 监控包体积 |

---

## 11. 分阶段开发计划

### Phase 1 — MVP Demo（前端部分，约 1 周）

| 任务 | 优先级 | 工时估算 | 说明 |
|------|--------|----------|------|
| 项目初始化 | P0 | 0.5d | Next.js + TailwindCSS + shadcn/ui + TypeScript 脚手架搭建 |
| 基础布局 | P0 | 0.5d | RootLayout、MainLayout、响应式 Header |
| 聊天页面 — 消息列表 | P0 | 1d | 消息气泡组件、Markdown 渲染、自动滚动 |
| 聊天页面 — 输入框 | P0 | 0.5d | 多行输入、Enter/Shift+Enter、发送按钮 |
| 聊天页面 — 流式输出 | P0 | 1d | SSE 客户端、流式渲染、逐 Token 显示 |
| 聊天页面 — 停止生成 | P0 | 0.5d | AbortController 取消 + 后端 stop 接口调用 |
| 联调测试 | P0 | 1d | 与后端 API 联调，修复问题 |

**交付物**：可以输入文字、看到 AI 流式回复、停止生成的最简聊天 UI。

### Phase 2 — 角色卡与会话管理（约 1 周）

| 任务 | 优先级 | 工时估算 | 说明 |
|------|--------|----------|------|
| 角色卡列表页 | P1 | 1d | 大封面卡片网格（5列）、搜索框、标签筛选 |
| 角色卡详情页 | P1 | 0.5d | 独立页面，封面左+信息右布局，角色介绍+开场白 |
| 角色卡编辑页（管理员） | P1 | 1d | 独立页面，完整表单、导入（PNG 解析 + JSON）、删除确认 |
| 会话列表侧边栏 | P1 | 1d | 桌面固定面板 + 移动端 Sheet 抽屉 |
| 重新生成 + 编辑消息 | P1 | 0.5d | 消息操作按钮、编辑态 UI |
| 联调测试 | P1 | 0.5d | — |

### Phase 3 — 世界书与上下文管理（约 1 周）

| 任务 | 优先级 | 工时估算 | 说明 |
|------|--------|----------|------|
| 世界书列表页 | P1 | 0.5d | 列表 + 新建 + 导入/导出 |
| 世界书编辑页 | P1 | 2d | 词条 CRUD、Accordion、内联编辑、Token 统计 |
| 上下文用量进度条 | P1 | 0.5d | 聊天窗口顶部进度条 |
| 模型选择下拉框 | P1 | 0.5d | 获取模型列表、切换模型 |
| 联调测试 | P1 | 0.5d | — |

### Phase 4 — 多用户系统与打磨（约 1 周）

| 任务 | 优先级 | 工时估算 | 说明 |
|------|--------|----------|------|
| 登录/注册页面 | P1 | 1d | 表单 + 校验 + JWT 存储 |
| 路由守卫 | P1 | 0.5d | Middleware 鉴权跳转 |
| 用户信息页 | P1 | 1d | 基本信息 + Token 统计图表（Recharts） |
| UI/UX 打磨 | P1 | 2d | Loading 骨架屏、错误边界、空状态、动画、暗色主题优化 |

---

## 12. 关键技术风险与应对

| 风险 | 影响 | 应对措施 |
|------|------|----------|
| SSE 流式渲染卡顿 | 高频 DOM 更新导致掉帧 | `requestAnimationFrame` 节流 + 只更新流式消息组件 |
| 移动端虚拟键盘遮挡输入框 | 用户体验差 | `visualViewport` API 监听 + 动态调整输入框位置 |
| PNG 解析兼容性 | 部分 SillyTavern 卡片格式不规范 | 容错解析 + 友好的错误提示，支持手动 JSON 导入兜底 |
| 长对话消息列表性能 | 消息过多导致渲染缓慢 | 虚拟滚动 + 消息 memo 缓存 |
| 网络不稳定的流式连接中断 | 部分回复丢失 | 保留已生成内容 + 显示"重试"按钮，不清空已有文字 |
| 世界书编辑页大量词条 | 表单状态复杂 | React Hook Form 的 `useFieldArray` + 虚拟化列表 |

---

## 13. 依赖清单

```json
{
  "dependencies": {
    "next": "^14.2",
    "react": "^18.3",
    "react-dom": "^18.3",
    "tailwindcss": "^3.4",
    "@tanstack/react-query": "^5.0",
    "@tanstack/react-virtual": "^3.0",
    "zustand": "^4.5",
    "react-hook-form": "^7.50",
    "@hookform/resolvers": "^3.3",
    "zod": "^3.22",
    "react-markdown": "^9.0",
    "remark-gfm": "^4.0",
    "rehype-highlight": "^7.0",
    "recharts": "^2.12",
    "next-themes": "^0.3",
    "clsx": "^2.1",
    "tailwind-merge": "^2.2",
    "lucide-react": "^0.350",
    "date-fns": "^3.3"
  },
  "devDependencies": {
    "typescript": "^5.4",
    "@types/react": "^18.3",
    "@types/node": "^20",
    "eslint": "^8.56",
    "eslint-config-next": "^14.2",
    "prettier": "^3.2",
    "prettier-plugin-tailwindcss": "^0.5",
    "@next/bundle-analyzer": "^14.2"
  }
}
```

---

## 14. 编码规范约定

| 类别 | 规范 |
|------|------|
| 文件命名 | 组件 `kebab-case.tsx`，hooks `use-xxx.ts`，工具 `kebab-case.ts` |
| 组件导出 | 命名导出（`export function ComponentName`），不使用默认导出 |
| 样式 | 优先 TailwindCSS 类名，复杂样式用 `cn()` 合并，避免内联 style |
| 状态管理 | Server State → TanStack Query；Client State → Zustand；Local → useState |
| 类型 | 严格模式 (`strict: true`)，API 响应类型通过 Zod schema 推导 |
| 错误处理 | 所有 API 调用包裹 try-catch，用户可见错误通过 Toast 展示 |
| 注释 | 仅注释非显而易见的业务逻辑和技术决策，不注释显而易见的代码 |
