import type { CharacterCard, CharacterSummary } from "@/types/character";
import type { ChatSession, ChatMessage } from "@/types/chat";
import type { WorldBookSummary, WorldBookDetail } from "@/types/worldbook";
import type { User, TokenUsageStats, ModelInfo } from "@/types/user";
import type { PaginatedResponse, ApiResponse, CursorResponse } from "@/types/api";

const delay = (ms: number) => new Promise((r) => setTimeout(r, ms));

// ──── Users ────

const defaultUser: User = {
  id: "usr_default",
  username: "Admin",
  email: "admin@example.com",
  role: "admin",
  createdAt: "2026-02-01T00:00:00.000Z",
};

// ──── Characters ────

const characters: CharacterCard[] = [
  {
    id: "chr_1", name: "艾莉丝", avatar: null, coverImage: null,
    description: "来自星辰学院的天才魔法少女，性格活泼开朗，擅长光系魔法。她总是充满好奇心，喜欢探索未知的魔法领域。",
    personality: "活泼开朗、好奇心强、善良正义、偶尔冒失",
    preset: "",
    scenario: "你在星辰学院的图书馆偶遇了正在研究禁忌魔法的艾莉丝",
    systemPrompt: "你是艾莉丝，一个来自星辰学院的魔法少女...",
    firstMessage: "啊！你、你看到了什么？我只是在研究一些...普通的魔法理论而已！才不是什么禁忌魔法呢！",
    alternateGreetings: [], exampleDialogue: "用户：你在研究什么？\n艾莉丝：嘿嘿~这可是秘密哦！",
    creatorNotes: "适合轻松的奇幻冒险对话", worldBookIds: ["wb_1"],
    tags: ["奇幻", "魔法", "校园", "冒险"], source: "sillytavern_png",
    createdBy: "usr_default", createdAt: "2026-02-15T00:00:00.000Z", updatedAt: "2026-02-15T00:00:00.000Z",
  },
  {
    id: "chr_2", name: "零号侦探", avatar: null, coverImage: null,
    description: "赛博朋克世界中的私人侦探，冷酷寡言但内心正义。左臂为义体改造，拥有超强的数据分析能力。",
    personality: "冷酷理性、寡言少语、正义感强、对弱者温柔",
    preset: "",
    scenario: "雨夜，一个神秘委托人来到你的破旧侦探事务所",
    systemPrompt: "你是零号侦探，赛博朋克世界中的一名私人侦探...",
    firstMessage: "*点燃一根电子烟，烟雾在全息屏幕的蓝光中缓缓升腾*\n\n...坐。说说你的麻烦。",
    alternateGreetings: [], exampleDialogue: "", creatorNotes: "赛博朋克风格硬汉侦探",
    worldBookIds: ["wb_2"], tags: ["赛博朋克", "侦探", "科幻", "硬汉"],
    source: "manual", createdBy: "usr_default",
    createdAt: "2026-02-20T00:00:00.000Z", updatedAt: "2026-02-20T00:00:00.000Z",
  },
  {
    id: "chr_3", name: "小雪", avatar: null, coverImage: null,
    description: "温柔体贴的邻家女孩，在咖啡店打工的大学生。喜欢读书和烘焙，总是带着温暖的笑容。",
    personality: "温柔体贴、善解人意、略带天然呆、喜欢照顾人",
    preset: "",
    scenario: "你是小雪工作的咖啡店的常客",
    systemPrompt: "你是小雪，一个在咖啡店打工的大学生...",
    firstMessage: "欢迎光临~今天的拿铁拉花是猫咪图案哦，要不要试试看？☕",
    alternateGreetings: [], exampleDialogue: "", creatorNotes: "日常温馨对话",
    worldBookIds: [], tags: ["日常", "温馨", "咖啡店", "校园"],
    source: "json_import", createdBy: "usr_default",
    createdAt: "2026-02-22T00:00:00.000Z", updatedAt: "2026-02-22T00:00:00.000Z",
  },
  {
    id: "chr_4", name: "暗影刺客·凯", avatar: null, coverImage: null,
    description: "暗影公会最年轻的大师级刺客，身手敏捷，精通各种暗杀技巧。表面冷酷无情，实则背负着沉重的过去。",
    personality: "沉默寡言、行动果断、表面冷漠内心脆弱",
    preset: "",
    scenario: "你在酒馆中偶遇了一个戴着兜帽的神秘人",
    systemPrompt: "你是暗影刺客凯...",
    firstMessage: "*从阴影中走出，兜帽下露出一双锐利的银色眼瞳*\n\n...你不应该出现在这里。",
    alternateGreetings: [], exampleDialogue: "", creatorNotes: "暗黑奇幻冒险",
    worldBookIds: [], tags: ["奇幻", "暗黑", "刺客", "冒险"],
    source: "sillytavern_png", createdBy: "usr_default",
    createdAt: "2026-02-25T00:00:00.000Z", updatedAt: "2026-02-25T00:00:00.000Z",
  },
  {
    id: "chr_5", name: "Dr. Nova", avatar: null, coverImage: null,
    description: "天才科学家，专注于量子物理与时间理论。在实验室爆炸事故后获得了感知时间流速的能力，性格古怪但充满魅力。",
    personality: "天才型人格、古怪幽默、对科学充满热情、社交能力差",
    preset: "",
    scenario: "你是Dr. Nova的新助手，第一天来到她混乱的实验室",
    systemPrompt: "你是Dr. Nova...",
    firstMessage: "哦！你就是新来的助手？太好了！快帮我扶稳这个——注意别碰那个蓝色按钮，上次有人按了之后我们的咖啡机穿越到了三天前。",
    alternateGreetings: [], exampleDialogue: "", creatorNotes: "科幻喜剧风格",
    worldBookIds: [], tags: ["科幻", "喜剧", "科学家", "时间"],
    source: "manual", createdBy: "usr_default",
    createdAt: "2026-02-28T00:00:00.000Z", updatedAt: "2026-02-28T00:00:00.000Z",
  },
  {
    id: "chr_6", name: "龙族末裔·焰", avatar: null, coverImage: null,
    description: "远古龙族的最后一位后裔，以人类形态生活在现代都市中。拥有操控火焰的能力，正在寻找其他幸存的龙族同胞。",
    personality: "骄傲自信、对龙族历史执着、偶尔展现龙的霸气",
    preset: "",
    scenario: "你在深夜的天台上看到一个周身围绕着微弱火焰的身影",
    systemPrompt: "你是焰，远古龙族的最后一位后裔...",
    firstMessage: "*火焰在指尖跳动，映照出金色的竖瞳*\n\n人类...你倒是胆子不小，居然不跑。",
    alternateGreetings: [], exampleDialogue: "", creatorNotes: "都市奇幻",
    worldBookIds: ["wb_3"], tags: ["奇幻", "都市", "龙族", "热血"],
    source: "sillytavern_png", createdBy: "usr_default",
    createdAt: "2026-03-01T00:00:00.000Z", updatedAt: "2026-03-01T00:00:00.000Z",
  },
];

// ──── Chat Sessions ────

const sessions: (ChatSession & { messages: ChatMessage[] })[] = [
  {
    id: "ses_1", characterId: "chr_1", characterName: "艾莉丝", characterAvatar: null, characterCoverImage: null,
    modelId: "qwen3-32b", title: "图书馆的秘密研究",
    lastMessage: "嘿嘿~那个魔法阵的核心其实是...",
    personalWorldBookIds: [], contextUsage: { usedTokens: 3200, maxTokens: 8192 },
    createdAt: "2026-03-03T06:00:00.000Z", updatedAt: "2026-03-03T06:30:00.000Z",
    messages: [
      { id: "msg_1", role: "assistant", content: "啊！你、你看到了什么？我只是在研究一些...普通的魔法理论而已！才不是什么禁忌魔法呢！\n\n*慌忙把桌上的书翻到背面，但你还是瞥见了封面上的古老符文*", tokenCount: 85, isCompressed: false, createdAt: "2026-03-03T06:00:00.000Z", editedAt: null },
      { id: "msg_2", role: "user", content: "别紧张，我不会告诉别人的。你在研究什么？看起来很有趣。", tokenCount: 32, isCompressed: false, createdAt: "2026-03-03T06:05:00.000Z", editedAt: null },
      { id: "msg_3", role: "assistant", content: "真、真的吗？你不会觉得奇怪？\n\n*小心翼翼地把书翻回正面*\n\n其实...我在研究一种失传的古代魔法——**时空折叠术**。传说中，星辰学院的创始人就是靠这种魔法穿越了次元壁垒才来到这个世界的！\n\n但是教授们说这只是传说，不让我们研究。可是你看这个魔法阵的结构...\n\n*眼睛闪闪发光地指着书上复杂的魔法阵*\n\n它和我们现在使用的空间转移魔法有 **73%** 的结构相似度！这不可能只是巧合！", tokenCount: 180, isCompressed: false, createdAt: "2026-03-03T06:06:00.000Z", editedAt: null },
      { id: "msg_4", role: "user", content: "73%的相似度？你是怎么计算出来的？", tokenCount: 20, isCompressed: false, createdAt: "2026-03-03T06:10:00.000Z", editedAt: null },
      { id: "msg_5", role: "assistant", content: "嘿嘿~那个魔法阵的核心其实是由 **十二个基本符文节点** 构成的。我把每个节点的魔力流向、连接方式和共振频率都做了量化分析！\n\n*从书包里掏出一叠写满密密麻麻公式的笔记*\n\n看！我用了魔力波动谱分析法。虽然时空折叠术的完整阵法已经失传了，但根据残存的文献碎片，我推算出了其中八个节点的参数——\n\n等等，你真的感兴趣吗？上次我跟室友说这些，她直接睡着了...", tokenCount: 195, isCompressed: false, createdAt: "2026-03-03T06:12:00.000Z", editedAt: null },
    ],
  },
  {
    id: "ses_2", characterId: "chr_2", characterName: "零号侦探", characterAvatar: null, characterCoverImage: null,
    modelId: "qwen3-32b", title: "雨夜委托",
    lastMessage: "...这件事比你想的要复杂得多。",
    personalWorldBookIds: [], contextUsage: { usedTokens: 1800, maxTokens: 8192 },
    createdAt: "2026-03-02T14:00:00.000Z", updatedAt: "2026-03-02T14:15:00.000Z",
    messages: [
      { id: "msg_6", role: "assistant", content: "*点燃一根电子烟，烟雾在全息屏幕的蓝光中缓缓升腾*\n\n...坐。说说你的麻烦。", tokenCount: 45, isCompressed: false, createdAt: "2026-03-02T14:00:00.000Z", editedAt: null },
      { id: "msg_7", role: "user", content: "我妹妹三天前失踪了，警方说是自愿离开，但我知道不是。", tokenCount: 28, isCompressed: false, createdAt: "2026-03-02T14:02:00.000Z", editedAt: null },
      { id: "msg_8", role: "assistant", content: "*义体左臂的指尖微微闪烁，数据流在视网膜投影中快速滚动*\n\n三天。在这座城市，三天足够让一个人消失五次。\n\n...你妹妹在哪里工作？最后一次联系是什么时候？\n\n*从抽屉里拿出一个老旧的数据板*\n\n还有——这件事比你想的要复杂得多。第六区最近有一系列类似的失踪案件。全是年轻女性，全是「自愿离开」。\n\n巧合？在我的行当里，没有巧合。", tokenCount: 160, isCompressed: false, createdAt: "2026-03-02T14:03:00.000Z", editedAt: null },
    ],
  },
  {
    id: "ses_3", characterId: "chr_3", characterName: "小雪", characterAvatar: null, characterCoverImage: null,
    modelId: "qwen3-32b", title: "午后咖啡时光",
    lastMessage: "明天我试试做抹茶蛋糕，给你留一块~",
    personalWorldBookIds: ["wb_4"], contextUsage: { usedTokens: 900, maxTokens: 8192 },
    createdAt: "2026-03-01T08:30:00.000Z", updatedAt: "2026-03-01T08:45:00.000Z",
    messages: [
      { id: "msg_9", role: "assistant", content: "欢迎光临~今天的拿铁拉花是猫咪图案哦，要不要试试看？☕", tokenCount: 30, isCompressed: false, createdAt: "2026-03-01T08:30:00.000Z", editedAt: null },
      { id: "msg_10", role: "user", content: "来一杯！对了小雪，今天有新的甜点吗？", tokenCount: 18, isCompressed: false, createdAt: "2026-03-01T08:32:00.000Z", editedAt: null },
      { id: "msg_11", role: "assistant", content: "有的有的！今天我自己做了蓝莓芝士蛋糕~嘿嘿，虽然卖相不是特别好看，但味道绝对没问题！\n\n店长说如果客人反馈好的话，可以加入正式菜单呢。所以...你愿意当我的第一个试吃员吗？🎂\n\n明天我试试做抹茶蛋糕，给你留一块~", tokenCount: 98, isCompressed: false, createdAt: "2026-03-01T08:33:00.000Z", editedAt: null },
    ],
  },
];

// ──── World Books ────

const worldBooks: WorldBookDetail[] = [
  { id: "wb_1", name: "星辰学院设定集", description: "包含星辰学院的历史、地理、魔法体系、重要人物等详细设定", scope: "global", userId: "usr_default", totalTokenCount: 4280, characterCount: 2, createdAt: "2026-02-10T00:00:00.000Z", updatedAt: "2026-02-10T00:00:00.000Z", entries: [
    { id: "e_1", keywords: ["星辰学院", "学院"], secondaryKeywords: [], content: "星辰学院是大陆上最负盛名的魔法学府，坐落于星辰山脉的最高峰——银月峰上。", position: "after_system", priority: 10, enabled: true, tokenCount: 120 },
    { id: "e_2", keywords: ["光系魔法", "光魔法"], secondaryKeywords: [], content: "光系魔法是星辰学院六大魔法系之一，以治愈、防护和净化为主要方向。", position: "after_system", priority: 8, enabled: true, tokenCount: 95 },
    { id: "e_3", keywords: ["禁忌魔法", "时空折叠"], secondaryKeywords: [], content: "时空折叠术是学院创始人塞拉斯·星辰的独创魔法，据说可以折叠时空、穿越次元。", position: "after_system", priority: 10, enabled: true, tokenCount: 108 },
  ]},
  { id: "wb_2", name: "赛博朋克·新东京", description: "赛博朋克世界观设定，包含城市结构、科技水平、社会阶层等信息", scope: "global", userId: "usr_default", totalTokenCount: 3150, characterCount: 1, createdAt: "2026-02-18T00:00:00.000Z", updatedAt: "2026-02-18T00:00:00.000Z", entries: [
    { id: "e_6", keywords: ["新东京", "城市"], secondaryKeywords: [], content: "新东京是建立在旧东京废墟上的超级都市。", position: "after_system", priority: 10, enabled: true, tokenCount: 95 },
  ]},
  { id: "wb_3", name: "龙族编年史", description: "远古龙族的历史、血脉、能力与现代龙裔设定", scope: "global", userId: "usr_default", totalTokenCount: 1820, characterCount: 1, createdAt: "2026-02-25T00:00:00.000Z", updatedAt: "2026-02-25T00:00:00.000Z", entries: [
    { id: "e_9", keywords: ["龙族", "远古龙"], secondaryKeywords: [], content: "龙族是这个世界最古老的种族。", position: "after_system", priority: 10, enabled: true, tokenCount: 85 },
  ]},
  { id: "wb_4", name: "我的角色扮演偏好", description: "个人定制的对话风格偏好，让AI的回复更符合我的口味", scope: "personal", userId: "usr_default", totalTokenCount: 520, characterCount: 0, createdAt: "2026-03-01T00:00:00.000Z", updatedAt: "2026-03-01T00:00:00.000Z", entries: [
    { id: "e_11", keywords: ["对话风格"], secondaryKeywords: [], content: "请使用更文学化的语言风格。", position: "after_system", priority: 8, enabled: true, tokenCount: 65 },
  ]},
  { id: "wb_5", name: "自定义魔法体系补充", description: "我自己补充的魔法体系细节", scope: "personal", userId: "usr_default", totalTokenCount: 380, characterCount: 0, createdAt: "2026-03-02T00:00:00.000Z", updatedAt: "2026-03-02T00:00:00.000Z", entries: [
    { id: "e_13", keywords: ["暗系魔法"], secondaryKeywords: [], content: "暗系魔法并非邪恶的代名词。", position: "after_system", priority: 7, enabled: true, tokenCount: 78 },
  ]},
];

const models: ModelInfo[] = [
  { id: "qwen3-32b", name: "Qwen3 32B (FP8)", maxContextLength: 8192, status: "online" },
  { id: "qwen3-8b", name: "Qwen3 8B (FP8)", maxContextLength: 8192, status: "offline" },
];

const tokenUsage: TokenUsageStats = {
  summary: { totalPromptTokens: 194900, totalCompletionTokens: 129400, totalTokens: 324300, totalSessions: 3, totalMessages: 12 },
  timeline: [
    { date: "2026-02-20", promptTokens: 12500, completionTokens: 8300, totalTokens: 20800 },
    { date: "2026-02-21", promptTokens: 15200, completionTokens: 10100, totalTokens: 25300 },
    { date: "2026-02-22", promptTokens: 8900, completionTokens: 5600, totalTokens: 14500 },
    { date: "2026-02-23", promptTokens: 22100, completionTokens: 14800, totalTokens: 36900 },
    { date: "2026-02-24", promptTokens: 18600, completionTokens: 12400, totalTokens: 31000 },
    { date: "2026-02-25", promptTokens: 9800, completionTokens: 6500, totalTokens: 16300 },
    { date: "2026-02-26", promptTokens: 14300, completionTokens: 9500, totalTokens: 23800 },
    { date: "2026-02-27", promptTokens: 20500, completionTokens: 13600, totalTokens: 34100 },
    { date: "2026-02-28", promptTokens: 16800, completionTokens: 11200, totalTokens: 28000 },
    { date: "2026-03-01", promptTokens: 25600, completionTokens: 17000, totalTokens: 42600 },
    { date: "2026-03-02", promptTokens: 19200, completionTokens: 12800, totalTokens: 32000 },
    { date: "2026-03-03", promptTokens: 11400, completionTokens: 7600, totalTokens: 19000 },
  ],
};

// ──── Streaming Demo Text ────

const streamingDemoText = `*翻开笔记本，指着一行复杂的魔法方程式*

你看这里！时空折叠术的核心原理其实和我们平时用的空间转移魔法是相通的——都是通过扭曲空间的"曲率"来实现位移。

但关键的区别在于，时空折叠术不仅仅扭曲空间，它还会影响 **时间流速**。

简单来说：
1. 空间转移 = 折叠空间的两个点使它们重合
2. 时空折叠 = 折叠时空的两个点，同时改变时间坐标

所以理论上，如果掌握了时空折叠术，不仅可以瞬间移动，甚至可以回到过去或者去往未来！

但问题是...折叠时间需要的魔力是折叠空间的 **指数级** 增长。

...所以创始人塞拉斯到底是怎么做到的呢？这才是真正的谜题。`;

// ──── Mock API Implementation ────

export const mockApi = {
  // Auth
  async getMe(): Promise<ApiResponse<User>> {
    await delay(100);
    return { data: defaultUser };
  },

  // Characters
  async getCharacters(params?: { search?: string; tag?: string[] }): Promise<PaginatedResponse<CharacterSummary>> {
    await delay(200);
    let filtered = characters;
    if (params?.search) {
      const s = params.search.toLowerCase();
      filtered = filtered.filter((c) => c.name.toLowerCase().includes(s) || c.description.toLowerCase().includes(s));
    }
    if (params?.tag?.length) {
      filtered = filtered.filter((c) => params.tag!.some((t) => c.tags.includes(t)));
    }
    const summaries: CharacterSummary[] = filtered.map(({ id, name, avatar, coverImage, description, personality, tags, source, createdAt, updatedAt }) => ({ id, name, avatar, coverImage, description, personality, tags, source, createdAt, updatedAt }));
    return { data: summaries, pagination: { page: 1, pageSize: 20, total: summaries.length, totalPages: 1 } };
  },

  async getCharacter(id: string): Promise<ApiResponse<CharacterCard>> {
    await delay(150);
    const c = characters.find((c) => c.id === id);
    if (!c) throw new Error("Not found");
    return { data: c };
  },

  async getTags(): Promise<ApiResponse<string[]>> {
    await delay(100);
    const tags = Array.from(new Set(characters.flatMap((c) => c.tags)));
    return { data: tags };
  },

  async createCharacter(data: Omit<CharacterCard, "id" | "createdBy" | "createdAt" | "updatedAt">): Promise<ApiResponse<CharacterCard>> {
    await delay(300);
    const now = new Date().toISOString();
    const card: CharacterCard = {
      ...data,
      id: `chr_${Date.now()}`,
      createdBy: "usr_default",
      createdAt: now,
      updatedAt: now,
    };
    characters.unshift(card);
    return { data: card };
  },

  async updateCharacter(id: string, data: Partial<CharacterCard>): Promise<ApiResponse<CharacterCard>> {
    await delay(300);
    const idx = characters.findIndex((c) => c.id === id);
    if (idx === -1) throw new Error("Not found");
    characters[idx] = { ...characters[idx], ...data, updatedAt: new Date().toISOString() };
    return { data: characters[idx] };
  },

  async deleteCharacter(id: string): Promise<ApiResponse<{ deleted: true }>> {
    await delay(200);
    const idx = characters.findIndex((c) => c.id === id);
    if (idx !== -1) characters.splice(idx, 1);
    return { data: { deleted: true } };
  },

  // Chat
  async getSessions(): Promise<PaginatedResponse<ChatSession>> {
    await delay(150);
    const list: ChatSession[] = sessions.map(({ messages, ...s }) => s);
    return { data: list, pagination: { page: 1, pageSize: 50, total: list.length, totalPages: 1 } };
  },

  async getMessages(sessionId: string): Promise<CursorResponse<ChatMessage>> {
    await delay(150);
    const s = sessions.find((s) => s.id === sessionId);
    return { data: s?.messages || [], hasMore: false, nextCursor: null };
  },

  async createSession(characterId: string): Promise<ApiResponse<ChatSession & { messages: ChatMessage[] }>> {
    await delay(200);
    const char = characters.find((c) => c.id === characterId)!;
    const msg: ChatMessage = {
      id: `msg_${Date.now()}`, role: "assistant", content: char.firstMessage,
      tokenCount: Math.ceil(char.firstMessage.length / 2), isCompressed: false,
      createdAt: new Date().toISOString(), editedAt: null,
    };
    const session = {
      id: `ses_${Date.now()}`, characterId, characterName: char.name, characterAvatar: null, characterCoverImage: char.coverImage,
      modelId: "qwen3-32b", title: "新会话", lastMessage: char.firstMessage.slice(0, 100),
      personalWorldBookIds: [] as string[], contextUsage: { usedTokens: 0, maxTokens: 8192 },
      createdAt: new Date().toISOString(), updatedAt: new Date().toISOString(),
      messages: [msg],
    };
    sessions.unshift(session);
    return { data: session };
  },

  getStreamingDemoText(): string {
    return streamingDemoText;
  },

  pushMessage(sessionId: string, message: ChatMessage): void {
    const s = sessions.find((s) => s.id === sessionId);
    if (s) {
      s.messages.push(message);
      s.lastMessage = message.content.slice(0, 50);
      s.updatedAt = new Date().toISOString();
    }
  },

  async updateSession(
    sessionId: string,
    input: { modelId?: string; personalWorldBookIds?: string[]; title?: string },
  ): Promise<ApiResponse<ChatSession>> {
    await delay(100);
    const s = sessions.find((s) => s.id === sessionId);
    if (!s) throw new Error("Not found");
    if (input.modelId !== undefined) s.modelId = input.modelId;
    if (input.personalWorldBookIds !== undefined) s.personalWorldBookIds = input.personalWorldBookIds;
    if (input.title !== undefined) s.title = input.title;
    s.updatedAt = new Date().toISOString();
    const { messages, ...rest } = s;
    return { data: rest };
  },

  // World Books
  async getWorldBooks(scope?: string): Promise<PaginatedResponse<WorldBookSummary>> {
    await delay(150);
    let filtered = worldBooks;
    if (scope) filtered = filtered.filter((b) => b.scope === scope);
    const summaries: WorldBookSummary[] = filtered.map((b) => ({
      ...b, entryCount: b.entries.length, entries: undefined as never,
    }));
    return { data: summaries, pagination: { page: 1, pageSize: 50, total: summaries.length, totalPages: 1 } };
  },

  async getWorldBook(id: string): Promise<ApiResponse<WorldBookDetail>> {
    await delay(150);
    const b = worldBooks.find((b) => b.id === id);
    if (!b) throw new Error("Not found");
    return { data: b };
  },

  // Models
  async getModels(): Promise<ApiResponse<ModelInfo[]>> {
    await delay(100);
    return { data: models };
  },

  // Usage
  async getUsage(): Promise<ApiResponse<TokenUsageStats>> {
    await delay(200);
    return { data: tokenUsage };
  },
};
