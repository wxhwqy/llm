"use client";

import { useState, useRef, useEffect, useCallback, use } from "react";
import {
  PanelLeftOpen,
  Plus,
  Send,
  Square,
  RotateCcw,
  Pencil,
  Check,
  X,
  User,
  Zap,
  BookOpen,
  Settings2,
  Trash2,
  Loader2,
  SlidersHorizontal,
  RotateCw,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Sheet,
  SheetContent,
  SheetTrigger,
  SheetTitle,
} from "@/components/ui/sheet";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { cn } from "@/lib/utils";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { getAvatarColor, timeAgo } from "@/lib/constants";
import { useChatStore } from "@/stores/chat-store";
import { useChatStream } from "@/hooks/use-chat-stream";
import {
  useSessions,
  useMessages,
  useModels,
  useWorldBooks,
  useDeleteSessions,
} from "@/hooks/use-queries";
import { Checkbox } from "@/components/ui/checkbox";
import { useUiStore } from "@/stores/ui-store";
import { useAuthGuard } from "@/hooks/use-require-auth";
import { api } from "@/lib/api-client";
import type { ChatMessage, SamplingParams } from "@/types/chat";
import type { ChatSession } from "@/types/chat";

/* ------------------------------------------------------------------ */
/*  CharacterAvatar                                                   */
/* ------------------------------------------------------------------ */

function CharacterAvatar({
  characterId,
  characterName,
  coverImage,
  size = "sm",
  clickable = false,
}: {
  characterId: string;
  characterName: string;
  coverImage?: string | null;
  size?: "sm" | "md";
  clickable?: boolean;
}) {
  const router = useRouter();
  const dim = size === "md" ? "h-9 w-9" : "h-8 w-8";

  const avatar = coverImage ? (
    <img
      src={coverImage}
      alt={characterName}
      className={cn(dim, "rounded-full object-cover shrink-0", clickable && "cursor-pointer")}
      onClick={clickable ? () => router.push(`/characters/${characterId}`) : undefined}
    />
  ) : (
    <div
      className={cn(
        dim,
        "rounded-full flex items-center justify-center text-xs font-bold text-white shrink-0",
        getAvatarColor(characterId),
        clickable && "cursor-pointer",
      )}
      onClick={clickable ? () => router.push(`/characters/${characterId}`) : undefined}
    >
      {characterName?.[0]}
    </div>
  );

  return avatar;
}

/* ------------------------------------------------------------------ */
/*  SessionList                                                       */
/* ------------------------------------------------------------------ */

function SessionList({
  sessions,
  currentId,
  onSelect,
}: {
  sessions: ChatSession[];
  currentId: string;
  onSelect?: () => void;
}) {
  const router = useRouter();
  const [managing, setManaging] = useState(false);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const deleteMutation = useDeleteSessions();

  const toggleSelect = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const handleConfirm = async () => {
    if (selected.size === 0) {
      setManaging(false);
      return;
    }
    await deleteMutation.mutateAsync([...selected]);
    const deletedCurrent = selected.has(currentId);
    setSelected(new Set());
    setManaging(false);
    if (deletedCurrent) {
      router.replace("/chat");
    }
  };

  const handleCancel = () => {
    setSelected(new Set());
    setManaging(false);
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between p-3 border-b">
        <h2 className="font-semibold text-sm">会话列表</h2>
        <div className="flex items-center gap-1">
          {managing ? (
            <>
              <Button
                variant="ghost"
                size="sm"
                className="h-7 px-2 text-xs text-muted-foreground"
                onClick={handleCancel}
                disabled={deleteMutation.isPending}
              >
                取消
              </Button>
              <Button
                variant="destructive"
                size="sm"
                className="h-7 px-2 text-xs"
                onClick={handleConfirm}
                disabled={selected.size === 0 || deleteMutation.isPending}
              >
                {deleteMutation.isPending ? (
                  <Loader2 className="h-3 w-3 animate-spin mr-1" />
                ) : (
                  <Trash2 className="h-3 w-3 mr-1" />
                )}
                删除{selected.size > 0 ? ` (${selected.size})` : ""}
              </Button>
            </>
          ) : (
            <>
              <Button
                variant="ghost"
                size="icon"
                className="h-7 w-7"
                onClick={() => setManaging(true)}
                title="管理"
              >
                <Settings2 className="h-4 w-4" />
              </Button>
              <Link href="/characters">
                <Button variant="ghost" size="icon" className="h-7 w-7">
                  <Plus className="h-4 w-4" />
                </Button>
              </Link>
            </>
          )}
        </div>
      </div>
      <ScrollArea className="flex-1 min-h-0 overscroll-none">
        <div className="p-2 space-y-1">
          {sessions.map((s) => {
            const isSelected = selected.has(s.id);
            const content = (
              <div
                className={cn(
                  "flex items-start gap-3 rounded-lg p-2.5 transition-colors text-left w-full",
                  managing
                    ? isSelected
                      ? "bg-destructive/10"
                      : "hover:bg-accent/50"
                    : s.id === currentId
                      ? "bg-accent"
                      : "hover:bg-accent/50",
                )}
              >
                {managing && (
                  <Checkbox
                    checked={isSelected}
                    className="mt-1 shrink-0"
                    onCheckedChange={() => toggleSelect(s.id)}
                  />
                )}
                <CharacterAvatar
                  characterId={s.characterId}
                  characterName={s.characterName}
                  coverImage={s.characterCoverImage}
                  size="md"
                />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium truncate">
                      {s.characterName}
                    </span>
                    <span className="text-[10px] text-muted-foreground shrink-0 ml-2">
                      {timeAgo(s.updatedAt)}
                    </span>
                  </div>
                  <p className="text-xs text-muted-foreground truncate mt-0.5">
                    {s.lastMessage}
                  </p>
                </div>
              </div>
            );

            if (managing) {
              return (
                <div
                  key={s.id}
                  className="cursor-pointer"
                  onClick={() => toggleSelect(s.id)}
                >
                  {content}
                </div>
              );
            }

            return (
              <Link
                key={s.id}
                href={`/chat/${s.id}`}
                onClick={onSelect}
              >
                {content}
              </Link>
            );
          })}
        </div>
      </ScrollArea>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  MessageBubble                                                     */
/* ------------------------------------------------------------------ */

function MessageBubble({
  message,
  characterName,
  characterId,
  coverImage,
  isStreaming,
  onEdit,
  onRegenerate,
}: {
  message: ChatMessage;
  characterName: string;
  characterId: string;
  coverImage?: string | null;
  isStreaming?: boolean;
  onEdit?: (id: string, content: string) => void;
  onRegenerate?: (id: string) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [editContent, setEditContent] = useState(message.content);
  const isUser = message.role === "user";

  return (
    <div
      className={cn(
        "flex gap-3 group w-full",
        isUser ? "flex-row-reverse" : "flex-row",
      )}
    >
      {isUser ? (
        <div className="h-8 w-8 rounded-full bg-violet-600 flex items-center justify-center shrink-0">
          <User className="h-4 w-4 text-white" />
        </div>
      ) : (
        <CharacterAvatar
          characterId={characterId}
          characterName={characterName}
          coverImage={coverImage}
          clickable
        />
      )}

      <div
        className={cn(
          "max-w-[75%] lg:max-w-[65%] space-y-1 flex flex-col",
          isUser ? "items-end" : "items-start",
        )}
      >
        <div
          className={cn(
            "rounded-2xl px-4 py-2.5 text-sm leading-relaxed whitespace-pre-wrap",
            isUser
              ? "bg-violet-600 text-white rounded-tr-sm"
              : "bg-muted rounded-tl-sm",
          )}
        >
          {editing ? (
            <div className="space-y-2">
              <textarea
                className="w-full bg-transparent border rounded p-2 text-sm min-h-[80px] resize-none focus:outline-none"
                value={editContent}
                onChange={(e) => setEditContent(e.target.value)}
              />
              <div className="flex gap-1 justify-end">
                <Button
                  size="sm"
                  variant="ghost"
                  className="h-7 px-2"
                  onClick={() => setEditing(false)}
                >
                  <X className="h-3 w-3" />
                </Button>
                <Button
                  size="sm"
                  className="h-7 px-2"
                  onClick={() => {
                    onEdit?.(message.id, editContent);
                    setEditing(false);
                  }}
                >
                  <Check className="h-3 w-3" />
                </Button>
              </div>
            </div>
          ) : (
            <>
              {message.content}
              {isStreaming && (
                <span className="inline-block w-0.5 h-4 bg-foreground ml-0.5 animate-pulse" />
              )}
            </>
          )}
        </div>

        {!editing && !isStreaming && (
          <div
            className={cn(
              "flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity",
              isUser ? "justify-end" : "justify-start",
            )}
          >
            {isUser && (
              <Button
                variant="ghost"
                size="sm"
                className="h-6 px-1.5 text-xs text-muted-foreground"
                onClick={() => {
                  setEditContent(message.content);
                  setEditing(true);
                }}
              >
                <Pencil className="h-3 w-3 mr-1" />
                编辑
              </Button>
            )}
            {!isUser && (
              <Button
                variant="ghost"
                size="sm"
                className="h-6 px-1.5 text-xs text-muted-foreground"
                onClick={() => onRegenerate?.(message.id)}
              >
                <RotateCcw className="h-3 w-3 mr-1" />
                重新生成
              </Button>
            )}
            {message.tokenCount > 0 && (
              <span className="text-[10px] text-muted-foreground/50 self-center ml-1">
                {message.tokenCount} tokens
              </span>
            )}
          </div>
        )}
      </div>

      <div className="flex-1" />
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Page                                                              */
/* ------------------------------------------------------------------ */

export default function ChatSessionPage({
  params,
}: {
  params: Promise<{ sessionId: string }>;
}) {
  const { sessionId } = use(params);

  /* ---- auth guard ---- */
  const { user: authUser, isLoading: authLoading } = useAuthGuard();

  /* ---- data fetching ---- */
  const { data: sessionsData } = useSessions();
  const { data: messagesData } = useMessages(sessionId);
  const { data: modelsData } = useModels();
  const { data: worldbooksData } = useWorldBooks("personal");

  const router = useRouter();
  const sessions = sessionsData?.data ?? [];
  const session = sessions.find((s) => s.id === sessionId);
  const modelList = modelsData?.data ?? [];
  const personalBooks = worldbooksData?.data ?? [];

  /* If sessions loaded but current session not found, redirect to /chat */
  useEffect(() => {
    if (sessionsData && !session) {
      router.replace("/chat");
    }
  }, [sessionsData, session, router]);

  /* ---- stores ---- */
  const messages = useChatStore((s) => s.messages);
  const isStreaming = useChatStore((s) => s.isStreaming);
  const streamingContent = useChatStore((s) => s.streamingContent);
  const sidebarOpen = useUiStore((s) => s.sidebarOpen);
  const setSidebarOpen = useUiStore((s) => s.setSidebarOpen);

  /* ---- local state ---- */
  const [input, setInput] = useState("");
  const [model, setModel] = useState("");
  const [worldbookOpen, setWorldbookOpen] = useState(false);
  const [enabledWorldbooks, setEnabledWorldbooks] = useState<string[]>([]);
  const [contextUsage, setContextUsage] = useState({
    usedTokens: 0,
    maxTokens: 0,
  });
  const [sampling, setSampling] = useState<SamplingParams>({
    temperature: null,
    topP: null,
    topK: null,
  });

  /* ---- chat stream ---- */
  const { sendMessage, stopGeneration, regenerate } =
    useChatStream(sessionId, setContextUsage);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  /* cleanup on session switch: reset UI (don't abort — let stream finish in background) */
  useEffect(() => {
    useChatStore.getState().resetStream();
  }, [sessionId]);

  /* sync server messages → store */
  useEffect(() => {
    if (messagesData?.data) {
      useChatStore.getState().setMessages(messagesData.data);
    }
  }, [messagesData]);

  /* sync session metadata → local state */
  useEffect(() => {
    if (!session) return;
    setModel(session.modelId);
    setEnabledWorldbooks(session.personalWorldBookIds);
    setContextUsage(session.contextUsage);
    if (session.samplingParams) {
      setSampling(session.samplingParams);
    }
  }, [session]);

  /* auto-scroll on new content */
  useEffect(() => {
    scrollToBottom();
  }, [messages, streamingContent, scrollToBottom]);

  /* ---- handlers ---- */

  const updateSession = useCallback(
    async (patch: {
      modelId?: string;
      personalWorldBookIds?: string[];
      temperature?: number | null;
      topP?: number | null;
      topK?: number | null;
    }) => {
      await api.put(`/chat/sessions/${sessionId}`, patch);
    },
    [sessionId],
  );

  const handleModelChange = useCallback(
    (newModel: string) => {
      setModel(newModel);
      updateSession({ modelId: newModel });
    },
    [updateSession],
  );

  const toggleWorldbook = useCallback(
    (id: string) => {
      setEnabledWorldbooks((prev) => {
        const next = prev.includes(id)
          ? prev.filter((x) => x !== id)
          : [...prev, id];
        updateSession({ personalWorldBookIds: next });
        return next;
      });
    },
    [updateSession],
  );

  const handleSend = () => {
    if (!input.trim() || isStreaming) return;
    const content = input.trim();
    setInput("");
    sendMessage(sessionId, content);
  };

  const handleStop = () => {
    stopGeneration();
  };

  const handleRegenerate = () => {
    regenerate(sessionId);
  };

  const handleEdit = async (id: string, content: string) => {
    // Optimistically update local store
    useChatStore.getState().editMessage(id, content);
    try {
      // Persist to backend
      await api.put(`/chat/sessions/${sessionId}/messages/${id}`, { content });
    } catch {
      // Revert on failure — reload from server
      const fresh = await api.get(`/chat/sessions/${sessionId}/messages`);
      useChatStore.getState().setMessages(fresh.data);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  /* Resolve actual context limit: prefer the model's maxContextLength over the
     hardcoded session maxTokens (which defaults to 8192) */
  const currentModel = modelList.find((m) => m.id === model);
  const effectiveMaxTokens = currentModel?.maxContextLength ?? contextUsage.maxTokens;
  const usagePercent =
    effectiveMaxTokens > 0
      ? (contextUsage.usedTokens / effectiveMaxTokens) * 100
      : 0;

  const characterName = session?.characterName ?? "";
  const characterId = session?.characterId ?? "";

  /* ---- render ---- */

  return (
    <div className="flex h-full">
      {/* Desktop sidebar */}
      <aside className="hidden md:flex w-72 lg:w-80 border-r flex-col shrink-0">
        <SessionList sessions={sessions} currentId={sessionId} />
      </aside>

      {/* Chat area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* ---- Header ---- */}
        <div className="flex items-center gap-3 px-4 py-2.5 border-b shrink-0">
          {/* Mobile sidebar trigger */}
          <Sheet open={sidebarOpen} onOpenChange={setSidebarOpen}>
            <SheetTrigger asChild className="md:hidden">
              <Button variant="ghost" size="icon" className="h-8 w-8">
                <PanelLeftOpen className="h-4 w-4" />
              </Button>
            </SheetTrigger>
            <SheetContent side="left" className="w-72 p-0">
              <SheetTitle className="sr-only">会话列表</SheetTitle>
              <SessionList
                sessions={sessions}
                currentId={sessionId}
                onSelect={() => setSidebarOpen(false)}
              />
            </SheetContent>
          </Sheet>

          {characterName && (
            <>
              <CharacterAvatar
                characterId={characterId}
                characterName={characterName}
                coverImage={session?.characterCoverImage}
                clickable
              />
              <div className="flex-1 min-w-0">
                <h2 className="text-sm font-semibold truncate">
                  {characterName}
                </h2>
                <p className="text-xs text-muted-foreground truncate">
                  {session?.title}
                </p>
              </div>
            </>
          )}

          {!characterName && <div className="flex-1" />}

          {/* Sampling params */}
          <Popover>
            <PopoverTrigger asChild>
              <Button
                variant="outline"
                size="icon"
                className={cn(
                  "h-8 w-8 hidden sm:flex shrink-0",
                  (sampling.temperature !== null || sampling.topP !== null || sampling.topK !== null)
                    && "border-violet-500/50 text-violet-600",
                )}
                title="采样参数"
              >
                <SlidersHorizontal className="h-3.5 w-3.5" />
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-72" align="end">
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h4 className="text-sm font-semibold">采样参数</h4>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-6 px-2 text-xs text-muted-foreground"
                    onClick={() => {
                      const reset: SamplingParams = { temperature: null, topP: null, topK: null };
                      setSampling(reset);
                      updateSession({ temperature: null, topP: null, topK: null });
                    }}
                  >
                    <RotateCw className="h-3 w-3 mr-1" />
                    重置
                  </Button>
                </div>
                <p className="text-xs text-muted-foreground -mt-2">
                  留空使用模型默认值
                </p>

                {/* Temperature */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <label className="text-xs font-medium">Temperature</label>
                    <span className="text-xs text-muted-foreground tabular-nums text-right">
                      {(sampling.temperature ?? 0.6).toFixed(1)}
                    </span>
                  </div>
                  <Slider
                    value={[sampling.temperature ?? 0.6]}
                    min={0}
                    max={2}
                    step={0.1}
                    onValueChange={([v]) => setSampling((s) => ({ ...s, temperature: v }))}
                    onValueCommit={([v]) => updateSession({ temperature: v })}
                  />
                  <div className="flex justify-between text-[10px] text-muted-foreground/60">
                    <span>精确</span>
                    <span>创意</span>
                  </div>
                </div>

                {/* Top-P */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <label className="text-xs font-medium">Top-P</label>
                    <span className="text-xs text-muted-foreground tabular-nums text-right">
                      {(sampling.topP ?? 0.95).toFixed(2)}
                    </span>
                  </div>
                  <Slider
                    value={[sampling.topP ?? 0.95]}
                    min={0}
                    max={1}
                    step={0.01}
                    onValueChange={([v]) => setSampling((s) => ({ ...s, topP: v }))}
                    onValueCommit={([v]) => updateSession({ topP: v })}
                  />
                  <div className="flex justify-between text-[10px] text-muted-foreground/60">
                    <span>集中</span>
                    <span>多样</span>
                  </div>
                </div>

                {/* Top-K */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <label className="text-xs font-medium">Top-K</label>
                    <span className="text-xs text-muted-foreground tabular-nums text-right">
                      {sampling.topK ?? 20}
                    </span>
                  </div>
                  <Slider
                    value={[sampling.topK ?? 20]}
                    min={0}
                    max={100}
                    step={1}
                    onValueChange={([v]) => setSampling((s) => ({ ...s, topK: v }))}
                    onValueCommit={([v]) => updateSession({ topK: v })}
                  />
                  <div className="flex justify-between text-[10px] text-muted-foreground/60">
                    <span>贪婪</span>
                    <span>宽泛</span>
                  </div>
                </div>
              </div>
            </PopoverContent>
          </Popover>

          {/* Model selector */}
          <Select value={model} onValueChange={handleModelChange}>
            <SelectTrigger className="w-[140px] h-8 text-xs hidden sm:flex">
              <Zap className="h-3 w-3 mr-1 text-amber-400" />
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {modelList
                .filter((m) => m.status !== "offline")
                .map((m) => (
                  <SelectItem key={m.id} value={m.id}>
                    <div className="flex items-center gap-1.5">
                      <span
                        className={cn(
                          "h-1.5 w-1.5 rounded-full shrink-0",
                          m.status === "online"
                            ? "bg-green-500"
                            : "bg-amber-500",
                        )}
                      />
                      {m.name}
                    </div>
                  </SelectItem>
                ))}
            </SelectContent>
          </Select>

          {/* Personal worldbook button */}
          <Button
            variant="outline"
            size="sm"
            className="h-8 text-xs hidden sm:flex gap-1.5"
            onClick={() => setWorldbookOpen(true)}
          >
            <BookOpen className="h-3 w-3" />
            世界书
            {enabledWorldbooks.length > 0 && (
              <Badge
                variant="secondary"
                className="h-4 px-1 text-[10px] ml-0.5"
              >
                {enabledWorldbooks.length}
              </Badge>
            )}
          </Button>

          {/* Context usage — desktop */}
          <div className="hidden lg:flex items-center gap-2 text-xs text-muted-foreground">
            <span>
              {contextUsage.usedTokens.toLocaleString()}/
              {effectiveMaxTokens.toLocaleString()}
            </span>
            <Progress value={usagePercent} className="w-20 h-1.5" />
          </div>
        </div>

        {/* ---- Worldbook dialog ---- */}
        <Dialog open={worldbookOpen} onOpenChange={setWorldbookOpen}>
          <DialogContent className="sm:max-w-md">
            <DialogHeader>
              <DialogTitle>个人世界书</DialogTitle>
              <p className="text-sm text-muted-foreground">
                选择要在当前会话中启用的个人世界书。全局世界书由角色卡自动关联，无需手动管理。
              </p>
            </DialogHeader>
            <div className="space-y-3 mt-2">
              {personalBooks.length === 0 ? (
                <p className="text-sm text-muted-foreground text-center py-6">
                  暂无个人世界书，前往世界书页面创建
                </p>
              ) : (
                personalBooks.map((book) => {
                  const enabled = enabledWorldbooks.includes(book.id);
                  return (
                    <div
                      key={book.id}
                      className={cn(
                        "flex items-center gap-3 rounded-lg border p-3 transition-colors cursor-pointer",
                        enabled
                          ? "border-sky-500/30 bg-sky-500/5"
                          : "hover:bg-muted/50",
                      )}
                      onClick={() => toggleWorldbook(book.id)}
                    >
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium">{book.name}</p>
                        <p className="text-xs text-muted-foreground truncate mt-0.5">
                          {book.description}
                        </p>
                        <p className="text-xs text-muted-foreground mt-1">
                          {book.entryCount} 条词条 · {book.totalTokenCount}{" "}
                          tokens
                        </p>
                      </div>
                      <Switch
                        checked={enabled}
                        onCheckedChange={() => toggleWorldbook(book.id)}
                      />
                    </div>
                  );
                })
              )}
            </div>
          </DialogContent>
        </Dialog>

        {/* ---- Messages ---- */}
        <div className="flex-1 overflow-auto overscroll-none">
          <div className="px-4 lg:px-8 py-4 space-y-5">
            {messages.map((msg) => (
              <MessageBubble
                key={msg.id}
                message={msg}
                characterName={characterName}
                characterId={characterId}
                coverImage={session?.characterCoverImage}
                onEdit={handleEdit}
                onRegenerate={handleRegenerate}
              />
            ))}

            {/* Streaming bubble */}
            {isStreaming && streamingContent && (
              <MessageBubble
                message={{
                  id: "streaming",
                  role: "assistant",
                  content: streamingContent,
                  tokenCount: 0,
                  isCompressed: false,
                  createdAt: "",
                  editedAt: null,
                }}
                characterName={characterName}
                characterId={characterId}
                coverImage={session?.characterCoverImage}
                isStreaming
              />
            )}

            {/* Loading dots */}
            {isStreaming && !streamingContent && characterName && (
              <div className="flex gap-3">
                <CharacterAvatar
                  characterId={characterId}
                  characterName={characterName}
                  coverImage={session?.characterCoverImage}
                />
                <div className="bg-muted rounded-2xl rounded-tl-sm px-4 py-3 flex gap-1">
                  <span className="w-2 h-2 bg-muted-foreground/40 rounded-full animate-bounce [animation-delay:0ms]" />
                  <span className="w-2 h-2 bg-muted-foreground/40 rounded-full animate-bounce [animation-delay:150ms]" />
                  <span className="w-2 h-2 bg-muted-foreground/40 rounded-full animate-bounce [animation-delay:300ms]" />
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* ---- Input area ---- */}
        <div className="border-t px-4 lg:px-8 py-3 shrink-0">
          {/* Context usage — mobile */}
          <div className="flex lg:hidden items-center gap-2 text-xs text-muted-foreground mb-2">
            <span>
              上下文:{" "}
              {contextUsage.usedTokens.toLocaleString()}/
              {effectiveMaxTokens.toLocaleString()}
            </span>
            <Progress value={usagePercent} className="flex-1 h-1.5" />
          </div>

          <div className="flex items-end gap-2">
            <Textarea
              placeholder="输入消息... (Enter 发送, Shift+Enter 换行)"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              rows={1}
              className="min-h-[40px] max-h-[120px] resize-none"
              disabled={isStreaming}
            />
            {isStreaming ? (
              <Button
                variant="destructive"
                size="icon"
                className="shrink-0 h-10 w-10"
                onClick={handleStop}
              >
                <Square className="h-4 w-4" />
              </Button>
            ) : (
              <Button
                size="icon"
                className="shrink-0 h-10 w-10"
                onClick={handleSend}
                disabled={!input.trim()}
              >
                <Send className="h-4 w-4" />
              </Button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
