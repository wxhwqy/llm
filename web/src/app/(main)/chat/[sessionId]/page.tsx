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
import { cn } from "@/lib/utils";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { getAvatarColor, timeAgo, USE_MOCK } from "@/lib/constants";
import { useChatStore } from "@/stores/chat-store";
import { useChatStream } from "@/hooks/use-chat-stream";
import {
  useSessions,
  useMessages,
  useModels,
  useWorldBooks,
} from "@/hooks/use-queries";
import { useUiStore } from "@/stores/ui-store";
import { api } from "@/lib/api-client";
import { mockApi } from "@/lib/mock-api";
import type { ChatMessage } from "@/types/chat";
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
  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between p-3 border-b">
        <h2 className="font-semibold text-sm">会话列表</h2>
        <Link href="/characters">
          <Button variant="ghost" size="icon" className="h-7 w-7">
            <Plus className="h-4 w-4" />
          </Button>
        </Link>
      </div>
      <ScrollArea className="flex-1 overscroll-none">
        <div className="p-2 space-y-1">
          {sessions.map((s) => (
            <Link
              key={s.id}
              href={`/chat/${s.id}`}
              onClick={onSelect}
              className={cn(
                "flex items-start gap-3 rounded-lg p-2.5 transition-colors text-left w-full",
                s.id === currentId ? "bg-accent" : "hover:bg-accent/50",
              )}
            >
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
            </Link>
          ))}
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

  /* ---- data fetching ---- */
  const { data: sessionsData } = useSessions();
  const { data: messagesData } = useMessages(sessionId);
  const { data: modelsData } = useModels();
  const { data: worldbooksData } = useWorldBooks("personal");

  const sessions = sessionsData?.data ?? [];
  const session = sessions.find((s) => s.id === sessionId);
  const modelList = modelsData?.data ?? [];
  const personalBooks = worldbooksData?.data ?? [];

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
  }, [session]);

  /* auto-scroll on new content */
  useEffect(() => {
    scrollToBottom();
  }, [messages, streamingContent, scrollToBottom]);

  /* ---- handlers ---- */

  const updateSession = useCallback(
    async (patch: { modelId?: string; personalWorldBookIds?: string[] }) => {
      if (USE_MOCK) {
        await mockApi.updateSession(sessionId, patch);
      } else {
        await api.put(`/chat/sessions/${sessionId}`, patch);
      }
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

  const handleEdit = (id: string, content: string) => {
    useChatStore.getState().editMessage(id, content);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const usagePercent =
    contextUsage.maxTokens > 0
      ? (contextUsage.usedTokens / contextUsage.maxTokens) * 100
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

          {/* Model selector */}
          <Select value={model} onValueChange={handleModelChange}>
            <SelectTrigger className="w-[140px] h-8 text-xs hidden sm:flex">
              <Zap className="h-3 w-3 mr-1 text-amber-400" />
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {modelList.map((m) => (
                <SelectItem key={m.id} value={m.id}>
                  {m.name}
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
              {contextUsage.maxTokens.toLocaleString()}
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
              {contextUsage.maxTokens.toLocaleString()}
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
