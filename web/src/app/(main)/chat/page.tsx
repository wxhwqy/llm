"use client";

import Link from "next/link";
import { MessageCircle, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Plus, PanelLeftOpen } from "lucide-react";
import {
  Sheet,
  SheetContent,
  SheetTrigger,
  SheetTitle,
} from "@/components/ui/sheet";
import { cn } from "@/lib/utils";
import { useSessions } from "@/hooks/use-queries";
import { useAuthGuard } from "@/hooks/use-require-auth";
import { useUiStore } from "@/stores/ui-store";
import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { getAvatarColor, timeAgo } from "@/lib/constants";
import type { ChatSession } from "@/types/chat";

function CharacterAvatar({
  characterId,
  characterName,
  coverImage,
}: {
  characterId: string;
  characterName: string;
  coverImage?: string | null;
}) {
  const dim = "h-9 w-9";
  if (coverImage) {
    return (
      <img
        src={coverImage}
        alt={characterName}
        className={cn(dim, "rounded-full object-cover shrink-0")}
      />
    );
  }
  return (
    <div
      className={cn(
        dim,
        "rounded-full flex items-center justify-center text-xs font-bold text-white shrink-0",
        getAvatarColor(characterId),
      )}
    >
      {characterName?.[0]}
    </div>
  );
}

function SessionList({
  sessions,
  onSelect,
}: {
  sessions: ChatSession[];
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
              className="flex items-start gap-3 rounded-lg p-2.5 transition-colors text-left w-full hover:bg-accent/50"
            >
              <CharacterAvatar
                characterId={s.characterId}
                characterName={s.characterName}
                coverImage={s.characterCoverImage}
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

export default function ChatPage() {
  const { isLoading: authLoading } = useAuthGuard();
  const { data: sessionsData, isLoading: sessionsLoading } = useSessions();
  const router = useRouter();
  const sidebarOpen = useUiStore((s) => s.sidebarOpen);
  const setSidebarOpen = useUiStore((s) => s.setSidebarOpen);

  const sessions = sessionsData?.data ?? [];
  const isLoading = authLoading || sessionsLoading;

  // If sessions exist, redirect to the first one
  useEffect(() => {
    if (!isLoading && sessions.length > 0) {
      router.replace(`/chat/${sessions[0].id}`);
    }
  }, [isLoading, sessions, router]);

  // Loading or redirecting
  if (isLoading || sessions.length > 0) {
    return null;
  }

  // Empty state — no sessions
  return (
    <div className="flex h-full">
      {/* Desktop sidebar */}
      <aside className="hidden md:flex w-72 lg:w-80 border-r flex-col shrink-0">
        <SessionList sessions={sessions} />
      </aside>

      {/* Main area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <div className="flex items-center gap-3 px-4 py-2.5 border-b shrink-0">
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
                onSelect={() => setSidebarOpen(false)}
              />
            </SheetContent>
          </Sheet>
          <div className="flex-1" />
        </div>

        {/* Empty state */}
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center space-y-4 px-6">
            <div className="mx-auto h-16 w-16 rounded-full bg-muted flex items-center justify-center">
              <MessageCircle className="h-8 w-8 text-muted-foreground" />
            </div>
            <div>
              <h2 className="text-lg font-semibold">还没有对话</h2>
              <p className="text-sm text-muted-foreground mt-1">
                前往角色卡页面，选择一个角色开始对话吧
              </p>
            </div>
            <Link href="/characters">
              <Button className="mt-2">
                <Sparkles className="h-4 w-4 mr-1.5" />
                浏览角色卡
              </Button>
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
