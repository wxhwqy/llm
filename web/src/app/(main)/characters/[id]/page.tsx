"use client";

import { use } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  ArrowLeft,
  MessageCircle,
  Pencil,
  Calendar,
  BookOpen,
  Loader2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { useCharacter, useCreateSession } from "@/hooks/use-queries";
import { getCoverGradient, timeAgo } from "@/lib/constants";
import { cn } from "@/lib/utils";
import type { CharacterCard } from "@/types/character";

function CoverImage({
  name,
  gradient,
  coverImage,
  className,
}: {
  name: string;
  gradient: string;
  coverImage?: string | null;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "relative overflow-hidden",
        coverImage ? "bg-muted" : `bg-gradient-to-br ${gradient}`,
        className
      )}
    >
      {coverImage ? (
        <img
          src={coverImage}
          alt={name}
          className="absolute inset-0 w-full h-full object-cover"
        />
      ) : (
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-[140px] lg:text-[180px] font-black text-white/10 select-none leading-none">
            {name[0]}
          </span>
        </div>
      )}
    </div>
  );
}

function sourceLabel(source: CharacterCard["source"]) {
  switch (source) {
    case "sillytavern_png":
      return "SillyTavern 导入";
    case "json_import":
      return "JSON 导入";
    default:
      return "手动创建";
  }
}

function DetailSkeleton() {
  return (
    <div className="mx-auto max-w-5xl px-4 py-6 lg:px-6 space-y-6">
      <Skeleton className="h-4 w-32" />
      <div className="flex flex-col md:flex-row gap-6 lg:gap-8">
        <Skeleton className="w-full md:w-[320px] lg:w-[360px] aspect-[3/4] rounded-xl" />
        <div className="flex-1 space-y-4">
          <Skeleton className="h-8 w-48" />
          <Skeleton className="h-4 w-40" />
          <div className="flex gap-2">
            <Skeleton className="h-6 w-14" />
            <Skeleton className="h-6 w-14" />
            <Skeleton className="h-6 w-14" />
          </div>
          <Skeleton className="h-20 w-full" />
          <Skeleton className="h-12 w-full mt-8" />
        </div>
      </div>
    </div>
  );
}

export default function CharacterDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const router = useRouter();
  const { data: characterData, isLoading, error } = useCharacter(id);
  const createSession = useCreateSession();
  const character = characterData?.data;

  const handleStartChat = () => {
    if (createSession.isPending) return;
    createSession.mutate(id, {
      onSuccess: (data) => {
        router.push(`/chat/${data.data.id}`);
      },
    });
  };

  if (isLoading) {
    return (
      <div className="h-full overflow-auto overscroll-none">
        <DetailSkeleton />
      </div>
    );
  }

  if (!character) {
    return (
      <div className="h-full flex items-center justify-center text-muted-foreground">
        角色卡不存在
      </div>
    );
  }

  const gradient = getCoverGradient(character.id);

  return (
    <div className="h-full overflow-auto overscroll-none">
      <div className="mx-auto max-w-5xl px-4 py-6 lg:px-6">
        <div className="flex items-center justify-between mb-5">
          <Link
            href="/characters"
            className="inline-flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            <ArrowLeft className="h-4 w-4" />
            返回角色卡列表
          </Link>
          <Link
            href={`/characters/${character.id}/edit`}
            className="text-muted-foreground/40 hover:text-muted-foreground transition-colors"
            title="编辑"
          >
            <Pencil className="h-3.5 w-3.5" />
          </Link>
        </div>

        <div className="flex flex-col md:flex-row gap-6 lg:gap-8">
          <div className="w-full md:w-[320px] lg:w-[360px] shrink-0">
            <CoverImage
              name={character.name}
              gradient={gradient}
              coverImage={character.coverImage}
              className="aspect-[3/4] rounded-xl w-full"
            />
          </div>

          <div className="flex-1 min-w-0 flex flex-col">
            <h1 className="text-2xl lg:text-3xl font-bold leading-tight">
              {character.name}
            </h1>

            <div className="flex items-center gap-3 mt-2 text-sm text-muted-foreground">
              <span>{sourceLabel(character.source)}</span>
              <span className="flex items-center gap-1">
                <Calendar className="h-3.5 w-3.5" />
                {timeAgo(character.createdAt)}
              </span>
            </div>

            <div className="flex flex-wrap gap-1.5 mt-4">
              {character.tags.map((tag) => (
                <Badge key={tag} variant="secondary">
                  {tag}
                </Badge>
              ))}
            </div>

            {character.worldBookIds.length > 0 && (
              <div className="flex items-center gap-2 mt-4 text-sm text-muted-foreground">
                <BookOpen className="h-4 w-4 shrink-0" />
                <span>关联世界书：</span>
                {character.worldBookIds.map((wbId) => (
                  <Link
                    key={wbId}
                    href={`/worldbooks/${wbId}`}
                    className="text-violet-500 hover:underline"
                  >
                    {wbId}
                  </Link>
                ))}
              </div>
            )}

            <div className="flex-1" />

            <div className="mt-6">
              <Button
                size="lg"
                className="w-full text-base h-12 bg-violet-600 hover:bg-violet-700"
                onClick={handleStartChat}
                disabled={createSession.isPending}
              >
                {createSession.isPending ? (
                  <Loader2 className="h-5 w-5 mr-2 animate-spin" />
                ) : (
                  <MessageCircle className="h-5 w-5 mr-2" />
                )}
                {createSession.isPending ? "创建中..." : "开始对话"}
              </Button>
            </div>
          </div>
        </div>

        {character.personality && (
          <>
            <Separator className="my-8" />
            <section>
              <h2 className="text-lg font-semibold flex items-center gap-2 mb-4">
                <span className="w-1 h-5 bg-violet-500 rounded-full" />
                角色简介
              </h2>
              <div
                className="text-sm text-muted-foreground leading-relaxed prose prose-sm prose-neutral dark:prose-invert max-w-none [&>*:first-child]:mt-0 [&>*:last-child]:mb-0"
                dangerouslySetInnerHTML={{ __html: character.personality }}
              />
            </section>
          </>
        )}

        {character.firstMessage && (
          <>
            <Separator className="my-8" />
            <section>
              <h2 className="text-lg font-semibold flex items-center gap-2 mb-4">
                <span className="w-1 h-5 bg-violet-500 rounded-full" />
                开场白
              </h2>
              <div className="rounded-xl border bg-muted/30 p-4 text-sm leading-relaxed whitespace-pre-wrap">
                {character.firstMessage}
              </div>
            </section>
          </>
        )}

        <div className="pb-8" />
      </div>
    </div>
  );
}
