"use client";

import { useState } from "react";
import Link from "next/link";
import {
  Plus,
  Upload,
  Download,
  Trash2,
  BookOpen,
  FileText,
  Users,
  Globe,
  User,
  Loader2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { Skeleton } from "@/components/ui/skeleton";
import { useWorldBooks, useDeleteWorldBook } from "@/hooks/use-queries";
import { api } from "@/lib/api-client";
import { cn } from "@/lib/utils";
import { timeAgo } from "@/lib/constants";
import type { WorldBookSummary } from "@/types/worldbook";

type ScopeFilter = "all" | "global" | "personal";

function WorldBookCard({ book }: { book: WorldBookSummary }) {
  const [deleteOpen, setDeleteOpen] = useState(false);
  const deleteMutation = useDeleteWorldBook();
  const isGlobal = book.scope === "global";

  const handleExport = async () => {
    try {
      const res = await fetch(`/api/worldbooks/${book.id}/export`);
      const data = await res.json();
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${book.name}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      console.error("Export failed:", e);
    }
  };

  const handleDelete = () => {
    deleteMutation.mutate(book.id, {
      onSuccess: () => setDeleteOpen(false),
    });
  };

  return (
    <>
      <div className="rounded-xl border bg-card p-5 transition-all hover:border-violet-500/50 hover:shadow-lg hover:shadow-violet-500/5">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            <div
              className={cn(
                "h-10 w-10 rounded-lg flex items-center justify-center",
                isGlobal ? "bg-violet-500/10" : "bg-sky-500/10"
              )}
            >
              <BookOpen
                className={cn(
                  "h-5 w-5",
                  isGlobal ? "text-violet-400" : "text-sky-400"
                )}
              />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="font-semibold">{book.name}</h3>
                <Badge
                  variant={isGlobal ? "default" : "outline"}
                  className={cn(
                    "text-[10px] px-1.5 h-5",
                    isGlobal
                      ? "bg-violet-600 hover:bg-violet-600"
                      : "text-sky-500 border-sky-500/30"
                  )}
                >
                  {isGlobal ? (
                    <>
                      <Globe className="h-2.5 w-2.5 mr-0.5" />
                      全局
                    </>
                  ) : (
                    <>
                      <User className="h-2.5 w-2.5 mr-0.5" />
                      个人
                    </>
                  )}
                </Badge>
              </div>
              <p className="text-xs text-muted-foreground mt-0.5">
                {timeAgo(book.createdAt)}
              </p>
            </div>
          </div>
        </div>

        <p className="text-sm text-muted-foreground mt-3 line-clamp-2">
          {book.description}
        </p>

        <div className="flex items-center gap-3 mt-4 text-xs text-muted-foreground">
          <div className="flex items-center gap-1">
            <FileText className="h-3.5 w-3.5" />
            {book.entryCount} 条词条
          </div>
          {isGlobal && (
            <div className="flex items-center gap-1">
              <Users className="h-3.5 w-3.5" />
              关联 {book.characterCount} 个角色
            </div>
          )}
          <Badge variant="secondary" className="text-xs">
            {book.totalTokenCount.toLocaleString()} tokens
          </Badge>
        </div>

        <div className="flex items-center gap-2 mt-4 pt-4 border-t">
          <Link href={`/worldbooks/${book.id}`} className="flex-1">
            <Button variant="outline" size="sm" className="w-full">
              编辑
            </Button>
          </Link>
          <Button
            variant="outline"
            size="icon"
            className="h-8 w-8 shrink-0"
            onClick={handleExport}
          >
            <Download className="h-3.5 w-3.5" />
          </Button>
          <Button
            variant="outline"
            size="icon"
            className="h-8 w-8 shrink-0 text-destructive hover:text-destructive"
            onClick={() => setDeleteOpen(true)}
          >
            <Trash2 className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>

      <Dialog open={deleteOpen} onOpenChange={setDeleteOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>删除世界书</DialogTitle>
            <DialogDescription>
              {isGlobal
                ? `确定要删除全局世界书「${book.name}」吗？关联的 ${book.characterCount} 个角色卡将失去该世界书。此操作不可撤销。`
                : `确定要删除个人世界书「${book.name}」吗？此操作不可撤销。`}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleteOpen(false)}>
              取消
            </Button>
            <Button
              variant="destructive"
              onClick={handleDelete}
              disabled={deleteMutation.isPending}
            >
              {deleteMutation.isPending && (
                <Loader2 className="h-4 w-4 mr-1.5 animate-spin" />
              )}
              删除
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}

function WorldBookCardSkeleton() {
  return (
    <div className="rounded-xl border bg-card p-5 space-y-3">
      <div className="flex items-center gap-3">
        <Skeleton className="h-10 w-10 rounded-lg" />
        <div className="space-y-2">
          <Skeleton className="h-4 w-32" />
          <Skeleton className="h-3 w-24" />
        </div>
      </div>
      <Skeleton className="h-4 w-full" />
      <Skeleton className="h-4 w-2/3" />
      <div className="flex gap-3 mt-4">
        <Skeleton className="h-4 w-20" />
        <Skeleton className="h-4 w-20" />
      </div>
      <Skeleton className="h-8 w-full mt-4" />
    </div>
  );
}

export default function WorldBooksPage() {
  const [filter, setFilter] = useState<ScopeFilter>("all");

  const { data: allData, isLoading: allLoading } = useWorldBooks();
  const { data: globalData, isLoading: globalLoading } = useWorldBooks("global");
  const { data: personalData, isLoading: personalLoading } = useWorldBooks("personal");

  const isLoading =
    filter === "all" ? allLoading : filter === "global" ? globalLoading : personalLoading;
  const activeData =
    filter === "all" ? allData : filter === "global" ? globalData : personalData;

  const books = activeData?.data ?? [];
  const allCount = allData?.pagination.total ?? 0;
  const globalCount = globalData?.pagination.total ?? 0;
  const personalCount = personalData?.pagination.total ?? 0;

  return (
    <div className="h-full overflow-auto overscroll-none">
      <div className="mx-auto max-w-4xl px-4 py-6 lg:px-6">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-2xl font-bold">世界书</h1>
            <p className="text-sm text-muted-foreground mt-1">
              管理全局与个人的世界观设定
            </p>
          </div>
          <div className="flex gap-2">
            <Link href="/worldbooks/import">
              <Button variant="outline" size="sm">
                <Upload className="h-4 w-4 mr-1.5" />
                导入
              </Button>
            </Link>
            <Link href="/worldbooks/new">
              <Button size="sm">
                <Plus className="h-4 w-4 mr-1.5" />
                新建
              </Button>
            </Link>
          </div>
        </div>

        <div className="flex items-center gap-1.5 mb-5 p-1 bg-muted/50 rounded-lg w-fit">
          {(
            [
              { key: "all", label: "全部", count: allCount },
              { key: "global", label: "全局", count: globalCount },
              { key: "personal", label: "个人", count: personalCount },
            ] as const
          ).map(({ key, label, count }) => (
            <button
              key={key}
              onClick={() => setFilter(key)}
              className={cn(
                "flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors",
                filter === key
                  ? "bg-background shadow-sm text-foreground"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              {label}
              <span
                className={cn(
                  "text-xs",
                  filter === key
                    ? "text-muted-foreground"
                    : "text-muted-foreground/60"
                )}
              >
                {count}
              </span>
            </button>
          ))}
        </div>

        {isLoading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {Array.from({ length: 4 }).map((_, i) => (
              <WorldBookCardSkeleton key={i} />
            ))}
          </div>
        ) : (
          <>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {books.map((book) => (
                <WorldBookCard key={book.id} book={book} />
              ))}
            </div>

            {books.length === 0 && (
              <div className="text-center py-16 text-muted-foreground">
                <BookOpen className="h-10 w-10 mx-auto mb-3 opacity-30" />
                <p>
                  没有
                  {filter === "personal"
                    ? "个人"
                    : filter === "global"
                      ? "全局"
                      : ""}
                  世界书
                </p>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
