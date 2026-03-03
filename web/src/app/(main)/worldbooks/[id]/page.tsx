"use client";

import { useState, use, useMemo } from "react";
import Link from "next/link";
import {
  ArrowLeft,
  Plus,
  Save,
  Download,
  Search,
  ChevronDown,
  ChevronRight,
  Trash2,
  GripVertical,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useWorldBook } from "@/hooks/use-queries";
import { cn } from "@/lib/utils";
import type { WorldBookEntry } from "@/types/worldbook";

function EntryEditor({
  entry,
  onChange,
}: {
  entry: WorldBookEntry;
  onChange: (updated: WorldBookEntry) => void;
}) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="rounded-lg border bg-card transition-colors hover:border-muted-foreground/20">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-3 w-full p-3 text-left"
      >
        <GripVertical className="h-4 w-4 text-muted-foreground/40 shrink-0 cursor-grab" />
        {expanded ? (
          <ChevronDown className="h-4 w-4 text-muted-foreground shrink-0" />
        ) : (
          <ChevronRight className="h-4 w-4 text-muted-foreground shrink-0" />
        )}
        <div className="flex-1 min-w-0 flex items-center gap-2 flex-wrap">
          {entry.keywords.map((kw) => (
            <Badge key={kw} variant="outline" className="text-xs shrink-0">
              {kw}
            </Badge>
          ))}
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <Badge
            variant={entry.enabled ? "default" : "secondary"}
            className="text-xs"
          >
            {entry.enabled ? "启用" : "禁用"}
          </Badge>
          <span className="text-xs text-muted-foreground">
            P{entry.priority}
          </span>
          <span className="text-xs text-muted-foreground">
            {entry.tokenCount} tk
          </span>
        </div>
      </button>

      {expanded && (
        <div className="px-3 pb-3 pt-1 space-y-3 border-t">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div>
              <label className="text-xs font-medium text-muted-foreground mb-1 block">
                关键词（逗号分隔）
              </label>
              <Input
                value={entry.keywords.join(", ")}
                className="text-sm"
                onChange={(e) =>
                  onChange({
                    ...entry,
                    keywords: e.target.value
                      .split(",")
                      .map((s) => s.trim())
                      .filter(Boolean),
                  })
                }
              />
            </div>
            <div className="flex gap-3">
              <div className="flex-1">
                <label className="text-xs font-medium text-muted-foreground mb-1 block">
                  注入位置
                </label>
                <Select
                  value={entry.position}
                  onValueChange={(v) =>
                    onChange({
                      ...entry,
                      position: v as WorldBookEntry["position"],
                    })
                  }
                >
                  <SelectTrigger className="text-sm">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="before_system">系统提示前</SelectItem>
                    <SelectItem value="after_system">系统提示后</SelectItem>
                    <SelectItem value="before_user">用户消息前</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="w-20">
                <label className="text-xs font-medium text-muted-foreground mb-1 block">
                  优先级
                </label>
                <Input
                  type="number"
                  value={entry.priority}
                  className="text-sm"
                  onChange={(e) =>
                    onChange({
                      ...entry,
                      priority: Number(e.target.value) || 0,
                    })
                  }
                />
              </div>
            </div>
          </div>

          <div>
            <label className="text-xs font-medium text-muted-foreground mb-1 block">
              内容
            </label>
            <Textarea
              value={entry.content}
              rows={4}
              className="text-sm resize-none"
              onChange={(e) =>
                onChange({ ...entry, content: e.target.value })
              }
            />
            <div className="text-xs text-muted-foreground mt-1 text-right">
              约 {entry.tokenCount} tokens
            </div>
          </div>

          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Switch
                checked={entry.enabled}
                onCheckedChange={(checked) =>
                  onChange({ ...entry, enabled: checked })
                }
              />
              <span className="text-sm">
                {entry.enabled ? "启用" : "禁用"}
              </span>
            </div>
            <Button
              variant="ghost"
              size="sm"
              className="text-destructive hover:text-destructive"
            >
              <Trash2 className="h-3.5 w-3.5 mr-1" />
              删除词条
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}

function EditPageSkeleton() {
  return (
    <div className="mx-auto max-w-4xl px-4 py-6 lg:px-6 space-y-4">
      <div className="flex items-center gap-3">
        <Skeleton className="h-8 w-8 rounded" />
        <Skeleton className="h-8 flex-1" />
        <Skeleton className="h-8 w-16" />
        <Skeleton className="h-8 w-16" />
      </div>
      <Skeleton className="h-16 w-full" />
      <div className="flex gap-4">
        <Skeleton className="h-4 w-24" />
        <Skeleton className="h-4 w-24" />
        <Skeleton className="h-4 w-24" />
      </div>
      <div className="flex gap-2">
        <Skeleton className="h-9 w-48" />
        <Skeleton className="h-9 w-24 ml-auto" />
      </div>
      {Array.from({ length: 3 }).map((_, i) => (
        <Skeleton key={i} className="h-14 w-full rounded-lg" />
      ))}
    </div>
  );
}

export default function WorldBookEditPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const { data, isLoading } = useWorldBook(id);
  const book = data?.data;

  const [entries, setEntries] = useState<WorldBookEntry[]>([]);
  const [initialized, setInitialized] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");

  if (book && !initialized) {
    setEntries(book.entries);
    setInitialized(true);
  }

  const totalTokens = entries.reduce((sum, e) => sum + e.tokenCount, 0);
  const enabledCount = entries.filter((e) => e.enabled).length;

  const filteredEntries = useMemo(() => {
    if (!searchQuery.trim()) return entries;
    const q = searchQuery.toLowerCase();
    return entries.filter(
      (e) =>
        e.keywords.some((kw) => kw.toLowerCase().includes(q)) ||
        e.content.toLowerCase().includes(q)
    );
  }, [entries, searchQuery]);

  if (isLoading || !book) {
    return (
      <div className="h-full overflow-auto overscroll-none">
        <EditPageSkeleton />
      </div>
    );
  }

  return (
    <div className="h-full overflow-auto overscroll-none">
      <div className="mx-auto max-w-4xl px-4 py-6 lg:px-6">
        <div className="flex items-center gap-3 mb-6">
          <Link href="/worldbooks">
            <Button variant="ghost" size="icon" className="h-8 w-8">
              <ArrowLeft className="h-4 w-4" />
            </Button>
          </Link>
          <div className="flex-1">
            <Input
              defaultValue={book.name}
              className="text-lg font-bold border-none px-0 h-auto focus-visible:ring-0 bg-transparent"
            />
          </div>
          <Button variant="outline" size="sm">
            <Download className="h-4 w-4 mr-1.5" />
            导出
          </Button>
          <Button size="sm">
            <Save className="h-4 w-4 mr-1.5" />
            保存
          </Button>
        </div>

        <Textarea
          defaultValue={book.description}
          placeholder="世界书描述..."
          rows={2}
          className="mb-4 resize-none text-sm"
        />

        <div className="flex items-center gap-4 text-sm text-muted-foreground mb-4">
          <span>
            共 <strong className="text-foreground">{entries.length}</strong> 条词条
          </span>
          <span>
            启用 <strong className="text-foreground">{enabledCount}</strong> 条
          </span>
          <span>
            总计{" "}
            <strong className="text-foreground">
              {totalTokens.toLocaleString()}
            </strong>{" "}
            tokens
          </span>
        </div>

        <div className="flex items-center gap-2 mb-4">
          <div className="relative max-w-xs flex-1">
            <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="搜索词条..."
              className="pl-8 text-sm"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>
          <div className="ml-auto">
            <Button size="sm">
              <Plus className="h-4 w-4 mr-1.5" />
              新增词条
            </Button>
          </div>
        </div>

        <div className="space-y-2">
          {filteredEntries.map((entry) => (
            <EntryEditor
              key={entry.id}
              entry={entry}
              onChange={(updated) => {
                setEntries((prev) =>
                  prev.map((e) => (e.id === updated.id ? updated : e))
                );
              }}
            />
          ))}
          {filteredEntries.length === 0 && searchQuery && (
            <div className="text-center py-12 text-muted-foreground">
              <Search className="h-8 w-8 mx-auto mb-2 opacity-30" />
              <p>没有匹配「{searchQuery}」的词条</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
