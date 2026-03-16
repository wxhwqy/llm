"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  ArrowLeft,
  Plus,
  Trash2,
  ChevronDown,
  ChevronRight,
  Loader2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useCreateWorldBook } from "@/hooks/use-queries";
import type { WorldBookEntry } from "@/types/worldbook";

type DraftEntry = Omit<WorldBookEntry, "id" | "tokenCount"> & { _key: string };

function emptyEntry(): DraftEntry {
  return {
    _key: `draft_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
    keywords: [],
    secondaryKeywords: [],
    content: "",
    position: "after_system",
    priority: 10,
    enabled: true,
  };
}

function DraftEntryEditor({
  entry,
  onChange,
  onDelete,
}: {
  entry: DraftEntry;
  onChange: (updated: DraftEntry) => void;
  onDelete: () => void;
}) {
  const [expanded, setExpanded] = useState(true);

  return (
    <div className="rounded-lg border bg-card">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-3 w-full p-3 text-left"
      >
        {expanded ? (
          <ChevronDown className="h-4 w-4 text-muted-foreground shrink-0" />
        ) : (
          <ChevronRight className="h-4 w-4 text-muted-foreground shrink-0" />
        )}
        <div className="flex-1 min-w-0 flex items-center gap-2 flex-wrap">
          {entry.keywords.length > 0 ? (
            entry.keywords.map((kw) => (
              <Badge key={kw} variant="outline" className="text-xs shrink-0">
                {kw}
              </Badge>
            ))
          ) : (
            <span className="text-sm text-muted-foreground">新词条</span>
          )}
        </div>
        <Badge
          variant={entry.enabled ? "default" : "secondary"}
          className="text-xs shrink-0"
        >
          {entry.enabled ? "启用" : "禁用"}
        </Badge>
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
                placeholder="关键词1, 关键词2"
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
                      position: v as DraftEntry["position"],
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
              placeholder="在这里填写词条内容..."
              className="text-sm resize-none"
              onChange={(e) =>
                onChange({ ...entry, content: e.target.value })
              }
            />
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
              onClick={onDelete}
            >
              <Trash2 className="h-3.5 w-3.5 mr-1" />
              删除
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}

export default function NewWorldBookPage() {
  const router = useRouter();
  const createMutation = useCreateWorldBook();

  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [scope, setScope] = useState<"global" | "personal">("global");
  const [entries, setEntries] = useState<DraftEntry[]>([]);

  const handleSubmit = () => {
    if (!name.trim()) return;
    createMutation.mutate(
      {
        name: name.trim(),
        description: description.trim(),
        scope,
        entries: entries.map(({ _key, ...rest }) => rest),
      },
      {
        onSuccess: (res) => {
          router.push(`/worldbooks/${res.data.id}`);
        },
      }
    );
  };

  return (
    <div className="h-full overflow-auto overscroll-none">
      <div className="mx-auto max-w-3xl px-4 py-6 lg:px-6">
        <div className="flex items-center gap-3 mb-6">
          <Link href="/worldbooks">
            <Button variant="ghost" size="icon" className="h-8 w-8">
              <ArrowLeft className="h-4 w-4" />
            </Button>
          </Link>
          <h1 className="text-xl font-bold flex-1">新建世界书</h1>
          <Button
            onClick={handleSubmit}
            disabled={!name.trim() || createMutation.isPending}
          >
            {createMutation.isPending && (
              <Loader2 className="h-4 w-4 mr-1.5 animate-spin" />
            )}
            创建
          </Button>
        </div>

        <div className="space-y-4 mb-6">
          <div>
            <label className="text-sm font-medium mb-1.5 block">名称</label>
            <Input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="世界书名称"
            />
          </div>

          <div>
            <label className="text-sm font-medium mb-1.5 block">描述</label>
            <Textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="简要描述这本世界书的内容..."
              rows={3}
              className="resize-none"
            />
          </div>

          <div>
            <label className="text-sm font-medium mb-1.5 block">作用范围</label>
            <Select value={scope} onValueChange={(v) => setScope(v as typeof scope)}>
              <SelectTrigger className="w-48">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="global">全局（所有角色可用）</SelectItem>
                <SelectItem value="personal">个人（仅自己可用）</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        <div className="border-t pt-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="font-semibold">
              词条 ({entries.length})
            </h2>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setEntries((prev) => [...prev, emptyEntry()])}
            >
              <Plus className="h-4 w-4 mr-1.5" />
              添加词条
            </Button>
          </div>

          {entries.length === 0 ? (
            <div className="text-center py-12 text-muted-foreground border rounded-lg border-dashed">
              <p className="text-sm">还没有词条</p>
              <p className="text-xs mt-1">点击上方按钮添加第一条词条，或者创建后再编辑</p>
            </div>
          ) : (
            <div className="space-y-2">
              {entries.map((entry, idx) => (
                <DraftEntryEditor
                  key={entry._key}
                  entry={entry}
                  onChange={(updated) =>
                    setEntries((prev) =>
                      prev.map((e, i) => (i === idx ? updated : e))
                    )
                  }
                  onDelete={() =>
                    setEntries((prev) => prev.filter((_, i) => i !== idx))
                  }
                />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
