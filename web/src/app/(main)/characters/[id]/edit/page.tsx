"use client";

import { use, useState, useEffect } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  ArrowLeft,
  Save,
  Trash2,
  Plus,
  X,
  Image as ImageIcon,
  Loader2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
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
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { useCharacter, useWorldBooks } from "@/hooks/use-queries";
import { getCoverGradient } from "@/lib/constants";
import { cn } from "@/lib/utils";
import type { CharacterCard } from "@/types/character";

function FormField({
  label,
  description,
  children,
}: {
  label: string;
  description?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-1.5">
      <label className="text-sm font-medium">{label}</label>
      {description && (
        <p className="text-xs text-muted-foreground">{description}</p>
      )}
      {children}
    </div>
  );
}

function EditSkeleton() {
  return (
    <div className="mx-auto max-w-3xl px-4 py-6 lg:px-6 space-y-8">
      <div className="flex items-center gap-3">
        <Skeleton className="h-8 w-8 rounded" />
        <Skeleton className="h-6 w-48" />
      </div>
      <Skeleton className="w-full aspect-[21/9] rounded-xl" />
      <div className="space-y-4">
        <Skeleton className="h-10 w-full" />
        <Skeleton className="h-10 w-full" />
        <Skeleton className="h-24 w-full" />
        <Skeleton className="h-24 w-full" />
      </div>
    </div>
  );
}

export default function CharacterEditPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const router = useRouter();
  const isNew = id === "new";

  const { data: characterData, isLoading } = useCharacter(isNew ? "" : id);
  const { data: worldBooksData } = useWorldBooks();

  const character = characterData?.data;
  const allWorldBooks = worldBooksData?.data ?? [];

  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [personality, setPersonality] = useState("");
  const [scenario, setScenario] = useState("");
  const [systemPrompt, setSystemPrompt] = useState("");
  const [firstMessage, setFirstMessage] = useState("");
  const [exampleDialogue, setExampleDialogue] = useState("");
  const [creatorNotes, setCreatorNotes] = useState("");
  const [source, setSource] = useState<CharacterCard["source"]>("manual");
  const [tags, setTags] = useState<string[]>([]);
  const [tagInput, setTagInput] = useState("");
  const [worldBookIds, setWorldBookIds] = useState<string[]>([]);
  const [deleteOpen, setDeleteOpen] = useState(false);

  useEffect(() => {
    if (character) {
      setName(character.name);
      setDescription(character.description);
      setPersonality(character.personality);
      setScenario(character.scenario);
      setSystemPrompt(character.systemPrompt);
      setFirstMessage(character.firstMessage);
      setExampleDialogue(character.exampleDialogue);
      setCreatorNotes(character.creatorNotes);
      setSource(character.source);
      setTags(character.tags);
      setWorldBookIds(character.worldBookIds);
    }
  }, [character]);

  if (!isNew && isLoading) {
    return (
      <div className="h-full overflow-auto overscroll-none">
        <EditSkeleton />
      </div>
    );
  }

  if (!isNew && !character) {
    return (
      <div className="h-full flex items-center justify-center text-muted-foreground">
        角色卡不存在
      </div>
    );
  }

  const gradient = character
    ? getCoverGradient(character.id)
    : "from-gray-500 to-gray-700";

  const addTag = () => {
    const t = tagInput.trim();
    if (t && !tags.includes(t)) {
      setTags([...tags, t]);
      setTagInput("");
    }
  };

  const removeWorldBook = (wbId: string) => {
    setWorldBookIds(worldBookIds.filter((id) => id !== wbId));
  };

  const availableWorldBooks = allWorldBooks.filter(
    (wb) => !worldBookIds.includes(wb.id)
  );

  const linkedWorldBooks = allWorldBooks.filter((wb) =>
    worldBookIds.includes(wb.id)
  );

  return (
    <div className="h-full overflow-auto overscroll-none">
      <div className="mx-auto max-w-3xl px-4 py-6 lg:px-6">
        <div className="flex items-center gap-3 mb-6">
          <Link href={isNew ? "/characters" : `/characters/${id}`}>
            <Button variant="ghost" size="icon" className="h-8 w-8">
              <ArrowLeft className="h-4 w-4" />
            </Button>
          </Link>
          <h1 className="text-xl font-bold flex-1">
            {isNew ? "新建角色卡" : `编辑 · ${name}`}
          </h1>
          <div className="flex gap-2">
            {!isNew && (
              <Button
                variant="outline"
                size="sm"
                className="text-destructive hover:text-destructive"
                onClick={() => setDeleteOpen(true)}
              >
                <Trash2 className="h-4 w-4 mr-1.5" />
                删除
              </Button>
            )}
            <Button size="sm">
              <Save className="h-4 w-4 mr-1.5" />
              保存
            </Button>
          </div>
        </div>

        <div className="space-y-8">
          <section className="space-y-4">
            <h2 className="text-base font-semibold">封面与头像</h2>
            <div
              className={cn(
                "relative w-full aspect-[21/9] rounded-xl bg-gradient-to-br overflow-hidden",
                gradient
              )}
            >
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-[80px] font-black text-white/10 select-none">
                  {name?.[0] || "?"}
                </span>
              </div>
              <div className="absolute inset-0 flex items-center justify-center opacity-0 hover:opacity-100 transition-opacity bg-black/40">
                <Button variant="secondary" size="sm">
                  <ImageIcon className="h-4 w-4 mr-1.5" />
                  更换封面
                </Button>
              </div>
            </div>
          </section>

          <Separator />

          <section className="space-y-4">
            <h2 className="text-base font-semibold">基本信息</h2>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <FormField label="角色名称">
                <Input
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="输入角色名称"
                />
              </FormField>

              <FormField label="来源">
                <Select
                  value={source}
                  onValueChange={(v) => setSource(v as CharacterCard["source"])}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="manual">手动创建</SelectItem>
                    <SelectItem value="sillytavern_png">
                      SillyTavern PNG
                    </SelectItem>
                    <SelectItem value="json_import">JSON 导入</SelectItem>
                  </SelectContent>
                </Select>
              </FormField>
            </div>

            <FormField
              label="角色描述"
              description="简短描述，在角色卡列表中展示"
            >
              <Textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="描述这个角色..."
                rows={3}
                className="resize-none"
              />
            </FormField>

            <FormField label="标签">
              <div className="flex flex-wrap gap-1.5 mb-2">
                {tags.map((tag) => (
                  <Badge key={tag} variant="secondary" className="gap-1 pr-1">
                    {tag}
                    <button
                      onClick={() => setTags(tags.filter((t) => t !== tag))}
                      className="hover:text-destructive"
                    >
                      <X className="h-3 w-3" />
                    </button>
                  </Badge>
                ))}
              </div>
              <div className="flex gap-2">
                <Input
                  value={tagInput}
                  onChange={(e) => setTagInput(e.target.value)}
                  placeholder="输入标签后回车"
                  className="max-w-xs"
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault();
                      addTag();
                    }
                  }}
                />
                <Button variant="outline" size="sm" onClick={addTag}>
                  <Plus className="h-4 w-4" />
                </Button>
              </div>
            </FormField>
          </section>

          <Separator />

          <section className="space-y-4">
            <h2 className="text-base font-semibold">角色设定</h2>

            <FormField
              label="性格描述"
              description="描述角色的性格特点，用于指导 AI 的对话风格"
            >
              <Textarea
                value={personality}
                onChange={(e) => setPersonality(e.target.value)}
                placeholder="例如：活泼开朗、好奇心强、善良正义..."
                rows={2}
                className="resize-none"
              />
            </FormField>

            <FormField
              label="场景设定"
              description="描述对话发生的背景场景"
            >
              <Textarea
                value={scenario}
                onChange={(e) => setScenario(e.target.value)}
                placeholder="例如：你在星辰学院的图书馆偶遇了..."
                rows={3}
                className="resize-none"
              />
            </FormField>

            <FormField
              label="系统提示词"
              description="发送给 AI 的 System Prompt，控制角色的整体行为"
            >
              <Textarea
                value={systemPrompt}
                onChange={(e) => setSystemPrompt(e.target.value)}
                placeholder="你是..., 你需要..."
                rows={5}
                className="resize-none font-mono text-xs"
              />
            </FormField>
          </section>

          <Separator />

          <section className="space-y-4">
            <h2 className="text-base font-semibold">对话内容</h2>

            <FormField
              label="开场白（首条消息）"
              description="创建会话后角色自动发送的第一条消息"
            >
              <Textarea
                value={firstMessage}
                onChange={(e) => setFirstMessage(e.target.value)}
                placeholder="角色的开场白..."
                rows={4}
                className="resize-none"
              />
            </FormField>

            <FormField
              label="示例对话"
              description="提供对话风格参考，帮助 AI 更好地模仿角色"
            >
              <Textarea
                value={exampleDialogue}
                onChange={(e) => setExampleDialogue(e.target.value)}
                placeholder="用户：...\n角色：..."
                rows={5}
                className="resize-none"
              />
            </FormField>
          </section>

          <Separator />

          <section className="space-y-4">
            <h2 className="text-base font-semibold">关联世界书</h2>
            <p className="text-xs text-muted-foreground">
              选择与此角色关联的世界书，聊天时会根据关键词自动注入世界观设定
            </p>

            <div className="space-y-2">
              {linkedWorldBooks.map((wb) => (
                <div
                  key={wb.id}
                  className="flex items-center justify-between rounded-lg border p-3"
                >
                  <div>
                    <p className="text-sm font-medium">{wb.name}</p>
                    <p className="text-xs text-muted-foreground">
                      {wb.totalTokenCount.toLocaleString()} tokens
                    </p>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="text-destructive h-7"
                    onClick={() => removeWorldBook(wb.id)}
                  >
                    <X className="h-3.5 w-3.5" />
                  </Button>
                </div>
              ))}
              {availableWorldBooks.length > 0 && (
                <Select
                  onValueChange={(wbId) =>
                    setWorldBookIds([...worldBookIds, wbId])
                  }
                >
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="关联世界书..." />
                  </SelectTrigger>
                  <SelectContent>
                    {availableWorldBooks.map((wb) => (
                      <SelectItem key={wb.id} value={wb.id}>
                        {wb.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              )}
            </div>
          </section>

          <Separator />

          <section className="space-y-4 pb-8">
            <FormField label="创作者备注" description="仅创作者可见的备注信息">
              <Textarea
                value={creatorNotes}
                onChange={(e) => setCreatorNotes(e.target.value)}
                placeholder="备注..."
                rows={2}
                className="resize-none"
              />
            </FormField>
          </section>
        </div>
      </div>

      <Dialog open={deleteOpen} onOpenChange={setDeleteOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>删除角色卡</DialogTitle>
            <DialogDescription>
              确定要删除「{name}」吗？相关的会话不会被删除，但将无法再使用此角色卡创建新会话。此操作不可撤销。
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleteOpen(false)}>
              取消
            </Button>
            <Button
              variant="destructive"
              onClick={() => {
                setDeleteOpen(false);
                router.push("/characters");
              }}
            >
              删除
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
