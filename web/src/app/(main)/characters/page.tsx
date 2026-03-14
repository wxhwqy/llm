"use client";

import { useState, useRef, useCallback } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Search, Upload, Plus, Sparkles, Tag, FileImage, FileJson, Loader2 } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useCharacters, useCharacterTags } from "@/hooks/use-queries";
import { useDebounce } from "@/hooks/use-debounce";
import { getCoverGradient, timeAgo } from "@/lib/constants";
import { cn, stripHtml } from "@/lib/utils";
import { parseCharacterPng, parseCharacterJson } from "@/lib/parse-character-file";
import { useImportStore } from "@/stores/import-store";
import type { CharacterSummary } from "@/types/character";

function CoverImage({ character }: { character: CharacterSummary }) {
  const gradient = character.coverImage ? undefined : getCoverGradient(character.id);
  return (
    <div
      className={cn(
        "relative w-full aspect-[3/4] overflow-hidden",
        character.coverImage ? "bg-muted" : `bg-gradient-to-br ${gradient}`
      )}
    >
      {character.coverImage ? (
        <img
          src={character.coverImage}
          alt={character.name}
          className="absolute inset-0 w-full h-full object-cover"
        />
      ) : (
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-[120px] font-black text-white/10 select-none leading-none">
            {character.name[0]}
          </span>
        </div>
      )}
      <div className="absolute inset-x-0 bottom-0 h-1/3 bg-gradient-to-t from-black/60 to-transparent" />
    </div>
  );
}

function CharacterCardItem({ character }: { character: CharacterSummary }) {
  return (
    <Link
      href={`/characters/${character.id}`}
      className="group relative flex flex-col rounded-xl border bg-card overflow-hidden text-left transition-all hover:border-violet-500/50 hover:shadow-xl hover:shadow-violet-500/5 hover:-translate-y-0.5"
    >
      <CoverImage character={character} />
      <div className="p-3.5 space-y-2">
        <h3 className="font-semibold text-base truncate">{character.name}</h3>
        <p className="text-xs text-muted-foreground line-clamp-2 leading-relaxed">
          {stripHtml(character.personality)}
        </p>
        <div className="flex items-end gap-1">
          <div className="flex flex-wrap gap-1 flex-1 min-w-0">
            {character.tags.map((tag) => (
              <Badge
                key={tag}
                variant="secondary"
                className="text-[10px] px-1.5 py-0 h-5"
              >
                {tag}
              </Badge>
            ))}
          </div>
          <span className="text-[10px] text-muted-foreground/60 shrink-0">
            {timeAgo(character.updatedAt)}
          </span>
        </div>
      </div>
    </Link>
  );
}

function CardSkeleton() {
  return (
    <div className="flex flex-col rounded-xl border bg-card overflow-hidden">
      <Skeleton className="w-full aspect-[3/4]" />
      <div className="p-3.5 space-y-2">
        <Skeleton className="h-5 w-2/3" />
        <Skeleton className="h-3 w-full" />
        <Skeleton className="h-3 w-4/5" />
        <div className="flex gap-1">
          <Skeleton className="h-5 w-10" />
          <Skeleton className="h-5 w-10" />
        </div>
      </div>
    </div>
  );
}

type ImportType = "sillytavern_png" | "json_import";

export default function CharactersPage() {
  const router = useRouter();
  const [search, setSearch] = useState("");
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const debouncedSearch = useDebounce(search);

  const { data: charactersData, isLoading } = useCharacters({
    search: debouncedSearch || undefined,
    tags: selectedTags.length > 0 ? selectedTags : undefined,
  });
  const { data: tagsData } = useCharacterTags();

  const characters = charactersData?.data ?? [];
  const allTags = tagsData?.data ?? [];

  /* ---- import dialog state ---- */
  const [importOpen, setImportOpen] = useState(false);
  const [importType, setImportType] = useState<ImportType>("sillytavern_png");
  const [importError, setImportError] = useState("");
  const [importing, setImporting] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const setImportData = useImportStore((s) => s.setData);

  const acceptMap: Record<ImportType, string> = {
    sillytavern_png: ".png",
    json_import: ".json",
  };

  const handleFile = useCallback(
    async (file: File) => {
      setImportError("");
      setImporting(true);
      try {
        const data =
          importType === "sillytavern_png"
            ? await parseCharacterPng(file)
            : await parseCharacterJson(file);
        setImportData(data);
        setImportOpen(false);
        router.push("/characters/new/edit");
      } catch (err) {
        setImportError(
          err instanceof Error ? err.message : "文件解析失败",
        );
      } finally {
        setImporting(false);
      }
    },
    [importType, setImportData, router],
  );

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
    e.target.value = "";
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  const toggleTag = (tag: string) => {
    setSelectedTags((prev) =>
      prev.includes(tag) ? prev.filter((t) => t !== tag) : [...prev, tag]
    );
  };

  return (
    <div className="h-full overflow-auto overscroll-none">
      <div className="mx-auto max-w-[1600px] px-4 py-6 lg:px-8">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-2xl font-bold">角色卡</h1>
            <p className="text-sm text-muted-foreground mt-1">
              选择一个角色开始对话
            </p>
          </div>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              className="hidden sm:flex"
              onClick={() => {
                setImportError("");
                setImportOpen(true);
              }}
            >
              <Upload className="h-4 w-4 mr-1.5" />
              导入
            </Button>
            <Link href="/characters/new/edit">
              <Button size="sm" className="hidden sm:flex">
                <Plus className="h-4 w-4 mr-1.5" />
                新建
              </Button>
            </Link>
          </div>
        </div>

        <div className="space-y-3 mb-6">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="搜索角色名称或描述..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="pl-9"
            />
          </div>
          {allTags.length > 0 && (
            <div className="flex items-center gap-2 overflow-x-auto pb-1">
              <Tag className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
              {allTags.map((tag) => (
                <Badge
                  key={tag}
                  variant={selectedTags.includes(tag) ? "default" : "outline"}
                  className="cursor-pointer shrink-0"
                  onClick={() => toggleTag(tag)}
                >
                  {tag}
                </Badge>
              ))}
            </div>
          )}
        </div>

        {isLoading ? (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
            {Array.from({ length: 10 }).map((_, i) => (
              <CardSkeleton key={i} />
            ))}
          </div>
        ) : characters.length > 0 ? (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
            {characters.map((c) => (
              <CharacterCardItem key={c.id} character={c} />
            ))}
          </div>
        ) : (
          <div className="text-center py-20 text-muted-foreground">
            <Sparkles className="h-10 w-10 mx-auto mb-3 opacity-30" />
            <p>没有找到匹配的角色卡</p>
          </div>
        )}
      </div>

      {/* ---- Import Dialog ---- */}
      <Dialog open={importOpen} onOpenChange={setImportOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>导入角色卡</DialogTitle>
          </DialogHeader>

          <div className="space-y-4">
            <div className="space-y-1.5">
              <label className="text-sm font-medium">导入来源</label>
              <Select
                value={importType}
                onValueChange={(v) => {
                  setImportType(v as ImportType);
                  setImportError("");
                }}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="sillytavern_png">
                    SillyTavern PNG
                  </SelectItem>
                  <SelectItem value="json_import">JSON 文件</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div
              className={cn(
                "relative flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed p-8 transition-colors cursor-pointer",
                dragOver
                  ? "border-violet-500 bg-violet-500/5"
                  : "border-muted-foreground/20 hover:border-violet-500/50",
              )}
              onClick={() => fileInputRef.current?.click()}
              onDragOver={(e) => {
                e.preventDefault();
                setDragOver(true);
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
            >
              {importing ? (
                <Loader2 className="h-8 w-8 text-violet-500 animate-spin" />
              ) : importType === "sillytavern_png" ? (
                <FileImage className="h-8 w-8 text-muted-foreground" />
              ) : (
                <FileJson className="h-8 w-8 text-muted-foreground" />
              )}
              <div className="text-center">
                <p className="text-sm font-medium">
                  {importing
                    ? "解析中..."
                    : importType === "sillytavern_png"
                      ? "上传 SillyTavern PNG 角色卡"
                      : "上传 JSON 角色文件"}
                </p>
                <p className="text-xs text-muted-foreground mt-1">
                  点击选择文件或拖拽到此处
                </p>
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept={acceptMap[importType]}
                className="hidden"
                onChange={handleFileChange}
              />
            </div>

            {importError && (
              <p className="text-sm text-destructive">{importError}</p>
            )}
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
