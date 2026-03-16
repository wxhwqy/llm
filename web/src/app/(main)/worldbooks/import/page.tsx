"use client";

import { useState, useRef } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  ArrowLeft,
  Upload,
  FileJson,
  CheckCircle2,
  AlertCircle,
  Loader2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { useImportWorldBook } from "@/hooks/use-queries";
import { cn } from "@/lib/utils";

export default function ImportWorldBookPage() {
  const router = useRouter();
  const importMutation = useImportWorldBook();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFileSelect = (file: File) => {
    if (file.type === "application/json" || file.name.endsWith(".json")) {
      setSelectedFile(file);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFileSelect(file);
  };

  const handleImport = () => {
    if (!selectedFile) return;
    importMutation.mutate(selectedFile, {
      onSuccess: (res) => {
        router.push(`/worldbooks/${res.data.id}`);
      },
    });
  };

  return (
    <div className="h-full overflow-auto overscroll-none">
      <div className="mx-auto max-w-2xl px-4 py-6 lg:px-6">
        <div className="flex items-center gap-3 mb-6">
          <Link href="/worldbooks">
            <Button variant="ghost" size="icon" className="h-8 w-8">
              <ArrowLeft className="h-4 w-4" />
            </Button>
          </Link>
          <h1 className="text-xl font-bold flex-1">导入世界书</h1>
        </div>

        <div className="space-y-6">
          <div>
            <p className="text-sm text-muted-foreground mb-4">
              支持导入以下格式的 JSON 文件：
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div className="rounded-lg border p-4">
                <h3 className="font-medium text-sm mb-1">原生格式</h3>
                <p className="text-xs text-muted-foreground">
                  从本平台导出的世界书 JSON 文件
                </p>
              </div>
              <div className="rounded-lg border p-4">
                <h3 className="font-medium text-sm mb-1">SillyTavern Lorebook</h3>
                <p className="text-xs text-muted-foreground">
                  SillyTavern 的 Lorebook / World Info 格式
                </p>
              </div>
            </div>
          </div>

          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={cn(
              "border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-colors",
              dragOver
                ? "border-violet-500 bg-violet-500/5"
                : selectedFile
                  ? "border-green-500/50 bg-green-500/5"
                  : "border-muted-foreground/20 hover:border-muted-foreground/40"
            )}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".json,application/json"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) handleFileSelect(file);
              }}
            />

            {selectedFile ? (
              <div className="space-y-2">
                <CheckCircle2 className="h-10 w-10 mx-auto text-green-500" />
                <div>
                  <p className="font-medium text-sm">{selectedFile.name}</p>
                  <p className="text-xs text-muted-foreground mt-0.5">
                    {(selectedFile.size / 1024).toFixed(1)} KB
                  </p>
                </div>
                <p className="text-xs text-muted-foreground">
                  点击重新选择文件
                </p>
              </div>
            ) : (
              <div className="space-y-2">
                <FileJson className="h-10 w-10 mx-auto text-muted-foreground/40" />
                <div>
                  <p className="text-sm font-medium">
                    拖放 JSON 文件到这里，或点击选择
                  </p>
                  <p className="text-xs text-muted-foreground mt-1">
                    支持 .json 文件
                  </p>
                </div>
              </div>
            )}
          </div>

          {importMutation.isError && (
            <div className="flex items-start gap-2 rounded-lg border border-destructive/50 bg-destructive/5 p-3">
              <AlertCircle className="h-4 w-4 text-destructive shrink-0 mt-0.5" />
              <div className="text-sm text-destructive">
                导入失败：{importMutation.error?.message || "未知错误"}
              </div>
            </div>
          )}

          <Button
            className="w-full"
            size="lg"
            onClick={handleImport}
            disabled={!selectedFile || importMutation.isPending}
          >
            {importMutation.isPending ? (
              <Loader2 className="h-4 w-4 mr-1.5 animate-spin" />
            ) : (
              <Upload className="h-4 w-4 mr-1.5" />
            )}
            {importMutation.isPending ? "导入中..." : "开始导入"}
          </Button>
        </div>
      </div>
    </div>
  );
}
