"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Sparkles, Eye, EyeOff, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import Link from "next/link";
import { useLogin } from "@/hooks/use-queries";
import { ApiClientError } from "@/lib/api-client";

export default function LoginPage() {
  const router = useRouter();
  const login = useLogin();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!email.trim() || !password.trim()) return;
    setError("");
    login.mutate(
      { email: email.trim(), password },
      {
        onSuccess: () => {
          router.push("/characters");
        },
        onError: (err) => {
          if (err instanceof ApiClientError) {
            if (err.error.code === "ACCOUNT_DISABLED") {
              setError("该账号已被禁用，请联系管理员");
            } else if (err.error.code === "INVALID_CREDENTIALS") {
              setError("邮箱或密码错误");
            } else {
              setError(err.error.message);
            }
          } else {
            setError("登录失败，请稍后再试");
          }
        },
      },
    );
  };

  return (
    <div className="min-h-dvh flex items-center justify-center bg-background p-4">
      <div className="w-full max-w-sm">
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center h-14 w-14 rounded-2xl bg-violet-600/10 mb-4">
            <Sparkles className="h-7 w-7 text-violet-400" />
          </div>
          <h1 className="text-2xl font-bold">AI Chat</h1>
          <p className="text-sm text-muted-foreground mt-1">
            角色扮演聊天平台
          </p>
        </div>

        <form
          onSubmit={handleSubmit}
          className="rounded-xl border bg-card p-6 space-y-4"
        >
          <div>
            <label className="text-sm font-medium mb-1.5 block">邮箱</label>
            <Input
              type="email"
              placeholder="name@example.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              autoComplete="email"
            />
          </div>

          <div>
            <label className="text-sm font-medium mb-1.5 block">密码</label>
            <div className="relative">
              <Input
                type={showPassword ? "text" : "password"}
                placeholder="输入密码"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                autoComplete="current-password"
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
              >
                {showPassword ? (
                  <EyeOff className="h-4 w-4" />
                ) : (
                  <Eye className="h-4 w-4" />
                )}
              </button>
            </div>
          </div>

          {error && (
            <p className="text-sm text-destructive">{error}</p>
          )}

          <Button
            type="submit"
            className="w-full mt-2"
            size="lg"
            disabled={login.isPending}
          >
            {login.isPending && (
              <Loader2 className="h-4 w-4 mr-1.5 animate-spin" />
            )}
            {login.isPending ? "登录中..." : "登录"}
          </Button>
        </form>

        <p className="text-center text-sm text-muted-foreground mt-4">
          没有账号？{" "}
          <Link
            href="/register"
            className="text-violet-400 hover:underline"
          >
            注册
          </Link>
        </p>
      </div>
    </div>
  );
}
