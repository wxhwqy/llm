"use client";

import { useState } from "react";
import { Sparkles, Eye, EyeOff } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import Link from "next/link";

export default function LoginPage() {
  const [showPassword, setShowPassword] = useState(false);

  return (
    <div className="min-h-dvh flex items-center justify-center bg-background p-4">
      <div className="w-full max-w-sm">
        {/* Logo */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center h-14 w-14 rounded-2xl bg-violet-600/10 mb-4">
            <Sparkles className="h-7 w-7 text-violet-400" />
          </div>
          <h1 className="text-2xl font-bold">AI Chat</h1>
          <p className="text-sm text-muted-foreground mt-1">
            角色扮演聊天平台
          </p>
        </div>

        {/* Form */}
        <div className="rounded-xl border bg-card p-6 space-y-4">
          <div>
            <label className="text-sm font-medium mb-1.5 block">
              邮箱
            </label>
            <Input
              type="email"
              placeholder="name@example.com"
              defaultValue="admin@example.com"
            />
          </div>

          <div>
            <label className="text-sm font-medium mb-1.5 block">
              密码
            </label>
            <div className="relative">
              <Input
                type={showPassword ? "text" : "password"}
                placeholder="输入密码"
                defaultValue="password"
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

          <Link href="/characters">
            <Button className="w-full mt-2" size="lg">
              登录
            </Button>
          </Link>
        </div>

        <p className="text-center text-sm text-muted-foreground mt-4">
          没有账号？{" "}
          <span className="text-violet-400 hover:underline cursor-pointer">
            注册
          </span>
        </p>
      </div>
    </div>
  );
}
