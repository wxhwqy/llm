"use client";

import { useState } from "react";
import {
  Search,
  Shield,
  ShieldOff,
  Trash2,
  Loader2,
  UserX,
  UserCheck,
  Zap,
  MessageCircle,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
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
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  useAdminUsers,
  useAdminUpdateUser,
  useAdminDeleteUser,
} from "@/hooks/use-queries";
import { useAuthGuard } from "@/hooks/use-require-auth";
import type { AdminUserItem } from "@/types/user";

function formatK(n: number) {
  if (n < 1000) return String(n);
  return (n / 1000).toFixed(1) + "K";
}

function UserRowSkeleton() {
  return (
    <div className="flex items-center gap-4 rounded-lg border bg-card p-4">
      <Skeleton className="h-10 w-10 rounded-full" />
      <div className="flex-1 space-y-2">
        <Skeleton className="h-4 w-32" />
        <Skeleton className="h-3 w-48" />
      </div>
      <Skeleton className="h-8 w-20" />
    </div>
  );
}

function UserRow({
  user,
  currentUserId,
}: {
  user: AdminUserItem;
  currentUserId: string;
}) {
  const updateUser = useAdminUpdateUser();
  const deleteUser = useAdminDeleteUser();
  const [deleteOpen, setDeleteOpen] = useState(false);

  const isSelf = user.id === currentUserId;
  const initial = user.username.charAt(0).toUpperCase();
  const joinDate = user.createdAt.split("T")[0];

  const handleToggleRole = () => {
    updateUser.mutate({
      id: user.id,
      role: user.role === "admin" ? "user" : "admin",
    });
  };

  const handleToggleStatus = () => {
    updateUser.mutate({
      id: user.id,
      status: user.status === "active" ? "disabled" : "active",
    });
  };

  const handleDelete = () => {
    deleteUser.mutate(user.id, {
      onSuccess: () => setDeleteOpen(false),
    });
  };

  return (
    <>
      <div className="flex items-center gap-4 rounded-lg border bg-card p-4">
        <div className="h-10 w-10 rounded-full bg-violet-600 flex items-center justify-center text-sm font-bold text-white shrink-0">
          {initial}
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="font-medium text-sm">{user.username}</span>
            <Badge variant={user.role === "admin" ? "default" : "secondary"}>
              {user.role === "admin" ? "管理员" : "用户"}
            </Badge>
            {user.status === "disabled" && (
              <Badge variant="destructive">已禁用</Badge>
            )}
          </div>
          <p className="text-xs text-muted-foreground mt-0.5 truncate">
            {user.email}
          </p>
          <div className="flex items-center gap-3 mt-1 text-xs text-muted-foreground">
            <span>{joinDate} 注册</span>
            <span className="flex items-center gap-0.5">
              <MessageCircle className="h-3 w-3" />
              {user.sessionCount} 会话
            </span>
            <span className="flex items-center gap-0.5">
              <Zap className="h-3 w-3" />
              {formatK(user.totalTokens)} tokens
            </span>
          </div>
        </div>

        {!isSelf && (
          <div className="flex items-center gap-1.5 shrink-0">
            <Button
              variant="ghost"
              size="sm"
              className="h-8 px-2 text-xs"
              onClick={handleToggleRole}
              disabled={updateUser.isPending}
              title={user.role === "admin" ? "降为用户" : "设为管理员"}
            >
              {user.role === "admin" ? (
                <ShieldOff className="h-3.5 w-3.5" />
              ) : (
                <Shield className="h-3.5 w-3.5" />
              )}
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="h-8 px-2 text-xs"
              onClick={handleToggleStatus}
              disabled={updateUser.isPending}
              title={user.status === "active" ? "禁用用户" : "启用用户"}
            >
              {user.status === "active" ? (
                <UserX className="h-3.5 w-3.5" />
              ) : (
                <UserCheck className="h-3.5 w-3.5" />
              )}
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="h-8 px-2 text-xs text-destructive hover:text-destructive"
              onClick={() => setDeleteOpen(true)}
              title="删除用户"
            >
              <Trash2 className="h-3.5 w-3.5" />
            </Button>
          </div>
        )}
      </div>

      <Dialog open={deleteOpen} onOpenChange={setDeleteOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>确认删除</DialogTitle>
            <DialogDescription>
              确定要删除用户 <strong>{user.username}</strong> 吗？该操作将同时删除该用户的所有会话记录，且无法撤销。
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleteOpen(false)}>
              取消
            </Button>
            <Button
              variant="destructive"
              onClick={handleDelete}
              disabled={deleteUser.isPending}
            >
              {deleteUser.isPending && (
                <Loader2 className="h-3.5 w-3.5 mr-1 animate-spin" />
              )}
              删除
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}

export default function AdminUsersPage() {
  const { user: currentUser, isLoading: authLoading } = useAuthGuard();
  const [search, setSearch] = useState("");
  const [roleFilter, setRoleFilter] = useState("all");
  const [statusFilter, setStatusFilter] = useState("all");

  const { data, isLoading: usersLoading } = useAdminUsers({
    search: search || undefined,
    role: roleFilter !== "all" ? roleFilter : undefined,
    status: statusFilter !== "all" ? statusFilter : undefined,
  });

  const isLoading = authLoading || usersLoading;
  const users = data?.data ?? [];

  return (
    <div className="h-full overflow-auto overscroll-none">
      <div className="mx-auto max-w-4xl px-4 py-6 lg:px-6">
        <h1 className="text-xl font-bold mb-6">用户管理</h1>

        {/* Filters */}
        <div className="flex flex-col sm:flex-row gap-3 mb-6">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="搜索用户名或邮箱..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="pl-9"
            />
          </div>
          <Select value={roleFilter} onValueChange={setRoleFilter}>
            <SelectTrigger className="w-[120px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">全部角色</SelectItem>
              <SelectItem value="admin">管理员</SelectItem>
              <SelectItem value="user">用户</SelectItem>
            </SelectContent>
          </Select>
          <Select value={statusFilter} onValueChange={setStatusFilter}>
            <SelectTrigger className="w-[120px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">全部状态</SelectItem>
              <SelectItem value="active">正常</SelectItem>
              <SelectItem value="disabled">已禁用</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* User list */}
        <div className="space-y-3">
          {isLoading ? (
            Array.from({ length: 5 }).map((_, i) => (
              <UserRowSkeleton key={i} />
            ))
          ) : users.length === 0 ? (
            <div className="text-center py-12 text-muted-foreground text-sm">
              没有找到匹配的用户
            </div>
          ) : (
            users.map((u) => (
              <UserRow
                key={u.id}
                user={u}
                currentUserId={currentUser?.id ?? ""}
              />
            ))
          )}
        </div>
      </div>
    </div>
  );
}
