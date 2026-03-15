"use client";

import { useState } from "react";
import {
  Mail,
  Calendar,
  MessageCircle,
  Zap,
  TrendingUp,
  Pencil,
  Lock,
  Check,
  X,
  Loader2,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { useTokenUsage, useUpdateProfile, useChangePassword } from "@/hooks/use-queries";
import { useAuthGuard } from "@/hooks/use-require-auth";
import { ApiClientError } from "@/lib/api-client";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";

type TimeRange = "daily" | "weekly" | "monthly";

function StatCard({
  icon: Icon,
  label,
  value,
  sub,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: string;
  sub?: string;
}) {
  return (
    <div className="rounded-xl border bg-card p-4">
      <div className="flex items-center gap-2 text-muted-foreground mb-2">
        <Icon className="h-4 w-4" />
        <span className="text-xs font-medium">{label}</span>
      </div>
      <div className="text-2xl font-bold">{value}</div>
      {sub && <p className="text-xs text-muted-foreground mt-1">{sub}</p>}
    </div>
  );
}

function StatCardSkeleton() {
  return (
    <div className="rounded-xl border bg-card p-4 space-y-2">
      <Skeleton className="h-4 w-24" />
      <Skeleton className="h-8 w-16" />
      <Skeleton className="h-3 w-20" />
    </div>
  );
}

function formatK(n: number) {
  return (n / 1000).toFixed(1) + "K";
}

export default function ProfilePage() {
  const { user, isLoading: userLoading } = useAuthGuard();
  const { data: usageData, isLoading: usageLoading } = useTokenUsage();
  const updateProfile = useUpdateProfile();
  const changePassword = useChangePassword();
  const [timeRange, setTimeRange] = useState<TimeRange>("daily");

  // Profile edit state
  const [editingProfile, setEditingProfile] = useState(false);
  const [editUsername, setEditUsername] = useState("");
  const [editEmail, setEditEmail] = useState("");
  const [profileError, setProfileError] = useState("");
  const [profileSuccess, setProfileSuccess] = useState("");

  // Password change state
  const [changingPassword, setChangingPassword] = useState(false);
  const [oldPassword, setOldPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [pwError, setPwError] = useState("");
  const [pwSuccess, setPwSuccess] = useState("");

  const usage = usageData?.data;
  const isLoading = userLoading || usageLoading;

  if (isLoading || !user || !usage) {
    return (
      <div className="h-full overflow-auto overscroll-none">
        <div className="mx-auto max-w-4xl px-4 py-6 lg:px-6 space-y-8">
          <div className="flex items-center gap-4">
            <Skeleton className="h-16 w-16 rounded-full" />
            <div className="space-y-2">
              <Skeleton className="h-6 w-32" />
              <Skeleton className="h-4 w-48" />
            </div>
          </div>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            {Array.from({ length: 4 }).map((_, i) => (
              <StatCardSkeleton key={i} />
            ))}
          </div>
          <Skeleton className="h-[380px] w-full rounded-xl" />
        </div>
      </div>
    );
  }

  const { summary, timeline } = usage;
  const initial = user.username.charAt(0).toUpperCase();
  const roleBadge = user.role === "admin" ? "管理员" : "用户";
  const joinDate = user.createdAt.split("T")[0];

  const tabs: { key: TimeRange; label: string }[] = [
    { key: "daily", label: "按日" },
    { key: "weekly", label: "按周" },
    { key: "monthly", label: "按月" },
  ];

  const startEditProfile = () => {
    setEditUsername(user.username);
    setEditEmail(user.email);
    setProfileError("");
    setProfileSuccess("");
    setEditingProfile(true);
  };

  const handleSaveProfile = () => {
    setProfileError("");
    const data: { username?: string; email?: string } = {};
    if (editUsername.trim() !== user.username) data.username = editUsername.trim();
    if (editEmail.trim() !== user.email) data.email = editEmail.trim();
    if (Object.keys(data).length === 0) {
      setEditingProfile(false);
      return;
    }
    updateProfile.mutate(data, {
      onSuccess: () => {
        setEditingProfile(false);
        setProfileSuccess("个人资料已更新");
        setTimeout(() => setProfileSuccess(""), 3000);
      },
      onError: (err) => {
        if (err instanceof ApiClientError && err.error.code === "CONFLICT") {
          setProfileError("用户名或邮箱已被占用");
        } else {
          setProfileError("更新失败");
        }
      },
    });
  };

  const handleChangePassword = () => {
    setPwError("");
    if (newPassword.length < 8) {
      setPwError("新密码至少 8 位");
      return;
    }
    changePassword.mutate(
      { oldPassword, newPassword },
      {
        onSuccess: () => {
          setChangingPassword(false);
          setOldPassword("");
          setNewPassword("");
          setPwSuccess("密码已修改");
          setTimeout(() => setPwSuccess(""), 3000);
        },
        onError: (err) => {
          if (err instanceof ApiClientError && err.error.code === "INVALID_CREDENTIALS") {
            setPwError("当前密码错误");
          } else {
            setPwError("修改失败");
          }
        },
      },
    );
  };

  return (
    <div className="h-full overflow-auto overscroll-none">
      <div className="mx-auto max-w-4xl px-4 py-6 lg:px-6">
        {/* ---- User Info ---- */}
        <div className="flex items-start gap-4 mb-8">
          <div className="h-16 w-16 rounded-full bg-violet-600 flex items-center justify-center text-2xl font-bold text-white shrink-0">
            {initial}
          </div>
          <div className="flex-1 min-w-0">
            {editingProfile ? (
              <div className="space-y-3">
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  <div>
                    <label className="text-xs text-muted-foreground mb-1 block">用户名</label>
                    <Input
                      value={editUsername}
                      onChange={(e) => setEditUsername(e.target.value)}
                      placeholder="2-20 个字符"
                    />
                  </div>
                  <div>
                    <label className="text-xs text-muted-foreground mb-1 block">邮箱</label>
                    <Input
                      type="email"
                      value={editEmail}
                      onChange={(e) => setEditEmail(e.target.value)}
                    />
                  </div>
                </div>
                {profileError && <p className="text-sm text-destructive">{profileError}</p>}
                <div className="flex gap-2">
                  <Button
                    size="sm"
                    onClick={handleSaveProfile}
                    disabled={updateProfile.isPending}
                  >
                    {updateProfile.isPending ? (
                      <Loader2 className="h-3.5 w-3.5 mr-1 animate-spin" />
                    ) : (
                      <Check className="h-3.5 w-3.5 mr-1" />
                    )}
                    保存
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => setEditingProfile(false)}
                  >
                    <X className="h-3.5 w-3.5 mr-1" />
                    取消
                  </Button>
                </div>
              </div>
            ) : (
              <>
                <div className="flex items-center gap-2">
                  <h1 className="text-xl font-bold">{user.username}</h1>
                  <Badge>{roleBadge}</Badge>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-7 px-2 text-muted-foreground"
                    onClick={startEditProfile}
                  >
                    <Pencil className="h-3 w-3" />
                  </Button>
                </div>
                <div className="flex items-center gap-4 mt-1 text-sm text-muted-foreground">
                  <span className="flex items-center gap-1">
                    <Mail className="h-3.5 w-3.5" />
                    {user.email}
                  </span>
                  <span className="flex items-center gap-1">
                    <Calendar className="h-3.5 w-3.5" />
                    {joinDate} 注册
                  </span>
                </div>
                {profileSuccess && (
                  <p className="text-sm text-emerald-500 mt-1">{profileSuccess}</p>
                )}
              </>
            )}
          </div>
        </div>

        {/* ---- Password Change ---- */}
        <div className="mb-8">
          {changingPassword ? (
            <div className="rounded-xl border bg-card p-4 space-y-3 max-w-md">
              <h3 className="text-sm font-semibold flex items-center gap-2">
                <Lock className="h-4 w-4" />
                修改密码
              </h3>
              <div>
                <label className="text-xs text-muted-foreground mb-1 block">当前密码</label>
                <Input
                  type="password"
                  value={oldPassword}
                  onChange={(e) => setOldPassword(e.target.value)}
                  autoComplete="current-password"
                />
              </div>
              <div>
                <label className="text-xs text-muted-foreground mb-1 block">新密码</label>
                <Input
                  type="password"
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                  placeholder="至少 8 位"
                  autoComplete="new-password"
                />
              </div>
              {pwError && <p className="text-sm text-destructive">{pwError}</p>}
              <div className="flex gap-2">
                <Button
                  size="sm"
                  onClick={handleChangePassword}
                  disabled={changePassword.isPending}
                >
                  {changePassword.isPending ? (
                    <Loader2 className="h-3.5 w-3.5 mr-1 animate-spin" />
                  ) : (
                    <Check className="h-3.5 w-3.5 mr-1" />
                  )}
                  确认修改
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => {
                    setChangingPassword(false);
                    setOldPassword("");
                    setNewPassword("");
                    setPwError("");
                  }}
                >
                  取消
                </Button>
              </div>
            </div>
          ) : (
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  setPwSuccess("");
                  setChangingPassword(true);
                }}
              >
                <Lock className="h-3.5 w-3.5 mr-1.5" />
                修改密码
              </Button>
              {pwSuccess && (
                <span className="text-sm text-emerald-500">{pwSuccess}</span>
              )}
            </div>
          )}
        </div>

        <Separator className="mb-6" />

        {/* ---- Stats ---- */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-8">
          <StatCard
            icon={Zap}
            label="总 Token 使用"
            value={formatK(summary.totalTokens)}
            sub="Prompt + Completion"
          />
          <StatCard
            icon={TrendingUp}
            label="Prompt Tokens"
            value={formatK(summary.totalPromptTokens)}
          />
          <StatCard
            icon={TrendingUp}
            label="Completion Tokens"
            value={formatK(summary.totalCompletionTokens)}
          />
          <StatCard
            icon={MessageCircle}
            label="总会话数"
            value={String(summary.totalSessions)}
            sub={`${summary.totalMessages} 条消息`}
          />
        </div>

        <Separator className="mb-6" />

        {/* ---- Usage Chart ---- */}
        <div>
          <h2 className="text-lg font-semibold mb-4">Token 使用趋势</h2>

          <div className="flex items-center gap-1.5 mb-4 p-1 bg-muted/50 rounded-lg w-fit">
            {tabs.map(({ key, label }) => (
              <button
                key={key}
                onClick={() => setTimeRange(key)}
                className={
                  "px-3 py-1.5 rounded-md text-sm font-medium transition-colors " +
                  (timeRange === key
                    ? "bg-background shadow-sm text-foreground"
                    : "text-muted-foreground hover:text-foreground")
                }
              >
                {label}
              </button>
            ))}
          </div>

          <div className="rounded-xl border bg-card p-4">
            <ResponsiveContainer width="100%" height={320}>
              <AreaChart data={timeline}>
                <defs>
                  <linearGradient
                    id="promptGrad"
                    x1="0"
                    y1="0"
                    x2="0"
                    y2="1"
                  >
                    <stop
                      offset="5%"
                      stopColor="oklch(0.488 0.243 264.376)"
                      stopOpacity={0.3}
                    />
                    <stop
                      offset="95%"
                      stopColor="oklch(0.488 0.243 264.376)"
                      stopOpacity={0}
                    />
                  </linearGradient>
                  <linearGradient
                    id="completionGrad"
                    x1="0"
                    y1="0"
                    x2="0"
                    y2="1"
                  >
                    <stop
                      offset="5%"
                      stopColor="oklch(0.696 0.17 162.48)"
                      stopOpacity={0.3}
                    />
                    <stop
                      offset="95%"
                      stopColor="oklch(0.696 0.17 162.48)"
                      stopOpacity={0}
                    />
                  </linearGradient>
                </defs>
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="oklch(1 0 0 / 5%)"
                />
                <XAxis
                  dataKey="date"
                  tick={{ fill: "oklch(0.708 0 0)", fontSize: 12 }}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  tick={{ fill: "oklch(0.708 0 0)", fontSize: 12 }}
                  axisLine={false}
                  tickLine={false}
                  tickFormatter={(v) => `${(v / 1000).toFixed(0)}K`}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "oklch(0.205 0 0)",
                    border: "1px solid oklch(1 0 0 / 10%)",
                    borderRadius: "8px",
                    fontSize: 12,
                  }}
                  formatter={(value) => [
                    Number(value).toLocaleString(),
                    "",
                  ]}
                />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Area
                  type="monotone"
                  dataKey="promptTokens"
                  name="Prompt"
                  stroke="oklch(0.488 0.243 264.376)"
                  fill="url(#promptGrad)"
                  strokeWidth={2}
                />
                <Area
                  type="monotone"
                  dataKey="completionTokens"
                  name="Completion"
                  stroke="oklch(0.696 0.17 162.48)"
                  fill="url(#completionGrad)"
                  strokeWidth={2}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="pb-8" />
      </div>
    </div>
  );
}
