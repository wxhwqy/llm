"use client";

import { useState } from "react";
import {
  Mail,
  Calendar,
  MessageCircle,
  Zap,
  TrendingUp,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { useCurrentUser, useTokenUsage } from "@/hooks/use-queries";
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
  const { data: userData, isLoading: userLoading } = useCurrentUser();
  const { data: usageData, isLoading: usageLoading } = useTokenUsage();
  const [timeRange, setTimeRange] = useState<TimeRange>("daily");

  const user = userData?.data;
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

  return (
    <div className="h-full overflow-auto overscroll-none">
      <div className="mx-auto max-w-4xl px-4 py-6 lg:px-6">
        <div className="flex items-center gap-4 mb-8">
          <div className="h-16 w-16 rounded-full bg-violet-600 flex items-center justify-center text-2xl font-bold text-white">
            {initial}
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold">{user.username}</h1>
              <Badge>{roleBadge}</Badge>
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
          </div>
        </div>

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
      </div>
    </div>
  );
}
