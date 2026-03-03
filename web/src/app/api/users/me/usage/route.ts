import { NextRequest } from "next/server";
import { withAuth, jsonOk } from "@/server/lib/response";
import { prisma } from "@/server/db/prisma";

export const GET = withAuth(async (req: NextRequest, _ctx, user) => {
  const url = req.nextUrl;
  const period = url.searchParams.get("period") ?? "daily";
  const fromStr = url.searchParams.get("from");
  const toStr = url.searchParams.get("to");

  const now = new Date();
  const from = fromStr ? new Date(fromStr) : new Date(now.getTime() - 30 * 86400000);
  const to = toStr ? new Date(toStr + "T23:59:59.999Z") : now;

  // Summary
  const usages = await prisma.tokenUsage.findMany({
    where: {
      userId: user.id,
      createdAt: { gte: from, lte: to },
    },
    select: {
      promptTokens: true,
      completionTokens: true,
      totalTokens: true,
      createdAt: true,
    },
  });

  const totalPromptTokens = usages.reduce((s, u) => s + u.promptTokens, 0);
  const totalCompletionTokens = usages.reduce((s, u) => s + u.completionTokens, 0);
  const totalTokens = usages.reduce((s, u) => s + u.totalTokens, 0);

  const [totalSessions, totalMessages] = await Promise.all([
    prisma.chatSession.count({ where: { userId: user.id } }),
    prisma.chatMessage.count({
      where: { session: { userId: user.id } },
    }),
  ]);

  // Timeline aggregation
  const buckets = new Map<string, { promptTokens: number; completionTokens: number; totalTokens: number }>();

  for (const u of usages) {
    let key: string;
    const d = u.createdAt;
    if (period === "monthly") {
      key = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-01`;
    } else if (period === "weekly") {
      const day = d.getDay();
      const monday = new Date(d);
      monday.setDate(d.getDate() - ((day + 6) % 7));
      key = monday.toISOString().slice(0, 10);
    } else {
      key = d.toISOString().slice(0, 10);
    }

    const existing = buckets.get(key) ?? { promptTokens: 0, completionTokens: 0, totalTokens: 0 };
    existing.promptTokens += u.promptTokens;
    existing.completionTokens += u.completionTokens;
    existing.totalTokens += u.totalTokens;
    buckets.set(key, existing);
  }

  const timeline = Array.from(buckets.entries())
    .map(([date, vals]) => ({ date, ...vals }))
    .sort((a, b) => a.date.localeCompare(b.date));

  return jsonOk({
    summary: { totalPromptTokens, totalCompletionTokens, totalTokens, totalSessions, totalMessages },
    timeline,
  });
});
