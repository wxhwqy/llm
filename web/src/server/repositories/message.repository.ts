import { prisma } from "../db/prisma";

// ─── Types ────────────────────────────────────────────────

export interface MessageRow {
  id: string;
  sessionId: string;
  role: string;
  content: string;
  tokenCount: number;
  isCompressed: boolean;
  createdAt: Date;
  editedAt: Date | null;
}

// ─── Queries ──────────────────────────────────────────────

export async function findMessageById(id: string) {
  return prisma.chatMessage.findUnique({ where: { id } });
}

export async function findMessagesBySession(
  sessionId: string,
  opts?: { excludeId?: string; select?: Record<string, boolean> },
) {
  const where: Record<string, unknown> = { sessionId };
  if (opts?.excludeId) where.id = { not: opts.excludeId };

  return prisma.chatMessage.findMany({
    where,
    orderBy: { createdAt: "asc" },
    ...(opts?.select ? { select: opts.select } : {}),
  });
}

export async function findMessagesCursor(
  sessionId: string,
  cursor: string | undefined,
  limit: number,
) {
  const where: Record<string, unknown> = { sessionId };
  if (cursor) {
    const cursorMsg = await prisma.chatMessage.findUnique({ where: { id: cursor } });
    if (cursorMsg) {
      where.createdAt = { lt: cursorMsg.createdAt };
    }
  }

  const messages = await prisma.chatMessage.findMany({
    where,
    orderBy: { createdAt: "desc" },
    take: limit + 1,
  });

  const hasMore = messages.length > limit;
  const result = (hasMore ? messages.slice(0, limit) : messages).reverse();
  const nextCursor = hasMore ? result[0]?.id ?? null : null;

  return { data: result, hasMore, nextCursor };
}

export async function findLastMessageByRole(sessionId: string, role: string) {
  return prisma.chatMessage.findFirst({
    where: { sessionId, role },
    orderBy: { createdAt: "desc" },
  });
}

// ─── Mutations ────────────────────────────────────────────

export async function createMessage(data: {
  sessionId: string;
  role: string;
  content: string;
  tokenCount: number;
}) {
  return prisma.chatMessage.create({ data });
}

export async function updateMessage(id: string, data: Record<string, unknown>) {
  return prisma.chatMessage.update({ where: { id }, data });
}

export async function deleteMessage(id: string) {
  return prisma.chatMessage.delete({ where: { id } });
}

// ─── Compression ─────────────────────────────────────────

export async function findUncompressedMessages(sessionId: string) {
  return prisma.chatMessage.findMany({
    where: { sessionId, isCompressed: false, role: { not: "system" } },
    orderBy: { createdAt: "asc" },
  });
}

export async function markMessagesCompressed(ids: string[]) {
  return prisma.chatMessage.updateMany({
    where: { id: { in: ids } },
    data: { isCompressed: true },
  });
}

// ─── Token Usage ──────────────────────────────────────────

export async function createTokenUsage(data: {
  userId: string;
  sessionId: string;
  messageId: string;
  modelId: string;
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
}) {
  return prisma.tokenUsage.create({ data });
}
