import { prisma } from "../db/prisma";
import { NotFoundError, ForbiddenError } from "../lib/errors";

// ─── Types ────────────────────────────────────────────────

export interface SessionRow {
  id: string;
  userId: string;
  characterId: string;
  modelId: string;
  title: string;
  usedTokens: number;
  maxTokens: number;
  createdAt: Date;
  updatedAt: Date;
  character: { name: string; avatar: string | null };
  messages?: { content: string }[];
  personalBooks?: { worldBookId: string }[];
}

// ─── Queries ──────────────────────────────────────────────

export async function findSessionById(id: string) {
  return prisma.chatSession.findUnique({ where: { id } });
}

export async function findSessionWithCharacter(id: string) {
  return prisma.chatSession.findUnique({
    where: { id },
    include: { character: true },
  });
}

/**
 * Find session and verify ownership. Throws if not found or not owned.
 */
export async function findOwnedSession(id: string, userId: string) {
  const session = await prisma.chatSession.findUnique({ where: { id } });
  if (!session) throw new NotFoundError("会话");
  if (session.userId !== userId) throw new ForbiddenError();
  return session;
}

export async function findOwnedSessionWithCharacter(id: string, userId: string) {
  const session = await prisma.chatSession.findUnique({
    where: { id },
    include: { character: true },
  });
  if (!session) throw new NotFoundError("会话");
  if (session.userId !== userId) throw new ForbiddenError();
  return session;
}

export async function listSessionsPaginated(userId: string, page: number, pageSize: number) {
  const [items, total] = await Promise.all([
    prisma.chatSession.findMany({
      where: { userId },
      orderBy: { updatedAt: "desc" },
      skip: (page - 1) * pageSize,
      take: pageSize,
      include: {
        character: { select: { name: true, avatar: true } },
        messages: { orderBy: { createdAt: "desc" }, take: 1, select: { content: true } },
        personalBooks: { select: { worldBookId: true } },
      },
    }),
    prisma.chatSession.count({ where: { userId } }),
  ]);
  return { items, total };
}

// ─── Mutations ────────────────────────────────────────────

export async function createSession(data: {
  userId: string;
  characterId: string;
  modelId: string;
  title: string;
  maxTokens: number;
}) {
  return prisma.chatSession.create({ data });
}

export async function updateSessionFields(
  id: string,
  data: Record<string, unknown>,
  include?: Record<string, unknown>,
) {
  return prisma.chatSession.update({ where: { id }, data, include });
}

export async function deleteSessionById(id: string) {
  return prisma.chatSession.delete({ where: { id } });
}

export async function replacePersonalWorldBooks(sessionId: string, worldBookIds: string[]) {
  await prisma.sessionWorldBook.deleteMany({ where: { sessionId } });
  if (worldBookIds.length) {
    await prisma.sessionWorldBook.createMany({
      data: worldBookIds.map((worldBookId) => ({ sessionId, worldBookId })),
    });
  }
}
