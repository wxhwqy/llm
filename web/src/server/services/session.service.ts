import { config } from "../lib/config";
import { estimateTokens } from "./token-counter.service";
import * as sessionRepo from "../repositories/session.repository";
import * as messageRepo from "../repositories/message.repository";

// ─── Formatters ───────────────────────────────────────────

function formatSessionSummary(s: {
  id: string;
  characterId: string;
  modelId: string;
  title: string;
  usedTokens: number;
  maxTokens: number;
  createdAt: Date;
  updatedAt: Date;
  character: { name: string; avatar: string | null };
  messages: { content: string }[];
  personalBooks: { worldBookId: string }[];
}) {
  return {
    id: s.id,
    characterId: s.characterId,
    characterName: s.character.name,
    characterAvatar: s.character.avatar,
    modelId: s.modelId,
    title: s.title,
    lastMessage: s.messages[0]?.content.slice(0, 100) ?? "",
    personalWorldBookIds: s.personalBooks.map((b) => b.worldBookId),
    contextUsage: { usedTokens: s.usedTokens, maxTokens: s.maxTokens },
    createdAt: s.createdAt.toISOString(),
    updatedAt: s.updatedAt.toISOString(),
  };
}

export function formatMessage(m: {
  id: string;
  role: string;
  content: string;
  tokenCount: number;
  isCompressed: boolean;
  createdAt: Date;
  editedAt: Date | null;
}) {
  return {
    id: m.id,
    role: m.role,
    content: m.content,
    tokenCount: m.tokenCount,
    isCompressed: m.isCompressed,
    createdAt: m.createdAt.toISOString(),
    editedAt: m.editedAt?.toISOString() ?? null,
  };
}

// ─── Session CRUD ─────────────────────────────────────────

export async function listSessions(userId: string, page: number, pageSize: number) {
  const { items, total } = await sessionRepo.listSessionsPaginated(userId, page, pageSize);
  const data = items.map((s) => formatSessionSummary(s as Parameters<typeof formatSessionSummary>[0]));
  return { data, pagination: { page, pageSize, total } };
}

export async function createSession(userId: string, characterId: string, modelId?: string) {
  const { prisma } = await import("../db/prisma");
  const character = await prisma.characterCard.findUnique({ where: { id: characterId } });
  if (!character) {
    const { NotFoundError } = await import("../lib/errors");
    throw new NotFoundError("角色卡");
  }

  const session = await sessionRepo.createSession({
    userId,
    characterId,
    modelId: modelId || config.defaultModelId,
    title: "新会话",
    maxTokens: 8192,
  });

  const messages = [];
  if (character.firstMessage) {
    const msg = await messageRepo.createMessage({
      sessionId: session.id,
      role: "assistant",
      content: character.firstMessage,
      tokenCount: estimateTokens(character.firstMessage),
    });
    messages.push(formatMessage(msg));
  }

  return {
    id: session.id,
    characterId,
    characterName: character.name,
    characterAvatar: character.avatar,
    modelId: session.modelId,
    title: session.title,
    lastMessage: character.firstMessage?.slice(0, 100) ?? "",
    personalWorldBookIds: [] as string[],
    contextUsage: { usedTokens: session.usedTokens, maxTokens: session.maxTokens },
    messages,
    createdAt: session.createdAt.toISOString(),
    updatedAt: session.updatedAt.toISOString(),
  };
}

export async function updateSession(
  sessionId: string,
  userId: string,
  data: { modelId?: string; personalWorldBookIds?: string[]; title?: string },
) {
  await sessionRepo.findOwnedSession(sessionId, userId);

  if (data.personalWorldBookIds !== undefined) {
    await sessionRepo.replacePersonalWorldBooks(sessionId, data.personalWorldBookIds);
  }

  const updated = await sessionRepo.updateSessionFields(
    sessionId,
    {
      ...(data.modelId !== undefined ? { modelId: data.modelId } : {}),
      ...(data.title !== undefined ? { title: data.title } : {}),
    },
    {
      character: { select: { name: true, avatar: true } },
      personalBooks: { select: { worldBookId: true } },
      messages: { orderBy: { createdAt: "desc" }, take: 1, select: { content: true } },
    },
  ) as unknown as {
    id: string; characterId: string; modelId: string; title: string;
    usedTokens: number; maxTokens: number; createdAt: Date; updatedAt: Date;
    character: { name: string; avatar: string | null };
    messages: { content: string }[];
    personalBooks: { worldBookId: string }[];
  };

  return {
    id: updated.id,
    characterId: updated.characterId,
    characterName: updated.character.name,
    characterAvatar: updated.character.avatar,
    modelId: updated.modelId,
    title: updated.title,
    lastMessage: updated.messages[0]?.content.slice(0, 100) ?? "",
    personalWorldBookIds: updated.personalBooks.map((b) => b.worldBookId),
    contextUsage: { usedTokens: updated.usedTokens, maxTokens: updated.maxTokens },
    createdAt: updated.createdAt.toISOString(),
    updatedAt: updated.updatedAt.toISOString(),
  };
}

export async function deleteSession(sessionId: string, userId: string) {
  await sessionRepo.findOwnedSession(sessionId, userId);
  await sessionRepo.deleteSessionById(sessionId);
}
