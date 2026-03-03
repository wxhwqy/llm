import { prisma } from "../db/prisma";
import { parseJsonArray, toJsonString } from "../lib/json-fields";
import { NotFoundError, ForbiddenError } from "../lib/errors";

function formatEntry(e: Record<string, unknown>) {
  return {
    id: e.id as string,
    keywords: parseJsonArray(e.keywords as string),
    secondaryKeywords: parseJsonArray(e.secondaryKeywords as string),
    content: e.content as string,
    position: e.position as string,
    priority: e.priority as number,
    enabled: e.enabled as boolean,
    tokenCount: e.tokenCount as number,
  };
}

function formatWorldbook(wb: Record<string, unknown>, entryCount?: number) {
  return {
    id: wb.id as string,
    name: wb.name as string,
    description: wb.description as string,
    scope: wb.scope as string,
    userId: wb.userId as string,
    totalTokenCount: wb.totalTokenCount as number,
    characterCount: 0,
    createdAt: (wb.createdAt as Date).toISOString(),
    updatedAt: (wb.updatedAt as Date).toISOString(),
    ...(entryCount !== undefined ? { entryCount } : {}),
  };
}

export async function listWorldbooks(userId: string, scope?: string) {
  const where: Record<string, unknown> = {
    OR: [
      { scope: "global" },
      { scope: "personal", userId },
    ],
  };
  if (scope) {
    if (scope === "personal") {
      (where as { OR?: unknown }).OR = undefined;
      Object.assign(where, { scope: "personal", userId });
    } else {
      (where as { OR?: unknown }).OR = undefined;
      Object.assign(where, { scope: "global" });
    }
  }

  const items = await prisma.worldBook.findMany({
    where,
    include: {
      entries: { select: { id: true } },
      characters: { select: { characterId: true } },
    },
    orderBy: { updatedAt: "desc" },
  });

  return items.map((wb) => ({
    ...formatWorldbook(wb as unknown as Record<string, unknown>),
    entryCount: wb.entries.length,
    characterCount: wb.characters.length,
  }));
}

function checkWritePermission(wb: { scope: string; userId: string }, userId: string, userRole: string) {
  if (wb.scope === "global" && userRole !== "admin") throw new ForbiddenError("仅管理员可修改全局世界书");
  if (wb.scope === "personal" && wb.userId !== userId) throw new ForbiddenError("只能修改自己的世界书");
}

function checkReadPermission(wb: { scope: string; userId: string }, userId: string) {
  if (wb.scope === "personal" && wb.userId !== userId) throw new NotFoundError("世界书");
}

export async function getWorldbookById(id: string, userId: string) {
  const wb = await prisma.worldBook.findUnique({
    where: { id },
    include: {
      entries: { orderBy: { priority: "desc" } },
      characters: { select: { characterId: true } },
    },
  });
  if (!wb) throw new NotFoundError("世界书");
  checkReadPermission(wb, userId);
  return {
    ...formatWorldbook(wb as unknown as Record<string, unknown>),
    characterCount: wb.characters.length,
    entries: wb.entries.map((e) => formatEntry(e as unknown as Record<string, unknown>)),
  };
}

function estimateTokenCount(text: string): number {
  // Rough estimate: ~1.5 chars per token for CJK, ~4 chars per token for English
  const cjk = (text.match(/[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]/g) || []).length;
  const other = text.length - cjk;
  return Math.ceil(cjk / 1.5 + other / 4);
}

export async function createWorldbook(
  data: { name: string; description?: string; scope: string; entries?: Array<Record<string, unknown>> },
  userId: string,
  userRole: string,
) {
  if (data.scope === "global" && userRole !== "admin") throw new ForbiddenError("普通用户只能创建个人世界书");

  let totalTokenCount = 0;
  const entryData = (data.entries ?? []).map((e) => {
    const tc = estimateTokenCount(e.content as string);
    totalTokenCount += tc;
    return {
      keywords: toJsonString(e.keywords as string[]),
      secondaryKeywords: toJsonString((e.secondaryKeywords as string[]) ?? []),
      content: e.content as string,
      position: (e.position as string) ?? "after_system",
      priority: (e.priority as number) ?? 0,
      enabled: (e.enabled as boolean) ?? true,
      tokenCount: tc,
    };
  });

  const wb = await prisma.worldBook.create({
    data: {
      name: data.name,
      description: data.description ?? "",
      scope: data.scope,
      userId,
      totalTokenCount,
      entries: entryData.length ? { create: entryData } : undefined,
    },
    include: {
      entries: { orderBy: { priority: "desc" } },
      characters: { select: { characterId: true } },
    },
  });

  return {
    ...formatWorldbook(wb as unknown as Record<string, unknown>),
    characterCount: wb.characters.length,
    entries: wb.entries.map((e) => formatEntry(e as unknown as Record<string, unknown>)),
  };
}

export async function updateWorldbook(
  id: string, data: { name?: string; description?: string },
  userId: string, userRole: string,
) {
  const wb = await prisma.worldBook.findUnique({ where: { id } });
  if (!wb) throw new NotFoundError("世界书");
  checkWritePermission(wb, userId, userRole);

  const updated = await prisma.worldBook.update({
    where: { id },
    data,
    include: {
      entries: { orderBy: { priority: "desc" } },
      characters: { select: { characterId: true } },
    },
  });
  return {
    ...formatWorldbook(updated as unknown as Record<string, unknown>),
    characterCount: updated.characters.length,
    entries: updated.entries.map((e) => formatEntry(e as unknown as Record<string, unknown>)),
  };
}

export async function deleteWorldbook(id: string, userId: string, userRole: string) {
  const wb = await prisma.worldBook.findUnique({ where: { id } });
  if (!wb) throw new NotFoundError("世界书");
  checkWritePermission(wb, userId, userRole);
  await prisma.worldBook.delete({ where: { id } });
}

export async function exportWorldbook(id: string, userId: string) {
  return getWorldbookById(id, userId);
}

export async function importWorldbook(
  jsonStr: string, userId: string, userRole: string, scope?: string,
) {
  const json = JSON.parse(jsonStr);
  const finalScope = (userRole === "admin" && scope === "global") ? "global" : "personal";

  // Detect SillyTavern lorebook format (entries is an object, not array)
  let entries: Array<Record<string, unknown>> = [];
  if (json.entries && !Array.isArray(json.entries)) {
    entries = Object.values(json.entries).map((e: unknown) => {
      const entry = e as Record<string, unknown>;
      return {
        keywords: entry.key ?? entry.keywords ?? [],
        secondaryKeywords: entry.secondary_keys ?? entry.secondaryKeywords ?? [],
        content: entry.content ?? "",
        position: entry.position ?? "after_system",
        priority: entry.order ?? entry.priority ?? 0,
        enabled: entry.enabled ?? entry.disable !== true,
      };
    });
  } else if (Array.isArray(json.entries)) {
    entries = json.entries;
  }

  return createWorldbook(
    { name: json.name ?? "导入的世界书", description: json.description ?? "", scope: finalScope, entries },
    userId, userRole,
  );
}

export async function addEntry(
  worldbookId: string, data: Record<string, unknown>,
  userId: string, userRole: string,
) {
  const wb = await prisma.worldBook.findUnique({ where: { id: worldbookId } });
  if (!wb) throw new NotFoundError("世界书");
  checkWritePermission(wb, userId, userRole);

  const tc = estimateTokenCount(data.content as string);
  const entry = await prisma.worldBookEntry.create({
    data: {
      worldBookId: worldbookId,
      keywords: toJsonString(data.keywords as string[]),
      secondaryKeywords: toJsonString((data.secondaryKeywords as string[]) ?? []),
      content: data.content as string,
      position: (data.position as string) ?? "after_system",
      priority: (data.priority as number) ?? 0,
      enabled: (data.enabled as boolean) ?? true,
      tokenCount: tc,
    },
  });
  await prisma.worldBook.update({
    where: { id: worldbookId },
    data: { totalTokenCount: { increment: tc } },
  });
  return formatEntry(entry as unknown as Record<string, unknown>);
}

export async function updateEntry(
  worldbookId: string, entryId: string, data: Record<string, unknown>,
  userId: string, userRole: string,
) {
  const wb = await prisma.worldBook.findUnique({ where: { id: worldbookId } });
  if (!wb) throw new NotFoundError("世界书");
  checkWritePermission(wb, userId, userRole);

  const existing = await prisma.worldBookEntry.findUnique({ where: { id: entryId } });
  if (!existing || existing.worldBookId !== worldbookId) throw new NotFoundError("词条");

  const updateData: Record<string, unknown> = {};
  if (data.keywords !== undefined) updateData.keywords = toJsonString(data.keywords as string[]);
  if (data.secondaryKeywords !== undefined) updateData.secondaryKeywords = toJsonString(data.secondaryKeywords as string[]);
  if (data.content !== undefined) {
    updateData.content = data.content;
    updateData.tokenCount = estimateTokenCount(data.content as string);
  }
  if (data.position !== undefined) updateData.position = data.position;
  if (data.priority !== undefined) updateData.priority = data.priority;
  if (data.enabled !== undefined) updateData.enabled = data.enabled;

  const entry = await prisma.worldBookEntry.update({ where: { id: entryId }, data: updateData });

  // Recalculate total token count
  const allEntries = await prisma.worldBookEntry.findMany({
    where: { worldBookId: worldbookId },
    select: { tokenCount: true },
  });
  await prisma.worldBook.update({
    where: { id: worldbookId },
    data: { totalTokenCount: allEntries.reduce((sum, e) => sum + e.tokenCount, 0) },
  });

  return formatEntry(entry as unknown as Record<string, unknown>);
}

export async function deleteEntry(
  worldbookId: string, entryId: string,
  userId: string, userRole: string,
) {
  const wb = await prisma.worldBook.findUnique({ where: { id: worldbookId } });
  if (!wb) throw new NotFoundError("世界书");
  checkWritePermission(wb, userId, userRole);

  const existing = await prisma.worldBookEntry.findUnique({ where: { id: entryId } });
  if (!existing || existing.worldBookId !== worldbookId) throw new NotFoundError("词条");

  await prisma.worldBookEntry.delete({ where: { id: entryId } });
  await prisma.worldBook.update({
    where: { id: worldbookId },
    data: { totalTokenCount: { decrement: existing.tokenCount } },
  });
}
