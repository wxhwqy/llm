import { prisma } from "../db/prisma";
import { parseJsonArray, toJsonString } from "../lib/json-fields";
import { NotFoundError } from "../lib/errors";

function formatCharacterSummary(c: Record<string, unknown>) {
  return {
    id: c.id,
    name: c.name,
    avatar: c.avatar ?? null,
    coverImage: c.coverImage ?? null,
    description: c.description,
    tags: parseJsonArray(c.tags as string),
    source: c.source,
    createdAt: (c.createdAt as Date).toISOString(),
    updatedAt: (c.updatedAt as Date).toISOString(),
  };
}

function formatCharacterDetail(c: Record<string, unknown>, worldBookIds: string[]) {
  return {
    ...formatCharacterSummary(c),
    personality: c.personality,
    scenario: c.scenario,
    systemPrompt: c.systemPrompt,
    firstMessage: c.firstMessage,
    alternateGreetings: parseJsonArray(c.alternateGreetings as string),
    exampleDialogue: c.exampleDialogue,
    creatorNotes: c.creatorNotes,
    worldBookIds,
    createdBy: c.createdBy,
  };
}

export async function listCharacters(params: {
  search?: string;
  tags?: string[];
  page: number;
  pageSize: number;
  sort: string;
  order: string;
}) {
  const where: Record<string, unknown> = {};
  if (params.search) {
    where.OR = [
      { name: { contains: params.search } },
      { description: { contains: params.search } },
    ];
  }

  const [items, total] = await Promise.all([
    prisma.characterCard.findMany({
      where,
      orderBy: { [params.sort]: params.order },
      skip: (params.page - 1) * params.pageSize,
      take: params.pageSize,
      select: {
        id: true, name: true, avatar: true, coverImage: true,
        description: true, tags: true, source: true,
        createdAt: true, updatedAt: true,
      },
    }),
    prisma.characterCard.count({ where }),
  ]);

  let result = items.map((c) => formatCharacterSummary(c as Record<string, unknown>));

  if (params.tags?.length) {
    result = result.filter((c) =>
      params.tags!.some((t) => (c.tags as string[]).includes(t)),
    );
  }

  return {
    data: result,
    pagination: { page: params.page, pageSize: params.pageSize, total },
  };
}

export async function getCharacterById(id: string) {
  const c = await prisma.characterCard.findUnique({
    where: { id },
    include: { worldBooks: { select: { worldBookId: true } } },
  });
  if (!c) throw new NotFoundError("角色卡");
  const worldBookIds = c.worldBooks.map((w) => w.worldBookId);
  return formatCharacterDetail(c as unknown as Record<string, unknown>, worldBookIds);
}

export async function getAllTags() {
  const chars = await prisma.characterCard.findMany({ select: { tags: true } });
  const tagSet = new Set<string>();
  for (const c of chars) {
    for (const t of parseJsonArray(c.tags)) tagSet.add(t);
  }
  return Array.from(tagSet).sort();
}

export async function createCharacter(
  data: {
    name: string;
    description?: string;
    personality?: string;
    scenario?: string;
    systemPrompt?: string;
    firstMessage?: string;
    alternateGreetings?: string[];
    exampleDialogue?: string;
    creatorNotes?: string;
    tags?: string[];
    worldBookIds?: string[];
    avatar?: string | null;
    coverImage?: string | null;
    source?: string;
  },
  userId: string,
) {
  const { worldBookIds, ...rest } = data;
  const c = await prisma.characterCard.create({
    data: {
      ...rest,
      alternateGreetings: toJsonString(rest.alternateGreetings),
      tags: toJsonString(rest.tags),
      description: rest.description ?? "",
      personality: rest.personality ?? "",
      scenario: rest.scenario ?? "",
      systemPrompt: rest.systemPrompt ?? "",
      firstMessage: rest.firstMessage ?? "",
      exampleDialogue: rest.exampleDialogue ?? "",
      creatorNotes: rest.creatorNotes ?? "",
      source: rest.source ?? "manual",
      avatar: rest.avatar ?? null,
      coverImage: rest.coverImage ?? null,
      createdBy: userId,
      worldBooks: worldBookIds?.length
        ? { create: worldBookIds.map((wbId) => ({ worldBookId: wbId })) }
        : undefined,
    },
    include: { worldBooks: { select: { worldBookId: true } } },
  });
  return formatCharacterDetail(c as unknown as Record<string, unknown>, c.worldBooks.map((w) => w.worldBookId));
}

export async function updateCharacter(
  id: string,
  data: {
    name?: string;
    description?: string;
    personality?: string;
    scenario?: string;
    systemPrompt?: string;
    firstMessage?: string;
    alternateGreetings?: string[];
    exampleDialogue?: string;
    creatorNotes?: string;
    tags?: string[];
    worldBookIds?: string[];
    avatar?: string | null;
    coverImage?: string | null;
  },
) {
  const existing = await prisma.characterCard.findUnique({ where: { id } });
  if (!existing) throw new NotFoundError("角色卡");

  const { worldBookIds, alternateGreetings, tags, ...rest } = data;
  const updateData: Record<string, unknown> = { ...rest };
  if (alternateGreetings !== undefined) updateData.alternateGreetings = toJsonString(alternateGreetings);
  if (tags !== undefined) updateData.tags = toJsonString(tags);

  if (worldBookIds !== undefined) {
    await prisma.characterWorldBook.deleteMany({ where: { characterId: id } });
    if (worldBookIds.length) {
      await prisma.characterWorldBook.createMany({
        data: worldBookIds.map((wbId) => ({ characterId: id, worldBookId: wbId })),
      });
    }
  }

  const c = await prisma.characterCard.update({
    where: { id },
    data: updateData,
    include: { worldBooks: { select: { worldBookId: true } } },
  });
  return formatCharacterDetail(c as unknown as Record<string, unknown>, c.worldBooks.map((w) => w.worldBookId));
}

export async function deleteCharacter(id: string) {
  const existing = await prisma.characterCard.findUnique({ where: { id } });
  if (!existing) throw new NotFoundError("角色卡");
  await prisma.characterCard.delete({ where: { id } });
}

export async function exportCharacter(id: string) {
  return getCharacterById(id);
}
