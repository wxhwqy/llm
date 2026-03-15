import { prisma } from "../db/prisma";

// ─── Queries ──────────────────────────────────────────────

export async function findAll() {
  return prisma.llmProvider.findMany({
    orderBy: [{ priority: "desc" }, { createdAt: "asc" }],
  });
}

export async function findEnabled() {
  return prisma.llmProvider.findMany({
    where: { enabled: true },
    orderBy: [{ priority: "desc" }, { createdAt: "asc" }],
  });
}

export async function findById(id: string) {
  return prisma.llmProvider.findUnique({ where: { id } });
}

// ─── Mutations ────────────────────────────────────────────

export async function create(data: {
  name: string;
  baseUrl: string;
  apiKey?: string;
  models?: string;
  autoDiscover?: boolean;
  enabled?: boolean;
  priority?: number;
}) {
  return prisma.llmProvider.create({ data });
}

export async function update(
  id: string,
  data: {
    name?: string;
    baseUrl?: string;
    apiKey?: string;
    models?: string;
    autoDiscover?: boolean;
    enabled?: boolean;
    priority?: number;
  },
) {
  return prisma.llmProvider.update({ where: { id }, data });
}

export async function deleteById(id: string) {
  return prisma.llmProvider.delete({ where: { id } });
}
