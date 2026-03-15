import { prisma } from "../db/prisma";

// ─── Queries ──────────────────────────────────────────────

export async function findUserById(id: string) {
  return prisma.user.findUnique({ where: { id } });
}

export async function findUserByEmail(email: string) {
  return prisma.user.findUnique({ where: { email } });
}

export async function findUserByUsername(username: string) {
  return prisma.user.findUnique({ where: { username } });
}

export async function listUsersPaginated(params: {
  search?: string;
  role?: string;
  status?: string;
  page: number;
  pageSize: number;
}) {
  const where: Record<string, unknown> = {};

  if (params.search) {
    where.OR = [
      { username: { contains: params.search } },
      { email: { contains: params.search } },
    ];
  }
  if (params.role) where.role = params.role;
  if (params.status) where.status = params.status;

  const [items, total] = await Promise.all([
    prisma.user.findMany({
      where,
      orderBy: { createdAt: "desc" },
      skip: (params.page - 1) * params.pageSize,
      take: params.pageSize,
      select: {
        id: true,
        username: true,
        email: true,
        role: true,
        status: true,
        createdAt: true,
        updatedAt: true,
        _count: { select: { sessions: true } },
      },
    }),
    prisma.user.count({ where }),
  ]);

  return { items, total };
}

// ─── Mutations ────────────────────────────────────────────

export async function createUser(data: {
  username: string;
  email: string;
  passwordHash: string;
  role?: string;
}) {
  return prisma.user.create({ data });
}

export async function updateUser(id: string, data: Record<string, unknown>) {
  return prisma.user.update({ where: { id }, data });
}

export async function deleteUserById(id: string) {
  return prisma.user.delete({ where: { id } });
}

// ─── Aggregations ─────────────────────────────────────────

export async function getUserTotalTokens(userId: string): Promise<number> {
  const result = await prisma.tokenUsage.aggregate({
    where: { userId },
    _sum: { totalTokens: true },
  });
  return result._sum.totalTokens ?? 0;
}
