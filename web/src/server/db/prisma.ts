import { PrismaClient } from "@/generated/prisma/client";
import { PrismaBetterSqlite3 } from "@prisma/adapter-better-sqlite3";
import path from "path";

function createPrismaClient() {
  const dbUrl = process.env.DATABASE_URL ?? "file:./dev.db";
  // Resolve relative path from project root
  let dbPath: string;
  if (dbUrl.startsWith("file:")) {
    const relative = dbUrl.replace("file:", "");
    dbPath = path.isAbsolute(relative) ? relative : path.join(process.cwd(), relative);
  } else {
    dbPath = path.join(process.cwd(), "dev.db");
  }
  const adapter = new PrismaBetterSqlite3({ url: `file:${dbPath}` });
  return new PrismaClient({ adapter });
}

const globalForPrisma = globalThis as unknown as { prisma: PrismaClient };

export const prisma = globalForPrisma.prisma ?? createPrismaClient();

if (process.env.NODE_ENV !== "production") globalForPrisma.prisma = prisma;
