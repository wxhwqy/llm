/**
 * 从 config/providers.json 导入 LLM Provider 配置到数据库。
 *
 * 用法: npx tsx scripts/import-providers.ts
 *
 * - enabled 为 false 的条目会跳过
 * - 已存在同名 Provider 会更新（按 name 匹配）
 * - 新 Provider 会创建
 */
import { PrismaClient } from "../src/generated/prisma/client";
import { PrismaBetterSqlite3 } from "@prisma/adapter-better-sqlite3";
import path from "path";
import fs from "fs";

const dbPath = path.join(__dirname, "..", "dev.db");
const adapter = new PrismaBetterSqlite3({ url: `file:${dbPath}` });
const prisma = new PrismaClient({ adapter });

interface ProviderConfig {
  name: string;
  baseUrl: string;
  apiKey: string;
  autoDiscover: boolean;
  enabled: boolean;
  priority: number;
  models: { id: string; name: string; maxContextLength: number }[];
}

async function main() {
  const configPath = path.join(__dirname, "..", "config", "providers.json");
  if (!fs.existsSync(configPath)) {
    console.error("配置文件不存在:", configPath);
    process.exit(1);
  }

  const providers: ProviderConfig[] = JSON.parse(fs.readFileSync(configPath, "utf-8"));
  console.log(`读取到 ${providers.length} 个 Provider 配置\n`);

  let created = 0;
  let updated = 0;
  let skipped = 0;

  for (const p of providers) {
    if (!p.enabled) {
      console.log(`  [跳过] ${p.name} (enabled: false)`);
      skipped++;
      continue;
    }

    if (!p.apiKey && p.baseUrl !== "http://localhost:8000" && !p.autoDiscover) {
      console.log(`  [跳过] ${p.name} (未配置 API Key)`);
      skipped++;
      continue;
    }

    // Check if provider with same name exists
    const existing = await prisma.llmProvider.findFirst({ where: { name: p.name } });

    const data = {
      name: p.name,
      baseUrl: p.baseUrl,
      apiKey: p.apiKey.startsWith("sk-你的") ? "" : p.apiKey,
      autoDiscover: p.autoDiscover,
      enabled: p.enabled,
      priority: p.priority,
      models: JSON.stringify(p.models),
    };

    if (existing) {
      await prisma.llmProvider.update({ where: { id: existing.id }, data });
      console.log(`  [更新] ${p.name} → ${p.baseUrl}`);
      updated++;
    } else {
      await prisma.llmProvider.create({ data });
      console.log(`  [创建] ${p.name} → ${p.baseUrl}`);
      created++;
    }
  }

  console.log(`\n完成: 创建 ${created}, 更新 ${updated}, 跳过 ${skipped}`);
}

main()
  .catch((e) => { console.error(e); process.exit(1); })
  .finally(() => prisma.$disconnect());
