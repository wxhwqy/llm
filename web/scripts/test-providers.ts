/**
 * 测试数据库中所有已启用的 LLM Provider 连通性。
 *
 * 用法: npx tsx scripts/test-providers.ts
 *
 * 测试内容:
 * 1. 连接测试 — 调用 GET {baseUrl}/v1/models
 * 2. 模型发现 — 列出可用模型
 * 3. 推理测试 — 用第一个模型发一条简单消息验证推理能力
 */
import { PrismaClient } from "../src/generated/prisma/client";
import { PrismaBetterSqlite3 } from "@prisma/adapter-better-sqlite3";
import path from "path";

const dbPath = path.join(__dirname, "..", "dev.db");
const adapter = new PrismaBetterSqlite3({ url: `file:${dbPath}` });
const prisma = new PrismaClient({ adapter });

interface ModelItem {
  id: string;
  name?: string;
  max_context_length?: number;
}

// ─── Helpers ──────────────────────────────────────────────

function maskKey(key: string): string {
  if (!key) return "(无)";
  if (key.length <= 8) return "***";
  return key.slice(0, 4) + "***" + key.slice(-4);
}

async function testConnection(baseUrl: string, apiKey: string): Promise<{ ok: boolean; models: ModelItem[]; error?: string }> {
  const headers: Record<string, string> = {};
  if (apiKey) headers["Authorization"] = `Bearer ${apiKey}`;

  try {
    const res = await fetch(`${baseUrl}/v1/models`, {
      headers,
      signal: AbortSignal.timeout(10_000),
    });
    if (!res.ok) {
      const body = await res.text().catch(() => "");
      return { ok: false, models: [], error: `HTTP ${res.status}: ${body.slice(0, 200)}` };
    }
    const json = await res.json();
    const models: ModelItem[] = json.data ?? [];
    return { ok: true, models };
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    return { ok: false, models: [], error: msg };
  }
}

async function testInference(
  baseUrl: string,
  apiKey: string,
  modelId: string,
): Promise<{ ok: boolean; content?: string; latencyMs?: number; error?: string }> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (apiKey) headers["Authorization"] = `Bearer ${apiKey}`;

  const start = Date.now();
  try {
    const res = await fetch(`${baseUrl}/v1/chat/completions`, {
      method: "POST",
      headers,
      body: JSON.stringify({
        model: modelId,
        messages: [{ role: "user", content: "请用一句话介绍你自己。" }],
        stream: false,
        max_tokens: 100,
        temperature: 0.3,
      }),
      signal: AbortSignal.timeout(30_000),
    });

    const latencyMs = Date.now() - start;

    if (!res.ok) {
      const body = await res.text().catch(() => "");
      return { ok: false, error: `HTTP ${res.status}: ${body.slice(0, 200)}`, latencyMs };
    }

    const json = await res.json();
    const content = json.choices?.[0]?.message?.content ?? "(空回复)";
    return { ok: true, content: content.trim(), latencyMs };
  } catch (e: unknown) {
    const latencyMs = Date.now() - start;
    const msg = e instanceof Error ? e.message : String(e);
    return { ok: false, error: msg, latencyMs };
  }
}

// ─── Main ─────────────────────────────────────────────────

async function main() {
  const providers = await prisma.llmProvider.findMany({
    where: { enabled: true },
    orderBy: { priority: "desc" },
  });

  if (providers.length === 0) {
    console.log("数据库中没有已启用的 Provider。请先运行 import-providers.ts 或在管理后台添加。");
    return;
  }

  console.log(`找到 ${providers.length} 个已启用的 Provider\n`);
  console.log("=".repeat(70));

  for (const p of providers) {
    console.log(`\n📡 ${p.name}`);
    console.log(`   地址: ${p.baseUrl}`);
    console.log(`   密钥: ${maskKey(p.apiKey)}`);
    console.log(`   优先级: ${p.priority} | 自动发现: ${p.autoDiscover}`);
    console.log("");

    // Step 1: Connection test
    process.stdout.write("   [1/3] 连接测试... ");
    const conn = await testConnection(p.baseUrl, p.apiKey);

    if (!conn.ok) {
      console.log(`❌ 失败`);
      console.log(`         错误: ${conn.error}`);
      console.log(`   [2/3] 跳过（连接失败）`);
      console.log(`   [3/3] 跳过（连接失败）`);
      console.log("\n" + "-".repeat(70));
      continue;
    }
    console.log(`✅ 成功`);

    // Step 2: Model discovery
    const manualModels: ModelItem[] = JSON.parse(p.models as string || "[]");
    const discoveredModels = conn.models;
    const allModels = [...discoveredModels];

    // Merge manual models not in discovered
    for (const m of manualModels) {
      if (!allModels.find((d) => d.id === m.id)) {
        allModels.push(m);
      }
    }

    process.stdout.write("   [2/3] 模型发现... ");
    if (allModels.length === 0) {
      console.log(`⚠️  未发现任何模型`);
      console.log(`   [3/3] 跳过（无可用模型）`);
      console.log("\n" + "-".repeat(70));
      continue;
    }
    console.log(`✅ 找到 ${allModels.length} 个模型`);
    for (const m of allModels) {
      const source = discoveredModels.find((d) => d.id === m.id) ? "自动" : "手动";
      const ctx = m.max_context_length ? ` (ctx: ${m.max_context_length})` : "";
      console.log(`         - ${m.id}${ctx} [${source}]`);
    }

    // Step 3: Inference test (use first model)
    const testModel = allModels[0];
    process.stdout.write(`   [3/3] 推理测试 (${testModel.id})... `);
    const inf = await testInference(p.baseUrl, p.apiKey, testModel.id);

    if (!inf.ok) {
      console.log(`❌ 失败 (${inf.latencyMs}ms)`);
      console.log(`         错误: ${inf.error}`);
    } else {
      console.log(`✅ 成功 (${inf.latencyMs}ms)`);
      console.log(`         回复: ${inf.content!.slice(0, 100)}${inf.content!.length > 100 ? "..." : ""}`);
    }

    console.log("\n" + "-".repeat(70));
  }

  console.log("\n测试完成。");
}

main()
  .catch((e) => { console.error(e); process.exit(1); })
  .finally(() => prisma.$disconnect());
