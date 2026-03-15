import { NextRequest } from "next/server";
import { withAuth, jsonOk } from "@/server/lib/response";
import { aggregateModels } from "@/server/services/provider.service";
import { listModels, getHealthStatus } from "@/server/services/llm-client.service";
import { prisma } from "@/server/db/prisma";

export const GET = withAuth(async (_req: NextRequest) => {
  // Check if any providers exist in DB
  const providerCount = await prisma.llmProvider.count();

  if (providerCount > 0) {
    // Use multi-provider aggregation
    const models = await aggregateModels();
    return jsonOk(models);
  }

  // Fallback: no providers configured, use legacy LLM_SERVICE_URL
  const [models, health] = await Promise.all([listModels(), getHealthStatus()]);

  const queueNearFull =
    health && health.queue.waiting >= health.queue.max_concurrent;

  const mapped = models.map((m) => {
    let status: "online" | "offline" | "busy" = "offline";
    if (m.status === "loaded") {
      status = queueNearFull ? "busy" : "online";
    }
    return {
      id: m.id,
      name: m.name,
      maxContextLength: m.max_context_length,
      status,
      providerId: null,
      providerName: null,
    };
  });

  return jsonOk(mapped);
});
