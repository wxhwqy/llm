import { NextRequest } from "next/server";
import { withAuth, jsonOk } from "@/server/lib/response";
import { listModels, getHealthStatus } from "@/server/services/llm-client.service";

export const GET = withAuth(async (_req: NextRequest) => {
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
    };
  });

  return jsonOk(mapped);
});
