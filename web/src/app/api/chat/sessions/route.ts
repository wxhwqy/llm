import { NextRequest } from "next/server";
import { withAuth, jsonCreated, jsonPaginated, validateBody } from "@/server/lib/response";
import { createSessionSchema } from "@/server/validators/chat";
import { listSessions, createSession } from "@/server/services/chat.service";

export const GET = withAuth(async (req: NextRequest, _ctx, user) => {
  const url = req.nextUrl;
  const page = parseInt(url.searchParams.get("page") ?? "1", 10);
  const pageSize = parseInt(url.searchParams.get("pageSize") ?? "50", 10);
  const result = await listSessions(user.id, page, pageSize);
  return jsonPaginated(result.data, result.pagination);
});

export const POST = withAuth(async (req: NextRequest, _ctx, user) => {
  const data = await validateBody(req, createSessionSchema);
  const result = await createSession(user.id, data.characterId, data.modelId);
  return jsonCreated(result);
});
