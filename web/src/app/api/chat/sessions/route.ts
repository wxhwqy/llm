import { NextRequest } from "next/server";
import { withAuth, jsonOk, jsonCreated, jsonPaginated } from "@/server/lib/response";
import { createSessionSchema } from "@/server/validators/chat";
import { listSessions, createSession } from "@/server/services/chat.service";
import { ValidationError } from "@/server/lib/errors";

export const GET = withAuth(async (req: NextRequest, _ctx, user) => {
  const url = req.nextUrl;
  const page = parseInt(url.searchParams.get("page") ?? "1", 10);
  const pageSize = parseInt(url.searchParams.get("pageSize") ?? "50", 10);
  const result = await listSessions(user.id, page, pageSize);
  return jsonPaginated(result.data, result.pagination);
});

export const POST = withAuth(async (req: NextRequest, _ctx, user) => {
  const body = await req.json();
  const data = createSessionSchema.safeParse(body);
  if (!data.success) {
    throw new ValidationError("参数校验失败", data.error.errors.map((e) => ({ field: e.path.join("."), message: e.message })));
  }
  const result = await createSession(user.id, data.data.characterId, data.data.modelId);
  return jsonCreated(result);
});
