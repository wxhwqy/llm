import { NextRequest } from "next/server";
import { withAuth, jsonOk } from "@/server/lib/response";
import { updateSessionSchema } from "@/server/validators/chat";
import { updateSession, deleteSession } from "@/server/services/chat.service";
import { ValidationError } from "@/server/lib/errors";

export const PUT = withAuth(async (req: NextRequest, ctx, user) => {
  const { id } = await ctx.params;
  const body = await req.json();
  const data = updateSessionSchema.safeParse(body);
  if (!data.success) {
    throw new ValidationError("参数校验失败", data.error.errors.map((e) => ({ field: e.path.join("."), message: e.message })));
  }
  const result = await updateSession(id, user.id, data.data);
  return jsonOk(result);
});

export const DELETE = withAuth(async (_req: NextRequest, ctx, user) => {
  const { id } = await ctx.params;
  await deleteSession(id, user.id);
  return jsonOk({ deleted: true });
});
