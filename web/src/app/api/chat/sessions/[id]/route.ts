import { NextRequest } from "next/server";
import { withAuth, jsonOk, validateBody } from "@/server/lib/response";
import { updateSessionSchema } from "@/server/validators/chat";
import { updateSession, deleteSession } from "@/server/services/chat.service";

export const PUT = withAuth(async (req: NextRequest, ctx, user) => {
  const { id } = await ctx.params;
  const data = await validateBody(req, updateSessionSchema);
  const result = await updateSession(id, user.id, data);
  return jsonOk(result);
});

export const DELETE = withAuth(async (_req: NextRequest, ctx, user) => {
  const { id } = await ctx.params;
  await deleteSession(id, user.id);
  return jsonOk({ deleted: true });
});
