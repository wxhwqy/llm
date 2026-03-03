import { NextRequest } from "next/server";
import { withAuth, jsonOk, validateBody } from "@/server/lib/response";
import { updateWorldbookSchema } from "@/server/validators/worldbook";
import { getWorldbookById, updateWorldbook, deleteWorldbook } from "@/server/services/worldbook.service";

export const GET = withAuth(async (_req: NextRequest, ctx, user) => {
  const { id } = await ctx.params;
  const wb = await getWorldbookById(id, user.id);
  return jsonOk(wb);
});

export const PUT = withAuth(async (req: NextRequest, ctx, user) => {
  const { id } = await ctx.params;
  const data = await validateBody(req, updateWorldbookSchema);
  const result = await updateWorldbook(id, data, user.id, user.role);
  return jsonOk(result);
});

export const DELETE = withAuth(async (_req: NextRequest, ctx, user) => {
  const { id } = await ctx.params;
  await deleteWorldbook(id, user.id, user.role);
  return jsonOk({ deleted: true });
});
