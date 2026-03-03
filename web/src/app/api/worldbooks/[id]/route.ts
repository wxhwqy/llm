import { NextRequest } from "next/server";
import { withAuth, jsonOk } from "@/server/lib/response";
import { updateWorldbookSchema } from "@/server/validators/worldbook";
import { getWorldbookById, updateWorldbook, deleteWorldbook } from "@/server/services/worldbook.service";
import { ValidationError } from "@/server/lib/errors";

export const GET = withAuth(async (_req: NextRequest, ctx, user) => {
  const { id } = await ctx.params;
  const wb = await getWorldbookById(id, user.id);
  return jsonOk(wb);
});

export const PUT = withAuth(async (req: NextRequest, ctx, user) => {
  const { id } = await ctx.params;
  const body = await req.json();
  const data = updateWorldbookSchema.safeParse(body);
  if (!data.success) {
    throw new ValidationError("参数校验失败", data.error.errors.map((e) => ({ field: e.path.join("."), message: e.message })));
  }
  const result = await updateWorldbook(id, data.data, user.id, user.role);
  return jsonOk(result);
});

export const DELETE = withAuth(async (_req: NextRequest, ctx, user) => {
  const { id } = await ctx.params;
  await deleteWorldbook(id, user.id, user.role);
  return jsonOk({ deleted: true });
});
