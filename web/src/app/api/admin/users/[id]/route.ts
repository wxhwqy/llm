import { NextRequest } from "next/server";
import { withAdmin, jsonOk, validateBody } from "@/server/lib/response";
import { adminUpdateUserSchema } from "@/server/validators/user";
import { adminUpdateUser, adminDeleteUser } from "@/server/services/user.service";

export const PUT = withAdmin(async (req: NextRequest, ctx, user) => {
  const { id } = await ctx.params;
  const data = await validateBody(req, adminUpdateUserSchema);
  const updated = await adminUpdateUser(user.id, id, data);
  return jsonOk(updated);
});

export const DELETE = withAdmin(async (_req: NextRequest, ctx, user) => {
  const { id } = await ctx.params;
  await adminDeleteUser(user.id, id);
  return jsonOk({ deleted: true });
});
