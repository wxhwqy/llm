import { NextRequest } from "next/server";
import { withAuth, jsonOk } from "@/server/lib/response";
import { updateEntrySchema } from "@/server/validators/worldbook";
import { updateEntry, deleteEntry } from "@/server/services/worldbook.service";
import { ValidationError } from "@/server/lib/errors";

export const PUT = withAuth(async (req: NextRequest, ctx, user) => {
  const { id, entryId } = await ctx.params;
  const body = await req.json();
  const data = updateEntrySchema.safeParse(body);
  if (!data.success) {
    throw new ValidationError("参数校验失败", data.error.errors.map((e) => ({ field: e.path.join("."), message: e.message })));
  }
  const entry = await updateEntry(id, entryId, data.data as unknown as Record<string, unknown>, user.id, user.role);
  return jsonOk(entry);
});

export const DELETE = withAuth(async (_req: NextRequest, ctx, user) => {
  const { id, entryId } = await ctx.params;
  await deleteEntry(id, entryId, user.id, user.role);
  return jsonOk({ deleted: true });
});
