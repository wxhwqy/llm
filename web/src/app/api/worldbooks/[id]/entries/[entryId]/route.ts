import { NextRequest } from "next/server";
import { withAuth, jsonOk, validateBody } from "@/server/lib/response";
import { updateEntrySchema } from "@/server/validators/worldbook";
import { updateEntry, deleteEntry } from "@/server/services/worldbook.service";

export const PUT = withAuth(async (req: NextRequest, ctx, user) => {
  const { id, entryId } = await ctx.params;
  const data = await validateBody(req, updateEntrySchema);
  const entry = await updateEntry(id, entryId, data as unknown as Record<string, unknown>, user.id, user.role);
  return jsonOk(entry);
});

export const DELETE = withAuth(async (_req: NextRequest, ctx, user) => {
  const { id, entryId } = await ctx.params;
  await deleteEntry(id, entryId, user.id, user.role);
  return jsonOk({ deleted: true });
});
