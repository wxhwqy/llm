import { NextRequest } from "next/server";
import { withAuth, jsonCreated, validateBody } from "@/server/lib/response";
import { createEntrySchema } from "@/server/validators/worldbook";
import { addEntry } from "@/server/services/worldbook.service";

export const POST = withAuth(async (req: NextRequest, ctx, user) => {
  const { id } = await ctx.params;
  const data = await validateBody(req, createEntrySchema);
  const entry = await addEntry(id, data as unknown as Record<string, unknown>, user.id, user.role);
  return jsonCreated(entry);
});
