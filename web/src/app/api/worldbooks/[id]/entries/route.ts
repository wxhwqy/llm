import { NextRequest } from "next/server";
import { withAuth, jsonCreated } from "@/server/lib/response";
import { createEntrySchema } from "@/server/validators/worldbook";
import { addEntry } from "@/server/services/worldbook.service";
import { ValidationError } from "@/server/lib/errors";

export const POST = withAuth(async (req: NextRequest, ctx, user) => {
  const { id } = await ctx.params;
  const body = await req.json();
  const data = createEntrySchema.safeParse(body);
  if (!data.success) {
    throw new ValidationError("参数校验失败", data.error.errors.map((e) => ({ field: e.path.join("."), message: e.message })));
  }
  const entry = await addEntry(id, data.data as unknown as Record<string, unknown>, user.id, user.role);
  return jsonCreated(entry);
});
