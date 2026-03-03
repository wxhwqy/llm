import { NextRequest } from "next/server";
import { withAuth, jsonOk } from "@/server/lib/response";
import { editMessageSchema } from "@/server/validators/chat";
import { editMessage } from "@/server/services/chat.service";
import { ValidationError } from "@/server/lib/errors";

export const PUT = withAuth(async (req: NextRequest, ctx, user) => {
  const { id, msgId } = await ctx.params;
  const body = await req.json();
  const data = editMessageSchema.safeParse(body);
  if (!data.success) {
    throw new ValidationError("参数校验失败", data.error.errors.map((e) => ({ field: e.path.join("."), message: e.message })));
  }
  const result = await editMessage(id, msgId, user.id, data.data.content);
  return jsonOk(result);
});
