import { NextRequest } from "next/server";
import { withAuth, jsonOk, validateBody } from "@/server/lib/response";
import { editMessageSchema } from "@/server/validators/chat";
import { editMessage } from "@/server/services/message.service";

export const PUT = withAuth(async (req: NextRequest, ctx, user) => {
  const { id, msgId } = await ctx.params;
  const data = await validateBody(req, editMessageSchema);
  const result = await editMessage(id, msgId, user.id, data.content);
  return jsonOk(result);
});
