export const runtime = "nodejs";

import { NextRequest } from "next/server";
import { withAuth, jsonCursor, validateBody } from "@/server/lib/response";
import { messagesQuerySchema, sendMessageSchema } from "@/server/validators/chat";
import { getMessages, sendMessageStream } from "@/server/services/chat.service";

export const GET = withAuth(async (req: NextRequest, ctx, user) => {
  const { id } = await ctx.params;
  const url = req.nextUrl;
  const params = messagesQuerySchema.parse({
    cursor: url.searchParams.get("cursor") ?? undefined,
    limit: url.searchParams.get("limit") ?? undefined,
  });
  const result = await getMessages(id, user.id, params.cursor, params.limit);
  return jsonCursor(result.data, result.hasMore, result.nextCursor);
});

export const POST = withAuth(async (req: NextRequest, ctx, user) => {
  const { id } = await ctx.params;
  const data = await validateBody(req, sendMessageSchema);

  const stream = await sendMessageStream(id, user.id, data.content, req.signal);

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "Connection": "keep-alive",
      "X-Accel-Buffering": "no",
    },
  });
});
