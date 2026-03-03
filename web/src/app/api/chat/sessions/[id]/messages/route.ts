export const runtime = "nodejs";

import { NextRequest } from "next/server";
import { withAuth, jsonCursor } from "@/server/lib/response";
import { messagesQuerySchema, sendMessageSchema } from "@/server/validators/chat";
import { getMessages, sendMessageStream } from "@/server/services/chat.service";
import { ValidationError } from "@/server/lib/errors";

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
  const body = await req.json();
  const data = sendMessageSchema.safeParse(body);
  if (!data.success) {
    throw new ValidationError("消息内容不能为空", data.error.errors.map((e) => ({ field: e.path.join("."), message: e.message })));
  }

  const stream = await sendMessageStream(id, user.id, data.data.content, req.signal);

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "Connection": "keep-alive",
      "X-Accel-Buffering": "no",
    },
  });
});
