export const runtime = "nodejs";

import { NextRequest } from "next/server";
import { withAuth } from "@/server/lib/response";
import { regenerateSchema } from "@/server/validators/chat";
import { regenerateStream } from "@/server/services/chat.service";

export const POST = withAuth(async (req: NextRequest, ctx, user) => {
  const { id } = await ctx.params;
  const body = await req.json().catch(() => ({}));
  const data = regenerateSchema.parse(body);

  const stream = await regenerateStream(id, user.id, data.modelId, req.signal);

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "Connection": "keep-alive",
      "X-Accel-Buffering": "no",
    },
  });
});
