import { NextRequest, NextResponse } from "next/server";
import { withAuth } from "@/server/lib/response";
import { exportWorldbook } from "@/server/services/worldbook.service";

export const GET = withAuth(async (_req: NextRequest, ctx, user) => {
  const { id } = await ctx.params;
  const wb = await exportWorldbook(id, user.id);
  const json = JSON.stringify(wb, null, 2);
  return new NextResponse(json, {
    headers: {
      "Content-Type": "application/json",
      "Content-Disposition": `attachment; filename="${encodeURIComponent(wb.name)}.json"`,
    },
  });
});
