import { NextRequest, NextResponse } from "next/server";
import { withAdmin } from "@/server/lib/response";
import { exportCharacter } from "@/server/services/character.service";

export const GET = withAdmin(async (_req: NextRequest, ctx) => {
  const { id } = await ctx.params;
  const character = await exportCharacter(id);
  const json = JSON.stringify(character, null, 2);
  return new NextResponse(json, {
    headers: {
      "Content-Type": "application/json",
      "Content-Disposition": `attachment; filename="${encodeURIComponent(character.name as string)}.json"`,
    },
  });
});
