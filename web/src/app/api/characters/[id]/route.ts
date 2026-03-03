import { NextRequest } from "next/server";
import { withAuth, jsonOk } from "@/server/lib/response";
import { getCharacterById } from "@/server/services/character.service";

export const GET = withAuth(async (_req: NextRequest, ctx) => {
  const { id } = await ctx.params;
  const character = await getCharacterById(id);
  return jsonOk(character);
});
