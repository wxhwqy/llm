import { NextRequest } from "next/server";
import { withAuth, jsonOk } from "@/server/lib/response";
import { getAllTags } from "@/server/services/character.service";

export const GET = withAuth(async (_req: NextRequest) => {
  const tags = await getAllTags();
  return jsonOk(tags);
});
