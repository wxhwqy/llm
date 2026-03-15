import { NextRequest } from "next/server";
import { apiHandler, jsonOk } from "@/server/lib/response";
import { getAllTags } from "@/server/services/character.service";

export const GET = apiHandler(async (_req: NextRequest) => {
  const tags = await getAllTags();
  return jsonOk(tags);
});
