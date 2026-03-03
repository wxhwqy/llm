import { NextRequest } from "next/server";
import { withAuth, jsonPaginated } from "@/server/lib/response";
import { characterSearchSchema } from "@/server/validators/character";
import { listCharacters } from "@/server/services/character.service";

export const GET = withAuth(async (req: NextRequest) => {
  const url = req.nextUrl;
  const rawTags = url.searchParams.getAll("tag");
  const params = characterSearchSchema.parse({
    search: url.searchParams.get("search") ?? undefined,
    page: url.searchParams.get("page") ?? undefined,
    pageSize: url.searchParams.get("pageSize") ?? undefined,
    sort: url.searchParams.get("sort") ?? undefined,
    order: url.searchParams.get("order") ?? undefined,
  });

  const result = await listCharacters({
    ...params,
    tags: rawTags.length ? rawTags : undefined,
  });

  return jsonPaginated(result.data, result.pagination);
});
