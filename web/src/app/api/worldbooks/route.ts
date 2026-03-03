import { NextRequest } from "next/server";
import { withAuth, jsonCreated, jsonPaginated, validateBody } from "@/server/lib/response";
import { worldbookListSchema, createWorldbookSchema } from "@/server/validators/worldbook";
import { listWorldbooks, createWorldbook } from "@/server/services/worldbook.service";

export const GET = withAuth(async (req: NextRequest, _ctx, user) => {
  const url = req.nextUrl;
  const params = worldbookListSchema.parse({
    scope: url.searchParams.get("scope") ?? undefined,
    page: url.searchParams.get("page") ?? undefined,
    pageSize: url.searchParams.get("pageSize") ?? undefined,
  });

  const items = await listWorldbooks(user.id, params.scope);
  const start = (params.page - 1) * params.pageSize;
  const paged = items.slice(start, start + params.pageSize);

  return jsonPaginated(paged, { page: params.page, pageSize: params.pageSize, total: items.length });
});

export const POST = withAuth(async (req: NextRequest, _ctx, user) => {
  const data = await validateBody(req, createWorldbookSchema);
  const result = await createWorldbook(
    { ...data, entries: data.entries as unknown as Record<string, unknown>[] },
    user.id, user.role,
  );
  return jsonCreated(result);
});
