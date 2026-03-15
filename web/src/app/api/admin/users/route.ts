import { NextRequest } from "next/server";
import { withAdmin, jsonPaginated } from "@/server/lib/response";
import { adminUserListSchema } from "@/server/validators/user";
import { listUsers } from "@/server/services/user.service";

export const GET = withAdmin(async (req: NextRequest) => {
  const url = req.nextUrl;
  const params = adminUserListSchema.parse({
    search: url.searchParams.get("search") ?? undefined,
    role: url.searchParams.get("role") ?? undefined,
    status: url.searchParams.get("status") ?? undefined,
    page: url.searchParams.get("page") ?? undefined,
    pageSize: url.searchParams.get("pageSize") ?? undefined,
  });

  const result = await listUsers(params);
  return jsonPaginated(result.data, result.pagination);
});
