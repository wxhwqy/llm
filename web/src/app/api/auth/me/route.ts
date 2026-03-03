import { NextRequest } from "next/server";
import { withAuth, jsonOk, type AuthUser } from "@/server/lib/response";

export const GET = withAuth(async (_req: NextRequest, _ctx, user: AuthUser) => {
  return jsonOk(user);
});
