import { NextRequest } from "next/server";
import type { AuthUser } from "../lib/response";
import { UnauthorizedError } from "../lib/errors";
import { verifyToken } from "../services/user.service";

export async function getCurrentUser(req: NextRequest): Promise<AuthUser> {
  const token = req.cookies.get("auth-token")?.value;
  if (!token) {
    throw new UnauthorizedError("请先登录");
  }

  const user = await verifyToken(token);
  return user;
}
