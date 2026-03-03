import { NextRequest } from "next/server";
import type { AuthUser } from "../lib/response";

const DEFAULT_USER: AuthUser = {
  id: "usr_default",
  username: "Admin",
  email: "admin@example.com",
  role: "admin",
  createdAt: "2026-02-01T00:00:00.000Z",
};

// Phase 1: hardcoded default user. Phase 4: JWT decode from cookie.
export async function getCurrentUser(_req: NextRequest): Promise<AuthUser> {
  return DEFAULT_USER;
}
