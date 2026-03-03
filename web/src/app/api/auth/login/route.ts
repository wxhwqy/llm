import { NextResponse } from "next/server";
import { apiHandler } from "@/server/lib/response";

// Phase 1 stub: always returns the default admin user
export const POST = apiHandler(async () => {
  return NextResponse.json({
    data: {
      user: {
        id: "usr_default",
        username: "Admin",
        email: "admin@example.com",
        role: "admin",
        createdAt: "2026-02-01T00:00:00.000Z",
      },
    },
  });
});
