import { NextResponse } from "next/server";
import { apiHandler } from "@/server/lib/response";

export const POST = apiHandler(async () => {
  return NextResponse.json(
    {
      data: {
        user: {
          id: "usr_default",
          username: "Admin",
          email: "admin@example.com",
          role: "admin",
          createdAt: "2026-02-01T00:00:00.000Z",
        },
      },
    },
    { status: 201 },
  );
});
