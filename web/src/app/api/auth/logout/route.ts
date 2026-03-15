import { NextResponse } from "next/server";
import { apiHandler } from "@/server/lib/response";

export const POST = apiHandler(async () => {
  const res = NextResponse.json({ data: { success: true } });
  res.cookies.set("auth-token", "", {
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    maxAge: 0,
    path: "/",
  });
  return res;
});
