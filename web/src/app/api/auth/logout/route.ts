import { NextResponse } from "next/server";
import { apiHandler } from "@/server/lib/response";

export const POST = apiHandler(async () => {
  const res = NextResponse.json({ data: { success: true } });
  res.cookies.delete("auth-token");
  return res;
});
