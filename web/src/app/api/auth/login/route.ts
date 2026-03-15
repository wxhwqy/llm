import { NextRequest, NextResponse } from "next/server";
import { apiHandler, validateBody } from "@/server/lib/response";
import { loginSchema } from "@/server/validators/user";
import { login } from "@/server/services/user.service";

export const POST = apiHandler(async (req: NextRequest) => {
  const data = await validateBody(req, loginSchema);
  const result = await login(data.email, data.password);

  const res = NextResponse.json({ data: { user: result.user } });
  res.cookies.set("auth-token", result.token, {
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    maxAge: 7 * 24 * 60 * 60, // 7 days
    path: "/",
  });
  return res;
});
