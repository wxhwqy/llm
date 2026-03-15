import { NextRequest, NextResponse } from "next/server";
import { apiHandler, validateBody } from "@/server/lib/response";
import { registerSchema } from "@/server/validators/user";
import { register } from "@/server/services/user.service";

export const POST = apiHandler(async (req: NextRequest) => {
  const data = await validateBody(req, registerSchema);
  const result = await register(data);

  const res = NextResponse.json({ data: { user: result.user } }, { status: 201 });
  res.cookies.set("auth-token", result.token, {
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    maxAge: 7 * 24 * 60 * 60,
    path: "/",
  });
  return res;
});
