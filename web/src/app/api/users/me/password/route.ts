import { NextRequest } from "next/server";
import { withAuth, jsonOk, validateBody } from "@/server/lib/response";
import { changePasswordSchema } from "@/server/validators/user";
import { changePassword } from "@/server/services/user.service";

export const PUT = withAuth(async (req: NextRequest, _ctx, user) => {
  const data = await validateBody(req, changePasswordSchema);
  await changePassword(user.id, data.oldPassword, data.newPassword);
  return jsonOk({ success: true });
});
