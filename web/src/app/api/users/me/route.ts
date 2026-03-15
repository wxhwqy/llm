import { NextRequest } from "next/server";
import { withAuth, jsonOk, validateBody } from "@/server/lib/response";
import { updateProfileSchema } from "@/server/validators/user";
import { updateProfile } from "@/server/services/user.service";

export const PUT = withAuth(async (req: NextRequest, _ctx, user) => {
  const data = await validateBody(req, updateProfileSchema);
  const updated = await updateProfile(user.id, data);
  return jsonOk(updated);
});
