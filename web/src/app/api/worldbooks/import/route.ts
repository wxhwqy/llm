import { NextRequest } from "next/server";
import { withAuth, jsonCreated } from "@/server/lib/response";
import { UnprocessableError } from "@/server/lib/errors";
import { importWorldbook } from "@/server/services/worldbook.service";

export const POST = withAuth(async (req: NextRequest, _ctx, user) => {
  const form = await req.formData();
  const file = form.get("file") as File | null;
  if (!file || file.size === 0) {
    throw new UnprocessableError("MISSING_FILE", "请上传文件");
  }
  if (!file.name.toLowerCase().endsWith(".json")) {
    throw new UnprocessableError("INVALID_FILE_TYPE", "仅支持 .json 文件");
  }

  const text = await file.text();
  const scope = form.get("scope") as string | undefined;
  const result = await importWorldbook(text, user.id, user.role, scope ?? undefined);
  return jsonCreated(result);
});
