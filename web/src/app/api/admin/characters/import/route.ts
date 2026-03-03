import { NextRequest } from "next/server";
import { withAdmin, jsonCreated } from "@/server/lib/response";
import { UnprocessableError } from "@/server/lib/errors";
import { importFromPng, importFromJson } from "@/server/services/character-import.service";

export const POST = withAdmin(async (req: NextRequest, _ctx, user) => {
  const form = await req.formData();
  const file = form.get("file") as File | null;
  if (!file || file.size === 0) {
    throw new UnprocessableError("MISSING_FILE", "请上传文件");
  }

  const name = file.name.toLowerCase();
  const buffer = Buffer.from(await file.arrayBuffer());

  if (name.endsWith(".png")) {
    const result = await importFromPng(buffer, user.id);
    return jsonCreated(result);
  }

  if (name.endsWith(".json")) {
    const text = buffer.toString("utf-8");
    const result = await importFromJson(text, user.id);
    return jsonCreated(result);
  }

  throw new UnprocessableError("INVALID_FILE_TYPE", "仅支持 .png 和 .json 文件");
});
