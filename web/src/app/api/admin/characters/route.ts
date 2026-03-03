import { NextRequest } from "next/server";
import { withAdmin, jsonCreated } from "@/server/lib/response";
import { createCharacterSchema } from "@/server/validators/character";
import { createCharacter } from "@/server/services/character.service";
import { ValidationError } from "@/server/lib/errors";
import { saveUploadedFile } from "@/server/services/upload.service";

export const POST = withAdmin(async (req: NextRequest, _ctx, user) => {
  const contentType = req.headers.get("content-type") ?? "";

  if (contentType.includes("multipart/form-data")) {
    const form = await req.formData();
    const raw: Record<string, unknown> = {};
    for (const [key, val] of form.entries()) {
      if (key === "coverImageFile") continue;
      if (["alternateGreetings", "tags", "worldBookIds"].includes(key)) {
        try { raw[key] = JSON.parse(val as string); } catch { raw[key] = []; }
      } else {
        raw[key] = val;
      }
    }
    const data = createCharacterSchema.parse(raw);
    const coverFile = form.get("coverImageFile") as File | null;
    let coverImage: string | null = null;
    if (coverFile && coverFile.size > 0) {
      coverImage = await saveUploadedFile(coverFile, "covers");
    }
    const result = await createCharacter({ ...data, coverImage }, user.id);
    return jsonCreated(result);
  }

  const body = await req.json();
  const data = createCharacterSchema.safeParse(body);
  if (!data.success) {
    throw new ValidationError("参数校验失败", data.error.errors.map((e) => ({ field: e.path.join("."), message: e.message })));
  }
  const result = await createCharacter(data.data, user.id);
  return jsonCreated(result);
});
