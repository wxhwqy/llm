import { NextRequest } from "next/server";
import { withAdmin, jsonCreated, validateBody } from "@/server/lib/response";
import { createCharacterSchema } from "@/server/validators/character";
import { createCharacter } from "@/server/services/character.service";
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

  const data = await validateBody(req, createCharacterSchema);
  const result = await createCharacter(data, user.id);
  return jsonCreated(result);
});
