import { NextRequest } from "next/server";
import { withAdmin, jsonOk, validateBody } from "@/server/lib/response";
import { updateCharacterSchema } from "@/server/validators/character";
import { updateCharacter, deleteCharacter } from "@/server/services/character.service";
import { saveUploadedFile } from "@/server/services/upload.service";

export const PUT = withAdmin(async (req: NextRequest, ctx) => {
  const { id } = await ctx.params;
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
    const data = updateCharacterSchema.parse(raw);
    const coverFile = form.get("coverImageFile") as File | null;
    let coverImage: string | undefined;
    if (coverFile && coverFile.size > 0) {
      coverImage = await saveUploadedFile(coverFile, "covers");
    }
    const result = await updateCharacter(id, { ...data, coverImage });
    return jsonOk(result);
  }

  const data = await validateBody(req, updateCharacterSchema);
  const result = await updateCharacter(id, data);
  return jsonOk(result);
});

export const DELETE = withAdmin(async (_req: NextRequest, ctx) => {
  const { id } = await ctx.params;
  await deleteCharacter(id);
  return jsonOk({ deleted: true });
});
