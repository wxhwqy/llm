import { NextRequest } from "next/server";
import { withAdmin, jsonOk, validateBody } from "@/server/lib/response";
import { updateProviderSchema } from "@/server/validators/provider";
import { updateProvider, deleteProvider } from "@/server/services/provider.service";

export const PUT = withAdmin(async (req: NextRequest, ctx) => {
  const { id } = await ctx.params;
  const input = await validateBody(req, updateProviderSchema);
  const provider = await updateProvider(id, input);
  return jsonOk(provider);
});

export const DELETE = withAdmin(async (_req: NextRequest, ctx) => {
  const { id } = await ctx.params;
  await deleteProvider(id);
  return jsonOk({ deleted: true });
});
