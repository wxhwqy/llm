import { NextRequest } from "next/server";
import { withAdmin, jsonOk } from "@/server/lib/response";
import { testProvider } from "@/server/services/provider.service";

export const POST = withAdmin(async (_req: NextRequest, ctx) => {
  const { id } = await ctx.params;
  const result = await testProvider(id);
  return jsonOk(result);
});
