import { NextRequest } from "next/server";
import { withAdmin, jsonOk, jsonCreated, validateBody } from "@/server/lib/response";
import { createProviderSchema } from "@/server/validators/provider";
import { listProviders, createProvider } from "@/server/services/provider.service";

export const GET = withAdmin(async () => {
  const providers = await listProviders();
  return jsonOk(providers);
});

export const POST = withAdmin(async (req: NextRequest) => {
  const input = await validateBody(req, createProviderSchema);
  const provider = await createProvider(input);
  return jsonCreated(provider);
});
