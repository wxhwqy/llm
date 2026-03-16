import { config } from "../lib/config";
import { ServiceUnavailableError } from "../lib/errors";

interface LLMModel {
  id: string;
  name: string;
  max_context_length: number;
  status: string; // "loaded" | "available"
}

interface LLMModelsResponse {
  data: LLMModel[];
}

interface HealthResponse {
  status: string;
  queue: { active: number; waiting: number; max_concurrent: number };
}

export interface SamplingParams {
  temperature?: number;
  topP?: number;
  topK?: number;
}

let modelCache: { data: LLMModel[]; ts: number } | null = null;
const MODEL_CACHE_TTL = 60_000;

/** Normalize baseUrl: strip trailing slash and /v1 suffix so we can append /v1/... uniformly */
function normalizeBaseUrl(url: string): string {
  return url.replace(/\/+$/, "").replace(/\/v1$/, "");
}

export async function listModels(): Promise<LLMModel[]> {
  if (modelCache && Date.now() - modelCache.ts < MODEL_CACHE_TTL) {
    return modelCache.data;
  }
  try {
    const res = await fetch(`${config.llmServiceUrl}/v1/models`, {
      signal: AbortSignal.timeout(5000),
    });
    if (!res.ok) throw new Error(`LLM Service returned ${res.status}`);
    const body: LLMModelsResponse = await res.json();
    modelCache = { data: body.data, ts: Date.now() };
    return body.data;
  } catch {
    if (modelCache) return modelCache.data;
    return [];
  }
}

export async function getHealthStatus(): Promise<HealthResponse | null> {
  try {
    const res = await fetch(`${config.llmServiceUrl}/health`, {
      signal: AbortSignal.timeout(5000),
    });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

export async function chatCompletion(
  messages: { role: string; content: string }[],
  modelId: string,
  provider?: { baseUrl: string; apiKey: string },
  timeout?: number,
): Promise<{ content: string; usage?: { prompt_tokens: number; completion_tokens: number; total_tokens: number } }> {
  const baseUrl = normalizeBaseUrl(provider?.baseUrl ?? config.llmServiceUrl);
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (provider?.apiKey) {
    headers["Authorization"] = `Bearer ${provider.apiKey}`;
  }

  const res = await fetch(`${baseUrl}/v1/chat/completions`, {
    method: "POST",
    headers,
    body: JSON.stringify({
      model: modelId,
      messages,
      stream: false,
      temperature: config.defaultTemperature,
      max_tokens: config.defaultMaxTokens,
    }),
    signal: AbortSignal.timeout(timeout ?? 30_000),
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({ error: { message: "推理服务异常" } }));
    throw new ServiceUnavailableError(body?.error?.message ?? "推理服务异常");
  }

  const body = await res.json();
  const choice = body.choices?.[0];
  return {
    content: choice?.message?.content ?? "",
    usage: body.usage,
  };
}

export async function streamChatCompletion(
  messages: { role: string; content: string }[],
  modelId: string,
  signal?: AbortSignal,
  sessionId?: string,
  provider?: { baseUrl: string; apiKey: string },
  sampling?: SamplingParams,
): Promise<Response> {
  const baseUrl = normalizeBaseUrl(provider?.baseUrl ?? config.llmServiceUrl);
  try {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (provider?.apiKey) {
      headers["Authorization"] = `Bearer ${provider.apiKey}`;
    }

    const res = await fetch(`${baseUrl}/v1/chat/completions`, {
      method: "POST",
      headers,
      body: JSON.stringify({
        model: modelId,
        messages,
        stream: true,
        no_think: true,
        temperature: sampling?.temperature ?? config.defaultTemperature,
        top_p: sampling?.topP ?? 0.95,
        max_tokens: config.defaultMaxTokens,
        ...(sampling?.topK !== undefined && { top_k: sampling.topK }),
        ...(sessionId && { session_id: sessionId }),
      }),
      signal,
    });

    if (!res.ok) {
      const body = await res.json().catch(() => ({ error: { message: "推理服务异常" } }));
      if (res.status === 503) throw new ServiceUnavailableError("推理服务繁忙，请稍后重试");
      throw new ServiceUnavailableError(body?.error?.message ?? "推理服务异常");
    }

    return res;
  } catch (err) {
    if (err instanceof ServiceUnavailableError) throw err;
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    throw new ServiceUnavailableError("推理服务不可用");
  }
}
