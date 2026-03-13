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

let modelCache: { data: LLMModel[]; ts: number } | null = null;
const MODEL_CACHE_TTL = 60_000;

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

export async function streamChatCompletion(
  messages: { role: string; content: string }[],
  modelId: string,
  signal?: AbortSignal,
  sessionId?: string,
): Promise<Response> {
  try {
    const res = await fetch(`${config.llmServiceUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: modelId,
        messages,
        stream: true,
        no_think: true,
        temperature: config.defaultTemperature,
        top_p: 0.95,
        max_tokens: config.defaultMaxTokens,
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
