import * as providerRepo from "../repositories/provider.repository";
import { NotFoundError } from "../lib/errors";
import type { CreateProviderInput, UpdateProviderInput, ModelItem } from "../validators/provider";

// ─── Types ────────────────────────────────────────────────

export interface ProviderWithModels {
  id: string;
  name: string;
  baseUrl: string;
  apiKey: string;
  models: ModelItem[];
  autoDiscover: boolean;
  enabled: boolean;
  priority: number;
  createdAt: string;
  updatedAt: string;
}

export interface AggregatedModel {
  id: string;
  name: string;
  maxContextLength: number;
  status: string;
  providerId: string;
  providerName: string;
}

// ─── Helpers ──────────────────────────────────────────────

function maskApiKey(key: string): string {
  if (!key) return "";
  if (key.length <= 6) return "***";
  return key.slice(0, 3) + "***" + key.slice(-3);
}

function parseModels(modelsJson: string): ModelItem[] {
  try {
    return JSON.parse(modelsJson);
  } catch {
    return [];
  }
}

function formatProvider(p: {
  id: string;
  name: string;
  baseUrl: string;
  apiKey: string;
  models: string;
  autoDiscover: boolean;
  enabled: boolean;
  priority: number;
  createdAt: Date;
  updatedAt: Date;
}): ProviderWithModels {
  return {
    id: p.id,
    name: p.name,
    baseUrl: p.baseUrl,
    apiKey: maskApiKey(p.apiKey),
    models: parseModels(p.models),
    autoDiscover: p.autoDiscover,
    enabled: p.enabled,
    priority: p.priority,
    createdAt: p.createdAt.toISOString(),
    updatedAt: p.updatedAt.toISOString(),
  };
}

// ─── Discovery cache ──────────────────────────────────────

interface DiscoveredModel {
  id: string;
  name: string;
  max_context_length: number;
  status: string;
}

const discoveryCache = new Map<string, { data: DiscoveredModel[]; ts: number }>();
const DISCOVERY_CACHE_TTL = 60_000;

async function discoverModels(
  baseUrl: string,
  apiKey: string,
  providerId: string,
): Promise<DiscoveredModel[]> {
  const cached = discoveryCache.get(providerId);
  if (cached && Date.now() - cached.ts < DISCOVERY_CACHE_TTL) {
    return cached.data;
  }

  try {
    const headers: Record<string, string> = {};
    if (apiKey) headers["Authorization"] = `Bearer ${apiKey}`;

    const res = await fetch(`${baseUrl}/v1/models`, {
      headers,
      signal: AbortSignal.timeout(5000),
    });
    if (!res.ok) throw new Error(`returned ${res.status}`);
    const body: { data: DiscoveredModel[] } = await res.json();
    discoveryCache.set(providerId, { data: body.data, ts: Date.now() });
    return body.data;
  } catch {
    const cached = discoveryCache.get(providerId);
    return cached ? cached.data : [];
  }
}

async function getProviderHealth(
  baseUrl: string,
  apiKey: string,
): Promise<{ queue: { waiting: number; max_concurrent: number } } | null> {
  try {
    const headers: Record<string, string> = {};
    if (apiKey) headers["Authorization"] = `Bearer ${apiKey}`;

    const res = await fetch(`${baseUrl}/health`, {
      headers,
      signal: AbortSignal.timeout(5000),
    });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

// ─── Service ──────────────────────────────────────────────

export async function listProviders(): Promise<ProviderWithModels[]> {
  const providers = await providerRepo.findAll();
  return providers.map(formatProvider);
}

export async function createProvider(input: CreateProviderInput): Promise<ProviderWithModels> {
  const provider = await providerRepo.create({
    name: input.name,
    baseUrl: input.baseUrl,
    apiKey: input.apiKey ?? "",
    models: JSON.stringify(input.models ?? []),
    autoDiscover: input.autoDiscover ?? true,
    enabled: input.enabled ?? true,
    priority: input.priority ?? 0,
  });
  return formatProvider(provider);
}

export async function updateProvider(
  id: string,
  input: UpdateProviderInput,
): Promise<ProviderWithModels> {
  const existing = await providerRepo.findById(id);
  if (!existing) throw new NotFoundError("Provider 不存在");

  const data: Record<string, unknown> = {};
  if (input.name !== undefined) data.name = input.name;
  if (input.baseUrl !== undefined) data.baseUrl = input.baseUrl;
  if (input.apiKey !== undefined) data.apiKey = input.apiKey;
  if (input.models !== undefined) data.models = JSON.stringify(input.models);
  if (input.autoDiscover !== undefined) data.autoDiscover = input.autoDiscover;
  if (input.enabled !== undefined) data.enabled = input.enabled;
  if (input.priority !== undefined) data.priority = input.priority;

  const provider = await providerRepo.update(id, data);
  // Clear discovery cache on update
  discoveryCache.delete(id);
  return formatProvider(provider);
}

export async function deleteProvider(id: string): Promise<void> {
  const existing = await providerRepo.findById(id);
  if (!existing) throw new NotFoundError("Provider 不存在");
  await providerRepo.deleteById(id);
  discoveryCache.delete(id);
}

export async function testProvider(
  id: string,
): Promise<{ success: boolean; models?: ModelItem[]; latencyMs?: number; error?: string }> {
  const provider = await providerRepo.findById(id);
  if (!provider) throw new NotFoundError("Provider 不存在");

  const start = Date.now();
  try {
    const headers: Record<string, string> = {};
    if (provider.apiKey) headers["Authorization"] = `Bearer ${provider.apiKey}`;

    const res = await fetch(`${provider.baseUrl}/v1/models`, {
      headers,
      signal: AbortSignal.timeout(10000),
    });
    const latencyMs = Date.now() - start;

    if (!res.ok) {
      return { success: false, error: `HTTP ${res.status}` };
    }

    const body: { data: DiscoveredModel[] } = await res.json();
    const models = body.data.map((m) => ({
      id: m.id,
      name: m.name,
      maxContextLength: m.max_context_length,
    }));

    return { success: true, models, latencyMs };
  } catch (err) {
    return {
      success: false,
      error: err instanceof Error ? err.message : "连接失败",
    };
  }
}

/**
 * Aggregate models from all enabled providers.
 * Returns models with provider info and status.
 */
export async function aggregateModels(): Promise<AggregatedModel[]> {
  const providers = await providerRepo.findEnabled();
  const results: AggregatedModel[] = [];

  await Promise.all(
    providers.map(async (provider) => {
      // Collect manually configured models
      const manualModels = parseModels(provider.models);
      for (const m of manualModels) {
        results.push({
          id: m.id,
          name: m.name,
          maxContextLength: m.maxContextLength,
          status: "online", // manual models assumed online
          providerId: provider.id,
          providerName: provider.name,
        });
      }

      // Auto-discover models
      if (provider.autoDiscover) {
        const [discovered, health] = await Promise.all([
          discoverModels(provider.baseUrl, provider.apiKey, provider.id),
          getProviderHealth(provider.baseUrl, provider.apiKey),
        ]);

        const queueNearFull =
          health && health.queue.waiting >= health.queue.max_concurrent;

        // Filter out models already in manual list
        const manualIds = new Set(manualModels.map((m) => m.id));
        for (const m of discovered) {
          if (manualIds.has(m.id)) continue;

          let status: "online" | "offline" | "busy" = "offline";
          if (m.status === "loaded") {
            status = queueNearFull ? "busy" : "online";
          }

          results.push({
            id: m.id,
            name: m.name,
            maxContextLength: m.max_context_length,
            status,
            providerId: provider.id,
            providerName: provider.name,
          });
        }
      }
    }),
  );

  // Sort by provider priority (already ordered) then by model name
  return results;
}

/**
 * Find the provider config for a given modelId.
 * Used by chat-stream to route requests to the correct provider.
 */
export async function resolveProviderForModel(
  modelId: string,
): Promise<{ baseUrl: string; apiKey: string } | null> {
  const models = await aggregateModels();
  const match = models.find((m) => m.id === modelId);
  if (!match) return null;

  const provider = await providerRepo.findById(match.providerId);
  if (!provider || !provider.enabled) return null;

  return { baseUrl: provider.baseUrl, apiKey: provider.apiKey };
}
