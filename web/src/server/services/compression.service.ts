import { config } from "../lib/config";
import { chatCompletion } from "./llm-client.service";
import { estimateTokens } from "./token-counter.service";
import * as messageRepo from "../repositories/message.repository";
import * as sessionRepo from "../repositories/session.repository";

// ─── Concurrency control ─────────────────────────────────

const compressingMap = new Map<string, Promise<void>>();

/**
 * Wait for any in-progress compression on this session to finish.
 * Returns immediately if no compression is running.
 */
export async function waitForCompression(sessionId: string): Promise<void> {
  const task = compressingMap.get(sessionId);
  if (task) await task;
}

/**
 * Check whether the next-round prompt (excluding user input) would
 * exceed the summary compression trigger threshold.
 */
export function shouldCompress(
  maxTokens: number,
  systemTokens: number,
  historyTokens: number,
): boolean {
  const availableBudget = maxTokens - config.defaultMaxTokens;
  if (availableBudget <= 0) return false;
  return (systemTokens + historyTokens) / availableBudget >= config.summaryCompressTrigger;
}

/**
 * Fire-and-forget: kick off compression for a session.
 * If compression is already running for this session, skip.
 */
export function startCompression(
  sessionId: string,
  modelId: string,
  provider?: { baseUrl: string; apiKey: string },
): void {
  if (compressingMap.has(sessionId)) return;

  const task = doCompress(sessionId, modelId, provider)
    .catch((err) => {
      console.error(`[compression] session=${sessionId} failed:`, err);
    })
    .finally(() => {
      compressingMap.delete(sessionId);
    });

  compressingMap.set(sessionId, task);
}

// ─── Internal ─────────────────────────────────────────────

async function doCompress(
  sessionId: string,
  modelId: string,
  provider?: { baseUrl: string; apiKey: string },
): Promise<void> {
  // 1. Load uncompressed conversation messages (user + assistant only)
  const messages = await messageRepo.findUncompressedMessages(sessionId);
  if (messages.length < 4) return; // need at least 2 pairs to compress

  // 2. Split: compress oldest 70%, keep newest 30%
  const splitIndex = alignToMessagePair(messages, Math.floor(messages.length * config.summaryCompressRatio));
  if (splitIndex < 2) return; // nothing meaningful to compress

  const toCompress = messages.slice(0, splitIndex);

  // 3. Load existing summary
  const session = await sessionRepo.findSessionById(sessionId);
  if (!session) return;
  const existingSummary = (session as Record<string, unknown>).contextSummary as string | null;

  // 4. Format conversation text
  const conversationText = toCompress
    .map((m) => `${m.role === "user" ? "用户" : "角色"}: ${m.content}`)
    .join("\n");

  // 5. Build summary prompt
  const userContent = existingSummary
    ? `[先前摘要]\n${existingSummary}\n\n[对话内容]\n${conversationText}`
    : `[对话内容]\n${conversationText}`;

  const summaryMessages = [
    {
      role: "system",
      content:
        "你是一个对话摘要助手。请对以下角色扮演对话内容进行简洁摘要，保留关键情节、角色状态变化、重要约定和情感发展。摘要应以第三人称叙述，方便后续对话继续时作为上下文参考。请直接输出摘要内容，不要添加标题或前缀。",
    },
    { role: "user", content: userContent },
  ];

  // 6. Call LLM (non-streaming, with timeout)
  const result = await chatCompletion(
    summaryMessages,
    modelId,
    provider,
    config.summaryCompressTimeout,
  );

  if (!result.content) return;

  // 7. Update session contextSummary
  const summaryTokens = estimateTokens(result.content);
  await sessionRepo.updateSessionFields(sessionId, {
    contextSummary: result.content,
    usedTokens: summaryTokens, // will be recalculated next turn
  });

  // 8. Mark compressed messages
  const compressedIds = toCompress.map((m) => m.id);
  await messageRepo.markMessagesCompressed(compressedIds);

  console.log(
    `[compression] session=${sessionId} compressed=${compressedIds.length} msgs, summary=${summaryTokens} tokens`,
  );
}

/**
 * Align split index to a message-pair boundary (user + assistant).
 * Ensures we don't split in the middle of a pair.
 */
function alignToMessagePair(
  messages: { role: string }[],
  rawIndex: number,
): number {
  let idx = rawIndex;
  // Walk backwards until we're at the start of a pair boundary
  // (i.e., the message at idx should be a "user" message, meaning
  // everything before idx is a complete pair)
  while (idx > 0 && idx < messages.length) {
    if (messages[idx].role === "user") break;
    idx--;
  }
  return idx;
}
