import { config } from "../lib/config";
import { estimateTokens, estimateMessagesTokens } from "./token-counter.service";
import { collectWorldbookEntries, buildScanText } from "./worldbook-injector.service";
import { buildPrompt } from "./prompt-builder.service";
import { applyContextWindow } from "./context-manager.service";
import { createLLMStream, type StreamResult } from "./sse-stream.service";
import { formatMessage } from "./session.service";
import { resolveProviderForModel } from "./provider.service";
import { waitForCompression, shouldCompress, startCompression } from "./compression.service";
import * as sessionRepo from "../repositories/session.repository";
import * as messageRepo from "../repositories/message.repository";

// ─── Shared helpers ───────────────────────────────────────

interface SessionWithCharacter {
  id: string;
  characterId: string;
  modelId: string;
  usedTokens: number;
  maxTokens: number;
  temperature: number | null;
  topP: number | null;
  topK: number | null;
  contextSummary: string | null;
  character: {
    preset: string;
    systemPrompt: string;
    description: string;
    scenario: string;
    exampleDialogue: string;
  };
}

/** Extract non-null sampling params from session */
function getSessionSampling(session: SessionWithCharacter) {
  const s: Record<string, number> = {};
  if (session.temperature !== null) s.temperature = session.temperature;
  if (session.topP !== null) s.topP = session.topP;
  if (session.topK !== null) s.topK = session.topK;
  return Object.keys(s).length ? s : undefined;
}

async function buildFinalMessages(
  session: SessionWithCharacter,
  history: { role: string; content: string }[],
  userInput: string,
) {
  const scanText = buildScanText(userInput, history);
  const wbBudget = Math.floor(session.maxTokens * config.worldbookTokenBudgetRatio);
  const entries = await collectWorldbookEntries(session.characterId, session.id, scanText, wbBudget);

  const character = session.character;
  const llmMessages = buildPrompt({
    character: {
      preset: character.preset,
      systemPrompt: character.systemPrompt,
      description: character.description,
      scenario: character.scenario,
      exampleDialogue: character.exampleDialogue,
    },
    worldBookEntries: entries,
    historyMessages: history.filter((m) => m.role !== "system"),
    userInput,
  });

  // Inject context summary after the character system message
  if (session.contextSummary) {
    const systemMessages = llmMessages.filter((m) => m.role === "system");
    const nonSystemMessages = llmMessages.filter((m) => m.role !== "system");
    const summaryMessage = { role: "system", content: `[对话回顾]\n${session.contextSummary}` };

    const systemTokens = estimateMessagesTokens([...systemMessages, summaryMessage]);
    const trimmed = applyContextWindow(nonSystemMessages, session.maxTokens, systemTokens);
    return [...systemMessages, summaryMessage, ...trimmed];
  }

  const systemTokens = estimateMessagesTokens(llmMessages.filter((m) => m.role === "system"));
  const historyPortion = llmMessages.filter((m) => m.role !== "system");
  const trimmed = applyContextWindow(historyPortion, session.maxTokens, systemTokens);

  return [...llmMessages.filter((m) => m.role === "system"), ...trimmed];
}

/**
 * After saving assistant message, check if we need to trigger compression
 * for the next round. Fires async — does not block the current response.
 */
function checkAndTriggerCompression(
  session: SessionWithCharacter,
  allHistoryTokens: number,
  systemTokens: number,
  provider?: { baseUrl: string; apiKey: string },
) {
  if (shouldCompress(session.maxTokens, systemTokens, allHistoryTokens)) {
    startCompression(session.id, session.modelId, provider);
  }
}

async function saveAssistantMessage(
  sessionId: string,
  userId: string,
  modelId: string,
  currentUsedTokens: number,
  maxTokens: number,
  result: StreamResult,
) {
  const assistantMsg = await messageRepo.createMessage({
    sessionId,
    role: "assistant",
    content: result.content,
    tokenCount: result.usage?.completion_tokens ?? estimateTokens(result.content),
  });

  if (result.usage) {
    await messageRepo.createTokenUsage({
      userId,
      sessionId,
      messageId: assistantMsg.id,
      modelId,
      promptTokens: result.usage.prompt_tokens,
      completionTokens: result.usage.completion_tokens,
      totalTokens: result.usage.total_tokens,
    });
  }

  const newUsedTokens = result.usage?.total_tokens ?? currentUsedTokens;
  await sessionRepo.updateSessionFields(sessionId, {
    usedTokens: newUsedTokens,
    updatedAt: new Date(),
  });

  return { assistantMsg, newUsedTokens, maxTokens };
}

// ─── Send message (new) ──────────────────────────────────

export async function sendMessageStream(
  sessionId: string,
  userId: string,
  content: string,
  abortSignal?: AbortSignal,
): Promise<ReadableStream<Uint8Array>> {
  // Wait for any in-progress compression to finish
  await waitForCompression(sessionId);

  const session = await sessionRepo.findOwnedSessionWithCharacter(sessionId, userId);

  // Save user message
  const userMsg = await messageRepo.createMessage({
    sessionId,
    role: "user",
    content,
    tokenCount: estimateTokens(content),
  });

  // Load history: only uncompressed messages, skip just-inserted user message
  const history = await messageRepo.findMessagesBySession(sessionId, {
    excludeId: userMsg.id,
    select: { role: true, content: true, isCompressed: true },
  });
  const uncompressedHistory = (history as { role: string; content: string; isCompressed: boolean }[])
    .filter((m) => !m.isCompressed);

  const sessionWithChar = session as unknown as SessionWithCharacter;

  const finalMessages = await buildFinalMessages(
    sessionWithChar,
    uncompressedHistory as { role: string; content: string }[],
    content,
  );

  // Resolve provider for this model
  const provider = await resolveProviderForModel(session.modelId);

  // Calculate system tokens for compression check (used in onComplete)
  const systemTokens = estimateMessagesTokens(finalMessages.filter((m) => m.role === "system"));

  return createLLMStream(finalMessages, session.modelId, {
    onBeforeStream: (send) => {
      send(JSON.stringify({ type: "user_message", message: formatMessage(userMsg) }));
    },
    onComplete: async (send, result) => {
      const { assistantMsg, newUsedTokens, maxTokens } = await saveAssistantMessage(
        sessionId, userId, session.modelId, session.usedTokens, session.maxTokens, result,
      );
      send(JSON.stringify({
        type: "message_complete",
        message: formatMessage(assistantMsg),
        contextUsage: { usedTokens: newUsedTokens, maxTokens },
      }));

      // Check if next round needs compression (async, non-blocking)
      const allUncompressed = await messageRepo.findUncompressedMessages(sessionId);
      const historyTokens = allUncompressed.reduce((sum, m) => sum + m.tokenCount, 0);
      checkAndTriggerCompression(sessionWithChar, historyTokens, systemTokens, provider ?? undefined);
    },
  }, { abortSignal, sessionId, provider: provider ?? undefined, sampling: getSessionSampling(sessionWithChar) });
}

// ─── Regenerate ──────────────────────────────────────────

export async function regenerateStream(
  sessionId: string,
  userId: string,
  modelId?: string,
  abortSignal?: AbortSignal,
): Promise<ReadableStream<Uint8Array>> {
  // Wait for any in-progress compression to finish
  await waitForCompression(sessionId);

  await sessionRepo.findOwnedSession(sessionId, userId);

  // Delete last assistant message
  const lastAssistant = await messageRepo.findLastMessageByRole(sessionId, "assistant");
  if (lastAssistant) {
    await messageRepo.deleteMessage(lastAssistant.id);
  }

  // Find last user message for re-generation
  const lastUser = await messageRepo.findLastMessageByRole(sessionId, "user");
  if (!lastUser) {
    const { NotFoundError } = await import("../lib/errors");
    throw new NotFoundError("消息");
  }

  // Override model if specified
  if (modelId) {
    await sessionRepo.updateSessionFields(sessionId, { modelId });
  }

  const sessionWithChar = await sessionRepo.findOwnedSessionWithCharacter(sessionId, userId);

  const history = await messageRepo.findMessagesBySession(sessionId, {
    select: { role: true, content: true, isCompressed: true },
  });

  const uncompressedHistory = (history as { role: string; content: string; isCompressed: boolean }[])
    .filter((m) => !m.isCompressed && m.role !== "system");
  const finalMessages = await buildFinalMessages(
    sessionWithChar as unknown as SessionWithCharacter,
    uncompressedHistory.slice(0, -1), // all except last user msg (it becomes userInput)
    lastUser.content,
  );

  const effectiveModelId = modelId || sessionWithChar.modelId;

  // Resolve provider for this model
  const provider = await resolveProviderForModel(effectiveModelId);

  // Calculate system tokens for compression check
  const systemTokens = estimateMessagesTokens(finalMessages.filter((m) => m.role === "system"));

  return createLLMStream(finalMessages, effectiveModelId, {
    onComplete: async (send, result) => {
      const { assistantMsg, newUsedTokens, maxTokens } = await saveAssistantMessage(
        sessionId, userId, effectiveModelId, sessionWithChar.usedTokens, sessionWithChar.maxTokens, result,
      );
      send(JSON.stringify({
        type: "message_complete",
        message: formatMessage(assistantMsg),
        contextUsage: { usedTokens: newUsedTokens, maxTokens },
      }));

      // Check if next round needs compression (async, non-blocking)
      const allUncompressed = await messageRepo.findUncompressedMessages(sessionId);
      const historyTokens = allUncompressed.reduce((sum, m) => sum + m.tokenCount, 0);
      checkAndTriggerCompression(
        sessionWithChar as unknown as SessionWithCharacter,
        historyTokens, systemTokens, provider ?? undefined,
      );
    },
  }, { abortSignal, sessionId, provider: provider ?? undefined, sampling: getSessionSampling(sessionWithChar as unknown as SessionWithCharacter) });
}
