import { config } from "../lib/config";
import { estimateTokens, estimateMessagesTokens } from "./token-counter.service";
import { collectWorldbookEntries, buildScanText } from "./worldbook-injector.service";
import { buildPrompt } from "./prompt-builder.service";
import { applyContextWindow } from "./context-manager.service";
import { createLLMStream, type StreamResult } from "./sse-stream.service";
import { formatMessage } from "./session.service";
import * as sessionRepo from "../repositories/session.repository";
import * as messageRepo from "../repositories/message.repository";

// ─── Shared helpers ───────────────────────────────────────

interface SessionWithCharacter {
  id: string;
  characterId: string;
  modelId: string;
  usedTokens: number;
  maxTokens: number;
  character: {
    preset: string;
    systemPrompt: string;
    description: string;
    scenario: string;
    exampleDialogue: string;
  };
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

  const systemTokens = estimateMessagesTokens(llmMessages.filter((m) => m.role === "system"));
  const historyPortion = llmMessages.filter((m) => m.role !== "system");
  const trimmed = applyContextWindow(historyPortion, session.maxTokens, systemTokens);

  return [...llmMessages.filter((m) => m.role === "system"), ...trimmed];
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
  const session = await sessionRepo.findOwnedSessionWithCharacter(sessionId, userId);

  // Save user message
  const userMsg = await messageRepo.createMessage({
    sessionId,
    role: "user",
    content,
    tokenCount: estimateTokens(content),
  });

  // Load history (skip just-inserted user message — it goes in prompt as userInput)
  const history = await messageRepo.findMessagesBySession(sessionId, {
    excludeId: userMsg.id,
    select: { role: true, content: true },
  });

  const finalMessages = await buildFinalMessages(
    session as unknown as SessionWithCharacter,
    history as { role: string; content: string }[],
    content,
  );

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
    },
  }, { abortSignal, sessionId });
}

// ─── Regenerate ──────────────────────────────────────────

export async function regenerateStream(
  sessionId: string,
  userId: string,
  modelId?: string,
  abortSignal?: AbortSignal,
): Promise<ReadableStream<Uint8Array>> {
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
    select: { role: true, content: true },
  });

  const nonSystemHistory = (history as { role: string; content: string }[]).filter((m) => m.role !== "system");
  const finalMessages = await buildFinalMessages(
    sessionWithChar as unknown as SessionWithCharacter,
    nonSystemHistory.slice(0, -1), // all except last user msg (it becomes userInput)
    lastUser.content,
  );

  const effectiveModelId = modelId || sessionWithChar.modelId;

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
    },
  }, { abortSignal, sessionId });
}
