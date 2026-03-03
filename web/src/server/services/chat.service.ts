import { prisma } from "../db/prisma";
import { parseJsonArray } from "../lib/json-fields";
import { NotFoundError, ForbiddenError } from "../lib/errors";
import { config } from "../lib/config";
import { estimateTokens, estimateMessagesTokens } from "./token-counter.service";
import { collectWorldbookEntries, buildScanText } from "./worldbook-injector.service";
import { buildPrompt } from "./prompt-builder.service";
import { applyContextWindow } from "./context-manager.service";
import { streamChatCompletion } from "./llm-client.service";

// ─── Session helpers ──────────────────────────────────────

export async function listSessions(userId: string, page: number, pageSize: number) {
  const [items, total] = await Promise.all([
    prisma.chatSession.findMany({
      where: { userId },
      orderBy: { updatedAt: "desc" },
      skip: (page - 1) * pageSize,
      take: pageSize,
      include: {
        character: { select: { name: true, avatar: true } },
        messages: { orderBy: { createdAt: "desc" }, take: 1, select: { content: true } },
        personalBooks: { select: { worldBookId: true } },
      },
    }),
    prisma.chatSession.count({ where: { userId } }),
  ]);

  const data = items.map((s) => ({
    id: s.id,
    characterId: s.characterId,
    characterName: s.character.name,
    characterAvatar: s.character.avatar,
    modelId: s.modelId,
    title: s.title,
    lastMessage: s.messages[0]?.content.slice(0, 100) ?? "",
    personalWorldBookIds: s.personalBooks.map((b) => b.worldBookId),
    contextUsage: { usedTokens: s.usedTokens, maxTokens: s.maxTokens },
    createdAt: s.createdAt.toISOString(),
    updatedAt: s.updatedAt.toISOString(),
  }));

  return { data, pagination: { page, pageSize, total } };
}

export async function createSession(userId: string, characterId: string, modelId?: string) {
  const character = await prisma.characterCard.findUnique({ where: { id: characterId } });
  if (!character) throw new NotFoundError("角色卡");

  const session = await prisma.chatSession.create({
    data: {
      userId,
      characterId,
      modelId: modelId || config.defaultModelId,
      title: "新会话",
      maxTokens: 8192,
    },
  });

  const messages = [];
  if (character.firstMessage) {
    const msg = await prisma.chatMessage.create({
      data: {
        sessionId: session.id,
        role: "assistant",
        content: character.firstMessage,
        tokenCount: estimateTokens(character.firstMessage),
      },
    });
    messages.push(formatMessage(msg));
  }

  return {
    id: session.id,
    characterId,
    characterName: character.name,
    characterAvatar: character.avatar,
    modelId: session.modelId,
    title: session.title,
    lastMessage: character.firstMessage?.slice(0, 100) ?? "",
    personalWorldBookIds: [] as string[],
    contextUsage: { usedTokens: session.usedTokens, maxTokens: session.maxTokens },
    messages,
    createdAt: session.createdAt.toISOString(),
    updatedAt: session.updatedAt.toISOString(),
  };
}

export async function updateSession(
  sessionId: string, userId: string,
  data: { modelId?: string; personalWorldBookIds?: string[]; title?: string },
) {
  const session = await prisma.chatSession.findUnique({ where: { id: sessionId } });
  if (!session) throw new NotFoundError("会话");
  if (session.userId !== userId) throw new ForbiddenError();

  if (data.personalWorldBookIds !== undefined) {
    await prisma.sessionWorldBook.deleteMany({ where: { sessionId } });
    if (data.personalWorldBookIds.length) {
      await prisma.sessionWorldBook.createMany({
        data: data.personalWorldBookIds.map((wbId) => ({ sessionId, worldBookId: wbId })),
      });
    }
  }

  const updated = await prisma.chatSession.update({
    where: { id: sessionId },
    data: {
      ...(data.modelId !== undefined ? { modelId: data.modelId } : {}),
      ...(data.title !== undefined ? { title: data.title } : {}),
    },
    include: {
      character: { select: { name: true, avatar: true } },
      personalBooks: { select: { worldBookId: true } },
      messages: { orderBy: { createdAt: "desc" }, take: 1, select: { content: true } },
    },
  });

  return {
    id: updated.id,
    characterId: updated.characterId,
    characterName: updated.character.name,
    characterAvatar: updated.character.avatar,
    modelId: updated.modelId,
    title: updated.title,
    lastMessage: updated.messages[0]?.content.slice(0, 100) ?? "",
    personalWorldBookIds: updated.personalBooks.map((b) => b.worldBookId),
    contextUsage: { usedTokens: updated.usedTokens, maxTokens: updated.maxTokens },
    createdAt: updated.createdAt.toISOString(),
    updatedAt: updated.updatedAt.toISOString(),
  };
}

export async function deleteSession(sessionId: string, userId: string) {
  const session = await prisma.chatSession.findUnique({ where: { id: sessionId } });
  if (!session) throw new NotFoundError("会话");
  if (session.userId !== userId) throw new ForbiddenError();
  await prisma.chatSession.delete({ where: { id: sessionId } });
}

// ─── Messages ─────────────────────────────────────────────

function formatMessage(m: { id: string; role: string; content: string; tokenCount: number; isCompressed: boolean; createdAt: Date; editedAt: Date | null }) {
  return {
    id: m.id,
    role: m.role,
    content: m.content,
    tokenCount: m.tokenCount,
    isCompressed: m.isCompressed,
    createdAt: m.createdAt.toISOString(),
    editedAt: m.editedAt?.toISOString() ?? null,
  };
}

export async function getMessages(sessionId: string, userId: string, cursor?: string, limit = 50) {
  const session = await prisma.chatSession.findUnique({ where: { id: sessionId } });
  if (!session) throw new NotFoundError("会话");
  if (session.userId !== userId) throw new ForbiddenError();

  const where: Record<string, unknown> = { sessionId };
  if (cursor) {
    const cursorMsg = await prisma.chatMessage.findUnique({ where: { id: cursor } });
    if (cursorMsg) {
      where.createdAt = { lt: cursorMsg.createdAt };
    }
  }

  const messages = await prisma.chatMessage.findMany({
    where,
    orderBy: { createdAt: "desc" },
    take: limit + 1,
  });

  const hasMore = messages.length > limit;
  const result = (hasMore ? messages.slice(0, limit) : messages).reverse();
  const nextCursor = hasMore ? result[0]?.id ?? null : null;

  return {
    data: result.map(formatMessage),
    hasMore,
    nextCursor,
  };
}

export async function editMessage(sessionId: string, messageId: string, userId: string, content: string) {
  const session = await prisma.chatSession.findUnique({ where: { id: sessionId } });
  if (!session) throw new NotFoundError("会话");
  if (session.userId !== userId) throw new ForbiddenError();

  const msg = await prisma.chatMessage.findUnique({ where: { id: messageId } });
  if (!msg || msg.sessionId !== sessionId) throw new NotFoundError("消息");
  if (msg.role !== "user") throw new ForbiddenError("只能编辑自己的消息");

  const updated = await prisma.chatMessage.update({
    where: { id: messageId },
    data: { content, tokenCount: estimateTokens(content), editedAt: new Date() },
  });
  return formatMessage(updated);
}

// ─── Streaming send message ───────────────────────────────

export async function sendMessageStream(
  sessionId: string,
  userId: string,
  content: string,
  abortSignal?: AbortSignal,
): Promise<ReadableStream<Uint8Array>> {
  const session = await prisma.chatSession.findUnique({
    where: { id: sessionId },
    include: { character: true },
  });
  if (!session) throw new NotFoundError("会话");
  if (session.userId !== userId) throw new ForbiddenError();

  // [1] Save user message
  const userMsg = await prisma.chatMessage.create({
    data: {
      sessionId,
      role: "user",
      content,
      tokenCount: estimateTokens(content),
    },
  });

  // [2] Load history (skip the message we just inserted, it will be in prompt as userInput)
  const history = await prisma.chatMessage.findMany({
    where: { sessionId, id: { not: userMsg.id } },
    orderBy: { createdAt: "asc" },
    select: { role: true, content: true },
  });

  // [3] Worldbook injection
  const scanText = buildScanText(content, history);
  const wbBudget = Math.floor(session.maxTokens * config.worldbookTokenBudgetRatio);
  const entries = await collectWorldbookEntries(session.characterId, sessionId, scanText, wbBudget);

  // [4] Build prompt
  const character = session.character;
  const llmMessages = buildPrompt({
    character: {
      systemPrompt: character.systemPrompt,
      personality: character.personality,
      scenario: character.scenario,
      exampleDialogue: character.exampleDialogue,
      firstMessage: character.firstMessage,
    },
    worldBookEntries: entries,
    historyMessages: history.filter((m) => m.role !== "system"),
    userInput: content,
  });

  // [5] Context window check & compression
  const systemTokens = estimateMessagesTokens(llmMessages.filter((m) => m.role === "system"));
  const historyPortion = llmMessages.filter((m) => m.role !== "system");
  const trimmed = applyContextWindow(historyPortion, session.maxTokens, systemTokens);
  const finalMessages = [
    ...llmMessages.filter((m) => m.role === "system"),
    ...trimmed,
  ];

  // [6] Create SSE stream
  const encoder = new TextEncoder();

  return new ReadableStream({
    async start(controller) {
      const send = (data: string) => controller.enqueue(encoder.encode(`data: ${data}\n\n`));

      // Send user_message event
      send(JSON.stringify({ type: "user_message", message: formatMessage(userMsg) }));

      let fullContent = "";
      let usage: { prompt_tokens: number; completion_tokens: number; total_tokens: number } | null = null;

      try {
        const llmResponse = await streamChatCompletion(finalMessages, session.modelId, abortSignal);
        const reader = llmResponse.body!.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() ?? "";

          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const data = line.slice(6);
            if (data === "[DONE]") break;

            try {
              const chunk = JSON.parse(data);
              if (chunk.error) {
                send(JSON.stringify({ type: "error", error: chunk.error }));
                break;
              }
              const delta = chunk.choices?.[0]?.delta?.content;
              if (delta) fullContent += delta;
              if (chunk.usage) usage = chunk.usage;
              send(data);
            } catch {
              // skip malformed chunks
            }
          }
        }
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") {
          // Client disconnected - save partial content
        } else {
          send(JSON.stringify({ type: "error", error: { code: "LLM_ERROR", message: "推理服务异常" } }));
        }
      }

      // [7] Save assistant message
      if (fullContent) {
        const assistantMsg = await prisma.chatMessage.create({
          data: {
            sessionId,
            role: "assistant",
            content: fullContent,
            tokenCount: usage?.completion_tokens ?? estimateTokens(fullContent),
          },
        });

        // Record token usage
        if (usage) {
          await prisma.tokenUsage.create({
            data: {
              userId,
              sessionId,
              messageId: assistantMsg.id,
              modelId: session.modelId,
              promptTokens: usage.prompt_tokens,
              completionTokens: usage.completion_tokens,
              totalTokens: usage.total_tokens,
            },
          });
        }

        const newUsedTokens = usage?.total_tokens ?? session.usedTokens;
        await prisma.chatSession.update({
          where: { id: sessionId },
          data: { usedTokens: newUsedTokens, updatedAt: new Date() },
        });

        // Send message_complete event
        send(JSON.stringify({
          type: "message_complete",
          message: formatMessage(assistantMsg),
          contextUsage: { usedTokens: newUsedTokens, maxTokens: session.maxTokens },
        }));
      }

      send("[DONE]");
      controller.close();
    },
  });
}

// ─── Regenerate ───────────────────────────────────────────

export async function regenerateStream(
  sessionId: string,
  userId: string,
  modelId?: string,
  abortSignal?: AbortSignal,
): Promise<ReadableStream<Uint8Array>> {
  const session = await prisma.chatSession.findUnique({ where: { id: sessionId } });
  if (!session) throw new NotFoundError("会话");
  if (session.userId !== userId) throw new ForbiddenError();

  // Find and delete the last assistant message
  const lastAssistant = await prisma.chatMessage.findFirst({
    where: { sessionId, role: "assistant" },
    orderBy: { createdAt: "desc" },
  });
  if (lastAssistant) {
    await prisma.chatMessage.delete({ where: { id: lastAssistant.id } });
  }

  // Find the last user message to re-trigger generation
  const lastUser = await prisma.chatMessage.findFirst({
    where: { sessionId, role: "user" },
    orderBy: { createdAt: "desc" },
  });

  if (!lastUser) throw new NotFoundError("消息");

  // Temporarily override model if specified
  if (modelId && modelId !== session.modelId) {
    await prisma.chatSession.update({ where: { id: sessionId }, data: { modelId } });
  }

  // Delete the last user message so sendMessageStream can re-create it properly
  // Actually, for regenerate we re-use the existing user message, so we build the stream differently
  // Let's build the stream directly without re-saving the user message

  const history = await prisma.chatMessage.findMany({
    where: { sessionId },
    orderBy: { createdAt: "asc" },
    select: { role: true, content: true },
  });

  const sessionWithChar = await prisma.chatSession.findUnique({
    where: { id: sessionId },
    include: { character: true },
  });
  if (!sessionWithChar) throw new NotFoundError("会话");

  const character = sessionWithChar.character;
  const scanText = buildScanText(lastUser.content, history);
  const wbBudget = Math.floor(sessionWithChar.maxTokens * config.worldbookTokenBudgetRatio);
  const entries = await collectWorldbookEntries(sessionWithChar.characterId, sessionId, scanText, wbBudget);

  // Build prompt from all history (the last message should be user's)
  const nonSystemHistory = history.filter((m) => m.role !== "system");
  const llmMessages = buildPrompt({
    character: {
      systemPrompt: character.systemPrompt,
      personality: character.personality,
      scenario: character.scenario,
      exampleDialogue: character.exampleDialogue,
      firstMessage: character.firstMessage,
    },
    worldBookEntries: entries,
    historyMessages: nonSystemHistory.slice(0, -1), // all except last user msg
    userInput: lastUser.content,
  });

  const systemTokens = estimateMessagesTokens(llmMessages.filter((m) => m.role === "system"));
  const historyPortion = llmMessages.filter((m) => m.role !== "system");
  const trimmed = applyContextWindow(historyPortion, sessionWithChar.maxTokens, systemTokens);
  const finalMessages = [
    ...llmMessages.filter((m) => m.role === "system"),
    ...trimmed,
  ];

  const effectiveModelId = modelId || sessionWithChar.modelId;
  const encoder = new TextEncoder();

  return new ReadableStream({
    async start(controller) {
      const send = (data: string) => controller.enqueue(encoder.encode(`data: ${data}\n\n`));

      let fullContent = "";
      let usage: { prompt_tokens: number; completion_tokens: number; total_tokens: number } | null = null;

      try {
        const llmResponse = await streamChatCompletion(finalMessages, effectiveModelId, abortSignal);
        const reader = llmResponse.body!.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() ?? "";

          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const data = line.slice(6);
            if (data === "[DONE]") break;
            try {
              const chunk = JSON.parse(data);
              if (chunk.error) {
                send(JSON.stringify({ type: "error", error: chunk.error }));
                break;
              }
              const delta = chunk.choices?.[0]?.delta?.content;
              if (delta) fullContent += delta;
              if (chunk.usage) usage = chunk.usage;
              send(data);
            } catch { /* skip */ }
          }
        }
      } catch (err) {
        if (!(err instanceof DOMException && err.name === "AbortError")) {
          send(JSON.stringify({ type: "error", error: { code: "LLM_ERROR", message: "推理服务异常" } }));
        }
      }

      if (fullContent) {
        const assistantMsg = await prisma.chatMessage.create({
          data: {
            sessionId,
            role: "assistant",
            content: fullContent,
            tokenCount: usage?.completion_tokens ?? estimateTokens(fullContent),
          },
        });

        if (usage) {
          await prisma.tokenUsage.create({
            data: {
              userId, sessionId, messageId: assistantMsg.id,
              modelId: effectiveModelId,
              promptTokens: usage.prompt_tokens,
              completionTokens: usage.completion_tokens,
              totalTokens: usage.total_tokens,
            },
          });
        }

        const newUsedTokens = usage?.total_tokens ?? sessionWithChar.usedTokens;
        await prisma.chatSession.update({
          where: { id: sessionId },
          data: { usedTokens: newUsedTokens, updatedAt: new Date() },
        });

        send(JSON.stringify({
          type: "message_complete",
          message: formatMessage(assistantMsg),
          contextUsage: { usedTokens: newUsedTokens, maxTokens: sessionWithChar.maxTokens },
        }));
      }

      send("[DONE]");
      controller.close();
    },
  });
}
