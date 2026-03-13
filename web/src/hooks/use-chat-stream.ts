"use client";

import { useRef, useCallback } from "react";
import { useChatStore } from "@/stores/chat-store";
import { mockApi } from "@/lib/mock-api";
import { USE_MOCK } from "@/lib/constants";
import { api } from "@/lib/api-client";
import { parseSSEStream } from "@/lib/sse-client";
import type { ChatMessage } from "@/types/chat";
import type { SSEChunkEvent, SSEUserMessageEvent, SSEMessageCompleteEvent, SSEErrorEvent } from "@/types/api";

interface UseChatStreamReturn {
  sendMessage: (sessionId: string, content: string) => Promise<void>;
  stopGeneration: () => void;
  regenerate: (sessionId: string) => Promise<void>;
}

/**
 * @param activeSessionId - the sessionId currently displayed on screen.
 *   Background streams whose sessionId differs will skip UI updates.
 */
export function useChatStream(
  activeSessionId: string,
  onContextUsageUpdate?: (usage: { usedTokens: number; maxTokens: number }) => void,
): UseChatStreamReturn {
  const abortRef = useRef<AbortController | null>(null);
  const activeRef = useRef(activeSessionId);
  activeRef.current = activeSessionId;

  const {
    addMessage,
    replaceMessageId,
    setStreaming,
    appendStreamContent,
    finalizeStream,
    resetStream,
    removeLastAssistant,
  } = useChatStore.getState();

  /** Whether this stream's session is still the one displayed */
  const isActive = (sid: string) => activeRef.current === sid;

  const sendMessage = useCallback(
    async (sessionId: string, content: string) => {
      const tempId = `temp_${Date.now()}`;
      const userMsg: ChatMessage = {
        id: tempId,
        role: "user",
        content,
        tokenCount: Math.ceil(content.length / 2),
        isCompressed: false,
        createdAt: new Date().toISOString(),
        editedAt: null,
      };

      addMessage(userMsg);
      setStreaming(true);

      const controller = new AbortController();
      abortRef.current = controller;

      if (USE_MOCK) {
        await simulateMockStream(sessionId, controller.signal, tempId, userMsg);
        return;
      }

      try {
        const res = await api.stream(
          `/chat/sessions/${sessionId}/messages`,
          { content },
          controller.signal,
        );

        if (!res.ok) {
          if (isActive(sessionId)) resetStream();
          return;
        }

        for await (const event of parseSSEStream(res)) {
          if ("type" in event) {
            if (event.type === "user_message") {
              const e = event as SSEUserMessageEvent;
              if (isActive(sessionId)) replaceMessageId(tempId, e.message);
            } else if (event.type === "message_complete") {
              const e = event as SSEMessageCompleteEvent;
              if (isActive(sessionId)) {
                finalizeStream(e.message);
                onContextUsageUpdate?.(e.contextUsage);
              } else {
                // Stream finished in background — no UI update needed,
                // server already persisted the message.
              }
            } else if (event.type === "error") {
              const e = event as SSEErrorEvent;
              console.error("Stream error:", e.error);
              if (isActive(sessionId)) resetStream();
            }
          } else if ("choices" in event) {
            const e = event as SSEChunkEvent;
            const delta = e.choices[0]?.delta?.content;
            if (delta && isActive(sessionId)) appendStreamContent(delta);
          }
        }
      } catch (err) {
        if ((err as Error).name !== "AbortError") {
          console.error("Stream failed:", err);
        }
        if (isActive(sessionId)) resetStream();
      } finally {
        if (abortRef.current === controller) abortRef.current = null;
      }
    },
    [addMessage, setStreaming, replaceMessageId, finalizeStream, appendStreamContent, resetStream, onContextUsageUpdate],
  );

  const stopGeneration = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
  }, []);

  const regenerate = useCallback(
    async (sessionId: string) => {
      removeLastAssistant();
      setStreaming(true);

      const controller = new AbortController();
      abortRef.current = controller;

      if (USE_MOCK) {
        await simulateMockStream(sessionId, controller.signal);
        return;
      }

      try {
        const res = await api.stream(
          `/chat/sessions/${sessionId}/regenerate`,
          {},
          controller.signal,
        );
        if (!res.ok) { if (isActive(sessionId)) resetStream(); return; }

        for await (const event of parseSSEStream(res)) {
          if ("type" in event && event.type === "message_complete") {
            if (isActive(sessionId)) {
              finalizeStream((event as SSEMessageCompleteEvent).message);
              onContextUsageUpdate?.((event as SSEMessageCompleteEvent).contextUsage);
            }
          } else if ("choices" in event) {
            const delta = (event as SSEChunkEvent).choices[0]?.delta?.content;
            if (delta && isActive(sessionId)) appendStreamContent(delta);
          }
        }
      } catch (err) {
        if ((err as Error).name !== "AbortError") console.error(err);
        if (isActive(sessionId)) resetStream();
      } finally {
        if (abortRef.current === controller) abortRef.current = null;
      }
    },
    [removeLastAssistant, setStreaming, finalizeStream, appendStreamContent, resetStream, onContextUsageUpdate],
  );

  async function simulateMockStream(sessionId: string, signal: AbortSignal, tempId?: string, userMsg?: ChatMessage) {
    const text = mockApi.getStreamingDemoText();
    if (isActive(sessionId)) useChatStore.setState({ streamingContent: "" });

    if (tempId && userMsg) {
      const realUserMsg = { ...userMsg, id: `msg_${Date.now()}` };
      if (isActive(sessionId)) replaceMessageId(tempId, realUserMsg);
      mockApi.pushMessage(sessionId, realUserMsg);
    }

    let accumulated = "";
    for (let i = 0; i < text.length; i++) {
      if (signal.aborted) break;
      accumulated += text[i];
      if (isActive(sessionId)) useChatStore.setState({ streamingContent: accumulated });
      const d = /[\s,，。！？\n]/.test(text[i]) ? 8 : 18;
      await new Promise((r) => setTimeout(r, d));
    }

    if (signal.aborted) return;

    const aiMsg: ChatMessage = {
      id: `msg_${Date.now()}`,
      role: "assistant",
      content: accumulated,
      tokenCount: Math.ceil(accumulated.length / 2),
      isCompressed: false,
      createdAt: new Date().toISOString(),
      editedAt: null,
    };
    if (isActive(sessionId)) {
      finalizeStream(aiMsg);
    }
    mockApi.pushMessage(sessionId, aiMsg);
  }

  return { sendMessage, stopGeneration, regenerate };
}
