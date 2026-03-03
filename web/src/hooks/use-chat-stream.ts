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

export function useChatStream(
  onContextUsageUpdate?: (usage: { usedTokens: number; maxTokens: number }) => void,
): UseChatStreamReturn {
  const abortRef = useRef<AbortController | null>(null);
  const {
    addMessage,
    replaceMessageId,
    setStreaming,
    appendStreamContent,
    finalizeStream,
    resetStream,
    removeLastAssistant,
  } = useChatStore.getState();

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

      if (USE_MOCK) {
        await simulateMockStream(tempId, userMsg);
        return;
      }

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        const res = await api.stream(
          `/chat/sessions/${sessionId}/messages`,
          { content },
          controller.signal,
        );

        if (!res.ok) {
          resetStream();
          return;
        }

        for await (const event of parseSSEStream(res)) {
          if ("type" in event) {
            if (event.type === "user_message") {
              const e = event as SSEUserMessageEvent;
              replaceMessageId(tempId, e.message);
            } else if (event.type === "message_complete") {
              const e = event as SSEMessageCompleteEvent;
              finalizeStream(e.message);
              onContextUsageUpdate?.(e.contextUsage);
            } else if (event.type === "error") {
              const e = event as SSEErrorEvent;
              console.error("Stream error:", e.error);
              resetStream();
            }
          } else if ("choices" in event) {
            const e = event as SSEChunkEvent;
            const delta = e.choices[0]?.delta?.content;
            if (delta) appendStreamContent(delta);
          }
        }
      } catch (err) {
        if ((err as Error).name !== "AbortError") {
          console.error("Stream failed:", err);
        }
        resetStream();
      } finally {
        abortRef.current = null;
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

      if (USE_MOCK) {
        await simulateMockStream();
        return;
      }

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        const res = await api.stream(
          `/chat/sessions/${sessionId}/regenerate`,
          {},
          controller.signal,
        );
        if (!res.ok) { resetStream(); return; }

        for await (const event of parseSSEStream(res)) {
          if ("type" in event && event.type === "message_complete") {
            finalizeStream((event as SSEMessageCompleteEvent).message);
            onContextUsageUpdate?.((event as SSEMessageCompleteEvent).contextUsage);
          } else if ("choices" in event) {
            const delta = (event as SSEChunkEvent).choices[0]?.delta?.content;
            if (delta) appendStreamContent(delta);
          }
        }
      } catch (err) {
        if ((err as Error).name !== "AbortError") console.error(err);
        resetStream();
      } finally {
        abortRef.current = null;
      }
    },
    [removeLastAssistant, setStreaming, finalizeStream, appendStreamContent, resetStream, onContextUsageUpdate],
  );

  async function simulateMockStream(tempId?: string, userMsg?: ChatMessage) {
    const text = mockApi.getStreamingDemoText();
    useChatStore.setState({ streamingContent: "" });

    if (tempId && userMsg) {
      replaceMessageId(tempId, { ...userMsg, id: `msg_${Date.now()}` });
    }

    let accumulated = "";
    for (let i = 0; i < text.length; i++) {
      if (abortRef.current?.signal.aborted) break;
      accumulated += text[i];
      useChatStore.setState({ streamingContent: accumulated });
      const d = /[\s,，。！？\n]/.test(text[i]) ? 8 : 18;
      await new Promise((r) => setTimeout(r, d));
    }

    const aiMsg: ChatMessage = {
      id: `msg_${Date.now()}`,
      role: "assistant",
      content: accumulated,
      tokenCount: Math.ceil(accumulated.length / 2),
      isCompressed: false,
      createdAt: new Date().toISOString(),
      editedAt: null,
    };
    finalizeStream(aiMsg);
  }

  return { sendMessage, stopGeneration, regenerate };
}
