import { streamChatCompletion, type SamplingParams } from "./llm-client.service";

// ─── Types ────────────────────────────────────────────────

export interface LLMUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface StreamResult {
  content: string;
  usage: LLMUsage | null;
}

type SSESender = (data: string) => void;

/**
 * Callbacks invoked during the stream lifecycle.
 * All callbacks are optional — provide only what you need.
 */
export interface StreamCallbacks {
  /** Called before LLM streaming starts (e.g. to send a user_message event). */
  onBeforeStream?: (send: SSESender) => void;
  /** Called after the assistant message is saved. */
  onComplete?: (send: SSESender, result: StreamResult) => Promise<void> | void;
  /** Called when an error occurs. */
  onError?: (send: SSESender, error: unknown) => void;
}

// ─── Core engine ──────────────────────────────────────────

/**
 * Creates a ReadableStream that:
 * 1. Calls the LLM via SSE
 * 2. Forwards each chunk to the client
 * 3. Collects full content + usage
 * 4. Invokes callbacks at lifecycle points
 */
export function createLLMStream(
  messages: { role: string; content: string }[],
  modelId: string,
  callbacks: StreamCallbacks,
  opts?: { abortSignal?: AbortSignal; sessionId?: string; provider?: { baseUrl: string; apiKey: string }; sampling?: SamplingParams },
): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();

  return new ReadableStream({
    async start(controller) {
      const send: SSESender = (data) =>
        controller.enqueue(encoder.encode(`data: ${data}\n\n`));

      callbacks.onBeforeStream?.(send);

      let fullContent = "";
      let usage: LLMUsage | null = null;
      let contentStarted = false; // 跳过开头空白字符

      try {
        const llmResponse = await streamChatCompletion(
          messages,
          modelId,
          opts?.abortSignal,
          opts?.sessionId,
          opts?.provider,
          opts?.sampling,
        );
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

              // 跳过开头的空白字符 chunk，避免流式输出前面出现空行
              if (delta && !contentStarted) {
                const trimmed = delta.trimStart();
                if (!trimmed) continue; // 整个 chunk 都是空白，跳过
                contentStarted = true;
                // 重写 chunk，把 trimmed 后的 delta 发送给客户端
                chunk.choices[0].delta.content = trimmed;
                send(JSON.stringify(chunk));
                continue;
              }
              send(data);
            } catch {
              // skip malformed chunks
            }
          }
        }
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") {
          // Client disconnected — fall through to save partial content
        } else {
          const defaultHandler = () =>
            send(JSON.stringify({ type: "error", error: { code: "LLM_ERROR", message: "推理服务异常" } }));
          callbacks.onError ? callbacks.onError(send, err) : defaultHandler();
        }
      }

      // 去除大模型返回内容开头的换行符和空格
      fullContent = fullContent.trimStart();

      if (fullContent) {
        try {
          await callbacks.onComplete?.(send, { content: fullContent, usage });
        } catch (err) {
          console.error("onComplete callback error:", err);
        }
      }

      send("[DONE]");
      controller.close();
    },
  });
}
