// Lightweight token estimation without heavy tokenizer dependency.
// CJK-heavy text ~1.5 chars/token; English/code ~4 chars/token.
// Calibrated against real usage from LLM Service responses.

export function estimateTokens(text: string): number {
  if (!text) return 0;
  const cjk = (text.match(/[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]/g) || []).length;
  const other = text.length - cjk;
  return Math.ceil(cjk / 1.5 + other / 4);
}

export function estimateMessagesTokens(
  messages: { role: string; content: string }[],
): number {
  let total = 0;
  for (const m of messages) {
    total += estimateTokens(m.content) + 4; // 4 tokens overhead per message
  }
  return total + 2; // <|im_start|> / <|im_end|> overhead
}
