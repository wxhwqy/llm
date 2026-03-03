import { estimateMessagesTokens } from "./token-counter.service";
import { config } from "../lib/config";

interface Message {
  role: string;
  content: string;
}

/**
 * Apply sliding window compression if the total prompt tokens exceed the budget.
 * Returns a trimmed list of history messages (does not touch system/firstMessage).
 */
export function applyContextWindow(
  historyMessages: Message[],
  maxContextLength: number,
  systemTokens: number,
): Message[] {
  const replyBudget = config.defaultMaxTokens;
  const availableBudget = maxContextLength - replyBudget;
  const threshold = availableBudget * config.contextCompressThreshold;

  const historyTokens = estimateMessagesTokens(historyMessages);
  const totalUsed = systemTokens + historyTokens;

  if (totalUsed <= threshold) return historyMessages;

  // Keep as many recent messages as fit within budget
  const targetHistoryTokens = availableBudget - systemTokens;
  const result: Message[] = [];
  let accumulated = 0;

  for (let i = historyMessages.length - 1; i >= 0; i--) {
    const msgTokens = estimateMessagesTokens([historyMessages[i]]);
    if (accumulated + msgTokens > targetHistoryTokens) break;
    accumulated += msgTokens;
    result.unshift(historyMessages[i]);
  }

  return result;
}
