import { ForbiddenError, NotFoundError } from "../lib/errors";
import { estimateTokens } from "./token-counter.service";
import { formatMessage } from "./session.service";
import * as sessionRepo from "../repositories/session.repository";
import * as messageRepo from "../repositories/message.repository";

export { formatMessage } from "./session.service";

export async function getMessages(sessionId: string, userId: string, cursor?: string, limit = 50) {
  await sessionRepo.findOwnedSession(sessionId, userId);

  const { data, hasMore, nextCursor } = await messageRepo.findMessagesCursor(sessionId, cursor, limit);

  return {
    data: data.map(formatMessage),
    hasMore,
    nextCursor,
  };
}

export async function editMessage(sessionId: string, messageId: string, userId: string, content: string) {
  await sessionRepo.findOwnedSession(sessionId, userId);

  const msg = await messageRepo.findMessageById(messageId);
  if (!msg || msg.sessionId !== sessionId) throw new NotFoundError("消息");
  if (msg.role !== "user") throw new ForbiddenError("只能编辑自己的消息");

  const updated = await messageRepo.updateMessage(messageId, {
    content,
    tokenCount: estimateTokens(content),
    editedAt: new Date(),
  });

  return formatMessage(updated);
}
