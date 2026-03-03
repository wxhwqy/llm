export interface ApiResponse<T> {
  data: T;
}

export interface PaginatedResponse<T> {
  data: T[];
  pagination: {
    page: number;
    pageSize: number;
    total: number;
    totalPages: number;
  };
}

export interface CursorResponse<T> {
  data: T[];
  hasMore: boolean;
  nextCursor: string | null;
}

export interface ApiError {
  code: string;
  message: string;
  details?: { field: string; message: string }[];
}

export interface SSEChunkEvent {
  id: string;
  object: "chat.completion.chunk";
  choices: {
    delta: { content?: string };
    finish_reason: "stop" | "length" | null;
    index: number;
  }[];
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

export interface SSEUserMessageEvent {
  type: "user_message";
  message: import("./chat").ChatMessage;
}

export interface SSEMessageCompleteEvent {
  type: "message_complete";
  message: import("./chat").ChatMessage;
  contextUsage: { usedTokens: number; maxTokens: number };
}

export interface SSEErrorEvent {
  type: "error";
  error: { code: string; message: string };
}

export type SSEEvent =
  | SSEChunkEvent
  | SSEUserMessageEvent
  | SSEMessageCompleteEvent
  | SSEErrorEvent;
