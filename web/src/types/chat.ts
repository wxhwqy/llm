export interface ChatMessage {
  id: string;
  role: "system" | "user" | "assistant";
  content: string;
  tokenCount: number;
  isCompressed: boolean;
  createdAt: string;
  editedAt: string | null;
}

export interface ChatSession {
  id: string;
  characterId: string;
  characterName: string;
  characterAvatar: string | null;
  characterCoverImage: string | null;
  modelId: string;
  title: string;
  lastMessage: string;
  personalWorldBookIds: string[];
  contextUsage: {
    usedTokens: number;
    maxTokens: number;
  };
  createdAt: string;
  updatedAt: string;
}

export interface ChatSessionDetail extends ChatSession {
  messages: ChatMessage[];
}

export interface CreateSessionInput {
  characterId: string;
  modelId?: string;
}

export interface UpdateSessionInput {
  modelId?: string;
  personalWorldBookIds?: string[];
  title?: string;
}
