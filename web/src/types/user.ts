export interface User {
  id: string;
  username: string;
  email: string;
  role: "admin" | "user";
  createdAt: string;
}

export interface TokenUsageStats {
  summary: {
    totalPromptTokens: number;
    totalCompletionTokens: number;
    totalTokens: number;
    totalSessions: number;
    totalMessages: number;
  };
  timeline: {
    date: string;
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  }[];
}

export interface ModelInfo {
  id: string;
  name: string;
  maxContextLength: number;
  status: "online" | "offline" | "busy";
}
