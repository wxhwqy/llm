import { create } from "zustand";
import type { ChatMessage } from "@/types/chat";

interface ChatStore {
  messages: ChatMessage[];
  isStreaming: boolean;
  streamingContent: string;

  setMessages: (messages: ChatMessage[]) => void;
  addMessage: (msg: ChatMessage) => void;
  replaceMessageId: (tempId: string, realMsg: ChatMessage) => void;
  removeLastAssistant: () => void;
  editMessage: (id: string, content: string) => void;

  setStreaming: (v: boolean) => void;
  appendStreamContent: (token: string) => void;
  finalizeStream: (msg: ChatMessage) => void;
  resetStream: () => void;
}

export const useChatStore = create<ChatStore>((set) => ({
  messages: [],
  isStreaming: false,
  streamingContent: "",

  setMessages: (messages) => set({ messages }),

  addMessage: (msg) =>
    set((s) => ({ messages: [...s.messages, msg] })),

  replaceMessageId: (tempId, realMsg) =>
    set((s) => ({
      messages: s.messages.map((m) => (m.id === tempId ? realMsg : m)),
    })),

  removeLastAssistant: () =>
    set((s) => {
      const idx = s.messages.findLastIndex((m) => m.role === "assistant");
      if (idx === -1) return s;
      return { messages: s.messages.filter((_, i) => i !== idx) };
    }),

  editMessage: (id, content) =>
    set((s) => ({
      messages: s.messages.map((m) =>
        m.id === id ? { ...m, content, editedAt: new Date().toISOString() } : m,
      ),
    })),

  setStreaming: (isStreaming) => set({ isStreaming }),

  appendStreamContent: (token) =>
    set((s) => ({ streamingContent: s.streamingContent + token })),

  finalizeStream: (msg) =>
    set((s) => ({
      messages: [...s.messages, msg],
      streamingContent: "",
      isStreaming: false,
    })),

  resetStream: () => set({ streamingContent: "", isStreaming: false }),
}));
