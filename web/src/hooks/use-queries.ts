"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { mockApi } from "@/lib/mock-api";
import { USE_MOCK } from "@/lib/constants";
import { api } from "@/lib/api-client";
import type { PaginatedResponse, ApiResponse, CursorResponse } from "@/types/api";
import type { CharacterCard, CharacterSummary } from "@/types/character";
import type { ChatSession, ChatMessage } from "@/types/chat";
import type { WorldBookSummary, WorldBookDetail } from "@/types/worldbook";
import type { User, TokenUsageStats, ModelInfo } from "@/types/user";

export function useCurrentUser() {
  return useQuery({
    queryKey: ["auth", "me"],
    queryFn: () =>
      USE_MOCK
        ? mockApi.getMe()
        : api.get<ApiResponse<User>>("/auth/me"),
    staleTime: Infinity,
  });
}

export function useCharacters(params?: { search?: string; tags?: string[] }) {
  return useQuery({
    queryKey: ["characters", params],
    queryFn: () => {
      if (USE_MOCK) return mockApi.getCharacters(params ? { search: params.search, tag: params.tags } : undefined);
      const sp = new URLSearchParams();
      if (params?.search) sp.set("search", params.search);
      params?.tags?.forEach((t) => sp.append("tag", t));
      return api.get<PaginatedResponse<CharacterSummary>>(`/characters?${sp}`);
    },
  });
}

export function useCharacter(id: string) {
  return useQuery({
    queryKey: ["character", id],
    queryFn: () =>
      USE_MOCK
        ? mockApi.getCharacter(id)
        : api.get<ApiResponse<CharacterCard>>(`/characters/${id}`),
    enabled: !!id,
  });
}

export function useCharacterTags() {
  return useQuery({
    queryKey: ["characters", "tags"],
    queryFn: () =>
      USE_MOCK ? mockApi.getTags() : api.get<ApiResponse<string[]>>("/characters/tags"),
    staleTime: 60_000,
  });
}

export function useSessions() {
  return useQuery({
    queryKey: ["chat", "sessions"],
    queryFn: () =>
      USE_MOCK
        ? mockApi.getSessions()
        : api.get<PaginatedResponse<ChatSession>>("/chat/sessions"),
    staleTime: 10_000,
  });
}

export function useMessages(sessionId: string) {
  return useQuery({
    queryKey: ["chat", "messages", sessionId],
    queryFn: () =>
      USE_MOCK
        ? mockApi.getMessages(sessionId)
        : api.get<CursorResponse<ChatMessage>>(`/chat/sessions/${sessionId}/messages?limit=50`),
    enabled: !!sessionId,
  });
}

export function useWorldBooks(scope?: string) {
  return useQuery({
    queryKey: ["worldbooks", scope],
    queryFn: () => {
      if (USE_MOCK) return mockApi.getWorldBooks(scope);
      const sp = scope ? `?scope=${scope}` : "";
      return api.get<PaginatedResponse<WorldBookSummary>>(`/worldbooks${sp}`);
    },
  });
}

export function useWorldBook(id: string) {
  return useQuery({
    queryKey: ["worldbook", id],
    queryFn: () =>
      USE_MOCK
        ? mockApi.getWorldBook(id)
        : api.get<ApiResponse<WorldBookDetail>>(`/worldbooks/${id}`),
    enabled: !!id,
  });
}

export function useCreateSession() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (characterId: string) =>
      USE_MOCK
        ? mockApi.createSession(characterId)
        : api.post<ApiResponse<ChatSession & { messages: ChatMessage[] }>>(
            "/chat/sessions",
            { characterId },
          ),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["chat", "sessions"] });
    },
  });
}

export function useModels() {
  return useQuery({
    queryKey: ["models"],
    queryFn: () =>
      USE_MOCK ? mockApi.getModels() : api.get<ApiResponse<ModelInfo[]>>("/models"),
    staleTime: 300_000,
  });
}

export function useTokenUsage() {
  return useQuery({
    queryKey: ["usage"],
    queryFn: () =>
      USE_MOCK ? mockApi.getUsage() : api.get<ApiResponse<TokenUsageStats>>("/users/me/usage"),
    staleTime: 60_000,
  });
}
