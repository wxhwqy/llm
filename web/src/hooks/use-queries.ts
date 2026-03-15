"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { mockApi } from "@/lib/mock-api";
import { USE_MOCK } from "@/lib/constants";
import { api } from "@/lib/api-client";
import type { PaginatedResponse, ApiResponse, CursorResponse } from "@/types/api";
import type { CharacterCard, CharacterSummary } from "@/types/character";
import type { ChatSession, ChatMessage } from "@/types/chat";
import type { WorldBookSummary, WorldBookDetail } from "@/types/worldbook";
import type { User, AdminUserItem, TokenUsageStats, ModelInfo } from "@/types/user";
import { useAuthStore } from "@/stores/auth-store";

export function useCurrentUser() {
  const setUser = useAuthStore((s) => s.setUser);
  return useQuery({
    queryKey: ["auth", "me"],
    queryFn: async () => {
      try {
        const res = USE_MOCK
          ? await mockApi.getMe()
          : await api.get<ApiResponse<User>>("/auth/me");
        setUser(res.data);
        return res;
      } catch {
        setUser(null);
        return { data: null as User | null };
      }
    },
    staleTime: Infinity,
    retry: false,
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

interface CharacterMutationPayload {
  name: string;
  description: string;
  personality: string;
  preset: string;
  scenario: string;
  systemPrompt: string;
  firstMessage: string;
  exampleDialogue: string;
  creatorNotes: string;
  source: string;
  tags: string[];
  worldBookIds: string[];
  coverImageDataUrl?: string | null;
}

function buildCharacterFormData(payload: CharacterMutationPayload): FormData {
  const fd = new FormData();
  const { coverImageDataUrl, tags, worldBookIds, ...fields } = payload;
  for (const [k, v] of Object.entries(fields)) {
    fd.append(k, v as string);
  }
  fd.append("tags", JSON.stringify(tags));
  fd.append("worldBookIds", JSON.stringify(worldBookIds));
  fd.append("alternateGreetings", JSON.stringify([]));
  if (coverImageDataUrl?.startsWith("data:")) {
    const [meta, b64] = coverImageDataUrl.split(",");
    const mime = meta.match(/:(.*?);/)?.[1] ?? "image/png";
    const bytes = atob(b64);
    const arr = new Uint8Array(bytes.length);
    for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i);
    fd.append("coverImageFile", new Blob([arr], { type: mime }), "cover.png");
  }
  return fd;
}

export function useCreateCharacter() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (payload: CharacterMutationPayload) => {
      if (USE_MOCK) {
        return mockApi.createCharacter({
          name: payload.name,
          description: payload.description,
          personality: payload.personality,
          preset: payload.preset,
          scenario: payload.scenario,
          systemPrompt: payload.systemPrompt,
          firstMessage: payload.firstMessage,
          alternateGreetings: [],
          exampleDialogue: payload.exampleDialogue,
          creatorNotes: payload.creatorNotes,
          source: payload.source as CharacterCard["source"],
          tags: payload.tags,
          worldBookIds: payload.worldBookIds,
          avatar: null,
          coverImage: payload.coverImageDataUrl ?? null,
        });
      }
      const fd = buildCharacterFormData(payload);
      return api.upload<ApiResponse<CharacterCard>>("/admin/characters", fd);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["characters"] });
    },
  });
}

export function useUpdateCharacter() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ id, ...payload }: CharacterMutationPayload & { id: string }) => {
      if (USE_MOCK) {
        return mockApi.updateCharacter(id, {
          name: payload.name,
          description: payload.description,
          personality: payload.personality,
          preset: payload.preset,
          scenario: payload.scenario,
          systemPrompt: payload.systemPrompt,
          firstMessage: payload.firstMessage,
          exampleDialogue: payload.exampleDialogue,
          creatorNotes: payload.creatorNotes,
          source: payload.source as CharacterCard["source"],
          tags: payload.tags,
          worldBookIds: payload.worldBookIds,
          coverImage: payload.coverImageDataUrl ?? null,
        });
      }
      const fd = buildCharacterFormData(payload);
      return api.uploadPut<ApiResponse<CharacterCard>>(`/admin/characters/${id}`, fd);
    },
    onSuccess: (_data, variables) => {
      queryClient.invalidateQueries({ queryKey: ["characters"] });
      queryClient.invalidateQueries({ queryKey: ["character", variables.id] });
    },
  });
}

export function useDeleteCharacter() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) =>
      USE_MOCK
        ? mockApi.deleteCharacter(id)
        : api.delete<ApiResponse<{ deleted: true }>>(`/admin/characters/${id}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["characters"] });
    },
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

// ─── Auth Mutations ──────────────────────────────────────

export function useLogin() {
  const queryClient = useQueryClient();
  const setUser = useAuthStore((s) => s.setUser);
  return useMutation({
    mutationFn: (data: { email: string; password: string }) =>
      USE_MOCK
        ? mockApi.login(data.email, data.password)
        : api.post<ApiResponse<{ user: User }>>("/auth/login", data),
    onSuccess: (res) => {
      queryClient.clear();
      setUser(res.data.user);
      queryClient.setQueryData(["auth", "me"], { data: res.data.user });
    },
  });
}

export function useRegister() {
  const queryClient = useQueryClient();
  const setUser = useAuthStore((s) => s.setUser);
  return useMutation({
    mutationFn: (data: { username: string; email: string; password: string }) =>
      USE_MOCK
        ? mockApi.register(data.username, data.email, data.password)
        : api.post<ApiResponse<{ user: User }>>("/auth/register", data),
    onSuccess: (res) => {
      queryClient.clear();
      setUser(res.data.user);
      queryClient.setQueryData(["auth", "me"], { data: res.data.user });
    },
  });
}

export function useLogout() {
  const queryClient = useQueryClient();
  const setUser = useAuthStore((s) => s.setUser);
  return useMutation({
    mutationFn: () =>
      USE_MOCK
        ? mockApi.logout()
        : api.post<ApiResponse<{ success: true }>>("/auth/logout"),
    onSuccess: () => {
      setUser(null);
      queryClient.clear();
    },
  });
}

export function useUpdateProfile() {
  const queryClient = useQueryClient();
  const setUser = useAuthStore((s) => s.setUser);
  return useMutation({
    mutationFn: (data: { username?: string; email?: string }) =>
      USE_MOCK
        ? mockApi.updateProfile(data)
        : api.put<ApiResponse<User>>("/users/me", data),
    onSuccess: (res) => {
      setUser(res.data);
      queryClient.setQueryData(["auth", "me"], { data: res.data });
    },
  });
}

export function useChangePassword() {
  return useMutation({
    mutationFn: (data: { oldPassword: string; newPassword: string }) =>
      USE_MOCK
        ? mockApi.changePassword()
        : api.put<ApiResponse<{ success: true }>>("/users/me/password", data),
  });
}

// ─── Admin User Management ──────────────────────────────

export function useAdminUsers(params?: { search?: string; role?: string; status?: string; page?: number }) {
  return useQuery({
    queryKey: ["admin", "users", params],
    queryFn: () => {
      const sp = new URLSearchParams();
      if (params?.search) sp.set("search", params.search);
      if (params?.role) sp.set("role", params.role);
      if (params?.status) sp.set("status", params.status);
      if (params?.page) sp.set("page", String(params.page));
      return api.get<PaginatedResponse<AdminUserItem>>(`/admin/users?${sp}`);
    },
  });
}

export function useAdminUpdateUser() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ id, ...data }: { id: string; role?: string; status?: string }) =>
      api.put<ApiResponse<User>>(`/admin/users/${id}`, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["admin", "users"] });
    },
  });
}

export function useAdminDeleteUser() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) =>
      api.delete<ApiResponse<{ deleted: true }>>(`/admin/users/${id}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["admin", "users"] });
    },
  });
}
