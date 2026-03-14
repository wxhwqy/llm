import { API_BASE } from "./constants";
import type { ApiError } from "@/types/api";

export class ApiClientError extends Error {
  constructor(
    public status: number,
    public error: ApiError,
  ) {
    super(error.message);
    this.name = "ApiClientError";
  }
}

async function request<T>(
  endpoint: string,
  options?: RequestInit,
): Promise<T> {
  const res = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({
      error: { code: "UNKNOWN", message: `HTTP ${res.status}` },
    }));
    throw new ApiClientError(res.status, body.error);
  }

  return res.json();
}

function streamRequest(
  endpoint: string,
  body: unknown,
  signal?: AbortSignal,
): Promise<Response> {
  return fetch(`${API_BASE}${endpoint}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
}

function uploadRequest<T>(
  endpoint: string,
  formData: FormData,
  method: "POST" | "PUT" = "POST",
): Promise<T> {
  return fetch(`${API_BASE}${endpoint}`, {
    method,
    body: formData,
  }).then(async (res) => {
    if (!res.ok) {
      const body = await res.json().catch(() => ({
        error: { code: "UNKNOWN", message: `HTTP ${res.status}` },
      }));
      throw new ApiClientError(res.status, body.error);
    }
    return res.json();
  });
}

export const api = {
  get: <T>(endpoint: string, signal?: AbortSignal) =>
    request<T>(endpoint, { signal }),

  post: <T>(endpoint: string, body?: unknown) =>
    request<T>(endpoint, {
      method: "POST",
      body: body ? JSON.stringify(body) : undefined,
    }),

  put: <T>(endpoint: string, body?: unknown) =>
    request<T>(endpoint, {
      method: "PUT",
      body: body ? JSON.stringify(body) : undefined,
    }),

  delete: <T>(endpoint: string) =>
    request<T>(endpoint, { method: "DELETE" }),

  stream: streamRequest,
  upload: <T>(endpoint: string, formData: FormData) =>
    uploadRequest<T>(endpoint, formData, "POST"),
  uploadPut: <T>(endpoint: string, formData: FormData) =>
    uploadRequest<T>(endpoint, formData, "PUT"),
};
