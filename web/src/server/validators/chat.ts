import { z } from "zod";

export const createSessionSchema = z.object({
  characterId: z.string().min(1),
  modelId: z.string().optional(),
});

export const updateSessionSchema = z.object({
  modelId: z.string().optional(),
  personalWorldBookIds: z.array(z.string()).optional(),
  title: z.string().optional(),
  temperature: z.number().min(0).max(2).optional(),
  topP: z.number().min(0).max(1).optional(),
  topK: z.number().int().min(0).optional(),
});

export const sendMessageSchema = z.object({
  content: z.string().min(1).max(50000),
});

export const editMessageSchema = z.object({
  content: z.string().min(1).max(50000),
});

export const messagesQuerySchema = z.object({
  cursor: z.string().optional(),
  limit: z.coerce.number().int().min(1).max(100).default(50),
});

export const regenerateSchema = z.object({
  modelId: z.string().optional(),
});
