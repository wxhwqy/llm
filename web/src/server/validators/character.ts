import { z } from "zod";

export const characterSearchSchema = z.object({
  search: z.string().optional(),
  tag: z.union([z.string(), z.array(z.string())]).optional(),
  page: z.coerce.number().int().positive().default(1),
  pageSize: z.coerce.number().int().min(1).max(100).default(20),
  sort: z.enum(["createdAt", "name", "updatedAt"]).default("createdAt"),
  order: z.enum(["asc", "desc"]).default("desc"),
});

export const createCharacterSchema = z.object({
  name: z.string().min(1).max(200),
  description: z.string().default(""),
  personality: z.string().default(""),
  preset: z.string().optional().default(""),
  scenario: z.string().default(""),
  systemPrompt: z.string().default(""),
  firstMessage: z.string().default(""),
  alternateGreetings: z.array(z.string()).optional().default([]),
  exampleDialogue: z.string().optional().default(""),
  creatorNotes: z.string().optional().default(""),
  worldBookIds: z.array(z.string()).optional().default([]),
  tags: z.array(z.string()).optional().default([]),
});

export const updateCharacterSchema = createCharacterSchema.partial();
