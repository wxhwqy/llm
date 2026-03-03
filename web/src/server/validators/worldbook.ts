import { z } from "zod";

export const worldbookListSchema = z.object({
  scope: z.enum(["global", "personal"]).optional(),
  page: z.coerce.number().int().positive().default(1),
  pageSize: z.coerce.number().int().min(1).max(100).default(50),
});

export const createWorldbookSchema = z.object({
  name: z.string().min(1).max(200),
  description: z.string().default(""),
  scope: z.enum(["global", "personal"]).default("personal"),
  entries: z
    .array(
      z.object({
        keywords: z.array(z.string().min(1)).min(1),
        secondaryKeywords: z.array(z.string()).optional().default([]),
        content: z.string().min(1),
        position: z
          .enum(["before_system", "after_system", "before_user"])
          .default("after_system"),
        priority: z.number().int().min(0).max(1000).default(0),
        enabled: z.boolean().default(true),
      }),
    )
    .optional()
    .default([]),
});

export const updateWorldbookSchema = z.object({
  name: z.string().min(1).max(200).optional(),
  description: z.string().optional(),
});

export const createEntrySchema = z.object({
  keywords: z.array(z.string().min(1)).min(1),
  secondaryKeywords: z.array(z.string()).optional().default([]),
  content: z.string().min(1),
  position: z
    .enum(["before_system", "after_system", "before_user"])
    .default("after_system"),
  priority: z.number().int().min(0).max(1000).default(0),
  enabled: z.boolean().default(true),
});

export const updateEntrySchema = createEntrySchema.partial();
