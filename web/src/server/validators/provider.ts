import { z } from "zod";

const modelItemSchema = z.object({
  id: z.string().min(1, "模型 ID 不能为空"),
  name: z.string().min(1, "模型名称不能为空"),
  maxContextLength: z.number().int().positive("上下文长度必须为正整数"),
});

export const createProviderSchema = z.object({
  name: z.string().min(1, "名称不能为空").max(100),
  baseUrl: z.string().url("请输入合法的 URL"),
  apiKey: z.string().optional().default(""),
  models: z.array(modelItemSchema).optional().default([]),
  autoDiscover: z.boolean().optional().default(true),
  enabled: z.boolean().optional().default(true),
  priority: z.number().int().optional().default(0),
});

export const updateProviderSchema = z.object({
  name: z.string().min(1).max(100).optional(),
  baseUrl: z.string().url("请输入合法的 URL").optional(),
  apiKey: z.string().optional(),
  models: z.array(modelItemSchema).optional(),
  autoDiscover: z.boolean().optional(),
  enabled: z.boolean().optional(),
  priority: z.number().int().optional(),
});

export type CreateProviderInput = z.infer<typeof createProviderSchema>;
export type UpdateProviderInput = z.infer<typeof updateProviderSchema>;
export type ModelItem = z.infer<typeof modelItemSchema>;
