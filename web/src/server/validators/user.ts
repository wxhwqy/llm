import { z } from "zod";

export const loginSchema = z.object({
  email: z.string().email("邮箱格式不正确"),
  password: z.string().min(1, "密码不能为空"),
});

export const registerSchema = z.object({
  username: z.string().min(2, "用户名至少2个字符").max(20, "用户名最多20个字符"),
  email: z.string().email("邮箱格式不正确"),
  password: z.string().min(8, "密码至少8个字符"),
});

export const updateProfileSchema = z.object({
  username: z.string().min(2).max(20).optional(),
  email: z.string().email().optional(),
});

export const changePasswordSchema = z.object({
  oldPassword: z.string().min(1, "请输入当前密码"),
  newPassword: z.string().min(8, "新密码至少8个字符"),
});

export const adminUpdateUserSchema = z.object({
  role: z.enum(["admin", "user"]).optional(),
  status: z.enum(["active", "disabled"]).optional(),
});

export const adminUserListSchema = z.object({
  search: z.string().optional(),
  role: z.enum(["admin", "user"]).optional(),
  status: z.enum(["active", "disabled"]).optional(),
  page: z.coerce.number().int().min(1).default(1),
  pageSize: z.coerce.number().int().min(1).max(100).default(20),
});
