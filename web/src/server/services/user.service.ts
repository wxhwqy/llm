import bcrypt from "bcryptjs";
import jwt from "jsonwebtoken";
import { config } from "../lib/config";
import {
  ConflictError,
  NotFoundError,
  ForbiddenError,
  UnauthorizedError,
  AppError,
} from "../lib/errors";
import {
  findUserById,
  findUserByEmail,
  findUserByUsername,
  createUser,
  updateUser,
  deleteUserById,
  listUsersPaginated,
  getUserTotalTokens,
} from "../repositories/user.repository";

// ─── Types ────────────────────────────────────────────────

interface UserPublic {
  id: string;
  username: string;
  email: string;
  role: string;
  status: string;
  createdAt: string;
}

function formatUser(u: {
  id: string;
  username: string;
  email: string;
  role: string;
  status: string;
  createdAt: Date;
}): UserPublic {
  return {
    id: u.id,
    username: u.username,
    email: u.email,
    role: u.role,
    status: u.status,
    createdAt: u.createdAt.toISOString(),
  };
}

function signToken(userId: string, role: string): string {
  return jwt.sign({ userId, role }, config.jwtSecret, { expiresIn: "7d" });
}

// ─── Auth ─────────────────────────────────────────────────

export async function register(data: {
  username: string;
  email: string;
  password: string;
}) {
  const [existingEmail, existingUsername] = await Promise.all([
    findUserByEmail(data.email),
    findUserByUsername(data.username),
  ]);
  if (existingEmail) throw new ConflictError("该邮箱已被注册");
  if (existingUsername) throw new ConflictError("该用户名已被使用");

  const passwordHash = await bcrypt.hash(data.password, 12);
  const user = await createUser({
    username: data.username,
    email: data.email,
    passwordHash,
  });

  const token = signToken(user.id, user.role);
  return { user: formatUser(user), token };
}

export async function login(email: string, password: string) {
  const user = await findUserByEmail(email);
  if (!user) {
    throw new AppError(401, "INVALID_CREDENTIALS", "邮箱或密码错误");
  }

  if (user.status !== "active") {
    throw new AppError(403, "ACCOUNT_DISABLED", "账号已被禁用");
  }

  const valid = await bcrypt.compare(password, user.passwordHash);
  if (!valid) {
    throw new AppError(401, "INVALID_CREDENTIALS", "邮箱或密码错误");
  }

  const token = signToken(user.id, user.role);
  return { user: formatUser(user), token };
}

export async function verifyToken(token: string): Promise<UserPublic> {
  let payload: { userId: string };
  try {
    payload = jwt.verify(token, config.jwtSecret) as { userId: string };
  } catch {
    throw new UnauthorizedError("登录已过期，请重新登录");
  }

  const user = await findUserById(payload.userId);
  if (!user) throw new UnauthorizedError("用户不存在");
  if (user.status !== "active") {
    throw new AppError(403, "ACCOUNT_DISABLED", "账号已被禁用");
  }

  return formatUser(user);
}

// ─── Self-service ─────────────────────────────────────────

export async function updateProfile(
  userId: string,
  data: { username?: string; email?: string },
) {
  if (data.username) {
    const existing = await findUserByUsername(data.username);
    if (existing && existing.id !== userId) {
      throw new ConflictError("该用户名已被使用");
    }
  }
  if (data.email) {
    const existing = await findUserByEmail(data.email);
    if (existing && existing.id !== userId) {
      throw new ConflictError("该邮箱已被注册");
    }
  }

  const user = await updateUser(userId, data);
  return formatUser(user);
}

export async function changePassword(
  userId: string,
  oldPassword: string,
  newPassword: string,
) {
  const user = await findUserById(userId);
  if (!user) throw new NotFoundError("用户");

  const valid = await bcrypt.compare(oldPassword, user.passwordHash);
  if (!valid) {
    throw new AppError(401, "INVALID_CREDENTIALS", "当前密码错误");
  }

  const passwordHash = await bcrypt.hash(newPassword, 12);
  await updateUser(userId, { passwordHash });
}

// ─── Admin ────────────────────────────────────────────────

export async function listUsers(params: {
  search?: string;
  role?: string;
  status?: string;
  page: number;
  pageSize: number;
}) {
  const { items, total } = await listUsersPaginated(params);

  const data = await Promise.all(
    items.map(async (u) => {
      const totalTokens = await getUserTotalTokens(u.id);
      return {
        id: u.id,
        username: u.username,
        email: u.email,
        role: u.role,
        status: u.status,
        sessionCount: u._count.sessions,
        totalTokens,
        createdAt: u.createdAt.toISOString(),
        updatedAt: u.updatedAt.toISOString(),
      };
    }),
  );

  return {
    data,
    pagination: { page: params.page, pageSize: params.pageSize, total },
  };
}

export async function adminUpdateUser(
  adminId: string,
  targetId: string,
  data: { role?: string; status?: string },
) {
  if (adminId === targetId) {
    throw new ForbiddenError("不能修改自己的角色或状态");
  }

  const user = await findUserById(targetId);
  if (!user) throw new NotFoundError("用户");

  const updated = await updateUser(targetId, data);
  return formatUser(updated);
}

export async function adminDeleteUser(adminId: string, targetId: string) {
  if (adminId === targetId) {
    throw new ForbiddenError("不能删除自己");
  }

  const user = await findUserById(targetId);
  if (!user) throw new NotFoundError("用户");

  await deleteUserById(targetId);
}
