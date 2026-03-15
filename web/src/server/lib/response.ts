import { NextRequest, NextResponse } from "next/server";
import { AppError, ValidationError } from "./errors";
import type { ZodType } from "zod";

type RouteContext = { params: Promise<Record<string, string>> };

export type AuthUser = {
  id: string;
  username: string;
  email: string;
  role: string;
  status: string;
  createdAt: string;
};

type Handler = (req: NextRequest, ctx: RouteContext) => Promise<Response>;
type AuthHandler = (req: NextRequest, ctx: RouteContext, user: AuthUser) => Promise<Response>;

export function apiHandler(handler: Handler): Handler {
  return async (req, ctx) => {
    try {
      return await handler(req, ctx);
    } catch (error) {
      if (error instanceof AppError) {
        const body: Record<string, unknown> = {
          error: { code: error.code, message: error.message },
        };
        if (error.details) {
          (body.error as Record<string, unknown>).details = error.details;
        }
        return NextResponse.json(body, { status: error.statusCode });
      }
      console.error("Unhandled error:", error);
      return NextResponse.json(
        { error: { code: "INTERNAL_ERROR", message: "服务器内部错误" } },
        { status: 500 },
      );
    }
  };
}

export function withAuth(handler: AuthHandler): Handler {
  return apiHandler(async (req, ctx) => {
    const { getCurrentUser } = await import("../middleware/auth");
    const user = await getCurrentUser(req);
    return handler(req, ctx, user);
  });
}

export function withAdmin(handler: AuthHandler): Handler {
  return withAuth(async (req, ctx, user) => {
    if (user.role !== "admin") {
      return NextResponse.json(
        { error: { code: "FORBIDDEN", message: "需要管理员权限" } },
        { status: 403 },
      );
    }
    return handler(req, ctx, user);
  });
}

export function jsonOk<T>(data: T, status = 200) {
  return NextResponse.json({ data }, { status });
}

export function jsonCreated<T>(data: T) {
  return jsonOk(data, 201);
}

export function jsonPaginated<T>(
  data: T[],
  pagination: { page: number; pageSize: number; total: number },
) {
  return NextResponse.json({
    data,
    pagination: {
      ...pagination,
      totalPages: Math.ceil(pagination.total / pagination.pageSize),
    },
  });
}

export function jsonCursor<T>(
  data: T[],
  hasMore: boolean,
  nextCursor: string | null,
) {
  return NextResponse.json({ data, hasMore, nextCursor });
}

/** Parse request body with a Zod schema; throws ValidationError on failure. */
export async function validateBody<T>(req: NextRequest, schema: ZodType<T>): Promise<T> {
  const body = await req.json();
  const result = schema.safeParse(body);
  if (!result.success) {
    const details = result.error.issues.map((issue) => ({
      field: issue.path.map(String).join("."),
      message: issue.message,
    }));
    throw new ValidationError("参数校验失败", details);
  }
  return result.data;
}
