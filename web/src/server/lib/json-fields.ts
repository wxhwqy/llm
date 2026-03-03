// SQLite stores arrays as JSON strings. These helpers bridge Prisma ↔ API responses.

export function parseJsonArray(val: string | null | undefined): string[] {
  if (!val) return [];
  try {
    const parsed = JSON.parse(val);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function toJsonString(arr: string[] | undefined | null): string {
  return JSON.stringify(arr ?? []);
}
