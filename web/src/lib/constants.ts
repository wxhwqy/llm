export const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "/api";

export const DEFAULT_MODEL_ID = "qwen3-32b";

export const DEFAULT_PAGE_SIZE = 20;

export const AVATAR_COLORS = [
  "bg-violet-600",
  "bg-rose-600",
  "bg-sky-600",
  "bg-emerald-600",
  "bg-amber-600",
  "bg-fuchsia-600",
  "bg-cyan-600",
  "bg-orange-600",
] as const;

export const COVER_GRADIENTS: Record<string, string> = {
  default: "from-gray-500 to-gray-700",
  violet: "from-violet-600 via-fuchsia-500 to-pink-500",
  cyber: "from-slate-800 via-cyan-700 to-blue-900",
  warm: "from-pink-300 via-rose-200 to-amber-200",
  dark: "from-gray-900 via-purple-900 to-slate-800",
  tech: "from-indigo-500 via-blue-400 to-cyan-300",
  fire: "from-orange-600 via-red-500 to-amber-500",
};

export function getAvatarColor(id: string) {
  let hash = 0;
  for (let i = 0; i < id.length; i++) hash = (hash * 31 + id.charCodeAt(i)) | 0;
  return AVATAR_COLORS[Math.abs(hash) % AVATAR_COLORS.length];
}

export function getCoverGradient(id: string) {
  const keys = Object.keys(COVER_GRADIENTS);
  let hash = 0;
  for (let i = 0; i < id.length; i++) hash = (hash * 31 + id.charCodeAt(i)) | 0;
  return COVER_GRADIENTS[keys[Math.abs(hash) % keys.length]];
}

export function timeAgo(dateStr: string) {
  const diff = Date.now() - new Date(dateStr).getTime();
  const days = Math.floor(diff / 86400000);
  if (days < 1) return "今天";
  if (days < 30) return `${days}天前`;
  const months = Math.floor(days / 30);
  if (months < 12) return `${months}个月前`;
  return `${Math.floor(months / 12)}年前`;
}
