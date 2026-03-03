import { prisma } from "../db/prisma";
import { parseJsonArray } from "../lib/json-fields";
import { config } from "../lib/config";

interface MatchedEntry {
  content: string;
  position: string;
  priority: number;
  tokenCount: number;
}

export async function collectWorldbookEntries(
  characterId: string,
  sessionId: string,
  scanText: string,
  tokenBudget: number,
): Promise<MatchedEntry[]> {
  // Gather worldbook IDs: character's global worldbooks + session's personal worldbooks
  const [charWbs, sessionWbs] = await Promise.all([
    prisma.characterWorldBook.findMany({
      where: { characterId },
      select: { worldBookId: true },
    }),
    prisma.sessionWorldBook.findMany({
      where: { sessionId },
      select: { worldBookId: true },
    }),
  ]);

  const wbIds = [
    ...charWbs.map((w) => w.worldBookId),
    ...sessionWbs.map((w) => w.worldBookId),
  ];
  if (!wbIds.length) return [];

  const entries = await prisma.worldBookEntry.findMany({
    where: { worldBookId: { in: wbIds }, enabled: true },
    orderBy: { priority: "desc" },
  });

  const normalizedScan = scanText.toLowerCase();
  const matched: MatchedEntry[] = [];

  for (const entry of entries) {
    const keywords = parseJsonArray(entry.keywords);
    const secondaryKws = parseJsonArray(entry.secondaryKeywords);

    const primaryHit = keywords.some((kw) => normalizedScan.includes(kw.toLowerCase()));
    if (!primaryHit) continue;

    if (secondaryKws.length > 0) {
      const secondaryHit = secondaryKws.some((kw) => normalizedScan.includes(kw.toLowerCase()));
      if (!secondaryHit) continue;
    }

    matched.push({
      content: entry.content,
      position: entry.position,
      priority: entry.priority,
      tokenCount: entry.tokenCount,
    });
  }

  // Token budget trimming
  let usedTokens = 0;
  const result: MatchedEntry[] = [];
  for (const e of matched) {
    if (usedTokens + e.tokenCount > tokenBudget) continue;
    usedTokens += e.tokenCount;
    result.push(e);
  }

  return result;
}

export function buildScanText(
  currentInput: string,
  historyMessages: { content: string }[],
): string {
  const rounds = config.worldbookScanRounds;
  const recent = historyMessages.slice(-rounds * 2);
  return recent.map((m) => m.content).join("\n") + "\n" + currentInput;
}
