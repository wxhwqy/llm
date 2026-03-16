import { UnprocessableError } from "../lib/errors";
import { saveBufferAsFile } from "./upload.service";
import { createCharacter } from "./character.service";
import { createWorldbook } from "./worldbook.service";
import { prisma } from "../db/prisma";

// ─── SillyTavern Character Book types ────────────────────

interface STCharacterBookEntry {
  keys?: string[];
  key?: string[];
  secondary_keys?: string[];
  content?: string;
  enabled?: boolean;
  disable?: boolean;
  insertion_order?: number;
  order?: number;
  priority?: number;
  position?: string | number;
  extensions?: Record<string, unknown>;
}

interface STCharacterBook {
  name?: string;
  description?: string;
  entries: Record<string, STCharacterBookEntry> | STCharacterBookEntry[];
}

interface STCharacterCardV2 {
  spec?: string;
  spec_version?: string;
  data: {
    name: string;
    description?: string;
    personality?: string;
    scenario?: string;
    first_mes?: string;
    mes_example?: string;
    system_prompt?: string;
    alternate_greetings?: string[];
    creator_notes?: string;
    tags?: string[];
    post_history_instructions?: string;
    character_book?: STCharacterBook;
  };
}

/**
 * 去除 HTML 标签，保留纯文本。
 * JanitorAI 等平台会在文本字段中嵌入 HTML（装饰图、样式标签等），
 * 导入时需要清理为纯文本。
 */
function stripHtml(html: string): string {
  if (!html) return "";
  return html
    .replace(/<img[^>]*>/gi, "")           // 移除图片标签
    .replace(/<br\s*\/?>/gi, "\n")         // <br> → 换行
    .replace(/<\/p>/gi, "\n")              // </p> → 换行
    .replace(/<[^>]+>/g, "")               // 移除其余所有标签
    .replace(/&nbsp;/gi, " ")
    .replace(/&amp;/gi, "&")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">")
    .replace(/&quot;/gi, '"')
    .replace(/&#39;/gi, "'")
    .replace(/\n{3,}/g, "\n\n")            // 合并多余空行
    .trim();
}

/**
 * 检测字符串是否包含 HTML 标签
 */
function containsHtml(text: string): boolean {
  return /<[a-z][\s\S]*>/i.test(text);
}

/**
 * 映射 SillyTavern / JanitorAI 角色卡字段到内部格式（1:1 映射，不交换字段）。
 *
 * 内部字段语义：
 * - description: 角色定义，组 prompt 给 AI 用（对应 ST.description）
 * - personality: 给用户看的角色介绍（对应 ST.personality）
 *
 * JanitorAI 等平台可能在 personality 中嵌入 HTML，导入时做清理。
 */
function mapSTCard(st: STCharacterCardV2) {
  const d = st.data;

  // personality 可能包含 JanitorAI 平台注入的 HTML，需要清理
  const rawPersonality = d.personality ?? "";
  const cleanedPersonality = containsHtml(rawPersonality)
    ? stripHtml(rawPersonality)
    : rawPersonality;

  return {
    name: d.name,
    description: d.description ?? "",
    personality: cleanedPersonality,
    preset: "",
    scenario: d.scenario ?? "",
    systemPrompt: d.system_prompt ?? "",
    firstMessage: d.first_mes ?? "",
    alternateGreetings: d.alternate_greetings ?? [],
    exampleDialogue: d.mes_example ?? "",
    creatorNotes: d.creator_notes ?? "",
    tags: d.tags ?? [],
  };
}

/**
 * Map SillyTavern character_book position field to our internal position string.
 * ST uses numbers: 0 = before_char (after_system), 1 = after_char (after_system),
 * or string values. Default to "after_system".
 */
function mapEntryPosition(pos?: string | number): string {
  if (pos === "before_system" || pos === "after_system" || pos === "before_user") return pos;
  if (pos === 0) return "before_system";
  if (pos === 1) return "after_system";
  return "after_system";
}

/**
 * Parse SillyTavern character_book → create WorldBook + Entries → associate to character.
 * Returns the created WorldBook ID, or null if no character_book.
 */
async function importCharacterBook(
  characterBook: STCharacterBook | undefined,
  characterId: string,
  characterName: string,
  userId: string,
): Promise<string | null> {
  if (!characterBook) return null;

  const rawEntries = characterBook.entries;
  if (!rawEntries) return null;

  // Normalize entries: ST uses object with numeric keys or array
  const entryList: STCharacterBookEntry[] = Array.isArray(rawEntries)
    ? rawEntries
    : Object.values(rawEntries);

  if (entryList.length === 0) return null;

  // Map entries to our format
  const entries = entryList.map((e) => ({
    keywords: e.keys ?? e.key ?? [],
    secondaryKeywords: e.secondary_keys ?? [],
    content: e.content ?? "",
    position: mapEntryPosition(e.position),
    priority: e.insertion_order ?? e.order ?? e.priority ?? 0,
    enabled: e.enabled ?? (e.disable !== true),
  }));

  // Create WorldBook as global scope (imported with character card)
  const wb = await createWorldbook(
    {
      name: characterBook.name || `${characterName} - 世界书`,
      description: characterBook.description ?? `从角色卡「${characterName}」导入的世界书`,
      scope: "global",
      entries: entries as Array<Record<string, unknown>>,
    },
    userId,
    "admin", // character import is admin-only, so use admin role
  );

  // Associate WorldBook to CharacterCard via CharacterWorldBook
  await prisma.characterWorldBook.create({
    data: { characterId, worldBookId: wb.id },
  });

  return wb.id;
}

export async function importFromPng(buffer: Buffer, userId: string) {
  // Parse PNG chunks to find tEXt chunk with keyword "chara"
  let charaData: string | null = null;

  try {
    // Simple PNG tEXt chunk parser (avoids native deps)
    charaData = extractCharaFromPng(buffer);
  } catch {
    throw new UnprocessableError("INVALID_CHARACTER_FILE", "PNG 文件解析失败");
  }

  if (!charaData) {
    throw new UnprocessableError("INVALID_CHARACTER_FILE", "PNG 文件不包含角色卡数据");
  }

  const decoded = Buffer.from(charaData, "base64").toString("utf-8");
  const json = JSON.parse(decoded);
  const stCard: STCharacterCardV2 = json.spec ? json : { data: json };
  const mapped = mapSTCard(stCard);

  const avatarPath = await saveBufferAsFile(buffer, "avatars");
  const coverPath = await saveBufferAsFile(buffer, "covers");

  const character = await createCharacter(
    { ...mapped, avatar: avatarPath, coverImage: coverPath, source: "sillytavern_png" },
    userId,
  );

  // Import character_book if present
  await importCharacterBook(stCard.data.character_book, character.id as string, mapped.name, userId);

  return character;
}

export async function importFromJson(jsonStr: string, userId: string) {
  const json = JSON.parse(jsonStr);

  // Detect SillyTavern format vs internal format
  if (json.spec === "chara_card_v2" || json.data?.name) {
    const stCard: STCharacterCardV2 = json.spec ? json : { data: json };
    const mapped = mapSTCard(stCard);
    const character = await createCharacter({ ...mapped, source: "json_import" }, userId);

    // Import character_book if present
    await importCharacterBook(stCard.data.character_book, character.id as string, mapped.name, userId);

    return character;
  }

  // Internal format: use fields directly
  return createCharacter({ ...json, source: "json_import" }, userId);
}

function extractCharaFromPng(buffer: Buffer): string | null {
  // PNG starts with 8-byte signature, then chunks: [4 len][4 type][data][4 crc]
  const PNG_SIG = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
  if (!buffer.subarray(0, 8).equals(PNG_SIG)) return null;

  let offset = 8;
  while (offset < buffer.length) {
    const length = buffer.readUInt32BE(offset);
    const type = buffer.subarray(offset + 4, offset + 8).toString("ascii");

    if (type === "tEXt") {
      const chunkData = buffer.subarray(offset + 8, offset + 8 + length);
      const nullIdx = chunkData.indexOf(0);
      if (nullIdx >= 0) {
        const keyword = chunkData.subarray(0, nullIdx).toString("ascii");
        if (keyword === "chara") {
          return chunkData.subarray(nullIdx + 1).toString("ascii");
        }
      }
    }

    // Move to next chunk: 4(len) + 4(type) + length(data) + 4(crc)
    offset += 12 + length;
  }
  return null;
}
