import { UnprocessableError } from "../lib/errors";
import { saveBufferAsFile } from "./upload.service";
import { createCharacter } from "./character.service";

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
    character_book?: unknown;
  };
}

function mapSTCard(st: STCharacterCardV2) {
  const d = st.data;
  return {
    name: d.name,
    description: d.description ?? "",
    personality: d.personality ?? "",
    scenario: d.scenario ?? "",
    systemPrompt: d.system_prompt ?? "",
    firstMessage: d.first_mes ?? "",
    alternateGreetings: d.alternate_greetings ?? [],
    exampleDialogue: d.mes_example ?? "",
    creatorNotes: d.creator_notes ?? "",
    tags: d.tags ?? [],
  };
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

  return createCharacter(
    { ...mapped, avatar: avatarPath, coverImage: coverPath, source: "sillytavern_png" },
    userId,
  );
}

export async function importFromJson(jsonStr: string, userId: string) {
  const json = JSON.parse(jsonStr);

  // Detect SillyTavern format vs internal format
  if (json.spec === "chara_card_v2" || json.data?.name) {
    const stCard: STCharacterCardV2 = json.spec ? json : { data: json };
    const mapped = mapSTCard(stCard);
    return createCharacter({ ...mapped, source: "json_import" }, userId);
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
