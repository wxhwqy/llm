import type { ImportedCharacterData } from "@/stores/import-store";

interface STCharacterCardV2 {
  spec?: string;
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
  };
}

function mapSTCard(
  st: STCharacterCardV2,
  source: ImportedCharacterData["source"],
): ImportedCharacterData {
  const d = st.data;
  return {
    name: d.name,
    description: d.description ?? "",
    personality: d.personality ?? "",
    scenario: d.scenario ?? "",
    systemPrompt: d.system_prompt ?? "",
    firstMessage: d.first_mes ?? "",
    exampleDialogue: d.mes_example ?? "",
    creatorNotes: d.creator_notes ?? "",
    tags: d.tags ?? [],
    source,
    imageDataUrl: null,
  };
}

/**
 * Extract "chara" tEXt chunk from a PNG ArrayBuffer (browser-compatible).
 */
function extractCharaFromPng(buf: ArrayBuffer): string | null {
  const view = new DataView(buf);
  const PNG_SIG = [137, 80, 78, 71, 13, 10, 26, 10];
  for (let i = 0; i < PNG_SIG.length; i++) {
    if (view.getUint8(i) !== PNG_SIG[i]) return null;
  }

  let offset = 8;
  const decoder = new TextDecoder("ascii");

  while (offset < buf.byteLength) {
    const length = view.getUint32(offset);
    const typeBytes = new Uint8Array(buf, offset + 4, 4);
    const type = decoder.decode(typeBytes);

    if (type === "tEXt") {
      const chunkData = new Uint8Array(buf, offset + 8, length);
      const nullIdx = chunkData.indexOf(0);
      if (nullIdx >= 0) {
        const keyword = decoder.decode(chunkData.subarray(0, nullIdx));
        if (keyword === "chara") {
          return decoder.decode(chunkData.subarray(nullIdx + 1));
        }
      }
    }

    offset += 12 + length; // 4(len) + 4(type) + data + 4(crc)
  }
  return null;
}

/** Convert a File to a data URL string */
function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

export async function parseCharacterPng(
  file: File,
): Promise<ImportedCharacterData> {
  const buf = await file.arrayBuffer();
  const charaData = extractCharaFromPng(buf);
  if (!charaData) {
    throw new Error("该 PNG 文件不包含 SillyTavern 角色卡数据");
  }

  const decoded = atob(charaData);
  const json = JSON.parse(decoded);
  const stCard: STCharacterCardV2 = json.spec ? json : { data: json };
  const data = mapSTCard(stCard, "sillytavern_png");
  data.imageDataUrl = await fileToDataUrl(file);
  return data;
}

export async function parseCharacterJson(
  file: File,
): Promise<ImportedCharacterData> {
  const text = await file.text();
  const json = JSON.parse(text);

  if (json.spec === "chara_card_v2" || json.data?.name) {
    const stCard: STCharacterCardV2 = json.spec ? json : { data: json };
    return mapSTCard(stCard, "json_import");
  }

  // Internal format
  return {
    name: json.name ?? "",
    description: json.description ?? "",
    personality: json.personality ?? "",
    scenario: json.scenario ?? "",
    systemPrompt: json.systemPrompt ?? "",
    firstMessage: json.firstMessage ?? "",
    exampleDialogue: json.exampleDialogue ?? "",
    creatorNotes: json.creatorNotes ?? "",
    tags: json.tags ?? [],
    source: "json_import",
    imageDataUrl: null,
  };
}
