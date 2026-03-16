import { create } from "zustand";

export interface CharacterBookEntry {
  keywords: string[];
  secondaryKeywords: string[];
  content: string;
  position: "before_system" | "after_system" | "before_user";
  priority: number;
  enabled: boolean;
}

export interface CharacterBookInfo {
  name: string | null;
  description: string | null;
  entryCount: number;
  entries: CharacterBookEntry[];
}

export interface ImportedCharacterData {
  name: string;
  description: string;
  personality: string;
  preset?: string;
  scenario: string;
  systemPrompt: string;
  firstMessage: string;
  exampleDialogue: string;
  creatorNotes: string;
  tags: string[];
  source: "sillytavern_png" | "json_import";
  /** PNG 文件的 data URL，用作封面和头像预览 */
  imageDataUrl: string | null;
  /** 角色卡自带的世界书信息 */
  characterBook: CharacterBookInfo | null;
}

interface ImportStore {
  data: ImportedCharacterData | null;
  setData: (data: ImportedCharacterData) => void;
  clear: () => void;
}

export const useImportStore = create<ImportStore>((set) => ({
  data: null,
  setData: (data) => set({ data }),
  clear: () => set({ data: null }),
}));
