import { create } from "zustand";

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
