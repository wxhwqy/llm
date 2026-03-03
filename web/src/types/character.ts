export interface CharacterCard {
  id: string;
  name: string;
  avatar: string | null;
  coverImage: string | null;
  description: string;
  personality: string;
  scenario: string;
  systemPrompt: string;
  firstMessage: string;
  alternateGreetings: string[];
  exampleDialogue: string;
  creatorNotes: string;
  worldBookIds: string[];
  tags: string[];
  source: "manual" | "sillytavern_png" | "json_import";
  createdBy: string;
  createdAt: string;
  updatedAt: string;
}

export type CharacterSummary = Pick<
  CharacterCard,
  "id" | "name" | "avatar" | "coverImage" | "description" | "tags" | "source" | "createdAt" | "updatedAt"
>;

export interface CharacterCreateInput {
  name: string;
  description: string;
  personality: string;
  scenario: string;
  systemPrompt: string;
  firstMessage: string;
  alternateGreetings?: string[];
  exampleDialogue?: string;
  creatorNotes?: string;
  worldBookIds?: string[];
  tags?: string[];
  coverImageFile?: File;
}

export type CharacterUpdateInput = Partial<CharacterCreateInput>;
