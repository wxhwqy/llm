export interface WorldBookEntry {
  id: string;
  keywords: string[];
  secondaryKeywords: string[];
  content: string;
  position: "before_system" | "after_system" | "before_user";
  priority: number;
  enabled: boolean;
  tokenCount: number;
}

export interface WorldBook {
  id: string;
  name: string;
  description: string;
  scope: "global" | "personal";
  userId: string;
  totalTokenCount: number;
  characterCount: number;
  createdAt: string;
  updatedAt: string;
}

export interface WorldBookSummary extends WorldBook {
  entryCount: number;
}

export interface WorldBookDetail extends WorldBook {
  entries: WorldBookEntry[];
}

export interface WorldBookCreateInput {
  name: string;
  description: string;
  scope: "global" | "personal";
  entries?: Omit<WorldBookEntry, "id" | "tokenCount">[];
}

export interface EntryCreateInput {
  keywords: string[];
  secondaryKeywords?: string[];
  content: string;
  position: "before_system" | "after_system" | "before_user";
  priority: number;
  enabled: boolean;
}
