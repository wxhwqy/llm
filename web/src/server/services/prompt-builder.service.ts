interface LLMMessage {
  role: string;
  content: string;
}

interface MatchedEntry {
  content: string;
  position: string;
  priority: number;
  tokenCount: number;
}

interface CharacterData {
  systemPrompt: string;
  personality: string;
  scenario: string;
  exampleDialogue: string;
  firstMessage: string;
}

export function buildPrompt(params: {
  character: CharacterData;
  worldBookEntries: MatchedEntry[];
  historyMessages: { role: string; content: string }[];
  userInput: string;
}): LLMMessage[] {
  const { character, worldBookEntries, historyMessages, userInput } = params;
  const messages: LLMMessage[] = [];

  // -- System message --
  const beforeSystem = worldBookEntries
    .filter((e) => e.position === "before_system")
    .map((e) => e.content)
    .join("\n\n");

  const afterSystem = worldBookEntries
    .filter((e) => e.position === "after_system")
    .map((e) => e.content)
    .join("\n\n");

  let systemContent = "";
  if (beforeSystem) systemContent += beforeSystem + "\n\n";
  if (character.systemPrompt) systemContent += character.systemPrompt + "\n\n";
  if (character.personality || character.scenario) {
    systemContent += "## 角色设定\n";
    if (character.personality) systemContent += `性格: ${character.personality}\n`;
    if (character.scenario) systemContent += `场景: ${character.scenario}\n`;
    systemContent += "\n";
  }
  if (afterSystem) systemContent += afterSystem + "\n\n";
  if (character.exampleDialogue) {
    systemContent += `## 示例对话\n${character.exampleDialogue}\n`;
  }

  if (systemContent.trim()) {
    messages.push({ role: "system", content: systemContent.trim() });
  }

  // -- First message --
  if (character.firstMessage) {
    messages.push({ role: "assistant", content: character.firstMessage });
  }

  // -- History --
  for (const msg of historyMessages) {
    messages.push({ role: msg.role, content: msg.content });
  }

  // -- Current user input (with before_user entries) --
  const beforeUser = worldBookEntries
    .filter((e) => e.position === "before_user")
    .map((e) => e.content)
    .join("\n\n");

  const userContent = beforeUser
    ? `${beforeUser}\n\n${userInput}`
    : userInput;

  messages.push({ role: "user", content: userContent });

  return messages;
}
