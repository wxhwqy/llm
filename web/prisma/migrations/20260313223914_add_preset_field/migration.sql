-- RedefineTables
PRAGMA defer_foreign_keys=ON;
PRAGMA foreign_keys=OFF;
CREATE TABLE "new_CharacterCard" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "name" TEXT NOT NULL,
    "avatar" TEXT,
    "coverImage" TEXT,
    "description" TEXT NOT NULL DEFAULT '',
    "personality" TEXT NOT NULL DEFAULT '',
    "preset" TEXT NOT NULL DEFAULT '',
    "scenario" TEXT NOT NULL DEFAULT '',
    "systemPrompt" TEXT NOT NULL DEFAULT '',
    "firstMessage" TEXT NOT NULL DEFAULT '',
    "alternateGreetings" TEXT NOT NULL DEFAULT '[]',
    "exampleDialogue" TEXT NOT NULL DEFAULT '',
    "creatorNotes" TEXT NOT NULL DEFAULT '',
    "tags" TEXT NOT NULL DEFAULT '[]',
    "source" TEXT NOT NULL DEFAULT 'manual',
    "createdBy" TEXT NOT NULL,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" DATETIME NOT NULL,
    CONSTRAINT "CharacterCard_createdBy_fkey" FOREIGN KEY ("createdBy") REFERENCES "User" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);
INSERT INTO "new_CharacterCard" ("alternateGreetings", "avatar", "coverImage", "createdAt", "createdBy", "creatorNotes", "description", "exampleDialogue", "firstMessage", "id", "name", "personality", "scenario", "source", "systemPrompt", "tags", "updatedAt") SELECT "alternateGreetings", "avatar", "coverImage", "createdAt", "createdBy", "creatorNotes", "description", "exampleDialogue", "firstMessage", "id", "name", "personality", "scenario", "source", "systemPrompt", "tags", "updatedAt" FROM "CharacterCard";
DROP TABLE "CharacterCard";
ALTER TABLE "new_CharacterCard" RENAME TO "CharacterCard";
CREATE INDEX "CharacterCard_name_idx" ON "CharacterCard"("name");
CREATE INDEX "CharacterCard_createdBy_idx" ON "CharacterCard"("createdBy");
PRAGMA foreign_keys=ON;
PRAGMA defer_foreign_keys=OFF;
