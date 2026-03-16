-- AlterTable
ALTER TABLE "ChatSession" ADD COLUMN "temperature" REAL;
ALTER TABLE "ChatSession" ADD COLUMN "topK" INTEGER;
ALTER TABLE "ChatSession" ADD COLUMN "topP" REAL;
