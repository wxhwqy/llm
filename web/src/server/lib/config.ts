export const config = {
  llmServiceUrl: process.env.LLM_SERVICE_URL || "http://localhost:8000",
  jwtSecret: process.env.JWT_SECRET || "dev-secret-key-change-in-production-at-least-32-chars",
  defaultModelId: process.env.DEFAULT_MODEL_ID || "qwen3-32b",
  defaultTemperature: parseFloat(process.env.DEFAULT_TEMPERATURE || "0.6"),
  defaultMaxTokens: parseInt(process.env.DEFAULT_MAX_TOKENS || "2048", 10),
  maxConcurrentInferences: parseInt(process.env.MAX_CONCURRENT_INFERENCES || "1", 10),
  maxQueueSize: parseInt(process.env.MAX_QUEUE_SIZE || "10", 10),
  worldbookTokenBudgetRatio: parseFloat(process.env.WORLDBOOK_TOKEN_BUDGET_RATIO || "0.25"),
  contextCompressThreshold: parseFloat(process.env.CONTEXT_COMPRESS_THRESHOLD || "0.8"),
  worldbookScanRounds: parseInt(process.env.WORLDBOOK_SCAN_ROUNDS || "5", 10),
  uploadDir: process.env.UPLOAD_DIR || "./uploads",
  maxUploadSizeMb: parseInt(process.env.MAX_UPLOAD_SIZE_MB || "10", 10),
};
