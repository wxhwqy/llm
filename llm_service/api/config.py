"""Application configuration loaded from environment variables."""

from __future__ import annotations

import json
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent.parent


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    id: str
    name: str
    path: str
    model_type: str = "qwen3"  # "qwen3", "qwen3_5", or "qwen3_5_moe"
    max_seq_len: int = 16384
    device: str = "nvidia"
    device_ids: list[int] = field(default_factory=lambda: [0, 1])
    tp_size: int = 2


@dataclass
class Settings:
    """Global application settings."""

    host: str = "0.0.0.0"
    port: int = 8000

    default_model: str = "qwen3-32b"
    model_configs: list[ModelConfig] = field(default_factory=list)

    default_temperature: float = 0.6
    default_top_k: int = 20
    default_top_p: float = 0.95
    default_max_tokens: int = 2048

    max_concurrent: int = 1
    max_queue_size: int = 16
    queue_timeout_seconds: float = 120.0

    @classmethod
    def from_env(cls) -> Settings:
        settings = cls(
            host=os.getenv("LLM_HOST", "0.0.0.0"),
            port=int(os.getenv("LLM_PORT", "8000")),
            default_model=os.getenv("LLM_DEFAULT_MODEL", "qwen3-32b"),
            default_temperature=float(os.getenv("LLM_DEFAULT_TEMPERATURE", "0.6")),
            default_top_k=int(os.getenv("LLM_DEFAULT_TOP_K", "20")),
            default_top_p=float(os.getenv("LLM_DEFAULT_TOP_P", "0.95")),
            default_max_tokens=int(os.getenv("LLM_DEFAULT_MAX_TOKENS", "2048")),
            max_concurrent=int(os.getenv("LLM_MAX_CONCURRENT", "1")),
            max_queue_size=int(os.getenv("LLM_MAX_QUEUE_SIZE", "16")),
            queue_timeout_seconds=float(os.getenv("LLM_QUEUE_TIMEOUT", "120")),
        )

        models_file = os.getenv("LLM_MODELS_CONFIG", str(PROJECT_DIR / "models.json"))
        if os.path.isfile(models_file):
            with open(models_file) as f:
                raw = json.load(f)
            settings.model_configs = [ModelConfig(**m) for m in raw]
        else:
            settings.model_configs = [_default_model_config()]

        return settings


def _default_model_config() -> ModelConfig:
    """Fallback model config when models.json is absent."""
    device = os.getenv("LLM_DEVICE", "nvidia")
    device_ids_raw = os.getenv("LLM_DEVICE_IDS", "[0,1]")
    tp_size = int(os.getenv("LLM_TP_SIZE", "2"))
    model_path = os.getenv("LLM_MODEL_PATH", "models/qwen3_32b_fp8")

    try:
        device_ids = json.loads(device_ids_raw)
    except (json.JSONDecodeError, TypeError):
        device_ids = [0]

    return ModelConfig(
        id=os.getenv("LLM_DEFAULT_MODEL", "qwen3-32b"),
        name="Qwen3 32B (FP8)",
        path=model_path,
        max_seq_len=int(os.getenv("LLM_MAX_SEQ_LEN", "16384")),
        device=device,
        device_ids=device_ids,
        tp_size=tp_size,
    )


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings
