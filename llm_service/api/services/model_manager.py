"""Model lifecycle management: loading, caching, listing."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from api.config import ModelConfig, PROJECT_DIR

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    config: ModelConfig
    model: Any
    tokenizer: Any
    loaded_at: float = field(default_factory=time.monotonic)


class ModelNotFoundError(Exception):
    def __init__(self, model_id: str):
        self.model_id = model_id
        super().__init__(f"Model not found: {model_id}")


class ModelLoadError(Exception):
    pass


class ModelManager:
    """Manages model instances.  Thread-unsafe loading is guarded by a lock."""

    def __init__(self, configs: list[ModelConfig]):
        self._configs: dict[str, ModelConfig] = {c.id: c for c in configs}
        self._loaded: dict[str, LoadedModel] = {}
        self._lock = asyncio.Lock()

    async def startup(self, default_model: str) -> None:
        logger.info("Pre-loading default model '%s' ...", default_model)
        await self.get_model(default_model)
        logger.info("Default model ready.")

    async def get_model(self, model_id: str) -> LoadedModel:
        if model_id in self._loaded:
            return self._loaded[model_id]

        async with self._lock:
            if model_id in self._loaded:
                return self._loaded[model_id]
            if model_id not in self._configs:
                raise ModelNotFoundError(model_id)
            loaded = await self._load(model_id)
            self._loaded[model_id] = loaded
            return loaded

    async def _load(self, model_id: str) -> LoadedModel:
        config = self._configs[model_id]
        loop = asyncio.get_event_loop()
        t0 = time.time()
        model, tokenizer = await loop.run_in_executor(
            None, self._load_sync, config
        )
        elapsed = time.time() - t0
        logger.info(
            "Model '%s' loaded in %.1fs (tp=%d)",
            model_id, elapsed, config.tp_size,
        )
        return LoadedModel(config=config, model=model, tokenizer=tokenizer)

    @staticmethod
    def _load_sync(config: ModelConfig) -> tuple[Any, Any]:
        import llaisys
        from llaisys.libllaisys import DeviceType
        from transformers import AutoTokenizer

        model_path = config.path
        if not os.path.isabs(model_path):
            model_path = str(PROJECT_DIR / model_path)

        device = DeviceType.NVIDIA if config.device == "nvidia" else DeviceType.CPU
        device_ids: Any = (
            config.device_ids if config.tp_size > 1 else config.device_ids[0]
        )

        if config.model_type == "qwen3_5_moe":
            model = llaisys.models.Qwen3_5Moe(model_path, device, config.device_ids[0])
        elif config.model_type == "qwen3_5":
            model = llaisys.models.Qwen3_5(model_path, device, config.device_ids[0])
        else:
            model = llaisys.models.Qwen3(model_path, device, device_ids)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True
        )
        return model, tokenizer

    def list_models(self) -> list[dict]:
        result = []
        for cfg in self._configs.values():
            result.append({
                "id": cfg.id,
                "name": cfg.name,
                "max_context_length": cfg.max_seq_len,
                "status": "loaded" if cfg.id in self._loaded else "available",
            })
        return result

    async def shutdown(self) -> None:
        for mid, loaded in list(self._loaded.items()):
            logger.info("Unloading model '%s'", mid)
            del loaded.model
        self._loaded.clear()
