from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    model_path: str
    served_model_name: str
    vllm_base_url: str
    vllm_port: int
    gateway_port: int
    web_port: int
    openai_api_key: str
    database_url: str
    default_max_tokens: int
    default_temperature: float
    default_top_p: float
    enable_vllm_metrics: bool
    web_origin: str
    log_level: str
    request_timeout_seconds: int
    bind_localhost_only: bool

    @property
    def database_path(self) -> Path:
        prefix = "sqlite:///"
        if not self.database_url.startswith(prefix):
            raise ValueError("DATABASE_URL must start with sqlite:///")
        return Path(self.database_url.removeprefix(prefix))

    @property
    def allowed_origins(self) -> list[str]:
        origins = [part.strip() for part in self.web_origin.split(",") if part.strip()]
        return origins or ["http://localhost:3000"]

    @property
    def masked_api_key(self) -> str:
        if len(self.openai_api_key) < 8:
            return "***"
        return f"{self.openai_api_key[:4]}...{self.openai_api_key[-4:]}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        model_path=os.getenv(
            "MODEL_PATH",
            "/media/shinkaman/INTEL_TUF/Sefetensors/nvfp4/Huihui-Qwen3.5-35B-A3B-abliterated-NVFP4",
        ),
        served_model_name=os.getenv(
            "SERVED_MODEL_NAME",
            "Huihui-Qwen3.5-35B-A3B-abliterated-NVFP4",
        ),
        vllm_base_url=os.getenv("VLLM_BASE_URL", "http://vllm:8010"),
        vllm_port=int(os.getenv("VLLM_PORT", "8010")),
        gateway_port=int(os.getenv("GATEWAY_PORT", "8000")),
        web_port=int(os.getenv("WEB_PORT", "3000")),
        openai_api_key=os.getenv("OPENAI_API_KEY", "local-dev-only-change-me"),
        database_url=os.getenv("DATABASE_URL", "sqlite:///data/sqlite/nvfp4studio.db"),
        default_max_tokens=int(os.getenv("DEFAULT_MAX_TOKENS", "2048")),
        default_temperature=float(os.getenv("DEFAULT_TEMPERATURE", "0.7")),
        default_top_p=float(os.getenv("DEFAULT_TOP_P", "0.95")),
        enable_vllm_metrics=_env_bool("ENABLE_VLLM_METRICS", True),
        web_origin=os.getenv("WEB_ORIGIN", "http://localhost:3000"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        request_timeout_seconds=int(os.getenv("REQUEST_TIMEOUT_SECONDS", "600")),
        bind_localhost_only=_env_bool("BIND_LOCALHOST_ONLY", True),
    )

