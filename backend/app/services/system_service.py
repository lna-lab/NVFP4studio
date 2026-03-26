from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

from app.core.config import Settings
from app.db.repository import BenchmarkRepository
from app.models.schemas import GPUStat, MetricSnapshot, SystemConfigResponse, SystemStatusResponse
from app.services.vllm_client import VllmClient
from app.telemetry.prometheus import parse_metrics


class SystemService:
    def __init__(
        self,
        settings: Settings,
        repository: BenchmarkRepository,
        vllm_client: VllmClient,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._vllm_client = vllm_client

    async def get_status(self) -> SystemStatusResponse:
        model_path = Path(self._settings.model_path)
        vllm_healthy, metrics_text = await asyncio.gather(
            self._vllm_client.health(),
            self._vllm_client.metrics() if self._settings.enable_vllm_metrics else asyncio.sleep(0, result=None),
        )
        metrics = parse_metrics(metrics_text or "") if metrics_text else {}
        return SystemStatusResponse(
            gateway_ok=True,
            vllm_healthy=vllm_healthy,
            model_path_exists=model_path.exists(),
            model_path=str(model_path),
            served_model_name=self._settings.served_model_name,
            database_path=str(self._settings.database_path),
            metrics_available=bool(metrics_text),
            gpu=self._read_gpu_stats(),
            metrics=MetricSnapshot(values=metrics),
            recent_benchmark_count=self._repository.count(),
        )

    def get_config(self) -> SystemConfigResponse:
        return SystemConfigResponse(
            model_path=self._settings.model_path,
            served_model_name=self._settings.served_model_name,
            vllm_base_url=self._settings.vllm_base_url,
            gateway_port=self._settings.gateway_port,
            web_port=self._settings.web_port,
            database_url=self._settings.database_url,
            default_max_tokens=self._settings.default_max_tokens,
            default_temperature=self._settings.default_temperature,
            default_top_p=self._settings.default_top_p,
            enable_vllm_metrics=self._settings.enable_vllm_metrics,
            bind_localhost_only=self._settings.bind_localhost_only,
            web_origin=self._settings.allowed_origins,
            openai_api_key_hint=self._settings.masked_api_key,
        )

    @staticmethod
    def _read_gpu_stats() -> list[GPUStat]:
        command = [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return []

        rows = []
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 4:
                continue
            rows.append(
                GPUStat(
                    name=parts[0],
                    memory_total_mb=int(parts[1]) if parts[1].isdigit() else None,
                    memory_used_mb=int(parts[2]) if parts[2].isdigit() else None,
                    utilization_gpu_percent=int(parts[3]) if parts[3].isdigit() else None,
                )
            )
        return rows

