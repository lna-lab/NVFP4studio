from __future__ import annotations

import asyncio
import json
import subprocess
from dataclasses import dataclass
from math import floor
from pathlib import Path

from app.core.config import Settings
from app.db.repository import BenchmarkRepository
from app.models.schemas import (
    GPUStat,
    MetricSnapshot,
    RuntimeAdvisory,
    RuntimeConfigApplyResponse,
    RuntimeConfigApplyRequest,
    RuntimeConfigValidation,
    RuntimeProfileName,
    SystemConfigResponse,
    SystemStatusResponse,
)
from app.services.gpu_monitor import read_gpu_snapshots
from app.services.vllm_client import VllmClient
from app.telemetry.prometheus import parse_metrics


CONTEXT_CAPACITY_HEADROOM_RATIO = 0.92
VRAM_CAPACITY_HEADROOM_RATIO = 0.97
LINEAR_ATTENTION_CACHE_FACTOR = 0.25
DEFAULT_RUNTIME_PROFILE: RuntimeProfileName = "speed"
RUNTIME_ENV_KEYS = (
    "VLLM_RUNTIME_PROFILE",
    "MAX_MODEL_LEN",
    "GPU_MEMORY_UTILIZATION",
    "MAX_NUM_SEQS",
    "MAX_NUM_BATCHED_TOKENS",
    "KV_CACHE_DTYPE",
    "KV_CACHE_MEMORY_BYTES",
    "CPU_OFFLOAD_GB",
    "SWAP_SPACE",
)


@dataclass(frozen=True)
class ModelContextProfile:
    native_context: int | None = None
    full_attention_layers: int = 0
    linear_attention_layers: int = 0
    num_key_value_heads: int | None = None
    head_dim: int | None = None
    cache_dtype_bytes: int = 2

    @property
    def effective_cache_layers(self) -> float:
        total = self.full_attention_layers + (self.linear_attention_layers * LINEAR_ATTENTION_CACHE_FACTOR)
        return total if total > 0 else 0.0

    @property
    def kv_cache_bytes_per_token(self) -> int | None:
        if self.num_key_value_heads is None or self.head_dim is None:
            return None
        effective_layers = self.effective_cache_layers
        if effective_layers <= 0:
            return None
        per_layer = 2 * self.num_key_value_heads * self.head_dim * self.cache_dtype_bytes
        return int(round(per_layer * effective_layers))


@dataclass(frozen=True)
class RuntimePreset:
    name: RuntimeProfileName
    label: str
    description: str
    gpu_memory_utilization: float
    max_num_seqs: int
    max_num_batched_tokens: int
    kv_cache_dtype: str
    kv_cache_memory_bytes: str | None
    cpu_offload_gb: float
    swap_space_gb: float

    def env_values(self, max_model_len: int) -> dict[str, str]:
        return {
            "VLLM_RUNTIME_PROFILE": self.name,
            "MAX_MODEL_LEN": str(max_model_len),
            "GPU_MEMORY_UTILIZATION": _format_float_env(self.gpu_memory_utilization),
            "MAX_NUM_SEQS": str(self.max_num_seqs),
            "MAX_NUM_BATCHED_TOKENS": str(self.max_num_batched_tokens),
            "KV_CACHE_DTYPE": self.kv_cache_dtype,
            "KV_CACHE_MEMORY_BYTES": self.kv_cache_memory_bytes or "",
            "CPU_OFFLOAD_GB": _format_float_env(self.cpu_offload_gb),
            "SWAP_SPACE": _format_float_env(self.swap_space_gb),
        }


@dataclass(frozen=True)
class RuntimeEnvConfig:
    profile: RuntimeProfileName
    max_model_len: int
    gpu_memory_utilization: float
    max_num_seqs: int
    max_num_batched_tokens: int
    kv_cache_dtype: str
    kv_cache_memory_bytes: str | None
    cpu_offload_gb: float
    swap_space_gb: float


RUNTIME_PRESETS: dict[RuntimeProfileName, RuntimePreset] = {
    "speed": RuntimePreset(
        name="speed",
        label="高速",
        description="現状に近い高速重視。VRAM 予約は厚めで、長い context と token/s を優先します。",
        gpu_memory_utilization=0.85,
        max_num_seqs=4,
        max_num_batched_tokens=8192,
        kv_cache_dtype="auto",
        kv_cache_memory_bytes=None,
        cpu_offload_gb=0,
        swap_space_gb=4,
    ),
    "balanced": RuntimePreset(
        name="balanced",
        label="バランス",
        description="単ユーザー前提で予約 VRAM をかなり絞りつつ、速度低下を抑える中庸設定です。",
        gpu_memory_utilization=0.60,
        max_num_seqs=1,
        max_num_batched_tokens=6144,
        kv_cache_dtype="auto",
        kv_cache_memory_bytes="12G",
        cpu_offload_gb=0,
        swap_space_gb=8,
    ),
    "memory": RuntimePreset(
        name="memory",
        label="省VRAM",
        description="VRAM を強く節約する設定です。KV cache を軽くし、単ユーザー前提で予約量を絞ります。",
        gpu_memory_utilization=0.45,
        max_num_seqs=1,
        max_num_batched_tokens=4096,
        kv_cache_dtype="fp8",
        kv_cache_memory_bytes="8G",
        cpu_offload_gb=0,
        swap_space_gb=16,
    ),
}


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
        vllm_healthy, metrics_text, runtime_max_context, native_context, runtime_env = await asyncio.gather(
            self._vllm_client.health(),
            self._vllm_client.metrics() if self._settings.enable_vllm_metrics else asyncio.sleep(0, result=None),
            self._detect_runtime_max_context(),
            asyncio.to_thread(self._read_native_model_context, model_path),
            asyncio.to_thread(self._read_runtime_env_config, Path(self._settings.project_env_file)),
        )
        metrics = parse_metrics(metrics_text or "") if metrics_text else {}
        gpu = self._read_gpu_stats()
        return SystemStatusResponse(
            gateway_ok=True,
            vllm_healthy=vllm_healthy,
            model_path_exists=model_path.exists(),
            model_path=str(model_path),
            served_model_name=self._settings.served_model_name,
            database_path=str(self._settings.database_path),
            metrics_available=bool(metrics_text),
            gpu=gpu,
            metrics=MetricSnapshot(values=metrics),
            recent_benchmark_count=self._repository.count(),
            advisory=self._build_runtime_advisory(
                gpu=gpu,
                metrics=metrics,
                runtime_max_context=runtime_max_context,
                native_context=native_context,
                runtime_env=runtime_env,
            ),
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

    async def apply_runtime_config(self, payload: RuntimeConfigApplyRequest) -> RuntimeConfigApplyResponse:
        requested_context = payload.max_model_len
        requested_profile = payload.runtime_profile
        status = await self.get_status()
        if status.advisory.runtime_max_context is not None:
            await asyncio.to_thread(self._remember_known_good_context, status.advisory.runtime_max_context)
        validation = self._validate_runtime_context_change(status, requested_context)
        current_runtime_context = status.advisory.runtime_max_context
        current_runtime_profile = status.advisory.runtime_profile
        env_path = Path(self._settings.project_env_file)
        requested_preset = self._resolve_runtime_profile(requested_profile)
        current_env_config = await asyncio.to_thread(self._read_runtime_env_config, env_path)

        if validation.fits_in_vram is False:
            return RuntimeConfigApplyResponse(
                accepted=False,
                restarted=False,
                previous_runtime_context=current_runtime_context,
                applied_runtime_context=current_runtime_context,
                previous_runtime_profile=current_runtime_profile,
                applied_runtime_profile=current_runtime_profile,
                validation=validation,
                message=validation.message,
            )

        if (
            current_runtime_context == requested_context
            and current_env_config.profile == requested_profile
            and current_env_config == self._runtime_env_from_preset(requested_preset, requested_context)
        ):
            return RuntimeConfigApplyResponse(
                accepted=True,
                restarted=False,
                previous_runtime_context=current_runtime_context,
                applied_runtime_context=current_runtime_context,
                previous_runtime_profile=current_runtime_profile,
                applied_runtime_profile=current_runtime_profile,
                validation=validation,
                message="現在の vLLM runtime 設定と同じ値です。再起動は行っていません。",
            )

        previous_env_values = await asyncio.to_thread(self._read_env_values, env_path, RUNTIME_ENV_KEYS)
        next_env_values = requested_preset.env_values(requested_context)

        await asyncio.to_thread(self._write_env_values, env_path, next_env_values)

        try:
            await asyncio.to_thread(self._recreate_vllm_container)
            applied_runtime_context = await self._wait_for_runtime_ready(requested_context)
        except Exception as exc:
            rollback_values = self._normalize_runtime_env_values(previous_env_values, current_runtime_context)
            await asyncio.to_thread(self._write_env_values, env_path, rollback_values)
            try:
                await asyncio.to_thread(self._recreate_vllm_container)
                await self._wait_for_runtime_ready(int(rollback_values["MAX_MODEL_LEN"]))
            except Exception:
                pass
            return RuntimeConfigApplyResponse(
                accepted=False,
                restarted=False,
                previous_runtime_context=current_runtime_context,
                applied_runtime_context=current_runtime_context,
                previous_runtime_profile=current_runtime_profile,
                applied_runtime_profile=current_runtime_profile,
                validation=validation,
                message=f"vLLM の再構成に失敗したため元の設定へ戻しました: {_format_exception(exc)}",
            )

        await asyncio.to_thread(self._remember_known_good_context, applied_runtime_context)
        return RuntimeConfigApplyResponse(
            accepted=True,
            restarted=True,
            previous_runtime_context=current_runtime_context,
            applied_runtime_context=applied_runtime_context,
            previous_runtime_profile=current_runtime_profile,
            applied_runtime_profile=requested_profile,
            validation=validation,
            message=f"runtime profile を {requested_profile} に切り替えて vLLM を再起動しました。",
        )

    @staticmethod
    def _read_gpu_stats() -> list[GPUStat]:
        rows = [
            GPUStat(
                name=snapshot.name,
                memory_total_mb=snapshot.memory_total_mb,
                memory_used_mb=snapshot.memory_used_mb,
                memory_free_mb=snapshot.memory_free_mb,
                utilization_gpu_percent=snapshot.utilization_gpu_percent,
                power_draw_watts=snapshot.power_draw_watts,
                power_limit_watts=snapshot.power_limit_watts,
            )
            for snapshot in read_gpu_snapshots()
        ]
        if rows:
            return rows

        command = [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,power.draw,power.limit",
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
            if len(parts) < 7:
                continue
            rows.append(
                GPUStat(
                    name=parts[0],
                    memory_total_mb=int(parts[1]) if parts[1].isdigit() else None,
                    memory_used_mb=int(parts[2]) if parts[2].isdigit() else None,
                    memory_free_mb=int(parts[3]) if parts[3].isdigit() else None,
                    utilization_gpu_percent=int(parts[4]) if parts[4].isdigit() else None,
                    power_draw_watts=float(parts[5]) if _is_float(parts[5]) else None,
                    power_limit_watts=float(parts[6]) if _is_float(parts[6]) else None,
                )
            )
        return rows

    async def _detect_runtime_max_context(self) -> int | None:
        try:
            response = await self._vllm_client.list_models()
            response.raise_for_status()
            payload = response.json()
        except Exception:
            return None

        models = payload.get("data") or []
        if not models:
            return None
        first_model = models[0]
        value = first_model.get("max_model_len")
        return int(value) if isinstance(value, int | float) else None

    @staticmethod
    def _read_native_model_context(model_path: Path) -> int | None:
        return SystemService._read_model_context_profile(model_path).native_context

    @staticmethod
    def _build_runtime_advisory(
        *,
        gpu: list[GPUStat],
        metrics: dict[str, float],
        runtime_max_context: int | None,
        native_context: int | None,
        runtime_env: RuntimeEnvConfig,
    ) -> RuntimeAdvisory:
        kv_cache_usage = metrics.get("vllm:kv_cache_usage_perc")
        num_running = int(metrics.get("vllm:num_requests_running", 0))
        cpu_offload_detected = metrics.get("vllm:engine_sleep_state:weights_offloaded", 0.0) >= 1.0
        safe_cache_capacity_tokens = SystemService._read_safe_cache_capacity_tokens(metrics)

        recommended_context = runtime_max_context
        if runtime_max_context is not None and kv_cache_usage is not None:
            recommended_context = floor(
                max(2048, min(runtime_max_context, runtime_max_context * max(0.35, 1.0 - kv_cache_usage)))
            )
        if recommended_context is not None and num_running > 0:
            recommended_context = floor(max(2048, recommended_context * 0.85))
        if recommended_context is not None and cpu_offload_detected:
            recommended_context = floor(max(2048, recommended_context * 0.5))

        fits_in_vram = None if runtime_max_context is None else not cpu_offload_detected
        if cpu_offload_detected:
            risk_level = "danger"
            message = "weights offload が検出されました。現在の設定では VRAM に収まらない可能性があります。"
        elif kv_cache_usage is not None and kv_cache_usage >= 0.85:
            risk_level = "danger"
            message = "KV cache 使用率が高く、長いコンテキストでは詰まりやすい状態です。設定を下げてください。"
        elif kv_cache_usage is not None and kv_cache_usage >= 0.65:
            risk_level = "warn"
            message = "KV cache に余裕が少なめです。長いコンテキストを使う前に設定確認をおすすめします。"
        elif num_running > 0:
            risk_level = "warn"
            message = "別リクエストが実行中です。安全側に見るなら少し短めのコンテキストが安心です。"
        else:
            risk_level = "ok"
            message = "現在は VRAM 内で安定動作している状態です。"

        if gpu and all(item.memory_total_mb for item in gpu) and all(item.memory_used_mb is not None for item in gpu):
            total_mb = sum(item.memory_total_mb or 0 for item in gpu)
            used_mb = sum(item.memory_used_mb or 0 for item in gpu)
            if total_mb and used_mb / total_mb >= 0.97 and risk_level == "ok":
                risk_level = "warn"
                message = "VRAM 使用率が非常に高いため、追加の余裕は大きくありません。"

        return RuntimeAdvisory(
            runtime_max_context=runtime_max_context,
            model_native_context=native_context,
            runtime_profile=runtime_env.profile,
            gpu_memory_utilization=metrics.get("vllm:cache_config_info:gpu_memory_utilization")
            or runtime_env.gpu_memory_utilization,
            max_num_seqs=runtime_env.max_num_seqs,
            max_num_batched_tokens=runtime_env.max_num_batched_tokens,
            kv_cache_dtype=runtime_env.kv_cache_dtype,
            kv_cache_memory_bytes=runtime_env.kv_cache_memory_bytes,
            cpu_offload_gb=metrics.get("vllm:cache_config_info:cpu_offload_gb") or runtime_env.cpu_offload_gb,
            swap_space_gb=metrics.get("vllm:cache_config_info:swap_space") or runtime_env.swap_space_gb,
            recommended_context=recommended_context,
            hard_context_limit=runtime_max_context,
            reserved_kv_capacity_tokens=safe_cache_capacity_tokens,
            kv_cache_usage_percent=round(kv_cache_usage * 100, 1) if kv_cache_usage is not None else None,
            cpu_offload_detected=cpu_offload_detected,
            fits_in_vram=fits_in_vram,
            risk_level=risk_level,
            message=message,
        )

    def _validate_runtime_context_change(
        self,
        status: SystemStatusResponse,
        requested_context: int,
    ) -> RuntimeConfigValidation:
        current_runtime_context = status.advisory.runtime_max_context
        native_context = status.advisory.model_native_context

        if native_context is not None and requested_context > native_context:
            return RuntimeConfigValidation(
                requested_context=requested_context,
                current_runtime_context=current_runtime_context,
                model_native_context=native_context,
                fits_in_vram=False,
                risk_level="danger",
                message=f"要求値 {requested_context} はモデル上限 {native_context} を超えています。",
            )

        primary_gpu = status.gpu[0] if status.gpu else None
        known_good_context = self._read_known_good_context()
        current_used_mb = primary_gpu.memory_used_mb if primary_gpu else None
        total_vram_mb = primary_gpu.memory_total_mb if primary_gpu else None
        model_profile = self._read_model_context_profile(Path(self._settings.model_path))
        estimated_required_vram_mb = self._estimate_runtime_vram_requirement_mb(
            current_vram_used_mb=current_used_mb,
            current_runtime_context=current_runtime_context,
            requested_context=requested_context,
            model_profile=model_profile,
        )
        fits_in_vram = None
        risk_level = "warn"
        message = "概算テストを通過しました。vLLM を再起動して確定検証します。"
        safe_capacity_mb = int(total_vram_mb * VRAM_CAPACITY_HEADROOM_RATIO) if total_vram_mb else None
        safe_cache_capacity_tokens = self._read_safe_cache_capacity_tokens(status.metrics.values)

        if known_good_context is not None and requested_context <= known_good_context:
            return RuntimeConfigValidation(
                requested_context=requested_context,
                current_runtime_context=current_runtime_context,
                model_native_context=native_context,
                current_vram_used_mb=current_used_mb,
                total_vram_mb=total_vram_mb,
                estimated_required_vram_mb=estimated_required_vram_mb or current_used_mb,
                fits_in_vram=True,
                risk_level="ok" if current_runtime_context and requested_context <= current_runtime_context else "warn",
                message=f"{known_good_context} tokens までは過去に正常起動を確認済みです。再適用を許可します。",
            )

        if status.advisory.cpu_offload_detected and requested_context > (current_runtime_context or 0):
            return RuntimeConfigValidation(
                requested_context=requested_context,
                current_runtime_context=current_runtime_context,
                model_native_context=native_context,
                current_vram_used_mb=current_used_mb,
                total_vram_mb=total_vram_mb,
                estimated_required_vram_mb=estimated_required_vram_mb,
                fits_in_vram=False,
                risk_level="danger",
                message="現在でも CPU offload が見えているため、より長い context の適用は拒否しました。",
            )

        if safe_cache_capacity_tokens is not None:
            fits_in_vram = requested_context <= safe_cache_capacity_tokens
            if fits_in_vram:
                risk_level = "warn" if requested_context > current_runtime_context else "ok"
                if requested_context > (current_runtime_context or 0):
                    detail = ""
                    if estimated_required_vram_mb is not None and safe_capacity_mb is not None:
                        detail = f"概算 VRAM {estimated_required_vram_mb} MB / 安全容量 {safe_capacity_mb} MB。"
                    message = (
                        f"vLLM の予約済み KV cache 容量では安全側で約 {safe_cache_capacity_tokens} tokens まで見込めます。"
                        f"要求値 {requested_context} はその範囲内です。{detail}"
                    )
                else:
                    message = "現在より小さいか同等の context なので、既存の予約済み cache 内で適用可能と判断しました。"
            else:
                risk_level = "danger"
                detail = ""
                if estimated_required_vram_mb is not None and safe_capacity_mb is not None:
                    detail = f" 概算 VRAM {estimated_required_vram_mb} MB / 安全容量 {safe_capacity_mb} MB。"
                message = (
                    f"vLLM の予約済み KV cache 容量では安全側で約 {safe_cache_capacity_tokens} tokens が上限です。"
                    f"要求値 {requested_context} はこの範囲を超えるため適用しません。{detail}"
                )
        elif estimated_required_vram_mb is not None and safe_capacity_mb is not None:
            fits_in_vram = estimated_required_vram_mb <= safe_capacity_mb
            if fits_in_vram:
                risk_level = "warn" if requested_context > (current_runtime_context or 0) else "ok"
                message = (
                    f"固定常駐分と KV cache 増分を分けて概算すると {estimated_required_vram_mb} MB 程度です。"
                    f"安全容量 {safe_capacity_mb} MB の範囲内なので再起動で検証します。"
                )
            else:
                risk_level = "danger"
                message = (
                    f"固定常駐分と KV cache 増分を分けて概算すると {estimated_required_vram_mb} MB 程度で、"
                    f"安全容量 {safe_capacity_mb} MB を超えます。OOM の可能性が高いため適用しません。"
                )

        return RuntimeConfigValidation(
            requested_context=requested_context,
            current_runtime_context=current_runtime_context,
            model_native_context=native_context,
            current_vram_used_mb=current_used_mb,
            total_vram_mb=total_vram_mb,
            estimated_required_vram_mb=estimated_required_vram_mb,
            fits_in_vram=fits_in_vram,
            risk_level=risk_level,
            message=message,
        )

    def _runtime_state_path(self) -> Path:
        return Path(self._settings.project_root) / "data/runtime/runtime_state.json"

    @staticmethod
    def _resolve_runtime_profile(profile_name: RuntimeProfileName) -> RuntimePreset:
        return RUNTIME_PRESETS.get(profile_name, RUNTIME_PRESETS[DEFAULT_RUNTIME_PROFILE])

    @classmethod
    def _runtime_env_from_preset(cls, preset: RuntimePreset, max_model_len: int) -> RuntimeEnvConfig:
        return RuntimeEnvConfig(
            profile=preset.name,
            max_model_len=max_model_len,
            gpu_memory_utilization=preset.gpu_memory_utilization,
            max_num_seqs=preset.max_num_seqs,
            max_num_batched_tokens=preset.max_num_batched_tokens,
            kv_cache_dtype=preset.kv_cache_dtype,
            kv_cache_memory_bytes=preset.kv_cache_memory_bytes,
            cpu_offload_gb=preset.cpu_offload_gb,
            swap_space_gb=preset.swap_space_gb,
        )

    @classmethod
    def _read_runtime_env_config(cls, env_path: Path) -> RuntimeEnvConfig:
        profile_name = cls._read_env_value(env_path, "VLLM_RUNTIME_PROFILE")
        normalized_profile: RuntimeProfileName = (
            profile_name if profile_name in RUNTIME_PRESETS else DEFAULT_RUNTIME_PROFILE
        )
        preset = cls._resolve_runtime_profile(normalized_profile)
        return RuntimeEnvConfig(
            profile=normalized_profile,
            max_model_len=_read_int_env(cls._read_env_value(env_path, "MAX_MODEL_LEN"), 8192),
            gpu_memory_utilization=_read_float_env(
                cls._read_env_value(env_path, "GPU_MEMORY_UTILIZATION"),
                preset.gpu_memory_utilization,
            ),
            max_num_seqs=_read_int_env(
                cls._read_env_value(env_path, "MAX_NUM_SEQS"),
                preset.max_num_seqs,
            ),
            max_num_batched_tokens=_read_int_env(
                cls._read_env_value(env_path, "MAX_NUM_BATCHED_TOKENS"),
                preset.max_num_batched_tokens,
            ),
            kv_cache_dtype=cls._read_env_value(env_path, "KV_CACHE_DTYPE") or preset.kv_cache_dtype,
            kv_cache_memory_bytes=cls._read_env_value(env_path, "KV_CACHE_MEMORY_BYTES") or None,
            cpu_offload_gb=_read_float_env(
                cls._read_env_value(env_path, "CPU_OFFLOAD_GB"),
                preset.cpu_offload_gb,
            ),
            swap_space_gb=_read_float_env(
                cls._read_env_value(env_path, "SWAP_SPACE"),
                preset.swap_space_gb,
            ),
        )

    @staticmethod
    def _read_model_context_profile(model_path: Path) -> ModelContextProfile:
        candidates: list[int] = []
        full_attention_layers = 0
        linear_attention_layers = 0
        num_key_value_heads = None
        head_dim = None
        cache_dtype_bytes = 2

        config_path = model_path / "config.json"
        if config_path.exists():
            try:
                with config_path.open() as handle:
                    config_payload = json.load(handle)
                text_config = config_payload.get("text_config")
                if not isinstance(text_config, dict):
                    text_config = config_payload
                for payload in (config_payload, text_config):
                    max_position_embeddings = payload.get("max_position_embeddings")
                    if isinstance(max_position_embeddings, int) and max_position_embeddings > 0:
                        candidates.append(max_position_embeddings)
                layer_types = text_config.get("layer_types")
                if isinstance(layer_types, list):
                    full_attention_layers = sum(1 for item in layer_types if item == "full_attention")
                    linear_attention_layers = sum(1 for item in layer_types if item == "linear_attention")
                if full_attention_layers == 0 and linear_attention_layers == 0:
                    num_hidden_layers = text_config.get("num_hidden_layers")
                    if isinstance(num_hidden_layers, int) and num_hidden_layers > 0:
                        full_attention_layers = num_hidden_layers
                num_key_value_heads = _positive_int(text_config.get("num_key_value_heads"))
                head_dim = _positive_int(text_config.get("head_dim"))
                dtype = text_config.get("dtype") or config_payload.get("dtype")
                cache_dtype_bytes = _dtype_size_bytes(dtype)
            except Exception:
                pass

        tokenizer_config_path = model_path / "tokenizer_config.json"
        if tokenizer_config_path.exists():
            try:
                with tokenizer_config_path.open() as handle:
                    tokenizer_payload = json.load(handle)
                model_max_length = tokenizer_payload.get("model_max_length")
                if isinstance(model_max_length, int) and model_max_length > 0:
                    candidates.append(model_max_length)
            except Exception:
                pass

        return ModelContextProfile(
            native_context=max(candidates) if candidates else None,
            full_attention_layers=full_attention_layers,
            linear_attention_layers=linear_attention_layers,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            cache_dtype_bytes=cache_dtype_bytes,
        )

    @staticmethod
    def _read_safe_cache_capacity_tokens(metrics: dict[str, float]) -> int | None:
        block_size = metrics.get("vllm:cache_config_info:block_size")
        num_gpu_blocks = metrics.get("vllm:cache_config_info:num_gpu_blocks")
        if block_size is None or num_gpu_blocks is None:
            return None
        raw_capacity = int(block_size) * int(num_gpu_blocks)
        if raw_capacity <= 0:
            return None
        return max(2048, floor(raw_capacity * CONTEXT_CAPACITY_HEADROOM_RATIO))

    @staticmethod
    def _estimate_runtime_vram_requirement_mb(
        *,
        current_vram_used_mb: int | None,
        current_runtime_context: int | None,
        requested_context: int,
        model_profile: ModelContextProfile,
    ) -> int | None:
        kv_bytes_per_token = model_profile.kv_cache_bytes_per_token
        if current_vram_used_mb is None or current_runtime_context is None or kv_bytes_per_token is None:
            return None
        current_kv_mb = (current_runtime_context * kv_bytes_per_token) / (1024 * 1024)
        baseline_vram_mb = max(0.0, current_vram_used_mb - current_kv_mb)
        requested_kv_mb = (requested_context * kv_bytes_per_token) / (1024 * 1024)
        return int(round(baseline_vram_mb + requested_kv_mb))

    def _read_known_good_context(self) -> int | None:
        state_path = self._runtime_state_path()
        if not state_path.exists():
            return None
        try:
            payload = json.loads(state_path.read_text())
        except Exception:
            return None
        value = payload.get("known_good_context")
        return value if isinstance(value, int) and value > 0 else None

    def _remember_known_good_context(self, context_length: int) -> None:
        state_path = self._runtime_state_path()
        state_path.parent.mkdir(parents=True, exist_ok=True)
        current_value = self._read_known_good_context() or 0
        if context_length <= current_value:
            return
        state_path.write_text(json.dumps({"known_good_context": context_length}, ensure_ascii=False, indent=2))

    def _recreate_vllm_container(self) -> None:
        command = [
            "/usr/local/bin/host-docker",
            "compose",
            "-f",
            self._settings.compose_file_path,
            "--env-file",
            self._settings.project_env_file,
            "--project-name",
            self._settings.compose_project_name,
            "--project-directory",
            self._settings.project_root,
            "up",
            "-d",
            "--force-recreate",
            "--no-deps",
            "vllm",
        ]
        subprocess.run(command, check=True, cwd=self._settings.project_root, capture_output=True, text=True)

    async def _wait_for_runtime_ready(self, requested_context: int) -> int:
        deadline = asyncio.get_running_loop().time() + self._settings.runtime_apply_timeout_seconds
        while asyncio.get_running_loop().time() < deadline:
            if await self._vllm_client.health():
                runtime_max_context = await self._detect_runtime_max_context()
                if runtime_max_context == requested_context:
                    return runtime_max_context
            await asyncio.sleep(5)
        raise TimeoutError(f"vLLM が {requested_context} context で healthy になりませんでした。")

    @staticmethod
    def _read_env_value(env_path: Path, key: str) -> str | None:
        if not env_path.exists():
            return None
        for line in env_path.read_text().splitlines():
            if line.startswith(f"{key}="):
                return line.split("=", 1)[1]
        return None

    @classmethod
    def _read_env_values(cls, env_path: Path, keys: tuple[str, ...]) -> dict[str, str | None]:
        return {key: cls._read_env_value(env_path, key) for key in keys}

    @staticmethod
    def _write_env_value(env_path: Path, key: str, value: str) -> None:
        env_path.parent.mkdir(parents=True, exist_ok=True)
        lines = env_path.read_text().splitlines() if env_path.exists() else []
        replaced = False
        for index, line in enumerate(lines):
            if line.startswith(f"{key}="):
                lines[index] = f"{key}={value}"
                replaced = True
                break
        if not replaced:
            lines.append(f"{key}={value}")
        env_path.write_text("\n".join(lines) + "\n")

    @classmethod
    def _write_env_values(cls, env_path: Path, values: dict[str, str]) -> None:
        for key, value in values.items():
            cls._write_env_value(env_path, key, value)

    @classmethod
    def _normalize_runtime_env_values(
        cls,
        values: dict[str, str | None],
        current_runtime_context: int | None,
    ) -> dict[str, str]:
        profile_name = values.get("VLLM_RUNTIME_PROFILE")
        normalized_profile: RuntimeProfileName = (
            profile_name if profile_name in RUNTIME_PRESETS else DEFAULT_RUNTIME_PROFILE
        )
        preset = cls._resolve_runtime_profile(normalized_profile)
        context_value = _read_int_env(values.get("MAX_MODEL_LEN"), current_runtime_context or 8192)
        resolved = preset.env_values(context_value)
        for key, raw in values.items():
            if raw is None:
                continue
            resolved[key] = raw
        resolved["VLLM_RUNTIME_PROFILE"] = normalized_profile
        resolved["MAX_MODEL_LEN"] = str(context_value)
        return resolved


def _is_float(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _positive_int(value: object) -> int | None:
    return value if isinstance(value, int) and value > 0 else None


def _read_int_env(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(float(value))
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _read_float_env(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _format_float_env(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.2f}".rstrip("0").rstrip(".")


def _dtype_size_bytes(dtype: object) -> int:
    if not isinstance(dtype, str):
        return 2
    normalized = dtype.lower()
    if normalized in {"float32", "fp32"}:
        return 4
    if normalized in {"float16", "fp16", "bfloat16", "bf16"}:
        return 2
    if "float8" in normalized or normalized in {"fp8"}:
        return 1
    return 2


def _format_exception(exc: Exception) -> str:
    if isinstance(exc, subprocess.CalledProcessError):
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        detail = stderr or stdout or str(exc)
        return detail
    return str(exc)
