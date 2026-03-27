from __future__ import annotations

from typing import Literal
from typing import Any

from pydantic import BaseModel, Field

RuntimeProfileName = Literal["speed", "balanced", "memory"]


class GPUStat(BaseModel):
    name: str
    memory_total_mb: int | None = None
    memory_used_mb: int | None = None
    memory_free_mb: int | None = None
    utilization_gpu_percent: int | None = None
    power_draw_watts: float | None = None
    power_limit_watts: float | None = None


class RuntimeAdvisory(BaseModel):
    runtime_max_context: int | None = None
    model_native_context: int | None = None
    runtime_profile: RuntimeProfileName = "speed"
    gpu_memory_utilization: float | None = None
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    kv_cache_dtype: str | None = None
    kv_cache_memory_bytes: str | None = None
    cpu_offload_gb: float | None = None
    swap_space_gb: float | None = None
    recommended_context: int | None = None
    hard_context_limit: int | None = None
    reserved_kv_capacity_tokens: int | None = None
    kv_cache_usage_percent: float | None = None
    cpu_offload_detected: bool = False
    fits_in_vram: bool | None = None
    risk_level: str = "unknown"
    message: str = "ランタイム診断を計算できませんでした。"


class RuntimeConfigApplyRequest(BaseModel):
    max_model_len: int = Field(ge=2048)
    runtime_profile: RuntimeProfileName = "speed"


class RuntimeConfigValidation(BaseModel):
    requested_context: int
    current_runtime_context: int | None = None
    model_native_context: int | None = None
    current_vram_used_mb: int | None = None
    total_vram_mb: int | None = None
    estimated_required_vram_mb: int | None = None
    fits_in_vram: bool | None = None
    risk_level: str = "unknown"
    message: str


class RuntimeConfigApplyResponse(BaseModel):
    accepted: bool
    restarted: bool
    previous_runtime_context: int | None = None
    applied_runtime_context: int | None = None
    previous_runtime_profile: RuntimeProfileName | None = None
    applied_runtime_profile: RuntimeProfileName | None = None
    validation: RuntimeConfigValidation
    message: str


class MetricSnapshot(BaseModel):
    values: dict[str, float] = Field(default_factory=dict)


class SystemStatusResponse(BaseModel):
    gateway_ok: bool
    vllm_healthy: bool
    model_path_exists: bool
    model_path: str
    served_model_name: str
    database_path: str
    metrics_available: bool
    gpu: list[GPUStat] = Field(default_factory=list)
    metrics: MetricSnapshot
    recent_benchmark_count: int
    advisory: RuntimeAdvisory


class SystemConfigResponse(BaseModel):
    model_path: str
    served_model_name: str
    vllm_base_url: str
    gateway_port: int
    web_port: int
    database_url: str
    default_max_tokens: int
    default_temperature: float
    default_top_p: float
    enable_vllm_metrics: bool
    bind_localhost_only: bool
    web_origin: list[str]
    openai_api_key_hint: str


class BenchmarkRecord(BaseModel):
    id: int
    request_id: str
    upstream_request_id: str | None = None
    model_name: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    started_at: str
    first_token_at: str | None = None
    finished_at: str
    ttft_ms: float | None = None
    e2e_latency_ms: float | None = None
    completion_tokens_per_sec: float | None = None
    total_tokens_per_sec: float | None = None
    streaming: bool
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    peak_power_watts: float | None = None
    peak_vram_used_mb: int | None = None
    power_limit_watts: float | None = None
    finish_reason: str | None = None
    error_message: str | None = None
    created_at: str


class BenchmarkListResponse(BaseModel):
    items: list[BenchmarkRecord]


class HealthResponse(BaseModel):
    status: str
    gateway_ok: bool
    vllm_healthy: bool


class ErrorResponse(BaseModel):
    detail: str
    request_id: str | None = None
    upstream_status_code: int | None = None
    upstream_body: Any | None = None
