from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any


def utcnow() -> datetime:
    return datetime.now(UTC)


def isoformat_or_none(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def _safe_rate(tokens: int | None, started_at: datetime | None, finished_at: datetime | None) -> float | None:
    if tokens is None or started_at is None or finished_at is None:
        return None
    seconds = (finished_at - started_at).total_seconds()
    if seconds <= 0:
        return None
    return round(tokens / seconds, 4)


def _safe_ms(started_at: datetime | None, finished_at: datetime | None) -> float | None:
    if started_at is None or finished_at is None:
        return None
    return round((finished_at - started_at).total_seconds() * 1000, 2)


@dataclass(slots=True)
class BenchmarkResult:
    request_id: str
    upstream_request_id: str | None
    model_name: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    started_at: str
    first_token_at: str | None
    finished_at: str
    ttft_ms: float | None
    e2e_latency_ms: float | None
    completion_tokens_per_sec: float | None
    total_tokens_per_sec: float | None
    streaming: bool
    temperature: float | None
    top_p: float | None
    max_tokens: int | None
    finish_reason: str | None
    error_message: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_benchmark_result(
    *,
    request_id: str,
    upstream_request_id: str | None,
    model_name: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    total_tokens: int | None,
    started_at: datetime,
    first_token_at: datetime | None,
    finished_at: datetime,
    streaming: bool,
    temperature: float | None,
    top_p: float | None,
    max_tokens: int | None,
    finish_reason: str | None,
    error_message: str | None,
) -> BenchmarkResult:
    return BenchmarkResult(
        request_id=request_id,
        upstream_request_id=upstream_request_id,
        model_name=model_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        started_at=isoformat_or_none(started_at) or "",
        first_token_at=isoformat_or_none(first_token_at),
        finished_at=isoformat_or_none(finished_at) or "",
        ttft_ms=_safe_ms(started_at, first_token_at),
        e2e_latency_ms=_safe_ms(started_at, finished_at),
        completion_tokens_per_sec=_safe_rate(completion_tokens, first_token_at, finished_at),
        total_tokens_per_sec=_safe_rate(total_tokens, started_at, finished_at),
        streaming=streaming,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        finish_reason=finish_reason,
        error_message=error_message,
    )


def extract_delta_text(chunk: dict[str, Any]) -> str:
    choices = chunk.get("choices") or []
    if not choices:
        return ""

    delta = choices[0].get("delta") or {}
    content = delta.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            piece.get("text", "")
            for piece in content
            if isinstance(piece, dict) and piece.get("type") == "text"
        )
    reasoning = delta.get("reasoning_content")
    if isinstance(reasoning, str):
        return reasoning
    return ""


def extract_finish_reason(chunk: dict[str, Any]) -> str | None:
    choices = chunk.get("choices") or []
    if not choices:
        return None
    return choices[0].get("finish_reason")

