from __future__ import annotations

import asyncio
import json
import logging
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Body, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from app.benchmark.metrics import build_benchmark_result, extract_delta_text, extract_finish_reason, utcnow
from app.core.config import get_settings
from app.db.repository import BenchmarkRepository
from app.services.gpu_monitor import GPUPeakSampler
from app.services.vllm_client import VllmClient


router = APIRouter()
logger = logging.getLogger(__name__)


def get_repository() -> BenchmarkRepository:
    from app.main import get_repository as main_get_repository

    return main_get_repository()


def get_vllm_client() -> VllmClient:
    from app.main import get_vllm_client as main_get_vllm_client

    return main_get_vllm_client()


def _ensure_api_key(authorization: str | None) -> None:
    settings = get_settings()
    if not authorization:
        return
    token = authorization.removeprefix("Bearer").strip()
    if token != settings.openai_api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _prepare_payload(payload: dict[str, Any]) -> dict[str, Any]:
    settings = get_settings()
    prepared = dict(payload)
    prepared.setdefault("model", settings.served_model_name)
    prepared.setdefault("temperature", settings.default_temperature)
    prepared.setdefault("top_p", settings.default_top_p)
    prepared.setdefault("max_tokens", settings.default_max_tokens)
    chat_template_kwargs = dict(prepared.get("chat_template_kwargs") or {})
    chat_template_kwargs.setdefault("enable_thinking", False)
    prepared["chat_template_kwargs"] = chat_template_kwargs

    if prepared.get("stream"):
        stream_options = dict(prepared.get("stream_options") or {})
        stream_options.setdefault("include_usage", True)
        prepared["stream_options"] = stream_options

    return prepared


@router.get("/v1/models")
async def list_models(authorization: str | None = Header(default=None)) -> Response:
    _ensure_api_key(authorization)
    upstream = await get_vllm_client().list_models()
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
    )


@router.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    payload: dict[str, Any] = Body(...),
    authorization: str | None = Header(default=None),
) -> Response:
    _ensure_api_key(authorization)

    repository = get_repository()
    vllm_client = get_vllm_client()
    prepared_payload = _prepare_payload(payload)
    request_id = f"req_{uuid4().hex}"
    started_at = utcnow()
    streaming = bool(prepared_payload.get("stream", False))
    model_name = str(prepared_payload.get("model"))
    temperature = prepared_payload.get("temperature")
    top_p = prepared_payload.get("top_p")
    max_tokens = prepared_payload.get("max_tokens")
    gpu_sampler = GPUPeakSampler()
    await gpu_sampler.start()

    if not streaming:
        try:
            upstream = await vllm_client.create_chat_completion(prepared_payload)
            finished_at = utcnow()
            try:
                body = upstream.json()
            except json.JSONDecodeError:
                body = {"raw": upstream.text}
            usage = body.get("usage") or {}
            gpu_peak = await gpu_sampler.stop()
            benchmark = build_benchmark_result(
                request_id=request_id,
                upstream_request_id=body.get("id"),
                model_name=body.get("model", model_name),
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
                started_at=started_at,
                first_token_at=None,
                finished_at=finished_at,
                streaming=False,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                peak_power_watts=gpu_peak.peak_power_watts,
                peak_vram_used_mb=gpu_peak.peak_vram_used_mb,
                power_limit_watts=gpu_peak.power_limit_watts,
                finish_reason=((body.get("choices") or [{}])[0]).get("finish_reason"),
                error_message=None if upstream.is_success else upstream.text,
            )
            record = await asyncio.to_thread(repository.insert, benchmark)
            return JSONResponse(
                content=body,
                status_code=upstream.status_code,
                headers={
                    "x-nvfp4studio-request-id": request_id,
                    "x-nvfp4studio-benchmark-id": str(record.id),
                },
            )
        finally:
            await gpu_sampler.stop()

    try:
        upstream = await vllm_client.create_chat_completion_stream(prepared_payload)
    except Exception:
        await gpu_sampler.stop()
        raise

    if upstream.status_code >= 400:
        finished_at = utcnow()
        body_bytes = await upstream.aread()
        body_text = body_bytes.decode("utf-8", errors="replace")
        gpu_peak = await gpu_sampler.stop()
        benchmark = build_benchmark_result(
            request_id=request_id,
            upstream_request_id=None,
            model_name=model_name,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            started_at=started_at,
            first_token_at=None,
            finished_at=finished_at,
            streaming=True,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            peak_power_watts=gpu_peak.peak_power_watts,
            peak_vram_used_mb=gpu_peak.peak_vram_used_mb,
            power_limit_watts=gpu_peak.power_limit_watts,
            finish_reason=None,
            error_message=body_text,
        )
        await asyncio.to_thread(repository.insert, benchmark)
        return Response(
            content=body_bytes,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/json"),
            headers={"x-nvfp4studio-request-id": request_id},
        )

    async def event_stream():
        first_token_at = None
        finished_at = None
        finish_reason = None
        usage: dict[str, Any] = {}
        upstream_request_id = None
        resolved_model_name = model_name

        try:
            async for line in upstream.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str and data_str != "[DONE]":
                        try:
                            chunk = json.loads(data_str)
                        except json.JSONDecodeError:
                            logger.warning("Failed to decode SSE chunk: %s", data_str[:200])
                        else:
                            upstream_request_id = upstream_request_id or chunk.get("id")
                            resolved_model_name = chunk.get("model", resolved_model_name)
                            usage = chunk.get("usage") or usage
                            if first_token_at is None and (
                                extract_delta_text(chunk) or chunk.get("choices")
                            ):
                                first_token_at = utcnow()
                            finish_reason = extract_finish_reason(chunk) or finish_reason
                yield (line + "\n").encode("utf-8")

            finished_at = utcnow()
            gpu_peak = await gpu_sampler.stop()
            benchmark = build_benchmark_result(
                request_id=request_id,
                upstream_request_id=upstream_request_id,
                model_name=resolved_model_name,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
                started_at=started_at,
                first_token_at=first_token_at,
                finished_at=finished_at,
                streaming=True,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                peak_power_watts=gpu_peak.peak_power_watts,
                peak_vram_used_mb=gpu_peak.peak_vram_used_mb,
                power_limit_watts=gpu_peak.power_limit_watts,
                finish_reason=finish_reason,
                error_message=None,
            )
            await asyncio.to_thread(repository.insert, benchmark)
        except Exception as exc:
            finished_at = utcnow()
            gpu_peak = await gpu_sampler.stop()
            benchmark = build_benchmark_result(
                request_id=request_id,
                upstream_request_id=upstream_request_id,
                model_name=resolved_model_name,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
                started_at=started_at,
                first_token_at=first_token_at,
                finished_at=finished_at,
                streaming=True,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                peak_power_watts=gpu_peak.peak_power_watts,
                peak_vram_used_mb=gpu_peak.peak_vram_used_mb,
                power_limit_watts=gpu_peak.power_limit_watts,
                finish_reason=finish_reason,
                error_message=str(exc),
            )
            await asyncio.to_thread(repository.insert, benchmark)
            raise
        finally:
            await gpu_sampler.stop()
            await upstream.aclose()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "x-nvfp4studio-request-id": request_id,
        },
    )
