from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.core.config import Settings, get_settings
from app.models.schemas import (
    HealthResponse,
    RuntimeConfigApplyRequest,
    RuntimeConfigApplyResponse,
    SystemConfigResponse,
    SystemStatusResponse,
)
from app.services.system_service import SystemService
from app.services.vllm_client import VllmClient


router = APIRouter()


def get_system_service() -> SystemService:
    from app.main import get_repository, get_vllm_client

    settings = get_settings()
    repository = get_repository()
    vllm_client = get_vllm_client()
    return SystemService(settings, repository, vllm_client)


@router.get("/health", response_model=HealthResponse)
async def health(service: SystemService = Depends(get_system_service)) -> HealthResponse:
    status = await service.get_status()
    return HealthResponse(
        status="ok" if status.vllm_healthy else "degraded",
        gateway_ok=True,
        vllm_healthy=status.vllm_healthy,
    )


@router.get("/api/system/status", response_model=SystemStatusResponse)
async def system_status(service: SystemService = Depends(get_system_service)) -> SystemStatusResponse:
    return await service.get_status()


@router.get("/api/system/config", response_model=SystemConfigResponse)
async def system_config(service: SystemService = Depends(get_system_service)) -> SystemConfigResponse:
    return service.get_config()


@router.post("/api/system/runtime-config", response_model=RuntimeConfigApplyResponse)
async def apply_runtime_config(
    payload: RuntimeConfigApplyRequest,
    service: SystemService = Depends(get_system_service),
) -> RuntimeConfigApplyResponse:
    result = await service.apply_runtime_config(payload)
    if not result.accepted:
        raise HTTPException(status_code=400, detail=result.message)
    return result
