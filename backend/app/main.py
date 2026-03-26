from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.benchmarks import router as benchmark_router
from app.api.routes.openai import router as openai_router
from app.api.routes.system import router as system_router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.repository import BenchmarkRepository
from app.services.vllm_client import VllmClient


repository: BenchmarkRepository | None = None
vllm_client: VllmClient | None = None


def get_repository() -> BenchmarkRepository:
    if repository is None:
        raise RuntimeError("Repository is not initialized")
    return repository


def get_vllm_client() -> VllmClient:
    if vllm_client is None:
        raise RuntimeError("vLLM client is not initialized")
    return vllm_client


@asynccontextmanager
async def lifespan(_: FastAPI):
    global repository
    global vllm_client

    settings = get_settings()
    configure_logging(settings.log_level)
    repository = BenchmarkRepository(settings.database_path)
    vllm_client = VllmClient(settings)
    try:
        yield
    finally:
        if vllm_client is not None:
            await vllm_client.close()


settings = get_settings()
app = FastAPI(
    title="NVFP4studio Gateway",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins + ["http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["x-nvfp4studio-request-id", "x-nvfp4studio-benchmark-id"],
)

app.include_router(system_router)
app.include_router(benchmark_router)
app.include_router(openai_router)

