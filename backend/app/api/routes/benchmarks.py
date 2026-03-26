from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, PlainTextResponse

from app.db.repository import BenchmarkRepository
from app.models.schemas import BenchmarkListResponse, BenchmarkRecord


router = APIRouter()


def get_repository() -> BenchmarkRepository:
    from app.main import get_repository as main_get_repository

    return main_get_repository()


@router.get("/api/benchmarks/recent", response_model=BenchmarkListResponse)
async def recent_benchmarks(limit: int = Query(default=25, ge=1, le=200)) -> BenchmarkListResponse:
    items = get_repository().recent(limit=limit)
    return BenchmarkListResponse(items=items)


@router.get("/api/benchmarks/request/{request_id}", response_model=BenchmarkRecord)
async def benchmark_by_request(request_id: str) -> BenchmarkRecord:
    record = get_repository().get_by_request_id(request_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Benchmark not found")
    return record


@router.get("/api/benchmarks/export")
async def export_benchmarks(format: str = Query(default="json")):
    repository = get_repository()
    if format == "json":
        return JSONResponse(content=repository.export_json())
    if format == "csv":
        content = repository.export_csv()
        return PlainTextResponse(
            content=content,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=benchmarks.csv"},
        )
    raise HTTPException(status_code=400, detail="format must be json or csv")


@router.get("/api/benchmarks/{benchmark_id}", response_model=BenchmarkRecord)
async def benchmark_detail(benchmark_id: int) -> BenchmarkRecord:
    record = get_repository().get(benchmark_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Benchmark not found")
    return record
