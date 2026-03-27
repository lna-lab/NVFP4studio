from __future__ import annotations

import csv
import io
import sqlite3
from pathlib import Path
from typing import Any

from app.benchmark.metrics import BenchmarkResult
from app.db.database import create_connection, initialize_database
from app.models.schemas import BenchmarkRecord


class BenchmarkRepository:
    def __init__(self, database_path: Path) -> None:
        self._connection = create_connection(database_path)
        initialize_database(self._connection)

    def insert(self, result: BenchmarkResult) -> BenchmarkRecord:
        payload = result.to_dict()
        self._connection.execute(
            """
            INSERT OR REPLACE INTO benchmarks (
                request_id,
                upstream_request_id,
                model_name,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                started_at,
                first_token_at,
                finished_at,
                ttft_ms,
                e2e_latency_ms,
                completion_tokens_per_sec,
                total_tokens_per_sec,
                streaming,
                temperature,
                top_p,
                max_tokens,
                peak_power_watts,
                peak_vram_used_mb,
                power_limit_watts,
                finish_reason,
                error_message
            ) VALUES (
                :request_id,
                :upstream_request_id,
                :model_name,
                :prompt_tokens,
                :completion_tokens,
                :total_tokens,
                :started_at,
                :first_token_at,
                :finished_at,
                :ttft_ms,
                :e2e_latency_ms,
                :completion_tokens_per_sec,
                :total_tokens_per_sec,
                :streaming,
                :temperature,
                :top_p,
                :max_tokens,
                :peak_power_watts,
                :peak_vram_used_mb,
                :power_limit_watts,
                :finish_reason,
                :error_message
            )
            """,
            {**payload, "streaming": int(payload["streaming"])},
        )
        self._connection.commit()
        cursor = self._connection.execute(
            "SELECT * FROM benchmarks WHERE request_id = ?",
            (result.request_id,),
        )
        return self._row_to_record(cursor.fetchone())

    def recent(self, limit: int = 25) -> list[BenchmarkRecord]:
        cursor = self._connection.execute(
            "SELECT * FROM benchmarks ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return [self._row_to_record(row) for row in cursor.fetchall()]

    def get(self, benchmark_id: int) -> BenchmarkRecord | None:
        cursor = self._connection.execute(
            "SELECT * FROM benchmarks WHERE id = ?",
            (benchmark_id,),
        )
        row = cursor.fetchone()
        return self._row_to_record(row) if row else None

    def get_by_request_id(self, request_id: str) -> BenchmarkRecord | None:
        cursor = self._connection.execute(
            "SELECT * FROM benchmarks WHERE request_id = ?",
            (request_id,),
        )
        row = cursor.fetchone()
        return self._row_to_record(row) if row else None

    def count(self) -> int:
        cursor = self._connection.execute("SELECT COUNT(*) AS value FROM benchmarks")
        return int(cursor.fetchone()["value"])

    def export_json(self) -> list[dict[str, Any]]:
        return [record.model_dump() for record in self.recent(limit=1000)]

    def export_csv(self) -> str:
        records = self.export_json()
        if not records:
            return ""
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
        return output.getvalue()

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> BenchmarkRecord:
        payload = dict(row)
        payload["streaming"] = bool(payload["streaming"])
        return BenchmarkRecord(**payload)
