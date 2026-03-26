from __future__ import annotations

import sqlite3
from pathlib import Path


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS benchmarks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL UNIQUE,
    upstream_request_id TEXT,
    model_name TEXT NOT NULL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    started_at TEXT NOT NULL,
    first_token_at TEXT,
    finished_at TEXT NOT NULL,
    ttft_ms REAL,
    e2e_latency_ms REAL,
    completion_tokens_per_sec REAL,
    total_tokens_per_sec REAL,
    streaming INTEGER NOT NULL,
    temperature REAL,
    top_p REAL,
    max_tokens INTEGER,
    finish_reason TEXT,
    error_message TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


def create_connection(database_path: Path) -> sqlite3.Connection:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database(connection: sqlite3.Connection) -> None:
    connection.executescript(SCHEMA_SQL)
    connection.commit()

