from __future__ import annotations

import json
from pathlib import Path

from app.core.config import Settings
from app.db.repository import BenchmarkRepository
from app.models.schemas import GPUStat, MetricSnapshot, RuntimeAdvisory, SystemStatusResponse
from app.services.system_service import SystemService


class DummyVllmClient:
    pass


def test_runtime_context_validation_prefers_live_cache_capacity(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "dtype": "bfloat16",
                "text_config": {
                    "dtype": "bfloat16",
                    "max_position_embeddings": 262144,
                    "num_hidden_layers": 40,
                    "num_key_value_heads": 2,
                    "head_dim": 256,
                    "layer_types": [
                        "linear_attention",
                        "linear_attention",
                        "linear_attention",
                        "full_attention",
                    ]
                    * 10,
                },
            }
        )
    )
    (model_dir / "tokenizer_config.json").write_text(json.dumps({"model_max_length": 262144}))

    settings = Settings(
        model_path=str(model_dir),
        served_model_name="test-model",
        vllm_base_url="http://127.0.0.1:8010",
        vllm_port=8010,
        gateway_port=8000,
        web_port=3000,
        openai_api_key="local-dev-only",
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
        default_max_tokens=2048,
        default_temperature=0.7,
        default_top_p=0.95,
        enable_vllm_metrics=True,
        web_origin="http://localhost:3000",
        log_level="INFO",
        request_timeout_seconds=30,
        bind_localhost_only=True,
        project_root=str(tmp_path),
        project_env_file=str(tmp_path / ".env"),
        compose_file_path=str(tmp_path / "docker-compose.yml"),
        compose_project_name="nvfp4studio_local_test",
        runtime_apply_timeout_seconds=60,
    )
    service = SystemService(settings, BenchmarkRepository(tmp_path / "repo.db"), DummyVllmClient())

    status = SystemStatusResponse(
        gateway_ok=True,
        vllm_healthy=True,
        model_path_exists=True,
        model_path=str(model_dir),
        served_model_name="test-model",
        database_path=str(tmp_path / "repo.db"),
        metrics_available=True,
        gpu=[
            GPUStat(
                name="Test GPU",
                memory_total_mb=97887,
                memory_used_mb=86063,
                memory_free_mb=11824,
                utilization_gpu_percent=0,
                power_draw_watts=50.0,
                power_limit_watts=600.0,
            )
        ],
        metrics=MetricSnapshot(
            values={
                "vllm:cache_config_info:block_size": 1056.0,
                "vllm:cache_config_info:num_gpu_blocks": 2721.0,
                "vllm:kv_cache_usage_perc": 0.0,
                "vllm:num_requests_running": 0.0,
            }
        ),
        recent_benchmark_count=0,
        advisory=RuntimeAdvisory(
            runtime_max_context=8192,
            model_native_context=262144,
            recommended_context=8192,
            hard_context_limit=8192,
            kv_cache_usage_percent=0.0,
            cpu_offload_detected=False,
            fits_in_vram=True,
            risk_level="ok",
            message="ok",
        ),
    )

    validation = service._validate_runtime_context_change(status, 262144)

    assert validation.fits_in_vram is True
    assert validation.estimated_required_vram_mb is not None
    assert validation.estimated_required_vram_mb < 100000
    assert "KV cache" in validation.message


def test_runtime_env_config_uses_selected_profile_defaults(tmp_path: Path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "VLLM_RUNTIME_PROFILE=memory",
                "MAX_MODEL_LEN=16384",
            ]
        )
        + "\n"
    )

    runtime_config = SystemService._read_runtime_env_config(env_path)

    assert runtime_config.profile == "memory"
    assert runtime_config.max_model_len == 16384
    assert runtime_config.gpu_memory_utilization == 0.45
    assert runtime_config.max_num_seqs == 1
    assert runtime_config.kv_cache_dtype == "fp8"
    assert runtime_config.kv_cache_memory_bytes == "8G"
    assert runtime_config.cpu_offload_gb == 0


def test_runtime_env_config_keeps_kv_budget_auto_when_env_is_blank(tmp_path: Path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "VLLM_RUNTIME_PROFILE=memory",
                "MAX_MODEL_LEN=262144",
                "KV_CACHE_MEMORY_BYTES=",
            ]
        )
        + "\n"
    )

    runtime_config = SystemService._read_runtime_env_config(env_path)

    assert runtime_config.profile == "memory"
    assert runtime_config.kv_cache_memory_bytes is None
