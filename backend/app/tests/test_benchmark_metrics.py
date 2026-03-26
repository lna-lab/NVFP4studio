from datetime import UTC, datetime, timedelta

from app.benchmark.metrics import build_benchmark_result, extract_delta_text


def test_build_benchmark_result_rates():
    started_at = datetime(2026, 1, 1, tzinfo=UTC)
    first_token_at = started_at + timedelta(milliseconds=500)
    finished_at = started_at + timedelta(seconds=5)

    result = build_benchmark_result(
        request_id="req_test",
        upstream_request_id="chatcmpl_test",
        model_name="demo-model",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        started_at=started_at,
        first_token_at=first_token_at,
        finished_at=finished_at,
        streaming=True,
        temperature=0.7,
        top_p=0.95,
        max_tokens=256,
        finish_reason="stop",
        error_message=None,
    )

    assert result.ttft_ms == 500.0
    assert result.e2e_latency_ms == 5000.0
    assert result.completion_tokens_per_sec is not None
    assert result.total_tokens_per_sec is not None


def test_extract_delta_text_string():
    chunk = {"choices": [{"delta": {"content": "こんにちは"}}]}
    assert extract_delta_text(chunk) == "こんにちは"

