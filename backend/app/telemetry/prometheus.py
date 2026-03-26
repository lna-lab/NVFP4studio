from __future__ import annotations


TARGET_METRICS = {
    "vllm:time_to_first_token_seconds",
    "vllm:inter_token_latency_seconds",
    "vllm:e2e_request_latency_seconds",
    "vllm:request_prompt_tokens",
    "vllm:request_generation_tokens",
}


def parse_metrics(text: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        metric_name = line.split("{", 1)[0].split(" ", 1)[0]
        if metric_name not in TARGET_METRICS:
            continue
        try:
            values[metric_name] = float(line.rsplit(" ", 1)[-1])
        except ValueError:
            continue
    return values

