from __future__ import annotations


TARGET_METRICS = {
    "vllm:time_to_first_token_seconds",
    "vllm:inter_token_latency_seconds",
    "vllm:e2e_request_latency_seconds",
    "vllm:request_prompt_tokens",
    "vllm:request_generation_tokens",
    "vllm:kv_cache_usage_perc",
    "vllm:num_requests_running",
}


def parse_metrics(text: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        metric_name = line.split("{", 1)[0].split(" ", 1)[0]
        if metric_name == "vllm:engine_sleep_state" and 'sleep_state="weights_offloaded"' in line:
            try:
                values["vllm:engine_sleep_state:weights_offloaded"] = float(line.rsplit(" ", 1)[-1])
            except ValueError:
                pass
            continue
        if metric_name == "vllm:cache_config_info":
            label_blob = _extract_label_blob(line)
            for label_name in ("block_size", "num_gpu_blocks", "gpu_memory_utilization", "cpu_offload_gb", "swap_space"):
                label_value = label_blob.get(label_name)
                if label_value is None:
                    continue
                try:
                    values[f"vllm:cache_config_info:{label_name}"] = float(label_value)
                except ValueError:
                    continue
            continue
        if metric_name not in TARGET_METRICS:
            continue
        try:
            values[metric_name] = float(line.rsplit(" ", 1)[-1])
        except ValueError:
            continue
    return values


def _extract_label_blob(line: str) -> dict[str, str]:
    if "{" not in line or "}" not in line:
        return {}
    raw = line.split("{", 1)[1].rsplit("}", 1)[0]
    labels: dict[str, str] = {}
    for part in raw.split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        labels[key] = value.strip().strip('"')
    return labels
