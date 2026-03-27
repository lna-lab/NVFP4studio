from app.telemetry.prometheus import parse_metrics


def test_parse_metrics_extracts_runtime_advisory_fields():
    payload = """
# HELP vllm:kv_cache_usage_perc KV cache usage
vllm:kv_cache_usage_perc{engine="0"} 0.42
vllm:num_requests_running{engine="0"} 1
vllm:engine_sleep_state{engine="0",sleep_state="weights_offloaded"} 1
vllm:cache_config_info{engine="0",block_size="1056",num_gpu_blocks="2721",gpu_memory_utilization="0.85",cpu_offload_gb="16",swap_space="8"} 1
"""

    values = parse_metrics(payload)

    assert values["vllm:kv_cache_usage_perc"] == 0.42
    assert values["vllm:num_requests_running"] == 1.0
    assert values["vllm:engine_sleep_state:weights_offloaded"] == 1.0
    assert values["vllm:cache_config_info:block_size"] == 1056.0
    assert values["vllm:cache_config_info:num_gpu_blocks"] == 2721.0
    assert values["vllm:cache_config_info:gpu_memory_utilization"] == 0.85
    assert values["vllm:cache_config_info:cpu_offload_gb"] == 16.0
    assert values["vllm:cache_config_info:swap_space"] == 8.0
