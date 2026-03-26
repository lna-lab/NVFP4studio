# Benchmark Methodology

## Request-Level Metrics

UI に表示する benchmark 値は vLLM の集計値ではなく、Gateway が request 単位で計測した実測値を主に使います。

### TTFT

- `started_at`: Gateway が upstream request を投げた時刻
- `first_token_at`: streaming の最初の token chunk を受信した時刻
- `ttft_ms = first_token_at - started_at`

### Completion token/s

- `completion_tokens / (finished_at - first_token_at)`
- 0 除算は回避

### Total token/s

- `total_tokens / (finished_at - started_at)`

### End-to-end latency

- `finished_at - started_at`

## Usage Tokens

- non-stream: upstream JSON body の `usage`
- stream: `stream_options.include_usage=true` を付け、最終 usage chunk を利用

usage が返らない場合は token 数が `null` になり、throughput も `null` になります。

## Server-Level Metrics

参考値として `/metrics` から次を取得します。

- `vllm:time_to_first_token_seconds`
- `vllm:inter_token_latency_seconds`
- `vllm:e2e_request_latency_seconds`
- `vllm:request_prompt_tokens`
- `vllm:request_generation_tokens`

これらは UI の Model Status パネルに表示し、会話 bubble には request-level 実測値を使います。

