# Parallel Request Probe

- generated_at: `2026-03-29T00:15:05.028050+00:00`
- runtime: `16K / 8G / TP4 / non-eager`
- model_name: `huihui-ai/Huihui-Qwen3.5-397B-A17B-abliterated-NVFP4`

## Summary

| concurrency | wall s | aggregate tok/s | avg total W | peak total W | peak single GPU W | avg util % | all passed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 6.937 | 73.8037 | 1029.35 | 1070.71 | 278.42 | 98.53 | PASS |
| 2 | 9.066 | 112.9441 | 903.25 | 1006.32 | 268.66 | 87.05 | PASS |
| 3 | 13.321 | 115.3108 | 992.95 | 1078.53 | 282.0 | 94.19 | PASS |

## Interpretation

- `3` concurrent requests delivered the highest raw aggregate throughput.
- `2` concurrent requests delivered almost all of that gain with lower average total GPU power, so it was the best efficiency point in this run.
- `tok/s per watt` was about `0.0717` at `1`, `0.1250` at `2`, and `0.1161` at `3`.

## Request Details

### Concurrency 1

| prompt | elapsed ms | completion tokens | pass |
| --- | ---: | ---: | --- |
| `nvfp4_vs_gguf` | 6935.99 | 512 | PASS |

### Concurrency 2

| prompt | elapsed ms | completion tokens | pass |
| --- | ---: | ---: | --- |
| `nvfp4_vs_gguf` | 9065.77 | 512 | PASS |
| `parallel_power` | 9065.77 | 512 | PASS |

### Concurrency 3

| prompt | elapsed ms | completion tokens | pass |
| --- | ---: | ---: | --- |
| `nvfp4_vs_gguf` | 10023.89 | 512 | PASS |
| `parallel_power` | 13319.87 | 512 | PASS |
| `throughput_tradeoff` | 13318.19 | 512 | PASS |
