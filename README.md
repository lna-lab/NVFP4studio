# NVFP4studio

Local vLLM + FastAPI + Next.js studio for NVFP4 models, with OpenAI-compatible chat APIs, runtime profile control, and benchmark tracking.

Language: [English](#english) | [中文](#中文) | [日本語](#日本語)

## Status

- MVP is working on the current target environment.
- Verified runtime target: Ubuntu 24.04 + NVIDIA GPU + Docker + NVIDIA Container Toolkit.
- UI, Gateway, and vLLM are running together via Docker Compose.
- English, Chinese, and Japanese documentation are included.
- Runtime profiles, GPU telemetry, KV cache budget control, and benchmark export are included.
- The current optimization focus especially targets `RTX PRO 6000 Blackwell` and `RTX 5090` users who want long-context NVFP4 serving with tighter VRAM budgets.

## Verified Model

- Primary Qwen3.5 targets:
  `Qwen3.5-27B`, `Qwen3.5-35B-A3B`, `Qwen3.5-4B`
- Example model path:
  `/absolute/path/to/your/nvfp4-model`
- Example served model name:
  `your-nvfp4-model`
- Public NVFP4 variants are available on Hugging Face and can be downloaded without purchase:
  [Qwen3.5-27B-NVFP4](https://huggingface.co/apolo13x/Qwen3.5-27B-NVFP4),
  [Qwen3.5-35B-A3B-NVFP4](https://huggingface.co/apolo13x/Qwen3.5-35B-A3B-NVFP4),
  [Qwen3.5-4B quantized models](https://huggingface.co/models?other=base_model%3Aquantized%3AQwen%2FQwen3.5-4B&p=1&sort=trending)

## Verified Runtime Findings

- `speed + -c 256K` used about `90.3GB` VRAM.
- `balanced + -c 256K` used about `65.3GB` VRAM.
- `memory + -c 256K + KV budget 4G` reached about `34.2GB` peak VRAM with quality canary `3/3`.
- `memory + -c 256K + KV budget 3G` reduced VRAM further, but quality canary fell to `2/3`, so it is not the current recommendation.
- `2 instances x 2 seqs` worked as a multi-tenant layout, but did not outperform `1 instance x 2 seqs` in aggregate throughput.
- On `4x RTX PRO 6000 Blackwell`, `huihui-ai/Huihui-Qwen3.5-397B-A17B-abliterated-NVFP4` at `16K / 8G / TP4` moved from about `18.5 tok/s` to about `83 tok/s` by switching to `ENFORCE_EAGER=false`; on this PCIe-only 4-GPU topology, vLLM still auto-falls back to NCCL for all-reduce.
- A more aggressive `speed` profile with higher VRAM reservation reached about `78.3 tok/s`, so the current fast-path recommendation is still the lighter `16K / 8G` setup with non-eager execution.
- `flashinfer_cutedsl + autotune` showed worker initialization instability in this environment, so it is not the current default recommendation.
- On the same `16K / 8G / TP4` runtime, aggregate throughput improved from about `73.8 tok/s` at single-request load to about `112.9 tok/s` at `2` concurrent requests and about `115.3 tok/s` at `3` concurrent requests; the best `tok/s per watt` point was `2` concurrent requests.

---

# English

## Overview

`NVFP4studio` is a local studio for serving a locally stored NVFP4 model through vLLM, exposing an OpenAI-compatible API through FastAPI, and visualizing TTFT, token/s, latency, GPU status, and benchmark history in a browser UI.

## Intended Users

- Especially useful for `RTX PRO 6000 Blackwell` and `RTX 5090` users
- Written for operators who want to keep long context while pushing VRAM use down without giving up acceptable quality

## Primary Qwen3.5 Targets

- `Qwen3.5-27B`
- `Qwen3.5-35B-A3B`
- `Qwen3.5-4B`
- Public NVFP4 variants are available on Hugging Face and can be downloaded without purchase:
  [Qwen3.5-27B-NVFP4](https://huggingface.co/apolo13x/Qwen3.5-27B-NVFP4),
  [Qwen3.5-35B-A3B-NVFP4](https://huggingface.co/apolo13x/Qwen3.5-35B-A3B-NVFP4),
  [Qwen3.5-4B quantized models](https://huggingface.co/models?other=base_model%3Aquantized%3AQwen%2FQwen3.5-4B&p=1&sort=trending)

## What Works

- vLLM serves the local model from an absolute host path
- Gateway exposes:
  `GET /health`
  `GET /api/system/status`
  `GET /api/system/config`
  `POST /api/system/runtime-config`
  `GET /api/benchmarks/recent`
  `GET /api/benchmarks/export?format=json|csv`
  `GET /v1/models`
  `POST /v1/chat/completions`
- Browser UI supports chat, streaming, model status, runtime profile controls, and benchmark history
- SQLite stores per-request benchmark records
- Docker Compose starts the full stack

## Verified Runtime Findings

- The main VRAM reduction lever was not model weights, but `KV cache` reservation.
- Explicit `--kv-cache-memory-bytes` was more effective than only lowering `gpu_memory_utilization`.
- The current best verified single-user point is:
  `memory` profile + `MAX_MODEL_LEN=262144` + `KV_CACHE_MEMORY_BYTES=4G`
- That setting kept quality canary at `3/3` while reducing peak VRAM to about `34.2GB`.
- `3G` reduced VRAM a bit more, but quality dropped, so it is currently rejected.
- `2x2` is useful for tenant isolation and comparison workloads, not as a guaranteed throughput upgrade.
- For `huihui-ai/Huihui-Qwen3.5-397B-A17B-abliterated-NVFP4` on `4x RTX PRO 6000 Blackwell`, the current fastest verified local point is:
  `MAX_MODEL_LEN=16384` + `KV_CACHE_MEMORY_BYTES=8G` + `TENSOR_PARALLEL_SIZE=4` + `ENFORCE_EAGER=false` + `DISABLE_CUSTOM_ALL_REDUCE=false`
- In the recorded local sweep, that path moved long streaming generation from about `18.5 tok/s` to about `81.4-83.0 tok/s`.
- On this PCIe-only 4-GPU setup, vLLM logged that custom all-reduce is not supported and fell back to NCCL automatically, so the dominant win came from disabling eager execution.
- A higher-reservation `speed` profile still worked, but landed at about `78.3 tok/s`, so it is not the current fast default for this model.
- `flashinfer_cutedsl + autotune` was not stable enough to adopt yet in this environment.
- A follow-up non-stream parallel-request probe on the same runtime showed:
  `1 concurrent = 73.8 tok/s`,
  `2 concurrent = 112.9 tok/s`,
  `3 concurrent = 115.3 tok/s`
- The highest raw aggregate throughput was `3 concurrent`, but the best efficiency point was `2 concurrent`, because it used less average total GPU power while delivering most of the gain.

## Verified Local URLs

- UI: `http://127.0.0.1:3000`
- Gateway: `http://127.0.0.1:8000`
- vLLM: `http://127.0.0.1:8010`

## Quick Start

```bash
git clone https://github.com/lna-lab/NVFP4studio.git
cd NVFP4studio
cp .env.example .env
./scripts/bootstrap.sh
./scripts/start.sh
```

Check the stack:

```bash
./scripts/healthcheck.sh
```

Run a smoke benchmark:

```bash
./scripts/benchmark_smoke.sh
```

Stop or restart:

```bash
./scripts/stop.sh
./scripts/restart.sh
```

## Requirements

- Ubuntu 24.04 class Linux environment
- NVIDIA GPU and working host `nvidia-smi`
- Docker
- Docker Compose
- NVIDIA Container Toolkit configured for Docker

## Architecture

- `vllm`: model serving
- `gateway`: FastAPI OpenAI-compatible proxy, runtime advisory, and benchmark recorder
- `frontend`: Next.js chat UI and runtime control panel
- `data/sqlite/nvfp4studio.db`: benchmark database
- `data/logs/`: service logs

## Configuration

Main runtime settings live in `.env`. Important values:

- `MODEL_PATH`
- `SERVED_MODEL_NAME`
- `VLLM_IMAGE`
- `TRANSFORMERS_GIT_REF`
- `MAX_MODEL_LEN`
- `VLLM_RUNTIME_PROFILE`
- `GPU_MEMORY_UTILIZATION`
- `MAX_NUM_SEQS`
- `MAX_NUM_BATCHED_TOKENS`
- `KV_CACHE_DTYPE`
- `KV_CACHE_MEMORY_BYTES`
- `CPU_OFFLOAD_GB`
- `SWAP_SPACE`
- `BIND_LOCALHOST_ONLY`
- `HOST_BIND_IP`

Default serving values start from a conservative compatibility setup:

- `TENSOR_PARALLEL_SIZE=1`
- `MAX_MODEL_LEN=8192`
- `TRUST_REMOTE_CODE=true`
- `GPU_MEMORY_UTILIZATION=0.85`
- `VLLM_RUNTIME_PROFILE=speed`

## Data and Exports

- SQLite DB: `data/sqlite/nvfp4studio.db`
- Logs: `data/logs/`
- Benchmark export directory: `data/exports/`
- Export API:
  `GET /api/benchmarks/export?format=json`
  `GET /api/benchmarks/export?format=csv`

## Known Note

The current verified model may emit reasoning-style text such as `Thinking Process` depending on prompt settings and model behavior. The stack itself is functioning, but prompt/response shaping may still be tuned.

## Documentation

- [docs/requirements.md](docs/requirements.md)
- [docs/architecture.md](docs/architecture.md)
- [docs/runbook.md](docs/runbook.md)
- [scripts/README.md](scripts/README.md)
- [docs/benchmark-methodology.md](docs/benchmark-methodology.md)
- [docs/vllm-optimization-notes.md](docs/vllm-optimization-notes.md)
- [docs/nvfp4-conversion-notes.md](docs/nvfp4-conversion-notes.md)
- [docs/model-research-notes.md](docs/model-research-notes.md)
- [docs/versions.md](docs/versions.md)

---

# 中文

## 概述

`NVFP4studio` 是一个本地化的 NVFP4 推理工作台。它使用 vLLM 从本地模型目录启动服务，通过 FastAPI 提供 OpenAI 兼容 API，并在浏览器 UI 中展示 TTFT、token/s、latency、GPU 状态和 benchmark 历史。

## 目标用户

- 特别面向 `RTX PRO 6000 Blackwell` 和 `RTX 5090` 用户
- 适合希望在保持长上下文的同时尽量压低 VRAM 占用、又不明显牺牲可用质量的使用者

## 主要 Qwen3.5 模型

- `Qwen3.5-27B`
- `Qwen3.5-35B-A3B`
- `Qwen3.5-4B`
- Hugging Face 上已有公开的 NVFP4 版本，用户可无偿获取:
  [Qwen3.5-27B-NVFP4](https://huggingface.co/apolo13x/Qwen3.5-27B-NVFP4),
  [Qwen3.5-35B-A3B-NVFP4](https://huggingface.co/apolo13x/Qwen3.5-35B-A3B-NVFP4),
  [Qwen3.5-4B quantized models](https://huggingface.co/models?other=base_model%3Aquantized%3AQwen%2FQwen3.5-4B&p=1&sort=trending)

## 当前已验证

- 已在当前目标环境完成启动与联调
- 已验证环境为 Ubuntu 24.04 + NVIDIA GPU + Docker + NVIDIA Container Toolkit
- 三个核心服务都可通过 Docker Compose 启动:
  `vllm`
  `gateway`
  `frontend`
- 已验证接口:
  `GET /health`
  `GET /api/system/status`
  `GET /api/system/config`
  `POST /api/system/runtime-config`
  `GET /v1/models`
  `POST /v1/chat/completions`
  `GET /api/benchmarks/recent`
  `GET /api/benchmarks/export?format=json|csv`

## 已验证的 VRAM 结论

- VRAM 差异的主因不是模型权重，而是 `KV cache` 预留量。
- 仅降低 `gpu_memory_utilization` 有帮助，但显式设置 `KV cache budget` 更有效。
- 当前最好的单用户已验证点是:
  `memory` profile + `MAX_MODEL_LEN=262144` + `KV_CACHE_MEMORY_BYTES=4G`
- 这一设置把峰值 VRAM 降到约 `34.2GB`，同时 quality canary 维持在 `3/3`。
- `3G` 虽然还能再省一点 VRAM，但 quality canary 下降到 `2/3`，当前不推荐。
- `2x2` 更适合多租户隔离与对比实验，不应直接理解为吞吐提升方案。
- 对于 `4x RTX PRO 6000 Blackwell` 上的 `huihui-ai/Huihui-Qwen3.5-397B-A17B-abliterated-NVFP4`，当前验证过的最快本地路径是:
  `MAX_MODEL_LEN=16384` + `KV_CACHE_MEMORY_BYTES=8G` + `TENSOR_PARALLEL_SIZE=4` + `ENFORCE_EAGER=false` + `DISABLE_CUSTOM_ALL_REDUCE=false`
- 这一路径把长文本 streaming 速度从约 `18.5 tok/s` 提升到约 `81.4-83.0 tok/s`。
- 在这个 4 卡 PCIe 拓扑下，vLLM 会记录 custom all-reduce 不受支持并自动回退到 NCCL，因此真正起决定作用的是关闭 eager。
- 更激进的 `speed` profile 也能工作，但本轮只有约 `78.3 tok/s`，因此当前不作为这个模型的默认快速设置。
- `flashinfer_cutedsl + autotune` 在当前环境中出现了 worker 初始化不稳定，暂不作为默认推荐。
- 在同一套 `16K / 8G / TP4` runtime 上，后续并行请求测试显示：
  `1 并发 = 73.8 tok/s`，
  `2 并发 = 112.9 tok/s`，
  `3 并发 = 115.3 tok/s`
- 其中 `3 并发` 的 aggregate throughput 最高，但若把吞吐和平均总功耗一起看，`2 并发` 是更均衡的效率点。

## 默认地址

- UI: `http://127.0.0.1:3000`
- Gateway: `http://127.0.0.1:8000`
- vLLM: `http://127.0.0.1:8010`

## 快速开始

```bash
git clone https://github.com/lna-lab/NVFP4studio.git
cd NVFP4studio
cp .env.example .env
./scripts/bootstrap.sh
./scripts/start.sh
```

健康检查:

```bash
./scripts/healthcheck.sh
```

基准测试:

```bash
./scripts/benchmark_smoke.sh
```

停止或重启:

```bash
./scripts/stop.sh
./scripts/restart.sh
```

## 主要功能

- 使用本地绝对路径加载 NVFP4 模型
- 提供 OpenAI 兼容聊天接口
- 支持运行时 profile 切换与上下文重配置
- 记录 TTFT、completion token/s、total token/s、total latency
- 记录 GPU 状态、KV cache 建议值和 benchmark 历史
- 将 benchmark 结果保存到 SQLite
- 支持 JSON 和 CSV 导出

## 主要配置

配置文件为 `.env`，重点参数包括:

- `MODEL_PATH`
- `SERVED_MODEL_NAME`
- `VLLM_IMAGE`
- `TRANSFORMERS_GIT_REF`
- `MAX_MODEL_LEN`
- `VLLM_RUNTIME_PROFILE`
- `GPU_MEMORY_UTILIZATION`
- `MAX_NUM_SEQS`
- `MAX_NUM_BATCHED_TOKENS`
- `KV_CACHE_DTYPE`
- `KV_CACHE_MEMORY_BYTES`
- `CPU_OFFLOAD_GB`
- `SWAP_SPACE`
- `BIND_LOCALHOST_ONLY`

默认推理参数:

- `TENSOR_PARALLEL_SIZE=1`
- `MAX_MODEL_LEN=8192`
- `TRUST_REMOTE_CODE=true`
- `GPU_MEMORY_UTILIZATION=0.85`
- `VLLM_RUNTIME_PROFILE=speed`

## 数据目录

- SQLite: `data/sqlite/nvfp4studio.db`
- 日志: `data/logs/`
- 导出: `data/exports/`

## 已知说明

当前已验证模型有时会输出类似 `Thinking Process` 的推理文本。这是模型行为层面的现象，不影响服务链路本身。

## 文档

- [docs/requirements.md](docs/requirements.md)
- [docs/architecture.md](docs/architecture.md)
- [docs/runbook.md](docs/runbook.md)
- [scripts/README.md](scripts/README.md)
- [docs/benchmark-methodology.md](docs/benchmark-methodology.md)
- [docs/vllm-optimization-notes.md](docs/vllm-optimization-notes.md)
- [docs/nvfp4-conversion-notes.md](docs/nvfp4-conversion-notes.md)
- [docs/model-research-notes.md](docs/model-research-notes.md)
- [docs/versions.md](docs/versions.md)

---

# 日本語

## 概要

`NVFP4studio` は、ローカルに保存した NVFP4 モデルを vLLM で起動し、FastAPI Gateway 経由で OpenAI 互換 API を公開しつつ、ブラウザ UI で TTFT、token/s、latency、GPU 状態、ベンチ履歴を見える化するローカル向けスタジオです。

## 想定ユーザー

- とくに `RTX PRO 6000 Blackwell` と `RTX 5090` のユーザーを強く意識しています
- 長コンテキストを維持しながら、品質を大きく崩さずに VRAM 使用量を詰めたい運用者向けです

## 主役となる Qwen3.5 モデル

- `Qwen3.5-27B`
- `Qwen3.5-35B-A3B`
- `Qwen3.5-4B`
- それぞれの NVFP4 系バリアントは Hugging Face 上で公開入手できます:
  [Qwen3.5-27B-NVFP4](https://huggingface.co/apolo13x/Qwen3.5-27B-NVFP4),
  [Qwen3.5-35B-A3B-NVFP4](https://huggingface.co/apolo13x/Qwen3.5-35B-A3B-NVFP4),
  [Qwen3.5-4B quantized models](https://huggingface.co/models?other=base_model%3Aquantized%3AQwen%2FQwen3.5-4B&p=1&sort=trending)

## 現在確認できていること

- Ubuntu 24.04 系 + NVIDIA GPU + Docker + NVIDIA Container Toolkit 上で起動確認済み
- `vllm`、`gateway`、`frontend` の 3 サービスが Compose で動作
- 次の API を確認済み:
  `GET /health`
  `GET /api/system/status`
  `GET /api/system/config`
  `POST /api/system/runtime-config`
  `GET /v1/models`
  `POST /v1/chat/completions`
  `GET /api/benchmarks/recent`
  `GET /api/benchmarks/export?format=json|csv`
- UI から会話、ベンチ履歴表示、runtime profile 切り替え、export が可能

## 確認できた VRAM 節約の知見

- VRAM 差の主因はモデル重みそのものより `KV cache` の予約量でした。
- `gpu_memory_utilization` を下げるだけでも効きますが、`KV_CACHE_MEMORY_BYTES` を明示した方が効果は大きいです。
- 現時点の単ユーザー暫定ベストは:
  `memory` profile + `MAX_MODEL_LEN=262144` + `KV_CACHE_MEMORY_BYTES=4G`
- この設定では peak VRAM が約 `34.2GB` まで下がり、quality canary も `3/3` を維持できました。
- `3G` まで下げると VRAM はさらに減りましたが、quality canary が `2/3` に落ちたため不採用です。
- `2インスタンス x 2シーケンス` は成立しましたが、総 throughput 強化というより multi-tenant 隔離向きでした。
- `huihui-ai/Huihui-Qwen3.5-397B-A17B-abliterated-NVFP4` を `4x RTX PRO 6000 Blackwell` で動かす場合、現時点の最速確認点は:
  `MAX_MODEL_LEN=16384` + `KV_CACHE_MEMORY_BYTES=8G` + `TENSOR_PARALLEL_SIZE=4` + `ENFORCE_EAGER=false` + `DISABLE_CUSTOM_ALL_REDUCE=false`
- この経路では長文 streaming の completion が約 `18.5 tok/s` から約 `81.4-83.0 tok/s` まで伸びました。
- ただしこの 4GPU PCIe 構成では、vLLM ログ上で custom all-reduce 非対応となり自動で NCCL にフォールバックしているため、実際の主因は `ENFORCE_EAGER=false` 側です。
- VRAM と電力をさらに厚く使う `speed` profile も成立しましたが、今回の sweep では約 `78.3 tok/s` に留まり、最速ではありませんでした。
- `flashinfer_cutedsl + autotune` はこの環境では worker 初期化が不安定で、現時点では既定値にしません。
- 同じ `16K / 8G / TP4` runtime で非 streaming の並列リクエスト probe を回したところ、
  `1 並列 = 73.8 tok/s`、
  `2 並列 = 112.9 tok/s`、
  `3 並列 = 115.3 tok/s`
  でした。
- 生の aggregate throughput だけなら `3 並列` が最高ですが、平均総消費電力まで含めた効率は `2 並列` が最良でした。

## 起動先

- UI: `http://127.0.0.1:3000`
- Gateway: `http://127.0.0.1:8000`
- vLLM: `http://127.0.0.1:8010`

## クイックスタート

```bash
git clone https://github.com/lna-lab/NVFP4studio.git
cd NVFP4studio
cp .env.example .env
./scripts/bootstrap.sh
./scripts/start.sh
```

動作確認:

```bash
./scripts/healthcheck.sh
```

スモークベンチ:

```bash
./scripts/benchmark_smoke.sh
```

停止と再起動:

```bash
./scripts/stop.sh
./scripts/restart.sh
```

## 主な機能

- ローカルモデルパスをそのまま bind mount して vLLM 起動
- OpenAI 互換 API を Gateway から公開
- runtime profile と context を UI/API から切り替え
- streaming 中に TTFT と token/s を計測
- GPU 状態、KV capacity、runtime advisory を取得
- リクエスト単位のベンチ結果を SQLite に保存
- ベンチ履歴の JSON / CSV export

## 主な設定

設定は `.env` で変更できます。よく使う項目:

- `MODEL_PATH`
- `SERVED_MODEL_NAME`
- `VLLM_IMAGE`
- `TRANSFORMERS_GIT_REF`
- `MAX_MODEL_LEN`
- `VLLM_RUNTIME_PROFILE`
- `GPU_MEMORY_UTILIZATION`
- `MAX_NUM_SEQS`
- `MAX_NUM_BATCHED_TOKENS`
- `KV_CACHE_DTYPE`
- `KV_CACHE_MEMORY_BYTES`
- `CPU_OFFLOAD_GB`
- `SWAP_SPACE`
- `BIND_LOCALHOST_ONLY`
- `HOST_BIND_IP`

初期値は次の方針です。

- `TENSOR_PARALLEL_SIZE=1`
- `MAX_MODEL_LEN=8192`
- `TRUST_REMOTE_CODE=true`
- `GPU_MEMORY_UTILIZATION=0.85`
- `VLLM_RUNTIME_PROFILE=speed`

## 保存先

- SQLite: `data/sqlite/nvfp4studio.db`
- ログ: `data/logs/`
- export: `data/exports/`

## 注意点

今回の検証モデルは、プロンプト条件によって `Thinking Process` のような推論文をそのまま出すことがあります。アプリ基盤の動作には問題ありませんが、prompt 設計や後処理にはまだ調整余地があります。

## 関連ドキュメント

- [docs/requirements.md](docs/requirements.md)
- [docs/architecture.md](docs/architecture.md)
- [docs/runbook.md](docs/runbook.md)
- [scripts/README.md](scripts/README.md)
- [docs/benchmark-methodology.md](docs/benchmark-methodology.md)
- [docs/vllm-optimization-notes.md](docs/vllm-optimization-notes.md)
- [docs/nvfp4-conversion-notes.md](docs/nvfp4-conversion-notes.md)
- [docs/model-research-notes.md](docs/model-research-notes.md)
- [docs/versions.md](docs/versions.md)
