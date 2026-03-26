# NVFP4studio

Local vLLM + FastAPI + Next.js studio for NVFP4 models, with OpenAI-compatible chat APIs and benchmark tracking.

Language: [English](#english) | [中文](#中文) | [日本語](#日本語)

## Status

- MVP is working on the current target environment.
- Verified runtime target: Ubuntu 24.04 + NVIDIA GPU + Docker + NVIDIA Container Toolkit.
- UI, Gateway, and vLLM are running together via Docker Compose.
- English, Chinese, and Japanese documentation are included.
- Windows and macOS are planned, but the current verified deployment target is Linux with NVIDIA GPU support.

## Verified Model

- Model path:
  `/media/shinkaman/INTEL_TUF/Sefetensors/nvfp4/Huihui-Qwen3.5-35B-A3B-abliterated-NVFP4`
- Default served model name:
  `Huihui-Qwen3.5-35B-A3B-abliterated-NVFP4`

---

# English

## Overview

`NVFP4studio` is a local studio for serving a locally stored NVFP4 model through vLLM, exposing an OpenAI-compatible API through FastAPI, and visualizing TTFT, token/s, latency, and usage in a browser UI.

## What Works

- vLLM serves the local model from an absolute host path
- Gateway exposes:
  `GET /health`
  `GET /api/system/status`
  `GET /api/benchmarks/recent`
  `GET /api/benchmarks/export?format=json|csv`
  `GET /v1/models`
  `POST /v1/chat/completions`
- Browser UI supports chat, streaming, model status, parameter controls, and benchmark history
- SQLite stores per-request benchmark records
- Docker Compose starts the full stack

## Verified Local URLs

- UI: `http://127.0.0.1:3000`
- Gateway: `http://127.0.0.1:8000`
- vLLM: `http://127.0.0.1:8010`

## Quick Start

```bash
cd /media/shinkaman/INTEL_TUF/Sefetensors/NVFP4studio
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
- `gateway`: FastAPI OpenAI-compatible proxy and benchmark recorder
- `frontend`: Next.js chat UI
- `data/sqlite/nvfp4studio.db`: benchmark database
- `data/logs/`: service logs

## Configuration

Main runtime settings live in `.env`. Important values:

- `MODEL_PATH`
- `SERVED_MODEL_NAME`
- `VLLM_IMAGE`
- `TRANSFORMERS_GIT_REF`
- `TENSOR_PARALLEL_SIZE`
- `MAX_MODEL_LEN`
- `GPU_MEMORY_UTILIZATION`
- `BIND_LOCALHOST_ONLY`
- `HOST_BIND_IP`

Default serving values follow the model-card style setup:

- `TENSOR_PARALLEL_SIZE=1`
- `MAX_MODEL_LEN=8192`
- `TRUST_REMOTE_CODE=true`
- `GPU_MEMORY_UTILIZATION=0.85`

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

- [docs/requirements.md](/media/shinkaman/INTEL_TUF/Sefetensors/NVFP4studio/docs/requirements.md)
- [docs/architecture.md](/media/shinkaman/INTEL_TUF/Sefetensors/NVFP4studio/docs/architecture.md)
- [docs/runbook.md](/media/shinkaman/INTEL_TUF/Sefetensors/NVFP4studio/docs/runbook.md)
- [docs/benchmark-methodology.md](/media/shinkaman/INTEL_TUF/Sefetensors/NVFP4studio/docs/benchmark-methodology.md)
- [docs/versions.md](/media/shinkaman/INTEL_TUF/Sefetensors/NVFP4studio/docs/versions.md)

---

# 中文

## 概述

`NVFP4studio` 是一个本地化的 NVFP4 推理工作台。它使用 vLLM 从本地模型目录启动服务，通过 FastAPI 提供 OpenAI 兼容 API，并在浏览器 UI 中展示 TTFT、token/s、latency 和 usage。

## 当前已验证

- 已在当前目标环境完成启动与联调
- 已验证环境为 Ubuntu 24.04 + NVIDIA GPU + Docker + NVIDIA Container Toolkit
- 三个核心服务都可通过 Docker Compose 启动:
  `vllm`
  `gateway`
  `frontend`
- 已验证接口:
  `GET /health`
  `GET /v1/models`
  `POST /v1/chat/completions`
  `GET /api/benchmarks/recent`
  `GET /api/benchmarks/export?format=json|csv`

## 默认地址

- UI: `http://127.0.0.1:3000`
- Gateway: `http://127.0.0.1:8000`
- vLLM: `http://127.0.0.1:8010`

## 快速开始

```bash
cd /media/shinkaman/INTEL_TUF/Sefetensors/NVFP4studio
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

## 主要功能

- 使用本地绝对路径加载 NVFP4 模型
- 提供 OpenAI 兼容聊天接口
- 记录 TTFT、completion token/s、total token/s、total latency
- 将 benchmark 结果保存到 SQLite
- 支持 JSON 和 CSV 导出

## 主要配置

配置文件为 `.env`，重点参数包括:

- `MODEL_PATH`
- `SERVED_MODEL_NAME`
- `VLLM_IMAGE`
- `TRANSFORMERS_GIT_REF`
- `MAX_MODEL_LEN`
- `GPU_MEMORY_UTILIZATION`
- `BIND_LOCALHOST_ONLY`

默认推理参数:

- `TENSOR_PARALLEL_SIZE=1`
- `MAX_MODEL_LEN=8192`
- `TRUST_REMOTE_CODE=true`
- `GPU_MEMORY_UTILIZATION=0.85`

## 数据目录

- SQLite: `data/sqlite/nvfp4studio.db`
- 日志: `data/logs/`
- 导出: `data/exports/`

## 已知说明

当前已验证模型有时会输出类似 `Thinking Process` 的推理文本。这是模型行为层面的现象，不影响服务链路本身。

## 文档

- [docs/requirements.md](/media/shinkaman/INTEL_TUF/Sefetensors/NVFP4studio/docs/requirements.md)
- [docs/architecture.md](/media/shinkaman/INTEL_TUF/Sefetensors/NVFP4studio/docs/architecture.md)
- [docs/runbook.md](/media/shinkaman/INTEL_TUF/Sefetensors/NVFP4studio/docs/runbook.md)
- [docs/benchmark-methodology.md](/media/shinkaman/INTEL_TUF/Sefetensors/NVFP4studio/docs/benchmark-methodology.md)
- [docs/versions.md](/media/shinkaman/INTEL_TUF/Sefetensors/NVFP4studio/docs/versions.md)

---

# 日本語

## 概要

`NVFP4studio` は、ローカルに保存した NVFP4 モデルを vLLM で起動し、FastAPI Gateway 経由で OpenAI 互換 API を公開しつつ、ブラウザ UI で TTFT、token/s、latency、usage を見える化するローカル向けスタジオです。

## 現在確認できていること

- Ubuntu 24.04 系 + NVIDIA GPU + Docker + NVIDIA Container Toolkit 上で起動確認済み
- `vllm`、`gateway`、`frontend` の 3 サービスが Compose で動作
- 次の API を確認済み:
  `GET /health`
  `GET /v1/models`
  `POST /v1/chat/completions`
  `GET /api/benchmarks/recent`
  `GET /api/benchmarks/export?format=json|csv`
- UI から会話、ベンチ履歴表示、export が可能

## 起動先

- UI: `http://127.0.0.1:3000`
- Gateway: `http://127.0.0.1:8000`
- vLLM: `http://127.0.0.1:8010`

## クイックスタート

```bash
cd /media/shinkaman/INTEL_TUF/Sefetensors/NVFP4studio
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
- streaming 中に TTFT と token/s を計測
- リクエスト単位のベンチ結果を SQLite に保存
- ベンチ履歴の JSON / CSV export
- ブラウザ UI で会話、状態確認、パラメータ調整

## 主な設定

設定は `.env` で変更できます。よく使う項目:

- `MODEL_PATH`
- `SERVED_MODEL_NAME`
- `VLLM_IMAGE`
- `TRANSFORMERS_GIT_REF`
- `MAX_MODEL_LEN`
- `GPU_MEMORY_UTILIZATION`
- `BIND_LOCALHOST_ONLY`
- `HOST_BIND_IP`

初期値は次の方針です。

- `TENSOR_PARALLEL_SIZE=1`
- `MAX_MODEL_LEN=8192`
- `TRUST_REMOTE_CODE=true`
- `GPU_MEMORY_UTILIZATION=0.85`

## 保存先

- SQLite: `data/sqlite/nvfp4studio.db`
- ログ: `data/logs/`
- export: `data/exports/`

## 注意点

今回の検証モデルは、プロンプト条件によって `Thinking Process` のような推論文をそのまま出すことがあります。アプリ基盤の動作には問題ありませんが、公開版では prompt 設計や後処理を追加調整する余地があります。

## 関連ドキュメント

- [docs/requirements.md](/media/shinkaman/INTEL_TUF/Sefetensors/NVFP4studio/docs/requirements.md)
- [docs/architecture.md](/media/shinkaman/INTEL_TUF/Sefetensors/NVFP4studio/docs/architecture.md)
- [docs/runbook.md](/media/shinkaman/INTEL_TUF/Sefetensors/NVFP4studio/docs/runbook.md)
- [docs/benchmark-methodology.md](/media/shinkaman/INTEL_TUF/Sefetensors/NVFP4studio/docs/benchmark-methodology.md)
- [docs/versions.md](/media/shinkaman/INTEL_TUF/Sefetensors/NVFP4studio/docs/versions.md)
