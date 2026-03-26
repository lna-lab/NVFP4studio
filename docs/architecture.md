# Architecture

## Overview

`NVFP4studio` は 3 サービス構成です。

1. `vllm`
   - ローカルモデルパスを bind mount
   - `vllm serve` で OpenAI 互換 API を公開
   - `/metrics` を expose
2. `gateway`
   - FastAPI
   - `/v1/*` を vLLM に中継
   - streaming を観測して TTFT / token/s / latency を計算
   - SQLite に保存
3. `frontend`
   - Next.js App Router
   - 状態表示、チャット、ベンチパネル、履歴 export

## Request Flow

1. Browser UI が `POST /v1/chat/completions` を Gateway に送る
2. Gateway が payload を整形し、streaming 時は `stream_options.include_usage=true` を付与
3. Gateway が vLLM upstream の最初の token chunk を観測し TTFT を計算
4. Gateway が usage と終了時刻から throughput / latency を計算
5. SQLite に benchmark row を保存
6. UI が `request_id` から benchmark detail を読み、会話 bubble に表示

## Data Model

主要カラム:

- `request_id`
- `model_name`
- `prompt_tokens`
- `completion_tokens`
- `total_tokens`
- `started_at`
- `first_token_at`
- `finished_at`
- `ttft_ms`
- `e2e_latency_ms`
- `completion_tokens_per_sec`
- `total_tokens_per_sec`
- `streaming`
- `temperature`
- `top_p`
- `max_tokens`
- `finish_reason`
- `error_message`

## Security Posture

- 既定は localhost bind
- API key はサポートするが、UI では未入力でもローカル利用を優先
- 外部公開前提の認証や multi-user 制御は含めない

