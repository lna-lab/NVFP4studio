#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
GPU_LOG="${ROOT_DIR}/data/logs/gpu-smoke.log"

if [ ! -f .env ]; then
  echo ".env がありません。先に ./scripts/bootstrap.sh を実行してください。" >&2
  exit 1
fi

set -a
source .env
set +a

if [ ! -d "${MODEL_PATH}" ]; then
  echo "MODEL_PATH が存在しません: ${MODEL_PATH}" >&2
  exit 1
fi

if [ "${BIND_LOCALHOST_ONLY:-true}" = "true" ]; then
  export HOST_BIND_IP="${HOST_BIND_IP:-127.0.0.1}"
else
  export HOST_BIND_IP="${HOST_BIND_IP:-0.0.0.0}"
fi

mkdir -p data/sqlite data/exports data/logs

if ! docker run --rm --pull never --gpus all "${GPU_SMOKE_IMAGE:-nvidia/cuda:13.0.1-base-ubuntu24.04}" nvidia-smi >"${GPU_LOG}" 2>&1; then
  echo "Docker GPU チェックに失敗しました: ${GPU_LOG}" >&2
  echo "先に ./scripts/bootstrap.sh と docs/runbook.md の GPU セクションを確認してください。" >&2
  exit 1
fi

docker compose up -d --build

echo "ヘルスチェック待機中..."
for attempt in $(seq 1 90); do
  if curl -fsS "http://${HOST_BIND_IP}:${GATEWAY_PORT}/health" >/dev/null 2>&1 \
    && curl -fsS "http://${HOST_BIND_IP}:${GATEWAY_PORT}/v1/models" >/dev/null 2>&1 \
    && curl -fsS "http://${HOST_BIND_IP}:${WEB_PORT}" >/dev/null 2>&1; then
    break
  fi
  sleep 5
  if [ "${attempt}" -eq 90 ]; then
    echo "起動確認に失敗しました。docker compose logs を確認してください。" >&2
    exit 1
  fi
done

echo "UI:      http://${HOST_BIND_IP}:${WEB_PORT}"
echo "Gateway: http://${HOST_BIND_IP}:${GATEWAY_PORT}"
echo "vLLM:    http://${HOST_BIND_IP}:${VLLM_PORT}"
echo "Logs:    ${ROOT_DIR}/data/logs"
