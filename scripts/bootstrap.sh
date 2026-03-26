#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
GPU_LOG="${ROOT_DIR}/data/logs/gpu-smoke.log"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker が見つかりません。" >&2
  exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "docker compose が使えません。" >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi が見つかりません。GPU ドライバを確認してください。" >&2
  exit 1
fi

if [ ! -f .env ]; then
  cp .env.example .env
  echo ".env を .env.example から生成しました。"
fi

mkdir -p data/sqlite data/exports data/logs
touch data/exports/.gitkeep data/logs/.gitkeep

set -a
source .env
set +a

if ! docker run --rm --pull missing --gpus all "${GPU_SMOKE_IMAGE}" nvidia-smi >"${GPU_LOG}" 2>&1; then
  echo "Docker から GPU を利用できません。" >&2
  echo "詳細: ${GPU_LOG}" >&2
  if grep -q 'could not select device driver' "${GPU_LOG}"; then
    echo "NVIDIA Container Toolkit の未設定が疑われます。" >&2
  fi
  echo "docs/runbook.md の GPU セクションを確認してください。" >&2
  exit 1
fi

docker compose pull vllm
docker compose build gateway frontend

echo "bootstrap 完了: .env と build キャッシュを準備しました。"
