#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEFETENSORS_ROOT="/models"
IMAGE_NAME="${IMAGE_NAME:-nvfp4studio-llm-compressor:latest}"
DOCKERFILE_PATH="${ROOT_DIR}/config/llm-compressor/Dockerfile"
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

MODEL_PATH="${MODEL_PATH:-/models/huihui/Huihui-Qwen3.5-4B-abliterated}"
OUTPUT_DIR="${OUTPUT_DIR:-/models/nvfp4/Huihui-Qwen3.5-4B-abliterated-NVFP4}"
RUNTIME_TRANSFORMERS_INSTALL="${RUNTIME_TRANSFORMERS_INSTALL:-git+https://github.com/huggingface/transformers.git@09832b2ae515cfbd020327f5d3ba2dafe6edf83c}"
EXTRA_ARGS=()
if (($# > 0)); then
  for arg in "$@"; do
    EXTRA_ARGS+=("$(printf '%q' "${arg}")")
  done
fi
EXTRA_ARGS_STR="${EXTRA_ARGS[*]:-}"

docker build -t "${IMAGE_NAME}" -f "${DOCKERFILE_PATH}" "${ROOT_DIR}"

docker run --rm \
  --gpus all \
  --ipc host \
  --shm-size 16gb \
  -e HOME=/tmp \
  -e HOST_UID="${HOST_UID}" \
  -e HOST_GID="${HOST_GID}" \
  -v "${SEFETENSORS_ROOT}:${SEFETENSORS_ROOT}" \
  -w "${ROOT_DIR}" \
  "${IMAGE_NAME}" \
  bash -lc "apt-get update >/tmp/nvfp4studio-apt.log 2>&1 \
    && apt-get install -y --no-install-recommends gcc g++ >>/tmp/nvfp4studio-apt.log 2>&1 \
    && python -m pip install --no-cache-dir \"${RUNTIME_TRANSFORMERS_INSTALL}\" >/tmp/nvfp4studio-transformers.log 2>&1 \
    && python ./scripts/quantize_4b_nvfp4.py \
    --model-path \"${MODEL_PATH}\" \
    --output-dir \"${OUTPUT_DIR}\" \
    ${EXTRA_ARGS_STR} \
    && if [ -d \"${OUTPUT_DIR}\" ]; then chown -R \"${HOST_UID}:${HOST_GID}\" \"${OUTPUT_DIR}\"; fi"
