#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

set -a
[ -f .env ] && source .env
set +a

HOST_BIND_IP="${HOST_BIND_IP:-127.0.0.1}"

echo "[host] nvidia-smi"
nvidia-smi || true

echo
echo "[gateway] /health"
curl -fsS "http://${HOST_BIND_IP}:${GATEWAY_PORT}/health" | python3 -m json.tool

echo
echo "[gateway] /v1/models"
curl -fsS "http://${HOST_BIND_IP}:${GATEWAY_PORT}/v1/models" | python3 -m json.tool

echo
echo "[gateway] /api/system/status"
curl -fsS "http://${HOST_BIND_IP}:${GATEWAY_PORT}/api/system/status" | python3 -m json.tool

