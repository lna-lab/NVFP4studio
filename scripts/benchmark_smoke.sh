#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

set -a
[ -f .env ] && source .env
set +a

HOST_BIND_IP="${HOST_BIND_IP:-127.0.0.1}"
OUT_FILE="data/exports/benchmark-smoke-$(date +%Y%m%d-%H%M%S).json"

python3 - <<'PY' "${HOST_BIND_IP}" "${GATEWAY_PORT}" "${OPENAI_API_KEY}" "${SERVED_MODEL_NAME}" "${OUT_FILE}"
import json
import sys
import time
import urllib.request

host, port, api_key, model_name, out_file = sys.argv[1:]
url = f"http://{host}:{port}/v1/chat/completions"
results = []

for index in range(3):
    payload = {
        "model": model_name,
        "stream": False,
        "messages": [
            {"role": "system", "content": "You are a concise benchmark assistant."},
            {"role": "user", "content": f"Say benchmark run {index + 1} in Japanese."},
        ],
        "temperature": 0.2,
        "max_tokens": 64,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    started = time.time()
    with urllib.request.urlopen(req, timeout=600) as response:
        body = json.loads(response.read().decode("utf-8"))
        results.append(
            {
                "run": index + 1,
                "elapsed_ms": round((time.time() - started) * 1000, 2),
                "request_id": response.headers.get("x-nvfp4studio-request-id"),
                "response": body,
            }
        )

with open(out_file, "w", encoding="utf-8") as handle:
    json.dump(results, handle, ensure_ascii=False, indent=2)

print(out_file)
PY

