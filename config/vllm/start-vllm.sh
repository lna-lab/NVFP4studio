#!/usr/bin/env bash
set -euo pipefail

if [ ! -d "${MODEL_PATH}" ]; then
  echo "MODEL_PATH not found: ${MODEL_PATH}" >&2
  exit 1
fi

mkdir -p /workspace/logs
TOKENIZER_DIR="/tmp/nvfp4studio-tokenizer"

python3 - <<'PY'
import json
import os
import shutil
from pathlib import Path

model_dir = Path(os.environ["MODEL_PATH"])
tokenizer_dir = Path("/tmp/nvfp4studio-tokenizer")
tokenizer_dir.mkdir(parents=True, exist_ok=True)

for name in ("tokenizer.json", "chat_template.jinja"):
    src = model_dir / name
    if src.exists():
        shutil.copy2(src, tokenizer_dir / name)

config_path = model_dir / "tokenizer_config.json"
with config_path.open() as handle:
    tokenizer_config = json.load(handle)

if tokenizer_config.get("tokenizer_class") == "TokenizersBackend":
    tokenizer_config.pop("backend", None)
    tokenizer_config.pop("is_local", None)
    tokenizer_config["tokenizer_class"] = "Qwen2Tokenizer"
    tokenizer_config.setdefault("add_bos_token", False)

with (tokenizer_dir / "tokenizer_config.json").open("w") as handle:
    json.dump(tokenizer_config, handle, ensure_ascii=False, indent=2)
PY

args=(
  serve
  "${MODEL_PATH}"
  --tokenizer "${TOKENIZER_DIR}"
  --host "${VLLM_HOST}"
  --port "${VLLM_PORT}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
)

if [ -n "${MAX_NUM_SEQS:-}" ]; then
  args+=(--max-num-seqs "${MAX_NUM_SEQS}")
fi

if [ -n "${MAX_NUM_BATCHED_TOKENS:-}" ]; then
  args+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
fi

if [ -n "${KV_CACHE_DTYPE:-}" ]; then
  args+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
fi

if [ -n "${KV_CACHE_MEMORY_BYTES:-}" ]; then
  args+=(--kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}")
fi

if [ -n "${CPU_OFFLOAD_GB:-}" ]; then
  args+=(--cpu-offload-gb "${CPU_OFFLOAD_GB}")
fi

if [ -n "${SWAP_SPACE:-}" ]; then
  args+=(--swap-space "${SWAP_SPACE}")
fi

if [ "${TRUST_REMOTE_CODE}" = "true" ]; then
  args+=(--trust-remote-code)
fi

if [ "${LANGUAGE_MODEL_ONLY:-false}" = "true" ]; then
  args+=(--language-model-only)
fi

vllm "${args[@]}" 2>&1 | tee -a /workspace/logs/vllm.log
