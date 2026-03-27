#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv-llm-compressor"

TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
TORCH_PACKAGES="${TORCH_PACKAGES:-torch torchvision}"
TRANSFORMERS_REF="${TRANSFORMERS_REF:-5a098a1}"
LLM_COMPRESSOR_REF="${LLM_COMPRESSOR_REF:-cf3bd6463e8d471ad6c8cc20a6a9b053c178e555}"

if ! python3 -m venv "${VENV_DIR}" 2>/tmp/nvfp4studio-venv.err; then
  cat /tmp/nvfp4studio-venv.err
  cat <<EOF

host 側では venv を作れませんでした。
このサーバーでは Docker 版を使う方が再現しやすいです。

推奨:
  ./scripts/quantize_4b_nvfp4_docker.sh

EOF
  exit 1
fi
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url "${TORCH_INDEX_URL}" ${TORCH_PACKAGES}
python -m pip install \
  "git+https://github.com/huggingface/transformers@${TRANSFORMERS_REF}" \
  accelerate \
  safetensors \
  sentencepiece
python -m pip install "git+https://github.com/vllm-project/llm-compressor@${LLM_COMPRESSOR_REF}"

cat <<EOF

llm-compressor 用の venv を用意しました。

activate:
  source "${VENV_DIR}/bin/activate"

確認:
  python -c "import torch, transformers, llmcompressor; print(torch.__version__)"

EOF
