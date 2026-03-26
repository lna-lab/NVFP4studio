ARG VLLM_BASE_IMAGE=vllm/vllm-openai:v0.17.1
FROM ${VLLM_BASE_IMAGE}

ARG TRANSFORMERS_GIT_REF=5a098a1

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN uv pip install --system --upgrade \
    "git+https://github.com/huggingface/transformers@${TRANSFORMERS_GIT_REF}"
