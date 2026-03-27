FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive
ARG TRANSFORMERS_REF=09832b2ae515cfbd020327f5d3ba2dafe6edf83c
ARG LLM_COMPRESSOR_REF=cf3bd6463e8d471ad6c8cc20a6a9b053c178e555

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install \
        "git+https://github.com/huggingface/transformers@${TRANSFORMERS_REF}" \
        accelerate \
        safetensors \
        sentencepiece \
    && python -m pip install \
        "git+https://github.com/vllm-project/llm-compressor@${LLM_COMPRESSOR_REF}" \
    && python -m pip install \
        "torchvision==0.25.0"

WORKDIR /workspace
