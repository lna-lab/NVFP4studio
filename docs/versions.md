# Versions

このリポジトリで固定した主要バージョンです。

| Component | Version |
| --- | --- |
| vLLM base image | `vllm/vllm-openai:v0.17.1` |
| Transformers git ref | `09832b2ae515cfbd020327f5d3ba2dafe6edf83c` |
| llm-compressor git ref | `cf3bd6463e8d471ad6c8cc20a6a9b053c178e555` |
| Python base image | `python:3.12-slim-bookworm` |
| FastAPI | `0.115.12` |
| Uvicorn | `0.34.0` |
| httpx | `0.28.1` |
| nvidia-ml-py | `12.575.51` |
| Node base image | `node:20-bookworm-slim` |
| Next.js | `14.2.13` |
| React | `18.3.1` |
| TypeScript | `5.8.2` |
| SQLite | host bundled |

## Notes

- vLLM の pin は Compose の `.env` から差し替え可能です。
- frontend / backend とも Docker build で再現する前提です。
- 実環境で pin を更新した場合はこのファイルと `README.md` を一緒に更新してください。
