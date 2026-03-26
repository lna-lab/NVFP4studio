# Versions

このリポジトリで固定した主要バージョンです。

| Component | Version |
| --- | --- |
| vLLM base image | `vllm/vllm-openai:v0.17.1` |
| Transformers git ref | `5a098a1` |
| Python base image | `python:3.12-slim-bookworm` |
| FastAPI | `0.115.12` |
| Uvicorn | `0.34.0` |
| httpx | `0.28.1` |
| Node base image | `node:20-bookworm-slim` |
| Next.js | `14.2.13` |
| React | `18.3.1` |
| TypeScript | `5.8.2` |
| SQLite | host bundled |

## Notes

- vLLM の pin は Compose の `.env` から差し替え可能です。
- frontend / backend とも Docker build で再現する前提です。
- 実環境で pin を更新した場合はこのファイルと `README.md` を一緒に更新してください。
