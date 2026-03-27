# Requirements Summary

このリポジトリは `NVFP4studio_requirements_codex_ja.md` を実装用の正本とし、その要点をここに整理します。

## MVP 目的

1. ローカル環境セットアップの自動化
2. vLLM で NVFP4 モデルを実行し OpenAI 互換 API を公開
3. ブラウザ UI から対話し、TTFT / token/s / latency を見える化

## 絶対条件

- モデル実体はホスト上の既存 NVFP4 ディレクトリをそのまま参照する
- 実装先はこのリポジトリ直下とする
- モデルファイルは移動・改変しない

## サービス要件

- `vLLM`: `/v1/models`, `/v1/chat/completions`, `/metrics`
- `Gateway`: health, config, benchmarks, OpenAI 互換 proxy
- `Web UI`: 1 画面型チャット、system prompt、パラメータ、ベンチ表示、履歴 export

## 計測要件

- TTFT
- completion token/s
- total token/s
- total latency
- prompt / completion / total tokens

## 保存要件

- SQLite に会話単位で保存
- CSV / JSON export
