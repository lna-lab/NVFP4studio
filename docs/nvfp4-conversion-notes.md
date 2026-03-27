# NVFP4 Conversion Notes

このメモは、ある `Qwen3.5` 派生 4B モデルを BF16 から NVFP4 へ変換したときの作業メモです。
通常運用の README とは分けて、量子化の再現手順だけを残します。

## 最初に結論

- 4B BF16 原本から NVFP4 変換自体は完了しました。
- 出力サイズは約 `6.3G` で、`model.safetensors`、`config.json`、`recipe.yaml` などを含む成果物を得られました。
- 量子化の再現には、`transformers` の新しめの commit、`llm-compressor` と `compressed-tensors` の互換 shim、C compiler の用意が重要でした。
- 現時点では、host 側の個別環境差を減らすため `quantize_4b_nvfp4_docker.sh` を既定ルートとして扱います。
- sample generation は必須成功条件ではないため、初回は `--skip-sample` を安全側の既定として使います。

## 今回の前提

- 入力モデル:
  `/models/huihui/Huihui-Qwen3.5-4B-abliterated`
- 出力先:
  `/models/nvfp4/Huihui-Qwen3.5-4B-abliterated-NVFP4`
- upstream 参照:
  `vllm-project/llm-compressor`
  `examples/quantization_w4a16_fp4/nvfp4/qwen3.5_example.py`
- llm-compressor ref:
  `cf3bd6463e8d471ad6c8cc20a6a9b053c178e555`
- transformers ref:
  `09832b2ae515cfbd020327f5d3ba2dafe6edf83c`

## 用意したもの

- 環境セットアップ:
  [setup_llm_compressor_env.sh](../scripts/setup_llm_compressor_env.sh)
- 4B 用の量子化スクリプト:
  [quantize_4b_nvfp4.py](../scripts/quantize_4b_nvfp4.py)
- Docker 実行ラッパー:
  [quantize_4b_nvfp4_docker.sh](../scripts/quantize_4b_nvfp4_docker.sh)
- Dockerfile:
  [config/llm-compressor/Dockerfile](../config/llm-compressor/Dockerfile)

## 今回の推奨ルート

- まず Docker ラッパーを使う
- 初回は `--skip-sample` で変換完了を優先する
- 変換後に BF16 と同じ canary を回して品質差を見る
- BF16 原本は上書きしない

## 手順

作業ディレクトリ:

```bash
cd /path/to/NVFP4studio
```

host 側で venv を使う場合:

```bash
./scripts/setup_llm_compressor_env.sh
source .venv-llm-compressor/bin/activate
```

このサーバーでは Docker 実行を推奨:

```bash
./scripts/quantize_4b_nvfp4_docker.sh --skip-sample
```

出力先を変えたいとき:

```bash
./scripts/quantize_4b_nvfp4_docker.sh \
  --model-path /models/huihui/Huihui-Qwen3.5-4B-abliterated \
  --output-dir /models/nvfp4/Huihui-Qwen3.5-4B-abliterated-NVFP4
```

## 量子化設定

upstream 例を踏襲して、次を ignore する。

- `lm_head`
- `re:.*visual.*`
- `re:.*linear_attn.*`
- `re:.*mtp.*`

scheme は `NVFP4A16`、targets は `Linear` を使う。

## 変換後に見たいこと

- まず short sample generation が通るか
- `vLLM` が `compressed-tensors` として起動できるか
- BF16 と比べて JSON / 日本語会話 / routing の品質差がどこに出るか
- 35B / 27B と役割分担したとき、4B が受付役として十分か

## 2026-03-26 の実績

- BF16 原本から NVFP4 変換が完了
- 出力先:
  `/models/nvfp4/Huihui-Qwen3.5-4B-abliterated-NVFP4`
- 出力サイズ:
  約 `6.3G`
- 主な出力ファイル:
  `model.safetensors`, `config.json`, `recipe.yaml`, `tokenizer.json`,
  `processor_config.json`, `nvfp4_conversion_metadata.json`

今回ハマった点:

- `Qwen3_5ForConditionalGeneration` は古い `transformers` pin では読めなかったため、
  `transformers` を `5.3.0.dev0` 系の commit へ上げる必要があった
- `llm-compressor` と `compressed-tensors` の dev snapshot 間で一部 import 名がずれており、
  量子化スクリプト側に互換 shim を入れて吸収した
- NVFP4 保存時に Triton / Inductor が C compiler を要求したため、
  Docker 実行時に `gcc` / `g++` を追加で入れて通した
- sample generation は量子化成功の必須条件ではないため、現時点では `--skip-sample` を既定の安全策として使う

## 注意

- BF16 原本を上書きしない
- 初回比較では BF16 と NVFP4 で同じ canary を回す
- 圧縮由来の癖とモデル固有の癖を混同しない
