# Model Research Notes

このメモは README とは分けて、モデル研究の観点から得られた知見を残すためのノートです。
運用手順ではなく、比較軸、仮説、観測、次に試す価値があることを蓄積します。

## 最初に結論

- このリポジトリで主役として見ているのは `Qwen3.5-27B`、`Qwen3.5-35B-A3B`、`Qwen3.5-4B` の 3 系統です。
- 現時点の主力候補は `35B-A3B NVFP4` です。
- `35B-A3B 4G` は `-c 256K` を維持しつつ、速度と安定性の両方で最も扱いやすい結果でした。
- `27B-abliterated` は sidecar や比較用途には有望ですが、full `256K` を守るには `9G` から `10G` 程度の厚めの KV 予算が必要でした。
- `27B-Claude-4.6-Opus-abliterated` は reasoning 漏れと strict JSON の崩れが強く、現時点の運用候補から外しています。
- `4B` は受付役として価値が高く、BF16 から NVFP4 への変換自体も通りました。
- 研究全体を通して分かったのは、単なるサイズ差よりも `KV budget`、thinking 制御、JSON 安定性の差が運用適性を大きく左右するということです。

## 今いちばん関心があること

- `MoE` と dense / sparse 系モデルで、同じ GPU 制約の中でどこまで品質と速度の両立が変わるか
- 事後学習や abliterated 派生によって、推論品質の性格がどう変わるか
- `NVFP4` のような強い圧縮をかけても、どの水準まで品質を守れるか
- 単体性能だけでなく、共存運用時にどのモデルが最も「使いやすい計算資源」になるか

## 研究テーマの整理

### 1. MoE / dense / sparse の比較

見たいのは単純な token/s だけではありません。
本当に見たいのは次の差です。

- 同じ `-c 256K` での VRAM 占有
- 同じ GPU 上での multi-instance 運用のしやすさ
- 低 VRAM 化したときの品質の壊れ方
- 長文、構造化出力、翻訳、推論タスクでの挙動差

関心としては、「速いモデル」よりも「資源制約下でどこまで気持ちよく使えるか」にあります。

### 2. 事後学習モデルの差

今回見ていきたいのは、base に近いものよりも

- `abliterated`
- Claude 系の蒸留・事後学習的な派生
- Huihui 系のローカライズや調整済みモデル

のような、性格づけが入った派生です。

ここでは次を比較したいです。

- 応答の素直さ
- 指示追従
- 構造化出力の安定性
- 冗長さ
- 推論文や hidden reasoning の漏れやすさ
- 日本語での自然さ

## 2026-03-26 時点の観測

## 現時点の運用候補

- `35B-A3B 4G`
  - 主力対話モデル向け
  - `-c 256K` を保ちつつ、VRAM と品質のバランスが最も良い
- `35B-A3B 4G + 27B-abliterated 10G`
  - 品質重視の mixed-model 構成
  - full `256K` と strict JSON を両立しやすい
- `35B-A3B 4G + 27B-abliterated 9G`
  - 省 VRAM 寄りの mixed-model 構成
  - full `256K` は保てるが、出力包装の guardrail を足したい

## 現時点で外している候補

- `35B 3G`
  - VRAM は下がるが、quality canary が崩れた
- `27B-Claude-4.6-Opus-abliterated`
  - hidden reasoning 漏れと strict JSON 崩れが強い
- `2x2` を throughput 強化策として採用する案
  - 成立はするが、総 token/s 向上は確認できなかった

### 35B NVFP4 で分かったこと

- VRAM 差の主因はモデル重みそのものより `KV cache` 予約だった
- `KV_CACHE_MEMORY_BYTES` を明示すると、`-c 256K` を維持したまま VRAM を大きく削れた
- `4G` までは quality canary を守れた
- `3G` は速度は出ても品質 canary が崩れたため不採用

要するに、速度より先に品質 guardrail を置く方針は正しかった。

### 2インスタンス x 2シーケンスで分かったこと

- `KV budget 6G` なら、各インスタンスを `約34.7GB` に収めつつ `2 * 256K` を確保できた
- 品質 canary と sustained exact JSON は通った
- ただし aggregate throughput は `1x2` を明確には上回らなかった
- したがって `2x2` は throughput 拡張というより、テナント分離や比較実験のための構成として有望

これはかなり大きい知見です。
GPU を「1つの巨大な推論器」としてだけでなく、「複数の独立した研究区画」として使える可能性が見えたからです。

### 27B 系を含めた初回ベースライン

2026-03-26 に `scripts/probe_model_matrix.py` を追加し、次の 3 モデルを同一条件で比較した。

- `Huihui-Qwen3.5-35B-A3B-abliterated-NVFP4`
- `Huihui-Qwen3.5-27B-abliterated-NVFP4`
- `Huihui-Qwen3.5-27B-Claude-4.6-Opus-abliterated-NVFP4`

比較条件は次の通り。

- `runtime_profile=memory`
- `MAX_MODEL_LEN=262144`
- `KV_CACHE_MEMORY_BYTES=4G`
- `GPU_MEMORY_UTILIZATION=0.45`
- `MAX_NUM_SEQS=1`
- `KV_CACHE_DTYPE=fp8`

参照レポート:

- `data/exports/model-matrix-probe-20260326-192424.json`
- `data/exports/model-matrix-probe-20260326-192424.md`

初回結果の要点:

- `35B-A3B`
  - idle `29924 MB`
  - peak `34295 MB`
  - `TTFT 79.47 ms`
  - `170.7 tok/s`
  - canary `3/3`
- `27B-abliterated`
  - idle `26677 MB`
  - peak `31065 MB`
  - `TTFT 89.32 ms`
  - `60.6 tok/s`
  - canary `3/3`
- `27B-Claude-4.6-Opus-abliterated`
  - idle `26681 MB`
  - peak `31067 MB`
  - `TTFT 87.95 ms`
  - `57.1 tok/s`
  - canary `1/3`

第一印象:

- 35B は、同じ `4G` 制約でも速度と安定性の両方でかなり強い
- 27B 2本は VRAM では有利だが、今回の短い streaming 実測では token/s が 35B よりかなり低い
- `27B-Claude-4.6-Opus-abliterated` は hidden reasoning 漏れが強く、`enable_thinking=false` を素直に守らない
- 同モデルは JSON strictness も崩しやすく、比較対象としては面白いが、現時点の運用適性は低い

とくに重要なのは、`27B-Claude-4.6-Opus-abliterated` が性能以前に「挙動制御」の問題を抱えていること。
この差は、単なるサイズ差ではなく、事後学習の性格差として見る価値がある。

### 35B と 27B の混在で分かったこと

`scripts/probe_mixed_models.py` を使い、`35B-A3B` と `27B-abliterated` を同じ GPU に同居させる検証を行った。
今回は次の構成を試した。

- `35B-A3B`: `KV budget 4G`
- `27B-abliterated`: `KV budget 10G`
- 両方とも `MAX_MODEL_LEN=262144`
- 両方とも `MAX_NUM_SEQS=1`
- `KV_CACHE_DTYPE=fp8`

参照レポート:

- `data/exports/mixed-model-probe-20260326-193932.json`
- `data/exports/mixed-model-probe-20260326-193932.md`

結果の要点:

- 総使用 VRAM は `約69.2GB` で、`96GB` 枠に十分収まった
- `35B-A3B` は `gpu_process_memory_mb 32766`、`reserved_kv_capacity_tokens 419200`
- `27B-abliterated` は `gpu_process_memory_mb 35506`、`reserved_kv_capacity_tokens 326144`
- どちらも `1 * 256K` を保持できた
- sustained exact JSON と strict JSON review は両モデルとも通過した

ここで重要なのは、`27B-abliterated` は単体 `KV budget 4G` だと `256K` を保持できなかったのに、混在検証では `10G` を与えることで `35B` と共存できたこと。
つまり、`35B` は比較的低い KV でも強い一方、`27B` は full `256K` を保つにはやや厚めの KV 予算を必要とする。

現時点では、この混在構成は throughput を最大化するというより、

- `35B` を主力対話モデル
- `27B` を比較・検証・補助ワークロード用の sidecar

として並べる用途に向いている。

### 35B 4G + 27B の KV budget sweep

混在構成で `27B-abliterated` 側の `KV budget` を `10G -> 9G -> 8G -> 7G -> 6G` と下げて追試した。
`35B-A3B` 側は `4G` 固定で、評価は mixed probe の sustained exact JSON と strict JSON review を使った。

参照レポート:

- `data/exports/mixed-model-probe-20260326-194641.json`
- `data/exports/mixed-model-probe-20260326-195011.json`
- `data/exports/mixed-model-probe-20260326-195341.json`
- `data/exports/mixed-model-probe-20260326-195711.json`

要点:

- `10G`
  - `reserved_kv_capacity_tokens 326144`
  - `gpu_process_memory_mb 35506`
  - `1 * 256K` を保持
  - sustained exact JSON 通過
  - strict JSON review 通過
- `9G`
  - `reserved_kv_capacity_tokens 294784`
  - `gpu_process_memory_mb 34514`
  - `1 * 256K` を保持
  - sustained exact JSON 通過
  - strict JSON review は markdown fence 付き JSON になって失敗
- `8G`
  - `reserved_kv_capacity_tokens 261856`
  - `gpu_process_memory_mb 33580`
  - `256K` にわずかに足りない
  - sustained exact JSON 通過
  - strict JSON review 失敗
- `7G` / `6G`
  - さらに VRAM は下がる
  - sustained exact JSON は通る
  - ただし `256K` を保持できず、strict JSON review も失敗

ここから言えることはかなりはっきりしている。

- `27B-abliterated` を `35B` と混在させる場合、full `256K` を守るには `9G` 以上が必要
- ただし、full `256K` と strict JSON の素直さを両立するには `10G` の方が安心
- `9G` はかなり面白い境界値で、内容自体はほぼ正しいが、出力包装が崩れ始める

したがって現時点の実運用候補は次の 2 つになる。

- `35B 4G + 27B 10G`
  - 品質重視
  - full `256K` と strict JSON を両立しやすい
- `35B 4G + 27B 9G`
  - 省 VRAM 寄り
  - full `256K` は保てるが、出力整形の guardrail を足したい

## 今日の仮説更新

- `35B-A3B NVFP4` は、現時点では「主力運用モデル」の最有力候補
- `27B-abliterated NVFP4` は、低 VRAM 区画向けの比較的素直なサブモデル候補
- `35B-A3B 4G + 27B-abliterated 10G` は、`-c 256K` を保ったまま 96GB GPU 上で混在可能
- `35B-A3B 4G + 27B-abliterated 9G` は、`256K` は保つが出力包装が崩れやすい境界値
- `27B-Claude-4.6-Opus-abliterated NVFP4` は、reasoning 漏れと構造化出力の崩れが強く、今回の検討材料から外す
- `4B` は受付役としての価値が高く、`9B` より先に BF16 原本から NVFP4 変換して探る価値がある
- モデルの使いやすさは、単なる size / tok/s だけでなく、「従順さ」「thinking 制御」「JSON 安定性」でかなり変わる

### 4B BF16 -> NVFP4 変換で分かったこと

`Huihui-Qwen3.5-4B-abliterated` の BF16 原本を `llm-compressor` で NVFP4 化し、
次の出力を得た。

- 出力:
  `/models/nvfp4/Huihui-Qwen3.5-4B-abliterated-NVFP4`
- サイズ:
  約 `6.3G`
- config 上の architecture は維持され、`quantization_config` も付与された

ここで重要だったのは、4B の変換自体は通ることが確認できた点。
つまり 4B を「受付役として NVFP4 化し、35B/27B と共存させる」研究は現実的な次フェーズに入った。

一方で、変換ルートには次の癖があった。

- `transformers` は `qwen3_5` を読める新しめの main 系 commit が必要
- `llm-compressor` と `compressed-tensors` の dev snapshot の噛み合わせに shim が要る
- NVFP4 の保存時に Triton / Inductor が C compiler を必要とする

このため、4B 研究は「モデル品質」だけでなく、「量子化ツールチェーンの安定性」も評価対象になる。

### 4B BF16 vs 4B NVFP4 の初回品質比較

`scripts/compare_4b_quality.py` を使い、BF16 原本と NVFP4 版を同じ vLLM runtime 条件で比較した。
条件は次の通り。

- `MAX_MODEL_LEN=8192`
- `GPU_MEMORY_UTILIZATION=0.60`
- `MAX_NUM_SEQS=1`
- `MAX_NUM_BATCHED_TOKENS=4096`
- thinking は無効
- 評価項目は日本語応答、数列、英訳、strict JSON、抽出 JSON、reasoning 抑制

参照レポート:

- `data/exports/quality-compare-4b-20260326-121646.json`
- `data/exports/quality-compare-4b-20260326-121646.md`

結果はかなり極端だった。

- `4B BF16`
  - canary は実質ほぼ通過
  - 日本語、数列、strict JSON、抽出 JSON は安定
  - 英訳 canary は validator 上は `trim VRAM while preserving quality` を false にしたが、意味的には妥当
  - streaming も自然文で返答
- `4B NVFP4`
  - 応答が `!` の連打に崩れた
  - strict JSON も抽出も不成立
  - streaming でも同様に `!` 連打
  - TTFT や token/s は出るが、品質評価としては不合格

この結果から言えること:

- 現在の 4B NVFP4 版は「少し劣化」ではなく、実用にならないレベルで壊れている
- 問題は vLLM runtime より、変換ルートか対象アーキテクチャとの相性にある可能性が高い
- BF16 側が正常なので、比較基準は確保できている

観測上の仮説:

- `Qwen3_5ForConditionalGeneration` の 4B 系は `linear_attention` を多く含み、今回の NVFP4 変換 recipe と相性が悪い可能性がある
- ignore は広く入っているが、それでも量子化対象に残った層のどこかで表現が崩壊している可能性がある
- tokenizer 破損ではなく、モデル出力自体が単一記号へ崩れている可能性が高い

したがって 4B の次の段階は、

- 現行 recipe のまま 4B NVFP4 を採用することではない
- どの層を追加で ignore すべきか、または 4B 系では別の圧縮方針が必要かを切り分けること
- BF16 を比較基準として維持しながら、量子化崩壊の原因箇所を狭めること

### 4B NVFP4 の hint 反映再試行でも崩壊は解消しなかった

`hint.md` に拾っておいた Hugging Face 上の 4B NVFP4 モデルカード記述から、
次のヒントを抽出して 2 回目の再試行を行った。

- `MLP-only`
- `MSE calibration`
- `CNN/DailyMail 256 samples`
- `KV-cache is not quantized`
- text-only 推論では `language-model-only` 相当の扱いを意識する

これに合わせて、`scripts/quantize_4b_calibrated_nvfp4.py` を次の方向に寄せた。

- `mixed` に加えて `mlp_only` preset を追加
- `mlp_only` では `mlp.(gate_proj|up_proj|down_proj)` のみを NVFP4 対象にする
- calibration 既定を `cnn_dailymail/3.0.0 train[:256]` に変更
- multimodal `processor` ではなく `tokenizer` で calibration text を token 化する

ここで分かったことはかなり重要だった。

- `nvidia/Nemotron-Post-Training-Dataset-v2` は gated で、そのままでは誰でも再現できない
- `cnn_dailymail` へ切り替えると calibration 自体は最後まで通る
- `mlp_only` recipe でも保存までは正常に完了する
- しかし品質比較では、出力崩壊は解消しなかった

参照レポート:

- `data/exports/quality-compare-4b-20260326-125630.json`
- `data/exports/quality-compare-4b-20260326-125630.md`

結果の要点:

- `4B BF16`
  - canary `6/6`
  - streaming も正常
- `4B NVFP4 calibrated-mlp_only`
  - canary `1/6`
  - 出力は依然として `!` の連打に崩壊
  - streaming でも同様に崩れた

この時点での仮説更新:

- 4B の崩壊原因は「calibration が薄い」だけではない
- `attention を触らない` だけでも直らない
- `llm-compressor` での保存形式、または 4B Qwen3.5 系の hybrid / linear-attention 構造に対して、さらに限定的な量子化対象が必要な可能性が高い
- `MLP-only` と呼んでいても、4B では `down_proj` を含めるのが強すぎる可能性がある

つまり、次の切り分けはより狭くする必要がある。

- `gate_proj + up_proj` だけを量子化
- `down_proj` を BF16 / FP8 に残す
- あるいは `Qwen/Qwen3.5-4B` 素体で同じ recipe を先に検証し、`Huihui` 派生との差を切る

## 次に試す価値があること

- `Huihui-Qwen3.5-4B-abliterated` の BF16 原本を llm-compressor で NVFP4 化する
- 4B BF16 と 4B NVFP4 で同じ canary を回し、圧縮由来の劣化を見分ける
- `35B-A3B 4G + 27B-abliterated 9G` に対して、post-process や stricter sampling で JSON fence 崩れを抑えられるか確認する
- `35B + 27B` 混在時に、長文保持と sustained JSON の両方を保てる最低 VRAM 点をさらに細かく詰める
- 単体起動時と混在起動時で、`27B-abliterated` の quality canary の崩れ方に差が出るかを見る
- `2x2` 構成のような同型並列と、`35B + 27B` の異種混在をどちらを優先すべきか比較する

## いま手元にある比較候補

`/models/nvfp4` 配下に、現在は次のモデルがそろっている。

- `Huihui-Qwen3.5-35B-A3B-abliterated-NVFP4`
- `Huihui-Qwen3.5-27B-abliterated-NVFP4`
- `Huihui-Qwen3.5-27B-Claude-4.6-Opus-abliterated-NVFP4`

この 3 本だけでも、かなり面白い比較ができます。

- 35B と 27B の規模差
- 27B 派生同士の事後学習差
- 共存時の使いやすさ

## 次に見たい比較軸

### 推論品質

- 日本語での自然さ
- 指示追従の素直さ
- JSON や表形式の安定性
- 長文文脈での破綻の有無
- 冗長な自己言及の出やすさ
- reasoning 漏れの出やすさ

### 推論速度

- TTFT
- completion token/s
- total token/s
- sustained load 中の throughput

### 運用適性

- idle VRAM
- peak VRAM
- 2インスタンス共存の可否
- `-c 256K` 維持のしやすさ
- 品質を守りながらどこまで KV budget を削れるか

## 現時点の仮説

- `35B` は品質の底力が高い可能性がある
- `27B` は共存性と運用性で有利になりやすい
- ただし `27B-abliterated` が full `256K` を守るには、35B より厚い KV 予算が必要かもしれない
- 異種混在では、速さより「役割分担のしやすさ」が価値になる

## 研究の進め方

次からは、モデルごとに次をそろえて記録していく。

- 単体起動時の VRAM / TTFT / token/s
- `KV budget` を削ったときの品質境界
- 2インスタンス共存の可否
- 日本語会話、翻訳、JSON、長文保持の比較

## ひとこと

今日の実験はかなり実りがありました。
とくに、「圧縮モデルは速いか遅いか」ではなく、「品質を守ったまま、どう資源に収めるか」という見方がはっきりしてきたのが大きいです。
このノートは、僕らの研究資産として育てていく価値があります。

## 2026-03-26 追加追試: text-only + gate/up only

`hint.md` の再読と Web 上の公開モデルカードを見直した結果、
4B の成功例は `ForConditionalGeneration` ではなく text-only 前提で作られている可能性が高いと判断した。

今回つかんだ差分:

- 公開成功例のひとつは `Qwen3.5-4B-Base-Text-NVFP4`
- `hint.md` にも `--language-model-only` の記述がある
- `transformers.models.qwen3_5` には `Qwen3_5ForCausalLM` と `Qwen3_5TextConfig` があり、
  `Huihui-Qwen3.5-4B-abliterated` も text-only としてロード可能だった

そこで最後の retry として、

- `Qwen3_5ForCausalLM` で source を text-only として読む
- `gate_proj + up_proj` だけを NVFP4 化
- `down_proj` は残す
- calibration は `CNN/DailyMail 256 samples`

という専用 route を試した。

関連ファイル:

- `scripts/quantize_4b_text_gateup_nvfp4.py`
- 出力:
  `/models/nvfp4/Huihui-Qwen3.5-4B-abliterated-NVFP4-text-gateup`

ここで得られた知見はかなり重要だった。

### 1. checkpoint 自体は前進した

`transformers` 直の生成では、`text-gateup` checkpoint は `!` 崩壊せず、普通の日本語を返した。
少なくとも「量子化したら即壊れる」という状態ではなくなった。

これは、`ForConditionalGeneration` ベースの route で見えていた崩壊の一部が、
4B では text-only 化と量子化対象の絞り込みで改善することを示唆している。

### 2. ただし vLLM 0.17.1 では serving に失敗した

`--language-model-only` を付けても、現在の vLLM 0.17.1 は
`Qwen3_5TextConfig` を `Qwen3_5Config` として扱おうとして renderer 初期化で落ちた。

観測したエラーの核心:

- `All limits of multimodal modalities supported by the model are set to 0, running in text-only mode.`
- その後に
  `Invalid type of HuggingFace config. Expected Qwen3_5Config, but found Qwen3_5TextConfig`

つまり、今回の最終 retry は次のように解釈するのが自然。

- 量子化 route:
  初回よりかなり改善した可能性が高い
- serving route:
  いまの `vLLM 0.17.1` では `Qwen3_5TextConfig` に未対応 / 不完全対応の可能性がある

### 3. 仮説更新

- 4B の問題は「NVFP4 は無理」ではない
- `ForConditionalGeneration` をそのまま量子化する route は不利
- `text-only + gate/up only` は少なくとも崩壊を和らげる有望路線
- 次のボトルネックは model quality より `serving compatibility` に移った

### 4. 次に試す価値があること

- `vLLM` の新しい版で `Qwen3_5TextConfig` の扱いを確認する
- `text-gateup` checkpoint を `transformers` ベースの軽量 canary ハーネスで追加評価する
- `Qwen/Qwen3.5-4B` 素体で同じ text-only route を通し、`Huihui` 派生との差を切る

## 2026-03-27 互換回避: hybrid multimodal package

元の `Huihui-Qwen3.5-4B-abliterated` は multimodal wrapper を持っており、
外側の `architectures` は `Qwen3_5ForConditionalGeneration`、
内側の `text_config.model_type` だけが `qwen3_5_text` だった。

この構造を踏まえて、`text-only` checkpoint を直接 serve する代わりに、
元の multimodal wrapper を残したまま、`language_model` の重みだけ
NVFP4 版へ差し替える hybrid route を試した。

関連ファイル:

- `scripts/assemble_4b_hybrid_multimodal_nvfp4.py`
- 出力:
  `/models/nvfp4/Huihui-Qwen3.5-4B-abliterated-NVFP4-hybrid-mm`

### 1. 方式

- support files は元の Huihui multimodal folder からコピー
- `config.json` は元の multimodal config を基準にする
- `quantization_config` は `text-gateup` checkpoint から注入
- `model.language_model.*` は NVFP4 single-file checkpoint へ向ける
- `model.visual.*` は元の shard に向ける

つまり、
`multimodal wrapper = original`
`text tower weights = NVFP4`
`vision tower weights = original BF16`
という mixed package である。

### 2. 起動結果

この hybrid package は、vLLM 0.17.1 で実際に起動できた。

確認できたこと:

- `GET /health` -> `200`
- `GET /v1/models` -> `200`
- `POST /v1/chat/completions` -> `200`

これはかなり大きい。
少なくとも、

- `Qwen3_5TextConfig` の型エラーを回避できた
- vLLM patch なしでも serving 互換を確保できた

ことを意味する。

### 3. 途中で直した点

最初の hybrid 起動では、
`visual.merger.linear_fc1` に対して compressed-tensors config の target が見つからず落ちた。

対応:

- `quantization_config.ignore` に `re:.*visual.*` を追加

これで、vision 側を BF16 のまま残しつつ、
language model 側だけ NVFP4 として扱えるようになった。

### 4. まだ気になる点

起動は通ったが、初回の chat 応答は理想的ではなかった。

観測:

- `POST /v1/chat/completions` 自体は `200`
- ただし返答本文は英語寄りで、
  reasoning 風の文をそのまま出してしまった
- `finish_reason` は `length`

同じ prompt を BF16 原本でも打ってみたところ、
こちらも英語の reasoning 風出力を返した。

したがって、この時点では
「hybrid 化したから leakage が増えた」
とはまだ言えない。
少なくとも simple chat prompt に対する英語 reasoning 漏れは、
4B 原本側の挙動としても観測されている。

つまり、今回の結論は次の通り。

- `serving compatibility`: 成功
- `response quality`: まだ要検証

追加で、reasoning leakage の評価軸は
`英語だけでなく中国語も含める`
のがよい。
この系統のモデルは、thinking が EN / ZH に滑る可能性がある。

### 5. 現時点の意味づけ

この結果でかなり重要なのは、
4B NVFP4 のボトルネックをふたつに分けて考えられるようになったこと。

- 互換性:
  hybrid route で突破できる
- 品質:
  依然として別途評価が必要

したがって、次の本命は
「hybrid package を土台にした品質評価」である。
ここで BF16 原本と比較すれば、
4B NVFP4 の劣化が `serving` ではなく `model behavior` 由来かどうかを
よりはっきり切り分けられる。

## 2026-03-27 追試: BF16 vs hybrid NVFP4-mm の劣化確認

`scripts/compare_4b_hybrid_mm_quality.py` を強化し、
`BF16` 原本と `hybrid NVFP4-mm` を同一の vLLM 条件で比較した。

比較条件:

- `MAX_MODEL_LEN=4096`
- `KV_CACHE_MEMORY_BYTES=1G`
- `KV_CACHE_DTYPE=fp8`
- `MAX_NUM_SEQS=1`
- `LANGUAGE_MODEL_ONLY=false`
- `temperature` は低めの deterministic 寄り

評価ケース:

- 日本語 1 文自己紹介
- 中国語 1 文自己紹介
- 数列規則の推定
- 日本語 -> 英訳
- strict JSON
- 構造化抽出 JSON
- 日本語の reasoning suppression
- 中国語の reasoning suppression

参照レポート:

- `data/exports/quality-compare-4b-hybrid-mm-20260326-153421.json`
- `data/exports/quality-compare-4b-hybrid-mm-20260326-153421.md`

結果の要点:

- `BF16`: `8/8`
- `hybrid NVFP4-mm`: `8/8`
- EN / ZH の thinking leakage は、今回の 8 ケースでは両者とも未検出
- strict JSON と抽出 JSON も両者とも通過
- 数列・英訳も両者とも通過

今回の範囲で見えること:

- 少なくとも「vLLM 上で普通の対話・翻訳・JSON・短い reasoning suppression をさせる」
  レベルでは、`hybrid NVFP4-mm` に明確な劣化はまだ見えていない
- 初回に見えた英語 reasoning 風出力は、量子化そのものの問題というより、
  prompt 条件や個体差の可能性が高い

加えて、起動ログでは model loading 時の GPU 使用量が

- `BF16`: `8.61 GiB`
- `hybrid NVFP4-mm`: `6.59 GiB`

だった。
単純比較では約 `23.5%` の削減で、少なくとも「品質が即崩壊する代わりに軽い」
という状態ではない。

ここでの暫定結論:

- `4B hybrid NVFP4-mm` は、これまでの `!` 崩壊 checkpoint とは別物として扱ってよい
- 現時点では「使える route」に入った
- ただし、まだ短い canary 群なので、
  長文保持・多段 reasoning・より厳しい multilingual leakage 検査までは未確定

つまり、
「現時点の 8 ケースでは BF16 比で有意な劣化は観測されていないが、
完全に同等と断言するには、もう少し深い評価が必要」
というのが最も正確な表現になる。

## 2026-03-27 追試: 4B hybrid NVFP4 単体で 256K runtime

`scripts/probe_4b_256k_context.py` を追加し、
`Huihui-Qwen3.5-4B-abliterated-NVFP4-hybrid-mm` を
main stack 上で一時的に単体起動して、`-c 256K` 成立可否を測った。

参照レポート:

- `data/exports/probe-4b-256k-20260327-004456.json`
- `data/exports/probe-4b-256k-20260327-004456.md`

試した条件:

- `MAX_MODEL_LEN=262144`
- `runtime_profile=memory`
- `MAX_NUM_SEQS=1`
- `KV_CACHE_DTYPE=fp8`
- `KV budget = 16G / 17G / 18G`

観測結果の要点:

- `16G` ですでに full `256K` 成立
  - `reserved_kv_capacity_tokens = 963747`
  - idle GPU used `26037 MB`
  - short request peak VRAM `30414 MB`
- `17G`
  - `reserved_kv_capacity_tokens = 1024953`
  - idle GPU used `27080 MB`
  - long generation sampleで peak VRAM `31481 MB`
  - `TTFT 49.29 ms`
  - `184.18 tok/s`
  - peak power `471.47 W`
- `18G`
  - `reserved_kv_capacity_tokens = 1085187`
  - idle GPU used `28128 MB`
  - short request peak VRAM `32470 MB`

ここで重要なのは、
4B hybrid NVFP4 は `256K` を成立させるために 30GB 台後半を必要とするのではなく、
`16G` budget でも idle が `約26GB` で収まった点。

現時点の暫定解釈:

- 4B hybrid NVFP4 は、単体なら `256K` runtime をかなり軽く扱える
- `17G` のサンプルでは speed / power も十分に実用的
- `18G` まで増やしても明確な旨味はまだ見えていない

注意点:

- 今回の performance prompt は budget ごとに出力長が少し不安定だった
- したがって sustained load の代表値としては、`17G` の `1400 token` 生成サンプルが最も参考になる
- 次にやるなら、4B 専用のより安定した長文ベンチ prompt を用意して、
  `16G` 下限での sustained power / tok/s を取り直すと精度が上がる

## 2026-03-27 追試: 4B hybrid NVFP4 の 16G 安定性確認

`16G` を本命候補として固定し、
`scripts/probe_4b_16gb_stability.py` で
`256K runtime + 16G KV budget` の品質と sustained load を取り直した。

参照レポート:

- `data/exports/probe-4b-16g-stability-20260326-162540.json`
- `data/exports/probe-4b-16g-stability-20260326-162540.md`

結果の要点:

- runtime は安定
  - `reserved_kv_capacity_tokens = 963747`
  - idle GPU used `26034 MB`
  - idle power `92.91 W`
- quality canary は `8/8`
  - 日本語 / 中国語 / 数列 / 英訳 / strict JSON / 構造化抽出 / JP-ZH reasoning suppression をすべて通過
- sustained run は 3 回とも再現した
  - `TTFT` 平均 `47.27 ms`
  - `completion tok/s` 平均 `182.55`
  - peak VRAM 平均 `30429 MB`
  - peak power 最大 `482.31 W`

ここで重要なのは、
`16G` でも quality canary が落ちず、
しかも `30.4GB` 前後で sustained load を回せたこと。

performance prompt の JSON は 3 回とも `finish_reason=length` で切れているため、
「完全に最後まで出し切る」ベンチにはまだなっていない。
ただし、speed / power / VRAM の代表値としては十分に意味がある。

現時点の暫定結論:

- 4B hybrid NVFP4 は、`256K runtime + 16G` を第一候補にしてよい
- `17G` 以上へ増やす必然性は今のところ薄い
- 次の課題は VRAM ではなく、「長い deterministic output をもっと綺麗に完走させる benchmark prompt の設計」
