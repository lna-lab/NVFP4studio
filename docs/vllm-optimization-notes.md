# vLLM Optimization Notes

このメモは、ある 35B 級 NVFP4 モデルを `vLLM` で最適化した過程と知見を残すためのノートです。
結論だけでなく、途中で外した仮説や再確認が必要な点も残します。

## 最初に結論

- 35B 級 NVFP4 では、VRAM 差の主因はモデル重みそのものより `KV cache` の予約量でした。
- `gpu_memory_utilization` を下げるだけでも効きますが、`--kv-cache-memory-bytes` を明示した方が効きが大きいです。
- 現時点の単ユーザー暫定ベストは:
  `memory` profile + `MAX_MODEL_LEN=262144` + `KV_CACHE_MEMORY_BYTES=4G`
- この設定では、quality canary `3/3` を維持したまま peak VRAM を約 `34.2GB` まで下げられました。
- `3G` はさらに VRAM を下げられましたが、quality canary が `2/3` に落ちたため不採用です。
- `2インスタンス x 2シーケンス` は成立しましたが、総 throughput を伸ばす設定というより multi-tenant 隔離向きでした。

## 397B / A17B 速度メモ

- `huihui-ai/Huihui-Qwen3.5-397B-A17B-abliterated-NVFP4` を `4x RTX PRO 6000 Blackwell` で回したとき、遅さの主因は GPU 能力不足ではなく `--enforce-eager` を含む保守設定だった。
- `MAX_MODEL_LEN=16384` + `KV_CACHE_MEMORY_BYTES=8G` + `TENSOR_PARALLEL_SIZE=4` のまま、`ENFORCE_EAGER=false` と `DISABLE_CUSTOM_ALL_REDUCE=false` に切り替えるだけで、長文 streaming は約 `18.5 tok/s` から約 `81.4-83.0 tok/s` まで上がった。
- ただしこの 4GPU PCIe 構成では、vLLM ログ上で custom all-reduce は非対応として自動で NCCL にフォールバックした。つまり実際の主因は `ENFORCE_EAGER=false` 側にある。
- `speed` profile 相当の厚い予約でも動いたが、本検証では約 `78.3 tok/s` で、最速ではなかった。
- `flashinfer_cutedsl + autotune` は worker 初期化が不安定で、現時点では既定値にしない。
- したがって、397B / A17B の現時点のローカル既定値は「`16K / 8G / TP4` を維持しつつ、non-eager + custom all-reduce 有効」を採る。

## 目的

- `-c 256K` を維持したまま VRAM 使用量をできるだけ削る
- 単ユーザー運用を前提に、token/s と TTFT を大きく崩さない
- 将来の `397B NVFP4` の検討に向けて、35B で効くレバーを先に整理する

## 現在採用している設定

- 現在の推奨設定は `memory` profile です。
- 安定している値は次の通りです。
  - `MAX_MODEL_LEN=262144`
  - `GPU_MEMORY_UTILIZATION=0.45`
  - `MAX_NUM_SEQS=1`
  - `MAX_NUM_BATCHED_TOKENS=4096`
  - `KV_CACHE_DTYPE=fp8`
  - `KV_CACHE_MEMORY_BYTES=4G`
  - `CPU_OFFLOAD_GB=0`
  - `SWAP_SPACE=16`
- ここまでは README に反映してよい結論として扱います。
- ただし 397B / A17B の速度重視運用では、同じ `memory` 系の予約方針でも `ENFORCE_EAGER=false` と `DISABLE_CUSTOM_ALL_REDUCE=false` を既定にする。

## README に反映している判断

- `speed + -c 256K` は約 `90.3GB`
- `balanced + -c 256K` は約 `65.3GB`
- `memory + -c 256K + KV budget 4G` は peak 約 `34.2GB` で canary `3/3`
- `memory + -c 256K + KV budget 3G` は canary `2/3` で不採用
- `2x2` は成立するが、throughput 強化策としては扱わない

## 実測メモ

### 397B / A17B の speed sweep

- 検証対象
  - モデル: `huihui-ai/Huihui-Qwen3.5-397B-A17B-abliterated-NVFP4`
  - GPU: `RTX PRO 6000 Blackwell x4`
  - context: `16384`
  - long streaming benchmark: `max_tokens=512`
- 参照 benchmark record
  - `id=66`
    - baseline safe
    - `ttft_ms=311.07`
    - `completion_tokens_per_sec=18.5252`
  - `id=68`
    - non-eager 系 memory path
    - `ttft_ms=332.85`
    - `completion_tokens_per_sec=81.4049`
  - `id=70`
    - non-eager 系 memory path
    - `ttft_ms=339.85`
    - `completion_tokens_per_sec=82.9839`
  - `id=72`
    - `speed` profile 寄り
    - `ttft_ms=341.65`
    - `completion_tokens_per_sec=78.2765`
- 解釈
- `18 tok/s` 台だった経路でも、GPU 電力は十分に使い切れておらず、CPU フォールバックではなかった。
- 速度の大部分は `eager` と通信経路の設定で決まり、VRAM をさらに厚く予約しても必ずしも速くはならない。
- `DISABLE_CUSTOM_ALL_REDUCE=false` を維持しても、この構成では結果的に NCCL フォールバックなので、ここは「明示 disable しない」程度の意味に留まる。
- このモデルでは、今のところ「軽い予約 + non-eager」が一番実用的だった。

### 比較の軸

- `speed + -c 256K`
  - 使用 VRAM は約 `90.3GB`
- `balanced + -c 256K`
  - 使用 VRAM は約 `65.3GB`
  - `speed` 比で約 `27.7%` 節約
- `memory + -c 256K + KV budget 8G`
  - streaming 長文ベンチでは `peak_vram_used_mb=49628`
  - 同ベンチで `ttft_ms=107.7`, `completion_tokens_per_sec=157.7565`
  - 短い非 streaming リクエストでは `peak_vram_used_mb=38174`
  - 短い streaming リクエストでは `peak_vram_used_mb=38161`, `ttft_ms=92.44`, `completion_tokens_per_sec=162.5271`
- `memory + -c 256K + KV budget 6G`
  - probe では `idle_vram_mb=31751`
  - `peak_vram_used_mb=36168`
  - `ttft_ms=83.96`
  - `completion_tokens_per_sec=161.6695`
  - quality canary `3/3`
- `memory + -c 256K + KV budget 4G`
  - probe では `idle_vram_mb=29836`
  - `peak_vram_used_mb=34218`
  - `ttft_ms=78.43`
  - `completion_tokens_per_sec=166.609`
  - quality canary `3/3`
- `memory + -c 256K + KV budget 3G`
  - probe では `idle_vram_mb=29169`
  - `peak_vram_used_mb=33298`
  - `ttft_ms=80.7`
  - `completion_tokens_per_sec=167.0317`
  - quality canary が `2/3` で、数列問題を誤答したため不採用

### 2インスタンス x 2シーケンスの比較

- 比較条件
  - `KV_CACHE_MEMORY_BYTES=6G`
  - `MAX_MODEL_LEN=262144`
  - `MAX_NUM_SEQS=2`
  - `GPU_MEMORY_UTILIZATION=0.45`
  - `KV_CACHE_DTYPE=fp8`
- 参照レポート
  - `data/exports/parallel-instance-probe-1x2-20260326-182836.json`
  - `data/exports/parallel-instance-probe-2x2-20260326-183111.json`
- `1インスタンス x 2シーケンス`
  - idle `35.6GB`
  - 平均電力 `313.0W`
  - peak 電力 `325.1W`
  - aggregate throughput `252.1 tok/s`
  - sustained exact JSON `2/2 PASS`
- `2インスタンス x 2シーケンス`
  - idle `70.3GB`
  - 各インスタンスの process memory `34.7GB`
  - 各インスタンスの reserved_kv_capacity_tokens `628800`
  - つまり `2 * 256K` を各インスタンス内で確保できた
  - 平均電力 `320.6W`
  - peak 電力 `343.6W`
  - aggregate throughput `244.2 tok/s`
  - sustained exact JSON `4/4 PASS`
- 解釈
  - `2x2` は VRAM 制約の観点では成立する
  - ただし raw throughput は `1x2` を上回らなかった
  - `tokens/s per watt` は `0.8054 -> 0.7617` で、電力効率は約 `5%` 悪化
  - したがって `2x2` は「速度を増やす設定」ではなく、「1枚の GPU 上に 2 テナントを無理なく共存させる設定」と解釈するのが妥当

### 読み方

- `peak_vram_used_mb` はベンチごとのピーク値
- idle の `nvidia-smi` と、長文 streaming 中のピークは一致しない
- 比較するときは、なるべく benchmark record 側の `peak_vram_used_mb` を基準にする

## 確認できたこと

- live process の実引数では `--kv-cache-memory-bytes 8G` が渡っている
- live process の実引数では `6G`, `4G`, `3G` も個別に確認できた
- 現在の live status は次の通り
  - `runtime_profile=memory`
  - `runtime_max_context=262144`
  - `kv_cache_memory_bytes=4G`
  - `num_gpu_blocks=200`
  - `block_size=2096`
  - `gpu_memory_utilization=0.45`
  - `reserved_kv_capacity_tokens=385664`
- つまり `256K` を維持しながら、品質を落とさず予約 VRAM を大きく削れた

## 重要な解釈

- モデル本体は過去ログ上で約 `21.9GiB`
- それ以外の差分は主に `KV cache`、workspace、runtime 予約
- したがって、35B の最適化では
  - 重み量子化を疑うより
  - `KV cache` の予約戦略を詰める
  方が先に効く

## 失敗・注意点

- `CPU_OFFLOAD_GB=16` はこの環境の `vLLM 0.17.1` だと
  `Cannot re-initialize the input batch when CPU weight offloading is enabled`
  の assertion に当たりやすく、`256K` の `memory` profile では不採用
- `6G / 4G / 3G / 2G` の追加 sweep は一度実行したが、途中で host compose の呼び出しミスが混ざった
  - この sweep 全体をそのまま採用するのではなく、個別に再確認できた値だけを README や preset に反映する
  - 現時点では `4G 採用 / 3G 不採用 / 2x2 は隔離向き` までを公開向けの結論とする

## 次に試す価値があること

- `2G` は reserved tokens 的に `256K` 運用の余裕が薄いので、試すとしても品質劣化前提の境界確認として扱う
- `KV budget` を UI/status に明示表示して、今どの budget で動いているか分かるようにする
- `397B` を見据えるなら、35B 側では次の優先度で検討する
  - `KV_CACHE_MEMORY_BYTES`
  - `kv offloading`
  - MoE なら `expert parallel`
  - `2x2` はテナント隔離には効くが、総 throughput を増やす手段ではないことを前提に設計する

## 品質 guardrail

- `scripts/probe_kv_budget.py` を追加した
- `scripts/probe_dual_instance.py` を追加した
- `scripts/probe_speed_paths.py` を追加した
- この script は budget ごとに
  - vLLM 再構成
  - process cmdline 検証
  - quality canary 3本
  - short streaming benchmark
  - JSON / Markdown 出力
  までを自動化する
- `probe_dual_instance.py` はさらに
  - `1x2` と `2x2` の比較
  - sustained exact JSON workload
  - 平均電力 / peak 電力 / aggregate throughput の記録
  までを扱える
- 現時点の解釈は次の通り
  - `8G`: 安定
  - `6G`: 安定
  - `4G`: 安定かつ最良
  - `3G`: 速度は維持したが quality canary が崩れたため不採用
  - `397B / A17B`: `16K / 8G / TP4` でも non-eager 化で `80 tok/s` 級まで伸びた

## 参照した benchmark record

- `id=8`
  - `peak_vram_used_mb=90313`
- `id=10`
  - `peak_vram_used_mb=63815`
  - `ttft_ms=241.11`
  - `completion_tokens_per_sec=151.2487`
- `id=13`
  - `peak_vram_used_mb=49628`
  - `ttft_ms=107.7`
  - `completion_tokens_per_sec=157.7565`
- `id=14`
  - `peak_vram_used_mb=38174`
- `id=15`
  - `peak_vram_used_mb=38161`
  - `ttft_ms=92.44`
  - `completion_tokens_per_sec=162.5271`
- probe `20260326-171539`
  - budget `6G`
  - canary `3/3`
- probe `20260326-171826`
  - budget `4G`
  - canary `3/3`
- probe `20260326-172119`
  - budget `3G`
  - canary `2/3`

## ひとことメモ

- 35B の段階でも、`256K` を維持したまま VRAM を大きく削る余地はまだある
- ただし、結論を急がず、`recreate` が本当に通ったかを `process cmdline` まで含めて確認すること
- 速度だけでは足りない。品質 canary を通った budget だけを採用すること
