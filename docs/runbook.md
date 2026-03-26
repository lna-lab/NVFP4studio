# Runbook

## 1. 初回セットアップ

```bash
cd /media/shinkaman/INTEL_TUF/Sefetensors/NVFP4studio
./scripts/bootstrap.sh
```

確認内容:

- `docker`
- `docker compose`
- `nvidia-smi`
- NVIDIA runtime
- `.env` 生成
- イメージ pull / build

## 2. 起動

```bash
./scripts/start.sh
```

ヘルス確認:

```bash
./scripts/healthcheck.sh
```

## 3. 停止

```bash
./scripts/stop.sh
```

## 4. 再起動

```bash
./scripts/restart.sh
```

## 5. ログ確認

```bash
docker compose logs -f vllm
docker compose logs -f gateway
docker compose logs -f frontend
```

ファイルログ:

- `data/logs/vllm.log`

## 6. よくある問題

### モデルが見つからない

- `.env` の `MODEL_PATH` を確認
- ホスト上にディレクトリが存在するか確認

### `docker compose up` で GPU が見えない

- `nvidia-smi`
- `docker run --rm --gpus all nvidia/cuda:13.0.1-base-ubuntu24.04 nvidia-smi`
- `data/logs/gpu-smoke.log`
- NVIDIA Container Toolkit を確認

現在の環境で次のエラーが出る場合:

```text
could not select device driver "" with capabilities: [[gpu]]
```

Docker の GPU runtime が未設定です。NVIDIA 公式ドキュメントの手順に沿って、概ね次を実行します。

```bash
sudo apt-get update && sudo apt-get install -y --no-install-recommends ca-certificates curl gnupg2
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.19.0-1
sudo apt-get install -y \
  nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
  nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
  libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
  libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:13.0.1-base-ubuntu24.04 nvidia-smi
```

その後に再度:

```bash
./scripts/bootstrap.sh
./scripts/start.sh
```

### `/v1/models` が 200 にならない

- vLLM の `/health` を先に確認
- `docker compose logs vllm`

### UI のメトリクスが空

- streaming リクエストか確認
- `GET /api/benchmarks/request/{request_id}` を確認
- vLLM usage chunk が返らないケースでは usage が空になることがある
