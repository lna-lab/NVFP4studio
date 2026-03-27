#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT_DIR / ".env"
COMPOSE_FILE = ROOT_DIR / "docker-compose.yml"
PROJECT_NAME = "nvfp4studio"
OUT_DIR = ROOT_DIR / "data" / "exports"
MODELS_ROOT = Path("/models/nvfp4")
MODEL_PORT = 8010
sys.path.append(str(ROOT_DIR / "backend"))

from app.telemetry.prometheus import parse_metrics  # type: ignore  # noqa: E402

INSTANCE_SPECS = [
    {"name": "mix-a", "container": "nvfp4studio-mixed-a", "host_port": 8210},
    {"name": "mix-b", "container": "nvfp4studio-mixed-b", "host_port": 8211},
]

MODEL_TASKS = {
    "chat": {
        "system": "Reply in Japanese. Keep it concise. Do not reveal hidden reasoning.",
        "user": "こんにちは。あなたの役割を一言で紹介して。",
        "max_tokens": 128,
    },
    "json": {
        "system": "Return strict JSON only.",
        "user": "mode を local、goal を preserve quality while reducing VRAM にして JSON を返して。",
        "max_tokens": 128,
    },
    "sustained": {
        "system": "Return strict JSON only. Do not include markdown fences or commentary.",
        "user": "JSON object with key pairs. pairs must be an array of 48 objects with keys n and triangular for n=1..48.",
        "max_tokens": 1024,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe heterogeneous 35B + 27B coexistence on a single GPU.")
    parser.add_argument("--model-a", default="Huihui-Qwen3.5-35B-A3B-abliterated-NVFP4")
    parser.add_argument("--budget-a", default="4G")
    parser.add_argument("--model-b", default="Huihui-Qwen3.5-27B-abliterated-NVFP4")
    parser.add_argument("--budget-b", default="10G")
    parser.add_argument("--max-model-len", type=int, default=262144)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--kv-cache-dtype", default="fp8")
    parser.add_argument("--cpu-offload-gb", type=int, default=0)
    parser.add_argument("--swap-space", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=900)
    return parser.parse_args()


def run_command(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)


def read_env() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def stop_primary_vllm() -> None:
    run_command(
        [
            "docker",
            "compose",
            "--env-file",
            str(ENV_PATH),
            "-p",
            PROJECT_NAME,
            "-f",
            str(COMPOSE_FILE),
            "stop",
            "vllm",
        ],
        cwd=ROOT_DIR,
    )


def restore_primary_vllm() -> None:
    run_command(
        [
            "docker",
            "compose",
            "--env-file",
            str(ENV_PATH),
            "-p",
            PROJECT_NAME,
            "-f",
            str(COMPOSE_FILE),
            "up",
            "-d",
            "vllm",
        ],
        cwd=ROOT_DIR,
    )


def remove_container(name: str) -> None:
    subprocess.run(["docker", "rm", "-f", name], check=False, capture_output=True, text=True)


def launch_instance(env: dict[str, str], spec: dict[str, Any], model_path: Path, budget: str, args: argparse.Namespace) -> None:
    log_dir = ROOT_DIR / "data" / "logs" / spec["name"]
    log_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "docker",
        "run",
        "-d",
        "--name",
        spec["container"],
        "--gpus",
        "all",
        "--ipc",
        "host",
        "--shm-size",
        "16gb",
        "-p",
        f"127.0.0.1:{spec['host_port']}:{MODEL_PORT}",
        "-v",
        f"{model_path}:{model_path}:ro",
        "-v",
        f"{ROOT_DIR / 'config' / 'vllm'}:/workspace/config/vllm:ro",
        "-v",
        f"{log_dir}:/workspace/logs",
        "-e",
        f"MODEL_PATH={model_path}",
        "-e",
        f"SERVED_MODEL_NAME={model_path.name}",
        "-e",
        "VLLM_HOST=0.0.0.0",
        "-e",
        f"VLLM_PORT={MODEL_PORT}",
        "-e",
        "TENSOR_PARALLEL_SIZE=1",
        "-e",
        f"MAX_MODEL_LEN={args.max_model_len}",
        "-e",
        f"GPU_MEMORY_UTILIZATION={args.gpu_memory_utilization}",
        "-e",
        f"MAX_NUM_SEQS={args.max_num_seqs}",
        "-e",
        f"MAX_NUM_BATCHED_TOKENS={args.max_num_batched_tokens}",
        "-e",
        f"KV_CACHE_DTYPE={args.kv_cache_dtype}",
        "-e",
        f"KV_CACHE_MEMORY_BYTES={budget}",
        "-e",
        f"CPU_OFFLOAD_GB={args.cpu_offload_gb}",
        "-e",
        f"SWAP_SPACE={args.swap_space}",
        "-e",
        f"TRUST_REMOTE_CODE={env.get('TRUST_REMOTE_CODE', 'true')}",
        "--entrypoint",
        "/bin/bash",
        "nvfp4studio-vllm:latest",
        "/workspace/config/vllm/start-vllm.sh",
    ]
    run_command(command, cwd=ROOT_DIR)


def fetch_json(url: str, payload: dict[str, Any] | None = None, *, timeout: int = 600) -> tuple[dict[str, Any], Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
        return body, response


def wait_for_instance(spec: dict[str, Any], expected_budget: str, expected_model_path: str, timeout: int) -> str:
    deadline = time.time() + timeout
    last_error = ""
    while time.time() < deadline:
        try:
            state = run_command(["docker", "inspect", "-f", "{{.State.Status}}", spec["container"]]).stdout.strip()
            if state not in {"created", "running"}:
                logs = subprocess.run(["docker", "logs", spec["container"]], check=False, capture_output=True, text=True).stdout
                raise RuntimeError(f"{spec['container']} state={state}\n{logs}")
            with urllib.request.urlopen(f"http://127.0.0.1:{spec['host_port']}/health", timeout=5):
                pass
            cmdline = run_command(["docker", "top", spec["container"], "-eo", "pid,ppid,cmd"]).stdout
            if f"--kv-cache-memory-bytes {expected_budget}" in cmdline and expected_model_path in cmdline:
                return cmdline
            last_error = cmdline
        except Exception as exc:
            last_error = str(exc)
        time.sleep(5)
    raise TimeoutError(f"{spec['container']} did not become ready: {last_error}")


def container_gpu_memory_mb(container_name: str) -> int:
    top_output = run_command(["docker", "top", container_name, "-eo", "pid,ppid,cmd"]).stdout.splitlines()[1:]
    pids = {line.split(None, 2)[0] for line in top_output if line.strip()}
    smi = run_command(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"]).stdout.splitlines()
    total = 0
    for line in smi:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        if parts[0] in pids:
            total += int(parts[1])
    return total


def total_gpu_snapshot() -> dict[str, float]:
    line = run_command(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free,power.draw,power.limit,utilization.gpu", "--format=csv,noheader,nounits"]
    ).stdout.splitlines()[0]
    used, free, power, limit, util = [part.strip() for part in line.split(",")]
    return {
        "memory_used_mb": float(used),
        "memory_free_mb": float(free),
        "power_draw_watts": float(power),
        "power_limit_watts": float(limit),
        "utilization_gpu_percent": float(util),
    }


def extract_text(body: dict[str, Any]) -> str:
    content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
    return content.strip() if isinstance(content, str) else ""


def validate(task_name: str, text: str) -> tuple[bool, str]:
    normalized = text.strip()
    lowered = normalized.lower()
    if task_name == "chat":
        return ("こんにちは" in normalized or "私は" in normalized), "日本語の自己紹介として成立しているか"
    if task_name == "json":
        try:
            payload = json.loads(normalized)
        except json.JSONDecodeError:
            return False, "JSON として解釈できない"
        return isinstance(payload, dict) and set(payload.keys()) == {"mode", "goal"}, "mode と goal のみを持つ JSON かを確認"
    if task_name == "sustained":
        try:
            payload = json.loads(normalized)
        except json.JSONDecodeError:
            return False, "strict JSON として解釈できない"
        expected = {"pairs": [{"n": n, "triangular": n * (n + 1) // 2} for n in range(1, 49)]}
        return payload == expected, "1..48 の triangular table が一致するかを確認"
    return False, "validator not found"


def run_request(spec: dict[str, Any], model_name: str, task_name: str, *, timeout: int = 900) -> dict[str, Any]:
    task = MODEL_TASKS[task_name]
    payload = {
        "model": model_name,
        "stream": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": task["max_tokens"],
        "messages": [
            {"role": "system", "content": task["system"]},
            {"role": "user", "content": task["user"]},
        ],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    started = time.time()
    body, response = fetch_json(f"http://127.0.0.1:{spec['host_port']}/v1/chat/completions", payload, timeout=timeout)
    elapsed_ms = round((time.time() - started) * 1000, 2)
    output = extract_text(body)
    passed, note = validate(task_name, output)
    usage = body.get("usage") or {}
    completion_tokens = usage.get("completion_tokens") or 0
    tok_per_s = round(completion_tokens / max(0.001, elapsed_ms / 1000), 4) if completion_tokens else None
    return {
        "instance": spec["name"],
        "task": task_name,
        "request_id": response.headers.get("x-nvfp4studio-request-id"),
        "elapsed_ms": elapsed_ms,
        "completion_tokens": completion_tokens,
        "tok_per_s": tok_per_s,
        "output": output,
        "passed": passed,
        "note": note,
    }


def sample_gpu(stop_event: threading.Event, sink: list[dict[str, float]]) -> None:
    while not stop_event.is_set():
        try:
            sink.append(total_gpu_snapshot())
        except Exception:
            pass
        time.sleep(0.2)


def main() -> int:
    args = parse_args()
    env = read_env()
    model_a = MODELS_ROOT / args.model_a
    model_b = MODELS_ROOT / args.model_b
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = OUT_DIR / f"mixed-model-probe-{timestamp}.json"
    md_path = OUT_DIR / f"mixed-model-probe-{timestamp}.md"

    report: dict[str, Any] = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": {
            "model_a": args.model_a,
            "budget_a": args.budget_a,
            "model_b": args.model_b,
            "budget_b": args.budget_b,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "kv_cache_dtype": args.kv_cache_dtype,
        },
        "restored_primary_vllm": False,
    }

    try:
        stop_primary_vllm()
        for spec in INSTANCE_SPECS:
            remove_container(spec["container"])
        launch_instance(env, INSTANCE_SPECS[0], model_a, args.budget_a, args)
        launch_instance(env, INSTANCE_SPECS[1], model_b, args.budget_b, args)
        wait_for_instance(INSTANCE_SPECS[0], args.budget_a, str(model_a), args.timeout)
        wait_for_instance(INSTANCE_SPECS[1], args.budget_b, str(model_b), args.timeout)

        warmup_a = run_request(INSTANCE_SPECS[0], model_a.name, "chat")
        warmup_b = run_request(INSTANCE_SPECS[1], model_b.name, "chat")

        states = []
        for spec, model_name in zip(INSTANCE_SPECS, [model_a.name, model_b.name], strict=True):
            metrics_text = urllib.request.urlopen(f"http://127.0.0.1:{spec['host_port']}/metrics", timeout=30).read().decode("utf-8")
            metrics = parse_metrics(metrics_text)
            states.append(
                {
                    "name": spec["name"],
                    "model_name": model_name,
                    "gpu_process_memory_mb": container_gpu_memory_mb(spec["container"]),
                    "reserved_kv_capacity_tokens": int(metrics.get("vllm:cache_config_info:block_size", 0))
                    * int(metrics.get("vllm:cache_config_info:num_gpu_blocks", 0)),
                    "block_size": metrics.get("vllm:cache_config_info:block_size"),
                    "num_gpu_blocks": metrics.get("vllm:cache_config_info:num_gpu_blocks"),
                }
            )

        idle_total = total_gpu_snapshot()
        report["idle"] = {"total_gpu": idle_total, "instances": states}
        report["warmup"] = [warmup_a, warmup_b]

        sample_buffer: list[dict[str, float]] = []
        stop_event = threading.Event()
        sampler = threading.Thread(target=sample_gpu, args=(stop_event, sample_buffer), daemon=True)
        sampler.start()
        started = time.time()
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(run_request, INSTANCE_SPECS[0], model_a.name, "sustained"),
                    executor.submit(run_request, INSTANCE_SPECS[1], model_b.name, "sustained"),
                ]
                sustained_results = [future.result() for future in futures]
        finally:
            ended = time.time()
            stop_event.set()
            sampler.join(timeout=2)

        review_results = [
            run_request(INSTANCE_SPECS[0], model_a.name, "json"),
            run_request(INSTANCE_SPECS[1], model_b.name, "json"),
        ]

        report["load"] = {
            "wall_seconds": round(max(0.001, ended - started), 3),
            "results": sustained_results,
            "review_results": review_results,
            "samples": sample_buffer,
            "peak_total_memory_used_mb": max((item["memory_used_mb"] for item in sample_buffer), default=idle_total["memory_used_mb"]),
            "peak_power_draw_watts": max((item["power_draw_watts"] for item in sample_buffer), default=idle_total["power_draw_watts"]),
            "average_power_draw_watts": round(sum(item["power_draw_watts"] for item in sample_buffer) / max(1, len(sample_buffer)), 2),
        }

        report["verdict"] = {
            "combined_under_96gb": idle_total["memory_used_mb"] < 96 * 1024,
            "model_a_supports_256k": states[0]["reserved_kv_capacity_tokens"] >= 262144,
            "model_b_supports_256k": states[1]["reserved_kv_capacity_tokens"] >= 262144,
            "all_sustained_passed": all(item["passed"] for item in sustained_results),
            "all_review_passed": all(item["passed"] for item in review_results),
        }
    finally:
        for spec in INSTANCE_SPECS:
            remove_container(spec["container"])
        restore_primary_vllm()
        report["restored_primary_vllm"] = True

    md_lines = [
        "# Mixed Model Probe",
        "",
        f"- model_a: `{args.model_a}` budget `{args.budget_a}`",
        f"- model_b: `{args.model_b}` budget `{args.budget_b}`",
        f"- idle_total_gpu_memory_used_mb: `{report['idle']['total_gpu']['memory_used_mb']}`",
        f"- peak_total_gpu_memory_used_mb: `{report['load']['peak_total_memory_used_mb']}`",
        f"- average_power_draw_watts: `{report['load']['average_power_draw_watts']}`",
        f"- peak_power_draw_watts: `{report['load']['peak_power_draw_watts']}`",
        f"- combined_under_96gb: `{report['verdict']['combined_under_96gb']}`",
        f"- model_a_supports_256k: `{report['verdict']['model_a_supports_256k']}`",
        f"- model_b_supports_256k: `{report['verdict']['model_b_supports_256k']}`",
        f"- all_sustained_passed: `{report['verdict']['all_sustained_passed']}`",
        f"- all_review_passed: `{report['verdict']['all_review_passed']}`",
        "",
        "## Sustained",
        "",
        "| instance | task | elapsed ms | tok/s | pass |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for item in report["load"]["results"]:
        md_lines.append(f"| {item['instance']} | {item['task']} | {item['elapsed_ms']} | {item['tok_per_s']} | {'PASS' if item['passed'] else 'FAIL'} |")
    md_lines.extend(["", "## Review", "", "| instance | task | pass |", "| --- | --- | --- |"])
    for item in report["load"]["review_results"]:
        md_lines.append(f"| {item['instance']} | {item['task']} | {'PASS' if item['passed'] else 'FAIL'} |")
    md_lines.append("")

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
