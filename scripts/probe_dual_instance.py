#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import re
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
MODEL_PORT = 8010
sys.path.append(str(ROOT_DIR / "backend"))

from app.telemetry.prometheus import parse_metrics  # type: ignore  # noqa: E402

INSTANCE_SPECS = [
    {"name": "probe-a", "container": "nvfp4studio-dual-a", "host_port": 8110},
    {"name": "probe-b", "container": "nvfp4studio-dual-b", "host_port": 8111},
]

CANARY_TASKS = [
    {
        "name": "sequence_rule",
        "system": "Reply in Japanese. One sentence only. Do not reveal hidden reasoning.",
        "user": "数列 2, 6, 12, 20 の次の2項と規則を1文で答えて。",
        "max_tokens": 128,
    },
    {
        "name": "translation_en",
        "system": "Translate to English only. No extra commentary.",
        "user": "品質を保ちながら VRAM を削る。",
        "max_tokens": 128,
    },
    {
        "name": "json_structure",
        "system": "Return strict JSON only.",
        "user": "mode は local、goal は preserve quality while reducing VRAM にして JSON を返して。",
        "max_tokens": 128,
    },
    {
        "name": "short_summary",
        "system": "Reply in Japanese with exactly two short sentences.",
        "user": "単一GPUで2インスタンスを並列運用する狙いを簡潔に述べて。",
        "max_tokens": 128,
    },
]

SUSTAINED_TASKS = [
    {
        "name": "square_pairs_a",
        "system": "Return strict JSON only. Do not include markdown fences or commentary.",
        "user": "JSON object with key pairs. pairs must be an array of 48 objects with keys n and square for n=1..48.",
        "max_tokens": 1024,
    },
    {
        "name": "triangle_pairs_a",
        "system": "Return strict JSON only. Do not include markdown fences or commentary.",
        "user": "JSON object with key pairs. pairs must be an array of 48 objects with keys n and triangular for n=1..48.",
        "max_tokens": 1024,
    },
    {
        "name": "square_pairs_b",
        "system": "Return strict JSON only. Do not include markdown fences or commentary.",
        "user": "JSON object with key pairs. pairs must be an array of 48 objects with keys n and square for n=1..48.",
        "max_tokens": 1024,
    },
    {
        "name": "triangle_pairs_b",
        "system": "Return strict JSON only. Do not include markdown fences or commentary.",
        "user": "JSON object with key pairs. pairs must be an array of 48 objects with keys n and triangular for n=1..48.",
        "max_tokens": 1024,
    },
]

WARMUP_TASK = {
    "name": "warmup",
    "system": "Reply in Japanese. One short sentence only. Do not reveal hidden reasoning.",
    "user": "ウォームアップです。一言だけ返してください。",
    "max_tokens": 64,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe 1x2 or 2x2 vLLM concurrency on a single GPU.")
    parser.add_argument("--instances", type=int, choices=(1, 2), default=2)
    parser.add_argument("--kv-budget", default="6G")
    parser.add_argument("--max-model-len", type=int, default=262144)
    parser.add_argument("--max-num-seqs", type=int, default=2)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
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


def launch_instance(env: dict[str, str], spec: dict[str, Any], args: argparse.Namespace) -> None:
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
        f"{env['MODEL_PATH']}:{env['MODEL_PATH']}:ro",
        "-v",
        f"{ROOT_DIR / 'config' / 'vllm'}:/workspace/config/vllm:ro",
        "-v",
        f"{log_dir}:/workspace/logs",
        "-e",
        f"MODEL_PATH={env['MODEL_PATH']}",
        "-e",
        f"SERVED_MODEL_NAME={env['SERVED_MODEL_NAME']}",
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
        f"KV_CACHE_MEMORY_BYTES={args.kv_budget}",
        "-e",
        f"CPU_OFFLOAD_GB={args.cpu_offload_gb}",
        "-e",
        f"SWAP_SPACE={args.swap_space}",
        "-e",
        "TRUST_REMOTE_CODE=true",
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


def wait_for_instance(spec: dict[str, Any], expected_budget: str, timeout: int) -> dict[str, Any]:
    deadline = time.time() + timeout
    last_error = ""
    while time.time() < deadline:
        try:
            state = run_command(["docker", "inspect", "-f", "{{.State.Status}}", spec["container"]]).stdout.strip()
            if state not in {"created", "running"}:
                logs = subprocess.run(
                    ["docker", "logs", spec["container"]],
                    check=False,
                    capture_output=True,
                    text=True,
                ).stdout
                raise RuntimeError(f"{spec['container']} state={state}\n{logs}")
            with urllib.request.urlopen(f"http://127.0.0.1:{spec['host_port']}/health", timeout=5):
                pass
            cmdline = run_command(["docker", "top", spec["container"], "-eo", "pid,ppid,cmd"]).stdout
            if f"--kv-cache-memory-bytes {expected_budget}" in cmdline:
                return {"cmdline": cmdline}
            last_error = cmdline
        except Exception as exc:  # pragma: no cover - startup timing
            last_error = str(exc)
        time.sleep(5)
    raise TimeoutError(f"{spec['container']} did not become ready: {last_error}")


def container_gpu_memory_mb(container_name: str) -> int:
    top_output = run_command(["docker", "top", container_name, "-eo", "pid,ppid,cmd"]).stdout.splitlines()[1:]
    pids = {line.split(None, 2)[0] for line in top_output if line.strip()}
    smi = run_command(
        ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"]
    ).stdout.splitlines()
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
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.free,power.draw,power.limit,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    ).stdout.splitlines()[0]
    used, free, power, limit, util = [part.strip() for part in line.split(",")]
    return {
        "memory_used_mb": float(used),
        "memory_free_mb": float(free),
        "power_draw_watts": float(power),
        "power_limit_watts": float(limit),
        "utilization_gpu_percent": float(util),
    }


def _expected_pairs(field_name: str, count: int, fn: Any) -> list[dict[str, int]]:
    return [{"n": index, field_name: fn(index)} for index in range(1, count + 1)]


def validate_canary(name: str, text: str) -> tuple[bool, str]:
    normalized = text.strip()
    lowered = normalized.lower()
    if name == "sequence_rule":
        return ("30" in normalized and "42" in normalized), "30 と 42 を含むかを確認"
    if name == "translation_en":
        ok = "quality" in lowered and "vram" in lowered and any(
            token in lowered for token in ("reduce", "lower", "maintain", "preserve")
        )
        return ok, "quality / VRAM と keep-or-reduce 系の語を含むかを確認"
    if name == "json_structure":
        try:
            payload = json.loads(normalized)
        except json.JSONDecodeError:
            return False, "JSON として解釈できない"
        return isinstance(payload, dict) and set(payload.keys()) == {"mode", "goal"}, "mode と goal のみを持つ JSON かを確認"
    if name == "short_summary":
        sentences = [part for part in re.split(r"[。.!?]\s*", normalized) if part.strip()]
        return len(sentences) == 2, "短い2文かを確認"
    return False, "validator not found"


def validate_sustained(name: str, text: str) -> tuple[bool, str]:
    try:
        payload = json.loads(text.strip())
    except json.JSONDecodeError:
        return False, "strict JSON として解釈できない"

    if name.startswith("square_pairs"):
        expected = {"pairs": _expected_pairs("square", 48, lambda n: n * n)}
        return payload == expected, "1..48 の square table が一致するかを確認"
    if name.startswith("triangle_pairs"):
        expected = {"pairs": _expected_pairs("triangular", 48, lambda n: n * (n + 1) // 2)}
        return payload == expected, "1..48 の triangular table が一致するかを確認"
    return False, "validator not found"


def extract_text(body: dict[str, Any]) -> str:
    content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
    if isinstance(content, str):
        return content.strip()
    return ""


def run_request(spec: dict[str, Any], task: dict[str, Any], model_name: str, *, quality_mode: str) -> dict[str, Any]:
    payload = {
        "model": model_name,
        "stream": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": task.get("max_tokens", 128),
        "messages": [
            {"role": "system", "content": task["system"]},
            {"role": "user", "content": task["user"]},
        ],
        "chat_template_kwargs": {
            "enable_thinking": False,
        },
    }
    started = time.time()
    body, _ = fetch_json(f"http://127.0.0.1:{spec['host_port']}/v1/chat/completions", payload, timeout=900)
    elapsed_ms = round((time.time() - started) * 1000, 2)
    output = extract_text(body)
    if quality_mode == "sustained":
        passed, note = validate_sustained(task["name"], output)
    else:
        passed, note = validate_canary(task["name"], output)
    usage = body.get("usage") or {}
    completion_tokens = usage.get("completion_tokens") or 0
    tok_per_s = round(completion_tokens / max(0.001, elapsed_ms / 1000), 4) if completion_tokens else None
    return {
        "instance": spec["name"],
        "task": task["name"],
        "elapsed_ms": elapsed_ms,
        "completion_tokens": completion_tokens,
        "prompt_tokens": usage.get("prompt_tokens"),
        "total_tokens": usage.get("total_tokens"),
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


def summarize_phase(results: list[dict[str, Any]], samples: list[dict[str, float]], started_at: float, ended_at: float) -> dict[str, Any]:
    wall_seconds = max(0.001, ended_at - started_at)
    total_completion_tokens = sum(item.get("completion_tokens") or 0 for item in results)
    return {
        "results": results,
        "samples": samples,
        "wall_seconds": round(wall_seconds, 3),
        "aggregate_completion_tokens": total_completion_tokens,
        "aggregate_completion_tokens_per_sec": round(total_completion_tokens / wall_seconds, 4),
        "peak_total_memory_used_mb": max((item["memory_used_mb"] for item in samples), default=0.0),
        "peak_power_draw_watts": max((item["power_draw_watts"] for item in samples), default=0.0),
        "average_power_draw_watts": round(
            sum(item["power_draw_watts"] for item in samples) / max(1, len(samples)),
            2,
        ),
        "average_gpu_utilization_percent": round(
            sum(item["utilization_gpu_percent"] for item in samples) / max(1, len(samples)),
            2,
        ),
        "all_passed": all(item["passed"] for item in results),
    }


def assign_tasks(specs: list[dict[str, Any]], tasks: list[dict[str, Any]]) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    if len(specs) == 1:
        return [(specs[0], tasks[:2])]
    midpoint = len(tasks) // 2
    return [
        (specs[0], tasks[:midpoint]),
        (specs[1], tasks[midpoint:]),
    ]


def run_phase(
    specs: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    model_name: str,
    *,
    quality_mode: str,
    sample_power: bool,
) -> dict[str, Any]:
    sample_buffer: list[dict[str, float]] = []
    stop_event = threading.Event()
    sampler: threading.Thread | None = None
    if sample_power:
        sampler = threading.Thread(target=sample_gpu, args=(stop_event, sample_buffer), daemon=True)
        sampler.start()

    started_at = time.time()
    try:
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(specs) * 2) as executor:
            for spec, task_pair in assign_tasks(specs, tasks):
                for task in task_pair:
                    futures.append(executor.submit(run_request, spec, task, model_name, quality_mode=quality_mode))
            results = [future.result() for future in futures]
    finally:
        ended_at = time.time()
        if sampler is not None:
            stop_event.set()
            sampler.join(timeout=2)

    return summarize_phase(results, sample_buffer, started_at, ended_at)


def build_report(report: dict[str, Any]) -> str:
    lines = [
        "# Parallel Instance Probe",
        "",
        f"- generated_at: `{report['generated_at']}`",
        f"- restored_primary_vllm: `{report['restored_primary_vllm']}`",
        "",
        "## Config",
        "",
        f"- instances: `{report['config']['instances']}`",
        f"- kv_budget: `{report['config']['kv_budget']}`",
        f"- max_model_len: `{report['config']['max_model_len']}`",
        f"- max_num_seqs: `{report['config']['max_num_seqs']}`",
        f"- max_num_batched_tokens: `{report['config']['max_num_batched_tokens']}`",
        "",
        "## Idle",
        "",
        f"- total_gpu_memory_used_mb: `{report['idle']['total_gpu']['memory_used_mb']}`",
    ]

    for instance_state in report["idle"]["instances"]:
        lines.append(f"- {instance_state['name']}_gpu_process_mb: `{instance_state['gpu_process_memory_mb']}`")
        lines.append(f"- {instance_state['name']}_reserved_kv_capacity_tokens: `{instance_state['reserved_kv_capacity_tokens']}`")
    lines.extend(
        [
            f"- per_instance_under_48gb: `{report['verdict']['per_instance_under_48gb']}`",
            f"- combined_under_96gb: `{report['verdict']['combined_under_96gb']}`",
            f"- supports_two_full_256k_sequences_per_instance: `{report['verdict']['supports_two_full_256k_sequences_per_instance']}`",
            "",
            "## Canary Phase",
            "",
            f"- all_passed: `{report['canary']['all_passed']}`",
            "",
            "| instance | task | elapsed ms | tok/s | pass |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )

    for result in report["canary"]["results"]:
        lines.append(
            f"| {result['instance']} | {result['task']} | {result['elapsed_ms']} | {result['tok_per_s']} | {'PASS' if result['passed'] else 'FAIL'} |"
        )

    lines.extend(
        [
            "",
            "## Sustained Phase",
            "",
            f"- wall_seconds: `{report['load']['wall_seconds']}`",
            f"- aggregate_completion_tokens: `{report['load']['aggregate_completion_tokens']}`",
            f"- aggregate_completion_tokens_per_sec: `{report['load']['aggregate_completion_tokens_per_sec']}`",
            f"- average_power_draw_watts: `{report['load']['average_power_draw_watts']}`",
            f"- peak_power_draw_watts: `{report['load']['peak_power_draw_watts']}`",
            f"- average_gpu_utilization_percent: `{report['load']['average_gpu_utilization_percent']}`",
            f"- peak_gpu_memory_used_mb: `{report['load']['peak_total_memory_used_mb']}`",
            f"- all_passed: `{report['load']['all_passed']}`",
            "",
            "| instance | task | elapsed ms | tok/s | pass |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )

    for result in report["load"]["results"]:
        lines.append(
            f"| {result['instance']} | {result['task']} | {result['elapsed_ms']} | {result['tok_per_s']} | {'PASS' if result['passed'] else 'FAIL'} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    env = read_env()
    specs = INSTANCE_SPECS[: args.instances]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    label = f"{args.instances}x2"
    json_path = OUT_DIR / f"parallel-instance-probe-{label}-{timestamp}.json"
    md_path = OUT_DIR / f"parallel-instance-probe-{label}-{timestamp}.md"

    report: dict[str, Any] = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": {
            "instances": args.instances,
            "kv_budget": args.kv_budget,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "kv_cache_dtype": args.kv_cache_dtype,
            "cpu_offload_gb": args.cpu_offload_gb,
            "swap_space": args.swap_space,
        },
        "restored_primary_vllm": False,
    }

    try:
        stop_primary_vllm()
        for spec in specs:
            remove_container(spec["container"])
            launch_instance(env, spec, args)
        for spec in specs:
            wait_for_instance(spec, args.kv_budget, args.timeout)
        for spec in specs:
            run_request(spec, WARMUP_TASK, env["SERVED_MODEL_NAME"], quality_mode="canary")

        instance_states = []
        for spec in specs:
            metrics_text = urllib.request.urlopen(f"http://127.0.0.1:{spec['host_port']}/metrics", timeout=30).read().decode("utf-8")
            metrics = parse_metrics(metrics_text)
            instance_states.append(
                {
                    "name": spec["name"],
                    "container": spec["container"],
                    "host_port": spec["host_port"],
                    "gpu_process_memory_mb": container_gpu_memory_mb(spec["container"]),
                    "reserved_kv_capacity_tokens": int(metrics.get("vllm:cache_config_info:block_size", 0))
                    * int(metrics.get("vllm:cache_config_info:num_gpu_blocks", 0)),
                }
            )

        idle_total = total_gpu_snapshot()
        report["idle"] = {
            "total_gpu": idle_total,
            "instances": instance_states,
        }

        report["canary"] = run_phase(specs, CANARY_TASKS, env["SERVED_MODEL_NAME"], quality_mode="canary", sample_power=False)
        report["load"] = run_phase(specs, SUSTAINED_TASKS, env["SERVED_MODEL_NAME"], quality_mode="sustained", sample_power=True)
        report["verdict"] = {
            "per_instance_under_48gb": all(item["gpu_process_memory_mb"] < 48 * 1024 for item in instance_states),
            "combined_under_96gb": idle_total["memory_used_mb"] < 96 * 1024,
            "supports_two_full_256k_sequences_per_instance": all(
                item["reserved_kv_capacity_tokens"] >= (2 * 262144) for item in instance_states
            ),
            "all_canaries_passed_under_concurrency": report["canary"]["all_passed"],
            "all_sustained_tasks_passed": report["load"]["all_passed"],
        }
    finally:
        for spec in specs:
            remove_container(spec["container"])
        restore_primary_vllm()
        report["restored_primary_vllm"] = True

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(build_report(report), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
