#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT_DIR / ".env"
COMPOSE_FILE = ROOT_DIR / "docker-compose.yml"
CONTAINER_NAME = "nvfp4studio-vllm"
PROJECT_NAME = "nvfp4studio"
OUT_DIR = ROOT_DIR / "data" / "exports"

RUNTIME_UPDATES = {
    "VLLM_RUNTIME_PROFILE": "memory",
    "MAX_MODEL_LEN": "262144",
    "GPU_MEMORY_UTILIZATION": "0.45",
    "MAX_NUM_SEQS": "1",
    "MAX_NUM_BATCHED_TOKENS": "4096",
    "KV_CACHE_DTYPE": "fp8",
    "CPU_OFFLOAD_GB": "0",
    "SWAP_SPACE": "16",
}

PERFORMANCE_PROMPT = {
    "system": "Reply in Japanese. Keep it short. Do not reveal hidden reasoning.",
    "user": "KV budget の現在状態を一言で説明して。"
}

QUALITY_CANARIES = [
    {
        "name": "sequence_rule",
        "system": "Reply in Japanese. Be concise. Do not reveal hidden reasoning.",
        "user": "数列 2, 6, 12, 20 の次の2項と規則を1文で答えて。",
    },
    {
        "name": "translation_en",
        "system": "Translate to English only. No extra commentary.",
        "user": "品質を保ちながら VRAM を削る。",
    },
    {
        "name": "json_structure",
        "system": "Return strict JSON only.",
        "user": "mode は local、goal は preserve quality while reducing VRAM にして JSON を返して。",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe vLLM KV cache budgets with quality canaries.")
    parser.add_argument("budgets", nargs="*", default=["8G", "6G"], help="Explicit KV budgets to test.")
    parser.add_argument("--gateway-host", default="127.0.0.1")
    parser.add_argument("--gateway-port", type=int, default=8000)
    parser.add_argument("--timeout", type=int, default=600, help="Timeout in seconds per vLLM reconfigure.")
    parser.add_argument("--keep-last", action="store_true", help="Keep the last tested runtime instead of restoring.")
    return parser.parse_args()


def read_env_file(path: Path) -> tuple[list[str], dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    values: dict[str, str] = {}
    for line in lines:
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return lines, values


def write_env_updates(path: Path, updates: dict[str, str]) -> None:
    lines, _ = read_env_file(path)
    for key, value in updates.items():
        prefix = f"{key}="
        for idx, line in enumerate(lines):
            if line.startswith(prefix):
                lines[idx] = f"{key}={value}"
                break
        else:
            lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_command(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)


def recreate_vllm() -> None:
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
            "--force-recreate",
            "--no-deps",
            "vllm",
        ],
        cwd=ROOT_DIR,
    )


def fetch_json(url: str, *, method: str = "GET", payload: dict[str, Any] | None = None, timeout: int = 600) -> tuple[dict[str, Any], Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
        return body, response


def fetch_status(host: str, port: int, timeout: int = 10) -> dict[str, Any]:
    body, _ = fetch_json(f"http://{host}:{port}/api/system/status", timeout=timeout)
    return body


def fetch_benchmark(host: str, port: int, request_id: str, timeout: int = 60) -> dict[str, Any]:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            body, _ = fetch_json(f"http://{host}:{port}/api/benchmarks/request/{request_id}", timeout=10)
            return body
        except Exception as exc:  # pragma: no cover - network timing variance
            last_error = exc
            time.sleep(1)
    raise RuntimeError(f"benchmark {request_id} was not persisted in time: {last_error}")


def wait_for_runtime(host: str, port: int, expected_context: int, expected_budget: str | None, timeout: int) -> tuple[dict[str, Any], str]:
    deadline = time.time() + timeout
    last_status: dict[str, Any] | None = None
    last_cmdline = ""
    while time.time() < deadline:
        try:
            status = fetch_status(host, port)
            last_status = status
            advisory = status.get("advisory", {})
            if status.get("vllm_healthy") and advisory.get("runtime_max_context") == expected_context:
                cmdline = run_command(["docker", "exec", CONTAINER_NAME, "ps", "-ef"]).stdout
                last_cmdline = cmdline
                has_budget_flag = "--kv-cache-memory-bytes " in cmdline
                if expected_budget:
                    if f"--kv-cache-memory-bytes {expected_budget}" in cmdline:
                        return status, cmdline
                elif not has_budget_flag:
                    return status, cmdline
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError(
        f"vLLM did not become ready for context={expected_context}, budget={expected_budget or 'auto'}.\n"
        f"Last status: {json.dumps(last_status, ensure_ascii=False)}\n"
        f"Last cmdline: {last_cmdline}"
    )


def extract_assistant_text(payload: dict[str, Any]) -> str:
    content = payload.get("choices", [{}])[0].get("message", {}).get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "".join(parts).strip()
    return ""


def run_non_stream(host: str, port: int, model_name: str, system: str, user: str) -> tuple[str | None, str]:
    payload = {
        "model": model_name,
        "stream": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 128,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"http://{host}:{port}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        body = json.loads(response.read().decode("utf-8"))
        return response.headers.get("x-nvfp4studio-request-id"), extract_assistant_text(body)


def run_stream(host: str, port: int, model_name: str, system: str, user: str) -> tuple[str | None, str]:
    payload = {
        "model": model_name,
        "stream": True,
        "temperature": 0.2,
        "top_p": 0.95,
        "max_tokens": 96,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"http://{host}:{port}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    collected: list[str] = []
    with urllib.request.urlopen(request, timeout=600) as response:
        request_id = response.headers.get("x-nvfp4studio-request-id")
        for raw_line in response.read().decode("utf-8").splitlines():
            if not raw_line.startswith("data: "):
                continue
            data_part = raw_line[6:].strip()
            if not data_part or data_part == "[DONE]":
                continue
            chunk = json.loads(data_part)
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                content = delta.get("content", "")
                if isinstance(content, str):
                    collected.append(content)
        return request_id, "".join(collected).strip()


def validate_canary(name: str, text: str) -> tuple[bool, str]:
    normalized = text.strip()
    lowered = normalized.lower()

    if name == "sequence_rule":
        passed = "30" in normalized and "42" in normalized
        return passed, "30 と 42 を含むかを確認"

    if name == "translation_en":
        passed = "quality" in lowered and "vram" in lowered and (
            "reduce" in lowered or "lower" in lowered or "decrease" in lowered
        )
        return passed, "quality / VRAM / reduce 系の語を含むかを確認"

    if name == "json_structure":
        try:
            payload = json.loads(normalized)
        except json.JSONDecodeError:
            return False, "JSON として解釈できない"
        passed = isinstance(payload, dict) and set(payload.keys()) == {"mode", "goal"}
        return passed, "mode と goal のみを持つ JSON かを確認"

    return False, "validator not found"


def summarize_status(status: dict[str, Any]) -> dict[str, Any]:
    gpu = (status.get("gpu") or [{}])[0]
    advisory = status.get("advisory", {})
    metrics = status.get("metrics", {}).get("values", {})
    return {
        "runtime_profile": advisory.get("runtime_profile"),
        "runtime_max_context": advisory.get("runtime_max_context"),
        "kv_cache_memory_bytes": advisory.get("kv_cache_memory_bytes"),
        "reserved_kv_capacity_tokens": advisory.get("reserved_kv_capacity_tokens"),
        "gpu_used_mb": gpu.get("memory_used_mb"),
        "gpu_free_mb": gpu.get("memory_free_mb"),
        "num_gpu_blocks": metrics.get("vllm:cache_config_info:num_gpu_blocks"),
        "block_size": metrics.get("vllm:cache_config_info:block_size"),
        "gpu_memory_utilization": metrics.get("vllm:cache_config_info:gpu_memory_utilization"),
    }


def build_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# KV Budget Probe",
        "",
        f"- generated_at: `{report['generated_at']}`",
        f"- restored_original_runtime: `{report['restored_original_runtime']}`",
        "",
        "## Summary",
        "",
        "| budget | cmdline | idle VRAM MB | peak VRAM MB | TTFT ms | tok/s | canaries |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]

    for item in report["results"]:
        performance = item.get("performance_benchmark") or {}
        passed = sum(1 for canary in item["canaries"] if canary["passed"])
        total = len(item["canaries"])
        lines.append(
            f"| {item['budget']} | {'ok' if item['cmdline_verified'] else 'ng'} | "
            f"{item['status_after_apply'].get('gpu_used_mb', 'N/A')} | "
            f"{performance.get('peak_vram_used_mb', 'N/A')} | "
            f"{performance.get('ttft_ms', 'N/A')} | "
            f"{performance.get('completion_tokens_per_sec', 'N/A')} | "
            f"{passed}/{total} |"
        )

    for item in report["results"]:
        lines.extend(
            [
                "",
                f"## {item['budget']}",
                "",
                f"- cmdline_verified: `{item['cmdline_verified']}`",
                f"- runtime_status: `{json.dumps(item['status_after_apply'], ensure_ascii=False)}`",
                f"- performance_text: `{item['performance_text']}`",
                "",
                "### Canaries",
            ]
        )
        for canary in item["canaries"]:
            lines.extend(
                [
                    f"- `{canary['name']}`: {'PASS' if canary['passed'] else 'FAIL'}",
                    f"  note: {canary['note']}",
                    f"  output: `{canary['output']}`",
                ]
            )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    original_env_text = ENV_PATH.read_text(encoding="utf-8")
    _, original_env = read_env_file(ENV_PATH)
    model_name = original_env.get("SERVED_MODEL_NAME", "your-nvfp4-model")
    original_context = int(original_env.get("MAX_MODEL_LEN", "8192"))
    original_budget = original_env.get("KV_CACHE_MEMORY_BYTES") or None

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = OUT_DIR / f"kv-budget-probe-{timestamp}.json"
    md_path = OUT_DIR / f"kv-budget-probe-{timestamp}.md"

    report: dict[str, Any] = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_name": model_name,
        "budgets": args.budgets,
        "results": [],
        "restored_original_runtime": False,
    }

    try:
        for budget in args.budgets:
            updates = dict(RUNTIME_UPDATES)
            updates["KV_CACHE_MEMORY_BYTES"] = budget
            write_env_updates(ENV_PATH, updates)
            recreate_vllm()
            status, cmdline = wait_for_runtime(args.gateway_host, args.gateway_port, 262144, budget, args.timeout)

            canaries = []
            for case in QUALITY_CANARIES:
                request_id, output_text = run_non_stream(
                    args.gateway_host,
                    args.gateway_port,
                    model_name,
                    case["system"],
                    case["user"],
                )
                benchmark = fetch_benchmark(args.gateway_host, args.gateway_port, request_id) if request_id else None
                passed, note = validate_canary(case["name"], output_text)
                canaries.append(
                    {
                        "name": case["name"],
                        "request_id": request_id,
                        "passed": passed,
                        "note": note,
                        "output": output_text,
                        "benchmark": benchmark,
                    }
                )

            perf_request_id, performance_text = run_stream(
                args.gateway_host,
                args.gateway_port,
                model_name,
                PERFORMANCE_PROMPT["system"],
                PERFORMANCE_PROMPT["user"],
            )
            performance_benchmark = (
                fetch_benchmark(args.gateway_host, args.gateway_port, perf_request_id) if perf_request_id else None
            )
            refreshed_status = fetch_status(args.gateway_host, args.gateway_port)

            report["results"].append(
                {
                    "budget": budget,
                    "cmdline_verified": f"--kv-cache-memory-bytes {budget}" in cmdline,
                    "status_after_apply": summarize_status(status),
                    "status_after_probe": summarize_status(refreshed_status),
                    "performance_request_id": perf_request_id,
                    "performance_text": performance_text,
                    "performance_benchmark": performance_benchmark,
                    "canaries": canaries,
                }
            )
    finally:
        if not args.keep_last:
            ENV_PATH.write_text(original_env_text, encoding="utf-8")
            recreate_vllm()
            wait_for_runtime(args.gateway_host, args.gateway_port, original_context, original_budget, args.timeout)
            report["restored_original_runtime"] = True

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(build_markdown_report(report), encoding="utf-8")

    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP error {exc.code}: {body}", file=sys.stderr)
        raise
