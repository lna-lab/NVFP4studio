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

MODEL_PATH = (
    ROOT_DIR.parent
    / "nvfp4"
    / "Huihui-Qwen3.5-4B-abliterated-NVFP4-hybrid-mm"
)
SERVED_MODEL_NAME = "Huihui-Qwen3.5-4B-hybrid-mm"

RUNTIME_UPDATES = {
    "MODEL_PATH": str(MODEL_PATH),
    "SERVED_MODEL_NAME": SERVED_MODEL_NAME,
    "VLLM_RUNTIME_PROFILE": "memory",
    "MAX_MODEL_LEN": "262144",
    "GPU_MEMORY_UTILIZATION": "0.45",
    "MAX_NUM_SEQS": "1",
    "MAX_NUM_BATCHED_TOKENS": "4096",
    "KV_CACHE_DTYPE": "fp8",
    "CPU_OFFLOAD_GB": "0",
    "SWAP_SPACE": "16",
    "TRUST_REMOTE_CODE": "true",
    "LANGUAGE_MODEL_ONLY": "false",
}

SANITY_CASE = {
    "name": "sequence_rule",
    "system": "Reply in Japanese. One sentence only. Do not reveal hidden reasoning.",
    "user": "数列 2, 6, 12, 20 の次の2項と規則を1文で答えて。",
}

PERFORMANCE_CASE = {
    "system": "Return strict JSON only. Do not include markdown fences or commentary.",
    "user": (
        "JSON object with key pairs. "
        "pairs must be an array of 96 objects with keys n and square for n=1..96."
    ),
    "max_tokens": 1400,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure 4B hybrid NVFP4 viability and performance at 256K runtime."
    )
    parser.add_argument("budgets", nargs="*", default=["16G", "17G", "18G"])
    parser.add_argument("--gateway-host", default="127.0.0.1")
    parser.add_argument("--gateway-port", type=int, default=8000)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--keep-last", action="store_true")
    return parser.parse_args()


def run_command(command: list[str], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, check=check, capture_output=True, text=True)


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


def fetch_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: int = 600,
) -> tuple[dict[str, Any], Any]:
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


def fetch_benchmark(host: str, port: int, request_id: str, timeout: int = 90) -> dict[str, Any]:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            body, _ = fetch_json(f"http://{host}:{port}/api/benchmarks/request/{request_id}", timeout=10)
            return body
        except Exception as exc:
            last_error = exc
            time.sleep(1)
    raise RuntimeError(f"benchmark {request_id} was not persisted in time: {last_error}")


def wait_for_runtime(
    host: str,
    port: int,
    expected_model_path: str,
    expected_context: int,
    expected_budget: str,
    timeout: int,
) -> tuple[dict[str, Any], str]:
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
                if expected_model_path not in cmdline:
                    time.sleep(5)
                    continue
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
        f"vLLM did not become ready for model={expected_model_path}, context={expected_context}, budget={expected_budget}.\n"
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


def run_non_stream(
    host: str,
    port: int,
    model_name: str,
    system: str,
    user: str,
    *,
    max_tokens: int = 128,
) -> tuple[str | None, str]:
    payload = {
        "model": model_name,
        "stream": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    body, response = fetch_json(
        f"http://{host}:{port}/v1/chat/completions",
        method="POST",
        payload=payload,
        timeout=900,
    )
    return response.headers.get("x-nvfp4studio-request-id"), extract_assistant_text(body)


def run_stream(
    host: str,
    port: int,
    model_name: str,
    system: str,
    user: str,
    *,
    max_tokens: int,
) -> tuple[str | None, str]:
    payload = {
        "model": model_name,
        "stream": True,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"http://{host}:{port}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    text_parts: list[str] = []
    with urllib.request.urlopen(request, timeout=900) as response:
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
                    text_parts.append(content)
    return request_id, "".join(text_parts).strip()


def summarize_status(status: dict[str, Any]) -> dict[str, Any]:
    gpu = (status.get("gpu") or [{}])[0]
    advisory = status.get("advisory", {})
    return {
        "runtime_profile": advisory.get("runtime_profile"),
        "runtime_max_context": advisory.get("runtime_max_context"),
        "reserved_kv_capacity_tokens": advisory.get("reserved_kv_capacity_tokens"),
        "kv_cache_memory_bytes": advisory.get("kv_cache_memory_bytes"),
        "fits_in_vram": advisory.get("fits_in_vram"),
        "risk_level": advisory.get("risk_level"),
        "message": advisory.get("message"),
        "gpu_process_memory_mb": status.get("gpu_process_memory_mb"),
        "gpu_used_mb": gpu.get("memory_used_mb"),
        "gpu_free_mb": gpu.get("memory_free_mb"),
        "power_draw_watts": gpu.get("power_draw_watts"),
        "power_limit_watts": gpu.get("power_limit_watts"),
        "utilization_gpu_percent": gpu.get("utilization_gpu_percent"),
    }


def validate_sanity(text: str) -> tuple[bool, str]:
    passed = "30" in text and "42" in text
    return passed, "30 と 42 を含むか"


def validate_performance_output(text: str) -> tuple[bool, str]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return False, "JSON parse failed"
    pairs = payload.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != 96:
        return False, "pairs length mismatch"
    for index, item in enumerate(pairs, start=1):
        if item != {"n": index, "square": index * index}:
            return False, f"mismatch at n={index}"
    return True, "96 square pairs"


def build_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# 4B NVFP4 256K Probe",
        "",
        f"- generated_at: `{report['generated_at']}`",
        f"- model_path: `{report['model_path']}`",
        f"- restored_original_runtime: `{report['restored_original_runtime']}`",
        "",
        "## Summary",
        "",
        "| kv budget | supports full 256K | idle VRAM MB | peak VRAM MB | TTFT ms | tok/s | peak power W | sanity | perf output |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]

    for item in report["results"]:
        benchmark = item.get("performance_benchmark") or {}
        lines.append(
            f"| {item['budget']} | "
            f"{'yes' if item['supports_full_256k'] else 'no'} | "
            f"{item['status_after_apply'].get('gpu_process_memory_mb', 'N/A')} | "
            f"{benchmark.get('peak_vram_used_mb', 'N/A')} | "
            f"{benchmark.get('ttft_ms', 'N/A')} | "
            f"{benchmark.get('completion_tokens_per_sec', 'N/A')} | "
            f"{benchmark.get('peak_power_watts', 'N/A')} | "
            f"{'PASS' if item['sanity']['passed'] else 'FAIL'} | "
            f"{'PASS' if item['performance_validation']['passed'] else 'FAIL'} |"
        )

    for item in report["results"]:
        lines.extend(
            [
                "",
                f"## {item['budget']}",
                "",
                f"- cmdline_verified: `{item['cmdline_verified']}`",
                f"- supports_full_256k: `{item['supports_full_256k']}`",
                f"- status_after_apply: `{json.dumps(item['status_after_apply'], ensure_ascii=False)}`",
                f"- status_after_probe: `{json.dumps(item['status_after_probe'], ensure_ascii=False)}`",
                f"- sanity_output: `{item['sanity']['output']}`",
                f"- sanity_note: `{item['sanity']['note']}`",
                f"- performance_preview: `{item['performance_preview']}`",
                f"- performance_validation: `{item['performance_validation']['note']}`",
                f"- performance_benchmark: `{json.dumps(item.get('performance_benchmark') or {}, ensure_ascii=False)}`",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    original_env_text = ENV_PATH.read_text(encoding="utf-8")
    _, original_env = read_env_file(ENV_PATH)
    original_context = int(original_env.get("MAX_MODEL_LEN", "8192"))
    original_budget = original_env.get("KV_CACHE_MEMORY_BYTES") or None
    original_model_path = original_env.get("MODEL_PATH", "")

    report: dict[str, Any] = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_path": str(MODEL_PATH),
        "budgets": args.budgets,
        "results": [],
        "restored_original_runtime": False,
        "restored_model_path": original_model_path,
    }

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = OUT_DIR / f"probe-4b-256k-{timestamp}.json"
    md_path = OUT_DIR / f"probe-4b-256k-{timestamp}.md"

    try:
        for budget in args.budgets:
            updates = dict(RUNTIME_UPDATES)
            updates["KV_CACHE_MEMORY_BYTES"] = budget
            write_env_updates(ENV_PATH, updates)
            recreate_vllm()
            status, cmdline = wait_for_runtime(
                args.gateway_host,
                args.gateway_port,
                expected_model_path=str(MODEL_PATH),
                expected_context=262144,
                expected_budget=budget,
                timeout=args.timeout,
            )

            summarized_status = summarize_status(status)
            supports_full_256k = (summarized_status.get("reserved_kv_capacity_tokens") or 0) >= 262144

            sanity_request_id, sanity_output = run_non_stream(
                args.gateway_host,
                args.gateway_port,
                SERVED_MODEL_NAME,
                SANITY_CASE["system"],
                SANITY_CASE["user"],
            )
            sanity_benchmark = (
                fetch_benchmark(args.gateway_host, args.gateway_port, sanity_request_id)
                if sanity_request_id
                else None
            )
            sanity_passed, sanity_note = validate_sanity(sanity_output)

            performance_request_id = None
            performance_preview = ""
            performance_benchmark = None
            performance_passed = False
            performance_note = "skipped because reserved_kv_capacity_tokens < 256K"
            if supports_full_256k:
                performance_request_id, performance_text = run_stream(
                    args.gateway_host,
                    args.gateway_port,
                    SERVED_MODEL_NAME,
                    PERFORMANCE_CASE["system"],
                    PERFORMANCE_CASE["user"],
                    max_tokens=PERFORMANCE_CASE["max_tokens"],
                )
                performance_preview = performance_text[:280].replace("\n", "\\n")
                performance_passed, performance_note = validate_performance_output(performance_text)
                if performance_request_id:
                    performance_benchmark = fetch_benchmark(
                        args.gateway_host,
                        args.gateway_port,
                        performance_request_id,
                        timeout=180,
                    )

            refreshed_status = fetch_status(args.gateway_host, args.gateway_port)

            report["results"].append(
                {
                    "budget": budget,
                    "cmdline_verified": (
                        str(MODEL_PATH) in cmdline and f"--kv-cache-memory-bytes {budget}" in cmdline
                    ),
                    "supports_full_256k": supports_full_256k,
                    "status_after_apply": summarized_status,
                    "status_after_probe": summarize_status(refreshed_status),
                    "sanity": {
                        "request_id": sanity_request_id,
                        "output": sanity_output,
                        "passed": sanity_passed,
                        "note": sanity_note,
                        "benchmark": sanity_benchmark,
                    },
                    "performance_request_id": performance_request_id,
                    "performance_preview": performance_preview,
                    "performance_validation": {
                        "passed": performance_passed,
                        "note": performance_note,
                    },
                    "performance_benchmark": performance_benchmark,
                }
            )
    finally:
        if not args.keep_last:
            ENV_PATH.write_text(original_env_text, encoding="utf-8")
            recreate_vllm()
            wait_for_runtime(
                args.gateway_host,
                args.gateway_port,
                expected_model_path=original_model_path,
                expected_context=original_context,
                expected_budget=original_budget or "",
                timeout=args.timeout,
            )
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
