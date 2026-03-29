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
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT_DIR / ".env"
COMPOSE_FILE = ROOT_DIR / "docker-compose.yml"
OUT_DIR = ROOT_DIR / "data" / "exports"
CONTAINER_NAME = "nvfp4studio-vllm"

PROMPTS = [
    {
        "name": "nvfp4_vs_gguf",
        "system": "Reply in Japanese. Do not reveal hidden reasoning.",
        "user": (
            "NVFP4 と GGUF の違い、4GPU 構成での長所短所、"
            "長コンテキスト運用で KV cache がどう効くかを、見出しなしで詳しく説明して。"
            "少なくとも 1200 文字以上。"
        ),
    },
    {
        "name": "parallel_power",
        "system": "Reply in Japanese. Do not reveal hidden reasoning.",
        "user": (
            "並列推論で消費電力が単純比例しない理由、GPU 使用率と通信待ちの関係、"
            "4GPU TP 構成で 2 並列や 3 並列を試す意味を、見出しなしで詳しく説明して。"
            "少なくとも 1200 文字以上。"
        ),
    },
    {
        "name": "throughput_tradeoff",
        "system": "Reply in Japanese. Do not reveal hidden reasoning.",
        "user": (
            "単発最速と aggregate throughput の違い、"
            "MAX_NUM_SEQS を上げたときに期待できる改善と副作用を、見出しなしで詳しく説明して。"
            "少なくとも 1200 文字以上。"
        ),
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe 397B TP4 parallel request throughput.")
    parser.add_argument(
        "--parallelism",
        nargs="*",
        type=int,
        default=[1, 2, 3],
        help="Concurrent request counts to test.",
    )
    parser.add_argument("--gateway-host", default="127.0.0.1")
    parser.add_argument("--gateway-port", type=int, default=8000)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--keep-last", action="store_true")
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


def compose_project_name(env_values: dict[str, str]) -> str:
    return env_values.get("COMPOSE_PROJECT_NAME") or ROOT_DIR.name.lower()


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


def run_command(command: list[str], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, check=check, capture_output=True, text=True)


def recreate_vllm() -> None:
    _, env_values = read_env_file(ENV_PATH)
    run_command(
        [
            "docker",
            "compose",
            "--env-file",
            str(ENV_PATH),
            "-p",
            compose_project_name(env_values),
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


def wait_for_runtime(host: str, port: int, expected_context: int, expected_seqs: int, timeout: int) -> tuple[dict[str, Any], str]:
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
                if f"--max-num-seqs {expected_seqs}" in cmdline:
                    return status, cmdline
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError(
        f"vLLM did not become ready for context={expected_context}, seqs={expected_seqs}.\n"
        f"Last status: {json.dumps(last_status, ensure_ascii=False)}\n"
        f"Last cmdline: {last_cmdline}"
    )


def fetch_benchmark(host: str, port: int, request_id: str, timeout: int = 120) -> dict[str, Any]:
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


def gpu_snapshot() -> list[dict[str, float]]:
    output = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.free,power.draw,power.limit,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    ).stdout.splitlines()
    rows: list[dict[str, float]] = []
    for line in output:
        index, used, free, power, limit, util = [part.strip() for part in line.split(",")]
        rows.append(
            {
                "index": float(index),
                "memory_used_mb": float(used),
                "memory_free_mb": float(free),
                "power_draw_watts": float(power),
                "power_limit_watts": float(limit),
                "utilization_gpu_percent": float(util),
            }
        )
    return rows


def detect_active_gpu_indices() -> list[int]:
    active = [int(item["index"]) for item in gpu_snapshot() if item["memory_used_mb"] > 10_000]
    return active or [0, 1, 2, 3]


def sample_gpu(stop_event: threading.Event, sink: list[dict[str, Any]], active_gpu_indices: set[int]) -> None:
    while not stop_event.is_set():
        try:
            rows = gpu_snapshot()
            selected = [row for row in rows if int(row["index"]) in active_gpu_indices]
            sink.append(
                {
                    "timestamp": time.time(),
                    "total_power_watts": round(sum(row["power_draw_watts"] for row in selected), 2),
                    "peak_single_gpu_power_watts": round(max((row["power_draw_watts"] for row in selected), default=0.0), 2),
                    "total_memory_used_mb": round(sum(row["memory_used_mb"] for row in selected), 2),
                    "average_gpu_utilization_percent": round(
                        sum(row["utilization_gpu_percent"] for row in selected) / max(1, len(selected)),
                        2,
                    ),
                }
            )
        except Exception:
            pass
        time.sleep(0.2)


def run_request(host: str, port: int, model_name: str, prompt: dict[str, str], max_tokens: int) -> dict[str, Any]:
    payload = {
        "model": model_name,
        "stream": False,
        "temperature": 0.2,
        "top_p": 0.95,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    started = time.time()
    error_message = None
    try:
        body, response = fetch_json(
            f"http://{host}:{port}/v1/chat/completions",
            method="POST",
            payload=payload,
            timeout=1800,
        )
        output = extract_assistant_text(body)
        usage = body.get("usage") or {}
        request_id = response.headers.get("x-nvfp4studio-request-id")
        benchmark = fetch_benchmark(host, port, request_id) if request_id else {}
    except urllib.error.HTTPError as exc:
        body = {}
        response = exc
        output = ""
        usage = {}
        request_id = exc.headers.get("x-nvfp4studio-request-id") if exc.headers else None
        benchmark = fetch_benchmark(host, port, request_id) if request_id else {}
        error_message = exc.read().decode("utf-8", errors="replace")
    elapsed_ms = round((time.time() - started) * 1000, 2)
    return {
        "prompt_name": prompt["name"],
        "request_id": request_id,
        "elapsed_ms": elapsed_ms,
        "output_chars": len(output),
        "output_preview": output[:220],
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "benchmark": benchmark,
        "error_message": error_message,
        "passed": error_message is None and len(output) >= 200 and (usage.get("completion_tokens") or 0) > 100,
    }


def summarize_run(
    concurrency: int,
    results: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    wall_seconds: float,
    active_gpu_indices: list[int],
) -> dict[str, Any]:
    total_completion_tokens = sum((item.get("completion_tokens") or 0) for item in results)
    benchmark_tok_s = [
        item.get("benchmark", {}).get("completion_tokens_per_sec")
        for item in results
        if item.get("benchmark", {}).get("completion_tokens_per_sec") is not None
    ]
    ttfts = [
        item.get("benchmark", {}).get("ttft_ms")
        for item in results
        if item.get("benchmark", {}).get("ttft_ms") is not None
    ]
    return {
        "concurrency": concurrency,
        "active_gpu_indices": active_gpu_indices,
        "wall_seconds": round(wall_seconds, 3),
        "aggregate_completion_tokens": total_completion_tokens,
        "aggregate_completion_tokens_per_sec": round(total_completion_tokens / max(0.001, wall_seconds), 4),
        "average_request_completion_tokens_per_sec": round(sum(benchmark_tok_s) / max(1, len(benchmark_tok_s)), 4)
        if benchmark_tok_s
        else None,
        "average_ttft_ms": round(sum(ttfts) / max(1, len(ttfts)), 2) if ttfts else None,
        "average_total_power_watts": round(
            sum(item["total_power_watts"] for item in samples) / max(1, len(samples)),
            2,
        ),
        "peak_total_power_watts": round(max((item["total_power_watts"] for item in samples), default=0.0), 2),
        "peak_single_gpu_power_watts": round(
            max((item["peak_single_gpu_power_watts"] for item in samples), default=0.0),
            2,
        ),
        "average_gpu_utilization_percent": round(
            sum(item["average_gpu_utilization_percent"] for item in samples) / max(1, len(samples)),
            2,
        ),
        "peak_total_memory_used_mb": round(max((item["total_memory_used_mb"] for item in samples), default=0.0), 2),
        "all_passed": all(item["passed"] for item in results),
        "results": results,
        "samples": samples,
    }


def build_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Parallel Request Probe",
        "",
        f"- generated_at: `{report['generated_at']}`",
        f"- restored_original_runtime: `{report['restored_original_runtime']}`",
        f"- model_name: `{report['model_name']}`",
        "",
        "## Summary",
        "",
        "| concurrency | wall s | aggregate tok/s | avg req tok/s | avg TTFT ms | avg total W | peak total W | peak single GPU W | avg util % | all passed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for item in report["results"]:
        lines.append(
            f"| {item['concurrency']} | {item['wall_seconds']} | {item['aggregate_completion_tokens_per_sec']} | "
            f"{item['average_request_completion_tokens_per_sec']} | {item['average_ttft_ms']} | "
            f"{item['average_total_power_watts']} | {item['peak_total_power_watts']} | "
            f"{item['peak_single_gpu_power_watts']} | {item['average_gpu_utilization_percent']} | "
            f"{'PASS' if item['all_passed'] else 'FAIL'} |"
        )

    for item in report["results"]:
        lines.extend(
            [
                "",
                f"## Concurrency {item['concurrency']}",
                "",
                f"- active_gpu_indices: `{item['active_gpu_indices']}`",
                f"- aggregate_completion_tokens: `{item['aggregate_completion_tokens']}`",
                f"- peak_total_memory_used_mb: `{item['peak_total_memory_used_mb']}`",
            ]
        )
        lines.extend(
            [
                "",
                "| prompt | elapsed ms | completion tokens | gateway tok/s | pass | note |",
                "| --- | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for result in item["results"]:
            note = "ok" if result["passed"] else (result.get("error_message") or "output too short")
            lines.append(
                f"| {result['prompt_name']} | {result['elapsed_ms']} | {result['completion_tokens']} | "
                f"{result['benchmark'].get('completion_tokens_per_sec')} | {'PASS' if result['passed'] else 'FAIL'} | {note[:80]} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if any(value < 1 for value in args.parallelism):
        raise SystemExit("parallelism must be >= 1")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    original_env_text = ENV_PATH.read_text(encoding="utf-8")
    _, original_env = read_env_file(ENV_PATH)
    model_name = original_env.get("SERVED_MODEL_NAME", "your-nvfp4-model")
    original_context = int(original_env.get("MAX_MODEL_LEN", "8192"))
    original_max_num_seqs = int(original_env.get("MAX_NUM_SEQS", "1"))
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = OUT_DIR / f"parallel-request-probe-{timestamp}.json"
    md_path = OUT_DIR / f"parallel-request-probe-{timestamp}.md"

    report: dict[str, Any] = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_name": model_name,
        "parallelism": args.parallelism,
        "restored_original_runtime": False,
        "results": [],
    }

    try:
        for concurrency in args.parallelism:
            updates = {
                "MAX_NUM_SEQS": str(concurrency),
            }
            write_env_updates(ENV_PATH, updates)
            recreate_vllm()
            status, cmdline = wait_for_runtime(
                args.gateway_host,
                args.gateway_port,
                original_context,
                concurrency,
                args.timeout,
            )

            warmup_prompt = PROMPTS[0]
            run_request(args.gateway_host, args.gateway_port, model_name, warmup_prompt, min(128, args.max_tokens))
            active_gpu_indices = detect_active_gpu_indices()

            sample_buffer: list[dict[str, Any]] = []
            stop_event = threading.Event()
            sampler = threading.Thread(
                target=sample_gpu,
                args=(stop_event, sample_buffer, set(active_gpu_indices)),
                daemon=True,
            )
            sampler.start()
            started = time.time()
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                    futures = [
                        executor.submit(
                            run_request,
                            args.gateway_host,
                            args.gateway_port,
                            model_name,
                            PROMPTS[index % len(PROMPTS)],
                            args.max_tokens,
                        )
                        for index in range(concurrency)
                    ]
                    results = [future.result() for future in futures]
            finally:
                wall_seconds = time.time() - started
                stop_event.set()
                sampler.join(timeout=2)

            run_summary = summarize_run(concurrency, results, sample_buffer, wall_seconds, active_gpu_indices)
            run_summary["status_after_apply"] = status.get("advisory", {})
            run_summary["cmdline"] = cmdline
            report["results"].append(run_summary)
    finally:
        if not args.keep_last:
            ENV_PATH.write_text(original_env_text, encoding="utf-8")
            recreate_vllm()
            wait_for_runtime(
                args.gateway_host,
                args.gateway_port,
                original_context,
                original_max_num_seqs,
                args.timeout,
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
