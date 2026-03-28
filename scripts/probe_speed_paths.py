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
OUT_DIR = ROOT_DIR / "data" / "exports"

LONG_STREAM_PROMPT = {
    "system": "Reply in Japanese. Do not reveal hidden reasoning.",
    "user": (
        "NVFP4 と GGUF の違い、4GPU 構成での長所短所、KV cache 予算の意味、"
        "16K context 運用時の注意点を、見出しなしでかなり詳しく説明して。"
        "少なくとも 1200 文字以上。"
    ),
}

QUALITY_CANARY = {
    "system": "Return strict JSON only.",
    "user": (
        "mode は local、goal は preserve quality while increasing throughput、"
        "tone は calm にして JSON を返して。"
    ),
}

COMMON_FLAG_RESETS = {
    "ENABLE_EXPERT_PARALLEL": "false",
    "EXPERT_PLACEMENT_STRATEGY": "",
    "ALL2ALL_BACKEND": "",
    "MAX_CUDAGRAPH_CAPTURE_SIZE": "",
    "MOE_BACKEND": "",
    "ENABLE_FLASHINFER_AUTOTUNE": "false",
    "VLLM_USE_FLASHINFER_MOE_FP4": "0",
}

FAST_FLAG_BASE = {
    **COMMON_FLAG_RESETS,
    "ENFORCE_EAGER": "false",
    "DISABLE_CUSTOM_ALL_REDUCE": "false",
}

SPEED_PROFILE_BASE = {
    **FAST_FLAG_BASE,
    "VLLM_RUNTIME_PROFILE": "speed",
    "MAX_MODEL_LEN": "16384",
    "GPU_MEMORY_UTILIZATION": "0.85",
    "MAX_NUM_SEQS": "4",
    "MAX_NUM_BATCHED_TOKENS": "8192",
    "KV_CACHE_DTYPE": "auto",
    "KV_CACHE_MEMORY_BYTES": "",
    "CPU_OFFLOAD_GB": "0",
    "SWAP_SPACE": "4",
}

VARIANT_DEFS = [
    {
        "name": "baseline_safe",
        "description": "Current stable 16K/8G runtime as-is.",
        "updates": {},
    },
    {
        "name": "no_eager",
        "description": "Disable eager mode only.",
        "updates": {
            **COMMON_FLAG_RESETS,
            "ENFORCE_EAGER": "false",
        },
    },
    {
        "name": "no_eager_nccl",
        "description": "Disable eager mode and re-enable custom all-reduce.",
        "updates": {
            **FAST_FLAG_BASE,
        },
    },
    {
        "name": "speed_no_eager_nccl",
        "description": "Switch to speed profile with no eager and NCCL/custom all-reduce enabled.",
        "updates": {
            **SPEED_PROFILE_BASE,
        },
    },
    {
        "name": "speed_flashinfer_fp4",
        "description": "Speed profile plus FlashInfer FP4 path.",
        "updates": {
            **SPEED_PROFILE_BASE,
            "VLLM_USE_FLASHINFER_MOE_FP4": "1",
            "MOE_BACKEND": "flashinfer_cutedsl",
            "ENABLE_FLASHINFER_AUTOTUNE": "true",
        },
    },
    {
        "name": "speed_expert_parallel",
        "description": "Speed profile plus expert parallel with round-robin placement.",
        "updates": {
            **SPEED_PROFILE_BASE,
            "ENABLE_EXPERT_PARALLEL": "true",
            "EXPERT_PLACEMENT_STRATEGY": "round_robin",
            "ALL2ALL_BACKEND": "allgather_reducescatter",
        },
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe speed-oriented vLLM runtime variants.")
    parser.add_argument(
        "--variants",
        nargs="*",
        default=[item["name"] for item in VARIANT_DEFS],
        help="Variant names to test. Defaults to the full built-in sweep.",
    )
    parser.add_argument("--gateway-host", default="127.0.0.1")
    parser.add_argument("--gateway-port", type=int, default=8000)
    parser.add_argument("--timeout", type=int, default=1800, help="Timeout in seconds per vLLM reconfigure.")
    parser.add_argument("--max-tokens", type=int, default=512, help="Completion length for the long streaming benchmark.")
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


def run_command(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)


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


def wait_for_runtime(host: str, port: int, expected_context: int, timeout: int) -> tuple[dict[str, Any], str]:
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
                if "vllm serve" in cmdline:
                    return status, cmdline
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError(
        f"vLLM did not become ready for context={expected_context}.\n"
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


def run_non_stream(host: str, port: int, model_name: str, system: str, user: str, *, max_tokens: int = 128) -> tuple[str | None, str]:
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
        timeout=600,
    )
    return response.headers.get("x-nvfp4studio-request-id"), extract_assistant_text(body)


def run_stream(host: str, port: int, model_name: str, system: str, user: str, *, max_tokens: int) -> tuple[str | None, str]:
    payload = {
        "model": model_name,
        "stream": True,
        "temperature": 0.2,
        "top_p": 0.95,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "chat_template_kwargs": {"enable_thinking": False},
        "stream_options": {"include_usage": True},
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"http://{host}:{port}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    collected: list[str] = []
    with urllib.request.urlopen(request, timeout=1800) as response:
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


def validate_json_canary(text: str) -> tuple[bool, str]:
    normalized = text.strip()
    try:
        payload = json.loads(normalized)
    except json.JSONDecodeError:
        return False, "JSON として解釈できない"

    passed = (
        isinstance(payload, dict)
        and payload.get("mode") == "local"
        and payload.get("tone") == "calm"
        and "increasing throughput" in payload.get("goal", "")
    )
    return passed, "strict JSON で mode / goal / tone が正しいかを確認"


def summarize_status(status: dict[str, Any]) -> dict[str, Any]:
    gpu = (status.get("gpu") or [{}])[0]
    advisory = status.get("advisory", {})
    metrics = status.get("metrics", {}).get("values", {})
    return {
        "runtime_profile": advisory.get("runtime_profile"),
        "runtime_max_context": advisory.get("runtime_max_context"),
        "kv_cache_memory_bytes": advisory.get("kv_cache_memory_bytes"),
        "gpu_used_mb": gpu.get("memory_used_mb"),
        "gpu_free_mb": gpu.get("memory_free_mb"),
        "gpu_memory_utilization": metrics.get("vllm:cache_config_info:gpu_memory_utilization"),
        "num_gpu_blocks": metrics.get("vllm:cache_config_info:num_gpu_blocks"),
    }


def build_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Speed Path Probe",
        "",
        f"- generated_at: `{report['generated_at']}`",
        f"- restored_original_runtime: `{report['restored_original_runtime']}`",
        "",
        "## Summary",
        "",
        "| variant | status | tok/s | TTFT ms | peak VRAM MB | JSON canary | notes |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]

    for item in report["results"]:
        benchmark = item.get("stream_benchmark") or {}
        lines.append(
            f"| {item['name']} | {item['status']} | "
            f"{benchmark.get('completion_tokens_per_sec', 'N/A')} | "
            f"{benchmark.get('ttft_ms', 'N/A')} | "
            f"{benchmark.get('peak_vram_used_mb', 'N/A')} | "
            f"{'PASS' if item.get('json_canary', {}).get('passed') else 'FAIL'} | "
            f"{item.get('note', '')} |"
        )

    for item in report["results"]:
        lines.extend(
            [
                "",
                f"## {item['name']}",
                "",
                f"- description: `{item['description']}`",
                f"- status: `{item['status']}`",
                f"- applied_updates: `{json.dumps(item.get('updates', {}), ensure_ascii=False)}`",
                f"- runtime_status: `{json.dumps(item.get('runtime_status', {}), ensure_ascii=False)}`",
                f"- note: `{item.get('note', '')}`",
            ]
        )
        if item.get("cmdline"):
            lines.append(f"- cmdline: `{item['cmdline']}`")
        if item.get("json_canary"):
            lines.extend(
                [
                    f"- json_canary: `{'PASS' if item['json_canary']['passed'] else 'FAIL'}`",
                    f"- json_canary_note: `{item['json_canary']['note']}`",
                    f"- json_canary_output: `{item['json_canary']['output']}`",
                ]
            )
        if item.get("stream_benchmark"):
            lines.append(f"- stream_benchmark: `{json.dumps(item['stream_benchmark'], ensure_ascii=False)}`")
        if item.get("stream_preview"):
            lines.append(f"- stream_preview: `{item['stream_preview']}`")

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    variant_map = {item["name"]: item for item in VARIANT_DEFS}
    unknown = [name for name in args.variants if name not in variant_map]
    if unknown:
        raise SystemExit(f"Unknown variants: {', '.join(unknown)}")

    original_env_text = ENV_PATH.read_text(encoding="utf-8")
    _, original_env = read_env_file(ENV_PATH)
    original_context = int(original_env.get("MAX_MODEL_LEN", "8192"))
    model_name = original_env.get("SERVED_MODEL_NAME", "your-nvfp4-model")

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = OUT_DIR / f"speed-path-probe-{timestamp}.json"
    md_path = OUT_DIR / f"speed-path-probe-{timestamp}.md"

    report: dict[str, Any] = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_name": model_name,
        "variants": args.variants,
        "results": [],
        "restored_original_runtime": False,
    }

    try:
        for name in args.variants:
            variant = variant_map[name]
            result: dict[str, Any] = {
                "name": name,
                "description": variant["description"],
                "updates": variant["updates"],
                "status": "startup_failed",
                "note": "",
            }
            try:
                ENV_PATH.write_text(original_env_text, encoding="utf-8")
                if variant["updates"]:
                    write_env_updates(ENV_PATH, variant["updates"])
                recreate_vllm()
                expected_context = int((variant["updates"] or {}).get("MAX_MODEL_LEN", original_env.get("MAX_MODEL_LEN", "8192")))
                status, cmdline = wait_for_runtime(args.gateway_host, args.gateway_port, expected_context, args.timeout)
                result["runtime_status"] = summarize_status(status)
                result["cmdline"] = cmdline

                canary_request_id, canary_output = run_non_stream(
                    args.gateway_host,
                    args.gateway_port,
                    model_name,
                    QUALITY_CANARY["system"],
                    QUALITY_CANARY["user"],
                    max_tokens=96,
                )
                canary_benchmark = fetch_benchmark(args.gateway_host, args.gateway_port, canary_request_id) if canary_request_id else None
                canary_passed, canary_note = validate_json_canary(canary_output)
                result["json_canary"] = {
                    "request_id": canary_request_id,
                    "passed": canary_passed,
                    "note": canary_note,
                    "output": canary_output,
                    "benchmark": canary_benchmark,
                }

                stream_request_id, stream_output = run_stream(
                    args.gateway_host,
                    args.gateway_port,
                    model_name,
                    LONG_STREAM_PROMPT["system"],
                    LONG_STREAM_PROMPT["user"],
                    max_tokens=args.max_tokens,
                )
                result["stream_benchmark"] = (
                    fetch_benchmark(args.gateway_host, args.gateway_port, stream_request_id) if stream_request_id else None
                )
                result["stream_preview"] = stream_output[:320]
                result["status"] = "ok" if canary_passed else "quality_warning"
                result["note"] = "stream benchmark completed"
            except Exception as exc:
                result["note"] = str(exc)
            report["results"].append(result)
    finally:
        if not args.keep_last:
            ENV_PATH.write_text(original_env_text, encoding="utf-8")
            recreate_vllm()
            wait_for_runtime(args.gateway_host, args.gateway_port, original_context, args.timeout)
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
