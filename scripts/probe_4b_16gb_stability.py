#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
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
    "KV_CACHE_MEMORY_BYTES": "16G",
    "CPU_OFFLOAD_GB": "0",
    "SWAP_SPACE": "16",
    "TRUST_REMOTE_CODE": "true",
    "LANGUAGE_MODEL_ONLY": "false",
}

QUALITY_CASES = [
    {
        "name": "jp_intro",
        "system": "Reply in Japanese. One sentence only. Do not reveal hidden reasoning.",
        "user": "日本語で一文だけ自己紹介して。",
        "expectation": "ja_sentence",
        "max_tokens": 64,
    },
    {
        "name": "zh_intro",
        "system": "请用中文回答，只用一句话，不要暴露推理过程。",
        "user": "请用中文一句话介绍你自己。",
        "expectation": "zh_sentence",
        "max_tokens": 64,
    },
    {
        "name": "sequence_rule",
        "system": "Reply in Japanese. One sentence only. Do not reveal hidden reasoning.",
        "user": "数列 2, 6, 12, 20 の次の2項と規則を1文で答えて。",
        "expectation": "sequence_rule",
        "max_tokens": 96,
    },
    {
        "name": "translation_en",
        "system": "Translate to English only. No extra commentary.",
        "user": "品質を保ちながら VRAM を削る。",
        "expectation": "translation_en",
        "max_tokens": 64,
    },
    {
        "name": "jp_json",
        "system": "Return strict JSON only.",
        "user": "mode は local、goal は preserve quality while reducing VRAM、tone は calm にして JSON を返して。",
        "expectation": "strict_json",
        "max_tokens": 96,
    },
    {
        "name": "structured_extract",
        "system": "Return strict JSON only.",
        "user": "次の文から name, gpu, context を抽出して JSON で返して。『Ken uses RTX PRO 6000 Blackwell with 256K context.』",
        "expectation": "structured_extract",
        "max_tokens": 96,
    },
    {
        "name": "jp_reasoning_suppression",
        "system": "Reply in Japanese. One short sentence. Do not reveal chain-of-thought, reasoning, or hidden planning.",
        "user": "VRAM節約と品質維持の方針を短く述べて。",
        "expectation": "no_leak_ja",
        "max_tokens": 64,
    },
    {
        "name": "zh_reasoning_suppression",
        "system": "请用中文简短回答，不要展示思维过程、推理过程或内部计划。",
        "user": "请简短说明如何在节省显存的同时保持质量。",
        "expectation": "no_leak_zh",
        "max_tokens": 64,
    },
]

PERFORMANCE_CASE = {
    "system": "Return strict JSON only. No markdown fences or commentary.",
    "user": (
        "Return a JSON object with key items. "
        "items must be an array of integers from 1 through 512 in ascending order."
    ),
    "max_tokens": 2200,
}

LEAK_PATTERNS_EN = [
    "the user asked",
    "i need to",
    "i should",
    "i'll",
    "reasoning",
]
LEAK_PATTERNS_ZH = [
    "用户要求",
    "我需要",
    "我应该",
    "思考过程",
    "推理过程",
]
LEAK_PATTERNS_GENERIC = [
    "<think>",
    "</think>",
    "thinking process",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe 4B hybrid NVFP4 stability at 16G KV budget and 256K runtime."
    )
    parser.add_argument("--gateway-host", default="127.0.0.1")
    parser.add_argument("--gateway-port", type=int, default=8000)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--performance-runs", type=int, default=3)
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


def wait_for_runtime(
    host: str,
    port: int,
    expected_model_path: str,
    expected_context: int,
    expected_budget: str | None,
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
        f"vLLM did not become ready for model={expected_model_path}, context={expected_context}, budget={expected_budget or 'auto'}.\n"
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
    max_tokens: int,
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


def detect_leakage(text: str) -> list[str]:
    lowered = text.lower()
    hits: list[str] = []
    for pattern in LEAK_PATTERNS_GENERIC:
        if pattern in lowered:
            hits.append(pattern)
    for pattern in LEAK_PATTERNS_EN:
        if pattern in lowered:
            hits.append(pattern)
    for pattern in LEAK_PATTERNS_ZH:
        if pattern in text:
            hits.append(pattern)
    return hits


def seems_japanese(text: str) -> bool:
    return any("\u3040" <= ch <= "\u30ff" or "\u4e00" <= ch <= "\u9fff" for ch in text)


def seems_chinese(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text) and not any(
        "\u3040" <= ch <= "\u30ff" for ch in text
    )


def validate_case(expectation: str, text: str) -> tuple[bool, str]:
    leaks = detect_leakage(text)
    if expectation == "ja_sentence":
        passed = seems_japanese(text) and len(leaks) == 0 and text.count("\n") <= 1
        return passed, f"ja_sentence leaks={leaks}"
    if expectation == "zh_sentence":
        passed = seems_chinese(text) and len(leaks) == 0 and text.count("\n") <= 1
        return passed, f"zh_sentence leaks={leaks}"
    if expectation == "sequence_rule":
        passed = "30" in text and "42" in text and len(leaks) == 0
        return passed, f"sequence_rule leaks={leaks}"
    if expectation == "translation_en":
        lowered = text.lower()
        passed = (
            "quality" in lowered
            and "vram" in lowered
            and ("reduce" in lowered or "lower" in lowered or "decrease" in lowered or "trim" in lowered)
            and len(leaks) == 0
        )
        return passed, f"translation_en leaks={leaks}"
    if expectation == "strict_json":
        try:
            payload = json.loads(text)
        except Exception:
            return False, "JSON parse failed"
        passed = (
            payload.get("mode") == "local"
            and payload.get("tone") == "calm"
            and "preserve quality" in payload.get("goal", "")
        )
        return passed, "strict_json"
    if expectation == "structured_extract":
        try:
            payload = json.loads(text)
        except Exception:
            return False, "JSON parse failed"
        passed = (
            payload.get("name") == "Ken"
            and "RTX PRO 6000 Blackwell" in payload.get("gpu", "")
            and "256K" in payload.get("context", "")
        )
        return passed, "structured_extract"
    if expectation in {"no_leak_ja", "no_leak_zh"}:
        passed = len(leaks) == 0
        return passed, f"leaks={leaks}"
    return False, "unknown expectation"


def validate_perf_output(text: str) -> tuple[bool, str]:
    try:
        payload = json.loads(text)
    except Exception:
        return False, "JSON parse failed"
    items = payload.get("items")
    if not isinstance(items, list):
        return False, "items missing"
    expected = list(range(1, 513))
    if items != expected:
        return False, "items mismatch"
    return True, "items 1..512"


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
        "gpu_used_mb": gpu.get("memory_used_mb"),
        "gpu_free_mb": gpu.get("memory_free_mb"),
        "power_draw_watts": gpu.get("power_draw_watts"),
        "power_limit_watts": gpu.get("power_limit_watts"),
        "utilization_gpu_percent": gpu.get("utilization_gpu_percent"),
    }


def average_or_none(values: list[float | None]) -> float | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return round(statistics.mean(filtered), 4)


def build_markdown_report(report: dict[str, Any]) -> str:
    quality = report["quality"]
    performance = report["performance"]
    lines = [
        "# 4B NVFP4 16G Stability Probe",
        "",
        f"- generated_at: `{report['generated_at']}`",
        f"- model_path: `{report['model_path']}`",
        f"- restored_original_runtime: `{report['restored_original_runtime']}`",
        "",
        "## Runtime",
        "",
        f"- status_after_apply: `{json.dumps(report['status_after_apply'], ensure_ascii=False)}`",
        f"- status_after_probe: `{json.dumps(report['status_after_probe'], ensure_ascii=False)}`",
        "",
        "## Quality",
        "",
        f"- score: `{quality['passed_count']}/{quality['total_cases']}`",
        "",
        "| Case | Pass | Note |",
        "| --- | --- | --- |",
    ]
    for case in quality["cases"]:
        lines.append(f"| {case['name']} | {'yes' if case['passed'] else 'no'} | {case['note']} |")

    lines.extend(
        [
            "",
            "## Performance",
            "",
            f"- runs: `{performance['runs']}`",
            f"- output_valid_runs: `{performance['valid_runs']}`",
            f"- average_ttft_ms: `{performance['average_ttft_ms']}`",
            f"- average_completion_tok_s: `{performance['average_completion_tok_s']}`",
            f"- average_peak_vram_mb: `{performance['average_peak_vram_mb']}`",
            f"- max_peak_power_watts: `{performance['max_peak_power_watts']}`",
            "",
            "| Run | Pass | Finish | TTFT ms | tok/s | peak VRAM MB | peak power W |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for run in performance["benchmarks"]:
        lines.append(
            f"| {run['run_index']} | {'yes' if run['passed'] else 'no'} | "
            f"{run['benchmark'].get('finish_reason')} | "
            f"{run['benchmark'].get('ttft_ms')} | "
            f"{run['benchmark'].get('completion_tokens_per_sec')} | "
            f"{run['benchmark'].get('peak_vram_used_mb')} | "
            f"{run['benchmark'].get('peak_power_watts')} |"
        )

    for case in quality["cases"]:
        lines.extend(
            [
                "",
                f"### {case['name']}",
                "",
                "```text",
                case["response_text"],
                "```",
            ]
        )

    for run in performance["benchmarks"]:
        lines.extend(
            [
                "",
                f"### performance_run_{run['run_index']}",
                "",
                f"- note: `{run['note']}`",
                f"- preview: `{run['preview']}`",
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
        "restored_original_runtime": False,
    }

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    json_path = OUT_DIR / f"probe-4b-16g-stability-{stamp}.json"
    md_path = OUT_DIR / f"probe-4b-16g-stability-{stamp}.md"

    try:
        write_env_updates(ENV_PATH, RUNTIME_UPDATES)
        recreate_vllm()
        status, _cmdline = wait_for_runtime(
            args.gateway_host,
            args.gateway_port,
            expected_model_path=str(MODEL_PATH),
            expected_context=262144,
            expected_budget=RUNTIME_UPDATES["KV_CACHE_MEMORY_BYTES"],
            timeout=args.timeout,
        )

        quality_cases = []
        passed_count = 0
        for case in QUALITY_CASES:
            _request_id, response_text = run_non_stream(
                args.gateway_host,
                args.gateway_port,
                SERVED_MODEL_NAME,
                case["system"],
                case["user"],
                max_tokens=case["max_tokens"],
            )
            passed, note = validate_case(case["expectation"], response_text)
            if passed:
                passed_count += 1
            quality_cases.append(
                {
                    "name": case["name"],
                    "response_text": response_text,
                    "passed": passed,
                    "note": note,
                }
            )

        benchmarks = []
        for index in range(1, args.performance_runs + 1):
            request_id, response_text = run_stream(
                args.gateway_host,
                args.gateway_port,
                SERVED_MODEL_NAME,
                PERFORMANCE_CASE["system"],
                PERFORMANCE_CASE["user"],
                max_tokens=PERFORMANCE_CASE["max_tokens"],
            )
            benchmark = fetch_benchmark(args.gateway_host, args.gateway_port, request_id, timeout=240) if request_id else {}
            passed, note = validate_perf_output(response_text)
            benchmarks.append(
                {
                    "run_index": index,
                    "request_id": request_id,
                    "passed": passed,
                    "note": note,
                    "preview": response_text[:280].replace("\n", "\\n"),
                    "benchmark": benchmark,
                }
            )

        refreshed_status = fetch_status(args.gateway_host, args.gateway_port)

        report["status_after_apply"] = summarize_status(status)
        report["status_after_probe"] = summarize_status(refreshed_status)
        report["quality"] = {
            "passed_count": passed_count,
            "total_cases": len(QUALITY_CASES),
            "cases": quality_cases,
        }
        report["performance"] = {
            "runs": args.performance_runs,
            "valid_runs": sum(1 for item in benchmarks if item["passed"]),
            "average_ttft_ms": average_or_none([item["benchmark"].get("ttft_ms") for item in benchmarks]),
            "average_completion_tok_s": average_or_none([item["benchmark"].get("completion_tokens_per_sec") for item in benchmarks]),
            "average_peak_vram_mb": average_or_none([item["benchmark"].get("peak_vram_used_mb") for item in benchmarks]),
            "max_peak_power_watts": max(
                (item["benchmark"].get("peak_power_watts") or 0.0 for item in benchmarks),
                default=0.0,
            ),
            "benchmarks": benchmarks,
        }
    finally:
        if not args.keep_last:
            ENV_PATH.write_text(original_env_text, encoding="utf-8")
            recreate_vllm()
            wait_for_runtime(
                args.gateway_host,
                args.gateway_port,
                expected_model_path=original_model_path,
                expected_context=original_context,
                expected_budget=original_budget,
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
