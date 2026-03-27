#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT_DIR / ".env"
COMPOSE_FILE = ROOT_DIR / "docker-compose.yml"
CONTAINER_NAME = "nvfp4studio-vllm"
PROJECT_NAME = "nvfp4studio"
OUT_DIR = ROOT_DIR / "data" / "exports"

BF16_PATH = Path("/models/huihui/Huihui-Qwen3.5-4B-abliterated")
NVFP4_PATH = Path("/models/nvfp4/Huihui-Qwen3.5-4B-abliterated-NVFP4")

RUNTIME_UPDATES = {
    "VLLM_RUNTIME_PROFILE": "balanced",
    "MAX_MODEL_LEN": "8192",
    "GPU_MEMORY_UTILIZATION": "0.60",
    "MAX_NUM_SEQS": "1",
    "MAX_NUM_BATCHED_TOKENS": "4096",
    "LANGUAGE_MODEL_ONLY": "true",
    "KV_CACHE_DTYPE": "",
    "KV_CACHE_MEMORY_BYTES": "",
    "CPU_OFFLOAD_GB": "0",
    "SWAP_SPACE": "8",
}

QUALITY_CASES = [
    {
        "name": "jp_identity",
        "system": "Reply in Japanese. Keep it concise. Do not reveal hidden reasoning.",
        "user": "あなたの役割を日本語で1文だけで自己紹介して。",
        "validator": "japanese_single_sentence",
        "max_tokens": 96,
    },
    {
        "name": "sequence_rule",
        "system": "Reply in Japanese. One sentence only. Do not reveal hidden reasoning.",
        "user": "数列 2, 6, 12, 20 の次の2項と規則を1文で答えて。",
        "validator": "sequence_rule",
        "max_tokens": 96,
    },
    {
        "name": "translation_en",
        "system": "Translate to English only. No extra commentary.",
        "user": "品質を保ちながら VRAM を削る。",
        "validator": "translation_en",
        "max_tokens": 64,
    },
    {
        "name": "strict_json",
        "system": "Return strict JSON only.",
        "user": "mode は local、goal は preserve quality while reducing VRAM、tone は calm にして JSON を返して。",
        "validator": "strict_json",
        "max_tokens": 96,
    },
    {
        "name": "structured_extract",
        "system": "Return strict JSON only.",
        "user": "次の文から name, gpu, context を抽出して JSON で返して。『Ken uses RTX PRO 6000 Blackwell with 256K context.』",
        "validator": "structured_extract",
        "max_tokens": 96,
    },
    {
        "name": "reasoning_suppression",
        "system": "Reply in Japanese. One short paragraph. Do not reveal hidden reasoning or chain-of-thought.",
        "user": "VRAM節約をしつつ品質を守る方針を短く説明して。",
        "validator": "no_thinking_leak",
        "max_tokens": 128,
    },
]

PERFORMANCE_CASE = {
    "system": "Reply in Japanese. Keep it short. Do not reveal hidden reasoning.",
    "user": "このモデルの雰囲気を一言で説明して。",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Huihui 4B BF16 and NVFP4 quality under the same vLLM runtime.")
    parser.add_argument("--gateway-host", default="127.0.0.1")
    parser.add_argument("--gateway-port", type=int, default=8000)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--keep-last", action="store_true")
    parser.add_argument("--bf16-path", type=Path, default=BF16_PATH)
    parser.add_argument("--bf16-name", default="Huihui-Qwen3.5-4B-abliterated-BF16")
    parser.add_argument("--nvfp4-path", type=Path, default=NVFP4_PATH)
    parser.add_argument("--nvfp4-name", default="Huihui-Qwen3.5-4B-abliterated-NVFP4")
    return parser.parse_args()


def run_command(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)


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


def wait_for_runtime(host: str, port: int, expected_model_path: str, expected_context: int, timeout: int) -> tuple[dict[str, Any], str]:
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
                if expected_model_path in cmdline:
                    return status, cmdline
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError(
        f"vLLM did not become ready for model={expected_model_path} context={expected_context}.\n"
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


def run_non_stream(host: str, port: int, model_name: str, system: str, user: str, *, max_tokens: int) -> tuple[str | None, str]:
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


def validate_case(case_name: str, text: str) -> tuple[bool, str]:
    normalized = text.strip()
    lowered = normalized.lower()

    if "<think>" in lowered or "thinking process" in lowered or "</think>" in lowered:
        return False, "thinking leak detected"

    if case_name == "japanese_single_sentence":
        passed = any(ch in normalized for ch in "あいうえおアイウエオ一-龯々。") and normalized.count("。") <= 1
        return passed, "日本語で1文に収まっているか"

    if case_name == "sequence_rule":
        passed = "30" in normalized and "42" in normalized
        return passed, "30 と 42 を含むか"

    if case_name == "translation_en":
        passed = "quality" in lowered and "vram" in lowered and (
            "reduce" in lowered or "lower" in lowered or "decrease" in lowered or "trim" in lowered
        )
        return passed, "quality / VRAM / reduce 系の英訳になっているか"

    if case_name == "strict_json":
        try:
            payload = json.loads(normalized)
        except Exception:
            return False, "JSON parse failed"
        passed = payload.get("mode") == "local" and payload.get("tone") == "calm" and "preserve quality" in payload.get("goal", "")
        return passed, "strict JSON で指定キーが揃うか"

    if case_name == "structured_extract":
        try:
            payload = json.loads(normalized)
        except Exception:
            return False, "JSON parse failed"
        passed = payload.get("name") == "Ken" and "RTX PRO 6000 Blackwell" in payload.get("gpu", "") and "256K" in payload.get("context", "")
        return passed, "抽出 JSON の3項目が正しいか"

    if case_name == "no_thinking_leak":
        passed = len(normalized) > 0
        return passed, "思考漏れなしで応答しているか"

    return False, "unknown validator"


def run_stream_probe(host: str, port: int, model_name: str) -> dict[str, Any]:
    payload = {
        "model": model_name,
        "stream": True,
        "temperature": 0.2,
        "top_p": 0.95,
        "max_tokens": 96,
        "messages": [
            {"role": "system", "content": PERFORMANCE_CASE["system"]},
            {"role": "user", "content": PERFORMANCE_CASE["user"]},
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
                    text_parts.append(content)
    benchmark = fetch_benchmark(host, port, request_id) if request_id else {}
    return {
        "request_id": request_id,
        "response_text": "".join(text_parts).strip(),
        "benchmark": benchmark,
    }


def write_report(results: dict[str, Any]) -> tuple[Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    json_path = OUT_DIR / f"quality-compare-4b-{stamp}.json"
    md_path = OUT_DIR / f"quality-compare-4b-{stamp}.md"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 4B Quality Compare",
        "",
        f"- created_at: `{results['created_at']}`",
        f"- runtime: `{results['runtime']}`",
        "",
    ]
    for model in results["models"]:
        lines.extend(
            [
                f"## {model['label']}",
                "",
                f"- model_path: `{model['model_path']}`",
                f"- idle_vram_mb: `{model['status'].get('gpu_process_memory_mb')}`",
                f"- runtime_context: `{model['status'].get('advisory', {}).get('runtime_max_context')}`",
                f"- canary_score: `{model['score']}/{model['total_cases']}`",
                "",
                "| Case | Pass | Note |",
                "| --- | --- | --- |",
            ]
        )
        for case in model["cases"]:
            lines.append(f"| {case['name']} | {'yes' if case['passed'] else 'no'} | {case['note']} |")
        lines.extend(
            [
                "",
                f"- stream_ttft_ms: `{model['stream'].get('benchmark', {}).get('ttft_ms')}`",
                f"- stream_tok_s: `{model['stream'].get('benchmark', {}).get('completion_tokens_per_second')}`",
                f"- stream_text: `{model['stream'].get('response_text', '')}`",
                "",
            ]
        )
        for case in model["cases"]:
            lines.extend(
                [
                    f"### {case['name']}",
                    "",
                    f"system: `{case['system']}`",
                    "",
                    f"user: `{case['user']}`",
                    "",
                    "response:",
                    "```text",
                    case["response_text"],
                    "```",
                    "",
                ]
            )

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def main() -> None:
    args = parse_args()
    _, original_env = read_env_file(ENV_PATH)
    original_subset = {key: original_env.get(key, "") for key in ["MODEL_PATH", "SERVED_MODEL_NAME", *RUNTIME_UPDATES.keys()]}

    models = [
        {"label": "BF16", "path": args.bf16_path, "served_name": args.bf16_name},
        {"label": "NVFP4", "path": args.nvfp4_path, "served_name": args.nvfp4_name},
    ]

    results: dict[str, Any] = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "runtime": RUNTIME_UPDATES,
        "models": [],
    }

    try:
        for model in models:
            updates = {
                "MODEL_PATH": str(model["path"]),
                "SERVED_MODEL_NAME": model["served_name"],
                **RUNTIME_UPDATES,
            }
            write_env_updates(ENV_PATH, updates)
            recreate_vllm()
            status, _cmdline = wait_for_runtime(
                args.gateway_host,
                args.gateway_port,
                expected_model_path=str(model["path"]),
                expected_context=int(RUNTIME_UPDATES["MAX_MODEL_LEN"]),
                timeout=args.timeout,
            )

            model_result: dict[str, Any] = {
                "label": model["label"],
                "model_path": str(model["path"]),
                "served_name": model["served_name"],
                "status": status,
                "cases": [],
            }

            passed_count = 0
            for case in QUALITY_CASES:
                _request_id, response_text = run_non_stream(
                    args.gateway_host,
                    args.gateway_port,
                    model["served_name"],
                    case["system"],
                    case["user"],
                    max_tokens=case["max_tokens"],
                )
                passed, note = validate_case(case["validator"], response_text)
                if passed:
                    passed_count += 1
                model_result["cases"].append(
                    {
                        "name": case["name"],
                        "system": case["system"],
                        "user": case["user"],
                        "response_text": response_text,
                        "passed": passed,
                        "note": note,
                    }
                )

            model_result["score"] = passed_count
            model_result["total_cases"] = len(QUALITY_CASES)
            model_result["stream"] = run_stream_probe(args.gateway_host, args.gateway_port, model["served_name"])
            results["models"].append(model_result)
    finally:
        if not args.keep_last:
            write_env_updates(ENV_PATH, original_subset)
            recreate_vllm()

    json_path, md_path = write_report(results)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nSaved reports:\n- {json_path}\n- {md_path}")


if __name__ == "__main__":
    main()
