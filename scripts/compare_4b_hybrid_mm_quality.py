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
SEFETENSORS_ROOT = ROOT_DIR.parent
OUT_DIR = ROOT_DIR / "data" / "exports"
VLLM_IMAGE = "nvfp4studio-vllm:latest"
VLLM_SCRIPT_DIR = ROOT_DIR / "config" / "vllm"

BF16_PATH = SEFETENSORS_ROOT / "huihui" / "Huihui-Qwen3.5-4B-abliterated"
HYBRID_PATH = (
    SEFETENSORS_ROOT
    / "nvfp4"
    / "Huihui-Qwen3.5-4B-abliterated-NVFP4-hybrid-mm"
)

COMMON_ENV = {
    "TENSOR_PARALLEL_SIZE": "1",
    "MAX_MODEL_LEN": "4096",
    "GPU_MEMORY_UTILIZATION": "0.25",
    "MAX_NUM_SEQS": "1",
    "KV_CACHE_DTYPE": "fp8",
    "KV_CACHE_MEMORY_BYTES": "1G",
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

LEAK_PATTERNS_EN = [
    "the user asked",
    "i need to",
    "i should",
    "i'll",
    "something like",
    "keep it concise",
    "one sentence",
    "one-sentence",
    "reasoning",
]
LEAK_PATTERNS_ZH = [
    "用户要求",
    "我需要",
    "我应该",
    "让我",
    "思考过程",
    "推理过程",
    "一句话介绍",
]
LEAK_PATTERNS_GENERIC = [
    "<think>",
    "</think>",
    "thinking process",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Huihui Qwen3.5 4B BF16 and hybrid NVFP4-mm using temporary vLLM containers."
    )
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--bf16-port", type=int, default=18082)
    parser.add_argument("--hybrid-port", type=int, default=18081)
    parser.add_argument("--keep-containers", action="store_true")
    return parser.parse_args()


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=check, capture_output=True, text=True)


def http_json(url: str, *, payload: dict[str, Any] | None = None, timeout: int = 600) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_status_code(url: str, timeout: int = 10) -> int:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        resp.read()
        return resp.status


def stop_container(name: str) -> None:
    run(["docker", "rm", "-f", name], check=False)


def start_container(
    *,
    name: str,
    model_path: Path,
    port: int,
    served_name: str,
    log_dir: Path,
) -> None:
    stop_container(name)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "vllm.log"
    if log_file.exists():
        log_file.unlink()

    command = [
        "docker",
        "run",
        "-d",
        "--name",
        name,
        "--entrypoint",
        "/bin/bash",
        "--gpus",
        "all",
        "--ipc",
        "host",
        "--shm-size",
        "8gb",
        "-p",
        f"127.0.0.1:{port}:{port}",
        "-e",
        f"MODEL_PATH={model_path}",
        "-e",
        "VLLM_HOST=0.0.0.0",
        "-e",
        f"VLLM_PORT={port}",
        "-e",
        f"SERVED_MODEL_NAME={served_name}",
    ]
    for key, value in COMMON_ENV.items():
        command.extend(["-e", f"{key}={value}"])
    command.extend(
        [
            "-v",
            f"{SEFETENSORS_ROOT}:{SEFETENSORS_ROOT}",
            "-v",
            f"{VLLM_SCRIPT_DIR}:/workspace/config/vllm:ro",
            "-v",
            f"{log_dir}:/workspace/logs",
            VLLM_IMAGE,
            "/workspace/config/vllm/start-vllm.sh",
        ]
    )
    run(command)


def wait_ready(port: int, timeout: int) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if http_status_code(f"http://127.0.0.1:{port}/health", timeout=5) == 200:
                return
        except Exception:
            pass
        time.sleep(3)
    raise TimeoutError(f"vLLM on port {port} did not become healthy in time")


def request_chat(port: int, model: str, system: str, user: str, max_tokens: int) -> dict[str, Any]:
    payload = {
        "model": model,
        "stream": False,
        "temperature": 0.2,
        "top_p": 0.95,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    return http_json(f"http://127.0.0.1:{port}/v1/chat/completions", payload=payload, timeout=600)


def extract_text(payload: dict[str, Any]) -> str:
    return payload["choices"][0]["message"]["content"].strip()


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


def compare_models(args: argparse.Namespace) -> dict[str, Any]:
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    models = [
        {
            "label": "BF16",
            "model_path": BF16_PATH,
            "served_name": "Huihui-Qwen3.5-4B-bf16",
            "container_name": "bf16-4b-test",
            "port": args.bf16_port,
            "log_dir": Path("/tmp/bf16-4b-logs"),
        },
        {
            "label": "HYBRID_NVFP4_MM",
            "model_path": HYBRID_PATH,
            "served_name": "Huihui-Qwen3.5-4B-hybrid-mm",
            "container_name": "hybrid4b-test",
            "port": args.hybrid_port,
            "log_dir": Path("/tmp/hybrid4b-logs"),
        },
    ]

    results: dict[str, Any] = {
        "created_at": timestamp,
        "common_env": COMMON_ENV,
        "models": [],
    }

    for model in models:
        start_container(
            name=model["container_name"],
            model_path=model["model_path"],
            port=model["port"],
            served_name=model["served_name"],
            log_dir=model["log_dir"],
        )
        wait_ready(model["port"], args.timeout)

        model_result = {
            "label": model["label"],
            "model_path": str(model["model_path"]),
            "served_name": model["served_name"],
            "port": model["port"],
            "health_status": http_status_code(f"http://127.0.0.1:{model['port']}/health"),
            "models_response": http_json(f"http://127.0.0.1:{model['port']}/v1/models"),
            "cases": [],
        }
        score = 0
        for case in QUALITY_CASES:
            payload = request_chat(
                model["port"],
                model["served_name"],
                case["system"],
                case["user"],
                case["max_tokens"],
            )
            text = extract_text(payload)
            passed, note = validate_case(case["expectation"], text)
            if passed:
                score += 1
            model_result["cases"].append(
                {
                    "name": case["name"],
                    "system": case["system"],
                    "user": case["user"],
                    "response_text": text,
                    "passed": passed,
                    "note": note,
                    "usage": payload.get("usage"),
                    "finish_reason": payload.get("choices", [{}])[0].get("finish_reason"),
                }
            )
        model_result["score"] = score
        model_result["total_cases"] = len(QUALITY_CASES)
        model_result["docker_stats"] = run(
            [
                "docker",
                "stats",
                "--no-stream",
                model["container_name"],
                "--format",
                "table {{.Name}}\t{{.MemUsage}}\t{{.CPUPerc}}\t{{.PIDs}}",
            ]
        ).stdout.strip()
        model_result["log_tail"] = run(
            ["tail", "-n", "120", str(model["log_dir"] / "vllm.log")]
        ).stdout
        results["models"].append(model_result)

        if not args.keep_containers:
            stop_container(model["container_name"])

    return results


def write_report(results: dict[str, Any]) -> tuple[Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    json_path = OUT_DIR / f"quality-compare-4b-hybrid-mm-{stamp}.json"
    md_path = OUT_DIR / f"quality-compare-4b-hybrid-mm-{stamp}.md"

    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 4B BF16 vs Hybrid NVFP4-MM",
        "",
        f"- created_at: `{results['created_at']}`",
        "",
    ]
    for model in results["models"]:
        lines.extend(
            [
                f"## {model['label']}",
                "",
                f"- model_path: `{model['model_path']}`",
                f"- score: `{model['score']}/{model['total_cases']}`",
                f"- health_status: `{model['health_status']}`",
                "",
                "| Case | Pass | Note |",
                "| --- | --- | --- |",
            ]
        )
        for case in model["cases"]:
            lines.append(
                f"| {case['name']} | {'yes' if case['passed'] else 'no'} | {case['note']} |"
            )
        lines.extend(["", "docker stats:", "```text", model["docker_stats"], "```", ""])
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
    try:
        results = compare_models(args)
    finally:
        if not args.keep_containers:
            stop_container("bf16-4b-test")
            stop_container("hybrid4b-test")

    json_path, md_path = write_report(results)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nSaved reports:\n- {json_path}\n- {md_path}")


if __name__ == "__main__":
    main()
