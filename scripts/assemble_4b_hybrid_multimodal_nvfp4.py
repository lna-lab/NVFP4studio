#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_ORIGINAL_MODEL_PATH = Path(
    "/models/huihui/Huihui-Qwen3.5-4B-abliterated"
)
DEFAULT_QUANT_MODEL_PATH = Path(
    "/models/nvfp4/Huihui-Qwen3.5-4B-abliterated-NVFP4-text-gateup"
)
DEFAULT_OUTPUT_PATH = Path(
    "/models/nvfp4/Huihui-Qwen3.5-4B-abliterated-NVFP4-hybrid-mm"
)
SUPPORT_FILE_SKIP_PREFIXES = ("model",)
SUPPORT_FILE_SKIP_NAMES = {"config.json"}
QUANT_FILE_NAME = "model.safetensors"
METADATA_FILE_NAME = "hybrid_mm_metadata.json"
HYBRID_IGNORE_PATTERNS = [
    "re:.*visual.*",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble a hybrid multimodal Huihui Qwen3.5 4B package by "
            "combining the original multimodal wrapper with a text-only "
            "NVFP4 language_model checkpoint."
        )
    )
    parser.add_argument(
        "--original-model-path",
        type=Path,
        default=DEFAULT_ORIGINAL_MODEL_PATH,
    )
    parser.add_argument(
        "--quant-model-path",
        type=Path,
        default=DEFAULT_QUANT_MODEL_PATH,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    parser.add_argument(
        "--link-mode",
        choices=("auto", "copy", "hardlink", "symlink"),
        default="auto",
        help="How to materialize large files into the output folder.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete the output directory first if it already exists.",
    )
    return parser.parse_args()


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"{label} が見つかりません: {path}")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def prepare_output_dir(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise SystemExit(
                f"output dir が既に存在します: {path}\n"
                "上書きする場合は --force を付けてください。"
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def materialize_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()

    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "hardlink":
        os.link(src, dst)
        return
    if mode == "symlink":
        os.symlink(src, dst)
        return

    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def should_copy_support_file(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.name in SUPPORT_FILE_SKIP_NAMES:
        return False
    return not path.name.startswith(SUPPORT_FILE_SKIP_PREFIXES)


def build_hybrid_config(original_cfg: dict, quant_cfg: dict) -> dict:
    hybrid_cfg = copy.deepcopy(original_cfg)
    hybrid_cfg["quantization_config"] = copy.deepcopy(quant_cfg["quantization_config"])
    hybrid_cfg["dtype"] = quant_cfg.get("dtype", hybrid_cfg.get("dtype"))
    hybrid_cfg["transformers_version"] = quant_cfg.get(
        "transformers_version",
        hybrid_cfg.get("transformers_version"),
    )

    text_cfg = hybrid_cfg.get("text_config", {})
    for key, value in quant_cfg.items():
        if key in text_cfg:
            text_cfg[key] = value
    hybrid_cfg["text_config"] = text_cfg

    ignore_list = hybrid_cfg["quantization_config"].get("ignore", [])
    ignore_list = [
        prefix_language_model_path(item) for item in ignore_list
    ]
    for pattern in HYBRID_IGNORE_PATTERNS:
        if pattern not in ignore_list:
            ignore_list.append(pattern)
    hybrid_cfg["quantization_config"]["ignore"] = ignore_list
    return hybrid_cfg


def prefix_language_model_path(item: str) -> str:
    if item.startswith("model."):
        return f"model.language_model.{item[len('model.'):]}"
    return item


def build_hybrid_index(
    original_index: dict,
    quant_file_name: str,
) -> dict:
    new_weight_map: dict[str, str] = {}
    for key, file_name in original_index["weight_map"].items():
        if key.startswith("model.language_model."):
            new_weight_map[key] = quant_file_name
        else:
            new_weight_map[key] = file_name
    return {"metadata": {}, "weight_map": new_weight_map}


def compute_total_size(output_dir: Path, index_payload: dict) -> int:
    file_names = set(index_payload["weight_map"].values())
    return sum((output_dir / name).stat().st_size for name in file_names)


def write_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    original_index: dict,
    hybrid_index: dict,
) -> None:
    non_language_files = sorted(
        {
            file_name
            for key, file_name in original_index["weight_map"].items()
            if not key.startswith("model.language_model.")
        }
    )
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "original_model_path": str(args.original_model_path),
        "quant_model_path": str(args.quant_model_path),
        "output_dir": str(args.output_dir),
        "link_mode": args.link_mode,
        "strategy": "original_multimodal_wrapper_plus_quantized_language_model",
        "language_model_weight_file": QUANT_FILE_NAME,
        "visual_weight_files": non_language_files,
        "language_model_weight_count": sum(
            1
            for key in hybrid_index["weight_map"]
            if key.startswith("model.language_model.")
        ),
        "visual_weight_count": sum(
            1
            for key in hybrid_index["weight_map"]
            if not key.startswith("model.language_model.")
        ),
        "notes": [
            "keeps original Huihui multimodal config and processor files",
            "injects NVFP4 quantization_config from text-only gate/up retry",
            "routes model.language_model.* to the quantized single-file checkpoint",
            "routes model.visual.* and other non-language weights to the original shard(s)",
        ],
    }
    save_json(output_dir / METADATA_FILE_NAME, payload)


def main() -> None:
    args = parse_args()
    ensure_exists(args.original_model_path, "original model path")
    ensure_exists(args.quant_model_path, "quant model path")
    prepare_output_dir(args.output_dir, force=args.force)

    original_cfg = load_json(args.original_model_path / "config.json")
    original_index = load_json(args.original_model_path / "model.safetensors.index.json")
    quant_cfg = load_json(args.quant_model_path / "config.json")

    print("[1/5] Copy support files from original multimodal folder")
    for path in sorted(args.original_model_path.iterdir()):
        if should_copy_support_file(path):
            materialize_file(path, args.output_dir / path.name, args.link_mode)

    print("[2/5] Materialize original non-language weight shard(s)")
    non_language_files = sorted(
        {
            file_name
            for key, file_name in original_index["weight_map"].items()
            if not key.startswith("model.language_model.")
        }
    )
    for file_name in non_language_files:
        materialize_file(
            args.original_model_path / file_name,
            args.output_dir / file_name,
            args.link_mode,
        )

    print("[3/5] Materialize quantized language_model checkpoint")
    materialize_file(
        args.quant_model_path / "model.safetensors",
        args.output_dir / QUANT_FILE_NAME,
        args.link_mode,
    )
    for extra_name in ("recipe.yaml", "nvfp4_text_gateup_metadata.json"):
        extra_path = args.quant_model_path / extra_name
        if extra_path.exists():
            materialize_file(extra_path, args.output_dir / extra_name, args.link_mode)

    print("[4/5] Write hybrid config.json")
    hybrid_cfg = build_hybrid_config(original_cfg, quant_cfg)
    save_json(args.output_dir / "config.json", hybrid_cfg)

    print("[5/5] Write hybrid index + metadata")
    hybrid_index = build_hybrid_index(original_index, QUANT_FILE_NAME)
    hybrid_index["metadata"]["total_size"] = compute_total_size(
        args.output_dir,
        hybrid_index,
    )
    save_json(args.output_dir / "model.safetensors.index.json", hybrid_index)
    write_metadata(args.output_dir, args, original_index, hybrid_index)

    print(f"Done: {args.output_dir}")


if __name__ == "__main__":
    main()
