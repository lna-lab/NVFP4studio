#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_MODEL_PATH = Path("/models/huihui/Huihui-Qwen3.5-4B-abliterated")
DEFAULT_OUTPUT_PATH = Path("/models/nvfp4/Huihui-Qwen3.5-4B-abliterated-NVFP4-text-gateup")
DEFAULT_IGNORE = [
    "lm_head",
    "re:.*visual.*",
    "re:.*linear_attn.*",
    "re:.*mtp.*",
]
TRANSFORMERS_REF = "09832b2ae515cfbd020327f5d3ba2dafe6edf83c"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Text-only gate/up-only NVFP4 retry for Huihui Qwen3.5 4B."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--dataset-id", default="cnn_dailymail")
    parser.add_argument("--dataset-config", default="3.0.0")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--num-calibration-samples", type=int, default=256)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--metadata-name", default="nvfp4_text_gateup_metadata.json")
    return parser.parse_args()


def load_dependencies() -> tuple[object, object, object, object, object]:
    try:
        from compressed_tensors.quantization.quant_scheme import NVFP4
        from datasets import load_dataset
        from transformers import AutoTokenizer
        from transformers.models.qwen3_5 import Qwen3_5ForCausalLM

        try:
            import compressed_tensors.entrypoints.convert as ct_convert
            from compressed_tensors.utils.safetensors_load import (
                find_config_path,
                get_checkpoint_files,
                is_weights_file,
                update_safetensors_index,
            )

            if not hasattr(ct_convert, "find_config_path"):
                ct_convert.find_config_path = find_config_path
            if not hasattr(ct_convert, "get_checkpoint_files"):
                ct_convert.get_checkpoint_files = get_checkpoint_files
            if not hasattr(ct_convert, "is_weights_file"):
                ct_convert.is_weights_file = is_weights_file
            if not hasattr(ct_convert, "update_safetensors_index"):
                ct_convert.update_safetensors_index = update_safetensors_index
        except Exception:
            pass

        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization import QuantizationModifier
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"必要な量子化依存の import に失敗しました: {exc}") from exc

    return (
        NVFP4,
        load_dataset,
        AutoTokenizer,
        Qwen3_5ForCausalLM,
        oneshot,
        QuantizationModifier,
    )


def write_metadata(output_dir: Path, args: argparse.Namespace) -> None:
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_model_path": str(args.model_path),
        "output_dir": str(args.output_dir),
        "transformers_ref": TRANSFORMERS_REF,
        "dataset_id": args.dataset_id,
        "dataset_config": args.dataset_config,
        "dataset_split": args.dataset_split,
        "num_calibration_samples": args.num_calibration_samples,
        "max_seq_length": args.max_seq_length,
        "strategy": "text_only_gate_up_only",
        "ignore": DEFAULT_IGNORE,
        "targets": ["re:.*mlp\\.(gate_proj|up_proj)$"],
        "notes": [
            "loads Huihui 4B as Qwen3_5ForCausalLM (text-only)",
            "uses CNN/DailyMail 256-sample calibration",
            "quantizes gate_proj and up_proj only",
            "down_proj remains unquantized",
        ],
    }
    (output_dir / args.metadata_name).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_text_from_example(example: dict[str, object]) -> str:
    article = str(example.get("article", "")).strip()
    highlights = str(example.get("highlights", "")).strip()
    if article and highlights:
        return f"{article}\n\nSummary:\n{highlights}"
    if article:
        return article
    if "text" in example and isinstance(example["text"], str):
        return example["text"]
    raise ValueError(f"Unsupported calibration example schema: {sorted(example.keys())}")


def main() -> None:
    args = parse_args()
    NVFP4, load_dataset, AutoTokenizer, Qwen3_5ForCausalLM, oneshot, QuantizationModifier = load_dependencies()

    if not args.model_path.exists():
        raise SystemExit(f"model path が見つかりません: {args.model_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Load text-only model from {args.model_path}")
    model = Qwen3_5ForCausalLM.from_pretrained(
        str(args.model_path),
        dtype=args.dtype,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)

    print(
        f"[2/5] Load calibration dataset {args.dataset_id}/{args.dataset_config}:{args.dataset_split} "
        f"({args.num_calibration_samples} samples)"
    )
    ds = load_dataset(
        args.dataset_id,
        args.dataset_config,
        split=f"{args.dataset_split}[:{args.num_calibration_samples}]",
    ).shuffle(seed=42)

    ds = ds.map(lambda ex: {"text": build_text_from_example(ex)})

    def tokenize(sample: dict[str, str]) -> dict[str, object]:
        return tokenizer(
            sample["text"],
            padding=False,
            truncation=True,
            max_length=args.max_seq_length,
            add_special_tokens=False,
        )

    ds = ds.map(tokenize, remove_columns=ds.column_names)

    print("[3/5] Build text-only gate/up-only recipe")
    scheme_nvfp4 = copy.deepcopy(NVFP4)
    scheme_nvfp4["targets"] = ["re:.*mlp\\.(gate_proj|up_proj)$"]
    recipe = QuantizationModifier(
        config_groups={"nvfp4_gate_up": scheme_nvfp4},
        ignore=DEFAULT_IGNORE,
    )

    print("[4/5] Apply calibrated oneshot quantization")
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=args.num_calibration_samples,
    )

    print(f"[5/5] Save compressed model to {args.output_dir}")
    model.save_pretrained(args.output_dir, save_compressed=True)
    tokenizer.save_pretrained(args.output_dir)
    write_metadata(args.output_dir, args)
    print("Done.")


if __name__ == "__main__":
    main()
