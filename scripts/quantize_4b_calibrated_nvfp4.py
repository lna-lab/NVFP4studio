#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_MODEL_PATH = Path("/models/huihui/Huihui-Qwen3.5-4B-abliterated")
DEFAULT_OUTPUT_PATH = Path("/models/nvfp4/Huihui-Qwen3.5-4B-abliterated-NVFP4-calibrated-mixed")
DEFAULT_IGNORE = [
    "lm_head",
    "re:.*visual.*",
    "re:.*linear_attn.*",
    "re:.*mtp.*",
]
TRANSFORMERS_REF = "09832b2ae515cfbd020327f5d3ba2dafe6edf83c"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrated mixed-precision NVFP4 conversion for Huihui Qwen3.5 4B."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--preset",
        choices=("mixed", "mlp_only"),
        default="mixed",
        help="量子化レシピの方向性。mixed は現行試作、mlp_only はモデルカード由来の保守的レシピ。",
    )
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--dataset-id", default="")
    parser.add_argument("--dataset-config", default="")
    parser.add_argument("--dataset-split", default="")
    parser.add_argument("--num-calibration-samples", type=int, default=256)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--metadata-name", default="nvfp4_calibrated_conversion_metadata.json")
    parser.add_argument("--skip-sample", action="store_true")
    return parser.parse_args()


def load_dependencies() -> tuple[object, object, object, object, object, object]:
    try:
        from compressed_tensors.offload import dispatch_model
        from compressed_tensors.quantization.quant_scheme import FP8_DYNAMIC, NVFP4
        from datasets import load_dataset
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

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
        dispatch_model,
        FP8_DYNAMIC,
        NVFP4,
        load_dataset,
        AutoProcessor,
        Qwen3_5ForConditionalGeneration,
        oneshot,
        QuantizationModifier,
    )


def write_metadata(output_dir: Path, args: argparse.Namespace) -> None:
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_model_path": str(args.model_path),
        "output_dir": str(args.output_dir),
        "preset": args.preset,
        "dataset_id": args.dataset_id,
        "dataset_config": args.dataset_config,
        "dataset_split": args.dataset_split,
        "num_calibration_samples": args.num_calibration_samples,
        "max_seq_length": args.max_seq_length,
        "transformers_ref": TRANSFORMERS_REF,
        "strategy": "calibrated_quantization",
        "ignore": DEFAULT_IGNORE,
        "notes": metadata_notes(args.preset, args.dataset_id),
    }
    (output_dir / args.metadata_name).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def metadata_notes(preset: str, dataset_id: str) -> list[str]:
    notes = [
        "linear_attn remains ignored",
        f"calibration dataset: {dataset_id}",
    ]
    if preset == "mixed":
        notes.extend(
            [
                "down_proj uses FP8 dynamic",
                "self_attn and gate/up proj use NVFP4",
            ]
        )
    else:
        notes.extend(
            [
                "MLP-only NVFP4 recipe",
                "attention projections stay in original precision",
            ]
        )
    return notes


def resolve_dataset_defaults(args: argparse.Namespace) -> tuple[str, str | None, str]:
    if args.dataset_id:
        return args.dataset_id, args.dataset_config or None, args.dataset_split or "train"

    if args.preset == "mlp_only":
        return "cnn_dailymail", "3.0.0", "train"

    return "HuggingFaceH4/ultrachat_200k", None, "train_sft"


def build_text_from_example(processor: object, example: dict[str, object]) -> str:
    if "messages" in example and isinstance(example["messages"], list):
        return processor.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )

    if "conversations" in example and isinstance(example["conversations"], list):
        return processor.apply_chat_template(
            example["conversations"],
            tokenize=False,
            add_generation_prompt=False,
        )

    if "text" in example and isinstance(example["text"], str):
        return example["text"]

    if "article" in example and isinstance(example["article"], str):
        article = example["article"].strip()
        highlights = str(example.get("highlights", "")).strip()
        if highlights:
            return f"{article}\n\nSummary:\n{highlights}"
        return article

    prompt = str(example.get("prompt", "")).strip()
    response = str(example.get("response", "")).strip()
    if prompt or response:
        messages = []
        if prompt:
            messages.append({"role": "user", "content": prompt})
        if response:
            messages.append({"role": "assistant", "content": response})
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    source = str(example.get("input", "")).strip()
    target = str(example.get("output", "")).strip()
    if source or target:
        joined = "\n".join(part for part in [source, target] if part)
        return joined

    raise ValueError(f"Unsupported calibration example schema: {sorted(example.keys())}")


def build_recipe(quantization_modifier_cls: object, fp8_dynamic: object, nvfp4: object, preset: str, ignore: list[str]) -> object:
    if preset == "mixed":
        scheme_fp8 = copy.deepcopy(fp8_dynamic)
        scheme_fp8["targets"] = ["re:.*down_proj.*"]

        scheme_nvfp4 = copy.deepcopy(nvfp4)
        scheme_nvfp4["targets"] = [
            "re:.*self_attn\\.(q_proj|k_proj|v_proj|o_proj)$",
            "re:.*mlp\\.(gate_proj|up_proj)$",
        ]

        return quantization_modifier_cls(
            config_groups={
                "fp8_down_proj": scheme_fp8,
                "nvfp4_core": scheme_nvfp4,
            },
            ignore=ignore,
        )

    scheme_nvfp4 = copy.deepcopy(nvfp4)
    scheme_nvfp4["targets"] = [
        "re:.*mlp\\.(gate_proj|up_proj|down_proj)$",
    ]
    return quantization_modifier_cls(
        config_groups={"nvfp4_mlp": scheme_nvfp4},
        ignore=ignore,
    )


def main() -> None:
    args = parse_args()
    (
        dispatch_model,
        FP8_DYNAMIC,
        NVFP4,
        load_dataset,
        AutoProcessor,
        Qwen3_5ForConditionalGeneration,
        oneshot,
        QuantizationModifier,
    ) = load_dependencies()

    if not args.model_path.exists():
        raise SystemExit(f"model path が見つかりません: {args.model_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Load model from {args.model_path}")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(args.model_path),
        dtype=args.dtype,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(str(args.model_path), trust_remote_code=True)
    tokenizer = getattr(processor, "tokenizer", processor)

    args.dataset_id, args.dataset_config, args.dataset_split = resolve_dataset_defaults(args)

    dataset_ref = args.dataset_id
    if args.dataset_config:
        dataset_ref = f"{dataset_ref}/{args.dataset_config}"
    print(
        f"[2/6] Load calibration dataset {dataset_ref}:{args.dataset_split} "
        f"({args.num_calibration_samples} samples)"
    )
    if args.dataset_config:
        ds = load_dataset(
            args.dataset_id,
            args.dataset_config,
            split=f"{args.dataset_split}[:{args.num_calibration_samples}]",
        ).shuffle(seed=42)
    else:
        ds = load_dataset(
            args.dataset_id,
            split=f"{args.dataset_split}[:{args.num_calibration_samples}]",
        ).shuffle(seed=42)

    def preprocess(example: dict[str, object]) -> dict[str, str]:
        return {"text": build_text_from_example(processor, example)}

    ds = ds.map(preprocess)

    def tokenize(sample: dict[str, str]) -> dict[str, object]:
        return tokenizer(
            sample["text"],
            padding=False,
            truncation=True,
            max_length=args.max_seq_length,
            add_special_tokens=False,
        )

    ds = ds.map(tokenize, remove_columns=ds.column_names)

    print(f"[3/6] Build {args.preset} recipe")
    recipe = build_recipe(QuantizationModifier, FP8_DYNAMIC, NVFP4, args.preset, DEFAULT_IGNORE)

    print("[4/6] Apply calibrated oneshot quantization")
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=args.num_calibration_samples,
    )

    if not args.skip_sample:
        print("[5/6] Run a short sample generation")
        try:
            dispatch_model(model)
            messages = [{"role": "user", "content": "あなたの役割を日本語で一言で教えて。"}]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            output = model.generate(**inputs, max_new_tokens=64)
            sample = tokenizer.decode(output[0], skip_special_tokens=True)
            print("\n========== SAMPLE GENERATION ==========")
            print(sample)
            print("======================================\n")
        except Exception as exc:  # pragma: no cover
            print(f"[warn] sample generation failed but save will continue: {exc}")
    else:
        print("[5/6] Skip sample generation")

    print(f"[6/6] Save compressed model to {args.output_dir}")
    model.save_pretrained(args.output_dir, save_compressed=True)
    processor.save_pretrained(args.output_dir)
    write_metadata(args.output_dir, args)
    print("Done.")


if __name__ == "__main__":
    main()
