#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_MODEL_PATH = Path("/models/huihui/Huihui-Qwen3.5-4B-abliterated")
DEFAULT_OUTPUT_PATH = Path("/models/nvfp4/Huihui-Qwen3.5-4B-abliterated-NVFP4")
UPSTREAM_EXAMPLE = (
    "https://github.com/vllm-project/llm-compressor/blob/"
    "cf3bd6463e8d471ad6c8cc20a6a9b053c178e555/examples/"
    "quantization_w4a16_fp4/nvfp4/qwen3.5_example.py"
)
DEFAULT_IGNORE = [
    "lm_head",
    "re:.*visual.*",
    "re:.*linear_attn.*",
    "re:.*mtp.*",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize a local Huihui Qwen3.5 model to NVFP4A16 using llm-compressor."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--sample-prompt", default="こんにちは。あなたの役割を一言で紹介して。")
    parser.add_argument("--sample-max-new-tokens", type=int, default=80)
    parser.add_argument("--scheme", default="NVFP4A16")
    parser.add_argument("--targets", default="Linear")
    parser.add_argument("--ignore", nargs="*", default=DEFAULT_IGNORE)
    parser.add_argument("--skip-sample", action="store_true")
    parser.add_argument("--metadata-name", default="nvfp4_conversion_metadata.json")
    return parser.parse_args()


def load_dependencies() -> tuple[object, object, object, object, object]:
    try:
        from compressed_tensors.offload import dispatch_model
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        # llm-compressor development snapshots sometimes expect these helpers to be
        # re-exported from compressed_tensors.entrypoints.convert. Add a small
        # compatibility shim before importing llmcompressor.
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
    except Exception as exc:  # pragma: no cover - runtime environment specific
        raise SystemExit(
            "llm-compressor 環境が見つかりません。"
            " 先に `./scripts/setup_llm_compressor_env.sh` を実行し、"
            " `source .venv-llm-compressor/bin/activate` してください。\n"
            f"詳細: {exc}"
        ) from exc

    return dispatch_model, AutoProcessor, Qwen3_5ForConditionalGeneration, oneshot, QuantizationModifier


def write_metadata(output_dir: Path, args: argparse.Namespace) -> None:
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_model_path": str(args.model_path),
        "output_dir": str(args.output_dir),
        "scheme": args.scheme,
        "targets": args.targets,
        "ignore": args.ignore,
        "sample_prompt": None if args.skip_sample else args.sample_prompt,
        "upstream_example": UPSTREAM_EXAMPLE,
    }
    (output_dir / args.metadata_name).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    dispatch_model, AutoProcessor, Qwen3_5ForConditionalGeneration, oneshot, QuantizationModifier = load_dependencies()

    if not args.model_path.exists():
        raise SystemExit(f"model path が見つかりません: {args.model_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Load model from {args.model_path}")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(args.model_path),
        dtype=args.dtype,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(str(args.model_path), trust_remote_code=True)

    print(f"[2/5] Build quantization recipe: {args.scheme}")
    recipe = QuantizationModifier(
        targets=args.targets,
        scheme=args.scheme,
        ignore=args.ignore,
    )

    print("[3/5] Apply oneshot quantization")
    oneshot(model=model, recipe=recipe)

    if not args.skip_sample:
        print("[4/5] Run a short sample generation")
        try:
            dispatch_model(model)
            messages = [{"role": "user", "content": args.sample_prompt}]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=prompt, return_tensors="pt").to(model.device)
            output = model.generate(**inputs, max_new_tokens=args.sample_max_new_tokens)
            sample = processor.decode(output[0], skip_special_tokens=True)
            print("\n========== SAMPLE GENERATION ==========")
            print(sample)
            print("======================================\n")
        except Exception as exc:  # pragma: no cover - runtime environment specific
            print(f"[warn] sample generation failed but quantization can still be saved: {exc}")
    else:
        print("[4/5] Skip sample generation")

    print(f"[5/5] Save compressed model to {args.output_dir}")
    model.save_pretrained(args.output_dir, save_compressed=True)
    processor.save_pretrained(args.output_dir)
    write_metadata(args.output_dir, args)
    print("Done.")


if __name__ == "__main__":
    main()
