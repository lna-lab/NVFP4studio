# Script Guide

This directory mixes day-to-day operator helpers with research and conversion utilities.

## Start Here

- Use `probe_kv_budget.py` when you want to reproduce the main VRAM-saving finding.
- Use `probe_dual_instance.py` when you want to test `1x2` versus `2x2`.
- Use `probe_parallel_requests.py` when you want to measure `1 / 2 / 3` concurrent requests on one TP runtime.
- Use `probe_mixed_models.py` when you want to evaluate colocating multiple models on one GPU.
- Use `quantize_4b_nvfp4_docker.sh` when you want the most reproducible 4B conversion path.

## Naming

- `probe_*`: runtime, VRAM, concurrency, or capacity investigations
- `compare_*`: behavior or quality comparisons between variants
- `quantize_*`: BF16 to NVFP4 conversion flows
- `assemble_*`: packaging helpers that combine artifacts into a runnable layout

## Primary Scripts

- `probe_kv_budget.py`: sweep KV cache budgets and record quality / VRAM tradeoffs
- `probe_dual_instance.py`: compare `1x2` and `2x2` multi-instance layouts
- `probe_parallel_requests.py`: compare `1 / 2 / 3` concurrent requests on one TP runtime
- `probe_mixed_models.py`: evaluate mixed-model colocated serving
- `probe_model_matrix.py`: compare multiple candidate models under one runtime preset
- `probe_4b_256k_context.py`: test 4B behavior at long context targets
- `probe_4b_16gb_stability.py`: check whether a 4B build stays usable in tighter VRAM envelopes
- `compare_4b_quality.py`: compare BF16 and NVFP4 4B outputs
- `compare_4b_hybrid_mm_quality.py`: compare multimodal hybrid packaging against BF16
- `quantize_4b_nvfp4.py`: base 4B NVFP4 conversion flow
- `quantize_4b_nvfp4_docker.sh`: Docker wrapper for the base 4B conversion flow
- `quantize_4b_calibrated_nvfp4.py`: calibrated mixed-precision conversion experiment
- `quantize_4b_calibrated_nvfp4_docker.sh`: Docker wrapper for the calibrated conversion flow
- `quantize_4b_text_gateup_nvfp4.py`: text-only gate/up-only retry route
- `assemble_4b_hybrid_multimodal_nvfp4.py`: rebuild a multimodal package around quantized text weights
