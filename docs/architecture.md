# Project Architecture

## Overview
The repository fine-tunes large vision-language models for multi-label hateful content classification. The core logic remains in the original training scripts; this document captures how the standardized layout organizes supporting assets.

## Components
- **Data**: CSVs with `text`, `image_path`, `labels` plus `class_names.txt`. Images are stored under `data/mmhs150k/images/`. Additional staging areas (`data/raw`, `data/interim`, `data/processed`) are available for other pipelines.
- **Models**: Training outputs live in `runs/` (kept from the original project). Optionally mirror/export into `models/`.
- **Code**: All Python logic resides in `src/text_image_multi_modal_vlm/` with three entrypoints: `qwen_vlm.py`, `paligemma.py`, and `smol_vlm.py`.
- **Configs**: YAML references in `configs/` capture common hyperparameters and paths; they are informational and map 1:1 to CLI flags.
- **Scripts**: `scripts/train.py` forwards arguments to the chosen model script without altering behavior.
- **Tests**: `tests/` includes import/smoke checks to ensure the package loads.

## Training Flow
1) Prepare data in `data/mmhs150k/` (or adjust paths via CLI flags).
2) Run `python -m text_image_multi_modal_vlm.<script>` with LoRA/QLoRA flags as desired.
3) Outputs (checkpoints, metrics, logs) are written to the `--out_dir` (default under `runs/`).
4) Optional inference adapters are saved under `<out_dir>/lora_adapter` for vLLM or downstream serving.

## Inference
Each script exposes the same `main()` entrypoint and utilities for loading adapters. Example usage is documented in `README.md` with Qwen2-VL + PEFT.

## Notes
- The refactor avoids modifying training/evaluation algorithms; only paths, packaging, and documentation were standardized.
- Add notebooks to `notebooks/` and figures to `reports/figures/` to keep artifacts organized.
