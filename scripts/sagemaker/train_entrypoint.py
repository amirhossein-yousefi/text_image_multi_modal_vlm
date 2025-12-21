#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""SageMaker training entrypoint.

This script is intended to be used as the SageMaker/HuggingFace Estimator `entry_point`.
It adapts SageMaker's training environment variables to this repo's existing trainers.

Expected input layout inside a data channel (typically `training`):
  train.csv
  val.csv
  test.csv (optional)
  class_names.txt (optional)
  images/ ...

SageMaker provides:
  - SM_CHANNEL_TRAINING: directory containing the `training` channel
  - SM_MODEL_DIR: directory where model artifacts should be saved

We translate these into the flags expected by:
  - text_image_multi_modal_vlm.qwen_vlm
  - text_image_multi_modal_vlm.paligemma
  - text_image_multi_modal_vlm.smol_vlm
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List


def _read_class_names(path: Path) -> List[str]:
    if not path.exists():
        return []
    labels: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            v = line.strip()
            if v:
                labels.append(v)
    return labels


def _as_comma(labels: List[str]) -> str:
    return ",".join([x.strip() for x in labels if x.strip()])


def _resolve_channel_dir(env_name: str, fallback: str | None = None) -> str:
    v = os.environ.get(env_name)
    if v:
        return v
    if fallback is not None:
        return fallback
    raise RuntimeError(f"Missing required SageMaker env var: {env_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SageMaker adapter for repo training scripts")
    parser.add_argument("--model", choices=["qwen", "paligemma", "smol"], required=True)

    # Where data lives in the container. Defaults to SageMaker's `training` channel.
    parser.add_argument("--data_dir", default=os.environ.get("SM_CHANNEL_TRAINING", ""))

    # Allow overriding file names if you packaged differently.
    parser.add_argument("--train_csv_name", default="train.csv")
    parser.add_argument("--val_csv_name", default="val.csv")
    parser.add_argument("--test_csv_name", default="test.csv")
    parser.add_argument("--class_names_file", default="class_names.txt")

    # Optional explicit class names (comma-separated). If omitted, we try class_names.txt.
    parser.add_argument("--class_names", default="")

    # Model selection / overrides
    parser.add_argument("--model_id", default="")

    # Forwarding: pass any additional arguments to the underlying module as-is.
    # Example: -- --learning_rate 1e-4 --num_train_epochs 3
    args, remainder = parser.parse_known_args()

    if not args.data_dir:
        # If user didn't pass --data_dir and SageMaker env is missing, still give a clear error.
        _resolve_channel_dir("SM_CHANNEL_TRAINING")

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir does not exist: {data_dir}")

    train_csv = data_dir / args.train_csv_name
    val_csv = data_dir / args.val_csv_name
    test_csv = data_dir / args.test_csv_name

    # For this repo, image_root should usually be the folder that contains `images/`.
    image_root = str(data_dir)

    class_names = args.class_names.strip()
    if not class_names:
        labels = _read_class_names(data_dir / args.class_names_file)
        class_names = _as_comma(labels)

    if not class_names:
        raise RuntimeError(
            "No class names provided. Supply --class_names 'a,b,c' or include class_names.txt in the channel."
        )

    out_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

    module_name = {
        "qwen": "text_image_multi_modal_vlm.qwen_vlm",
        "paligemma": "text_image_multi_modal_vlm.paligemma",
        "smol": "text_image_multi_modal_vlm.smol_vlm",
    }[args.model]

    # Import lazily so this script can be validated without transformers installed.
    import importlib

    module = importlib.import_module(module_name)
    if not hasattr(module, "main"):
        raise SystemExit(f"Selected module {module_name} has no main()")

    forwarded: List[str] = []
    forwarded += ["--train_csv", str(train_csv)]
    forwarded += ["--val_csv", str(val_csv)]
    if test_csv.exists():
        forwarded += ["--test_csv", str(test_csv)]
    forwarded += ["--image_root", image_root]
    forwarded += ["--class_names", class_names]
    forwarded += ["--out_dir", out_dir]

    if args.model_id.strip():
        forwarded += ["--model_id", args.model_id.strip()]

    # Append any extra flags intended for the underlying trainer.
    forwarded += remainder

    # Make sure our package is importable inside the training container.
    # (SageMaker's HF containers run from /opt/ml/code.)
    if str(Path(__file__).resolve().parents[2] / "src") not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

    sys.argv = [module.__file__] + forwarded
    module.main()


if __name__ == "__main__":
    main()
