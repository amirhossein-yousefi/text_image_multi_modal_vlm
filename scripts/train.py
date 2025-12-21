"""
Thin wrapper to route training runs to the model-specific scripts without altering their logic.

Usage:
  python scripts/train.py --model qwen -- --train_csv ... --val_csv ...

Arguments after `--` are forwarded untouched to the selected script's argparse.
"""

import argparse
import importlib
import sys
from pathlib import Path
from typing import Dict


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


MODEL_MODULES: Dict[str, str] = {
    "qwen": "text_image_multi_modal_vlm.qwen_vlm",
    "paligemma": "text_image_multi_modal_vlm.paligemma",
    "smol": "text_image_multi_modal_vlm.smol_vlm",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dispatch training to a specific VLM script.",
        epilog="Pass all model-specific flags after -- (see configs/*.yaml for examples).",
    )
    parser.add_argument("--model", choices=sorted(MODEL_MODULES.keys()), required=True)

    args, remainder = parser.parse_known_args()

    if not remainder:
        parser.error("no downstream arguments supplied; append -- <flags> for the target script")

    module_name = MODEL_MODULES[args.model]
    module = importlib.import_module(module_name)
    if not hasattr(module, "main"):
        raise SystemExit(f"Selected module {module_name} has no main() entrypoint")

    # Forward arguments to the target script
    sys.argv = [module.__file__] + remainder
    module.main()


if __name__ == "__main__":
    main()
