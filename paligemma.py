"""
Backward-compatible wrapper for the PaliGemma training script after the src/ layout refactor.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from text_image_multi_modal_vlm.paligemma import main


if __name__ == "__main__":
    main()
