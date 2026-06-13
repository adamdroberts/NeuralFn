from __future__ import annotations

import runpy
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
CANONICAL_SCRIPT = SCRIPT_DIR / "train_gpt.py"


if __name__ == "__main__":
    sys.argv[0] = str(Path(__file__).resolve())
    runpy.run_path(str(CANONICAL_SCRIPT), run_name="__main__")
else:
    from train_gpt import *  # noqa: F401,F403
