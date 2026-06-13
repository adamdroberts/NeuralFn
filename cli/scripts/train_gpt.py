from __future__ import annotations

import runpy
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
COMPAT_SCRIPT = SCRIPT_DIR / "train_gpt2.py"


if __name__ == "__main__":
    sys.argv[0] = str(Path(__file__).resolve())
    runpy.run_path(str(COMPAT_SCRIPT), run_name="__main__")
