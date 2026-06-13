from __future__ import annotations

import runpy
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
CANONICAL_SCRIPT = SCRIPT_DIR / "train_gpt.py"
SCRIPT_DIR_STR = str(SCRIPT_DIR)
if SCRIPT_DIR_STR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR_STR)


if __name__ == "__main__":
    sys.argv[0] = str(Path(__file__).resolve())
    runpy.run_path(str(CANONICAL_SCRIPT), run_name="__main__")
else:
    from train_gpt import *  # noqa: F401,F403
    from train_gpt import _fast_compiled_cli_argv, _fast_compiled_cli_main  # noqa: F401
