from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
COMPAT_SCRIPT = SCRIPT_DIR / "infer_gpt2.py"
SCRIPT_DIR_STR = str(SCRIPT_DIR)
if SCRIPT_DIR_STR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR_STR)


if __name__ == "__main__":
    sys.argv[0] = str(Path(__file__).resolve())
    from infer_gpt2 import main

    raise SystemExit(main())
else:
    from infer_gpt2 import *  # noqa: F401,F403
