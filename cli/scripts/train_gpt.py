from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_gpt2 import _fast_compiled_cli_main, main  # noqa: E402


if __name__ == "__main__":
    fast_exit = _fast_compiled_cli_main(sys.argv[1:])
    if fast_exit is not None:
        raise SystemExit(fast_exit)
    raise SystemExit(main())
