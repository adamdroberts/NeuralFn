#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "neuralfn" / "csrc" / "native_train" / "shipped_gpt_template_presets.h"


def _catalog() -> tuple[str, ...]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from neuralfn.config import SHIPPED_GPT_TEMPLATE_PRESETS

    return tuple(SHIPPED_GPT_TEMPLATE_PRESETS)


def render_header(presets: tuple[str, ...]) -> str:
    items = "\n".join(f'        "{preset}",' for preset in presets)
    return f"""#pragma once

#include <string>
#include <vector>

// Generated from neuralfn.config.SHIPPED_GPT_TEMPLATE_PRESETS by
// tools/generate_native_gpt_template_catalog.py.
namespace neuralfn_native {{

inline const std::vector<std::string>& shipped_gpt_template_presets() {{
    static const std::vector<std::string> presets = {{
{items}
    }};
    return presets;
}}

}}  // namespace neuralfn_native
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the native GPT preset catalog header.")
    parser.add_argument("--check", action="store_true", help="fail if the checked-in header is stale")
    args = parser.parse_args(argv)

    expected = render_header(_catalog())
    if args.check:
        current = HEADER.read_text(encoding="utf-8")
        if current == expected:
            return 0
        diff = difflib.unified_diff(
            current.splitlines(keepends=True),
            expected.splitlines(keepends=True),
            fromfile=str(HEADER),
            tofile=f"{HEADER} (generated)",
        )
        sys.stderr.writelines(diff)
        return 1

    HEADER.write_text(expected, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
