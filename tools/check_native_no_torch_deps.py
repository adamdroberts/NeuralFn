#!/usr/bin/env python3
"""Fail if native training artifacts link Python/Torch runtime libraries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


DEFAULT_ARTIFACTS = (
    Path("build/nfn_gpt_native_train"),
    Path("build/libnfn_native_train_tile_ops.so"),
)
FORBIDDEN_LIBRARY_MARKERS = (
    "libtorch",
    "libtorch_cpu",
    "libtorch_cuda",
    "libc10",
    "libc10_cuda",
    "libpython",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "artifacts",
        nargs="*",
        type=Path,
        default=list(DEFAULT_ARTIFACTS),
        help="Native executable/shared-library artifacts to inspect with ldd.",
    )
    parser.add_argument("--json", action="store_true", help="Print a machine-readable report.")
    return parser.parse_args()


def ldd_output(path: Path) -> str:
    proc = subprocess.run(
        ["ldd", str(path)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc.stdout


def main() -> int:
    args = parse_args()
    report: list[dict[str, object]] = []
    failed = False
    for artifact in args.artifacts:
        path = artifact.expanduser()
        entry: dict[str, object] = {"artifact": str(path), "exists": path.exists(), "forbidden": []}
        if not path.exists():
            entry["error"] = "missing"
            failed = True
            report.append(entry)
            continue
        output = ldd_output(path)
        forbidden = [
            line.strip()
            for line in output.splitlines()
            if any(marker in line for marker in FORBIDDEN_LIBRARY_MARKERS)
        ]
        entry["forbidden"] = forbidden
        if forbidden:
            failed = True
        report.append(entry)

    if args.json:
        print(json.dumps({"passed": not failed, "artifacts": report}, indent=2))
    else:
        for entry in report:
            if not entry["exists"]:
                print(f"{entry['artifact']}: missing", file=sys.stderr)
                continue
            forbidden = entry["forbidden"]
            if forbidden:
                print(f"{entry['artifact']}: forbidden native dependency detected", file=sys.stderr)
                for line in forbidden:
                    print(f"  {line}", file=sys.stderr)
            else:
                print(f"{entry['artifact']}: ok")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
