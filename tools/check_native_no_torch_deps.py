#!/usr/bin/env python3
"""Fail if native training paths depend on Python/Torch runtime libraries."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from textwrap import dedent


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
FORBIDDEN_PYTHON_IMPORT_ROOTS = (
    "torch",
    "numpy",
    "tiktoken",
    "server.dataset_manager",
    "nfn_impl",
)
DEFAULT_PYTHON_ENTRYPOINTS = (
    (
        "train_gpt_fast_command",
        (
            sys.executable,
            "cli/scripts/train_gpt.py",
            "--tinystories",
            "--native-cuda-dry-run",
            "--native-cuda-print-command",
            "--native-cuda-no-checkpoint",
        ),
    ),
    (
        "nfn_train_fast_command",
        (
            sys.executable,
            "cli/nfn.py",
            "train",
            "--tinystories",
            "--native-cuda-dry-run",
            "--native-cuda-print-command",
            "--no-checkpoint",
        ),
    ),
    (
        "native_sdk_imports",
        (
            sys.executable,
            "-c",
            "import neuralfn; import neuralfn.native_gpt; import neuralfn.native_gpt2; print('native-sdk-ok')",
        ),
    ),
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
    parser.add_argument(
        "--skip-artifacts",
        action="store_true",
        help="Skip ldd inspection of compiled artifacts.",
    )
    parser.add_argument(
        "--skip-python-entrypoints",
        action="store_true",
        help="Skip import-blocked checks for native Python entrypoints.",
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


def _write_native_cli_stub(root: Path) -> Path:
    stub = root / "nfn_gpt_native_train"
    stub.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$@\"\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    return stub


def _write_import_blocker(root: Path) -> None:
    root.joinpath("sitecustomize.py").write_text(
        dedent(
            f"""
            import importlib.abc

            FORBIDDEN = {FORBIDDEN_PYTHON_IMPORT_ROOTS!r}

            class _NativeNoTorchImportBlocker(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    for root in FORBIDDEN:
                        if fullname == root or fullname.startswith(root + "."):
                            raise ImportError(
                                "native no-Torch verifier blocked import of " + fullname
                            )
                    return None

            import sys
            sys.meta_path.insert(0, _NativeNoTorchImportBlocker())
            """
        ).lstrip(),
        encoding="utf-8",
    )


def python_entrypoint_report(repo_root: Path) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="nfn-native-no-torch-") as tmp:
        temp_root = Path(tmp)
        _write_import_blocker(temp_root)
        native_cli = _write_native_cli_stub(temp_root)
        env = os.environ.copy()
        env["NFN_NATIVE_GPT_CLI"] = str(native_cli)
        env["NFN_NATIVE_GPT2_CLI"] = str(native_cli)
        env["PYTHONPATH"] = os.pathsep.join(
            part for part in (str(temp_root), str(repo_root), env.get("PYTHONPATH", "")) if part
        )
        env.setdefault("CUDA_VISIBLE_DEVICES", "0")
        env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
        for name, command in DEFAULT_PYTHON_ENTRYPOINTS:
            proc = subprocess.run(
                list(command),
                cwd=repo_root,
                env=env,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            entries.append(
                {
                    "name": name,
                    "command": list(command),
                    "returncode": proc.returncode,
                    "passed": proc.returncode == 0,
                    "stdout": proc.stdout.strip(),
                    "stderr": proc.stderr.strip(),
                }
            )
    return entries


def main() -> int:
    args = parse_args()
    artifact_report: list[dict[str, object]] = []
    failed = False
    if not args.skip_artifacts:
        for artifact in args.artifacts:
            path = artifact.expanduser()
            entry: dict[str, object] = {"artifact": str(path), "exists": path.exists(), "forbidden": []}
            if not path.exists():
                entry["error"] = "missing"
                failed = True
                artifact_report.append(entry)
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
            artifact_report.append(entry)

    python_report: list[dict[str, object]] = []
    if not args.skip_python_entrypoints:
        repo_root = Path(__file__).resolve().parents[1]
        python_report = python_entrypoint_report(repo_root)
        failed = failed or any(not bool(entry["passed"]) for entry in python_report)

    if args.json:
        print(
            json.dumps(
                {
                    "passed": not failed,
                    "forbidden_python_import_roots": list(FORBIDDEN_PYTHON_IMPORT_ROOTS),
                    "artifacts": artifact_report,
                    "python_entrypoints": python_report,
                },
                indent=2,
            )
        )
    else:
        for entry in artifact_report:
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
        for entry in python_report:
            if entry["passed"]:
                print(f"{entry['name']}: ok")
            else:
                print(f"{entry['name']}: failed", file=sys.stderr)
                if entry["stderr"]:
                    print(str(entry["stderr"]), file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
