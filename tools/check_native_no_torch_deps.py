#!/usr/bin/env python3
"""Fail if native training paths depend on Python/Torch runtime libraries."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import struct
import sys
import tempfile
import time
from textwrap import dedent
import tomllib


REQUIRED_DEFAULT_ARTIFACTS = (
    Path("build/nfn_gpt_native_train"),
    Path("build/libnfn_native_train_tile_ops.so"),
)
OPTIONAL_DEFAULT_ARTIFACTS = (
    Path("build/nfn_native_train"),
    Path("build/nfn_gpt2_native_train"),
    Path("build/nfn_gpt2_evo_native_train"),
    Path("build/nfn_nanogpt_native_train"),
    Path("build/nfn_llama_native_train"),
    Path("build/nfn_mixllama_native_train"),
    Path("build/nfn_jepa_native_train"),
    Path("build/nfn_semantic_router_moe_native_train"),
    Path("build/nfn_deepseek_v4_native_train"),
    Path("build/libnfn_native_train_tile_ops_tk.so"),
)
OPTIONAL_DEFAULT_ARTIFACT_GLOBS = (
    "neuralfn/_native*.so",
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
    "train_gpt_native",
    "infer_gpt",
    "nfn_impl",
)
FORBIDDEN_PROJECT_DEPENDENCY_PREFIXES = (
    "alembic",
    "datasets",
    "fastapi",
    "mcp",
    "networkx",
    "numpy",
    "pydantic",
    "pymysql",
    "python-multipart",
    "redis",
    "sqlalchemy",
    "tiktoken",
    "torch",
    "torchvision",
    "torchaudio",
    "uvicorn",
)
REQUIRED_OPTIONAL_DEPENDENCY_PREFIXES = (
    "datasets",
    "fastapi",
    "networkx",
    "numpy",
    "tiktoken",
)
FORBIDDEN_OPTIONAL_EXTRA_NAMES = ("torch",)
FORBIDDEN_OPTIONAL_EXTRA_DEPENDENCY_PREFIXES = {
    "all": (
        "torch",
        "torchvision",
        "torchaudio",
    ),
    "torch": (
        "torch",
        "torchvision",
        "torchaudio",
    ),
}
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
        "nfn_train_default_fast_command",
        (
            sys.executable,
            "cli/nfn.py",
            "train",
            "--native-cuda-dry-run",
            "--native-cuda-print-command",
            "--no-checkpoint",
        ),
    ),
    (
        "nfn_console_train_fast_command",
        (
            sys.executable,
            "-c",
            "\n".join(
                [
                    "import sys",
                    "sys.argv = [",
                    "    'nfn',",
                    "    'train',",
                    "    '--tinystories',",
                    "    '--native-cuda-dry-run',",
                    "    '--native-cuda-print-command',",
                    "    '--no-checkpoint',",
                    "]",
                    "from nfn import main",
                    "raise SystemExit(int(main() or 0))",
                ]
            ),
        ),
    ),
    (
        "native_sdk_imports",
        (
            sys.executable,
            "-c",
            "import neuralfn; import neuralfn.native_gpt; import neuralfn.native_gpt2; import neuralfn.native_train; print('native-sdk-ok')",
        ),
    ),
    (
        "native_sdk_public_exports",
        (
            sys.executable,
            "-c",
            "\n".join(
                [
                    "import neuralfn",
                    "from neuralfn import NativeGptRunConfig, NativeGpt2RunConfig, NativeTrainRunConfig",
                    "from neuralfn import build_native_gpt_compiled_cli_run_config",
                    "from neuralfn import build_native_train_run_config, run_native_train",
                    "from neuralfn import native_gpt_kernel_backend, native_gpt_parameter_count",
                    "from neuralfn import native_train_model_registry, native_train_runner_status",
                    "from neuralfn import resolve_native_gpt_binding_command, resolve_native_gpt2_binding_command",
                    "from neuralfn import resolve_native_train_binding_command",
                    "assert NativeGptRunConfig.__name__ == 'NativeGptRunConfig'",
                    "assert NativeGpt2RunConfig.__name__ == 'NativeGpt2RunConfig'",
                    "assert NativeTrainRunConfig.__name__ == 'NativeTrainRunConfig'",
                    "assert native_gpt_kernel_backend('tile-cuda') == 'tile-cuda'",
                    "assert native_gpt_parameter_count(max_seq_len=1024, padded_vocab_size=50304, num_layers=12, channels=768) > 0",
                    "assert callable(build_native_gpt_compiled_cli_run_config)",
                    "assert callable(build_native_train_run_config)",
                    "assert callable(run_native_train)",
                    "assert callable(resolve_native_gpt_binding_command)",
                    "assert callable(resolve_native_gpt2_binding_command)",
                    "assert callable(resolve_native_train_binding_command)",
                    "registry = native_train_model_registry()",
                    "assert isinstance(registry, dict) and registry",
                    "assert native_train_runner_status('compiled-cli').available in (True, False)",
                    "print('native-sdk-public-exports-ok')",
                ]
            ),
        ),
    ),
    (
        "native_sdk_binding_imports",
        (
            sys.executable,
            "-c",
            "\n".join(
                [
                    "import importlib",
                    "import importlib.util",
                    "loaded = []",
                    "for name in ('neuralfn._native_gpt', 'neuralfn._native_gpt2', 'neuralfn._native_train'):",
                    "    if importlib.util.find_spec(name) is not None:",
                    "        importlib.import_module(name)",
                    "        loaded.append(name)",
                    "print('native-sdk-binding-imports-ok:' + ','.join(loaded))",
                ]
            ),
        ),
    ),
)
DEFAULT_SCRIPT_NATIVE_DISPATCH_ENTRYPOINTS = (
    (
        "train_gpt2_evo_default_native_dispatch",
        (
            sys.executable,
            "cli/scripts/train_gpt2_evo.py",
        ),
    ),
    (
        "train_nanogpt_default_native_dispatch",
        (
            sys.executable,
            "cli/scripts/train_nanogpt.py",
        ),
    ),
    (
        "train_llama_fast_default_native_dispatch",
        (
            sys.executable,
            "cli/scripts/train_llama_fast.py",
        ),
    ),
    (
        "train_llama_megakernel_default_native_dispatch",
        (
            sys.executable,
            "cli/scripts/train_llama_megakernel.py",
        ),
    ),
    (
        "train_mixllama_fast_default_native_dispatch",
        (
            sys.executable,
            "cli/scripts/train_mixllama_fast.py",
        ),
    ),
    (
        "train_jepa_semantic_default_native_dispatch",
        (
            sys.executable,
            "cli/scripts/train_jepa_semantic.py",
        ),
    ),
    (
        "train_semantic_router_moe_default_native_dispatch",
        (
            sys.executable,
            "cli/scripts/train_semantic_router_moe.py",
        ),
    ),
    (
        "train_semantic_router_moe_overnight_default_native_dispatch",
        (
            sys.executable,
            "cli/scripts/train_semantic_router_moe-overnight.py",
        ),
    ),
    (
        "train_deepseek_v4_default_native_dispatch",
        (
            sys.executable,
            "cli/scripts/train_deepseek_v4.py",
        ),
    ),
)
NATIVE_GPT_CHECKPOINT_MAGIC = 20240326
NATIVE_GPT_CHECKPOINT_HEADER_INTS = 256
DEFAULT_MAX_ENTRYPOINT_SECONDS = 2.0


def _native_gpt_parameter_count(
    *,
    max_seq_len: int,
    padded_vocab_size: int,
    num_layers: int,
    channels: int,
) -> int:
    c = int(channels)
    l = int(num_layers)
    return int(
        int(padded_vocab_size) * c
        + int(max_seq_len) * c
        + l * c
        + l * c
        + l * 3 * c * c
        + l * 3 * c
        + l * c * c
        + l * c
        + l * c
        + l * c
        + l * 4 * c * c
        + l * 4 * c
        + l * c * 4 * c
        + l * c
        + c
        + c
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "artifacts",
        nargs="*",
        type=Path,
        default=(),
        help=(
            "Native executable/shared-library artifacts to inspect with ldd. "
            "Explicit paths are required to exist; the default scan includes "
            "required native artifacts plus optional compiled trainer artifacts "
            "that are present in build/."
        ),
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
    parser.add_argument(
        "--max-entrypoint-seconds",
        type=float,
        default=DEFAULT_MAX_ENTRYPOINT_SECONDS,
        help=(
            "Maximum wall-clock seconds allowed for each native Python fast-path entrypoint. "
            "Use 0 to disable this startup budget."
        ),
    )
    return parser.parse_args()


def default_artifacts() -> list[Path]:
    artifacts = list(REQUIRED_DEFAULT_ARTIFACTS)
    artifacts.extend(path for path in OPTIONAL_DEFAULT_ARTIFACTS if path.exists())
    for pattern in OPTIONAL_DEFAULT_ARTIFACT_GLOBS:
        artifacts.extend(sorted(Path().glob(pattern)))
    return artifacts


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


def _write_native_checkpoint_stub(root: Path) -> Path:
    checkpoint = root / "model_00000010.bin"
    max_seq_len = 8
    vocab_size = 16
    num_layers = 1
    num_heads = 1
    channels = 4
    padded_vocab_size = 16
    header = [0] * NATIVE_GPT_CHECKPOINT_HEADER_INTS
    header[:8] = [
        NATIVE_GPT_CHECKPOINT_MAGIC,
        5,
        max_seq_len,
        vocab_size,
        num_layers,
        num_heads,
        channels,
        padded_vocab_size,
    ]
    parameter_count = _native_gpt_parameter_count(
        max_seq_len=max_seq_len,
        padded_vocab_size=padded_vocab_size,
        num_layers=num_layers,
        channels=channels,
    )
    checkpoint.write_bytes(
        struct.pack("<" + "i" * NATIVE_GPT_CHECKPOINT_HEADER_INTS, *header) +
        b"\0" * (parameter_count * 2)
    )
    (root / "DONE_00000010").write_text("", encoding="utf-8")
    return checkpoint


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


def python_entrypoint_report(repo_root: Path, *, max_entrypoint_seconds: float) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    startup_budget_seconds = max(0.0, float(max_entrypoint_seconds))
    with tempfile.TemporaryDirectory(prefix="nfn-native-no-torch-") as tmp:
        temp_root = Path(tmp)
        _write_import_blocker(temp_root)
        native_cli = _write_native_cli_stub(temp_root)
        native_checkpoint = _write_native_checkpoint_stub(temp_root)
        env = os.environ.copy()
        env["NFN_NATIVE_GPT_CLI"] = str(native_cli)
        env["NFN_NATIVE_GPT2_CLI"] = str(native_cli)
        env["NFN_NATIVE_GPT2_EVO_CLI"] = str(native_cli)
        env["NFN_NATIVE_NANOGPT_CLI"] = str(native_cli)
        env["NFN_NATIVE_LLAMA_CLI"] = str(native_cli)
        env["NFN_NATIVE_MIXLLAMA_CLI"] = str(native_cli)
        env["NFN_NATIVE_JEPA_CLI"] = str(native_cli)
        env["NFN_NATIVE_DEEPSEEK_V4_CLI"] = str(native_cli)
        env["NFN_NATIVE_SEMANTIC_ROUTER_MOE_CLI"] = str(native_cli)
        cli_root = repo_root / "cli"
        env["PYTHONPATH"] = os.pathsep.join(
            part
            for part in (
                str(temp_root),
                str(cli_root),
                str(repo_root),
                env.get("PYTHONPATH", ""),
            )
            if part
        )
        env.setdefault("CUDA_VISIBLE_DEVICES", "0")
        env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
        entrypoints = [
            *DEFAULT_PYTHON_ENTRYPOINTS,
            *DEFAULT_SCRIPT_NATIVE_DISPATCH_ENTRYPOINTS,
            (
                "train_gpt2_evo_fast_command",
                (
                    sys.executable,
                    "cli/scripts/train_gpt2_evo.py",
                    "--tinystories",
                    "--native-cuda-dry-run",
                    "--native-cuda-print-command",
                    "--native-cuda-no-checkpoint",
                ),
            ),
            (
                "train_nanogpt_fast_command",
                (
                    sys.executable,
                    "cli/scripts/train_nanogpt.py",
                    "--tinystories",
                    "--native-cuda-dry-run",
                    "--native-cuda-print-command",
                    "--native-cuda-no-checkpoint",
                ),
            ),
            (
                "train_llama_fast_command",
                (
                    sys.executable,
                    "cli/scripts/train_llama_fast.py",
                    "--tinystories",
                    "--native-cuda-dry-run",
                    "--native-cuda-print-command",
                    "--native-cuda-no-checkpoint",
                ),
            ),
            (
                "train_llama_megakernel_fast_command",
                (
                    sys.executable,
                    "cli/scripts/train_llama_megakernel.py",
                    "--tinystories",
                    "--native-cuda-dry-run",
                    "--native-cuda-print-command",
                    "--native-cuda-no-checkpoint",
                ),
            ),
            (
                "train_mixllama_fast_command",
                (
                    sys.executable,
                    "cli/scripts/train_mixllama_fast.py",
                    "--tinystories",
                    "--native-cuda-dry-run",
                    "--native-cuda-print-command",
                    "--native-cuda-no-checkpoint",
                ),
            ),
            (
                "train_jepa_semantic_fast_command",
                (
                    sys.executable,
                    "cli/scripts/train_jepa_semantic.py",
                    "--tinystories",
                    "--native-cuda-dry-run",
                    "--native-cuda-print-command",
                    "--native-cuda-no-checkpoint",
                ),
            ),
            (
                "train_semantic_router_moe_fast_command",
                (
                    sys.executable,
                    "cli/scripts/train_semantic_router_moe.py",
                    "--tinystories",
                    "--native-cuda-dry-run",
                    "--native-cuda-print-command",
                    "--native-cuda-no-checkpoint",
                ),
            ),
            (
                "train_semantic_router_moe_overnight_fast_command",
                (
                    sys.executable,
                    "cli/scripts/train_semantic_router_moe-overnight.py",
                    "--tinystories",
                    "--native-cuda-dry-run",
                    "--native-cuda-print-command",
                    "--native-cuda-no-checkpoint",
                ),
            ),
            (
                "train_deepseek_v4_fast_command",
                (
                    sys.executable,
                    "cli/scripts/train_deepseek_v4.py",
                    "--tinystories",
                    "--native-cuda-dry-run",
                    "--native-cuda-print-command",
                    "--native-cuda-no-checkpoint",
                ),
            ),
            (
                "infer_gpt_native_info",
                (
                    sys.executable,
                    "cli/scripts/infer_gpt.py",
                    "--native-checkpoint",
                    str(native_checkpoint),
                    "--native-info",
                ),
            ),
            (
                "infer_gpt_native_sample_prompt_tokens",
                (
                    sys.executable,
                    "cli/scripts/infer_gpt.py",
                    "--native-checkpoint",
                    str(native_checkpoint),
                    "--prompt-tokens",
                    "1,2,3",
                    "--max-new-tokens",
                    "2",
                ),
            ),
            (
                "nfn_infer_native_info",
                (
                    sys.executable,
                    "cli/nfn.py",
                    "infer",
                    "--native-checkpoint",
                    str(native_checkpoint),
                    "--native-info",
                ),
            ),
            (
                "nfn_infer_native_sample_prompt_tokens",
                (
                    sys.executable,
                    "cli/nfn.py",
                    "infer",
                    "--native-checkpoint",
                    str(native_checkpoint),
                    "--prompt-tokens",
                    "1,2,3",
                    "--max-new-tokens",
                    "2",
                ),
            ),
            (
                "nfn_console_infer_native_info",
                (
                    sys.executable,
                    "-c",
                    "\n".join(
                        [
                            "import sys",
                            "sys.argv = [",
                            "    'nfn',",
                            "    'infer',",
                            "    '--native-checkpoint',",
                            f"    {str(native_checkpoint)!r},",
                            "    '--native-info',",
                            "]",
                            "from nfn import main",
                            "raise SystemExit(int(main() or 0))",
                        ]
                    ),
                ),
            ),
            (
                "nfn_console_infer_native_sample_prompt_tokens",
                (
                    sys.executable,
                    "-c",
                    "\n".join(
                        [
                            "import sys",
                            "sys.argv = [",
                            "    'nfn',",
                            "    'infer',",
                            "    '--native-checkpoint',",
                            f"    {str(native_checkpoint)!r},",
                            "    '--prompt-tokens',",
                            "    '1,2,3',",
                            "    '--max-new-tokens',",
                            "    '2',",
                            "]",
                            "from nfn import main",
                            "raise SystemExit(int(main() or 0))",
                        ]
                    ),
                ),
            ),
        ]
        for name, command in entrypoints:
            started = time.perf_counter()
            proc = subprocess.run(
                list(command),
                cwd=repo_root,
                env=env,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            elapsed_seconds = time.perf_counter() - started
            startup_within_budget = (
                True if startup_budget_seconds <= 0.0 else elapsed_seconds <= startup_budget_seconds
            )
            entries.append(
                {
                    "name": name,
                    "command": list(command),
                    "returncode": proc.returncode,
                    "passed": proc.returncode == 0 and startup_within_budget,
                    "elapsed_seconds": elapsed_seconds,
                    "startup_budget_seconds": startup_budget_seconds,
                    "startup_within_budget": startup_within_budget,
                    "stdout": proc.stdout.strip(),
                    "stderr": proc.stderr.strip(),
                }
            )
    return entries


def project_dependency_report(repo_root: Path) -> dict[str, object]:
    pyproject = repo_root / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    dependencies = list(data.get("project", {}).get("dependencies", []))
    optional_dependencies = dict(data.get("project", {}).get("optional-dependencies", {}))
    forbidden_extra_names = sorted(
        extra_name
        for extra_name in optional_dependencies
        if str(extra_name).strip().lower().replace("_", "-") in FORBIDDEN_OPTIONAL_EXTRA_NAMES
    )
    offenders: list[str] = []
    for dependency in dependencies:
        normalized = str(dependency).strip().lower().replace("_", "-")
        for prefix in FORBIDDEN_PROJECT_DEPENDENCY_PREFIXES:
            if (
                normalized == prefix
                or normalized.startswith(prefix + ">")
                or normalized.startswith(prefix + "=")
                or normalized.startswith(prefix + "[")
            ):
                offenders.append(str(dependency))
                break
    optional_hits: dict[str, list[str]] = {prefix: [] for prefix in REQUIRED_OPTIONAL_DEPENDENCY_PREFIXES}
    forbidden_optional_hits: dict[str, list[str]] = {
        extra_name: [] for extra_name in FORBIDDEN_OPTIONAL_EXTRA_DEPENDENCY_PREFIXES
    }
    for extra_name, extra_dependencies in optional_dependencies.items():
        for dependency in extra_dependencies:
            normalized = str(dependency).strip().lower().replace("_", "-")
            for prefix in REQUIRED_OPTIONAL_DEPENDENCY_PREFIXES:
                if (
                    normalized == prefix
                    or normalized.startswith(prefix + ">")
                    or normalized.startswith(prefix + "=")
                    or normalized.startswith(prefix + "[")
                ):
                    optional_hits[prefix].append(str(extra_name))
            for prefix in FORBIDDEN_OPTIONAL_EXTRA_DEPENDENCY_PREFIXES.get(str(extra_name), ()):
                if (
                    normalized == prefix
                    or normalized.startswith(prefix + ">")
                    or normalized.startswith(prefix + "=")
                    or normalized.startswith(prefix + "[")
                ):
                    forbidden_optional_hits.setdefault(str(extra_name), []).append(str(dependency))
    missing_optional = [
        prefix for prefix, extra_names in optional_hits.items() if not extra_names
    ]
    forbidden_optional_hits = {
        extra_name: hits for extra_name, hits in forbidden_optional_hits.items() if hits
    }
    return {
        "name": "pyproject_default_dependencies",
        "path": str(pyproject),
        "passed": not offenders and not missing_optional and not forbidden_optional_hits and not forbidden_extra_names,
        "offenders": offenders,
        "forbidden_default_dependency_prefixes": list(FORBIDDEN_PROJECT_DEPENDENCY_PREFIXES),
        "forbidden_optional_extra_names": list(FORBIDDEN_OPTIONAL_EXTRA_NAMES),
        "forbidden_optional_extra_name_hits": forbidden_extra_names,
        "forbidden_optional_extra_dependency_prefixes": {
            extra_name: list(prefixes)
            for extra_name, prefixes in FORBIDDEN_OPTIONAL_EXTRA_DEPENDENCY_PREFIXES.items()
        },
        "forbidden_optional_extra_hits": forbidden_optional_hits,
        "optional_dependency_hits": optional_hits,
        "missing_optional_dependency_prefixes": missing_optional,
    }


def main() -> int:
    args = parse_args()
    artifact_report: list[dict[str, object]] = []
    failed = False
    if not args.skip_artifacts:
        artifacts = list(args.artifacts) if args.artifacts else default_artifacts()
        for artifact in artifacts:
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
    dependency_report: dict[str, object] | None = None
    if not args.skip_python_entrypoints:
        repo_root = Path(__file__).resolve().parents[1]
        dependency_report = project_dependency_report(repo_root)
        failed = failed or not bool(dependency_report["passed"])
        python_report = python_entrypoint_report(
            repo_root,
            max_entrypoint_seconds=float(args.max_entrypoint_seconds),
        )
        failed = failed or any(not bool(entry["passed"]) for entry in python_report)

    if args.json:
        print(
            json.dumps(
                {
                    "passed": not failed,
                    "forbidden_python_import_roots": list(FORBIDDEN_PYTHON_IMPORT_ROOTS),
                    "artifacts": artifact_report,
                    "project_dependencies": dependency_report,
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
        if dependency_report is not None:
            if dependency_report["passed"]:
                print(f"{dependency_report['name']}: ok")
            else:
                print(f"{dependency_report['name']}: failed", file=sys.stderr)
                for dependency in dependency_report["offenders"]:
                    print(f"  hard dependency: {dependency}", file=sys.stderr)
                for extra_name in dependency_report["forbidden_optional_extra_name_hits"]:
                    print(f"  forbidden optional extra: {extra_name}", file=sys.stderr)
                for prefix in dependency_report["missing_optional_dependency_prefixes"]:
                    print(f"  missing optional dependency coverage: {prefix}", file=sys.stderr)
        for entry in python_report:
            if entry["passed"]:
                print(f"{entry['name']}: ok ({float(entry['elapsed_seconds']):.3f}s)")
            else:
                print(f"{entry['name']}: failed", file=sys.stderr)
                if not entry.get("startup_within_budget", True):
                    print(
                        "startup budget exceeded: "
                        f"{float(entry['elapsed_seconds']):.3f}s > "
                        f"{float(entry['startup_budget_seconds']):.3f}s",
                        file=sys.stderr,
                    )
                if entry["stderr"]:
                    print(str(entry["stderr"]), file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
