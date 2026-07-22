#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neuralfn.native_train import NATIVE_TEMPLATE_FAMILY_ALIASES  # noqa: E402


DIRECT_DENSE_GPT_SELECTORS = ("gpt", "gpt3")


def shipped_gpt_template_presets() -> tuple[str, ...]:
    """Read the literal preset declarations without importing optional SDK deps."""

    config_source = (ROOT / "neuralfn" / "config.py").read_text(encoding="utf-8")
    module = ast.parse(config_source, filename=str(ROOT / "neuralfn" / "config.py"))
    declarations: dict[str, tuple[str, ...]] = {}
    for node in module.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            value_node = node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target = node.target
            value_node = node.value
        else:
            continue
        if not isinstance(target, ast.Name) or target.id not in {
            "MODERN_BASE_PRESETS",
            "SHIPPED_GPT_TEMPLATE_BASE_PRESETS",
        }:
            continue
        value = ast.literal_eval(value_node)
        if not isinstance(value, tuple) or not all(isinstance(item, str) for item in value):
            raise RuntimeError(f"{target.id} must remain a literal tuple of strings")
        declarations[target.id] = tuple(value)
    try:
        base = declarations["SHIPPED_GPT_TEMPLATE_BASE_PRESETS"]
        modern = declarations["MODERN_BASE_PRESETS"]
    except KeyError as exc:
        raise RuntimeError(f"missing shipped preset declaration: {exc.args[0]}") from exc
    return tuple(dict.fromkeys((*base, *(f"{preset}_modern" for preset in modern))))


def _family_key(template_name: str) -> str:
    return str(template_name or "").strip().lower().replace("_", "-")


def normalize_dense_gpt_template_name(template_name: str) -> str:
    normalized = str(template_name or "").strip().lower().replace("-", "_")
    if normalized == "nano_gpt":
        return "nanogpt"
    return normalized


def sanitize_template_dir(template_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", template_name).strip("._") or "template"


def covered_dense_gpt_templates() -> list[str]:
    family_aliases = {_family_key(name) for name in NATIVE_TEMPLATE_FAMILY_ALIASES}
    templates: list[str] = []
    for template in (*DIRECT_DENSE_GPT_SELECTORS, *shipped_gpt_template_presets()):
        normalized = normalize_dense_gpt_template_name(template)
        if _family_key(normalized) in family_aliases:
            continue
        if normalized not in templates:
            templates.append(normalized)
    return templates


def select_templates(raw: str) -> list[str]:
    covered = covered_dense_gpt_templates()
    by_name = {name: name for name in covered}
    if not raw.strip():
        return covered
    selected: list[str] = []
    unknown: list[str] = []
    for item in raw.split(","):
        normalized = normalize_dense_gpt_template_name(item)
        template = by_name.get(normalized)
        if template is None:
            unknown.append(normalized)
        elif template not in selected:
            selected.append(template)
    if unknown:
        raise SystemExit(f"unknown covered dense GPT template(s): {', '.join(unknown)}")
    return selected


def _load_json(stdout: str) -> dict[str, Any]:
    return json.loads(stdout)


def run_template_smoke(
    *,
    binary: Path,
    output_dir: Path,
    template_name: str,
    train_seq_len: int,
    max_steps: int,
    dry_run: bool,
    include_output: bool,
) -> dict[str, Any]:
    template_dir = output_dir / sanitize_template_dir(template_name)
    smoke_argv = [
        str(binary),
        "--checkpoint-metadata-smoke",
        "--output-dir",
        str(template_dir),
        "--template-name",
        template_name,
        "--train-seq-len",
        str(int(train_seq_len)),
        "--max-steps",
        str(int(max_steps)),
    ]
    result: dict[str, Any] = {
        "template_name": template_name,
        "binary": str(binary),
        "output_dir": str(template_dir),
        "smoke_argv": smoke_argv,
        "smoke_returncode": None,
        "info_returncode": None,
        "passed": True,
    }
    if dry_run:
        result["checkpoint_path"] = str(template_dir / f"model_{int(max_steps):08d}.bin")
        result["info_argv"] = [
            str(binary),
            "--native-info",
            "--native-checkpoint",
            result["checkpoint_path"],
        ]
        return result

    smoke_proc = subprocess.run(
        smoke_argv,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    result["smoke_returncode"] = smoke_proc.returncode
    if include_output or smoke_proc.returncode != 0:
        result["smoke_stdout"] = smoke_proc.stdout
        result["smoke_stderr"] = smoke_proc.stderr
    try:
        smoke_payload = _load_json(smoke_proc.stdout)
    except Exception as exc:
        result["passed"] = False
        result["error"] = f"failed to parse checkpoint metadata JSON: {exc}"
        return result

    checkpoint_path = Path(str(smoke_payload.get("checkpoint_path", ""))).expanduser()
    result.update(
        {
            "checkpoint_path": str(checkpoint_path),
            "resolved_native_template_name": smoke_payload.get("resolved_native_template_name", ""),
            "metadata_only": bool(smoke_payload.get("metadata_only")),
            "size_matches": bool(smoke_payload.get("size_matches")),
            "num_layers": int(smoke_payload.get("num_layers", 0)),
            "num_heads": int(smoke_payload.get("num_heads", 0)),
            "model_dim": int(smoke_payload.get("model_dim", 0)),
            "max_seq_len": int(smoke_payload.get("max_seq_len", 0)),
            "parameter_count": int(smoke_payload.get("parameter_count", 0)),
        }
    )

    info_argv = [str(binary), "--native-info", "--native-checkpoint", str(checkpoint_path)]
    result["info_argv"] = info_argv
    info_proc = subprocess.run(
        info_argv,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    result["info_returncode"] = info_proc.returncode
    if include_output or info_proc.returncode != 0:
        result["info_stdout"] = info_proc.stdout
        result["info_stderr"] = info_proc.stderr
    try:
        info_payload = _load_json(info_proc.stdout)
    except Exception as exc:
        result["passed"] = False
        result["error"] = f"failed to parse native-info JSON: {exc}"
        return result

    checks = {
        "smoke_returncode_ok": smoke_proc.returncode == 0,
        "smoke_passed": bool(smoke_payload.get("passed")),
        "metadata_only": bool(smoke_payload.get("metadata_only")),
        "checkpoint_exists": checkpoint_path.exists(),
        "smoke_size_matches": bool(smoke_payload.get("size_matches")),
        "info_returncode_ok": info_proc.returncode == 0,
        "info_status": info_payload.get("status") == "native-checkpoint-info",
        "info_size_matches": bool(info_payload.get("size_matches")),
        "info_done_marker_exists": bool(info_payload.get("done_marker_exists")),
        "info_shape_matches_smoke": (
            int(info_payload.get("max_seq_len", 0)) == int(smoke_payload.get("max_seq_len", -1))
            and int(info_payload.get("num_layers", 0)) == int(smoke_payload.get("num_layers", -1))
            and int(info_payload.get("num_heads", 0)) == int(smoke_payload.get("num_heads", -1))
            and int(info_payload.get("channels", 0)) == int(smoke_payload.get("model_dim", -1))
            and int(info_payload.get("parameter_count", 0)) == int(smoke_payload.get("parameter_count", -1))
        ),
    }
    result["checks"] = checks
    result["passed"] = all(checks.values())
    if not result["passed"]:
        failed = [name for name, ok in checks.items() if not ok]
        result["error"] = "failed checks: " + ", ".join(failed)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run no-CUDA native dense GPT checkpoint metadata smokes for every "
            "covered dense template and inspect each checkpoint with native-info."
        )
    )
    parser.add_argument("--native-bin", default=str(ROOT / "build" / "nfn_gpt_native_train"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--templates",
        default="",
        help="Comma-separated dense GPT template subset. Defaults to every covered dense GPT selector.",
    )
    parser.add_argument("--train-seq-len", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-smoke-output", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    binary = Path(args.native_bin).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    templates = select_templates(args.templates)
    output_dir.mkdir(parents=True, exist_ok=True)

    missing_binary = not binary.exists()
    smoke_results: list[dict[str, Any]] = []
    if not missing_binary:
        for template_name in templates:
            result = run_template_smoke(
                binary=binary,
                output_dir=output_dir,
                template_name=template_name,
                train_seq_len=max(1, int(args.train_seq_len)),
                max_steps=max(0, int(args.max_steps)),
                dry_run=bool(args.dry_run),
                include_output=bool(args.include_smoke_output),
            )
            smoke_results.append(result)
            if not bool(result["passed"]) and not args.keep_going:
                break

    passed = (
        not missing_binary
        and len(smoke_results) == len(templates)
        and all(bool(result["passed"]) for result in smoke_results)
    )
    payload = {
        "status": "native-dense-gpt-template-checkpoint-smoke-sweep",
        "native_bin": str(binary),
        "output_dir": str(output_dir),
        "template_count": len(templates),
        "smoke_count": len(smoke_results),
        "passed_count": sum(1 for result in smoke_results if bool(result["passed"])),
        "failed_count": sum(1 for result in smoke_results if not bool(result["passed"])),
        "missing_binary": missing_binary,
        "passed": passed,
        "smokes": smoke_results,
    }
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(
            "Native dense GPT template checkpoint smoke sweep: "
            f"{payload['smoke_count']}/{payload['template_count']} smokes, "
            f"missing_binary={str(missing_binary).lower()}, passed={str(passed).lower()}"
        )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
