#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neuralfn.native_family import (  # noqa: E402
    audit_native_family_checkpoint_template_coverage,
    normalize_native_family_template_name,
    parse_native_family_template_list,
)
from neuralfn.native_train import (  # noqa: E402
    NATIVE_TEMPLATE_FAMILY_ALIASES,
    NATIVE_TRAIN_FAMILY_TARGETS,
)


NON_FAMILY_CHECKPOINT_TARGETS = {"gpt", "gpt2", "gpt3", "nanogpt", "gpt2-evo", "nano-gpt"}


def covered_native_family_templates() -> dict[str, str]:
    targets = {
        normalize_native_family_template_name(name): normalize_native_family_template_name(name)
        for name in NATIVE_TRAIN_FAMILY_TARGETS
        if normalize_native_family_template_name(name) not in NON_FAMILY_CHECKPOINT_TARGETS
    }
    aliases = {
        normalize_native_family_template_name(template): normalize_native_family_template_name(family)
        for template, family in NATIVE_TEMPLATE_FAMILY_ALIASES.items()
    }
    targets.update(aliases)
    return dict(sorted(targets.items()))


def select_templates(raw: str) -> dict[str, str]:
    covered = covered_native_family_templates()
    if not raw.strip():
        return covered
    selected: dict[str, str] = {}
    unknown: list[str] = []
    for template in parse_native_family_template_list(raw):
        family = covered.get(template)
        if family is None:
            unknown.append(template)
        else:
            selected[template] = family
    if unknown:
        raise SystemExit(f"unknown covered native-family template(s): {', '.join(unknown)}")
    return selected


def run_smoke(
    *,
    binary: Path,
    output_dir: Path,
    template_name: str,
    dry_run: bool,
    include_output: bool,
) -> dict[str, Any]:
    argv = [
        str(binary),
        "--smoke-family-layout-checkpoint-step",
        "--output-dir",
        str(output_dir),
        "--template-name",
        template_name,
    ]
    if dry_run:
        return {
            "template_name": template_name,
            "binary": str(binary),
            "argv": argv,
            "returncode": None,
            "passed": True,
        }
    proc = subprocess.run(
        argv,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    result = {
        "template_name": template_name,
        "binary": str(binary),
        "argv": argv,
        "returncode": proc.returncode,
        "passed": proc.returncode == 0,
    }
    if include_output or proc.returncode != 0:
        result["stdout"] = proc.stdout
        result["stderr"] = proc.stderr
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run no-CUDA native-family checkpoint smokes for covered template "
            "aliases and verify that each template produced a loadable artifact."
        )
    )
    parser.add_argument("--native-bin-dir", default=str(ROOT / "build"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--templates",
        default="",
        help="Comma-separated covered template subset. Defaults to every native-family checkpoint template.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-smoke-output", action="store_true")
    parser.add_argument(
        "--require-architecture-forward",
        action="store_true",
        help="Require real architecture-forward inference from persistent parameter state.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    native_bin_dir = Path(args.native_bin_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    templates = select_templates(args.templates)
    output_dir.mkdir(parents=True, exist_ok=True)

    smoke_results: list[dict[str, Any]] = []
    missing_binaries: list[dict[str, str]] = []
    for template_name, family in templates.items():
        target = NATIVE_TRAIN_FAMILY_TARGETS.get(family)
        binary = native_bin_dir / str(target or "")
        if not target or not binary.exists():
            missing_binaries.append(
                {
                    "template_name": template_name,
                    "native_family": family,
                    "expected_binary": str(binary),
                }
            )
            if not args.keep_going:
                break
            continue
        result = run_smoke(
            binary=binary,
            output_dir=output_dir,
            template_name=template_name,
            dry_run=bool(args.dry_run),
            include_output=bool(args.include_smoke_output),
        )
        smoke_results.append(result)
        if not bool(result["passed"]) and not args.keep_going:
            break

    verification: dict[str, Any] = {}
    if not args.dry_run:
        verification = audit_native_family_checkpoint_template_coverage(
            output_dir,
            required_templates=templates,
            max_new_tokens=max(1, int(args.max_new_tokens)),
            require_architecture_forward=bool(args.require_architecture_forward),
        )
    passed = (
        not missing_binaries
        and all(bool(result["passed"]) for result in smoke_results)
        and (bool(args.dry_run) or bool(verification.get("passed")))
        and len(smoke_results) == len(templates)
    )
    payload = {
        "status": "native-family-template-checkpoint-smoke-sweep",
        "native_bin_dir": str(native_bin_dir),
        "output_dir": str(output_dir),
        "template_count": len(templates),
        "smoke_count": len(smoke_results),
        "missing_binary_count": len(missing_binaries),
        "passed": passed,
        "missing_binaries": missing_binaries,
        "smokes": smoke_results,
        "verification": verification,
    }
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(
            "Native-family template checkpoint smoke sweep: "
            f"{payload['smoke_count']}/{payload['template_count']} smokes, "
            f"missing_binaries={payload['missing_binary_count']}, "
            f"passed={str(passed).lower()}"
        )
        if verification:
            print(
                "Coverage: "
                f"passed_templates={verification.get('passed_template_count', 0)}/"
                f"{verification.get('required_template_count', 0)}, "
                f"missing_templates={verification.get('missing_template_count', 0)}"
            )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
