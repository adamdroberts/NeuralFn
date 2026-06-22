from __future__ import annotations

import argparse
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from native_training_guard import reject_torch_training_by_default
from train_semantic_router_moe import ROUTER_DEFAULTS, _env_str, build_parser as _build_router_parser
from train_semantic_router_moe import default_output_path as _router_default_output_path


MODE_NAME = "semantic_router_moe_overnight"
GRAPH_NAME = "semantic_router_moe_sdk"


def mode_name(*, megakernel: bool = False) -> str:
    return "semantic_router_moe_megakernel" if megakernel else "semantic_router_moe"


def default_output_path(*, megakernel: bool = False) -> Path:
    if megakernel:
        return _router_default_output_path(megakernel=True).with_name("semantic_router_moe_overnight_megakernel.pt")
    return _router_default_output_path(megakernel=False).with_name("semantic_router_moe_overnight.pt")


def interrupted_output_path(*, megakernel: bool = False) -> Path:
    return default_output_path(megakernel=megakernel).with_name(
        f"{default_output_path(megakernel=megakernel).stem}.interrupted.pt"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = _build_router_parser()
    parser.description = "Train semantic_router_moe overnight through the NeuralFn native CUDA harness."
    parser.set_defaults(output=_env_str("OUTPUT", ""))
    return parser


def resolve_mode_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if not str(getattr(args, "output", "") or "").strip():
        args.output = str(default_output_path(megakernel=bool(getattr(args, "megakernel", False))))
    return args


def main(argv: list[str] | None = None) -> int:
    """Dispatch overnight semantic-router MoE training/preflight to native C++."""

    original_argv = sys.argv
    if argv is not None:
        sys.argv = [str(Path(__file__).resolve()), *argv]
    try:
        reject_torch_training_by_default(
            "train_semantic_router_moe-overnight.py",
            native_target="nfn train --base-model semantic-router-moe",
            model_family="semantic-router-moe",
            family_native_cli_env="NFN_NATIVE_SEMANTIC_ROUTER_MOE_CLI",
            family_native_cli_name="nfn_semantic_router_moe_native_train",
        )
    finally:
        if argv is not None:
            sys.argv = original_argv
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
