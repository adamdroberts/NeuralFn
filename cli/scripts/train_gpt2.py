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
    from train_gpt import _apply_cached_vocab_from_alias, _fast_compiled_cli_argv, _fast_compiled_cli_main  # noqa: F401


def main(argv: list[str] | None = None) -> int:
    explicit = list(sys.argv[1:] if argv is None else argv)
    configure_console_logging()
    parser = build_parser()
    args = parser.parse_args(explicit)
    apply_tinystories_dataset_defaults(args)
    dataset_name = resolve_dataset_selector_args(args)
    resolve_mode_defaults(args)
    dataset_name, _dataset_path, _dataset_meta = resolve_or_download_dataset(dataset_name)
    _apply_cached_vocab_from_alias(args, dataset_name)
    graph, spec = build_graph(args, dataset_name)
    trainer_cfg = build_trainer_config(args, resolved_epochs=1)
    print_resolved_summary(args, spec, trainer_cfg, {"steps_per_epoch": 1})
    return 0
