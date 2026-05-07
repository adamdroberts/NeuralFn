from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def _neuralfn_repo_root() -> Path:
    current = Path(__file__).resolve()
    if current.parent.name == "scripts":
        return current.parents[2]
    return current.parents[1]


NEURALFN_REPO_ROOT = _neuralfn_repo_root()
CLI_SCRIPTS_DIR = Path(__file__).resolve().parent
if CLI_SCRIPTS_DIR.name != "scripts":
    CLI_SCRIPTS_DIR = CLI_SCRIPTS_DIR / "scripts"
for path in (NEURALFN_REPO_ROOT, CLI_SCRIPTS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


class HarnessArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_help(sys.stderr)
        self.exit(2, f"{self.prog}: error: {message}\n")


def create_argument_parser(*args, **kwargs) -> argparse.ArgumentParser:
    kwargs.setdefault("allow_abbrev", False)
    kwargs.setdefault("formatter_class", argparse.ArgumentDefaultsHelpFormatter)
    return HarnessArgumentParser(*args, **kwargs)


DEFAULT_ARTIFACTS_DIR = Path.home() / "NeuralFn" / "artifacts"


def artifact_root() -> Path:
    configured = os.getenv("NEURALFN_ARTIFACTS_DIR")
    if configured:
        return Path(configured).expanduser()
    return DEFAULT_ARTIFACTS_DIR


def artifact_path(name: str) -> Path:
    return artifact_root() / name
