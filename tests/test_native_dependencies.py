from __future__ import annotations

import tomllib
from pathlib import Path
import subprocess
import sys


def test_torch_is_optional_not_a_core_dependency() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    dependencies = pyproject["project"]["dependencies"]
    optional = pyproject["project"]["optional-dependencies"]

    assert not any(str(item).startswith("torch") for item in dependencies)
    assert any(str(item).startswith("torch") for item in optional["torch"])
    assert any(str(item).startswith("ninja") for item in optional["tile-cuda"])
    assert not any(str(item).startswith("torch") for item in optional["tile-cuda"])

    egg_info_requires = Path("neuralfn.egg-info/requires.txt")
    if egg_info_requires.exists():
        current_group = "core"
        requirements_by_group: dict[str, list[str]] = {"core": []}
        for raw_line in egg_info_requires.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                current_group = line.strip("[]")
                requirements_by_group.setdefault(current_group, [])
                continue
            requirements_by_group.setdefault(current_group, []).append(line)
        assert not any(item.startswith("torch") for item in requirements_by_group["core"])
        assert not any(item.startswith("torch") for item in requirements_by_group.get("tile-cuda", []))
        assert any(item.startswith("torch") for item in requirements_by_group.get("torch", []))


def test_native_train_sdk_import_does_not_import_torch() -> None:
    code = """
import sys
from neuralfn.native_train import build_native_train_run_config, native_train_runner_status
cfg = build_native_train_run_config("gpt2", ["--dry-run"])
status = native_train_runner_status("subprocess")
print("ARGV", " ".join(cfg.argv()[:3]))
print("STATUS", status.resolved)
print("TORCH_LOADED", "torch" in sys.modules)
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "ARGV" in proc.stdout
    assert "STATUS subprocess" in proc.stdout
    assert "TORCH_LOADED False" in proc.stdout
