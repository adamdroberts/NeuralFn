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
    assert not any(str(item).startswith("torch") for item in optional["all"])

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
        assert not any(item.startswith("torch") for item in requirements_by_group.get("all", []))
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


def test_native_gpt_sdk_alias_import_does_not_import_torch() -> None:
    code = """
import sys
from neuralfn.native_gpt import build_native_gpt_compiled_cli_run_config
cfg = build_native_gpt_compiled_cli_run_config(
    dataset_alias="/tmp/native-cache",
    executable="/bin/echo",
    output_dir="/tmp/native-output",
    eval_every_steps=1000,
    sample_every_steps=20000,
    generate_tokens=144,
    checkpoint_every_steps=200,
    batch_size=64,
    seq_len=1024,
    train_batch_tokens=524288,
    learning_rate=0.0006,
    min_lr=None,
    warmup_steps=60,
    weight_decay=0.1,
    max_steps=20000,
    num_layers=12,
    activation="gelu",
)
print("ARGV", " ".join(cfg.compiled_cli_argv("/tmp/nfn_gpt2_native_train")[:3]))
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
    assert "TORCH_LOADED False" in proc.stdout
