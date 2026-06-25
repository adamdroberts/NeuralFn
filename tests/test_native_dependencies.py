from __future__ import annotations

import tomllib
from pathlib import Path
import importlib.util
import subprocess
import sys


def test_torch_is_not_an_install_dependency() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    dependencies = pyproject["project"]["dependencies"]
    optional = pyproject["project"]["optional-dependencies"]

    assert not any(str(item).startswith("torch") for item in dependencies)
    assert "torch" not in optional
    assert any(str(item).startswith("ninja") for item in optional["tile-cuda"])
    assert not any(str(item).startswith("torch") for item in optional["tile-cuda"])
    assert not any(str(item).startswith("torch") for item in optional["all"])
    for extra_name, requirements in optional.items():
        assert not any(str(item).startswith(("torch", "torchvision", "torchaudio")) for item in requirements), extra_name

    requires_path = Path("neuralfn.egg-info/requires.txt")
    pkg_info_path = Path("neuralfn.egg-info/PKG-INFO")
    if requires_path.exists() and pkg_info_path.exists():
        requires_txt = requires_path.read_text(encoding="utf-8")
        pkg_info = pkg_info_path.read_text(encoding="utf-8")
        assert "[torch]" not in requires_txt
        assert "torch>=2.0" not in requires_txt
        assert 'Provides-Extra: torch' not in pkg_info
        assert 'Requires-Dist: torch' not in pkg_info


def test_native_train_sdk_import_does_not_import_torch() -> None:
    code = """
import sys
from neuralfn.native_train import build_native_train_run_config, exec_native_train, native_train_runner_status
cfg = build_native_train_run_config(
    "gpt2",
    ["--dry-run"],
    require_cooperative_lm_head_backward=True,
)
status = native_train_runner_status("subprocess")
print("ARGV", " ".join(cfg.argv()[:3]))
print("STRICT_FLAG", "--require-cooperative-lm-head-backward" in cfg.argv())
print("STATUS", status.resolved)
print("EXEC_CALLABLE", callable(exec_native_train))
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
    assert "STRICT_FLAG True" in proc.stdout
    assert "STATUS subprocess" in proc.stdout
    assert "EXEC_CALLABLE True" in proc.stdout
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


def test_no_torch_verifier_rejects_stale_egg_info(tmp_path: Path) -> None:
    module_path = Path("tools/check_native_no_torch_deps.py")
    spec = importlib.util.spec_from_file_location("check_native_no_torch_deps", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    egg_info = tmp_path / "neuralfn.egg-info"
    egg_info.mkdir()
    (egg_info / "requires.txt").write_text("[torch]\ntorch>=2.0\n", encoding="utf-8")
    (egg_info / "PKG-INFO").write_text(
        'Provides-Extra: torch\nRequires-Dist: torch>=2.0; extra == "torch"\n',
        encoding="utf-8",
    )

    report = module.egg_info_dependency_report(tmp_path)
    assert report["passed"] is False
    assert report["offenders"]
    assert report["forbidden_optional_extra_hits"]


def test_no_torch_verifier_covers_console_train_fast_path() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "tools/check_native_no_torch_deps.py",
            "--skip-artifacts",
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    import json

    report = json.loads(proc.stdout)
    assert report["egg_info_dependencies"]["passed"] is True
    entries = {
        str(entry["name"]): entry
        for entry in report["python_entrypoints"]
    }
    console_entry = entries["nfn_console_train_fast_command"]
    assert console_entry["passed"] is True
    assert "--train-transformer-lm" in str(console_entry["stdout"])
    assert "--backend" in str(console_entry["stdout"])
    assert "tile-cuda" in str(console_entry["stdout"])
