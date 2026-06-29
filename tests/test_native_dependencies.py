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


def test_surrogate_trainer_module_import_is_lean_without_torch_numpy() -> None:
    code = r'''
import importlib.abc
import sys

class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in {"torch", "numpy"}:
            raise ImportError(f"blocked optional dependency: {fullname}")
        return None

sys.meta_path.insert(0, BlockOptional())
from neuralfn.trainer import SurrogateTrainer, TrainConfig

trainer = SurrogateTrainer(object(), TrainConfig(epochs=1))
print("TRAINER", type(trainer).__name__)
print("TORCH_LOADED", "torch" in sys.modules)
print("NUMPY_LOADED", "numpy" in sys.modules)
try:
    trainer.build_surrogates()
except ImportError as exc:
    print("OPTIONAL_ERROR", "optional legacy graph/Torch stack" in str(exc))
else:
    raise SystemExit("build_surrogates unexpectedly succeeded")
'''
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "TRAINER SurrogateTrainer" in proc.stdout
    assert "TORCH_LOADED False" in proc.stdout
    assert "NUMPY_LOADED False" in proc.stdout
    assert "OPTIONAL_ERROR True" in proc.stdout


def test_inference_module_import_is_lean_without_torch() -> None:
    code = r'''
import importlib.abc
import sys

class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] == "torch":
            raise ImportError(f"blocked optional dependency: {fullname}")
        return None

sys.meta_path.insert(0, BlockTorch())
from neuralfn.inference import InferenceCache, export_to_pt, load_pt_checkpoint

print("CACHE", InferenceCache.__name__)
print("EXPORT_CALLABLE", callable(export_to_pt))
print("LOAD_CALLABLE", callable(load_pt_checkpoint))
print("TORCH_LOADED", "torch" in sys.modules)
try:
    load_pt_checkpoint("/tmp/does-not-matter.pt")
except ImportError as exc:
    print("OPTIONAL_ERROR", "require PyTorch" in str(exc))
else:
    raise SystemExit("load_pt_checkpoint unexpectedly succeeded")
'''
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "CACHE InferenceCache" in proc.stdout
    assert "EXPORT_CALLABLE True" in proc.stdout
    assert "LOAD_CALLABLE True" in proc.stdout
    assert "TORCH_LOADED False" in proc.stdout
    assert "OPTIONAL_ERROR True" in proc.stdout


def test_evolutionary_module_import_is_lean_without_numpy() -> None:
    code = r'''
import importlib.abc
import sys

class BlockNumpy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] == "numpy":
            raise ImportError(f"blocked optional dependency: {fullname}")
        return None

sys.meta_path.insert(0, BlockNumpy())
from neuralfn.evolutionary import EvoConfig, EvolutionaryTrainer

trainer = EvolutionaryTrainer(object(), EvoConfig(generations=1))
print("TRAINER", type(trainer).__name__)
print("NUMPY_LOADED", "numpy" in sys.modules)
try:
    trainer.train([], [])
except ImportError as exc:
    print("OPTIONAL_ERROR", "requires NumPy" in str(exc))
else:
    raise SystemExit("train unexpectedly succeeded")
'''
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "TRAINER EvolutionaryTrainer" in proc.stdout
    assert "NUMPY_LOADED False" in proc.stdout
    assert "OPTIONAL_ERROR True" in proc.stdout


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


def test_no_torch_verifier_covers_universal_gpt_native_routes() -> None:
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
    python_entries = {
        str(entry["name"]): entry
        for entry in report["python_entrypoints"]
    }
    shell_entries = {
        str(entry["name"]): entry
        for entry in report["shell_entrypoints"]
    }

    custom_graph_entry = python_entries["nfn_train_gpt_custom_graph_command"]
    assert custom_graph_entry["passed"] is True
    assert "--model-family gpt3" in str(custom_graph_entry["stdout"])
    assert "--graph-file" in str(custom_graph_entry["stdout"])
    assert "--backend tile-cuda" in str(custom_graph_entry["stdout"])

    sdk_entry = python_entries["native_sdk_public_exports"]
    assert sdk_entry["passed"] is True
    assert "native-sdk-public-exports-ok" in str(sdk_entry["stdout"])

    sm120_gpt3_entry = shell_entries["train_gpt_sm120_gpt3_dry_run"]
    assert sm120_gpt3_entry["passed"] is True
    assert "--model-family gpt3" in str(sm120_gpt3_entry["stdout"])
    assert "--template-name gpt3" in str(sm120_gpt3_entry["stdout"])
    assert "--train-seq-len 2048" in str(sm120_gpt3_entry["stdout"])

    catalog_entry = shell_entries["native_gpt_linked_list_templates"]
    assert catalog_entry["passed"] is True
    catalog = json.loads(str(catalog_entry["stdout"]))
    templates_by_name = {
        str(template["name"]): template
        for template in catalog["templates"]
    }
    expected_native_templates = {
        "gpt": "gpt2",
        "gpt2": "gpt2",
        "gpt2_modern": "gpt2_modern",
        "gpt2_megakernel": "gpt2_megakernel",
        "gpt2_moa": "gpt2_moa",
        "gpt3": "gpt3",
        "nanogpt": "nanogpt",
        "nanogpt_modern": "nanogpt_modern",
        "nanogpt_megakernel": "nanogpt_megakernel",
    }
    for template_name, resolved_name in expected_native_templates.items():
        template = templates_by_name[template_name]
        assert template["resolved_native_template_name"] == resolved_name
        assert template["selected_graph_support_status"] == "native-transformer-lm"
        assert template["selected_graph_native_runnable"] is True

    gpt3_geometry = templates_by_name["gpt3"]["selected_template_geometry"]
    assert gpt3_geometry["seq_len"] == 2048
    assert gpt3_geometry["model_dim"] == 768
    assert gpt3_geometry["num_heads"] == 12
    assert gpt3_geometry["num_layers"] == 12

    for template_name in ("nanogpt", "nanogpt_modern", "nanogpt_megakernel"):
        geometry = templates_by_name[template_name]["selected_template_geometry"]
        assert geometry["model_dim"] == 320
        assert geometry["num_heads"] == 5
        assert geometry["num_layers"] == 5
        assert geometry["seq_len"] == 1024

    llama_template = templates_by_name["llama"]
    assert llama_template["selected_graph_support_status"] == "template-native-trainer-missing"
    assert llama_template["selected_graph_native_runnable"] is False
