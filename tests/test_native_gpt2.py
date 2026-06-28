from __future__ import annotations

import json
import importlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import struct
from types import SimpleNamespace

import neuralfn
import neuralfn.native_cuda_device as native_cuda_device_module
import neuralfn.native_train as native_train_module
import neuralfn.native_gpt2 as native_gpt2_module
import pytest

from neuralfn.config import SHIPPED_GPT_TEMPLATE_PRESETS
from neuralfn.native_gpt import (
    NativeGptRunConfig,
    NativeGptRunnerStatus,
    build_native_gpt_compiled_cli_run_config,
    exec_native_gpt,
    native_gpt_activation,
    native_gpt_encoding_vocab_size,
    native_gpt_kernel_backend,
    native_gpt_runner_status,
    normalize_native_gpt_encoding_name,
    read_native_gpt_checkpoint_info,
    resolve_native_gpt_binding_command,
    run_native_gpt,
)
from neuralfn.native_gpt2 import (
    NativeGpt2RunConfig,
    build_native_gpt2_compiled_cli_run_config,
    build_native_gpt2_run_config,
    exec_native_gpt2,
    latest_native_gpt2_checkpoint,
    native_gpt2_activation,
    native_gpt2_checkpoint_sampler_argv,
    native_gpt2_checkpoint_sampler_env,
    native_gpt2_parameter_count,
    native_gpt2_prompt_tokens,
    native_gpt2_runner_status,
    read_native_gpt2_checkpoint_info,
    render_native_gpt2_checkpoint_sampler_text,
    resolve_native_gpt2_cli,
    resolve_native_gpt2_binding_command,
    resolve_native_gpt2_executable,
    resolve_native_gpt2_launcher,
    resolve_native_gpt2_token_shards,
    run_native_gpt2,
    run_native_gpt2_checkpoint_sampler,
    write_native_gpt2_run_config,
)
from neuralfn.native_train import (
    NativeTrainRunnerStatus,
    build_native_gpt_launcher_run_config,
    build_native_sm120_gpt_run_config,
    build_native_train_run_config,
    exec_native_train,
    native_train_model_registry,
    native_train_runner_status,
    resolve_native_gpt_launcher_train_cli,
    resolve_native_sm120_train_cli,
    resolve_native_train_binding_command,
    resolve_native_train_cli,
    run_native_train,
)


def _write_raw_text_dataset(root: Path) -> tuple[Path, dict[str, object]]:
    dataset_path = root / "tiny"
    dataset_path.mkdir()
    (dataset_path / "data.txt").write_text("hello world. " * 128, encoding="utf-8")
    (dataset_path / "val.txt").write_text("validation story. " * 64, encoding="utf-8")
    meta = {"data_format": "raw_text", "source": "local"}
    (dataset_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return dataset_path, meta


def _write_uint16_shard_dataset(root: Path) -> Path:
    dataset_path = root / "uint16"
    dataset_path.mkdir()
    token_bytes = struct.pack("<" + "H" * 2048, *[idx % 256 for idx in range(2048)])
    (dataset_path / "fineweb_train_000000.bin").write_bytes(token_bytes)
    (dataset_path / "fineweb_val_000000.bin").write_bytes(token_bytes)
    (dataset_path / "meta.json").write_text(
        json.dumps(
            {
                "data_format": "uint16_shards",
                "tokenizer_encoding": "gpt2",
                "tokenizer_vocab_size": 50257,
            }
        ),
        encoding="utf-8",
    )
    return dataset_path


def _load_train_gpt_native_script_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "cli" / "scripts" / "train_gpt_native.py"
    spec = importlib.util.spec_from_file_location("train_gpt_native_direct_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_native_checkpoint(path: Path, *, step: int | None = None, version: int = 5) -> Path:
    max_seq_len = 8
    vocab_size = 16
    num_layers = 1
    num_heads = 1
    channels = 4
    padded_vocab_size = 16
    header = [0] * 256
    header[:8] = [
        20240326,
        version,
        max_seq_len,
        vocab_size,
        num_layers,
        num_heads,
        channels,
        padded_vocab_size,
    ]
    bytes_per_param = 4 if version == 3 else 2
    parameter_count = native_gpt2_parameter_count(
        max_seq_len=max_seq_len,
        padded_vocab_size=padded_vocab_size,
        num_layers=num_layers,
        channels=channels,
    )
    path.write_bytes(struct.pack("<" + "i" * 256, *header) + b"\0" * (parameter_count * bytes_per_param))
    if step is not None:
        (path.parent / f"DONE_{step:08d}").write_text("", encoding="utf-8")
    return path


def test_native_no_torch_dependency_verifier_covers_python_entrypoints() -> None:
    root = Path(__file__).resolve().parents[1]
    verifier_source = (root / "tools" / "check_native_no_torch_deps.py").read_text(encoding="utf-8")
    assert 'cli_root = repo_root / "cli"' in verifier_source
    assert "str(cli_root)" in verifier_source
    proc = subprocess.run(
        [
            sys.executable,
            str(root / "tools" / "check_native_no_torch_deps.py"),
            "--skip-artifacts",
            "--json",
        ],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["passed"] is True
    assert payload["artifacts"] == []
    assert set(payload["forbidden_python_import_roots"]) >= {
        "torch",
        "numpy",
        "tiktoken",
        "server.dataset_manager",
        "infer_gpt",
        "nfn_impl",
    }
    project_dependencies = payload["project_dependencies"]
    assert project_dependencies["forbidden_optional_extra_names"] == ["torch"]
    assert project_dependencies["forbidden_optional_extra_name_hits"] == []
    assert project_dependencies["forbidden_optional_extra_dependency_prefixes"]["all"] == [
        "torch",
        "torchvision",
        "torchaudio",
    ]
    assert project_dependencies["forbidden_optional_extra_dependency_prefixes"]["torch"] == [
        "torch",
        "torchvision",
        "torchaudio",
    ]
    assert project_dependencies["forbidden_optional_extra_hits"] == {}
    assert project_dependencies["optional_dependency_hits"].get("torch") is None
    requirements_dependencies = payload["requirements_dependencies"]
    assert requirements_dependencies["name"] == "requirements_default_dependencies"
    assert requirements_dependencies["exists"] is True
    assert requirements_dependencies["passed"] is True
    assert requirements_dependencies["offenders"] == []
    assert requirements_dependencies["forbidden_dependency_prefixes"] == [
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
    ]
    entrypoints = {entry["name"]: entry for entry in payload["python_entrypoints"]}
    assert entrypoints["train_gpt_fast_command"]["passed"] is True
    assert entrypoints["train_gpt_fast_command"]["elapsed_seconds"] >= 0.0
    assert entrypoints["train_gpt_fast_command"]["startup_budget_seconds"] == 2.0
    assert entrypoints["train_gpt_fast_command"]["startup_within_budget"] is True
    assert "--eval-batches 20" in entrypoints["train_gpt_fast_command"]["stdout"]
    assert entrypoints["train_gpt2_compat_fast_command"]["passed"] is True
    assert entrypoints["train_gpt2_compat_fast_command"]["startup_within_budget"] is True
    assert "--train-transformer-lm" in entrypoints["train_gpt2_compat_fast_command"]["stdout"]
    assert entrypoints["train_gpt2_compat_template_name_command"]["passed"] is True
    assert "--template-name gpt2_moa" in entrypoints["train_gpt2_compat_template_name_command"]["stdout"]
    assert "--native-cuda-activation moa" in entrypoints["train_gpt2_compat_template_name_command"]["stdout"]
    assert entrypoints["train_gpt2_compat_custom_graph_command"]["passed"] is True
    assert "--graph-file" in entrypoints["train_gpt2_compat_custom_graph_command"]["stdout"]
    shell_entrypoints = {entry["name"]: entry for entry in payload["shell_entrypoints"]}
    assert shell_entrypoints["train_gpt_sm120_dry_run"]["passed"] is True
    assert "--model-family gpt" in shell_entrypoints["train_gpt_sm120_dry_run"]["stdout"]
    assert "--template-name gpt" in shell_entrypoints["train_gpt_sm120_dry_run"]["stdout"]
    assert "--batch-size 64" in shell_entrypoints["train_gpt_sm120_dry_run"]["stdout"]
    assert "--train-seq-len 1024" in shell_entrypoints["train_gpt_sm120_dry_run"]["stdout"]
    assert shell_entrypoints["train_gpt_sm120_gpt3_dry_run"]["passed"] is True
    assert "--model-family gpt3" in shell_entrypoints["train_gpt_sm120_gpt3_dry_run"]["stdout"]
    assert "--template-name gpt3" in shell_entrypoints["train_gpt_sm120_gpt3_dry_run"]["stdout"]
    assert "--batch-size 32" in shell_entrypoints["train_gpt_sm120_gpt3_dry_run"]["stdout"]
    assert "--train-seq-len 2048" in shell_entrypoints["train_gpt_sm120_gpt3_dry_run"]["stdout"]
    assert shell_entrypoints["train_gpt_sm120_custom_graph_dry_run"]["passed"] is True
    assert "--template-name gpt2_moa" in shell_entrypoints["train_gpt_sm120_custom_graph_dry_run"]["stdout"]
    assert "--native-cuda-activation moa" in shell_entrypoints["train_gpt_sm120_custom_graph_dry_run"]["stdout"]
    assert "--graph-file /tmp/native-compatible-gpt-graph.json" in shell_entrypoints[
        "train_gpt_sm120_custom_graph_dry_run"
    ]["stdout"]
    assert shell_entrypoints["train_gpt_sm120_compiled_dry_run"]["passed"] is True
    assert "--model-family gpt" in shell_entrypoints["train_gpt_sm120_compiled_dry_run"]["stdout"]
    assert "--template-name gpt" in shell_entrypoints["train_gpt_sm120_compiled_dry_run"]["stdout"]
    assert "--batch-size 64" in shell_entrypoints["train_gpt_sm120_compiled_dry_run"]["stdout"]
    assert "--train-seq-len 1024" in shell_entrypoints["train_gpt_sm120_compiled_dry_run"]["stdout"]
    assert shell_entrypoints["train_gpt_sm120_compiled_gpt3_dry_run"]["passed"] is True
    assert "--model-family gpt3" in shell_entrypoints["train_gpt_sm120_compiled_gpt3_dry_run"]["stdout"]
    assert "--template-name gpt3" in shell_entrypoints["train_gpt_sm120_compiled_gpt3_dry_run"]["stdout"]
    assert "--batch-size 32" in shell_entrypoints["train_gpt_sm120_compiled_gpt3_dry_run"]["stdout"]
    assert "--train-seq-len 2048" in shell_entrypoints["train_gpt_sm120_compiled_gpt3_dry_run"]["stdout"]
    assert shell_entrypoints["train_gpt_sm120_compiled_custom_graph_dry_run"]["passed"] is True
    assert "--template-name gpt2_moa" in shell_entrypoints[
        "train_gpt_sm120_compiled_custom_graph_dry_run"
    ]["stdout"]
    assert "--native-cuda-activation moa" in shell_entrypoints[
        "train_gpt_sm120_compiled_custom_graph_dry_run"
    ]["stdout"]
    assert "--graph-file /tmp/native-compatible-gpt-graph.json" in shell_entrypoints[
        "train_gpt_sm120_compiled_custom_graph_dry_run"
    ]["stdout"]
    assert shell_entrypoints["train_gpt_compiled_dry_run"]["passed"] is True
    assert "--model-family gpt" in shell_entrypoints["train_gpt_compiled_dry_run"]["stdout"]
    assert "--template-name gpt" in shell_entrypoints["train_gpt_compiled_dry_run"]["stdout"]
    assert shell_entrypoints["train_gpt_compiled_generic_env_dry_run"]["passed"] is True
    assert "--model-family gpt3" in shell_entrypoints["train_gpt_compiled_generic_env_dry_run"]["stdout"]
    assert "--template-name gpt3" in shell_entrypoints["train_gpt_compiled_generic_env_dry_run"]["stdout"]
    assert "--batch-size 32" in shell_entrypoints["train_gpt_compiled_generic_env_dry_run"]["stdout"]
    assert "--train-seq-len 2048" in shell_entrypoints["train_gpt_compiled_generic_env_dry_run"]["stdout"]
    assert shell_entrypoints["native_gpt_linked_list_templates"]["passed"] is True
    assert shell_entrypoints["native_gpt_linked_list_templates"]["startup_within_budget"] is True
    assert "shipped_template_catalog" in shell_entrypoints["native_gpt_linked_list_templates"]["stdout"]
    assert shell_entrypoints["native_gpt2_compat_list_templates"]["passed"] is True
    assert shell_entrypoints["native_gpt2_compat_list_templates"]["startup_within_budget"] is True
    assert "shipped_template_catalog" in shell_entrypoints["native_gpt2_compat_list_templates"]["stdout"]
    assert shell_entrypoints["native_train_registry_list_models"]["passed"] is True
    assert shell_entrypoints["native_train_registry_list_models"]["startup_within_budget"] is True
    assert '"models"' in shell_entrypoints["native_train_registry_list_models"]["stdout"]
    assert '"name": "gpt"' in shell_entrypoints["native_train_registry_list_models"]["stdout"]
    assert '"status": "implemented"' in shell_entrypoints["native_train_registry_list_models"]["stdout"]
    assert shell_entrypoints["native_train_gpt_list_templates"]["passed"] is True
    assert shell_entrypoints["native_train_gpt_list_templates"]["startup_within_budget"] is True
    assert '"action": "list_templates"' in shell_entrypoints["native_train_gpt_list_templates"]["stdout"]
    assert '"token_shards_resolved": false' in shell_entrypoints["native_train_gpt_list_templates"]["stdout"]
    assert shell_entrypoints["native_train_gpt_wrapper_list_templates"]["passed"] is True
    assert shell_entrypoints["native_train_gpt_wrapper_list_templates"]["startup_within_budget"] is True
    assert '"action": "list_templates"' in shell_entrypoints["native_train_gpt_wrapper_list_templates"]["stdout"]
    assert '"token_shards_resolved": false' in shell_entrypoints[
        "native_train_gpt_wrapper_list_templates"
    ]["stdout"]
    assert "--train-seq-len 2048" not in entrypoints["train_gpt2_compat_custom_graph_command"]["stdout"]
    assert entrypoints["train_gpt_native_fast_command"]["passed"] is True
    assert entrypoints["train_gpt_native_fast_command"]["startup_within_budget"] is True
    assert "nfn_gpt_native_train" in entrypoints["train_gpt_native_fast_command"]["stdout"]
    metadata_entry = entrypoints["train_gpt_native_metadata_missing_alias_command"]
    assert metadata_entry["passed"] is True
    assert metadata_entry["startup_within_budget"] is True
    assert "--dataset-alias" in metadata_entry["stdout"]
    assert "missing_alias_for_cpp_resolver" in metadata_entry["stdout"]
    assert "--print-plan" in metadata_entry["stdout"]
    assert "Dataset alias 'missing_alias_for_cpp_resolver' was not found" not in metadata_entry["stderr"]
    assert entrypoints["train_gpt2_evo_fast_command"]["passed"] is True
    assert entrypoints["train_nanogpt_fast_command"]["passed"] is True
    for name in (
        "train_gpt2_evo_default_native_dispatch",
        "train_nanogpt_default_native_dispatch",
        "train_llama_fast_default_native_dispatch",
        "train_llama_megakernel_default_native_dispatch",
        "train_mixllama_fast_default_native_dispatch",
        "train_jepa_semantic_default_native_dispatch",
        "train_semantic_router_moe_default_native_dispatch",
        "train_semantic_router_moe_overnight_default_native_dispatch",
        "train_deepseek_v4_default_native_dispatch",
    ):
        assert entrypoints[name]["passed"] is True
        assert entrypoints[name]["startup_within_budget"] is True
    assert entrypoints["nfn_train_fast_command"]["passed"] is True
    assert entrypoints["nfn_train_default_fast_command"]["passed"] is True
    assert entrypoints["nfn_train_gpt_template_name_command"]["passed"] is True
    assert "--template-name gpt2_moa" in entrypoints["nfn_train_gpt_template_name_command"]["stdout"]
    assert "--native-cuda-activation moa" in entrypoints["nfn_train_gpt_template_name_command"]["stdout"]
    assert entrypoints["nfn_train_gpt_custom_graph_command"]["passed"] is True
    assert "--model-family gpt" in entrypoints["nfn_train_gpt_custom_graph_command"]["stdout"]
    assert "--graph-file" in entrypoints["nfn_train_gpt_custom_graph_command"]["stdout"]
    assert "--train-seq-len 2048" not in entrypoints["nfn_train_gpt_custom_graph_command"]["stdout"]
    assert entrypoints["infer_gpt_native_info"]["passed"] is True
    assert entrypoints["infer_gpt_native_sample_prompt_tokens"]["passed"] is True
    assert entrypoints["nfn_infer_native_info"]["passed"] is True
    assert entrypoints["nfn_infer_native_directory_info"]["passed"] is True
    assert "model_00000020.bin" in entrypoints["nfn_infer_native_directory_info"]["stdout"]
    assert entrypoints["nfn_infer_native_sample_prompt_tokens"]["passed"] is True
    assert entrypoints["nfn_console_infer_native_info"]["passed"] is True
    assert entrypoints["nfn_console_infer_native_info"]["startup_within_budget"] is True
    assert entrypoints["nfn_console_infer_native_sample_prompt_tokens"]["passed"] is True
    assert (
        entrypoints["nfn_console_infer_native_sample_prompt_tokens"]["startup_within_budget"]
        is True
    )
    assert entrypoints["native_sdk_imports"]["passed"] is True
    assert entrypoints["native_sdk_public_exports"]["passed"] is True
    assert entrypoints["native_sdk_binding_imports"]["passed"] is True
    shell_entrypoints = {entry["name"]: entry for entry in payload["shell_entrypoints"]}
    assert shell_entrypoints["bench_linear_backward_dry_run"]["passed"] is True
    assert "--candidate-symbol" in shell_entrypoints["bench_linear_backward_dry_run"]["stdout"]
    assert shell_entrypoints["bench_lm_head_backward_dry_run"]["passed"] is True
    assert "--candidate-first" in shell_entrypoints["bench_lm_head_backward_dry_run"]["stdout"]
    assert "--candidate-symbol" in shell_entrypoints["bench_lm_head_backward_dry_run"]["stdout"]
    assert shell_entrypoints["bench_native_gpt_linear_hot_matrix_dry_run"]["passed"] is True
    assert "smoke-dinput" in shell_entrypoints["bench_native_gpt_linear_hot_matrix_dry_run"]["stdout"]
    assert "smoke-dweight" in shell_entrypoints["bench_native_gpt_linear_hot_matrix_dry_run"]["stdout"]


def test_native_no_torch_dependency_verifier_requires_compiled_gpt_artifacts() -> None:
    root = Path(__file__).resolve().parents[1]
    module_path = root / "tools" / "check_native_no_torch_deps.py"
    spec = importlib.util.spec_from_file_location("check_native_no_torch_deps", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    artifacts = module.default_artifacts()
    required = {str(path) for path in module.REQUIRED_DEFAULT_ARTIFACTS}
    assert {str(path) for path in artifacts} >= required

    present_optional = [
        path for path in module.OPTIONAL_DEFAULT_ARTIFACTS if (root / path).exists()
    ]
    present_optional_globs = [
        path for pattern in module.OPTIONAL_DEFAULT_ARTIFACT_GLOBS for path in sorted(root.glob(pattern))
    ]
    assert all(path in artifacts for path in present_optional)
    assert all(path.relative_to(root) in artifacts for path in present_optional_globs)
    assert Path("build/nfn_gpt_native_train") in module.REQUIRED_DEFAULT_ARTIFACTS
    assert Path("build/libnfn_native_train_tile_ops.so") in module.REQUIRED_DEFAULT_ARTIFACTS
    assert Path("build/nfn_gpt_native_train_linked") in module.REQUIRED_DEFAULT_ARTIFACTS
    assert Path("build/nfn_gpt2_native_train") in module.REQUIRED_DEFAULT_ARTIFACTS
    assert Path("build/nfn_train_gpt") in module.REQUIRED_DEFAULT_ARTIFACTS
    assert Path("build/nfn_train_gpt_sm120") in module.REQUIRED_DEFAULT_ARTIFACTS
    assert Path("build/nfn_native_train") in module.REQUIRED_DEFAULT_ARTIFACTS
    assert Path("build/nfn_native_train") not in module.OPTIONAL_DEFAULT_ARTIFACTS
    assert Path("build/nfn_train_gpt_sm120") not in module.OPTIONAL_DEFAULT_ARTIFACTS
    assert Path("build/nfn_train_gpt") not in module.OPTIONAL_DEFAULT_ARTIFACTS
    assert Path("build/nfn_gpt_native_train_linked") not in module.OPTIONAL_DEFAULT_ARTIFACTS
    assert Path("build/nfn_gpt2_native_train") not in module.OPTIONAL_DEFAULT_ARTIFACTS
    assert Path("build/nfn_gpt2_evo_native_train") in module.OPTIONAL_DEFAULT_ARTIFACTS
    assert Path("build/nfn_nanogpt_native_train") in module.OPTIONAL_DEFAULT_ARTIFACTS
    assert Path("build/linear_backward_bench") in module.OPTIONAL_DEFAULT_ARTIFACTS
    assert Path("build/lm_head_backward_bench") in module.OPTIONAL_DEFAULT_ARTIFACTS
    assert module.artifact_rebuild_command(
        root / "build" / "libnfn_native_train_tile_ops_tk.so",
        root,
    ) == (
        "bash",
        "tools/build_native_train_tile_ops.sh",
        "build/libnfn_native_train_tile_ops_tk.so",
    )
    assert module.artifact_rebuild_command(
        root / "build" / "linear_backward_bench",
        root,
    ) == ("bash", "tools/build_linear_backward_bench.sh")
    assert module.artifact_rebuild_command(
        root / "build" / "lm_head_backward_bench",
        root,
    ) == ("bash", "tools/build_lm_head_backward_bench.sh")
    assert module.artifact_rebuild_command(
        root / "build" / "nfn_gpt2_evo_native_train",
        root,
    ) == ("bash", "tools/build_native_missing_trainers.sh")
    assert module.artifact_rebuild_command(
        root / "build" / "nfn_nanogpt_native_train",
        root,
    ) == ("bash", "tools/build_native_missing_trainers.sh")
    assert "neuralfn/_native_gpt.*.so" in module.REQUIRED_DEFAULT_ARTIFACT_GLOBS
    assert "neuralfn/_native_gpt2.*.so" in module.REQUIRED_DEFAULT_ARTIFACT_GLOBS
    assert "neuralfn/_native_train.*.so" in module.REQUIRED_DEFAULT_ARTIFACT_GLOBS
    assert not module.OPTIONAL_DEFAULT_ARTIFACT_GLOBS


def test_native_no_torch_dependency_verifier_detects_stale_artifacts(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    module_path = root / "tools" / "check_native_no_torch_deps.py"
    spec = importlib.util.spec_from_file_location("check_native_no_torch_deps", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    artifact = tmp_path / "build" / "nfn_gpt_native_train"
    source = tmp_path / "neuralfn" / "csrc" / "native_gpt2" / "nfn_gpt2_native_train.cpp"
    artifact.parent.mkdir(parents=True)
    source.parent.mkdir(parents=True)
    artifact.write_text("old binary", encoding="utf-8")
    source.write_text("new source", encoding="utf-8")
    os.utime(artifact, (100.0, 100.0))
    os.utime(source, (200.0, 200.0))

    dependencies = module.artifact_source_dependencies(artifact, tmp_path)
    stale_sources = module.stale_artifact_sources(artifact, tmp_path)

    assert Path("neuralfn/csrc/native_gpt2/nfn_gpt2_native_train.cpp") in dependencies
    assert stale_sources
    assert stale_sources[0]["source"] == "neuralfn/csrc/native_gpt2/nfn_gpt2_native_train.cpp"
    assert stale_sources[0]["exists"] is True


def test_native_no_torch_dependency_verifier_maps_optional_family_sources() -> None:
    root = Path(__file__).resolve().parents[1]
    module_path = root / "tools" / "check_native_no_torch_deps.py"
    spec = importlib.util.spec_from_file_location("check_native_no_torch_deps", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    evo_dependencies = module.artifact_source_dependencies(
        root / "build" / "nfn_gpt2_evo_native_train",
        root,
    )
    nano_dependencies = module.artifact_source_dependencies(
        root / "build" / "nfn_nanogpt_native_train",
        root,
    )
    llama_dependencies = module.artifact_source_dependencies(
        root / "build" / "nfn_llama_native_train",
        root,
    )

    assert Path("neuralfn/csrc/native_train/gpt2_evo_native_train.cpp") in evo_dependencies
    assert Path("neuralfn/csrc/native_train/shipped_gpt_template_presets.h") in evo_dependencies
    assert Path("neuralfn/csrc/native_train/nanogpt_native_train.cpp") in nano_dependencies
    assert Path("neuralfn/csrc/native_train/token_shards.cpp") in nano_dependencies
    assert Path("neuralfn/csrc/native_train/missing_native_train.cpp") in llama_dependencies
    assert Path("tools/build_native_missing_trainers.sh") in llama_dependencies


def test_native_no_torch_dependency_verifier_maps_sdk_bindings() -> None:
    root = Path(__file__).resolve().parents[1]
    module_path = root / "tools" / "check_native_no_torch_deps.py"
    spec = importlib.util.spec_from_file_location("check_native_no_torch_deps", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    generic = module.artifact_source_dependencies(
        root / "neuralfn" / "_native_gpt.cpython-313-x86_64-linux-gnu.so",
        root,
    )
    compat = module.artifact_source_dependencies(
        root / "neuralfn" / "_native_gpt2.cpython-313-x86_64-linux-gnu.so",
        root,
    )
    unified = module.artifact_source_dependencies(
        root / "neuralfn" / "_native_train.cpython-313-x86_64-linux-gnu.so",
        root,
    )

    assert Path("tools/build_native_gpt_binding.sh") in generic
    assert Path("tools/build_native_gpt2_binding.sh") in compat
    assert Path("tools/build_native_train_binding.sh") in unified
    assert Path("neuralfn/csrc/native_gpt2/binding.cpp") in generic
    assert Path("neuralfn/csrc/native_train/binding.cpp") in unified


def test_resolve_native_gpt2_token_shards_materializes_uint16_cache(tmp_path: Path) -> None:
    dataset_path, meta = _write_raw_text_dataset(tmp_path)

    cached_meta, train_shard, val_shard = resolve_native_gpt2_token_shards(
        "tiny",
        dataset_path=dataset_path,
        dataset_meta=meta,
        encoding_name="gpt2",
    )

    assert cached_meta["data_format"] == "uint16_shards"
    assert train_shard.name == "fineweb_train_000000.bin"
    assert val_shard.name == "fineweb_val_000000.bin"
    assert train_shard.exists()
    assert val_shard.exists()


def test_read_native_gpt2_checkpoint_info_and_latest_done_marker(tmp_path: Path) -> None:
    old_checkpoint = _write_native_checkpoint(tmp_path / "model_00000100.bin", step=100)
    checkpoint = _write_native_checkpoint(tmp_path / "model_00000200.bin", step=200)

    info = read_native_gpt2_checkpoint_info(checkpoint)

    assert info.path == str(checkpoint)
    assert info.precision == "bf16"
    assert info.version == 5
    assert info.max_seq_len == 8
    assert info.vocab_size == 16
    assert info.num_layers == 1
    assert info.num_heads == 1
    assert info.channels == 4
    assert info.padded_vocab_size == 16
    assert info.size_matches is True
    assert info.step == 200
    assert info.done_marker_exists is True
    assert latest_native_gpt2_checkpoint(tmp_path) == checkpoint
    assert old_checkpoint.exists()


def test_read_native_gpt_checkpoint_info_uses_generic_sdk_class(tmp_path: Path) -> None:
    checkpoint = _write_native_checkpoint(tmp_path / "model_00000001.bin", step=1)

    info = read_native_gpt_checkpoint_info(checkpoint)

    assert type(info).__name__ == "NativeGptCheckpointInfo"
    assert type(info) is neuralfn.NativeGptCheckpointInfo
    assert info.path == str(checkpoint)
    assert info.done_marker_exists is True


def test_nfn_infer_checkpoint_directory_uses_latest_native_checkpoint(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    _write_native_checkpoint(tmp_path / "model_00000010.bin", step=10)
    checkpoint = _write_native_checkpoint(tmp_path / "model_00000020.bin", step=20)

    proc = subprocess.run(
        [
            sys.executable,
            str(root / "cli" / "nfn.py"),
            "infer",
            "--checkpoint",
            str(tmp_path),
            "--native-info",
        ],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "Native GPT checkpoint detected" in proc.stdout
    assert f"path: {checkpoint}" in proc.stdout
    assert "checkpoint_step: 20" in proc.stdout
    assert "Traceback" not in proc.stderr


def test_native_gpt_checkpoint_sampler_sdk_builds_no_torch_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _write_native_checkpoint(tmp_path / "model_00000001.bin", step=1)
    native_cli = tmp_path / "nfn_gpt_native_train"
    native_cli.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    native_cli.chmod(0o755)
    monkeypatch.setenv("NFN_NATIVE_GPT_CLI", str(native_cli))

    assert native_gpt2_prompt_tokens(prompt_tokens="1,2,3") == "1,2,3"
    assert native_gpt2_prompt_tokens(prompt="") == "50256"
    with pytest.raises(RuntimeError, match="token-id only by default"):
        native_gpt2_prompt_tokens(prompt="Hello")

    argv = native_gpt2_checkpoint_sampler_argv(
        checkpoint,
        prompt_tokens="1,2,3",
        max_new_tokens=7,
        temperature=0.7,
        top_k=12,
        repetition_penalty=1.1,
        seed=42,
    )

    assert argv == [
        str(native_cli),
        "--sample-checkpoint",
        str(checkpoint),
        "--prompt-tokens",
        "1,2,3",
        "--max-new-tokens",
        "7",
        "--temperature",
        "0.7",
        "--top-k",
        "12",
        "--repetition-penalty",
        "1.1",
        "--seed",
        "42",
    ]
    env = native_gpt2_checkpoint_sampler_env(cuda_visible_devices="2")
    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert env["CUDA_DEVICE_MAX_CONNECTIONS"] == "1"
    assert env["CUDA_MODULE_LOADING"] == "LAZY"
    monkeypatch.setattr(native_gpt2_module, "resolve_cuda_visible_devices_value", lambda value: "7")
    default_env = native_gpt2_checkpoint_sampler_env()
    assert default_env["CUDA_VISIBLE_DEVICES"] == "7"
    rendered = render_native_gpt2_checkpoint_sampler_text('{"generated_tokens": [1, 2, 3]}')
    assert "Generated token ids: [1, 2, 3]" in rendered


def test_nfn_native_checkpoint_sampler_uses_sdk_binding_helper() -> None:
    source = (Path(__file__).resolve().parents[1] / "cli" / "nfn.py").read_text(encoding="utf-8")
    function_body = source.rsplit("def _run_lightweight_native_gpt_sampler(", 1)[1].split("\ndef ", 1)[0]

    assert "run_native_gpt_checkpoint_sampler" in function_body
    assert "temperature=" in function_body
    assert "top_k=" in function_body
    assert "repetition_penalty=" in function_body
    assert "seed=" in function_body
    assert "native_gpt_checkpoint_sampler_argv" not in function_body
    assert "subprocess.run" not in function_body


def test_packed_qkv_attention_backward_chunks_large_batches() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "neuralfn"
        / "csrc"
        / "tile_cuda"
        / "kernels.cu"
    ).read_text(encoding="utf-8")

    assert "kTkPackedAttentionBackwardDefaultMaxBatchPerLaunch = 64" in source
    assert "NFN_NATIVE_GPT_PACKED_ATTENTION_BACKWARD_BATCH_CAP" in source
    assert "NFN_NATIVE_GPT2_PACKED_ATTENTION_BACKWARD_BATCH_CAP" in source
    assert "tk_packed_attention_backward_max_batch_per_launch()" in source
    assert "record_attention_backward_tk_chunk_batch(chunk_batch)" in source
    assert "attention_backward_tk_batch_cap()" in source
    assert "attention_backward_tk_chunk_batch_total()" in source
    assert "attention_backward_tk_chunk_batch_max()" in source
    assert "attention_backward_tk_chunk_batch_min()" in source
    assert "attention_backward_tk_chunk_batch_last()" in source
    assert "for (std::int64_t batch_begin = 0; batch_begin < batch;" in source
    assert "chunk_batch = std::min(max_batch_per_launch, batch - batch_begin)" in source
    assert "qkv_bf16_bits + batch_begin * packed_elements_per_batch" in source
    assert "out_bf16_bits + batch_begin * merged_elements_per_batch" in source
    assert "grad_out + batch_begin * merged_elements_per_batch" in source
    assert "saved_lse != nullptr ? saved_lse : workspace->lse" in source
    assert "workspace->packed_grad_bf + batch_begin * packed_elements_per_batch" in source


def test_native_gpt_direct_u16_path_elides_int64_token_arena() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "neuralfn"
        / "csrc"
        / "native_gpt2"
        / "nfn_gpt2_native_train.cpp"
    ).read_text(encoding="utf-8")

    assert "token_i64_arena_elements = direct_u16_token_ids_enabled ? 0 : rows * 2" in source
    assert "token_device_arena_suballocation_count = direct_u16_token_ids_enabled ? 1 : 2" in source
    assert "targets = direct_u16_token_ids_enabled ? nullptr : (token_i64_arena + rows)" in source
    assert "active_targets = direct_u16_token_ids_enabled ? nullptr : (token_i64_arena + active_rows)" in source
    assert "direct_u16_token_ids_enabled ? nullptr : (active_targets + row_start)" in source
    assert "token_i64_device_arena_elided" in source
    assert "token_i64_device_arena_bytes_elided" in source


def test_native_gpt_dense_modern_template_aliases_are_classified_explicitly() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "neuralfn"
        / "csrc"
        / "native_gpt2"
        / "nfn_gpt2_native_train.cpp"
    ).read_text(encoding="utf-8")

    assert 'name == "gpt2_modern"' in source
    assert 'name == "nanogpt_modern"' in source
    assert 'name == "nanogpt_megakernel"' in source
    assert '\\"gpt2_modern\\"' in source
    assert '\\"nanogpt_modern\\"' in source
    assert '\\"nanogpt_megakernel\\"' in source
    assert 'selector == "nanogpt" || selector == "nanogpt_megakernel" || selector == "nanogpt_modern"' in source


def test_native_gpt_layer_evo_candidate_loss_stays_device_resident() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "neuralfn"
        / "csrc"
        / "native_gpt2"
        / "nfn_gpt2_native_train.cpp"
    ).read_text(encoding="utf-8")

    assert "copy_loss_to_host" in source
    assert "native-forward-loss-device-resident-current-batch" in source
    assert "candidate_loss_transport" in source
    assert "device-to-device" in source
    assert "layer_evo.candidate_loss.copy_device_to_device" in source
    assert "layer_evo.candidate_loss.copy_host_to_device" not in source
    assert "layer_evo_candidate_loss_device_copy_count" in source
    assert "layer_evo_candidate_loss_host_roundtrips_elided" in source


def test_native_training_guard_sets_fast_cuda_ordinal_default() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "cli"
        / "scripts"
        / "native_training_guard.py"
    ).read_text(encoding="utf-8")

    assert "from neuralfn.native_cuda_device import" not in source
    assert "resolve_cuda_visible_devices_value" in source
    assert '_set_env_default_if_empty(env, "CUDA_VISIBLE_DEVICES", resolve_cuda_visible_devices_value("0"))' in source
    assert '_set_env_default_if_empty(env, "CUDA_DEVICE_MAX_CONNECTIONS", "1")' in source
    assert '_set_env_default_if_empty(env, "CUDA_MODULE_LOADING", "LAZY")' in source


def test_nfn_direct_native_train_sets_lazy_cuda_module_loading() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "cli"
        / "nfn.py"
    ).read_text(encoding="utf-8")

    assert "from neuralfn.native_cuda_device import" not in source
    assert "resolve_cuda_visible_devices_value" in source
    assert '_set_env_default_if_empty(env, "CUDA_VISIBLE_DEVICES", resolve_cuda_visible_devices_value("0"))' in source
    assert '_set_env_default_if_empty(env, "CUDA_DEVICE_MAX_CONNECTIONS", "1")' in source
    assert '_set_env_default_if_empty(env, "CUDA_MODULE_LOADING", "LAZY")' in source


def test_native_cuda_device_resolves_dedicated_to_display_disabled_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(argv)
        return SimpleNamespace(
            stdout=(
                "0, Enabled, 1\n"
                "1, Disabled, 12\n"
                "2, Disabled, 3\n"
            )
        )

    monkeypatch.setattr(native_cuda_device_module.subprocess, "run", fake_run)

    assert native_cuda_device_module.resolve_cuda_visible_devices_value("dedicated") == "2"
    assert native_cuda_device_module.resolve_cuda_visible_devices_value("auto") == "2"
    assert native_cuda_device_module.resolve_cuda_visible_devices_value("3") == "3"
    assert native_cuda_device_module.resolve_cuda_visible_devices_value("0") == "0"
    assert calls


def test_sm120_cuda13_validator_covers_native_cuda_smokes() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "validate_sm120_cuda13.sh"
    ).read_text(encoding="utf-8")

    assert "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES:-dedicated" in source
    assert 'LINKED_TRAIN_BIN="${ROOT_DIR}/build/nfn_gpt_native_train_linked"' in source
    assert 'elif [[ -x "${LINKED_TRAIN_BIN}" ]]; then' in source
    assert 'TILE_OPS_LIB="linked"' in source
    assert '"${TILE_OPS_LIB}" != "linked" && ! -f "${TILE_OPS_LIB}"' in source
    assert "--check-tile-ops" in source
    assert "--smoke-tile-ops" in source
    assert "--smoke-nvfp4-pack" in source
    assert "--smoke-transformer-lm-step" in source
    assert "--tinystories" in source
    assert "NFN_SM120_CUDA13_RUN_NO_TORCH" in source
    assert "tools/check_native_no_torch_deps.py --json" in source
    assert "tests/test_native_gpt2.py -q" in source
    assert "NFN_SM120_CUDA13_RUN_PYTEST" in source
    assert "NFN_SM120_CUDA13_RUN_BENCH" in source
    assert "NFN_SM120_CUDA13_CHECK_BENCH_CONTRACT" in source
    assert "bench_native_gpt_sm120_candidate.sh" in source
    assert "NFN_SM120_NATIVE_INCLUDE_LLMK_REFERENCE" in source
    assert "NFN_SM120_NATIVE_DISABLE_METRIC_RATIO_GATES" in source
    assert "candidate_native_metric_values" in source
    assert "candidate_native_metrics" in source
    assert "optimizer_tile_strategy" in source
    assert "tile-size-1024-sumsq-scale-adamw" in source
    assert "lm_head_classifier_backward_path_class" in source
    assert "diagnostic-cuda-graph-wrapper" in source
    assert "lm_head_ce_kernel_strategy" in source
    assert "no-loss-default-specialized-dlogits-vec8-loads-scalar-stores" in source
    assert "no-loss-specialized-dlogits-vec8-loads-normal-vec8-stores" in source
    assert "lm_head_fused_graph_prewarm_success_count" in source
    assert "lm_head_fused_graph_prewarm_duplicate_skip_count" in source
    assert "block_backward_input_linear_strategy" in source
    assert "tk-sm120-bf16-dinput" in source
    assert "block_backward_weight_linear_strategy" in source
    assert "shape-gated-bf16-cublaslt-dweight-bgrad-first-write-then-accumulate" in source
    assert "token_weight_init_strategy" in source
    assert "device-vector4-strided-power2-deterministic-fused-bf16-shadow" in source
    assert "SM120 CUDA 13.3 benchmark contract passed" in source
    assert "CUDA 13.3 SM120 validation passed." in source


def test_native_gpt_transformer_lm_reports_opt_in_async_allocator() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "neuralfn"
        / "csrc"
        / "native_gpt2"
        / "nfn_gpt2_native_train.cpp"
    ).read_text(encoding="utf-8")

    assert "NFN_NATIVE_GPT_CUDA_MALLOC_ASYNC" in source
    assert "cudaMallocAsync" in source
    assert "cudaFreeAsync" in source
    assert "device_cuda_malloc_async_requested" in source
    assert "device_cuda_malloc_async_enabled" in source
    assert "device_cuda_malloc_async_fallback_count" in source
    assert "cudaDeviceSynchronize after cudaFreeAsync" in source
    assert "setup_timing_accounted_ms" in source
    assert "setup_timing_unattributed_ms" in source
    assert "setup_timing_record_count" in source
    assert "setup.load_tile_ops" in source
    assert "setup.load_cuda_runtime" in source
    assert "setup.cuda_runtime_symbols" in source
    assert 'append_existing("/usr/local/cuda/lib64/libcudart.so.13")' in source
    assert 'append_existing("/usr/local/cuda/lib64/libcudart.so")' in source
    assert 'append("libcudart.so.13")' in source
    assert 'append("libcudart.so")' in source
    assert "tile_ops_dlopen_binding_strategy" in source
    assert "RTLD_LAZY | RTLD_LOCAL" in source
    assert "tile_ops_dlopen_wall_ms" in source
    assert "tile_ops_required_symbol_scan_wall_ms" in source
    assert "tile_ops_typed_symbol_load_wall_ms" in source
    assert "cuda_runtime_symbol_load_wall_ms" in source
    assert "cuda_runtime_version_preflight_wall_ms" in source
    assert "NFN_NATIVE_GPT_COMBINED_DEVICE_ARENA" in source
    assert "transformer_device_arena_requested" in source
    assert "transformer_device_arena_enabled" in source
    assert "transformer_device_arena_cuda_malloc_count" in source
    assert "cudaMalloc transformer_lm_combined_device_arena" in source


def test_native_gpt_transformer_lm_supports_linked_tile_ops_loader() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root
        / "neuralfn"
        / "csrc"
        / "native_gpt2"
        / "nfn_gpt2_native_train.cpp"
    ).read_text(encoding="utf-8")
    linked_build = (root / "tools" / "build_native_gpt_cli_linked.sh").read_text(
        encoding="utf-8"
    )
    build_script = (root / "tools" / "build_native_gpt_cli.sh").read_text(
        encoding="utf-8"
    )
    train_gpt_source = (root / "cli" / "scripts" / "train_gpt.py").read_text(
        encoding="utf-8"
    )
    nfn_source = (root / "cli" / "nfn.py").read_text(encoding="utf-8")
    native_train_sdk_source = (root / "neuralfn" / "native_train.py").read_text(
        encoding="utf-8"
    )
    native_train_source = (
        root / "neuralfn" / "csrc" / "native_train" / "nfn_native_train.cpp"
    ).read_text(encoding="utf-8")
    gpt2_evo_source = (
        root / "neuralfn" / "csrc" / "native_train" / "gpt2_evo_native_train.cpp"
    ).read_text(encoding="utf-8")
    build_all = (root / "tools" / "build_native_gpt2_all.sh").read_text(encoding="utf-8")
    train_gpt_build = (root / "tools" / "build_train_gpt_cli.sh").read_text(encoding="utf-8")
    rebuild_sm120 = (root / "tools" / "rebuild_native_sm120.sh").read_text(encoding="utf-8")
    train_sm120 = (root / "tools" / "train_gpt_sm120.sh").read_text(encoding="utf-8")
    train_sm120_cpp = (
        root / "neuralfn" / "csrc" / "native_train" / "train_gpt_sm120.cpp"
    ).read_text(encoding="utf-8")
    tile_ops_build = (root / "tools" / "build_native_train_tile_ops.sh").read_text(
        encoding="utf-8"
    )
    parity_bench = (root / "tools" / "bench_native_gpt_sm120_parity.sh").read_text(encoding="utf-8")
    candidate_bench = (root / "tools" / "bench_native_gpt_sm120_candidate.sh").read_text(encoding="utf-8")
    paired_speed = (root / "tools" / "paired_kernel_speed.py").read_text(encoding="utf-8")
    no_torch_verifier = (root / "tools" / "check_native_no_torch_deps.py").read_text(
        encoding="utf-8"
    )

    assert "linked_tile_ops_requested" in source
    assert "open_tile_ops_library" in source
    assert "dlopen(tile_lib_path.c_str(), RTLD_NOW | RTLD_LOCAL)" not in source
    assert 'executable_name == "nfn_gpt_native_train_linked"' in source
    assert 'return "linked";' in source
    assert "bool linked_tile_ops_requested(std::string_view path)" in source
    assert "void* open_tile_ops_library(const std::string& path, int flags, bool* linked_requested)" in source
    assert 'normalized_tile_lib_path == "linked"' in source
    assert 'normalized_tile_lib_path == "builtin"' in source
    assert 'normalized_tile_lib_path == "rtld-default"' in source
    assert 'normalized_tile_lib_path == "rtld_default"' in source
    assert "tile_handle = RTLD_DEFAULT" in source
    assert "open_tile_ops_library(tile_lib_path, RTLD_NOW | RTLD_LOCAL, &linked_tile_ops)" in source
    assert "if (tile_handle == nullptr && !linked_tile_ops)" in source
    assert "tile_handle != nullptr && !linked_tile_ops" in source
    assert "RTLD_DEFAULT-linked" in source
    assert "if (!linked_tile_ops_requested && tile_handle == nullptr)" in source
    assert "tile_handle != nullptr && !linked_tile_ops_requested" in source
    assert "tile_ops_dlopen_wall_ms" in source
    assert "tile_ops_dlopen_binding_strategy" in source

    assert "nfn_gpt_native_train_linked" in linked_build
    assert "libnfn_native_train_tile_ops.so" in linked_build
    assert "-Wl,--export-dynamic" in linked_build
    assert "-Wl,--no-as-needed" in linked_build
    assert "-Wl,-rpath" in linked_build
    assert '"${ROOT_DIR}/neuralfn/csrc/tile_cuda/kernels.cu" -nt "${TILE_OPS_LIB}"' in linked_build
    assert '"${ROOT_DIR}/neuralfn/csrc/native_train/tile_ops.cu" -nt "${TILE_OPS_LIB}"' in linked_build
    assert '"${ROOT_DIR}/neuralfn/csrc/native_train/tile_ops.h" -nt "${TILE_OPS_LIB}"' in linked_build
    assert '"${ROOT_DIR}/tools/build_native_train_tile_ops.sh" -nt "${TILE_OPS_LIB}"' in linked_build
    assert "nfn_gpt_native_train_linked" in train_gpt_source
    assert "_native_cli_uses_linked_tile_ops" in train_gpt_source
    assert '_append_value(out, "--tile-ops-lib", "linked")' in train_gpt_source
    assert "from neuralfn.native_cuda_device import" not in train_gpt_source
    assert "resolve_cuda_visible_devices_value" in train_gpt_source
    assert '_set_env_default_if_empty(env, "CUDA_VISIBLE_DEVICES", resolve_cuda_visible_devices_value("0"))' in train_gpt_source
    assert "DEFAULT_NATIVE_GPT_TRAIN_CLI_LINKED" in native_train_sdk_source
    assert "nfn_gpt_native_train_linked" in native_train_sdk_source
    assert "nfn_gpt_native_train_linked" in nfn_source
    assert "_native_gpt_cli_uses_linked_tile_ops" in nfn_source
    assert '_append_value_arg(out, "--tile-ops-lib", "linked")' in nfn_source
    assert "nfn_gpt_native_train_linked" in native_train_source
    assert "linked_local_build" in native_train_source
    assert "nfn_gpt_native_train_linked" in gpt2_evo_source
    assert "linked_build_path" in gpt2_evo_source
    assert "NATIVE_GPT_TRAIN_BIN" in train_sm120
    assert "COMPILED_SM120_LAUNCHER" in train_sm120
    assert "NFN_NATIVE_SM120_CLI" in train_sm120
    assert "NFN_SM120_USE_COMPILED_LAUNCHER" in train_sm120
    assert 'exec "${COMPILED_SM120_LAUNCHER}" "$@"' in train_sm120
    assert "select_auto_cuda_device" in train_sm120
    assert "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES" in train_sm120
    assert "NFN_SM120_CUDA_VISIBLE_DEVICES" in train_sm120
    assert "display_active,utilization.gpu" in train_sm120
    assert "select_display_disabled_cuda_device" in train_sm120_cpp
    assert "resolve_cuda_visible_devices_default" in train_sm120_cpp
    assert "NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES" in train_sm120_cpp
    assert "NFN_SM120_CUDA_VISIBLE_DEVICES" in train_sm120_cpp
    assert "NFN_NATIVE_GPT_CUDA_VISIBLE_DEVICES" in train_sm120_cpp
    assert "NFN_NATIVE_GPT_MODEL_FAMILY" in train_sm120_cpp
    assert "NFN_NATIVE_GPT_TEMPLATE_NAME" in train_sm120_cpp
    assert "NFN_NATIVE_GPT_TRAIN_BATCH_TOKENS" in train_sm120_cpp
    assert 'setenv_default_if_empty("CUDA_VISIBLE_DEVICES", resolve_cuda_visible_devices_default())' in train_sm120_cpp
    assert "build/nfn_gpt_native_train_linked" in train_sm120
    assert 'TILE_OPS_ARGS=(--tile-ops-lib linked)' in train_sm120
    assert "build_native_gpt_cli_linked.sh" in train_sm120
    assert "GPT_LINKED_CLI_OUT" in build_all
    assert "GPT_TRAIN_CLI_OUT" in build_all
    assert "build_native_gpt_cli_linked.sh" in build_all
    assert "build_train_gpt_cli.sh" in build_all
    assert "SM120_CLI_OUT" in build_all
    assert "build_train_gpt_sm120_cli.sh" in build_all
    assert "nfn_train_gpt" in train_gpt_build
    assert "train_gpt_sm120.cpp" in train_gpt_build
    assert build_all.index("build_native_train_tile_ops.sh") < build_all.index(
        "build_native_gpt_cli_linked.sh"
    )
    assert build_all.index("build_native_gpt_cli_linked.sh") < build_all.index(
        "build_train_gpt_cli.sh"
    )
    assert build_all.index("build_train_gpt_cli.sh") < build_all.index(
        "build_train_gpt_sm120_cli.sh"
    )
    assert "build_native_gpt_cli_linked.sh" in rebuild_sm120
    assert "build_train_gpt_cli.sh" in rebuild_sm120
    assert "build_train_gpt_sm120_cli.sh" in rebuild_sm120
    assert rebuild_sm120.index("build_native_gpt_cli_linked.sh") < rebuild_sm120.index(
        "build_train_gpt_cli.sh"
    )
    assert rebuild_sm120.index("build_train_gpt_cli.sh") < rebuild_sm120.index(
        "build_train_gpt_sm120_cli.sh"
    )
    assert rebuild_sm120.index("build_native_gpt_cli_linked.sh") < rebuild_sm120.index(
        "build_train_gpt_sm120_cli.sh"
    )
    assert "NFN_NATIVE_REBUILD_BINDINGS" in rebuild_sm120
    assert "build_native_gpt_binding.sh" in rebuild_sm120
    assert "build_native_gpt2_binding.sh" in rebuild_sm120
    assert "build_native_train_binding.sh" in rebuild_sm120
    assert "libnfn_native_train_tile_ops_tk.so" in rebuild_sm120
    assert "NFN_TILE_CUDA_TK_EXTRA_NVCC_FLAGS" in rebuild_sm120
    assert "-DLLMK_SM120_USE_TK_FUSED_DGELU_DINP" in tile_ops_build
    assert "-DLLMK_SM120_APPROX_DGELU_TANH=1" in tile_ops_build
    assert tile_ops_build.index("-DLLMK_SM120_USE_CUBLASLT_GEMM") < tile_ops_build.index(
        "-DLLMK_SM120_USE_TK_FUSED_DGELU_DINP"
    )
    assert "build_linear_backward_bench.sh" in rebuild_sm120
    assert "build_lm_head_backward_bench.sh" in rebuild_sm120
    assert rebuild_sm120.index("build_native_train_tile_ops.sh") < rebuild_sm120.index(
        "build_linear_backward_bench.sh"
    )
    assert rebuild_sm120.index("build_linear_backward_bench.sh") < rebuild_sm120.index(
        "build_native_missing_trainers.sh"
    )
    assert 'export NFN_NATIVE_FORCE_REBUILD="${NFN_NATIVE_FORCE_REBUILD:-1}"' in rebuild_sm120
    assert "NFN_NATIVE_GPT_FORCE_REBUILD" in build_script
    assert "NFN_NATIVE_FORCE_REBUILD" in build_script
    assert "source_newer_than_out" in build_script
    assert "TOKEN_SHARDS_HEADER" in build_script
    assert '"${SRC}" "${TOKEN_SHARDS_SRC}" -pthread -ldl -o "${OUT}"' in build_script
    assert "NFN_NATIVE_GPT_FORCE_REBUILD" in linked_build
    assert "source_newer_than_out" in linked_build
    assert '! source_newer_than_out "${TILE_OPS_LIB}"' in linked_build
    assert 'Path("tools/build_native_train_tile_ops.sh")' in no_torch_verifier
    assert 'Path("build/nfn_train_gpt")' in no_torch_verifier
    assert '"tools/build_train_gpt_cli.sh"' in no_torch_verifier
    assert '-pthread -ldl -o "${OUT}"' in linked_build
    assert "nfn_gpt_native_train_linked" in parity_bench
    assert "NFN_NATIVE_GPT_TRAIN_BIN_EXPLICIT" in parity_bench
    assert "ensure_default_native_gpt_trainer_current" in parity_bench
    assert "native_gpt_source_newer_than" in parity_bench
    assert "tile_ops_source_newer_than" in parity_bench
    assert '"$ROOT_DIR/tools/build_native_train_tile_ops.sh" -nt "$target"' in parity_bench
    assert '"$ROOT_DIR/tools/build_native_train_tile_ops.sh" -nt "$target"' in candidate_bench
    assert 'bash "$ROOT_DIR/tools/build_native_gpt_cli_linked.sh" "$NFN_NATIVE_GPT_TRAIN_BIN"' in parity_bench
    assert 'bash "$ROOT_DIR/tools/build_native_gpt_cli.sh" "$NFN_NATIVE_GPT_TRAIN_BIN"' in parity_bench
    assert 'NFN_NATIVE_TILE_OPS_ARG="linked"' in parity_bench
    assert '--tile-ops-lib "$NFN_NATIVE_TILE_OPS_ARG"' in parity_bench
    assert "build/nfn_gpt_native_train_linked --backend tile-cuda --tile-ops-lib linked" in (
        root / "docs" / "cli.md"
    ).read_text(encoding="utf-8")
    assert "TRAIN_LOOP_EVENT_TIMING=\"$(env_or_alias3 NFN_SM120_NATIVE_TRAIN_LOOP_EVENT_TIMING NFN_SM120_PARITY_TRAIN_LOOP_EVENT_TIMING NFN_SM120_TRAIN_LOOP_EVENT_TIMING 1)\"" in parity_bench
    assert "SAMPLES=\"$(env_or_alias3 NFN_SM120_NATIVE_SAMPLES NFN_SM120_PARITY_SAMPLES NFN_SM120_SAMPLES 3)\"" in parity_bench
    assert "WARMUP=\"$(env_or_alias3 NFN_SM120_NATIVE_WARMUP NFN_SM120_PARITY_WARMUP NFN_SM120_WARMUP 1)\"" in parity_bench
    assert "ENFORCE_GATE=\"$(env_or_alias3 NFN_SM120_NATIVE_ENFORCE_PARITY_GATE NFN_SM120_PARITY_ENFORCE_GATE NFN_SM120_ENFORCE_PARITY_GATE 1)\"" in parity_bench
    assert "DEFAULT_MAX_TRAIN_LOOP_RATIO=\"$(env_or_alias3 NFN_SM120_NATIVE_PARITY_MAX_TRAIN_LOOP_RATIO NFN_SM120_PARITY_MAX_TRAIN_LOOP_RATIO NFN_SM120_MAX_TRAIN_LOOP_RATIO 1.003)\"" in parity_bench
    assert "DEFAULT_MAX_STEADY_STATE_RATIO=\"$(env_or_alias3 NFN_SM120_NATIVE_PARITY_MAX_STEADY_STATE_RATIO NFN_SM120_PARITY_MAX_STEADY_STATE_RATIO NFN_SM120_MAX_STEADY_STATE_RATIO 1.003)\"" in parity_bench
    assert 'MAX_CANDIDATE_RATIO_RAW="${gate_stat_prefix}train_loop_wall_ms_per_step=${DEFAULT_MAX_TRAIN_LOOP_RATIO}"' in parity_bench
    assert 'MAX_CANDIDATE_RATIO_RAW+=" ${gate_stat_prefix}train_loop_cuda_event_steady_state_wall_ms_per_step=${DEFAULT_MAX_STEADY_STATE_RATIO}"' in parity_bench
    assert 'paired_args+=(--candidate-env "NFN_NATIVE_GPT_TRAIN_LOOP_EVENT_TIMING=1")' in parity_bench
    assert "CANDIDATE_PROFILE_RAW=\"$(env_or_alias4 NFN_SM120_NATIVE_CANDIDATE_PROFILE NFN_SM120_NATIVE_PARITY_PROFILE NFN_SM120_PARITY_CANDIDATE_PROFILE NFN_SM120_PARITY_PROFILE \"\")\"" in parity_bench
    assert "NFN_SM120_NATIVE_CANDIDATE_PROFILE/NFN_SM120_PARITY_CANDIDATE_PROFILE/NFN_SM120_PARITY_PROFILE is not supported" in parity_bench
    assert (
        "ALLOW_STALE_GPU_UTILIZATION_WITHOUT_COMPUTE=\"$(env_or_alias3 NFN_SM120_NATIVE_ALLOW_STALE_GPU_UTILIZATION_WITHOUT_COMPUTE NFN_SM120_PARITY_ALLOW_STALE_GPU_UTILIZATION_WITHOUT_COMPUTE NFN_SM120_ALLOW_STALE_GPU_UTILIZATION_WITHOUT_COMPUTE 1)\""
        in parity_bench
    )
    assert "--allow-stale-selected-gpu-utilization-without-compute-processes" in parity_bench
    assert "Refusing to run because a parity profile would otherwise be ignored" in parity_bench
    assert "def bool_value(value)" in parity_bench
    assert 'lowered in {"1", "true", "yes", "on"}' in parity_bench
    assert "def candidate_profile_path()" in parity_bench
    assert 'glob.glob(os.path.join(profile_dir, "candidate_*.json"))' in parity_bench
    assert "Candidate native profile sidecar:" in parity_bench
    assert "Top candidate setup timings:" in parity_bench
    assert "Top candidate stage timings:" in parity_bench
    assert "float_arena_request_stats" in parity_bench
    assert "uint16_arena_request_stats" in parity_bench
    assert "top_families" in parity_bench
    assert "total_allocated_bytes" in parity_bench
    assert "float_arena_allocated_bytes" in paired_speed
    assert "uint16_arena_allocated_bytes" in paired_speed
    assert "transformer_arena_allocated_bytes" in paired_speed
    assert "activation_storage_bytes" in paired_speed
    assert "lm_head_bf16_logit_bytes" in paired_speed
    assert "nfn_gpt_native_train_linked" in candidate_bench
    assert "NFN_SM120_NATIVE_MAX_CANDIDATE_REFERENCE_RATIO" in candidate_bench
    assert "NFN_SM120_NATIVE_MIN_CANDIDATE_REFERENCE_RATIO" in candidate_bench
    assert "NFN_NATIVE_GPT_TRAIN_BIN_EXPLICIT" in candidate_bench
    assert "NFN_SM120_NATIVE_CANDIDATE_TRAIN_BIN_EXPLICIT" in candidate_bench
    assert "ensure_native_gpt_trainer_current" in candidate_bench
    assert 'ensure_native_gpt_trainer_current "$NFN_NATIVE_GPT_TRAIN_BIN" "$NFN_NATIVE_GPT_TRAIN_BIN_EXPLICIT"' in candidate_bench
    assert 'ensure_native_gpt_trainer_current "$NFN_SM120_NATIVE_CANDIDATE_TRAIN_BIN" "$NFN_SM120_NATIVE_CANDIDATE_TRAIN_BIN_EXPLICIT"' in candidate_bench
    assert 'bash "$ROOT_DIR/tools/build_native_gpt_cli_linked.sh" "$train_bin"' in candidate_bench
    assert 'bash "$ROOT_DIR/tools/build_native_gpt_cli.sh" "$train_bin"' in candidate_bench
    assert "--max-candidate-reference-ratio" in candidate_bench
    assert "--min-candidate-reference-ratio" in candidate_bench
    assert "--max-candidate-reference-ratio" in paired_speed
    assert "--min-candidate-reference-ratio" in paired_speed
    assert "candidate_reference_metric_ratio_gates" in paired_speed
    assert 'ratio_key="candidate_over_reference_native_metrics"' in paired_speed
    assert "DEFAULT_VS_LEGACY_PROFILE=0" in candidate_bench
    assert "candidate_gate_scope=default-vs-legacy" in candidate_bench
    assert "filter_generated_candidate_ratio_gates" in candidate_bench
    assert 'metric" == "train_loop_cuda_event_steady_state_wall_ms_per_step"' in candidate_bench
    assert '"$metric" == stage.*' in candidate_bench
    assert "tile_ops_arg_for" in candidate_bench
    assert 'NFN_SM120_NATIVE_CANDIDATE_TILE_OPS_LIB_EXPLICIT="generated"' in candidate_bench
    assert '--tile-ops-lib "$NFN_NATIVE_TILE_OPS_ARG"' in candidate_bench
    assert '--tile-ops-lib "$NFN_SM120_NATIVE_CANDIDATE_TILE_OPS_ARG"' in candidate_bench
    assert "lm_head_tk_dinput_32768" in candidate_bench
    assert "routed LM-head dHidden through TK dInput" in candidate_bench
    assert "stage.lm_head_backward.dhidden.total_ms to 1.132973x" in candidate_bench
    assert "lm_head_cublaslt_dhidden_32768" in candidate_bench
    assert "moved 48 LM-head dHidden calls from BF16 GEMMEx to cuBLASLt" in candidate_bench
    assert "stage.block_backward.total_ms to 1.001504x" in candidate_bench
    assert "lm_head_dhidden_fast16bf_32768" in candidate_bench
    assert "stage.lm_head_backward.total_ms to 1.004489x" in candidate_bench
    assert "stage.lm_head_backward.dhidden.total_ms stayed flat at 1.000265x" in candidate_bench
    assert "lm_head_tk_dweight_32768" in candidate_bench
    assert "train_loop_wall_ms_per_step to 1.052253x" in candidate_bench
    assert "stage.lm_head_backward.dweight.total_ms to 1.337552x" in candidate_bench
    assert "lm_head_tk_dweight_49152" in candidate_bench
    assert "NFN_NATIVE_LINEAR_TK_DWEIGHT_ENABLE_SHAPE=768,50304,49152,N,T" in candidate_bench
    assert "qkv_concurrent_dinput_dweight" in candidate_bench
    assert "block_backward_qkv_concurrent_dinput_dweight_count from 0 to 288" in candidate_bench
    assert "block_backward_qkv_dinput_before_dweight_count from 288 to 0" in candidate_bench
    assert "train_loop_wall_ms_per_step regressed to 1.001725x" in candidate_bench
    assert "candidate-over-llm.kittens train_loop_wall_ms_per_step stayed at 1.003575x" in candidate_bench
    assert "lm_head_concurrent_dhidden_dweight" in candidate_bench
    assert "CUDA 13.3 RTX 5090 3-sample same-script confirmation" in candidate_bench
    assert "NFN_NATIVE_GPT_LM_HEAD_CONCURRENT_DHIDDEN_DWEIGHT=1" in candidate_bench
    assert "NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_BACKWARD=0 NFN_NATIVE_GPT_LM_HEAD_CONCURRENT_DHIDDEN_DWEIGHT=1" in candidate_bench
    assert "lm_head_dweight_before_dhidden" in candidate_bench
    assert "NFN_NATIVE_GPT_LM_HEAD_DWEIGHT_BEFORE_DHIDDEN=1" in candidate_bench
    assert "NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_BACKWARD=0 NFN_NATIVE_GPT_LM_HEAD_DWEIGHT_BEFORE_DHIDDEN=1" in candidate_bench
    assert "NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_BACKWARD=0 NFN_NATIVE_GPT_LM_HEAD_PIPELINE_CHUNKS=1" in candidate_bench
    assert "cuda_device_max_connections_1" in candidate_bench
    assert "This profile is a no-op in the SM120 paired wrapper" in candidate_bench
    assert "CUDA_DEVICE_MAX_CONNECTIONS already defaults to 1" in candidate_bench
    assert "bf16_workspace_prewarm" in candidate_bench
    assert "NFN_NATIVE_GPT_PREWARM_BF16_WORKSPACE=1" in candidate_bench
    assert "combined_device_arena" in candidate_bench
    assert "setup_wall_ms to 1.031475x" in candidate_bench
    assert "setup.uint16_arena_materialize.total_ms to 2.339592x" in candidate_bench
    assert "setup.token_weight_init.total_ms to 1.289567x" in candidate_bench
    assert "lm_head_cooperative_backward" in candidate_bench
    assert "activated the cooperative LM-head CUDA Graph path" in candidate_bench
    assert "steady-state CUDA-event timing to 1.005839x" in candidate_bench
    assert "candidate-over-llm.kittens wall to 1.018312x" in candidate_bench
    assert "token_weight_threaded" in candidate_bench
    assert "setup.token_weight_init.total_ms to 1.025016x" in candidate_bench
    assert "adamw_token_shadow_refresh" in candidate_bench
    assert "NFN_NATIVE_GPT_FUSE_TOKEN_WEIGHT_BF16_ADAMW_REFRESH=0" in candidate_bench
    assert "NFN_NATIVE_GPT_FUSE_TOKEN_WEIGHT_BF16_ADAMW_REFRESH=1" in candidate_bench
    assert "older two-launch token-shadow refresh path" in candidate_bench
    assert "stage.adamw_update.total_ms=1.004" in candidate_bench
    assert "token_weight_bf16_fused_adamw_refresh_count" in paired_speed
    assert "token_weight_bf16_adamw_refresh_fusion_enabled" in paired_speed
    assert "adamw_bf16_shadow_refresh_strategy" in paired_speed
    assert "token_weight_vector4_strided" in candidate_bench
    assert "NFN_NATIVE_GPT_TOKEN_WEIGHT_VECTOR4_STRIDED_INIT=0" in candidate_bench
    assert "NFN_NATIVE_GPT_TOKEN_WEIGHT_VECTOR4_STRIDED_INIT=1" in candidate_bench
    assert "1.012563x" not in candidate_bench
    assert "token_weight_bf16_pattern" in candidate_bench
    assert "setup_wall_ms regressed to 1.005840x" in candidate_bench
    assert "setup.token_weight_init.total_ms to 1.015463x" in candidate_bench
    assert "setup.uint16_arena_materialize.total_ms stayed noise-flat at 0.998215x" in candidate_bench
    assert "token_weight_padded_init" in candidate_bench
    assert "NFN_NATIVE_GPT_FUSE_TOKEN_WEIGHT_PADDED_INIT=1" in candidate_bench
    assert "moving token_weight_bf16_padding_memset_count from 1 to 0" in candidate_bench
    assert "train_loop_wall_ms_per_step to 0.999280x" in candidate_bench
    assert "candidate/reference train_loop_wall_ms_per_step=1.000957x" in candidate_bench
    assert "Keep the route opt-in" in candidate_bench
    assert "host_descriptor_reserve" in candidate_bench
    assert "NFN_NATIVE_GPT_HOST_DESCRIPTOR_RESERVE=1" in candidate_bench
    assert "setup_wall_ms to 1.016598x" in candidate_bench
    assert "keep it diagnostic-only" in candidate_bench
    assert "host_descriptor_reserve_enabled" in source
    assert "host_descriptor_reserve_count" in source
    assert "NFN_NATIVE_GPT_HOST_DESCRIPTOR_RESERVE" in source
    assert "false);" in source[source.index("NFN_NATIVE_GPT_HOST_DESCRIPTOR_RESERVE") :]
    assert "token_weight_fast_int32" in candidate_bench
    assert "setup_wall_ms regressed to 1.014148x" in candidate_bench
    assert "setup.token_weight_init.total_ms to 0.960426x" in candidate_bench
    assert "token_weight_two_pass_bf16" in candidate_bench
    assert "setup_wall_ms regressed to 1.005287x" in candidate_bench
    assert "setup.token_weight_init.total_ms to 1.005229x" in candidate_bench
    assert "mlp_fc_dinput_before_dweight" in candidate_bench
    assert "block_backward_mlp_fc_dinput_before_dweight_count from 0 to 288" in candidate_bench
    assert "post-reinstall recheck proved the route" in candidate_bench
    assert "steady-state CUDA-event timing regressed to 1.001167x" in candidate_bench
    assert "The native default is restored to dWeight+bias before dInput" in candidate_bench
    assert '"mlp_proj_concurrent_dinput_dweight"|"mlp-proj-concurrent-dinput-dweight"' in candidate_bench
    assert "NFN_NATIVE_GPT_BLOCK_MLP_PROJ_CONCURRENT_DINPUT_DWEIGHT=1" in candidate_bench
    assert "block_backward_mlp_proj_concurrent_dinput_dweight_count from 0 to 288" in candidate_bench
    assert "stage.block_backward.mlp_proj.total_ms to 1.025216x" in candidate_bench
    assert "stage.block_backward.mlp_proj.total_ms=1.000" in candidate_bench
    assert "ce_bf16_threads_512" in candidate_bench
    assert "lm_head_ce_bf16_threads_per_row from 1024 to 512" in candidate_bench
    assert "stage.lm_head_backward.ce.total_ms to 1.430612x" in candidate_bench
    assert "linked_startup" in candidate_bench
    assert "linked_tile_ops" in candidate_bench
    assert "NFN_SM120_NATIVE_BASELINE_TRAIN_BIN" in candidate_bench
    assert "NFN_SM120_NATIVE_LINKED_STARTUP_CANDIDATE_BIN" in candidate_bench
    assert "FORCE_DISABLE_ROUTE_CHANGE=1" in candidate_bench
    assert 'if [[ "$FORCE_DISABLE_ROUTE_CHANGE" == "1" ]]; then' in candidate_bench
    assert "REQUIRE_NATIVE_ROUTE_CHANGE=0" in candidate_bench
    assert 'Path("build/nfn_gpt_native_train_linked")' in no_torch_verifier
    assert 'Path("build/linear_backward_bench")' in no_torch_verifier
    assert "--rebuild-stale" in no_torch_verifier
    assert "ARTIFACT_REBUILD_COMMANDS" in no_torch_verifier
    assert '"tools/build_train_gpt_sm120_cli.sh"' in no_torch_verifier
    assert '"tools/build_native_gpt2_cli.sh"' in no_torch_verifier
    assert '"tools/build_native_gpt_binding.sh"' in no_torch_verifier
    assert '"tools/build_linear_backward_bench.sh"' in no_torch_verifier
    assert '"tools/build_lm_head_backward_bench.sh"' in no_torch_verifier
    assert "def rebuild_stale_artifact" in no_torch_verifier


def test_native_gpt_transformer_lm_smoke_uses_linked_tile_ops_loader() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root
        / "neuralfn"
        / "csrc"
        / "native_gpt2"
        / "nfn_gpt2_native_train.cpp"
    ).read_text(encoding="utf-8")
    body = source.split("int print_transformer_lm_step_smoke_json(", 1)[1].split(
        "int print_norm_residual_step_smoke_json(", 1
    )[0]

    assert (
        "open_tile_ops_library(tile_lib_path, RTLD_NOW | RTLD_LOCAL, &linked_tile_ops)"
        in body
    )
    assert "if (tile_handle == nullptr && !linked_tile_ops)" in body
    assert "if (tile_handle != nullptr && !linked_tile_ops)" in body
    assert "dlopen(tile_lib_path.c_str(), RTLD_NOW | RTLD_LOCAL)" not in body


def test_native_gpt_cli_supports_json_output_file_aliases() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "neuralfn"
        / "csrc"
        / "native_gpt2"
        / "nfn_gpt2_native_train.cpp"
    ).read_text(encoding="utf-8")

    assert "ScopedStdoutRedirect" in source
    assert "--json-out PATH" in source
    assert "--profile-json PATH" in source
    assert '"--json-out"' in source
    assert '"--profile-json"' in source
    assert '"--stage-profile-json"' in source
    assert "cfg.json_out_path" in source


def test_native_gpt_transformer_lm_defaults_to_bf16_attention_grad_out_handoff() -> None:
    root = Path(__file__).resolve().parents[1]
    gpt_source = (root / "neuralfn" / "csrc" / "native_gpt2" / "nfn_gpt2_native_train.cpp").read_text(
        encoding="utf-8"
    )
    tile_header = (root / "neuralfn" / "csrc" / "native_train" / "tile_ops.h").read_text(
        encoding="utf-8"
    )
    tile_source = (root / "neuralfn" / "csrc" / "native_train" / "tile_ops.cu").read_text(
        encoding="utf-8"
    )
    kernels_source = (root / "neuralfn" / "csrc" / "tile_cuda" / "kernels.cu").read_text(
        encoding="utf-8"
    )

    assert "NFN_NATIVE_GPT_BF16_ATTENTION_GRAD_OUT" in gpt_source
    assert "NFN_NATIVE_GPT2_BF16_ATTENTION_GRAD_OUT" in gpt_source
    assert 'NFN_NATIVE_GPT2_BF16_ATTENTION_GRAD_OUT"}),\n            true)' in gpt_source
    assert "attention_backward_bf16_grad_out_handoff_enabled" in gpt_source
    assert "attention_backward_grad_out_dtype" in gpt_source
    assert "attention_backward_bf16_grad_out_scratch_elements" in gpt_source
    assert "linear_backward_input_weight_bf16_to_bf16_bits" in gpt_source
    assert "packed_attention_backward_to_qkv_bf16_bits_from_bf16_grad" in gpt_source
    assert "tk-sm120-packed-qkv-bf16-grad-out-direct-bf16-qkv-handoff" in gpt_source

    assert "packed_attention_dprep_bf16_grad_kernel" in kernels_source
    assert "linear_backward_input_weight_bf16_bits_to_bf16_bits_float32_kernel" in kernels_source
    assert "launch_linear_backward_input_weight_bf16_to_bf16_bits_float32" in kernels_source
    assert "launch_tk_attention_packed_qkv_backward_to_qkv_bf16_bits_from_bf16_grad_bits" in kernels_source
    assert "launch_scaled_dot_product_attention_packed_qkv_backward_to_qkv_bf16_bits_from_bf16_merged_grad_float32" in kernels_source
    assert "launch_scaled_dot_product_attention_packed_qkv_backward_to_qkv_bf16_bits_from_saved_lse_bf16_from_bf16_merged_grad_float32" in kernels_source

    assert "nfn_native_tile_linear_backward_input_weight_bf16_to_bf16_bits_float32" in tile_header
    assert "nfn_native_tile_linear_backward_input_weight_bf16_to_bf16_bits_float32" in tile_source
    assert "nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_bf16_bits_from_bf16_merged_grad_float32" in tile_header
    assert "nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_bf16_bits_from_bf16_merged_grad_float32" in tile_source
    assert "nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_bf16_bits_from_saved_lse_bf16_from_bf16_merged_grad_float32" in tile_header
    assert "nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_bf16_bits_from_saved_lse_bf16_from_bf16_merged_grad_float32" in tile_source


def test_native_gpt_transformer_lm_exposes_opt_in_lm_head_chunk_pipeline() -> None:
    root = Path(__file__).resolve().parents[1]
    gpt_source = (root / "neuralfn" / "csrc" / "native_gpt2" / "nfn_gpt2_native_train.cpp").read_text(
        encoding="utf-8"
    )

    assert "NFN_NATIVE_GPT_LM_HEAD_PIPELINE_CHUNKS" in gpt_source
    assert "NFN_NATIVE_GPT2_LM_HEAD_PIPELINE_CHUNKS" in gpt_source
    assert "lm_head_pipeline_chunks_requested" in gpt_source
    assert "lm_head_pipeline_chunks_enabled" in gpt_source
    assert "lm_head_pipeline_logit_buffer_count" in gpt_source
    assert "lm_head_pipeline_extra_bf16_logit_bytes" in gpt_source
    assert "lm_head_pipeline_slot_event_wait_count" in gpt_source
    assert "lm_head_pipeline_done_event_record_count" in gpt_source
    assert "lm_head_backward.pipeline_buffer_wait" in gpt_source
    assert "lm_head_backward.pipeline_queue" in gpt_source
    assert "lm_head_backward.pipeline_final_wait" in gpt_source
    assert "cudaEventRecord lm_head_pipeline_dhidden_done" in gpt_source
    assert "cudaEventRecord lm_head_pipeline_dweight_done" in gpt_source
    assert "lm_head.pipeline.wait_dhidden_slot" in gpt_source
    assert "lm_head.pipeline.wait_dweight_slot" in gpt_source
    assert "lm_head.pipeline.wait_dhidden_buffer" not in gpt_source
    assert "lm_head.pipeline.wait_dweight_buffer" not in gpt_source
    assert "double-buffered-logits-ce-default-stream-side-stream-dhidden-ordered-dweight-slot-events" in gpt_source


def test_native_tile_linear_exposes_cublaslt_grouped_layout_probe() -> None:
    root = Path(__file__).resolve().parents[1]
    gpt_source = (root / "neuralfn" / "csrc" / "native_gpt2" / "nfn_gpt2_native_train.cpp").read_text(
        encoding="utf-8"
    )
    tile_header = (root / "neuralfn" / "csrc" / "native_train" / "tile_ops.h").read_text(
        encoding="utf-8"
    )
    tile_source = (root / "neuralfn" / "csrc" / "native_train" / "tile_ops.cu").read_text(
        encoding="utf-8"
    )
    kernels_source = (root / "neuralfn" / "csrc" / "tile_cuda" / "kernels.cu").read_text(
        encoding="utf-8"
    )
    speed_tool = (root / "tools" / "paired_kernel_speed.py").read_text(encoding="utf-8")

    assert "nfn_native_tile_trainer_linear_cublaslt_grouped_layout_probe_status" in tile_header
    assert "nfn_native_tile_trainer_linear_cublaslt_grouped_layout_probe_status" in tile_source
    assert "trainer_linear_cublaslt_grouped_layout_probe_status" in kernels_source
    assert "cublasLtGroupedMatrixLayoutCreate" in kernels_source
    assert "nfn_native_tile_trainer_linear_cublaslt_grouped_matmul_probe_status" in tile_header
    assert "nfn_native_tile_trainer_linear_cublaslt_grouped_matmul_probe_status" in tile_source
    assert "trainer_linear_cublaslt_grouped_matmul_probe_status" in kernels_source
    assert "NFN_NATIVE_GPT_PROBE_CUBLASLT_GROUPED_MATMUL" in gpt_source
    assert "linear_cublaslt_grouped_matmul_probe_status" in gpt_source
    assert "nfn_native_tile_trainer_linear_cublas_grouped_bf16_gemm_probe_status" in tile_header
    assert "nfn_native_tile_trainer_linear_cublas_grouped_bf16_gemm_probe_status" in tile_source
    assert "trainer_linear_cublas_grouped_bf16_gemm_probe_status" in kernels_source
    assert "cublasGemmGroupedBatchedEx" in kernels_source
    assert "nfn_native_tile_trainer_linear_cublas_prewarm" in tile_header
    assert "nfn_native_tile_trainer_linear_cublas_prewarm" in tile_source
    assert "trainer_linear_cublas_prewarm" in kernels_source
    assert "nfn_native_tile_trainer_linear_bf16_workspace_prewarm" in tile_header
    assert "nfn_native_tile_trainer_linear_bf16_workspace_prewarm" in tile_source
    assert "trainer_linear_bf16_workspace_prewarm" in kernels_source
    assert "nfn_native_tile_trainer_linear_cublaslt_prewarm_bf16_plan" in tile_header
    assert "nfn_native_tile_trainer_linear_cublaslt_prewarm_bf16_plan" in tile_source
    assert "trainer_linear_cublaslt_prewarm_bf16_plan" in kernels_source
    assert "nfn_native_tile_trainer_linear_cublaslt_plan_cache_count" in tile_header
    assert "nfn_native_tile_trainer_linear_cublaslt_plan_cache_count" in tile_source
    assert "trainer_linear_cublaslt_plan_cache_count" in kernels_source
    assert "nfn_native_tile_trainer_linear_cublaslt_plan_cache_entry" in tile_header
    assert "nfn_native_tile_trainer_linear_cublaslt_plan_cache_entry" in tile_source
    assert "trainer_linear_cublaslt_plan_cache_entry" in kernels_source
    assert "NFN_NATIVE_GPT_PREWARM_CUBLASLT_PLANS" in gpt_source
    assert "NFN_NATIVE_GPT2_PREWARM_CUBLASLT_PLANS" in gpt_source
    assert "NFN_TILE_CUDA_LINEAR_CUBLASLT_PREWARM" in gpt_source
    assert "NFN_NATIVE_GPT_PREWARM_CUBLASLT_PLAN_MODE" in gpt_source
    assert "NFN_NATIVE_GPT2_PREWARM_CUBLASLT_PLAN_MODE" in gpt_source
    assert "NFN_TILE_CUDA_LINEAR_CUBLASLT_PREWARM_MODE" in gpt_source
    assert 'linear_cublaslt_plan_prewarm_mode = "all"' in gpt_source
    assert 'linear_cublaslt_plan_prewarm_mode == "block_only"' in gpt_source
    assert 'linear_cublaslt_plan_prewarm_mode == "lm_head_only"' in gpt_source
    assert '"NFN_TILE_CUDA_LINEAR_CUBLASLT_PREWARM"}),\n            false)' in gpt_source
    assert "linear_cublaslt_grouped_layout_probe_available" in gpt_source
    assert "linear_cublaslt_grouped_layout_probe_requested" in gpt_source
    assert "NFN_NATIVE_GPT_PROBE_CUBLASLT_GROUPED_LAYOUT" in gpt_source
    assert "linear_cublaslt_grouped_layout_probe_status" in gpt_source
    assert "linear_cublaslt_grouped_layout_supported" in gpt_source
    assert "linear_cublaslt_grouped_matmul_probe_available" in gpt_source
    assert "linear_cublaslt_grouped_matmul_probe_requested" in gpt_source
    assert "linear_cublaslt_grouped_matmul_supported" in gpt_source
    assert "linear_cublas_grouped_bf16_gemm_probe_available" in gpt_source
    assert "linear_cublas_grouped_bf16_gemm_probe_requested" in gpt_source
    assert "linear_cublas_grouped_bf16_gemm_probe_status" in gpt_source
    assert "linear_cublas_grouped_bf16_gemm_supported" in gpt_source
    assert "requested cuBLAS grouped BF16 GEMM probe failed with status" in gpt_source
    assert "requested cuBLAS grouped BF16 GEMM probe is unavailable in the Tile ops library" in gpt_source
    assert "NFN_NATIVE_GPT_PREWARM_CUBLAS_HANDLE" in gpt_source
    assert "NFN_NATIVE_GPT2_PREWARM_CUBLAS_HANDLE" in gpt_source
    assert "NFN_TILE_CUDA_LINEAR_CUBLAS_PREWARM" in gpt_source
    assert (
        '"NFN_TILE_CUDA_LINEAR_CUBLAS_PREWARM"}),\n'
        "            true)"
    ) in gpt_source
    assert "linear_cublas_handle_prewarm_available" in gpt_source
    assert "linear_cublas_handle_prewarm_enabled" in gpt_source
    assert "linear_cublas_handle_prewarm_success_count" in gpt_source
    assert "setup.cublas_handle_prewarm" in gpt_source
    assert "NFN_NATIVE_GPT_PREWARM_BF16_WORKSPACE" in gpt_source
    assert "NFN_NATIVE_GPT2_PREWARM_BF16_WORKSPACE" in gpt_source
    assert "NFN_TILE_CUDA_LINEAR_BF16_WORKSPACE_PREWARM" in gpt_source
    assert (
        '"NFN_TILE_CUDA_LINEAR_BF16_WORKSPACE_PREWARM"}),\n'
        "            true)"
    ) in gpt_source
    assert "linear_bf16_workspace_prewarm_available" in gpt_source
    assert "linear_bf16_workspace_prewarm_enabled" in gpt_source
    assert "linear_bf16_workspace_prewarm_success_count" in gpt_source
    assert "setup.linear_bf16_workspace_prewarm" in gpt_source
    assert "NFN_NATIVE_GPT_PREWARM_TK_QKV_FORWARD" in gpt_source
    assert "NFN_NATIVE_GPT_PREWARM_TK_QKV_FORWARD_ROWS" in gpt_source
    assert "setup.tk_qkv_forward_prewarm" in gpt_source
    assert "NFN_NATIVE_GPT_FAST_STARTUP" in gpt_source
    assert "NFN_NATIVE_GPT2_FAST_STARTUP" in gpt_source
    assert "NFN_TILE_CUDA_FAST_STARTUP" in gpt_source
    assert "--fast-startup" in gpt_source
    assert "--native-cuda-fast-startup" in gpt_source
    assert "native_fast_startup_requested" in gpt_source
    assert "native_fast_startup_prewarm_policy" in gpt_source
    assert "cfg.fast_startup ||" in gpt_source
    assert "!native_fast_startup_requested && !cfg.startup_only" in gpt_source
    assert "cfg.startup_only\n            ? 0\n            : std::min<std::int64_t>(" in gpt_source
    assert "startup-only-skip-throughput-prewarms-by-default" in gpt_source
    assert (
        'linear_tk_qkv_first_use_prewarm_env,\n'
        "            native_fast_startup_prewarm_default)"
    ) in gpt_source
    assert "linear_tk_qkv_first_use_prewarm_requested_rows" in gpt_source
    assert "linear_tk_qkv_first_use_prewarm_effective_rows" in gpt_source
    assert "linear_tk_qkv_first_use_prewarm_success_count" in gpt_source
    assert "linear_cublaslt_plan_prewarm_available" in gpt_source
    assert "linear_cublaslt_plan_prewarm_mode" in gpt_source
    assert "linear_cublaslt_plan_prewarm_attempted_count" in gpt_source
    assert "linear_cublaslt_plan_prewarm_skipped_count" in gpt_source
    assert "linear_cublaslt_plan_prewarm_success_count" in gpt_source
    assert "linear_cublaslt_plan_cache_available" in gpt_source
    assert "linear_cublaslt_plan_cache_count" in gpt_source
    assert "linear_cublaslt_plan_cache" in gpt_source
    assert "setup.cublaslt_plan_prewarm" in gpt_source
    assert "linear_cublaslt_grouped_layout_probe_status" in speed_tool
    assert "linear_cublaslt_grouped_layout_supported" in speed_tool
    assert '"native_fast_startup_requested"' in speed_tool
    assert '"native_fast_startup_prewarm_policy"' in speed_tool
    assert "linear_cublaslt_grouped_matmul_probe_requested" in speed_tool
    assert "linear_cublaslt_grouped_matmul_probe_status" in speed_tool
    assert "linear_cublaslt_grouped_matmul_supported" in speed_tool
    assert "linear_cublas_grouped_bf16_gemm_probe_status" in speed_tool
    assert "linear_cublas_grouped_bf16_gemm_probe_requested" in speed_tool
    assert "linear_cublas_grouped_bf16_gemm_supported" in speed_tool
    assert "linear_cublas_handle_prewarm_enabled" in speed_tool
    assert "linear_cublas_handle_prewarm_requested" in speed_tool
    assert "linear_cublas_handle_prewarm_success_count" in speed_tool
    assert "linear_cublas_handle_prewarm_failure_count" in speed_tool
    assert "linear_bf16_workspace_prewarm_enabled" in speed_tool
    assert "linear_bf16_workspace_prewarm_requested" in speed_tool
    assert "linear_bf16_workspace_prewarm_success_count" in speed_tool
    assert "linear_bf16_workspace_prewarm_failure_count" in speed_tool
    assert '"graph_editor_tensor_flow"' in speed_tool
    assert '"torch_required"' in speed_tool
    assert '"train_loss_host_d2h_count"' in speed_tool
    assert "native_runtime_contract_gate" in speed_tool
    assert "candidate native training must report graph_editor_tensor_flow=false" in speed_tool
    assert "and train_loss_host_d2h_count=0" in speed_tool
    assert "linear_tk_qkv_first_use_prewarm_requested_count" in speed_tool
    assert "linear_tk_qkv_first_use_prewarm_requested_rows" in speed_tool
    assert "linear_tk_qkv_first_use_prewarm_effective_rows" in speed_tool
    assert "linear_tk_qkv_first_use_prewarm_success_count" in speed_tool
    assert "stored_packed_attention_lse_enabled" in speed_tool
    assert "stored_packed_attention_lse_elements" in speed_tool
    assert "stored_packed_attention_lse_bytes" in speed_tool
    assert "lm_head_classifier_chunk_launch_count" in speed_tool
    assert "lm_head_classifier_last_row_stride" in speed_tool
    assert "lm_head_classifier_ce_no_loss_requested" in speed_tool
    assert "lm_head_classifier_ce_no_loss_enabled" in speed_tool
    assert "lm_head_classifier_no_loss_chunk_count" in speed_tool
    assert "lm_head_ce_no_loss_default_specialized_requested" in speed_tool
    assert "lm_head_ce_no_loss_default_specialized_enabled" in speed_tool
    assert "lm_head_ce_no_loss_llmk_style_specialized_requested" in speed_tool
    assert "lm_head_ce_no_loss_llmk_style_specialized_enabled" in speed_tool
    assert "lm_head_ce_row_loss_reduction_enabled" in speed_tool
    assert "lm_head_ce_row_loss_sum_accumulate_requested" in speed_tool
    assert "lm_head_ce_row_loss_sum_accumulate_enabled" in speed_tool
    assert "lm_head_ce_llmk_style_specialized_requested" in speed_tool
    assert "lm_head_ce_llmk_style_specialized_enabled" in speed_tool
    assert "lm_head_ce_loss_bins_default_specialized_requested" in speed_tool
    assert "lm_head_ce_loss_bins_default_specialized_enabled" in speed_tool


def test_native_gpt2_exposes_lm_head_last_dweight_overlap_candidate() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "neuralfn/csrc/native_gpt2/nfn_gpt2_native_train.cpp").read_text(
        encoding="utf-8"
    )
    candidate_script = (root / "tools/bench_native_gpt_sm120_candidate.sh").read_text(
        encoding="utf-8"
    )
    lm_head_bench_script = (root / "tools/bench_lm_head_backward_candidate.sh").read_text(
        encoding="utf-8"
    )

    assert "NFN_NATIVE_GPT_LM_HEAD_OVERLAP_LAST_DWEIGHT" in source
    assert "lm_head_overlap_last_dweight_requested" in source
    assert "lm_head_overlap_last_dweight_available" in source
    assert "lm_head_overlap_last_dweight_enabled" in source
    assert "lm_head_overlap_last_dweight_queue_count" in source
    assert "lm_head_overlap_last_dweight_sync_count" in source
    assert "lm_head_side_stream_count" in source
    assert "lm_head_dhidden_stream_enabled" in source
    assert "lm_head_dweight_stream_enabled" in source
    assert "last-processed-row-chunk-dweight-side-stream-overlaps-final-norm-block-backward" in source
    assert "lm_head_backward.last_dweight_overlap_queue" in source
    assert "lm_head_backward.last_dweight_overlap_final_wait" in source
    assert "lm_head_overlap_last_dweight" in candidate_script
    assert "REJECTED_CANDIDATE_PROFILE=\"$CANDIDATE_PROFILE\"" in candidate_script
    assert "regressed train_loop_wall_ms_per_step to 1.020764x" in candidate_script
    assert "NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_BACKWARD=0" in candidate_script
    assert "NFN_NATIVE_GPT_LM_HEAD_OVERLAP_LAST_DWEIGHT=1" in candidate_script
    speed_tool = (root / "tools/paired_kernel_speed.py").read_text(encoding="utf-8")
    assert "lm_head_overlap_last_dweight_queue_count" in speed_tool
    assert "lm_head_overlap_last_dweight_sync_count" in speed_tool
    assert "lm_head_side_stream_count" in speed_tool
    assert "snapshot_selected_gpu_load_json" in lm_head_bench_script
    assert "merge_gpu_load_context_json" in lm_head_bench_script
    assert "NFN_LM_HEAD_BACKWARD_REQUIRE_IDLE_SELECTED_GPU" in lm_head_bench_script
    assert "NFN_LM_HEAD_BACKWARD_MAX_SELECTED_GPU_UTILIZATION_PCT" in lm_head_bench_script
    assert "NFN_LM_HEAD_BACKWARD_GPU_BENCHMARK_LOCK" in lm_head_bench_script
    assert "nfn_lm_head_backward_gpu_${safe_gpu}.lock" in lm_head_bench_script
    assert "validate_selected_gpu_idle_snapshot" in lm_head_bench_script
    assert "require_selected_gpu_idle" in lm_head_bench_script
    assert "selected GPU {selected} has {compute_count} active compute process(es)" in lm_head_bench_script
    assert '"gpu_load_context"' in lm_head_bench_script
    assert '"compute_process_count"' in lm_head_bench_script
    assert "GPU_LOAD_BEFORE" in lm_head_bench_script
    assert "GPU_LOAD_AFTER" in lm_head_bench_script
    assert "BENCH_STDOUT" in lm_head_bench_script


def test_build_native_gpt2_run_config_matches_sm120_cli_shape(tmp_path: Path) -> None:
    dataset_path, meta = _write_raw_text_dataset(tmp_path)

    cfg, cached_meta = build_native_gpt2_run_config(
        dataset_name="tiny",
        dataset_path=dataset_path,
        dataset_meta=meta,
        encoding_name="gpt2",
        executable="/opt/nfn/train_gpt2cu",
        output_dir=tmp_path / "log124M" / "5090_S",
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
        activation="sd_prelu",
    )

    argv = cfg.argv()
    assert cached_meta["token_cache_format"] == "raw_text_uint16_shards"
    assert cfg.lm_head_row_chunk_size == 32768
    assert argv[:3] == ["/opt/nfn/train_gpt2cu", "-i", str(dataset_path / "fineweb_train_000000.bin")]
    assert argv[argv.index("-j") + 1] == str(dataset_path / "fineweb_val_000000.bin")
    assert argv[argv.index("-v") + 1] == "1000"
    assert argv[argv.index("-b") + 1] == "64"
    assert argv[argv.index("-t") + 1] == "1024"
    assert argv[argv.index("-d") + 1] == "524288"
    assert argv[argv.index("-l") + 1] == "0.0006"
    assert argv[argv.index("-q") + 1] == "0.0"
    assert argv[argv.index("-e") + 1] == "d12"
    assert argv[argv.index("-af") + 1] == "sd-prelu"
    assert argv[argv.index("-x") + 1] == "20000"


def test_build_native_gpt2_compiled_cli_config_passes_dataset_alias_without_shard_inspection(tmp_path: Path) -> None:
    cfg = build_native_gpt2_compiled_cli_run_config(
        dataset_alias="roneneldan__TinyStories__TinyStoriesV2-GPT4",
        executable="/opt/nfn/train_gpt2cu",
        output_dir=tmp_path / "log124M" / "5090_S",
        eval_every_steps=250,
        train_loss_every_steps=1000,
        lm_head_row_chunk_size=8192,
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
        kernel_backend="tile-cuda",
        tile_ops_lib="/opt/nfn/libnfn_native_train_tile_ops.so",
        smoke_tile_ops=True,
        smoke_nvfp4_pack=True,
        smoke_optimizer_step=True,
        smoke_lm_step=True,
        smoke_attention_step=True,
        smoke_mlp_step=True,
        smoke_norm_residual_step=True,
        smoke_transformer_block_step=True,
        smoke_transformer_lm_step=True,
        smoke_embedding_lm_step=True,
        train_embedding_lm=True,
        train_transformer_lm=True,
        checkpoint_metadata_smoke=True,
        cuda_runtime_lib="/usr/local/cuda/lib64/libcudart.so",
    )

    argv = cfg.compiled_cli_argv("/opt/nfn/nfn_gpt2_native_train")

    assert cfg.dataset_alias == "roneneldan__TinyStories__TinyStoriesV2-GPT4"
    assert cfg.train_data == ""
    assert argv[:9] == [
        "/opt/nfn/nfn_gpt2_native_train",
        "--model-family",
        "gpt",
        "--dataset-alias",
        "roneneldan__TinyStories__TinyStoriesV2-GPT4",
        "--backend",
        "tile-cuda",
        "--output-dir",
        str(tmp_path / "log124M" / "5090_S"),
    ]
    assert "--target" not in argv
    assert argv[argv.index("--train-batch-tokens") + 1] == "524288"
    assert argv[argv.index("--tile-ops-lib") + 1] == "/opt/nfn/libnfn_native_train_tile_ops.so"
    assert "--smoke-tile-ops" in argv
    assert "--smoke-nvfp4-pack" in argv
    assert "--smoke-optimizer-step" in argv
    assert "--smoke-lm-step" in argv
    assert "--smoke-attention-step" in argv
    assert "--smoke-mlp-step" in argv
    assert "--smoke-norm-residual-step" in argv
    assert "--smoke-transformer-block-step" in argv
    assert "--smoke-transformer-lm-step" in argv
    assert "--smoke-embedding-lm-step" in argv
    assert "--train-embedding-lm" in argv
    assert "--train-transformer-lm" in argv
    assert "--checkpoint-metadata-smoke" in argv
    assert argv[argv.index("--eval-batches") + 1] == "1"
    assert argv[argv.index("--eval-batch-size") + 1] == "0"
    assert cfg.train_loss_every_steps == 1000
    assert argv[argv.index("--train-loss-every-steps") + 1] == "1000"
    assert argv[argv.index("--lm-head-row-chunk-size") + 1] == "8192"
    assert argv[argv.index("--cuda-runtime-lib") + 1] == "/usr/local/cuda/lib64/libcudart.so"
    assert cfg.template_name == "gpt"
    assert argv[argv.index("--template-name") + 1] == "gpt"
    assert cfg.write_checkpoint is True
    assert "--no-checkpoint" not in argv


def test_build_native_gpt2_compiled_cli_config_can_defer_shape_to_graph_metadata(tmp_path: Path) -> None:
    cfg = build_native_gpt2_compiled_cli_run_config(
        dataset_alias="roneneldan__TinyStories__TinyStoriesV2-GPT4",
        executable="/opt/nfn/train_gpt2cu",
        output_dir=tmp_path / "graph-shaped-gpt",
        eval_every_steps=250,
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
        graph_file="/tmp/native-compatible-gpt-graph.json",
        batch_size_explicit=False,
        seq_len_explicit=False,
        num_layers_explicit=False,
    )

    argv = cfg.compiled_cli_argv("/opt/nfn/nfn_gpt_native_train")

    assert "--graph-file" in argv
    assert argv[argv.index("--graph-file") + 1] == "/tmp/native-compatible-gpt-graph.json"
    assert "--batch-size" not in argv
    assert "--train-seq-len" not in argv
    assert "--num-layers" not in argv
    assert cfg.batch_size_explicit is False
    assert cfg.seq_len_explicit is False
    assert cfg.num_layers_explicit is False

    with pytest.raises(ValueError, match="kernel backend must be tile-cuda"):
        build_native_gpt2_compiled_cli_run_config(
            dataset_alias="roneneldan__TinyStories__TinyStoriesV2-GPT4",
            executable="/opt/nfn/train_gpt2cu",
            output_dir=tmp_path / "gpt2-llm",
            eval_every_steps=250,
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
            kernel_backend="llm-kittens",
        )


def test_build_native_gpt2_compiled_cli_config_can_skip_checkpoint_export(tmp_path: Path) -> None:
    cfg = build_native_gpt2_compiled_cli_run_config(
        dataset_alias="cached-shards",
        executable=None,
        output_dir=tmp_path / "gpt",
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
        max_steps=1,
        num_layers=12,
        activation="gelu",
        write_checkpoint=False,
        startup_only=True,
    )
    generic_cfg = build_native_gpt_compiled_cli_run_config(
        dataset_alias="cached-shards",
        executable=None,
        output_dir=tmp_path / "generic-gpt",
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
        max_steps=1,
        num_layers=12,
        activation="gelu",
        write_checkpoint=False,
    )

    assert cfg.write_checkpoint is False
    cfg_argv = cfg.compiled_cli_argv("/opt/nfn/nfn_gpt_native_train")
    assert "--no-checkpoint" in cfg_argv
    assert "--startup-only" in cfg_argv
    assert isinstance(generic_cfg, NativeGptRunConfig)
    assert generic_cfg.write_checkpoint is False
    assert "--no-checkpoint" in generic_cfg.compiled_cli_argv("/opt/nfn/nfn_gpt_native_train")


def test_build_native_gpt_compiled_cli_config_preserves_zero_cadences(tmp_path: Path) -> None:
    cfg = build_native_gpt2_compiled_cli_run_config(
        dataset_alias="cached-shards",
        executable=None,
        output_dir=tmp_path / "gpt2",
        eval_every_steps=0,
        sample_every_steps=0,
        generate_tokens=144,
        checkpoint_every_steps=0,
        batch_size=64,
        seq_len=1024,
        train_batch_tokens=524288,
        learning_rate=0.0006,
        min_lr=None,
        warmup_steps=60,
        weight_decay=0.1,
        max_steps=5,
        num_layers=12,
        activation="gelu",
    )
    generic_cfg = build_native_gpt_compiled_cli_run_config(
        dataset_alias="cached-shards",
        executable=None,
        output_dir=tmp_path / "gpt",
        eval_every_steps=0,
        sample_every_steps=0,
        generate_tokens=144,
        checkpoint_every_steps=0,
        batch_size=64,
        seq_len=1024,
        train_batch_tokens=524288,
        learning_rate=0.0006,
        min_lr=None,
        warmup_steps=60,
        weight_decay=0.1,
        max_steps=5,
        num_layers=12,
        activation="gelu",
    )

    cfg_argv = cfg.compiled_cli_argv("/opt/nfn/nfn_gpt_native_train")
    generic_argv = generic_cfg.compiled_cli_argv("/opt/nfn/nfn_gpt_native_train")

    assert cfg.eval_every_steps == 0
    assert cfg.sample_every_steps == 0
    assert cfg.checkpoint_every_steps == 0
    assert cfg_argv[cfg_argv.index("--eval-every-steps") + 1] == "0"
    assert cfg_argv[cfg_argv.index("--native-cuda-sample-every") + 1] == "0"
    assert cfg_argv[cfg_argv.index("--native-cuda-checkpoint-every") + 1] == "0"
    assert generic_cfg.eval_every_steps == 0
    assert generic_cfg.sample_every_steps == 0
    assert generic_cfg.checkpoint_every_steps == 0
    assert generic_argv[generic_argv.index("--eval-every-steps") + 1] == "0"
    assert generic_argv[generic_argv.index("--native-cuda-sample-every") + 1] == "0"
    assert generic_argv[generic_argv.index("--native-cuda-checkpoint-every") + 1] == "0"


def test_build_native_gpt2_compiled_cli_config_defaults_to_neuralfn_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native_cli = tmp_path / "nfn_gpt_native_train"
    monkeypatch.setenv("NFN_NATIVE_GPT_CLI", str(native_cli))
    monkeypatch.setenv("NFN_NATIVE_GPT_TRAIN_BIN", "/opt/nfn/train_gpt2cu")

    cfg = build_native_gpt2_compiled_cli_run_config(
        dataset_alias="cached-shards",
        executable=None,
        output_dir=tmp_path / "gpt",
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
        max_steps=1,
        num_layers=12,
        activation="gelu",
    )
    assert cfg.executable == str(native_cli)
    assert cfg.kernel_backend == "tile-cuda"
    assert "--target" not in cfg.compiled_cli_argv()
    with pytest.raises(ValueError, match="kernel backend must be tile-cuda"):
        build_native_gpt2_compiled_cli_run_config(
            dataset_alias="cached-shards",
            executable=None,
            output_dir=tmp_path / "gpt-llm",
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
            max_steps=1,
            num_layers=12,
            activation="gelu",
            kernel_backend="llm-kittens",
        )


def test_native_gpt2_compiled_cli_prefers_linked_workstation_binary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    linked_cli = tmp_path / "nfn_gpt_native_train_linked"
    dynamic_cli = tmp_path / "nfn_gpt_native_train"
    linked_cli.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    dynamic_cli.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    linked_cli.chmod(0o755)
    dynamic_cli.chmod(0o755)

    monkeypatch.delenv("NFN_NATIVE_GPT_CLI", raising=False)
    monkeypatch.delenv("NFN_NATIVE_GPT2_CLI", raising=False)
    monkeypatch.setattr(native_gpt2_module, "DEFAULT_NATIVE_GPT_CLI_LINKED", str(linked_cli))
    monkeypatch.setattr(native_gpt2_module, "DEFAULT_NATIVE_GPT2_CLI", str(dynamic_cli))

    cfg = build_native_gpt2_compiled_cli_run_config(
        dataset_alias="cached-shards",
        executable=None,
        output_dir=tmp_path / "gpt",
        eval_every_steps=0,
        sample_every_steps=0,
        generate_tokens=0,
        checkpoint_every_steps=0,
        batch_size=64,
        seq_len=1024,
        train_batch_tokens=524288,
        learning_rate=0.0006,
        min_lr=None,
        warmup_steps=60,
        weight_decay=0.1,
        max_steps=1,
        num_layers=12,
        activation="gelu",
    )
    argv = cfg.compiled_cli_argv()

    assert resolve_native_gpt2_cli() == str(linked_cli)
    assert argv[0] == str(linked_cli)
    assert argv[argv.index("--tile-ops-lib") + 1] == "linked"

    explicit_tile_ops = build_native_gpt2_compiled_cli_run_config(
        dataset_alias="cached-shards",
        executable=None,
        output_dir=tmp_path / "gpt-explicit",
        eval_every_steps=0,
        sample_every_steps=0,
        generate_tokens=0,
        checkpoint_every_steps=0,
        batch_size=64,
        seq_len=1024,
        train_batch_tokens=524288,
        learning_rate=0.0006,
        min_lr=None,
        warmup_steps=60,
        weight_decay=0.1,
        max_steps=1,
        num_layers=12,
        activation="gelu",
        tile_ops_lib="/tmp/libcandidate.so",
    ).compiled_cli_argv()
    assert explicit_tile_ops[0] == str(linked_cli)
    assert explicit_tile_ops[explicit_tile_ops.index("--tile-ops-lib") + 1] == "/tmp/libcandidate.so"

    monkeypatch.setenv("NFN_NATIVE_GPT_CLI", str(dynamic_cli))
    assert resolve_native_gpt2_cli() == str(dynamic_cli)


def test_native_gpt_external_bridge_defaults_are_removed_from_training_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    native_sdk_source = (root / "neuralfn" / "native_gpt2.py").read_text(encoding="utf-8")
    train_gpt_source = (root / "cli" / "scripts" / "train_gpt.py").read_text(encoding="utf-8")
    train_gpt_native_source = (root / "cli" / "scripts" / "train_gpt_native.py").read_text(
        encoding="utf-8"
    )
    native_cli_source = (
        root / "neuralfn" / "csrc" / "native_gpt2" / "nfn_gpt2_native_train.cpp"
    ).read_text(encoding="utf-8")

    assert 'DEFAULT_NATIVE_GPT2_EXECUTABLE = "nfn_gpt_native_train"' in native_sdk_source
    assert '_DEFAULT_NATIVE_GPT_TARGET = "train_gpt2cu"' not in train_gpt_source
    assert "/mnt/disk2/dev/open-source/llm.kittens/train_gpt2cu" not in native_sdk_source
    assert "/mnt/disk2/dev/open-source/llm.kittens/train_gpt2cu" not in train_gpt_source
    assert "/mnt/disk2/dev/open-source/llm.kittens/train_gpt2cu" not in native_cli_source
    assert 'return "train_gpt2cu"' not in native_cli_source
    assert '"status": "external-fast-path"' not in native_cli_source
    assert "build_command(const Config& cfg" not in native_cli_source
    assert "os.execvpe(command[0], command, _compiled_cli_env(config))" in train_gpt_native_source
    assert '_set_env_default_if_empty(env, "CUDA_MODULE_LOADING", "LAZY")' in train_gpt_native_source
    assert (
        'if runner_status.resolved == "compiled-cli":\n'
        "        return _exec_compiled_cli(compiled_cli_args or native_cfg.compiled_cli_argv(), native_cfg)"
    ) in train_gpt_native_source
    assert "subprocess.run(compiled_cli_args or native_cfg.compiled_cli_argv()" not in train_gpt_native_source


def test_native_gpt_lm_smoke_uses_stable_cuda_13_3_expectations() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "neuralfn" / "csrc" / "native_gpt2" / "nfn_gpt2_native_train.cpp").read_text(
        encoding="utf-8"
    )

    assert "static_cast<double>(kRows) * std::log(static_cast<double>(kPaddedVocab))" in source
    assert "token == host_targets[0] || token == host_targets[1]" in source
    assert "max_grad_abs_error <= 1e-5 && max_weight_abs_error <= 1e-4" in source


def test_native_gpt_bf16_ce_vector_stores_reuse_vec_loads() -> None:
    root = Path(__file__).resolve().parents[1]
    kernels_text = (root / "neuralfn" / "csrc" / "tile_cuda" / "kernels.cu").read_text(encoding="utf-8")
    packed_load = (
        "const int4 packed = vec_loads ? load_bf16_vec8(row_logits + col) : make_int4(0, 0, 0, 0);"
    )
    raw_load = "const std::uint16_t raw = vec_loads ? int4_u16_at(packed, offset) : row_logits[current_col];"
    assert kernels_text.count(packed_load) >= 5
    assert kernels_text.count(raw_load) >= 5
    assert "(vec_normal_stores && vec_loads) ? load_bf16_vec8" not in kernels_text
    assert "(vec_normal_stores && vec_loads) ? int4_u16_at" not in kernels_text
    assert "cross_entropy_bf16_scalar_streaming_stores_enabled" in kernels_text
    assert "NFN_NATIVE_GPT_CE_BF16_SCALAR_STREAMING_STORES" in kernels_text
    assert "NFN_NATIVE_GPT2_CE_BF16_SCALAR_STREAMING_STORES" in kernels_text
    assert "NFN_TILE_CUDA_CE_BF16_SCALAR_STREAMING_STORES" in kernels_text
    assert "store_bf16_scalar(" in kernels_text
    assert "st.global.cs.u16" in kernels_text


def test_native_gpt_lm_head_cooperative_abi_is_typed_and_graph_prewarm_default_on() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "neuralfn" / "csrc" / "native_gpt2" / "nfn_gpt2_native_train.cpp").read_text(
        encoding="utf-8"
    )
    tile_ops_source = (root / "neuralfn" / "csrc" / "native_train" / "tile_ops.cu").read_text(
        encoding="utf-8"
    )
    tile_ops_header = (root / "neuralfn" / "csrc" / "native_train" / "tile_ops.h").read_text(
        encoding="utf-8"
    )
    assert "using LmHeadClassifierBackwardCooperativeBf16U16Fn = int (*)();" not in source
    for required_arg in [
        "std::uint16_t* logits_bf16",
        "const std::uint16_t* targets_u16",
        "float* row_losses",
        "const std::uint16_t* hidden_bf16",
        "const float* hidden_float",
        "const std::uint16_t* token_weight_bf16",
        "const float* token_weight_float",
        "float* grad_hidden",
        "float* grad_weight",
        "std::int64_t row_stride",
        "float dweight_beta",
        "int flags",
        "void* stream",
    ]:
        assert required_arg in source
        assert required_arg.replace("stream", "cuda_stream") in tile_ops_source or required_arg in tile_ops_source
    assert "nfn_native_tile_lm_head_classifier_backward_cooperative_bf16_u16" in tile_ops_source
    assert "launch_lm_head_classifier_backward_row_losses_inplace_strided_no_pad_zero_bf16_bits_u16_targets" in tile_ops_source
    assert "kLmHeadCooperativeFlagLossBins" in tile_ops_source
    assert "kLmHeadCooperativeFlagNoLoss" in tile_ops_source
    assert "launch_lm_head_classifier_backward_loss_bins_inplace_strided_no_pad_zero_bf16_bits_u16_targets" in tile_ops_source
    assert "launch_lm_head_classifier_backward_inplace_strided_no_pad_zero_bf16_bits_u16_targets_with_workspace" in tile_ops_source
    assert "record_loss ? 0 : kLmHeadCooperativeFlagNoLoss" in source
    assert "kLmHeadCooperativeLossBinCountShift" in source
    assert "lm_head_classifier_no_loss_chunk_count += 1" in source
    assert "launch_linear_backward_input_bf16_bits_weight_bf16_float32" in tile_ops_source
    assert "launch_linear_backward_input_bf16_bits_weight_bf16_strided_float32" in tile_ops_source
    assert "launch_linear_backward_weight_accumulate_bf16_bits_bf16_bits_float32_beta" in tile_ops_source
    assert "launch_linear_backward_weight_accumulate_bf16_bits_bf16_bits_strided_float32_beta" in tile_ops_source
    assert "nfn_native_tile_linear_backward_input_bf16_bits_weight_bf16_strided_float32" in tile_ops_header
    assert "nfn_native_tile_linear_backward_weight_accumulate_bf16_bits_bf16_bits_strided_float32_beta" in tile_ops_header
    assert "NFN_NATIVE_GPT_LM_HEAD_PUBLIC_VOCAB_STRIDED_GEMM" in source
    assert "NFN_NATIVE_GPT2_LM_HEAD_PUBLIC_VOCAB_STRIDED_GEMM" in source
    assert "lm_head_public_vocab_strided_gemm_requested" in source
    default_off_strided_gemm = (
        'env_or_empty_any({"NFN_NATIVE_GPT_LM_HEAD_PUBLIC_VOCAB_STRIDED_GEMM",\n'
        + '                              "NFN_NATIVE_GPT2_LM_HEAD_PUBLIC_VOCAB_STRIDED_GEMM"}),\n'
        + "            false);"
    )
    assert default_off_strided_gemm in source
    assert "public-vocab-strided-bf16-dinput-dhidden" in source
    assert "public-vocab-strided-bf16-dlogit-dweight" in source
    assert "NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_BACKWARD" in source
    assert "NFN_NATIVE_GPT2_LM_HEAD_COOPERATIVE_BACKWARD" in source
    assert "NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_LOSS_BINS" in source
    assert "NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_CUDA_GRAPH" in source
    assert "NFN_NATIVE_GPT2_LM_HEAD_COOPERATIVE_CUDA_GRAPH" in source
    assert "NFN_NATIVE_GPT_LM_HEAD_FORCE_SEQUENCE_WRAPPER_DIAGNOSTIC" in source
    assert "NFN_NATIVE_GPT2_LM_HEAD_FORCE_SEQUENCE_WRAPPER_DIAGNOSTIC" in source
    assert "NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_GRAPH_PREWARM" in source
    assert "NFN_NATIVE_GPT2_LM_HEAD_COOPERATIVE_GRAPH_PREWARM" in source
    assert "native_fast_startup_prewarm_default" in source
    assert "cfg.fast_startup ||" in source
    assert "!native_fast_startup_requested && !cfg.startup_only" in source
    assert "startup-only-skip-throughput-prewarms-by-default" in source
    assert (
        '"NFN_NATIVE_GPT2_LM_HEAD_COOPERATIVE_GRAPH_PREWARM"}),\n'
        "            native_fast_startup_prewarm_default)"
    ) in source
    assert "NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_CUBLASLT" in source
    assert "NFN_NATIVE_GPT2_LM_HEAD_COOPERATIVE_CUBLASLT" in source
    assert "lm_head_cooperative_loss_bins_requested" in source
    assert "lm_head_cooperative_backward_cuda_graph_requested" in source
    assert "lm_head_cooperative_cublaslt_requested" in source
    assert "lm_head_force_sequence_wrapper_diagnostic_enabled" in source
    assert "lm_head_cooperative_backward_graph_prewarm_requested" in source
    assert "lm_head_cooperative_backward_graph_prewarm_enabled" in source
    assert "nfn_native_tile_lm_head_classifier_backward_cooperative_fused_bf16_u16" in source
    assert "nfn_native_tile_lm_head_classifier_backward_cooperative_cublaslt_bf16_u16" in source
    assert "nfn_native_tile_lm_head_classifier_backward_fused_kernel_bf16_u16" in source
    assert "nfn_native_tile_lm_head_classifier_backward_fused_graph_prewarm_bf16_u16" in source
    assert "nfn_native_tile_lm_head_classifier_backward_fused_kernel_is_true_fused" in source
    assert "nfn_native_tile_lm_head_classifier_backward_llmk_classifier_matmul_parity" in source
    assert "lm_head_classifier_backward_true_fused_kernel_bf16_u16" in source
    assert "lm_head_classifier_backward_true_fused_capability" in source
    assert "lm_head_classifier_backward_llmk_parity_capability" in source
    assert "lm_head_cooperative_backward_fused_kernel_symbol_available" in source
    assert "lm_head_cooperative_backward_fused_kernel_raw_capability_available" in source
    assert "lm_head_cooperative_backward_fused_kernel_capability_available" in source
    assert "lm_head_true_fused_cooperative_requested" in source
    assert "lm_head_true_fused_cooperative_production_shape" in source
    assert "lm_head_true_fused_cooperative_allow_production" in source
    assert "lm_head_true_fused_cooperative_shape_allowed" in source
    assert "!lm_head_true_fused_cooperative_requested &&" in source
    assert "lm_head_llmk_classifier_matmul_parity_available" in source
    assert "lm_head_cooperative_backward_cuda_graph_available" in source
    assert "lm_head_cooperative_backward_cuda_graph_enabled" in source
    assert "lm_head_classifier_backward_path_class" in source
    assert "lm_head_cooperative_backward_fused_kernel_abi_path_class" in source
    assert "strict-true-fused-tile-kernel" in source
    assert "diagnostic-cuda-graph-wrapper" in source
    assert "diagnostic-cublaslt-sequence-wrapper" in source
    assert "diagnostic-sequence-wrapper" in source
    assert "legacy-abi-sequence-wrapper" in source
    assert "cooperative_lm_head_backward_requirement_error" in source
    assert (
        "required cooperative LM-head backward Tile path is unavailable"
    ) in source
    assert "must return true before training starts" in source
    assert "does not satisfy the strict true-fused requirement" in source
    assert (
        source.index("cooperative_lm_head_backward_requirement_error(cfg, argv[0])")
        < source.index("resolve_token_shards(")
    )
    assert "auto* cooperative_backward_fn =" in source
    assert (
        "lm_head_cooperative_backward_kernel_enabled ||\n"
        "                         lm_head_cooperative_backward_cuda_graph_enabled"
    ) in source
    assert "? lm_head_classifier_backward_true_fused_kernel_bf16_u16" in source
    assert "lm_head_classifier_backward_cooperative_cublaslt_bf16_u16" in source
    assert "lm_head_classifier_backward_cooperative_fused_bf16_u16" in source
    assert (
        "cooperative LM-head backward route selected without a callable Tile function"
    ) in source
    assert "lm_head_cooperative_backward_cublaslt_wrapper_available" in source
    assert "lm_head_cooperative_backward_cublaslt_wrapper_enabled" in source
    assert "lm_head_cooperative_backward_sequence_wrapper_available" in source
    assert "lm_head_cooperative_backward_sequence_wrapper_enabled" in source
    assert "lm_head.backward.cooperative.bf16_u16" in source
    assert "abi-wrapper-sequences-existing-ce-dhidden-dweight-kernels-not-parity" in source
    assert "diagnostic-sequence-wrapper-ce-side-stream-dhidden-dweight-not-parity" in source
    assert "strict-true-fused-cooperative-classifier-backward" in source
    assert "diagnostic-sequence-wrapper-loss-bins-ce-side-stream-dhidden-dweight-not-parity" in source
    assert "diagnostic-cuda-graph-ce-fork-join-dhidden-dweight-not-single-kernel" in source
    assert "diagnostic-cublaslt-sequence-wrapper-ce-dhidden-dweight-not-parity" in source
    assert "linear_tk_sm120_config_symbol_loaded = false" in source
    assert 'dlsym(handle, "nfn_native_tile_trainer_linear_tk_sm120_k_tile")' in source
    assert (
        'dlsym(handle, "nfn_native_tile_trainer_linear_tk_sm120_approx_dgelu_tanh_enabled")'
        in source
    )
    assert '"linear_tk_sm120_config_symbol_loaded": false' not in source
    assert '"linear_tk_sm120_fast_dgelu_enabled": false' not in source
    assert (
        "diagnostic-cuda-graph-loss-bins-ce-fork-join-dhidden-dweight-not-single-kernel"
        in source
    )
    assert "strict-true-fused-cooperative-classifier-loss-bins-backward" in source
    assert "nfn_native_tile_lm_head_classifier_backward_cooperative_fused_bf16_u16" in tile_ops_source
    assert "nfn_native_tile_lm_head_classifier_backward_cooperative_fused_bf16_u16" in tile_ops_header
    assert "nfn_native_tile_lm_head_classifier_backward_cooperative_cublaslt_bf16_u16" in tile_ops_source
    assert "nfn_native_tile_lm_head_classifier_backward_cooperative_cublaslt_bf16_u16" in tile_ops_header
    assert "cublaslt_linear_backward_input_bf16_bits_weight_bf16_strided_float32" in tile_ops_source
    assert (
        "cublaslt_linear_backward_weight_accumulate_bf16_bits_bf16_bits_strided_float32_beta"
        in tile_ops_source
    )
    assert "nfn_native_tile_lm_head_classifier_backward_fused_kernel_bf16_u16" in tile_ops_source
    assert "nfn_native_tile_lm_head_classifier_backward_fused_kernel_bf16_u16" in tile_ops_header
    assert "nfn_native_tile_lm_head_classifier_backward_fused_graph_prewarm_bf16_u16" in tile_ops_source
    assert "nfn_native_tile_lm_head_classifier_backward_fused_graph_prewarm_bf16_u16" in tile_ops_header
    assert "nfn_native_tile_lm_head_classifier_backward_fused_kernel_is_true_fused" in tile_ops_source
    assert "nfn_native_tile_lm_head_classifier_backward_fused_kernel_is_true_fused" in tile_ops_header
    assert "nfn_native_tile_lm_head_classifier_backward_fused_kernel_path_class" in tile_ops_source
    assert "nfn_native_tile_lm_head_classifier_backward_fused_kernel_path_class" in tile_ops_header
    assert "nfn_native_tile_lm_head_classifier_true_fused_launch_count" in tile_ops_source
    assert "nfn_native_tile_lm_head_classifier_true_fused_launch_count" in tile_ops_header
    assert "nfn_native_tile_lm_head_classifier_backward_fused_kernel_graph_body_node_count" in tile_ops_source
    assert "nfn_native_tile_lm_head_classifier_backward_fused_kernel_graph_body_node_count" in tile_ops_header
    assert "nfn_native_tile_lm_head_classifier_backward_fused_kernel_graph_body_ce_node_count" in tile_ops_source
    assert "nfn_native_tile_lm_head_classifier_backward_fused_kernel_graph_body_ce_node_count" in tile_ops_header
    assert "nfn_native_tile_lm_head_classifier_backward_fused_kernel_graph_body_dhidden_node_count" in tile_ops_source
    assert "nfn_native_tile_lm_head_classifier_backward_fused_kernel_graph_body_dhidden_node_count" in tile_ops_header
    assert "nfn_native_tile_lm_head_classifier_backward_fused_kernel_graph_body_dweight_node_count" in tile_ops_source
    assert "nfn_native_tile_lm_head_classifier_backward_fused_kernel_graph_body_dweight_node_count" in tile_ops_header
    assert "kLmHeadCooperativeBackwardTrueFusedPathClassSymbol" in source
    assert "kLmHeadCooperativeBackwardGraphBodyNodeCountSymbol" in source
    assert "candidate_symbol_abi_path_class" in (root / "neuralfn" / "csrc" / "native_train" / "lm_head_backward_bench.cpp").read_text(encoding="utf-8")
    assert "nfn_native_tile_lm_head_classifier_backward_llmk_classifier_matmul_parity" in tile_ops_source
    assert "nfn_native_tile_lm_head_classifier_backward_llmk_classifier_matmul_parity" in tile_ops_header
    assert "nfn_native_tile_lm_head_prob_only_target_correction_threads" in tile_ops_source
    assert "nfn_native_tile_lm_head_prob_only_target_correction_threads" in tile_ops_header
    assert "lm_head_prob_only_target_correction_threads" in source
    assert "cudaGraphLaunch(exec, stream)" in tile_ops_source
    assert "cudaGraphUpload(exec, stream)" in tile_ops_source
    assert "NFN_NATIVE_GPT_LM_HEAD_GRAPH_UPLOAD" in tile_ops_source
    assert "prewarm_lm_head_classifier_backward_graph_bf16_u16" in tile_ops_source
    assert "cudaStreamCreateWithFlags" in tile_ops_source
    assert "cudaEventRecord(cooperative_streams.ce_done, stream)" in tile_ops_source
    assert (
        "cudaStreamWaitEvent(cooperative_streams.dhidden, cooperative_streams.ce_done, 0)"
        in tile_ops_source
    )
    assert (
        "cudaStreamWaitEvent(cooperative_streams.dweight, cooperative_streams.ce_done, 0)"
        in tile_ops_source
    )
    assert "cooperative_streams.dhidden_done" in tile_ops_source
    assert "cooperative_streams.dweight_done" in tile_ops_source
    assert "cudaError_t launch_lm_head_classifier_backward_true_fused_cooperative_bf16_bits_u16" in tile_ops_source
    assert "return status == cudaSuccess ? launch_status() : static_cast<int>(status);" in tile_ops_source
    kernels_source = (root / "neuralfn" / "csrc" / "tile_cuda" / "kernels.cu").read_text(
        encoding="utf-8"
    )
    assert "NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE_ALLOW_PRODUCTION" in kernels_source
    assert "NFN_NATIVE_GPT_LM_HEAD_TRUE_FUSED_COOPERATIVE_ALLOW_PRODUCTION" in kernels_source
    assert "NFN_NATIVE_GPT2_LM_HEAD_TRUE_FUSED_COOPERATIVE_ALLOW_PRODUCTION" in kernels_source
    assert "lm_head_true_fused_cooperative_allow_production_enabled" in kernels_source
    assert "return cudaErrorNotSupported;" in kernels_source
    assert "lm_head_classifier_backward_true_fused_cooperative_bf16_bits_u16_kernel" in kernels_source
    assert "cg::this_grid()" in kernels_source
    true_fused_kernel_body = kernels_source.split(
        "lm_head_classifier_backward_true_fused_cooperative_bf16_bits_u16_kernel",
        1,
    )[1].split(
        "__global__ void token_cross_entropy_backward_inplace_strided_no_pad_zero_bf16_bits_u16_targets_llmk_style_kernel",
        1,
    )[0]
    assert "if (!no_loss && threadIdx.x == 0 && row_losses != nullptr) {\n      const float target_logit" in true_fused_kernel_body
    assert "constexpr int kMatTile = kLmHeadTrueFusedMatTile;" in true_fused_kernel_body
    assert "#ifndef NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_MAT_TILE" in kernels_source
    assert "NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_MAT_TILE == 4" in kernels_source
    assert "NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_MAT_TILE == 8" in kernels_source
    assert "NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_MAT_TILE == 16" in kernels_source
    assert "NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_MAT_TILE == 32" in kernels_source
    assert (
        "constexpr int kLmHeadTrueFusedMatTile = "
        "NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_MAT_TILE;"
    ) in kernels_source
    assert "kLmHeadTrueFusedRequiredThreads" in kernels_source
    assert "lm_head_true_fused_mat_tile()" in kernels_source
    assert "lm_head_true_fused_required_threads()" in kernels_source
    assert "case 16:" in kernels_source
    assert "case 64:" in kernels_source
    assert (
        "if (threads != kLmHeadTrueFusedRequiredThreads) {\n"
        "    return cudaErrorNotSupported;"
    ) in kernels_source
    assert "__shared__ float tile_a[kMatTile][kMatTile];" in true_fused_kernel_body
    assert "__shared__ float tile_b[kMatTile][kMatTile];" in true_fused_kernel_body
    assert "hidden_tiles = hidden_row_tiles * hidden_col_tiles" in true_fused_kernel_body
    assert "weight_tiles = weight_row_tiles * weight_col_tiles" in true_fused_kernel_body
    assert "tile_a[tile_row][k] * tile_b[k][tile_col]" in true_fused_kernel_body
    assert "tile_a[k][tile_col] * tile_b[k][tile_row]" in true_fused_kernel_body
    assert "g_lm_head_classifier_true_fused_launch_count" in kernels_source
    assert "lm_head_classifier_true_fused_launch_count()" in kernels_source
    assert "g_lm_head_classifier_true_fused_launch_count.fetch_add" in kernels_source
    assert "status = cudaLaunchCooperativeKernel(" in kernels_source
    assert (
        "if (status == cudaSuccess && cudaPeekAtLastError() == cudaSuccess) {\n"
        "    g_lm_head_classifier_true_fused_launch_count.fetch_add"
    ) in kernels_source
    assert "NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE" in tile_ops_source
    assert "NFN_NATIVE_GPT_LM_HEAD_TRUE_FUSED_COOPERATIVE" in tile_ops_source
    assert "NFN_NATIVE_GPT2_LM_HEAD_TRUE_FUSED_COOPERATIVE" in tile_ops_source
    assert (
        "int nfn_native_tile_lm_head_classifier_backward_fused_kernel_is_true_fused() {\n"
        "    return lm_head_true_fused_cooperative_enabled() ? 1 : 0;\n"
        "}"
    ) in tile_ops_source
    assert (
        "return neuralfn::tile_cuda::token_cross_entropy_bf16_threads_per_row() ==\n"
        "           neuralfn::tile_cuda::lm_head_true_fused_required_threads();"
        in tile_ops_source
    )
    assert "nfn_native_tile_lm_head_true_fused_mat_tile" in tile_ops_header
    assert "nfn_native_tile_lm_head_true_fused_required_threads" in tile_ops_header
    assert "nfn_native_tile_lm_head_true_fused_mat_tile" in source
    assert "nfn_native_tile_lm_head_true_fused_required_threads" in source
    assert "lm_head_true_fused_mat_tile_fn" in source
    assert "lm_head_true_fused_required_threads_fn" in source
    assert '\\"lm_head_true_fused_mat_tile\\"' in source
    assert '\\"lm_head_true_fused_required_threads\\"' in source
    assert (
        "const char* nfn_native_tile_lm_head_classifier_backward_fused_kernel_path_class() {\n"
        "    if (lm_head_true_fused_cooperative_enabled()) {\n"
        '        return "strict-true-fused-tile-kernel";'
    ) in tile_ops_source
    assert '"diagnostic-cuda-graph-wrapper"' in tile_ops_source
    assert (
        "int nfn_native_tile_lm_head_classifier_backward_fused_kernel_graph_body_node_count() {\n"
        "    return 3;\n"
        "}"
    ) in tile_ops_source
    assert "lm_head_fused_graph_body_ce_node_count_per_replay" in source
    assert "lm_head_fused_graph_body_dhidden_node_replay_total" in source
    assert "lm_head_fused_graph_body_dweight_node_replay_total" in source
    assert (
        "int nfn_native_tile_lm_head_classifier_backward_llmk_classifier_matmul_parity() {\n"
        "    return 1;\n"
        "}"
    ) in tile_ops_source
    assert "run_lm_head_classifier_backward_cooperative_sequence_bf16_u16" in tile_ops_source
    assert "LmHeadCooperativeStreams" in tile_ops_source
    assert "cudaStreamWaitEvent" in tile_ops_source
    wrapper_source = (root / "tools" / "bench_lm_head_backward_candidate.sh").read_text(
        encoding="utf-8"
    )
    assert "trainer-chunk-true-fused-tile16" in wrapper_source
    assert "trainer-chunk-true-fused-tile8" in wrapper_source
    assert "trainer-chunk-true-fused-tile4" in wrapper_source
    assert "-DNFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_MAT_TILE=16" in wrapper_source
    assert "-DNFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_MAT_TILE=8" in wrapper_source
    assert "-DNFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_MAT_TILE=4" in wrapper_source
    assert 'NFN_TILE_CUDA_CE_BF16_THREADS="${NFN_TILE_CUDA_CE_BF16_THREADS:-256}"' in wrapper_source
    assert 'NFN_TILE_CUDA_CE_BF16_THREADS="${NFN_TILE_CUDA_CE_BF16_THREADS:-64}"' in wrapper_source
    assert 'NFN_TILE_CUDA_CE_BF16_THREADS="${NFN_TILE_CUDA_CE_BF16_THREADS:-16}"' in wrapper_source
    assert "nfn_lm_head_backward_tile_ops_true_fused_tile16.so" in wrapper_source
    assert "nfn_lm_head_backward_tile_ops_true_fused_tile8.so" in wrapper_source
    assert "nfn_lm_head_backward_tile_ops_true_fused_tile4.so" in wrapper_source
    assert "37.738071x candidate/baseline" in wrapper_source
    assert "113.697403x candidate/reference-summed" in wrapper_source
    assert "4510.827989 ms slower" in wrapper_source
    assert "FORCE_REBUILD_TILE_OPS=1" in wrapper_source
    bench_source = (root / "neuralfn" / "csrc" / "native_train" / "lm_head_backward_bench.cpp").read_text(
        encoding="utf-8"
    )
    assert "true_fused_launch_count" in bench_source
    assert "nfn_native_tile_lm_head_classifier_true_fused_launch_count" in bench_source
    assert "g_lm_head_cooperative_sequence_launch_count" in tile_ops_source
    assert "linked_tile_ops_requested ? RTLD_DEFAULT : dlopen" in source
    assert "if (!linked_tile_ops_requested && handle == nullptr)" in source
    assert "nfn_native_tile_lm_head_cooperative_sequence_launch_count" in tile_ops_source
    assert "nfn_native_tile_lm_head_cooperative_sequence_ce_launch_count" in tile_ops_header
    assert "nfn_native_tile_lm_head_cooperative_sequence_dhidden_launch_count" in tile_ops_header
    assert "nfn_native_tile_lm_head_cooperative_sequence_dweight_launch_count" in tile_ops_header
    assert "nfn_native_tile_lm_head_cooperative_sequence_concurrent_count" in tile_ops_header
    assert "nfn_native_tile_lm_head_cooperative_sequence_legacy_count" in tile_ops_header
    assert "nfn_native_tile_lm_head_cooperative_sequence_loss_bin_count" in tile_ops_header
    assert "nfn_native_tile_lm_head_fused_graph_capture_attempt_count" in tile_ops_header
    assert "nfn_native_tile_lm_head_fused_graph_capture_success_count" in tile_ops_header
    assert "nfn_native_tile_lm_head_fused_graph_upload_success_count" in tile_ops_header
    assert "nfn_native_tile_lm_head_fused_graph_upload_failure_count" in tile_ops_header
    assert "g_lm_head_fused_graph_upload_success_count.store(0, std::memory_order_relaxed)" in tile_ops_source
    assert "g_lm_head_fused_graph_upload_failure_count.store(0, std::memory_order_relaxed)" in tile_ops_source
    assert "struct LmHeadGraphLocalStats" in tile_ops_source
    assert "thread_local LmHeadGraphLocalStats stats" in tile_ops_source
    assert "reset_lm_head_graph_local_stats()" in tile_ops_source
    assert "LmHeadGraphLocalStats& stats = lm_head_graph_local_stats()" in tile_ops_source
    assert "stats.replay_count += 1" in tile_ops_source
    assert "stats.replay_success_count += 1" in tile_ops_source
    assert "lm_head_graph_local_stats().fallback_count += 1" in tile_ops_source
    assert "nfn_native_tile_lm_head_fused_graph_cache_hit_count" in tile_ops_header
    assert "nfn_native_tile_lm_head_fused_graph_thread_cache_hit_count" in tile_ops_header
    assert "nfn_native_tile_lm_head_fused_graph_cache_entry_count" in tile_ops_header
    assert "nfn_native_tile_lm_head_fused_graph_replay_count" in tile_ops_header
    assert "nfn_native_tile_lm_head_fused_graph_replay_success_count" in tile_ops_header
    assert "nfn_native_tile_lm_head_fused_graph_fallback_count" in tile_ops_header
    assert "nfn_native_tile_lm_head_graph_body_cublaslt_dhidden_launch_count" in tile_ops_header
    assert "nfn_native_tile_lm_head_graph_body_cublaslt_dweight_launch_count" in tile_ops_header
    assert "nfn_native_tile_lm_head_graph_body_tile_dhidden_fallback_count" in tile_ops_header
    assert "nfn_native_tile_lm_head_graph_body_tile_dweight_fallback_count" in tile_ops_header
    assert "g_lm_head_graph_body_cublaslt_dhidden_launch_count" in tile_ops_source
    assert "g_lm_head_graph_body_cublaslt_dweight_launch_count" in tile_ops_source
    assert "g_lm_head_graph_body_tile_dhidden_fallback_count" in tile_ops_source
    assert "g_lm_head_graph_body_tile_dweight_fallback_count" in tile_ops_source
    assert "lm_head_cooperative_sequence_launch_count" in source
    assert "lm_head_classifier_true_fused_launch_count" in source
    assert "lm_head_fused_graph_replay_success_count" in source
    assert "lm_head_fused_graph_upload_success_count" in source
    assert "lm_head_fused_graph_upload_failure_count" in source
    assert "lm_head_fused_graph_thread_cache_hit_count" in source
    assert "lm_head_graph_body_cublaslt_dhidden_launch_count" in source
    assert "lm_head_graph_body_cublaslt_dweight_launch_count" in source
    assert "lm_head_graph_body_tile_dhidden_fallback_count" in source
    assert "lm_head_graph_body_tile_dweight_fallback_count" in source
    assert "lm_head_fused_graph_prewarm_success_count" in source
    assert "lm_head_fused_graph_prewarm_duplicate_skip_count" in source
    assert "lm_head_fused_graph_prewarm_dedup_enabled" in source
    assert "NFN_NATIVE_GPT_LM_HEAD_GRAPH_PREWARM_DEDUP" in source
    assert "NFN_NATIVE_GPT2_LM_HEAD_GRAPH_PREWARM_DEDUP" in source
    assert "lm_head_fused_graph_prewarm_failure_count" in source
    assert "lm_head_fused_graph_prewarm_last_error_code" in source
    assert "lm_head_fused_graph_prewarm_cache_entry_count" in source
    assert "struct LmHeadGraphPrewarmKey" in source
    assert "std::uint16_t* logits_bf16 = nullptr;" in source
    assert "existing.logits_bf16 == key.logits_bf16" in source
    assert "existing.grad_hidden == key.grad_hidden" in source
    assert "bf16_logit_chunk," in source
    assert "grad_hidden_chunk," in source
    assert "already_prewarmed(key)" in source
    assert "prewarm_row_loss_reduction_available" in source
    assert "prewarm_loss_bin_reduction_available" in source
    assert "const int prewarm_flags[] = {kLmHeadCooperativeFlagNoLoss, prewarm_loss_flags};" in source
    assert "lm_head.fused_graph.prewarm_loss_bins.zero" in source
    assert "lm_head_fused_graph_prewarm_last_rows = row_count" in source
    assert "lm_head_classifier_last_rows = lm_head_fused_graph_prewarm_last_rows" in source
    assert "g_lm_head_fused_graph_fallback_count" in tile_ops_source
    assert "store_lm_head_backward_thread_graph(key, exec)" in tile_ops_source
    assert "find_lm_head_backward_thread_graph(key)" in tile_ops_source
    assert "NFN_NATIVE_GPT_LM_HEAD_GRAPH_PREWARM_THREAD_CACHE" in tile_ops_source
    graph_body = tile_ops_source.split(
        "void launch_lm_head_classifier_backward_graph_body_bf16_u16",
        1,
    )[1].split(
        "int capture_lm_head_classifier_backward_graph_bf16_u16",
        1,
    )[0]
    sequence_body = tile_ops_source.split(
        "static int run_lm_head_classifier_backward_cooperative_sequence_bf16_u16(",
        1,
    )[1].split(
        "static int run_lm_head_classifier_backward_cooperative_sequence_bf16_u16_legacy_order",
        1,
    )[0]
    legacy_sequence_body = tile_ops_source.split(
        "static int run_lm_head_classifier_backward_cooperative_sequence_bf16_u16_legacy_order",
        1,
    )[1].split(
        "int nfn_native_tile_lm_head_classifier_backward_cooperative_bf16_u16",
        1,
    )[0]
    for cooperative_body in (graph_body, sequence_body, legacy_sequence_body):
        assert "launch_linear_backward_input_bf16_bits_weight_bf16_strided_float32" not in cooperative_body
        assert (
            "launch_linear_backward_weight_accumulate_bf16_bits_bf16_bits_strided_float32_beta"
            not in cooperative_body
        )
        assert "launch_linear_backward_input_bf16_bits_weight_bf16_float32" in cooperative_body
        assert "launch_linear_backward_weight_accumulate_bf16_bits_bf16_bits_float32_beta" in cooperative_body
    strict_fused_body = tile_ops_source.split(
        "int nfn_native_tile_lm_head_classifier_backward_fused_kernel_bf16_u16",
        1,
    )[1].split(
        "int nfn_native_tile_lm_head_classifier_backward_fused_kernel_is_true_fused",
        1,
    )[0]
    assert "g_lm_head_cooperative_sequence_launch_count.fetch_add" not in strict_fused_body
    assert "g_lm_head_fused_graph_fallback_count.fetch_add" not in strict_fused_body
    assert "lm_head_graph_local_stats().fallback_count += 1" in strict_fused_body
    assert "NFN_NATIVE_GPT_LM_HEAD_GRAPH_BODY_SERIAL" in tile_ops_source
    assert "diagnostic-cuda-graph-wrapper-serial-body" in tile_ops_source
    assert "return include_symbol_check ? (loaded && all_symbols && plan_passed) : false;" in source
    assert "const bool lm_head_cooperative_backward_route_integrated = false;" not in source
    bench_source = (root / "tools" / "bench_native_gpt_sm120_candidate.sh").read_text(
        encoding="utf-8"
    )
    assert '"lm_head_cooperative_backward"|"lm-head-cooperative-backward")' in bench_source
    assert '"lm_head_graph_prewarm"|"lm-head-graph-prewarm"' in bench_source
    assert '"lm_head_graph_prewarm_dedup"|"lm-head-graph-prewarm-dedup"' in bench_source
    assert "NFN_NATIVE_GPT_LM_HEAD_GRAPH_PREWARM_DEDUP=0" in bench_source
    assert "NFN_NATIVE_GPT_LM_HEAD_GRAPH_PREWARM_DEDUP=1" in bench_source
    assert "pointer-aware dedup key path" in bench_source
    assert "Equal-sized row chunks with different buffers are intentionally distinct keys" in bench_source
    assert "checks deterministic prewarm work rather than setup timing or route-change gates" in bench_source
    assert "FORCE_DISABLE_ROUTE_CHANGE=1" in bench_source
    assert "lm_head_fused_graph_prewarm_success_count=1.000" in bench_source
    assert '"lm_head_graph_thread_cache_prewarm"|"lm-head-graph-thread-cache-prewarm"' in bench_source
    assert '"lm_head_graph_upload_off"|"lm-head-graph-upload-off"' in bench_source
    assert "lm_head_fused_graph_upload_success_count from 3 to 0" in bench_source
    assert "1.001492x" in bench_source
    assert '"lm_head_graph_serial_body"|"lm-head-graph-serial-body"' in bench_source
    assert '"lm_head_true_fused_cooperative"|"lm-head-true-fused-cooperative"' in bench_source
    assert '"lm_head_true_fused_tile16"|"lm-head-true-fused-tile16"' in bench_source
    assert '"lm_head_true_fused_tile8"|"lm-head-true-fused-tile8"' in bench_source
    assert '"lm_head_true_fused_tile4"|"lm-head-true-fused-tile4"' in bench_source
    assert "NFN_NATIVE_GPT_LM_HEAD_GRAPH_UPLOAD=0" in bench_source
    assert "NFN_NATIVE_GPT_LM_HEAD_GRAPH_UPLOAD=1" in bench_source
    assert "NFN_NATIVE_GPT_LM_HEAD_GRAPH_BODY_SERIAL=1" in bench_source
    assert "NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE=1" in bench_source
    assert "NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE_ALLOW_PRODUCTION=1" in bench_source
    assert "-DNFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_MAT_TILE=16" in bench_source
    assert "-DNFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_MAT_TILE=8" in bench_source
    assert "-DNFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_MAT_TILE=4" in bench_source
    assert "LM_HEAD_BACKWARD_PREFLIGHT_PROFILE=\"trainer-chunk-true-fused\"" in bench_source
    assert "LM_HEAD_BACKWARD_PREFLIGHT_PROFILE=\"trainer-chunk-true-fused-tile16\"" in bench_source
    assert "LM_HEAD_BACKWARD_PREFLIGHT_PROFILE=\"trainer-chunk-true-fused-tile8\"" in bench_source
    assert "LM_HEAD_BACKWARD_PREFLIGHT_PROFILE=\"trainer-chunk-true-fused-tile4\"" in bench_source
    assert "NFN_SM120_NATIVE_LM_HEAD_BACKWARD_PREFLIGHT" in bench_source
    assert "NFN_SM120_NATIVE_LM_HEAD_BACKWARD_MAX_REFERENCE_GAP_MS" in bench_source
    assert "lm_head_backward_preflight_profile=" in bench_source
    assert "run_lm_head_backward_preflight" in bench_source
    assert "tools/bench_lm_head_backward_candidate.sh" in bench_source
    assert "lm_head_cooperative_sequence_wrapper" in bench_source
    assert "train_loop_wall_ms_per_step regressed to 1.012109x" in bench_source
    assert "stage.lm_head_backward.cooperative.total_ms regressed to 1.073406x" in bench_source
    assert "true fused/reference-aligned classifier-backward path" in bench_source
    assert "NFN_NATIVE_GPT_CE_BF16_THREADS=256" in bench_source
    assert "NFN_NATIVE_GPT_CE_BF16_THREADS=64" in bench_source
    assert "NFN_NATIVE_GPT_CE_BF16_THREADS=16" in bench_source
    assert "candidate_true_fused_cooperative_env=NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE=1" in bench_source
    assert (
        "candidate_true_fused_production_env="
        "NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE_ALLOW_PRODUCTION=1"
    ) in bench_source
    assert "focused trainer-chunk preflight proved the default 32x32 strict body" in bench_source
    assert "6.708146x candidate/current-wrapper" in bench_source
    assert "22.033921x candidate/reference-summed time" in bench_source
    assert "753.913597 ms slower than the reference CE+dHidden+dWeight components" in bench_source
    assert "full-loop gate also regressed train_loop_wall_ms_per_step to 5.991992x" in bench_source
    assert "stage.lm_head_backward.total_ms to 22.660619x" in bench_source
    assert "must remain rejected until it passes the promotion gate" in bench_source
    assert '"fast_startup"|"fast-startup"|"native_fast_startup"|"native-fast-startup"' in bench_source
    assert "NFN_NATIVE_GPT_FAST_STARTUP=1" in bench_source
    assert "NFN_NATIVE_GPT_FAST_STARTUP=0" in bench_source
    assert "setup_wall_ms=0.850" in bench_source
    assert (
        '"fast_startup_full"|"fast-startup-full"|"native_fast_startup_full"|'
        '"native-fast-startup-full"'
    ) in bench_source
    assert "setup_wall_ms to 0.655522x" in bench_source
    assert "train_loop_wall_ms_per_step regressed to 1.017654x" in bench_source
    assert "startup_plus_first_step_wall_ms=1.000" in bench_source
    assert "NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_GRAPH_PREWARM=1" in bench_source
    assert "NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_GRAPH_PREWARM=0" in bench_source
    assert 'ACCEPTED_CANDIDATE_PROFILE="$CANDIDATE_PROFILE"' in bench_source
    assert "post-MLP-FC-rollback rerun" in bench_source
    assert "0.985915x train_loop_wall_ms_per_step" in bench_source
    assert "0.999199x steady-state CUDA-event timing" in bench_source
    assert "0.957549x stage.lm_head_backward.total_ms" in bench_source
    assert "0.997858x stage.block_backward.total_ms" in bench_source
    assert "0.992403x stage.block_backward.mlp_proj.total_ms" in bench_source
    assert "graph capture attempts 3->0 and graph cache hits 45->48" in bench_source
    assert "train_loop_cuda_event_steady_state_wall_ms_per_step=1.002" in bench_source
    graph_prewarm_block = bench_source.split('"lm_head_graph_prewarm"|', 1)[1].split(
        "    ;;", 1
    )[0]
    assert "startup_plus_first_step_wall_ms=1.000" in graph_prewarm_block
    assert '"lm_head_cooperative_sequence_wrapper"|"lm-head-cooperative-sequence-wrapper"' in bench_source
    assert '"lm_head_cooperative_cublaslt"|"lm-head-cooperative-cublaslt"' in bench_source
    assert (
        "lm_head_dhidden_fast16bf_32768, lm_head_cooperative_cublaslt, lm_head_tk_dweight_32768"
        in bench_source
    )
    assert "1.335573x stage.lm_head_backward.total_ms" in bench_source
    assert "1.477219x stage.lm_head_backward.cooperative.total_ms" in bench_source
    assert '"lm_head_cooperative_loss_bins"|"lm-head-cooperative-loss-bins")' in bench_source
    assert "NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_BACKWARD=1" in bench_source
    assert "NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_CUDA_GRAPH=0" in bench_source
    assert "NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_LOSS_BINS=1" in bench_source
    assert "NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_CUBLASLT=1" in bench_source
    speed_tool = (root / "tools" / "paired_kernel_speed.py").read_text(encoding="utf-8")
    assert "stage.lm_head_backward.cooperative.total_ms" in speed_tool
    assert "lm_head_cooperative_backward_sequence_wrapper_enabled" in speed_tool
    assert "lm_head_classifier_backward_path_class" in speed_tool
    assert "lm_head_cooperative_backward_fused_kernel_abi_path_class" in speed_tool
    assert "lm_head_cooperative_backward_cuda_graph_enabled" in speed_tool
    assert "lm_head_cooperative_backward_graph_prewarm_enabled" in speed_tool
    assert "lm_head_prob_only_target_correction_threads" in speed_tool
    assert "lm_head_fused_graph_replay_success_count" in speed_tool
    assert "graph_replay_success_rate" in speed_tool
    assert "graph_capture_success_per_replay_mean" in speed_tool
    assert "graph_upload_success_per_replay_mean" in speed_tool
    assert "graph_prewarm_success_per_replay_mean" in speed_tool
    assert "lm_head_fused_graph_upload_success_count" in speed_tool
    assert "lm_head_fused_graph_upload_failure_count" in speed_tool
    assert "lm_head_fused_graph_thread_cache_hit_count" in speed_tool
    assert "lm_head_classifier_true_fused_launch_count" in speed_tool
    assert "lm_head_fused_graph_fallback_count" in speed_tool
    assert "lm_head_fused_graph_body_node_count_per_replay" in speed_tool
    assert "graph_body_total_node_replays_mean" in speed_tool
    assert "lm_head_fused_graph_body_ce_node_replay_total" in speed_tool
    assert "lm_head_fused_graph_body_dhidden_node_replay_total" in speed_tool
    assert "lm_head_fused_graph_body_dweight_node_replay_total" in speed_tool
    assert "lm_head_graph_body_cublaslt_dhidden_launch_count" in speed_tool
    assert "lm_head_graph_body_cublaslt_dweight_launch_count" in speed_tool
    assert "lm_head_graph_body_tile_dhidden_fallback_count" in speed_tool
    assert "lm_head_graph_body_tile_dweight_fallback_count" in speed_tool
    assert "lm_head_fused_graph_prewarm_success_count" in speed_tool
    assert "lm_head_fused_graph_prewarm_failure_count" in speed_tool
    assert "lm_head_fused_graph_prewarm_duplicate_skip_count" in speed_tool
    assert "lm_head_dhidden_strided_vocab_gemm_count" in speed_tool
    assert "lm_head_dweight_strided_vocab_gemm_count" in speed_tool
    assert "strict_true_fused_but_slow" in speed_tool
    assert "strict-true-fused-slow" in speed_tool
    assert "failed candidate/reference parity gates" in speed_tool


def test_native_gpt_lm_head_backward_microbench_compares_strict_symbol() -> None:
    root = Path(__file__).resolve().parents[1]
    bench_source = (
        root / "neuralfn" / "csrc" / "native_train" / "lm_head_backward_bench.cpp"
    ).read_text(encoding="utf-8")
    build_script = (root / "tools" / "build_lm_head_backward_bench.sh").read_text(
        encoding="utf-8"
    )
    wrapper = (root / "tools" / "bench_lm_head_backward_candidate.sh").read_text(
        encoding="utf-8"
    )
    assert "using LmHeadBackwardFn = int (*)(" in bench_source
    assert "nfn_native_tile_lm_head_classifier_backward_cooperative_bf16_u16" in bench_source
    assert "nfn_native_tile_lm_head_classifier_backward_cooperative_cublaslt_bf16_u16" in bench_source
    assert "nfn_native_tile_lm_head_classifier_backward_fused_kernel_bf16_u16" in bench_source
    assert "nfn_native_tile_lm_head_classifier_backward_fused_kernel_is_true_fused" in bench_source
    assert "reference_cublaslt_components" in bench_source
    assert "candidate_true_fused_capability" in bench_source
    assert "candidate_true_fused_production_shape" in bench_source
    assert "candidate_true_fused_allow_production_env" in bench_source
    assert "candidate_true_fused_forced_production_debug" in bench_source
    assert "candidate_true_fused_production_ready" in bench_source
    assert "true_fused_candidate_production_shape && true_fused_allow_production_env" in bench_source
    assert "candidate_path_class == \"strict-true-fused-tile-kernel\"" in bench_source
    assert "true_fused_allow_production_env_enabled" in bench_source
    assert "NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE_ALLOW_PRODUCTION" in bench_source
    assert "candidate_sequence_wrapper_only" in bench_source
    assert "candidate_strict_symbol_is_placeholder_sequence" in bench_source
    assert "candidate_cuda_graph_wrapper_only" in bench_source
    assert "candidate_path_class" in bench_source
    assert "true_fused_replacement_required" in bench_source
    assert "next_required_kernel_body" in bench_source
    assert "row-chunked-ce-dhidden-dweight-single-tile-kernel" in bench_source
    assert "strict experimental gate only" in bench_source
    assert "reference_classifier_fusion_scope" in bench_source
    assert "ce-dlogits-only-logits-dhidden-dweight-remain-separate" in bench_source
    assert "reference_alignment_target" in bench_source
    assert "next_reference_aligned_kernel_body" in bench_source
    assert "fused-ce-dlogits-separate-classifier-matmuls" in bench_source
    assert "next_required_symbol=" in bench_source
    assert "next_required_capability_symbol=" in bench_source
    assert "next_required_path_class=strict-true-fused-tile-kernel" in bench_source
    assert "next_required_kernel_body=" in bench_source
    assert "next_reference_aligned_kernel_body=" in bench_source
    assert "candidate_component_gap" in bench_source
    assert "candidate_to_reference_ce_ms_per_iter_ratio" in bench_source
    assert "candidate_to_reference_dhidden_ms_per_iter_ratio" in bench_source
    assert "candidate_to_reference_dweight_ms_per_iter_ratio" in bench_source
    assert "candidate_reference_gap" in bench_source
    assert "candidate_minus_reference_summed_ms_per_iter" in bench_source
    assert "candidate_minus_reference_summed_with_logits_ms_per_iter" in bench_source
    assert "candidate_minus_reference_cublaslt_summed_ms_per_iter" in bench_source
    assert "candidate_minus_reference_cublaslt_summed_with_logits_ms_per_iter" in bench_source
    assert "reference_bottleneck_component" in bench_source
    assert "reference_cublaslt_bottleneck_component" in bench_source
    assert "graph_replay_success_count > 0" in bench_source
    assert "ce_launch_count > 0" in bench_source
    assert "dhidden_launch_count > 0" in bench_source
    assert "dweight_launch_count > 0" in bench_source
    assert "candidate_to_baseline_ms_per_iter_ratio" in bench_source
    assert "candidate_to_reference_summed_ms_per_iter_ratio" in bench_source
    assert "candidate_to_reference_cublaslt_summed_ms_per_iter_ratio" in bench_source
    assert "candidate_to_reference_summed_with_logits_ms_per_iter_ratio" in bench_source
    assert "candidate_to_reference_cublaslt_summed_with_logits_ms_per_iter_ratio" in bench_source
    assert "int reference_component_warmup = -1" in bench_source
    assert "--reference-component-warmup" in bench_source
    assert "int effective_reference_component_warmup(const Options& options)" in bench_source
    assert "return std::max(1, options.warmup)" in bench_source
    assert '"  \\"reference_component_warmup\\": " << effective_reference_component_warmup(options)' in bench_source
    assert "label + \" warmup prepare synchronize\"" in bench_source
    assert "label + \" warmup synchronize\"" in bench_source
    assert "--no-loss" in bench_source
    assert '"  \\"no_loss\\": "' in bench_source
    assert "nfn_native_tile_lm_head_classifier_backward_inplace_strided_no_pad_zero_bf16_bits_u16_targets_with_workspace" in bench_source
    assert "reference_components" in bench_source
    assert "logits_ms_per_iter" in bench_source
    assert "ce_ms_per_iter" in bench_source
    assert "dhidden_ms_per_iter" in bench_source
    assert "dweight_ms_per_iter" in bench_source
    assert "summed_ms_per_iter" in bench_source
    assert "summed_with_logits_ms_per_iter" in bench_source
    assert "nfn_native_tile_linear_bf16_input_weight_bf16_output_float32" in bench_source
    assert (
        "nfn_native_tile_lm_head_classifier_backward_row_losses_inplace_strided_no_pad_zero_bf16_bits_u16_targets"
        in bench_source
    )
    assert "nfn_native_tile_linear_backward_input_bf16_bits_weight_bf16_strided_float32" in bench_source
    assert (
        "nfn_native_tile_linear_backward_weight_accumulate_bf16_bits_bf16_bits_strided_float32_beta"
        in bench_source
    )
    assert "nfn_native_tile_lm_head_cooperative_sequence_launch_count" in bench_source
    assert "nfn_native_tile_lm_head_cooperative_sequence_concurrent_count" in bench_source
    assert "nfn_native_tile_lm_head_cooperative_sequence_legacy_count" in bench_source
    assert "graph_capture_attempt_count" in bench_source
    assert "graph_capture_success_count" in bench_source
    assert "graph_cache_hit_count" in bench_source
    assert "graph_thread_cache_hit_count" in bench_source
    assert "graph_cache_entry_count" in bench_source
    assert "graph_replay_count" in bench_source
    assert "graph_replay_success_count" in bench_source
    assert "graph_fallback_count" in bench_source
    assert "nfn_native_tile_lm_head_fused_graph_capture_attempt_count" in bench_source
    assert "nfn_native_tile_lm_head_fused_graph_replay_success_count" in bench_source
    assert "nfn_native_tile_lm_head_fused_graph_fallback_count" in bench_source
    assert "cudaEventElapsedTime" in bench_source
    assert "timed_reset_between_iterations" in bench_source
    assert "cuda_check(cudaDeviceSynchronize(), name + \" warmup synchronize\");\n    }\n    reset_stats();" in bench_source
    assert "timed pre-reset logits memset" in bench_source
    assert "cudaEventRecord(start)" in bench_source
    assert "neuralfn/csrc/native_train/lm_head_backward_bench.cpp" in build_script
    assert "-lcudart -ldl" in build_script
    assert "NFN_LM_HEAD_BACKWARD_CANDIDATE_SYMBOL" in wrapper
    assert "NFN_LM_HEAD_BACKWARD_BASELINE_SYMBOL" in wrapper
    assert "NFN_LM_HEAD_BACKWARD_REFERENCE_COMPONENT_WARMUP" in wrapper
    assert 'BENCH_ARGS+=(--reference-component-warmup "${REFERENCE_COMPONENT_WARMUP}")' in wrapper
    assert "NFN_LM_HEAD_BACKWARD_CUDA_VISIBLE_DEVICES" in wrapper
    assert "select_auto_cuda_device" in wrapper
    assert "nvidia-smi --query-gpu=index,display_active,utilization.gpu" in wrapper
    assert "if ! query_output=\"$(nvidia-smi --query-gpu=index,display_active,utilization.gpu" in wrapper
    assert "printf '%s\\n' \"0\"" in wrapper
    assert '"auto"|"dedicated"|"dedicated-auto")' in wrapper
    assert 'export CUDA_VISIBLE_DEVICES="${SELECTED_CUDA_VISIBLE_DEVICE}"' in wrapper
    assert "NFN_LM_HEAD_BACKWARD_PROFILE" in wrapper
    assert "trainer-chunk|trainer_chunk" in wrapper
    assert "trainer-chunk-strict|trainer_chunk_strict" in wrapper
    assert "trainer-chunk-true-fused|trainer_chunk_true_fused" in wrapper
    assert "true-fused-cooperative-smoke|true_fused_cooperative_smoke" in wrapper
    assert "strict-true-fused-smoke|strict_true_fused_smoke" in wrapper
    assert "NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE" in wrapper
    assert "NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE_ALLOW_PRODUCTION" in wrapper
    assert "Production-shape focused strict true-fused LM-head profile" in wrapper
    assert "DEFAULT_MAX_RATIO=1.000" in wrapper
    assert "DEFAULT_MAX_REFERENCE_RATIO=1.000" in wrapper
    assert "DEFAULT_MAX_CUBLASLT_REFERENCE_RATIO=1.000" in wrapper
    assert "candidate/current-wrapper and candidate/reference parity" in wrapper
    assert 'MAX_RATIO="${NFN_LM_HEAD_BACKWARD_MAX_RATIO:-${DEFAULT_MAX_RATIO:-}}"' in wrapper
    assert 'MAX_REFERENCE_RATIO="${NFN_LM_HEAD_BACKWARD_MAX_REFERENCE_RATIO:-${DEFAULT_MAX_REFERENCE_RATIO:-}}"' in wrapper
    assert 'MAX_CUBLASLT_REFERENCE_RATIO="${NFN_LM_HEAD_BACKWARD_MAX_CUBLASLT_REFERENCE_RATIO:-${DEFAULT_MAX_CUBLASLT_REFERENCE_RATIO:-}}"' in wrapper
    assert "DRY_RUN_ENV_PREFIX" in wrapper
    assert "HIDDEN_DIM=8" in wrapper
    assert "VOCAB=16" in wrapper
    assert "ROW_STRIDE=16" in wrapper
    assert "trainer-chunk-cublaslt|trainer_chunk_cublaslt" in wrapper
    assert "REJECTED_PROFILE=\"${PROFILE}\"" in wrapper
    assert "NFN_LM_HEAD_BACKWARD_ALLOW_REJECTED_PROFILE" in wrapper
    assert "candidate/baseline ratio 1.466890" in wrapper
    assert "DEFAULT_REQUIRE_TRUE_FUSED=1" in wrapper
    assert "trainer-row-loss|trainer_row_loss" in wrapper
    assert "trainer-row-loss-cublaslt|trainer_row_loss_cublaslt" in wrapper
    assert "trainer-loss-bins|trainer_loss_bins" in wrapper
    assert "DEFAULT_CANDIDATE_SYMBOL" in wrapper
    assert "BASELINE_SYMBOL_OVERRIDE" in wrapper
    assert "CANDIDATE_SYMBOL_OVERRIDE" in wrapper
    assert "DEFAULT_NO_LOSS=1" in wrapper
    assert "NFN_LM_HEAD_BACKWARD_NO_LOSS" in wrapper
    assert "NFN_LM_HEAD_BACKWARD_MAX_RATIO" in wrapper
    assert "NFN_LM_HEAD_BACKWARD_MAX_REFERENCE_RATIO" in wrapper
    assert "NFN_LM_HEAD_BACKWARD_MAX_REFERENCE_WITH_LOGITS_RATIO" in wrapper
    assert "NFN_LM_HEAD_BACKWARD_MAX_CUBLASLT_REFERENCE_RATIO" in wrapper
    assert "NFN_LM_HEAD_BACKWARD_MAX_CUBLASLT_REFERENCE_WITH_LOGITS_RATIO" in wrapper
    assert "NFN_LM_HEAD_BACKWARD_MAX_REFERENCE_GAP_MS" in wrapper
    assert "NFN_LM_HEAD_BACKWARD_MAX_REFERENCE_WITH_LOGITS_GAP_MS" in wrapper
    assert "NFN_LM_HEAD_BACKWARD_MAX_CUBLASLT_REFERENCE_GAP_MS" in wrapper
    assert "NFN_LM_HEAD_BACKWARD_MAX_CUBLASLT_REFERENCE_WITH_LOGITS_GAP_MS" in wrapper
    assert "check_json_ratio" in wrapper
    assert "check_json_gap" in wrapper
    assert "NFN_LM_HEAD_BACKWARD_REQUIRE_TRUE_FUSED" in wrapper
    assert "REQUIRE_TRUE_FUSED_ARG=(--require-true-fused-candidate)" in wrapper
    assert "NFN_LM_HEAD_BACKWARD_CANDIDATE_FIRST" in wrapper
    assert "NFN_LM_HEAD_BACKWARD_DRY_RUN" in wrapper
    assert "CANDIDATE_FIRST_ARG=(--candidate-first)" in wrapper
    assert "BENCH_ARGS=(" in wrapper
    assert "printf '%q' \"${BENCH_BIN}\"" in wrapper
    assert "BENCH_DEPS=(" in wrapper
    assert '"${ROOT_DIR}/neuralfn/csrc/native_train/lm_head_backward_bench.cpp"' in wrapper
    assert '"${ROOT_DIR}/neuralfn/csrc/native_train/tile_ops.h"' in wrapper
    assert 'if [[ "${DEP}" -nt "${BENCH_BIN}" ]]; then' in wrapper
    assert "candidate strict symbol is still sequencing CE/dHidden/dWeight" in wrapper
    assert "candidate strict symbol is a CUDA Graph wrapper around CE/dHidden/dWeight" in wrapper
    assert "candidate_true_fused_capability is false" in wrapper
    assert "candidate.true_fused_launch_count is zero" in wrapper
    assert "candidate_reference_gap:" in wrapper
    assert "candidate_minus_reference_summed_ms_per_iter" in wrapper
    assert "reference_bottleneck_component" in wrapper
    assert "next_required_symbol" in wrapper
    assert "next_required_capability_symbol" in wrapper
    assert "next_required_path_class" in wrapper
    assert "next_required_kernel_body" in wrapper
    assert "emit_true_fused_requirement_message" in wrapper
    assert "LM-head true-fused replacement required" in wrapper
    assert "BENCH_STATUS=$?" in wrapper
    assert "candidate_to_baseline_ms_per_iter_ratio" in wrapper
    assert "candidate_first" in bench_source
    assert "--candidate-first" in bench_source
    assert "--require-true-fused-candidate" in bench_source
    assert "require_true_fused_candidate" in bench_source
    assert "candidate strict symbol is not a true fused" in bench_source
    assert '\\"run_order\\": \\"' in bench_source
    assert "candidate-first" in bench_source
    assert "baseline-first" in bench_source
    assert "tools/build_native_train_tile_ops.sh" in wrapper
    assert "TILE_OPS_DEPS=(" in wrapper
    assert '"${ROOT_DIR}/neuralfn/csrc/native_train/tile_ops.cu"' in wrapper
    assert '"${ROOT_DIR}/neuralfn/csrc/native_train/tile_ops.h"' in wrapper
    assert '"${ROOT_DIR}/neuralfn/csrc/tile_cuda/kernels.cu"' in wrapper
    assert '"${ROOT_DIR}/tools/build_native_train_tile_ops.sh"' in wrapper
    assert 'if [[ "${DEP}" -nt "${TILE_OPS_LIB}" ]]; then' in wrapper
    assert "--candidate-symbol" in wrapper
    assert "--baseline-symbol" in wrapper


def test_native_gpt_lm_head_true_fused_gate_rejects_slow_strict_kernel() -> None:
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "paired_kernel_speed_for_test",
        root / "tools" / "paired_kernel_speed.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    target = module.summarize_lm_head_true_fused_target(
        {
            "candidate_native_metrics": {
                "lm_head_classifier_true_fused_launch_count": {"mean": 16.0},
            },
            "candidate_native_metric_values": {
                "lm_head_classifier_backward_path_class": [
                    "strict-true-fused-tile-kernel"
                ],
                "lm_head_cooperative_backward_fused_kernel_capability_available": [
                    "true"
                ],
                "lm_head_cooperative_backward_fused_kernel_symbol_available": [
                    "true"
                ],
            },
            "candidate_reference_metric_ratio_gates": {
                "enabled": True,
                "passed": False,
            },
        }
    )
    assert target["required"] is True
    assert target["status"] == "strict-true-fused-slow"
    assert "failed candidate/reference parity gates" in target["reason"]

    gate = module.evaluate_lm_head_true_fused_gate(required=True, target=target)
    assert gate["passed"] is False
    assert gate["status"] == "strict-true-fused-slow"


def test_native_gpt_linear_backward_microbench_profiles_block_and_lm_head_shapes() -> None:
    root = Path(__file__).resolve().parents[1]
    bench_source = (
        root / "neuralfn" / "csrc" / "native_train" / "linear_backward_bench.cpp"
    ).read_text(encoding="utf-8")
    build_script = (root / "tools" / "build_linear_backward_bench.sh").read_text(
        encoding="utf-8"
    )
    wrapper = (root / "tools" / "bench_linear_backward_candidate.sh").read_text(
        encoding="utf-8"
    )
    assert "using LinearDinputStridedFn = int (*)(" in bench_source
    assert "using LinearDweightStridedFn = int (*)(" in bench_source
    assert "nfn_native_tile_linear_backward_input_bf16_bits_weight_bf16_strided_float32" in (
        bench_source
    )
    assert "nfn_native_tile_linear_backward_weight_accumulate_bf16_bits_bf16_bits_strided_float32_beta" in (
        bench_source
    )
    tile_ops_header = (root / "neuralfn" / "csrc" / "native_train" / "tile_ops.h").read_text(
        encoding="utf-8"
    )
    tile_ops_source = (root / "neuralfn" / "csrc" / "native_train" / "tile_ops.cu").read_text(
        encoding="utf-8"
    )
    kernels_source = (root / "neuralfn" / "csrc" / "tile_cuda" / "kernels.cu").read_text(
        encoding="utf-8"
    )
    assert (
        "nfn_native_tile_linear_backward_input_bf16_bits_weight_bf16_strided_cublaslt_float32"
        in tile_ops_header
    )
    assert (
        "nfn_native_tile_linear_backward_weight_accumulate_bf16_bits_bf16_bits_strided_cublaslt_float32_beta"
        in tile_ops_header
    )
    assert "cudaErrorNotSupported" in tile_ops_source
    assert "cublaslt_linear_backward_input_bf16_bits_weight_bf16_strided_float32" in (
        kernels_source
    )
    assert (
        "cublaslt_linear_backward_weight_accumulate_bf16_bits_bf16_bits_strided_float32_beta"
        in kernels_source
    )
    assert "linear_backward_tile_ops" in bench_source
    assert "candidate_to_baseline_ms_per_iter_ratio" in bench_source
    assert "candidate_symbol_changed" in bench_source
    assert "baseline.symbol != candidate.symbol" in bench_source
    assert "cudaEventElapsedTime" in bench_source
    assert "timed_reset_between_iterations" in bench_source
    assert "candidate_first" in bench_source
    assert "--candidate-first" in bench_source
    assert '\\"run_order\\": \\"' in bench_source
    assert "candidate-first" in bench_source
    assert "baseline-first" in bench_source
    assert "neuralfn/csrc/native_train/linear_backward_bench.cpp" in build_script
    assert "-lcudart -ldl" in build_script
    assert "NFN_LINEAR_BACKWARD_PROFILE" in wrapper
    assert "NFN_LINEAR_BACKWARD_CANDIDATE_SYMBOL" in wrapper
    assert "NFN_LINEAR_BACKWARD_BASELINE_SYMBOL" in wrapper
    assert "NFN_LINEAR_BACKWARD_CANDIDATE_FIRST" in wrapper
    assert "CANDIDATE_FIRST_ARGS=(--candidate-first)" in wrapper
    assert "NFN_LINEAR_BACKWARD_CUDA_VISIBLE_DEVICES" in wrapper
    assert "select_auto_cuda_device" in wrapper
    assert "if ! query_output=\"$(nvidia-smi --query-gpu=index,display_active,utilization.gpu" in wrapper
    assert "printf '%s\\n' \"0\"" in wrapper
    assert '"auto"|"dedicated"|"dedicated-auto")' in wrapper
    assert 'export CUDA_VISIBLE_DEVICES="${SELECTED_CUDA_VISIBLE_DEVICE}"' in wrapper
    assert "mlp-proj-dinput|mlp_proj_dinput" in wrapper
    assert "mlp-fc-dweight|mlp_fc_dweight" in wrapper
    assert "qkv-dinput|qkv_dinput" in wrapper
    assert "attn-proj-dweight|attn_proj_dweight" in wrapper
    assert "lm-head-dinput|lm_head_dinput" in wrapper
    assert "lm-head-dinput-cublaslt|lm_head_dinput_cublaslt" in wrapper
    assert "lm-head-dweight|lm_head_dweight" in wrapper
    assert "lm-head-dweight-cublaslt|lm_head_dweight_cublaslt" in wrapper
    assert "DEFAULT_CANDIDATE_SYMBOL" in wrapper
    assert "strided_cublaslt_float32" in wrapper
    assert "DEFAULT_ROWS=65536" in wrapper
    assert "DEFAULT_ROWS=32768" in wrapper
    assert "NFN_LINEAR_BACKWARD_MAX_RATIO" in wrapper
    assert "NFN_LINEAR_BACKWARD_REQUIRE_ROUTE_CHANGE" in wrapper
    assert "candidate_symbol_changed is false; candidate and baseline symbols are identical" in wrapper
    assert "BENCH_DEPS=(" in wrapper
    assert '"${ROOT_DIR}/neuralfn/csrc/native_train/linear_backward_bench.cpp"' in wrapper
    assert '"${ROOT_DIR}/neuralfn/csrc/native_train/tile_ops.h"' in wrapper
    assert 'if [[ "${DEP}" -nt "${BENCH_BIN}" ]]; then' in wrapper
    assert "tools/build_native_train_tile_ops.sh" in wrapper
    assert "TILE_OPS_DEPS=(" in wrapper
    assert '"${ROOT_DIR}/neuralfn/csrc/native_train/tile_ops.cu"' in wrapper
    assert '"${ROOT_DIR}/neuralfn/csrc/native_train/tile_ops.h"' in wrapper
    assert '"${ROOT_DIR}/neuralfn/csrc/tile_cuda/kernels.cu"' in wrapper
    assert '"${ROOT_DIR}/tools/build_native_train_tile_ops.sh"' in wrapper
    assert 'if [[ "${DEP}" -nt "${TILE_OPS_LIB}" ]]; then' in wrapper
    assert "--grad-out-row-stride" in wrapper
    matrix_wrapper = (root / "tools" / "bench_native_gpt_linear_hot_matrix.sh").read_text(
        encoding="utf-8"
    )
    assert "DEFAULT_PROFILES=(" in matrix_wrapper
    for profile in (
        "mlp-proj-dinput",
        "mlp-proj-dweight",
        "mlp-fc-dinput",
        "mlp-fc-dweight",
        "qkv-dinput",
        "qkv-dweight",
        "attn-proj-dinput",
        "attn-proj-dweight",
        "lm-head-dinput",
        "lm-head-dweight",
    ):
        assert profile in matrix_wrapper
    assert "NFN_LINEAR_HOT_MATRIX_PROFILES" in matrix_wrapper
    assert "NFN_LINEAR_HOT_DINPUT_CANDIDATE_SYMBOL" in matrix_wrapper
    assert "NFN_LINEAR_HOT_DWEIGHT_CANDIDATE_SYMBOL" in matrix_wrapper
    assert "NFN_LINEAR_HOT_${PROFILE_ENV}_CANDIDATE_SYMBOL" in matrix_wrapper
    assert "NFN_LINEAR_BACKWARD_CANDIDATE_SYMBOL=${CANDIDATE_SYMBOL}" in matrix_wrapper
    assert "NFN_LINEAR_BACKWARD_MAX_RATIO=${MAX_RATIO}" in matrix_wrapper
    assert "NFN_LINEAR_HOT_MATRIX_REQUIRE_ROUTE_CHANGE" in matrix_wrapper
    assert "NFN_LINEAR_BACKWARD_REQUIRE_ROUTE_CHANGE=1" in matrix_wrapper
    assert "candidate_symbol_changed_count" in matrix_wrapper
    assert "same_symbol_profile_count" in matrix_wrapper
    assert "measurement_only_profile_count" in matrix_wrapper
    assert "route_change_required" in matrix_wrapper
    assert "route_change_passed" in matrix_wrapper
    assert "route_change_failure_reason" in matrix_wrapper
    assert "hot-matrix comparisons must use a distinct candidate symbol" in matrix_wrapper
    assert "native_gpt_linear_hot_matrix" in matrix_wrapper
    assert "max_candidate_to_baseline_ms_per_iter_ratio" in matrix_wrapper
    assert "mean_candidate_to_baseline_ms_per_iter_ratio" in matrix_wrapper
    assert "bash \"${LINEAR_WRAPPER}\"" in matrix_wrapper


def test_native_gpt_cuda_error_35_reports_runtime_visibility_hint() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "neuralfn" / "csrc" / "native_gpt2" / "nfn_gpt2_native_train.cpp").read_text(
        encoding="utf-8"
    )
    assert "void append_cuda_error_message(" in source
    assert "code == 35" in source
    assert "CUDA runtime/driver mismatch or blocked GPU device access" in source
    assert "verify unsandboxed nvidia-smi" in source
    assert "--cuda-runtime-lib/NFN_CUDA_RUNTIME_LIB" in source
    assert source.count("append_cuda_error_message(out, code, cuda_get_error_string(code));") >= 10


def test_native_sm120_candidate_wrapper_covers_attention_and_ordering_profiles() -> None:
    root = Path(__file__).resolve().parents[1]
    bench_source = (root / "tools" / "bench_native_gpt_sm120_candidate.sh").read_text(
        encoding="utf-8"
    )
    speed_source = (root / "tools" / "paired_kernel_speed.py").read_text(encoding="utf-8")
    assert "NATIVE_HOT_SUMMARY_METRIC_KEYS" in speed_source
    assert "float_arena_cuda_malloc_wall_ms" in speed_source
    assert "float_arena_pointer_assign_wall_ms" in speed_source
    assert "uint16_arena_cuda_malloc_wall_ms" in speed_source
    assert "uint16_arena_pointer_assign_wall_ms" in speed_source
    assert "token_weight_bf16_padding_memset_count" in speed_source
    assert "uint16_arena_first_enabled" in speed_source
    assert "arena_materialize_order" in speed_source
    assert "transformer_device_arena_cuda_malloc_wall_ms" in speed_source
    assert "transformer_device_arena_pointer_assign_wall_ms" in speed_source
    assert "setup.float_uint16_arena_materialize_concurrent.total_ms" in speed_source
    assert "concurrent_arena_materialize_requested" in speed_source
    assert "concurrent_arena_materialize_enabled" in speed_source
    assert "concurrent_arena_materialize_count" in speed_source
    assert "baseline_env: {json.dumps(baseline_env, sort_keys=True)}" in speed_source
    assert "candidate_env: {json.dumps(candidate_env, sort_keys=True)}" in speed_source

    expected_profiles = {
        "bf16_attention_grad_out": "NFN_NATIVE_GPT_BF16_ATTENTION_GRAD_OUT=1",
        "bf16_attention_dprep_grad_out": "NFN_NATIVE_GPT_BF16_ATTENTION_DPREP_GRAD_OUT=1",
        "attention_dprep_grid3d": "NFN_NATIVE_GPT_PACKED_ATTENTION_DPREP_GRID3D=1",
        "attention_dprep_float_hd64_specialized": "NFN_NATIVE_GPT_PACKED_ATTENTION_DPREP_FLOAT_HD64_SPECIALIZED=1",
        "packed_attention_saved_lse_off": "NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LSE=0",
        "mlp_proj_dinput_before_dweight": "NFN_NATIVE_GPT_MLP_PROJ_DINPUT_BEFORE_DWEIGHT=1",
        "mlp_proj_concurrent_dinput_dweight": "NFN_NATIVE_GPT_BLOCK_MLP_PROJ_CONCURRENT_DINPUT_DWEIGHT=1",
        "mlp_fc_dinput_before_dweight": "NFN_NATIVE_GPT_MLP_FC_DINPUT_BEFORE_DWEIGHT=1",
        "attn_proj_dinput_before_dweight": "NFN_NATIVE_GPT_ATTN_PROJ_DINPUT_BEFORE_DWEIGHT=1",
        "qkv_dinput_before_dweight": "NFN_NATIVE_GPT_QKV_DINPUT_BEFORE_DWEIGHT=1",
        "qkv_dinput_ln128": "NFN_NATIVE_GPT_QKV_DINPUT_BEFORE_DWEIGHT=1 NFN_NATIVE_GPT_LAYERNORM_AFFINE_ROW_CHUNK_SIZE=128",
        "qkv_dinput_ln64": "NFN_NATIVE_GPT_QKV_DINPUT_BEFORE_DWEIGHT=1 NFN_NATIVE_GPT_LAYERNORM_AFFINE_ROW_CHUNK_SIZE=64",
        "lm_head_fused_loss_backward_off": "NFN_NATIVE_GPT_LM_HEAD_FUSED_LOSS_BACKWARD=0",
        "lm_head_classifier_ce_no_loss": "NFN_NATIVE_GPT_LM_HEAD_CLASSIFIER_CE_NO_LOSS=1",
        "cuda_module_eager": "CUDA_MODULE_LOADING=EAGER",
        "lm_head_prepack_bf16_hidden_off": "NFN_NATIVE_GPT_LM_HEAD_PREPACK_BF16_HIDDEN=0",
        "lm_head_tk_dweight_49152": "NFN_NATIVE_LINEAR_TK_DWEIGHT_ENABLE_SHAPE=768,50304,49152,N,T",
        "mlp_proj_tk_dweight_65536": "NFN_NATIVE_LINEAR_TK_DWEIGHT_ENABLE_SHAPE=3072,768,65536,N,T",
        "fused_ln2_bf16_out_off": "NFN_NATIVE_GPT_FUSE_LN2_BF16_OUT=0",
        "mlp_residual_next_ln1_off": "NFN_NATIVE_GPT_FUSE_MLP_RESIDUAL_NEXT_LN1=0",
        "block_split_bgrad_65536": "NFN_NATIVE_LINEAR_BF16_BF16_BGRAD_DISABLE_SHAPE=768,2304,65536,N,T:768,768,65536,N,T:768,3072,65536,N,T:3072,768,65536,N,T",
        "mlp_proj_split_bgrad_65536": "NFN_NATIVE_LINEAR_BF16_BF16_BGRAD_DISABLE_SHAPE=3072,768,65536,N,T",
        "linear_bias_row_chunk_256": "NFN_NATIVE_GPT_LINEAR_BACKWARD_BIAS_ROW_CHUNK_SIZE=256",
        "linear_bias_row_chunk_1024": "NFN_NATIVE_GPT_LINEAR_BACKWARD_BIAS_ROW_CHUNK_SIZE=1024",
        "linear_bias_threads_512": "NFN_NATIVE_GPT_LINEAR_BACKWARD_BIAS_THREADS=512",
        "lm_head_loss_bins": "NFN_NATIVE_GPT_LM_HEAD_LOSS_BIN_REDUCTION=1",
        "lm_head_loss_bins_bf16_workspace_prewarm": (
            "NFN_NATIVE_GPT_LM_HEAD_LOSS_BIN_REDUCTION=1 "
            "NFN_NATIVE_GPT_PREWARM_BF16_WORKSPACE=1"
        ),
        "lm_head_ce_loss_bins_default_specialized": "NFN_NATIVE_GPT_LM_HEAD_CE_LOSS_BINS_DEFAULT_SPECIALIZED=1",
        "lm_head_ce_llmk_style_specialized": "NFN_NATIVE_GPT_LM_HEAD_CE_LLMK_STYLE_SPECIALIZED=1",
        "lm_head_prob_only_corrections": "NFN_NATIVE_GPT_LM_HEAD_PROB_ONLY_CORRECTIONS=1",
        "lm_head_prob_only_combined_corrections": "NFN_NATIVE_GPT_LM_HEAD_PROB_ONLY_COMBINED_CORRECTIONS=1",
        "lm_head_prob_only_ce_target_corrections": "NFN_NATIVE_GPT_LM_HEAD_PROB_ONLY_CE_TARGET_CORRECTIONS=1",
        "lm_head_prob_only_combined_corrections_threads_512": "NFN_NATIVE_GPT_LM_HEAD_PROB_ONLY_TARGET_CORRECTION_THREADS=512",
        "lm_head_ce_no_loss_llmk_style_specialized": "NFN_NATIVE_GPT_LM_HEAD_CE_NO_LOSS_LLMK_STYLE_SPECIALIZED=1",
        "lm_head_ce_no_loss_vec8_normal_store_specialized": "NFN_NATIVE_GPT_LM_HEAD_CE_NO_LOSS_VEC8_NORMAL_STORE_SPECIALIZED=1",
        "lm_head_ce_loss_bins_llmk_style_specialized": "NFN_NATIVE_GPT_LM_HEAD_LOSS_BIN_REDUCTION=1 NFN_NATIVE_GPT_LM_HEAD_CE_LLMK_STYLE_SPECIALIZED=1",
        "bf16_persistent_block_outputs6": "NFN_NATIVE_GPT_BF16_PERSISTENT_BLOCK_OUTPUT_COUNT=6",
        "bf16_persistent_block_outputs_last6": "NFN_NATIVE_GPT_BF16_PERSISTENT_BLOCK_OUTPUT_PLACEMENT=tail",
        "lm_head_row_chunk_65536": "NFN_NATIVE_GPT_ALLOW_UNSAFE_LM_HEAD_ROW_CHUNK=1",
        "lm_head_row_chunk_16384": "--lm-head-row-chunk-size 16384",
        "cublaslt_plan_prewarm_block_only": "NFN_NATIVE_GPT_PREWARM_CUBLASLT_PLAN_MODE=block_only",
        "cublaslt_plan_prewarm_lm_head_only": "NFN_NATIVE_GPT_PREWARM_CUBLASLT_PLAN_MODE=lm_head_only",
        "cublaslt_plan_prewarm_off": "NFN_NATIVE_GPT_PREWARM_CUBLASLT_PLANS=0",
        "cublaslt_heavy_shape_flip": "NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=3072,768,65536,N,T,1:768,3072,65536,N,T,1:768,65536,3072,N,N,0:768,65536,2304,N,N,0:768,2304,65536,N,T,0",
        "cublaslt_block_dinput": "NFN_NATIVE_LINEAR_BF16_CUBLASLT_ENABLE_SHAPE=3072,65536,768,N,N:768,65536,3072,N,N:768,65536,2304,N,N:768,65536,768,N,N",
        "cublaslt_block_dinput_h3_65536": "NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE=768,65536,3072,N,N,3:768,65536,2304,N,N,3",
        "lm_head_public_vocab_strided_gemm": "NFN_NATIVE_GPT_LM_HEAD_PUBLIC_VOCAB_STRIDED_GEMM=1",
        "packed_attention_bwd_batch_96": "NFN_NATIVE_GPT_PACKED_ATTENTION_BACKWARD_BATCH_CAP=96",
        "cublaslt_grouped_probe_required": "NFN_NATIVE_GPT_PROBE_CUBLASLT_GROUPED_LAYOUT=1 NFN_NATIVE_GPT_PROBE_CUBLASLT_GROUPED_MATMUL=1",
        "lm_head_row_loss_sum_accumulate": "NFN_NATIVE_GPT_LM_HEAD_ROW_LOSS_SUM_ACCUMULATE=1",
        "lm_head_row_loss_partial_reduce": "NFN_NATIVE_GPT_LM_HEAD_ROW_LOSS_SUM_ACCUMULATE=0",
        "lm_head_cooperative_no_loss_backward": "NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_BACKWARD=1 NFN_NATIVE_GPT_LM_HEAD_CLASSIFIER_CE_NO_LOSS=1 NFN_NATIVE_GPT_LM_HEAD_CE_NO_LOSS_DEFAULT_SPECIALIZED=1",
        "adamw_token_shadow_refresh": "NFN_NATIVE_GPT_FUSE_TOKEN_WEIGHT_BF16_ADAMW_REFRESH=1",
        "combined_device_arena": "NFN_NATIVE_GPT_COMBINED_DEVICE_ARENA=1",
        "cuda_malloc_async": "NFN_NATIVE_GPT_CUDA_MALLOC_ASYNC=1",
        "lm_head_cooperative_sequence_wrapper": (
            "NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_BACKWARD=1 "
            "NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_CUDA_GRAPH=0 "
            "NFN_NATIVE_GPT_LM_HEAD_FORCE_SEQUENCE_WRAPPER_DIAGNOSTIC=1"
        ),
        "lm_head_cooperative_backward_off": "NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_BACKWARD=0",
        "concurrent_arena_materialize": "NFN_NATIVE_GPT_CONCURRENT_ARENA_MATERIALIZE=1",
        "uint16_arena_first": "NFN_NATIVE_GPT_UINT16_ARENA_FIRST=1",
        "store_mlp_blocks3": "NFN_NATIVE_GPT_STORE_MLP_BLOCKS=3",
        "store_mlp_blocks6": "NFN_NATIVE_GPT_STORE_MLP_BLOCKS=6",
        "store_mlp_blocks9": "NFN_NATIVE_GPT_STORE_MLP_BLOCKS=9",
        "store_mlp_blocks11": "NFN_NATIVE_GPT_STORE_MLP_BLOCKS=11",
        "store_mlp_blocks6_tail": "NFN_NATIVE_GPT_STORE_MLP_BLOCK_PLACEMENT=tail",
        "store_packed_attention_blocks6": "NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_BLOCKS=6",
        "store_packed_attention_blocks6_tail": "NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_BLOCK_PLACEMENT=tail",
        "store_packed_attention_ln1_bf16_off": "NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LN1_BF16=0",
        "store_residual1_off": "NFN_NATIVE_GPT_STORE_RESIDUAL1_ACTIVATIONS=0",
        "full_activation_tape": "NFN_NATIVE_GPT_FULL_ACTIVATION_TAPE=1",
        "bgrad_first_write_direct": "NFN_NATIVE_GPT_BGRAD_FIRST_WRITE_DIRECT=1",
        "bgrad_first_write_direct_qkv_65536": "NFN_NATIVE_LINEAR_BGRAD_FIRST_WRITE_DIRECT_ENABLE_SHAPE=768,2304,65536,N,T",
        "bgrad_first_write_direct_attn_proj_65536": "NFN_NATIVE_LINEAR_BGRAD_FIRST_WRITE_DIRECT_ENABLE_SHAPE=768,768,65536,N,T",
        "bgrad_first_write_direct_mlp_fc_65536": "NFN_NATIVE_LINEAR_BGRAD_FIRST_WRITE_DIRECT_ENABLE_SHAPE=768,3072,65536,N,T",
        "bgrad_first_write_direct_mlp_proj_65536": "NFN_NATIVE_LINEAR_BGRAD_FIRST_WRITE_DIRECT_ENABLE_SHAPE=3072,768,65536,N,T",
    }
    for profile, env_assignment in expected_profiles.items():
        assert profile in bench_source
        assert env_assignment in bench_source
    for profile in (
        "cublaslt_plan_prewarm_block_only",
        "cublaslt_plan_prewarm_lm_head_only",
        "cublaslt_plan_prewarm_off",
    ):
        profile_block = bench_source.split(f'"{profile}"|', 1)[1].split("    ;;", 1)[0]
        assert "train_loop_cuda_event_first_step_wall_ms_per_step=1.000" in profile_block
        assert "startup_plus_first_step_wall_ms=1.000" in profile_block
    for profile in ("combined_device_arena", "cuda_malloc_async"):
        profile_block = bench_source.split(f'"{profile}"|', 1)[1].split("    ;;", 1)[0]
        assert "STARTUP_ONLY=1" in profile_block
        assert "STEPS=0" in profile_block
        assert "FORCE_DISABLE_ROUTE_CHANGE=1" in profile_block
    assert "setup_wall_ms to 3.467504x" in bench_source
    assert "first-step CUDA-event timing to 1.048444x" in bench_source
    assert "mixed-fp32-direct-output-plus-fused-bf16-persistent-store" in (
        root / "neuralfn" / "csrc" / "native_gpt2" / "nfn_gpt2_native_train.cpp"
    ).read_text(encoding="utf-8")
    gpt2_source = (
        root / "neuralfn" / "csrc" / "native_gpt2" / "nfn_gpt2_native_train.cpp"
    ).read_text(encoding="utf-8")
    assert "bf16_persistent_block_output_count" in gpt2_source
    assert "stored_mlp_activation_block_placement" in gpt2_source
    assert "stored_packed_attention_block_placement" in gpt2_source
    assert "stored_mlp_activation_for" in gpt2_source
    assert "stored_packed_attention_for" in gpt2_source
    assert "train_loop_cuda_event_steady_state_wall_ms_per_step=1.002" in bench_source
    assert "kept 512 as the default" in bench_source
    assert "stage.block_backward.mlp_fc.dweight_bias.total_ms=0.972707x" in bench_source
    assert "block_backward_mlp_proj_dinput_before_dweight_count moved 0->288" in bench_source
    assert "stage.block_backward.mlp_proj.dinput.total_ms regressed to 1.101843x" in bench_source
    assert "PUBLIC_VOCAB_STRIDED_GEMM" in bench_source
    assert "NFN_NATIVE_GPT_LM_HEAD_PROB_ONLY_TARGET_CORRECTION_THREADS=256" in bench_source
    assert "NFN_NATIVE_GPT_LM_HEAD_PROB_ONLY_TARGET_CORRECTION_THREADS=512" in bench_source
    assert "NFN_NATIVE_GPT_LM_HEAD_CE_NO_LOSS_VEC8_NORMAL_STORE_SPECIALIZED=0" in bench_source
    assert "STRICT_GROUPED_CUBLASLT_PROBE=1" in bench_source
    assert "required cuBLASLt grouped probe failed" in bench_source
    assert "grouped layout and grouped matmul probe statuses are both 0" in bench_source
    assert 'data.get("candidate_native_metrics", {})' in bench_source
    assert "linear_cublaslt_grouped_matmul_probe_status" in bench_source
    assert "stage.lm_head_backward.dhidden.total_ms=1.000" in bench_source
    assert "stage.lm_head_backward.dweight.total_ms=1.000" in bench_source
    assert (
        'BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_LOSS_BIN_REDUCTION=0"'
        in bench_source
    )
    assert "CANDIDATE_NOTE=" in bench_source
    assert "keeps the loss-bin train-loss logging route as the default" in bench_source
    assert "--lm-head-row-chunk-size 65536" in bench_source
    assert "changed lm_head_classifier_last_rows from 32768 to 65536" in bench_source
    assert "train_loop_wall_ms_per_step regressed to 7.368793x" in bench_source
    assert '"lm_head_row_chunk_16384"|"lm-head-row-chunk-16384"' in bench_source
    assert "halved lm_head_bf16_logit_bytes to 0.500000x" in bench_source
    assert "setup_wall_ms to 0.931740x" in bench_source
    assert "train_loop_wall_ms_per_step regressed to 1.001777x" in bench_source
    assert "startup_plus_first_step_wall_ms=1.000" in bench_source
    assert "8.037520x versus llm.kittens" in bench_source
    assert "stage.block_backward.attn_sdpa.to_qkv.total_ms collapsed to 63.207371x" in bench_source
    assert "changed attention_backward_tk_batch_cap from 64 to 96" in bench_source
    assert "stage.block_backward.attn_sdpa.to_qkv.total_ms to 1.000638x" in bench_source
    assert "candidate-over-llm.kittens train_loop_wall_ms_per_step remained 1.035674x" in bench_source
    assert "train_loop_wall_ms_per_step to 0.981541x" in bench_source
    assert "stage.block_backward.total_ms to 0.999905x" in bench_source
    assert (
        'COMMON_EXTRA_ARGS_RAW="${COMMON_EXTRA_ARGS_RAW:+$COMMON_EXTRA_ARGS_RAW }--train-loss-every-steps 1"'
        in bench_source
    )
    assert (
        'BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_CLASSIFIER_CE_NO_LOSS=0"'
        in bench_source
    )
    assert (
        'BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_ATTN_PROJ_DINPUT_BEFORE_DWEIGHT=0"'
        in bench_source
    )
    assert (
        'BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_LINEAR_BACKWARD_BIAS_ROW_CHUNK_SIZE=512"'
        in bench_source
    )
    assert "The Tile-CUDA default remains 256" in bench_source
    assert "1.002081x steady-state CUDA-event step time" in bench_source
    assert "1.000470x stage.block_backward.mlp_fc.dweight_bias.total_ms" in bench_source
    assert '"cublaslt_heavy_shape_flip"|"cublaslt-heavy-shape-flip"' in bench_source
    assert "The candidate proved plan-cache and linear-shape changes" in bench_source
    assert "steady-state CUDA-event timing to 1.005491x" in bench_source
    assert "MLP FC backward to 1.031422x" in bench_source
    assert "full-final-norm BF16 prepack to per-chunk BF16 packing" in bench_source
    assert "stage.lm_head_backward.total_ms to 1.055161x" in bench_source
    assert (
        'BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_FUSE_LN2_BF16_OUT=1"'
        in bench_source
    )
    assert (
        'BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_FUSE_MLP_RESIDUAL_NEXT_LN1=1"'
        in bench_source
    )
    assert (
        'BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_LM_HEAD_ROW_LOSS_SUM_ACCUMULATE=1"'
        in bench_source
    )
    assert '"tk_dgelu_dinput"|"tk-dgelu-dinput")' in bench_source
    assert '"tk_dgelu_approx_tanh"|"tk-dgelu-approx-tanh")' in bench_source
    assert "baseline already reports linear_tk_dgelu_dinput_gemm_count=288" in bench_source
    assert "The tanh approximation variant is therefore historical/diagnostic-only" in bench_source
    assert "-DLLMK_SM120_USE_TK_FUSED_DGELU_DINP" in bench_source
    assert "-DLLMK_SM120_APPROX_DGELU_TANH=1" in bench_source
    assert '"mlp_proj_dgelu_fallback"|"mlp-proj-dgelu-fallback"' in bench_source
    assert "NFN_NATIVE_LINEAR_TK_DGELU_DINPUT_DISABLE_SHAPE=3072,65536,768,N,N" in bench_source
    assert "linear_tk_dgelu_dinput_gemm_count dropping" in bench_source
    assert "MLP projection dInput to 1.207964x" in bench_source
    assert '"tk_forward_no_n96"|"tk-forward-no-n96"|"llmk_forward_no_n96"|"llmk-forward-no-n96")' in bench_source
    assert "stage.lm_head_backward.total_ms=1.001484x" in bench_source
    assert "stage.block_backward.mlp_proj.total_ms=1.001994x" in bench_source
    assert "setup_wall_ms to 0.884111x" in bench_source
    assert "MLP projection to 1.546830x" in bench_source
    assert "setup_wall_ms to 0.958626x" in bench_source
    assert "backward_recompute_blocks 11->0" in bench_source
    assert (
        'BASELINE_ENV_RAW="${BASELINE_ENV_RAW:+$BASELINE_ENV_RAW }NFN_NATIVE_GPT_FULL_ACTIVATION_TAPE=0"'
        in bench_source
    )
    assert "full_activation_tape" in bench_source
    assert "no_recompute" in bench_source
    assert "activation_tape_count" in speed_source
    assert '("activation_tape_count", ("block_state_layout", "activation_tape_count"))' in speed_source
    assert "full_activation_tape_enabled" in speed_source
    assert '("block_state_layout", "full_activation_tape_enabled")' in speed_source
    assert "backward_recompute_blocks" in speed_source
    assert (
        '("backward_recompute_blocks", ("block_state_layout", "backward_recompute_blocks"))'
        in speed_source
    )
    assert "activation_tape_strategy" in speed_source
    assert "attention dprep timing to 1.000231x" in bench_source
    assert "AUTO_ATTENTION_SECTION_TIMING=1" in bench_source
    assert "grouped layout status 0 with grouped matmul status 15" in bench_source
    assert "NFN_NATIVE_GPT_STORE_RESIDUAL1_ACTIVATIONS=1" in bench_source
    assert "stored_mlp_activation_blocks" in speed_source
    assert "stored_packed_attention_activation_blocks" in speed_source
    assert "stored_packed_attention_ln1_bf16_blocks" in speed_source
    assert "stored_residual1_activation_blocks" in speed_source
    assert '"llmk_sm120_reference_flags"|"llmk-sm120-reference-flags"' in bench_source
    assert "-DLLMK_SM120_DWEIGHT_SUPER_M=2" in bench_source
    assert "-DLLMK_SM120_FAST_DGELU=1" in bench_source
    assert "-DLLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1" in bench_source
    assert "2026-06-28 3-step, 2-sample, stage-timed rerun" in bench_source
    assert "candidate-over-llm.kittens gates at train_loop_wall_ms_per_step=0.999113x" in bench_source
    assert "no hot route counters" in bench_source
    assert "no cuBLASLt plan-cache entries" in bench_source
    assert "flat/slightly slower versus the current linked native baseline" in bench_source
    assert "FORCE_DISABLE_ROUTE_CHANGE=1" in bench_source
    assert "Known profiles:" in bench_source
    assert "mlp_proj_concurrent_dinput_dweight" in bench_source
    assert (
        "cublaslt_plan_prewarm_block_only, "
        "cublaslt_plan_prewarm_lm_head_only, "
        "cublaslt_plan_prewarm_off"
    ) in bench_source
    assert '"tk_sm120_super_m7"|"tk-sm120-super-m7"' in bench_source
    assert "-DLLMK_SM120_SUPER_M=7 -DLLMK_SM120_DINP_SUPER_M=7" in bench_source
    assert "strategy telemetry changed super_m and dinput_super_m from 8 to 7" in bench_source
    assert "steady-state CUDA-event timing regressed to 1.000992x" in bench_source
    assert '"tk_sm120_super_m13"|"tk-sm120-super-m13"' in bench_source
    assert '"tk_qkv_forward_prewarm_32768"|"tk-qkv-forward-prewarm-32768"' in bench_source
    assert "NFN_NATIVE_GPT_PREWARM_TK_QKV_FORWARD_ROWS=32768" in bench_source
    assert "setup_wall_ms to 0.961917x" in bench_source
    assert "train_loop_wall_ms_per_step to 1.002107x" in bench_source
    assert "-DLLMK_SM120_SUPER_M=13 -DLLMK_SM120_DINP_SUPER_M=13" in bench_source
    assert "strategy telemetry changed super_m and dinput_super_m from 8 to 13" in bench_source
    assert "block backward to 1.011813x" in bench_source
    assert "CUDA 13.3 RTX 5090 same-script gate moved 192 MLP projection dWeight calls to TK" in bench_source
    assert "lm_head_only_candidate_gate=1" in bench_source
    assert 'if [[ "$lm_head_only_candidate_gate" != "1" ]]; then' in bench_source
    assert "SKIP_LM_HEAD_CE_STAGE_GATE=0" in bench_source
    assert "SKIP_LM_HEAD_CE_STAGE_GATE=1" in bench_source
    assert 'if [[ "$SKIP_LM_HEAD_CE_STAGE_GATE" != "1" ]]; then' in bench_source

    tile_source = (root / "neuralfn" / "csrc" / "tile_cuda" / "kernels.cu").read_text(
        encoding="utf-8"
    )
    assert "LinearShapeList parse_linear_shape_list" in tile_source
    assert "current != ':' && current != ';'" in tile_source
    assert "return linear_shape_list_matches(disabled_shapes, m, n, k, op_a, op_b);" in tile_source
    assert "CublasLtHeuristicShapeOverrideList parse_cublaslt_heuristic_shape_override_list" in tile_source
    assert "parse_cublaslt_heuristic_shape_override_token(token_start, &shape)" in tile_source

    for gated_metric in [
        "stage.block_backward.total_ms=1.000",
        "stage.block_backward.attn_sdpa.total_ms=1.000",
        "stage.block_backward.attn_sdpa.to_qkv.total_ms=1.000",
        "stage.block_backward.mlp_proj.dinput.total_ms=1.000",
        "stage.block_backward.mlp_proj.dweight_bias.total_ms=1.000",
        "stage.block_backward.mlp_fc.dweight_bias.total_ms=1.000",
        "stage.block_backward.mlp_fc.total_ms=1.000",
        "stage.block_backward.attn_proj.total_ms=1.000",
        "stage.lm_head_backward.ce.total_ms=1.000",
    ]:
        assert gated_metric in bench_source
    assert "*LM_HEAD_CE_NO_LOSS_DEFAULT_SPECIALIZED*|*lm_head_ce_no_loss_default_specialized*" not in bench_source
    assert "lm_head_logits_bf16_fallback_32768" in bench_source
    assert "moved lm_head_logits_tk_gemm_count from 48 to 0" in bench_source
    assert "1.003097x train_loop_wall_ms_per_step" in bench_source
    assert "AUTO_ATTENTION_SECTION_TIMING=1" in bench_source
    assert '--baseline-env "NFN_NATIVE_GPT_ATTENTION_BACKWARD_SECTION_TIMING=1"' in bench_source
    assert '--candidate-env "NFN_NATIVE_GPT_ATTENTION_BACKWARD_SECTION_TIMING=1"' in bench_source
    assert "block_state_layout.linear_backward_bias_row_chunk_size" in speed_source
    assert "fused_ln2_bf16_out_enabled" in speed_source
    assert "stored_mlp_forward_strategy" in speed_source
    assert "stored_mlp_ln2_bf16_prepack_strategy" in speed_source
    assert "block_state_layout.mlp_residual_next_ln1_fusion_count" in speed_source
    assert "attention_backward_float_hd64_dprep_launch_count" in speed_source


def test_build_native_gpt_compiled_cli_config_defaults_to_universal_gpt(tmp_path: Path) -> None:
    cfg = build_native_gpt_compiled_cli_run_config(
        dataset_alias="cached-shards",
        executable="/opt/nfn/train_gpt2cu",
        output_dir=tmp_path / "gpt",
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
        template_name="gpt2_megakernel",
        graph_file="/tmp/custom-gpt.json",
    )

    argv = cfg.compiled_cli_argv("/opt/nfn/nfn_gpt_native_train")

    assert isinstance(cfg, NativeGptRunConfig)
    assert type(cfg).__name__ == "NativeGptRunConfig"
    assert repr(cfg).startswith("NativeGptRunConfig(")
    assert cfg.model_family == "gpt"
    assert cfg.template_name == "gpt2_megakernel"
    assert cfg.graph_file == "/tmp/custom-gpt.json"
    assert argv[:3] == ["/opt/nfn/nfn_gpt_native_train", "--model-family", "gpt"]
    assert argv[argv.index("--template-name") + 1] == "gpt2_megakernel"
    assert argv[argv.index("--graph-file") + 1] == "/tmp/custom-gpt.json"
    assert normalize_native_gpt_encoding_name("tokgpt2") == "gpt2"
    assert native_gpt_encoding_vocab_size("gpt2") == 50257
    assert native_gpt_activation("sd_prelu") == "sd-prelu"
    assert native_gpt_kernel_backend("tile-cuda") == "tile-cuda"


def test_build_native_gpt2_compiled_cli_config_canonicalizes_dense_gpt_family(tmp_path: Path) -> None:
    cfg = build_native_gpt2_compiled_cli_run_config(
        dataset_alias="cached-shards",
        executable="/opt/nfn/train_gpt2cu",
        output_dir=tmp_path / "gpt3",
        eval_every_steps=1000,
        sample_every_steps=20000,
        generate_tokens=144,
        checkpoint_every_steps=200,
        batch_size=64,
        seq_len=2048,
        train_batch_tokens=524288,
        learning_rate=0.0006,
        min_lr=None,
        warmup_steps=60,
        weight_decay=0.1,
        max_steps=20000,
        num_layers=12,
        activation="gelu",
        model_family="gpt3",
    )

    argv = cfg.compiled_cli_argv("/opt/nfn/nfn_gpt_native_train")

    assert cfg.model_family == "gpt"
    assert argv[argv.index("--model-family") + 1] == "gpt"
    assert argv[argv.index("--train-seq-len") + 1] == "2048"

    nanogpt_cfg = build_native_gpt_compiled_cli_run_config(
        dataset_alias="cached-shards",
        executable="/opt/nfn/train_gpt2cu",
        output_dir=tmp_path / "nanogpt",
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
        model_family="nanogpt",
    )

    nanogpt_argv = nanogpt_cfg.compiled_cli_argv("/opt/nfn/nfn_gpt_native_train")

    assert isinstance(nanogpt_cfg, NativeGptRunConfig)
    assert nanogpt_cfg.model_family == "gpt"
    assert nanogpt_cfg.template_name == "nanogpt"
    assert nanogpt_argv[nanogpt_argv.index("--model-family") + 1] == "gpt"
    assert nanogpt_argv[nanogpt_argv.index("--template-name") + 1] == "nanogpt"

    nanogpt_graph_cfg = build_native_gpt2_compiled_cli_run_config(
        dataset_alias="cached-shards",
        executable="/opt/nfn/train_gpt2cu",
        output_dir=tmp_path / "nanogpt-graph",
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
        model_family="nanogpt",
        graph_file="/tmp/custom-gpt.json",
    )

    assert isinstance(nanogpt_graph_cfg, NativeGpt2RunConfig)
    assert nanogpt_graph_cfg.model_family == "gpt"
    assert nanogpt_graph_cfg.template_name == "gpt"
    assert nanogpt_graph_cfg.graph_file == "/tmp/custom-gpt.json"

    nano_gpt_alias_cfg = build_native_gpt2_compiled_cli_run_config(
        dataset_alias="cached-shards",
        executable="/opt/nfn/train_gpt2cu",
        output_dir=tmp_path / "nano-gpt-alias",
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
        model_family="nano_gpt",
    )

    assert nano_gpt_alias_cfg.model_family == "gpt"
    assert nano_gpt_alias_cfg.template_name == "nanogpt"


def test_train_gpt_native_direct_wrapper_accepts_nanogpt_selector(capsys) -> None:
    module = _load_train_gpt_native_script_module()

    rc = module.main(
        [
            "--model-family",
            "nanogpt",
            "--dataset-alias",
            "/tmp/native-cache",
            "--native-cuda-dry-run",
            "--native-cuda-print-command",
            "--eval-every-steps",
            "1000",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 0
    assert "Native CUDA model family: gpt" in captured.out
    assert "Native CUDA template: nanogpt" in captured.out
    assert "--model-family gpt" in captured.out
    assert "--template-name nanogpt" in captured.out
    assert "--dataset-alias /tmp/native-cache" in captured.out
    assert "--eval-every-steps 1000" in captured.out
    assert "--train-transformer-lm" in captured.out
    assert "--base-model" not in captured.out


def test_build_native_gpt2_compiled_cli_config_maps_gpt2_moa_template_to_native_activation(tmp_path: Path) -> None:
    cfg = build_native_gpt2_compiled_cli_run_config(
        dataset_alias="/tmp/native-cache",
        executable="/opt/nfn/train_gpt2cu",
        output_dir=tmp_path / "gpt2-moa",
        eval_every_steps=250,
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
        template_name="gpt2_moa",
    )

    argv = cfg.compiled_cli_argv("/opt/nfn/nfn_gpt2_native_train")

    assert cfg.template_name == "gpt2_moa"
    assert cfg.activation == "moa"
    assert argv[argv.index("--template-name") + 1] == "gpt2_moa"
    assert argv[argv.index("--native-cuda-activation") + 1] == "moa"


def test_write_native_gpt2_run_config_includes_command(tmp_path: Path) -> None:
    cfg = NativeGpt2RunConfig(
        executable="train_gpt2cu",
        train_data="train.bin",
        val_data="val.bin",
        output_dir="out",
        model_descriptor="d12",
        eval_every_steps=250,
        sample_every_steps=20000,
        generate_tokens=144,
        checkpoint_every_steps=200,
        batch_size=64,
        seq_len=1024,
        train_batch_tokens=524288,
        learning_rate=0.0006,
        final_lr_fraction=0.0,
        warmup_steps=60,
        weight_decay=0.1,
        max_steps=20000,
    )
    output = tmp_path / "native.json"

    write_native_gpt2_run_config(cfg, output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["argv"][0] == "train_gpt2cu"
    assert "train_gpt2cu -i train.bin -j val.bin" in payload["command"]
    assert payload["runner"]["requested"] == "auto"


def test_native_gpt2_activation_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="Unsupported native GPT activation"):
        native_gpt2_activation("not-real")


def test_native_gpt2_kernel_backend_rejects_unknown_value(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="kernel backend"):
        build_native_gpt2_compiled_cli_run_config(
            dataset_alias="cached-shards",
            executable="/opt/nfn/train_gpt2cu",
            output_dir=tmp_path / "out",
            eval_every_steps=250,
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
            kernel_backend="external",
        )
    with pytest.raises(ValueError, match="kernel backend"):
        build_native_gpt2_compiled_cli_run_config(
            dataset_alias="cached-shards",
            executable="/opt/nfn/train_gpt2cu",
            output_dir=tmp_path / "out",
            eval_every_steps=250,
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
            kernel_backend="tile_cuda",
        )


def test_native_gpt_compiled_cli_exposes_template_catalog_action() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "neuralfn" / "csrc" / "native_gpt2" / "nfn_gpt2_native_train.cpp").read_text(
        encoding="utf-8"
    )

    assert "--list-templates" in source
    assert "print_template_catalog_json" in source
    assert '\\"action\\": \\"list_templates\\"' in source
    assert '\\"token_shards_resolved\\": false' in source
    assert "selected_graph_support_status" in source
    assert "selected_graph_native_runnable" in source


def test_native_gpt_compiled_cli_lists_template_catalog_when_built() -> None:
    root = Path(__file__).resolve().parents[1]
    cli = root / "build" / "nfn_gpt_native_train"
    if not cli.exists():
        pytest.skip("build/nfn_gpt_native_train is not built")

    proc = subprocess.run(
        [str(cli), "--list-templates"],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["action"] == "list_templates"
    assert payload["token_shards_resolved"] is False
    assert payload["shipped_template_catalog_count"] == len(SHIPPED_GPT_TEMPLATE_PRESETS)
    assert payload["selector_count"] == len(SHIPPED_GPT_TEMPLATE_PRESETS) + 2
    statuses = {item["name"]: item["selected_graph_support_status"] for item in payload["templates"]}
    assert statuses["gpt"] == "native-transformer-lm"
    assert statuses["gpt2"] == "native-transformer-lm"
    assert statuses["gpt3"] == "native-transformer-lm"
    assert statuses["nanogpt"] == "native-transformer-lm"
    assert statuses["semantic_router_moe"] == "template-native-trainer-missing"


def test_native_gpt2_runner_status_auto_requires_neuralfn_native_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NFN_NATIVE_GPT2_BINDING", "0")
    monkeypatch.setenv("NFN_NATIVE_GPT2_CLI", str(tmp_path / "missing-native-cli"))
    monkeypatch.setenv("NFN_NATIVE_GPT2_LAUNCHER", str(tmp_path / "missing-launcher"))
    status = native_gpt2_runner_status("auto")

    assert status.requested == "auto"
    assert status.resolved == "compiled-cli"
    assert status.available is False
    assert "binding unavailable" in status.reason
    assert "compiled native GPT CLI/launcher not found" in status.reason
    assert "subprocess" not in status.reason


def test_native_gpt2_runner_status_uses_compiled_cli_when_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NFN_NATIVE_GPT2_BINDING", "0")
    cli = tmp_path / "nfn_gpt2_native_train"
    cli.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    cli.chmod(0o755)
    monkeypatch.setenv("NFN_NATIVE_GPT2_CLI", str(cli))

    explicit = native_gpt2_runner_status("compiled-cli")
    automatic = native_gpt2_runner_status("auto")

    assert resolve_native_gpt2_cli() == str(cli)
    assert explicit.resolved == "compiled-cli"
    assert explicit.available is True
    with pytest.raises(ValueError, match="compiled-cli"):
        native_gpt2_runner_status("cli")
    assert automatic.resolved == "compiled-cli"
    assert automatic.available is True


def test_native_gpt_generic_env_names_take_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generic_cli = tmp_path / "nfn_gpt_native_train"
    legacy_cli = tmp_path / "nfn_gpt2_native_train"
    for cli in (generic_cli, legacy_cli):
        cli.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        cli.chmod(0o755)

    monkeypatch.setenv("NFN_NATIVE_GPT_BINDING", "0")
    monkeypatch.setenv("NFN_NATIVE_GPT2_BINDING", "1")
    monkeypatch.setenv("NFN_NATIVE_GPT_CLI", str(generic_cli))
    monkeypatch.setenv("NFN_NATIVE_GPT2_CLI", str(legacy_cli))
    monkeypatch.setenv("NFN_NATIVE_GPT_TRAIN_BIN", "/opt/nfn/train_gpt")
    monkeypatch.setenv("NFN_NATIVE_GPT2_TRAIN_BIN", "/opt/nfn/train_gpt2cu")

    assert resolve_native_gpt2_cli() == str(generic_cli)
    assert resolve_native_gpt2_executable() == "/opt/nfn/train_gpt"
    status = native_gpt2_runner_status("auto")
    assert status.resolved == "compiled-cli"
    assert "NFN_NATIVE_GPT_BINDING" in status.reason
    generic_status = native_gpt_runner_status("auto")
    assert isinstance(generic_status, NativeGptRunnerStatus)
    assert type(generic_status).__name__ == "NativeGptRunnerStatus"
    assert generic_status.resolved == "compiled-cli"


def test_native_gpt_template_catalog_action_reaches_wrappers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[1]
    native_cli = tmp_path / "nfn_gpt_native_train"
    args_file = tmp_path / "native-args.txt"
    native_cli.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\" > \"$NFN_TEST_NATIVE_ARGS\"\n"
        "exit 17\n",
        encoding="utf-8",
    )
    native_cli.chmod(0o755)
    monkeypatch.setenv("NFN_NATIVE_GPT_CLI", str(native_cli))
    monkeypatch.setenv("NFN_TEST_NATIVE_ARGS", str(args_file))
    env = os.environ.copy()

    nfn_proc = subprocess.run(
        [
            sys.executable,
            str(root / "cli" / "nfn.py"),
            "train",
            "--base-model",
            "gpt",
            "--list-templates",
        ],
        cwd=root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert nfn_proc.returncode == 17
    nfn_args = args_file.read_text(encoding="utf-8").splitlines()
    assert "--model-family" in nfn_args
    assert "--list-templates" in nfn_args
    assert "--train-transformer-lm" not in nfn_args

    gpt_proc = subprocess.run(
        [
            sys.executable,
            str(root / "cli" / "scripts" / "train_gpt.py"),
            "--native-cuda-list-templates",
        ],
        cwd=root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert gpt_proc.returncode == 17
    gpt_args = args_file.read_text(encoding="utf-8").splitlines()
    assert "--list-templates" in gpt_args
    assert "--train-transformer-lm" not in gpt_args
    assert "--dataset-alias" not in gpt_args
    assert "--dataset-path" not in gpt_args
    assert "--tinystories" not in gpt_args
    assert "--eval-batches" not in gpt_args


def test_native_gpt2_runner_status_uses_compiled_launcher_when_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NFN_NATIVE_GPT2_BINDING", "0")
    monkeypatch.setenv("NFN_NATIVE_GPT2_CLI", str(tmp_path / "missing-native-cli"))
    launcher = tmp_path / "nfn_gpt2_tile_train"
    generic_sm120_launcher = tmp_path / "nfn_train_gpt"
    launcher.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    launcher.chmod(0o755)
    monkeypatch.setenv("NFN_NATIVE_GPT2_LAUNCHER", str(launcher))

    explicit = native_gpt2_runner_status("launcher")
    automatic = native_gpt2_runner_status("auto")

    assert resolve_native_gpt2_launcher() == str(launcher)
    assert explicit.resolved == "launcher"
    assert explicit.available is True
    assert automatic.resolved == "launcher"
    assert automatic.available is True


def test_native_gpt2_binding_runner_invokes_in_process_module(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_run(payload: dict[str, object]) -> int:
        calls.append(payload)
        return 17

    def fake_resolve(payload: dict[str, object]) -> list[str]:
        return ["native-gpt2-binding", str(payload["train_data"])]

    monkeypatch.delitem(sys.modules, "neuralfn_native_gpt", raising=False)
    monkeypatch.delitem(sys.modules, "neuralfn._native_gpt", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "neuralfn_native_gpt2",
        SimpleNamespace(run_gpt2=fake_run, resolve_native_gpt2_command=fake_resolve),
    )
    monkeypatch.setattr(native_gpt2_module, "NATIVE_GPT2_BINDING_MODULES", ("neuralfn_native_gpt2",))
    monkeypatch.setattr(native_gpt2_module, "resolve_cuda_visible_devices_value", lambda value: "7")
    cfg = NativeGpt2RunConfig(
        executable="train_gpt2cu",
        train_data="train.bin",
        val_data="val.bin",
        output_dir="out",
        model_descriptor="d12",
        eval_every_steps=250,
        sample_every_steps=20000,
        generate_tokens=144,
        checkpoint_every_steps=200,
        batch_size=64,
        seq_len=1024,
        train_batch_tokens=524288,
        learning_rate=0.0006,
        final_lr_fraction=0.0,
        warmup_steps=60,
        weight_decay=0.1,
        max_steps=20000,
    )

    assert native_gpt2_runner_status("auto").resolved == "binding"
    assert run_native_gpt2(cfg, runner="binding") == 17
    assert calls[0]["train_data"] == "train.bin"
    assert calls[0]["cuda_visible_devices"] == "7"
    assert calls[0]["cuda_device_max_connections"] == "1"


def test_native_gpt2_checkpoint_sampler_prefers_binding_capture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model_00000001.bin"
    checkpoint.write_bytes(b"placeholder")
    calls: list[dict[str, object]] = []

    def fake_run(payload: dict[str, object]) -> int:
        raise AssertionError("sampler should use capture runner, not training runner")

    def fake_capture(payload: dict[str, object]) -> dict[str, object]:
        calls.append(payload)
        return {"returncode": 0, "stdout": '{"generated_tokens": [42]}\\n', "stderr": ""}

    def fake_resolve(payload: dict[str, object]) -> list[str]:
        return list(payload["compiled_cli_argv"])  # type: ignore[index]

    monkeypatch.delitem(sys.modules, "neuralfn_native_gpt", raising=False)
    monkeypatch.delitem(sys.modules, "neuralfn._native_gpt", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "neuralfn_native_gpt2",
        SimpleNamespace(
            run_gpt2=fake_run,
            run_gpt2_capture=fake_capture,
            resolve_native_gpt2_command=fake_resolve,
        ),
    )
    monkeypatch.setattr(native_gpt2_module, "NATIVE_GPT2_BINDING_MODULES", ("neuralfn_native_gpt2",))
    monkeypatch.setenv("NFN_NATIVE_GPT2_CLI", "/opt/nfn/nfn_gpt_native_train")

    result = run_native_gpt2_checkpoint_sampler(
        checkpoint,
        prompt_tokens="1,2,3",
        max_new_tokens=4,
        cuda_visible_devices="2",
        runner="binding",
    )

    assert result.returncode == 0
    assert result.stdout == '{"generated_tokens": [42]}\\n'
    assert calls[0]["compiled_cli_argv"] == [
        "/opt/nfn/nfn_gpt_native_train",
        "--sample-checkpoint",
        str(checkpoint),
        "--prompt-tokens",
        "1,2,3",
        "--max-new-tokens",
        "4",
        "--temperature",
        "0.8",
        "--top-k",
        "32",
        "--repetition-penalty",
        "1.0",
        "--seed",
        "1337",
    ]
    assert calls[0]["cuda_visible_devices"] == "2"
    assert calls[0]["cuda_device_max_connections"] == "1"


def test_native_gpt2_binding_runner_errors_when_explicit_and_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NFN_NATIVE_GPT2_BINDING", "0")
    status = native_gpt2_runner_status("binding")

    assert status.resolved == "binding"
    assert status.available is False
    assert status.reason


def test_native_gpt2_launcher_runner_executes_launcher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = tmp_path / "nfn_gpt2_tile_train"
    launcher.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\" > \"$NFN_TEST_LAUNCHER_ARGS\"\n"
        "exit 23\n",
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    output = tmp_path / "launcher-args.txt"
    monkeypatch.setenv("NFN_NATIVE_GPT2_LAUNCHER", str(launcher))
    monkeypatch.setenv("NFN_TEST_LAUNCHER_ARGS", str(output))
    cfg = NativeGpt2RunConfig(
        executable="/opt/nfn/train_gpt2cu",
        train_data="train.bin",
        val_data="val.bin",
        output_dir="out",
        model_descriptor="d12",
        eval_every_steps=250,
        sample_every_steps=20000,
        generate_tokens=144,
        checkpoint_every_steps=200,
        batch_size=64,
        seq_len=1024,
        train_batch_tokens=524288,
        learning_rate=0.0006,
        final_lr_fraction=0.0,
        warmup_steps=60,
        weight_decay=0.1,
        max_steps=20000,
    )

    assert run_native_gpt2(cfg, runner="launcher") == 23
    args = output.read_text(encoding="utf-8").splitlines()
    assert args[:4] == ["--target", "/opt/nfn/train_gpt2cu", "--", "-i"]


def test_native_gpt2_compiled_cli_runner_executes_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = tmp_path / "nfn_gpt2_native_train"
    cli.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\" > \"$NFN_TEST_NATIVE_CLI_ARGS\"\n"
        "printf 'CUDA_VISIBLE_DEVICES=%s\\nCUDA_DEVICE_MAX_CONNECTIONS=%s\\nCUDA_MODULE_LOADING=%s\\n' "
        "\"$CUDA_VISIBLE_DEVICES\" \"$CUDA_DEVICE_MAX_CONNECTIONS\" \"$CUDA_MODULE_LOADING\" > \"$NFN_TEST_NATIVE_CLI_ENV\"\n"
        "exit 19\n",
        encoding="utf-8",
    )
    cli.chmod(0o755)
    output = tmp_path / "native-cli-args.txt"
    env_output = tmp_path / "native-cli-env.txt"
    monkeypatch.setenv("NFN_NATIVE_GPT2_CLI", str(cli))
    monkeypatch.setenv("NFN_TEST_NATIVE_CLI_ARGS", str(output))
    monkeypatch.setenv("NFN_TEST_NATIVE_CLI_ENV", str(env_output))
    cfg = NativeGpt2RunConfig(
        executable="/opt/nfn/train_gpt2cu",
        train_data=str(tmp_path / "dataset" / "fineweb_train_000000.bin"),
        val_data=str(tmp_path / "dataset" / "fineweb_val_000000.bin"),
        output_dir="out",
        model_descriptor="d12",
        eval_every_steps=250,
        sample_every_steps=20000,
        generate_tokens=144,
        checkpoint_every_steps=200,
        batch_size=64,
        seq_len=1024,
        train_batch_tokens=524288,
        learning_rate=0.0006,
        final_lr_fraction=0.0,
        warmup_steps=60,
        weight_decay=0.1,
        max_steps=20000,
    )

    assert run_native_gpt2(cfg, runner="compiled-cli") == 19
    args = output.read_text(encoding="utf-8").splitlines()
    assert args[:6] == [
        "--model-family",
        "gpt",
        "--dataset-alias",
        str(tmp_path / "dataset"),
        "--backend",
        "tile-cuda",
    ]
    assert "--target" not in args
    assert "--train-transformer-lm" in args
    assert "--eval-every-steps" in args
    assert "--final-lr-fraction" in args
    env_lines = env_output.read_text(encoding="utf-8").splitlines()
    assert "CUDA_VISIBLE_DEVICES=0" in env_lines
    assert "CUDA_DEVICE_MAX_CONNECTIONS=1" in env_lines
    assert "CUDA_MODULE_LOADING=LAZY" in env_lines


def test_native_gpt_exec_handoff_uses_compiled_cli_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = tmp_path / "nfn_gpt_native_train"
    cli.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    cli.chmod(0o755)
    monkeypatch.setenv("NFN_NATIVE_GPT_CLI", str(cli))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_DEVICE_MAX_CONNECTIONS", raising=False)
    monkeypatch.delenv("CUDA_MODULE_LOADING", raising=False)
    calls: list[tuple[str, list[str], dict[str, str]]] = []

    def fake_execvpe(file: str, args: list[str], env: dict[str, str]) -> None:
        calls.append((file, list(args), dict(env)))

    monkeypatch.setattr(native_gpt2_module.os, "execvpe", fake_execvpe)
    cfg = build_native_gpt2_compiled_cli_run_config(
        dataset_alias=str(tmp_path / "dataset"),
        executable=None,
        output_dir=tmp_path / "gpt",
        eval_every_steps=250,
        sample_every_steps=20000,
        generate_tokens=144,
        checkpoint_every_steps=200,
        max_steps=1,
        batch_size=64,
        seq_len=1024,
        train_batch_tokens=524288,
        learning_rate=0.0006,
        min_lr=None,
        warmup_steps=60,
        weight_decay=0.1,
        num_layers=12,
        activation="gelu",
    )

    assert exec_native_gpt2(cfg, runner="compiled-cli") == 127
    generic_cfg = build_native_gpt_compiled_cli_run_config(
        dataset_alias=str(tmp_path / "dataset"),
        executable=None,
        output_dir=tmp_path / "gpt",
        eval_every_steps=250,
        sample_every_steps=20000,
        generate_tokens=144,
        checkpoint_every_steps=200,
        max_steps=1,
        batch_size=64,
        seq_len=1024,
        train_batch_tokens=524288,
        learning_rate=0.0006,
        min_lr=None,
        warmup_steps=60,
        weight_decay=0.1,
        num_layers=12,
        activation="gelu",
    )
    assert exec_native_gpt(generic_cfg, runner="compiled-cli") == 127

    assert len(calls) == 2
    for file, args, env in calls:
        assert file == str(cli)
        assert args[:4] == [str(cli), "--model-family", "gpt", "--dataset-alias"]
        assert "--train-transformer-lm" in args
        assert env["CUDA_VISIBLE_DEVICES"] == "0"
        assert env["CUDA_DEVICE_MAX_CONNECTIONS"] == "1"
        assert env["CUDA_MODULE_LOADING"] == "LAZY"


def test_native_gpt2_cpp_launcher_builds_and_execs(tmp_path: Path) -> None:
    if shutil.which("c++") is None:
        pytest.skip("c++ compiler not available")
    root = Path(__file__).resolve().parents[1]
    launcher = tmp_path / "nfn_gpt2_tile_train"

    build = subprocess.run(
        ["bash", str(root / "tools" / "build_native_gpt2_launcher.sh"), str(launcher)],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert build.returncode == 0, build.stderr
    assert launcher.exists()

    dry_run = subprocess.run(
        [str(launcher), "--target", "/bin/echo", "--dry-run", "--print-command", "--", "-i", "train.bin"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert dry_run.returncode == 0, dry_run.stderr
    assert dry_run.stdout.strip() == "/bin/echo -i train.bin"

    executed = subprocess.run(
        [str(launcher), "--target", "/bin/echo", "--", "hello"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert executed.returncode == 0, executed.stderr
    assert executed.stdout.strip() == "hello"


def test_native_gpt2_cpp_binding_builds_and_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if shutil.which("c++") is None:
        pytest.skip("c++ compiler not available")
    root = Path(__file__).resolve().parents[1]
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    package_dir = tmp_path / "neuralfn"
    binding = package_dir / f"_native_gpt2{ext_suffix}"

    build = subprocess.run(
        ["bash", str(root / "tools" / "build_native_gpt2_binding.sh"), str(binding)],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert build.returncode == 0, build.stderr
    assert binding.exists()

    monkeypatch.setattr(neuralfn, "__path__", [str(package_dir), *list(neuralfn.__path__)])
    monkeypatch.delitem(sys.modules, "neuralfn_native_gpt", raising=False)
    monkeypatch.delitem(sys.modules, "neuralfn._native_gpt", raising=False)
    monkeypatch.delitem(sys.modules, "neuralfn_native_gpt2", raising=False)
    monkeypatch.delitem(sys.modules, "neuralfn._native_gpt2", raising=False)
    monkeypatch.setattr(native_gpt2_module, "NATIVE_GPT2_BINDING_MODULES", ("neuralfn._native_gpt2",))
    cfg = NativeGpt2RunConfig(
        executable="/bin/true",
        train_data="train.bin",
        val_data="val.bin",
        output_dir="out",
        model_descriptor="d12",
        eval_every_steps=250,
        sample_every_steps=20000,
        generate_tokens=144,
        checkpoint_every_steps=200,
        batch_size=64,
        seq_len=1024,
        train_batch_tokens=524288,
        learning_rate=0.0006,
        final_lr_fraction=0.0,
        warmup_steps=60,
        weight_decay=0.1,
        max_steps=20000,
    )

    status = native_gpt2_runner_status("auto")

    assert status.resolved == "binding"
    assert status.binding_module == "neuralfn._native_gpt2"
    assert run_native_gpt2(cfg, runner="auto") == 0
    binding_module = importlib.import_module("neuralfn._native_gpt2")
    captured = binding_module.run_gpt2_capture(
        {
            "compiled_cli_argv": [
                sys.executable,
                "-c",
                "import sys; print('native-sampler-ok'); print('native-stderr-ok', file=sys.stderr)",
            ]
        }
    )
    assert captured == {
        "returncode": 0,
        "stdout": "native-sampler-ok\n",
        "stderr": "native-stderr-ok\n",
    }


def test_native_gpt_cpp_binding_builds_and_runs_generic_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if shutil.which("c++") is None:
        pytest.skip("c++ compiler not available")
    root = Path(__file__).resolve().parents[1]
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    package_dir = tmp_path / "neuralfn"
    binding = package_dir / f"_native_gpt{ext_suffix}"

    build = subprocess.run(
        ["bash", str(root / "tools" / "build_native_gpt_binding.sh"), str(binding)],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert build.returncode == 0, build.stderr
    assert binding.exists()

    monkeypatch.setattr(neuralfn, "__path__", [str(package_dir), *list(neuralfn.__path__)])
    monkeypatch.delitem(sys.modules, "neuralfn_native_gpt", raising=False)
    monkeypatch.delitem(sys.modules, "neuralfn._native_gpt", raising=False)
    monkeypatch.delitem(sys.modules, "neuralfn_native_gpt2", raising=False)
    monkeypatch.delitem(sys.modules, "neuralfn._native_gpt2", raising=False)
    cfg = NativeGptRunConfig(
        executable="/bin/true",
        train_data="train.bin",
        val_data="val.bin",
        output_dir="out",
        model_descriptor="d12",
        eval_every_steps=250,
        sample_every_steps=20000,
        generate_tokens=144,
        checkpoint_every_steps=200,
        batch_size=64,
        seq_len=1024,
        train_batch_tokens=524288,
        learning_rate=0.0006,
        final_lr_fraction=0.0,
        warmup_steps=60,
        weight_decay=0.1,
        max_steps=20000,
    )

    status = native_gpt_runner_status("auto")

    assert status.resolved == "binding"
    assert status.binding_module == "neuralfn._native_gpt"
    assert run_native_gpt(cfg, runner="auto") == 0


def test_native_gpt_cpp_binding_uses_spawn_and_lazy_cuda_module_loading() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "neuralfn" / "csrc" / "native_gpt2" / "binding.cpp").read_text(encoding="utf-8")

    assert "#include <spawn.h>" in source
    assert "#include <sys/select.h>" in source
    assert "posix_spawnp(&pid" in source
    assert "pipe(stdout_pipe)" in source
    assert "pipe(stderr_pipe)" in source
    assert "drain_child_output_pipes" in source
    assert "posix_spawn_file_actions_adddup2" in source
    assert "STDERR_FILENO" in source
    assert '"run_gpt_capture"' in source
    assert '"run_infer"' in source
    assert 'setenv_default_if_empty("CUDA_MODULE_LOADING", "LAZY")' in source
    assert "fork()" not in source


def test_native_train_cpp_binding_uses_spawn_and_lazy_cuda_module_loading() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "neuralfn" / "csrc" / "native_train" / "binding.cpp").read_text(encoding="utf-8")

    assert "#include <spawn.h>" in source
    assert "posix_spawnp(&pid" in source
    assert 'setenv_default_if_empty("CUDA_MODULE_LOADING", "LAZY")' in source
    assert "fork()" not in source


def test_native_train_cpp_binding_resolves_and_runs_compiled_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if shutil.which("c++") is None:
        pytest.skip("c++ compiler not available")
    root = Path(__file__).resolve().parents[1]
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    package_dir = tmp_path / "neuralfn"
    binding = package_dir / f"_native_train{ext_suffix}"

    build = subprocess.run(
        ["bash", str(root / "tools" / "build_native_train_binding.sh"), str(binding)],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert build.returncode == 0, build.stderr
    assert binding.exists()

    monkeypatch.setattr(neuralfn, "__path__", [str(package_dir), *list(neuralfn.__path__)])
    monkeypatch.delitem(sys.modules, "neuralfn_native_train", raising=False)
    monkeypatch.delitem(sys.modules, "neuralfn._native_train", raising=False)

    cfg = build_native_train_run_config(
        "gpt",
        args=("--dry-run", "--print-command", "--no-checkpoint"),
        native_train_cli="/bin/true",
    )
    status = native_train_runner_status("auto")

    assert isinstance(status, NativeTrainRunnerStatus)
    assert status.resolved == "binding"
    assert status.binding_module == "neuralfn._native_train"
    assert status.command_resolver_available is True
    assert resolve_native_train_binding_command(cfg) == cfg.argv()
    assert run_native_train(cfg, runner="auto") == 0


def test_native_train_cpp_binding_requires_command_resolver_symbol() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "neuralfn" / "csrc" / "native_train" / "binding.cpp").read_text(encoding="utf-8")
    python_source = (root / "neuralfn" / "native_train.py").read_text(encoding="utf-8")

    assert "resolve_command" in source
    assert "resolve_native_train_command" in source
    assert "command_from_config(config, &command, &command_error)" in source
    assert "strict_native_command" in source
    assert "is_forbidden_native_launcher" in source
    assert "requires a compiled C++ command" in source
    assert "validate_strict_native_train_command" in python_source
    assert "command_resolver_available" in python_source
    assert "resolve_native_train_binding_command" in python_source
    assert "missing run_train(config_dict)/run_native_train(config_dict)" in python_source
    assert "resolve_command(config_dict)/resolve_native_train_command(config_dict)" in python_source


def test_native_gpt_cpp_binding_requires_command_resolver_symbol() -> None:
    root = Path(__file__).resolve().parents[1]
    binding_source = (root / "neuralfn" / "csrc" / "native_gpt2" / "binding.cpp").read_text(encoding="utf-8")
    python_source = (root / "neuralfn" / "native_gpt2.py").read_text(encoding="utf-8")
    generic_source = (root / "neuralfn" / "native_gpt.py").read_text(encoding="utf-8")

    assert '"resolve_command"' in binding_source
    assert '"resolve_native_gpt_command"' in binding_source
    assert '"resolve_native_gpt2_command"' in binding_source
    assert "command_resolver_available" in python_source
    assert "resolve_native_gpt2_binding_command" in python_source
    assert "resolve_native_gpt_binding_command" in generic_source
    assert "resolve_command(config_dict)/resolve_native_gpt_command(config_dict)/resolve_native_gpt2_command(config_dict)" in python_source


def test_native_gpt2_cpp_binding_uses_compiled_cli_for_alias_only_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if shutil.which("c++") is None:
        pytest.skip("c++ compiler not available")
    root = Path(__file__).resolve().parents[1]
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    package_dir = tmp_path / "neuralfn"
    binding = package_dir / f"_native_gpt2{ext_suffix}"

    build = subprocess.run(
        ["bash", str(root / "tools" / "build_native_gpt2_binding.sh"), str(binding)],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert build.returncode == 0, build.stderr
    assert binding.exists()

    compiled_cli = tmp_path / "nfn_gpt2_native_train"
    observed_args = tmp_path / "compiled-cli-args.txt"
    compiled_cli.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\" > \"$NFN_TEST_COMPILED_CLI_ARGS\"\n"
        "exit 37\n",
        encoding="utf-8",
    )
    compiled_cli.chmod(0o755)
    monkeypatch.setenv("NFN_NATIVE_GPT2_CLI", str(compiled_cli))
    monkeypatch.setenv("NFN_TEST_COMPILED_CLI_ARGS", str(observed_args))
    monkeypatch.setattr(neuralfn, "__path__", [str(package_dir), *list(neuralfn.__path__)])
    monkeypatch.delitem(sys.modules, "neuralfn_native_gpt", raising=False)
    monkeypatch.delitem(sys.modules, "neuralfn._native_gpt", raising=False)
    monkeypatch.delitem(sys.modules, "neuralfn_native_gpt2", raising=False)
    monkeypatch.delitem(sys.modules, "neuralfn._native_gpt2", raising=False)
    cfg = build_native_gpt2_compiled_cli_run_config(
        dataset_alias="cached-shards",
        executable="/tmp/should-not-run-raw-train-gpt2cu",
        output_dir=tmp_path / "out",
        eval_every_steps=250,
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

    assert cfg.train_data == ""
    status = native_gpt2_runner_status("auto")
    assert status.resolved == "binding"
    assert status.command_resolver_available is True
    assert resolve_native_gpt2_binding_command(cfg) == cfg.compiled_cli_argv()
    generic_cfg = NativeGptRunConfig(
        **{key: value for key, value in cfg.to_dict().items() if key in NativeGptRunConfig.__dataclass_fields__}
    )
    assert resolve_native_gpt_binding_command(generic_cfg) == cfg.compiled_cli_argv()
    assert run_native_gpt2(cfg, runner="auto") == 37
    args = observed_args.read_text(encoding="utf-8").splitlines()
    assert args[:6] == [
        "--model-family",
        "gpt",
        "--dataset-alias",
        "cached-shards",
        "--backend",
        "tile-cuda",
    ]
    assert "--target" not in args
    assert "--train-transformer-lm" in args


def test_native_train_run_config_and_subprocess_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = tmp_path / "nfn_native_train"
    cli.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\" > \"$NFN_TEST_NATIVE_TRAIN_ARGS\"\n"
        "printf 'CUDA_VISIBLE_DEVICES=%s\\nCUDA_DEVICE_MAX_CONNECTIONS=%s\\nCUDA_MODULE_LOADING=%s\\n' "
        "\"$CUDA_VISIBLE_DEVICES\" \"$CUDA_DEVICE_MAX_CONNECTIONS\" \"$CUDA_MODULE_LOADING\" > \"$NFN_TEST_NATIVE_TRAIN_ENV\"\n"
        "exit 29\n",
        encoding="utf-8",
    )
    cli.chmod(0o755)
    output = tmp_path / "native-train-args.txt"
    env_output = tmp_path / "native-train-env.txt"
    monkeypatch.setenv("NFN_NATIVE_TRAIN_CLI", str(cli))
    monkeypatch.setenv("NFN_NATIVE_TRAIN_BINDING", "0")
    monkeypatch.setenv("NFN_TEST_NATIVE_TRAIN_ARGS", str(output))
    monkeypatch.setenv("NFN_TEST_NATIVE_TRAIN_ENV", str(env_output))
    monkeypatch.setattr(native_train_module, "resolve_cuda_visible_devices_value", lambda value: "7")
    cfg = build_native_train_run_config("nano_gpt", ["--tinystories", "--dry-run"])

    assert resolve_native_train_cli() == str(cli)
    assert cfg.to_dict()["model_family"] == "nanogpt"
    assert cfg.argv()[:3] == [str(cli), "--base-model", "nanogpt"]
    status = native_train_runner_status("auto")
    assert status.resolved == "compiled-cli"
    assert status.available is True
    assert run_native_train(cfg, runner="auto") == 29
    args = output.read_text(encoding="utf-8").splitlines()
    assert args[:4] == ["--base-model", "nanogpt", "--tinystories", "--dry-run"]
    env_lines = env_output.read_text(encoding="utf-8").splitlines()
    assert "CUDA_VISIBLE_DEVICES=7" in env_lines
    assert "CUDA_DEVICE_MAX_CONNECTIONS=1" in env_lines
    assert "CUDA_MODULE_LOADING=LAZY" in env_lines

    token_lm_cfg = build_native_train_run_config(
        "nanogpt",
        [
            "--train-token-lm",
            "--tile-ops-lib",
            "/tmp/libnfn_native_train_tile_ops.so",
            "--dataset-alias",
            "/tmp/native-cache",
            "--max-steps",
            "2",
        ],
    )

    assert run_native_train(token_lm_cfg, runner="compiled-cli") == 29
    token_lm_args = output.read_text(encoding="utf-8").splitlines()
    assert token_lm_args[:3] == ["--base-model", "nanogpt", "--train-token-lm"]
    assert "--tile-ops-lib" in token_lm_args
    assert "--dataset-alias" in token_lm_args
    assert "--max-steps" in token_lm_args


def test_native_sm120_gpt_run_config_uses_compiled_launcher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sm120_cli = tmp_path / "nfn_train_gpt_sm120"
    sm120_cli.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\" > \"$NFN_TEST_NATIVE_TRAIN_ARGS\"\n"
        "exit 47\n",
        encoding="utf-8",
    )
    sm120_cli.chmod(0o755)
    output = tmp_path / "sm120-train-args.txt"
    monkeypatch.setenv("NFN_NATIVE_SM120_CLI", str(sm120_cli))
    monkeypatch.setenv("NFN_NATIVE_TRAIN_BINDING", "0")
    monkeypatch.setenv("NFN_TEST_NATIVE_TRAIN_ARGS", str(output))

    cfg = build_native_sm120_gpt_run_config("gpt3", ["--tinystories", "--dry-run"])

    assert resolve_native_sm120_train_cli() == str(sm120_cli)
    assert cfg.argv() == [str(sm120_cli), "--base-model", "gpt3", "--tinystories", "--dry-run"]
    assert cfg.to_dict()["model_family"] == "gpt3"
    assert run_native_train(cfg, runner="compiled-cli") == 47
    assert output.read_text(encoding="utf-8").splitlines() == [
        "--base-model",
        "gpt3",
        "--tinystories",
        "--dry-run",
    ]


def test_native_sm120_gpt_run_config_runs_when_global_status_probe_misses_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sm120_cli = tmp_path / "nfn_train_gpt_sm120"
    sm120_cli.write_text("#!/usr/bin/env bash\nexit 48\n", encoding="utf-8")
    sm120_cli.chmod(0o755)
    missing_global_cli = tmp_path / "missing-nfn-native-train"
    monkeypatch.setenv("NFN_NATIVE_TRAIN_BINDING", "0")
    monkeypatch.setattr(
        native_train_module,
        "resolve_available_native_train_cli_for_status",
        lambda: missing_global_cli,
    )

    cfg = build_native_sm120_gpt_run_config("gpt", ["--dry-run"], native_sm120_cli=str(sm120_cli))

    assert native_train_runner_status("compiled-cli").available is False
    assert run_native_train(cfg, runner="compiled-cli") == 48


def test_native_sm120_gpt_run_config_rejects_non_gpt_family_and_python_launcher() -> None:
    with pytest.raises(ValueError, match="dense GPT"):
        build_native_sm120_gpt_run_config("llama", ["--dry-run"])

    cfg = build_native_sm120_gpt_run_config(
        "gpt",
        ["--dry-run"],
        native_sm120_cli=sys.executable,
    )

    with pytest.raises(ValueError, match="compiled C\\+\\+ command"):
        cfg.argv()


def test_native_gpt_launcher_run_config_uses_generic_compiled_launcher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generic_cli = tmp_path / "nfn_train_gpt"
    generic_cli.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\" > \"$NFN_TEST_NATIVE_TRAIN_ARGS\"\n"
        "exit 46\n",
        encoding="utf-8",
    )
    generic_cli.chmod(0o755)
    output = tmp_path / "gpt-train-args.txt"
    monkeypatch.setenv("NFN_NATIVE_GPT_TRAIN_CLI", str(generic_cli))
    monkeypatch.setenv("NFN_NATIVE_TRAIN_BINDING", "0")
    monkeypatch.setenv("NFN_TEST_NATIVE_TRAIN_ARGS", str(output))

    cfg = build_native_gpt_launcher_run_config("gpt3", ["--tinystories", "--dry-run"])

    assert resolve_native_gpt_launcher_train_cli() == str(generic_cli)
    assert cfg.argv() == [str(generic_cli), "--base-model", "gpt3", "--tinystories", "--dry-run"]
    assert cfg.to_dict()["model_family"] == "gpt3"
    assert run_native_train(cfg, runner="compiled-cli") == 46
    assert output.read_text(encoding="utf-8").splitlines() == [
        "--base-model",
        "gpt3",
        "--tinystories",
        "--dry-run",
    ]


def test_native_gpt_launcher_run_config_runs_when_global_status_probe_misses_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generic_cli = tmp_path / "nfn_train_gpt"
    generic_cli.write_text("#!/usr/bin/env bash\nexit 45\n", encoding="utf-8")
    generic_cli.chmod(0o755)
    missing_global_cli = tmp_path / "missing-nfn-native-train"
    monkeypatch.setenv("NFN_NATIVE_TRAIN_BINDING", "0")
    monkeypatch.setattr(
        native_train_module,
        "resolve_available_native_train_cli_for_status",
        lambda: missing_global_cli,
    )

    cfg = build_native_gpt_launcher_run_config("gpt", ["--dry-run"], native_gpt_launcher_cli=str(generic_cli))

    assert native_train_runner_status("compiled-cli").available is False
    assert run_native_train(cfg, runner="compiled-cli") == 45


def test_native_gpt_launcher_run_config_rejects_non_gpt_family_and_python_launcher() -> None:
    with pytest.raises(ValueError, match="dense GPT"):
        build_native_gpt_launcher_run_config("llama", ["--dry-run"])

    cfg = build_native_gpt_launcher_run_config(
        "gpt",
        ["--dry-run"],
        native_gpt_launcher_cli=sys.executable,
    )

    with pytest.raises(ValueError, match="compiled C\\+\\+ command"):
        cfg.argv()


def test_compiled_sm120_launcher_honors_native_env_defaults(tmp_path: Path) -> None:
    if shutil.which("c++") is None:
        pytest.skip("c++ compiler not available")
    root = Path(__file__).resolve().parents[1]
    sm120_launcher = tmp_path / "nfn_train_gpt_sm120"
    generic_launcher = tmp_path / "nfn_train_gpt"
    build = subprocess.run(
        ["bash", str(root / "tools" / "build_train_gpt_sm120_cli.sh"), str(sm120_launcher)],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert build.returncode == 0, build.stderr
    generic_build = subprocess.run(
        ["bash", str(root / "tools" / "build_train_gpt_cli.sh"), str(generic_launcher)],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert generic_build.returncode == 0, generic_build.stderr

    fake_native = tmp_path / "nfn_gpt_native_train"
    observed = tmp_path / "native-argv.txt"
    observed_env = tmp_path / "native-env.txt"
    fake_native.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\" > \"$NFN_TEST_NATIVE_GPT_ARGV\"\n"
        "printf 'CUDA_VISIBLE_DEVICES=%s\\nCUDA_DEVICE_MAX_CONNECTIONS=%s\\nCUDA_MODULE_LOADING=%s\\n' "
        "\"$CUDA_VISIBLE_DEVICES\" \"$CUDA_DEVICE_MAX_CONNECTIONS\" \"$CUDA_MODULE_LOADING\" > \"$NFN_TEST_NATIVE_GPT_ENV\"\n",
        encoding="utf-8",
    )
    fake_native.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "NFN_NATIVE_GPT_TRAIN_BIN": str(fake_native),
            "NFN_TEST_NATIVE_GPT_ARGV": str(observed),
            "NFN_TEST_NATIVE_GPT_ENV": str(observed_env),
            "CUDA_VISIBLE_DEVICES": "",
            "CUDA_DEVICE_MAX_CONNECTIONS": "",
            "CUDA_MODULE_LOADING": "",
            "NFN_SM120_NATIVE_EVAL_EVERY_STEPS": "1000",
            "NFN_SM120_NATIVE_EVAL_BATCHES": "7",
            "NFN_SM120_NATIVE_SAMPLE_EVERY": "0",
            "NFN_SM120_NATIVE_GENERATE_TOKENS": "32",
            "NFN_SM120_NATIVE_CHECKPOINT_EVERY": "0",
            "NFN_SM120_NATIVE_TRAIN_BATCH_TOKENS": "262144",
            "NFN_SM120_NATIVE_LEARNING_RATE": "0.0003",
            "NFN_SM120_NATIVE_FINAL_LR_FRACTION": "0.1",
            "NFN_SM120_NATIVE_WEIGHT_DECAY": "0.2",
            "NFN_SM120_NATIVE_WARMUP_STEPS": "12",
            "NFN_SM120_NATIVE_MAX_STEPS": "123",
            "NFN_SM120_NATIVE_TRAIN_LOSS_EVERY_STEPS": "50",
            "NFN_SM120_NATIVE_BATCH_SIZE": "16",
            "NFN_SM120_NATIVE_TRAIN_SEQ_LEN": "512",
            "NFN_NATIVE_GPT_CUDA_VISIBLE_DEVICES": "0",
        }
    )
    proc = subprocess.run(
        [str(sm120_launcher), "--dataset-alias", "/tmp/native-cache"],
        cwd=root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    args = observed.read_text(encoding="utf-8").splitlines()
    expected_pairs = {
        "--eval-every-steps": "1000",
        "--eval-batches": "7",
        "--native-cuda-sample-every": "0",
        "--native-cuda-generate-tokens": "32",
        "--native-cuda-checkpoint-every": "0",
        "--train-batch-tokens": "262144",
        "--learning-rate": "0.0003",
        "--final-lr-fraction": "0.1",
        "--weight-decay": "0.2",
        "--warmup-steps": "12",
        "--max-steps": "123",
        "--train-loss-every-steps": "50",
        "--batch-size": "16",
        "--train-seq-len": "512",
    }
    for flag, value in expected_pairs.items():
        assert args[args.index(flag) + 1] == value
    assert "--train-transformer-lm" in args
    env_lines = observed_env.read_text(encoding="utf-8").splitlines()
    assert "CUDA_VISIBLE_DEVICES=0" in env_lines
    assert "CUDA_DEVICE_MAX_CONNECTIONS=1" in env_lines
    assert "CUDA_MODULE_LOADING=LAZY" in env_lines

    generic_observed = tmp_path / "generic-native-argv.txt"
    generic_env = env.copy()
    generic_env.update(
        {
            "NFN_TEST_NATIVE_GPT_ARGV": str(generic_observed),
            "NFN_NATIVE_GPT_MODEL_FAMILY": "gpt3",
            "NFN_NATIVE_GPT_TEMPLATE_NAME": "gpt3",
            "NFN_NATIVE_GPT_BATCH_SIZE": "32",
            "NFN_NATIVE_GPT_TRAIN_SEQ_LEN": "2048",
            "NFN_NATIVE_GPT_EVAL_EVERY_STEPS": "1000",
            "NFN_NATIVE_GPT_TRAIN_BATCH_TOKENS": "524288",
        }
    )
    generic_proc = subprocess.run(
        [str(generic_launcher), "--dataset-alias", "/tmp/native-cache"],
        cwd=root,
        env=generic_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert generic_proc.returncode == 0, generic_proc.stderr
    generic_args = generic_observed.read_text(encoding="utf-8").splitlines()
    assert generic_args[generic_args.index("--model-family") + 1] == "gpt3"
    assert generic_args[generic_args.index("--template-name") + 1] == "gpt3"
    assert generic_args[generic_args.index("--batch-size") + 1] == "32"
    assert generic_args[generic_args.index("--train-seq-len") + 1] == "2048"
    assert generic_args[generic_args.index("--eval-every-steps") + 1] == "1000"
    assert generic_args[generic_args.index("--train-batch-tokens") + 1] == "524288"


def test_sm120_shell_fallback_honors_native_env_defaults(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    fake_native = tmp_path / "nfn_gpt_native_train"
    observed = tmp_path / "native-argv.txt"
    fake_native.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\" > \"$NFN_TEST_NATIVE_GPT_ARGV\"\n",
        encoding="utf-8",
    )
    fake_native.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "NFN_SM120_USE_COMPILED_LAUNCHER": "0",
            "NFN_NATIVE_GPT_TRAIN_BIN": str(fake_native),
            "NFN_TEST_NATIVE_GPT_ARGV": str(observed),
            "NFN_NATIVE_GPT_DATASET_ALIAS": "/tmp/native-cache",
            "NFN_NATIVE_GPT_MODEL_FAMILY": "gpt3",
            "NFN_NATIVE_GPT_TEMPLATE_NAME": "gpt3",
            "NFN_NATIVE_GPT_BATCH_SIZE": "32",
            "NFN_NATIVE_GPT_TRAIN_SEQ_LEN": "2048",
            "NFN_SM120_NATIVE_EVAL_EVERY_STEPS": "1000",
            "NFN_SM120_NATIVE_EVAL_BATCHES": "7",
            "NFN_SM120_NATIVE_SAMPLE_EVERY": "0",
            "NFN_SM120_NATIVE_GENERATE_TOKENS": "32",
            "NFN_SM120_NATIVE_CHECKPOINT_EVERY": "0",
            "NFN_SM120_NATIVE_TRAIN_BATCH_TOKENS": "262144",
            "NFN_SM120_NATIVE_LEARNING_RATE": "0.0003",
            "NFN_SM120_NATIVE_FINAL_LR_FRACTION": "0.1",
            "NFN_SM120_NATIVE_WEIGHT_DECAY": "0.2",
            "NFN_SM120_NATIVE_WARMUP_STEPS": "12",
            "NFN_SM120_NATIVE_MAX_STEPS": "123",
            "NFN_SM120_NATIVE_TRAIN_LOSS_EVERY_STEPS": "50",
            "NFN_NATIVE_GPT_CUDA_VISIBLE_DEVICES": "0",
        }
    )
    proc = subprocess.run(
        ["bash", str(root / "tools" / "train_gpt_sm120.sh")],
        cwd=root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    args = observed.read_text(encoding="utf-8").splitlines()
    expected_pairs = {
        "--model-family": "gpt3",
        "--template-name": "gpt3",
        "--dataset-alias": "/tmp/native-cache",
        "--eval-every-steps": "1000",
        "--eval-batches": "7",
        "--native-cuda-sample-every": "0",
        "--native-cuda-generate-tokens": "32",
        "--native-cuda-checkpoint-every": "0",
        "--train-batch-tokens": "262144",
        "--learning-rate": "0.0003",
        "--final-lr-fraction": "0.1",
        "--weight-decay": "0.2",
        "--warmup-steps": "12",
        "--max-steps": "123",
        "--train-loss-every-steps": "50",
        "--batch-size": "32",
        "--train-seq-len": "2048",
    }
    for flag, value in expected_pairs.items():
        assert args[args.index(flag) + 1] == value
    assert "--train-transformer-lm" in args


def test_compiled_and_shell_sm120_launchers_default_to_dedicated_gpu_selector(
    tmp_path: Path,
) -> None:
    if shutil.which("c++") is None:
        pytest.skip("c++ compiler not available")
    root = Path(__file__).resolve().parents[1]
    sm120_launcher = tmp_path / "nfn_train_gpt_sm120"
    build = subprocess.run(
        ["bash", str(root / "tools" / "build_train_gpt_sm120_cli.sh"), str(sm120_launcher)],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert build.returncode == 0, build.stderr

    fake_native = tmp_path / "nfn_gpt_native_train"
    compiled_env_out = tmp_path / "compiled-env.txt"
    shell_env_out = tmp_path / "shell-env.txt"
    fake_native.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'CUDA_VISIBLE_DEVICES=%s\\n' \"$CUDA_VISIBLE_DEVICES\" > \"$NFN_TEST_NATIVE_GPT_ENV\"\n",
        encoding="utf-8",
    )
    fake_native.chmod(0o755)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_nvidia_smi = fake_bin / "nvidia-smi"
    fake_nvidia_smi.write_text(
        "#!/usr/bin/env bash\n"
        "printf '0, Enabled, 3\\n1, Disabled, 8\\n2, Disabled, 1\\n'\n",
        encoding="utf-8",
    )
    fake_nvidia_smi.chmod(0o755)

    base_env = os.environ.copy()
    base_env.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{base_env.get('PATH', '')}",
            "NFN_NATIVE_GPT_TRAIN_BIN": str(fake_native),
            "CUDA_VISIBLE_DEVICES": "",
            "CUDA_DEVICE_MAX_CONNECTIONS": "",
            "CUDA_MODULE_LOADING": "",
        }
    )
    base_env.pop("NFN_NATIVE_GPT_CUDA_VISIBLE_DEVICES", None)
    base_env.pop("NFN_SM120_NATIVE_CUDA_VISIBLE_DEVICES", None)
    base_env.pop("NFN_SM120_CUDA_VISIBLE_DEVICES", None)

    compiled_env = base_env.copy()
    compiled_env["NFN_TEST_NATIVE_GPT_ENV"] = str(compiled_env_out)
    compiled_proc = subprocess.run(
        [str(sm120_launcher), "--dataset-alias", "/tmp/native-cache"],
        cwd=root,
        env=compiled_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert compiled_proc.returncode == 0, compiled_proc.stderr
    assert compiled_env_out.read_text(encoding="utf-8").strip() == "CUDA_VISIBLE_DEVICES=2"

    shell_env = base_env.copy()
    shell_env["NFN_TEST_NATIVE_GPT_ENV"] = str(shell_env_out)
    shell_env["NFN_SM120_USE_COMPILED_LAUNCHER"] = "0"
    shell_proc = subprocess.run(
        ["bash", str(root / "tools" / "train_gpt_sm120.sh"), "--dataset-alias", "/tmp/native-cache"],
        cwd=root,
        env=shell_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert shell_proc.returncode == 0, shell_proc.stderr
    assert shell_env_out.read_text(encoding="utf-8").strip() == "CUDA_VISIBLE_DEVICES=2"


def test_native_train_run_config_rejects_python_launchers_by_default() -> None:
    cfg = build_native_train_run_config(
        "gpt",
        ["--dry-run"],
        native_train_cli=sys.executable,
    )

    with pytest.raises(ValueError, match="compiled C\\+\\+ command"):
        cfg.argv()

    diagnostic_cfg = build_native_train_run_config(
        "gpt",
        ["--dry-run"],
        native_train_cli=sys.executable,
        strict_native_command=False,
    )
    assert diagnostic_cfg.argv()[:3] == [sys.executable, "--base-model", "gpt"]
    assert diagnostic_cfg.to_dict()["strict_native_command"] is False


def test_exec_native_train_replaces_process_with_compiled_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = tmp_path / "nfn_native_train"
    cli.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    cli.chmod(0o755)
    monkeypatch.setenv("NFN_NATIVE_TRAIN_CLI", str(cli))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.delenv("CUDA_DEVICE_MAX_CONNECTIONS", raising=False)
    monkeypatch.delenv("CUDA_MODULE_LOADING", raising=False)
    monkeypatch.setattr(native_train_module, "resolve_cuda_visible_devices_value", lambda value: "7")
    observed: dict[str, object] = {}

    def fake_execvpe(file: str, argv: list[str], env: dict[str, str]) -> None:
        observed["file"] = file
        observed["argv"] = argv
        observed["env"] = env
        raise SystemExit(0)

    monkeypatch.setattr(native_train_module.os, "execvpe", fake_execvpe)
    cfg = build_native_train_run_config("gpt", ["--tinystories", "--dry-run"])

    with pytest.raises(SystemExit) as exc:
        exec_native_train(cfg, runner="compiled-cli")

    assert exc.value.code == 0
    assert observed["file"] == str(cli)
    assert observed["argv"] == [str(cli), "--base-model", "gpt", "--tinystories", "--dry-run"]
    env = observed["env"]
    assert isinstance(env, dict)
    assert env["CUDA_VISIBLE_DEVICES"] == "7"
    assert env["CUDA_DEVICE_MAX_CONNECTIONS"] == "1"
    assert env["CUDA_MODULE_LOADING"] == "LAZY"


def test_native_train_run_config_uses_direct_dense_gpt_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    family_cli = tmp_path / "nfn_gpt_native_train"
    family_cli.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\" > \"$NFN_TEST_NATIVE_TRAIN_ARGS\"\n"
        "printf 'CUDA_MODULE_LOADING=%s\\n' \"$CUDA_MODULE_LOADING\" > \"$NFN_TEST_NATIVE_TRAIN_ENV\"\n"
        "exit 43\n",
        encoding="utf-8",
    )
    family_cli.chmod(0o755)
    output = tmp_path / "native-train-direct-args.txt"
    env_output = tmp_path / "native-train-direct-env.txt"
    monkeypatch.delenv("NFN_NATIVE_TRAIN_CLI", raising=False)
    monkeypatch.setenv("NFN_NATIVE_GPT_CLI", str(family_cli))
    monkeypatch.setenv("NFN_NATIVE_TRAIN_BINDING", "0")
    monkeypatch.setenv("NFN_TEST_NATIVE_TRAIN_ARGS", str(output))
    monkeypatch.setenv("NFN_TEST_NATIVE_TRAIN_ENV", str(env_output))

    cfg = build_native_train_run_config("gpt2", ["--tinystories", "--dry-run"])

    assert cfg.argv()[:5] == [str(family_cli), "--model-family", "gpt2", "--tinystories", "--dry-run"]
    status = native_train_runner_status("compiled-cli")
    assert status.available is True
    assert run_native_train(cfg, runner="compiled-cli") == 43
    args = output.read_text(encoding="utf-8").splitlines()
    assert args[:4] == ["--model-family", "gpt2", "--tinystories", "--dry-run"]
    assert "--base-model" not in args
    assert "CUDA_MODULE_LOADING=LAZY" in env_output.read_text(encoding="utf-8").splitlines()


def test_native_train_run_config_can_require_strict_dense_gpt_lm_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    family_cli = tmp_path / "nfn_gpt_native_train"
    family_cli.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    family_cli.chmod(0o755)
    monkeypatch.delenv("NFN_NATIVE_TRAIN_CLI", raising=False)
    monkeypatch.setenv("NFN_NATIVE_GPT_CLI", str(family_cli))

    cfg = build_native_train_run_config(
        "gpt3",
        ["--tinystories", "--dry-run"],
        require_cooperative_lm_head_backward=True,
        fast_startup=True,
    )

    assert cfg.to_dict()["require_cooperative_lm_head_backward"] is True
    assert cfg.to_dict()["fast_startup"] is True
    assert cfg.argv() == [
        str(family_cli),
        "--model-family",
        "gpt3",
        "--tinystories",
        "--dry-run",
        "--fast-startup",
        "--require-cooperative-lm-head-backward",
    ]

    duplicate_cfg = build_native_train_run_config(
        "gpt",
        ["--dry-run", "--native-cuda-require-cooperative-lm-head-backward"],
        require_cooperative_lm_head_backward=True,
    )

    assert duplicate_cfg.argv().count("--require-cooperative-lm-head-backward") == 0
    assert duplicate_cfg.argv().count("--native-cuda-require-cooperative-lm-head-backward") == 1

    fast_startup_duplicate_cfg = build_native_train_run_config(
        "gpt",
        ["--dry-run", "--native-cuda-fast-startup"],
        fast_startup=True,
    )

    assert fast_startup_duplicate_cfg.argv().count("--fast-startup") == 0
    assert fast_startup_duplicate_cfg.argv().count("--native-cuda-fast-startup") == 1

    unsupported_cfg = build_native_train_run_config(
        "llama",
        ["--dry-run"],
        fast_startup=True,
    )

    with pytest.raises(ValueError, match="only supported for dense GPT"):
        unsupported_cfg.argv()


def test_native_train_run_config_prefers_linked_dense_gpt_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    linked_cli = tmp_path / "nfn_gpt_native_train_linked"
    dynamic_cli = tmp_path / "nfn_gpt_native_train"
    linked_cli.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'linked\\n' > \"$NFN_TEST_NATIVE_TRAIN_SELECTED\"\n"
        "printf '%s\\n' \"$@\" > \"$NFN_TEST_NATIVE_TRAIN_ARGS\"\n"
        "exit 44\n",
        encoding="utf-8",
    )
    dynamic_cli.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'dynamic\\n' > \"$NFN_TEST_NATIVE_TRAIN_SELECTED\"\n"
        "exit 45\n",
        encoding="utf-8",
    )
    linked_cli.chmod(0o755)
    dynamic_cli.chmod(0o755)
    output = tmp_path / "native-train-linked-args.txt"
    selected = tmp_path / "native-train-selected.txt"
    monkeypatch.delenv("NFN_NATIVE_TRAIN_CLI", raising=False)
    monkeypatch.delenv("NFN_NATIVE_GPT_CLI", raising=False)
    monkeypatch.setenv("NFN_NATIVE_TRAIN_BINDING", "0")
    monkeypatch.setenv("NFN_TEST_NATIVE_TRAIN_ARGS", str(output))
    monkeypatch.setenv("NFN_TEST_NATIVE_TRAIN_SELECTED", str(selected))
    monkeypatch.setattr(native_train_module, "DEFAULT_NATIVE_GPT_TRAIN_CLI_LINKED", str(linked_cli))
    monkeypatch.setattr(native_train_module, "DEFAULT_NATIVE_GPT_TRAIN_CLI", str(dynamic_cli))

    cfg = build_native_train_run_config("gpt3", ["--tinystories", "--dry-run"])

    assert cfg.argv()[:5] == [str(linked_cli), "--model-family", "gpt3", "--tinystories", "--dry-run"]
    assert run_native_train(cfg, runner="compiled-cli") == 44
    assert selected.read_text(encoding="utf-8").strip() == "linked"
    args = output.read_text(encoding="utf-8").splitlines()
    assert args[:4] == ["--model-family", "gpt3", "--tinystories", "--dry-run"]
    assert "--base-model" not in args


def test_native_train_run_config_uses_direct_family_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    family_cli = tmp_path / "nfn_gpt2_evo_native_train"
    family_cli.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\" > \"$NFN_TEST_NATIVE_TRAIN_ARGS\"\n"
        "exit 46\n",
        encoding="utf-8",
    )
    family_cli.chmod(0o755)
    output = tmp_path / "native-train-family-args.txt"
    monkeypatch.delenv("NFN_NATIVE_TRAIN_CLI", raising=False)
    monkeypatch.setenv("NFN_NATIVE_GPT2_EVO_CLI", str(family_cli))
    monkeypatch.setenv("NFN_NATIVE_TRAIN_BINDING", "0")
    monkeypatch.setenv("NFN_TEST_NATIVE_TRAIN_ARGS", str(output))

    cfg = build_native_train_run_config(
        "gpt2_evo",
        ["--tinystories", "--native-cuda-dry-run", "--native-cuda-print-command"],
    )

    assert cfg.to_dict()["model_family"] == "gpt2-evo"
    assert cfg.argv() == [
        str(family_cli),
        "--tinystories",
        "--native-cuda-dry-run",
        "--native-cuda-print-command",
    ]
    status = native_train_runner_status("compiled-cli")
    assert status.available is True
    assert run_native_train(cfg, runner="compiled-cli") == 46
    args = output.read_text(encoding="utf-8").splitlines()
    assert args == ["--tinystories", "--native-cuda-dry-run", "--native-cuda-print-command"]
    assert "--base-model" not in args
    assert "--model-family" not in args


def test_native_train_explicit_unified_cli_overrides_direct_dense_gpt_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    family_cli = tmp_path / "nfn_gpt_native_train"
    family_cli.write_text("#!/usr/bin/env bash\nexit 44\n", encoding="utf-8")
    family_cli.chmod(0o755)
    unified_cli = tmp_path / "nfn_native_train"
    unified_cli.write_text("#!/usr/bin/env bash\nexit 45\n", encoding="utf-8")
    unified_cli.chmod(0o755)
    monkeypatch.setenv("NFN_NATIVE_GPT_CLI", str(family_cli))
    monkeypatch.setenv("NFN_NATIVE_TRAIN_CLI", str(unified_cli))

    cfg = build_native_train_run_config("gpt", ["--dry-run"])

    assert cfg.argv()[:4] == [str(unified_cli), "--base-model", "gpt", "--dry-run"]

    evo_cfg = build_native_train_run_config("gpt2-evo", ["--dry-run"])

    assert evo_cfg.argv()[:4] == [str(unified_cli), "--base-model", "gpt2-evo", "--dry-run"]


def test_native_train_cpp_binding_builds_and_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if shutil.which("c++") is None:
        pytest.skip("c++ compiler not available")
    root = Path(__file__).resolve().parents[1]
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    package_dir = tmp_path / "neuralfn"
    binding = package_dir / f"_native_train{ext_suffix}"

    build = subprocess.run(
        ["bash", str(root / "tools" / "build_native_train_binding.sh"), str(binding)],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert build.returncode == 0, build.stderr
    assert binding.exists()

    cli = tmp_path / "nfn_native_train"
    cli.write_text("#!/usr/bin/env bash\nexit 31\n", encoding="utf-8")
    cli.chmod(0o755)
    monkeypatch.setattr(neuralfn, "__path__", list(neuralfn.__path__) + [str(package_dir)])
    monkeypatch.delitem(sys.modules, "neuralfn_native_train", raising=False)
    monkeypatch.delitem(sys.modules, "neuralfn._native_train", raising=False)
    monkeypatch.setenv("NFN_NATIVE_TRAIN_CLI", str(cli))
    cfg = build_native_train_run_config("gpt2", ["--dry-run"])

    status = native_train_runner_status("auto")

    assert status.resolved == "binding"
    assert status.binding_module == "neuralfn._native_train"
    assert run_native_train(cfg, runner="auto") == 31

    raw_cli = tmp_path / "raw_train_gpt2cu"
    raw_cli.write_text("#!/usr/bin/env bash\nexit 41\n", encoding="utf-8")
    raw_cli.chmod(0o755)
    compiled_cli = tmp_path / "nfn_gpt_native_train"
    observed_args = tmp_path / "native-train-compiled-cli-args.txt"
    compiled_cli.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\" > \"$NFN_TEST_NATIVE_TRAIN_COMPILED_ARGS\"\n"
        "exit 33\n",
        encoding="utf-8",
    )
    compiled_cli.chmod(0o755)
    monkeypatch.setenv("NFN_TEST_NATIVE_TRAIN_COMPILED_ARGS", str(observed_args))

    module = importlib.import_module("neuralfn._native_train")
    assert module.run_train(
        {
            "train_data": "",
            "val_data": "",
            "argv": [str(raw_cli), "--should-not-run"],
            "compiled_cli_argv": [str(compiled_cli), "--dataset-alias", "cached-shards"],
            "launcher_argv": [str(raw_cli), "--launcher-fallback"],
        }
    ) == 33
    assert observed_args.read_text(encoding="utf-8").splitlines() == [
        "--dataset-alias",
        "cached-shards",
    ]


def test_native_gpt2_cpp_cli_builds_and_uses_sm120_defaults(tmp_path: Path) -> None:
    if shutil.which("c++") is None:
        pytest.skip("c++ compiler not available")
    root = Path(__file__).resolve().parents[1]
    cli = tmp_path / "nfn_gpt2_native_train"
    dataset_path = _write_uint16_shard_dataset(tmp_path)
    default_dataset_path = tmp_path / "roneneldan__TinyStories__TinyStoriesV2-GPT4"
    shutil.copytree(dataset_path, default_dataset_path)

    build = subprocess.run(
        ["bash", str(root / "tools" / "build_native_gpt2_cli.sh"), str(cli)],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert build.returncode == 0, build.stderr
    assert cli.exists()

    help_proc = subprocess.run(
        [str(cli), "--help"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert help_proc.returncode == 0, help_proc.stderr
    assert "Native no-Python dense GPT trainer entrypoint" in help_proc.stdout
    assert "--smoke-tile-ops" in help_proc.stdout
    assert "--smoke-nvfp4-pack" in help_proc.stdout
    assert "--smoke-optimizer-step" in help_proc.stdout
    assert "--smoke-lm-step" in help_proc.stdout
    assert "--smoke-attention-step" in help_proc.stdout
    assert "--smoke-mlp-step" in help_proc.stdout
    assert "--smoke-norm-residual-step" in help_proc.stdout
    assert "--smoke-transformer-block-step" in help_proc.stdout
    assert "--smoke-transformer-lm-step" in help_proc.stdout
    assert "--smoke-embedding-lm-step" in help_proc.stdout
    assert "--train-embedding-lm" in help_proc.stdout
    assert "--native-info --native-checkpoint PATH" in help_proc.stdout
    assert "--inspect-checkpoint PATH" in help_proc.stdout
    assert "--sample-checkpoint PATH --prompt-tokens IDS" in help_proc.stdout
    assert "--temperature VALUE" in help_proc.stdout
    assert "--top-k K" in help_proc.stdout
    assert "--repetition-penalty VALUE" in help_proc.stdout
    assert "--seed N" in help_proc.stdout
    assert "--checkpoint-logits-smoke --native-checkpoint PATH --prompt-tokens IDS" in help_proc.stdout
    assert "--checkpoint-qkv-smoke --native-checkpoint PATH --prompt-tokens IDS" in help_proc.stdout
    assert "--checkpoint-attention-smoke --native-checkpoint PATH --prompt-tokens IDS" in help_proc.stdout
    assert "--checkpoint-attention-residual-smoke --native-checkpoint PATH --prompt-tokens IDS" in help_proc.stdout
    assert "--checkpoint-block-smoke --native-checkpoint PATH --prompt-tokens IDS" in help_proc.stdout
    assert "--checkpoint-block-logits-smoke --native-checkpoint PATH --prompt-tokens IDS" in help_proc.stdout
    assert "--checkpoint-forward-logits-smoke --native-checkpoint PATH --prompt-tokens IDS" in help_proc.stdout
    assert "--checkpoint-block-index N" in help_proc.stdout
    assert "--checkpoint-load-smoke --native-checkpoint PATH" in help_proc.stdout
    assert "--checkpoint-load-tensor NAME" in help_proc.stdout
    assert "--checkpoint-layout --native-checkpoint PATH" in help_proc.stdout
    assert "--train-transformer-lm" in help_proc.stdout
    assert "--startup-only" in help_proc.stdout
    assert "--cuda-runtime-lib PATH" in help_proc.stdout
    assert "--model-family gpt|gpt2|gpt3|nanogpt" in help_proc.stdout
    assert "--template-name NAME" in help_proc.stdout
    assert "--graph-file PATH" in help_proc.stdout
    assert "compatible dense GPT template_spec metadata can drive native geometry" in help_proc.stdout
    assert "Tile-CUDA smokes/training" in help_proc.stdout
    assert "dense GPT registered parameter layout" in help_proc.stdout

    dry_run = subprocess.run(
        [
            str(cli),
            "--dataset-alias",
            str(dataset_path),
            "--target",
            "/bin/echo",
            "--dry-run",
            "--eval-every-steps",
            "1000",
            "--tile-cuda-activation-dtype",
            "nvfp4",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert dry_run.returncode == 0, dry_run.stderr
    default_payload = json.loads(dry_run.stdout)
    assert default_payload["model_family"] == "gpt"
    assert default_payload["backend"] == "tile-cuda"
    assert default_payload["status"] == "native-transformer-lm-ready"
    assert default_payload["template_name"] == "gpt"
    assert default_payload["resolved_native_template_name"] == "gpt2"
    assert default_payload["graph_file"] == ""
    assert default_payload["graph_file_exists"] is False
    assert default_payload["graph_file_size_bytes"] == -1
    assert default_payload["architecture_source"] == "template"
    assert default_payload["architecture_contract"] == "gpt-template-preset"
    assert (
        default_payload["model_family_context_policy"]
        == "dense-gpt-selectors-canonicalize-to-gpt-template-or-graph-selects-architecture"
    )
    assert default_payload["native_geometry_contract"] == {
        "name": "native-dense-gpt-transformer",
        "shape_source": "selected_dense_gpt_geometry",
        "template_selector": "gpt",
        "resolved_template_selector": "gpt2",
        "graph_file": "",
        "graph_file_exists": False,
        "graph_file_size_bytes": -1,
        "selector_native_runnable": True,
        "template_geometry_dynamic": False,
        "custom_graph_geometry_dynamic": False,
        "selected_template_geometry": {
            "source": "template",
            "model_dim": 768,
            "num_heads": 12,
            "head_dim": 64,
            "mlp_multiplier": 4,
            "vocab_size": 50257,
            "padded_vocab_size": 50304,
            "num_layers": 12,
            "seq_len": 1024,
            "dropout_p": 0,
        },
        "geometry_matches_compiled_loop": True,
        "model_dim": 768,
        "num_heads": 12,
        "head_dim": 64,
        "mlp_multiplier": 4,
        "vocab_size": 50257,
        "padded_vocab_size": 50304,
        "num_layers": 12,
        "seq_len": 1024,
        "position_encoding": "absolute",
        "norm": "layernorm",
        "attention": "causal-packed-qkv-sm120-tk-bf16",
        "mlp": "gelu-4x",
        "dropout_p": 0,
        "supported_template_selectors": [
            "gpt",
            "gpt2",
            "gpt2_modern",
            "gpt3",
            "gpt2_megakernel",
            "gpt2_moa",
            "nanogpt",
            "nanogpt_modern",
            "nanogpt_megakernel",
        ],
        "unsupported_geometry_next_step": "add-native-non-dense-variant-and-non-gpt-vocab-training-plans",
    }
    assert default_payload["lm_head_classifier_strategy_contract"] == {
        "reference_strategy": (
            "llm.kittens-full-resident-logits-fused-ce-dlogits-separate-classifier-matmuls"
        ),
        "native_strategy": (
            "row-chunked-bf16-logits-public-vocab-fused-ce-dlogits-separate-classifier-matmuls-tile-abi"
        ),
        "reference_classifier_fusion_scope": (
            "ce-dlogits-only-logits-dhidden-dweight-remain-separate"
        ),
        "native_classifier_fusion_scope": (
            "ce-dlogits-only-logits-dhidden-dweight-remain-separate"
        ),
        "reference_full_logit_rows": 64 * 1024,
        "native_logit_chunk_rows": 32768,
        "native_logit_chunk_count": 2,
        "padded_vocab_size": 50304,
        "reference_full_bf16_logit_elements": 64 * 1024 * 50304,
        "reference_full_bf16_logit_bytes": 64 * 1024 * 50304 * 2,
        "reference_full_float32_logit_bytes": 64 * 1024 * 50304 * 4,
        "native_chunk_bf16_logit_elements": 32768 * 50304,
        "native_chunk_bf16_logit_bytes": 32768 * 50304 * 2,
        "native_chunk_float32_logit_bytes": 32768 * 50304 * 4,
        "resident_logit_reduction_ratio": 2.0,
        "dlogits_storage": "in-place-over-bf16-logits",
        "graph_editor_tensor_flow": False,
        "torch_required": False,
        "same_script_benchmark_target": (
            "tools/paired_kernel_speed.py stage.lm_head_backward.total_ms and train_loop_wall_ms"
        ),
        "reference_alignment_target": (
            "match-fused-ce-dlogits-and-optimize-separate-logits-dhidden-dweight-stages"
        ),
        "strict_true_fused_experimental_path": "strict-true-fused-tile-kernel",
        "required_kernel_next_step": (
            "match-reference-fused-ce-dlogits-and-optimize-separate-logits-dhidden-dweight-stages"
        ),
    }
    assert default_payload["selected_graph_native_runnable"] is True
    assert default_payload["checkpoint_export_enabled"] is True
    assert default_payload["lm_head_cooperative_backward_requested"] is True
    assert default_payload["lm_head_cooperative_backward_sequence_wrapper_available"] is True
    assert isinstance(default_payload["lm_head_cooperative_backward_kernel_available"], bool)
    assert isinstance(default_payload["lm_head_cooperative_backward_fused_kernel_available"], bool)
    assert default_payload["lm_head_cooperative_backward_route_integrated"] is True
    assert default_payload["lm_head_cooperative_backward_fused_kernel_capability_available"] is False
    assert default_payload["lm_head_cooperative_backward_kernel_available"] is False
    assert default_payload["lm_head_cooperative_backward_fused_kernel_available"] is False
    assert default_payload["lm_head_llmk_classifier_matmul_parity_available"] is True
    assert isinstance(default_payload["lm_head_cooperative_backward_kernel_enabled"], bool)
    assert isinstance(default_payload["lm_head_cooperative_backward_cuda_graph_enabled"], bool)
    assert (
        default_payload["lm_head_cooperative_backward_strategy"]
        == "diagnostic-cuda-graph-ce-fork-join-dhidden-dweight-not-single-kernel"
    )
    assert default_payload["validation_shards_required"] is True
    assert default_payload["validation_shards_resolved"] is True
    assert default_payload["lm_head_row_chunk_size"] == 32768
    assert default_payload["lm_head_row_chunk_safe_cap"] == 49152
    assert default_payload["lm_head_row_chunk_unsafe_override_enabled"] is False

    nanogpt_model_plan = subprocess.run(
        [
            str(cli),
            "--dataset-alias",
            str(dataset_path),
            "--backend",
            "tile-cuda",
            "--model-family",
            "nanogpt",
            "--print-plan",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert nanogpt_model_plan.returncode == 0, nanogpt_model_plan.stderr
    nanogpt_model_payload = json.loads(nanogpt_model_plan.stdout)
    assert nanogpt_model_payload["model_family"] == "gpt"
    assert nanogpt_model_payload["template_name"] == "nanogpt"
    assert nanogpt_model_payload["resolved_native_template_name"] == "nanogpt"
    assert nanogpt_model_payload["selected_graph_support_status"] == "native-transformer-lm"
    assert nanogpt_model_payload["selected_graph_native_runnable"] is True

    unsafe_lm_head_chunk = subprocess.run(
        [
            str(cli),
            "--dataset-alias",
            str(dataset_path),
            "--dry-run",
            "--eval-every-steps",
            "0",
            "--lm-head-row-chunk-size",
            "65536",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert unsafe_lm_head_chunk.returncode == 0, unsafe_lm_head_chunk.stderr
    unsafe_payload = json.loads(unsafe_lm_head_chunk.stdout)
    assert unsafe_payload["passed"] is False
    assert unsafe_payload["status"] == "native-transformer-lm-failed"
    assert unsafe_payload["lm_head_row_chunk_size"] == 65536
    assert unsafe_payload["lm_head_row_chunk_safe_cap"] == 49152
    assert unsafe_payload["lm_head_row_chunk_unsafe_override_enabled"] is False
    assert "NFN_NATIVE_GPT_ALLOW_UNSAFE_LM_HEAD_ROW_CHUNK=1" in unsafe_payload["error"]

    unsafe_override = subprocess.run(
        [
            str(cli),
            "--dataset-alias",
            str(dataset_path),
            "--dry-run",
            "--eval-every-steps",
            "0",
            "--lm-head-row-chunk-size",
            "65536",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env={**os.environ, "NFN_NATIVE_GPT_ALLOW_UNSAFE_LM_HEAD_ROW_CHUNK": "1"},
    )
    assert unsafe_override.returncode == 0, unsafe_override.stderr
    unsafe_override_payload = json.loads(unsafe_override.stdout)
    assert unsafe_override_payload["passed"] is True
    assert unsafe_override_payload["lm_head_row_chunk_size"] == 65536
    assert unsafe_override_payload["lm_head_row_chunk_unsafe_override_enabled"] is True

    train_only_dataset = tmp_path / "train_only_uint16"
    train_only_dataset.mkdir()
    (train_only_dataset / "fineweb_train_000000.bin").write_bytes(struct.pack("<64H", *range(64)))
    train_only_eval_required = subprocess.run(
        [str(cli), "--dataset-alias", str(train_only_dataset), "--dry-run"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert train_only_eval_required.returncode == 2
    assert "no native uint16 validation token bin found" in train_only_eval_required.stderr

    train_only_no_eval = subprocess.run(
        [
            str(cli),
            "--dataset-alias",
            str(train_only_dataset),
            "--dry-run",
            "--eval-every-steps",
            "0",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert train_only_no_eval.returncode == 0, train_only_no_eval.stderr
    train_only_payload = json.loads(train_only_no_eval.stdout)
    assert train_only_payload["validation_shards_required"] is False
    assert train_only_payload["validation_shards_resolved"] is False
    assert train_only_payload["val_shard"] == ""

    train_only_startup = subprocess.run(
        [
            str(cli),
            "--dataset-alias",
            str(train_only_dataset),
            "--dry-run",
            "--startup-only",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert train_only_startup.returncode == 0, train_only_startup.stderr
    train_only_startup_payload = json.loads(train_only_startup.stdout)
    assert train_only_startup_payload["validation_shards_required"] is False
    assert train_only_startup_payload["validation_shards_resolved"] is False
    assert train_only_startup_payload["val_shard"] == ""
    assert default_payload["checkpoint_export_startup_only_elided"] is False
    assert default_payload["train_shard"].endswith("fineweb_train_000000.bin")
    assert default_payload["val_shard"].endswith("fineweb_val_000000.bin")
    assert default_payload["training_step_plan"]["status"] == "ready"

    llm_tinystories_dir = tmp_path / "llm-kittens" / "dev" / "data" / "tinystories"
    llm_tinystories_dir.mkdir(parents=True)
    tinystories_bytes = struct.pack("<" + "H" * 2048, *[idx % 256 for idx in range(2048)])
    (llm_tinystories_dir / "TinyStories_train.bin").write_bytes(tinystories_bytes)
    (llm_tinystories_dir / "TinyStories_val.bin").write_bytes(tinystories_bytes)
    tinystories_env = {**os.environ, "NFN_LLM_KITTENS_TINYSTORIES_DIR": str(llm_tinystories_dir)}
    tinystories_dry_run = subprocess.run(
        [
            str(cli),
            "--tinystories",
            "--target",
            "/bin/echo",
            "--dry-run",
        ],
        env=tinystories_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert tinystories_dry_run.returncode == 0, tinystories_dry_run.stderr
    tinystories_payload = json.loads(tinystories_dry_run.stdout)
    assert tinystories_payload["dataset_path"] == str(llm_tinystories_dir)
    assert tinystories_payload["train_shard"] == str(llm_tinystories_dir / "TinyStories_train.bin")
    assert tinystories_payload["val_shard"] == str(llm_tinystories_dir / "TinyStories_val.bin")
    assert tinystories_payload["token_shards_resolved"] is True

    direct_train_file_dry_run = subprocess.run(
        [
            str(cli),
            "--dataset-alias",
            str(llm_tinystories_dir / "TinyStories_train.bin"),
            "--target",
            "/bin/echo",
            "--dry-run",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert direct_train_file_dry_run.returncode == 0, direct_train_file_dry_run.stderr
    direct_train_file_payload = json.loads(direct_train_file_dry_run.stdout)
    assert direct_train_file_payload["dataset_path"] == str(llm_tinystories_dir / "TinyStories_train.bin")
    assert direct_train_file_payload["train_shard"] == str(llm_tinystories_dir / "TinyStories_train.bin")
    assert direct_train_file_payload["val_shard"] == str(llm_tinystories_dir / "TinyStories_val.bin")

    external_plan = subprocess.run(
        [
            str(cli),
            "--dataset-alias",
            str(dataset_path),
            "--backend",
            "llm-kittens",
            "--target",
            "/bin/echo",
            "--print-plan",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert external_plan.returncode == 2
    assert "Invalid backend: llm-kittens" in external_plan.stderr
    assert external_plan.stdout == ""

    tile_plan = subprocess.run(
        [
            str(cli),
            "--dataset-alias",
            str(dataset_path),
            "--backend",
            "tile-cuda",
            "--print-plan",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert tile_plan.returncode == 0, tile_plan.stderr
    tile_payload = json.loads(tile_plan.stdout)
    assert tile_payload["model_family"] == "gpt"
    assert tile_payload["backend"] == "tile-cuda"
    assert tile_payload["status"] == "native-transformer-lm-ready"
    assert tile_payload["template_name"] == "gpt"
    assert tile_payload["resolved_native_template_name"] == "gpt2"
    assert tile_payload["architecture_source"] == "template"
    assert tile_payload["architecture_contract"] == "gpt-template-preset"
    assert (
        tile_payload["model_family_context_policy"]
        == "dense-gpt-selectors-canonicalize-to-gpt-template-or-graph-selects-architecture"
    )
    assert tile_payload["native_geometry_contract"]["name"] == "native-dense-gpt-transformer"
    assert tile_payload["native_geometry_contract"]["shape_source"] == "selected_dense_gpt_geometry"
    assert tile_payload["native_geometry_contract"]["selector_native_runnable"] is True
    assert tile_payload["native_geometry_contract"]["template_geometry_dynamic"] is False
    assert tile_payload["native_geometry_contract"]["custom_graph_geometry_dynamic"] is False
    assert tile_payload["native_geometry_contract"]["model_dim"] == 768
    assert tile_payload["native_geometry_contract"]["num_heads"] == 12
    assert tile_payload["native_geometry_contract"]["vocab_size"] == 50257
    assert tile_payload["native_geometry_contract"]["padded_vocab_size"] == 50304
    assert tile_payload["template_known"] is True
    assert tile_payload["checkpoint_export_enabled"] is True
    assert tile_payload["checkpoint_export_startup_only_elided"] is False

    startup_tile_plan = subprocess.run(
        [
            str(cli),
            "--dataset-alias",
            str(dataset_path),
            "--backend",
            "tile-cuda",
            "--startup-only",
            "--print-plan",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert startup_tile_plan.returncode == 0, startup_tile_plan.stderr
    startup_tile_payload = json.loads(startup_tile_plan.stdout)
    assert startup_tile_payload["checkpoint_export_enabled"] is False
    assert startup_tile_payload["checkpoint_export_startup_only_elided"] is True
    assert startup_tile_payload["validation_shards_required"] is False
    assert startup_tile_payload["validation_shards_resolved"] is False
    assert startup_tile_payload["val_shard"] == ""
    assert tile_payload["shipped_template_catalog_count"] == len(SHIPPED_GPT_TEMPLATE_PRESETS)
    assert tile_payload["shipped_template_catalog"] == list(SHIPPED_GPT_TEMPLATE_PRESETS)
    assert tile_payload["selected_graph_support_status"] == "native-transformer-lm"
    assert tile_payload["selected_graph_native_runnable"] is True
    assert tile_payload["train_shard"].endswith("fineweb_train_000000.bin")
    assert tile_payload["val_shard"].endswith("fineweb_val_000000.bin")
    assert tile_payload["parameter_layout"]["buffer_count"] == 2 + (12 * 12) + 2
    assert tile_payload["parameter_layout"]["buffers"][0]["name"] == "wte.weight"
    assert tile_payload["shape"]["vocab_size"] == 50257
    assert tile_payload["shape"]["padded_vocab_size"] == 50304
    assert tile_payload["parameter_layout"]["buffers"][0]["shape"] == [50304, 768]
    assert tile_payload["training_step_plan"]["status"] == "ready"
    assert tile_payload["training_step_plan"]["forward_stage_count"] > 0
    assert tile_payload["training_step_plan"]["backward_stage_count"] > 0
    assert any(stage["name"] == "h.0.attn.sdpa.forward" for stage in tile_payload["training_step_plan"]["stages"])
    assert any(stage["name"] == "adamw_step" for stage in tile_payload["training_step_plan"]["stages"])
    assert tile_payload["layer_evo"] == {
        "enabled": False,
        "graph_editor_tensor_flow": False,
        "target_parameter": "block_6.ln1.weight",
        "target_parameter_dtype": "float32",
        "layer_index": 6,
        "interval": 10,
        "population": 8,
        "mutation_scale": 0.02,
        "forward_candidate_eval_enabled": True,
        "candidate_loss_source": "native-forward-loss-device-resident-current-batch",
        "candidate_loss_transport": "device-to-device",
    }
    assert tile_payload["attention_forward_strategy"] == "tk-sm120-packed-qkv-bf16-flashattention"
    assert tile_payload["attention_forward_score_reuse_value_dim"] == 64
    assert tile_payload["attention_forward_scalar_cta_elision_factor"] == 64
    assert tile_payload["attention_forward_value_chunk_size"] == 64
    assert tile_payload["attention_forward_scalar_launch_fallback_available"] is True
    assert tile_payload["attention_forward_scalar_launch_fallback_enabled"] is False
    assert tile_payload["attention_forward_scalar_launch_allowed"] is False
    assert tile_payload["optimized_attention_required"] is True
    assert tile_payload["attention_forward_row_launch_auto_disable_enabled"] is True
    assert tile_payload["attention_forward_row_count"] * 64 == tile_payload["attention_forward_scalar_output_count"]
    assert tile_payload["packed_qkv_attention_enabled"] is True
    assert tile_payload["packed_qkv_attention_bf16_elements"] == 64 * 1024 * 768 * 4
    assert tile_payload["packed_qkv_attention_bf16_bytes"] == 64 * 1024 * 768 * 4 * 2
    assert tile_payload["packed_qkv_float_attention_tape_elided"] is True
    assert tile_payload["packed_qkv_float_attention_tape_elements_elided"] == 64 * 1024 * 768 * 8
    assert tile_payload["packed_qkv_float_attention_tape_bytes_elided"] == 64 * 1024 * 768 * 8 * 4
    assert tile_payload["qkv_forward_layout_strategy"] == "packed-qkv-bf16-no-split"
    assert tile_payload["qkv_forward_layout_kernel_launches_per_block"] == 0
    assert tile_payload["qkv_forward_layout_legacy_launches_per_block"] == 4
    assert tile_payload["qkv_forward_layout_launches_elided_per_block"] == 4
    assert tile_payload["qkv_bias_layout_strategy"] == "packed-qkv-bf16-bias-fused-tk-gemm"
    assert tile_payload["qkv_bias_fused_tk_gemm_enabled"] is True
    assert tile_payload["qkv_bias_layout_kernel_launches_per_block"] == 0
    assert tile_payload["qkv_bias_layout_legacy_launches_per_block"] == 2
    assert tile_payload["qkv_bias_layout_launches_elided_per_block"] == 2
    assert tile_payload["qkv_backward_layout_strategy"] == "packed-qkv-bf16-gradient-handoff"
    assert tile_payload["qkv_backward_layout_kernel_launches_per_block"] == 1
    assert tile_payload["qkv_backward_layout_legacy_launches_per_block"] == 4
    assert tile_payload["qkv_backward_layout_launches_elided_per_block"] == 3
    assert tile_payload["attention_backward_bf16_qkv_grad_handoff_enabled"] is True
    assert tile_payload["attention_backward_direct_bf16_qkv_grad_scratch_enabled"] is True
    assert tile_payload["attention_backward_direct_bf16_qkv_grad_scratch_elements"] == 64 * 1024 * 768 * 3
    assert tile_payload["attention_backward_qkv_bridge_strategy"] == (
        "tk-sm120-packed-qkv-bf16-grad-out-direct-bf16-qkv-handoff"
    )
    assert tile_payload["attention_backward_qkv_bridge_kernel_launches_per_block"] == 2
    assert tile_payload["attention_backward_qkv_bridge_legacy_launches_per_block"] == 4
    assert tile_payload["attention_backward_qkv_bridge_launches_elided_per_block"] == 3
    assert tile_payload["bf16_projection_residual_enabled"] is True
    assert tile_payload["attention_projection_input_strategy"] == (
        "packed-o-bf16-direct-gemm-bf16-residual-consumer"
    )
    assert tile_payload["attention_packed_output_unpack_strategy"] == "elided-direct-bf16-projection"
    assert tile_payload["mlp_fc_bias_gelu_strategy"] == "fused-bias-preactivation-gelu"
    assert tile_payload["mlp_fc_bias_gelu_kernel_launches_per_block"] == 1
    assert tile_payload["mlp_fc_bias_gelu_legacy_launches_per_block"] == 2
    assert tile_payload["mlp_fc_bias_gelu_launches_elided_per_block"] == 1
    assert tile_payload["mlp_proj_forward_activation_strategy"] == "fused-gelu-bf16-act-direct-bf16-output-gemm"
    assert tile_payload["mlp_forward_act_bf16_elements"] == 0
    assert tile_payload["mlp_forward_act_bf16_bytes"] == 0
    assert tile_payload["projection_bf16_scratch_elements"] == 64 * 1024 * 768
    assert tile_payload["projection_bf16_scratch_bytes"] == 64 * 1024 * 768 * 2
    assert tile_payload["projection_bias_residual_strategy"] == "fused-bf16-linear-bias-residual-add"
    assert tile_payload["projection_bias_residual_kernel_launches_per_block"] == 2
    assert tile_payload["projection_bias_residual_legacy_launches_per_block"] == 4
    assert tile_payload["projection_bias_residual_launches_elided_per_block"] == 2
    assert tile_payload["attention_residual_ln2_strategy"] == "fused-bf16-linear-bias-residual-layernorm"
    assert tile_payload["attention_residual_ln2_kernel_launches_per_block"] == 1
    assert tile_payload["attention_residual_ln2_legacy_launches_per_block"] == 2
    assert tile_payload["attention_residual_ln2_launches_elided_per_block"] == 1
    assert tile_payload["attention_backward_grad_layout_strategy"] == "merged-grad-out-direct"
    assert tile_payload["attention_backward_grad_layout_kernel_launches_per_block"] == 0
    assert tile_payload["attention_backward_grad_layout_legacy_launches_per_block"] == 1
    assert tile_payload["attention_backward_grad_layout_launches_elided_per_block"] == 1
    assert (
        tile_payload["attention_backward_strategy"]
        == "tk-sm120-packed-qkv-bf16-saved-activation-backward-direct-bf16-grad-scratch-handoff"
    )
    assert tile_payload["attention_backward_reuses_forward_workspace"] is True
    assert tile_payload["attention_backward_uses_saved_forward_workspace"] is True
    assert tile_payload["attention_activation_storage_strategy"] == "disabled"
    assert tile_payload["packed_attention_activation_storage_strategy"] == (
        "packed-qkv-o-bf16-forward-store-direct-backward"
    )
    assert tile_payload["stored_packed_attention_activation_blocks"] == 12
    assert tile_payload["stored_packed_attention_block_placement"] == "head"
    assert tile_payload["stored_packed_attention_block_start"] == 0
    assert tile_payload["stored_packed_attention_bf16_elements"] > 0
    assert tile_payload["stored_packed_attention_bf16_bytes"] == (
        tile_payload["stored_packed_attention_bf16_elements"] * 2
    )
    assert tile_payload["stored_packed_attention_lse_elements"] > 0
    assert tile_payload["stored_packed_attention_lse_bytes"] == (
        tile_payload["stored_packed_attention_lse_elements"] * 4
    )
    assert tile_payload["stored_packed_attention_lse_enabled"] is True
    assert tile_payload["stored_packed_attention_ln1_bf16_enabled"] is True
    assert tile_payload["stored_packed_attention_ln1_bf16_blocks"] == 11
    assert tile_payload["stored_packed_attention_ln1_bf16_elements"] > 0
    assert tile_payload["stored_packed_attention_ln1_bf16_bytes"] == (
        tile_payload["stored_packed_attention_ln1_bf16_elements"] * 2
    )
    assert tile_payload["stored_packed_attention_store_blocks"] == 0
    assert tile_payload["stored_packed_attention_restore_blocks"] == 0
    assert tile_payload["stored_packed_attention_backward_kernel_launches"] == 0
    assert tile_payload["stored_packed_attention_backward_consumer_strategy"] == (
        "saved-packed-qkv-o-lse-bf16-backward-to-qkv"
    )
    assert tile_payload["attention_backward_recompute_forward_elided_per_block"] == 1
    assert tile_payload["attention_backward_score_reuse_dim"] == 64
    assert tile_payload["attention_backward_scalar_cta_elision_factor"] == 192
    assert tile_payload["attention_backward_row_count"] * 192 == tile_payload["attention_backward_scalar_output_count"]
    disabled_packed_store_plan = subprocess.run(
        [
            str(cli),
            "--dataset-alias",
            str(dataset_path),
            "--backend",
            "tile-cuda",
            "--print-plan",
        ],
        env={
            **os.environ,
            "NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_ACTIVATIONS": "0",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert disabled_packed_store_plan.returncode == 0, disabled_packed_store_plan.stderr
    disabled_packed_store_payload = json.loads(disabled_packed_store_plan.stdout)
    assert disabled_packed_store_payload["packed_attention_activation_storage_strategy"] == "disabled"
    assert disabled_packed_store_payload["stored_packed_attention_activation_blocks"] == 0
    assert disabled_packed_store_payload["stored_packed_attention_bf16_elements"] == 0
    assert disabled_packed_store_payload["stored_packed_attention_bf16_bytes"] == 0
    assert disabled_packed_store_payload["stored_packed_attention_backward_consumer_strategy"] == "disabled"
    gpt2_stages_by_name = {stage["name"]: stage for stage in tile_payload["training_step_plan"]["stages"]}
    assert (
        gpt2_stages_by_name["lm_head.backward_weight_tied"]["kernel_abi"]
        == "nfn_native_tile_linear_backward_weight_accumulate_float32"
    )
    assert "nfn_native_tile_scaled_dot_product_attention_backward_float32" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_evo_mutate_candidates_float32" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_evo_select_best_loss_float32" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_evo_adopt_candidate_float32" in tile_payload["available_native_kernels"]
    assert (
        "nfn_native_tile_scaled_dot_product_attention_backward_from_merged_grad_float32"
        in tile_payload["available_native_kernels"]
    )
    assert (
        "nfn_native_tile_scaled_dot_product_attention_backward_to_qkv_from_merged_grad_float32"
        in tile_payload["available_native_kernels"]
    )
    assert (
        "nfn_native_tile_scaled_dot_product_attention_backward_to_qkv_reuse_forward_from_merged_grad_float32"
        in tile_payload["available_native_kernels"]
    )
    assert "nfn_native_tile_attention_tk_store_forward_workspace_bf16" in tile_payload["available_native_kernels"]
    assert (
        "nfn_native_tile_scaled_dot_product_attention_backward_to_qkv_from_saved_tk_bf16_from_merged_grad_float32"
        in tile_payload["available_native_kernels"]
    )
    assert "nfn_native_tile_bf16_bits_add_bias_inplace_float32" in tile_payload["available_native_kernels"]
    assert (
        "nfn_native_tile_scaled_dot_product_attention_packed_qkv_bf16_float32"
        in tile_payload["available_native_kernels"]
    )
    assert (
        "nfn_native_tile_scaled_dot_product_attention_packed_qkv_store_lse_bf16_float32"
        in tile_payload["available_native_kernels"]
    )
    assert (
        "nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_from_merged_grad_float32"
        in tile_payload["available_native_kernels"]
    )
    assert (
        "nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_from_saved_lse_bf16_from_merged_grad_float32"
        in tile_payload["available_native_kernels"]
    )
    assert (
        "nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_bf16_bits_from_merged_grad_float32"
        in tile_payload["available_native_kernels"]
    )
    assert (
        "nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_bf16_bits_from_saved_lse_bf16_from_merged_grad_float32"
        in tile_payload["available_native_kernels"]
    )
    assert "nfn_native_tile_attention_forward_stats_reset" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_attention_forward_row_launch_count" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_attention_backward_dprep_timing_us" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_attention_backward_tk_timing_us" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_attention_forward_row_fallback_count" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_attention_forward_scalar_launch_count" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_attention_forward_row_prelaunch_clear_error" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_attention_forward_row_prelaunch_peek_error" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_attention_forward_row_grid_x" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_attention_forward_row_block_x" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_attention_forward_row_attr_status" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_attention_forward_row_attr_const_size_bytes" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_linear_bf16_float32" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_linear_backward_weight_accumulate_bf16_float32" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_bf16_bits_to_float32" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_store_mlp_activations_bf16_float32" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_restore_mlp_activations_bf16_float32" in tile_payload["available_native_kernels"]
    assert (
        "nfn_native_tile_layer_norm_backward_input_residual_add_with_stats_float32"
        in tile_payload["available_native_kernels"]
    )
    assert (
        "nfn_native_tile_layer_norm_backward_input_residual_add_with_stats_bf16_bits_float32"
        in tile_payload["available_native_kernels"]
    )
    assert (
        "nfn_native_tile_layer_norm_backward_affine_accumulate_with_stats_bf16_bits_float32"
        in tile_payload["available_native_kernels"]
    )
    assert "nfn_native_tile_linear_backward_weight_accumulate_bf16_bits_float32" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_linear_backward_input_dgelu_bf16_bits_float32" in tile_payload["available_native_kernels"]
    assert (
        "nfn_native_tile_linear_backward_input_dgelu_weight_bf16_bits_float32"
        in tile_payload["available_native_kernels"]
    )
    assert (
        "nfn_native_tile_linear_backward_input_dgelu_weight_bf16_bits_only_float32"
        in tile_payload["available_native_kernels"]
    )
    assert (
        "nfn_native_tile_linear_backward_input_dgelu_bf16_bits_weight_bf16_bits_only_float32"
        in tile_payload["available_native_kernels"]
    )
    assert "nfn_native_tile_linear_weight_bf16_gelu_bf16_float32" in tile_payload["available_native_kernels"]
    assert (
        "nfn_native_tile_linear_bf16_input_weight_bf16_gelu_bf16_float32"
        in tile_payload["available_native_kernels"]
    )
    assert "nfn_native_tile_gelu_backward_inplace_bf16_bits_float32" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_trainer_linear_stats_reset" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_trainer_linear_bf16_cache_reset" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_trainer_linear_bf16_gemm_count" in tile_payload["available_native_kernels"]
    assert (
        "nfn_native_tile_trainer_linear_bf16_gemm_fast16bf_request_count"
        in tile_payload["available_native_kernels"]
    )
    assert "nfn_native_tile_trainer_linear_tk_gemm_count" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_trainer_linear_tk_float_out_gemm_count" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_trainer_linear_tk_dweight_gemm_count" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_trainer_linear_tk_dgelu_dinput_gemm_count" in tile_payload[
        "available_native_kernels"
    ]
    assert "nfn_native_tile_trainer_linear_tk_sm120_k_tile" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_trainer_linear_tk_sm120_grad_k_tile" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_trainer_linear_tk_sm120_super_m" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_trainer_linear_tk_sm120_dinput_super_m" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_trainer_linear_tk_sm120_dweight_super_m" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_trainer_linear_tk_sm120_huge_n_k_tile" in tile_payload["available_native_kernels"]
    assert (
        "nfn_native_tile_trainer_linear_tk_sm120_fast_dgelu_enabled"
        in tile_payload["available_native_kernels"]
    )
    assert (
        "nfn_native_tile_trainer_linear_tk_sm120_approx_dgelu_tanh_enabled"
        in tile_payload["available_native_kernels"]
    )
    assert "nfn_native_tile_trainer_linear_cublaslt_gemm_count" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_trainer_linear_cublaslt_bgrad_gemm_count" in tile_payload["available_native_kernels"]
    assert (
        "nfn_native_tile_trainer_linear_cublaslt_bgrad_direct_write_count"
        in tile_payload["available_native_kernels"]
    )
    assert (
        "nfn_native_tile_trainer_linear_cublaslt_bgrad_accumulate_count"
        in tile_payload["available_native_kernels"]
    )
    assert "nfn_native_tile_trainer_linear_sgemm_count" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_trainer_bf16_to_f32_vec4_count" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_trainer_linear_bf16_a_pack_count" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_trainer_linear_bf16_a_cache_hit_count" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_trainer_linear_bf16_cache_reset_count" in tile_payload["available_native_kernels"]
    assert (
        "nfn_native_tile_trainer_linear_bf16_workspace_allocation_count"
        in tile_payload["available_native_kernels"]
    )
    assert "nfn_native_tile_trainer_linear_bf16_cached_a_capacity" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_trainer_linear_bf16_cache_entry_count" in tile_payload["available_native_kernels"]
    assert (
        "nfn_native_tile_trainer_linear_cublas_grouped_bf16_gemm_probe_status"
        in tile_payload["available_native_kernels"]
    )
    assert "nfn_native_tile_split_qkv_to_heads_float32" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_split_qkv_to_heads_add_bias_float32" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_merge_heads_to_qkv_float32" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_gelu_add_bias_float32" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_linear_bias_residual_add_float32" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_linear_bias_residual_add_bf16_linear_float32" in tile_payload[
        "available_native_kernels"
    ]
    assert tile_payload["required_native_work"] == []
    assert any("SM120 throughput gap" in item for item in tile_payload["remaining_validation"])
    assert tile_payload["schedule"]["sample_every_steps"] == 20000
    assert tile_payload["schedule"]["generate_tokens"] == 144
    assert tile_payload["schedule"]["checkpoint_every_steps"] == 200

    megakernel_template_plan = subprocess.run(
        [
            str(cli),
            "--dataset-alias",
            str(dataset_path),
            "--backend",
            "tile-cuda",
            "--template-name",
            "gpt2_megakernel",
            "--print-plan",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert megakernel_template_plan.returncode == 0, megakernel_template_plan.stderr
    megakernel_template_payload = json.loads(megakernel_template_plan.stdout)
    assert megakernel_template_payload["template_name"] == "gpt2_megakernel"
    assert megakernel_template_payload["resolved_native_template_name"] == "gpt2_megakernel"
    assert megakernel_template_payload["selected_graph_support_status"] == "native-transformer-lm"
    assert megakernel_template_payload["selected_graph_native_runnable"] is True

    moa_template_plan = subprocess.run(
        [
            str(cli),
            "--dataset-alias",
            str(dataset_path),
            "--backend",
            "tile-cuda",
            "--template-name",
            "gpt2_moa",
            "--print-plan",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert moa_template_plan.returncode == 0, moa_template_plan.stderr
    moa_template_payload = json.loads(moa_template_plan.stdout)
    assert moa_template_payload["template_name"] == "gpt2_moa"
    assert moa_template_payload["resolved_native_template_name"] == "gpt2_moa"
    assert moa_template_payload["native_cuda_activation"] == "moa"
    assert moa_template_payload["selected_graph_support_status"] == "native-transformer-lm"
    assert moa_template_payload["selected_graph_native_runnable"] is True

    for preset in SHIPPED_GPT_TEMPLATE_PRESETS:
        preset_plan = subprocess.run(
            [
                str(cli),
                "--dataset-alias",
                str(dataset_path),
                "--backend",
                "tile-cuda",
                "--template-name",
                preset,
                "--print-plan",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert preset_plan.returncode == 0, f"{preset}: {preset_plan.stderr}"
        preset_payload = json.loads(preset_plan.stdout)
        assert preset_payload["template_name"] == preset
        assert preset_payload["template_known"] is True
        assert preset_payload["shipped_template_catalog_count"] == len(SHIPPED_GPT_TEMPLATE_PRESETS)
        if preset in {"gpt2", "gpt2_modern", "gpt2_megakernel", "gpt2_moa"}:
            assert preset_payload["selected_graph_support_status"] == "native-transformer-lm"
            assert preset_payload["selected_graph_native_runnable"] is True
            assert preset_payload["native_geometry_contract"]["shape_source"] == "selected_dense_gpt_geometry"
            assert preset_payload["native_geometry_contract"]["template_geometry_dynamic"] is False
            assert preset_payload["native_geometry_contract"]["geometry_matches_compiled_loop"] is True
        elif preset in {"nanogpt", "nanogpt_modern", "nanogpt_megakernel"}:
            assert preset_payload["selected_graph_support_status"] == "native-transformer-lm"
            assert preset_payload["selected_graph_native_runnable"] is True
            assert preset_payload["native_geometry_contract"]["selected_template_geometry"] == {
                "source": "template",
                "model_dim": 320,
                "num_heads": 5,
                "head_dim": 64,
                "mlp_multiplier": 4,
                "vocab_size": 50257,
                "padded_vocab_size": 50304,
                "num_layers": 5,
                "seq_len": 1024,
                "dropout_p": 0.1,
            }
            assert preset_payload["native_geometry_contract"]["template_geometry_dynamic"] is True
            assert preset_payload["native_geometry_contract"]["geometry_matches_compiled_loop"] is True
        else:
            assert preset_payload["selected_graph_support_status"] == "template-native-trainer-missing"
            assert preset_payload["selected_graph_native_runnable"] is False

    gpt3_template_plan = subprocess.run(
        [
            str(cli),
            "--dataset-alias",
            str(dataset_path),
            "--backend",
            "tile-cuda",
            "--template-name",
            "gpt3",
            "--print-plan",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert gpt3_template_plan.returncode == 0, gpt3_template_plan.stderr
    gpt3_template_payload = json.loads(gpt3_template_plan.stdout)
    assert gpt3_template_payload["template_name"] == "gpt3"
    assert gpt3_template_payload["resolved_native_template_name"] == "gpt3"
    assert gpt3_template_payload["template_known"] is True
    assert gpt3_template_payload["selected_graph_support_status"] == "native-transformer-lm"
    assert gpt3_template_payload["selected_graph_native_runnable"] is True
    assert gpt3_template_payload["native_geometry_contract"]["selected_template_geometry"]["seq_len"] == 2048
    assert gpt3_template_payload["native_geometry_contract"]["seq_len"] == 2048
    assert gpt3_template_payload["shape"]["seq_len"] == 2048
    assert gpt3_template_payload["shape"]["batch_size"] == 32
    assert gpt3_template_payload["schedule"]["train_batch_tokens"] == 524288

    unsupported_template = subprocess.run(
        [
            str(cli),
            "--dataset-alias",
            str(dataset_path),
            "--backend",
            "tile-cuda",
            "--template-name",
            "llama",
            "--max-steps",
            "1",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert unsupported_template.returncode == 2
    unsupported_payload = json.loads(unsupported_template.stdout)
    assert unsupported_payload["template_name"] == "llama"
    assert unsupported_payload["selected_graph_support_status"] == "template-native-trainer-missing"
    assert unsupported_payload["selected_graph_native_runnable"] is False
    assert unsupported_payload["status"] == "selected-graph-native-trainer-missing"
    assert unsupported_payload["template_known"] is True
    assert unsupported_payload["token_shards_resolved"] is False

    unknown_template = subprocess.run(
        [
            str(cli),
            "--dataset-alias",
            str(dataset_path),
            "--backend",
            "tile-cuda",
            "--template-name",
            "typo-not-a-shipped-template",
            "--max-steps",
            "1",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert unknown_template.returncode == 2
    unknown_payload = json.loads(unknown_template.stdout)
    assert unknown_payload["template_name"] == "typo_not_a_shipped_template"
    assert unknown_payload["template_known"] is False
    assert unknown_payload["selected_graph_support_status"] == "unknown-template"
    assert unknown_payload["selected_graph_native_runnable"] is False
    assert unknown_payload["status"] == "unknown-template"
    assert unknown_payload["token_shards_resolved"] is False

    unsupported_missing_dataset = subprocess.run(
        [
            str(cli),
            "--dataset-alias",
            str(tmp_path / "missing-native-cache"),
            "--backend",
            "tile-cuda",
            "--template-name",
            "llama",
            "--max-steps",
            "1",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert unsupported_missing_dataset.returncode == 2
    assert unsupported_missing_dataset.stderr == ""
    unsupported_missing_payload = json.loads(unsupported_missing_dataset.stdout)
    assert unsupported_missing_payload["selected_graph_support_status"] == "template-native-trainer-missing"
    assert unsupported_missing_payload["status"] == "selected-graph-native-trainer-missing"
    assert unsupported_missing_payload["dataset_alias"].endswith("missing-native-cache")
    assert unsupported_missing_payload["token_shards_resolved"] is False
    assert unsupported_missing_payload["dataset_path"] == ""

    custom_graph_path = tmp_path / "custom-graph.json"
    custom_graph_path.write_text('{"nodes": {}, "edges": {}}\n', encoding="utf-8")
    custom_graph = subprocess.run(
        [
            str(cli),
            "--dataset-alias",
            str(dataset_path),
            "--backend",
            "tile-cuda",
            "--graph-file",
            str(custom_graph_path),
            "--print-plan",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert custom_graph.returncode == 0, custom_graph.stderr
    custom_payload = json.loads(custom_graph.stdout)
    assert custom_payload["graph_file"].endswith("custom-graph.json")
    assert custom_payload["graph_file_exists"] is True
    assert custom_payload["graph_file_size_bytes"] == custom_graph_path.stat().st_size
    assert custom_payload["template_known"] is True
    assert custom_payload["shipped_template_catalog_count"] == len(SHIPPED_GPT_TEMPLATE_PRESETS)
    assert custom_payload["selected_graph_support_status"] == "custom-graph-native-trainer-missing"
    assert custom_payload["selected_graph_native_runnable"] is False
    assert custom_payload["native_geometry_contract"]["graph_file"].endswith("custom-graph.json")
    assert custom_payload["native_geometry_contract"]["graph_file_exists"] is True
    assert custom_payload["native_geometry_contract"]["graph_file_size_bytes"] == custom_graph_path.stat().st_size
    assert custom_payload["native_geometry_contract"]["selector_native_runnable"] is False
    assert custom_payload["native_geometry_contract"]["custom_graph_geometry_dynamic"] is False

    native_custom_graph_path = tmp_path / "native-compatible-gpt-graph.json"
    native_custom_graph_path.write_text(
        json.dumps(
            {
                "graph_settings": {
                    "torch_config": {
                        "template_spec": {
                            "model_dim": 768,
                            "num_layers": 12,
                            "vocab_size": 50257,
                            "padded_vocab_size": 50304,
                            "seq_len": 2048,
                            "block_spec": {
                                "family": "gpt2",
                                "num_heads": 12,
                                "dropout_p": 0.0,
                                "mlp_multiplier": 4.0,
                            },
                            "template": {
                                "objective": "ar",
                                "backbone": "gpt2",
                                "sparsity": "dense",
                                "router_mode": "none",
                            },
                        }
                    }
                },
                "nodes": {},
                "edges": {},
            }
        ),
        encoding="utf-8",
    )
    native_custom_graph = subprocess.run(
        [
            str(cli),
            "--dataset-alias",
            str(dataset_path),
            "--backend",
            "tile-cuda",
            "--graph-file",
            str(native_custom_graph_path),
            "--print-plan",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert native_custom_graph.returncode == 0, native_custom_graph.stderr
    native_custom_payload = json.loads(native_custom_graph.stdout)
    assert native_custom_payload["graph_file"].endswith("native-compatible-gpt-graph.json")
    assert native_custom_payload["architecture_source"] == "custom_graph"
    assert native_custom_payload["selected_graph_support_status"] == "native-transformer-lm"
    assert native_custom_payload["selected_graph_native_runnable"] is True
    assert native_custom_payload["status"] == "native-transformer-lm-ready"
    assert native_custom_payload["native_geometry_contract"]["shape_source"] == "custom_graph_template_spec"
    assert native_custom_payload["native_geometry_contract"]["custom_graph_geometry_dynamic"] is True
    assert native_custom_payload["native_geometry_contract"]["selected_template_geometry"]["source"] == "custom_graph_template_spec"
    assert native_custom_payload["native_geometry_contract"]["geometry_matches_compiled_loop"] is True
    assert native_custom_payload["native_geometry_contract"]["selector_native_runnable"] is True
    assert native_custom_payload["shape"]["seq_len"] == 2048
    assert native_custom_payload["shape"]["batch_size"] == 32
    assert native_custom_payload["schedule"]["train_batch_tokens"] == 524288

    missing_custom_graph = subprocess.run(
        [
            str(cli),
            "--dataset-alias",
            str(dataset_path),
            "--backend",
            "tile-cuda",
            "--graph-file",
            str(tmp_path / "missing-custom-graph.json"),
            "--print-plan",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert missing_custom_graph.returncode == 0, missing_custom_graph.stderr
    missing_custom_payload = json.loads(missing_custom_graph.stdout)
    assert missing_custom_payload["graph_file"].endswith("missing-custom-graph.json")
    assert missing_custom_payload["graph_file_exists"] is False
    assert missing_custom_payload["graph_file_size_bytes"] == -1
    assert missing_custom_payload["selected_graph_support_status"] == "custom-graph-file-missing"
    assert missing_custom_payload["status"] == "custom-graph-file-missing"
    assert missing_custom_payload["selected_graph_native_runnable"] is False
    assert missing_custom_payload["native_geometry_contract"]["graph_file_exists"] is False
    assert missing_custom_payload["native_geometry_contract"]["graph_file_size_bytes"] == -1

    missing_ops = subprocess.run(
        [
            str(cli),
            "--dataset-alias",
            str(dataset_path),
            "--check-tile-ops",
            "--tile-ops-lib",
            str(tmp_path / "missing-tile-ops.so"),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert missing_ops.returncode == 2
    missing_payload = json.loads(missing_ops.stdout)
    assert missing_payload["tile_ops_check"]["loaded"] is False
    assert missing_payload["tile_ops_check"]["all_required_symbols_found"] is False
    assert missing_payload["tile_ops_check"]["optimized_optimizer_contract_loaded"] is False
    assert (
        missing_payload["tile_ops_check"]["optimized_optimizer_contract_error"]
        == "missing optimized many-tensor/device-scale AdamW Tile-CUDA symbols"
    )
    assert missing_payload["optimized_kernel_contract_required"] is True
    assert missing_payload["optimized_kernel_contract_basic_fallback_allowed"] is False
    assert missing_payload["optimized_kernel_contract_passed"] is False
    assert (
        missing_payload["optimized_kernel_contract_error"]
        == "missing optimized many-tensor/device-scale AdamW Tile-CUDA symbols"
    )
    assert "nfn_native_tile_adamw_step_many_with_device_scale_float32" in (
        missing_payload["tile_ops_check"]["optimized_optimizer_missing_symbols"]
    )
    assert missing_payload["token_shards_resolved"] is False
    assert missing_payload["train_shard"] == ""
    assert missing_payload["val_shard"] == ""

    missing_ops_profile_json = tmp_path / "missing-tile-ops-profile.json"
    missing_ops_profile = subprocess.run(
        [
            str(cli),
            "--dataset-alias",
            str(dataset_path),
            "--check-tile-ops",
            "--tile-ops-lib",
            str(tmp_path / "missing-tile-ops.so"),
            "--profile-json",
            str(missing_ops_profile_json),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert missing_ops_profile.returncode == 2
    assert missing_ops_profile.stdout == ""
    assert "native-tile-ops-check-failed:" in missing_ops_profile.stderr
    assert "missing-tile-ops.so" in missing_ops_profile.stderr
    missing_ops_profile_payload = json.loads(missing_ops_profile_json.read_text(encoding="utf-8"))
    assert missing_ops_profile_payload["tile_ops_check"]["loaded"] is False
    assert missing_ops_profile_payload["tile_ops_check"]["all_required_symbols_found"] is False
    assert missing_ops_profile_payload["tile_ops_check"]["optimized_optimizer_contract_loaded"] is False
    assert missing_ops_profile_payload["optimized_kernel_contract_required"] is True
    assert missing_ops_profile_payload["optimized_kernel_contract_passed"] is False

    missing_dataset_check = subprocess.run(
        [
            str(cli),
            "--dataset-alias",
            str(tmp_path / "does-not-exist"),
            "--check-tile-ops",
            "--tile-ops-lib",
            str(tmp_path / "missing-tile-ops.so"),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert missing_dataset_check.returncode == 2
    missing_dataset_payload = json.loads(missing_dataset_check.stdout)
    assert missing_dataset_payload["tile_ops_check"]["loaded"] is False
    assert missing_dataset_payload["token_shards_resolved"] is False
    assert "fineweb_train" not in missing_dataset_check.stderr

    missing_smoke = subprocess.run(
        [
            str(cli),
            "--dataset-alias",
            str(dataset_path),
            "--smoke-tile-ops",
            "--tile-ops-lib",
            str(tmp_path / "missing-tile-ops.so"),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert missing_smoke.returncode == 2
    smoke_payload = json.loads(missing_smoke.stdout)
    assert smoke_payload["model_family"] == "gpt"
    assert smoke_payload["backend"] == "tile-cuda"
    assert smoke_payload["smoke"] == "tile_ops_fill"
    assert smoke_payload["loaded"] is False
    assert smoke_payload["kernel_loaded"] is False
    assert smoke_payload["passed"] is False

    missing_dataset_smoke = subprocess.run(
        [
            str(cli),
            "--dataset-alias",
            str(tmp_path / "does-not-exist"),
            "--smoke-lm-step",
            "--tile-ops-lib",
            str(tmp_path / "missing-tile-ops.so"),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert missing_dataset_smoke.returncode == 2
    missing_dataset_smoke_payload = json.loads(missing_dataset_smoke.stdout)
    assert missing_dataset_smoke_payload["smoke"] == "lm_step"
    assert missing_dataset_smoke_payload["token_shards_resolved"] is False
    assert missing_dataset_smoke_payload["loaded"] is False
    assert "fineweb_train" not in missing_dataset_smoke.stderr

    missing_optimizer_smoke = subprocess.run(
        [
            str(cli),
            "--dataset-alias",
            str(dataset_path),
            "--smoke-optimizer-step",
            "--tile-ops-lib",
            str(tmp_path / "missing-tile-ops.so"),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert missing_optimizer_smoke.returncode == 2
    optimizer_payload = json.loads(missing_optimizer_smoke.stdout)
    assert optimizer_payload["model_family"] == "gpt"
    assert optimizer_payload["backend"] == "tile-cuda"
    assert optimizer_payload["smoke"] == "optimizer_step"
    assert optimizer_payload["loaded"] is False
    assert optimizer_payload["fill_kernel_loaded"] is False
    assert optimizer_payload["optimizer_kernel_loaded"] is False
    assert optimizer_payload["parameter_buffer_count"] == 2 + (12 * 12) + 2
    assert optimizer_payload["total_parameters"] == tile_payload["parameter_layout"]["total_parameters"]
    assert optimizer_payload["passed"] is False

    missing_lm_smoke = subprocess.run(
        [
            str(cli),
            "--dataset-alias",
            str(dataset_path),
            "--smoke-lm-step",
            "--tile-ops-lib",
            str(tmp_path / "missing-tile-ops.so"),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert missing_lm_smoke.returncode == 2
    lm_payload = json.loads(missing_lm_smoke.stdout)
    assert lm_payload["model_family"] == "gpt"
    assert lm_payload["backend"] == "tile-cuda"
    assert lm_payload["smoke"] == "lm_step"
    assert lm_payload["loaded"] is False
    assert lm_payload["rows"] == 2
    assert lm_payload["vocab"] == 50257
    assert lm_payload["padded_vocab"] == 50304
    assert lm_payload["model_dim"] == 768
    assert "nfn_native_tile_token_cross_entropy_backward_with_workspace_float32" in lm_payload["kernels"]
    assert lm_payload["passed"] is False

    missing_attention_smoke = subprocess.run(
        [
            str(cli),
            "--dataset-alias",
            str(dataset_path),
            "--smoke-attention-step",
            "--tile-ops-lib",
            str(tmp_path / "missing-tile-ops.so"),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert missing_attention_smoke.returncode == 2
    attention_payload = json.loads(missing_attention_smoke.stdout)
    assert attention_payload["model_family"] == "gpt"
    assert attention_payload["backend"] == "tile-cuda"
    assert attention_payload["smoke"] == "attention_step"
    assert attention_payload["loaded"] is False
    assert attention_payload["batch"] == 1
    assert attention_payload["seq"] == 2
    assert attention_payload["heads"] == 1
    assert attention_payload["model_dim"] == 768
    assert "nfn_native_tile_scaled_dot_product_attention_backward_float32" in attention_payload["kernels"]
    assert attention_payload["passed"] is False

    missing_mlp_smoke = subprocess.run(
        [
            str(cli),
            "--dataset-alias",
            str(dataset_path),
            "--smoke-mlp-step",
            "--tile-ops-lib",
            str(tmp_path / "missing-tile-ops.so"),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert missing_mlp_smoke.returncode == 2
    mlp_payload = json.loads(missing_mlp_smoke.stdout)
    assert mlp_payload["model_family"] == "gpt"
    assert mlp_payload["backend"] == "tile-cuda"
    assert mlp_payload["smoke"] == "mlp_step"
    assert mlp_payload["loaded"] is False
    assert mlp_payload["rows"] == 2
    assert mlp_payload["model_dim"] == 768
    assert mlp_payload["hidden_dim"] == 3072
    assert "nfn_native_tile_gelu_backward_float32" in mlp_payload["kernels"]
    assert mlp_payload["passed"] is False

    missing_norm_residual_smoke = subprocess.run(
        [
            str(cli),
            "--dataset-alias",
            str(dataset_path),
            "--smoke-norm-residual-step",
            "--tile-ops-lib",
            str(tmp_path / "missing-tile-ops.so"),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert missing_norm_residual_smoke.returncode == 2
    norm_residual_payload = json.loads(missing_norm_residual_smoke.stdout)
    assert norm_residual_payload["model_family"] == "gpt"
    assert norm_residual_payload["backend"] == "tile-cuda"
    assert norm_residual_payload["smoke"] == "norm_residual_step"
    assert norm_residual_payload["loaded"] is False
    assert norm_residual_payload["rows"] == 2
    assert norm_residual_payload["model_dim"] == 768
    assert "nfn_native_tile_layer_norm_backward_input_float32" in norm_residual_payload["kernels"]
    assert "nfn_native_tile_gradient_accumulate_float32" in norm_residual_payload["kernels"]
    assert norm_residual_payload["passed"] is False

    missing_transformer_block_smoke = subprocess.run(
        [
            str(cli),
            "--dataset-alias",
            str(dataset_path),
            "--smoke-transformer-block-step",
            "--tile-ops-lib",
            str(tmp_path / "missing-tile-ops.so"),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert missing_transformer_block_smoke.returncode == 2
    transformer_block_payload = json.loads(missing_transformer_block_smoke.stdout)
    assert transformer_block_payload["model_family"] == "gpt"
    assert transformer_block_payload["backend"] == "tile-cuda"
    assert transformer_block_payload["smoke"] == "transformer_block_step"
    assert transformer_block_payload["loaded"] is False
    assert transformer_block_payload["batch"] == 1
    assert transformer_block_payload["seq"] == 2
    assert transformer_block_payload["heads"] == 12
    assert transformer_block_payload["head_dim"] == 64
    assert transformer_block_payload["model_dim"] == 768
    assert transformer_block_payload["hidden_dim"] == 3072
    assert transformer_block_payload["weight_update_count"] == 12
    assert "nfn_native_tile_reshape_heads_float32" in transformer_block_payload["kernels"]
    assert "nfn_native_tile_merge_heads_float32" in transformer_block_payload["kernels"]
    assert "nfn_native_tile_linear_backward_bias_float32" in transformer_block_payload["kernels"]
    assert "nfn_native_tile_scaled_dot_product_attention_backward_float32" in transformer_block_payload["kernels"]
    assert "nfn_native_tile_gradient_accumulate_float32" in transformer_block_payload["kernels"]
    assert transformer_block_payload["passed"] is False

    missing_transformer_lm_smoke = subprocess.run(
        [
            str(cli),
            "--dataset-alias",
            str(dataset_path),
            "--smoke-transformer-lm-step",
            "--tile-ops-lib",
            str(tmp_path / "missing-tile-ops.so"),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert missing_transformer_lm_smoke.returncode == 2
    transformer_lm_payload = json.loads(missing_transformer_lm_smoke.stdout)
    assert transformer_lm_payload["model_family"] == "gpt"
    assert transformer_lm_payload["backend"] == "tile-cuda"
    assert transformer_lm_payload["smoke"] == "transformer_lm_step"
    assert transformer_lm_payload["loaded"] is False
    assert transformer_lm_payload["batch_loaded"] is True
    assert transformer_lm_payload["batch"] == 1
    assert transformer_lm_payload["heads"] == 1
    assert transformer_lm_payload["head_dim"] == 4
    assert transformer_lm_payload["seq"] == 2
    assert transformer_lm_payload["vocab"] == 50257
    assert transformer_lm_payload["padded_vocab"] == 50304
    assert transformer_lm_payload["model_dim"] == 4
    assert transformer_lm_payload["hidden_dim"] == 8
    assert transformer_lm_payload["weight_update_count"] == 16
    assert "nfn_native_tile_token_embedding_float32" in transformer_lm_payload["kernels"]
    assert "nfn_native_tile_scaled_dot_product_attention_backward_float32" in transformer_lm_payload["kernels"]
    assert "nfn_native_tile_token_cross_entropy_backward_with_workspace_float32" in transformer_lm_payload["kernels"]
    assert "nfn_native_tile_token_embedding_backward_weight_float32" in transformer_lm_payload["kernels"]
    assert transformer_lm_payload["passed"] is False

    missing_embedding_lm_smoke = subprocess.run(
        [
            str(cli),
            "--dataset-alias",
            str(dataset_path),
            "--smoke-embedding-lm-step",
            "--tile-ops-lib",
            str(tmp_path / "missing-tile-ops.so"),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert missing_embedding_lm_smoke.returncode == 2
    embedding_lm_payload = json.loads(missing_embedding_lm_smoke.stdout)
    assert embedding_lm_payload["model_family"] == "gpt"
    assert embedding_lm_payload["backend"] == "tile-cuda"
    assert embedding_lm_payload["smoke"] == "embedding_lm_step"
    assert embedding_lm_payload["loaded"] is False
    assert embedding_lm_payload["batch_loaded"] is True
    assert embedding_lm_payload["batch"] == 1
    assert embedding_lm_payload["seq"] == 2
    assert embedding_lm_payload["rows"] == 2
    assert embedding_lm_payload["vocab"] == 50257
    assert embedding_lm_payload["padded_vocab"] == 50304
    assert embedding_lm_payload["model_dim"] == 768
    assert embedding_lm_payload["weight_update_count"] == 4
    assert "nfn_native_tile_absolute_position_embedding_float32" in embedding_lm_payload["kernels"]
    assert "nfn_native_tile_token_cross_entropy_backward_with_workspace_float32" in embedding_lm_payload["kernels"]
    assert "nfn_native_tile_absolute_position_embedding_backward_float32" in embedding_lm_payload["kernels"]
    assert embedding_lm_payload["passed"] is False

    missing_train_embedding_lm = subprocess.run(
        [
            str(cli),
            "--dataset-alias",
            str(dataset_path),
            "--train-embedding-lm",
            "--tile-ops-lib",
            str(tmp_path / "missing-tile-ops.so"),
            "--batch-size",
            "1",
            "--train-seq-len",
            "2",
            "--max-steps",
            "2",
            "--eval-every-steps",
            "1",
            "--eval-batches",
            "1",
            "--eval-batch-size",
            "1",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert missing_train_embedding_lm.returncode == 2
    train_embedding_payload = json.loads(missing_train_embedding_lm.stdout)
    assert train_embedding_payload["model_family"] == "gpt"
    assert train_embedding_payload["backend"] == "tile-cuda"
    assert train_embedding_payload["status"] == "native-embedding-lm-failed"
    assert train_embedding_payload["loaded"] is False
    assert train_embedding_payload["batch_size"] == 1
    assert train_embedding_payload["eval_batch_size"] == 1
    assert train_embedding_payload["seq_len"] == 2
    assert train_embedding_payload["vocab"] == 50257
    assert train_embedding_payload["padded_vocab"] == 50304
    assert train_embedding_payload["max_steps"] == 2
    assert train_embedding_payload["eval_every_steps"] == 1
    assert train_embedding_payload["eval_batches"] == 1
    assert train_embedding_payload["steps_completed"] == 0
    assert "nfn_native_tile_absolute_position_embedding_float32" in train_embedding_payload["kernels"]
    assert "nfn_native_tile_layer_norm_backward_input_float32" in train_embedding_payload["kernels"]
    assert "nfn_native_tile_absolute_position_embedding_backward_float32" in train_embedding_payload["kernels"]
    assert train_embedding_payload["passed"] is False

    missing_train_transformer_lm = subprocess.run(
        [
            str(cli),
            "--dataset-alias",
            str(dataset_path),
            "--train-transformer-lm",
            "--tile-ops-lib",
            str(tmp_path / "missing-tile-ops.so"),
            "--batch-size",
            "1",
            "--train-seq-len",
            "2",
            "--max-steps",
            "2",
            "--eval-every-steps",
            "1",
            "--eval-batches",
            "1",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert missing_train_transformer_lm.returncode == 2
    train_transformer_payload = json.loads(missing_train_transformer_lm.stdout)
    assert train_transformer_payload["model_family"] == "gpt"
    assert train_transformer_payload["backend"] == "tile-cuda"
    assert train_transformer_payload["status"] == "native-transformer-lm-failed"
    assert train_transformer_payload["loaded"] is False
    assert train_transformer_payload["cuda_runtime_loaded"] is False

    startup_zero_steps_missing_ops = subprocess.run(
        [
            str(cli),
            "--dataset-alias",
            str(dataset_path),
            "--train-transformer-lm",
            "--startup-only",
            "--tile-ops-lib",
            str(tmp_path / "missing-tile-ops.so"),
            "--batch-size",
            "1",
            "--train-seq-len",
            "2",
            "--max-steps",
            "0",
            "--eval-every-steps",
            "0",
            "--eval-batches",
            "1",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert startup_zero_steps_missing_ops.returncode == 2
    startup_zero_steps_payload = json.loads(startup_zero_steps_missing_ops.stdout)
    assert startup_zero_steps_payload["status"] == "native-transformer-lm-failed"
    assert startup_zero_steps_payload["startup_only"] is True
    assert startup_zero_steps_payload["max_steps"] == 0
    assert "max_steps" not in startup_zero_steps_payload["error"]
    assert "missing-tile-ops.so" in startup_zero_steps_payload["error"]

    missing_train_transformer_profile_json = tmp_path / "missing-transformer-lm-profile.json"
    missing_train_transformer_profile = subprocess.run(
        [
            str(cli),
            "--dataset-alias",
            str(dataset_path),
            "--train-transformer-lm",
            "--tile-ops-lib",
            str(tmp_path / "missing-tile-ops.so"),
            "--batch-size",
            "1",
            "--train-seq-len",
            "2",
            "--max-steps",
            "2",
            "--eval-every-steps",
            "1",
            "--eval-batches",
            "1",
            "--profile-json",
            str(missing_train_transformer_profile_json),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert missing_train_transformer_profile.returncode == 2
    assert missing_train_transformer_profile.stdout == ""
    assert "native-transformer-lm-failed:" in missing_train_transformer_profile.stderr
    assert "missing-tile-ops.so" in missing_train_transformer_profile.stderr
    train_transformer_profile_payload = json.loads(
        missing_train_transformer_profile_json.read_text(encoding="utf-8")
    )
    assert train_transformer_profile_payload["status"] == "native-transformer-lm-failed"
    assert train_transformer_profile_payload["loaded"] is False
    assert train_transformer_profile_payload["cuda_runtime_loaded"] is False
    assert train_transformer_profile_payload["sample_every_steps"] == 20000
    assert train_transformer_profile_payload["generate_tokens"] == 144
    assert train_transformer_profile_payload["checkpoint_every_steps"] == 200
    assert train_transformer_profile_payload["train_time_sampling_enabled"] is False
    assert train_transformer_profile_payload["periodic_checkpoint_enabled"] is False
    assert train_transformer_profile_payload["final_checkpoint_export_enabled"] is True

    assert train_transformer_payload["native_geometry_contract"]["name"] == "native-dense-gpt-transformer"
    assert train_transformer_payload["native_geometry_contract"]["shape_source"] == "selected_dense_gpt_geometry"
    assert train_transformer_payload["native_geometry_contract"]["selector_native_runnable"] is True
    assert train_transformer_payload["native_geometry_contract"]["template_geometry_dynamic"] is True
    assert train_transformer_payload["native_geometry_contract"]["custom_graph_geometry_dynamic"] is False
    assert train_transformer_payload["native_geometry_contract"]["model_dim"] == 768
    assert train_transformer_payload["native_geometry_contract"]["seq_len"] == 2
    assert train_transformer_payload["batch_size"] == 1
    assert train_transformer_payload["seq_len"] == 2
    assert train_transformer_payload["train_loss_every_steps"] == 0
    assert (
        train_transformer_payload["train_loss_device_accumulation_strategy"]
        == "optimizer-step-device-scalar-accumulate"
    )
    assert train_transformer_payload["train_loss_host_copy_scope"] == "once-per-logged-optimizer-step"
    assert train_transformer_payload["train_loss_host_d2h_count"] == 0
    assert train_transformer_payload["train_loss_host_d2h_copies_per_logged_step"] == 0
    assert (
        train_transformer_payload["train_loss_microbatch_host_d2h_copies_elided_per_logged_step"]
        == 0
    )
    assert train_transformer_payload["sample_every_steps"] == 20000
    assert train_transformer_payload["generate_tokens"] == 144
    assert train_transformer_payload["checkpoint_every_steps"] == 200
    assert train_transformer_payload["train_time_sampling_enabled"] is False
    assert train_transformer_payload["periodic_checkpoint_enabled"] is False
    assert train_transformer_payload["final_checkpoint_export_enabled"] is True
    assert train_transformer_payload["trained_layers"] == 12
    assert train_transformer_payload["target_layers"] == 12
    assert train_transformer_payload["layer_evo"] == {
        "enabled": False,
        "runtime_enabled": False,
        "graph_editor_tensor_flow": False,
        "target_parameter": "block_6.ln1.weight",
        "target_parameter_dtype": "float32",
        "layer_index": 6,
        "interval": 10,
        "population": 8,
        "mutation_scale": 0.02,
        "parameter_elements": 0,
        "candidate_elements": 0,
        "runs": 0,
        "mutate_kernel_launches": 0,
        "select_kernel_launches": 0,
        "adopt_kernel_launches": 0,
        "forward_candidate_evals": 0,
        "forward_candidate_eval_enabled": False,
        "candidate_loss_source": "native-forward-loss-device-resident-current-batch",
        "candidate_loss_transport": "device-to-device",
        "candidate_loss_device_copy_count": 0,
        "candidate_loss_host_roundtrips_elided": 0,
        "workspace_allocation_strategy": "float-arena-plus-int64-device",
        "float_workspace_request_count": 0,
        "float_workspace_cuda_mallocs_elided": 0,
        "int64_workspace_cuda_malloc_count": 0,
        "required_next_step": "expand candidate targets beyond block ln1.weight when broader evo mutations are enabled",
    }
    assert train_transformer_payload["vocab"] == 50257
    assert train_transformer_payload["padded_vocab"] == 50304
    assert train_transformer_payload["lm_head_public_vocab_ce_enabled"] is True
    assert train_transformer_payload["lm_head_softmax_vocab"] == 50257
    assert train_transformer_payload["lm_head_logit_row_stride"] == 50304
    assert train_transformer_payload["lm_head_padded_dlogits_zeroed"] is False
    assert train_transformer_payload["lm_head_ce_pad_zero_skipped"] is True
    assert train_transformer_payload["token_weight_padding_zero_enabled"] is True
    assert train_transformer_payload["token_weight_init_elements"] == 50257 * 768
    assert train_transformer_payload["token_weight_padding_elements"] == (50304 - 50257) * 768
    assert train_transformer_payload["lm_head_classifier_strategy_contract"] == {
        "reference_strategy": (
            "llm.kittens-full-resident-logits-fused-ce-dlogits-separate-classifier-matmuls"
        ),
        "native_strategy": (
            "row-chunked-bf16-logits-public-vocab-fused-ce-dlogits-separate-classifier-matmuls-tile-abi"
        ),
        "reference_classifier_fusion_scope": (
            "ce-dlogits-only-logits-dhidden-dweight-remain-separate"
        ),
        "native_classifier_fusion_scope": (
            "ce-dlogits-only-logits-dhidden-dweight-remain-separate"
        ),
        "reference_full_logit_rows": 2,
        "native_logit_chunk_rows": 2,
        "native_logit_chunk_count": 1,
        "padded_vocab_size": 50304,
        "reference_full_bf16_logit_elements": 2 * 50304,
        "reference_full_bf16_logit_bytes": 2 * 50304 * 2,
        "reference_full_float32_logit_bytes": 2 * 50304 * 4,
        "native_chunk_bf16_logit_elements": 2 * 50304,
        "native_chunk_bf16_logit_bytes": 2 * 50304 * 2,
        "native_chunk_float32_logit_bytes": 2 * 50304 * 4,
        "resident_logit_reduction_ratio": 1,
        "dlogits_storage": "in-place-over-bf16-logits",
        "graph_editor_tensor_flow": False,
        "torch_required": False,
        "same_script_benchmark_target": (
            "tools/paired_kernel_speed.py stage.lm_head_backward.total_ms and train_loop_wall_ms"
        ),
        "reference_alignment_target": (
            "match-fused-ce-dlogits-and-optimize-separate-logits-dhidden-dweight-stages"
        ),
        "strict_true_fused_experimental_path": "strict-true-fused-tile-kernel",
        "required_kernel_next_step": (
            "match-reference-fused-ce-dlogits-and-optimize-separate-logits-dhidden-dweight-stages"
        ),
    }
    assert train_transformer_payload["model_dim"] == 768
    assert train_transformer_payload["hidden_dim"] == 3072
    assert train_transformer_payload["lm_head_row_chunk_size"] == 2
    assert train_transformer_payload["lm_head_row_chunk_count"] == 1
    assert train_transformer_payload["loss_partial_count"] == 1
    assert train_transformer_payload["logit_workspace_elements"] == 0
    assert train_transformer_payload["grad_logit_workspace_elements"] == 0
    assert train_transformer_payload["lm_head_training_logits_dtype"] == "bf16"
    assert train_transformer_payload["lm_head_training_dlogits_dtype"] == "bf16"
    assert train_transformer_payload["lm_head_loss_logits_dtype"] == "bf16"
    assert train_transformer_payload["lm_head_bf16_loss_enabled"] is True
    source = (
        Path(__file__).resolve().parents[1]
        / "neuralfn"
        / "csrc"
        / "native_gpt2"
        / "nfn_gpt2_native_train.cpp"
    ).read_text(encoding="utf-8")
    assert "float_logits_small_chunk" not in source
    assert train_transformer_payload["lm_head_bf16_logits_enabled"] is True
    assert train_transformer_payload["lm_head_bf16_logit_elements"] == 0
    assert train_transformer_payload["lm_head_bf16_logit_bytes"] == 0
    assert (
        train_transformer_payload["lm_head_ce_backward_strategy"]
        == "public-vocab-strided-no-pad-zero-fused-row-bf16-logits-dlogits"
    )
    assert "lm_head_ce_loss_backward_fused_available" in train_transformer_payload
    assert (
        train_transformer_payload["lm_head_ce_loss_backward_strategy"]
        in {
            "no-loss-dlogits-public-vocab-no-pad-zero-bf16-u16-targets",
            "no-loss-dlogits-public-vocab-bf16-u16-targets",
            "fused-loss-accumulate-and-dlogits-public-vocab-no-pad-zero-bf16-u16-targets",
            "fused-loss-accumulate-and-dlogits-public-vocab-bf16-u16-targets",
            "separate-loss-partials-reduction-then-dlogits",
        }
    )
    assert "lm_head_classifier_chunk_kernel_available" in train_transformer_payload
    assert train_transformer_payload["lm_head_classifier_chunk_kernel_enabled"] is False
    assert train_transformer_payload["lm_head_classifier_chunk_strategy"] in {
        "row-chunked-public-vocab-bf16-u16-loss-dlogits-tile-abi",
        "unavailable",
    }
    assert train_transformer_payload["lm_head_classifier_chunk_launch_count"] == 0
    assert train_transformer_payload["lm_head_classifier_last_rows"] == 0
    assert train_transformer_payload["lm_head_classifier_last_vocab"] == 0
    assert train_transformer_payload["lm_head_classifier_last_row_stride"] == 0
    assert train_transformer_payload["lm_head_grad_logits_workspace_allocated"] is False
    assert train_transformer_payload["linear_backend_strategy"] == "not-run"
    assert train_transformer_payload["block_forward_linear_strategy"] == "bf16-shadow-weight-gemmex-forward"
    assert train_transformer_payload["block_backward_input_linear_strategy"] == "bf16-shadow-weight-gemmex-dinput"
    assert (
        train_transformer_payload["block_backward_weight_linear_strategy"]
        == "forced-bf16-gemmex-dweight-plus-bias-accumulate-fallback"
    )
    assert train_transformer_payload["dweight_first_microbatch_beta_zero_enabled"] is True
    assert (
        train_transformer_payload["dweight_first_microbatch_beta_strategy"]
        == "first-gradient-accumulation-microbatch-uses-gemm-beta-zero"
    )
    assert (
        train_transformer_payload["lm_head_dweight_beta_zero_scope"]
        == "first-gradient-accumulation-microbatch-first-processed-row-chunk-only"
    )
    assert (
        train_transformer_payload["non_block_forward_backward_linear_strategy"]
        == "padded-lm-head-bf16-gemmex-fallback"
    )
    assert train_transformer_payload["lm_head_logits_linear_strategy"] == "bf16-gemmex-fallback"
    assert train_transformer_payload["lm_head_dhidden_linear_strategy"] == "bf16-gemmex-dinput-dhidden-default"
    assert train_transformer_payload["lm_head_loss_copy_device_synchronize_enabled"] is False
    assert train_transformer_payload["lm_head_loss_copy_ordering"] == "blocking-cudaMemcpy-d2h"
    assert train_transformer_payload["lm_head_ce_row_loss_reduction_requested"] is True
    assert train_transformer_payload["lm_head_ce_row_loss_reduction_available"] is False
    assert train_transformer_payload["lm_head_ce_row_loss_reduction_enabled"] is False
    assert train_transformer_payload["lm_head_ce_row_loss_sum_accumulate_available"] is False
    assert train_transformer_payload["lm_head_ce_row_loss_sum_accumulate_requested"] is False
    assert train_transformer_payload["lm_head_ce_row_loss_sum_accumulate_enabled"] is False
    assert train_transformer_payload["linear_bf16_gemm_count"] == 0
    assert train_transformer_payload["linear_tk_gemm_count"] == 0
    assert train_transformer_payload["linear_cublaslt_gemm_count"] == 0
    assert train_transformer_payload["linear_cublaslt_bgrad_gemm_count"] == 0
    assert train_transformer_payload["linear_cublaslt_bgrad_direct_write_count"] == 0
    assert train_transformer_payload["linear_cublaslt_bgrad_accumulate_count"] == 0
    assert train_transformer_payload["block_backward_dinput_tk_gemm_count"] == 0
    assert train_transformer_payload["block_backward_dinput_cublaslt_gemm_count"] == 0
    assert train_transformer_payload["block_backward_dinput_bf16_gemm_count"] == 0
    assert train_transformer_payload["block_backward_mlp_proj_dinput_before_dweight_count"] == 0
    assert train_transformer_payload["block_backward_mlp_fc_dinput_before_dweight_count"] == 0
    assert train_transformer_payload["block_backward_attn_proj_dinput_before_dweight_count"] == 0
    assert "block_backward_attn_proj_first_step_concurrent_dinput_dweight_count" in train_transformer_payload
    assert train_transformer_payload["block_backward_qkv_dinput_before_dweight_count"] == 0
    assert train_transformer_payload["linear_cublaslt_descriptor_cache_enabled"] is True
    assert train_transformer_payload["linear_sgemm_count"] == 0
    assert train_transformer_payload["bf16_to_f32_vec4_count"] == 0
    assert "float_arena_cuda_malloc_wall_ms" in train_transformer_payload
    assert "float_arena_pointer_assign_wall_ms" in train_transformer_payload
    assert "uint16_arena_cuda_malloc_wall_ms" in train_transformer_payload
    assert "uint16_arena_pointer_assign_wall_ms" in train_transformer_payload
    assert "transformer_device_arena_cuda_malloc_wall_ms" in train_transformer_payload
    assert "transformer_device_arena_pointer_assign_wall_ms" in train_transformer_payload
    assert train_transformer_payload["linear_bf16_a_pack_count"] == 0
    assert train_transformer_payload["linear_bf16_a_cache_hit_count"] == 0
    assert train_transformer_payload["linear_bf16_a_cache_strategy"] == "unused"
    assert train_transformer_payload["linear_bf16_cache_reset_count"] == 0
    assert train_transformer_payload["linear_bf16_workspace_allocation_strategy"] == "unused"
    assert train_transformer_payload["linear_bf16_workspace_allocation_count"] == 0
    assert train_transformer_payload["linear_bf16_workspace_a_capacity"] == 0
    assert train_transformer_payload["linear_bf16_workspace_b_capacity"] == 0
    assert train_transformer_payload["linear_bf16_cached_a_capacity"] == 0
    assert train_transformer_payload["linear_bf16_cache_entry_count"] == 0
    assert train_transformer_payload["timing"]["stage_timing_enabled"] is False
    assert train_transformer_payload["timing"]["stage_timing_max_events"] == 20000
    assert train_transformer_payload["timing"]["stage_timing_event_count"] == 0
    assert train_transformer_payload["timing"]["stage_timing_dropped_event_count"] == 0
    assert train_transformer_payload["timing"]["stage_timing_prealloc_event_pairs_requested"] == 8192
    assert train_transformer_payload["timing"]["stage_timing_event_pair_create_count"] == 0
    assert train_transformer_payload["timing"]["stage_timing_event_pair_preallocated_count"] == 0
    assert train_transformer_payload["timing"]["stage_timing_event_pair_hot_create_count"] == 0
    assert train_transformer_payload["timing"]["stage_timing_event_pair_unused_destroy_count"] == 0
    assert train_transformer_payload["timing"]["stage_timing"] == []
    assert train_transformer_payload["timing"]["post_train_diagnostic_samples_elided"] is False
    assert train_transformer_payload["timing"]["post_train_diagnostic_sample_d2h_count"] == 0
    assert train_transformer_payload["timing"]["post_train_diagnostic_sample_d2h_count_elided"] == 0
    assert train_transformer_payload["attention_forward_strategy"] == "row-vector-tile-score-reuse"
    assert train_transformer_payload["attention_forward_row_count"] == 24
    assert train_transformer_payload["attention_forward_scalar_output_count"] == 1536
    assert train_transformer_payload["attention_forward_score_reuse_value_dim"] == 64
    assert train_transformer_payload["attention_forward_scalar_cta_elision_factor"] == 64
    assert train_transformer_payload["attention_forward_value_chunk_size"] == 64
    assert train_transformer_payload["attention_forward_scalar_launch_fallback_available"] is True
    assert train_transformer_payload["attention_forward_scalar_launch_fallback_enabled"] is False
    assert train_transformer_payload["attention_forward_scalar_launch_allowed"] is False
    assert train_transformer_payload["optimized_attention_required"] is True
    assert train_transformer_payload["optimized_kernel_contract_required"] is True
    assert train_transformer_payload["optimized_kernel_contract_basic_fallback_allowed"] is False
    assert train_transformer_payload["optimized_kernel_contract_passed"] is False
    assert (
        "missing optimized many-tensor/device-scale AdamW Tile-CUDA symbols"
        in train_transformer_payload["optimized_kernel_contract_error"]
    )
    assert train_transformer_payload["attention_forward_row_launch_auto_disable_enabled"] is True
    assert train_transformer_payload["attention_forward_row_launch_auto_disabled"] is False
    assert train_transformer_payload["attention_forward_row_launch_count"] == 0
    assert train_transformer_payload["attention_forward_row_launch_success_count"] == 0
    assert train_transformer_payload["attention_forward_row_launch_fallback_count"] == 0
    assert train_transformer_payload["attention_forward_scalar_launch_count"] == 0
    assert train_transformer_payload["packed_qkv_attention_enabled"] is False
    assert train_transformer_payload["packed_qkv_attention_bf16_elements"] == 0
    assert train_transformer_payload["packed_qkv_attention_bf16_bytes"] == 0
    assert train_transformer_payload["packed_qkv_float_attention_tape_elided"] is False
    assert train_transformer_payload["packed_qkv_float_attention_tape_elements_elided"] == 0
    assert train_transformer_payload["qkv_forward_layout_strategy"] == "fused-split-to-heads"
    assert train_transformer_payload["qkv_forward_layout_kernel_launches_per_block"] == 1
    assert train_transformer_payload["qkv_forward_layout_legacy_launches_per_block"] == 4
    assert train_transformer_payload["qkv_forward_layout_launches_elided_per_block"] == 3
    assert train_transformer_payload["qkv_bias_layout_strategy"] == "fused-qkv-bias-split-to-heads"
    assert train_transformer_payload["qkv_bias_fused_tk_gemm_enabled"] is False
    assert train_transformer_payload["qkv_bias_layout_kernel_launches_per_block"] == 1
    assert train_transformer_payload["qkv_bias_layout_legacy_launches_per_block"] == 2
    assert train_transformer_payload["qkv_bias_layout_launches_elided_per_block"] == 1
    assert train_transformer_payload["qkv_backward_layout_strategy"] == "fused-heads-to-qkv"
    assert train_transformer_payload["qkv_backward_layout_kernel_launches_per_block"] == 1
    assert train_transformer_payload["qkv_backward_layout_legacy_launches_per_block"] == 4
    assert train_transformer_payload["qkv_backward_layout_launches_elided_per_block"] == 3
    assert train_transformer_payload["attention_backward_bf16_qkv_grad_handoff_enabled"] is False
    assert train_transformer_payload["attention_backward_direct_bf16_qkv_grad_scratch_enabled"] is False
    assert train_transformer_payload["attention_backward_direct_bf16_qkv_grad_scratch_elements"] == 0
    assert (
        train_transformer_payload["attention_backward_qkv_bridge_strategy"]
        == "fused-bf16-heads-to-row-qkv"
    )
    assert train_transformer_payload["attention_backward_qkv_bridge_kernel_launches_per_block"] == 1
    assert train_transformer_payload["attention_backward_qkv_bridge_legacy_launches_per_block"] == 4
    assert train_transformer_payload["attention_backward_qkv_bridge_launches_elided_per_block"] == 3
    assert train_transformer_payload["bf16_projection_residual_enabled"] is True
    assert train_transformer_payload["attention_projection_input_strategy"] == (
        "float32-attention-output-bf16-gemm-bf16-residual-consumer"
    )
    assert train_transformer_payload["attention_packed_output_unpack_strategy"] == "not-packed"
    assert train_transformer_payload["mlp_fc_bias_gelu_strategy"] == "fused-bias-preactivation-gelu"
    assert train_transformer_payload["mlp_fc_bias_gelu_kernel_launches_per_block"] == 1
    assert train_transformer_payload["mlp_fc_bias_gelu_legacy_launches_per_block"] == 2
    assert train_transformer_payload["mlp_fc_bias_gelu_launches_elided_per_block"] == 1
    assert train_transformer_payload["mlp_proj_forward_activation_strategy"] == (
        "fused-gelu-bf16-act-direct-bf16-output-gemm"
    )
    assert train_transformer_payload["mlp_forward_act_bf16_elements"] == 0
    assert train_transformer_payload["mlp_forward_act_bf16_bytes"] == 0
    assert train_transformer_payload["projection_bf16_scratch_elements"] == 0
    assert train_transformer_payload["projection_bf16_scratch_bytes"] == 0
    assert train_transformer_payload["float_projection_outputs_elided"] is True
    assert train_transformer_payload["float_attention_projection_output_elided"] is True
    assert train_transformer_payload["float_mlp_projection_output_elided"] is True
    assert (
        train_transformer_payload[
            "saved_packed_attention_recompute_needs_float_attention_projection"
        ]
        is False
    )
    assert train_transformer_payload["float_projection_output_elements_elided"] == 1 * 1 * 2 * 768 * 2
    assert train_transformer_payload["projection_bias_residual_strategy"] == "fused-bf16-linear-bias-residual-add"
    assert train_transformer_payload["projection_bias_residual_kernel_launches_per_block"] == 2
    assert train_transformer_payload["projection_bias_residual_legacy_launches_per_block"] == 4
    assert train_transformer_payload["projection_bias_residual_launches_elided_per_block"] == 2
    assert (
        train_transformer_payload["attention_residual_ln2_strategy"]
        == "fused-bf16-linear-bias-residual-layernorm-bf16-norm-fp32-store-elided"
    )
    assert train_transformer_payload["fused_ln2_bf16_norm_float_store_elision_enabled"] is True
    assert train_transformer_payload["stored_mlp_ln2_bf16_float_store_elided_count"] == 0
    assert train_transformer_payload["stored_mlp_ln2_bf16_float_store_elided_elements"] == 0
    assert train_transformer_payload["attention_residual_ln2_kernel_launches_per_block"] == 1
    assert train_transformer_payload["attention_residual_ln2_legacy_launches_per_block"] == 2
    assert train_transformer_payload["attention_residual_ln2_launches_elided_per_block"] == 1
    assert train_transformer_payload["attention_backward_grad_layout_strategy"] == "merged-grad-out-direct"
    assert train_transformer_payload["attention_backward_grad_layout_kernel_launches_per_block"] == 0
    assert train_transformer_payload["attention_backward_grad_layout_legacy_launches_per_block"] == 1
    assert train_transformer_payload["attention_backward_grad_layout_launches_elided_per_block"] == 1
    assert (
        train_transformer_payload["attention_backward_strategy"]
        == "query-row-atomic-tile-score-reuse"
    )
    assert train_transformer_payload["attention_backward_reuses_forward_workspace"] is False
    assert train_transformer_payload["attention_backward_uses_saved_forward_workspace"] is False
    assert train_transformer_payload["attention_backward_recompute_forward_elided_per_block"] == 0
    assert train_transformer_payload["attention_backward_score_reuse_dim"] == 64
    assert train_transformer_payload["attention_backward_scalar_cta_elision_factor"] == 192
    assert (
        train_transformer_payload["attention_backward_row_count"] * 192
        == train_transformer_payload["attention_backward_scalar_output_count"]
    )
    assert train_transformer_payload["train_batch_tokens"] == 524288
    assert train_transformer_payload["requested_train_batch_tokens"] == 524288
    assert train_transformer_payload["microbatch_tokens"] == 2
    assert train_transformer_payload["grad_accum_steps"] == 262144
    assert train_transformer_payload["effective_train_batch_tokens"] == 524288
    assert train_transformer_payload["gradient_accumulation_strategy"] == "device-microbatch-average"
    assert train_transformer_payload["gradient_accumulation_scale"] == pytest.approx(1.0 / 262144.0)
    assert train_transformer_payload["token_gradient_accumulation_strategy"] == "direct-device-accumulation-buffer"
    assert train_transformer_payload["token_gradient_scratch_buffer_allocated"] is False
    assert train_transformer_payload["token_gradient_microbatch_full_copy_elided"] is True
    assert train_transformer_payload["token_gradient_microbatch_zero_elided"] is True
    assert (
        train_transformer_payload["block_linear_weight_gradient_accumulation_strategy"]
        == "direct-device-accumulation-buffer"
    )
    assert train_transformer_payload["block_linear_weight_gradient_scratch_buffers_allocated"] is False
    assert train_transformer_payload["block_linear_weight_gradient_microbatch_full_copy_elided"] is True
    assert (
        train_transformer_payload["layer_norm_affine_gradient_accumulation_strategy"]
        == "direct-device-accumulation-buffer"
    )
    assert train_transformer_payload["layer_norm_affine_gradient_scratch_buffers_allocated"] is False
    assert train_transformer_payload["layer_norm_affine_gradient_microbatch_full_copy_elided"] is True
    assert (
        train_transformer_payload["linear_bias_gradient_accumulation_strategy"]
        == "direct-device-accumulation-buffer"
    )
    assert train_transformer_payload["linear_bias_gradient_scratch_buffers_allocated"] is False
    assert train_transformer_payload["linear_bias_gradient_microbatch_full_copy_elided"] is True
    assert train_transformer_payload["position_gradient_accumulation_strategy"] == "direct-device-accumulation-buffer"
    assert train_transformer_payload["position_gradient_scratch_buffer_allocated"] is False
    assert train_transformer_payload["position_gradient_microbatch_full_copy_elided"] is True
    assert train_transformer_payload["position_gradient_microbatch_zero_elided"] is True
    assert train_transformer_payload["block_state_layout"]["layer_norm_stats_strategy"] in {
        "forward-store-mean-rstd-backward-reuse",
        "backward-recompute",
    }
    assert isinstance(train_transformer_payload["block_state_layout"]["layer_norm_backward_reuses_forward_stats"], bool)
    assert train_transformer_payload["attention_activation_storage_strategy"] == "disabled"
    assert train_transformer_payload["stored_attention_activation_blocks"] == 0
    assert train_transformer_payload["stored_attention_bf16_elements"] == 0
    assert train_transformer_payload["stored_attention_bf16_bytes"] == 0
    assert train_transformer_payload["stored_attention_lse_elements"] == 0
    assert train_transformer_payload["stored_attention_lse_bytes"] == 0
    assert train_transformer_payload["stored_attention_store_kernel_launches"] == 0
    assert train_transformer_payload["stored_attention_restore_kernel_launches"] == 0
    assert train_transformer_payload["stored_attention_backward_kernel_launches"] == 0
    assert train_transformer_payload["stored_attention_backward_consumer_strategy"] == "disabled"
    assert train_transformer_payload["packed_attention_activation_storage_strategy"] == "disabled"
    assert train_transformer_payload["stored_packed_attention_activation_blocks"] == 0
    assert train_transformer_payload["stored_packed_attention_block_placement"] == "head"
    assert train_transformer_payload["stored_packed_attention_block_start"] == 0
    assert train_transformer_payload["stored_packed_attention_bf16_elements"] == 0
    assert train_transformer_payload["stored_packed_attention_bf16_bytes"] == 0
    assert train_transformer_payload["stored_packed_attention_lse_elements"] == 0
    assert train_transformer_payload["stored_packed_attention_lse_bytes"] == 0
    assert train_transformer_payload["stored_packed_attention_lse_enabled"] is False
    assert train_transformer_payload["stored_packed_attention_ln1_bf16_enabled"] is False
    assert train_transformer_payload["stored_packed_attention_ln1_bf16_blocks"] == 0
    assert train_transformer_payload["stored_packed_attention_ln1_bf16_elements"] == 0
    assert train_transformer_payload["stored_packed_attention_ln1_bf16_bytes"] == 0
    assert train_transformer_payload["stored_packed_attention_store_blocks"] == 0
    assert train_transformer_payload["stored_packed_attention_restore_blocks"] == 0
    assert train_transformer_payload["stored_packed_attention_backward_kernel_launches"] == 0
    assert train_transformer_payload["stored_packed_attention_backward_consumer_strategy"] == "disabled"
    assert train_transformer_payload["max_steps"] == 2
    assert train_transformer_payload["eval_every_steps"] == 1
    assert train_transformer_payload["eval_batches"] == 1
    assert train_transformer_payload["validation"]["eval_batch_size"] == 1
    assert train_transformer_payload["train_loss_eval_count"] == 0
    assert train_transformer_payload["train_loss_last_step"] == 0
    assert train_transformer_payload["train_loss_sparse"] is False
    assert train_transformer_payload["train_loss_sampling"] == "disabled"
    assert train_transformer_payload["train_loss_on_validation_steps"] is False
    assert train_transformer_payload["token_id_direct_u16_enabled"] is True
    assert train_transformer_payload["token_id_upload_strategy"] == (
        "uint16-pinned-async-h2d-direct-kernel-consumption"
    )
    assert train_transformer_payload["token_id_host_staging"] == "pinned"
    assert train_transformer_payload["token_id_h2d_copy"] == "cudaMemcpyAsync-contiguous-arena"
    assert train_transformer_payload["token_id_h2d_copy_calls_per_microbatch"] == 1
    assert train_transformer_payload["token_id_h2d_copy_calls_elided_per_microbatch"] == 1
    assert train_transformer_payload["token_id_widen_strategy"] == "elided-direct-u16-kernels"
    assert train_transformer_payload["token_id_widen_kernel_launches_per_microbatch"] == 0
    assert train_transformer_payload["token_id_widen_kernel_launches_elided_per_microbatch"] == 2
    assert train_transformer_payload["token_batch_staging_strategy"] == "direct-sampler-to-pinned-arena"
    assert train_transformer_payload["token_batch_vector_materialization"] is False
    assert train_transformer_payload["token_batch_vector_copy_to_pinned_elided"] is True
    assert train_transformer_payload["token_id_host_validation"] is False
    assert train_transformer_payload["token_buffer_allocation_strategy"] == "combined-arenas"
    assert train_transformer_payload["token_device_allocation_strategy"] == "single-device-arena"
    assert train_transformer_payload["token_device_arena_cuda_malloc_count"] == 0
    assert train_transformer_payload["token_device_arena_requested_bytes"] == 0
    assert train_transformer_payload["token_device_arena_bytes"] == 0
    assert train_transformer_payload["token_device_arena_suballocation_count"] == 0
    assert train_transformer_payload["token_device_cuda_mallocs_elided"] == 1
    assert train_transformer_payload["token_i64_arena_cuda_malloc_count"] == 0
    assert train_transformer_payload["token_u16_device_arena_cuda_malloc_count"] == 0
    assert train_transformer_payload["token_u16_pinned_arena_cuda_host_alloc_count"] == 0
    assert train_transformer_payload["token_i64_arena_elements"] == 0
    assert train_transformer_payload["token_u16_device_arena_elements"] == 0
    assert train_transformer_payload["token_u16_pinned_arena_elements"] == 0
    assert (
        train_transformer_payload["token_weight_init_strategy"]
        == "device-vector4-strided-power2-deterministic"
    )
    assert train_transformer_payload["token_weight_threaded_init_enabled"] is False
    assert train_transformer_payload["token_weight_vector4_init_enabled"] is True
    assert train_transformer_payload["token_weight_vector4_strided_init_requested"] is True
    assert train_transformer_payload["token_weight_bf16_pattern_init_requested"] is False
    assert train_transformer_payload["token_weight_fast_int32_init_enabled"] is False
    assert train_transformer_payload["token_weight_init_legacy_mod17_enabled"] is False
    assert train_transformer_payload["token_weight_bf16_initial_refresh_fusion_enabled"] is True
    assert train_transformer_payload["token_weight_bf16_initial_refresh_elided"] is False
    assert train_transformer_payload["token_weight_host_materialization"] is False
    assert train_transformer_payload["float_allocation_strategy"] == "single-arena"
    assert train_transformer_payload["float_allocation_cuda_malloc_count"] == 0
    assert train_transformer_payload["float_allocation_request_count"] == 0
    assert train_transformer_payload["float_arena_requested_elements"] == 0
    assert train_transformer_payload["float_arena_allocated_elements"] == 0
    assert train_transformer_payload["uint16_allocation_strategy"] == "single-arena"
    assert train_transformer_payload["uint16_allocation_cuda_malloc_count"] == 0
    assert train_transformer_payload["uint16_allocation_request_count"] == 0
    assert train_transformer_payload["uint16_arena_requested_elements"] == 0
    assert train_transformer_payload["uint16_arena_allocated_elements"] == 0
    assert train_transformer_payload["uint16_arena_cuda_malloc_count"] == 0
    assert train_transformer_payload["uint16_arena_suballocation_count"] == 0
    assert train_transformer_payload["float_arena_zero_init_strategy"] == "adamw-state-contiguous-range-cuda-memset"
    assert train_transformer_payload["startup_cuda_memset_zero_enabled"] is True
    assert isinstance(train_transformer_payload["startup_cuda_memset_zero_available"], bool)
    assert train_transformer_payload["float_arena_zero_fill_count"] == 0
    assert train_transformer_payload["adamw_state_zero_fill_count"] == 0
    assert train_transformer_payload["startup_cuda_memset_zero_fill_count"] == 0
    assert train_transformer_payload["startup_tile_zero_fill_count"] == 0
    assert train_transformer_payload["adamw_state_zero_range_count"] == 0
    assert train_transformer_payload["adamw_state_zero_range_elements"] == 0
    assert train_transformer_payload["startup_per_buffer_zero_fill_elided"] is True
    assert train_transformer_payload["startup_per_buffer_zero_fill_launches_elided"] == 369
    assert train_transformer_payload["descriptor_allocation_strategy"] == "single-device-arena"
    assert train_transformer_payload["descriptor_arena_cuda_malloc_count"] == 0
    assert train_transformer_payload["descriptor_arena_requested_bytes"] == 0
    assert train_transformer_payload["descriptor_arena_bytes"] == 0
    assert train_transformer_payload["descriptor_arena_suballocation_count"] == 0
    assert train_transformer_payload["descriptor_upload_strategy"] == "single-host-packed-arena-copy"
    assert train_transformer_payload["descriptor_arena_copy_count"] == 0
    assert train_transformer_payload["descriptor_arena_copy_calls_elided"] == 37
    assert train_transformer_payload["descriptor_cuda_mallocs_elided"] == 37
    assert train_transformer_payload["parameter_initialization_strategy"] in {
        "fused-multi-buffer-fill-values",
        "mixed-float32-bf16-fill-many-values",
    }
    assert train_transformer_payload["parameter_initialization_descriptor_count"] == 0
    assert train_transformer_payload["bf16_parameter_initialization_descriptor_count"] == 0
    assert train_transformer_payload["parameter_initialization_max_elements"] == 0
    assert train_transformer_payload["bf16_parameter_initialization_max_elements"] == 0
    assert train_transformer_payload["parameter_initialization_kernel_launches"] == 0
    assert train_transformer_payload["bf16_parameter_initialization_kernel_launches"] == 0
    assert train_transformer_payload.get("mixed_parameter_initialization_kernel_launches", 0) == 0
    assert train_transformer_payload["parameter_initialization_kernel_launches_per_startup"] == 0
    assert train_transformer_payload["parameter_initialization_per_buffer_launches_elided"] == 75
    assert train_transformer_payload["direct_bf16_block_weight_initialization_enabled"] is True
    assert train_transformer_payload["token_weight_bf16_shadow_enabled"] is True
    assert train_transformer_payload["token_weight_bf16_refresh_count"] == 0
    assert train_transformer_payload["adamw_update_strategy"] in {
        "fused-multi-buffer-device-scale",
        "split-float32-and-bf16-param-multi-buffer-device-scale",
        "split-float32-token-shadow-and-bf16-param-multi-buffer-device-scale",
    }
    assert train_transformer_payload["optimizer_tile_size"] == 1024
    assert isinstance(train_transformer_payload["optimizer_tile_size_symbol_loaded"], bool)
    assert train_transformer_payload["optimizer_tile_strategy"] == "tile-size-1024-sumsq-scale-adamw"
    assert train_transformer_payload["attention_backward_tk_block_size"] in {0, 16, 32, 64}
    assert isinstance(train_transformer_payload["attention_backward_tk_block_size_symbol_loaded"], bool)
    assert isinstance(train_transformer_payload["linear_tk_sm120_config_symbol_loaded"], bool)
    assert isinstance(train_transformer_payload["linear_tk_sm120_fast_dgelu_enabled"], bool)
    assert isinstance(train_transformer_payload["linear_tk_sm120_approx_dgelu_tanh_enabled"], bool)
    assert train_transformer_payload["linear_tk_sm120_k_tile"] in {0, 16, 32, 64}
    assert train_transformer_payload["linear_tk_sm120_grad_k_tile"] in {0, 32, 64}
    assert train_transformer_payload["linear_tk_sm120_super_m"] in {0, 4, 7, 8, 13}
    assert train_transformer_payload["linear_tk_sm120_dinput_super_m"] in {0, 4, 7, 8, 13}
    assert train_transformer_payload["linear_tk_sm120_dweight_super_m"] in {0, 1, 2}
    assert train_transformer_payload["linear_tk_sm120_huge_n_k_tile"] in {0, 32, 64, 128}
    assert train_transformer_payload["adamw_descriptor_count"] == 0
    assert train_transformer_payload["adamw_float_update_descriptor_count"] == 0
    assert train_transformer_payload["adamw_bf16_param_descriptor_count"] == 0
    assert train_transformer_payload["adamw_bf16_param_bf16_grad_descriptor_count"] == 0
    assert train_transformer_payload["adamw_kernel_launches"] == 0
    assert train_transformer_payload["adamw_float_update_kernel_launches"] == 0
    assert train_transformer_payload["adamw_bf16_param_kernel_launches"] == 0
    assert train_transformer_payload["adamw_bf16_param_bf16_grad_kernel_launches"] == 0
    assert train_transformer_payload["adamw_step_kernel_launches_per_optimizer_step"] == 0
    assert train_transformer_payload["adamw_per_buffer_step_launches_elided"] == 147
    assert train_transformer_payload["gradient_zero_strategy"] == "fused-multi-buffer-accumulation-zero"
    assert train_transformer_payload["gradient_cuda_memset_zero_enabled"] is True
    assert isinstance(train_transformer_payload["gradient_cuda_memset_zero_available"], bool)
    assert train_transformer_payload["gradient_zero_range_count"] == 0
    assert train_transformer_payload["gradient_zero_range_elements"] == 0
    assert train_transformer_payload["gradient_zero_cuda_memset_count"] == 0
    assert train_transformer_payload["gradient_zero_tile_fill_count"] == 0
    assert train_transformer_payload["accumulation_zero_kernel_launches"] == 0
    assert train_transformer_payload["gradient_zero_kernel_launches_per_optimizer_step"] == 0
    assert train_transformer_payload["gradient_zero_per_buffer_launches_elided"] == 147
    assert train_transformer_payload["gradient_clip_strategy"] == "fused-multi-buffer-sumsq-device-scale"
    assert train_transformer_payload["gradient_sumsq_kernel_launches"] == 0
    assert train_transformer_payload["gradient_sumsq_kernel_launches_per_optimizer_step"] == 0
    assert train_transformer_payload["gradient_sumsq_per_buffer_launches_elided"] == 147
    assert train_transformer_payload["steps_completed"] == 0
    assert train_transformer_payload["train_microbatches_completed"] == 0
    assert train_transformer_payload["weight_update_count"] == 148
    assert train_transformer_payload["block_dweight_bf16_staging_enabled"] is False
    assert train_transformer_payload["block_dweight_bf16_staging_elements"] == 0
    assert train_transformer_payload["block_dweight_bf16_staging_bytes"] == 0
    assert train_transformer_payload["block_dweight_bf16_staging_zero_count"] == 0
    assert train_transformer_payload["block_dweight_bf16_staging_convert_kernel_launches"] == 0
    assert train_transformer_payload["block_dweight_bf16_staging_strategy"] == (
        "disabled-fp32-accumulation-default"
    )
    assert train_transformer_payload["block_backward_mlp_fc_grad_out_float_buffer_elided"] is True
    assert train_transformer_payload["block_backward_mlp_fc_grad_out_float_elements"] == 0
    assert train_transformer_payload["block_backward_mlp_fc_grad_out_float_bytes_elided"] == 2 * 1 * 3072 * 4
    assert train_transformer_payload["bf16_persistent_block_outputs_enabled"] is False
    assert train_transformer_payload["bf16_persistent_block_output_count"] == 0
    assert train_transformer_payload["bf16_persistent_block_output_placement"] == "head"
    assert train_transformer_payload["bf16_persistent_block_output_start"] == 0
    assert train_transformer_payload["bf16_persistent_block_output_store_count"] == 0
    assert train_transformer_payload["bf16_persistent_block_output_restore_count"] == 0
    assert train_transformer_payload["fp32_persistent_block_output_elements_elided"] == 0
    assert train_transformer_payload["fp32_persistent_block_output_bytes_elided"] == 0
    assert train_transformer_payload["block_state_layout"] == {
            "allocated_block_count": 12,
                "target_block_count": 12,
                "activation_tape_count": 1,
                "full_activation_tape_enabled": False,
                "packed_qkv_float_attention_tape_elided": False,
                "packed_qkv_float_attention_tape_elements_elided": 0,
        "persistent_block_outputs": 11,
        "persistent_block_output_write_strategy": "direct-residual2-output",
        "persistent_block_output_copy_elided_count": 0,
        "bf16_persistent_block_outputs_enabled": False,
        "bf16_persistent_block_output_count": 0,
        "bf16_persistent_block_output_placement": "head",
        "bf16_persistent_block_output_start": 0,
        "bf16_persistent_block_output_store_count": 0,
        "bf16_persistent_block_output_restore_count": 0,
        "bf16_persistent_block_input_ln1_backward_requested": False,
        "bf16_persistent_block_input_ln1_backward_enabled": False,
        "bf16_persistent_block_input_ln1_backward_count": 0,
        "fp32_persistent_block_output_elements_elided": 0,
        "final_block_output_copy_elided": True,
        "validation_persistent_block_outputs": 0,
        "validation_block_output_copies_elided": True,
        "backward_recompute_blocks": 11,
        "stored_mlp_activation_block_placement": "head",
        "stored_mlp_activation_block_start": 0,
        "stored_packed_attention_block_placement": "head",
        "stored_packed_attention_block_start": 0,
        "final_block_backward_recompute_elided": True,
        "backward_recompute_mlp_fc_gelu_elided": True,
                "backward_recompute_attention_qkv_sdpa_elided": False,
                "backward_recompute_attention_uses_saved_o": False,
        "backward_recompute_mlp_projection_elided": True,
        "backward_recompute_final_residual_elided": True,
        "mlp_proj_backward_gelu_inplace": True,
        "mlp_proj_backward_grad_act_scratch_allocated": False,
                "mlp_residual_next_ln1_fusion_enabled": False,
                "mlp_residual_next_ln1_fusion_count": 0,
                "mlp_residual_next_ln1_strategy": "separate-mlp-residual-and-next-ln1",
            "float_projection_output_buffers_allocated": 0,
            "float_projection_output_buffers_elided": 2,
            "float_attention_projection_output_elided": True,
            "float_mlp_projection_output_elided": True,
            "saved_packed_attention_recompute_needs_float_attention_projection": False,
            "float_projection_output_elements_elided": 1 * 1 * 2 * 768 * 2,
        "mlp_fc_grad_out_float_buffer_elided": True,
        "mlp_fc_grad_out_float_elements": 0,
        "mlp_fc_grad_out_float_bytes_elided": 2 * 1 * 3072 * 4,
                    "activation_tape_strategy": "scratch-recompute-bf16-stored-mlp-direct-backward-opt-in",
        "forward_row_qkv_scratch_allocated": False,
        "forward_row_qkv_scratch_buffers_elided": 3,
        "per_block_parameter_buffers": 12,
        "per_block_gradient_buffers": 0,
        "per_block_direct_accum_gradient_buffers": 12,
        "per_block_accum_gradient_buffers": 12,
        "per_block_adamw_state_buffers": 24,
        "per_block_gradient_partials": train_transformer_payload["block_state_layout"]["per_block_gradient_partials"],
        "global_gradient_partials": train_transformer_payload["block_state_layout"]["global_gradient_partials"],
        "global_parameter_buffers": 4,
        "parameter_allocation_loop": True,
        "parameter_initialization_loop": False,
        "parameter_initialization_loop_elided": True,
        "parameter_initialization_strategy": train_transformer_payload[
            "parameter_initialization_strategy"
        ],
        "parameter_initialization_descriptor_count": 0,
        "bf16_parameter_initialization_descriptor_count": 0,
        "mixed_parameter_initialization_kernel_launches": 0,
        "parameter_initialization_kernel_launches_per_startup": 0,
        "parameter_initialization_per_buffer_launches_elided": 75,
        "startup_zero_init_strategy": "adamw-state-contiguous-range-cuda-memset",
        "startup_cuda_memset_zero_enabled": True,
        "startup_cuda_memset_zero_available": train_transformer_payload["startup_cuda_memset_zero_available"],
        "startup_arena_zero_fill_count": 0,
        "startup_adamw_state_zero_fill_count": 0,
        "startup_cuda_memset_zero_fill_count": 0,
        "startup_tile_zero_fill_count": 0,
        "startup_adamw_state_zero_range_count": 0,
        "startup_adamw_state_zero_range_elements": 0,
        "startup_per_buffer_zero_fill_elided": True,
        "startup_per_buffer_zero_fill_launches_elided": 369,
        "descriptor_allocation_strategy": "single-device-arena",
        "descriptor_arena_cuda_malloc_count": 0,
        "descriptor_arena_suballocation_count": 0,
        "descriptor_upload_strategy": "single-host-packed-arena-copy",
        "descriptor_arena_copy_count": 0,
        "descriptor_arena_copy_calls_elided": 37,
        "descriptor_cuda_mallocs_elided": 37,
        "block0_duplicate_allocation_elided": True,
        "block0_duplicate_activation_allocation_elided": True,
        "block0_duplicate_parameter_initialization_elided": True,
        "block0_duplicate_adamw_state_zero_elided": True,
        "gradient_zero_loop": False,
        "gradient_zero_loop_elided": True,
        "gradient_zero_strategy": "fused-multi-buffer-accumulation-zero",
        "gradient_cuda_memset_zero_enabled": True,
        "gradient_cuda_memset_zero_available": train_transformer_payload["gradient_cuda_memset_zero_available"],
        "gradient_zero_range_count": 0,
        "gradient_zero_range_elements": 0,
        "gradient_zero_cuda_memset_count": 0,
        "gradient_zero_tile_fill_count": 0,
        "gradient_zeroed_buffer_count": 0,
        "gradient_zero_descriptor_count": 0,
        "gradient_zero_kernel_launches_per_optimizer_step": 0,
        "gradient_zero_per_buffer_launches_elided": 147,
        "gradient_accumulation_loop": False,
        "gradient_accumulation_buffers": True,
        "gradient_accumulation_copy_loop_elided": True,
        "gradient_accumulation_zero_strategy": "all-accumulation-buffers",
        "token_gradient_accumulation_direct": True,
        "token_gradient_scratch_buffer_allocated": False,
        "block_linear_weight_gradient_accumulation_direct": True,
        "block_dweight_bf16_staging_enabled": False,
        "block_dweight_bf16_staging_elements": 0,
        "block_dweight_bf16_staging_zero_count": 0,
        "block_dweight_bf16_staging_convert_kernel_launches": 0,
        "block_linear_weight_gradient_scratch_buffers_allocated": False,
        "block_linear_weight_gradient_microbatch_full_copy_elided": True,
        "layer_norm_affine_gradient_accumulation_direct": True,
        "layer_norm_affine_gradient_scratch_buffers_allocated": False,
        "layer_norm_affine_gradient_microbatch_full_copy_elided": True,
        "linear_bias_gradient_accumulation_direct": True,
        "linear_bias_gradient_scratch_buffers_allocated": False,
        "linear_bias_gradient_microbatch_full_copy_elided": True,
        "linear_backward_bias_row_chunk_size": 256,
        "linear_backward_bias_threads_per_block": 512,
        "position_gradient_accumulation_direct": True,
        "position_gradient_scratch_buffer_allocated": False,
        "position_gradient_microbatch_full_copy_elided": True,
        "layer_norm_backward_affine_strategy": "auto-chunked-atomic-accumulate",
        "layer_norm_backward_affine_row_chunk_size": 128,
        "layer_norm_stats_strategy": "forward-store-mean-rstd-backward-reuse",
        "layer_norm_backward_reuses_forward_stats": True,
        "layer_norm_stats_disabled_by_fused_residual_ln2": False,
        "layer_norm_backward_residual_fusion_enabled": True,
        "layer_norm_backward_affine_residual_fusion_enabled": True,
        "layer_norm_backward_affine_residual_fused_kernel_launches": 0,
        "layer_norm_backward_residual_strategy": "fused-affine-dinput-residual-add-with-forward-stats",
        "layer_norm_backward_residual_scratch_buffers_allocated": False,
        "layer_norm_backward_residual_scratch_buffers_elided": 2,
        "layer_norm_backward_residual_scratch_elements_elided": 2 * 2 * 768,
        "residual1_backward_consumer_strategy": "bf16-layernorm-backward",
        "gradient_clip_loop": False,
        "gradient_clip_loop_elided": True,
        "gradient_clip_strategy": "fused-multi-buffer-sumsq-device-scale",
        "optimizer_tile_size": 1024,
        "optimizer_tile_strategy": "tile-size-1024-sumsq-scale-adamw",
        "optimized_optimizer_contract_loaded": False,
        "optimized_optimizer_contract_error": "",
        "attention_backward_tk_block_size": train_transformer_payload["attention_backward_tk_block_size"],
        "attention_backward_tk_block_size_symbol_loaded": train_transformer_payload[
            "attention_backward_tk_block_size_symbol_loaded"
        ],
        "gradient_clip_descriptor_count": 0,
        "gradient_clip_bf16_sumsq_kernel_loaded": False,
        "gradient_sumsq_kernel_launches_per_optimizer_step": 0,
        "gradient_sumsq_per_buffer_launches_elided": 147,
        "adamw_device_clip_scale_fused": True,
        "adamw_bf16_shadow_refresh_strategy": "elided-bf16-primary-params",
            "block_weight_bf16_initialization_strategy": "direct-bf16-fill-many-values",
            "token_weight_bf16_shadow_enabled": True,
            "token_weight_bf16_refresh_count": 0,
            "token_weight_bf16_initial_refresh_fusion_enabled": True,
            "token_weight_bf16_adamw_refresh_fusion_enabled": True,
            "token_weight_padded_init_fusion_enabled": False,
            "token_weight_padding_zero_launches_elided": 0,
            "token_weight_bf16_padding_memset_count": 0,
            "token_weight_bf16_initial_refresh_elided": False,
            "block_weight_bf16_primary_param_update_enabled": True,
        "direct_bf16_block_weight_initialization_enabled": True,
        "block_weight_bf16_gradient_storage_strategy": "float32-accumulation-buffer",
        "adamw_bf16_param_bf16_grad_kernel_loaded": False,
        "adamw_update_loop": False,
        "adamw_update_loop_elided": True,
        "adamw_update_strategy": "split-float32-and-bf16-param-multi-buffer-device-scale",
        "adamw_descriptor_count": 0,
        "adamw_float_update_descriptor_count": 0,
        "adamw_bf16_param_descriptor_count": 0,
        "adamw_bf16_param_bf16_grad_descriptor_count": 0,
        "adamw_step_kernel_launches_per_optimizer_step": 0,
        "adamw_per_buffer_step_launches_elided": 147,
        "checkpoint_export_loop": True,
        "activation_tape_loop": True,
        "forward_block_loop": True,
        "backward_block_loop": True,
        "residual_backward_fused": True,
    }
    assert train_transformer_payload["block_state_layout"]["per_block_gradient_partials"] > 0
    assert train_transformer_payload["block_state_layout"]["global_gradient_partials"] > 0
    assert train_transformer_payload["gradient_partial_count"] > 0
    assert train_transformer_payload["gradient_clip_norm"] == 1.0
    assert "sample_gradient_clip_scale" in train_transformer_payload
    assert train_transformer_payload["checkpoint"] == {
        "enabled": True,
        "requested": True,
        "startup_only_elided": False,
        "checkpoint_written": False,
        "checkpoint_path": "",
        "done_marker": "",
        "checkpoint_step": 0,
        "version": 5,
        "precision": "bf16",
        "num_layers": 12,
        "num_heads": 12,
        "channels": 768,
        "padded_vocab": 50304,
        "payload_pack_strategy": "device-many-float32-to-bf16-bits-contiguous",
        "payload_pack_kernel": "nfn_native_tile_float32_to_bf16_bits_many",
        "payload_copy_strategy": "single-contiguous-device-payload-d2h",
        "payload_cpu_bf16_conversion": False,
        "tensor_count": 0,
        "payload_elements": 0,
        "bf16_param_sync_kernel_launches": 0,
        "device_pack_kernel_launches": 0,
        "d2h_copy_count": 0,
        "d2h_bytes": 0,
        "float32_d2h_bytes_elided": 0,
        "expected_file_size": 0,
        "actual_file_size": 0,
    }
    assert train_transformer_payload["validation"] == {
        "eval_every_steps": 1,
        "eval_batches": 1,
        "requested_eval_batch_size": 1,
        "eval_batch_size": 1,
        "runtime_enabled": True,
        "sampler_constructed": True,
        "eval_count": 0,
        "losses": [],
    }
    assert "nfn_native_tile_sumsq_partials_float32" in train_transformer_payload["kernels"]
    assert "nfn_native_tile_sumsq_partials_many_float32" in train_transformer_payload["kernels"]
    assert "nfn_native_tile_sumsq_partials_many_bf16_bits_float32" in train_transformer_payload["kernels"]
    assert "nfn_native_tile_optimizer_tile_size" in train_transformer_payload["kernels"]
    assert "nfn_native_tile_attention_backward_tk_block_size" in train_transformer_payload["kernels"]
    assert "nfn_native_tile_global_norm_clip_scale_float32" in train_transformer_payload["kernels"]
    assert "nfn_native_tile_scale_inplace_by_device_float32" in train_transformer_payload["kernels"]
    assert "nfn_native_tile_scaled_dot_product_attention_backward_float32" in train_transformer_payload["kernels"]
    assert (
        "nfn_native_tile_scaled_dot_product_attention_backward_from_merged_grad_float32"
        in train_transformer_payload["kernels"]
    )
    assert "nfn_native_tile_bf16_bits_add_bias_inplace_float32" in train_transformer_payload["kernels"]
    assert (
        "nfn_native_tile_scaled_dot_product_attention_packed_qkv_bf16_float32"
        in train_transformer_payload["kernels"]
    )
    assert (
        "nfn_native_tile_scaled_dot_product_attention_packed_qkv_store_lse_bf16_float32"
        in train_transformer_payload["kernels"]
    )
    assert (
        "nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_from_merged_grad_float32"
        in train_transformer_payload["kernels"]
    )
    assert (
        "nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_from_saved_lse_bf16_from_merged_grad_float32"
        in train_transformer_payload["kernels"]
    )
    assert (
        "nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_bf16_bits_from_merged_grad_float32"
        in train_transformer_payload["kernels"]
    )
    assert (
        "nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_bf16_bits_from_saved_lse_bf16_from_merged_grad_float32"
        in train_transformer_payload["kernels"]
    )
    assert (
        "nfn_native_tile_token_cross_entropy_backward_inplace_with_workspace_float32"
        in train_transformer_payload["kernels"]
    )
    assert "nfn_native_tile_token_cross_entropy_partials_strided_float32" in train_transformer_payload["kernels"]
    assert "nfn_native_tile_token_cross_entropy_partials_strided_bf16_bits" in train_transformer_payload["kernels"]
    assert (
        "nfn_native_tile_token_cross_entropy_partials_strided_bf16_bits_u16_targets"
        in train_transformer_payload["kernels"]
    )
    assert (
        "nfn_native_tile_token_cross_entropy_backward_inplace_strided_with_workspace_float32"
        in train_transformer_payload["kernels"]
    )
    assert (
        "nfn_native_tile_token_cross_entropy_backward_inplace_strided_bf16_bits_with_workspace"
        in train_transformer_payload["kernels"]
    )
    assert (
        "nfn_native_tile_token_cross_entropy_backward_inplace_strided_bf16_bits_u16_targets_with_workspace"
        in train_transformer_payload["kernels"]
    )
    assert "nfn_native_tile_linear_backward_weight_accumulate_float32" in train_transformer_payload["kernels"]
    assert "nfn_native_tile_merge_heads_to_qkv_float32" in train_transformer_payload["kernels"]
    assert "nfn_native_tile_copy_float32" in train_transformer_payload["kernels"]
    assert "nfn_native_tile_fill_many_float32" in train_transformer_payload["kernels"]
    assert "nfn_native_tile_fill_many_values_float32" in train_transformer_payload["kernels"]
    assert "nfn_native_tile_fill_many_values_bf16_bits_float32" in train_transformer_payload["kernels"]
    assert "nfn_native_tile_init_gpt2_token_weight_float32" in train_transformer_payload["kernels"]
    assert "nfn_native_tile_init_gpt2_token_weight_fast_float32" in train_transformer_payload["kernels"]
    assert (
        "nfn_native_tile_linear_backward_weight_accumulate_bf16_bits_bf16_bits_float32_beta"
        in train_transformer_payload["kernels"]
    )
    assert (
        "nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_bits_bf16_bits_float32_beta"
        in train_transformer_payload["kernels"]
    )
    assert (
        "nfn_native_tile_linear_backward_input_dgelu_bf16_bits_weight_bf16_bits_only_float32"
        in train_transformer_payload["kernels"]
    )
    assert "nfn_native_tile_evo_mutate_candidates_float32" in train_transformer_payload["kernels"]
    assert "nfn_native_tile_evo_select_best_loss_float32" in train_transformer_payload["kernels"]
    assert "nfn_native_tile_evo_adopt_candidate_float32" in train_transformer_payload["kernels"]
    assert "nfn_native_tile_uint16_to_int64" in train_transformer_payload["kernels"]
    assert "nfn_native_tile_token_embedding_u16_float32" in train_transformer_payload["kernels"]
    assert "nfn_native_tile_token_embedding_backward_weight_u16_float32" in train_transformer_payload["kernels"]
    assert "nfn_native_tile_float32_to_bf16_bits" in train_transformer_payload["kernels"]
    assert "nfn_native_tile_bf16_bits_to_float32" in train_transformer_payload["kernels"]
    assert "nfn_native_tile_store_mlp_activations_bf16_float32" in train_transformer_payload["kernels"]
    assert "nfn_native_tile_restore_mlp_activations_bf16_float32" in train_transformer_payload["kernels"]
    assert "nfn_native_tile_linear_backward_weight_accumulate_bf16_bits_float32" in train_transformer_payload["kernels"]
    assert "nfn_native_tile_linear_bf16_input_weight_bf16_gelu_bf16_float32" in train_transformer_payload["kernels"]
    assert "nfn_native_tile_gelu_backward_inplace_bf16_bits_float32" in train_transformer_payload["kernels"]
    assert "nfn_native_tile_float32_to_bf16_bits_many" in train_transformer_payload["kernels"]
    assert "nfn_native_tile_adamw_step_with_device_scale_float32" in train_transformer_payload["kernels"]
    assert "nfn_native_tile_adamw_step_many_with_device_scale_float32" in train_transformer_payload["kernels"]
    assert "nfn_native_tile_adamw_step_many_with_device_scale_bf16_shadow_float32" in train_transformer_payload["kernels"]
    assert "nfn_native_tile_adamw_step_many_with_device_scale_bf16_param_float32" in train_transformer_payload["kernels"]
    assert (
        "nfn_native_tile_adamw_step_many_with_device_scale_bf16_param_bf16_grad_float32"
        in train_transformer_payload["kernels"]
    )
    assert train_transformer_payload["passed"] is False

    smaller_eval_transformer_lm = subprocess.run(
        [
            str(cli),
            "--dataset-alias",
            str(dataset_path),
            "--train-transformer-lm",
            "--tile-ops-lib",
            str(tmp_path / "missing-tile-ops.so"),
            "--batch-size",
            "2",
            "--train-seq-len",
            "2",
            "--max-steps",
            "1",
            "--eval-every-steps",
            "1",
            "--eval-batches",
            "1",
            "--eval-batch-size",
            "1",
            "--train-batch-tokens",
            "4",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert smaller_eval_transformer_lm.returncode == 2
    smaller_eval_payload = json.loads(smaller_eval_transformer_lm.stdout)
    assert smaller_eval_payload["batch_size"] == 2
    assert smaller_eval_payload["validation"]["requested_eval_batch_size"] == 1
    assert smaller_eval_payload["validation"]["eval_batch_size"] == 1
    assert smaller_eval_payload["validation"]["eval_batches"] == 1

    checkpoint_out = tmp_path / "checkpoint-metadata-out"
    checkpoint_metadata = subprocess.run(
        [
            str(cli),
            "--dataset-alias",
            str(dataset_path),
            "--checkpoint-metadata-smoke",
            "--output-dir",
            str(checkpoint_out),
            "--train-seq-len",
            "8",
            "--max-steps",
            "2",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert checkpoint_metadata.returncode == 0, checkpoint_metadata.stderr
    checkpoint_payload = json.loads(checkpoint_metadata.stdout)
    assert checkpoint_payload["model_family"] == "gpt"
    assert checkpoint_payload["backend"] == "tile-cuda"
    assert checkpoint_payload["status"] == "native-checkpoint-metadata-written"
    assert checkpoint_payload["checkpoint_metadata_smoke"] is True
    assert checkpoint_payload["metadata_only"] is True
    assert checkpoint_payload["trained_layers"] == 0
    assert checkpoint_payload["target_layers"] == 12
    assert checkpoint_payload["num_layers"] == 12
    assert checkpoint_payload["num_heads"] == 12
    assert checkpoint_payload["vocab"] == 50257
    assert checkpoint_payload["padded_vocab"] == 50304
    assert checkpoint_payload["model_dim"] == 768
    assert checkpoint_payload["max_seq_len"] == 8
    assert checkpoint_payload["checkpoint_step"] == 2
    assert checkpoint_payload["size_matches"] is True
    assert checkpoint_payload["passed"] is True
    checkpoint_path = Path(checkpoint_payload["checkpoint_path"])
    checkpoint_info = read_native_gpt2_checkpoint_info(checkpoint_path)
    assert checkpoint_info.version == 5
    assert checkpoint_info.precision == "bf16"
    assert checkpoint_info.max_seq_len == 8
    assert checkpoint_info.vocab_size == 50257
    assert checkpoint_info.num_layers == 12
    assert checkpoint_info.num_heads == 12
    assert checkpoint_info.channels == 768
    assert checkpoint_info.padded_vocab_size == 50304
    assert checkpoint_info.size_matches is True
    assert checkpoint_info.done_marker_exists is True
    assert latest_native_gpt2_checkpoint(checkpoint_out) == checkpoint_path

    native_info = subprocess.run(
        [str(cli), "--native-info", "--native-checkpoint", str(checkpoint_path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert native_info.returncode == 0, native_info.stderr
    native_info_payload = json.loads(native_info.stdout)
    assert native_info_payload["status"] == "native-checkpoint-info"
    assert native_info_payload["runtime"] == "native-cpp"
    assert native_info_payload["backend"] == "tile-cuda"
    assert native_info_payload["path"] == str(checkpoint_path)
    assert native_info_payload["precision"] == "bf16"
    assert native_info_payload["max_seq_len"] == 8
    assert native_info_payload["vocab_size"] == 50257
    assert native_info_payload["padded_vocab_size"] == 50304
    assert native_info_payload["num_layers"] == 12
    assert native_info_payload["num_heads"] == 12
    assert native_info_payload["channels"] == 768
    assert native_info_payload["size_matches"] is True
    assert native_info_payload["checkpoint_step"] == 2
    assert native_info_payload["done_marker_exists"] is True
    assert native_info_payload["prompt_generation_status"] == "native-token-sampler-available"

    inspect_info = subprocess.run(
        [str(cli), "--inspect-checkpoint", str(checkpoint_path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert inspect_info.returncode == 0, inspect_info.stderr
    assert json.loads(inspect_info.stdout)["status"] == "native-checkpoint-info"

    sample_plan = subprocess.run(
        [
            str(cli),
            "--sample-checkpoint",
            str(checkpoint_path),
            "--prompt-tokens",
            "1,2,3",
            "--max-new-tokens",
            "4",
            "--temperature",
            "0.7",
            "--top-k",
            "8",
            "--repetition-penalty",
            "1.1",
            "--seed",
            "42",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert sample_plan.returncode in {0, 2}
    sample_plan_payload = json.loads(sample_plan.stdout)
    assert sample_plan_payload["status"] == "native-checkpoint-sampler"
    assert sample_plan_payload["runtime"] == "native-cpp"
    assert sample_plan_payload["backend"] == "tile-cuda"
    assert sample_plan_payload["path"] == str(checkpoint_path)
    assert sample_plan_payload["prompt_token_count"] == 3
    assert sample_plan_payload["sequence_token_count"] in {3, 7}
    assert sample_plan_payload["max_new_tokens"] == 4
    assert sample_plan_payload["sampling_strategy"] == "top-k-temperature"
    assert sample_plan_payload["temperature"] == pytest.approx(0.7)
    assert sample_plan_payload["top_k"] == 8
    assert sample_plan_payload["repetition_penalty"] == pytest.approx(1.1)
    assert sample_plan_payload["seed"] == 42
    assert sample_plan_payload["blocks_executed"] == 12
    assert sample_plan_payload["transformer_blocks_executed"] is True
    assert sample_plan_payload["final_logits_executed"] is True
    assert sample_plan_payload["torch_required"] is False
    assert sample_plan_payload["graph_editor_node_flow"] is False
    if sample_plan.returncode == 0:
        assert sample_plan_payload["forward_pass_status"] == "cuda-tile-forward-executed"
        assert sample_plan_payload["sequence_token_count"] == 7
        assert sample_plan_payload["generated_token_count"] == 4
        assert len(sample_plan_payload["generated_tokens"]) == 4
    else:
        assert sample_plan_payload["forward_pass_status"] == "cuda-tile-forward-failed"
        assert sample_plan_payload["sequence_token_count"] == 3
        assert sample_plan_payload["generated_token_count"] == 0
        assert sample_plan_payload["generated_tokens"] == []

    logits_bad_token = subprocess.run(
        [
            str(cli),
            "--checkpoint-logits-smoke",
            "--native-checkpoint",
            str(checkpoint_path),
            "--prompt-tokens",
            "999999",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert logits_bad_token.returncode == 2
    assert "outside checkpoint vocab" in logits_bad_token.stderr

    qkv_bad_block = subprocess.run(
        [
            str(cli),
            "--checkpoint-qkv-smoke",
            "--native-checkpoint",
            str(checkpoint_path),
            "--prompt-tokens",
            "1,2,3",
            "--checkpoint-block-index",
            "999",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert qkv_bad_block.returncode == 2
    assert "outside checkpoint layer range" in qkv_bad_block.stderr

    missing_load_smoke = subprocess.run(
        [
            str(cli),
            "--checkpoint-load-smoke",
            "--native-checkpoint",
            str(checkpoint_path.with_name("missing.bin")),
            "--checkpoint-load-elements",
            "8",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert missing_load_smoke.returncode == 2
    assert "failed to open native checkpoint" in missing_load_smoke.stderr

    bad_load_tensor = subprocess.run(
        [
            str(cli),
            "--checkpoint-load-smoke",
            "--native-checkpoint",
            str(checkpoint_path),
            "--checkpoint-load-tensor",
            "h.999.not_a_tensor",
            "--checkpoint-load-elements",
            "8",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert bad_load_tensor.returncode == 2
    assert "checkpoint tensor not found" in bad_load_tensor.stderr

    layout_proc = subprocess.run(
        [
            str(cli),
            "--checkpoint-layout",
            "--native-checkpoint",
            str(checkpoint_path),
            "--checkpoint-layout-sample-buffers",
            "3",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert layout_proc.returncode == 0, layout_proc.stderr
    layout_payload = json.loads(layout_proc.stdout)
    assert layout_payload["status"] == "native-checkpoint-layout"
    assert layout_payload["runtime"] == "native-cpp"
    assert layout_payload["torch_required"] is False
    assert layout_payload["cuda_required"] is False
    assert layout_payload["graph_editor_node_flow"] is False
    assert layout_payload["checkpoint_payload_layout_matches_writer"] is True
    assert layout_payload["layout_parameter_count"] == layout_payload["parameter_count"]
    assert layout_payload["parameter_layout"]["buffer_count"] == 2 + (12 * 12) + 2
    assert layout_payload["parameter_layout"]["buffers"][0]["name"] == "wte.weight"
    assert layout_payload["parameter_layout"]["buffers"][0]["offset"] == 0
    assert layout_payload["parameter_layout"]["buffers"][1]["name"] == "wpe.weight"
    assert len(layout_payload["payload_samples"]) == 3
    assert layout_payload["payload_samples"][0]["file_offset_bytes"] == 1024

    bad_backend = subprocess.run(
        [str(cli), "--dataset-alias", str(dataset_path), "--backend", "tile_cuda", "--print-plan"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert bad_backend.returncode == 2
    assert "Invalid backend: tile_cuda" in bad_backend.stderr

    odd_dataset = tmp_path / "odd_uint16"
    odd_dataset.mkdir()
    (odd_dataset / "fineweb_train_000000.bin").write_bytes(b"\x01")
    (odd_dataset / "fineweb_val_000000.bin").write_bytes(struct.pack("<2H", 1, 2))
    odd_run = subprocess.run(
        [str(cli), "--dataset-alias", str(odd_dataset), "--target", "/bin/echo", "--dry-run"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert odd_run.returncode == 2
    assert "uint16 token shard has odd byte size" in odd_run.stderr

    default_env = os.environ.copy()
    default_env["NFN_DATASETS_DIR"] = str(tmp_path)
    default_dry_run = subprocess.run(
        [str(cli), "--target", "/bin/echo", "--dry-run"],
        env=default_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert default_dry_run.returncode == 0, default_dry_run.stderr
    default_alias_payload = json.loads(default_dry_run.stdout)
    assert default_alias_payload["backend"] == "tile-cuda"
    assert default_alias_payload["train_shard"] == str(default_dataset_path / "fineweb_train_000000.bin")
    assert default_alias_payload["val_shard"] == str(default_dataset_path / "fineweb_val_000000.bin")

    executed = subprocess.run(
        [
            str(cli),
            "--dataset-alias",
            str(dataset_path),
            "--backend",
            "llm-kittens",
            "--target",
            "/bin/echo",
            "--activation",
            "moa",
            "--moa-interval",
            "7",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert executed.returncode == 2
    assert "Invalid backend: llm-kittens" in executed.stderr
    assert executed.stdout == ""


def test_unified_native_train_cli_builds_dispatches_dense_gpt_aliases_and_rejects_unsupported(tmp_path: Path) -> None:
    if shutil.which("c++") is None:
        pytest.skip("c++ compiler not available")
    root = Path(__file__).resolve().parents[1]
    unified = tmp_path / "nfn_native_train"
    fake_gpt = tmp_path / "nfn_gpt_native_train"
    fake_gpt.write_text("#!/usr/bin/env bash\nprintf '%s\\n' \"$@\"\n", encoding="utf-8")
    fake_gpt.chmod(0o755)

    build = subprocess.run(
        ["bash", str(root / "tools" / "build_native_train_cli.sh"), str(unified)],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert build.returncode == 0, build.stderr
    assert unified.exists()

    for model in ("gpt", "gpt2", "gpt3"):
        dense_gpt = subprocess.run(
            [
                str(unified),
                "train",
                "--base-model",
                model,
                "--native-gpt-cli",
                str(fake_gpt),
                "--dataset-alias",
                "/tmp/native-cache",
                "--dry-run",
                "--eval-every-steps",
                "1000",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert dense_gpt.returncode == 0, dense_gpt.stderr
        assert f"--model-family\n{model}" in dense_gpt.stdout
        assert "--dataset-alias\n/tmp/native-cache" in dense_gpt.stdout
        assert "--eval-every-steps\n1000" in dense_gpt.stdout
        assert "--train-transformer-lm" in dense_gpt.stdout
        assert "--backend\ntile-cuda" in dense_gpt.stdout
        assert "--base-model" not in dense_gpt.stdout

    nanogpt_dense = subprocess.run(
        [
            str(unified),
            "train",
            "--base-model",
            "nanogpt",
            "--native-gpt-cli",
            str(fake_gpt),
            "--dataset-alias",
            "/tmp/native-cache",
            "--dry-run",
            "--eval-every-steps",
            "1000",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert nanogpt_dense.returncode == 0, nanogpt_dense.stderr
    assert "--model-family\ngpt" in nanogpt_dense.stdout
    assert "--template-name\nnanogpt" in nanogpt_dense.stdout
    assert "--dataset-alias\n/tmp/native-cache" in nanogpt_dense.stdout
    assert "--eval-every-steps\n1000" in nanogpt_dense.stdout
    assert "--train-transformer-lm" in nanogpt_dense.stdout
    assert "--backend\ntile-cuda" in nanogpt_dense.stdout
    assert "--base-model" not in nanogpt_dense.stdout

    nano_gpt_alias = subprocess.run(
        [
            str(unified),
            "train",
            "--base-model",
            "nano-gpt",
            "--native-gpt-cli",
            str(fake_gpt),
            "--dataset-alias",
            "/tmp/native-cache",
            "--dry-run",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert nano_gpt_alias.returncode == 0, nano_gpt_alias.stderr
    assert "--model-family\ngpt" in nano_gpt_alias.stdout
    assert "--template-name\nnanogpt" in nano_gpt_alias.stdout
    assert "--base-model" not in nano_gpt_alias.stdout

    high_level_aliases = subprocess.run(
        [
            str(unified),
            "train",
            "--base-model",
            "gpt3",
            "--native-gpt-cli",
            str(fake_gpt),
            "--dataset",
            "tinystories",
            "--native-cuda-print-command",
            "--native-cuda-no-checkpoint",
            "--native-cuda-startup-only",
            "--kernel-backend",
            "tile-cuda",
            "--output",
            "/tmp/native-model.pt",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert high_level_aliases.returncode == 0, high_level_aliases.stderr
    assert "--model-family gpt3" in high_level_aliases.stdout
    assert "--tinystories" in high_level_aliases.stdout
    assert "--no-checkpoint" in high_level_aliases.stdout
    assert "--startup-only" in high_level_aliases.stdout
    assert "--backend tile-cuda" in high_level_aliases.stdout
    assert "--output-dir /tmp/native-model" in high_level_aliases.stdout
    assert "--train-transformer-lm" in high_level_aliases.stdout
    assert "--train-seq-len 2048" in high_level_aliases.stdout
    assert "--native-cuda-print-command" not in high_level_aliases.stdout
    assert "--native-cuda-startup-only" not in high_level_aliases.stdout
    assert "--kernel-backend" not in high_level_aliases.stdout
    assert "--output " not in high_level_aliases.stdout

    for catalog_flag in ("--list-templates", "--native-cuda-list-templates"):
        catalog = subprocess.run(
            [
                str(unified),
                "train",
                "--base-model",
                "gpt",
                "--native-gpt-cli",
                str(fake_gpt),
                catalog_flag,
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert catalog.returncode == 0, catalog.stderr
        assert "--model-family\ngpt" in catalog.stdout
        assert "--list-templates" in catalog.stdout
        assert "--native-cuda-list-templates" not in catalog.stdout
        assert "--train-transformer-lm" not in catalog.stdout
        assert "--dataset-alias" not in catalog.stdout
        assert "TinyStories" not in catalog.stdout

    coverage = subprocess.run(
        [str(unified), "--list-models", "--json"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert coverage.returncode == 0, coverage.stderr
    payload = json.loads(coverage.stdout)
    statuses = {item["name"]: item["status"] for item in payload["models"]}
    native_targets = {item["name"]: item["native_target"] for item in payload["models"]}
    transformer_statuses = {item["name"]: item["transformer_lm_status"] for item in payload["models"]}
    token_statuses = {item["name"]: item["token_lm_status"] for item in payload["models"]}
    geometry_statuses = {item["name"]: item["geometry_status"] for item in payload["models"]}
    assert statuses["gpt"] == "implemented"
    assert statuses["gpt2"] == "implemented"
    assert statuses["gpt3"] == "implemented"
    assert statuses["gpt2-evo"] == "implemented"
    assert statuses["nanogpt"] == "implemented"
    assert native_targets["gpt"] == "nfn_gpt_native_train"
    assert native_targets["gpt2"] == "nfn_gpt_native_train"
    assert native_targets["gpt3"] == "nfn_gpt_native_train"
    assert native_targets["gpt2-evo"] == "nfn_gpt2_evo_native_train"
    assert native_targets["nanogpt"] == "nfn_gpt_native_train"
    assert transformer_statuses["gpt"] == "native-transformer-lm"
    assert transformer_statuses["gpt2"] == "native-transformer-lm"
    assert transformer_statuses["gpt3"] == "native-transformer-lm"
    assert transformer_statuses["gpt2-evo"] == "native-dense-gpt-layer-evo-delegate"
    assert transformer_statuses["nanogpt"] == "native-transformer-lm"
    assert token_statuses["gpt2-evo"] == "not-applicable"
    assert token_statuses["nanogpt"] == "implemented"
    assert geometry_statuses["gpt2-evo"] == "dense-gpt2-compatible-layer-evo-delegate"
    assert geometry_statuses["nanogpt"] == "dense-gpt-template-geometry"
    sdk_payload = native_train_model_registry(native_train_cli=str(unified))
    sdk_statuses = {item["name"]: item["status"] for item in sdk_payload["models"]}
    sdk_transformer_statuses = {item["name"]: item["transformer_lm_status"] for item in sdk_payload["models"]}
    assert sdk_statuses == statuses
    assert sdk_transformer_statuses == transformer_statuses

    llama = subprocess.run(
        [str(unified), "--base-model", "llama", "--tinystories"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert llama.returncode == 2
    assert "native CUDA Tile trainer for llama is not implemented yet" in llama.stderr
    assert "implement this family's CUDA Tile C++ kernels first" in llama.stderr


def test_missing_family_native_trainers_build_and_unified_frontend_dispatches(tmp_path: Path) -> None:
    if shutil.which("c++") is None:
        pytest.skip("c++ compiler not available")
    root = Path(__file__).resolve().parents[1]
    unified = tmp_path / "nfn_native_train"

    build_unified = subprocess.run(
        ["bash", str(root / "tools" / "build_native_train_cli.sh"), str(unified)],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert build_unified.returncode == 0, build_unified.stderr

    build_missing = subprocess.run(
        ["bash", str(root / "tools" / "build_native_missing_trainers.sh"), str(tmp_path)],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert build_missing.returncode == 0, build_missing.stderr
    gpt2_evo = tmp_path / "nfn_gpt2_evo_native_train"
    assert gpt2_evo.exists()
    nanogpt = tmp_path / "nfn_nanogpt_native_train"
    assert nanogpt.exists()
    dense_gpt_source = (root / "neuralfn" / "csrc" / "native_gpt2" / "nfn_gpt2_native_train.cpp").read_text(
        encoding="utf-8"
    )
    assert "std::string tile_cuda_activation_dtype = \"nvfp4\";" in dense_gpt_source
    assert "--tile-cuda-activation-dtype nvfp4|float32|none" in dense_gpt_source
    assert "--require-native-nvfp4-activation-packing" in dense_gpt_source
    assert "required-nvfp4-native-packing-missing" in dense_gpt_source
    assert "native_tile_cuda_activation_json" in dense_gpt_source
    assert "requested-nvfp4-not-yet-packed-native-dense-gpt" in dense_gpt_source
    assert "json_escape(cfg.tile_cuda_activation_dtype)" in dense_gpt_source
    fake_gpt = tmp_path / "nfn_gpt_native_train"
    fake_gpt.write_text("#!/usr/bin/env bash\nprintf '%s\\n' \"$@\"\n", encoding="utf-8")
    fake_gpt.chmod(0o755)

    evo_help = subprocess.run(
        [str(gpt2_evo), "--help"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert evo_help.returncode == 0, evo_help.stderr
    assert "Compiled NeuralFn GPT-2 evo native training preflight" in evo_help.stdout
    assert "--template-name NAME" in evo_help.stdout
    assert "--graph-file PATH" in evo_help.stdout
    assert "--tile-cuda-activation-dtype nvfp4|float32|none" in evo_help.stdout
    assert "--evo-layer-index N" in evo_help.stdout
    assert "--evo-layer-population N" in evo_help.stdout
    assert "--tile-ops-lib PATH" in evo_help.stdout
    assert "--cuda-runtime-lib PATH" in evo_help.stdout
    assert "--require-native-nvfp4-activation-packing" in evo_help.stdout
    assert "--smoke-evo-kernels" in evo_help.stdout
    assert "--native-cuda-*" in evo_help.stdout

    evo_plan_proc = subprocess.run(
        [
            str(gpt2_evo),
            "--native-cuda-print-plan",
            "--dataset-alias",
            "/tmp/native-cache",
            "--eval-every-steps",
            "1000",
            "--template-name",
            "gpt2-moa",
            "--tile-cuda-activation-dtype",
            "nvfp4",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert evo_plan_proc.returncode == 0, evo_plan_proc.stderr
    evo_plan = json.loads(evo_plan_proc.stdout)
    assert evo_plan["model_family"] == "gpt2-evo"
    assert evo_plan["status"] == "native-preflight-dense-gpt-layer-evo-delegate"
    assert evo_plan["template_name"] == "gpt2_moa"
    assert evo_plan["graph_file"] == ""
    assert evo_plan["template_known"] is True
    assert evo_plan["selected_graph_support_status"] == "native-dense-gpt-layer-evo-delegate"
    assert evo_plan["selected_graph_native_runnable"] is True
    assert evo_plan["shipped_template_catalog_count"] == len(SHIPPED_GPT_TEMPLATE_PRESETS)
    assert evo_plan["shipped_template_catalog"] == list(SHIPPED_GPT_TEMPLATE_PRESETS)
    assert evo_plan["shape"]["num_layers"] == 12
    assert evo_plan["shape"]["model_dim"] == 768
    assert evo_plan["shape"]["num_heads"] == 12
    assert evo_plan["shape"]["vocab_size"] == 50257
    assert evo_plan["schedule"]["eval_every_steps"] == 1000
    assert evo_plan["schedule"]["grad_accum_steps"] == 8
    assert evo_plan["optimizer"]["profile"] == "adamw"
    assert evo_plan["tile_cuda"]["activation_dtype"] == "nvfp4"
    assert evo_plan["tile_cuda"]["requested_activation_dtype"] == "nvfp4"
    assert evo_plan["tile_cuda"]["effective_activation_dtype"] == "bf16-float32-mixed"
    assert evo_plan["tile_cuda"]["native_activation_packing_active"] is False
    assert evo_plan["tile_cuda"]["nvfp4_activation_packing_active"] is False
    assert evo_plan["tile_cuda"]["native_activation_packing_required"] is False
    assert evo_plan["tile_cuda"]["native_activation_packing_error"] == ""
    assert (
        evo_plan["tile_cuda"]["activation_dtype_status"]
        == "requested-nvfp4-not-yet-packed-native-dense-gpt"
    )
    assert evo_plan["layer_evo"]["enabled"] is True
    assert evo_plan["layer_evo"]["layer_index"] == 6
    assert evo_plan["layer_evo"]["interval"] == 10
    assert evo_plan["layer_evo"]["population"] == 8
    assert evo_plan["layer_evo"]["evo_block_parameters"] > 0
    assert evo_plan["estimated_parameters"] > evo_plan["layer_evo"]["evo_block_parameters"]
    assert "NVFP4 activation intent preserved in the compiled native plan" in evo_plan["available_native_kernels"]
    assert "template/custom graph selector parsed before graph-backed runtime import" in evo_plan["available_native_kernels"]
    assert (
        "device-side evo candidate mutation, best-loss selection, and best-candidate adoption Tile ABI"
        in evo_plan["available_native_kernels"]
    )
    assert (
        "dense GPT native transformer trainer delegate with --layer-evo for GPT-2-compatible templates"
        in evo_plan["available_native_kernels"]
    )
    assert (
        "native CUDA forward-only candidate evaluation for current plus mutated evo-layer weights"
        in evo_plan["available_native_kernels"]
    )
    assert (
        "wire the native evo Tile ABI into the layer-evolution loop without host graph-editor tensor flow"
        not in evo_plan["required_native_kernels"]
    )
    assert "copy/adopt best evo block candidate without host graph-editor tensor flow" not in evo_plan["required_native_kernels"]

    evo_modern_plan_proc = subprocess.run(
        [
            str(gpt2_evo),
            "--print-plan",
            "--dataset-alias",
            "/tmp/native-cache",
            "--template-name",
            "gpt2_modern",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert evo_modern_plan_proc.returncode == 0, evo_modern_plan_proc.stderr
    evo_modern_plan = json.loads(evo_modern_plan_proc.stdout)
    assert evo_modern_plan["template_name"] == "gpt2_modern"
    assert evo_modern_plan["template_known"] is True
    assert evo_modern_plan["selected_graph_support_status"] == "native-dense-gpt-layer-evo-delegate"
    assert evo_modern_plan["selected_graph_native_runnable"] is True

    evo_custom_graph_path = tmp_path / "gpt2-evo-custom.json"
    evo_custom_graph_path.write_text('{"nodes": {}, "edges": {}}\n', encoding="utf-8")
    evo_custom_graph = subprocess.run(
        [
            str(gpt2_evo),
            "--print-plan",
            "--dataset-alias",
            "/tmp/native-cache",
            "--graph-file",
            str(evo_custom_graph_path),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert evo_custom_graph.returncode == 0, evo_custom_graph.stderr
    evo_custom_graph_plan = json.loads(evo_custom_graph.stdout)
    assert evo_custom_graph_plan["graph_file"].endswith("gpt2-evo-custom.json")
    assert evo_custom_graph_plan["graph_file_exists"] is True
    assert evo_custom_graph_plan["graph_file_size_bytes"] == evo_custom_graph_path.stat().st_size
    assert evo_custom_graph_plan["template_name"] == "gpt2"
    assert evo_custom_graph_plan["template_known"] is True
    assert evo_custom_graph_plan["selected_graph_support_status"] == "custom-graph-native-trainer-missing"
    assert evo_custom_graph_plan["selected_graph_native_runnable"] is False

    evo_missing_custom_graph = subprocess.run(
        [
            str(gpt2_evo),
            "--print-plan",
            "--dataset-alias",
            "/tmp/native-cache",
            "--graph-file",
            str(tmp_path / "missing-gpt2-evo-custom.json"),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert evo_missing_custom_graph.returncode == 0, evo_missing_custom_graph.stderr
    evo_missing_custom_graph_plan = json.loads(evo_missing_custom_graph.stdout)
    assert evo_missing_custom_graph_plan["graph_file"].endswith("missing-gpt2-evo-custom.json")
    assert evo_missing_custom_graph_plan["graph_file_exists"] is False
    assert evo_missing_custom_graph_plan["graph_file_size_bytes"] == -1
    assert evo_missing_custom_graph_plan["selected_graph_support_status"] == "custom-graph-file-missing"
    assert evo_missing_custom_graph_plan["status"] == "custom-graph-file-missing"
    assert evo_missing_custom_graph_plan["selected_graph_native_runnable"] is False

    evo_unknown_template = subprocess.run(
        [
            str(gpt2_evo),
            "--print-plan",
            "--preset",
            "typo-not-a-shipped-template",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert evo_unknown_template.returncode == 0, evo_unknown_template.stderr
    evo_unknown_plan = json.loads(evo_unknown_template.stdout)
    assert evo_unknown_plan["template_name"] == "typo_not_a_shipped_template"
    assert evo_unknown_plan["template_known"] is False
    assert evo_unknown_plan["selected_graph_support_status"] == "unknown-template"
    assert evo_unknown_plan["selected_graph_native_runnable"] is False

    evo_required_nvfp4 = subprocess.run(
        [
            str(gpt2_evo),
            "--print-plan",
            "--require-native-nvfp4-activation-packing",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert evo_required_nvfp4.returncode == 0, evo_required_nvfp4.stderr
    evo_required_nvfp4_plan = json.loads(evo_required_nvfp4.stdout)
    assert evo_required_nvfp4_plan["tile_cuda"]["native_activation_packing_required"] is True
    assert (
        evo_required_nvfp4_plan["tile_cuda"]["activation_dtype_status"]
        == "required-nvfp4-native-packing-missing"
    )
    assert "intent only" in evo_required_nvfp4_plan["tile_cuda"]["native_activation_packing_error"]

    bad_evo_optimizer = subprocess.run(
        [str(gpt2_evo), "--print-plan", "--optimizer-profile", "sm120_adamw"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert bad_evo_optimizer.returncode == 2
    assert "--optimizer-profile must be adamw" in bad_evo_optimizer.stderr

    help_proc = subprocess.run(
        [str(nanogpt), "--help"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert help_proc.returncode == 0, help_proc.stderr
    assert "Compiled NeuralFn NanoGPT native trainer" in help_proc.stdout
    assert "--optimizer-profile adamw" in help_proc.stdout
    assert "--check-tile-ops" in help_proc.stdout
    assert "--smoke-tile-ops" in help_proc.stdout
    assert "--smoke-optimizer-step" in help_proc.stdout
    assert "--smoke-training-loop-step" in help_proc.stdout
    assert "--smoke-lm-step" in help_proc.stdout
    assert "--smoke-token-train-step" in help_proc.stdout
    assert "--smoke-embedding-norm-step" in help_proc.stdout
    assert "--smoke-qkv-layout-step" in help_proc.stdout
    assert "--smoke-fused-qkv-attention-step" in help_proc.stdout
    assert "--smoke-transformer-block-step" in help_proc.stdout
    assert "--smoke-mlp-step" in help_proc.stdout
    assert "--smoke-attention-step" in help_proc.stdout
    assert "--train-token-lm" in help_proc.stdout
    assert "--eval-every-steps N" in help_proc.stdout
    assert "--eval-batches N" in help_proc.stdout
    assert "Validation batches per eval, default 20" in help_proc.stdout
    assert "--eval-batch-size N" in help_proc.stdout
    assert "Validation microbatch rows, default 64" in help_proc.stdout
    assert "--tile-ops-lib PATH" in help_proc.stdout
    assert "--cuda-runtime-lib PATH" in help_proc.stdout

    plan_proc = subprocess.run(
        [
            str(nanogpt),
            "--print-plan",
            "--dataset-alias",
            str(tmp_path / "missing-native-cache"),
            "--eval-every-steps",
            "1000",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert plan_proc.returncode == 0, plan_proc.stderr
    plan = json.loads(plan_proc.stdout)
    assert plan["model_family"] == "nanogpt"
    assert plan["shape"]["num_layers"] == 5
    assert plan["shape"]["model_dim"] == 320
    assert plan["shape"]["num_heads"] == 5
    assert plan["shape"]["vocab_size"] == 50257
    assert plan["shape"]["dropout_p"] == 0
    assert plan["schedule"]["eval_every_steps"] == 1000
    assert plan["optimizer"]["profile"] == "adamw"
    optimizer_groups = plan["optimizer"]["parameter_groups"]
    assert optimizer_groups["adamw_step_abi"] == "nfn_native_tile_adamw_step_float32"
    assert plan["token_shards"] is None
    layout = plan["parameter_layout"]
    buffers = layout["buffers"]
    assert layout["total_parameters"] == plan["estimated_parameters"]
    assert layout["buffer_count"] == 43
    assert layout["parameter_dtype"] == "float32"
    assert layout["gradient_dtype"] == "float32"
    assert layout["optimizer_state_dtype"] == "float32"
    assert layout["required_device_buffers"]["parameters"] == layout["total_parameters"]
    assert layout["required_device_buffers"]["gradients"] == layout["total_parameters"]
    assert layout["required_device_buffers"]["adamw_exp_avg"] == layout["total_parameters"]
    assert layout["required_device_buffers"]["adamw_exp_avg_sq"] == layout["total_parameters"]
    assert layout["required_device_buffers"]["clip_scale"] == 1
    assert buffers[0] == {
        "name": "tok_emb.weight",
        "shape": [50257, 320],
        "offset": 0,
        "count": 16082240,
        "weight_decay": True,
    }
    assert buffers[-1]["name"] == "ln_f.weight"
    assert buffers[-1]["weight_decay"] is False
    for previous, current in zip(buffers, buffers[1:]):
        assert current["offset"] == previous["offset"] + previous["count"]
    groups = optimizer_groups["groups"]
    assert [group["name"] for group in groups] == ["decay", "no_decay"]
    assert groups[0]["weight_decay"] == plan["optimizer"]["weight_decay"]
    assert groups[1]["weight_decay"] == 0
    assert groups[0]["total_elements"] + groups[1]["total_elements"] == layout["total_parameters"]
    for group in groups:
        assert group["total_elements"] == sum(buffers[index]["count"] for index in group["buffer_indices"])
    assert all(buffers[index]["weight_decay"] for index in groups[0]["buffer_indices"])
    assert not any(buffers[index]["weight_decay"] for index in groups[1]["buffer_indices"])
    step_plan = plan["training_step_plan"]
    stages = step_plan["stages"]
    assert step_plan["stage_count"] == 114
    assert step_plan["ready_stage_count"] == 114
    assert step_plan["requires_wiring_stage_count"] == 0
    assert step_plan["missing_abi_stage_count"] == 0
    assert stages[0] == {
        "name": "token_embedding.forward",
        "phase": "forward",
        "status": "ready",
        "kernel_abi": "nfn_native_tile_token_embedding_float32",
        "elements": 20971520,
    }
    assert stages[2]["kernel_abi"] == "nfn_native_tile_scaled_residual_add_float32"
    stages_by_name = {stage["name"]: stage for stage in stages}
    assert stages_by_name["lm_head.backward_input"]["kernel_abi"] == "nfn_native_tile_linear_backward_input_float32"
    assert stages_by_name["lm_head.backward_input"]["elements"] == 20971520
    assert stages_by_name["lm_head.backward_weight_tied"]["kernel_abi"] == "nfn_native_tile_linear_backward_weight_float32"
    assert stages_by_name["lm_head.backward_weight_tied"]["elements"] == 16082240
    assert stages_by_name["blocks.0.attn.qkv.split"]["status"] == "ready"
    assert stages_by_name["blocks.0.attn.qkv.split"]["kernel_abi"] == "nfn_native_tile_split_qkv_float32"
    assert stages_by_name["blocks.0.attn.sdpa.backward"]["kernel_abi"] == (
        "nfn_native_tile_scaled_dot_product_attention_backward_float32"
    )
    assert stages_by_name["blocks.0.attn.qkv.grad_merge"]["kernel_abi"] == "nfn_native_tile_merge_qkv_float32"
    assert stages_by_name["blocks.0.attn.qkv.backward"]["status"] == "ready"
    assert stages_by_name["blocks.0.mlp.fc.forward"]["status"] == "ready"
    assert stages_by_name["blocks.0.mlp.gelu.forward"]["kernel_abi"] == "nfn_native_tile_gelu_float32"
    assert stages_by_name["blocks.0.mlp.proj.forward"]["status"] == "ready"
    assert stages_by_name["blocks.0.mlp.proj.backward"]["status"] == "ready"
    assert stages_by_name["blocks.0.mlp.gelu.backward"]["kernel_abi"] == "nfn_native_tile_gelu_backward_float32"
    assert stages_by_name["blocks.0.mlp.fc.backward"]["status"] == "ready"
    assert stages_by_name["lm_head.forward"]["status"] == "ready"
    assert stages[-4]["name"] == "gradient_zero"
    assert stages[-4]["status"] == "ready"
    assert stages[-4]["kernel_abi"] == "nfn_native_tile_fill_float32"
    assert stages[-4]["elements"] == layout["total_parameters"]
    assert stages[-1]["name"] == "adamw_step"
    assert stages[-1]["status"] == "ready"
    assert stages[-1]["elements"] == layout["total_parameters"]
    assert (
        "chunked row-wise token and masked token cross entropy logits backward for full GPT-class vocabularies"
        in plan["available_native_kernels"]
    )
    assert "trainer-wide parameter/gradient buffer registry" in plan["available_native_kernels"]
    assert "AdamW parameter-group planner over registered buffers" in plan["available_native_kernels"]
    assert "token embedding weight backward" in plan["available_native_kernels"]
    assert "absolute position embedding backward" in plan["available_native_kernels"]
    assert "linear input backward" in plan["available_native_kernels"]
    assert "linear weight backward" in plan["available_native_kernels"]
    assert "linear bias backward" in plan["available_native_kernels"]
    assert "tied LM head input and weight backward via linear native ABI" in plan["available_native_kernels"]
    assert "scaled residual add forward" in plan["available_native_kernels"]
    assert "GELU activation forward" in plan["available_native_kernels"]
    assert "GELU activation backward" in plan["available_native_kernels"]
    assert "dropout forward/backward native Tile ABI for nonzero dropout_p" in plan["available_native_kernels"]
    assert "MLP projection/GELU forward/backward/update smoke over raw native kernels" in plan["available_native_kernels"]
    assert "scaled dot-product attention backward" in plan["available_native_kernels"]
    assert "fused QKV attention forward/backward/update smoke over raw native kernels" in plan["available_native_kernels"]
    assert "multi-step tied token-LM trainer loop over cached native token shards" in plan["available_native_kernels"]
    assert "global norm clipping scale finalizer and device-scalar gradient scaling" in plan["available_native_kernels"]
    assert "registered-buffer AdamW iteration over decay and no-decay parameter groups" in plan["available_native_kernels"]
    assert "LayerNorm input backward" in plan["available_native_kernels"]
    assert "LayerNorm affine parameter backward" in plan["available_native_kernels"]
    assert "RMSNorm input backward" in plan["available_native_kernels"]
    assert "token embedding weight backward" not in plan["required_native_kernels"]
    assert "causal attention backward and attention-stage wiring" not in plan["required_native_kernels"]
    assert "tied LM head weight/input backward" not in plan["required_native_kernels"]
    assert "attention-stage QKV/output projection wiring" not in plan["required_native_kernels"]
    assert "MLP-stage activation and projection wiring" not in plan["required_native_kernels"]
    assert "global norm clipping over native gradient buffers" not in plan["required_native_kernels"]
    assert "trainer-wide parameter/gradient buffer registry and optimizer wiring" not in plan["required_native_kernels"]
    assert "trainer loop optimizer step wiring over registered parameter buffers" not in plan["required_native_kernels"]
    assert "full trainer loop integration over ready forward, backward, and optimizer stages" in plan[
        "required_native_kernels"
    ]
    assert "dropout forward/backward native Tile ABI for nonzero dropout_p" not in plan["required_native_kernels"]

    dropout_plan_proc = subprocess.run(
        [str(nanogpt), "--print-plan", "--dropout-p", "0.1"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert dropout_plan_proc.returncode == 0, dropout_plan_proc.stderr
    dropout_plan = json.loads(dropout_plan_proc.stdout)
    dropout_step_plan = dropout_plan["training_step_plan"]
    assert dropout_plan["shape"]["dropout_p"] == 0.1
    assert dropout_step_plan["stage_count"] == 124
    assert dropout_step_plan["ready_stage_count"] == 124
    assert dropout_step_plan["missing_abi_stage_count"] == 0
    dropout_stages = {stage["name"]: stage for stage in dropout_step_plan["stages"]}
    assert dropout_stages["blocks.0.dropout.forward"]["kernel_abi"] == "nfn_native_tile_dropout_forward_float32"
    assert dropout_stages["blocks.0.dropout.backward"]["kernel_abi"] == "nfn_native_tile_dropout_backward_float32"
    assert "dropout forward/backward native Tile ABI for nonzero dropout_p" in dropout_plan["available_native_kernels"]
    assert "dropout forward/backward native Tile ABI for nonzero dropout_p" not in dropout_plan["required_native_kernels"]

    shard_dir = tmp_path / "token_shards"
    shard_dir.mkdir()
    (shard_dir / "fineweb_train_000001.bin").write_bytes(struct.pack("<8H", *range(8)))
    (shard_dir / "fineweb_train_000000.bin").write_bytes(
        b"\x88\xd8\x34\x01" + b"\0" * 1020 + struct.pack("<8H", *range(8, 16))
    )
    (shard_dir / "fineweb_val_000000.bin").write_bytes(struct.pack("<4H", *range(16, 20)))
    shard_plan_proc = subprocess.run(
        [
            str(nanogpt),
            "--print-plan",
            "--dataset-alias",
            str(shard_dir),
            "--require-token-shards",
            "--sample-token-batch",
            "--train-seq-len",
            "4",
            "--batch-size",
            "2",
            "--train-batch-tokens",
            "16",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert shard_plan_proc.returncode == 0, shard_plan_proc.stderr
    shard_plan = json.loads(shard_plan_proc.stdout)
    assert shard_plan["token_shards"]["dataset_path"] == str(shard_dir)
    assert shard_plan["token_shards"]["batch_read_strategy"] == "contiguous_shard_segments"
    assert shard_plan["token_shards"]["train_tokens"] == 16
    assert shard_plan["token_shards"]["val_tokens"] == 4
    assert [Path(item["path"]).name for item in shard_plan["token_shards"]["train_shards"]] == [
        "fineweb_train_000000.bin",
        "fineweb_train_000001.bin",
    ]
    assert shard_plan["token_shards"]["train_shards"][0]["header_uint16"] == 512
    assert shard_plan["token_shards"]["batch_plan"]["microbatch_tokens"] == 8
    assert shard_plan["token_shards"]["batch_plan"]["grad_accum_steps"] == 2
    assert shard_plan["token_shards"]["batch_plan"]["train_sequences"] == 2
    assert shard_plan["token_shards"]["batch_plan"]["train_microbatches"] == 1
    assert shard_plan["token_shards"]["batch_plan"]["train_optimizer_steps_per_epoch"] == 1
    assert shard_plan["sample_batch"]["tokens"] == [8, 9, 10, 11, 0, 1, 2, 3]
    assert shard_plan["sample_batch"]["targets"] == [9, 10, 11, 12, 1, 2, 3, 4]

    missing_shards = subprocess.run(
        [str(nanogpt), "--print-plan", "--dataset-alias", str(tmp_path / "missing"), "--require-token-shards"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert missing_shards.returncode == 2
    assert "dataset directory not found" in missing_shards.stderr

    bad_optimizer = subprocess.run(
        [str(nanogpt), "--print-plan", "--optimizer-profile", "sm120_adamw"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert bad_optimizer.returncode == 2
    assert "--optimizer-profile must be adamw" in bad_optimizer.stderr

    missing_tile_ops = subprocess.run(
        [str(nanogpt), "--check-tile-ops", "--tile-ops-lib", str(tmp_path / "missing-tile-ops.so")],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert missing_tile_ops.returncode == 2
    missing_tile_ops_payload = json.loads(missing_tile_ops.stdout)
    assert missing_tile_ops_payload["loaded"] is False
    assert missing_tile_ops_payload["all_required_symbols_found"] is False
    assert missing_tile_ops_payload["required_symbol_count"] >= 30
    assert "error" in missing_tile_ops_payload

    missing_train_loop_tile_ops = subprocess.run(
        [
            str(nanogpt),
            "--train-token-lm",
            "--tile-ops-lib",
            str(tmp_path / "missing-tile-ops.so"),
            "--dataset-alias",
            str(shard_dir),
            "--train-seq-len",
            "4",
            "--batch-size",
            "2",
            "--train-batch-tokens",
            "8",
            "--vocab-size",
            "20",
            "--model-dim",
            "4",
            "--num-heads",
            "1",
            "--max-steps",
            "2",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert missing_train_loop_tile_ops.returncode == 2
    missing_train_loop_payload = json.loads(missing_train_loop_tile_ops.stdout)
    assert missing_train_loop_payload["status"] == "native-token-lm-failed"
    assert missing_train_loop_payload["dataset_loaded"] is True
    assert missing_train_loop_payload["loaded"] is False
    assert missing_train_loop_payload["steps_completed"] == 0
    assert "error" in missing_train_loop_payload

    train_loop_dry_run = subprocess.run(
        [
            str(nanogpt),
            "--train-token-lm",
            "--dry-run",
            "--dataset-alias",
            str(shard_dir),
            "--train-seq-len",
            "4",
            "--batch-size",
            "2",
            "--train-batch-tokens",
            "8",
            "--vocab-size",
            "20",
            "--model-dim",
            "4",
            "--num-heads",
            "1",
            "--max-steps",
            "2",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert train_loop_dry_run.returncode == 0, train_loop_dry_run.stderr
    train_loop_dry_run_payload = json.loads(train_loop_dry_run.stdout)
    assert train_loop_dry_run_payload["model_family"] == "nanogpt"
    assert train_loop_dry_run_payload["token_shards"]["dataset_path"] == str(shard_dir)
    assert train_loop_dry_run_payload["token_shards"]["batch_read_strategy"] == "contiguous_shard_segments"
    assert train_loop_dry_run_payload["schedule"]["max_steps"] == 2

    unified_missing_train_loop = subprocess.run(
        [
            str(unified),
            "--base-model",
            "nanogpt",
            "--train-token-lm",
            "--tile-ops-lib",
            str(tmp_path / "missing-tile-ops.so"),
            "--dataset-alias",
            str(shard_dir),
            "--train-seq-len",
            "4",
            "--batch-size",
            "2",
            "--train-batch-tokens",
            "8",
            "--vocab-size",
            "20",
            "--model-dim",
            "4",
            "--num-heads",
            "1",
            "--max-steps",
            "2",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert unified_missing_train_loop.returncode == 2
    unified_missing_train_loop_payload = json.loads(unified_missing_train_loop.stdout)
    assert unified_missing_train_loop_payload["status"] == "native-token-lm-failed"
    assert unified_missing_train_loop_payload["dataset_loaded"] is True
    assert unified_missing_train_loop_payload["loaded"] is False
    assert unified_missing_train_loop_payload["steps_completed"] == 0

    unified_print_command = subprocess.run(
        [
            str(unified),
            "--base-model",
            "nanogpt",
            "--train-token-lm",
            "--dataset-alias",
            str(shard_dir),
            "--dry-run",
            "--print-command",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert unified_print_command.returncode == 0, unified_print_command.stderr
    assert str(nanogpt) in unified_print_command.stdout
    assert "--train-token-lm" in unified_print_command.stdout
    assert "--dry-run" in unified_print_command.stdout
    assert "--print-command" in unified_print_command.stdout

    dry_run = subprocess.run(
        [
            str(unified),
            "--base-model",
            "nanogpt",
            "--native-gpt-cli",
            str(fake_gpt),
            "--dataset-alias",
            "/tmp/native-cache",
            "--dry-run",
            "--eval-every-steps",
            "1000",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert dry_run.returncode == 0, dry_run.stderr
    assert "--model-family\ngpt" in dry_run.stdout
    assert "--template-name\nnanogpt" in dry_run.stdout
    assert "--dataset-alias\n/tmp/native-cache" in dry_run.stdout
    assert "--eval-every-steps\n1000" in dry_run.stdout
    assert "--base-model" not in dry_run.stdout

    evo_dry_run = subprocess.run(
        [
            str(unified),
            "--base-model",
            "gpt2-evo",
            "--dataset-alias",
            "/tmp/native-cache",
            "--dry-run",
            "--eval-every-steps",
            "1000",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert evo_dry_run.returncode == 0
    evo_dry_run_json_start = evo_dry_run.stdout.find("{")
    assert evo_dry_run_json_start >= 0
    evo_dry_run_plan = json.loads(evo_dry_run.stdout[evo_dry_run_json_start:])
    assert evo_dry_run_plan["schedule"]["eval_every_steps"] == 1000
    assert evo_dry_run_plan["layer_evo"]["enabled"] is True
    assert evo_dry_run_plan["selected_graph_support_status"] == "native-dense-gpt-layer-evo-delegate"

    evo_print_command = subprocess.run(
        [
            str(unified),
            "--base-model",
            "gpt2-evo",
            "--dataset-alias",
            "/tmp/native-cache",
            "--dry-run",
            "--print-command",
            "--eval-every-steps",
            "1000",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert evo_print_command.returncode == 0
    assert str(gpt2_evo) in evo_print_command.stdout
    assert "--dry-run" in evo_print_command.stdout
    assert "--print-command" in evo_print_command.stdout
    assert "--native-cuda-print-command" not in evo_print_command.stdout

    evo_delegate_print_command = subprocess.run(
        [
            str(gpt2_evo),
            "--native-cuda-dry-run",
            "--native-cuda-print-command",
            "--native-cuda-startup-only",
            "--dataset-alias",
            "/tmp/native-cache",
            "--eval-every-steps",
            "1000",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert evo_delegate_print_command.returncode == 0
    assert str(fake_gpt) in evo_delegate_print_command.stdout
    assert evo_delegate_print_command.stdout.count(str(fake_gpt)) == 1
    assert "--train-transformer-lm" in evo_delegate_print_command.stdout
    assert "--layer-evo" in evo_delegate_print_command.stdout
    assert "--tile-cuda-activation-dtype nvfp4" in evo_delegate_print_command.stdout
    assert "--dataset-alias /tmp/native-cache" in evo_delegate_print_command.stdout
    assert "--eval-every-steps 1000" in evo_delegate_print_command.stdout
    assert "--native-cuda-print-command" not in evo_delegate_print_command.stdout
    assert "--startup-only" in evo_delegate_print_command.stdout
    assert "--native-cuda-startup-only" not in evo_delegate_print_command.stdout

    evo_required_nvfp4_delegate = subprocess.run(
        [
            str(gpt2_evo),
            "--native-cuda-dry-run",
            "--native-cuda-print-command",
            "--require-native-nvfp4-activation-packing",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert evo_required_nvfp4_delegate.returncode == 0
    assert "--require-native-nvfp4-activation-packing" in evo_required_nvfp4_delegate.stdout


def test_native_gpt2_build_all_script_supports_temp_outputs(tmp_path: Path) -> None:
    if shutil.which("c++") is None:
        pytest.skip("c++ compiler not available")
    if shutil.which("nvcc") is None:
        pytest.skip("nvcc compiler not available")
    root = Path(__file__).resolve().parents[1]
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    env = os.environ.copy()
    env["NFN_NATIVE_GPT_BINDING_OUT"] = str(tmp_path / f"_native_gpt{ext_suffix}")
    env["NFN_NATIVE_GPT2_BINDING_OUT"] = str(tmp_path / f"_native_gpt2{ext_suffix}")
    env["NFN_NATIVE_TRAIN_BINDING_OUT"] = str(tmp_path / f"_native_train{ext_suffix}")
    env["NFN_NATIVE_GPT2_LAUNCHER_OUT"] = str(tmp_path / "nfn_gpt2_tile_train")
    env["NFN_NATIVE_GPT_CLI_OUT"] = str(tmp_path / "nfn_gpt_native_train")
    env["NFN_NATIVE_GPT2_CLI_OUT"] = str(tmp_path / "nfn_gpt2_native_train")
    env["NFN_NATIVE_TRAIN_CLI_OUT"] = str(tmp_path / "nfn_native_train")
    env["NFN_NATIVE_TRAIN_TILE_OPS_OUT"] = str(tmp_path / "libnfn_native_train_tile_ops.so")
    env["NFN_NATIVE_MISSING_TRAINERS_OUT_DIR"] = str(tmp_path / "missing")

    proc = subprocess.run(
        ["bash", str(root / "tools" / "build_native_gpt2_all.sh")],
        cwd=root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert Path(env["NFN_NATIVE_GPT_BINDING_OUT"]).exists()
    assert Path(env["NFN_NATIVE_GPT2_BINDING_OUT"]).exists()
    assert Path(env["NFN_NATIVE_TRAIN_BINDING_OUT"]).exists()
    assert Path(env["NFN_NATIVE_GPT2_LAUNCHER_OUT"]).exists()
    assert Path(env["NFN_NATIVE_GPT_CLI_OUT"]).exists()
    assert Path(env["NFN_NATIVE_GPT2_CLI_OUT"]).exists()
    assert Path(env["NFN_NATIVE_TRAIN_CLI_OUT"]).exists()
    assert Path(env["NFN_NATIVE_TRAIN_TILE_OPS_OUT"]).exists()
    assert (Path(env["NFN_NATIVE_MISSING_TRAINERS_OUT_DIR"]) / "nfn_nanogpt_native_train").exists()
    assert (Path(env["NFN_NATIVE_MISSING_TRAINERS_OUT_DIR"]) / "nfn_llama_native_train").exists()


def test_native_gpt_cuda_tile_startup_smoke_without_torch(tmp_path: Path) -> None:
    if os.environ.get("NFN_NATIVE_TILE_CUDA_TEST") != "1":
        pytest.skip("set NFN_NATIVE_TILE_CUDA_TEST=1 to run native CUDA Tile smoke coverage")
    root = Path(__file__).resolve().parents[1]
    cli = root / "build" / "nfn_gpt_native_train"
    tile_ops = root / "build" / "libnfn_native_train_tile_ops.so"
    if not cli.exists():
        pytest.skip("build/nfn_gpt_native_train is not built")
    if not tile_ops.exists():
        pytest.skip("build/libnfn_native_train_tile_ops.so is not built")

    dataset_path = _write_uint16_shard_dataset(tmp_path)
    proc = subprocess.run(
        [
            str(cli),
            "--dataset-alias",
            str(dataset_path),
            "--train-transformer-lm",
            "--startup-only",
            "--no-checkpoint",
            "--tile-ops-lib",
            str(tile_ops),
            "--batch-size",
            "1",
            "--train-seq-len",
            "2",
            "--max-steps",
            "0",
            "--layer-evo",
            "--evo-layer-interval",
            "1",
            "--eval-every-steps",
            "0",
            "--eval-batches",
            "0",
        ],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["passed"] is True
    assert payload["status"] == "native-transformer-lm-startup-ready"
    assert payload["backend"] == "tile-cuda"
    assert payload["loaded"] is True
    assert payload["cuda_runtime_loaded"] is True
    assert payload["startup_only"] is True
    assert payload["steps_completed"] == 0
    assert payload["graph_editor_tensor_flow"] is False
    assert payload["torch_required"] is False
    assert payload["lm_head_classifier_strategy_contract"]["graph_editor_tensor_flow"] is False
    assert payload["lm_head_classifier_strategy_contract"]["torch_required"] is False
    assert payload["layer_evo"]["graph_editor_tensor_flow"] is False
    assert payload["layer_evo"]["runtime_enabled"] is True
    assert payload["layer_evo"]["workspace_allocation_strategy"] == "float-arena-plus-int64-device"
    assert payload["layer_evo"]["float_workspace_request_count"] == 3
    assert payload["layer_evo"]["float_workspace_cuda_mallocs_elided"] == 3
    assert payload["layer_evo"]["int64_workspace_cuda_malloc_count"] == 1
    assert payload["native_geometry_contract"]["selector_native_runnable"] is True
    assert payload["tile_ops_library"] == str(tile_ops)
    assert "nfn_native_tile_scaled_dot_product_attention_packed_qkv_bf16_float32" in payload["kernels"]


def test_large_row_reduction_fallbacks_use_tiled_dweight_and_shared_bias_chunks() -> None:
    root = Path(__file__).resolve().parents[1]
    kernels_text = (root / "neuralfn" / "csrc" / "tile_cuda" / "kernels.cu").read_text()
    gpt2_source_text = (
        root / "neuralfn" / "csrc" / "native_gpt2" / "nfn_gpt2_native_train.cpp"
    ).read_text()

    assert "kLayerNormBackwardAffineDefaultRowChunkSize = 128" in kernels_text
    assert "constexpr std::int64_t kDefaultRowChunkSize = 128;" in gpt2_source_text
    assert "NFN_NATIVE_GPT_LAYERNORM_AFFINE_ROW_CHUNK_SIZE" in gpt2_source_text
    assert '\\"layer_norm_backward_affine_row_chunk_size\\"' in gpt2_source_text
    assert '\\"linear_backward_bias_row_chunk_size\\"' in gpt2_source_text
    assert "NFN_TILE_CUDA_LAYERNORM_AFFINE_ROW_CHUNK_SIZE" in kernels_text
    assert "NFN_NATIVE_GPT_LAYERNORM_AFFINE_ROW_CHUNK_SIZE" in kernels_text
    assert "kLinearBackwardBiasRowChunkSize = 256" in kernels_text
    assert "linear_backward_bias_row_chunk_size()" in kernels_text
    assert "NFN_TILE_CUDA_LINEAR_BACKWARD_BIAS_ROW_CHUNK_SIZE" in kernels_text
    assert "NFN_NATIVE_GPT_LINEAR_BACKWARD_BIAS_ROW_CHUNK_SIZE" in kernels_text
    assert "NFN_NATIVE_GPT2_LINEAR_BACKWARD_BIAS_ROW_CHUNK_SIZE" in kernels_text
    assert "resolved_linear_backward_bias_row_chunk_size" in gpt2_source_text
    for function_name in (
        "launch_layer_norm_backward_affine_float32",
        "launch_layer_norm_backward_affine_accumulate_float32",
        "launch_layer_norm_backward_affine_accumulate_with_stats_float32",
    ):
        function_body = kernels_text.rsplit(f"void {function_name}(", 1)[1].split("\nvoid ", 1)[0]
        assert "kRowChunkSize = layer_norm_backward_affine_row_chunk_size()" in function_body
        assert "kRowChunkSize = kLinearBackwardBiasRowChunkSize" not in function_body
    for function_name in (
        "launch_linear_backward_weight_accumulate_bf16_bits_float32",
        "launch_linear_backward_weight_accumulate_float32_bf16_bits",
        "launch_linear_backward_weight_accumulate_bf16_bits_bf16_bits_float32_beta",
        "launch_linear_backward_weight_bias_accumulate_float32_bf16_bits_beta",
    ):
        function_body = kernels_text.rsplit(f"void {function_name}(", 1)[1].split("\nvoid ", 1)[0]
        assert "launch_linear_backward_weight_tiled_float32_fallback" in function_body
        assert "linear_backward_weight_chunked_atomic_" not in function_body
        assert "kRowChunkSize = 256" not in function_body
    for function_name in (
        "launch_linear_backward_bias_float32",
        "launch_linear_backward_bias_accumulate_float32",
    ):
        function_body = kernels_text.rsplit(f"void {function_name}(", 1)[1].split("\nvoid ", 1)[0]
        assert "kRowChunkSize = linear_backward_bias_row_chunk_size()" in function_body
        assert "kRowChunkSize = 256" not in function_body


def test_packed_qkv_uint16_arena_reserves_full_scratch_layout() -> None:
    root = Path(__file__).resolve().parents[1]
    gpt2_source_text = (root / "neuralfn" / "csrc" / "native_gpt2" / "nfn_gpt2_native_train.cpp").read_text()

    assert "const std::int64_t elements_per_tape = qkv_activation_elements + activation_elements * 2;" in gpt2_source_text


def test_packed_attention_ln1_recompute_uses_stats_only_tile_abi() -> None:
    root = Path(__file__).resolve().parents[1]
    gpt2_source_text = (root / "neuralfn" / "csrc" / "native_gpt2" / "nfn_gpt2_native_train.cpp").read_text()
    header_text = (root / "neuralfn" / "csrc" / "native_train" / "tile_ops.h").read_text()
    source_text = (root / "neuralfn" / "csrc" / "native_train" / "tile_ops.cu").read_text()
    kernels_text = (root / "neuralfn" / "csrc" / "tile_cuda" / "kernels.cu").read_text()

    assert "NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LN1_STATS" in gpt2_source_text
    assert "stored_packed_attention_ln1_stats_enabled" in gpt2_source_text
    assert "stored_packed_attention_ln1_stats_blocks" in gpt2_source_text
    assert "stored_packed_attention_ln1_stats_elements" in gpt2_source_text
    assert "stored_packed_attention_ln1_stats_bytes" in gpt2_source_text
    assert "std::uint16_t* ln1 = nullptr" not in gpt2_source_text
    assert "NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LN1_BF16" in gpt2_source_text
    assert "NFN_NATIVE_GPT2_STORE_PACKED_ATTENTION_LN1_BF16" in gpt2_source_text
    assert "stored_packed_attention_ln1_bf16_enabled" in gpt2_source_text
    assert "stored_packed_attention_ln1_bf16_strategy" in gpt2_source_text
    assert "saved-forward-ln1-bf16-direct-qkv-dweight" in gpt2_source_text
    assert "nfn_native_tile_layer_norm_apply_stats_bf16_out_float32" in gpt2_source_text
    assert "nfn_native_tile_layer_norm_apply_stats_bf16_out_float32" in header_text
    assert "launch_layer_norm_apply_stats_bf16_out_float32" in source_text
    assert "layer_norm_apply_stats_bf16_out_float32_kernel" in kernels_text
    assert "NFN_NATIVE_GPT_PACKED_ATTENTION_DPREP_HD64_SPECIALIZED" in kernels_text
    assert (
        "if (value == nullptr || value[0] == '\\0') {\n"
        "      return true;\n"
        "    }"
    ) in kernels_text


def test_native_train_tile_ops_builds_torch_free_c_abi(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    gpt2_source = root / "neuralfn" / "csrc" / "native_gpt2" / "nfn_gpt2_native_train.cpp"
    token_shards_header = root / "neuralfn" / "csrc" / "native_train" / "token_shards.h"
    token_shards_source = root / "neuralfn" / "csrc" / "native_train" / "token_shards.cpp"
    header = root / "neuralfn" / "csrc" / "native_train" / "tile_ops.h"
    source = root / "neuralfn" / "csrc" / "native_train" / "tile_ops.cu"
    kernels = root / "neuralfn" / "csrc" / "tile_cuda" / "kernels.cu"
    gpt2_evo_source = root / "neuralfn" / "csrc" / "native_train" / "gpt2_evo_native_train.cpp"
    build_script = root / "tools" / "build_native_train_tile_ops.sh"
    candidate_bench = root / "tools" / "bench_native_gpt_sm120_candidate.sh"
    speed_tool = root / "tools" / "paired_kernel_speed.py"
    gpt2_source_text = gpt2_source.read_text()
    gpt2_evo_source_text = gpt2_evo_source.read_text()
    token_shards_header_text = token_shards_header.read_text()
    token_shards_source_text = token_shards_source.read_text()
    header_text = header.read_text()
    source_text = source.read_text()
    kernels_text = kernels.read_text()
    script_text = build_script.read_text()
    candidate_bench_text = candidate_bench.read_text()
    speed_source = speed_tool.read_text(encoding="utf-8")

    assert "-Xlinker -Bsymbolic" in script_text
    assert "cudaRuntimeGetVersion" in gpt2_source_text
    assert "cudaDriverGetVersion" in gpt2_source_text
    assert "NFN_NATIVE_GPT_CUDA_VERSION_PREFLIGHT" in gpt2_source_text
    assert '"requested": ' in gpt2_source_text
    assert "cuda_runtime_preflight" in gpt2_source_text
    assert "CUDA driver is unavailable to the native trainer" in gpt2_source_text
    assert "CUDA runtime/driver mismatch" in gpt2_source_text
    assert "cudaMemcpyAsync" in gpt2_source_text
    assert "cudaHostAlloc" in gpt2_source_text
    assert "cudaFreeHost" in gpt2_source_text
    assert "copy_async" in gpt2_source_text
    assert "print_invocation_command(argc, argv)" in gpt2_source_text
    tile_print_command_guard = 'if (cfg.backend == "tile-cuda" && cfg.print_command)'
    assert tile_print_command_guard in gpt2_source_text
    assert gpt2_source_text.index(tile_print_command_guard) < gpt2_source_text.index("resolve_token_shards")
    assert "tile-cuda exits before CUDA/shard setup" in gpt2_source_text
    assert "Print the train_gpt2cu command" not in gpt2_source_text
    assert "token_ids_pinned" in gpt2_source_text
    assert "transformer_lm_token_device_arena" in gpt2_source_text
    assert "transformer_lm_token_u16_pinned_arena" in gpt2_source_text
    assert "token_buffer_allocation_strategy" in gpt2_source_text
    assert "token_device_allocation_strategy" in gpt2_source_text
    assert "token_device_arena_cuda_malloc_count" in gpt2_source_text
    assert "token_i64_arena_cuda_malloc_count" in gpt2_source_text
    assert "token_i64_arena_elements = direct_u16_token_ids_enabled ? 0 : rows * 2" in gpt2_source_text
    assert "token_i64_device_arena_elided" in gpt2_source_text
    assert "token_i64_device_arena_bytes_elided" in gpt2_source_text
    assert "targets = direct_u16_token_ids_enabled ? nullptr : (token_i64_arena + rows)" in gpt2_source_text
    assert "active_targets = direct_u16_token_ids_enabled ? nullptr : (token_i64_arena + active_rows)" in gpt2_source_text
    assert "cudaMalloc transformer_lm_token_i64_arena" not in gpt2_source_text
    assert "cudaMalloc transformer_lm_token_u16_device_arena" not in gpt2_source_text
    assert "token_weight.init_device" in gpt2_source_text
    assert "nfn_native_tile_init_gpt2_token_weight_float32" in gpt2_source_text
    assert "nfn_native_tile_init_gpt2_token_weight_fast_float32" in gpt2_source_text
    assert "#define NFN_TILE_CUDA_TOKEN_WEIGHT_INIT_TILE_SIZE 4096" in kernels_text
    assert "NFN_TILE_CUDA_TOKEN_WEIGHT_INIT_TILE_SHAPE 4096_ic" in kernels_text
    assert "NFN_TILE_CUDA_TOKEN_WEIGHT_INIT_TILE_SHAPE 8192_ic" in kernels_text
    assert "ct::shape{NFN_TILE_CUDA_TOKEN_WEIGHT_INIT_TILE_SHAPE}" in kernels_text
    assert "NFN_NATIVE_GPT_TOKEN_WEIGHT_THREADED_INIT" in kernels_text
    assert "NFN_TILE_CUDA_TOKEN_WEIGHT_THREADED_INIT" in kernels_text
    assert "NFN_NATIVE_GPT_TOKEN_WEIGHT_VECTOR4_INIT" in kernels_text
    assert "NFN_TILE_CUDA_TOKEN_WEIGHT_VECTOR4_INIT" in kernels_text
    assert "NFN_NATIVE_GPT_TOKEN_WEIGHT_BF16_PATTERN_INIT" in kernels_text
    assert "NFN_TILE_CUDA_TOKEN_WEIGHT_BF16_PATTERN_INIT" in kernels_text
    assert "NFN_NATIVE_GPT_TOKEN_WEIGHT_FAST_INT32_INIT" in kernels_text
    assert "NFN_TILE_CUDA_TOKEN_WEIGHT_FAST_INT32_INIT" in kernels_text
    assert "init_gpt2_token_weight_fast_int32_with_bf16_shadow_float32_kernel" in kernels_text
    assert "init_gpt2_token_weight_vector4_with_bf16_shadow_float32_kernel" in kernels_text
    assert "init_gpt2_token_weight_vector4_with_bf16_shadow_convert_float32_kernel" in kernels_text
    assert "bool token_weight_bf16_pattern_init_enabled()" in kernels_text
    assert "gpt2_token_weight_init_float_pattern4" in kernels_text
    assert "make_float4(-0.08f, -0.07f, -0.06f, -0.05f)" in kernels_text
    assert "gpt2_token_weight_init_bf16_pattern4" in kernels_text
    assert "gpt2_token_weight_init_bf16_pattern1" in kernels_text
    assert "make_ushort4(0xbda4u, 0xbd8fu, 0xbd76u, 0xbd4du)" in kernels_text
    vector4_shadow_init = kernels_text[
        kernels_text.index("init_gpt2_token_weight_vector4_with_bf16_shadow_float32_kernel") :
        kernels_text.index("init_gpt2_token_weight_vector4_with_bf16_shadow_convert_float32_kernel")
    ]
    assert "gpt2_token_weight_init_bf16_pattern4(bucket)" in vector4_shadow_init
    assert "shadow_bf16_bits[tail] = gpt2_token_weight_init_bf16_pattern1(bucket)" in vector4_shadow_init
    assert "bf16_bits_from_float(value0)" not in vector4_shadow_init
    vector4_convert_shadow_init = kernels_text[
        kernels_text.index("init_gpt2_token_weight_vector4_with_bf16_shadow_convert_float32_kernel") :
        kernels_text.index("init_gpt2_token_weight_vector4_strided_float32_kernel")
    ]
    assert "const float4 pattern = gpt2_token_weight_init_float_pattern4(bucket)" in vector4_convert_shadow_init
    assert "bf16_bits_from_float(pattern.x)" in vector4_convert_shadow_init
    assert "bf16_bits_from_float(value0)" not in vector4_convert_shadow_init
    token_threaded_init_helper = kernels_text[
        kernels_text.index("bool token_weight_threaded_init_enabled()") :
        kernels_text.index("void launch_init_gpt2_token_weight_threaded_float32")
    ]
    assert "return false;" in token_threaded_init_helper
    token_vector4_init_helper = kernels_text[
        kernels_text.index("bool token_weight_vector4_init_enabled()") :
        kernels_text.index("void launch_init_gpt2_token_weight_threaded_float32")
    ]
    assert "return true;" in token_vector4_init_helper
    token_fast_int32_init_helper = kernels_text[
        kernels_text.index("bool token_weight_fast_int32_tile_init_enabled()") :
        kernels_text.index("void launch_init_gpt2_token_weight_threaded_float32")
    ]
    assert "return true;" in token_fast_int32_init_helper
    assert "FloatArenaRequest" in gpt2_source_text
    assert "Uint16ArenaRequest" in gpt2_source_text
    assert "cudaMalloc transformer_lm_float_arena" in gpt2_source_text
    assert "cudaMalloc transformer_lm_uint16_arena" in gpt2_source_text
    assert "cudaMalloc transformer_lm_combined_device_arena" in gpt2_source_text
    assert "float_allocation_strategy" in gpt2_source_text
    assert "float_allocation_cuda_malloc_count" in gpt2_source_text
    assert "uint16_allocation_strategy" in gpt2_source_text
    assert "uint16_arena_cuda_malloc_count" in gpt2_source_text
    assert "NFN_NATIVE_GPT_CONCURRENT_ARENA_MATERIALIZE" in gpt2_source_text
    assert "materialize_float_and_uint16_arenas_concurrently" in gpt2_source_text
    assert "concurrent_arena_materialize_requested" in gpt2_source_text
    assert "concurrent_arena_materialize_enabled" in gpt2_source_text
    assert "concurrent_arena_materialize_count" in gpt2_source_text
    assert "transformer_device_arena_requested_bytes" in gpt2_source_text
    assert "transformer_device_arena_allocated_bytes" in gpt2_source_text
    assert "DescriptorArenaRequest" in gpt2_source_text
    assert "host_descriptor_arena" in gpt2_source_text
    assert "cudaMalloc transformer_lm_descriptor_arena" in gpt2_source_text
    assert "cudaMemcpy transformer_lm_descriptor_arena" in gpt2_source_text
    assert "descriptor_allocation_strategy" in gpt2_source_text
    assert "descriptor_upload_strategy" in gpt2_source_text
    assert "descriptor_arena_cuda_malloc_count" in gpt2_source_text
    assert "cudaMalloc adamw_param_ptrs" not in gpt2_source_text
    assert "cudaMalloc parameter_fill_ptrs" not in gpt2_source_text
    assert "cudaMemcpy adamw_param_ptrs" not in gpt2_source_text
    assert "cudaMemcpy parameter_fill_ptrs" not in gpt2_source_text
    assert "float*& ln1_weight = block0.ln1_weight" not in gpt2_source_text
    assert '"ln1_weight.fill"' not in gpt2_source_text
    assert '"qkv_weight.fill"' not in gpt2_source_text
    assert "block0_duplicate_allocation_elided" in gpt2_source_text
    assert "block0_duplicate_activation_allocation_elided" in gpt2_source_text
    assert "block0_duplicate_parameter_initialization_elided" in gpt2_source_text
    assert "block0_duplicate_adamw_state_zero_elided" in gpt2_source_text
    assert "torch/" not in header_text
    assert "torch/" not in source_text
    assert "bindings.cpp" not in script_text
    assert "tile_cuda/kernels.cu" in script_text
    assert "-DNFN_TILE_CUDA_USE_CUBLAS_LINEAR=1" in script_text
    assert "-lcublas" in script_text
    assert "nfn_native_tile_adamw_step_float32" in header_text
    assert "nfn_native_tile_adamw_step_with_device_scale_float32" in header_text
    assert "nfn_native_tile_adamw_step_many_with_device_scale_float32" in header_text
    assert "nfn_native_tile_adamw_step_many_with_device_scale_bf16_shadow_float32" in header_text
    assert "nfn_native_tile_adamw_step_many_with_device_scale_bf16_param_float32" in header_text
    assert "nfn_native_tile_adamw_step_many_with_device_scale_bf16_param_bf16_grad_float32" in header_text
    assert "AdamWManyWithDeviceScaleBf16ParamBf16GradFn" in gpt2_source_text
    assert "adamw_many_with_device_scale_bf16_param_bf16_grad" in gpt2_source_text
    assert "adamw_bf16_param_bf16_grad_host" in gpt2_source_text
    assert "adamw_bf16_param_bf16_grad_partial_offsets" in gpt2_source_text
    assert "accumulated_gradients.bf16_params_bf16_grads.sumsq_partials_many" in gpt2_source_text
    assert "qkv-fc-bf16-dweight-staging-direct-bf16-param-adamw" in gpt2_source_text
    assert "adamw_bf16_param_bf16_grad_kernel_loaded" in gpt2_source_text
    assert "block_weight_bf16_gradient_storage_strategy" in gpt2_source_text
    assert "optimized_optimizer_contract_loaded" in gpt2_source_text
    assert "optimized_optimizer_contract_error" in gpt2_source_text
    assert "optimized_optimizer_missing_symbols" in gpt2_source_text
    assert "optimized_optimizer_contract_symbols()" in gpt2_source_text
    assert "missing optimized many-tensor/device-scale AdamW Tile-CUDA symbols" in gpt2_source_text
    assert "optimized_optimizer_contract_loaded =\n                    fill_many != nullptr" in gpt2_source_text
    assert "adamw_many_with_device_scale != nullptr" in gpt2_source_text
    assert "adamw_many_with_device_scale_bf16_shadow != nullptr" in gpt2_source_text
    assert "adamw_many_with_device_scale_bf16_param != nullptr" in gpt2_source_text
    assert "adamw_many_with_device_scale_bf16_param_bf16_grad != nullptr" in gpt2_source_text
    assert "NFN_NATIVE_GPT_TOKEN_WEIGHT_BF16_SHADOW" in gpt2_source_text
    assert "NFN_NATIVE_GPT_FUSE_TOKEN_WEIGHT_BF16_INIT" in gpt2_source_text
    assert "NFN_NATIVE_GPT_FUSE_TOKEN_WEIGHT_BF16_ADAMW_REFRESH" in gpt2_source_text
    assert "NFN_NATIVE_GPT_FUSE_TOKEN_WEIGHT_PADDED_INIT" in gpt2_source_text
    assert "token_weight_bf16_shadow_enabled" in gpt2_source_text
    assert "token_weight_bf16_refresh_count" in gpt2_source_text
    assert "token_weight_bf16_fused_adamw_refresh_count" in gpt2_source_text
    assert "token_weight_bf16_adamw_refresh_fusion_enabled" in gpt2_source_text
    assert "token_weight_bf16_initial_refresh_elided" in gpt2_source_text
    assert "token_weight_padded_init_fusion_enabled" in gpt2_source_text
    assert "token_weight_padding_zero_launches_elided" in gpt2_source_text
    assert "token_weight_bf16.initial_refresh" in gpt2_source_text
    assert "nfn_native_tile_init_gpt2_token_weight_fast_with_bf16_shadow_padded_float32" in header_text
    assert "launch_init_gpt2_token_weight_fast_with_bf16_shadow_padded_float32" in source_text
    assert "init_gpt2_token_weight_vector4_with_bf16_shadow_padded_float32_kernel" in kernels_text
    assert "init_gpt2_token_weight_vector4_strided_with_bf16_shadow_padded_float32_kernel" in kernels_text
    padded_token_init_kernel = kernels_text[
        kernels_text.index("init_gpt2_token_weight_vector4_with_bf16_shadow_padded_float32_kernel") :
        kernels_text.index("init_gpt2_token_weight_vector4_strided_float32_kernel")
    ]
    assert "const float4 pattern = gpt2_token_weight_init_float_pattern4(bucket)" in padded_token_init_kernel
    assert "bf16_bits_from_float(pattern.x)" in padded_token_init_kernel
    assert "shadow_bf16_bits[tail] = bf16_bits_from_float(value)" in padded_token_init_kernel
    assert "bf16_bits_from_float(value0)" not in padded_token_init_kernel
    assert "adamw_float_update_bf16_shadow_offsets" in gpt2_source_text
    assert "adamw_many_with_device_scale_bf16_shadow.float_params_token_shadow" in gpt2_source_text
    assert "split-float32-token-shadow-and-bf16-param-multi-buffer-device-scale" in gpt2_source_text
    assert "elided-block-bf16-primary-token-shadow-fused-adamw" in gpt2_source_text
    assert "nfn_native_tile_linear_backward_input_bf16_bits_weight_bf16_float32" in header_text
    assert "launch_adamw_step_many_with_device_scale_float32" in source_text
    assert "launch_adamw_step_many_with_device_scale_bf16_shadow_float32" in source_text
    assert "launch_adamw_step_many_with_device_scale_bf16_param_float32" in source_text
    assert "launch_adamw_step_many_with_device_scale_bf16_param_bf16_grad_float32" in source_text
    assert "adamw_step_many_with_device_scale_bf16_shadow_float32_kernel" in kernels_text
    assert "adamw_step_many_with_device_scale_bf16_param_float32_kernel" in kernels_text
    assert "adamw_step_many_with_device_scale_bf16_param_bf16_grad_float32_kernel" in kernels_text
    assert "params_bf16_bits" in kernels_text
    assert "grads_bf16_bits" in kernels_text
    assert "bf16_shadow_offsets" in kernels_text
    assert "ct::element_cast<__nv_bfloat16>(next_p)" in kernels_text
    assert "nfn_native_tile_fill_float32" in header_text
    assert "nfn_native_tile_fill_many_float32" in header_text
    assert "nfn_native_tile_fill_many_values_float32" in header_text
    assert "nfn_native_tile_fill_many_values_bf16_bits_float32" in header_text
    assert "launch_fill_many_float32" in source_text
    assert "launch_fill_many_values_float32" in source_text
    assert "launch_fill_many_values_bf16_bits_float32" in source_text
    assert "fill_many_values_float32_kernel" in kernels_text
    assert "fill_many_values_bf16_bits_float32_kernel" in kernels_text
    assert "block_weight_bf16_initialization_strategy" in gpt2_source_text
    assert "bf16_parameter_initialization_descriptor_count" in gpt2_source_text
    assert "setup_timing" in gpt2_source_text
    assert "post_train_sample_wall_ms" in gpt2_source_text
    assert "post_train_diagnostic_samples_elided = cfg.startup_only || cfg.sample_every_steps <= 0" in gpt2_source_text
    assert "post_train_diagnostic_sample_d2h_count_elided" in gpt2_source_text
    assert "post_train_diagnostic_sample_d2h_count_elided = 2" in gpt2_source_text
    assert "cleanup_wall_ms" in gpt2_source_text
    assert "setup.float_arena_materialize" in gpt2_source_text
    assert "setup.zero_init" in gpt2_source_text
    assert "setup.block_weight_bf16_initial_refresh" in gpt2_source_text
    assert "NFN_NATIVE_GPT_SETUP_EVENT_TIMING" in gpt2_source_text
    assert "setup_cuda_event_timing" in gpt2_source_text
    assert "run_setup_cuda_timed(\"setup.token_weight_init\"" in gpt2_source_text
    assert "--layer-evo" in gpt2_source_text
    assert "--native-cuda-layer-evo" in gpt2_source_text
    assert "layer_evo.graph_editor_tensor_flow" not in gpt2_source_text
    assert "graph_editor_tensor_flow" in gpt2_source_text
    assert "layer_evo.mutate_candidates.ln1_weight" in gpt2_source_text
    assert "layer_evo.select_best_loss" in gpt2_source_text
    assert "layer_evo.adopt_candidate.ln1_weight" in gpt2_source_text
    assert "native-forward-loss-device-resident-current-batch" in gpt2_source_text
    assert "candidate_loss.copy_device_to_device" in gpt2_source_text
    assert "candidate_loss.copy_host_to_device" not in gpt2_source_text
    assert "layer_evo_candidate_loss_host_roundtrips_elided" in gpt2_source_text
    assert "layer_evo_forward_candidate_evals" in gpt2_source_text
    assert "request_layer_evo_workspace();" in gpt2_source_text
    assert "layer_evo.float_workspace_cuda_mallocs_elided" not in gpt2_source_text
    assert '\\"float_workspace_cuda_mallocs_elided\\"' in gpt2_source_text
    assert '\\"workspace_allocation_strategy\\": \\"float-arena-plus-int64-device\\"' in gpt2_source_text
    assert "native_tile_cuda_activation_json" in gpt2_source_text
    assert "requested-nvfp4-not-yet-packed-native-dense-gpt" in gpt2_source_text
    assert "--smoke-nvfp4-pack" in gpt2_source_text
    assert "--native-cuda-smoke-nvfp4-pack" in gpt2_source_text
    assert "print_nvfp4_pack_smoke_json" in gpt2_source_text
    assert "native_nvfp4_pack" in gpt2_source_text
    assert "nonzero_scale_bytes" in gpt2_source_text
    assert '\\"effective_activation_dtype\\"' in gpt2_evo_source_text
    assert '\\"native_activation_packing_active\\"' in gpt2_evo_source_text
    assert "nfn_native_tile_sumsq_partials_many_float32" in header_text
    assert "nfn_native_tile_sumsq_partials_many_bf16_bits_float32" in header_text
    assert "nfn_native_tile_optimizer_tile_size" in header_text
    assert "nfn_native_tile_attention_backward_tk_block_size" in header_text
    assert "NFN_TILE_CUDA_OPTIMIZER_TILE_SIZE" in source_text
    assert "nfn_native_tile_optimizer_tile_size" in gpt2_source_text
    assert "optimizer_tile_strategy" in gpt2_source_text
    assert "launch_sumsq_partials_many_float32" in source_text
    assert "launch_sumsq_partials_many_bf16_bits_float32" in source_text
    assert "sumsq_partials_many_float32_kernel" in kernels_text
    assert "sumsq_partials_many_bf16_bits_float32_kernel" in kernels_text
    assert "nfn_native_tile_fill_many_values_mixed_float32_bf16_bits" in header_text
    assert "nfn_native_tile_fill_many_values_mixed_float32_bf16_bits" in source_text
    assert "fill_many_values_mixed_float32_bf16_bits_kernel" in kernels_text
    assert "FillManyValuesMixedFn" in gpt2_source_text
    assert "mixed-float32-bf16-fill-many-values" in gpt2_source_text
    assert "mixed_parameter_initialization_kernel_launches" in gpt2_source_text
    assert "SumsqPartialsManyBf16BitsFn" in gpt2_source_text
    assert "gradient_clip_bf16_sumsq_kernel_loaded" in gpt2_source_text
    assert "gradient_partial_offsets" in gpt2_source_text
    assert "fused-multi-buffer-fill-values" in gpt2_source_text
    assert "fused-multi-buffer-sumsq-device-scale" in gpt2_source_text
    assert "nfn_native_tile_init_gpt2_token_weight_float32" in header_text
    assert "nfn_native_tile_init_gpt2_token_weight_fast_float32" in header_text
    assert "nfn_native_tile_init_gpt2_token_weight_with_bf16_shadow_float32" in header_text
    assert "nfn_native_tile_init_gpt2_token_weight_fast_with_bf16_shadow_float32" in header_text
    assert "launch_init_gpt2_token_weight_with_bf16_shadow_float32" in source_text
    assert "launch_init_gpt2_token_weight_fast_with_bf16_shadow_float32" in source_text
    assert "init_gpt2_token_weight_with_bf16_shadow_float32_kernel" in kernels_text
    assert "init_gpt2_token_weight_fast_with_bf16_shadow_float32_kernel" in kernels_text
    assert "init_gpt2_token_weight_threaded_with_bf16_shadow_float32_kernel" in kernels_text
    assert "nfn_native_tile_copy_float32" in header_text
    assert "nfn_native_tile_uint16_to_int64" in header_text
    assert "nfn_native_tile_float32_to_bf16_bits" in header_text
    assert "nfn_native_tile_bf16_bits_to_float32" in header_text
    assert "nfn_native_tile_float32_to_nvfp4_packed" in header_text
    assert "nfn_native_tile_nvfp4_packed_to_float32" in header_text
    assert "nfn_native_tile_float32_to_nvfp4_packed" in gpt2_source_text
    assert "nfn_native_tile_nvfp4_packed_to_float32" in gpt2_source_text
    assert "nfn_native_tile_bf16_bits_add_bias_inplace_float32" in header_text
    assert "nfn_native_tile_store_mlp_activations_bf16_float32" in header_text
    assert "nfn_native_tile_restore_mlp_activations_bf16_float32" in header_text
    assert "nfn_native_tile_linear_weight_bf16_float32" in header_text
    assert "nfn_native_tile_linear_weight_bf16_output_float32" in header_text
    assert "nfn_native_tile_linear_bf16_input_float_weight_bf16_output_float32" in header_text
    assert "nfn_native_tile_linear_bf16_input_bits_float32" in header_text
    assert "nfn_native_tile_linear_bf16_input_weight_bf16_float32" in header_text
    assert "nfn_native_tile_linear_bf16_gelu_bf16_float32" in header_text
    assert "nfn_native_tile_linear_weight_bf16_gelu_bf16_float32" in header_text
    assert "nfn_native_tile_linear_bf16_input_weight_bf16_gelu_bf16_float32" in header_text
    assert "nfn_native_tile_gelu_add_bias_bf16_act_float32" in header_text
    assert "nfn_native_tile_dropout_forward_float32" in header_text
    assert "nfn_native_tile_dropout_backward_float32" in header_text
    assert "nfn_native_tile_linear_backward_weight_accumulate_bf16_bits_float32" in header_text
    assert "nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_float32" in header_text
    assert "nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_bits_float32" in header_text
    assert "nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_bits_float32_beta" in header_text
    assert "nfn_native_tile_linear_bf16_output_float32" in header_text
    assert "nfn_native_tile_linear_backward_input_bf16_bits_float32" in header_text
    assert "nfn_native_tile_linear_backward_input_weight_bf16_float32" in header_text
    assert "nfn_native_tile_linear_backward_weight_accumulate_float32_bf16_bits" in header_text
    assert "nfn_native_tile_linear_backward_weight_accumulate_bf16_bits_bf16_bits_float32" in header_text
    assert "nfn_native_tile_linear_backward_weight_accumulate_bf16_bits_bf16_bits_float32_beta" in header_text
    assert "nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_bits_bf16_bits_float32_beta" in header_text
    assert "nfn_native_tile_linear_backward_weight_bias_accumulate_float32_bf16_bits_beta" in header_text
    assert "NFN_NATIVE_GPT_LM_HEAD_BF16_DWEIGHT" in gpt2_source_text
    assert "lm_head_bf16_dweight_enabled" in gpt2_source_text
    assert "NFN_NATIVE_GPT_LM_HEAD_PREPACK_BF16_HIDDEN" in gpt2_source_text
    assert "lm_head_prepack_bf16_hidden_enabled" in gpt2_source_text
    assert (
        'env_or_empty_any({"NFN_NATIVE_GPT_LM_HEAD_PREPACK_BF16_HIDDEN",\n'
        '                              "NFN_NATIVE_GPT2_LM_HEAD_PREPACK_BF16_HIDDEN"}),\n'
        "            true);"
    ) in gpt2_source_text
    assert "NFN_NATIVE_GPT_REUSE_FORWARD_LM_HEAD_LOGITS" in gpt2_source_text
    assert "NFN_NATIVE_GPT2_REUSE_FORWARD_LM_HEAD_LOGITS" in gpt2_source_text
    assert "lm_head_reuse_forward_logits_enabled" in gpt2_source_text
    assert "NFN_NATIVE_GPT_FULL_BATCH_LM_HEAD_REUSE" in gpt2_source_text
    assert "NFN_NATIVE_GPT2_FULL_BATCH_LM_HEAD_REUSE" in gpt2_source_text
    assert "lm_head_full_batch_reuse_schedule_enabled" in gpt2_source_text
    assert "resident-full-logit-single-row-batch-gemms" in gpt2_source_text
    assert "lm_head_forward_logits_for_backward" in gpt2_source_text
    assert "lm_head_full_logit_elements" in gpt2_source_text
    assert "lm_head_dweight_strategy" in gpt2_source_text
    assert "full-final-norm-bf16-prepack-bf16-dlogit-dweight-accumulate" in gpt2_source_text
    assert "full-final-norm-bf16-prepack-bf16-dlogit-dweight-first-write-then-accumulate" in gpt2_source_text
    assert "NFN_NATIVE_GPT_LM_HEAD_BF16_HIDDEN_FROM_FINAL_NORM" in gpt2_source_text
    assert "lm_head_bf16_hidden_from_final_norm_requested" in gpt2_source_text
    assert "lm_head_bf16_hidden_from_final_norm_enabled" in gpt2_source_text
    assert "final-norm-direct-bf16-hidden-bf16-dlogit-dweight-accumulate" in gpt2_source_text
    assert "dweight_first_microbatch_beta_zero_enabled" in gpt2_source_text
    assert "first_lm_head_dweight_chunk" in gpt2_source_text
    assert "chunk_index == 0 && dweight_first_microbatch_beta_zero_enabled && !dweight_accumulate" in gpt2_source_text
    assert "lm_head_dweight_beta_zero_scope" in gpt2_source_text
    assert "NFN_NATIVE_GPT_LM_HEAD_DWEIGHT_BEFORE_DHIDDEN" in gpt2_source_text
    assert "lm_head_dweight_before_dhidden_enabled" in gpt2_source_text
    assert "NFN_NATIVE_GPT_LM_HEAD_CONCURRENT_DHIDDEN_DWEIGHT" in gpt2_source_text
    assert "NFN_NATIVE_GPT2_LM_HEAD_CONCURRENT_DHIDDEN_DWEIGHT" in gpt2_source_text
    assert "lm_head_concurrent_dhidden_dweight_enabled" in gpt2_source_text
    assert "two-nonblocking-cuda-streams-after-ce-event" in gpt2_source_text
    assert "CUDA 13.3 RTX 5090 3-sample same-script confirmation" in candidate_bench_text
    assert "NFN_NATIVE_GPT_LM_HEAD_CONCURRENT_DHIDDEN_DWEIGHT=1" in candidate_bench_text
    assert "NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_BACKWARD=0 NFN_NATIVE_GPT_LM_HEAD_CONCURRENT_DHIDDEN_DWEIGHT=1" in candidate_bench_text
    assert "serial dWeight-before-dHidden schedule" in candidate_bench_text
    assert "NFN_NATIVE_GPT_LM_HEAD_DWEIGHT_BEFORE_DHIDDEN=1" in candidate_bench_text
    assert "NFN_NATIVE_GPT_LM_HEAD_COOPERATIVE_BACKWARD=0 NFN_NATIVE_GPT_LM_HEAD_DWEIGHT_BEFORE_DHIDDEN=1" in candidate_bench_text
    assert "NFN_NATIVE_GPT_LM_HEAD_REVERSE_CHUNKS" in gpt2_source_text
    assert "lm_head_reverse_chunk_order_enabled" in gpt2_source_text
    assert "reverse-row-chunk-order-default-cuda-13-3-rtx-5090" in gpt2_source_text
    assert "lm_head_backward.hidden_prepack" in gpt2_source_text
    assert "launch_linear_bf16_input_float_weight_bf16_output_float32" in source_text
    assert "linear_bf16_input_float_weight_bf16_output_float32_kernel" in kernels_text
    assert "cublas_linear_gemm_ex_float32_a_bf16_bits_b_to_bf16_bits" in kernels_text
    assert "nfn_native_tile_gelu_backward_inplace_bf16_bits_float32" in header_text
    assert "dropout_forward_float32_kernel" in kernels_text
    assert "dropout_backward_float32_kernel" in kernels_text
    assert "nfn_native_tile_float32_to_bf16_bits_many" in header_text
    assert "nfn_native_tile_trainer_linear_stats_reset" in header_text
    assert "nfn_native_tile_trainer_linear_bf16_cache_reset" in header_text
    assert "nfn_native_tile_trainer_linear_bf16_gemm_count" in header_text
    assert "nfn_native_tile_trainer_linear_bf16_gemm_fast16bf_request_count" in header_text
    assert "nfn_native_tile_trainer_linear_tk_gemm_count" in header_text
    assert "nfn_native_tile_trainer_linear_tk_float_out_gemm_count" in header_text
    assert "nfn_native_tile_trainer_linear_tk_dweight_gemm_count" in header_text
    assert "nfn_native_tile_trainer_linear_tk_dgelu_dinput_gemm_count" in header_text
    assert "nfn_native_tile_trainer_linear_tk_sm120_k_tile" in header_text
    assert "nfn_native_tile_trainer_linear_tk_sm120_grad_k_tile" in header_text
    assert "nfn_native_tile_trainer_linear_tk_sm120_super_m" in header_text
    assert "nfn_native_tile_trainer_linear_tk_sm120_dinput_super_m" in header_text
    assert "nfn_native_tile_trainer_linear_tk_sm120_dweight_super_m" in header_text
    assert "nfn_native_tile_trainer_linear_tk_sm120_huge_n_k_tile" in header_text
    assert "nfn_native_tile_trainer_linear_tk_sm120_fast_dgelu_enabled" in header_text
    assert "nfn_native_tile_trainer_linear_tk_sm120_approx_dgelu_tanh_enabled" in header_text
    assert "nfn_native_tile_trainer_linear_cublaslt_gemm_count" in header_text
    assert "nfn_native_tile_trainer_linear_cublaslt_bgrad_gemm_count" in header_text
    assert "nfn_native_tile_trainer_linear_cublaslt_bgrad_direct_write_count" in header_text
    assert "nfn_native_tile_trainer_linear_cublaslt_bgrad_accumulate_count" in header_text
    assert "nfn_native_tile_trainer_linear_sgemm_count" in header_text
    assert "nfn_native_tile_trainer_bf16_to_f32_vec4_count" in header_text
    assert "nfn_native_tile_trainer_linear_bf16_a_pack_count" in header_text
    assert "nfn_native_tile_trainer_linear_bf16_a_cache_hit_count" in header_text
    assert "nfn_native_tile_trainer_linear_bf16_cache_reset_count" in header_text
    assert "nfn_native_tile_trainer_linear_cublas_grouped_bf16_gemm_probe_status" in header_text
    assert "nfn_native_tile_trainer_linear_cublas_prewarm" in header_text
    assert "launch_float32_to_bf16_bits" in source_text
    assert "launch_bf16_bits_to_float32" in source_text
    assert "launch_float32_to_nvfp4_packed" in source_text
    assert "launch_nvfp4_packed_to_float32" in source_text
    assert "launch_bf16_bits_add_bias_inplace_float32" in source_text
    assert "launch_store_mlp_activations_bf16_float32" in source_text
    assert "launch_restore_mlp_activations_bf16_float32" in source_text
    assert "launch_linear_weight_bf16_float32" in source_text
    assert "launch_linear_weight_bf16_output_float32" in source_text
    assert "launch_linear_bf16_input_bits_float32" in source_text
    assert "launch_linear_bf16_input_weight_bf16_float32" in source_text
    assert "launch_linear_bf16_gelu_bf16_float32" in source_text
    assert "launch_linear_weight_bf16_gelu_bf16_float32" in source_text
    assert "launch_linear_bf16_input_weight_bf16_gelu_bf16_float32" in source_text
    assert "launch_gelu_add_bias_bf16_act_float32" in source_text
    assert "launch_linear_backward_weight_accumulate_bf16_bits_float32" in source_text
    assert "launch_linear_backward_weight_bias_accumulate_bf16_float32" in source_text
    assert "launch_linear_backward_weight_bias_accumulate_bf16_bits_float32" in source_text
    assert "launch_linear_bf16_output_float32" in source_text
    assert "launch_linear_backward_input_bf16_bits_float32" in source_text
    assert "launch_linear_backward_input_weight_bf16_float32" in source_text
    assert "launch_linear_backward_weight_accumulate_float32_bf16_bits" in source_text
    assert "launch_linear_backward_weight_accumulate_bf16_bits_bf16_bits_float32" in source_text
    assert "launch_linear_backward_weight_accumulate_bf16_bits_bf16_bits_float32_beta" in source_text
    assert "launch_linear_backward_weight_bias_accumulate_bf16_bits_float32_beta" in source_text
    assert "launch_gelu_backward_inplace_bf16_bits_float32" in source_text
    assert "launch_float32_to_bf16_bits_many" in source_text
    assert "nfn_native_tile_trainer_linear_bf16_workspace_allocation_count" in source_text
    assert "nfn_native_tile_trainer_linear_bf16_cached_a_capacity" in source_text
    assert "nfn_native_tile_trainer_linear_bf16_cache_entry_count" in source_text
    assert "nfn_native_tile_trainer_linear_cublas_grouped_bf16_gemm_probe_status" in source_text
    assert "f32_to_bf16_bits_kernel" in kernels_text
    assert "f32_to_bf16_bits_vec4_kernel" in kernels_text
    assert "float32_to_nvfp4_packed_kernel" in kernels_text
    assert "nvfp4_packed_to_float32_kernel" in kernels_text
    assert "nvfp4_float_to_e4m3fn_device" in kernels_text
    assert "nvfp4_float_to_e2m1_code_device" in kernels_text
    assert "NFN_TILE_CUDA_F32_TO_BF16_VEC4" in kernels_text
    assert "NFN_NATIVE_GPT_F32_TO_BF16_VEC4" in kernels_text
    assert "NFN_NATIVE_GPT2_F32_TO_BF16_VEC4" in kernels_text
    assert "bf16_bits_to_f32_kernel" in kernels_text
    assert "bf16_bits_to_f32_vec4_kernel" in kernels_text
    assert "NFN_TILE_CUDA_BF16_TO_F32_VEC4" in kernels_text
    assert "NFN_NATIVE_GPT_BF16_TO_F32_VEC4" in kernels_text
    assert "NFN_NATIVE_GPT2_BF16_TO_F32_VEC4" in kernels_text
    assert "g_bf16_to_f32_vec4_count" in kernels_text
    assert "trainer_bf16_to_f32_vec4_count" in kernels_text
    assert "f32_to_bf16_bits_many_kernel" in kernels_text
    assert "f32_to_bf16_bits_many_vec4_kernel" in kernels_text
    assert "NFN_TILE_CUDA_F32_TO_BF16_MANY_VEC4" in kernels_text
    assert "NFN_NATIVE_GPT_F32_TO_BF16_MANY_VEC4" in kernels_text
    assert "NFN_NATIVE_GPT2_F32_TO_BF16_MANY_VEC4" in kernels_text
    assert "store_mlp_activations_bf16_float32_vec4_kernel" in kernels_text
    assert "restore_mlp_activations_bf16_float32_vec4_kernel" in kernels_text
    assert "NFN_TILE_CUDA_STORE_MLP_ACTIVATIONS_VEC4" in kernels_text
    assert "NFN_NATIVE_GPT_STORE_MLP_ACTIVATIONS_VEC4" in kernels_text
    assert "NFN_NATIVE_GPT2_STORE_MLP_ACTIVATIONS_VEC4" in kernels_text
    assert "NFN_NATIVE_GPT_TOKEN_WEIGHT_VECTOR4_INIT" in gpt2_source_text
    assert "NFN_TILE_CUDA_TOKEN_WEIGHT_VECTOR4_INIT" in kernels_text
    assert "return true;" in kernels_text[kernels_text.index("bool token_weight_vector4_init_enabled()") :]
    assert "bf16_bits_add_bias_inplace_kernel" in kernels_text
    assert "bf16_bits_add_bias_inplace_tile_float32_kernel" in kernels_text
    assert "launch_linear_bf16_float32" in kernels_text
    assert "cublas_linear_gemm_ex_bf16_float32_to_bf16_bits" in kernels_text
    assert "cublas_linear_gemm_ex_bf16_bits_a_float32_to_bf16_bits" in kernels_text
    assert "cublas_linear_gemm_ex_bf16_bits_ab_float32" in kernels_text
    assert "Cache the stable weight operand only; activation pointers are reused with new contents." in kernels_text
    assert "tk_linear_gemm_bf16_forward_to_bf16_bits" in kernels_text
    assert "tk_linear_gemm_bf16_forward_to_float32" in kernels_text
    assert "tk_linear_gemm_bf16_forward_gelu_to_bf16_bits" in kernels_text
    assert "tk_linear_gemm_bf16_forward_gelu_weight_bf16_to_bf16_bits" in kernels_text
    assert "linear_bf16_input_weight_bf16_gelu_bf16_float32_kernel" in kernels_text

    assert "cublas_linear_gemm_ex_bf16_bits_b_float32" in kernels_text
    assert "linear_bf16_output_float32_kernel" in kernels_text
    assert "linear_weight_bf16_output_float32_kernel" in kernels_text
    assert "linear_weight_bf16_bits_float32_kernel" in kernels_text
    assert "linear_bf16_input_bits_float32_kernel" in kernels_text
    assert "linear_bf16_input_weight_bf16_bits_float32_kernel" in kernels_text
    assert "linear_bf16_gelu_bf16_float32_kernel" in kernels_text
    assert "linear_weight_bf16_gelu_bf16_float32_kernel" in kernels_text
    assert "launch_linear_bf16_input_weight_bf16_gelu_bf16_float32" in kernels_text
    assert "gelu_add_bias_bf16_act_float32_kernel" in kernels_text
    assert "__tile_global__ void gelu_add_bias_bf16_act_float32_kernel" in kernels_text
    assert "gelu_add_bias_bf16_act_float32_kernel<<<blocks, 1, 0, stream>>>" in kernels_text
    assert "ct::element_cast<__nv_bfloat16>(result)" in kernels_text
    assert "__tile_global__ void gelu_backward_inplace_bf16_bits_float32_kernel" in kernels_text
    assert "gelu_backward_inplace_bf16_bits_float32_kernel<<<blocks, 1, 0, stream>>>" in kernels_text
    assert "fused-gelu-bf16-act-direct-gemm" in gpt2_source_text
    assert ".mlp.proj.forward.no_bias.bf16_act" in gpt2_source_text
    assert ".mlp.bias_gelu.forward.bf16_act" in gpt2_source_text
    assert "nfn_native_tile_linear_bf16_float32" in header_text
    assert "nfn_native_tile_linear_bf16_float32" in source_text
    assert "launch_linear_backward_input_bf16_float32" in kernels_text
    assert "linear_backward_input_bf16_bits_float32_kernel" in kernels_text
    assert "nfn_native_tile_linear_backward_input_bf16_float32" in header_text
    assert "nfn_native_tile_linear_backward_input_bf16_float32" in source_text
    assert "bf16-shadow-weight-shape-gated-cublaslt-forward" in gpt2_source_text
    assert "persistent-fp32-master-bf16-shadow-refresh-after-adamw" in gpt2_source_text
    assert "NFN_NATIVE_GPT_BF16_BLOCK_WEIGHT_PARAMS" in gpt2_source_text
    assert "persistent-bf16-primary-block-weight-adamw" in gpt2_source_text
    assert "split-float32-and-bf16-param-multi-buffer-device-scale" in gpt2_source_text
    assert "adamw_many_with_device_scale_bf16_param" in gpt2_source_text
    assert "sync_bf16_param_to_fp32_checkpoint" in gpt2_source_text
    assert "launch_linear_backward_weight_accumulate_bf16_float32" in kernels_text
    assert "linear_backward_weight_tiled_float32_kernel" in kernels_text
    assert "launch_linear_backward_weight_tiled_float32_fallback" in kernels_text
    assert "nfn_native_tile_linear_backward_weight_accumulate_bf16_float32" in header_text
    assert "nfn_native_tile_linear_backward_weight_accumulate_bf16_float32" in source_text
    assert "cublas_linear_gemm_ex_bf16_float32" in kernels_text
    assert "CUDA_R_16BF" in kernels_text
    assert "CUBLAS_COMPUTE_32F" in kernels_text
    assert "cublasLtMatmul" in kernels_text
    assert "CUBLASLT_EPILOGUE_BGRADB" in kernels_text
    assert "CUBLASLT_MATMUL_DESC_BIAS_POINTER" in kernels_text
    assert "NFN_TILE_CUDA_LINEAR_BF16_GEMM_EX_FAST_16BF" in kernels_text
    assert "NFN_NATIVE_LINEAR_BF16_GEMM_EX_FAST_16BF" in kernels_text
    assert "trainer_linear_bf16_gemm_ex_compute_type" in kernels_text
    assert "NFN_TILE_CUDA_LINEAR_BF16_GEMM_EX_ALGO" in kernels_text
    assert "NFN_NATIVE_LINEAR_BF16_GEMM_EX_ALGO" in kernels_text
    assert "NFN_TILE_CUDA_LINEAR_BF16_GEMM_EX_ALGO_SHAPE" in kernels_text
    assert "NFN_NATIVE_LINEAR_BF16_GEMM_EX_ALGO_SHAPE" in kernels_text
    assert "parse_bf16_gemm_ex_algo_token" in kernels_text
    assert "trainer_linear_bf16_gemm_ex_algo" in kernels_text
    assert '"%d,%d,%d,%7[^,],%7[^,],%31s"' in kernels_text
    assert "CUBLAS_GEMM_ALGO0_TENSOR_OP + parsed" in kernels_text
    assert "cublas_linear_gemm_ex_bf16_float32_with_bgrad" in kernels_text
    assert "cublas_linear_gemm_ex_bf16_bits_a_float32_with_bgrad" in kernels_text
    assert "cublas_linear_gemm_ex_bf16_bits_b_float32_with_bgrad" in kernels_text
    assert "trainer_linear_bgrad_first_write_direct_enabled" in kernels_text
    assert "trainer_linear_bgrad_first_write_direct_shape_enabled" in kernels_text
    assert "NFN_NATIVE_GPT_BGRAD_FIRST_WRITE_DIRECT" in kernels_text
    assert "NFN_NATIVE_GPT2_BGRAD_FIRST_WRITE_DIRECT" in kernels_text
    assert "NFN_TILE_CUDA_LINEAR_BGRAD_FIRST_WRITE_DIRECT" in kernels_text
    assert "NFN_NATIVE_GPT_BGRAD_FIRST_WRITE_DIRECT_ENABLE_SHAPE" in kernels_text
    assert "NFN_NATIVE_GPT2_BGRAD_FIRST_WRITE_DIRECT_ENABLE_SHAPE" in kernels_text
    assert "NFN_NATIVE_LINEAR_BGRAD_FIRST_WRITE_DIRECT_ENABLE_SHAPE" in kernels_text
    assert "NFN_TILE_CUDA_LINEAR_BGRAD_FIRST_WRITE_DIRECT_ENABLE_SHAPE" in kernels_text
    assert "first_write_bias ? grad_bias : ensure_trainer_linear_bgrad_workspace(output_dim)" in kernels_text
    assert "if (!first_write_bias)" in kernels_text
    assert "bgrad_first_write_direct_enabled" in gpt2_source_text
    assert "linear_bias_gradient_first_write_bgrad_direct_enabled" in gpt2_source_text
    assert "NFN_NATIVE_GPT_FUSE_FLOAT32_BF16_DWEIGHT_BGRAD" in kernels_text
    assert "NFN_TILE_CUDA_LINEAR_FLOAT32_BF16_BGRAD" in kernels_text
    assert "trainer_linear_bf16_bf16_bgrad_disabled_for_shape" in kernels_text
    assert "NFN_TILE_CUDA_LINEAR_BF16_BF16_BGRAD_DISABLE_SHAPE" in kernels_text
    assert "NFN_NATIVE_LINEAR_BF16_BF16_BGRAD_DISABLE_SHAPE" in kernels_text
    assert (
        "!trainer_linear_bf16_bf16_bgrad_enabled() ||\n"
        "         trainer_linear_bf16_bf16_bgrad_disabled_for_shape"
    ) in kernels_text
    assert "cross_entropy_bf16_exp2_enabled" in kernels_text
    assert "cross_entropy_exp_device" in kernels_text
    assert "NFN_NATIVE_GPT_CE_BF16_EXP2" in kernels_text
    assert "NFN_NATIVE_GPT2_CE_BF16_EXP2" in kernels_text
    assert "NFN_TILE_CUDA_CE_BF16_EXP2" in kernels_text
    assert "lm_head_ce_bf16_exp2_enabled" in gpt2_source_text
    assert "CUBLAS_COMPUTE_32F_FAST_TF32" in kernels_text
    assert "CUBLAS_COMPUTE_32F_FAST_16BF" in kernels_text
    assert "NFN_TILE_CUDA_LINEAR_BF16" in kernels_text
    assert "NFN_NATIVE_LINEAR_BF16" in kernels_text
    assert "NFN_TILE_CUDA_LINEAR_BF16_CUBLASLT" in kernels_text
    assert "NFN_NATIVE_LINEAR_BF16_CUBLASLT" in kernels_text
    assert "NFN_TILE_CUDA_LINEAR_BF16_CUBLASLT_LARGE_SHAPES" in kernels_text
    assert "NFN_NATIVE_LINEAR_BF16_CUBLASLT_LARGE_SHAPES" in kernels_text
    assert "NFN_TILE_CUDA_LINEAR_BF16_CUBLASLT_DISABLE_SHAPE" in kernels_text
    assert "NFN_NATIVE_LINEAR_BF16_CUBLASLT_DISABLE_SHAPE" in kernels_text
    assert "NFN_TILE_CUDA_LINEAR_CUBLASLT" in kernels_text
    assert "NFN_NATIVE_LINEAR_CUBLASLT" in kernels_text
    assert "NFN_TILE_CUDA_LINEAR_TK_GEMM" in kernels_text
    assert "NFN_NATIVE_LINEAR_TK_GEMM" in kernels_text
    assert "NFN_TILE_CUDA_LINEAR_TK_FLOAT_OUT" in kernels_text
    assert "NFN_NATIVE_LINEAR_TK_FLOAT_OUT" in kernels_text
    assert "NFN_TILE_CUDA_LINEAR_SHAPE_STATS" in kernels_text
    assert "NFN_NATIVE_LINEAR_SHAPE_STATS" in kernels_text
    assert "NFN_NATIVE_GPT_LINEAR_SHAPE_STATS" in kernels_text
    assert "NFN_NATIVE_GPT2_LINEAR_SHAPE_STATS" in kernels_text
    assert "NFN_TILE_CUDA_LINEAR_TK_FORWARD_DISABLE_SHAPE" in kernels_text
    assert "NFN_NATIVE_LINEAR_TK_FORWARD_DISABLE_SHAPE" in kernels_text
    assert "NFN_TILE_CUDA_LINEAR_TK_FORWARD_ENABLE_SHAPE" in kernels_text
    assert "NFN_NATIVE_LINEAR_TK_FORWARD_ENABLE_SHAPE" in kernels_text
    assert "token_position_embedding_residual_float32_kernel" in kernels_text
    assert "token_position_embedding_residual_u16_float32_kernel" in kernels_text
    assert "nfn_native_tile_token_position_embedding_residual_float32" in gpt2_source_text
    assert "nfn_native_tile_token_position_embedding_residual_u16_float32" in gpt2_source_text
    assert "NFN_NATIVE_GPT_FUSE_EMBEDDING_RESIDUAL" in gpt2_source_text
    assert "NFN_NATIVE_GPT2_FUSE_EMBEDDING_RESIDUAL" in gpt2_source_text
    assert "NFN_TILE_CUDA_FUSE_EMBEDDING_RESIDUAL" in gpt2_source_text
    assert "embedding_residual_fusion_enabled" in gpt2_source_text
    assert "embedding_residual_intermediate_float_buffers_elided" in gpt2_source_text
    assert "fuse_embedding_residual_enabled ? 0 : activation_elements" in gpt2_source_text
    assert "default_disabled_lm_head_logits_shape" not in kernels_text
    bf16_output_forward = kernels_text[
        kernels_text.index("void launch_linear_bf16_input_weight_bf16_output_float32") :
        kernels_text.index("void launch_linear_backward_input_bf16_bits_float32")
    ]
    assert "tk_linear_gemm_bf16_forward_to_bf16_bits" in bf16_output_forward
    assert "cublas_linear_gemm_ex_bf16_bits_ab_to_bf16_bits" in bf16_output_forward
    assert bf16_output_forward.index("tk_linear_gemm_bf16_forward_to_bf16_bits") < bf16_output_forward.index(
        "cublas_linear_gemm_ex_bf16_bits_ab_to_bf16_bits"
    )
    assert "NFN_TILE_CUDA_CE_BF16_THREADS" in kernels_text
    assert "NFN_NATIVE_GPT_CE_BF16_THREADS" in kernels_text
    assert "NFN_NATIVE_GPT2_CE_BF16_THREADS" in kernels_text
    assert "cross_entropy_bf16_threads_per_row" in kernels_text
    assert "token_cross_entropy_bf16_threads_per_row" in kernels_text
    assert "nfn_native_tile_token_cross_entropy_bf16_threads_per_row" in header_text
    assert "nfn_native_tile_token_cross_entropy_bf16_threads_per_row" in source_text
    assert "token_cross_entropy_bf16_threads_per_row_fn" in gpt2_source_text
    assert "lm_head_ce_bf16_threads_per_row" in gpt2_source_text
    assert "nfn_native_tile_lm_head_true_fused_mat_tile" in gpt2_source_text
    assert "nfn_native_tile_lm_head_true_fused_required_threads" in gpt2_source_text
    assert "lm_head_true_fused_mat_tile" in gpt2_source_text
    assert "lm_head_true_fused_required_threads" in gpt2_source_text
    assert "NFN_TILE_CUDA_CE_BF16_VEC_STORES" in kernels_text
    assert "NFN_NATIVE_GPT_CE_BF16_VEC_STORES" in kernels_text
    assert "NFN_NATIVE_GPT2_CE_BF16_VEC_STORES" in kernels_text
    assert "cross_entropy_bf16_vec_stores_enabled" in kernels_text
    assert "NFN_TILE_CUDA_CE_BF16_VEC_NORMAL_STORES" in kernels_text
    assert "NFN_NATIVE_GPT_CE_BF16_VEC_NORMAL_STORES" in kernels_text
    assert "NFN_NATIVE_GPT2_CE_BF16_VEC_NORMAL_STORES" in kernels_text
    assert "cross_entropy_bf16_vec_normal_stores_enabled" in kernels_text
    assert "NFN_TILE_CUDA_CE_BF16_VEC_LOADS" in kernels_text
    assert "NFN_NATIVE_GPT_CE_BF16_VEC_LOADS" in kernels_text
    assert "NFN_NATIVE_GPT2_CE_BF16_VEC_LOADS" in kernels_text
    assert "cross_entropy_bf16_vec_loads_enabled" in kernels_text
    assert "NFN_TILE_CUDA_CE_BF16_SCALAR_STREAMING_STORES" in kernels_text
    assert "NFN_NATIVE_GPT_CE_BF16_SCALAR_STREAMING_STORES" in kernels_text
    assert "NFN_NATIVE_GPT2_CE_BF16_SCALAR_STREAMING_STORES" in kernels_text
    assert "cross_entropy_bf16_scalar_streaming_stores_enabled" in kernels_text
    assert "lm_head_ce_bf16_scalar_streaming_stores_enabled" in gpt2_source_text
    assert "NFN_TILE_CUDA_LM_HEAD_CE_DEFAULT_SPECIALIZED" in kernels_text
    assert "NFN_NATIVE_GPT_LM_HEAD_CE_DEFAULT_SPECIALIZED" in kernels_text
    assert "NFN_NATIVE_GPT2_LM_HEAD_CE_DEFAULT_SPECIALIZED" in kernels_text
    assert "NFN_TILE_CUDA_LM_HEAD_CE_LLMK_STYLE_SPECIALIZED" in kernels_text
    assert "NFN_NATIVE_GPT_LM_HEAD_CE_LLMK_STYLE_SPECIALIZED" in kernels_text
    assert "NFN_NATIVE_GPT2_LM_HEAD_CE_LLMK_STYLE_SPECIALIZED" in kernels_text
    assert "NFN_TILE_CUDA_LM_HEAD_CE_NO_LOSS_LLMK_STYLE_SPECIALIZED" in kernels_text
    assert "NFN_NATIVE_GPT_LM_HEAD_CE_NO_LOSS_LLMK_STYLE_SPECIALIZED" in kernels_text
    assert "NFN_NATIVE_GPT2_LM_HEAD_CE_NO_LOSS_LLMK_STYLE_SPECIALIZED" in kernels_text
    assert "NFN_TILE_CUDA_LM_HEAD_CE_NO_LOSS_VEC8_NORMAL_STORE_SPECIALIZED" in kernels_text
    assert "NFN_NATIVE_GPT_LM_HEAD_CE_NO_LOSS_VEC8_NORMAL_STORE_SPECIALIZED" in kernels_text
    assert "NFN_NATIVE_GPT2_LM_HEAD_CE_NO_LOSS_VEC8_NORMAL_STORE_SPECIALIZED" in kernels_text
    assert "lm_head_ce_no_loss_vec8_normal_store_specialized_enabled" in kernels_text
    no_loss_normal_store_body = kernels_text.split(
        "bool lm_head_ce_no_loss_vec8_normal_store_specialized_enabled()", 1
    )[1].split("bool lm_head_ce_llmk_style_specialized_enabled()", 1)[0]
    assert "if (raw == nullptr) {\n      return true;\n    }" in no_loss_normal_store_body
    assert "NFN_TILE_CUDA_LM_HEAD_PROB_ONLY_TARGET_CORRECTION_THREADS" in kernels_text
    assert "lm_head_prob_only_target_correction_threads()" in kernels_text
    prob_only_thread_body = kernels_text.split(
        "int lm_head_prob_only_target_correction_threads_value()", 1
    )[1].split("constexpr std::int64_t kLinearBackwardBiasRowChunkSize", 1)[0]
    assert "return 512;" in prob_only_thread_body
    linear_bias_thread_body = kernels_text.split(
        "int linear_backward_bias_threads_per_block_value()", 1
    )[1].split("std::int64_t layer_norm_affine_row_chunk_size()", 1)[0]
    assert "return 512;" in linear_bias_thread_body
    prob_only_body = kernels_text.split(
        "token_cross_entropy_backward_inplace_strided_no_pad_zero_bf16_bits_u16_targets_prob_only_kernel",
        1,
    )[1].split(
        "__global__ void lm_head_classifier_backward_prob_only_ce_target_correction_bf16_bits_kernel",
        1,
    )[0]
    assert "load_bf16_vec8(row_logits + col)" in prob_only_body
    assert "store_bf16_vec8_normal(" in prob_only_body
    ce_target_body = kernels_text.split(
        "__global__ void lm_head_classifier_backward_prob_only_ce_target_correction_bf16_bits_kernel",
        1,
    )[1].split(
        "__global__ void token_cross_entropy_backward_inplace_strided_no_pad_zero_bf16_bits_u16_targets_llmk_style_kernel",
        1,
    )[0]
    assert "store_bf16_vec8_normal(" in ce_target_body
    assert "grad_hidden[row_dim] -= loss_scale * weight" in ce_target_body
    assert "atomicAdd(" in ce_target_body
    assert "NFN_TILE_CUDA_LM_HEAD_CE_LOSS_BINS_DEFAULT_SPECIALIZED" in kernels_text
    assert "NFN_NATIVE_GPT_LM_HEAD_CE_LOSS_BINS_DEFAULT_SPECIALIZED" in kernels_text
    assert "NFN_NATIVE_GPT2_LM_HEAD_CE_LOSS_BINS_DEFAULT_SPECIALIZED" in kernels_text
    assert "lm_head_ce_default_specialized_enabled" in gpt2_source_text
    assert "lm_head_ce_no_loss_default_specialized_enabled" in gpt2_source_text
    assert "NFN_NATIVE_GPT_LM_HEAD_CE_NO_LOSS_DEFAULT_SPECIALIZED" in gpt2_source_text
    assert "lm_head_ce_no_loss_vec8_normal_store_specialized_enabled" in gpt2_source_text
    assert "NFN_NATIVE_GPT_LM_HEAD_CE_NO_LOSS_VEC8_NORMAL_STORE_SPECIALIZED" in gpt2_source_text
    assert (
        'env_or_empty_any({"NFN_NATIVE_GPT_LM_HEAD_CE_NO_LOSS_VEC8_NORMAL_STORE_SPECIALIZED",\n'
        '                              "NFN_NATIVE_GPT2_LM_HEAD_CE_NO_LOSS_VEC8_NORMAL_STORE_SPECIALIZED",\n'
        '                              "NFN_TILE_CUDA_LM_HEAD_CE_NO_LOSS_VEC8_NORMAL_STORE_SPECIALIZED"}),\n'
        "            true);"
    ) in gpt2_source_text
    assert "lm_head_ce_no_loss_llmk_style_specialized_enabled" in gpt2_source_text
    assert "lm_head_ce_llmk_style_specialized_enabled" in gpt2_source_text
    assert "lm_head_ce_loss_bins_default_specialized_enabled" in gpt2_source_text
    assert "lm_head_ce_loss_bin_reduction_runtime_enabled" in gpt2_source_text
    assert "lm_head_classifier_loss_bin_launch_count > 0" in gpt2_source_text
    assert (
        "lm_head_ce_loss_bins_default_specialized_requested &&\n"
        "        lm_head_loss_bin_reduction_requested &&"
    ) in gpt2_source_text
    assert (
        'env_or_empty_any({"NFN_NATIVE_GPT_LM_HEAD_LOSS_BIN_REDUCTION",\n'
        '                              "NFN_NATIVE_GPT2_LM_HEAD_LOSS_BIN_REDUCTION"}),\n'
        "            true);"
    ) in gpt2_source_text
    assert "lm_head_ce_kernel_strategy" in gpt2_source_text
    assert "lm_head_classifier_fusion_scope" in gpt2_source_text
    assert "ce-dlogits-only-logits-dhidden-dweight-remain-separate" in gpt2_source_text
    assert "lm_head_schedule_parity_status" in gpt2_source_text
    assert "reference-parity-separate-logits-ce-dhidden-dweight" in gpt2_source_text
    assert "non-reference-cooperative-lm-head-schedule-enabled" in gpt2_source_text
    assert "default-specialized-row-loss-vec8-loads-scalar-stores" in gpt2_source_text
    assert "default-specialized-loss-bins-vec8-loads-scalar-stores" in gpt2_source_text
    assert "no-loss-specialized-dlogits-vec8-loads-normal-vec8-stores" in gpt2_source_text
    assert "no-loss-llmk-style-dlogits-vec8-loads-streaming-vec8-stores" in gpt2_source_text
    assert "llmk-style-row-loss-vec8-loads-streaming-vec8-stores" in gpt2_source_text
    assert "llmk-style-loss-bins-vec8-loads-streaming-vec8-stores" in gpt2_source_text
    assert "token_cross_entropy_backward_inplace_strided_no_pad_zero_bf16_bits_u16_targets_llmk_style_kernel" in kernels_text
    assert "token_cross_entropy_backward_inplace_strided_no_pad_zero_bf16_bits_u16_targets_vec8_normal_store_kernel" in kernels_text
    assert "lm_head_classifier_backward_loss_bins_default_bf16_bits_u16_targets_kernel" in kernels_text
    assert "lm_head_classifier_backward_row_losses_llmk_style_bf16_bits_u16_targets_kernel" in kernels_text
    assert "lm_head_classifier_backward_loss_bins_llmk_style_bf16_bits_u16_targets_kernel" in kernels_text
    assert "store_bf16_vec8_streaming(" in kernels_text
    assert "vec8-loads-scalar-streaming-stores" in gpt2_source_text
    assert "bf16_row_max_vec8_or_scalar" in kernels_text
    assert "bf16_row_exp_sum_vec8_or_scalar" in kernels_text
    assert "if (vec_loads) {" in kernels_text
    assert "const int4 packed = load_bf16_vec8(row_logits + col);" in kernels_text
    assert (
        "const int4 packed = vec_loads ? load_bf16_vec8(row_logits + col) : make_int4(0, 0, 0, 0);"
        in kernels_text
    )
    assert "(vec_normal_stores && vec_loads) ? load_bf16_vec8" not in kernels_text
    assert "bf16_bits_to_f32_device(int4_u16_at(packed, offset))" in kernels_text
    assert "store_bf16_vec8_streaming" in kernels_text
    assert "store_bf16_vec8_normal" in kernels_text
    assert "store_bf16_vec8(row_logits + col, grad, vec_stores)" in kernels_text
    assert "store_bf16_scalar(" in kernels_text
    assert "load_bf16_vec8" in kernels_text
    assert "__stcs(reinterpret_cast<int4*>(dst)" in kernels_text
    assert "st.global.cs.u16" in kernels_text
    assert "ensure_llmk_sm120_cublaslt_initialized" in kernels_text
    assert "llmk::cublaslt_sm120::init()" in kernels_text
    assert "record_linear_shape_stat(4, m, n, k, op_a, op_b, elapsed_us)" in kernels_text
    assert "record_linear_shape_stat(3, output_dim, rows, input_dim, op_a, op_b, elapsed_us)" in kernels_text
    assert (
        "record_linear_shape_stat(2, output_dim, rows, input_dim, CUBLAS_OP_T, CUBLAS_OP_N, elapsed_us)"
        in kernels_text
    )
    assert (
        "record_linear_shape_stat(2, input_dim, rows, output_dim, kOpA, kOpB, elapsed_us)"
        in kernels_text
    )
    assert "begin_linear_shape_timing(stream)" in kernels_text
    assert "finish_linear_shape_timing(&timing)" in kernels_text
    assert "finish_linear_shape_timing_with_host_fallback(&timing, host_start)" in kernels_text
    assert "cudaDeviceSynchronize() != cudaSuccess" in kernels_text
    assert "cudaStreamIsCapturing(stream, &capture_status)" in kernels_text
    assert "capture_status != cudaStreamCaptureStatusNone" in kernels_text
    assert 'return "cublas_gemmex_bf16"' in gpt2_source_text
    assert "trainer_linear_shape_stats_entry" in kernels_text
    assert "total_us" in kernels_text
    assert "return false;" in kernels_text
    assert 'std::strcmp(value, "1") == 0' in kernels_text
    assert "tf32-cublaslt-optimized" in gpt2_source_text
    assert "tf32-sgemm-optimized" in gpt2_source_text
    assert "lm_head_logits_linear_strategy" in gpt2_source_text
    assert "lm_head_logits_tk_shape_used" in gpt2_source_text
    assert "lm_head_dhidden_linear_strategy" in gpt2_source_text
    assert "lm_head_dhidden_gemmex_shape_used" in gpt2_source_text
    assert "NFN_NATIVE_LINEAR_TK_DINPUT" in kernels_text
    assert "tk_linear_backward_input_bf16_bits_weight_bf16_bits_float32" in kernels_text
    assert "linear_tk_gemm_count" in gpt2_source_text
    assert "bf16_to_f32_vec4_count" in gpt2_source_text
    assert "linear_tk_float_out_gemm_count" in gpt2_source_text
    assert "linear_tk_dweight_gemm_count" in gpt2_source_text
    assert "linear_tk_dgelu_dinput_gemm_count" in gpt2_source_text
    assert "linear_tk_sm120_config_symbol_loaded" in gpt2_source_text
    assert "linear_tk_sm120_dweight_super_m" in gpt2_source_text
    assert "linear_tk_sm120_approx_dgelu_tanh_enabled" in gpt2_source_text
    assert "linear_shape_stats" in gpt2_source_text
    assert '\\"total_us\\"' in gpt2_source_text
    assert '\\"avg_us\\"' in gpt2_source_text
    assert "cublaslt" in gpt2_source_text
    assert "tk_bf16" in gpt2_source_text
    assert "cublas_sgemm" in gpt2_source_text
    assert "block-bf16-cublaslt-shape-gated-lm-head-tk-sm120-default" in gpt2_source_text
    assert "padded-lm-head-tk-sm120-bf16-gemm-default" in gpt2_source_text
    assert "block-forward-dinput-dweight-bf16-lm-head-tf32" in gpt2_source_text
    assert "bf16-shadow-weight-shape-gated-cublaslt-forward" in gpt2_source_text
    assert "bf16-shadow-weight-shape-gated-cublaslt-dinput" in gpt2_source_text
    assert "block_backward_mlp_proj_dgelu_strategy" in gpt2_source_text
    assert "NFN_NATIVE_GPT_BF16_MLP_GRAD_HANDOFF" in gpt2_source_text
    assert 'NFN_NATIVE_GPT2_BF16_MLP_GRAD_HANDOFF"}),\n            true)' in gpt2_source_text
    assert "NFN_NATIVE_GPT_ELIDE_MLP_DGELU_FLOAT_GRAD" in gpt2_source_text
    assert "block_backward_mlp_dgelu_float_grad_elided" in gpt2_source_text
    assert "NFN_NATIVE_GPT_BF16_PERSISTENT_BLOCK_OUTPUTS" in gpt2_source_text
    assert "fp32_persistent_block_output_bytes_elided" in gpt2_source_text
    assert "scratch-residual2-output-plus-fused-bf16-persistent-store" in gpt2_source_text
    assert "NFN_NATIVE_GPT_REUSE_MLP_PROJ_BF16_GRAD_OUT" in gpt2_source_text
    assert "block_backward_mlp_proj_bf16_grad_out_reuse_enabled" in gpt2_source_text
    assert "tk-sm120-fused-dinput-dgelu-reused-bf16-grad-out-bf16-store-bf16-shadow-weight" in gpt2_source_text
    assert "tk-sm120-fused-dinput-dgelu-bf16-store-bf16-shadow-weight-bf16-grad-handoff-no-float-grad" in gpt2_source_text
    assert "tk-sm120-fused-dinput-dgelu-bf16-store-bf16-shadow-weight-bf16-grad-handoff-float-grad" in gpt2_source_text
    assert "tk-sm120-fused-dinput-dgelu-bf16-store-bf16-shadow-weight-float32-grad" in gpt2_source_text
    assert "shape-gated-bf16-cublaslt-dweight-bgrad-accumulate" in gpt2_source_text
    assert "shape-gated-bf16-cublaslt-dweight-bgrad-first-write-then-accumulate" in gpt2_source_text
    assert "forced-bf16-gemmex-dweight-plus-bias-accumulate-fallback" in gpt2_source_text
    assert "tk-sm120-bf16-scratch-to-float32-dweight-diagnostic" in gpt2_source_text
    assert "bf16-shadow-weight-gemmex-forward" in gpt2_source_text
    assert "bf16-shadow-weight-gemmex-dinput" in gpt2_source_text
    assert "non_block_forward_backward_linear_strategy" in gpt2_source_text
    assert ".forward.no_bias.bf16" in gpt2_source_text
    assert ".backward_input.bf16" in gpt2_source_text
    assert "trainer_linear_bf16_gemm_count" in kernels_text
    assert "trainer_linear_tk_gemm_count" in kernels_text
    assert "trainer_linear_tk_float_out_gemm_count" in kernels_text
    assert "trainer_linear_tk_dweight_gemm_count" in kernels_text
    assert "trainer_linear_tk_dgelu_dinput_gemm_count" in kernels_text
    assert "g_linear_tk_dgelu_dinput_gemm_count.fetch_add" in kernels_text
    assert "trainer_linear_cublaslt_bgrad_gemm_count" in kernels_text
    assert "trainer_linear_cublaslt_bgrad_direct_write_count" in kernels_text
    assert "trainer_linear_cublaslt_bgrad_accumulate_count" in kernels_text
    assert "NFN_NATIVE_LINEAR_TK_DWEIGHT" in kernels_text
    assert "trainer_linear_bf16_b_operand" in kernels_text
    assert "trainer_linear_bf16_a_operand" in kernels_text
    assert "trainer_linear_cublaslt_heuristic_index_override" in kernels_text
    assert 'std::getenv("NFN_TILE_CUDA_CUBLASLT_HEURISTIC_INDEX")' in kernels_text
    assert 'std::getenv("NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_INDEX")' in kernels_text
    assert "trainer_linear_cublaslt_shape_heuristic_index_override" in kernels_text
    assert 'std::getenv("NFN_TILE_CUDA_CUBLASLT_HEURISTIC_SHAPE")' in kernels_text
    assert 'std::getenv("NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_SHAPE")' in kernels_text
    assert '"%d,%d,%d,%7[^,],%7[^,],%d"' in kernels_text
    assert "trainer_linear_shape_stats_entry_v2" in kernels_text
    assert "cublaslt_selected_heuristic" in kernels_text
    assert "cublaslt_returned_heuristics" in kernels_text
    assert "cublaslt_workspace_bytes" in kernels_text
    assert "trainer_linear_cublaslt_plan_cache_count" in kernels_text
    assert "trainer_linear_cublaslt_plan_cache_entry" in kernels_text
    assert "linear_cublaslt_plan_cache_available" in gpt2_source_text
    assert "linear_cublaslt_plan_cache_count" in gpt2_source_text
    assert "linear_cublaslt_plan_cache" in gpt2_source_text
    cublaslt_enable_body = kernels_text.split(
        "bool trainer_linear_bf16_cublaslt_shape_enabled(", 1
    )[1].split("\nbool trainer_linear_bf16_cublaslt_shape_supported", 1)[0]
    assert "LinearShapeList enabled_shapes" in cublaslt_enable_body
    assert "parse_linear_shape_list(value)" in cublaslt_enable_body
    assert "linear_shape_list_matches(enabled_shapes" in cublaslt_enable_body
    assert "trainer_linear_cublaslt_descriptor_cache_enabled" in kernels_text
    assert 'std::getenv("NFN_TILE_CUDA_CUBLASLT_DESCRIPTOR_CACHE")' in kernels_text
    assert 'std::getenv("NFN_NATIVE_LINEAR_CUBLASLT_DESCRIPTOR_CACHE")' in kernels_text
    assert "int selected = returned > 1 ? 1 : 0" in kernels_text
    assert "tk_linear_backward_input_dgelu_bf16_bits_float32" in kernels_text
    assert "tk_linear_backward_input_dgelu_weight_bf16_bits_float32" in kernels_text
    assert "tk_linear_backward_input_dgelu_bf16_bits_weight_bf16_bits_float32" in kernels_text
    assert "launch_linear_backward_input_dgelu_weight_bf16_bits_only_float32" in kernels_text
    assert "launch_linear_backward_input_dgelu_bf16_bits_weight_bf16_bits_only_float32" in kernels_text
    assert "gelu_backward_inplace_bf16_bits_to_bf16_bits_float32_kernel" in kernels_text
    bf16_only_dgelu_body = kernels_text.split(
        "void launch_linear_backward_input_dgelu_bf16_bits_weight_bf16_bits_only_float32(", 1
    )[1].split("\nvoid launch_linear_backward_weight_float32", 1)[0]
    assert "cublas_linear_gemm_ex_bf16_bits_ab_to_bf16_bits" in bf16_only_dgelu_body
    assert "launch_gelu_backward_inplace_bf16_bits_to_bf16_bits_float32" in bf16_only_dgelu_body
    assert "trainer_linear_tk_dinput_shape_enabled" in kernels_text
    assert "trainer_linear_tk_dinput_shape_disabled" in kernels_text
    assert 'std::getenv("NFN_TILE_CUDA_LINEAR_TK_DINPUT_ENABLE_SHAPE")' in kernels_text
    assert 'std::getenv("NFN_NATIVE_LINEAR_TK_DINPUT_ENABLE_SHAPE")' in kernels_text
    assert 'std::getenv("NFN_TILE_CUDA_LINEAR_TK_DINPUT_DISABLE_SHAPE")' in kernels_text
    assert 'std::getenv("NFN_NATIVE_LINEAR_TK_DINPUT_DISABLE_SHAPE")' in kernels_text
    assert "trainer_linear_tk_dgelu_dinput_shape_disabled" in kernels_text
    assert 'std::getenv("NFN_TILE_CUDA_LINEAR_TK_DGELU_DINPUT_DISABLE_SHAPE")' in kernels_text
    assert 'std::getenv("NFN_NATIVE_LINEAR_TK_DGELU_DINPUT_DISABLE_SHAPE")' in kernels_text
    assert "trainer_linear_tk_dinput_default_shape_enabled" in kernels_text
    assert "trainer_linear_tk_dinput_default_block_enabled" in kernels_text
    assert 'std::getenv("NFN_NATIVE_LINEAR_TK_DINPUT_DEFAULT_BLOCK")' in kernels_text
    assert 'std::getenv("NFN_TILE_CUDA_LINEAR_TK_DINPUT_DEFAULT_BLOCK")' in kernels_text
    assert "return m <= 4096 && k <= 4096;" in kernels_text
    tk_dinput_body = kernels_text.split(
        "bool tk_linear_backward_input_bf16_bits_weight_bf16_bits_float32(", 1
    )[1].split("\nbool tk_linear_gemm_bf16_forward_gelu_to_bf16_bits", 1)[0]
    assert "!trainer_linear_tk_dinput_enabled() && !shape_enabled && !default_block_enabled" in tk_dinput_body
    assert "trainer_linear_tk_dinput_shape_disabled(input_dim, rows, output_dim, kOpA, kOpB)" in tk_dinput_body
    for dgelu_function_name in (
        "bool tk_linear_backward_input_dgelu_bf16_bits_float32(",
        "bool tk_linear_backward_input_dgelu_weight_bf16_bits_float32(",
        "bool tk_linear_backward_input_dgelu_bf16_bits_weight_bf16_bits_float32(",
    ):
        dgelu_body = kernels_text.split(dgelu_function_name, 1)[1].split("\nbool ", 1)[0]
        assert "trainer_linear_tk_dgelu_dinput_shape_disabled(input_dim, rows, output_dim, kOpA, kOpB)" in dgelu_body
    assert "write_float_grad" in kernels_text
    assert "matmul_dispatch_tk_ab" in kernels_text
    assert "kLayerNormBackwardAffineDefaultRowChunkSize = 128" in kernels_text
    assert "constexpr std::int64_t kDefaultRowChunkSize = 128;" in gpt2_source_text
    assert "NFN_TILE_CUDA_LAYERNORM_AFFINE_ROW_CHUNK_SIZE" in kernels_text
    assert "kLinearBackwardBiasRowChunkSize = 256" in kernels_text
    for function_name in (
        "launch_linear_backward_weight_accumulate_bf16_bits_float32",
        "launch_linear_backward_weight_accumulate_float32_bf16_bits",
        "launch_linear_backward_weight_accumulate_bf16_bits_bf16_bits_float32_beta",
    ):
        function_body = kernels_text.rsplit(f"void {function_name}(", 1)[1].split("\nvoid ", 1)[0]
        assert "launch_linear_backward_weight_tiled_float32_fallback" in function_body
        assert "linear_backward_weight_chunked_atomic_" not in function_body
        assert "kRowChunkSize = 256" not in function_body
    assert "cached-first-gemm-operand-with-optimizer-reset" in gpt2_source_text
    assert "nfn_native_tile_trainer_linear_bf16_cache_reset" in gpt2_source_text
    assert "payload_pack_strategy" in gpt2_source_text
    assert "device-many-float32-to-bf16-bits-contiguous" in gpt2_source_text
    assert "single-contiguous-device-payload-d2h" in gpt2_source_text
    assert "payload_cpu_bf16_conversion" in gpt2_source_text
    assert "nfn_native_tile_global_norm_clip_scale_float32" in header_text
    assert "nfn_native_tile_scale_inplace_by_device_float32" in header_text
    assert "nfn_native_tile_linear_backward_input_float32" in header_text
    assert "nfn_native_tile_linear_backward_input_dgelu_bf16_bits_float32" in header_text
    assert "nfn_native_tile_linear_backward_input_dgelu_weight_bf16_bits_float32" in header_text
    assert "nfn_native_tile_linear_backward_input_dgelu_weight_bf16_bits_only_float32" in header_text
    assert "nfn_native_tile_linear_backward_input_dgelu_bf16_bits_weight_bf16_bits_only_float32" in header_text
    assert "nfn_native_tile_split_qkv_float32" in header_text
    assert "nfn_native_tile_split_qkv_to_heads_float32" in header_text
    assert "nfn_native_tile_split_qkv_to_heads_add_bias_float32" in header_text
    assert "nfn_native_tile_merge_qkv_float32" in header_text
    assert "nfn_native_tile_merge_heads_to_qkv_float32" in header_text
    assert "nfn_native_tile_reshape_heads_float32" in header_text
    assert "nfn_native_tile_merge_heads_float32" in header_text
    assert "nfn_native_tile_linear_backward_weight_float32" in header_text
    assert "nfn_native_tile_linear_backward_weight_accumulate_float32" in header_text
    assert "nfn_native_tile_linear_backward_bias_float32" in header_text
    assert "nfn_native_tile_linear_backward_bias_accumulate_float32" in header_text
    assert "nfn_native_tile_scaled_residual_add_float32" in header_text
    assert "nfn_native_tile_linear_bias_residual_add_float32" in header_text
    assert "nfn_native_tile_linear_bias_residual_add_bf16_linear_float32" in header_text
    assert "nfn_native_tile_linear_bias_residual_add_bf16_linear_bf16_residual_float32" in header_text
    assert "nfn_native_tile_linear_bias_residual_layer_norm_float32" in header_text
    assert "nfn_native_tile_linear_bias_residual_layer_norm_with_stats_float32" in header_text
    assert "nfn_native_tile_linear_bias_residual_layer_norm_with_stats_bf16_linear_float32" in header_text
    assert "nfn_native_tile_linear_bias_residual_layer_norm_with_stats_bf16_residual_float32" in header_text
    assert (
        "nfn_native_tile_linear_bias_residual_layer_norm_with_stats_bf16_linear_bf16_residual_float32"
        in header_text
    )
    assert "nfn_native_tile_linear_bias_residual_layer_norm_with_stats_bf16_residual_bf16_norm_float32" in header_text
    assert (
        "nfn_native_tile_linear_bias_residual_layer_norm_with_stats_bf16_linear_bf16_residual_bf16_norm_float32"
        in header_text
    )
    assert "launch_linear_bias_residual_layer_norm_float32" in source_text
    assert "launch_linear_bias_residual_add_bf16_linear_float32" in source_text
    assert "launch_linear_bias_residual_add_bf16_linear_bf16_residual_float32" in source_text
    assert "launch_linear_bias_residual_layer_norm_with_stats_float32" in source_text
    assert "launch_linear_bias_residual_layer_norm_with_stats_bf16_linear_float32" in source_text
    assert "launch_linear_bias_residual_layer_norm_with_stats_bf16_residual_float32" in source_text
    assert "launch_linear_bias_residual_layer_norm_with_stats_bf16_linear_bf16_residual_float32" in source_text
    assert "launch_linear_bias_residual_layer_norm_with_stats_bf16_residual_bf16_norm_float32" in source_text
    assert (
        "launch_linear_bias_residual_layer_norm_with_stats_bf16_linear_bf16_residual_bf16_norm_float32"
        in source_text
    )
    assert "linear_bias_residual_add_bf16_linear_float32_kernel" in kernels_text
    assert "linear_bias_residual_add_bf16_linear_bf16_residual_float32_kernel" in kernels_text
    assert "linear_bias_residual_layer_norm_bf16_linear_float32_kernel" in kernels_text
    assert "NFN_NATIVE_GPT_BF16_PROJECTION_RESIDUAL" in gpt2_source_text
    assert "bf16_projection_residual_enabled" in gpt2_source_text
    assert "projection_bf16_scratch_elements" in gpt2_source_text
    assert "fused-bf16-linear-bias-residual-add" in gpt2_source_text
    assert "fused-bf16-linear-bias-residual-layernorm" in gpt2_source_text
    assert "nfn_native_tile_gelu_float32" in header_text
    assert "nfn_native_tile_gelu_add_bias_float32" in header_text
    assert "nfn_native_tile_gelu_backward_float32" in header_text
    assert "nfn_native_tile_gelu_backward_inplace_float32" in header_text
    assert "launch_dropout_forward_float32" in source_text
    assert "launch_dropout_backward_float32" in source_text
    assert "launch_gelu_backward_inplace_float32" in source_text
    assert "nfn_native_tile_token_embedding_float32" in header_text
    assert "nfn_native_tile_token_embedding_backward_weight_float32" in header_text
    assert "nfn_native_tile_absolute_position_embedding_float32" in header_text
    assert "nfn_native_tile_absolute_position_embedding_backward_float32" in header_text
    assert "nfn_native_tile_absolute_position_embedding_backward_accumulate_float32" in header_text
    assert "nfn_native_tile_layer_norm_float32" in header_text
    assert "nfn_native_tile_layer_norm_with_stats_float32" in header_text
    assert "nfn_native_tile_layer_norm_apply_stats_bf16_out_float32" in header_text
    assert "nfn_native_tile_layer_norm_backward_input_float32" in header_text
    assert "nfn_native_tile_layer_norm_backward_input_with_stats_float32" in header_text
    assert "nfn_native_tile_layer_norm_backward_input_residual_add_with_stats_float32" in header_text
    assert "nfn_native_tile_layer_norm_backward_input_residual_add_with_stats_bf16_bits_float32" in header_text
    assert "nfn_native_tile_layer_norm_backward_affine_float32" in header_text
    assert "nfn_native_tile_layer_norm_backward_affine_accumulate_float32" in header_text
    assert "nfn_native_tile_layer_norm_backward_affine_accumulate_with_stats_float32" in header_text
    assert "nfn_native_tile_layer_norm_backward_affine_accumulate_with_stats_bf16_bits_float32" in header_text
    assert "launch_layer_norm_with_stats_float32" in source_text
    assert "launch_layer_norm_apply_stats_bf16_out_float32" in source_text
    assert "launch_layer_norm_backward_input_with_stats_float32" in source_text
    assert "launch_layer_norm_backward_input_residual_add_with_stats_float32" in source_text
    assert "launch_layer_norm_backward_affine_accumulate_with_stats_float32" in source_text
    assert "layer_norm_backward_input_residual_add_with_stats_float32_kernel" in kernels_text
    assert "layer_norm_apply_stats_bf16_out_float32_kernel" in kernels_text
    assert "launch_layer_norm_backward_input_residual_add_with_stats_bf16_bits_float32" in source_text
    assert "launch_layer_norm_backward_affine_accumulate_with_stats_bf16_bits_float32" in source_text
    assert "layer_norm_backward_input_residual_add_with_stats_bf16_bits_float32_kernel" in kernels_text
    assert "layer_norm_backward_affine_chunked_atomic_with_stats_bf16_bits_float32_kernel" in kernels_text
    assert "fused-dinput-residual-add-with-forward-stats" in gpt2_source_text
    assert "nfn_native_tile_rms_norm_float32" in header_text
    assert "nfn_native_tile_rms_norm_backward_input_float32" in header_text
    assert "nfn_native_tile_softmax_lastdim_float32" in header_text
    assert "nfn_native_tile_token_cross_entropy_partials_float32" in header_text
    assert "nfn_native_tile_token_cross_entropy_partials_bf16_bits" in header_text
    assert "nfn_native_tile_token_cross_entropy_partials_strided_float32" in header_text
    assert "nfn_native_tile_token_cross_entropy_partials_strided_bf16_bits" in header_text
    assert "nfn_native_tile_token_cross_entropy_partials_strided_bf16_bits_u16_targets" in header_text
    assert "nfn_native_tile_token_cross_entropy_partials_strided_bf16_bits_u16_targets" in source_text
    assert "nfn_native_tile_masked_token_cross_entropy_partials_float32" in header_text
    assert "nfn_native_tile_token_cross_entropy_backward_float32" in header_text
    assert "nfn_native_tile_masked_token_cross_entropy_backward_float32" in header_text
    assert "nfn_native_tile_token_cross_entropy_workspace_allocation_count" in header_text
    assert "nfn_native_tile_token_cross_entropy_workspace_row_capacity" in header_text
    assert "nfn_native_tile_token_cross_entropy_workspace_allocation_count" in source_text
    assert "nfn_native_tile_token_cross_entropy_workspace_row_capacity" in source_text
    assert "nfn_native_tile_token_cross_entropy_backward_with_workspace_float32" in header_text
    assert "nfn_native_tile_token_cross_entropy_backward_inplace_with_workspace_float32" in header_text
    assert "nfn_native_tile_token_cross_entropy_backward_inplace_bf16_bits_with_workspace" in header_text
    assert "nfn_native_tile_token_cross_entropy_backward_inplace_strided_with_workspace_float32" in header_text
    assert "nfn_native_tile_token_cross_entropy_backward_inplace_strided_bf16_bits_with_workspace" in header_text
    assert (
        "nfn_native_tile_token_cross_entropy_backward_inplace_strided_bf16_bits_u16_targets_with_workspace"
        in header_text
    )
    assert (
        "nfn_native_tile_token_cross_entropy_backward_inplace_strided_bf16_bits_u16_targets_with_workspace"
        in source_text
    )
    assert (
        "nfn_native_tile_token_cross_entropy_backward_loss_inplace_strided_bf16_bits_u16_targets"
        in header_text
    )
    assert (
        "nfn_native_tile_token_cross_entropy_backward_loss_inplace_strided_bf16_bits_u16_targets"
        in source_text
    )
    assert (
        "nfn_native_tile_lm_head_classifier_backward_loss_inplace_strided_no_pad_zero_bf16_bits_u16_targets"
        in header_text
    )
    assert (
        "nfn_native_tile_lm_head_classifier_backward_loss_inplace_strided_no_pad_zero_bf16_bits_u16_targets"
        in source_text
    )
    assert (
        "nfn_native_tile_lm_head_classifier_backward_row_losses_inplace_strided_no_pad_zero_bf16_bits_u16_targets"
        in header_text
    )
    assert (
        "nfn_native_tile_lm_head_classifier_backward_row_losses_inplace_strided_no_pad_zero_bf16_bits_u16_targets"
        in source_text
    )
    assert "nfn_native_tile_sum_accumulate_float32" in header_text
    assert "nfn_native_tile_sum_accumulate_float32" in source_text
    assert (
        "nfn_native_tile_lm_head_classifier_backward_inplace_strided_no_pad_zero_bf16_bits_u16_targets_with_workspace"
        in header_text
    )
    assert (
        "nfn_native_tile_lm_head_classifier_backward_inplace_strided_no_pad_zero_bf16_bits_u16_targets_with_workspace"
        in source_text
    )
    assert "nfn_native_tile_lm_head_classifier_chunk_launch_count" in header_text
    assert "nfn_native_tile_lm_head_classifier_chunk_launch_count" in source_text
    assert "nfn_native_tile_lm_head_classifier_true_fused_launch_count" in header_text
    assert "nfn_native_tile_lm_head_classifier_true_fused_launch_count" in source_text
    assert "launch_lm_head_classifier_backward_loss_inplace_strided_no_pad_zero_bf16_bits_u16_targets" in kernels_text
    assert "launch_lm_head_classifier_backward_row_losses_inplace_strided_no_pad_zero_bf16_bits_u16_targets" in kernels_text
    assert "launch_lm_head_classifier_backward_inplace_strided_no_pad_zero_bf16_bits_u16_targets_with_workspace" in kernels_text
    assert "sum_accumulate_float32_kernel" in kernels_text
    assert "launch_sum_accumulate_float32" in kernels_text
    assert "token_cross_entropy_partials_strided_float32_kernel" in kernels_text
    assert "token_cross_entropy_partials_strided_bf16_bits_kernel" in kernels_text
    assert "token_cross_entropy_partials_strided_bf16_bits_u16_targets_kernel" in kernels_text
    assert "token_cross_entropy_backward_inplace_strided_float32_fused_kernel" in kernels_text
    assert "token_cross_entropy_backward_inplace_strided_bf16_bits_fused_kernel" in kernels_text
    assert "token_cross_entropy_backward_inplace_strided_bf16_bits_u16_targets_fused_kernel" in kernels_text
    assert (
        "token_cross_entropy_backward_inplace_strided_no_pad_zero_bf16_bits_u16_targets_default_kernel"
        in kernels_text
    )
    assert (
        "token_cross_entropy_backward_loss_inplace_strided_bf16_bits_u16_targets_fused_kernel"
        in kernels_text
    )
    loss_backward_body = kernels_text.split(
        "token_cross_entropy_backward_loss_inplace_strided_bf16_bits_u16_targets_fused_kernel",
        1,
    )[1].split("\n__tile_global__ void token_cross_entropy_backward_chunked_float32_kernel", 1)[0]
    assert "loss_out[row] = loss" in loss_backward_body
    assert "atomicAdd(loss_out" in loss_backward_body
    assert "const float target_logit = bf16_bits_to_f32_device" in loss_backward_body
    assert loss_backward_body.index("const float target_logit") < loss_backward_body.index(
        "loss_out[row] = loss"
    )
    assert loss_backward_body.index("const float target_logit") < loss_backward_body.index(
        "if (vec_stores || vec_normal_stores)"
    )
    assert "__syncthreads();" not in loss_backward_body
    assert "nfn_native_tile_masked_token_cross_entropy_backward_with_workspace_float32" in header_text
    assert "nfn_native_tile_scaled_dot_product_attention_float32" in header_text
    assert "nfn_native_tile_scaled_dot_product_attention_packed_qkv_bf16_float32" in header_text
    assert "nfn_native_tile_scaled_dot_product_attention_packed_qkv_store_lse_bf16_float32" in header_text
    assert (
        "nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_from_merged_grad_float32"
        in header_text
    )
    assert (
        "nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_from_saved_lse_bf16_from_merged_grad_float32"
        in header_text
    )
    assert (
        "nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_bf16_bits_from_merged_grad_float32"
        in header_text
    )
    assert (
        "nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_bf16_bits_from_saved_lse_bf16_from_merged_grad_float32"
        in header_text
    )
    assert "nfn_native_tile_attention_forward_stats_reset" in header_text
    assert "nfn_native_tile_attention_forward_row_launch_count" in header_text
    assert "nfn_native_tile_attention_forward_row_fallback_count" in header_text
    assert "nfn_native_tile_attention_forward_scalar_launch_count" in header_text
    assert "nfn_native_tile_attention_forward_row_prelaunch_clear_error" in header_text
    assert "nfn_native_tile_attention_forward_row_prelaunch_peek_error" in header_text
    assert "nfn_native_tile_attention_forward_row_grid_x" in header_text
    assert "nfn_native_tile_attention_forward_row_block_x" in header_text
    assert "nfn_native_tile_attention_forward_row_attr_status" in header_text
    assert "nfn_native_tile_attention_forward_row_attr_const_size_bytes" in header_text
    assert "cudaFuncGetAttributes" in kernels_text
    assert "nfn_native_tile_scaled_dot_product_attention_backward_float32" in header_text
    assert "nfn_native_tile_scaled_dot_product_attention_backward_from_merged_grad_float32" in header_text
    assert "nfn_native_tile_scaled_dot_product_attention_backward_to_qkv_from_merged_grad_float32" in header_text
    assert (
        "nfn_native_tile_scaled_dot_product_attention_backward_to_qkv_reuse_forward_from_merged_grad_float32"
        in header_text
    )
    assert "nfn_native_tile_attention_tk_store_forward_workspace_bf16" in header_text
    assert (
        "nfn_native_tile_scaled_dot_product_attention_backward_to_qkv_from_saved_tk_bf16_from_merged_grad_float32"
        in header_text
    )
    assert "launch_scaled_dot_product_attention_packed_qkv_bf16_float32" in source_text
    assert "launch_scaled_dot_product_attention_packed_qkv_store_lse_bf16_float32" in source_text
    assert (
        "launch_scaled_dot_product_attention_packed_qkv_backward_to_qkv_from_merged_grad_float32"
        in source_text
    )
    assert (
        "launch_scaled_dot_product_attention_packed_qkv_backward_to_qkv_from_saved_lse_bf16_from_merged_grad_float32"
        in source_text
    )
    assert (
        "launch_scaled_dot_product_attention_packed_qkv_backward_to_qkv_bf16_bits_from_merged_grad_float32"
        in source_text
    )
    assert (
        "launch_scaled_dot_product_attention_packed_qkv_backward_to_qkv_bf16_bits_from_saved_lse_bf16_from_merged_grad_float32"
        in source_text
    )
    assert "launch_tk_attention_packed_qkv_forward_bf16_float32" in kernels_text
    assert "launch_tk_attention_packed_qkv_forward_store_lse_bf16_float32" in kernels_text
    assert "launch_tk_attention_packed_qkv_backward_to_qkv_float32" in kernels_text
    assert "launch_tk_attention_packed_qkv_backward_to_qkv_bf16_bits" in kernels_text
    assert "packed_attention_dprep_kernel" in kernels_text
    assert "launch_forward_causal_packed_qkv_btc" in kernels_text
    assert "launch_backward_causal_packed_qkv_packed_grads" in kernels_text
    assert "packed-qkv-bf16-no-split" in gpt2_source_text
    assert "packed-qkv-bf16-bias-inplace" in gpt2_source_text
    assert "packed-o-bf16-direct-gemm" in gpt2_source_text
    assert "elided-direct-bf16-projection" in gpt2_source_text
    assert ".attn.out.forward.no_bias.packed_o_bf16_bits" in gpt2_source_text
    assert ".attn.out.backward_weight_bias.accumulate.packed_o_bf16_bits" in gpt2_source_text
    assert ".attn.unpack_packed_out_bf16" not in gpt2_source_text
    assert "tk-sm120-packed-qkv-bf16-backward-bridge" in gpt2_source_text
    assert "linear_backward_weight_tiled_float32_kernel" in kernels_text
    assert "launch_linear_backward_weight_tiled_float32_fallback" in kernels_text
    assert "linear_backward_bias_chunked_atomic_float32_kernel" in kernels_text
    assert "cublas_linear_forward_float32" in kernels_text
    assert "cublas_linear_backward_input_float32" in kernels_text
    assert "cublas_linear_backward_weight_float32" in kernels_text
    assert "cublas_linear_backward_bias_float32" in kernels_text
    assert "rows <= kTileSize && cublas_linear_backward_bias_float32" in kernels_text
    assert "kAttentionValueChunkSize = 64" in kernels_text
    assert "cudaPeekAtLastError" in kernels_text
    assert "cudaGetLastError" in kernels_text
    assert "g_attention_forward_row_prelaunch_clear_error" in kernels_text
    assert "g_attention_forward_row_prelaunch_peek_error" in kernels_text
    assert "g_attention_forward_row_grid_x" in kernels_text
    assert "g_attention_forward_row_launch_disabled" in kernels_text
    assert "ensure_trainer_bias_ones" in kernels_text
    assert "linear_add_bias_float32_kernel" in kernels_text
    assert "gelu_add_bias_float32_kernel" in kernels_text
    assert "linear_bias_residual_add_float32_kernel" in kernels_text
    assert "linear_bias_residual_layer_norm_float32_kernel" in kernels_text
    assert "mean_out[row] = static_cast<float>(mean)" in kernels_text
    assert "rstd_out[row] = static_cast<float>(norm_scale)" in kernels_text
    assert "NFN_NATIVE_GPT_FUSE_ATTENTION_RESIDUAL_LN2" in gpt2_source_text
    assert "NFN_NATIVE_GPT2_FUSE_ATTENTION_RESIDUAL_LN2" in gpt2_source_text
    assert "nfn_native_tile_linear_bias_residual_layer_norm_with_stats_float32" in gpt2_source_text
    assert "nfn_native_tile_linear_bias_residual_layer_norm_with_stats_bf16_residual_float32" in gpt2_source_text
    assert "nfn_native_tile_linear_bias_residual_layer_norm_with_stats_bf16_residual_bf16_norm_float32" in gpt2_source_text
    assert "NFN_NATIVE_GPT_FUSE_LN2_BF16_OUT" in gpt2_source_text
    assert "NFN_NATIVE_GPT_ELIDE_LN2_BF16_NORM_FLOAT_STORE" in gpt2_source_text
    assert "fused_ln2_bf16_norm_float_store_elision_enabled" in gpt2_source_text
    assert "NFN_NATIVE_GPT_FUSE_MLP_RESIDUAL_NEXT_LN1" in gpt2_source_text
    assert "NFN_NATIVE_GPT2_FUSE_MLP_RESIDUAL_NEXT_LN1" in gpt2_source_text
    assert ".mlp.bias_residual_next_ln1_bf16_linear_bf16_norm" in gpt2_source_text
    assert "mlp_residual_next_ln1_fusion_count" in gpt2_source_text
    assert "fused-mlp-bias-residual-next-ln1-when-packed-ln1-storage-is-available" in gpt2_source_text
    assert "stored_mlp_ln2_bf16_float_store_elided_elements" in gpt2_source_text
    assert "stored_mlp_ln2_bf16_prepack_strategy" in gpt2_source_text
    assert "stored_mlp_ln2_bf16_fused_store_kernel_launches" in gpt2_source_text
    assert "attention_residual_ln2_strategy" in gpt2_source_text
    assert "norm_out != nullptr" in kernels_text
    assert "fused-linear-bias-residual-layernorm" in gpt2_source_text
    assert "token_cross_entropy_backward_rowwise_float32_kernel" in kernels_text
    assert "token_cross_entropy_backward_rowwise_inplace_float32_kernel" in kernels_text
    assert "token_cross_entropy_row_stats_float32_kernel" in kernels_text
    assert "token_cross_entropy_bf16_bits_row_stats_kernel" in kernels_text
    assert "token_cross_entropy_backward_chunked_float32_kernel" in kernels_text
    assert "token_cross_entropy_backward_chunked_inplace_float32_kernel" in kernels_text
    assert "ensure_token_cross_entropy_workspace(rows)" in kernels_text
    assert "g_token_cross_entropy_workspace_allocation_count.fetch_add" in kernels_text
    ce_backward_body = kernels_text.split("void launch_token_cross_entropy_backward_float32(", 1)[1].split(
        "\nvoid launch_token_cross_entropy_backward_with_workspace_float32(",
        1,
    )[0]
    masked_ce_backward_body = kernels_text.split(
        "void launch_masked_token_cross_entropy_backward_float32(",
        1,
    )[1].split("\nvoid launch_masked_token_cross_entropy_backward_with_workspace_float32(", 1)[0]
    assert "cudaMalloc" not in ce_backward_body
    assert "cudaFree" not in ce_backward_body
    assert "cudaMalloc" not in masked_ce_backward_body
    assert "cudaFree" not in masked_ce_backward_body
    assert "token_cross_entropy_backward_inplace_bf16_bits_kernel" in kernels_text
    assert "token_cross_entropy_backward_inplace_bf16_bits_fused_kernel" in kernels_text
    assert "block_reduce_max_f32" in kernels_text
    assert "block_reduce_sum_f32" in kernels_text
    assert "token_cross_entropy_backward_elementwise_float32_kernel" not in kernels_text
    assert "gelu_float32_kernel" in kernels_text
    assert "gelu_backward_float32_kernel" in kernels_text
    assert "gelu_backward_inplace_float32_kernel" in kernels_text
    assert "linear_bias_residual_add_bf16_linear_dim768_float32_kernel" in kernels_text
    assert "dim768_bf16_residual_add_enabled()" in kernels_text
    assert "NFN_TILE_CUDA_DIM768_BF16_RESIDUAL_ADD" in kernels_text
    assert "if (output_dim == 768 && dim768_bf16_residual_add_enabled())" in kernels_text
    assert "constexpr int kRowsPerBlock = 2" in kernels_text
    assert "token_embedding_backward_weight_float32_kernel" in kernels_text
    assert "token_embedding_u16_float32_kernel" in kernels_text
    assert "token_embedding_backward_weight_u16_float32_kernel" in kernels_text
    assert "nfn_native_tile_token_embedding_u16_float32" in header_text
    assert "nfn_native_tile_token_embedding_u16_float32" in source_text
    assert "nfn_native_tile_token_embedding_backward_weight_u16_float32" in header_text
    assert "nfn_native_tile_token_embedding_backward_weight_u16_float32" in source_text
    assert "init_gpt2_token_weight_float32_kernel" in kernels_text
    assert "uint16_to_int64_kernel" in kernels_text
    assert "NFN_NATIVE_GPT_DIRECT_U16_TOKENS" in gpt2_source_text
    assert "NFN_NATIVE_GPT2_DIRECT_U16_TOKENS" in gpt2_source_text
    assert "direct_u16_token_ids_enabled" in gpt2_source_text
    assert "active_targets_u16 = token_u16_device_arena + active_rows" in gpt2_source_text
    assert ".wte.forward.u16" in gpt2_source_text
    assert ".ce.forward.public_vocab_strided_bf16_bits_u16_targets" in gpt2_source_text
    assert "ce.backward.inplace.public_vocab_strided_bf16_bits_u16_targets" in gpt2_source_text
    assert "lm_head_classifier_backward_loss_bf16_u16" in gpt2_source_text
    assert "lm_head_classifier_backward_row_losses_bf16_u16" in gpt2_source_text
    assert "NFN_NATIVE_GPT_LM_HEAD_FUSED_LOSS_BACKWARD" in gpt2_source_text
    assert "NFN_NATIVE_GPT2_LM_HEAD_FUSED_LOSS_BACKWARD" in gpt2_source_text
    assert "NFN_NATIVE_GPT_LM_HEAD_CLASSIFIER_CE_NO_LOSS" in gpt2_source_text
    assert "NFN_NATIVE_GPT2_LM_HEAD_CLASSIFIER_CE_NO_LOSS" in gpt2_source_text
    assert "NFN_NATIVE_GPT_LM_HEAD_PROB_ONLY_TARGET_CORRECTION_THREADS" in gpt2_source_text
    assert "NFN_NATIVE_GPT2_LM_HEAD_PROB_ONLY_TARGET_CORRECTION_THREADS" in gpt2_source_text
    assert "NFN_NATIVE_GPT_LM_HEAD_PROB_ONLY_CE_TARGET_CORRECTIONS" in gpt2_source_text
    assert "NFN_NATIVE_GPT2_LM_HEAD_PROB_ONLY_CE_TARGET_CORRECTIONS" in gpt2_source_text
    assert "NFN_NATIVE_GPT_LM_HEAD_ROW_LOSS_REDUCTION" in gpt2_source_text
    assert "NFN_NATIVE_GPT2_LM_HEAD_ROW_LOSS_REDUCTION" in gpt2_source_text
    assert "NFN_NATIVE_GPT_LM_HEAD_ROW_LOSS_SUM_ACCUMULATE" in gpt2_source_text
    assert "NFN_NATIVE_GPT2_LM_HEAD_ROW_LOSS_SUM_ACCUMULATE" in gpt2_source_text
    assert "lm_head_fused_loss_backward_enabled" in gpt2_source_text
    assert "lm_head_classifier_ce_no_loss_requested" in gpt2_source_text
    assert "lm_head_classifier_ce_no_loss_enabled" in gpt2_source_text
    assert "lm_head_classifier_no_loss_chunk_count" in gpt2_source_text
    assert "lm_head_ce_loss_backward_fused_enabled" in gpt2_source_text
    assert "lm_head_ce_row_loss_reduction_enabled" in gpt2_source_text
    assert "lm_head_ce_row_loss_sum_accumulate_enabled" in gpt2_source_text
    assert "no-loss-dlogits-public-vocab-no-pad-zero-bf16-u16-targets" in gpt2_source_text
    assert "no-loss-prob-only-dlogits-vec8-loads-normal-vec8-stores-plus-target-corrections" in gpt2_source_text
    assert (
        "no-loss-prob-only-dlogits-vec8-loads-normal-vec8-stores-plus-combined-target-correction"
        in gpt2_source_text
    )
    assert (
        "no-loss-prob-only-dlogits-vec8-loads-normal-vec8-stores-plus-ce-target-correction"
        in gpt2_source_text
    )
    assert "lm_head_prob_only_corrections_chunk_count > 0" in gpt2_source_text
    assert "lm_head_prob_only_ce_target_correction_chunk_count" in gpt2_source_text
    assert "no-loss-dlogits-vec8-loads-scalar-stores" in gpt2_source_text
    assert "no-loss-default-specialized-dlogits-vec8-loads-scalar-stores" in gpt2_source_text
    assert "fused-row-losses-sum-accumulate-and-dlogits-public-vocab-no-pad-zero-bf16-u16-targets" in gpt2_source_text
    assert "row-chunked-public-vocab-bf16-u16-loss-dlogits-tile-abi" in gpt2_source_text
    assert "ce_backward_inplace_strided_no_pad_zero_bf16_bits_u16_targets_workspace" in gpt2_source_text
    assert "nfn_native_tile_token_cross_entropy_backward_inplace_strided_no_pad_zero_bf16_bits_u16_targets_with_workspace" in header_text
    assert "nfn_native_tile_token_cross_entropy_backward_loss_inplace_strided_no_pad_zero_bf16_bits_u16_targets" in source_text
    assert "zero_token_padding_enabled" in gpt2_source_text
    assert "token_weight_bf16_padding_memset_count" in gpt2_source_text
    assert "cuda_memset_async(\n                            token_weight_bf16 + public_token_weight_elements" in gpt2_source_text
    assert '\\"token_weight_bf16_padding_memset_count\\"' in gpt2_source_text
    assert "wte.backward_weight.u16" in gpt2_source_text
    assert "elided-direct-u16-kernels" in gpt2_source_text
    assert "token_u16_arena.copy_async" in gpt2_source_text
    assert "token_i64_arena.device_widen" in gpt2_source_text
    assert "next_into(token_ids_pinned, targets_pinned, active_rows)" in gpt2_source_text
    assert "next_into(token_ids_pinned, active_targets_pinned, active_rows)" in gpt2_source_text
    assert "direct-sampler-to-pinned-arena" in gpt2_source_text
    assert (
        "attention(tape.q_heads, tape.k_heads, tape.v_heads, tape.attn_heads, active_activation_elements"
        in gpt2_source_text
    )
    assert "attention(tape.q_heads, tape.k_heads, tape.v_heads, tape.attn_heads, active_batch_size" not in gpt2_source_text
    assert "forward_row_qkv_scratch_allocated" in gpt2_source_text
    assert "forward_row_qkv_scratch_buffers_elided" in gpt2_source_text
    assert 'visit(&tape.q, activation_elements, prefix + ".attn.q")' not in gpt2_source_text
    assert 'visit(&tape.k, activation_elements, prefix + ".attn.k")' not in gpt2_source_text
    assert 'visit(&tape.v, activation_elements, prefix + ".attn.v")' not in gpt2_source_text
    assert "steady_clock_host_wall_ms" in gpt2_source_text
    assert "setup_wall_ms" in gpt2_source_text
    assert "train_loop_wall_ms" in gpt2_source_text
    assert "NFN_NATIVE_GPT_TRAIN_LOOP_EVENT_TIMING" in gpt2_source_text
    assert "train_loop_cuda_event_wall_ms_per_step" in gpt2_source_text
    assert "train_loop_cuda_event_first_step_wall_ms_per_step" in gpt2_source_text
    assert "train_loop_cuda_event_steady_state_wall_ms_per_step" in gpt2_source_text
    assert 'run(cuda_device_synchronize(), "train_loop.complete");' in gpt2_source_text
    assert (
        gpt2_source_text.index('run(cuda_device_synchronize(), "train_loop.complete");')
        < gpt2_source_text.index("train_loop_wall_ms = elapsed_ms(train_loop_start_time, train_loop_end_time);")
        < gpt2_source_text.index('"token_weight.sample"')
    )
    assert "validation_wall_ms" in gpt2_source_text
    assert "const bool validation_runtime_enabled = cfg.eval_every_steps > 0 && cfg.eval_batches > 0;" in gpt2_source_text
    assert "std::optional<neuralfn::native_train::SequentialTokenBatchSampler> val_sampler" in gpt2_source_text
    assert "const bool validation_sampler_constructed = val_sampler.has_value();" in gpt2_source_text
    assert "checkpoint_wall_ms" in gpt2_source_text
    assert 'setenv_default_if_empty("CUDA_VISIBLE_DEVICES", "0")' in gpt2_source_text
    assert 'setenv_default_if_empty("CUDA_DEVICE_MAX_CONNECTIONS", "1")' in gpt2_source_text
    assert 'setenv_default_if_empty("CUDA_MODULE_LOADING", "LAZY")' in gpt2_source_text
    assert '\\"cuda_module_loading\\"' in gpt2_source_text
    assert "--no-checkpoint" in gpt2_source_text
    assert "--native-cuda-no-checkpoint" in gpt2_source_text
    assert "cfg.write_checkpoint = false" in gpt2_source_text
    assert "checkpoint_export_enabled" in gpt2_source_text
    assert '"  \\"checkpoint_export_enabled\\": "\n        << (final_checkpoint_export_enabled ? "true" : "false")' in gpt2_source_text
    assert "checkpoint_export_startup_only_elided" in gpt2_source_text
    assert "const bool final_checkpoint_export_enabled = cfg.write_checkpoint && !cfg.startup_only" in gpt2_source_text
    assert "if (passed && final_checkpoint_export_enabled)" in gpt2_source_text
    assert "require_optimized_attention = true" in gpt2_source_text
    assert "require_optimized_kernels = true" in gpt2_source_text
    assert "--allow-basic-kernel-fallback" in gpt2_source_text
    assert "--require-optimized-kernels" in gpt2_source_text
    assert "optimized native GPT kernel contract failed" in gpt2_source_text
    assert "basic TF32/SGEMM linear fallback launched" in gpt2_source_text
    assert '\\"optimized_kernel_contract_required\\"' in gpt2_source_text
    assert '\\"optimized_kernel_contract_passed\\"' in gpt2_source_text
    assert "--allow-scalar-attention-fallback" in gpt2_source_text
    assert "optimized attention required, but scalar attention fallback launched" in gpt2_source_text
    assert '\\"enabled\\": ' in gpt2_source_text
    assert "train_tokens_per_second" in gpt2_source_text
    assert "NFN_NATIVE_GPT_STAGE_TIMING" in gpt2_source_text
    assert "NFN_NATIVE_GPT2_STAGE_TIMING" in gpt2_source_text
    assert "NFN_NATIVE_GPT_STAGE_TIMING_MAX_EVENTS" in gpt2_source_text
    assert "NFN_NATIVE_GPT2_STAGE_TIMING_MAX_EVENTS" in gpt2_source_text
    assert "cudaEventCreateWithFlags" in gpt2_source_text
    assert "cudaEventElapsedTime" in gpt2_source_text
    assert "cudaStreamCreateWithFlags" in gpt2_source_text
    assert "cudaStreamWaitEvent" in gpt2_source_text
    assert "cudaStreamSynchronize" in gpt2_source_text
    assert "cudaStreamDestroy" in gpt2_source_text
    assert "NFN_NATIVE_GPT_BLOCK_MLP_FC_CONCURRENT_DINPUT_DWEIGHT" in gpt2_source_text
    assert "NFN_NATIVE_GPT2_BLOCK_MLP_FC_CONCURRENT_DINPUT_DWEIGHT" in gpt2_source_text
    assert "NFN_NATIVE_GPT_BLOCK_QKV_CONCURRENT_DINPUT_DWEIGHT" in gpt2_source_text
    assert "NFN_NATIVE_GPT2_BLOCK_QKV_CONCURRENT_DINPUT_DWEIGHT" in gpt2_source_text
    assert "NFN_NATIVE_GPT_QKV_DINPUT_BEFORE_DWEIGHT" in gpt2_source_text
    assert "NFN_NATIVE_GPT2_QKV_DINPUT_BEFORE_DWEIGHT" in gpt2_source_text
    assert "NFN_NATIVE_GPT_BLOCK_ATTN_PROJ_CONCURRENT_DINPUT_DWEIGHT" in gpt2_source_text
    assert "NFN_NATIVE_GPT2_BLOCK_ATTN_PROJ_CONCURRENT_DINPUT_DWEIGHT" in gpt2_source_text
    assert "NFN_NATIVE_GPT_BLOCK_ATTN_PROJ_FIRST_STEP_CONCURRENT_DINPUT_DWEIGHT" in gpt2_source_text
    assert "NFN_NATIVE_GPT2_BLOCK_ATTN_PROJ_FIRST_STEP_CONCURRENT_DINPUT_DWEIGHT" in gpt2_source_text
    assert (
        'env_or_empty_any({"NFN_NATIVE_GPT_ATTN_PROJ_DINPUT_BEFORE_DWEIGHT",\n'
        '                              "NFN_NATIVE_GPT2_ATTN_PROJ_DINPUT_BEFORE_DWEIGHT"}),\n'
        "            false);"
    ) in gpt2_source_text
    assert "block_backward_mlp_fc_concurrent_dinput_dweight_requested" in gpt2_source_text
    assert "block_backward_pair_streams_available" in gpt2_source_text
    assert "block_backward_mlp_fc_concurrent_dinput_dweight_enabled" in gpt2_source_text
    assert "block_backward_mlp_fc_concurrent_dinput_dweight_count" in gpt2_source_text
    assert '"block_backward_mlp_fc_concurrent_dinput_dweight_count",' in speed_source
    assert "block_backward.mlp_fc.dinput_dweight_concurrent" in gpt2_source_text
    assert "NFN_NATIVE_GPT_BLOCK_MLP_PROJ_CONCURRENT_DINPUT_DWEIGHT" in gpt2_source_text
    assert "NFN_NATIVE_GPT2_BLOCK_MLP_PROJ_CONCURRENT_DINPUT_DWEIGHT" in gpt2_source_text
    assert "block_backward_mlp_proj_concurrent_dinput_dweight_requested" in gpt2_source_text
    assert "block_backward_mlp_proj_concurrent_dinput_dweight_enabled" in gpt2_source_text
    assert "block_backward_mlp_proj_concurrent_dinput_dweight_count" in gpt2_source_text
    assert '"block_backward_mlp_proj_concurrent_dinput_dweight_count",' in speed_source
    assert "block_backward.mlp_proj.dinput_dweight_concurrent" in gpt2_source_text
    assert "block_backward_qkv_concurrent_dinput_dweight_requested" in gpt2_source_text
    assert "block_backward_qkv_concurrent_dinput_dweight_enabled" in gpt2_source_text
    assert "block_backward_qkv_concurrent_dinput_dweight_count" in gpt2_source_text
    assert '"block_backward_qkv_concurrent_dinput_dweight_count",' in speed_source
    assert "block_backward.qkv.dinput_dweight_concurrent" in gpt2_source_text
    assert "block_backward_mlp_proj_tk_dweight_requested" in gpt2_source_text
    assert "block_backward_mlp_proj_tk_dweight_enabled" in gpt2_source_text
    assert "diagnostic-tk-sm120-mlp-proj-dweight-plus-tile-bias" in gpt2_source_text
    assert "block_backward_attn_proj_concurrent_dinput_dweight_requested" in gpt2_source_text
    assert "block_backward_attn_proj_concurrent_dinput_dweight_enabled" in gpt2_source_text
    assert "block_backward_attn_proj_concurrent_dinput_dweight_count" in gpt2_source_text
    assert '"block_backward_attn_proj_concurrent_dinput_dweight_count",' in speed_source
    assert "block_backward_attn_proj_first_step_concurrent_dinput_dweight_requested" in gpt2_source_text
    assert "block_backward_attn_proj_first_step_concurrent_dinput_dweight_enabled" in gpt2_source_text
    assert "block_backward_attn_proj_first_step_concurrent_dinput_dweight_count" in gpt2_source_text
    assert "block_backward.attn_proj.dinput_dweight_concurrent" in gpt2_source_text
    assert "cudaStreamCreateWithFlags block_backward_dinput" in gpt2_source_text
    assert "cudaStreamCreateWithFlags block_backward_dweight" in gpt2_source_text
    assert "cudaEventCreateWithFlags block_backward_pair_ready" in gpt2_source_text
    assert "cudaEventDestroy block_backward_pair_ready" in gpt2_source_text
    assert "cudaStreamDestroy block_backward_dinput" in gpt2_source_text
    assert "cudaStreamDestroy block_backward_dweight" in gpt2_source_text
    assert "stage_timing_enabled" in gpt2_source_text
    assert "stage_timing_max_events" in gpt2_source_text
    assert "kDefaultStageTimingEventsPerOptimizerStep = 4096" in gpt2_source_text
    assert "stage_timing_default_steps" in gpt2_source_text
    assert "stage_timing_event_count" in gpt2_source_text
    assert "stage_timing_dropped_event_count" in gpt2_source_text
    assert "stage_timing_prealloc_event_pairs_requested" in gpt2_source_text
    assert "stage_timing_event_pair_preallocated_count" in gpt2_source_text
    assert "stage_timing_event_pair_hot_create_count" in gpt2_source_text
    assert "stage_timing" in gpt2_source_text
    assert "first_step_total_ms" in gpt2_source_text
    assert "steady_state_total_ms" in gpt2_source_text
    assert "block_backward" in gpt2_source_text
    assert "block_recompute" in gpt2_source_text
    assert "lm_head_backward" in gpt2_source_text
    assert "lm_head_logits_tk_gemm_count" in gpt2_source_text
    assert "lm_head_logits_cublaslt_gemm_count" in gpt2_source_text
    assert "lm_head_logits_bf16_gemm_count" in gpt2_source_text
    assert "lm_head_dhidden_tk_gemm_count" in gpt2_source_text
    assert "lm_head_dhidden_cublaslt_gemm_count" in gpt2_source_text
    assert "lm_head_dhidden_bf16_gemm_count" in gpt2_source_text
    assert "block_backward_dinput_tk_gemm_count" in gpt2_source_text
    assert "block_backward_dinput_cublaslt_gemm_count" in gpt2_source_text
    assert "block_backward_dinput_bf16_gemm_count" in gpt2_source_text
    assert "block_backward_mlp_proj_dinput_before_dweight_count" in gpt2_source_text
    assert "block_backward_mlp_fc_dinput_before_dweight_count" in gpt2_source_text
    assert (
        'env_or_empty_any({"NFN_NATIVE_GPT_MLP_FC_DINPUT_BEFORE_DWEIGHT",\n'
        '                              "NFN_NATIVE_GPT2_MLP_FC_DINPUT_BEFORE_DWEIGHT"}),\n'
        "            false);"
    ) in gpt2_source_text
    assert "block_backward_attn_proj_dinput_before_dweight_count" in gpt2_source_text
    assert "block_backward_qkv_dinput_before_dweight_count" in gpt2_source_text
    assert "float_arena_cuda_malloc_wall_ms" in gpt2_source_text
    assert "float_arena_pointer_assign_wall_ms" in gpt2_source_text
    assert "uint16_arena_cuda_malloc_wall_ms" in gpt2_source_text
    assert "uint16_arena_pointer_assign_wall_ms" in gpt2_source_text
    assert '\\"float_arena_allocated_bytes\\"' in gpt2_source_text
    assert '\\"uint16_arena_allocated_bytes\\"' in gpt2_source_text
    assert '\\"transformer_arena_allocated_bytes\\"' in gpt2_source_text
    assert '\\"activation_storage_bytes\\"' in gpt2_source_text
    assert '\\"lm_head_bf16_logit_bytes\\"' in gpt2_source_text
    assert '\\"total_requested_elements\\"' in gpt2_source_text
    assert '\\"total_allocated_elements\\"' in gpt2_source_text
    assert '\\"total_requested_bytes\\"' in gpt2_source_text
    assert '\\"total_allocated_bytes\\"' in gpt2_source_text
    assert "transformer_device_arena_cuda_malloc_wall_ms" in gpt2_source_text
    assert "transformer_device_arena_pointer_assign_wall_ms" in gpt2_source_text
    assert "lm_head_logits_tk_used" in gpt2_source_text
    assert "lm_head_dhidden_cublaslt_shape_used" in gpt2_source_text
    assert "lm_head_backward.dhidden" in gpt2_source_text
    assert "lm_head_backward.dweight" in gpt2_source_text
    assert "lm_head_backward.dhidden_dweight_concurrent" in gpt2_source_text
    assert 'stage_name + ".attention"' in gpt2_source_text
    assert 'stage_name + ".attention.qkv"' in gpt2_source_text
    assert 'stage_name + ".attention.sdpa"' in gpt2_source_text
    assert 'stage_name + ".attention.proj"' in gpt2_source_text
    assert 'stage_name + ".mlp_fc_gelu"' in gpt2_source_text
    assert 'stage_name + ".mlp_fc_gelu.fc"' in gpt2_source_text
    assert 'stage_name + ".mlp_fc_gelu.gelu"' in gpt2_source_text
    assert 'stage_name + ".mlp_proj.proj"' in gpt2_source_text
    assert "block_backward.mlp_proj" in gpt2_source_text
    assert "block_backward.attn_sdpa" in gpt2_source_text
    assert "block_backward.qkv" in gpt2_source_text
    assert "mlp.gelu.backward_inplace" in gpt2_source_text
    assert "mlp_proj_backward_gelu_inplace" in gpt2_source_text
    assert "mlp_proj_backward_grad_act_scratch_allocated" in gpt2_source_text
    assert "layer_norm_backward_residual_scratch_buffers_allocated" in gpt2_source_text
    assert "layer_norm_backward_residual_scratch_buffers_elided" in gpt2_source_text
    assert "layer_norm_backward_residual_scratch_elements_elided" in gpt2_source_text
    assert "{&grad_residual1_from_mlp,\n              fuse_ln_backward_residual_enabled ? 0 : activation_elements,\n              \"residual1.grad_from_mlp\"}" in gpt2_source_text
    assert "{&grad_x_from_attn,\n              fuse_ln_backward_residual_enabled ? 0 : activation_elements,\n              \"embedding_residual.grad_from_attention\"}" in gpt2_source_text
    assert "compute_final_output" in gpt2_source_text
    assert "stored_mlp_activation_store_kernel_launches" in gpt2_source_text
    assert "stored_mlp_layer_norm_stats_elements" in gpt2_source_text
    assert "stored_mlp_layer_norm_stats_bytes" in gpt2_source_text
    assert "stored_mlp_activation_backward_consumer_strategy" in gpt2_source_text
    assert "reuse_packed_ln2_fc_gelu_enabled" in gpt2_source_text
    assert "stored_mlp_forward_strategy" in gpt2_source_text
    assert "tk-sm120-fused-fc-bias-gelu-prepacked-ln2-bf16-shadow-weight" in gpt2_source_text
    assert "tk-sm120-fused-fc-bias-gelu-bf16-store-bf16-shadow-weight" in gpt2_source_text
    assert "mlp.gelu.backward_inplace.bf16_bits" in gpt2_source_text
    assert "NFN_NATIVE_GPT_STORE_MLP_ACTIVATIONS" in gpt2_source_text
    assert "NFN_NATIVE_GPT2_STORE_MLP_ACTIVATIONS" in gpt2_source_text
    assert "NFN_NATIVE_GPT_STORE_MLP_BLOCKS" in gpt2_source_text
    assert "NFN_NATIVE_GPT2_STORE_MLP_BLOCKS" in gpt2_source_text
    assert "kDefaultStoredMlpBlocks = 12" in gpt2_source_text
    assert "NFN_NATIVE_GPT_STORE_ATTENTION_ACTIVATIONS" in gpt2_source_text
    assert "NFN_NATIVE_GPT2_STORE_ATTENTION_ACTIVATIONS" in gpt2_source_text
    assert "NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_ACTIVATIONS" in gpt2_source_text
    assert "NFN_NATIVE_GPT2_STORE_PACKED_ATTENTION_ACTIVATIONS" in gpt2_source_text
    assert "NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_BLOCKS" in gpt2_source_text
    assert "NFN_NATIVE_GPT2_STORE_PACKED_ATTENTION_BLOCKS" in gpt2_source_text
    assert "NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LSE" in gpt2_source_text
    assert "NFN_NATIVE_GPT2_STORE_PACKED_ATTENTION_LSE" in gpt2_source_text
    assert "NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LN1_STATS" in gpt2_source_text
    assert "NFN_NATIVE_GPT2_STORE_PACKED_ATTENTION_LN1_STATS" in gpt2_source_text
    assert "kDefaultStoredPackedAttentionBlocks = 12" in gpt2_source_text
    assert "NFN_NATIVE_GPT_STORE_RESIDUAL1_ACTIVATIONS" in gpt2_source_text
    assert "NFN_NATIVE_GPT2_STORE_RESIDUAL1_ACTIVATIONS" in gpt2_source_text
    assert "NFN_NATIVE_GPT_FUSE_RESIDUAL1_STORE" in gpt2_source_text
    assert "NFN_NATIVE_GPT2_FUSE_RESIDUAL1_STORE" in gpt2_source_text
    assert "env_flag_enabled_or_default(store_residual1_activations_env, true)" in gpt2_source_text
    assert "stored_residual1_activation_blocks" in gpt2_source_text
    assert "residual1_activation_store_strategy" in gpt2_source_text
    assert "saved_packed_attention_recompute_needs_float_attention_projection" in gpt2_source_text
    assert "float_attention_projection_output_elided" in gpt2_source_text
    assert "float_mlp_projection_output_elided" in gpt2_source_text
    assert "fused-attention-residual-layernorm-bf16-store" in gpt2_source_text
    assert "separate-float32-to-bf16-store" in gpt2_source_text
    assert "stored_residual1_activation_elements" in gpt2_source_text
    assert "stored_residual1_activation_bytes" in gpt2_source_text
    assert "stored_residual1_activation_store_kernel_launches" in gpt2_source_text
    assert "stored_residual1_activation_restore_kernel_launches" in gpt2_source_text
    assert "bf16-forward-store-recompute-restore" in gpt2_source_text
    assert "NFN_NATIVE_GPT_BF16_RESIDUAL1_LN_BACKWARD" in gpt2_source_text
    assert "NFN_NATIVE_GPT2_BF16_RESIDUAL1_LN_BACKWARD" in gpt2_source_text
    assert "bf16-forward-store-direct-ln-backward" in gpt2_source_text
    assert "residual1_backward_consumer_strategy" in gpt2_source_text
    assert "bf16-layernorm-backward" in gpt2_source_text
    assert "restore-float32-layernorm-backward" in gpt2_source_text
    assert "kDefaultLmHeadRowChunkSize = 32768" in gpt2_source_text
    assert "kDefaultSafeLmHeadRowChunkSize = 49152" in gpt2_source_text
    assert "NFN_NATIVE_GPT_ALLOW_UNSAFE_LM_HEAD_ROW_CHUNK" in gpt2_source_text
    assert "NFN_NATIVE_GPT2_ALLOW_UNSAFE_LM_HEAD_ROW_CHUNK" in gpt2_source_text
    assert "lm_head_row_chunk_safe_cap" in gpt2_source_text
    assert "lm_head_row_chunk_unsafe_override_enabled" in gpt2_source_text
    assert "NFN_NATIVE_GPT_PACKED_ATTENTION_BACKWARD_BATCH_CAP" in kernels_text
    assert "NFN_NATIVE_GPT2_PACKED_ATTENTION_BACKWARD_BATCH_CAP" in kernels_text
    assert "NFN_NATIVE_GPT_PACKED_ATTENTION_DPREP_GRID3D" in kernels_text
    assert "NFN_NATIVE_GPT2_PACKED_ATTENTION_DPREP_GRID3D" in kernels_text
    assert (
        "bool tk_packed_attention_dprep_grid3d_enabled() {\n"
        "  static const bool enabled = []() {\n"
        "    const char* value = std::getenv(\"NFN_NATIVE_GPT_PACKED_ATTENTION_DPREP_GRID3D\");\n"
        "    if (value == nullptr) {\n"
        "      value = std::getenv(\"NFN_NATIVE_GPT2_PACKED_ATTENTION_DPREP_GRID3D\");\n"
        "    }\n"
        "    if (value == nullptr || value[0] == '\\0') {\n"
        "      return false;\n"
        "    }\n"
        "    if (std::strcmp(value, \"0\") == 0"
        in kernels_text
    )
    assert "NFN_NATIVE_GPT_PACKED_ATTENTION_DPREP_WARPS" in kernels_text
    assert "NFN_NATIVE_GPT2_PACKED_ATTENTION_DPREP_WARPS" in kernels_text
    assert "packed_attention_dprep_grid3d_kernel" in kernels_text
    assert "packed_attention_dprep_bf16_grad_grid3d_kernel" in kernels_text
    assert "NFN_NATIVE_GPT_DIRECT_BF16_QKV_GRAD_SCRATCH" in gpt2_source_text
    assert "NFN_NATIVE_GPT_DIRECT_BF16_BLOCK_WEIGHT_INIT" in gpt2_source_text
    assert "NFN_NATIVE_GPT_FUSE_QKV_BIAS_TK_GEMM" in gpt2_source_text
    assert "NFN_NATIVE_GPT_BF16_QKV_DWEIGHT" in gpt2_source_text
    assert "attention_backward_direct_bf16_qkv_grad_scratch_enabled" in gpt2_source_text
    assert "qkv_bias_fused_tk_gemm_enabled" in gpt2_source_text
    assert "block_backward_bf16_qkv_dweight_enabled" in gpt2_source_text
    assert "tk-sm120-packed-qkv-direct-bf16-grad-scratch-handoff" in gpt2_source_text
    assert "grad_qkv_bf16_bits != qkv_bf16_bits" in kernels_text
    assert "kTkPackedAttentionBackwardDefaultMaxBatchPerLaunch = 64" in kernels_text
    assert "nfn_native_tile_scaled_dot_product_attention_store_tk_bf16_float32" in gpt2_source_text
    assert "forward_store_tk_bf16" in gpt2_source_text
    assert "tk-bf16-direct-forward-store-saved-backward" in gpt2_source_text
    assert "launch_tk_attention_forward_store_bf16_float32" in kernels_text
    assert "stored_attention_store_kernel_launches" in gpt2_source_text
    assert "stored_attention_backward_kernel_launches" in gpt2_source_text
    assert "stored_attention_backward_consumer_strategy" in gpt2_source_text
    assert "packed_attention_activation_storage_strategy" in gpt2_source_text
    assert "stored_packed_attention_lse_enabled" in gpt2_source_text
    assert "stored_packed_attention_lse_elements" in gpt2_source_text
    assert "stored_packed_attention_lse_bytes" in gpt2_source_text
    assert "stored_packed_attention_ln1_stats_enabled" in gpt2_source_text
    assert "stored_packed_attention_ln1_stats_blocks" in gpt2_source_text
    assert "stored_packed_attention_ln1_stats_elements" in gpt2_source_text
    assert "stored_packed_attention_ln1_stats_bytes" in gpt2_source_text
    assert "stored_packed_attention_ln1_bf16_enabled" in gpt2_source_text
    assert "stored_packed_attention_ln1_bf16_blocks" in gpt2_source_text
    assert "stored_packed_attention_ln1_bf16_elements" in gpt2_source_text
    assert "stored_packed_attention_ln1_bf16_bytes" in gpt2_source_text
    assert "stored_packed_attention_backward_consumer_strategy" in gpt2_source_text
    assert "recompute_block_from_saved_packed_attention" in gpt2_source_text
    assert "recompute_block_from_saved_attention" in gpt2_source_text
    assert "attention_backward_uses_saved_forward_workspace" in gpt2_source_text
    assert "backward_recompute_mlp_fc_gelu_elided" in gpt2_source_text
    assert "backward_recompute_attention_qkv_sdpa_elided" in gpt2_source_text
    assert "backward_recompute_attention_uses_saved_o" in gpt2_source_text
    assert "backward_recompute_mlp_projection_elided" in gpt2_source_text
    assert "backward_recompute_final_residual_elided" in gpt2_source_text
    assert "NFN_NATIVE_GPT_FULL_ACTIVATION_TAPE" in gpt2_source_text
    assert "full-forward-tape-bf16-stored-packed-attention-and-mlp-direct-backward" in gpt2_source_text
    assert "NFN_NATIVE_LINEAR_BF16_CUBLASLT_EXTRA_LARGE_K" in kernels_text
    assert "NFN_NATIVE_LINEAR_CUBLASLT_WORKSPACE_MB" in kernels_text
    assert "bool next_into(std::uint16_t* tokens, std::uint16_t* targets" in token_shards_header_text
    assert "SequentialTokenBatchSampler::next_into" in token_shards_source_text
    assert "append_contiguous_chunks_into" in token_shards_source_text
    assert "token_ids.device_widen" not in gpt2_source_text
    assert "targets.device_widen" not in gpt2_source_text
    assert "fill_float32_kernel" in kernels_text
    assert "scaled_dot_product_attention_row_float32_kernel" in kernels_text
    assert "NFN_TILE_CUDA_USE_TK_ATTENTION" in kernels_text
    assert "llmc/tk/attention_sm120.cuh" in kernels_text
    assert "launch_tk_attention_forward_float32" in kernels_text
    assert "launch_tk_attention_backward_float32" in kernels_text
    assert "launch_tk_attention_store_forward_workspace_bf16" in kernels_text
    assert "launch_tk_attention_backward_to_qkv_from_saved_bf16_float32" in kernels_text
    assert "cudaMemcpyDeviceToDevice" in kernels_text
    assert "tk-sm120-bf16-bridge" in gpt2_source_text
    assert "attention_forward_tk_launch_count" in gpt2_source_text
    assert "attention_backward_tk_launch_count" in gpt2_source_text
    assert "attention_backward_tk_block_size" in gpt2_source_text
    assert "attention_backward_dprep_timing_us" in gpt2_source_text
    assert "attention_backward_float_hd64_dprep_launch_count" in gpt2_source_text
    assert "attention_backward_tk_timing_us" in gpt2_source_text
    assert "nfn_native_tile_attention_forward_tk_launch_count" in header_text
    assert "nfn_native_tile_attention_backward_tk_launch_count" in header_text
    assert "nfn_native_tile_attention_backward_tk_block_size" in header_text
    assert "nfn_native_tile_attention_backward_float_hd64_dprep_launch_count" in header_text
    assert "nfn_native_tile_attention_backward_dprep_timing_us" in header_text
    assert "nfn_native_tile_attention_backward_tk_timing_us" in header_text
    assert "nfn_native_tile_attention_forward_tk_launch_count" in source_text
    assert "nfn_native_tile_attention_backward_float_hd64_dprep_launch_count" in source_text
    assert "nfn_native_tile_attention_backward_dprep_timing_us" in source_text
    assert "NFN_NATIVE_GPT_ATTENTION_BACKWARD_SECTION_TIMING" in kernels_text
    assert "packed_attention_dprep_bf16_grad_hd64_h12_kernel" in kernels_text
    assert "packed_attention_dprep_float_grad_hd64_h12_kernel" in kernels_text
    assert "NFN_NATIVE_GPT_PACKED_ATTENTION_DPREP_HD64_SPECIALIZED" in kernels_text
    assert "NFN_NATIVE_GPT_PACKED_ATTENTION_DPREP_FLOAT_HD64_SPECIALIZED" in kernels_text
    assert (
        "if (value == nullptr || value[0] == '\\0') {\n"
        "      return true;\n"
        "    }"
    ) in kernels_text
    assert "NFN_TILE_CUDA_TOKEN_WEIGHT_VECTOR4_STRIDED_INIT" in kernels_text
    assert "NFN_NATIVE_GPT_TOKEN_WEIGHT_VECTOR4_STRIDED_INIT" in kernels_text
    assert (
        'env_or_empty_any({"NFN_NATIVE_GPT_TOKEN_WEIGHT_VECTOR4_STRIDED_INIT",\n'
        '                              "NFN_NATIVE_GPT2_TOKEN_WEIGHT_VECTOR4_STRIDED_INIT",\n'
        '                              "NFN_TILE_CUDA_TOKEN_WEIGHT_VECTOR4_STRIDED_INIT"}),\n'
        "            true);"
    ) in gpt2_source_text
    assert "init_gpt2_token_weight_vector4_strided_with_bf16_shadow_float32_kernel" in kernels_text
    assert "init_gpt2_token_weight_vector4_strided_with_bf16_shadow_padded_float32_kernel" in kernels_text
    assert (
        '"device-vector4-strided-power2-deterministic-fused-bf16-shadow-padded-zero"'
        in gpt2_source_text
    )
    assert "launch_init_gpt2_token_weight_vector4_strided_float32" in kernels_text
    assert "NFN_TILE_CUDA_USE_TK_ATTENTION:-1" in script_text
    assert "LLM_KITTENS_ROOT" in script_text
    assert "TK_ROOT" in script_text
    assert 'if [[ "${CUDA_ARCH}" == "sm_120" ]]' in script_text
    assert 'CUDA_ARCH="sm_120a"' in script_text
    assert 'CUDA_ARCH="compute_120a"' in script_text
    assert "sm_120a" in script_text
    assert "--threads=0" in script_text
    assert "-t=0" in script_text
    assert "-forward-unknown-to-host-compiler" in script_text
    assert "-Xcompiler=-Wno-psabi" in script_text
    assert "-Xcompiler=-fno-strict-aliasing" in script_text
    assert "-ftemplate-backtrace-limit=0" in script_text
    assert "-DLLMK_SM120_DPREP_WARPS=3" in script_text
    assert "-DLLMK_SM120_MEMORY_BLOCK_SIZE=1024" in script_text
    assert "-DLLMK_SM120_LAYERNORM_BWD_BLOCKS_PER_SM=1" in script_text
    assert "-DLLMK_SM120_USE_CUBLASLT_GEMM" in script_text
    assert "kGpt2AttentionHeads = 12" in kernels_text
    assert "kGpt2AttentionHeadDim = 64" in kernels_text
    assert "kGpt2AttentionValueChunks" in kernels_text
    assert "for (std::int64_t d_out = 0; d_out < kGpt2AttentionHeadDim; ++d_out)" in kernels_text
    assert "const dim3 row_grid(" in kernels_text
    assert "static_cast<unsigned int>(row_count)," in kernels_text
    assert "static_cast<unsigned int>(kGpt2AttentionValueChunks)" in kernels_text
    assert "row_grid, row_block, 0, stream" in kernels_text
    assert "row-vector-tile-score-reuse" in gpt2_source_text
    assert "scaled_dot_product_attention_backward_q_float32_kernel" in kernels_text
    assert "scaled_dot_product_attention_backward_k_float32_kernel" in kernels_text
    assert "scaled_dot_product_attention_backward_v_float32_kernel" in kernels_text
    assert "scaled_dot_product_attention_backward_q_row_float32_kernel" in kernels_text
    assert "scaled_dot_product_attention_backward_k_row_float32_kernel" in kernels_text
    assert "scaled_dot_product_attention_backward_v_row_float32_kernel" in kernels_text
    assert "scaled_dot_product_attention_backward_query_row_atomic_float32_kernel" in kernels_text
    assert "bf16_heads_to_qkv_float32_kernel" in kernels_text
    assert "launch_scaled_dot_product_attention_backward_from_merged_grad_float32" in kernels_text
    assert "launch_scaled_dot_product_attention_backward_to_qkv_from_merged_grad_float32" in kernels_text
    assert "launch_scaled_dot_product_attention_backward_to_qkv_reuse_forward_from_merged_grad_float32" in kernels_text
    assert "global_norm_clip_scale_float32_kernel" in kernels_text
    assert "scale_inplace_by_device_float32_kernel" in kernels_text
    assert "scaled_residual_add_float32_kernel" in kernels_text
    assert "split_qkv_float32_kernel" in kernels_text
    assert "split_qkv_to_heads_float32_kernel" in kernels_text
    assert "split_qkv_to_heads_add_bias_float32_kernel" in kernels_text
    assert "merge_qkv_float32_kernel" in kernels_text
    assert "merge_heads_to_qkv_float32_kernel" in kernels_text
    assert "reshape_heads_float32_kernel" in kernels_text
    assert "merge_heads_float32_kernel" in kernels_text
    assert "launch_scaled_residual_add_float32" in source_text
    assert "launch_split_qkv_float32" in source_text
    assert "launch_split_qkv_to_heads_float32" in source_text
    assert "launch_split_qkv_to_heads_add_bias_float32" in source_text
    assert "launch_merge_qkv_float32" in source_text
    assert "launch_merge_heads_to_qkv_float32" in source_text
    assert "launch_reshape_heads_float32" in source_text
    assert "launch_merge_heads_float32" in source_text
    assert "launch_evo_mutate_candidates_float32" in source_text
    assert "launch_evo_select_best_loss_float32" in source_text
    assert "launch_evo_adopt_candidate_float32" in source_text
    assert "qkv_forward_layout_strategy" in gpt2_source_text
    assert "fused-split-to-heads" in gpt2_source_text
    assert "qkv_bias_layout_strategy" in gpt2_source_text
    assert "fused-qkv-bias-split-to-heads" in gpt2_source_text
    assert "qkv_backward_layout_strategy" in gpt2_source_text
    assert "fused-heads-to-qkv" in gpt2_source_text
    assert "attention_backward_qkv_bridge_strategy" in gpt2_source_text
    assert "fused-bf16-heads-to-row-qkv" in gpt2_source_text
    assert "attention_backward_grad_layout_strategy" in gpt2_source_text
    assert "attention_backward_strategy" in gpt2_source_text
    assert "attention_backward_score_reuse_dim" in gpt2_source_text
    assert "merged-grad-out-direct" in gpt2_source_text
    assert "mlp_fc_bias_gelu_strategy" in gpt2_source_text
    assert "fused-bias-preactivation-gelu" in gpt2_source_text
    assert "launch_fill_float32" in source_text
    assert "launch_init_gpt2_token_weight_float32" in source_text
    assert "launch_init_gpt2_token_weight_fast_float32" in source_text
    assert "NFN_TILE_CUDA_TOKEN_WEIGHT_INIT_TILE_SHAPE" in kernels_text
    assert "launch_uint16_to_int64" in source_text
    assert "atomic_add_masked" in kernels_text
    assert "vocab <= kTileSize" in kernels_text

    syntax = subprocess.run(
        ["bash", "-n", str(build_script)],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr
    if shutil.which("nvcc") is None:
        pytest.skip("nvcc compiler not available")

    out = tmp_path / "libnfn_native_train_tile_ops.so"
    build = subprocess.run(
        ["bash", str(build_script), str(out)],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert build.returncode == 0, build.stderr
    assert out.exists()

    if shutil.which("c++") is not None:
        native_out = tmp_path / "native"
        build_missing = subprocess.run(
            ["bash", str(root / "tools" / "build_native_missing_trainers.sh"), str(native_out)],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert build_missing.returncode == 0, build_missing.stderr
        gpt2_evo = native_out / "nfn_gpt2_evo_native_train"
        nanogpt = native_out / "nfn_nanogpt_native_train"
        evo_smoke = subprocess.run(
            [str(gpt2_evo), "--smoke-evo-kernels", "--tile-ops-lib", str(out)],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        evo_payload = json.loads(evo_smoke.stdout)
        if evo_smoke.returncode != 0:
            pytest.skip(f"CUDA runtime/device not available for native GPT-2 evo smoke: {evo_payload.get('error')}")
        assert evo_payload["model_family"] == "gpt2-evo"
        assert evo_payload["smoke"] == "evo_kernels"
        assert evo_payload["loaded"] is True
        assert evo_payload["cuda_runtime_loaded"] is True
        assert evo_payload["mutate_kernel_loaded"] is True
        assert evo_payload["select_kernel_loaded"] is True
        assert evo_payload["adopt_kernel_loaded"] is True
        assert evo_payload["elements"] == 4
        assert evo_payload["candidate_count"] == 3
        assert evo_payload["mutation_scale"] == 0
        assert evo_payload["best_index"] == 1
        assert evo_payload["best_loss"] == 1.25
        assert evo_payload["max_adopt_abs_error"] <= 1e-6
        assert evo_payload["passed"] is True
        tile_check = subprocess.run(
            [str(nanogpt), "--check-tile-ops", "--tile-ops-lib", str(out)],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert tile_check.returncode == 0, tile_check.stderr
        payload = json.loads(tile_check.stdout)
        assert payload["loaded"] is True
        assert payload["abi_version"] == 1
        assert payload["all_required_symbols_found"] is True
        assert payload["required_symbol_count"] >= 30
        symbols = {item["name"]: item["found"] for item in payload["symbols"]}
        assert symbols["nfn_native_tile_linear_backward_weight_float32"] is True
        assert symbols["nfn_native_tile_scaled_dot_product_attention_backward_float32"] is True
        assert symbols["nfn_native_tile_fill_float32"] is True
        assert symbols["nfn_native_tile_scaled_residual_add_float32"] is True
        assert symbols["nfn_native_tile_split_qkv_float32"] is True
        assert symbols["nfn_native_tile_merge_qkv_float32"] is True
        tile_smoke = subprocess.run(
            [str(nanogpt), "--smoke-tile-ops", "--tile-ops-lib", str(out)],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        smoke_payload = json.loads(tile_smoke.stdout)
        if tile_smoke.returncode != 0:
            pytest.skip(f"CUDA runtime/device not available for native Tile smoke: {smoke_payload.get('error')}")
        assert smoke_payload["loaded"] is True
        assert smoke_payload["cuda_runtime_loaded"] is True
        assert smoke_payload["kernel"] == "nfn_native_tile_fill_float32"
        assert smoke_payload["kernel_loaded"] is True
        assert smoke_payload["elements"] == 16
        assert smoke_payload["expected_value"] == 3.25
        assert smoke_payload["max_abs_error"] <= 1e-6
        assert smoke_payload["passed"] is True
        optimizer_smoke = subprocess.run(
            [
                str(nanogpt),
                "--smoke-optimizer-step",
                "--tile-ops-lib",
                str(out),
                "--vocab-size",
                "8",
                "--train-seq-len",
                "4",
                "--num-layers",
                "1",
                "--model-dim",
                "4",
                "--num-heads",
                "1",
            ],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        optimizer_payload = json.loads(optimizer_smoke.stdout)
        if optimizer_smoke.returncode != 0:
            pytest.skip(f"CUDA runtime/device not available for native AdamW smoke: {optimizer_payload.get('error')}")
        assert optimizer_payload["loaded"] is True
        assert optimizer_payload["cuda_runtime_loaded"] is True
        assert optimizer_payload["fill_kernel_loaded"] is True
        assert optimizer_payload["optimizer_kernel"] == "nfn_native_tile_adamw_step_float32"
        assert optimizer_payload["optimizer_kernel_loaded"] is True
        assert optimizer_payload["parameter_buffer_count"] == 11
        assert optimizer_payload["total_parameters"] == 260
        assert optimizer_payload["adamw_step_calls"] == 11
        assert optimizer_payload["decay_buffer_count"] == 6
        assert optimizer_payload["no_decay_buffer_count"] == 5
        assert optimizer_payload["decay_elements"] + optimizer_payload["no_decay_elements"] == 260
        assert optimizer_payload["max_param_abs_error"] <= 1e-5
        assert optimizer_payload["max_exp_avg_abs_error"] <= 1e-6
        assert optimizer_payload["max_exp_avg_sq_abs_error"] <= 1e-6
        assert optimizer_payload["passed"] is True
        training_loop_smoke = subprocess.run(
            [
                str(nanogpt),
                "--smoke-training-loop-step",
                "--tile-ops-lib",
                str(out),
                "--vocab-size",
                "8",
                "--train-seq-len",
                "4",
                "--num-layers",
                "1",
                "--model-dim",
                "4",
                "--num-heads",
                "1",
            ],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        training_loop_payload = json.loads(training_loop_smoke.stdout)
        if training_loop_smoke.returncode != 0:
            pytest.skip(
                "CUDA runtime/device not available for native training-loop smoke: "
                f"{training_loop_payload.get('error')}"
            )
        assert training_loop_payload["loaded"] is True
        assert training_loop_payload["cuda_runtime_loaded"] is True
        assert training_loop_payload["parameter_buffer_count"] == 11
        assert training_loop_payload["total_parameters"] == 260
        assert training_loop_payload["partial_count"] == 1
        assert training_loop_payload["adamw_step_calls"] == 11
        assert training_loop_payload["decay_buffer_count"] == 6
        assert training_loop_payload["no_decay_buffer_count"] == 5
        assert "nfn_native_tile_fill_float32" in training_loop_payload["kernels"]
        assert "nfn_native_tile_sumsq_partials_float32" in training_loop_payload["kernels"]
        assert "nfn_native_tile_global_norm_clip_scale_float32" in training_loop_payload["kernels"]
        assert "nfn_native_tile_scale_inplace_by_device_float32" in training_loop_payload["kernels"]
        assert "nfn_native_tile_adamw_step_float32" in training_loop_payload["kernels"]
        assert 0.0 < training_loop_payload["expected_clip_scale"] < 1.0
        assert 0.0 < training_loop_payload["expected_scaled_grad"] < 0.25
        assert training_loop_payload["clip_scale_abs_error"] <= 1e-6
        assert training_loop_payload["max_grad_abs_error"] <= 1e-6
        assert training_loop_payload["max_param_abs_error"] <= 1e-5
        assert training_loop_payload["max_exp_avg_abs_error"] <= 1e-6
        assert training_loop_payload["max_exp_avg_sq_abs_error"] <= 1e-6
        assert training_loop_payload["passed"] is True
        lm_smoke = subprocess.run(
            [str(nanogpt), "--smoke-lm-step", "--tile-ops-lib", str(out)],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        lm_payload = json.loads(lm_smoke.stdout)
        if lm_smoke.returncode != 0:
            pytest.skip(f"CUDA runtime/device not available for native LM-step smoke: {lm_payload.get('error')}")
        assert lm_payload["loaded"] is True
        assert lm_payload["cuda_runtime_loaded"] is True
        assert lm_payload["rows"] == 2
        assert lm_payload["vocab"] == 4
        assert lm_payload["model_dim"] == 4
        assert "nfn_native_tile_token_embedding_float32" in lm_payload["kernels"]
        assert "nfn_native_tile_token_cross_entropy_backward_float32" in lm_payload["kernels"]
        assert "nfn_native_tile_token_embedding_backward_weight_float32" in lm_payload["kernels"]
        assert "nfn_native_tile_adamw_step_float32" in lm_payload["kernels"]
        assert lm_payload["loss_abs_error"] <= 1e-5
        assert lm_payload["max_grad_abs_error"] <= 1e-5
        assert lm_payload["max_weight_abs_error"] <= 1e-5
        assert lm_payload["passed"] is True
        token_shard_dir = tmp_path / "tile_token_shards"
        token_shard_dir.mkdir()
        (token_shard_dir / "fineweb_train_000000.bin").write_bytes(struct.pack("<64H", *[i % 8 for i in range(64)]))
        (token_shard_dir / "fineweb_val_000000.bin").write_bytes(struct.pack("<32H", *[i % 8 for i in range(32)]))
        token_smoke = subprocess.run(
            [
                str(nanogpt),
                "--smoke-token-train-step",
                "--tile-ops-lib",
                str(out),
                "--dataset-alias",
                str(token_shard_dir),
                "--train-seq-len",
                "4",
                "--batch-size",
                "2",
                "--train-batch-tokens",
                "8",
                "--vocab-size",
                "8",
                "--model-dim",
                "4",
                "--num-heads",
                "1",
            ],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        token_payload = json.loads(token_smoke.stdout)
        if token_smoke.returncode != 0:
            pytest.skip(
                f"CUDA runtime/device not available for native sampled-token train-step smoke: {token_payload.get('error')}"
            )
        assert token_payload["loaded"] is True
        assert token_payload["cuda_runtime_loaded"] is True
        assert token_payload["dataset_loaded"] is True
        assert token_payload["batch_loaded"] is True
        assert token_payload["token_shards"]["dataset_path"] == str(token_shard_dir)
        assert token_payload["sample_batch"]["tokens"] == [0, 1, 2, 3, 4, 5, 6, 7]
        assert token_payload["sample_batch"]["targets"] == [1, 2, 3, 4, 5, 6, 7, 0]
        assert token_payload["rows"] == 8
        assert token_payload["vocab"] == 8
        assert token_payload["model_dim"] == 4
        assert "nfn_native_tile_token_embedding_float32" in token_payload["kernels"]
        assert "nfn_native_tile_token_cross_entropy_backward_float32" in token_payload["kernels"]
        assert "nfn_native_tile_token_embedding_backward_weight_float32" in token_payload["kernels"]
        assert "nfn_native_tile_adamw_step_float32" in token_payload["kernels"]
        assert token_payload["loss_abs_error"] <= 1e-5
        assert token_payload["max_grad_abs_error"] <= 2e-5
        assert token_payload["max_weight_abs_error"] <= 2e-5
        assert token_payload["passed"] is True
        token_train = subprocess.run(
            [
                str(nanogpt),
                "--train-token-lm",
                "--tile-ops-lib",
                str(out),
                "--dataset-alias",
                str(token_shard_dir),
                "--train-seq-len",
                "4",
                "--batch-size",
                "2",
                "--train-batch-tokens",
                "8",
                "--vocab-size",
                "8",
                "--model-dim",
                "4",
                "--num-heads",
                "1",
                "--max-steps",
                "2",
                "--eval-every-steps",
                "1",
                "--eval-batches",
                "1",
                "--eval-batch-size",
                "2",
            ],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        token_train_payload = json.loads(token_train.stdout)
        assert token_train.returncode == 0, token_train_payload.get("error")
        assert token_train_payload["status"] == "native-token-lm-trained"
        assert token_train_payload["loaded"] is True
        assert token_train_payload["cuda_runtime_loaded"] is True
        assert token_train_payload["dataset_loaded"] is True
        assert token_train_payload["token_shards"]["dataset_path"] == str(token_shard_dir)
        assert token_train_payload["token_shards"]["batch_read_strategy"] == "contiguous_shard_segments"
        assert token_train_payload["rows"] == 8
        assert token_train_payload["vocab"] == 8
        assert token_train_payload["model_dim"] == 4
        assert token_train_payload["max_steps"] == 2
        assert token_train_payload["eval_every_steps"] == 1
        assert token_train_payload["eval_batches"] == 1
        assert token_train_payload["eval_batch_size"] == 2
        assert token_train_payload["steps_completed"] == 2
        assert token_train_payload["tokens_processed"] == 16
        assert token_train_payload["final_loss_mean"] > 0.0
        assert token_train_payload["validation"]["eval_count"] == 2
        assert [item["step"] for item in token_train_payload["validation"]["losses"]] == [1, 2]
        assert all(item["tokens"] == 8 for item in token_train_payload["validation"]["losses"])
        assert all(item["loss_mean"] > 0.0 for item in token_train_payload["validation"]["losses"])
        assert "nfn_native_tile_token_embedding_float32" in token_train_payload["kernels"]
        assert "nfn_native_tile_token_cross_entropy_backward_float32" in token_train_payload["kernels"]
        assert "nfn_native_tile_adamw_step_float32" in token_train_payload["kernels"]
        assert token_train_payload["passed"] is True
        embedding_norm_smoke = subprocess.run(
            [
                str(nanogpt),
                "--smoke-embedding-norm-step",
                "--tile-ops-lib",
                str(out),
                "--dataset-alias",
                str(token_shard_dir),
                "--train-seq-len",
                "4",
                "--batch-size",
                "2",
                "--train-batch-tokens",
                "8",
                "--vocab-size",
                "8",
                "--model-dim",
                "4",
                "--num-heads",
                "1",
            ],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        embedding_norm_payload = json.loads(embedding_norm_smoke.stdout)
        if embedding_norm_smoke.returncode != 0:
            pytest.skip(
                "CUDA runtime/device not available for native sampled embedding/norm train-step smoke: "
                f"{embedding_norm_payload.get('error')}"
            )
        assert embedding_norm_payload["loaded"] is True
        assert embedding_norm_payload["cuda_runtime_loaded"] is True
        assert embedding_norm_payload["dataset_loaded"] is True
        assert embedding_norm_payload["batch_loaded"] is True
        assert embedding_norm_payload["sample_batch"]["tokens"] == [0, 1, 2, 3, 4, 5, 6, 7]
        assert embedding_norm_payload["sample_batch"]["targets"] == [1, 2, 3, 4, 5, 6, 7, 0]
        assert embedding_norm_payload["rows"] == 8
        assert embedding_norm_payload["vocab"] == 8
        assert embedding_norm_payload["model_dim"] == 4
        assert "nfn_native_tile_absolute_position_embedding_float32" in embedding_norm_payload["kernels"]
        assert "nfn_native_tile_scaled_residual_add_float32" in embedding_norm_payload["kernels"]
        assert "nfn_native_tile_layer_norm_float32" in embedding_norm_payload["kernels"]
        assert "nfn_native_tile_layer_norm_backward_input_float32" in embedding_norm_payload["kernels"]
        assert "nfn_native_tile_absolute_position_embedding_backward_float32" in embedding_norm_payload["kernels"]
        assert "nfn_native_tile_adamw_step_float32" in embedding_norm_payload["kernels"]
        assert embedding_norm_payload["max_residual_abs_error"] <= 1e-6
        assert embedding_norm_payload["max_norm_abs_error"] <= 1e-6
        assert embedding_norm_payload["loss_abs_error"] <= 1e-5
        assert embedding_norm_payload["max_token_grad_abs_error"] <= 1e-6
        assert embedding_norm_payload["max_position_grad_abs_error"] <= 1e-6
        assert embedding_norm_payload["max_ln_grad_abs_error"] <= 1e-6
        assert embedding_norm_payload["max_token_weight_abs_error"] <= 1e-5
        assert embedding_norm_payload["max_position_weight_abs_error"] <= 1e-5
        assert embedding_norm_payload["max_ln_weight_abs_error"] <= 1e-5
        assert embedding_norm_payload["max_ln_bias_abs_error"] <= 1e-6
        assert embedding_norm_payload["passed"] is True
        qkv_layout_smoke = subprocess.run(
            [str(nanogpt), "--smoke-qkv-layout-step", "--tile-ops-lib", str(out)],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        qkv_layout_payload = json.loads(qkv_layout_smoke.stdout)
        if qkv_layout_smoke.returncode != 0:
            pytest.skip(
                f"CUDA runtime/device not available for native QKV layout smoke: {qkv_layout_payload.get('error')}"
            )
        assert qkv_layout_payload["loaded"] is True
        assert qkv_layout_payload["cuda_runtime_loaded"] is True
        assert qkv_layout_payload["rows"] == 2
        assert qkv_layout_payload["model_dim"] == 4
        assert qkv_layout_payload["qkv_elements"] == 24
        assert "nfn_native_tile_split_qkv_float32" in qkv_layout_payload["kernels"]
        assert "nfn_native_tile_merge_qkv_float32" in qkv_layout_payload["kernels"]
        assert qkv_layout_payload["max_q_abs_error"] <= 1e-6
        assert qkv_layout_payload["max_k_abs_error"] <= 1e-6
        assert qkv_layout_payload["max_v_abs_error"] <= 1e-6
        assert qkv_layout_payload["max_merged_abs_error"] <= 1e-6
        assert qkv_layout_payload["passed"] is True
        fused_qkv_attention_smoke = subprocess.run(
            [str(nanogpt), "--smoke-fused-qkv-attention-step", "--tile-ops-lib", str(out)],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        fused_qkv_attention_payload = json.loads(fused_qkv_attention_smoke.stdout)
        if fused_qkv_attention_smoke.returncode != 0:
            pytest.skip(
                "CUDA runtime/device not available for native fused-QKV attention smoke: "
                f"{fused_qkv_attention_payload.get('error')}"
            )
        assert fused_qkv_attention_payload["loaded"] is True
        assert fused_qkv_attention_payload["cuda_runtime_loaded"] is True
        assert fused_qkv_attention_payload["batch"] == 1
        assert fused_qkv_attention_payload["heads"] == 1
        assert fused_qkv_attention_payload["seq"] == 2
        assert fused_qkv_attention_payload["model_dim"] == 4
        assert "nfn_native_tile_split_qkv_float32" in fused_qkv_attention_payload["kernels"]
        assert "nfn_native_tile_scaled_dot_product_attention_float32" in fused_qkv_attention_payload["kernels"]
        assert "nfn_native_tile_scaled_dot_product_attention_backward_float32" in fused_qkv_attention_payload["kernels"]
        assert "nfn_native_tile_merge_qkv_float32" in fused_qkv_attention_payload["kernels"]
        assert "nfn_native_tile_linear_backward_weight_float32" in fused_qkv_attention_payload["kernels"]
        assert "nfn_native_tile_adamw_step_float32" in fused_qkv_attention_payload["kernels"]
        assert fused_qkv_attention_payload["max_q_abs_error"] <= 1e-4
        assert fused_qkv_attention_payload["max_k_abs_error"] <= 1e-4
        assert fused_qkv_attention_payload["max_v_abs_error"] <= 1e-4
        assert fused_qkv_attention_payload["max_attn_abs_error"] <= 1e-4
        assert fused_qkv_attention_payload["max_out_abs_error"] <= 1e-4
        assert fused_qkv_attention_payload["max_grad_x_abs_error"] <= 1e-4
        assert fused_qkv_attention_payload["max_grad_qkv_weight_abs_error"] <= 1e-5
        assert fused_qkv_attention_payload["max_grad_out_weight_abs_error"] <= 1e-4
        assert fused_qkv_attention_payload["max_qkv_weight_abs_error"] <= 1e-5
        assert fused_qkv_attention_payload["max_out_weight_abs_error"] <= 1e-5
        assert fused_qkv_attention_payload["passed"] is True
        transformer_block_smoke = subprocess.run(
            [str(nanogpt), "--smoke-transformer-block-step", "--tile-ops-lib", str(out)],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        transformer_block_payload = json.loads(transformer_block_smoke.stdout)
        if transformer_block_smoke.returncode != 0:
            pytest.skip(
                "CUDA runtime/device not available for native transformer-block smoke: "
                f"{transformer_block_payload.get('error')}"
            )
        assert transformer_block_payload["loaded"] is True
        assert transformer_block_payload["cuda_runtime_loaded"] is True
        assert transformer_block_payload["batch"] == 1
        assert transformer_block_payload["heads"] == 1
        assert transformer_block_payload["seq"] == 2
        assert transformer_block_payload["model_dim"] == 4
        assert transformer_block_payload["hidden_dim"] == 8
        assert transformer_block_payload["weight_update_count"] == 8
        assert "nfn_native_tile_layer_norm_float32" in transformer_block_payload["kernels"]
        assert "nfn_native_tile_linear_float32" in transformer_block_payload["kernels"]
        assert "nfn_native_tile_split_qkv_float32" in transformer_block_payload["kernels"]
        assert "nfn_native_tile_scaled_dot_product_attention_float32" in transformer_block_payload["kernels"]
        assert "nfn_native_tile_scaled_residual_add_float32" in transformer_block_payload["kernels"]
        assert "nfn_native_tile_gelu_float32" in transformer_block_payload["kernels"]
        assert "nfn_native_tile_linear_backward_weight_float32" in transformer_block_payload["kernels"]
        assert "nfn_native_tile_linear_backward_input_float32" in transformer_block_payload["kernels"]
        assert "nfn_native_tile_gelu_backward_float32" in transformer_block_payload["kernels"]
        assert "nfn_native_tile_layer_norm_backward_affine_float32" in transformer_block_payload["kernels"]
        assert "nfn_native_tile_layer_norm_backward_input_float32" in transformer_block_payload["kernels"]
        assert "nfn_native_tile_gradient_accumulate_float32" in transformer_block_payload["kernels"]
        assert "nfn_native_tile_scaled_dot_product_attention_backward_float32" in transformer_block_payload["kernels"]
        assert "nfn_native_tile_merge_qkv_float32" in transformer_block_payload["kernels"]
        assert "nfn_native_tile_adamw_step_float32" in transformer_block_payload["kernels"]
        assert transformer_block_payload["forward_finite"] is True
        assert transformer_block_payload["backward_finite"] is True
        assert transformer_block_payload["optimizer_finite"] is True
        assert transformer_block_payload["residual2_max_abs"] > 0.0
        assert transformer_block_payload["grad_x_max_abs"] > 0.0
        assert transformer_block_payload["grad_qkv_weight_max_abs"] > 0.0
        assert transformer_block_payload["grad_attn_proj_weight_max_abs"] > 0.0
        assert transformer_block_payload["grad_fc_weight_max_abs"] > 0.0
        assert transformer_block_payload["grad_mlp_proj_weight_max_abs"] > 0.0
        assert transformer_block_payload["qkv_weight_max_delta"] > 0.0
        assert transformer_block_payload["attn_proj_weight_max_delta"] > 0.0
        assert transformer_block_payload["fc_weight_max_delta"] > 0.0
        assert transformer_block_payload["mlp_proj_weight_max_delta"] > 0.0
        assert transformer_block_payload["passed"] is True
        mlp_smoke = subprocess.run(
            [str(nanogpt), "--smoke-mlp-step", "--tile-ops-lib", str(out)],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        mlp_payload = json.loads(mlp_smoke.stdout)
        if mlp_smoke.returncode != 0:
            pytest.skip(f"CUDA runtime/device not available for native MLP-step smoke: {mlp_payload.get('error')}")
        assert mlp_payload["loaded"] is True
        assert mlp_payload["cuda_runtime_loaded"] is True
        assert mlp_payload["rows"] == 2
        assert mlp_payload["model_dim"] == 4
        assert mlp_payload["hidden_dim"] == 8
        assert "nfn_native_tile_linear_float32" in mlp_payload["kernels"]
        assert "nfn_native_tile_gelu_float32" in mlp_payload["kernels"]
        assert "nfn_native_tile_gelu_backward_float32" in mlp_payload["kernels"]
        assert "nfn_native_tile_linear_backward_weight_float32" in mlp_payload["kernels"]
        assert "nfn_native_tile_adamw_step_float32" in mlp_payload["kernels"]
        assert mlp_payload["max_out_abs_error"] <= 1e-4
        assert mlp_payload["max_grad_x_abs_error"] <= 1e-4
        assert mlp_payload["max_fc_grad_abs_error"] <= 1e-4
        assert mlp_payload["max_proj_grad_abs_error"] <= 1e-5
        assert mlp_payload["max_fc_weight_abs_error"] <= 1e-5
        assert mlp_payload["max_proj_weight_abs_error"] <= 1e-5
        assert mlp_payload["passed"] is True
        attention_smoke = subprocess.run(
            [str(nanogpt), "--smoke-attention-step", "--tile-ops-lib", str(out)],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        attention_payload = json.loads(attention_smoke.stdout)
        if attention_smoke.returncode != 0:
            pytest.skip(
                f"CUDA runtime/device not available for native attention-step smoke: {attention_payload.get('error')}"
            )
        assert attention_payload["loaded"] is True
        assert attention_payload["cuda_runtime_loaded"] is True
        assert attention_payload["batch"] == 1
        assert attention_payload["heads"] == 1
        assert attention_payload["seq"] == 2
        assert attention_payload["model_dim"] == 4
        assert "nfn_native_tile_scaled_dot_product_attention_float32" in attention_payload["kernels"]
        assert "nfn_native_tile_scaled_dot_product_attention_backward_float32" in attention_payload["kernels"]
        assert "nfn_native_tile_linear_backward_weight_float32" in attention_payload["kernels"]
        assert attention_payload["max_attn_abs_error"] <= 1e-4
        assert attention_payload["max_out_abs_error"] <= 1e-4
        assert attention_payload["max_grad_q_weight_abs_error"] <= 1e-6
        assert attention_payload["max_grad_k_weight_abs_error"] <= 1e-6
        assert attention_payload["max_grad_v_weight_abs_error"] <= 1e-4
        assert attention_payload["max_grad_out_weight_abs_error"] <= 1e-4
        assert attention_payload["max_q_weight_abs_error"] <= 1e-5
        assert attention_payload["max_k_weight_abs_error"] <= 1e-5
        assert attention_payload["max_v_weight_abs_error"] <= 1e-5
        assert attention_payload["max_out_weight_abs_error"] <= 1e-5
        assert attention_payload["passed"] is True

    nm = shutil.which("nm")
    if nm is not None:
        symbols = subprocess.run(
            [nm, "-D", str(out)],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert symbols.returncode == 0, symbols.stderr
        exported = symbols.stdout
        assert "nfn_native_tile_fill_float32" in exported
        assert "nfn_native_tile_init_gpt2_token_weight_float32" in exported
        assert "nfn_native_tile_init_gpt2_token_weight_fast_float32" in exported
        assert "nfn_native_tile_init_gpt2_token_weight_fast_with_bf16_shadow_float32" in exported
        assert "nfn_native_tile_copy_float32" in exported
        assert "nfn_native_tile_uint16_to_int64" in exported
        assert "nfn_native_tile_float32_to_bf16_bits" in exported
        assert "nfn_native_tile_bf16_bits_to_float32" in exported
        assert "nfn_native_tile_float32_to_nvfp4_packed" in exported
        assert "nfn_native_tile_nvfp4_packed_to_float32" in exported
        assert "nfn_native_tile_bf16_bits_add_bias_inplace_float32" in exported
        assert (
            "nfn_native_tile_token_cross_entropy_backward_loss_inplace_strided_bf16_bits_u16_targets"
            in exported
        )
        assert (
            "nfn_native_tile_lm_head_classifier_backward_loss_inplace_strided_no_pad_zero_bf16_bits_u16_targets"
            in exported
        )
        assert (
            "nfn_native_tile_lm_head_classifier_backward_row_losses_inplace_strided_no_pad_zero_bf16_bits_u16_targets"
            in exported
        )
        assert "nfn_native_tile_trainer_linear_cublaslt_grouped_matmul_probe_status" in exported
        assert "nfn_native_tile_sum_accumulate_float32" in exported
        assert (
            "nfn_native_tile_lm_head_classifier_backward_inplace_strided_no_pad_zero_bf16_bits_u16_targets_with_workspace"
            in exported
        )
        assert "nfn_native_tile_lm_head_classifier_chunk_launch_count" in exported
        assert "nfn_native_tile_store_mlp_activations_bf16_float32" in exported
        assert "nfn_native_tile_restore_mlp_activations_bf16_float32" in exported
        assert "nfn_native_tile_float32_to_bf16_bits_many" in exported
        assert "nfn_native_tile_linear_bf16_float32" in exported
        assert "nfn_native_tile_linear_weight_bf16_float32" in exported
        assert "nfn_native_tile_linear_bf16_output_float32" in exported
        assert "nfn_native_tile_linear_weight_bf16_output_float32" in exported
        assert "nfn_native_tile_linear_bf16_input_float_weight_bf16_output_float32" in exported
        assert "nfn_native_tile_linear_bf16_input_bits_float32" in exported
        assert "nfn_native_tile_linear_bf16_input_weight_bf16_float32" in exported
        assert "nfn_native_tile_linear_bf16_gelu_bf16_float32" in exported
        assert "nfn_native_tile_linear_weight_bf16_gelu_bf16_float32" in exported
        assert "nfn_native_tile_linear_bf16_input_weight_bf16_gelu_bf16_float32" in exported
        assert "nfn_native_tile_linear_backward_input_bf16_float32" in exported
        assert "nfn_native_tile_linear_backward_input_weight_bf16_float32" in exported
        assert "nfn_native_tile_linear_backward_input_bf16_bits_float32" in exported
        assert "nfn_native_tile_linear_backward_input_dgelu_weight_bf16_bits_float32" in exported
        assert "nfn_native_tile_linear_backward_input_dgelu_weight_bf16_bits_only_float32" in exported
        assert "nfn_native_tile_linear_backward_input_dgelu_bf16_bits_weight_bf16_bits_only_float32" in exported
        assert "nfn_native_tile_gelu_add_bias_bf16_act_float32" in exported
        assert "nfn_native_tile_dropout_forward_float32" in exported
        assert "nfn_native_tile_dropout_backward_float32" in exported
        assert "nfn_native_tile_trainer_linear_stats_reset" in exported
        assert "nfn_native_tile_trainer_linear_bf16_cache_reset" in exported
        assert "nfn_native_tile_trainer_linear_bf16_gemm_count" in exported
        assert "nfn_native_tile_trainer_linear_bf16_gemm_fast16bf_request_count" in exported
        assert "nfn_native_tile_trainer_linear_tk_gemm_count" in exported
        assert "nfn_native_tile_trainer_linear_tk_float_out_gemm_count" in exported
        assert "nfn_native_tile_trainer_linear_tk_dweight_gemm_count" in exported
        assert "nfn_native_tile_trainer_linear_tk_dgelu_dinput_gemm_count" in exported
        assert "nfn_native_tile_trainer_linear_cublaslt_gemm_count" in exported
        assert "nfn_native_tile_trainer_linear_cublaslt_bgrad_gemm_count" in exported
        assert "nfn_native_tile_trainer_linear_cublaslt_bgrad_direct_write_count" in exported
        assert "nfn_native_tile_trainer_linear_cublaslt_bgrad_accumulate_count" in exported
        assert "nfn_native_tile_trainer_linear_sgemm_count" in exported
        assert "nfn_native_tile_trainer_bf16_to_f32_vec4_count" in exported
        assert "nfn_native_tile_trainer_linear_bf16_a_pack_count" in exported
        assert "nfn_native_tile_trainer_linear_bf16_a_cache_hit_count" in exported
        assert "nfn_native_tile_trainer_linear_bf16_cache_reset_count" in exported
        assert "nfn_native_tile_trainer_linear_bf16_workspace_allocation_count" in exported
        assert "nfn_native_tile_trainer_linear_bf16_cached_a_capacity" in exported
        assert "nfn_native_tile_trainer_linear_bf16_cache_entry_count" in exported
        assert "nfn_native_tile_trainer_linear_cublas_grouped_bf16_gemm_probe_status" in exported
        assert "nfn_native_tile_trainer_linear_cublas_prewarm" in exported
        assert "nfn_native_tile_trainer_linear_shape_stats_count" in exported
        assert "nfn_native_tile_trainer_linear_shape_stats_entry" in exported
        assert "nfn_native_tile_trainer_linear_cublaslt_plan_cache_count" in exported
        assert "nfn_native_tile_trainer_linear_cublaslt_plan_cache_entry" in exported
        assert "nfn_native_tile_adamw_step_with_device_scale_float32" in exported
        assert "nfn_native_tile_global_norm_clip_scale_float32" in exported
        assert "nfn_native_tile_scale_inplace_by_device_float32" in exported
        assert "nfn_native_tile_linear_backward_input_float32" in exported
        assert "nfn_native_tile_linear_backward_weight_float32" in exported
        assert "nfn_native_tile_linear_backward_weight_accumulate_float32" in exported
        assert "nfn_native_tile_linear_backward_weight_accumulate_bf16_float32" in exported
        assert "nfn_native_tile_linear_backward_weight_accumulate_bf16_bits_float32" in exported
        assert "nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_float32" in exported
        assert "nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_bits_float32" in exported
        assert "nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_bits_float32_beta" in exported
        assert "nfn_native_tile_linear_backward_weight_accumulate_float32_bf16_bits" in exported
        assert "nfn_native_tile_linear_backward_weight_accumulate_bf16_bits_bf16_bits_float32" in exported
        assert "nfn_native_tile_linear_backward_weight_accumulate_bf16_bits_bf16_bits_float32_beta" in exported
        assert "nfn_native_tile_linear_backward_weight_bias_accumulate_float32_bf16_bits" in exported
        assert "nfn_native_tile_linear_backward_weight_bias_accumulate_float32_bf16_bits_beta" in exported
        assert "nfn_native_tile_linear_backward_bias_float32" in exported
        assert "nfn_native_tile_linear_backward_bias_accumulate_float32" in exported
        assert "nfn_native_tile_evo_mutate_candidates_float32" in exported
        assert "nfn_native_tile_evo_select_best_loss_float32" in exported
        assert "nfn_native_tile_evo_adopt_candidate_float32" in exported
        assert "nfn_native_tile_scaled_residual_add_float32" in exported
        assert "nfn_native_tile_linear_bias_residual_add_bf16_linear_float32" in exported
        assert "nfn_native_tile_linear_bias_residual_add_bf16_linear_bf16_residual_float32" in exported
        assert "nfn_native_tile_linear_bias_residual_layer_norm_with_stats_bf16_linear_float32" in exported
        assert (
            "nfn_native_tile_linear_bias_residual_layer_norm_with_stats_bf16_linear_bf16_residual_float32"
            in exported
        )
        assert (
            "nfn_native_tile_linear_bias_residual_layer_norm_with_stats_bf16_linear_bf16_residual_bf16_norm_float32"
            in exported
        )
        assert "nfn_native_tile_gelu_float32" in exported
        assert "nfn_native_tile_gelu_backward_float32" in exported
        assert "nfn_native_tile_gelu_backward_inplace_bf16_bits_float32" in exported
        assert "nfn_native_tile_dropout_forward_float32" in exported
        assert "nfn_native_tile_dropout_backward_float32" in exported
        assert "nfn_native_tile_token_embedding_float32" in exported
        assert "nfn_native_tile_token_embedding_backward_weight_float32" in exported
        assert "nfn_native_tile_absolute_position_embedding_float32" in exported
        assert "nfn_native_tile_absolute_position_embedding_backward_float32" in exported
        assert "nfn_native_tile_absolute_position_embedding_backward_accumulate_float32" in exported
        assert "nfn_native_tile_layer_norm_float32" in exported
        assert "nfn_native_tile_layer_norm_with_stats_float32" in exported
        assert "nfn_native_tile_layer_norm_backward_input_float32" in exported
        assert "nfn_native_tile_layer_norm_backward_input_with_stats_float32" in exported
        assert "nfn_native_tile_layer_norm_backward_input_residual_add_with_stats_float32" in exported
        assert "nfn_native_tile_layer_norm_backward_input_residual_add_with_stats_bf16_bits_float32" in exported
        assert "nfn_native_tile_layer_norm_backward_affine_float32" in exported
        assert "nfn_native_tile_layer_norm_backward_affine_accumulate_float32" in exported
        assert "nfn_native_tile_layer_norm_backward_affine_accumulate_with_stats_float32" in exported
        assert "nfn_native_tile_layer_norm_backward_affine_accumulate_with_stats_bf16_bits_float32" in exported
        assert "nfn_native_tile_rms_norm_float32" in exported
        assert "nfn_native_tile_rms_norm_backward_input_float32" in exported
        assert "nfn_native_tile_softmax_lastdim_float32" in exported
        assert "nfn_native_tile_token_cross_entropy_partials_bf16_bits" in exported
        assert "nfn_native_tile_token_cross_entropy_partials_strided_float32" in exported
        assert "nfn_native_tile_token_cross_entropy_partials_strided_bf16_bits" in exported
        assert "nfn_native_tile_masked_token_cross_entropy_partials_float32" in exported
        assert "nfn_native_tile_token_cross_entropy_backward_float32" in exported
        assert "nfn_native_tile_masked_token_cross_entropy_backward_float32" in exported
        assert "nfn_native_tile_token_cross_entropy_workspace_allocation_count" in exported
        assert "nfn_native_tile_token_cross_entropy_workspace_row_capacity" in exported
        assert "nfn_native_tile_token_cross_entropy_backward_with_workspace_float32" in exported
        assert "nfn_native_tile_token_cross_entropy_backward_inplace_with_workspace_float32" in exported
        assert "nfn_native_tile_token_cross_entropy_backward_inplace_bf16_bits_with_workspace" in exported
        assert "nfn_native_tile_token_cross_entropy_backward_inplace_strided_with_workspace_float32" in exported
        assert "nfn_native_tile_token_cross_entropy_backward_inplace_strided_bf16_bits_with_workspace" in exported
        assert "nfn_native_tile_masked_token_cross_entropy_backward_with_workspace_float32" in exported
        assert "nfn_native_tile_scaled_dot_product_attention_float32" in exported
        assert "nfn_native_tile_scaled_dot_product_attention_packed_qkv_bf16_float32" in exported
        assert "nfn_native_tile_scaled_dot_product_attention_packed_qkv_store_lse_bf16_float32" in exported
        assert (
            "nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_from_merged_grad_float32"
            in exported
        )
        assert (
            "nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_from_saved_lse_bf16_from_merged_grad_float32"
            in exported
        )
        assert (
            "nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_bf16_bits_from_merged_grad_float32"
            in exported
        )
        assert (
            "nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_bf16_bits_from_saved_lse_bf16_from_merged_grad_float32"
            in exported
        )
        assert "nfn_native_tile_scaled_dot_product_attention_backward_float32" in exported
        assert (
            "nfn_native_tile_scaled_dot_product_attention_backward_to_qkv_reuse_forward_from_merged_grad_float32"
            in exported
        )
        assert "nfn_native_tile_attention_tk_store_forward_workspace_bf16" in exported
        assert (
            "nfn_native_tile_scaled_dot_product_attention_backward_to_qkv_from_saved_tk_bf16_from_merged_grad_float32"
            in exported
        )


def test_native_tile_token_weight_bf16_pattern_initializer_is_opt_in() -> None:
    root = Path(__file__).resolve().parents[1]
    kernels_text = (root / "neuralfn" / "csrc" / "tile_cuda" / "kernels.cu").read_text()
    assert "NFN_NATIVE_GPT_TOKEN_WEIGHT_BF16_PATTERN_INIT" in kernels_text
    assert "NFN_NATIVE_GPT2_TOKEN_WEIGHT_BF16_PATTERN_INIT" in kernels_text
    assert "NFN_TILE_CUDA_TOKEN_WEIGHT_BF16_PATTERN_INIT" in kernels_text
    pattern_env_helper = kernels_text[
        kernels_text.index("bool token_weight_bf16_pattern_init_enabled()") :
        kernels_text.index("bool token_weight_vector4_strided_init_enabled()")
    ]
    assert "return false;" in pattern_env_helper
    vector4_dispatch = kernels_text[
        kernels_text.index("if (token_weight_bf16_pattern_init_enabled())") :
        kernels_text.index("constexpr int kTokenInitTileSize", kernels_text.index("if (token_weight_bf16_pattern_init_enabled())"))
    ]
    assert "init_gpt2_token_weight_vector4_with_bf16_shadow_float32_kernel" in vector4_dispatch
    assert "init_gpt2_token_weight_vector4_with_bf16_shadow_convert_float32_kernel" in vector4_dispatch
    pattern_kernel = kernels_text[
        kernels_text.index("init_gpt2_token_weight_vector4_with_bf16_shadow_float32_kernel") :
        kernels_text.index("init_gpt2_token_weight_vector4_with_bf16_shadow_convert_float32_kernel")
    ]
    assert "gpt2_token_weight_init_bf16_pattern4(bucket)" in pattern_kernel
    assert "bf16_bits_from_float(value0)" not in pattern_kernel
    convert_kernel = kernels_text[
        kernels_text.index("init_gpt2_token_weight_vector4_with_bf16_shadow_convert_float32_kernel") :
        kernels_text.index("init_gpt2_token_weight_vector4_strided_float32_kernel")
    ]
    assert "const float4 pattern = gpt2_token_weight_init_float_pattern4(bucket)" in convert_kernel
    assert "bf16_bits_from_float(pattern.x)" in convert_kernel
    assert "bf16_bits_from_float(value0)" not in convert_kernel


def test_cli_install_script_help_and_no_native_mode() -> None:
    root = Path(__file__).resolve().parents[1]

    help_proc = subprocess.run(
        ["bash", str(root / "cli" / "install.sh"), "--help"],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert help_proc.returncode == 0, help_proc.stderr
    assert "--no-native" in help_proc.stdout
    assert "native C++ bindings" in help_proc.stdout
    assert "raw CUDA Tile trainer ops shared library" in help_proc.stdout
    assert "per-family native trainer entrypoints" in help_proc.stdout


def test_top_level_nfn_train_defaults_to_native_gpt_without_base_model(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    native_cli = tmp_path / "nfn_gpt_native_train"
    native_cli.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n", encoding="utf-8")
    native_cli.chmod(0o755)

    env = os.environ.copy()
    env["NFN_NATIVE_GPT_CLI"] = str(native_cli)
    proc = subprocess.run(
        [
            sys.executable,
            str(root / "cli" / "nfn.py"),
            "train",
            "--tinystories",
            "--native-cuda-print-command",
            "--native-cuda-no-checkpoint",
            "--native-cuda-startup-only",
        ],
        cwd=root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "--model-family\ngpt" in proc.stdout
    assert "--train-transformer-lm" in proc.stdout
    assert "--tinystories" in proc.stdout
    assert "--no-checkpoint" in proc.stdout
    assert "--startup-only" in proc.stdout
    assert "--native-cuda-no-checkpoint" not in proc.stdout
    assert "--native-cuda-startup-only" not in proc.stdout
    assert "TorchTrainer path" not in proc.stderr


def test_paired_speed_gates_native_runtime_contract(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "nfn_gpt_native_train"
    baseline.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' '{\"status\":\"baseline\",\"train_loop_wall_ms\":1,\"steps_completed\":1}'\n",
        encoding="utf-8",
    )
    candidate.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' '{\"status\":\"native-transformer-lm-trained\","
        "\"graph_editor_tensor_flow\":false,\"torch_required\":false,"
        "\"optimized_kernel_contract_passed\":true,"
        "\"train_loss_host_d2h_count\":0,"
        "\"train_loop_wall_ms\":1,\"steps_completed\":1}'\n",
        encoding="utf-8",
    )
    baseline.chmod(0o755)
    candidate.chmod(0o755)

    base_cmd = [
        sys.executable,
        str(root / "tools" / "paired_kernel_speed.py"),
        "--baseline",
        str(baseline),
        "--candidate",
        str(candidate),
        "--samples",
        "1",
        "--warmup",
        "0",
        "--cuda-visible-devices",
        "",
    ]
    passing = subprocess.run(
        base_cmd,
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert passing.returncode == 0, passing.stderr
    assert "native_runtime_contract_gate" in passing.stdout
    assert "graph_editor_tensor_flow: expected=false observed=false passed=true" in passing.stdout
    assert "torch_required: expected=false observed=false passed=true" in passing.stdout
    assert "optimized_kernel_contract_passed: expected=true observed=true passed=true" in passing.stdout
    assert "train_loss_host_d2h_count: expected=0 observed=0 passed=true" in passing.stdout

    candidate.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' '{\"status\":\"native-transformer-lm-trained\","
        "\"graph_editor_tensor_flow\":true,\"torch_required\":false,"
        "\"optimized_kernel_contract_passed\":true,"
        "\"train_loss_host_d2h_count\":1,"
        "\"train_loop_wall_ms\":1,\"steps_completed\":1}'\n",
        encoding="utf-8",
    )
    candidate.chmod(0o755)
    failing = subprocess.run(
        base_cmd,
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert failing.returncode == 1
    assert "native runtime contract gate failed" in failing.stderr
    assert "graph_editor_tensor_flow: expected=false observed=true passed=false" in failing.stdout
    assert "train_loss_host_d2h_count: expected=0 observed=1 passed=false" in failing.stdout


def test_canonical_infer_gpt_wrapper_reports_generic_program_name() -> None:
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [
            sys.executable,
            str(root / "cli" / "scripts" / "infer_gpt.py"),
            "--help",
        ],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "usage: infer_gpt.py" in proc.stdout
    assert "usage: infer_gpt2.py" not in proc.stdout
    assert "Print native GPT checkpoint metadata" in proc.stdout


def test_top_level_nfn_train_prefers_direct_family_native_cli(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    family_cli = tmp_path / "nfn_gpt2_evo_native_train"
    family_cli.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n", encoding="utf-8")
    family_cli.chmod(0o755)

    env = os.environ.copy()
    env.pop("NFN_NATIVE_TRAIN_CLI", None)
    env["NFN_NATIVE_GPT2_EVO_CLI"] = str(family_cli)
    proc = subprocess.run(
        [
            sys.executable,
            str(root / "cli" / "nfn.py"),
            "train",
            "--base-model",
            "gpt2-evo",
            "--tinystories",
            "--native-cuda-print-command",
            "--native-cuda-dry-run",
            "--native-cuda-startup-only",
            "--eval-every-steps",
            "1000",
        ],
        cwd=root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "--tinystories" in proc.stdout
    assert "--print-command" in proc.stdout
    assert "--dry-run" in proc.stdout
    assert "--startup-only" in proc.stdout
    assert "--eval-every-steps\n1000" in proc.stdout
    assert "--base-model" not in proc.stdout
    assert "--model-family" not in proc.stdout
    assert "TorchTrainer path" not in proc.stderr


def test_top_level_nfn_train_gpt2_evo_prints_family_delegate_command(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    family_cli = tmp_path / "nfn_gpt2_evo_native_train"
    family_cli.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'delegate-native %s\\n' \"$*\"\n",
        encoding="utf-8",
    )
    family_cli.chmod(0o755)

    env = os.environ.copy()
    env.pop("NFN_NATIVE_TRAIN_CLI", None)
    env["NFN_NATIVE_GPT2_EVO_CLI"] = str(family_cli)
    proc = subprocess.run(
        [
            sys.executable,
            str(root / "cli" / "nfn.py"),
            "train",
            "--base-model",
            "gpt2-evo",
            "--tinystories",
            "--native-cuda-print-command",
            "--native-cuda-dry-run",
            "--eval-every-steps",
            "1000",
        ],
        cwd=root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith("delegate-native ")
    assert str(family_cli) not in proc.stdout
    assert "--tinystories" in proc.stdout
    assert "--dry-run" in proc.stdout
    assert "--print-command" in proc.stdout
    assert "--eval-every-steps 1000" in proc.stdout
    assert "--base-model" not in proc.stdout
    assert "--model-family" not in proc.stdout


def test_top_level_nfn_train_explicit_unified_cli_overrides_direct_family_native_cli(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    family_cli = tmp_path / "nfn_gpt2_evo_native_train"
    family_cli.write_text("#!/bin/sh\nexit 44\n", encoding="utf-8")
    family_cli.chmod(0o755)
    unified_cli = tmp_path / "nfn_native_train"
    unified_cli.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n", encoding="utf-8")
    unified_cli.chmod(0o755)

    env = os.environ.copy()
    env["NFN_NATIVE_GPT2_EVO_CLI"] = str(family_cli)
    env["NFN_NATIVE_TRAIN_CLI"] = str(unified_cli)
    proc = subprocess.run(
        [
            sys.executable,
            str(root / "cli" / "nfn.py"),
            "train",
            "--base-model",
            "gpt2-evo",
            "--tinystories",
            "--native-cuda-print-command",
            "--native-cuda-dry-run",
        ],
        cwd=root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert str(unified_cli) in proc.stdout
    assert "--base-model gpt2-evo" in proc.stdout
    assert "--tinystories" in proc.stdout
    assert "--print-command" in proc.stdout
    assert "--dry-run" in proc.stdout
    assert "--model-family" not in proc.stdout


def test_native_gpt2_command_installer_links_temp_bin(tmp_path: Path) -> None:
    if shutil.which("c++") is None:
        pytest.skip("c++ compiler not available")
    root = Path(__file__).resolve().parents[1]
    native_cli = tmp_path / "nfn_gpt_native_train"
    linked_native_cli = tmp_path / "nfn_gpt_native_train_linked"
    compat_native_cli = tmp_path / "nfn_gpt2_native_train"
    native_train_cli = tmp_path / "nfn_native_train"
    launcher = tmp_path / "nfn_gpt2_tile_train"
    missing_dir = tmp_path / "missing"

    build_cli = subprocess.run(
        ["bash", str(root / "tools" / "build_native_gpt_cli.sh"), str(native_cli)],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert build_cli.returncode == 0, build_cli.stderr
    linked_native_cli.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"${1:-}\" == \"--help\" ]]; then\n"
        "  printf '%s\\n' 'Native no-Python dense GPT trainer'\n"
        "  exit 0\n"
        "fi\n"
        "exec \"${NFN_TEST_DYNAMIC_NATIVE_CLI}\" \"$@\"\n",
        encoding="utf-8",
    )
    linked_native_cli.chmod(0o755)
    build_compat_cli = subprocess.run(
        ["bash", str(root / "tools" / "build_native_gpt2_cli.sh"), str(compat_native_cli)],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert build_compat_cli.returncode == 0, build_compat_cli.stderr
    build_native_train = subprocess.run(
        ["bash", str(root / "tools" / "build_native_train_cli.sh"), str(native_train_cli)],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert build_native_train.returncode == 0, build_native_train.stderr
    build_launcher = subprocess.run(
        ["bash", str(root / "tools" / "build_native_gpt2_launcher.sh"), str(launcher)],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert build_launcher.returncode == 0, build_launcher.stderr
    sm120_launcher = tmp_path / "nfn_train_gpt_sm120"
    generic_sm120_launcher = tmp_path / "nfn_train_gpt"
    build_generic_sm120_launcher = subprocess.run(
        ["bash", str(root / "tools" / "build_train_gpt_cli.sh"), str(generic_sm120_launcher)],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert build_generic_sm120_launcher.returncode == 0, build_generic_sm120_launcher.stderr
    build_sm120_launcher = subprocess.run(
        ["bash", str(root / "tools" / "build_train_gpt_sm120_cli.sh"), str(sm120_launcher)],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert build_sm120_launcher.returncode == 0, build_sm120_launcher.stderr
    build_missing = subprocess.run(
        ["bash", str(root / "tools" / "build_native_missing_trainers.sh"), str(missing_dir)],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert build_missing.returncode == 0, build_missing.stderr

    bin_dir = tmp_path / "bin"
    env = os.environ.copy()
    env["NFN_NATIVE_GPT2_BIN_DIR"] = str(bin_dir)
    env.pop("NFN_NATIVE_GPT_CLI", None)
    env["NFN_NATIVE_GPT_LINKED_CLI"] = str(linked_native_cli)
    env["NFN_TEST_DYNAMIC_NATIVE_CLI"] = str(native_cli)
    env["NFN_NATIVE_GPT2_CLI"] = str(compat_native_cli)
    env["NFN_NATIVE_TRAIN_CLI"] = str(native_train_cli)
    env["NFN_NATIVE_GPT2_LAUNCHER"] = str(launcher)
    env["NFN_NATIVE_GPT_TRAIN_CLI"] = str(generic_sm120_launcher)
    env["NFN_NATIVE_SM120_CLI"] = str(sm120_launcher)
    env["NFN_NATIVE_MISSING_TRAINERS_DIR"] = str(missing_dir)
    install = subprocess.run(
        ["bash", str(root / "tools" / "install_native_gpt2_commands.sh")],
        cwd=root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert install.returncode == 0, install.stderr

    linked_native = bin_dir / "nfn-gpt2-native"
    linked_train = bin_dir / "nfn-gpt2-native-train"
    linked_gpt_native = bin_dir / "nfn-gpt-native"
    linked_gpt_train = bin_dir / "nfn-gpt-native-train"
    linked_gpt2_compat = bin_dir / "nfn-gpt2-native-compat"
    linked_unified = bin_dir / "nfn-native-train"
    linked_launcher = bin_dir / "nfn-gpt2-tile-launcher"
    linked_gpt_launcher = bin_dir / "nfn-train-gpt"
    linked_gpt_launcher_alias = bin_dir / "nfn-gpt-train"
    linked_sm120 = bin_dir / "nfn-train-gpt-sm120"
    linked_sm120_alias = bin_dir / "nfn-gpt-sm120-train"
    linked_nanogpt_underscore = bin_dir / "nfn_nanogpt_native_train"
    linked_nanogpt = bin_dir / "nfn-nanogpt-native-train"
    assert linked_native.is_symlink()
    assert linked_train.is_symlink()
    assert linked_gpt_native.is_symlink()
    assert linked_gpt_train.is_symlink()
    assert linked_gpt2_compat.is_symlink()
    assert linked_unified.is_symlink()
    assert linked_launcher.is_symlink()
    assert linked_gpt_launcher.is_symlink()
    assert linked_gpt_launcher_alias.is_symlink()
    assert linked_sm120.is_symlink()
    assert linked_sm120_alias.is_symlink()
    assert linked_nanogpt_underscore.is_symlink()
    assert linked_nanogpt.is_symlink()
    assert linked_native.resolve() == linked_native_cli
    assert linked_train.resolve() == linked_native_cli
    assert linked_gpt_native.resolve() == linked_native_cli
    assert linked_gpt_train.resolve() == linked_native_cli
    assert linked_gpt2_compat.resolve() == compat_native_cli
    assert linked_unified.resolve() == native_train_cli
    assert linked_launcher.resolve() == launcher
    assert linked_gpt_launcher.resolve() == generic_sm120_launcher
    assert linked_gpt_launcher_alias.resolve() == generic_sm120_launcher
    assert linked_sm120.resolve() == sm120_launcher
    assert linked_sm120_alias.resolve() == sm120_launcher
    assert linked_nanogpt_underscore.resolve() == missing_dir / "nfn_nanogpt_native_train"
    assert linked_nanogpt.resolve() == missing_dir / "nfn_nanogpt_native_train"

    help_proc = subprocess.run(
        [str(linked_native), "--help"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert help_proc.returncode == 0, help_proc.stderr
    assert "Native no-Python dense GPT trainer" in help_proc.stdout

    unified_help = subprocess.run(
        [str(linked_unified), "--help"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert unified_help.returncode == 0, unified_help.stderr
    assert "Unified no-Python NeuralFn native training frontend" in unified_help.stdout

    sm120_help = subprocess.run(
        [str(linked_sm120), "--help"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert sm120_help.returncode == 0, sm120_help.stderr
    assert "Compiled dense GPT training helper" in sm120_help.stdout

    nanogpt_help = subprocess.run(
        [str(linked_nanogpt), "--help"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert nanogpt_help.returncode == 0, nanogpt_help.stderr
    assert "Compiled NeuralFn NanoGPT native training preflight" in nanogpt_help.stdout

    installed_dispatch = subprocess.run(
        [
            str(linked_unified),
            "--base-model",
            "nanogpt",
            "--dataset-alias",
            "/tmp/native-cache",
            "--dry-run",
            "--print-command",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert installed_dispatch.returncode == 0, installed_dispatch.stderr
    assert "nfn_gpt_native_train" in installed_dispatch.stdout
    assert "--model-family gpt" in installed_dispatch.stdout
    assert "--template-name nanogpt" in installed_dispatch.stdout
    assert "--dataset-alias /tmp/native-cache" in installed_dispatch.stdout

    installed_print_command = subprocess.run(
        [
            str(linked_unified),
            "--base-model",
            "nanogpt",
            "--train-token-lm",
            "--dataset-alias",
            "/tmp/native-cache",
            "--dry-run",
            "--print-command",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert installed_print_command.returncode == 0, installed_print_command.stderr
    assert str(linked_nanogpt_underscore) in installed_print_command.stdout
    assert "--train-token-lm" in installed_print_command.stdout
    assert "--dry-run" in installed_print_command.stdout
    assert "--print-command" in installed_print_command.stdout
