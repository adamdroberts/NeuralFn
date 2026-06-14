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
import pytest

from neuralfn.config import SHIPPED_GPT_TEMPLATE_PRESETS
from neuralfn.native_gpt import (
    NativeGptRunConfig,
    NativeGptRunnerStatus,
    build_native_gpt_compiled_cli_run_config,
    native_gpt_activation,
    native_gpt_encoding_vocab_size,
    native_gpt_kernel_backend,
    native_gpt_runner_status,
    normalize_native_gpt_encoding_name,
    read_native_gpt_checkpoint_info,
)
from neuralfn.native_gpt2 import (
    NativeGpt2RunConfig,
    build_native_gpt2_compiled_cli_run_config,
    build_native_gpt2_run_config,
    latest_native_gpt2_checkpoint,
    native_gpt2_activation,
    native_gpt2_parameter_count,
    native_gpt2_runner_status,
    read_native_gpt2_checkpoint_info,
    resolve_native_gpt2_cli,
    resolve_native_gpt2_executable,
    resolve_native_gpt2_launcher,
    resolve_native_gpt2_token_shards,
    run_native_gpt2,
    write_native_gpt2_run_config,
)
from neuralfn.native_train import (
    build_native_train_run_config,
    native_train_model_registry,
    native_train_runner_status,
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
    assert "for (std::int64_t batch_begin = 0; batch_begin < batch;" in source
    assert "chunk_batch = std::min(max_batch_per_launch, batch - batch_begin)" in source
    assert "qkv_bf16_bits + batch_begin * packed_elements_per_batch" in source
    assert "out_bf16_bits + batch_begin * merged_elements_per_batch" in source
    assert "grad_out + batch_begin * merged_elements_per_batch" in source
    assert "saved_lse != nullptr ? saved_lse : workspace->lse" in source
    assert "workspace->packed_grad_bf + batch_begin * packed_elements_per_batch" in source


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
    assert cfg.lm_head_row_chunk_size == 8192
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
    assert argv[argv.index("--lm-head-row-chunk-size") + 1] == "8192"
    assert argv[argv.index("--cuda-runtime-lib") + 1] == "/usr/local/cuda/lib64/libcudart.so"
    assert cfg.template_name == "gpt"
    assert argv[argv.index("--template-name") + 1] == "gpt"
    assert cfg.write_checkpoint is True
    assert "--no-checkpoint" not in argv

    llm_cfg = build_native_gpt2_compiled_cli_run_config(
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
    llm_argv = llm_cfg.compiled_cli_argv("/opt/nfn/nfn_gpt2_native_train")
    assert llm_argv[llm_argv.index("--backend") + 1] == "llm-kittens"
    assert llm_argv[llm_argv.index("--target") + 1] == "/opt/nfn/train_gpt2cu"


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
    assert "--no-checkpoint" in cfg.compiled_cli_argv("/opt/nfn/nfn_gpt_native_train")
    assert isinstance(generic_cfg, NativeGptRunConfig)
    assert generic_cfg.write_checkpoint is False
    assert "--no-checkpoint" in generic_cfg.compiled_cli_argv("/opt/nfn/nfn_gpt_native_train")


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


def test_native_gpt2_runner_status_falls_back_to_subprocess_without_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NFN_NATIVE_GPT2_BINDING", "0")
    monkeypatch.setenv("NFN_NATIVE_GPT2_CLI", str(tmp_path / "missing-native-cli"))
    monkeypatch.setenv("NFN_NATIVE_GPT2_LAUNCHER", str(tmp_path / "missing-launcher"))
    status = native_gpt2_runner_status("auto")

    assert status.requested == "auto"
    assert status.resolved == "subprocess"
    assert status.available is True
    assert "binding unavailable" in status.reason
    assert "compiled native GPT CLI/launcher not found" in status.reason


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


def test_native_gpt2_runner_status_uses_compiled_launcher_when_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NFN_NATIVE_GPT2_BINDING", "0")
    monkeypatch.setenv("NFN_NATIVE_GPT2_CLI", str(tmp_path / "missing-native-cli"))
    launcher = tmp_path / "nfn_gpt2_tile_train"
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

    monkeypatch.setitem(sys.modules, "neuralfn_native_gpt2", SimpleNamespace(run_gpt2=fake_run))
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
    assert calls[0]["cuda_visible_devices"] == "0"
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
        "printf 'CUDA_VISIBLE_DEVICES=%s\\nCUDA_DEVICE_MAX_CONNECTIONS=%s\\n' "
        "\"$CUDA_VISIBLE_DEVICES\" \"$CUDA_DEVICE_MAX_CONNECTIONS\" > \"$NFN_TEST_NATIVE_CLI_ENV\"\n"
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
    monkeypatch.delitem(sys.modules, "neuralfn_native_gpt2", raising=False)
    monkeypatch.delitem(sys.modules, "neuralfn._native_gpt2", raising=False)
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
    assert native_gpt2_runner_status("auto").resolved == "binding"
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
        "exit 29\n",
        encoding="utf-8",
    )
    cli.chmod(0o755)
    output = tmp_path / "native-train-args.txt"
    monkeypatch.setenv("NFN_NATIVE_TRAIN_CLI", str(cli))
    monkeypatch.setenv("NFN_NATIVE_TRAIN_BINDING", "0")
    monkeypatch.setenv("NFN_TEST_NATIVE_TRAIN_ARGS", str(output))
    cfg = build_native_train_run_config("nano_gpt", ["--tinystories", "--dry-run"])

    assert resolve_native_train_cli() == str(cli)
    assert cfg.to_dict()["model_family"] == "nano-gpt"
    assert cfg.argv()[:3] == [str(cli), "--base-model", "nano-gpt"]
    status = native_train_runner_status("auto")
    assert status.resolved == "compiled-cli"
    assert status.available is True
    assert run_native_train(cfg, runner="auto") == 29
    args = output.read_text(encoding="utf-8").splitlines()
    assert args[:4] == ["--base-model", "nano-gpt", "--tinystories", "--dry-run"]

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
    assert "--smoke-optimizer-step" in help_proc.stdout
    assert "--smoke-lm-step" in help_proc.stdout
    assert "--smoke-attention-step" in help_proc.stdout
    assert "--smoke-mlp-step" in help_proc.stdout
    assert "--smoke-norm-residual-step" in help_proc.stdout
    assert "--smoke-transformer-block-step" in help_proc.stdout
    assert "--smoke-transformer-lm-step" in help_proc.stdout
    assert "--smoke-embedding-lm-step" in help_proc.stdout
    assert "--train-embedding-lm" in help_proc.stdout
    assert "--train-transformer-lm" in help_proc.stdout
    assert "--cuda-runtime-lib PATH" in help_proc.stdout
    assert "--template-name NAME" in help_proc.stdout
    assert "--graph-file PATH" in help_proc.stdout
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
    assert default_payload["architecture_source"] == "template"
    assert default_payload["architecture_contract"] == "gpt-template-preset"
    assert (
        default_payload["model_family_context_policy"]
        == "dense-gpt-selectors-canonicalize-to-gpt-template-or-graph-selects-architecture"
    )
    assert default_payload["selected_graph_native_runnable"] is True
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
    assert external_plan.returncode == 0, external_plan.stderr
    external_payload = json.loads(external_plan.stdout)
    assert external_payload["backend"] == "llm-kittens"
    assert external_payload["status"] == "external-fast-path"
    assert external_payload["template_name"] == "gpt"
    assert external_payload["resolved_native_template_name"] == "gpt2"
    assert external_payload["graph_file"] == ""
    assert external_payload["architecture_source"] == "template"
    assert external_payload["architecture_contract"] == "gpt-template-preset"

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
    assert tile_payload["template_known"] is True
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
    assert tile_payload["attention_forward_strategy"] == "tk-sm120-packed-qkv-bf16-flashattention"
    assert tile_payload["attention_forward_score_reuse_value_dim"] == 64
    assert tile_payload["attention_forward_scalar_cta_elision_factor"] == 64
    assert tile_payload["attention_forward_value_chunk_size"] == 64
    assert tile_payload["attention_forward_scalar_launch_fallback_enabled"] is True
    assert tile_payload["attention_forward_row_launch_auto_disable_enabled"] is True
    assert tile_payload["attention_forward_row_count"] * 64 == tile_payload["attention_forward_scalar_output_count"]
    assert tile_payload["packed_qkv_attention_enabled"] is True
    assert tile_payload["packed_qkv_attention_bf16_elements"] > 0
    assert tile_payload["packed_qkv_attention_bf16_bytes"] > 0
    assert tile_payload["packed_qkv_float_attention_tape_elided"] is True
    assert tile_payload["packed_qkv_float_attention_tape_elements_elided"] == 64 * 1024 * 768 * 8
    assert tile_payload["packed_qkv_float_attention_tape_bytes_elided"] == 64 * 1024 * 768 * 8 * 4
    assert tile_payload["qkv_forward_layout_strategy"] == "packed-qkv-bf16-no-split"
    assert tile_payload["qkv_forward_layout_kernel_launches_per_block"] == 0
    assert tile_payload["qkv_forward_layout_legacy_launches_per_block"] == 4
    assert tile_payload["qkv_forward_layout_launches_elided_per_block"] == 4
    assert tile_payload["qkv_bias_layout_strategy"] == "packed-qkv-bf16-bias-inplace"
    assert tile_payload["qkv_bias_layout_kernel_launches_per_block"] == 1
    assert tile_payload["qkv_bias_layout_legacy_launches_per_block"] == 2
    assert tile_payload["qkv_bias_layout_launches_elided_per_block"] == 1
    assert tile_payload["qkv_backward_layout_strategy"] == "packed-qkv-bf16-gradient-unpack"
    assert tile_payload["qkv_backward_layout_kernel_launches_per_block"] == 1
    assert tile_payload["qkv_backward_layout_legacy_launches_per_block"] == 4
    assert tile_payload["qkv_backward_layout_launches_elided_per_block"] == 3
    assert tile_payload["attention_backward_qkv_bridge_strategy"] == "tk-sm120-packed-qkv-packed-grad-bridge"
    assert tile_payload["attention_backward_qkv_bridge_kernel_launches_per_block"] == 2
    assert tile_payload["attention_backward_qkv_bridge_legacy_launches_per_block"] == 4
    assert tile_payload["attention_backward_qkv_bridge_launches_elided_per_block"] == 3
    assert tile_payload["attention_projection_input_strategy"] == "packed-o-bf16-direct-gemm"
    assert tile_payload["attention_packed_output_unpack_strategy"] == "elided-direct-bf16-projection"
    assert tile_payload["mlp_fc_bias_gelu_strategy"] == "fused-bias-preactivation-gelu"
    assert tile_payload["mlp_fc_bias_gelu_kernel_launches_per_block"] == 1
    assert tile_payload["mlp_fc_bias_gelu_legacy_launches_per_block"] == 2
    assert tile_payload["mlp_fc_bias_gelu_launches_elided_per_block"] == 1
    assert tile_payload["mlp_proj_forward_activation_strategy"] == "fused-gelu-bf16-act-direct-gemm"
    assert tile_payload["mlp_forward_act_bf16_elements"] == 0
    assert tile_payload["mlp_forward_act_bf16_bytes"] == 0
    assert tile_payload["projection_bias_residual_strategy"] == "fused-linear-bias-residual-add"
    assert tile_payload["projection_bias_residual_kernel_launches_per_block"] == 2
    assert tile_payload["projection_bias_residual_legacy_launches_per_block"] == 4
    assert tile_payload["projection_bias_residual_launches_elided_per_block"] == 2
    assert tile_payload["attention_residual_ln2_strategy"] == "fused-linear-bias-residual-layernorm"
    assert tile_payload["attention_residual_ln2_kernel_launches_per_block"] == 1
    assert tile_payload["attention_residual_ln2_legacy_launches_per_block"] == 2
    assert tile_payload["attention_residual_ln2_launches_elided_per_block"] == 1
    assert tile_payload["attention_backward_grad_layout_strategy"] == "merged-grad-out-direct"
    assert tile_payload["attention_backward_grad_layout_kernel_launches_per_block"] == 0
    assert tile_payload["attention_backward_grad_layout_legacy_launches_per_block"] == 1
    assert tile_payload["attention_backward_grad_layout_launches_elided_per_block"] == 1
    assert tile_payload["attention_backward_strategy"] == "tk-sm120-packed-qkv-bf16-backward-bridge"
    assert tile_payload["attention_backward_reuses_forward_workspace"] is True
    assert tile_payload["attention_backward_uses_saved_forward_workspace"] is True
    assert tile_payload["attention_activation_storage_strategy"] == "disabled"
    assert tile_payload["packed_attention_activation_storage_strategy"] == (
        "packed-qkv-o-bf16-forward-store-direct-backward"
    )
    assert tile_payload["stored_packed_attention_activation_blocks"] == 12
    assert tile_payload["stored_packed_attention_bf16_elements"] > 0
    assert tile_payload["stored_packed_attention_bf16_bytes"] > 0
    assert tile_payload["stored_packed_attention_lse_elements"] > 0
    assert tile_payload["stored_packed_attention_lse_bytes"] > 0
    assert tile_payload["stored_packed_attention_lse_enabled"] is True
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
    assert "nfn_native_tile_attention_forward_stats_reset" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_attention_forward_row_launch_count" in tile_payload["available_native_kernels"]
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
    assert "nfn_native_tile_linear_weight_bf16_gelu_bf16_float32" in tile_payload["available_native_kernels"]
    assert (
        "nfn_native_tile_linear_bf16_input_weight_bf16_gelu_bf16_float32"
        in tile_payload["available_native_kernels"]
    )
    assert "nfn_native_tile_gelu_backward_inplace_bf16_bits_float32" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_trainer_linear_stats_reset" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_trainer_linear_bf16_cache_reset" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_trainer_linear_bf16_gemm_count" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_trainer_linear_tk_gemm_count" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_trainer_linear_tk_float_out_gemm_count" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_trainer_linear_cublaslt_gemm_count" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_trainer_linear_sgemm_count" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_trainer_linear_bf16_a_pack_count" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_trainer_linear_bf16_a_cache_hit_count" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_trainer_linear_bf16_cache_reset_count" in tile_payload["available_native_kernels"]
    assert (
        "nfn_native_tile_trainer_linear_bf16_workspace_allocation_count"
        in tile_payload["available_native_kernels"]
    )
    assert "nfn_native_tile_trainer_linear_bf16_cached_a_capacity" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_trainer_linear_bf16_cache_entry_count" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_split_qkv_to_heads_float32" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_split_qkv_to_heads_add_bias_float32" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_merge_heads_to_qkv_float32" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_gelu_add_bias_float32" in tile_payload["available_native_kernels"]
    assert "nfn_native_tile_linear_bias_residual_add_float32" in tile_payload["available_native_kernels"]
    assert tile_payload["required_native_work"] == []
    assert any("SM120 throughput" in item for item in tile_payload["remaining_validation"])

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
        if preset in {"gpt2", "gpt2_megakernel", "gpt2_moa"}:
            assert preset_payload["selected_graph_support_status"] == "native-transformer-lm"
            assert preset_payload["selected_graph_native_runnable"] is True
        else:
            assert preset_payload["selected_graph_support_status"] == "template-native-trainer-missing"
            assert preset_payload["selected_graph_native_runnable"] is False

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

    custom_graph = subprocess.run(
        [
            str(cli),
            "--dataset-alias",
            str(dataset_path),
            "--backend",
            "tile-cuda",
            "--graph-file",
            str(tmp_path / "custom-graph.json"),
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
    assert custom_payload["template_known"] is True
    assert custom_payload["shipped_template_catalog_count"] == len(SHIPPED_GPT_TEMPLATE_PRESETS)
    assert custom_payload["selected_graph_support_status"] == "custom-graph-native-trainer-missing"
    assert custom_payload["selected_graph_native_runnable"] is False

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
    assert missing_payload["token_shards_resolved"] is False
    assert missing_payload["train_shard"] == ""
    assert missing_payload["val_shard"] == ""

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
    assert train_transformer_payload["batch_size"] == 1
    assert train_transformer_payload["seq_len"] == 2
    assert train_transformer_payload["trained_layers"] == 12
    assert train_transformer_payload["target_layers"] == 12
    assert train_transformer_payload["vocab"] == 50257
    assert train_transformer_payload["padded_vocab"] == 50304
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
    assert train_transformer_payload["lm_head_bf16_logits_enabled"] is True
    assert train_transformer_payload["lm_head_bf16_logit_elements"] == 0
    assert train_transformer_payload["lm_head_bf16_logit_bytes"] == 0
    assert train_transformer_payload["lm_head_ce_backward_strategy"] == "fused-row-bf16-logits-dlogits"
    assert train_transformer_payload["lm_head_grad_logits_workspace_allocated"] is False
    assert train_transformer_payload["linear_backend_strategy"] == "not-run"
    assert train_transformer_payload["block_forward_linear_strategy"] == "bf16-shadow-weight-gemmex-forward"
    assert train_transformer_payload["block_backward_input_linear_strategy"] == "bf16-shadow-weight-gemmex-dinput"
    assert (
        train_transformer_payload["block_backward_weight_linear_strategy"]
        == "forced-bf16-gemmex-dweight-plus-bias-accumulate-fallback"
    )
    assert (
        train_transformer_payload["non_block_forward_backward_linear_strategy"]
        == "padded-lm-head-bf16-gemmex-fallback"
    )
    assert train_transformer_payload["lm_head_logits_linear_strategy"] == "bf16-gemmex-fallback"
    assert train_transformer_payload["linear_bf16_gemm_count"] == 0
    assert train_transformer_payload["linear_tk_gemm_count"] == 0
    assert train_transformer_payload["linear_cublaslt_gemm_count"] == 0
    assert train_transformer_payload["linear_sgemm_count"] == 0
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
    assert train_transformer_payload["timing"]["stage_timing_event_count"] == 0
    assert train_transformer_payload["timing"]["stage_timing_dropped_event_count"] == 0
    assert train_transformer_payload["timing"]["stage_timing"] == []
    assert train_transformer_payload["attention_forward_strategy"] == "tk-sm120-packed-qkv-bf16-flashattention"
    assert train_transformer_payload["attention_forward_row_count"] == 24
    assert train_transformer_payload["attention_forward_scalar_output_count"] == 1536
    assert train_transformer_payload["attention_forward_score_reuse_value_dim"] == 64
    assert train_transformer_payload["attention_forward_scalar_cta_elision_factor"] == 64
    assert train_transformer_payload["attention_forward_value_chunk_size"] == 64
    assert train_transformer_payload["attention_forward_scalar_launch_fallback_enabled"] is True
    assert train_transformer_payload["attention_forward_row_launch_auto_disable_enabled"] is True
    assert train_transformer_payload["attention_forward_row_launch_auto_disabled"] is False
    assert train_transformer_payload["attention_forward_row_launch_count"] == 0
    assert train_transformer_payload["attention_forward_row_launch_success_count"] == 0
    assert train_transformer_payload["attention_forward_row_launch_fallback_count"] == 0
    assert train_transformer_payload["attention_forward_scalar_launch_count"] == 0
    assert train_transformer_payload["packed_qkv_attention_enabled"] is True
    assert train_transformer_payload["packed_qkv_attention_bf16_elements"] == 0
    assert train_transformer_payload["packed_qkv_attention_bf16_bytes"] == 0
    assert train_transformer_payload["qkv_forward_layout_strategy"] == "packed-qkv-bf16-no-split"
    assert train_transformer_payload["qkv_forward_layout_kernel_launches_per_block"] == 0
    assert train_transformer_payload["qkv_forward_layout_legacy_launches_per_block"] == 4
    assert train_transformer_payload["qkv_forward_layout_launches_elided_per_block"] == 4
    assert train_transformer_payload["qkv_bias_layout_strategy"] == "packed-qkv-bf16-bias-inplace"
    assert train_transformer_payload["qkv_bias_layout_kernel_launches_per_block"] == 1
    assert train_transformer_payload["qkv_bias_layout_legacy_launches_per_block"] == 2
    assert train_transformer_payload["qkv_bias_layout_launches_elided_per_block"] == 1
    assert train_transformer_payload["qkv_backward_layout_strategy"] == "packed-qkv-bf16-gradient-unpack"
    assert train_transformer_payload["qkv_backward_layout_kernel_launches_per_block"] == 1
    assert train_transformer_payload["qkv_backward_layout_legacy_launches_per_block"] == 4
    assert train_transformer_payload["qkv_backward_layout_launches_elided_per_block"] == 3
    assert train_transformer_payload["attention_backward_qkv_bridge_strategy"] == "tk-sm120-packed-qkv-packed-grad-bridge"
    assert train_transformer_payload["attention_backward_qkv_bridge_kernel_launches_per_block"] == 2
    assert train_transformer_payload["attention_backward_qkv_bridge_legacy_launches_per_block"] == 4
    assert train_transformer_payload["attention_backward_qkv_bridge_launches_elided_per_block"] == 3
    assert train_transformer_payload["attention_projection_input_strategy"] == "packed-o-bf16-direct-gemm"
    assert train_transformer_payload["attention_packed_output_unpack_strategy"] == "elided-direct-bf16-projection"
    assert train_transformer_payload["mlp_fc_bias_gelu_strategy"] == "fused-bias-preactivation-gelu"
    assert train_transformer_payload["mlp_fc_bias_gelu_kernel_launches_per_block"] == 1
    assert train_transformer_payload["mlp_fc_bias_gelu_legacy_launches_per_block"] == 2
    assert train_transformer_payload["mlp_fc_bias_gelu_launches_elided_per_block"] == 1
    assert train_transformer_payload["mlp_proj_forward_activation_strategy"] == "fused-gelu-bf16-act-direct-gemm"
    assert train_transformer_payload["mlp_forward_act_bf16_elements"] == 0
    assert train_transformer_payload["mlp_forward_act_bf16_bytes"] == 0
    assert train_transformer_payload["projection_bias_residual_strategy"] == "fused-linear-bias-residual-add"
    assert train_transformer_payload["projection_bias_residual_kernel_launches_per_block"] == 2
    assert train_transformer_payload["projection_bias_residual_legacy_launches_per_block"] == 4
    assert train_transformer_payload["projection_bias_residual_launches_elided_per_block"] == 2
    assert train_transformer_payload["attention_residual_ln2_strategy"] == "fused-linear-bias-residual-layernorm"
    assert train_transformer_payload["attention_residual_ln2_kernel_launches_per_block"] == 1
    assert train_transformer_payload["attention_residual_ln2_legacy_launches_per_block"] == 2
    assert train_transformer_payload["attention_residual_ln2_launches_elided_per_block"] == 1
    assert train_transformer_payload["attention_backward_grad_layout_strategy"] == "merged-grad-out-direct"
    assert train_transformer_payload["attention_backward_grad_layout_kernel_launches_per_block"] == 0
    assert train_transformer_payload["attention_backward_grad_layout_legacy_launches_per_block"] == 1
    assert train_transformer_payload["attention_backward_grad_layout_launches_elided_per_block"] == 1
    assert train_transformer_payload["attention_backward_strategy"] == "tk-sm120-packed-qkv-bf16-backward-bridge"
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
    assert train_transformer_payload["packed_attention_activation_storage_strategy"] == (
        "packed-qkv-o-bf16-forward-store-direct-backward"
    )
    assert train_transformer_payload["stored_packed_attention_activation_blocks"] == 12
    assert train_transformer_payload["stored_packed_attention_bf16_elements"] == 0
    assert train_transformer_payload["stored_packed_attention_bf16_bytes"] == 0
    assert train_transformer_payload["stored_packed_attention_lse_elements"] == 0
    assert train_transformer_payload["stored_packed_attention_lse_bytes"] == 0
    assert train_transformer_payload["stored_packed_attention_lse_enabled"] is True
    assert train_transformer_payload["stored_packed_attention_store_blocks"] == 0
    assert train_transformer_payload["stored_packed_attention_restore_blocks"] == 0
    assert train_transformer_payload["stored_packed_attention_backward_kernel_launches"] == 0
    assert train_transformer_payload["stored_packed_attention_backward_consumer_strategy"] == (
        "saved-packed-qkv-o-lse-bf16-backward-to-qkv"
    )
    assert train_transformer_payload["max_steps"] == 2
    assert train_transformer_payload["eval_every_steps"] == 1
    assert train_transformer_payload["eval_batches"] == 1
    assert train_transformer_payload["validation"]["eval_batch_size"] == 1
    assert train_transformer_payload["train_loss_eval_count"] == 0
    assert train_transformer_payload["train_loss_last_step"] == 0
    assert train_transformer_payload["train_loss_sparse"] is False
    assert train_transformer_payload["train_loss_sampling"] == "disabled"
    assert train_transformer_payload["train_loss_on_validation_steps"] is False
    assert train_transformer_payload["token_id_upload_strategy"] == "uint16-pinned-async-h2d-device-widen"
    assert train_transformer_payload["token_id_host_staging"] == "pinned"
    assert train_transformer_payload["token_id_h2d_copy"] == "cudaMemcpyAsync-contiguous-arena"
    assert train_transformer_payload["token_id_h2d_copy_calls_per_microbatch"] == 1
    assert train_transformer_payload["token_id_h2d_copy_calls_elided_per_microbatch"] == 1
    assert train_transformer_payload["token_id_widen_strategy"] == "single-contiguous-arena-kernel"
    assert train_transformer_payload["token_id_widen_kernel_launches_per_microbatch"] == 1
    assert train_transformer_payload["token_id_widen_kernel_launches_elided_per_microbatch"] == 1
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
    assert train_transformer_payload["token_weight_init_strategy"] == "device-tile-deterministic"
    assert train_transformer_payload["token_weight_host_materialization"] is False
    assert train_transformer_payload["float_allocation_strategy"] == "single-arena"
    assert train_transformer_payload["float_allocation_cuda_malloc_count"] == 0
    assert train_transformer_payload["float_allocation_request_count"] == 0
    assert train_transformer_payload["float_arena_requested_elements"] == 0
    assert train_transformer_payload["float_arena_allocated_elements"] == 0
    assert train_transformer_payload["float_arena_zero_init_strategy"] == "single-arena-fill"
    assert train_transformer_payload["float_arena_zero_fill_count"] == 0
    assert train_transformer_payload["startup_per_buffer_zero_fill_elided"] is True
    assert train_transformer_payload["startup_per_buffer_zero_fill_launches_elided"] == 369
    assert train_transformer_payload["descriptor_allocation_strategy"] == "single-device-arena"
    assert train_transformer_payload["descriptor_arena_cuda_malloc_count"] == 0
    assert train_transformer_payload["descriptor_arena_requested_bytes"] == 0
    assert train_transformer_payload["descriptor_arena_bytes"] == 0
    assert train_transformer_payload["descriptor_arena_suballocation_count"] == 0
    assert train_transformer_payload["descriptor_upload_strategy"] == "single-host-packed-arena-copy"
    assert train_transformer_payload["descriptor_arena_copy_count"] == 0
    assert train_transformer_payload["descriptor_arena_copy_calls_elided"] == 13
    assert train_transformer_payload["descriptor_cuda_mallocs_elided"] == 13
    assert train_transformer_payload["parameter_initialization_strategy"] == "fused-multi-buffer-fill-values"
    assert train_transformer_payload["parameter_initialization_descriptor_count"] == 0
    assert train_transformer_payload["parameter_initialization_max_elements"] == 0
    assert train_transformer_payload["parameter_initialization_kernel_launches"] == 0
    assert train_transformer_payload["parameter_initialization_kernel_launches_per_startup"] == 0
    assert train_transformer_payload["parameter_initialization_per_buffer_launches_elided"] == 74
    assert train_transformer_payload["adamw_update_strategy"] == "fused-multi-buffer-device-scale"
    assert train_transformer_payload["adamw_descriptor_count"] == 0
    assert train_transformer_payload["adamw_kernel_launches"] == 0
    assert train_transformer_payload["adamw_step_kernel_launches_per_optimizer_step"] == 0
    assert train_transformer_payload["adamw_per_buffer_step_launches_elided"] == 147
    assert train_transformer_payload["gradient_zero_strategy"] == "fused-multi-buffer-accumulation-zero"
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
    assert train_transformer_payload["block_state_layout"] == {
        "allocated_block_count": 12,
        "target_block_count": 12,
        "activation_tape_count": 1,
        "packed_qkv_float_attention_tape_elided": True,
        "packed_qkv_float_attention_tape_elements_elided": 2 * 1 * 768 * 8,
        "persistent_block_outputs": 11,
        "final_block_output_copy_elided": True,
        "validation_persistent_block_outputs": 0,
        "validation_block_output_copies_elided": True,
        "backward_recompute_blocks": 11,
        "final_block_backward_recompute_elided": True,
        "backward_recompute_mlp_fc_gelu_elided": True,
        "backward_recompute_attention_qkv_sdpa_elided": True,
        "backward_recompute_attention_uses_saved_o": True,
        "backward_recompute_mlp_projection_elided": True,
        "backward_recompute_final_residual_elided": True,
        "mlp_proj_backward_gelu_inplace": True,
        "mlp_proj_backward_grad_act_scratch_allocated": False,
        "activation_tape_strategy": "scratch-recompute-bf16-stored-packed-attention-and-mlp-direct-backward",
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
        "parameter_initialization_strategy": "fused-multi-buffer-fill-values",
        "parameter_initialization_descriptor_count": 0,
        "parameter_initialization_kernel_launches_per_startup": 0,
        "parameter_initialization_per_buffer_launches_elided": 74,
        "startup_zero_init_strategy": "single-arena-fill",
        "startup_arena_zero_fill_count": 0,
        "startup_per_buffer_zero_fill_elided": True,
        "startup_per_buffer_zero_fill_launches_elided": 369,
        "descriptor_allocation_strategy": "single-device-arena",
        "descriptor_arena_cuda_malloc_count": 0,
        "descriptor_arena_suballocation_count": 0,
        "descriptor_upload_strategy": "single-host-packed-arena-copy",
        "descriptor_arena_copy_count": 0,
        "descriptor_arena_copy_calls_elided": 13,
        "descriptor_cuda_mallocs_elided": 13,
        "block0_duplicate_allocation_elided": True,
        "block0_duplicate_activation_allocation_elided": True,
        "block0_duplicate_parameter_initialization_elided": True,
        "block0_duplicate_adamw_state_zero_elided": True,
        "gradient_zero_loop": False,
        "gradient_zero_loop_elided": True,
        "gradient_zero_strategy": "fused-multi-buffer-accumulation-zero",
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
        "block_linear_weight_gradient_scratch_buffers_allocated": False,
        "block_linear_weight_gradient_microbatch_full_copy_elided": True,
        "layer_norm_affine_gradient_accumulation_direct": True,
        "layer_norm_affine_gradient_scratch_buffers_allocated": False,
        "layer_norm_affine_gradient_microbatch_full_copy_elided": True,
        "linear_bias_gradient_accumulation_direct": True,
        "linear_bias_gradient_scratch_buffers_allocated": False,
        "linear_bias_gradient_microbatch_full_copy_elided": True,
        "position_gradient_accumulation_direct": True,
        "position_gradient_scratch_buffer_allocated": False,
        "position_gradient_microbatch_full_copy_elided": True,
        "layer_norm_backward_affine_strategy": "auto-chunked-atomic-accumulate",
            "layer_norm_stats_strategy": "forward-store-mean-rstd-backward-reuse",
            "layer_norm_backward_reuses_forward_stats": True,
            "layer_norm_stats_disabled_by_fused_residual_ln2": False,
            "layer_norm_backward_residual_fusion_enabled": True,
            "layer_norm_backward_residual_strategy": "fused-dinput-residual-add-with-forward-stats",
            "residual1_backward_consumer_strategy": "bf16-layernorm-backward",
            "gradient_clip_loop": False,
        "gradient_clip_loop_elided": True,
        "gradient_clip_strategy": "fused-multi-buffer-sumsq-device-scale",
        "gradient_clip_descriptor_count": 0,
        "gradient_sumsq_kernel_launches_per_optimizer_step": 0,
        "gradient_sumsq_per_buffer_launches_elided": 147,
        "adamw_device_clip_scale_fused": True,
        "adamw_bf16_shadow_refresh_strategy": "separate-many-pack-after-adamw",
        "adamw_update_loop": False,
        "adamw_update_loop_elided": True,
        "adamw_update_strategy": "fused-multi-buffer-device-scale",
        "adamw_descriptor_count": 0,
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
        "eval_batch_size": 1,
        "eval_count": 0,
        "losses": [],
    }
    assert "nfn_native_tile_sumsq_partials_float32" in train_transformer_payload["kernels"]
    assert "nfn_native_tile_sumsq_partials_many_float32" in train_transformer_payload["kernels"]
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
        "nfn_native_tile_token_cross_entropy_backward_inplace_with_workspace_float32"
        in train_transformer_payload["kernels"]
    )
    assert "nfn_native_tile_linear_backward_weight_accumulate_float32" in train_transformer_payload["kernels"]
    assert "nfn_native_tile_merge_heads_to_qkv_float32" in train_transformer_payload["kernels"]
    assert "nfn_native_tile_copy_float32" in train_transformer_payload["kernels"]
    assert "nfn_native_tile_fill_many_float32" in train_transformer_payload["kernels"]
    assert "nfn_native_tile_fill_many_values_float32" in train_transformer_payload["kernels"]
    assert "nfn_native_tile_init_gpt2_token_weight_float32" in train_transformer_payload["kernels"]
    assert "nfn_native_tile_uint16_to_int64" in train_transformer_payload["kernels"]
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
    assert executed.returncode == 0, executed.stderr
    assert "-af moa" in executed.stdout
    assert "-ak 7" in executed.stdout


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
        assert "--base-model" not in dense_gpt.stdout

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
    assert statuses["gpt"] == "implemented"
    assert statuses["gpt2"] == "implemented"
    assert statuses["gpt3"] == "implemented"
    assert statuses["nanogpt"] == "partial-native-trainer"
    assert native_targets["gpt"] == "nfn_gpt_native_train"
    assert native_targets["gpt2"] == "nfn_gpt_native_train"
    assert native_targets["gpt3"] == "nfn_gpt_native_train"
    sdk_payload = native_train_model_registry(native_train_cli=str(unified))
    sdk_statuses = {item["name"]: item["status"] for item in sdk_payload["models"]}
    assert sdk_statuses == statuses

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

    evo_plan_proc = subprocess.run(
        [
            str(gpt2_evo),
            "--print-plan",
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
    assert evo_plan["status"] == "native-preflight-missing-evo-trainer"
    assert evo_plan["template_name"] == "gpt2_moa"
    assert evo_plan["graph_file"] == ""
    assert evo_plan["template_known"] is True
    assert evo_plan["selected_graph_support_status"] == "native-gpt2-evo-trainer-missing"
    assert evo_plan["selected_graph_native_runnable"] is False
    assert evo_plan["shipped_template_catalog_count"] == len(SHIPPED_GPT_TEMPLATE_PRESETS)
    assert evo_plan["shipped_template_catalog"] == list(SHIPPED_GPT_TEMPLATE_PRESETS)
    assert evo_plan["shape"]["num_layers"] == 12
    assert evo_plan["shape"]["model_dim"] == 768
    assert evo_plan["shape"]["num_heads"] == 12
    assert evo_plan["schedule"]["eval_every_steps"] == 1000
    assert evo_plan["schedule"]["grad_accum_steps"] == 8
    assert evo_plan["optimizer"]["profile"] == "adamw"
    assert evo_plan["tile_cuda"]["activation_dtype"] == "nvfp4"
    assert evo_plan["layer_evo"]["enabled"] is True
    assert evo_plan["layer_evo"]["layer_index"] == 6
    assert evo_plan["layer_evo"]["interval"] == 10
    assert evo_plan["layer_evo"]["population"] == 8
    assert evo_plan["layer_evo"]["evo_block_parameters"] > 0
    assert evo_plan["estimated_parameters"] > evo_plan["layer_evo"]["evo_block_parameters"]
    assert "NVFP4 activation intent preserved in the compiled native plan" in evo_plan["available_native_kernels"]
    assert "template/custom graph selector parsed before graph-backed runtime import" in evo_plan["available_native_kernels"]
    assert "forward-only candidate evaluation for current plus mutated evo-layer weights" in evo_plan["required_native_kernels"]
    assert "copy/adopt best evo block candidate without host graph-editor tensor flow" in evo_plan["required_native_kernels"]

    evo_custom_graph = subprocess.run(
        [
            str(gpt2_evo),
            "--print-plan",
            "--dataset-alias",
            "/tmp/native-cache",
            "--graph-file",
            str(tmp_path / "gpt2-evo-custom.json"),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert evo_custom_graph.returncode == 0, evo_custom_graph.stderr
    evo_custom_graph_plan = json.loads(evo_custom_graph.stdout)
    assert evo_custom_graph_plan["graph_file"].endswith("gpt2-evo-custom.json")
    assert evo_custom_graph_plan["template_name"] == "gpt2"
    assert evo_custom_graph_plan["template_known"] is True
    assert evo_custom_graph_plan["selected_graph_support_status"] == "custom-graph-native-trainer-missing"
    assert evo_custom_graph_plan["selected_graph_native_runnable"] is False

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
    assert "--eval-batch-size N" in help_proc.stdout
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
        "shape": [1024, 320],
        "offset": 0,
        "count": 327680,
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
    assert stages_by_name["lm_head.backward_weight_tied"]["elements"] == 327680
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

    dry_run = subprocess.run(
        [
            str(unified),
            "--base-model",
            "nanogpt",
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
    assert dry_run.returncode == 2
    assert "nfn_nanogpt_native_train: native CUDA Tile trainer for nanogpt is not implemented yet" in dry_run.stderr
    dry_run_json_start = dry_run.stdout.find("{")
    assert dry_run_json_start >= 0
    dry_run_plan = json.loads(dry_run.stdout[dry_run_json_start:])
    assert dry_run_plan["schedule"]["eval_every_steps"] == 1000

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
    assert evo_dry_run.returncode == 2
    assert "nfn_gpt2_evo_native_train: native CUDA Tile trainer for gpt2-evo is not implemented yet" in evo_dry_run.stderr
    evo_dry_run_json_start = evo_dry_run.stdout.find("{")
    assert evo_dry_run_json_start >= 0
    evo_dry_run_plan = json.loads(evo_dry_run.stdout[evo_dry_run_json_start:])
    assert evo_dry_run_plan["schedule"]["eval_every_steps"] == 1000
    assert evo_dry_run_plan["layer_evo"]["enabled"] is True


def test_native_gpt2_build_all_script_supports_temp_outputs(tmp_path: Path) -> None:
    if shutil.which("c++") is None:
        pytest.skip("c++ compiler not available")
    if shutil.which("nvcc") is None:
        pytest.skip("nvcc compiler not available")
    root = Path(__file__).resolve().parents[1]
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    env = os.environ.copy()
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
    assert Path(env["NFN_NATIVE_GPT2_BINDING_OUT"]).exists()
    assert Path(env["NFN_NATIVE_TRAIN_BINDING_OUT"]).exists()
    assert Path(env["NFN_NATIVE_GPT2_LAUNCHER_OUT"]).exists()
    assert Path(env["NFN_NATIVE_GPT_CLI_OUT"]).exists()
    assert Path(env["NFN_NATIVE_GPT2_CLI_OUT"]).exists()
    assert Path(env["NFN_NATIVE_TRAIN_CLI_OUT"]).exists()
    assert Path(env["NFN_NATIVE_TRAIN_TILE_OPS_OUT"]).exists()
    assert (Path(env["NFN_NATIVE_MISSING_TRAINERS_OUT_DIR"]) / "nfn_nanogpt_native_train").exists()
    assert (Path(env["NFN_NATIVE_MISSING_TRAINERS_OUT_DIR"]) / "nfn_llama_native_train").exists()


def test_large_row_reduction_fallbacks_use_shared_row_chunks() -> None:
    root = Path(__file__).resolve().parents[1]
    kernels_text = (root / "neuralfn" / "csrc" / "tile_cuda" / "kernels.cu").read_text()

    assert "kLayerNormBackwardAffineDefaultRowChunkSize = 256" in kernels_text
    assert "NFN_TILE_CUDA_LAYERNORM_AFFINE_ROW_CHUNK_SIZE" in kernels_text
    assert "NFN_NATIVE_GPT_LAYERNORM_AFFINE_ROW_CHUNK_SIZE" in kernels_text
    assert "kLinearBackwardBiasRowChunkSize = 512" in kernels_text
    for function_name in (
        "launch_layer_norm_backward_affine_float32",
        "launch_layer_norm_backward_affine_accumulate_float32",
        "launch_layer_norm_backward_affine_accumulate_with_stats_float32",
    ):
        function_body = kernels_text.split(f"void {function_name}", 1)[1].split("\nvoid ", 1)[0]
        assert "kRowChunkSize = layer_norm_backward_affine_row_chunk_size()" in function_body
        assert "kRowChunkSize = kLinearBackwardBiasRowChunkSize" not in function_body
    for function_name in (
        "launch_linear_backward_weight_accumulate_bf16_bits_float32",
        "launch_linear_backward_weight_accumulate_float32_bf16_bits",
        "launch_linear_backward_bias_float32",
        "launch_linear_backward_bias_accumulate_float32",
    ):
        function_body = kernels_text.split(f"void {function_name}", 1)[1].split("\nvoid ", 1)[0]
        assert "kRowChunkSize = kLinearBackwardBiasRowChunkSize" in function_body
        assert "kRowChunkSize = 256" not in function_body


def test_native_train_tile_ops_builds_torch_free_c_abi(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    gpt2_source = root / "neuralfn" / "csrc" / "native_gpt2" / "nfn_gpt2_native_train.cpp"
    token_shards_header = root / "neuralfn" / "csrc" / "native_train" / "token_shards.h"
    token_shards_source = root / "neuralfn" / "csrc" / "native_train" / "token_shards.cpp"
    header = root / "neuralfn" / "csrc" / "native_train" / "tile_ops.h"
    source = root / "neuralfn" / "csrc" / "native_train" / "tile_ops.cu"
    kernels = root / "neuralfn" / "csrc" / "tile_cuda" / "kernels.cu"
    build_script = root / "tools" / "build_native_train_tile_ops.sh"
    gpt2_source_text = gpt2_source.read_text()
    token_shards_header_text = token_shards_header.read_text()
    token_shards_source_text = token_shards_source.read_text()
    header_text = header.read_text()
    source_text = source.read_text()
    kernels_text = kernels.read_text()
    script_text = build_script.read_text()

    assert "cudaRuntimeGetVersion" in gpt2_source_text
    assert "cudaDriverGetVersion" in gpt2_source_text
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
    assert "cudaMalloc transformer_lm_token_i64_arena" not in gpt2_source_text
    assert "cudaMalloc transformer_lm_token_u16_device_arena" not in gpt2_source_text
    assert "token_weight.init_device" in gpt2_source_text
    assert "nfn_native_tile_init_gpt2_token_weight_float32" in gpt2_source_text
    assert "FloatArenaRequest" in gpt2_source_text
    assert "cudaMalloc transformer_lm_float_arena" in gpt2_source_text
    assert "float_allocation_strategy" in gpt2_source_text
    assert "float_allocation_cuda_malloc_count" in gpt2_source_text
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
    assert "launch_adamw_step_many_with_device_scale_float32" in source_text
    assert "launch_adamw_step_many_with_device_scale_bf16_shadow_float32" in source_text
    assert "adamw_step_many_with_device_scale_bf16_shadow_float32_kernel" in kernels_text
    assert "bf16_shadow_offsets" in kernels_text
    assert "ct::element_cast<__nv_bfloat16>(next_p)" in kernels_text
    assert "nfn_native_tile_fill_float32" in header_text
    assert "nfn_native_tile_fill_many_float32" in header_text
    assert "nfn_native_tile_fill_many_values_float32" in header_text
    assert "launch_fill_many_float32" in source_text
    assert "launch_fill_many_values_float32" in source_text
    assert "fill_many_values_float32_kernel" in kernels_text
    assert "nfn_native_tile_sumsq_partials_many_float32" in header_text
    assert "launch_sumsq_partials_many_float32" in source_text
    assert "sumsq_partials_many_float32_kernel" in kernels_text
    assert "gradient_partial_offsets" in gpt2_source_text
    assert "fused-multi-buffer-fill-values" in gpt2_source_text
    assert "fused-multi-buffer-sumsq-device-scale" in gpt2_source_text
    assert "nfn_native_tile_init_gpt2_token_weight_float32" in header_text
    assert "nfn_native_tile_copy_float32" in header_text
    assert "nfn_native_tile_uint16_to_int64" in header_text
    assert "nfn_native_tile_float32_to_bf16_bits" in header_text
    assert "nfn_native_tile_bf16_bits_to_float32" in header_text
    assert "nfn_native_tile_bf16_bits_add_bias_inplace_float32" in header_text
    assert "nfn_native_tile_store_mlp_activations_bf16_float32" in header_text
    assert "nfn_native_tile_restore_mlp_activations_bf16_float32" in header_text
    assert "nfn_native_tile_linear_weight_bf16_float32" in header_text
    assert "nfn_native_tile_linear_weight_bf16_output_float32" in header_text
    assert "nfn_native_tile_linear_bf16_input_bits_float32" in header_text
    assert "nfn_native_tile_linear_bf16_input_weight_bf16_float32" in header_text
    assert "nfn_native_tile_linear_bf16_gelu_bf16_float32" in header_text
    assert "nfn_native_tile_linear_weight_bf16_gelu_bf16_float32" in header_text
    assert "nfn_native_tile_linear_bf16_input_weight_bf16_gelu_bf16_float32" in header_text
    assert "nfn_native_tile_gelu_add_bias_bf16_act_float32" in header_text
    assert "nfn_native_tile_linear_backward_weight_accumulate_bf16_bits_float32" in header_text
    assert "nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_float32" in header_text
    assert "nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_bits_float32" in header_text
    assert "nfn_native_tile_linear_bf16_output_float32" in header_text
    assert "nfn_native_tile_linear_backward_input_bf16_bits_float32" in header_text
    assert "nfn_native_tile_linear_backward_input_weight_bf16_float32" in header_text
    assert "nfn_native_tile_linear_backward_weight_accumulate_float32_bf16_bits" in header_text
    assert "nfn_native_tile_gelu_backward_inplace_bf16_bits_float32" in header_text
    assert "nfn_native_tile_float32_to_bf16_bits_many" in header_text
    assert "nfn_native_tile_trainer_linear_stats_reset" in header_text
    assert "nfn_native_tile_trainer_linear_bf16_cache_reset" in header_text
    assert "nfn_native_tile_trainer_linear_bf16_gemm_count" in header_text
    assert "nfn_native_tile_trainer_linear_tk_gemm_count" in header_text
    assert "nfn_native_tile_trainer_linear_tk_float_out_gemm_count" in header_text
    assert "nfn_native_tile_trainer_linear_cublaslt_gemm_count" in header_text
    assert "nfn_native_tile_trainer_linear_sgemm_count" in header_text
    assert "nfn_native_tile_trainer_linear_bf16_a_pack_count" in header_text
    assert "nfn_native_tile_trainer_linear_bf16_a_cache_hit_count" in header_text
    assert "nfn_native_tile_trainer_linear_bf16_cache_reset_count" in header_text
    assert "launch_float32_to_bf16_bits" in source_text
    assert "launch_bf16_bits_to_float32" in source_text
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
    assert "launch_gelu_backward_inplace_bf16_bits_float32" in source_text
    assert "launch_float32_to_bf16_bits_many" in source_text
    assert "nfn_native_tile_trainer_linear_bf16_workspace_allocation_count" in source_text
    assert "nfn_native_tile_trainer_linear_bf16_cached_a_capacity" in source_text
    assert "nfn_native_tile_trainer_linear_bf16_cache_entry_count" in source_text
    assert "f32_to_bf16_bits_kernel" in kernels_text
    assert "f32_to_bf16_bits_many_kernel" in kernels_text
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
    assert "launch_linear_backward_weight_accumulate_bf16_float32" in kernels_text
    assert "linear_backward_weight_chunked_atomic_float32_bf16_bits_kernel" in kernels_text
    assert "nfn_native_tile_linear_backward_weight_accumulate_bf16_float32" in header_text
    assert "nfn_native_tile_linear_backward_weight_accumulate_bf16_float32" in source_text
    assert "cublas_linear_gemm_ex_bf16_float32" in kernels_text
    assert "CUDA_R_16BF" in kernels_text
    assert "CUBLAS_COMPUTE_32F" in kernels_text
    assert "cublasLtMatmul" in kernels_text
    assert "CUBLASLT_EPILOGUE_BGRADB" in kernels_text
    assert "CUBLASLT_MATMUL_DESC_BIAS_POINTER" in kernels_text
    assert "cublas_linear_gemm_ex_bf16_float32_with_bgrad" in kernels_text
    assert "cublas_linear_gemm_ex_bf16_bits_a_float32_with_bgrad" in kernels_text
    assert "CUBLAS_COMPUTE_32F_FAST_TF32" in kernels_text
    assert "CUBLAS_COMPUTE_32F_FAST_16BF" in kernels_text
    assert "NFN_TILE_CUDA_LINEAR_BF16" in kernels_text
    assert "NFN_NATIVE_LINEAR_BF16" in kernels_text
    assert "NFN_TILE_CUDA_LINEAR_BF16_CUBLASLT" in kernels_text
    assert "NFN_NATIVE_LINEAR_BF16_CUBLASLT" in kernels_text
    assert "NFN_TILE_CUDA_LINEAR_CUBLASLT" in kernels_text
    assert "NFN_NATIVE_LINEAR_CUBLASLT" in kernels_text
    assert "NFN_TILE_CUDA_LINEAR_TK_GEMM" in kernels_text
    assert "NFN_NATIVE_LINEAR_TK_GEMM" in kernels_text
    assert "NFN_TILE_CUDA_LINEAR_TK_FLOAT_OUT" in kernels_text
    assert "NFN_NATIVE_LINEAR_TK_FLOAT_OUT" in kernels_text
    assert "return false;" in kernels_text
    assert 'std::strcmp(value, "1") == 0' in kernels_text
    assert "tf32-cublaslt-optimized" in gpt2_source_text
    assert "tf32-sgemm-optimized" in gpt2_source_text
    assert "lm_head_logits_linear_strategy" in gpt2_source_text
    assert "linear_tk_gemm_count" in gpt2_source_text
    assert "linear_tk_float_out_gemm_count" in gpt2_source_text
    assert "block-bf16-cublaslt-shape-gated-lm-head-tk-sm120-default" in gpt2_source_text
    assert "padded-lm-head-tk-sm120-bf16-gemm-default" in gpt2_source_text
    assert "block-forward-dinput-dweight-bf16-lm-head-tf32" in gpt2_source_text
    assert "bf16-shadow-weight-shape-gated-cublaslt-forward" in gpt2_source_text
    assert "bf16-shadow-weight-shape-gated-cublaslt-dinput" in gpt2_source_text
    assert "block_backward_mlp_proj_dgelu_strategy" in gpt2_source_text
    assert "tk-sm120-fused-dinput-dgelu-bf16-store-bf16-shadow-weight-float32-grad" in gpt2_source_text
    assert "shape-gated-bf16-cublaslt-dweight-bgrad-accumulate" in gpt2_source_text
    assert "forced-bf16-gemmex-dweight-plus-bias-accumulate-fallback" in gpt2_source_text
    assert "bf16-shadow-weight-gemmex-forward" in gpt2_source_text
    assert "bf16-shadow-weight-gemmex-dinput" in gpt2_source_text
    assert "non_block_forward_backward_linear_strategy" in gpt2_source_text
    assert ".forward.no_bias.bf16" in gpt2_source_text
    assert ".backward_input.bf16" in gpt2_source_text
    assert "trainer_linear_bf16_gemm_count" in kernels_text
    assert "trainer_linear_tk_gemm_count" in kernels_text
    assert "trainer_linear_tk_float_out_gemm_count" in kernels_text
    assert "trainer_linear_bf16_b_operand" in kernels_text
    assert "trainer_linear_bf16_a_operand" in kernels_text
    assert "trainer_linear_cublaslt_heuristic_index_override" in kernels_text
    assert 'std::getenv("NFN_TILE_CUDA_CUBLASLT_HEURISTIC_INDEX")' in kernels_text
    assert 'std::getenv("NFN_NATIVE_LINEAR_CUBLASLT_HEURISTIC_INDEX")' in kernels_text
    assert "int selected = returned > 1 ? 1 : 0" in kernels_text
    assert "tk_linear_backward_input_dgelu_bf16_bits_float32" in kernels_text
    assert "tk_linear_backward_input_dgelu_weight_bf16_bits_float32" in kernels_text
    assert "matmul_dispatch_tk_ab" in kernels_text
    assert "kLayerNormBackwardAffineDefaultRowChunkSize = 256" in kernels_text
    assert "NFN_TILE_CUDA_LAYERNORM_AFFINE_ROW_CHUNK_SIZE" in kernels_text
    assert "kLinearBackwardBiasRowChunkSize = 512" in kernels_text
    for function_name in (
        "launch_linear_backward_weight_accumulate_bf16_bits_float32",
        "launch_linear_backward_weight_accumulate_float32_bf16_bits",
    ):
        function_body = kernels_text.split(f"void {function_name}", 1)[1].split("\nvoid ", 1)[0]
        assert "kRowChunkSize = kLinearBackwardBiasRowChunkSize" in function_body
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
    assert "nfn_native_tile_linear_bias_residual_layer_norm_float32" in header_text
    assert "nfn_native_tile_linear_bias_residual_layer_norm_with_stats_float32" in header_text
    assert "nfn_native_tile_linear_bias_residual_layer_norm_with_stats_bf16_residual_float32" in header_text
    assert "launch_linear_bias_residual_layer_norm_float32" in source_text
    assert "launch_linear_bias_residual_layer_norm_with_stats_float32" in source_text
    assert "launch_linear_bias_residual_layer_norm_with_stats_bf16_residual_float32" in source_text
    assert "nfn_native_tile_gelu_float32" in header_text
    assert "nfn_native_tile_gelu_add_bias_float32" in header_text
    assert "nfn_native_tile_gelu_backward_float32" in header_text
    assert "nfn_native_tile_gelu_backward_inplace_float32" in header_text
    assert "launch_gelu_backward_inplace_float32" in source_text
    assert "nfn_native_tile_token_embedding_float32" in header_text
    assert "nfn_native_tile_token_embedding_backward_weight_float32" in header_text
    assert "nfn_native_tile_absolute_position_embedding_float32" in header_text
    assert "nfn_native_tile_absolute_position_embedding_backward_float32" in header_text
    assert "nfn_native_tile_absolute_position_embedding_backward_accumulate_float32" in header_text
    assert "nfn_native_tile_layer_norm_float32" in header_text
    assert "nfn_native_tile_layer_norm_with_stats_float32" in header_text
    assert "nfn_native_tile_layer_norm_backward_input_float32" in header_text
    assert "nfn_native_tile_layer_norm_backward_input_with_stats_float32" in header_text
    assert "nfn_native_tile_layer_norm_backward_input_residual_add_with_stats_float32" in header_text
    assert "nfn_native_tile_layer_norm_backward_input_residual_add_with_stats_bf16_bits_float32" in header_text
    assert "nfn_native_tile_layer_norm_backward_affine_float32" in header_text
    assert "nfn_native_tile_layer_norm_backward_affine_accumulate_float32" in header_text
    assert "nfn_native_tile_layer_norm_backward_affine_accumulate_with_stats_float32" in header_text
    assert "nfn_native_tile_layer_norm_backward_affine_accumulate_with_stats_bf16_bits_float32" in header_text
    assert "launch_layer_norm_with_stats_float32" in source_text
    assert "launch_layer_norm_backward_input_with_stats_float32" in source_text
    assert "launch_layer_norm_backward_input_residual_add_with_stats_float32" in source_text
    assert "launch_layer_norm_backward_affine_accumulate_with_stats_float32" in source_text
    assert "layer_norm_backward_input_residual_add_with_stats_float32_kernel" in kernels_text
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
    assert "nfn_native_tile_masked_token_cross_entropy_partials_float32" in header_text
    assert "nfn_native_tile_token_cross_entropy_backward_float32" in header_text
    assert "nfn_native_tile_masked_token_cross_entropy_backward_float32" in header_text
    assert "nfn_native_tile_token_cross_entropy_backward_with_workspace_float32" in header_text
    assert "nfn_native_tile_token_cross_entropy_backward_inplace_with_workspace_float32" in header_text
    assert "nfn_native_tile_token_cross_entropy_backward_inplace_bf16_bits_with_workspace" in header_text
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
    assert "launch_tk_attention_packed_qkv_forward_bf16_float32" in kernels_text
    assert "launch_tk_attention_packed_qkv_forward_store_lse_bf16_float32" in kernels_text
    assert "launch_tk_attention_packed_qkv_backward_to_qkv_float32" in kernels_text
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
    assert "linear_backward_weight_chunked_atomic_float32_kernel" in kernels_text
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
    assert "attention_residual_ln2_strategy" in gpt2_source_text
    assert "fused-linear-bias-residual-layernorm" in gpt2_source_text
    assert "token_cross_entropy_backward_rowwise_float32_kernel" in kernels_text
    assert "token_cross_entropy_backward_rowwise_inplace_float32_kernel" in kernels_text
    assert "token_cross_entropy_row_stats_float32_kernel" in kernels_text
    assert "token_cross_entropy_bf16_bits_row_stats_kernel" in kernels_text
    assert "token_cross_entropy_backward_chunked_float32_kernel" in kernels_text
    assert "token_cross_entropy_backward_chunked_inplace_float32_kernel" in kernels_text
    assert "token_cross_entropy_backward_inplace_bf16_bits_kernel" in kernels_text
    assert "token_cross_entropy_backward_inplace_bf16_bits_fused_kernel" in kernels_text
    assert "block_reduce_max_f32" in kernels_text
    assert "block_reduce_sum_f32" in kernels_text
    assert "token_cross_entropy_backward_elementwise_float32_kernel" not in kernels_text
    assert "gelu_float32_kernel" in kernels_text
    assert "gelu_backward_float32_kernel" in kernels_text
    assert "gelu_backward_inplace_float32_kernel" in kernels_text
    assert "token_embedding_backward_weight_float32_kernel" in kernels_text
    assert "init_gpt2_token_weight_float32_kernel" in kernels_text
    assert "uint16_to_int64_kernel" in kernels_text
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
    assert "validation_wall_ms" in gpt2_source_text
    assert "checkpoint_wall_ms" in gpt2_source_text
    assert 'std::getenv("CUDA_VISIBLE_DEVICES")' in gpt2_source_text
    assert 'setenv("CUDA_VISIBLE_DEVICES", "0", 0)' in gpt2_source_text
    assert 'std::getenv("CUDA_DEVICE_MAX_CONNECTIONS")' in gpt2_source_text
    assert "--no-checkpoint" in gpt2_source_text
    assert "--native-cuda-no-checkpoint" in gpt2_source_text
    assert "cfg.write_checkpoint = false" in gpt2_source_text
    assert "checkpoint_export_enabled" in gpt2_source_text
    assert '\\"enabled\\": ' in gpt2_source_text
    assert "train_tokens_per_second" in gpt2_source_text
    assert "NFN_NATIVE_GPT_STAGE_TIMING" in gpt2_source_text
    assert "NFN_NATIVE_GPT2_STAGE_TIMING" in gpt2_source_text
    assert "cudaEventCreateWithFlags" in gpt2_source_text
    assert "cudaEventElapsedTime" in gpt2_source_text
    assert "stage_timing_enabled" in gpt2_source_text
    assert "stage_timing_event_count" in gpt2_source_text
    assert "stage_timing_dropped_event_count" in gpt2_source_text
    assert "stage_timing" in gpt2_source_text
    assert "block_backward" in gpt2_source_text
    assert "block_recompute" in gpt2_source_text
    assert "lm_head_backward" in gpt2_source_text
    assert "lm_head_backward.dhidden" in gpt2_source_text
    assert "lm_head_backward.dweight" in gpt2_source_text
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
    assert "compute_final_output" in gpt2_source_text
    assert "stored_mlp_activation_store_kernel_launches" in gpt2_source_text
    assert "stored_mlp_layer_norm_stats_elements" in gpt2_source_text
    assert "stored_mlp_layer_norm_stats_bytes" in gpt2_source_text
    assert "stored_mlp_activation_backward_consumer_strategy" in gpt2_source_text
    assert "NFN_NATIVE_GPT_REUSE_PACKED_LN2_FC_GELU" in gpt2_source_text
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
    assert "kDefaultStoredPackedAttentionBlocks = 12" in gpt2_source_text
    assert "NFN_NATIVE_GPT_STORE_RESIDUAL1_ACTIVATIONS" in gpt2_source_text
    assert "NFN_NATIVE_GPT2_STORE_RESIDUAL1_ACTIVATIONS" in gpt2_source_text
    assert "NFN_NATIVE_GPT_FUSE_RESIDUAL1_STORE" in gpt2_source_text
    assert "NFN_NATIVE_GPT2_FUSE_RESIDUAL1_STORE" in gpt2_source_text
    assert "env_flag_enabled_or_default(store_residual1_activations_env, true)" in gpt2_source_text
    assert "stored_residual1_activation_blocks" in gpt2_source_text
    assert "residual1_activation_store_strategy" in gpt2_source_text
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
    assert "kDefaultLmHeadRowChunkSize = 8192" in gpt2_source_text
    assert "NFN_NATIVE_GPT_PACKED_ATTENTION_BACKWARD_BATCH_CAP" in kernels_text
    assert "NFN_NATIVE_GPT2_PACKED_ATTENTION_BACKWARD_BATCH_CAP" in kernels_text
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
    assert "stored_packed_attention_backward_consumer_strategy" in gpt2_source_text
    assert "recompute_block_from_saved_packed_attention" in gpt2_source_text
    assert "recompute_block_from_saved_attention" in gpt2_source_text
    assert "attention_backward_uses_saved_forward_workspace" in gpt2_source_text
    assert "backward_recompute_mlp_fc_gelu_elided" in gpt2_source_text
    assert "backward_recompute_attention_qkv_sdpa_elided" in gpt2_source_text
    assert "backward_recompute_attention_uses_saved_o" in gpt2_source_text
    assert "backward_recompute_mlp_projection_elided" in gpt2_source_text
    assert "backward_recompute_final_residual_elided" in gpt2_source_text
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
    assert "nfn_native_tile_attention_forward_tk_launch_count" in header_text
    assert "nfn_native_tile_attention_backward_tk_launch_count" in header_text
    assert "nfn_native_tile_attention_forward_tk_launch_count" in source_text
    assert "NFN_TILE_CUDA_USE_TK_ATTENTION:-1" in script_text
    assert "LLM_KITTENS_ROOT" in script_text
    assert "TK_ROOT" in script_text
    assert "sm_120a" in script_text
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
        nanogpt = native_out / "nfn_nanogpt_native_train"
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
        assert lm_payload["max_grad_abs_error"] <= 1e-6
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
        assert fused_qkv_attention_payload["max_q_abs_error"] <= 1e-5
        assert fused_qkv_attention_payload["max_k_abs_error"] <= 1e-5
        assert fused_qkv_attention_payload["max_v_abs_error"] <= 1e-5
        assert fused_qkv_attention_payload["max_attn_abs_error"] <= 1e-5
        assert fused_qkv_attention_payload["max_out_abs_error"] <= 1e-5
        assert fused_qkv_attention_payload["max_grad_x_abs_error"] <= 1e-5
        assert fused_qkv_attention_payload["max_grad_qkv_weight_abs_error"] <= 1e-5
        assert fused_qkv_attention_payload["max_grad_out_weight_abs_error"] <= 1e-5
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
        assert mlp_payload["max_out_abs_error"] <= 1e-5
        assert mlp_payload["max_grad_x_abs_error"] <= 1e-5
        assert mlp_payload["max_fc_grad_abs_error"] <= 1e-5
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
        assert attention_payload["max_attn_abs_error"] <= 1e-5
        assert attention_payload["max_out_abs_error"] <= 1e-5
        assert attention_payload["max_grad_q_weight_abs_error"] <= 1e-6
        assert attention_payload["max_grad_k_weight_abs_error"] <= 1e-6
        assert attention_payload["max_grad_v_weight_abs_error"] <= 1e-5
        assert attention_payload["max_grad_out_weight_abs_error"] <= 1e-5
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
        assert "nfn_native_tile_copy_float32" in exported
        assert "nfn_native_tile_uint16_to_int64" in exported
        assert "nfn_native_tile_float32_to_bf16_bits" in exported
        assert "nfn_native_tile_bf16_bits_to_float32" in exported
        assert "nfn_native_tile_bf16_bits_add_bias_inplace_float32" in exported
        assert "nfn_native_tile_store_mlp_activations_bf16_float32" in exported
        assert "nfn_native_tile_restore_mlp_activations_bf16_float32" in exported
        assert "nfn_native_tile_float32_to_bf16_bits_many" in exported
        assert "nfn_native_tile_linear_bf16_float32" in exported
        assert "nfn_native_tile_linear_weight_bf16_float32" in exported
        assert "nfn_native_tile_linear_bf16_output_float32" in exported
        assert "nfn_native_tile_linear_weight_bf16_output_float32" in exported
        assert "nfn_native_tile_linear_bf16_input_bits_float32" in exported
        assert "nfn_native_tile_linear_bf16_input_weight_bf16_float32" in exported
        assert "nfn_native_tile_linear_bf16_gelu_bf16_float32" in exported
        assert "nfn_native_tile_linear_weight_bf16_gelu_bf16_float32" in exported
        assert "nfn_native_tile_linear_bf16_input_weight_bf16_gelu_bf16_float32" in exported
        assert "nfn_native_tile_linear_backward_input_bf16_float32" in exported
        assert "nfn_native_tile_linear_backward_input_weight_bf16_float32" in exported
        assert "nfn_native_tile_linear_backward_input_bf16_bits_float32" in exported
        assert "nfn_native_tile_linear_backward_input_dgelu_weight_bf16_bits_float32" in exported
        assert "nfn_native_tile_gelu_add_bias_bf16_act_float32" in exported
        assert "nfn_native_tile_trainer_linear_stats_reset" in exported
        assert "nfn_native_tile_trainer_linear_bf16_cache_reset" in exported
        assert "nfn_native_tile_trainer_linear_bf16_gemm_count" in exported
        assert "nfn_native_tile_trainer_linear_tk_gemm_count" in exported
        assert "nfn_native_tile_trainer_linear_tk_float_out_gemm_count" in exported
        assert "nfn_native_tile_trainer_linear_cublaslt_gemm_count" in exported
        assert "nfn_native_tile_trainer_linear_sgemm_count" in exported
        assert "nfn_native_tile_trainer_linear_bf16_a_pack_count" in exported
        assert "nfn_native_tile_trainer_linear_bf16_a_cache_hit_count" in exported
        assert "nfn_native_tile_trainer_linear_bf16_cache_reset_count" in exported
        assert "nfn_native_tile_trainer_linear_bf16_workspace_allocation_count" in exported
        assert "nfn_native_tile_trainer_linear_bf16_cached_a_capacity" in exported
        assert "nfn_native_tile_trainer_linear_bf16_cache_entry_count" in exported
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
        assert "nfn_native_tile_linear_backward_weight_accumulate_float32_bf16_bits" in exported
        assert "nfn_native_tile_linear_backward_bias_float32" in exported
        assert "nfn_native_tile_linear_backward_bias_accumulate_float32" in exported
        assert "nfn_native_tile_scaled_residual_add_float32" in exported
        assert "nfn_native_tile_gelu_float32" in exported
        assert "nfn_native_tile_gelu_backward_float32" in exported
        assert "nfn_native_tile_gelu_backward_inplace_bf16_bits_float32" in exported
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
        assert "nfn_native_tile_masked_token_cross_entropy_partials_float32" in exported
        assert "nfn_native_tile_token_cross_entropy_backward_float32" in exported
        assert "nfn_native_tile_masked_token_cross_entropy_backward_float32" in exported
        assert "nfn_native_tile_token_cross_entropy_backward_with_workspace_float32" in exported
        assert "nfn_native_tile_token_cross_entropy_backward_inplace_with_workspace_float32" in exported
        assert "nfn_native_tile_token_cross_entropy_backward_inplace_bf16_bits_with_workspace" in exported
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


def test_native_gpt2_command_installer_links_temp_bin(tmp_path: Path) -> None:
    if shutil.which("c++") is None:
        pytest.skip("c++ compiler not available")
    root = Path(__file__).resolve().parents[1]
    native_cli = tmp_path / "nfn_gpt_native_train"
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
    env["NFN_NATIVE_GPT_CLI"] = str(native_cli)
    env["NFN_NATIVE_GPT2_CLI"] = str(compat_native_cli)
    env["NFN_NATIVE_TRAIN_CLI"] = str(native_train_cli)
    env["NFN_NATIVE_GPT2_LAUNCHER"] = str(launcher)
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
    linked_nanogpt_underscore = bin_dir / "nfn_nanogpt_native_train"
    linked_nanogpt = bin_dir / "nfn-nanogpt-native-train"
    assert linked_native.is_symlink()
    assert linked_train.is_symlink()
    assert linked_gpt_native.is_symlink()
    assert linked_gpt_train.is_symlink()
    assert linked_gpt2_compat.is_symlink()
    assert linked_unified.is_symlink()
    assert linked_launcher.is_symlink()
    assert linked_nanogpt_underscore.is_symlink()
    assert linked_nanogpt.is_symlink()
    assert linked_native.resolve() == native_cli
    assert linked_train.resolve() == native_cli
    assert linked_gpt_native.resolve() == native_cli
    assert linked_gpt_train.resolve() == native_cli
    assert linked_gpt2_compat.resolve() == compat_native_cli
    assert linked_unified.resolve() == native_train_cli
    assert linked_launcher.resolve() == launcher
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
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert installed_dispatch.returncode == 2
    installed_dispatch_json_start = installed_dispatch.stdout.find("{")
    assert installed_dispatch_json_start >= 0
    installed_dispatch_plan = json.loads(installed_dispatch.stdout[installed_dispatch_json_start:])
    assert installed_dispatch_plan["model_family"] == "nanogpt"
    assert installed_dispatch_plan["dataset_alias"] == "/tmp/native-cache"
    assert "nfn_nanogpt_native_train: native CUDA Tile trainer for nanogpt is not implemented yet" in installed_dispatch.stderr

    installed_print_command = subprocess.run(
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
    assert installed_print_command.returncode == 0, installed_print_command.stderr
    assert str(linked_nanogpt_underscore) in installed_print_command.stdout
    assert "--dry-run" in installed_print_command.stdout
    assert "--print-command" in installed_print_command.stdout
