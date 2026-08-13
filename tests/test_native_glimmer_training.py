from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

import pytest
import torch

from server.dataset_manager import build_token_shard_v2_header, write_structured_sft_v1


ROOT = Path(__file__).resolve().parents[1]
GEOMETRY = "8,16,2,1,2,4,13,3,16"


def _compile(command: list[str]) -> None:
    completed = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=False
    )
    assert completed.returncode == 0, completed.stderr


@pytest.fixture(scope="module")
def glimmer_native_train_tools(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    compiler = shutil.which("g++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++ compiler is unavailable")
    root = tmp_path_factory.mktemp("glimmer-native-train")
    runtime = root / "libfake_cuda_runtime.so"
    tile = root / "libfake_glimmer_tile_ops.so"
    trainer = root / "nfn_muse_glimmer_native_train"
    _compile(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-Wall",
            "-Wextra",
            "-pedantic",
            "-fPIC",
            "-shared",
            str(ROOT / "tests/cpp/fake_cuda_runtime.cpp"),
            "-o",
            str(runtime),
        ]
    )
    _compile(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-Wall",
            "-Wextra",
            "-pedantic",
            "-fPIC",
            "-shared",
            "-I",
            str(ROOT / "neuralfn/csrc/native_train"),
            str(ROOT / "tests/cpp/fake_glimmer_tile_ops.cpp"),
            "-o",
            str(tile),
        ]
    )
    _compile(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-Wall",
            "-Wextra",
            "-pedantic",
            "-I",
            str(ROOT / "neuralfn/csrc/native_train"),
            "-I",
            str(ROOT / "neuralfn/csrc/native_gpt2"),
            str(ROOT / "neuralfn/csrc/native_train/muse_glimmer_native_train.cpp"),
            str(ROOT / "neuralfn/csrc/native_train/token_shards.cpp"),
            "-ldl",
            "-o",
            str(trainer),
        ]
    )
    return trainer, runtime, tile


def _tiny_layout() -> list[tuple[str, tuple[int, int], bool]]:
    result: list[tuple[str, tuple[int, int], bool]] = [
        ("token_embedding.weight", (13, 8), False)
    ]
    for layer in range(4):
        prefix = f"layers.{layer}."
        result.extend(
            [
                (prefix + "input_layernorm.weight", (1, 8), True),
                (prefix + "post_attention_layernorm.weight", (1, 8), True),
                (prefix + "pre_feedforward_layernorm.weight", (1, 8), True),
                (prefix + "post_feedforward_layernorm.weight", (1, 8), True),
                (prefix + "q_proj.weight", (4, 8), False),
                (prefix + "k_proj.weight", (2, 8), False),
                (prefix + "v_proj.weight", (2, 8), False),
                (prefix + "attn_gate_proj.weight", (4, 8), False),
                (prefix + "o_proj.weight", (8, 4), False),
                (prefix + "gate_proj.weight", (16, 8), False),
                (prefix + "up_proj.weight", (16, 8), False),
                (prefix + "down_proj.weight", (8, 16), False),
            ]
        )
    result.extend(
        [("final_norm.weight", (1, 8), False), ("lm_head.weight", (13, 8), False)]
    )
    return result


def _write_checkpoint(path: Path) -> str:
    generator = torch.Generator().manual_seed(20260813)
    with path.open("wb") as stream:
        for name, shape, centered in _tiny_layout():
            if centered:
                value = torch.randn(shape, generator=generator) * 0.01
            elif name == "final_norm.weight":
                value = torch.ones(shape)
            else:
                value = torch.randn(shape, generator=generator) * 0.04
            stream.write(value.to(torch.bfloat16).view(torch.uint16).numpy().tobytes())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_shard(path: Path, values: list[int], *, split: str) -> None:
    payload = struct.pack("<" + "I" * len(values), *values)
    path.write_bytes(
        build_token_shard_v2_header(
            token_count=len(values),
            tokenizer_vocab_size=13,
            tokenizer_sha256="a" * 64,
            tokenizer_revision="tiny-glimmer-v1",
            tokenizer_name="tiny-glimmer",
            split=split,
            objective="ar",
        )
        + payload
    )


def _base_command(tools: tuple[Path, Path, Path]) -> list[str]:
    trainer, runtime, tile = tools
    return [
        str(trainer),
        "--cuda-runtime-lib",
        str(runtime),
        "--tile-ops-lib",
        str(tile),
        "--tiny-geometry",
        GEOMETRY,
        "--sequence-length",
        "4",
        "--batch-size",
        "1",
        "--activation-checkpoint-interval",
        "2",
    ]


def test_nfn_train_routes_muse_glimmer_to_dedicated_family_binary(
    tmp_path: Path,
) -> None:
    trainer = tmp_path / "nfn_muse_glimmer_native_train"
    trainer.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n", encoding="utf-8")
    trainer.chmod(0o755)
    env = os.environ.copy()
    env.pop("NFN_NATIVE_TRAIN_CLI", None)
    env["NFN_NATIVE_MUSE_GLIMMER_CLI"] = str(trainer)

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "cli/nfn.py"),
            "train",
            "--base-model",
            "muse-glimmer",
            "--checkpoint",
            "/models/glimmer.bf16",
            "--dataset",
            "/datasets/glimmer-sft",
            "--objective",
            "sft",
            "--adapter",
            "qlora",
            "--sequence-length",
            "2048",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == [
        "--checkpoint",
        "/models/glimmer.bf16",
        "--dataset",
        "/datasets/glimmer-sft",
        "--objective",
        "sft",
        "--adapter",
        "qlora",
        "--sequence-length",
        "2048",
    ]


def test_native_glimmer_compiled_layout_and_training_kernel_contract(
    glimmer_native_train_tools: tuple[Path, Path, Path],
) -> None:
    trainer, _runtime, tile = glimmer_native_train_tools
    layout = subprocess.run(
        [str(trainer), "--print-parameter-layout", "--sequence-length", "128"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(layout.stdout)
    assert payload["tensor_count"] == 627
    assert payload["parameter_count"] == 27_854_780_928
    assert payload["tensors"][0]["name"] == "token_embedding.weight"
    assert payload["tensors"][-1]["name"] == "lm_head.weight"

    check = subprocess.run(
        [
            str(trainer),
            "--kernel-check",
            "--tile-ops-lib",
            str(tile),
            "--tiny-geometry",
            GEOMETRY,
            "--sequence-length",
            "4",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(check.stdout)["passed"] is True


def test_native_glimmer_train_save_and_strict_resume(
    glimmer_native_train_tools: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    source = tmp_path / "source.bf16"
    source_sha = _write_checkpoint(source)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    values = [1, 2, 3, 4, 5, 3, 2, 1, 6]
    _write_shard(dataset / "fineweb_train_000000.bin", values, split="train")
    _write_shard(dataset / "fineweb_val_000000.bin", values, split="validation")
    output = tmp_path / "output"
    first = subprocess.run(
        _base_command(glimmer_native_train_tools)
        + [
            "--checkpoint",
            str(source),
            "--checkpoint-sha256",
            source_sha,
            "--dataset",
            str(dataset),
            "--output-dir",
            str(output),
            "--max-steps",
            "1",
            "--checkpoint-every-steps",
            "1",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert first.returncode == 0, first.stderr
    checkpoint1 = output / "checkpoint-step-1"
    state1 = json.loads((checkpoint1 / "trainer_state.json").read_text())
    assert state1["schema"] == "neuralfn.muse_glimmer_native_training.v2"
    assert state1["completed_step"] == 1
    assert state1["source_sha256"] == source_sha
    assert state1["tokenizer_sha256"] == "a" * 64
    assert (checkpoint1 / "model.bf16").read_bytes() != source.read_bytes()

    second = subprocess.run(
        _base_command(glimmer_native_train_tools)
        + [
            "--resume-from-checkpoint",
            str(checkpoint1),
            "--checkpoint-sha256",
            source_sha,
            "--dataset",
            str(dataset),
            "--output-dir",
            str(output),
            "--max-steps",
            "2",
            "--checkpoint-every-steps",
            "1",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert second.returncode == 0, second.stderr
    checkpoint2 = output / "checkpoint-step-2"
    state2 = json.loads((checkpoint2 / "trainer_state.json").read_text())
    assert state2["completed_step"] == 2
    assert state2["source_sha256"] == source_sha
    assert state2["model_sha256"] != state1["model_sha256"]

    corrupt = bytearray((checkpoint2 / "optimizer.f32").read_bytes())
    corrupt[-1] ^= 1
    (checkpoint2 / "optimizer.f32").write_bytes(corrupt)
    rejected = subprocess.run(
        _base_command(glimmer_native_train_tools)
        + [
            "--resume-from-checkpoint",
            str(checkpoint2),
            "--dataset",
            str(dataset),
            "--output-dir",
            str(output),
            "--max-steps",
            "3",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert rejected.returncode == 2
    assert "authentication failed" in rejected.stderr


def test_native_glimmer_structured_sft_masks_boundaries_and_resume_lineage(
    glimmer_native_train_tools: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    source = tmp_path / "source.bf16"
    source_sha = _write_checkpoint(source)
    dataset = tmp_path / "sft"
    template_sha = "b" * 64
    common = {
        "sequence_length": 4,
        "tokenizer_vocab_size": 13,
        "pad_token_id": 0,
        "tokenizer_sha256": "a" * 64,
        "chat_template_sha256": template_sha,
        "tokenizer_revision": "tiny-glimmer-v1",
    }
    records = [
        {
            "input_ids": [1, 2, 3, 4],
            "targets": [-100, 3, -100, 5],
            "loss_mask": [0.0, 1.0, 0.0, 1.0],
            "sequence_ids": [0, 0, 1, 1],
        }
    ]
    write_structured_sft_v1(
        dataset / "sft_train_000000.sft", records, split="train", **common
    )
    write_structured_sft_v1(
        dataset / "sft_val_000000.sft", records, split="validation", **common
    )
    output = tmp_path / "out"
    command = _base_command(glimmer_native_train_tools) + [
        "--checkpoint",
        str(source),
        "--checkpoint-sha256",
        source_sha,
        "--dataset",
        str(dataset),
        "--output-dir",
        str(output),
        "--objective",
        "sft",
        "--chat-template-sha256",
        template_sha,
        "--max-steps",
        "1",
        "--checkpoint-every-steps",
        "1",
    ]
    completed = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    state = json.loads((output / "checkpoint-step-1/trainer_state.json").read_text())
    assert state["objective"] == "sft"
    assert state["chat_template_sha256"] == template_sha
    assert state["packed_sequence_boundaries"] is True

    wrong_lineage = subprocess.run(
        _base_command(glimmer_native_train_tools)
        + [
            "--resume-from-checkpoint",
            str(output / "checkpoint-step-1"),
            "--dataset",
            str(dataset),
            "--output-dir",
            str(output),
            "--objective",
            "sft",
            "--chat-template-sha256",
            "c" * 64,
            "--max-steps",
            "2",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert wrong_lineage.returncode == 2
    assert "template" in wrong_lineage.stderr.lower()


def test_native_glimmer_lora_is_adapter_only_deterministic_and_strictly_resumable(
    glimmer_native_train_tools: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    source = tmp_path / "source.bf16"
    source_sha = _write_checkpoint(source)
    source_bytes = source.read_bytes()
    dataset = tmp_path / "sft"
    template_sha = "b" * 64
    common = {
        "sequence_length": 4,
        "tokenizer_vocab_size": 13,
        "pad_token_id": 0,
        "tokenizer_sha256": "a" * 64,
        "chat_template_sha256": template_sha,
        "tokenizer_revision": "tiny-glimmer-v1",
    }
    records = [
        {
            "input_ids": [1, 2, 3, 4],
            "targets": [-100, 3, 4, 5],
            "loss_mask": [0.0, 1.0, 1.0, 1.0],
            "sequence_ids": [0, 0, 0, 0],
        }
    ]
    write_structured_sft_v1(
        dataset / "sft_train_000000.sft", records, split="train", **common
    )
    write_structured_sft_v1(
        dataset / "sft_val_000000.sft", records, split="validation", **common
    )

    def command(output: Path) -> list[str]:
        return _base_command(glimmer_native_train_tools) + [
            "--checkpoint",
            str(source),
            "--checkpoint-sha256",
            source_sha,
            "--dataset",
            str(dataset),
            "--output-dir",
            str(output),
            "--objective",
            "sft",
            "--chat-template-sha256",
            template_sha,
            "--adapter",
            "lora",
            "--lora-rank",
            "2",
            "--lora-alpha",
            "4",
            "--lora-dropout",
            "0.25",
            "--lora-seed",
            "1234",
            "--max-steps",
            "1",
            "--checkpoint-every-steps",
            "1",
        ]

    output = tmp_path / "out"
    first = subprocess.run(
        command(output), cwd=ROOT, text=True, capture_output=True, check=False
    )
    assert first.returncode == 0, first.stderr
    checkpoint1 = output / "checkpoint-step-1"
    assert source.read_bytes() == source_bytes
    assert not (checkpoint1 / "model.bf16").exists()
    assert (checkpoint1 / "adapter.bf16").is_file()
    assert (checkpoint1 / "adapter_optimizer.f32").is_file()
    assert (checkpoint1 / "trainer_state.lora.v1").is_file()
    state1 = json.loads((checkpoint1 / "trainer_state.json").read_text())
    manifest1 = json.loads((checkpoint1 / "adapter_manifest.json").read_text())
    assert state1["schema"] == "neuralfn.muse_glimmer_native_lora_training.v1"
    assert state1["adapter_only"] is True
    assert state1["base_frozen"] is True
    assert state1["source_sha256"] == source_sha
    assert state1["rank"] == 2
    assert state1["alpha"] == 4
    assert state1["dropout"] == pytest.approx(0.25)
    assert manifest1["format"] == "neuralfn.native_muse_glimmer_lora.bf16.v1"
    assert manifest1["base_sha256"] == source_sha
    assert len(manifest1["tensors"]) == 4 * 8 * 2
    assert all(len(tensor["sha256"]) == 64 for tensor in manifest1["tensors"])
    assert manifest1["tensors"][0]["name"] == "layers.0.q_proj.weight.lora_A"
    assert manifest1["tensors"][1]["name"] == "layers.0.q_proj.weight.lora_B"

    # A fresh run with the same seed/data produces the same dropout masks,
    # initialization, gradients, and adapter artifact.
    deterministic = tmp_path / "deterministic"
    repeated = subprocess.run(
        command(deterministic), cwd=ROOT, text=True, capture_output=True, check=False
    )
    assert repeated.returncode == 0, repeated.stderr
    assert (
        deterministic / "checkpoint-step-1/adapter.bf16"
    ).read_bytes() == (checkpoint1 / "adapter.bf16").read_bytes()

    resumed_command = command(output)
    resumed_command[resumed_command.index("--max-steps") + 1] = "2"
    resumed_command.extend(["--resume-from-checkpoint", str(checkpoint1)])
    resumed = subprocess.run(
        resumed_command, cwd=ROOT, text=True, capture_output=True, check=False
    )
    assert resumed.returncode == 0, resumed.stderr
    checkpoint2 = output / "checkpoint-step-2"
    state2 = json.loads((checkpoint2 / "trainer_state.json").read_text())
    assert state2["completed_step"] == 2
    assert state2["adapter_sha256"] != state1["adapter_sha256"]
    assert source.read_bytes() == source_bytes

    wrong_rank = resumed_command.copy()
    wrong_rank[wrong_rank.index("--lora-rank") + 1] = "3"
    rejected = subprocess.run(
        wrong_rank, cwd=ROOT, text=True, capture_output=True, check=False
    )
    assert rejected.returncode == 2
    assert "configuration mismatch" in rejected.stderr


def test_native_glimmer_qlora_keeps_nf4_base_immutable_and_resumes_strictly(
    glimmer_native_train_tools: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    source = tmp_path / "source.bf16"
    source_sha = _write_checkpoint(source)
    source_bytes = source.read_bytes()
    dataset = tmp_path / "sft"
    template_sha = "d" * 64
    common = {
        "sequence_length": 4,
        "tokenizer_vocab_size": 13,
        "pad_token_id": 0,
        "tokenizer_sha256": "a" * 64,
        "chat_template_sha256": template_sha,
        "tokenizer_revision": "tiny-glimmer-v1",
    }
    records = [
        {
            "input_ids": [1, 2, 3, 4],
            "targets": [-100, 3, 4, 5],
            "loss_mask": [0.0, 1.0, 1.0, 1.0],
            "sequence_ids": [0, 0, 0, 0],
        }
    ]
    write_structured_sft_v1(
        dataset / "sft_train_000000.sft", records, split="train", **common
    )
    write_structured_sft_v1(
        dataset / "sft_val_000000.sft", records, split="validation", **common
    )

    def command(output: Path) -> list[str]:
        return _base_command(glimmer_native_train_tools) + [
            "--checkpoint",
            str(source),
            "--checkpoint-sha256",
            source_sha,
            "--dataset",
            str(dataset),
            "--output-dir",
            str(output),
            "--objective",
            "sft",
            "--chat-template-sha256",
            template_sha,
            "--adapter",
            "qlora",
            "--lora-targets",
            "q_proj,v_proj,down_proj",
            "--lora-rank",
            "2",
            "--lora-alpha",
            "4",
            "--qlora-group-size",
            "64",
            "--max-steps",
            "1",
            "--checkpoint-every-steps",
            "1",
        ]

    output = tmp_path / "out"
    first = subprocess.run(
        command(output), cwd=ROOT, text=True, capture_output=True, check=False
    )
    assert first.returncode == 0, first.stderr
    checkpoint1 = output / "checkpoint-step-1"
    state1 = json.loads((checkpoint1 / "trainer_state.json").read_text())
    manifest1 = json.loads((checkpoint1 / "adapter_manifest.json").read_text())
    assert state1["adapter"] == "qlora"
    assert state1["base_frozen"] is True
    assert state1["adapter_only"] is True
    assert manifest1["base_weight_precision"] == "bf16"
    assert manifest1["training_base_precision"] == "nf4-group64-fp32-scale"
    assert manifest1["training_adapter"] == "qlora"
    assert manifest1["targets"] == ["q_proj", "v_proj", "down_proj"]
    assert len(manifest1["tensors"]) == 4 * 3 * 2
    assert not (checkpoint1 / "model.bf16").exists()
    assert source.read_bytes() == source_bytes

    deterministic = tmp_path / "deterministic"
    repeated = subprocess.run(
        command(deterministic), cwd=ROOT, text=True, capture_output=True, check=False
    )
    assert repeated.returncode == 0, repeated.stderr
    assert (
        deterministic / "checkpoint-step-1/adapter.bf16"
    ).read_bytes() == (checkpoint1 / "adapter.bf16").read_bytes()

    resumed_command = command(output)
    resumed_command[resumed_command.index("--max-steps") + 1] = "2"
    resumed_command.extend(["--resume-from-checkpoint", str(checkpoint1)])
    resumed = subprocess.run(
        resumed_command, cwd=ROOT, text=True, capture_output=True, check=False
    )
    assert resumed.returncode == 0, resumed.stderr
    assert (output / "checkpoint-step-2/adapter.bf16").is_file()
    assert source.read_bytes() == source_bytes

    wrong_mode = resumed_command.copy()
    wrong_mode[wrong_mode.index("qlora")] = "lora"
    rejected = subprocess.run(
        wrong_mode, cwd=ROOT, text=True, capture_output=True, check=False
    )
    assert rejected.returncode == 2
    assert "configuration mismatch" in rejected.stderr
