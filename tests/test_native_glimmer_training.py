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

from server.dataset_manager import (
    build_token_shard_v2_header,
    write_structured_preference_v1,
    write_structured_ppo_prompt_v1,
    write_structured_sft_v1,
)


ROOT = Path(__file__).resolve().parents[1]
GEOMETRY = "8,16,2,1,2,4,13,3,16"
KQUANT_GEOMETRY = "256,512,4,2,64,4,17,3,16"


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


@pytest.fixture(scope="module")
def fake_nccl_library(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("g++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++ compiler is unavailable")
    root = tmp_path_factory.mktemp("glimmer-fake-nccl")
    library = root / "libfake_nccl.so"
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
            str(ROOT / "tests/cpp/fake_nccl.cpp"),
            "-o",
            str(library),
        ]
    )
    return library


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


def _write_checkpoint(path: Path, *, seed: int = 20260813) -> str:
    generator = torch.Generator().manual_seed(seed)
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


def _write_tiny_kquant_gguf(
    path: Path, *, unsupported_encoding: bool = False
) -> str:
    """Write a small GGUF-v3 model with real typed/strided K descriptors.

    Quantized blocks are all-zero correctness fixtures; layer zero and the
    embedding/head remain F32 so the LoRA optimizer receives nonzero gradients.
    """

    dim, intermediate, query_width, kv_width, head_dim, layers, vocab = (
        256,
        512,
        256,
        128,
        64,
        4,
        17,
    )
    tensors: list[tuple[str, int, int, int, bytes]] = []

    def f32_values(rows: int, cols: int, *, scale: float = 1.0) -> bytes:
        values = [scale * (((index * 13) % 29) - 14) / 700.0 for index in range(rows * cols)]
        return struct.pack("<" + "f" * len(values), *values)

    def add(name: str, rows: int, cols: int, encoding: int, payload: bytes) -> None:
        tensors.append((name, rows, cols, encoding, payload))

    add("token_embd.weight", vocab, dim, 0, f32_values(vocab, dim, scale=0.7))
    projection_shapes = {
        "attn_q": (query_width, dim),
        "attn_k": (kv_width, dim),
        "attn_v": (kv_width, dim),
        "attn_gate": (query_width, dim),
        "attn_output": (dim, query_width),
        "ffn_gate": (intermediate, dim),
        "ffn_up": (intermediate, dim),
        "ffn_down": (dim, intermediate),
    }
    norm_shapes = {
        "attn_norm": dim,
        "post_attention_norm": dim,
        "ffn_norm": dim,
        "post_ffw_norm": dim,
        "attn_q_norm": head_dim,
        "attn_k_norm": head_dim,
    }
    quant_types = (12, 13, 14)
    for layer in range(layers):
        prefix = f"blk.{layer}."
        for suffix, width in norm_shapes.items():
            value = 3.87 if suffix == "attn_q_norm" else 1.0
            add(
                prefix + suffix + ".weight",
                1,
                width,
                0,
                struct.pack("<" + "f" * width, *([value] * width)),
            )
        for projection_index, (suffix, (rows, cols)) in enumerate(
            projection_shapes.items()
        ):
            if layer == 0:
                encoding = 0
                payload = f32_values(rows, cols, scale=0.35)
            else:
                encoding = quant_types[(layer + projection_index) % len(quant_types)]
                block_bytes = {12: 144, 13: 176, 14: 210}[encoding]
                payload = bytes(rows * (cols // 256) * block_bytes)
            add(prefix + suffix + ".weight", rows, cols, encoding, payload)
    add("output.weight", vocab, dim, 0, f32_values(vocab, dim, scale=0.8))
    add("output_norm.weight", 1, dim, 0, struct.pack("<" + "f" * dim, *([1.0] * dim)))
    if unsupported_encoding:
        name, rows, cols, _encoding, payload = tensors[-3]
        tensors[-3] = (name, rows, cols, 99, payload)

    relative_offsets: list[int] = []
    data = bytearray()
    for _name, _rows, _cols, _encoding, payload in tensors:
        while len(data) % 32:
            data.append(0)
        relative_offsets.append(len(data))
        data.extend(payload)
    table = bytearray()
    for (name, rows, cols, encoding, _payload), offset in zip(
        tensors, relative_offsets, strict=True
    ):
        encoded = name.encode("utf-8")
        table.extend(struct.pack("<Q", len(encoded)))
        table.extend(encoded)
        dimensions = (cols,) if rows == 1 else (cols, rows)
        table.extend(struct.pack("<I", len(dimensions)))
        table.extend(struct.pack("<" + "Q" * len(dimensions), *dimensions))
        table.extend(struct.pack("<IQ", encoding, offset))
    header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 0))
    header.extend(table)
    while len(header) % 32:
        header.append(0)
    path.write_bytes(header + data)
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


def test_nfn_train_launches_one_glimmer_process_per_pipeline_rank(
    tmp_path: Path,
) -> None:
    trainer = tmp_path / "nfn_muse_glimmer_native_train"
    captured = tmp_path / "captured.txt"
    trainer.write_text(
        "#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$NFN_TEST_PIPELINE_CAPTURE\"\n",
        encoding="utf-8",
    )
    trainer.chmod(0o755)
    env = os.environ.copy()
    env.pop("NFN_NATIVE_TRAIN_CLI", None)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["NFN_NATIVE_MUSE_GLIMMER_CLI"] = str(trainer)
    env["NFN_TEST_PIPELINE_CAPTURE"] = str(captured)
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "cli/nfn.py"),
            "train",
            "--base-model",
            "muse-glimmer",
            "--checkpoint",
            "/models/glimmer.bf16",
            "--checkpoint-sha256",
            "a" * 64,
            "--dataset",
            "/datasets/glimmer",
            "--pipeline-parallel-size",
            "2",
            "--pipeline-cuda-devices",
            "3,4",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    lines = captured.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert any("--pipeline-parallel-rank 0" in line and "--cuda-device 3" in line for line in lines)
    assert any("--pipeline-parallel-rank 1" in line and "--cuda-device 4" in line for line in lines)
    bootstrap_values = {
        line.split("--distributed-id-file ", 1)[1].split()[0] for line in lines
    }
    assert len(bootstrap_values) == 1
    assert "--pipeline-cuda-devices" not in "\n".join(lines)


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


def test_native_glimmer_pipeline_plan_is_stage_local_and_fits_80gb_class(
    glimmer_native_train_tools: tuple[Path, Path, Path],
) -> None:
    trainer, _runtime, _tile = glimmer_native_train_tools
    completed = subprocess.run(
        [
            str(trainer),
            "--print-distributed-plan",
            "--pipeline-parallel-size",
            "8",
            "--pipeline-parallel-rank",
            "0",
            "--sequence-length",
            "128",
            "--batch-size",
            "1",
            "--activation-checkpoint-interval",
            "4",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    plan = json.loads(completed.stdout)
    assert plan["schema"] == "neuralfn.muse_glimmer_pipeline_plan.v1"
    assert plan["world_size"] == 8
    assert [(stage["layer_begin"], stage["layer_end"]) for stage in plan["stages"]] == [
        (0, 6),
        (6, 13),
        (13, 19),
        (19, 26),
        (26, 32),
        (32, 39),
        (39, 45),
        (45, 52),
    ]
    assert plan["stages"][0]["owns_embedding"] is True
    assert plan["stages"][-1]["owns_final_norm_and_head"] is True
    assert all(
        stage["required_bytes_before_reserve"] + plan["reserve_bytes_per_rank"]
        < 80 * 1024**3
        for stage in plan["stages"]
    )
    assert sum(stage["parameter_elements"] for stage in plan["stages"]) == 27_854_780_928


def test_native_glimmer_two_rank_pipeline_matches_single_rank_and_checkpoints(
    glimmer_native_train_tools: tuple[Path, Path, Path],
    fake_nccl_library: Path,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.bf16"
    source_sha = _write_checkpoint(source)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    values = [1, 2, 3, 4, 5, 3, 2, 1, 6]
    _write_shard(dataset / "fineweb_train_000000.bin", values, split="train")
    _write_shard(dataset / "fineweb_val_000000.bin", values, split="validation")

    single_output = tmp_path / "single"
    single = subprocess.run(
        _base_command(glimmer_native_train_tools)
        + [
            "--checkpoint",
            str(source),
            "--checkpoint-sha256",
            source_sha,
            "--dataset",
            str(dataset),
            "--output-dir",
            str(single_output),
            "--max-steps",
            "1",
            "--checkpoint-every-steps",
            "1",
            "--max-grad-norm",
            "1000000000",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert single.returncode == 0, single.stderr

    distributed_output = tmp_path / "distributed"
    bootstrap = tmp_path / "nccl-id.bin"
    processes: list[subprocess.Popen[str]] = []
    for rank in range(2):
        command = _base_command(glimmer_native_train_tools) + [
            "--checkpoint",
            str(source),
            "--checkpoint-sha256",
            source_sha,
            "--dataset",
            str(dataset),
            "--output-dir",
            str(distributed_output),
            "--max-steps",
            "1",
            "--checkpoint-every-steps",
            "1",
            "--max-grad-norm",
            "1000000000",
            "--pipeline-parallel-size",
            "2",
            "--pipeline-parallel-rank",
            str(rank),
            "--cuda-device",
            "0",
            "--nccl-lib",
            str(fake_nccl_library),
            "--distributed-id-file",
            str(bootstrap),
            "--distributed-reserve-bytes",
            "0",
        ]
        processes.append(
            subprocess.Popen(
                command,
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        )
    results = [process.communicate(timeout=60) for process in processes]
    for process, (stdout, stderr) in zip(processes, results, strict=True):
        assert process.returncode == 0, f"stdout={stdout}\nstderr={stderr}"

    distributed_checkpoint = distributed_output / "checkpoint-step-1"
    assert (distributed_checkpoint / "DONE").is_file()
    assert (distributed_checkpoint / "distributed_state.v1").is_file()
    manifest = json.loads(
        (distributed_checkpoint / "distributed_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["schema"] == "neuralfn.muse_glimmer_distributed_checkpoint.v1"
    assert manifest["world_size"] == 2
    assert [(stage["layer_begin"], stage["layer_end"]) for stage in manifest["stages"]] == [
        (0, 2),
        (2, 4),
    ]
    reconstructed_model = b"".join(
        (distributed_checkpoint / f"model-rank-{rank:05d}.bf16").read_bytes()
        for rank in range(2)
    )
    reconstructed_optimizer = b"".join(
        (distributed_checkpoint / f"optimizer-rank-{rank:05d}.f32").read_bytes()
        for rank in range(2)
    )
    assert reconstructed_model == (
        single_output / "checkpoint-step-1" / "model.bf16"
    ).read_bytes()
    # Optimizer shards are stored rank-major while the single-rank format is
    # moment-major, so compare each rank slice to the corresponding moment
    # slices rather than concatenating layouts as if they were identical.
    full_optimizer = (
        single_output / "checkpoint-step-1" / "optimizer.f32"
    ).read_bytes()
    rank_model_bytes = [
        (distributed_checkpoint / f"model-rank-{rank:05d}.bf16").stat().st_size
        for rank in range(2)
    ]
    rank_elements = [value // 2 for value in rank_model_bytes]
    full_elements = sum(rank_elements)
    expected_rank_major = b"".join(
        full_optimizer[
            moment * full_elements * 4
            + sum(rank_elements[:rank]) * 4 :
            moment * full_elements * 4
            + sum(rank_elements[: rank + 1]) * 4
        ]
        for rank in range(2)
        for moment in range(2)
    )
    assert reconstructed_optimizer == expected_rank_major

    resume_bootstrap = tmp_path / "nccl-resume-id.bin"
    resumed_processes: list[subprocess.Popen[str]] = []
    for rank in range(2):
        resumed_processes.append(
            subprocess.Popen(
                _base_command(glimmer_native_train_tools)
                + [
                    "--resume-from-checkpoint",
                    str(distributed_checkpoint),
                    "--checkpoint-sha256",
                    source_sha,
                    "--dataset",
                    str(dataset),
                    "--output-dir",
                    str(distributed_output),
                    "--max-steps",
                    "2",
                    "--checkpoint-every-steps",
                    "1",
                    "--max-grad-norm",
                    "1000000000",
                    "--pipeline-parallel-size",
                    "2",
                    "--pipeline-parallel-rank",
                    str(rank),
                    "--cuda-device",
                    "0",
                    "--nccl-lib",
                    str(fake_nccl_library),
                    "--distributed-id-file",
                    str(resume_bootstrap),
                    "--distributed-reserve-bytes",
                    "0",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        )
    resumed_results = [
        process.communicate(timeout=60) for process in resumed_processes
    ]
    for process, (stdout, stderr) in zip(
        resumed_processes, resumed_results, strict=True
    ):
        assert process.returncode == 0, f"stdout={stdout}\nstderr={stderr}"
    resumed_checkpoint = distributed_output / "checkpoint-step-2"
    assert (resumed_checkpoint / "DONE").is_file()
    assert b"".join(
        (resumed_checkpoint / f"model-rank-{rank:05d}.bf16").read_bytes()
        for rank in range(2)
    ) != reconstructed_model


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


def test_native_glimmer_dpo_uses_frozen_reference_and_resumes_strictly(
    glimmer_native_train_tools: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    source = tmp_path / "source.bf16"
    source_sha = _write_checkpoint(source)
    reference = tmp_path / "reference.bf16"
    reference_sha = _write_checkpoint(reference, seed=20260814)
    dataset = tmp_path / "preference"
    template_sha = "e" * 64
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
            "chosen": {
                "input_ids": [1, 2, 3, 4],
                "targets": [-100, 3, 4, 5],
                "loss_mask": [0.0, 1.0, 1.0, 1.0],
                "sequence_ids": [0, 0, 0, 0],
            },
            "rejected": {
                "input_ids": [1, 2, 6, 7],
                "targets": [-100, 6, 7, 8],
                "loss_mask": [0.0, 1.0, 1.0, 1.0],
                "sequence_ids": [0, 0, 0, 0],
            },
        }
    ]
    write_structured_preference_v1(
        dataset / "preference_train_000000.preference",
        records,
        split="train",
        **common,
    )
    write_structured_preference_v1(
        dataset / "preference_val_000000.preference",
        records,
        split="validation",
        **common,
    )
    output = tmp_path / "out"
    command = _base_command(glimmer_native_train_tools) + [
        "--checkpoint",
        str(source),
        "--checkpoint-sha256",
        source_sha,
        "--reference-checkpoint",
        str(reference),
        "--reference-checkpoint-sha256",
        reference_sha,
        "--dataset",
        str(dataset),
        "--output-dir",
        str(output),
        "--objective",
        "dpo",
        "--chat-template-sha256",
        template_sha,
        "--dpo-beta",
        "0.2",
        "--dpo-label-smoothing",
        "0.1",
        "--dpo-loss-type",
        "sigmoid",
        "--max-steps",
        "1",
        "--checkpoint-every-steps",
        "1",
    ]
    first = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=False
    )
    assert first.returncode == 0, first.stderr
    events = [json.loads(line) for line in first.stdout.splitlines()]
    assert events[0]["objective"] == "dpo"
    assert events[0]["reference_sha256"] == reference_sha
    checkpoint1 = output / "checkpoint-step-1"
    state1 = json.loads((checkpoint1 / "trainer_state.json").read_text())
    assert state1["schema"] == "neuralfn.muse_glimmer_native_dpo_training.v1"
    assert state1["reference_frozen"] is True
    assert state1["source_sha256"] == source_sha
    assert state1["reference_sha256"] == reference_sha
    assert state1["loss_type"] == "sigmoid"
    assert (checkpoint1 / "trainer_state.dpo.v1").is_file()
    assert (checkpoint1 / "model.bf16").read_bytes() != source.read_bytes()
    assert reference_sha == hashlib.sha256(reference.read_bytes()).hexdigest()

    resumed_command = command.copy()
    resumed_command[resumed_command.index("--max-steps") + 1] = "2"
    resumed_command.extend(["--resume-from-checkpoint", str(checkpoint1)])
    resumed = subprocess.run(
        resumed_command, cwd=ROOT, text=True, capture_output=True, check=False
    )
    assert resumed.returncode == 0, resumed.stderr
    assert (output / "checkpoint-step-2/model.bf16").is_file()

    wrong_reference = resumed_command.copy()
    wrong_reference[
        wrong_reference.index("--reference-checkpoint-sha256") + 1
    ] = source_sha
    rejected = subprocess.run(
        wrong_reference, cwd=ROOT, text=True, capture_output=True, check=False
    )
    assert rejected.returncode == 2
    assert "configuration mismatch" in rejected.stderr


def test_native_glimmer_reward_model_trains_masked_head_and_freezes_lm_head(
    glimmer_native_train_tools: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    source = tmp_path / "source.bf16"
    source_sha = _write_checkpoint(source)
    dataset = tmp_path / "preference"
    template_sha = "f" * 64
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
            "chosen": {
                "input_ids": [1, 2, 3, 4],
                "targets": [-100, 3, 4, 5],
                "loss_mask": [0.0, 1.0, 1.0, 1.0],
                "sequence_ids": [0, 0, 0, 0],
            },
            "rejected": {
                "input_ids": [1, 2, 6, 7],
                "targets": [-100, 6, 7, 8],
                "loss_mask": [0.0, 1.0, 1.0, 1.0],
                "sequence_ids": [0, 0, 0, 0],
            },
        }
    ]
    write_structured_preference_v1(
        dataset / "preference_train_000000.preference",
        records,
        split="train",
        **common,
    )
    write_structured_preference_v1(
        dataset / "preference_val_000000.preference",
        records,
        split="validation",
        **common,
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
        "reward_model",
        "--chat-template-sha256",
        template_sha,
        "--reward-head-seed",
        "123",
        "--max-steps",
        "1",
        "--checkpoint-every-steps",
        "1",
    ]
    first = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=False
    )
    assert first.returncode == 0, first.stderr
    checkpoint1 = output / "checkpoint-step-1"
    state1 = json.loads((checkpoint1 / "trainer_state.json").read_text())
    manifest = json.loads(
        (checkpoint1 / "reward_model_manifest.json").read_text()
    )
    assert state1["schema"] == "neuralfn.muse_glimmer_native_reward_training.v1"
    assert state1["lm_head_frozen"] is True
    assert state1["pool"] == "last_selected_token"
    assert state1["reward_head_seed"] == 123
    assert manifest["format"] == "neuralfn.native_muse_glimmer_reward.bf16.v1"
    assert manifest["pretrained_source_sha256"] == source_sha
    assert (checkpoint1 / "reward_head.bf16").stat().st_size == 8 * 2
    assert (checkpoint1 / "trainer_state.reward.v1").is_file()
    # The untied LM head is the final tensor in the model artifact and is not
    # part of reward optimization.
    lm_head_bytes = 13 * 8 * 2
    assert (checkpoint1 / "model.bf16").read_bytes()[-lm_head_bytes:] == (
        source.read_bytes()[-lm_head_bytes:]
    )

    resumed_command = command.copy()
    resumed_command[resumed_command.index("--max-steps") + 1] = "2"
    resumed_command.extend(["--resume-from-checkpoint", str(checkpoint1)])
    resumed = subprocess.run(
        resumed_command, cwd=ROOT, text=True, capture_output=True, check=False
    )
    assert resumed.returncode == 0, resumed.stderr
    checkpoint2 = output / "checkpoint-step-2"
    assert (checkpoint2 / "reward_head.bf16").is_file()

    corrupt = bytearray((checkpoint2 / "reward_head_optimizer.f32").read_bytes())
    corrupt[-1] ^= 1
    (checkpoint2 / "reward_head_optimizer.f32").write_bytes(corrupt)
    rejected_command = command.copy()
    rejected_command[rejected_command.index("--max-steps") + 1] = "3"
    rejected_command.extend(["--resume-from-checkpoint", str(checkpoint2)])
    rejected = subprocess.run(
        rejected_command, cwd=ROOT, text=True, capture_output=True, check=False
    )
    assert rejected.returncode == 2
    assert "authentication failed" in rejected.stderr


def test_native_glimmer_ppo_runs_online_rollout_reference_reward_gae_and_resume(
    glimmer_native_train_tools: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    source = tmp_path / "source.bf16"
    source_sha = _write_checkpoint(source)
    template_sha = "9" * 64

    # Produce the independently authenticated frozen reward model consumed by
    # online PPO rather than substituting synthetic/zero rewards.
    preference = tmp_path / "preference"
    preference_common = {
        "sequence_length": 4,
        "tokenizer_vocab_size": 13,
        "pad_token_id": 0,
        "tokenizer_sha256": "a" * 64,
        "chat_template_sha256": template_sha,
        "tokenizer_revision": "tiny-glimmer-v1",
    }
    preference_records = [
        {
            "chosen": {
                "input_ids": [1, 2, 3, 4],
                "targets": [-100, 3, 4, 5],
                "loss_mask": [0.0, 1.0, 1.0, 1.0],
                "sequence_ids": [0, 0, 0, 0],
            },
            "rejected": {
                "input_ids": [1, 2, 6, 7],
                "targets": [-100, 6, 7, 8],
                "loss_mask": [0.0, 1.0, 1.0, 1.0],
                "sequence_ids": [0, 0, 0, 0],
            },
        }
    ]
    write_structured_preference_v1(
        preference / "preference_train_000000.preference",
        preference_records,
        split="train",
        **preference_common,
    )
    write_structured_preference_v1(
        preference / "preference_val_000000.preference",
        preference_records,
        split="validation",
        **preference_common,
    )
    reward_output = tmp_path / "reward-output"
    reward_run = subprocess.run(
        _base_command(glimmer_native_train_tools)
        + [
            "--checkpoint",
            str(source),
            "--checkpoint-sha256",
            source_sha,
            "--dataset",
            str(preference),
            "--output-dir",
            str(reward_output),
            "--objective",
            "reward_model",
            "--chat-template-sha256",
            template_sha,
            "--reward-head-seed",
            "77",
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
    assert reward_run.returncode == 0, reward_run.stderr
    reward_checkpoint = reward_output / "checkpoint-step-1"
    reward_manifest = reward_checkpoint / "reward_model_manifest.json"
    reward_manifest_sha = hashlib.sha256(reward_manifest.read_bytes()).hexdigest()

    prompts = tmp_path / "prompts"
    prompt_common = {
        "records": [
            {
                "input_ids": [1, 2, 0, 0],
                "attention_mask": [1.0, 1.0, 0.0, 0.0],
            }
        ],
        "sequence_length": 4,
        "tokenizer_vocab_size": 13,
        "pad_token_id": 0,
        "tokenizer_sha256": "a" * 64,
        "chat_template_sha256": template_sha,
        "tokenizer_revision": "tiny-glimmer-v1",
    }
    write_structured_ppo_prompt_v1(
        prompts / "ppo_prompt_train_000000.ppo_prompt",
        split="train",
        **prompt_common,
    )
    write_structured_ppo_prompt_v1(
        prompts / "ppo_prompt_val_000000.ppo_prompt",
        split="validation",
        **prompt_common,
    )
    output = tmp_path / "ppo-output"
    command = _base_command(glimmer_native_train_tools) + [
        "--checkpoint",
        str(source),
        "--checkpoint-sha256",
        source_sha,
        "--reference-checkpoint",
        str(source),
        "--reference-checkpoint-sha256",
        source_sha,
        "--reward-checkpoint",
        str(reward_checkpoint),
        "--reward-checkpoint-sha256",
        reward_manifest_sha,
        "--dataset",
        str(prompts),
        "--output-dir",
        str(output),
        "--objective",
        "ppo",
        "--chat-template-sha256",
        template_sha,
        "--rollout-length",
        "2",
        "--ppo-epochs-per-rollout",
        "2",
        "--ppo-minibatch-size",
        "1",
        "--rollout-top-k",
        "1",
        "--rollout-seed",
        "123",
        "--ppo-value-head-seed",
        "456",
        "--eos-token-ids",
        "11,12",
        "--max-steps",
        "1",
        "--checkpoint-every-steps",
        "1",
    ]
    first = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=False
    )
    assert first.returncode == 0, first.stderr
    events = [json.loads(line) for line in first.stdout.splitlines()]
    epoch_events = [event for event in events if event["event"] == "ppo_epoch"]
    rollout_events = [event for event in events if event["event"] == "rollout"]
    assert len(epoch_events) == 2
    assert all(event["loss"] != 0.0 for event in epoch_events)
    assert rollout_events[0]["actions"] > 0
    assert rollout_events[0]["reference_sha256"] == source_sha
    assert rollout_events[0]["reward_manifest_sha256"] == reward_manifest_sha
    checkpoint1 = output / "checkpoint-rollout-1"
    state1 = json.loads((checkpoint1 / "trainer_state.json").read_text())
    assert state1["schema"] == "neuralfn.muse_glimmer_native_ppo_training.v1"
    assert state1["online_rollout"] is True
    assert state1["reference_frozen"] is True
    assert state1["reward_frozen"] is True
    assert state1["optimizer_step"] == 2
    assert (checkpoint1 / "trainer_state.ppo.v1").is_file()
    assert (checkpoint1 / "value_head.bf16").stat().st_size == 8 * 2
    assert (checkpoint1 / "model.bf16").read_bytes() != source.read_bytes()

    resumed_command = command.copy()
    resumed_command[resumed_command.index("--max-steps") + 1] = "2"
    resumed_command.extend(["--resume-from-checkpoint", str(checkpoint1)])
    resumed = subprocess.run(
        resumed_command, cwd=ROOT, text=True, capture_output=True, check=False
    )
    assert resumed.returncode == 0, resumed.stderr
    checkpoint2 = output / "checkpoint-rollout-2"
    state2 = json.loads((checkpoint2 / "trainer_state.json").read_text())
    assert state2["completed_rollout"] == 2
    assert state2["optimizer_step"] == 4

    corrupted = bytearray((checkpoint2 / "value_head_optimizer.f32").read_bytes())
    corrupted[-1] ^= 1
    (checkpoint2 / "value_head_optimizer.f32").write_bytes(corrupted)
    rejected_command = command.copy()
    rejected_command[rejected_command.index("--max-steps") + 1] = "3"
    rejected_command.extend(["--resume-from-checkpoint", str(checkpoint2)])
    rejected = subprocess.run(
        rejected_command, cwd=ROOT, text=True, capture_output=True, check=False
    )
    assert rejected.returncode == 2
    assert "authentication failed" in rejected.stderr


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


def test_native_glimmer_kquant_lora_streams_typed_base_and_pins_profile(
    glimmer_native_train_tools: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    source = tmp_path / "tiny-kquant.gguf"
    source_sha = _write_tiny_kquant_gguf(source)
    source_bytes = source.read_bytes()
    dataset = tmp_path / "sft-kquant"
    template_sha = "e" * 64
    common = {
        "sequence_length": 4,
        "tokenizer_vocab_size": 17,
        "pad_token_id": 0,
        "tokenizer_sha256": "a" * 64,
        "chat_template_sha256": template_sha,
        "tokenizer_revision": "tiny-glimmer-kquant-v1",
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

    trainer, runtime, tile = glimmer_native_train_tools

    def command(output: Path) -> list[str]:
        return [
            str(trainer),
            "--cuda-runtime-lib",
            str(runtime),
            "--tile-ops-lib",
            str(tile),
            "--tiny-geometry",
            KQUANT_GEOMETRY,
            "--sequence-length",
            "4",
            "--batch-size",
            "1",
            "--activation-checkpoint-interval",
            "2",
            "--checkpoint",
            str(source),
            "--checkpoint-sha256",
            source_sha,
            "--kquant-profile",
            "test",
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
            "--max-steps",
            "1",
            "--checkpoint-every-steps",
            "1",
        ]

    output = tmp_path / "out-kquant"
    completed = subprocess.run(
        command(output), cwd=ROOT, text=True, capture_output=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    checkpoint1 = output / "checkpoint-step-1"
    state = json.loads((checkpoint1 / "trainer_state.json").read_text())
    manifest = json.loads((checkpoint1 / "adapter_manifest.json").read_text())
    assert state["base_weight_precision"] == "test"
    assert state["source_sha256"] == source_sha
    assert state["base_frozen"] is True
    assert manifest["base_weight_precision"] == "test"
    assert manifest["training_base_precision"] == "test"
    assert manifest["training_adapter"] == "lora"
    assert manifest["base_sha256"] == source_sha
    assert not (checkpoint1 / "model.bf16").exists()
    assert source.read_bytes() == source_bytes
    assert any((checkpoint1 / "adapter.bf16").read_bytes())

    resumed_command = command(output)
    resumed_command[resumed_command.index("--max-steps") + 1] = "2"
    resumed_command.extend(["--resume-from-checkpoint", str(checkpoint1)])
    resumed = subprocess.run(
        resumed_command, cwd=ROOT, text=True, capture_output=True, check=False
    )
    assert resumed.returncode == 0, resumed.stderr
    assert (output / "checkpoint-step-2/adapter.bf16").is_file()
    assert source.read_bytes() == source_bytes

    wrong_public_profile = command(tmp_path / "wrong-profile")
    wrong_public_profile[wrong_public_profile.index("test")] = "k-quant-17gb"
    rejected = subprocess.run(
        wrong_public_profile, cwd=ROOT, text=True, capture_output=True, check=False
    )
    assert rejected.returncode == 2
    assert "production Muse Glimmer geometry" in rejected.stderr

    unsupported = tmp_path / "unsupported.gguf"
    unsupported_sha = _write_tiny_kquant_gguf(
        unsupported, unsupported_encoding=True
    )
    unsupported_command = command(tmp_path / "unsupported-out")
    unsupported_command[unsupported_command.index(str(source))] = str(unsupported)
    unsupported_command[unsupported_command.index(source_sha)] = unsupported_sha
    rejected = subprocess.run(
        unsupported_command, cwd=ROOT, text=True, capture_output=True, check=False
    )
    assert rejected.returncode == 2
    assert "encoding is unsupported" in rejected.stderr
