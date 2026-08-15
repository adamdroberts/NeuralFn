from __future__ import annotations

import math
import os
from pathlib import Path
import struct
import subprocess

import pytest
import torch
import torch.nn.functional as F

from neuralfn.torch_backend import (
    MuseGlimmerPerceptionNormStage,
    MuseGlimmerVisionTowerStage,
)


ROOT = Path(__file__).resolve().parents[1]
HIDDEN = 8
INTERMEDIATE = 12
LAYERS = 2
HEADS = 2
PATCH_WIDTH = 6
MERGE = 2
POSITION_SIDE = 2
ADAPTER = 5
OUTPUT = 8


def _bf16_bits(value: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    if bits & 0x7F800000 == 0x7F800000:
        return (bits >> 16) & 0xFFFF
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


def _bf16_value(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def _shape_size(shape: tuple[int, ...]) -> int:
    result = 1
    for value in shape:
        result *= value
    return result


def _layout() -> list[tuple[str, tuple[int, ...], str]]:
    rows: list[tuple[str, tuple[int, ...], str]] = [
        ("patch", (HIDDEN, PATCH_WIDTH), "weight"),
        ("position", (POSITION_SIDE * POSITION_SIDE, HIDDEN), "weight"),
        ("pre_norm.weight", (HIDDEN,), "norm_weight"),
        ("pre_norm.bias", (HIDDEN,), "bias"),
        ("post_norm.weight", (HIDDEN,), "norm_weight"),
        ("post_norm.bias", (HIDDEN,), "bias"),
    ]
    for layer in range(LAYERS):
        prefix = f"layers.{layer}."
        for projection in ("q", "k", "v", "proj"):
            rows.extend(
                [
                    (prefix + projection + ".weight", (HIDDEN, HIDDEN), "weight"),
                    (prefix + projection + ".bias", (HIDDEN,), "bias"),
                ]
            )
        rows.extend(
            [
                (prefix + "norm1.weight", (HIDDEN,), "norm_weight"),
                (prefix + "norm1.bias", (HIDDEN,), "bias"),
                (prefix + "norm2.weight", (HIDDEN,), "norm_weight"),
                (prefix + "norm2.bias", (HIDDEN,), "bias"),
                (prefix + "fc1.weight", (INTERMEDIATE, HIDDEN), "weight"),
                (prefix + "fc1.bias", (INTERMEDIATE,), "bias"),
                (prefix + "fc2.weight", (HIDDEN, INTERMEDIATE), "weight"),
                (prefix + "fc2.bias", (HIDDEN,), "bias"),
            ]
        )
    rows.extend(
        [
            ("adapter.fc1", (ADAPTER, MERGE * MERGE * HIDDEN), "weight"),
            ("adapter.fc2", (ADAPTER, ADAPTER), "weight"),
            ("projection", (OUTPUT, ADAPTER), "weight"),
        ]
    )
    return rows


def _write_payload(path: Path) -> dict[str, torch.Tensor]:
    payload = bytearray()
    tensors: dict[str, torch.Tensor] = {}
    global_index = 0
    for name, shape, kind in _layout():
        values: list[float] = []
        bits: list[int] = []
        for local_index in range(_shape_size(shape)):
            if kind == "norm_weight":
                value = 0.93 + 0.09 * math.cos(global_index * 0.037)
            elif kind == "bias":
                value = 0.021 * math.sin(global_index * 0.071)
            else:
                value = (
                    0.083 * math.sin(global_index * 0.031 + local_index * 0.17)
                    + 0.027 * math.cos(global_index * 0.019 - local_index * 0.11)
                )
            encoded = _bf16_bits(value)
            bits.append(encoded)
            values.append(_bf16_value(encoded))
            global_index += 1
        payload.extend(struct.pack(f"<{len(bits)}H", *bits))
        tensors[name] = torch.tensor(values, dtype=torch.float32).reshape(shape)
    path.write_bytes(payload)
    return tensors


@pytest.fixture(scope="session")
def vision_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("native-glimmer-vision-probe") / "probe"
    subprocess.run(
        [
            "c++",
            "-std=c++20",
            "-O2",
            "-pthread",
            "-I",
            str(ROOT / "neuralfn" / "csrc" / "native_gpt2"),
            str(ROOT / "tests" / "cpp" / "resident_glimmer_vision_probe.cpp"),
            str(
                ROOT
                / "neuralfn"
                / "csrc"
                / "native_gpt2"
                / "resident_glimmer_vision.cpp"
            ),
            str(
                ROOT
                / "neuralfn"
                / "csrc"
                / "native_gpt2"
                / "resident_glimmer_cuda.cpp"
            ),
            "-ldl",
            "-o",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return output


@pytest.fixture(scope="session")
def fake_vision_cuda(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path]:
    output = tmp_path_factory.mktemp("fake-glimmer-vision-cuda")
    runtime = output / "libcudart-fake.so"
    tile = output / "libnfn-glimmer-tile-fake.so"
    subprocess.run(
        [
            "c++", "-std=c++20", "-O2", "-fPIC", "-shared",
            str(ROOT / "tests" / "cpp" / "fake_cuda_runtime.cpp"),
            "-o", str(runtime),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "c++", "-std=c++20", "-O2", "-fPIC", "-shared",
            "-I", str(ROOT / "neuralfn" / "csrc" / "native_train"),
            str(ROOT / "tests" / "cpp" / "fake_glimmer_tile_ops.cpp"),
            "-o", str(tile),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return runtime, tile


def _oracle(weights: dict[str, torch.Tensor]) -> torch.Tensor:
    tower = MuseGlimmerVisionTowerStage(
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_heads=HEADS,
        num_layers=LAYERS,
        patch_size=1,
        patch_temporal=2,
        merge_size=MERGE,
        pos_emb_height=POSITION_SIDE,
        pos_emb_width=POSITION_SIDE,
        rope_theta=10_000.0,
        eps=1.0e-5,
    )
    with torch.no_grad():
        tower.patch_embedding.weight.copy_(weights["patch"])
        tower.position_embedding.weight.copy_(weights["position"])
        tower.pre_norm.weight.copy_(weights["pre_norm.weight"])
        tower.pre_norm.bias.copy_(weights["pre_norm.bias"])
        tower.post_norm.weight.copy_(weights["post_norm.weight"])
        tower.post_norm.bias.copy_(weights["post_norm.bias"])
        for index, layer in enumerate(tower.layers):
            prefix = f"layers.{index}."
            for source, target in (
                ("q", layer.attn.q_proj),
                ("k", layer.attn.k_proj),
                ("v", layer.attn.v_proj),
                ("proj", layer.attn.proj),
            ):
                target.weight.copy_(weights[prefix + source + ".weight"])
                target.bias.copy_(weights[prefix + source + ".bias"])
            layer.norm1.weight.copy_(weights[prefix + "norm1.weight"])
            layer.norm1.bias.copy_(weights[prefix + "norm1.bias"])
            layer.norm2.weight.copy_(weights[prefix + "norm2.weight"])
            layer.norm2.bias.copy_(weights[prefix + "norm2.bias"])
            layer.fc1.weight.copy_(weights[prefix + "fc1.weight"])
            layer.fc1.bias.copy_(weights[prefix + "fc1.bias"])
            layer.fc2.weight.copy_(weights[prefix + "fc2.weight"])
            layer.fc2.bias.copy_(weights[prefix + "fc2.bias"])
    patches = torch.tensor(
        [
            0.19 * math.sin(index * 0.17) + 0.07 * math.cos(index * 0.11)
            for index in range(8 * PATCH_WIDTH)
        ],
        dtype=torch.float32,
    ).reshape(8, PATCH_WIDTH)
    grid = torch.tensor([[1, 2, 4]], dtype=torch.int64)
    with torch.no_grad():
        hidden = tower(patches, grid)
        hidden = F.gelu(F.linear(hidden, weights["adapter.fc1"]))
        hidden = F.gelu(F.linear(hidden, weights["adapter.fc2"]))
        hidden = F.linear(hidden, weights["projection"])
        return MuseGlimmerPerceptionNormStage(1.0e-5)(hidden)


def test_native_vision_matches_independent_torch_window_and_merge_oracle(
    vision_probe: Path,
    tmp_path: Path,
) -> None:
    payload = tmp_path / "tiny-vision.bf16"
    weights = _write_payload(payload)
    completed = subprocess.run(
        [str(vision_probe), str(payload)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = completed.stdout.splitlines()
    actual = torch.tensor([float(value) for value in lines[0].split(",")]).reshape(2, OUTPUT)
    assert actual == pytest.approx(_oracle(weights), abs=2.5e-5)
    assert lines[1] == "cancelled"


def test_native_cuda_vision_matches_cpu_and_torch_without_cpu_model_compute(
    vision_probe: Path,
    fake_vision_cuda: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    payload = tmp_path / "tiny-vision-cuda.bf16"
    weights = _write_payload(payload)
    runtime, tile = fake_vision_cuda
    environment = os.environ.copy()
    environment["NFN_TEST_GLIMMER_CUDA_RUNTIME"] = str(runtime)
    environment["NFN_TEST_GLIMMER_TILE_OPS"] = str(tile)
    completed = subprocess.run(
        [str(vision_probe), str(payload)],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = completed.stdout.splitlines()
    cpu = torch.tensor([float(value) for value in lines[0].split(",")]).reshape(2, OUTPUT)
    cuda = torch.tensor([float(value) for value in lines[2].split(",")]).reshape(2, OUTPUT)
    expected = _oracle(weights)
    assert cpu == pytest.approx(expected, abs=2.5e-5)
    assert cuda == pytest.approx(cpu, abs=2.5e-5)
    resident, workspace, launches = map(int, lines[3].split(","))
    # Norms, biases, and the learned position table are expanded once to F32
    # at load time; request-time model compute still remains on the CUDA path.
    assert resident > payload.stat().st_size
    assert workspace > 0
    assert launches > 0
    assert lines[1] == "cancelled"
    assert lines[4] == "cuda-cancelled"


def test_native_vision_rejects_truncated_payload(
    vision_probe: Path,
    tmp_path: Path,
) -> None:
    payload = tmp_path / "truncated-vision.bf16"
    _write_payload(payload)
    payload.write_bytes(payload.read_bytes()[:-2])
    completed = subprocess.run(
        [str(vision_probe), str(payload)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    assert "payload" in completed.stderr
