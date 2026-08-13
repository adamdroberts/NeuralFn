from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import struct
import subprocess

import pytest

import neuralfn.native_gguf as gguf
from neuralfn.native_ir import NativeExecutionManifest


ROOT = Path(__file__).resolve().parents[1]


def _string(value: bytes) -> bytes:
    return struct.pack("<Q", len(value)) + value


def _build_sparse_muse_fixture(
    path: Path,
    *,
    mutate_tensor=None,
) -> None:
    metadata = []
    for index in range(32):
        metadata.append(_string(f"fixture.{index}".encode()) + struct.pack("<II", 4, index))

    table = bytearray()
    relative_offset = 0
    for index, name in enumerate(gguf.expected_muse_glimmer_gguf_tensor_names()):
        shape, _native_name = gguf._tensor_contract(name)
        dimensions = tuple(reversed(shape))
        tensor_type = gguf.GGML_TYPE_F32
        tensor_offset = relative_offset
        if mutate_tensor is not None:
            dimensions, tensor_type, tensor_offset = mutate_tensor(
                index,
                name,
                dimensions,
                tensor_type,
                tensor_offset,
            )
        table.extend(_string(name.encode()))
        table.extend(struct.pack("<I", len(dimensions)))
        table.extend(struct.pack("<" + "Q" * len(dimensions), *dimensions))
        table.extend(struct.pack("<IQ", tensor_type, tensor_offset))
        row_bytes = dimensions[0] * 4
        tensor_bytes = row_bytes
        for dimension in dimensions[1:]:
            tensor_bytes *= dimension
        relative_offset = gguf._align(tensor_offset + tensor_bytes, 32)

    header = (
        b"GGUF"
        + struct.pack("<IQQ", 3, 731, 32)
        + b"".join(metadata)
        + bytes(table)
    )
    data_offset = gguf._align(len(header), 32)
    with path.open("wb") as stream:
        stream.write(header)
        stream.write(bytes(data_offset - len(header)))
        stream.truncate(data_offset + tensor_offset + tensor_bytes)


def _build_sparse_dflash_fixture(path: Path, *, mutate_tensor=None) -> None:
    metadata = []
    for index in range(33):
        metadata.append(_string(f"fixture.{index}".encode()) + struct.pack("<II", 4, index))
    table = bytearray()
    relative_offset = 0
    f32_names = {"enc.output_norm.weight", "output_norm.weight"}
    for layer in range(5):
        f32_names.update(
            {
                f"blk.{layer}.attn_norm.weight",
                f"blk.{layer}.ffn_norm.weight",
                f"blk.{layer}.attn_k_norm.weight",
                f"blk.{layer}.attn_q_norm.weight",
            }
        )
    q6_names = {
        *(f"blk.{layer}.ffn_down.weight" for layer in range(5)),
        *(f"blk.{layer}.attn_v.weight" for layer in range(5)),
    }
    for index, name in enumerate(gguf.expected_muse_glimmer_dflash_gguf_tensor_names()):
        shape, _native_name = gguf._dflash_tensor_contract(name)
        dimensions = tuple(reversed(shape))
        tensor_type = (
            gguf.GGML_TYPE_F32
            if name in f32_names
            else gguf.GGML_TYPE_Q6_K
            if name in q6_names
            else gguf.GGML_TYPE_Q4_K
        )
        tensor_offset = relative_offset
        if mutate_tensor is not None:
            dimensions, tensor_type, tensor_offset = mutate_tensor(
                index, name, dimensions, tensor_type, tensor_offset
            )
        table.extend(_string(name.encode()))
        table.extend(struct.pack("<I", len(dimensions)))
        table.extend(struct.pack("<" + "Q" * len(dimensions), *dimensions))
        table.extend(struct.pack("<IQ", tensor_type, tensor_offset))
        _encoding, block_elements, block_bytes = gguf.GGML_TYPE_LAYOUTS[tensor_type]
        row_bytes = dimensions[0] // block_elements * block_bytes
        tensor_bytes = row_bytes
        for dimension in dimensions[1:]:
            tensor_bytes *= dimension
        relative_offset = gguf._align(tensor_offset + tensor_bytes, 32)
    header = b"GGUF" + struct.pack("<IQQ", 3, 58, 33) + b"".join(metadata) + bytes(table)
    data_offset = gguf._align(len(header), 32)
    with path.open("wb") as stream:
        stream.write(header)
        stream.write(bytes(data_offset - len(header)))
        stream.truncate(data_offset + tensor_offset + tensor_bytes)


def _build_sparse_mmproj_fixture(path: Path, *, mutate_tensor=None) -> None:
    def entry(key: str, value_type: int, payload: bytes) -> bytes:
        return _string(key.encode()) + struct.pack("<I", value_type) + payload

    metadata = [
        entry("general.architecture", 8, _string(b"clip")),
        entry("general.type", 8, _string(b"mmproj")),
        entry("general.name", 8, _string(b"Muse Glimmer Hf")),
        entry("general.size_label", 8, _string(b"1.9B")),
        entry("clip.has_vision_encoder", 7, b"\x01"),
        entry("clip.vision.projection_dim", 4, struct.pack("<I", 6_656)),
        entry("clip.vision.image_size", 4, struct.pack("<I", 896)),
        entry("clip.vision.patch_size", 4, struct.pack("<I", 14)),
        entry("clip.vision.embedding_length", 4, struct.pack("<I", 1_536)),
        entry("clip.vision.feed_forward_length", 4, struct.pack("<I", 8_960)),
        entry("clip.vision.block_count", 4, struct.pack("<I", 50)),
        entry("clip.vision.attention.head_count", 4, struct.pack("<I", 16)),
        entry(
            "clip.vision.image_mean",
            9,
            struct.pack("<IQfff", 6, 3, 0.5, 0.5, 0.5),
        ),
        entry(
            "clip.vision.image_std",
            9,
            struct.pack("<IQfff", 6, 3, 0.5, 0.5, 0.5),
        ),
        entry("clip.projector_type", 8, _string(b"muse-glimmer")),
        entry(
            "clip.vision.attention.layer_norm_epsilon",
            6,
            struct.pack("<f", 1.0e-5),
        ),
        entry("clip.vision.spatial_merge_size", 4, struct.pack("<I", 2)),
        entry("general.quantization_version", 4, struct.pack("<I", 2)),
        entry("general.file_type", 4, struct.pack("<I", 15)),
    ]
    table = bytearray()
    relative_offset = 0
    names: list[str] = []
    for layer in sorted(range(50), key=str):
        for suffix in (
            "attn_k",
            "attn_out",
            "attn_q",
            "attn_v",
            "ffn_up",
            "ffn_down",
            "ln1",
            "ln2",
        ):
            names.extend(
                (f"v.blk.{layer}.{suffix}.bias", f"v.blk.{layer}.{suffix}.weight")
            )
    names.extend(
        (
            "v.post_ln.bias",
            "v.post_ln.weight",
            "v.pre_ln.bias",
            "v.pre_ln.weight",
            "v.patch_embd.weight",
            "v.position_embd.weight",
            "mm.0.weight",
            "mm.1.weight",
            "mm.2.weight",
        )
    )
    for index, name in enumerate(names):
        shape, _native_name, tensor_type = gguf._mmproj_tensor_contract(name)
        dimensions = tuple(reversed(shape))
        tensor_offset = relative_offset
        if mutate_tensor is not None:
            dimensions, tensor_type, tensor_offset = mutate_tensor(
                index, name, dimensions, tensor_type, tensor_offset
            )
        table.extend(_string(name.encode()))
        table.extend(struct.pack("<I", len(dimensions)))
        table.extend(struct.pack("<" + "Q" * len(dimensions), *dimensions))
        table.extend(struct.pack("<IQ", tensor_type, tensor_offset))
        _encoding, block_elements, block_bytes = gguf.GGML_TYPE_LAYOUTS[tensor_type]
        row_bytes = dimensions[0] // block_elements * block_bytes
        tensor_bytes = row_bytes
        for dimension in dimensions[1:]:
            tensor_bytes *= dimension
        relative_offset = gguf._align(tensor_offset + tensor_bytes, 32)
    header = b"GGUF" + struct.pack("<IQQ", 3, 809, 19) + b"".join(metadata) + bytes(table)
    data_offset = gguf._align(len(header), 32)
    with path.open("wb") as stream:
        stream.write(header)
        stream.write(bytes(data_offset - len(header)))
        stream.truncate(data_offset + tensor_offset + tensor_bytes)


def test_canonical_profile_contracts_are_pinned() -> None:
    assert gguf.KQUANT_PROFILES["k-quant-17gb"] == {
        "filename": "Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf",
        "nbytes": 16_756_683_904,
        "sha256": "4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e",
        "tensor_table_sha256": "44cd4374970ce14d5f944a1e0627831615a482874b656f0eb7bb5753817cc8fa",
        "inventory": {"F32": 313, "Q4_K": 365, "Q5_K": 1, "Q6_K": 52},
    }
    assert gguf.KQUANT_PROFILES["k-quant-dynamic"]["inventory"] == {
        "F32": 313,
        "Q4_K": 51,
        "Q5_K": 130,
        "Q6_K": 237,
    }
    assert len(gguf.expected_muse_glimmer_gguf_tensor_names()) == 731
    assert gguf.DFLASH_KQUANT_PROFILE == {
        "filename": "dflash-Muse-Glimmer-30B-Q4_K_M.gguf",
        "nbytes": 1_631_208_128,
        "sha256": "b2e808bf656086fe86bd0d0bd990f01d33e377537a07c02d45371517c8b264ef",
        "tensor_table_sha256": "cfb4c50ed5e0e760f5601b84d5ddbbce03d08fedcee41c4c9ed10c298def0b30",
        "tokenizer_metadata_sha256": "f7b318c5ce1048bc5775efc7f7dbbd92c50e02177b4dba5763ceaa0f5874f7de",
        "inventory": {"F32": 22, "Q4_K": 26, "Q6_K": 10},
    }
    assert len(gguf.expected_muse_glimmer_dflash_gguf_tensor_names()) == 58
    assert gguf.MMPROJ_KQUANT_PROFILE == {
        "filename": "mmproj-Muse-Glimmer-30B-Q4_K_M.gguf",
        "nbytes": 1_400_328_928,
        "sha256": "f48b452316f9b213758e8659444029b961a24a07f99a1abb2a9f88b06f7c00c6",
        "tensor_table_sha256": "47a880e1fde666694bf591879b3e8bbab6cff1a72ba883d959d3bf3cae4bea78",
        "inventory": {"F32": 506, "Q4_K": 200, "Q6_K": 100, "BF16": 3},
    }
    assert len(gguf.expected_muse_glimmer_mmproj_gguf_tensor_names()) == 809


def test_mmproj_parser_and_descriptor_cover_mixed_vision_types(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mmproj-fixture.gguf"
    _build_sparse_mmproj_fixture(path)
    parsed = gguf.inspect_muse_glimmer_mmproj_gguf_header_fixture(path)
    assert len(parsed.tensors) == 809
    assert parsed.encoding_inventory == {
        "Q4_K": 200,
        "Q6_K": 100,
        "F32": 506,
        "BF16": 3,
    }
    assert parsed.tensors[-3].native_name == "vision.adapter.fc1.weight"
    assert parsed.tensors[-3].shape == (4_096, 6_144)
    canonical = replace(
        parsed,
        path=tmp_path / gguf.MMPROJ_KQUANT_PROFILE["filename"],
        file_nbytes=gguf.MMPROJ_KQUANT_PROFILE["nbytes"],
        file_sha256=gguf.MMPROJ_KQUANT_PROFILE["sha256"],
        tensor_table_sha256=gguf.MMPROJ_KQUANT_PROFILE["tensor_table_sha256"],
    )
    profile = "k-quant-17gb"
    target_expected = gguf.KQUANT_PROFILES[profile]
    target = gguf.GGUFModel(
        path=tmp_path / target_expected["filename"],
        version=3,
        metadata={},
        tensors=(),
        alignment=32,
        data_offset=13_000_000,
        file_nbytes=target_expected["nbytes"],
        file_sha256=target_expected["sha256"],
        tensor_table_sha256=target_expected["tensor_table_sha256"],
        tokenizer_metadata_sha256=gguf.MUSE_GLIMMER_GGUF_TOKENIZER_METADATA_SHA256,
    )
    payload = gguf.build_muse_glimmer_kquant_execution_manifest_payload(
        {profile: target},
        mmproj_model=canonical,
    )
    descriptor = payload["companion_checkpoints"]["mmproj"]
    assert descriptor["encoding_inventory"] == gguf.MMPROJ_KQUANT_PROFILE["inventory"]
    assert descriptor["target_compatibility"]["packed_patch_width"] == 588
    assert descriptor["target_compatibility"]["temporal_patch_reduction"] == "sum"
    assert descriptor["target_compatibility"]["media_token_ids"] == {
        "image": 200_090,
        "video": 200_091,
        "patch": 200_092,
    }
    manifest = NativeExecutionManifest.from_dict(payload)
    assert manifest.capabilities["vision"] is True
    assert manifest.capabilities["video"] is False
    assert payload["kernel_abi"]["media_encoder"]["status"] == "ready"


def test_native_mmproj_loader_accepts_the_exact_sparse_canonical_table(
    tmp_path: Path,
) -> None:
    path = tmp_path / gguf.MMPROJ_KQUANT_PROFILE["filename"]
    _build_sparse_mmproj_fixture(path)
    executable = tmp_path / "mmproj-probe"
    subprocess.run(
        [
            "c++",
            "-std=c++20",
            "-O2",
            "-pthread",
            "-I",
            str(ROOT / "neuralfn/csrc/native_gpt2"),
            str(ROOT / "tests/cpp/resident_glimmer_mmproj_probe.cpp"),
            str(ROOT / "neuralfn/csrc/native_gpt2/resident_glimmer_vision.cpp"),
            "-o",
            str(executable),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    completed = subprocess.run(
        [str(executable), str(path)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "6656,1400328928"


def test_dflash_parser_validates_distinct_tensor_table_and_one_based_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "dflash-fixture.gguf"
    _build_sparse_dflash_fixture(path)
    monkeypatch.setattr(
        gguf,
        "_validate_muse_glimmer_dflash_metadata",
        lambda _metadata: "d" * 64,
    )
    model = gguf._parse_dflash_gguf(
        path,
        require_complete_file=True,
        compute_file_sha256=False,
    )
    assert len(model.tensors) == 58
    assert model.encoding_inventory == {"Q4_K": 26, "F32": 22, "Q6_K": 10}
    assert model.tensors[0].name == "fc.weight"
    assert model.tensors[0].shape == (6_656, 33_280)
    assert model.tensors[0].native_name == "assistant.context_projection.weight"
    assert model.tensors[-1].name == "output_norm.weight"
    assert model.tensors[-1].absolute_offset + model.tensors[-1].nbytes == path.stat().st_size

    bad = tmp_path / "dflash-bad.gguf"

    def mutate(index, _name, dimensions, tensor_type, tensor_offset):
        if index == 3:
            dimensions = (dimensions[0] + 256, *dimensions[1:])
        return dimensions, tensor_type, tensor_offset

    _build_sparse_dflash_fixture(bad, mutate_tensor=mutate)
    with pytest.raises(gguf.GGUFError, match="logical shape"):
        gguf._parse_dflash_gguf(
            bad,
            require_complete_file=True,
            compute_file_sha256=False,
        )


def test_parser_validates_shapes_offsets_and_exact_extent(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "fixture.gguf"
    _build_sparse_muse_fixture(path)
    monkeypatch.setattr(gguf, "_validate_muse_glimmer_metadata", lambda _metadata: "a" * 64)
    model = gguf._parse_gguf(
        path,
        require_complete_file=True,
        compute_file_sha256=False,
    )
    assert len(model.tensors) == 731
    assert model.tensors[0].shape == (202_048, 6_656)
    assert model.tensors[0].row_elements == 6_656
    assert model.tensors[0].native_name == "text.embedding.weight"
    assert model.tensors[-1].absolute_offset + model.tensors[-1].nbytes == path.stat().st_size
    assert model.file_sha256 is None
    assert len(model.tensor_table_sha256) == 64


@pytest.mark.parametrize("corruption", ["unknown_type", "wrong_shape", "overlap"])
def test_parser_fails_closed_on_tensor_table_corruption(
    tmp_path: Path,
    monkeypatch,
    corruption: str,
) -> None:
    path = tmp_path / f"{corruption}.gguf"

    def mutate(index, _name, dimensions, tensor_type, tensor_offset):
        if index == 1 and corruption == "unknown_type":
            tensor_type = 12345
        if index == 1 and corruption == "wrong_shape":
            dimensions = (dimensions[0] + 1, *dimensions[1:])
        if index == 1 and corruption == "overlap":
            tensor_offset = 0
        return dimensions, tensor_type, tensor_offset

    _build_sparse_muse_fixture(path, mutate_tensor=mutate)
    monkeypatch.setattr(gguf, "_validate_muse_glimmer_metadata", lambda _metadata: "b" * 64)
    with pytest.raises(gguf.GGUFError):
        gguf._parse_gguf(
            path,
            require_complete_file=True,
            compute_file_sha256=False,
        )


def test_reference_dequantizers_match_pinned_block_layouts() -> None:
    assert gguf.dequantize_ggml_blocks(struct.pack("<f", 1.25), gguf.GGML_TYPE_F32) == (
        1.25,
    )
    assert gguf.dequantize_ggml_blocks(bytes.fromhex("803f"), gguf.GGML_TYPE_BF16) == (
        1.0,
    )

    scales = bytes((1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0))
    q4 = struct.pack("<ee", 1.0, 0.5) + scales + bytes([0x21] * 128)
    q4_values = gguf.dequantize_ggml_blocks(q4, gguf.GGML_TYPE_Q4_K)
    assert len(q4_values) == 256
    assert q4_values[:32] == (-1.5,) * 32
    assert q4_values[32:64] == (1.0,) * 32
    assert q4_values[64:96] == (-0.5,) * 32
    assert q4_values[96:128] == (4.0,) * 32

    high = bytearray(32)
    high[0] = 3
    q5 = struct.pack("<ee", 1.0, 0.5) + scales + bytes(high) + bytes([0x21] * 128)
    q5_values = gguf.dequantize_ggml_blocks(q5, gguf.GGML_TYPE_Q5_K)
    assert len(q5_values) == 256
    assert q5_values[0] == 14.5
    assert q5_values[1] == -1.5
    assert q5_values[32] == 33.0
    assert q5_values[33] == 1.0

    q6 = bytes(128) + bytes(64) + struct.pack("<16b", *range(1, 17)) + struct.pack("<e", 0.5)
    q6_values = gguf.dequantize_ggml_blocks(q6, gguf.GGML_TYPE_Q6_K)
    assert len(q6_values) == 256
    assert (q6_values[0], q6_values[16], q6_values[32], q6_values[64]) == (
        -16.0,
        -32.0,
        -48.0,
        -80.0,
    )
    assert q6_values[128] == -144.0


def test_dequantizer_rejects_unknown_or_partial_blocks() -> None:
    with pytest.raises(gguf.GGUFError, match="Unsupported ggml type"):
        gguf.dequantize_ggml_blocks(b"", 999)
    with pytest.raises(gguf.GGUFError, match="not a multiple"):
        gguf.dequantize_ggml_blocks(bytes(143), gguf.GGML_TYPE_Q4_K)


def test_header_fixture_requires_pinned_table_and_inventory(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "header.bin"
    _build_sparse_muse_fixture(path)
    monkeypatch.setattr(gguf, "_validate_muse_glimmer_metadata", lambda _metadata: "c" * 64)
    monkeypatch.setitem(
        gguf.KQUANT_PROFILES,
        "fixture",
        {
            "filename": path.name,
            "nbytes": path.stat().st_size,
            "sha256": hashlib.sha256(b"not-read").hexdigest(),
            "tensor_table_sha256": "0" * 64,
            "inventory": {"F32": 731},
        },
    )
    with pytest.raises(gguf.GGUFError, match="tensor-table SHA-256 mismatch"):
        gguf.inspect_muse_glimmer_gguf_header_fixture(path, profile="fixture")


def test_raw_cuda_uses_a_distinct_typed_packed_weight_abi() -> None:
    root = Path(__file__).resolve().parents[1]
    header = (root / "neuralfn/csrc/native_train/tile_ops.h").read_text(encoding="utf-8")
    wrapper = (root / "neuralfn/csrc/native_train/tile_ops.cu").read_text(encoding="utf-8")
    kernels = (root / "neuralfn/csrc/tile_cuda/kernels.cu").read_text(encoding="utf-8")

    assert "struct NfnNativeTilePackedWeightDescriptorV1" in header
    assert "NFN_NATIVE_TILE_PACKED_WEIGHT_Q4_K = 12" in header
    assert "NFN_NATIVE_TILE_PACKED_WEIGHT_Q5_K = 13" in header
    assert "NFN_NATIVE_TILE_PACKED_WEIGHT_Q6_K = 14" in header
    assert "NFN_NATIVE_TILE_PACKED_WEIGHT_BF16 = 30" in header
    assert "nfn_native_tile_linear_packed_weight_float32_v1" in header
    assert "nfn_native_tile_linear_backward_input_packed_weight_float32_v1" in header
    assert "normalize_packed_weight_descriptor" in wrapper
    assert "source->data_nbytes != expected_nbytes" in wrapper
    assert "source->row_stride_bytes != expected_row_stride" in wrapper
    assert "if (kind < 1 || kind > 3)" in wrapper
    assert "packed_weight_value_device" in kernels
    assert "linear_packed_weight_float32_kernel" in kernels
    assert "linear_backward_input_packed_weight_float32_kernel" in kernels


def test_authenticated_kquant_descriptor_and_manifest_are_cpu_runnable(tmp_path: Path) -> None:
    profile = "k-quant-17gb"
    expected = gguf.KQUANT_PROFILES[profile]
    model = gguf.GGUFModel(
        path=tmp_path / expected["filename"],
        version=3,
        metadata={},
        tensors=(),
        alignment=32,
        data_offset=13_000_000,
        file_nbytes=expected["nbytes"],
        file_sha256=expected["sha256"],
        tensor_table_sha256=expected["tensor_table_sha256"],
        tokenizer_metadata_sha256=gguf.MUSE_GLIMMER_GGUF_TOKENIZER_METADATA_SHA256,
    )
    descriptor = gguf.kquant_checkpoint_descriptor(model, profile=profile)
    assert descriptor["target_nbytes"] == expected["nbytes"]
    assert descriptor["target_sha256"] == expected["sha256"]
    assert descriptor["required_kernel_profile"] == (
        "muse-glimmer-gguf-kquant-mapped-v1"
    )
    assert descriptor["capabilities"] == {
        "resident_cpu": True,
        "whole_model_cuda": True,
        "post_training": False,
    }
    payload = gguf.build_muse_glimmer_kquant_execution_manifest_payload(
        {profile: model}
    )
    parsed = NativeExecutionManifest.from_dict(payload)
    assert parsed.primary_checkpoint_variant == profile
    assert parsed.checkpoint == {
        "format": "neuralfn.native_family_muse_glimmer.gguf.kquant.v1",
        "artifact_path": expected["filename"],
        "target_nbytes": expected["nbytes"],
        "target_sha256": expected["sha256"],
    }
    assert parsed.capabilities["resident_inference"] is True
    assert parsed.capabilities["whole_model_cuda"] is True


def test_packed_dflash_descriptor_binds_both_kquant_targets(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "dflash-fixture.gguf"
    _build_sparse_dflash_fixture(path)
    monkeypatch.setattr(
        gguf,
        "_validate_muse_glimmer_dflash_metadata",
        lambda _metadata: gguf.DFLASH_KQUANT_PROFILE["tokenizer_metadata_sha256"],
    )
    parsed_dflash = gguf._parse_dflash_gguf(
        path,
        require_complete_file=True,
        compute_file_sha256=False,
    )
    canonical_dflash = replace(
        parsed_dflash,
        path=tmp_path / gguf.DFLASH_KQUANT_PROFILE["filename"],
        file_nbytes=gguf.DFLASH_KQUANT_PROFILE["nbytes"],
        file_sha256=gguf.DFLASH_KQUANT_PROFILE["sha256"],
        tensor_table_sha256=gguf.DFLASH_KQUANT_PROFILE["tensor_table_sha256"],
    )
    targets: dict[str, gguf.GGUFModel] = {}
    for profile, expected in gguf.KQUANT_PROFILES.items():
        targets[profile] = gguf.GGUFModel(
            path=tmp_path / expected["filename"],
            version=3,
            metadata={},
            tensors=(),
            alignment=32,
            data_offset=13_000_000,
            file_nbytes=expected["nbytes"],
            file_sha256=expected["sha256"],
            tensor_table_sha256=expected["tensor_table_sha256"],
            tokenizer_metadata_sha256=gguf.MUSE_GLIMMER_GGUF_TOKENIZER_METADATA_SHA256,
        )
    payload = gguf.build_muse_glimmer_kquant_execution_manifest_payload(
        targets,
        dflash_model=canonical_dflash,
    )
    descriptor = payload["companion_checkpoints"]["dflash"]
    assert descriptor["format"] == (
        "neuralfn.native_family_muse_glimmer_dflash.gguf.kquant.v1"
    )
    assert descriptor["encoding_inventory"] == {"F32": 22, "Q4_K": 26, "Q6_K": 10}
    assert descriptor["target_compatibility"]["target_layer_ids_gguf_one_based"] == [
        2,
        14,
        26,
        38,
        50,
    ]
    assert set(
        descriptor["target_compatibility"]["allowed_target_checkpoint_sha256"]
    ) == {row["sha256"] for row in gguf.KQUANT_PROFILES.values()}
    assert descriptor["capabilities"]["resident_cpu"] is True
    assert descriptor["capabilities"]["resident_cuda"] is True
    manifest = NativeExecutionManifest.from_dict(payload)
    assert manifest.capabilities["speculative_decoding"] is True
    assert manifest.kernel_abi["speculative_decoding"]["version"] == 1
