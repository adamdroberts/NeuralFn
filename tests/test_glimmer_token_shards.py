from __future__ import annotations

import json
from pathlib import Path
import shutil
import struct
import subprocess

import numpy as np
import pytest

from server.dataset_manager import (
    TOKEN_SHARD_V2_ENDIAN_MARKER,
    TOKEN_SHARD_V2_HEADER_BYTES,
    build_token_shard_v2_header,
    inspect_structured_sft_v1,
    inspect_token_shard,
    write_structured_sft_v1,
)


ROOT = Path(__file__).resolve().parents[1]
BOUNDARY_IDS = [65_535, 65_536, 201_818, 202_047, 1]


def _write_v2(path: Path, values: list[int], *, split: str = "train", vocab_size: int = 202_048) -> None:
    payload = np.asarray(values, dtype=np.dtype("<u4")).tobytes()
    path.write_bytes(
        build_token_shard_v2_header(
            token_count=len(values),
            tokenizer_vocab_size=vocab_size,
            tokenizer_sha256="a" * 64,
            tokenizer_revision="fixture-r1",
            tokenizer_name="muse-glimmer-fixture",
            split=split,
            objective="ar",
        )
        + payload
    )


def test_uint32_token_shard_v2_round_trips_wide_ids(tmp_path: Path) -> None:
    shard = tmp_path / "fineweb_train_000000.bin"
    _write_v2(shard, BOUNDARY_IDS)

    inspected = inspect_token_shard(shard)
    assert inspected == {
        "schema": "neuralfn.native_token_shard.v2",
        "dtype": "uint32_le",
        "element_bytes": 4,
        "header_bytes": TOKEN_SHARD_V2_HEADER_BYTES,
        "token_count": len(BOUNDARY_IDS),
        "tokenizer_vocab_size": 202_048,
        "tokenizer_sha256": "a" * 64,
        "tokenizer_revision": "fixture-r1",
        "tokenizer_name": "muse-glimmer-fixture",
        "split": "train",
        "objective": "ar",
        "max_token_id": 202_047,
    }


@pytest.mark.parametrize("corruption", ["endian", "reserved", "truncated", "out_of_range"])
def test_uint32_token_shard_v2_rejects_corruption(tmp_path: Path, corruption: str) -> None:
    shard = tmp_path / "fineweb_train_000000.bin"
    values = list(BOUNDARY_IDS)
    _write_v2(shard, values)
    content = bytearray(shard.read_bytes())
    if corruption == "endian":
        struct.pack_into("<I", content, 20, TOKEN_SHARD_V2_ENDIAN_MARKER + 1)
    elif corruption == "reserved":
        content[511] = 1
    elif corruption == "truncated":
        content.pop()
    else:
        struct.pack_into("<I", content, TOKEN_SHARD_V2_HEADER_BYTES, 202_048)
    shard.write_bytes(content)

    with pytest.raises(ValueError):
        inspect_token_shard(shard)


def test_legacy_uint16_token_shard_remains_header_compatible(tmp_path: Path) -> None:
    shard = tmp_path / "fineweb_train_000000.bin"
    shard.write_bytes(b"\x88\xd8\x34\x01" + bytes(1020) + np.asarray([1, 2, 65_535], dtype="<u2").tobytes())
    inspected = inspect_token_shard(shard)
    assert inspected["schema"] == "legacy.uint16"
    assert inspected["header_bytes"] == 1024
    assert inspected["token_count"] == 3
    assert inspected["max_token_id"] == 65_535


def test_cpp_resolver_and_wide_sampler_read_v2_and_reject_legacy_sampler(tmp_path: Path) -> None:
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is unavailable")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _write_v2(dataset / "fineweb_train_000000.bin", BOUNDARY_IDS)
    _write_v2(dataset / "fineweb_val_000000.bin", BOUNDARY_IDS, split="validation")
    helper = tmp_path / "wide_sampler.cpp"
    helper.write_text(
        r"""
#include "token_shards.h"
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        auto dataset = neuralfn::native_train::resolve_token_shards(argv[1], false);
        neuralfn::native_train::SequentialTokenBatchSampler32 sampler(dataset.train_shards, 4, 1);
        neuralfn::native_train::TokenBatch32 batch;
        if (!sampler.next(batch)) return 3;
        bool legacy_rejected = false;
        try {
            neuralfn::native_train::SequentialTokenBatchSampler legacy(dataset.train_shards, 4, 1);
        } catch (const std::exception&) {
            legacy_rejected = true;
        }
        if (!legacy_rejected) return 4;
        std::cout << neuralfn::native_train::token_shard_dataset_json(dataset) << "\n";
        std::cout << neuralfn::native_train::token_batch_json(batch) << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 2;
    }
}
""",
        encoding="utf-8",
    )
    binary = tmp_path / "wide_sampler"
    compile_result = subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-I",
            str(ROOT / "neuralfn" / "csrc" / "native_train"),
            str(helper),
            str(ROOT / "neuralfn" / "csrc" / "native_train" / "token_shards.cpp"),
            "-o",
            str(binary),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert compile_result.returncode == 0, compile_result.stderr
    run = subprocess.run(
        [str(binary), str(dataset)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    dataset_json, batch_json = [json.loads(line) for line in run.stdout.splitlines()]
    assert dataset_json["train_shards"][0]["dtype"] == "uint32_le"
    assert dataset_json["train_shards"][0]["max_token_id"] == 202_047
    assert batch_json["dtype"] == "uint32"
    assert batch_json["tokens"] == BOUNDARY_IDS[:4]
    assert batch_json["targets"] == BOUNDARY_IDS[1:]


def test_native_tile_exports_uint32_token_conversion() -> None:
    header = (ROOT / "neuralfn" / "csrc" / "native_train" / "tile_ops.h").read_text(encoding="utf-8")
    source = (ROOT / "neuralfn" / "csrc" / "native_train" / "tile_ops.cu").read_text(encoding="utf-8")
    kernels = (ROOT / "neuralfn" / "csrc" / "tile_cuda" / "kernels.cu").read_text(encoding="utf-8")

    assert "nfn_native_tile_uint32_to_int64" in header
    assert "nfn_native_tile_uint32_to_int64" in source
    assert "launch_uint32_to_int64" in source
    assert "uint32_to_int64_kernel" in kernels
    assert "launch_uint32_to_int64" in kernels


def test_structured_sft_v1_round_trip_and_corruption_gates(tmp_path: Path) -> None:
    path = tmp_path / "sft_train_000000.sft"
    record = {
        "input_ids": [65_535, 65_536, 201_818, 202_047],
        "targets": [-100, 201_818, -100, 200_008],
        "loss_mask": [0.0, 1.0, 0.0, 1.0],
        "sequence_ids": [0, 0, 1, 1],
    }
    write_structured_sft_v1(
        path,
        [record],
        sequence_length=4,
        tokenizer_vocab_size=202_048,
        pad_token_id=200_018,
        tokenizer_sha256="a" * 64,
        chat_template_sha256="b" * 64,
        tokenizer_revision="fixture-r1",
        split="train",
    )
    inspected = inspect_structured_sft_v1(path)
    assert inspected["schema"] == "neuralfn.native_structured_sft.v1"
    assert inspected["record_count"] == 1
    assert inspected["chat_template_sha256"] == "b" * 64

    bad = bytearray(path.read_bytes())
    # The ignored first target must have a zero loss mask.
    struct.pack_into("<f", bad, 512 + 4 * 4 + 4 * 4, 1.0)
    path.write_bytes(bad)
    with pytest.raises(ValueError, match="Ignored targets"):
        inspect_structured_sft_v1(path)
