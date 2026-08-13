from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

import pytest

import neuralfn.native_muse_glimmer_checkpoint as checkpoint
from neuralfn.native_ir import NativeExecutionManifest


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_safetensors(
    path: Path,
    tensors: list[tuple[str, tuple[int, ...], bytes]],
    *,
    dtype: str = "BF16",
    mutate_header=None,
) -> None:
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    payload = bytearray()
    for name, shape, value in tensors:
        start = len(payload)
        payload.extend(value)
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [start, len(payload)],
        }
    if mutate_header is not None:
        mutate_header(header)
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def _fixture_bundle(tmp_path: Path):
    contracts = (
        checkpoint.MuseGlimmerTensorContract("source.a", "text.a", (2, 2), "text"),
        checkpoint.MuseGlimmerTensorContract(
            "source.b", "text.b", (3,), "text", "centered_delta"
        ),
    )
    shard_a = tmp_path / "model-00001-of-00002.safetensors"
    shard_b = tmp_path / "model-00002-of-00002.safetensors"
    _write_safetensors(shard_a, [("source.a", (2, 2), bytes(range(8)))])
    _write_safetensors(shard_b, [("source.b", (3,), bytes(range(8, 14)))])
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(
        json.dumps(
            {
                "metadata": {"total_parameters": 7, "total_size": 14},
                "weight_map": {
                    "source.a": shard_a.name,
                    "source.b": shard_b.name,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    auth = {
        shard_a.name: (shard_a.stat().st_size, _sha(shard_a)),
        shard_b.name: (shard_b.stat().st_size, _sha(shard_b)),
    }
    return contracts, index, auth


def _inspect(tmp_path: Path):
    contracts, index, auth = _fixture_bundle(tmp_path)
    bundle = checkpoint.inspect_safetensors_index(
        tmp_path,
        index_filename=index.name,
        index_sha256=_sha(index),
        contracts=contracts,
        shard_authentication=auth,
        expected_metadata={"total_parameters": 7, "total_size": 14},
    )
    return contracts, bundle


def _write_production_lora_checkpoint(
    root: Path,
    *,
    base_sha256: str = "c" * 64,
    training_adapter: str = "lora",
    base_weight_precision: str = "bf16",
) -> tuple[Path, dict[str, object]]:
    root.mkdir()
    rank = 1
    tensors: list[dict[str, object]] = []
    payload = bytearray()
    for layer in range(52):
        for suffix, rows, cols in (
            ("lora_A", rank, 6_656),
            ("lora_B", 4_096, rank),
        ):
            value = bytes([(layer * 2 + len(tensors)) % 251 + 1]) * (rows * cols * 2)
            offset = len(payload)
            payload.extend(value)
            tensors.append(
                {
                    "name": f"layers.{layer}.q_proj.weight.{suffix}",
                    "rows": rows,
                    "cols": cols,
                    "byte_offset": offset,
                    "nbytes": len(value),
                    "sha256": hashlib.sha256(value).hexdigest(),
                }
            )
    adapter = root / "adapter.bf16"
    adapter.write_bytes(payload)
    manifest: dict[str, object] = {
        "format": checkpoint.NATIVE_LORA_FORMAT,
        "architecture": "muse_glimmer",
        "base_weight_precision": base_weight_precision,
        "training_base_precision": (
            "nf4-group64-fp32-scale"
            if training_adapter == "qlora"
            else base_weight_precision
        ),
        "training_adapter": training_adapter,
        "layers": 52,
        "hidden_size": 6_656,
        "attention_size": 4_096,
        "kv_size": 256,
        "intermediate_size": 19_968,
        "adapter_path": "adapter.bf16",
        "adapter_sha256": hashlib.sha256(payload).hexdigest(),
        "base_sha256": base_sha256,
        "graph_topology_sha256": checkpoint.NATIVE_TRAIN_TOPOLOGY_SHA256,
        "graph_fingerprint": "d" * 64,
        "tokenizer_sha256": checkpoint.MUSE_GLIMMER_TOKENIZER_SHA256,
        "chat_template_sha256": checkpoint.MUSE_GLIMMER_ATEM_TEMPLATE_SHA256,
        "rank": rank,
        "alpha": 2.0,
        "scaling": 2.0,
        "dropout": 0.0,
        "seed": 17,
        "dtype": "bfloat16",
        "targets": ["q_proj"],
        "tensors": tensors,
    }
    (root / "adapter_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    return root, manifest


def test_production_tensor_contracts_cover_every_parameter_exactly() -> None:
    main = checkpoint.muse_glimmer_main_tensor_contracts()
    assistant = checkpoint.muse_glimmer_assistant_tensor_contracts()

    assert len(main) == 1_436
    assert sum(contract.elements for contract in main) == checkpoint.MAIN_PARAMETER_COUNT
    assert sum(contract.nbytes for contract in main) == checkpoint.MAIN_PAYLOAD_BYTES
    assert sum(contract.component == "text" for contract in main) == 627
    assert sum(contract.component == "vision" for contract in main) == 809
    assert sum(contract.parameterization == "centered_delta" for contract in main) == 208
    assert len({contract.source_name for contract in main}) == len(main)
    assert len({contract.native_name for contract in main}) == len(main)

    assert len(assistant) == 58
    assert sum(contract.elements for contract in assistant) == checkpoint.ASSISTANT_PARAMETER_COUNT
    assert sum(contract.nbytes for contract in assistant) == checkpoint.ASSISTANT_PAYLOAD_BYTES
    context = next(
        contract for contract in assistant if contract.source_name == "encoder.fc.weight"
    )
    assert context.shape == (6_656, 33_280)


def test_strict_sharded_inspection_and_bounded_native_conversion(tmp_path: Path) -> None:
    _contracts, bundle = _inspect(tmp_path)
    assert bundle.parameter_count == 7
    assert bundle.payload_bytes == 14
    assert [entry.contract.native_name for entry in bundle.entries] == ["text.a", "text.b"]

    converted = checkpoint._publish_converted_checkpoint(
        bundle,
        tmp_path / "native",
        entries=bundle.entries,
        format_name=checkpoint.MAIN_FORMAT,
        component="text",
        source_revision="fixture-revision",
        source_files={"fixture": {"sha256": "0" * 64, "nbytes": 1}},
        compatible_target_sha256=None,
    )
    assert converted.artifact_path.read_bytes() == bytes(range(14))
    metadata = json.loads(converted.metadata_path.read_text(encoding="utf-8"))
    assert metadata["artifact"]["target_nbytes"] == 14
    assert metadata["artifact"]["target_sha256"] == hashlib.sha256(bytes(range(14))).hexdigest()
    assert metadata["tensors"][1]["parameterization"] == "centered_delta"
    assert metadata["capabilities"]["resident_cpu"] is True
    assert metadata["capabilities"]["resident_cuda"] is True
    assert converted.done_path.is_file()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        checkpoint._publish_converted_checkpoint(
            bundle,
            tmp_path / "native",
            entries=bundle.entries,
            format_name=checkpoint.MAIN_FORMAT,
            component="text",
            source_revision="fixture-revision",
            source_files={},
            compatible_target_sha256=None,
        )


@pytest.mark.parametrize("corruption", ["dtype", "shape", "overlap", "truncated"])
def test_safetensors_inspection_rejects_tensor_corruption(
    tmp_path: Path,
    corruption: str,
) -> None:
    contracts, index, auth = _fixture_bundle(tmp_path)
    shard = tmp_path / "model-00001-of-00002.safetensors"

    def mutate(header):
        tensor = header["source.a"]
        if corruption == "shape":
            tensor["shape"] = [4, 1]
        elif corruption == "overlap":
            tensor["data_offsets"] = [1, 9]

    if corruption == "dtype":
        _write_safetensors(shard, [("source.a", (2, 2), bytes(range(8)))], dtype="F16")
    elif corruption == "truncated":
        shard.write_bytes(shard.read_bytes()[:-1])
    else:
        _write_safetensors(
            shard,
            [("source.a", (2, 2), bytes(range(8)))],
            mutate_header=mutate,
        )
    auth[shard.name] = (shard.stat().st_size, _sha(shard))
    with pytest.raises(checkpoint.MuseGlimmerCheckpointError):
        checkpoint.inspect_safetensors_index(
            tmp_path,
            index_filename=index.name,
            index_sha256=_sha(index),
            contracts=contracts,
            shard_authentication=auth,
            expected_metadata={"total_parameters": 7, "total_size": 14},
        )


def test_safetensors_index_rejects_hash_missing_unexpected_and_traversal(tmp_path: Path) -> None:
    contracts, index, auth = _fixture_bundle(tmp_path)
    with pytest.raises(checkpoint.MuseGlimmerCheckpointError, match="SHA-256 mismatch"):
        checkpoint.inspect_safetensors_index(
            tmp_path,
            index_filename=index.name,
            index_sha256="0" * 64,
            contracts=contracts,
            shard_authentication=auth,
        )

    payload = json.loads(index.read_text(encoding="utf-8"))
    payload["weight_map"].pop("source.b")
    index.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(checkpoint.MuseGlimmerCheckpointError, match="allowlist mismatch"):
        checkpoint.inspect_safetensors_index(
            tmp_path,
            index_filename=index.name,
            index_sha256=_sha(index),
            contracts=contracts,
            shard_authentication=auth,
        )

    payload["weight_map"]["source.b"] = "../escape.safetensors"
    index.write_text(json.dumps(payload), encoding="utf-8")
    traversal_auth = dict(auth)
    traversal_auth.pop("model-00002-of-00002.safetensors")
    traversal_auth["../escape.safetensors"] = (1, "0" * 64)
    with pytest.raises(checkpoint.MuseGlimmerCheckpointError, match="Unsafe"):
        checkpoint.inspect_safetensors_index(
            tmp_path,
            index_filename=index.name,
            index_sha256=_sha(index),
            contracts=contracts,
            shard_authentication=traversal_auth,
        )


def test_safetensors_index_rejects_duplicate_json_tensor_name(tmp_path: Path) -> None:
    contracts, _index, auth = _fixture_bundle(tmp_path)
    duplicate = tmp_path / "duplicate.index.json"
    duplicate.write_text(
        '{"metadata":{},"weight_map":{"source.a":"model-00001-of-00002.safetensors",'
        '"source.a":"model-00001-of-00002.safetensors",'
        '"source.b":"model-00002-of-00002.safetensors"}}',
        encoding="utf-8",
    )
    with pytest.raises(checkpoint.MuseGlimmerCheckpointError, match="duplicate key"):
        checkpoint.inspect_safetensors_index(
            tmp_path,
            index_filename=duplicate.name,
            index_sha256=_sha(duplicate),
            contracts=contracts,
            shard_authentication=auth,
        )


def test_assistant_conversion_requires_exact_target_digest(tmp_path: Path) -> None:
    with pytest.raises(checkpoint.MuseGlimmerCheckpointError, match="target checkpoint"):
        checkpoint.convert_official_muse_glimmer_assistant_safetensors(
            tmp_path,
            tmp_path / "out",
            target_checkpoint_sha256="not-a-digest",
        )


def test_runnable_text_manifest_is_additive_strict_cpu_and_cuda(tmp_path: Path) -> None:
    converted = checkpoint.ConvertedMuseGlimmerCheckpoint(
        metadata_path=tmp_path / "checkpoint.json",
        artifact_path=tmp_path / "muse-glimmer-text.bf16",
        done_path=tmp_path / "DONE",
        format=checkpoint.MAIN_FORMAT,
        component="text",
        nbytes=55_709_561_856,
        sha256="a" * 64,
        tensor_count=627,
    )
    row = {
        "name": "fixture",
        "source_name": "fixture",
        "dtype": "bfloat16",
        "shape": [1],
        "offset": 0,
        "nbytes": 2,
        "sha256": "b" * 64,
        "byte_order": "little",
        "layout": "row_major",
        "parameterization": "ordinary",
        "component": "text",
    }
    metadata = {
        "schema": "neuralfn.native_muse_glimmer_checkpoint",
        "version": 1,
        "component": "text",
        "artifact": {
            "target_nbytes": converted.nbytes,
            "target_sha256": converted.sha256,
        },
        "tensors": [dict(row) for _ in range(627)],
    }
    payload = checkpoint.build_muse_glimmer_execution_manifest_payload(
        converted, metadata
    )
    parsed = NativeExecutionManifest.from_dict(payload)
    assert parsed.primary_checkpoint_variant == "bf16"
    assert parsed.checkpoint_variants["bf16"]["required_kernel_profile"] == (
        "muse-glimmer-bf16-mapped-v1"
    )
    assert parsed.capabilities["resident_inference"] is True
    assert parsed.capabilities["whole_model_cuda"] is True
    assert payload["kernel_abi"]["whole_model_cuda"] == {
        "version": 1,
        "status": "ready",
        "profile": "muse-glimmer-hybrid-gqa-bf16-cache-v1",
        "feature_abi_symbol": "nfn_native_tile_glimmer_inference_abi_version",
        "load_operation": "load_model_with_options",
    }
    assert parsed.capabilities["vision"] is False
    assert parsed.stop_tokens == (200_001, 200_008)
    assert parsed.model["template_spec"]["block_spec"]["layer_attention_pattern"] == [
        {
            "kind": "local",
            "window_size": 2_048,
            "pos_encoding": "rope",
            "rope_theta": 500_000.0,
        },
        {
            "kind": "local",
            "window_size": 2_048,
            "pos_encoding": "rope",
            "rope_theta": 500_000.0,
        },
        {
            "kind": "local",
            "window_size": 2_048,
            "pos_encoding": "rope",
            "rope_theta": 500_000.0,
        },
        {
            "kind": "full",
            "window_size": None,
            "pos_encoding": "none",
            "rope_theta": 500_000.0,
        },
    ]


def test_native_lora_inspector_authenticates_every_tensor(tmp_path: Path) -> None:
    root, manifest = _write_production_lora_checkpoint(tmp_path / "lora")
    descriptor = checkpoint.inspect_native_muse_glimmer_lora_checkpoint(root)

    assert descriptor["format"] == checkpoint.NATIVE_LORA_FORMAT
    assert descriptor["rank"] == 1
    assert descriptor["alpha"] == 2.0
    assert descriptor["targets"] == ["q_proj"]
    assert descriptor["target_sha256"] == manifest["adapter_sha256"]
    assert descriptor["target_nbytes"] == 52 * (6_656 + 4_096) * 2
    assert descriptor["target_compatibility"] == {
        "allowed_target_checkpoint_sha256": ["c" * 64],
        "base_weight_precision": "bf16",
        "tokenizer_sha256": checkpoint.MUSE_GLIMMER_TOKENIZER_SHA256,
        "chat_template_sha256": checkpoint.MUSE_GLIMMER_ATEM_TEMPLATE_SHA256,
    }

    adapter = root / "adapter.bf16"
    damaged = bytearray(adapter.read_bytes())
    damaged[-1] ^= 1
    adapter.write_bytes(damaged)
    with pytest.raises(
        checkpoint.MuseGlimmerCheckpointError, match="failed SHA-256 validation"
    ):
        checkpoint.inspect_native_muse_glimmer_lora_checkpoint(root)


def test_native_qlora_adapter_inspector_preserves_training_provenance(
    tmp_path: Path,
) -> None:
    root, _manifest = _write_production_lora_checkpoint(
        tmp_path / "qlora", training_adapter="qlora"
    )
    descriptor = checkpoint.inspect_native_muse_glimmer_lora_checkpoint(root)
    assert descriptor["source"]["training_adapter"] == "qlora"
    assert descriptor["source"]["training_base_precision"] == (
        "nf4-group64-fp32-scale"
    )
    assert descriptor["capabilities"]["qlora"] is True

    manifest_path = root / "adapter_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["training_base_precision"] = "nf4-unknown"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(
        checkpoint.MuseGlimmerCheckpointError,
        match="architecture/provenance contract",
    ):
        checkpoint.inspect_native_muse_glimmer_lora_checkpoint(root)


@pytest.mark.parametrize("profile", ["k-quant-17gb", "k-quant-dynamic"])
def test_native_kquant_lora_adapter_inspector_preserves_packed_base_lineage(
    tmp_path: Path, profile: str
) -> None:
    root, _manifest = _write_production_lora_checkpoint(
        tmp_path / profile,
        training_adapter="lora",
        base_weight_precision=profile,
    )
    descriptor = checkpoint.inspect_native_muse_glimmer_lora_checkpoint(root)
    assert descriptor["target_compatibility"]["base_weight_precision"] == profile
    assert descriptor["source"]["training_base_precision"] == profile
    assert descriptor["capabilities"]["kquant_lora"] is True
    assert descriptor["capabilities"]["qlora"] is False


def test_native_lora_attachment_is_atomic_and_lineage_bound(tmp_path: Path) -> None:
    base_sha = "c" * 64
    lora, manifest = _write_production_lora_checkpoint(
        tmp_path / "lora", base_sha256=base_sha
    )
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    execution = {
        "schema": "neuralfn.native_execution_manifest",
        "version": 1,
        "checkpoint_variants": {
            "bf16": {
                "target_sha256": base_sha,
                "weight_precision": "bf16",
            }
        },
        "tokenizer": {"sha256": checkpoint.MUSE_GLIMMER_TOKENIZER_SHA256},
        "chat_template": {
            "sha256": checkpoint.MUSE_GLIMMER_ATEM_TEMPLATE_SHA256
        },
        "companion_checkpoints": {},
        "kernel_abi": {},
        "capabilities": {},
    }
    manifest_path = bundle / "native-execution-manifest.json"
    manifest_path.write_text(json.dumps(execution), encoding="utf-8")
    done_path = bundle / "native-execution-DONE"
    done_path.write_text(
        json.dumps(
            {
                "schema": "neuralfn.native_execution_bundle.done",
                "version": 1,
                "manifest_sha256": _sha(manifest_path),
                "checkpoint_sha256": base_sha,
            }
        ),
        encoding="utf-8",
    )

    assert checkpoint.attach_native_muse_glimmer_lora(bundle, lora) == manifest_path
    updated = json.loads(manifest_path.read_text(encoding="utf-8"))
    descriptor = updated["companion_checkpoints"]["lora"]
    assert descriptor["target_sha256"] == manifest["adapter_sha256"]
    assert updated["capabilities"]["native_lora"] is True
    assert updated["capabilities"]["post_training"] is True
    assert (bundle / "muse-glimmer-lora.bf16").read_bytes() == (
        lora / "adapter.bf16"
    ).read_bytes()
    assert json.loads(done_path.read_text(encoding="utf-8"))[
        "lora_checkpoint_sha256"
    ] == manifest["adapter_sha256"]
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        checkpoint.attach_native_muse_glimmer_lora(bundle, lora)


def test_native_lora_cli_attach_round_trips_qlora_provenance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from cli import nfn

    base_sha = "c" * 64
    qlora, adapter_manifest = _write_production_lora_checkpoint(
        tmp_path / "qlora", base_sha256=base_sha, training_adapter="qlora"
    )
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    execution = {
        "schema": "neuralfn.native_execution_manifest",
        "version": 1,
        "checkpoint_variants": {
            "bf16": {
                "target_sha256": base_sha,
                "weight_precision": "bf16",
            }
        },
        "tokenizer": {"sha256": checkpoint.MUSE_GLIMMER_TOKENIZER_SHA256},
        "chat_template": {
            "sha256": checkpoint.MUSE_GLIMMER_ATEM_TEMPLATE_SHA256
        },
        "companion_checkpoints": {},
        "kernel_abi": {},
        "capabilities": {},
    }
    manifest_path = bundle / "native-execution-manifest.json"
    manifest_path.write_text(json.dumps(execution), encoding="utf-8")
    (bundle / "native-execution-DONE").write_text(
        json.dumps(
            {
                "schema": "neuralfn.native_execution_bundle.done",
                "version": 1,
                "manifest_sha256": _sha(manifest_path),
                "checkpoint_sha256": base_sha,
            }
        ),
        encoding="utf-8",
    )

    assert (
        nfn.main(
            [
                "migrate",
                "muse-glimmer-lora-to-native",
                "--artifact",
                str(bundle),
                "--checkpoint",
                str(qlora),
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["adapter_sha256"] == adapter_manifest["adapter_sha256"]
    assert output["training_adapter"] == "qlora"
    assert output["resident_cpu"] is True
    assert output["resident_cuda"] is True
