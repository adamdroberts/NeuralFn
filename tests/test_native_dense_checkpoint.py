from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

import pytest

from neuralfn.native_dense_checkpoint import (
    NATIVE_DENSE_GPT_CHECKPOINT_FORMAT,
    inspect_native_dense_checkpoint,
)


def _write_checkpoint(path: Path, *, channels: int = 4, num_heads: int = 2) -> None:
    max_seq_len = 8
    vocab_size = 7
    padded_vocab_size = 8
    num_layers = 2
    header = [0] * 256
    header[:8] = [
        20240326,
        5,
        max_seq_len,
        vocab_size,
        num_layers,
        num_heads,
        channels,
        padded_vocab_size,
    ]
    parameter_count = (
        padded_vocab_size * channels
        + max_seq_len * channels
        + num_layers * (12 * channels * channels + 13 * channels)
        + 2 * channels
    )
    payload = bytes((index * 17 + 3) & 0xFF for index in range(parameter_count * 2))
    path.write_bytes(struct.pack("<256i", *header) + payload)


def _model(**overrides):
    template_overrides = overrides.pop("template_spec", {})
    block = {
        "norm_type": "layernorm",
        "mlp_type": "gelu",
        "pos_encoding": "absolute",
        "attention_variant": "dense",
        "residual_type": "add",
        "compression": "none",
        "activation_mode": "single",
        "linear_bias": True,
        "use_qk_norm": False,
        "dropout_p": 0.0,
        "num_heads": 2,
        "num_kv_heads": 2,
    }
    block.update(overrides.pop("block", {}))
    template_spec = {
        "model_dim": 4,
        "num_layers": 2,
        "vocab_size": 7,
        "tie_embeddings": True,
        "logit_softcap": 0.0,
        "block_spec": block,
    }
    template_spec.update(template_overrides)
    model = {
        "family": "gpt2",
        "family_class": "autoregressive_transformer",
        "template_spec": template_spec,
    }
    model.update(overrides)
    return model


def test_inspector_emits_exact_geometry_tensor_offsets_and_checksums(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model_00000001.bin"
    _write_checkpoint(checkpoint)
    info = inspect_native_dense_checkpoint(checkpoint)

    assert info.max_seq_len == 8
    assert info.vocab_size == 7
    assert info.padded_vocab_size == 8
    assert info.num_layers == 2
    assert info.num_heads == 2
    assert info.channels == 4
    assert info.sha256 == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert info.tensors[0].name == "transformer.wte.weight"
    assert info.tensors[0].shape == (8, 4)
    assert info.tensors[0].offset == 1024
    assert info.tensors[-1].name == "transformer.ln_f.bias"
    assert info.tensors[-1].offset + info.tensors[-1].nbytes == info.file_size
    for tensor in info.tensors:
        raw = checkpoint.read_bytes()[tensor.offset : tensor.offset + tensor.nbytes]
        assert tensor.sha256 == hashlib.sha256(raw).hexdigest()

    descriptor = info.checkpoint_descriptor(artifact_path="model.bin")
    assert descriptor["format"] == NATIVE_DENSE_GPT_CHECKPOINT_FORMAT
    assert descriptor["artifact_path"] == "model.bin"
    assert descriptor["target_sha256"] == info.sha256
    assert descriptor["geometry"]["max_seq_len"] == 8


def test_inspector_rejects_bad_header_geometry_and_length(tmp_path: Path) -> None:
    truncated = tmp_path / "truncated.bin"
    truncated.write_bytes(b"short")
    with pytest.raises(ValueError, match="header is truncated"):
        inspect_native_dense_checkpoint(truncated)

    bad_geometry = tmp_path / "bad-geometry.bin"
    _write_checkpoint(bad_geometry, channels=3, num_heads=2)
    with pytest.raises(ValueError, match="invalid model geometry"):
        inspect_native_dense_checkpoint(bad_geometry)

    bad_length = tmp_path / "bad-length.bin"
    _write_checkpoint(bad_length)
    bad_length.write_bytes(bad_length.read_bytes()[:-2])
    with pytest.raises(ValueError, match="file size"):
        inspect_native_dense_checkpoint(bad_length)


@pytest.mark.parametrize(
    ("model", "message"),
    [
        (_model(family="llama"), "dense GPT-family"),
        (_model(block={"pos_encoding": "rope"}), "pos_encoding"),
        (_model(block={"use_qk_norm": "yes"}), "boolean use_qk_norm"),
        (_model(template_spec={"logit_softcap": -1.0}), "non-negative logit_softcap"),
        (_model(block={"num_kv_heads": 1}), "requires MHA"),
        (_model(template_spec={"model_dim": 8}), "model_dim"),
    ],
)
def test_model_contract_fails_closed(model, message: str, tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.bin"
    _write_checkpoint(checkpoint)
    info = inspect_native_dense_checkpoint(checkpoint)
    with pytest.raises(ValueError, match=message):
        info.validate_model(model)


@pytest.mark.parametrize(
    "model",
    [
        _model(block={"use_qk_norm": True}),
        _model(template_spec={
            "model_dim": 4,
            "num_layers": 2,
            "vocab_size": 7,
            "tie_embeddings": True,
            "logit_softcap": 30.0,
            "block_spec": {
                "norm_type": "layernorm",
                "mlp_type": "gelu",
                "pos_encoding": "absolute",
                "attention_variant": "dense",
                "residual_type": "add",
                "compression": "none",
                "activation_mode": "single",
                "linear_bias": True,
                "use_qk_norm": False,
                "dropout_p": 0.0,
                "num_heads": 2,
                "num_kv_heads": 2,
            },
        }),
    ],
)
def test_model_contract_accepts_parameter_free_dense_inference_variants(
    model,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.bin"
    _write_checkpoint(checkpoint)
    inspect_native_dense_checkpoint(checkpoint).validate_model(model)


def test_inspector_is_dependency_light(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.bin"
    _write_checkpoint(checkpoint)
    code = (
        "import json,sys; "
        "from neuralfn.native_dense_checkpoint import inspect_native_dense_checkpoint; "
        f"info=inspect_native_dense_checkpoint({str(checkpoint)!r}); "
        "print(json.dumps({'sha256':info.sha256,'heavy':[n for n in ('torch','numpy','networkx') if n in sys.modules]}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert payload["heavy"] == []
