from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

import neuralfn.native_inference as native_inference
from neuralfn.native_cli import NativeArtifactCLIConfig
from neuralfn.native_inference import (
    NativeInferenceCapabilityError,
    NativeModelLoadConfig,
    _cuda_runtime_candidates,
    _load_manifest_with_path_pin,
    _resolve_model_tile_ops_library,
    _select_checkpoint_variant,
    _variant_runtime_bytes,
)
from neuralfn.native_serve import NativeServeConfig


class _Binding:
    def __init__(self, *, cuda: bool = True, profiles: tuple[str, ...] = ()) -> None:
        self.cuda = cuda
        self.profiles = profiles

    def resident_inference_capabilities(self) -> Mapping[str, Any]:
        return {
            "whole_model_cuda": self.cuda,
            "weight_kernel_profiles": list(self.profiles),
        }


def _descriptor(root: Path, profile: str, size: int, *, kernel: str) -> dict[str, Any]:
    payload = (profile + "\n").encode()
    path = root / f"{profile}.bin"
    path.write_bytes(payload)
    return {
        "format": f"fixture.{profile}",
        "artifact_path": path.name,
        "target_nbytes": len(payload),
        "target_sha256": hashlib.sha256(payload).hexdigest(),
        "required_kernel_profile": kernel,
        "resident_weight_bytes": size,
        "peak_load_staging_bytes": 10,
        "max_workspace_bytes": 20,
        "memory_profile": {
            "version": 1,
            "minimum_total_vram_bytes": 1_000,
            "backend_fingerprint": "fixture-gpu-v1",
            "fixed_runtime_bytes": 30,
            "kv_cache_bytes_per_context_token_per_session": 2,
            "session_bytes": 40,
        },
    }


def _manifest(root: Path) -> dict[str, Any]:
    bf16 = _descriptor(root, "bf16", 900, kernel="bf16-kernel")
    dynamic = _descriptor(root, "k-quant-dynamic", 600, kernel="k-kernel")
    compact = _descriptor(root, "k-quant-17gb", 400, kernel="k-kernel")
    return {
        "schema": "neuralfn.native_execution_manifest",
        "version": 1,
        "checkpoint": {
            key: bf16[key]
            for key in ("format", "artifact_path", "target_nbytes", "target_sha256")
        },
        "primary_checkpoint_variant": "bf16",
        "checkpoint_variants": {
            "bf16": bf16,
            "k-quant-dynamic": dynamic,
            "k-quant-17gb": compact,
        },
        "companion_checkpoints": {},
        "context_limits": {"max_context_tokens": 100},
        "capabilities": {"whole_model_cuda": True},
    }


def _add_dflash(root: Path, manifest: dict[str, Any], *, resident_bytes: int) -> None:
    payload = b"fixture-dflash"
    (root / "dflash.bin").write_bytes(payload)
    manifest["companion_checkpoints"] = {
        "dflash": {
            "format": "neuralfn.native_family_muse_glimmer_dflash.bf16.v1",
            "artifact_path": "dflash.bin",
            "target_nbytes": len(payload),
            "target_sha256": hashlib.sha256(payload).hexdigest(),
            "resident_weight_bytes": resident_bytes,
            "target_compatibility": {
                "allowed_target_checkpoint_sha256": [
                    row["target_sha256"]
                    for row in manifest["checkpoint_variants"].values()
                ],
                "target_layer_ids_zero_based": [1, 13, 25, 37, 49],
                "block_size": 16,
                "proposal_tokens": 15,
                "mask_token_id": 201818,
                "shared_embedding": True,
                "shared_lm_head": True,
            },
        }
    }


def test_load_config_defaults_and_public_config_round_trip(tmp_path: Path) -> None:
    load = NativeModelLoadConfig()
    assert load.weight_precision == "auto"
    assert load.speculative_decoding == "auto"
    assert load.to_dict()["weight_precision"] == "auto"
    assert NativeArtifactCLIConfig(tmp_path).model_load == load
    assert NativeServeConfig(tmp_path).model_load == load

    for precision in ("auto", "bf16", "k-quant-dynamic", "k-quant-17gb"):
        configured = NativeModelLoadConfig(
            weight_precision=precision,
            runtime="native-cuda",
            cuda_device=3,
            cuda_runtime_lib="libcudart-fixture.so",
        )
        assert configured.weight_precision == precision
        assert configured.cuda_device == 3


def test_model_cuda_device_is_independent_from_tile_cache_config() -> None:
    load = NativeModelLoadConfig(cuda_device=2)
    assert load.cuda_device == 2
    # Constructing the model policy no longer requires putting this model-only
    # device into KVCacheConfig's Tile-only fields.
    assert NativeArtifactCLIConfig(Path("artifact"), model_load=load).kv_cache.cuda_device == 0


def test_cuda13_runtime_discovery_covers_workspace_scratch_and_explicit_pin(
    tmp_path: Path,
) -> None:
    volume = tmp_path / "volume"
    workspace = volume / "dev" / "project"
    workspace.mkdir(parents=True)
    runtime_dir = volume / "tmp" / "cuda-wheel" / "nvidia" / "cu13" / "lib"
    runtime_dir.mkdir(parents=True)
    runtime = runtime_dir / "libcudart.so.13"
    runtime.write_bytes(b"fixture")

    candidates = _cuda_runtime_candidates(
        NativeModelLoadConfig(),
        environ={},
        search_paths=(),
        anchors=(workspace,),
    )

    assert str(runtime) in candidates
    assert "libcudart.so.13" in candidates
    assert candidates.index(str(runtime)) < candidates.index("libcudart.so.13")
    assert _cuda_runtime_candidates(
        NativeModelLoadConfig(cuda_runtime_lib="/pinned/libcudart.so.13"),
        environ={"NFN_CUDA_RUNTIME_LIB": "/ignored/libcudart.so.13"},
        search_paths=(),
        anchors=(workspace,),
    ) == ("/pinned/libcudart.so.13",)


def test_model_tile_ops_resolver_prefers_installed_strict_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripts = tmp_path / "bin"
    scripts.mkdir()
    python = scripts / "python"
    python.write_bytes(b"fixture")
    strict = scripts / "libnfn_native_train_tile_ops_strict.so"
    strict.write_bytes(b"strict-fixture")
    monkeypatch.setattr(native_inference.sys, "executable", str(python))
    monkeypatch.delenv("NFN_NATIVE_TILE_OPS_LIB", raising=False)

    assert _resolve_model_tile_ops_library(NativeModelLoadConfig()) == str(strict.resolve())


def test_cpu_auto_selects_authenticated_primary_without_vram_probe(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)

    def must_not_probe(_config):
        raise AssertionError("CPU auto must not query CUDA")

    effective, proof = _select_checkpoint_variant(
        tmp_path,
        manifest,
        _Binding(cuda=False, profiles=("bf16-kernel", "k-kernel")),
        NativeModelLoadConfig(runtime="auto"),
        memory_probe=must_not_probe,
    )
    assert proof["effective_weight_precision"] == "bf16"
    assert proof["weight_precision_selection"] == "auto-primary"
    assert effective["checkpoint"]["artifact_path"] == "bf16.bin"


def test_cuda_auto_uses_quality_order_and_exact_byte_budget(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    config = NativeModelLoadConfig(
        runtime="native-cuda",
        vram_reserve_bytes=100,
        context_tokens=100,
        session_count=1,
    )
    # BF16 needs 900+10+20+30+200+40+100=1300. Dynamic needs 1000.
    effective, proof = _select_checkpoint_variant(
        tmp_path,
        manifest,
        _Binding(profiles=("bf16-kernel", "k-kernel")),
        config,
        memory_probe=lambda _config: (1_100, 2_000),
    )
    assert proof["effective_weight_precision"] == "k-quant-dynamic"
    assert proof["required_free_vram_bytes"] == 1_000
    assert proof["candidates"]["bf16"]["status"] == "insufficient-vram"
    assert effective["checkpoint"]["artifact_path"] == "k-quant-dynamic.bin"


def test_optional_dflash_never_lowers_target_quality_but_required_is_budgeted_first(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    _add_dflash(tmp_path, manifest, resident_bytes=100)
    base = dict(
        runtime="native-cuda",
        vram_reserve_bytes=0,
        context_tokens=100,
        session_count=1,
    )
    effective, optional = _select_checkpoint_variant(
        tmp_path,
        manifest,
        _Binding(profiles=("bf16-kernel", "k-kernel")),
        NativeModelLoadConfig(**base, speculative_decoding="auto"),
        memory_probe=lambda _config: (1_250, 2_000),
    )
    assert effective["checkpoint"]["artifact_path"] == "bf16.bin"
    assert optional["effective_speculative_decoding"] == "off"
    assert optional["speculative_decoding_selection"] == (
        "auto-skipped-to-preserve-target-precision"
    )

    effective, required = _select_checkpoint_variant(
        tmp_path,
        manifest,
        _Binding(profiles=("bf16-kernel", "k-kernel")),
        NativeModelLoadConfig(**base, speculative_decoding="required"),
        memory_probe=lambda _config: (1_250, 2_000),
    )
    assert effective["checkpoint"]["artifact_path"] == "k-quant-dynamic.bin"
    assert required["effective_speculative_decoding"] == "dflash"
    assert required["companion_checkpoints"] == ["dflash"]

    with pytest.raises(NativeInferenceCapabilityError, match="Explicit"):
        _select_checkpoint_variant(
            tmp_path,
            manifest,
            _Binding(profiles=("bf16-kernel", "k-kernel")),
            NativeModelLoadConfig(
                **base,
                weight_precision="bf16",
                speculative_decoding="required",
            ),
            memory_probe=lambda _config: (1_250, 2_000),
        )


def test_auto_skips_only_missing_or_unsupported_candidates(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    (tmp_path / "bf16.bin").unlink()
    _effective, proof = _select_checkpoint_variant(
        tmp_path,
        manifest,
        _Binding(profiles=("k-kernel",)),
        NativeModelLoadConfig(runtime="native-cuda", vram_reserve_bytes=0),
        memory_probe=lambda _config: (10_000, 10_000),
    )
    assert proof["effective_weight_precision"] == "k-quant-dynamic"
    assert proof["candidates"]["bf16"]["status"] == "missing-file"


def test_explicit_precision_never_downgrades(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    with pytest.raises(NativeInferenceCapabilityError, match="Explicit"):
        _select_checkpoint_variant(
            tmp_path,
            manifest,
            _Binding(profiles=("bf16-kernel", "k-kernel")),
            NativeModelLoadConfig(
                weight_precision="bf16",
                runtime="native-cuda",
                vram_reserve_bytes=100,
            ),
            memory_probe=lambda _config: (1_000, 2_000),
        )


def test_path_pin_and_conflicting_explicit_precision(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    (tmp_path / "native-execution-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    root, _path, loaded, pin = _load_manifest_with_path_pin(
        tmp_path / "k-quant-17gb.bin"
    )
    assert root == tmp_path.resolve()
    assert pin == "k-quant-17gb"
    with pytest.raises(NativeInferenceCapabilityError, match="conflicting"):
        _select_checkpoint_variant(
            root,
            loaded,
            _Binding(cuda=False, profiles=("bf16-kernel", "k-kernel")),
            NativeModelLoadConfig(weight_precision="bf16", runtime="cpu"),
            path_pin=pin,
        )


def test_manifest_variant_invariant_and_unknown_ids_fail_closed(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    manifest["checkpoint"]["target_sha256"] = "0" * 64
    with pytest.raises(NativeInferenceCapabilityError, match="primary checkpoint"):
        _select_checkpoint_variant(
            tmp_path,
            manifest,
            _Binding(cuda=False, profiles=("bf16-kernel",)),
            NativeModelLoadConfig(runtime="cpu"),
        )

    manifest = _manifest(tmp_path)
    manifest["checkpoint_variants"]["marketing-alias"] = manifest[
        "checkpoint_variants"
    ]["bf16"]
    with pytest.raises(NativeInferenceCapabilityError, match="Unsupported checkpoint variant"):
        _select_checkpoint_variant(
            tmp_path,
            manifest,
            _Binding(cuda=False, profiles=("bf16-kernel",)),
            NativeModelLoadConfig(runtime="cpu"),
        )


def test_cuda_probe_failure_and_no_fit_are_fail_closed(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)

    def broken_probe(_config):
        raise NativeInferenceCapabilityError("probe failed")

    with pytest.raises(NativeInferenceCapabilityError, match="probe failed"):
        _select_checkpoint_variant(
            tmp_path,
            manifest,
            _Binding(profiles=("bf16-kernel", "k-kernel")),
            NativeModelLoadConfig(runtime="native-cuda"),
            memory_probe=broken_probe,
        )
    with pytest.raises(NativeInferenceCapabilityError, match="No weight precision fits"):
        _select_checkpoint_variant(
            tmp_path,
            manifest,
            _Binding(profiles=("bf16-kernel", "k-kernel")),
            NativeModelLoadConfig(runtime="native-cuda", vram_reserve_bytes=0),
            memory_probe=lambda _config: (1, 1_000),
        )


def test_hybrid_kv_budget_caps_local_layers_and_scales_global_layers() -> None:
    descriptor = {
        "resident_weight_bytes": 100,
        "peak_load_staging_bytes": 10,
        "max_workspace_bytes": 20,
        "memory_profile": {
            "fixed_runtime_bytes": 30,
            "session_bytes": 40,
            "kv_cache_bytes_per_context_token_per_session": 0,
            "hybrid_kv_cache": {
                "local_layers": 3,
                "global_layers": 1,
                "local_window": 4,
                "kv_heads": 1,
                "head_dim": 2,
                "key_value_components": 2,
                "bytes_per_element": 2,
                "final_hidden_elements": 8,
            },
        },
    }
    # At context 6: (3*4 + 1*6) * 1*2*2*2 + 8*2 = 160 B/session.
    assert _variant_runtime_bytes(
        descriptor,
        context_tokens=6,
        session_count=2,
        companion_bytes=50,
    ) == 100 + 50 + 10 + 20 + 30 + 2 * (160 + 40)
