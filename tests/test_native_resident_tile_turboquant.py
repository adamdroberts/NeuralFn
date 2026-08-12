from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sysconfig
from types import ModuleType

import pytest

from neuralfn.native_inference import (
    GenerationConfig,
    KVCacheConfig,
    NativeInferenceModel,
    _turboquant_binding_tables,
)
from tests.test_native_resident_binding import _manifest, _write_tiny_dense_v5


ROOT = Path(__file__).resolve().parents[1]
LIVE_CUDA = os.environ.get("NFN_NATIVE_TURBOQUANT_CUDA_TEST") == "1"


@pytest.fixture(scope="session")
def resident_tile_binding(tmp_path_factory: pytest.TempPathFactory) -> ModuleType:
    output = tmp_path_factory.mktemp("native-resident-tile-binding") / (
        "_native_inference" + (sysconfig.get_config_var("EXT_SUFFIX") or ".so")
    )
    subprocess.run(
        ["bash", str(ROOT / "tools" / "build_native_inference_binding.sh"), str(output)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    spec = importlib.util.spec_from_file_location("_native_inference", output)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_tiny_model(
    binding: ModuleType,
    root: Path,
    *,
    max_seq_len: int = 8,
    num_heads: int = 1,
    channels: int = 8,
):
    checkpoint = root / "model.bin"
    _write_tiny_dense_v5(
        checkpoint,
        nontrivial=True,
        max_seq_len=max_seq_len,
        vocab_size=8,
        num_layers=1,
        num_heads=num_heads,
        channels=channels,
    )
    return binding.load_model(str(root), _manifest(checkpoint, turboquant=True))


def _fake_tile_sidecar(
    root: Path,
    *,
    base_abi: int = 1,
    strict_abi: int = 1,
    feature_abi: int = 1,
    include_forward: bool = True,
) -> Path:
    source = root / "fake_tile_ops.c"
    output = root / "libfake_tile_ops.so"
    forward = (
        "int nfn_native_tile_turboquant_attention_forward_v1(const void *descriptor) "
        "{ (void)descriptor; return 0; }"
        if include_forward
        else ""
    )
    source.write_text(
        f"""
#include <stdint.h>
int nfn_native_tile_ops_abi_version(void) {{ return {base_abi}; }}
int nfn_native_tile_strict_math_abi_version(void) {{ return {strict_abi}; }}
int nfn_native_tile_turboquant_attention_abi_version(void) {{ return {feature_abi}; }}
{forward}
const char *nfn_native_tile_ops_error_string(int code) {{ (void)code; return "fake"; }}
void nfn_native_tile_turboquant_attention_stats_reset(void) {{}}
int64_t nfn_native_tile_turboquant_attention_launch_count(void) {{ return 0; }}
""",
        encoding="utf-8",
    )
    subprocess.run(
        ["cc", "-std=c11", "-shared", "-fPIC", str(source), "-o", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    return output.resolve()


def _configure(binding: ModuleType, model, tile_ops_lib: Path, **extra):
    return binding.configure_model_turboquant_attention(
        model,
        {
            "backend": "tile-cuda",
            "tile_ops_lib": str(tile_ops_lib),
            "device": 0,
            **extra,
        },
    )


def _cache_payload(profile: str, backend: str) -> dict:
    return {
        "effective_mode": "turboquant",
        "turboquant_profile": profile,
        "turboquant_attention_backend": backend,
        "tables": _turboquant_binding_tables(
            channels=8,
            num_heads=1,
            profile=profile,
        ),
    }


def test_binding_exposes_additive_tile_capability_without_changing_cpu_backend(
    resident_tile_binding: ModuleType,
) -> None:
    assert resident_tile_binding.resident_inference_abi_version() == 1
    capabilities = resident_tile_binding.resident_inference_capabilities()
    assert capabilities["backend"] == "cpu-reference-resident"
    assert capabilities["turboquant_tile_attention"] is True
    assert callable(resident_tile_binding.configure_model_turboquant_attention)


def test_missing_tile_sidecar_fails_closed_before_session_creation(
    resident_tile_binding: ModuleType,
    tmp_path: Path,
) -> None:
    model = _load_tiny_model(resident_tile_binding, tmp_path)
    try:
        with pytest.raises(RuntimeError, match="not a readable regular file"):
            _configure(
                resident_tile_binding,
                model,
                (tmp_path / "missing-tile-ops.so").resolve(),
            )
        stats = resident_tile_binding.model_stats(model)
        assert stats["turboquant_tile_attention_configured"] is False
        assert stats["turboquant_attention_backend"] == "cpu"
    finally:
        resident_tile_binding.close_model(model)


@pytest.mark.parametrize(
    ("sidecar_options", "message"),
    [
        ({"base_abi": 0}, "base Tile ops ABI version 1"),
        ({"strict_abi": 0}, "strict-math ABI version 1"),
        ({"feature_abi": 0}, "attention feature ABI version 1"),
        ({"include_forward": False}, "missing required symbol.*forward_v1"),
    ],
)
def test_stale_or_incomplete_tile_sidecar_fails_closed_without_cuda(
    resident_tile_binding: ModuleType,
    tmp_path: Path,
    sidecar_options: dict,
    message: str,
) -> None:
    model = _load_tiny_model(resident_tile_binding, tmp_path)
    sidecar = _fake_tile_sidecar(tmp_path, **sidecar_options)
    try:
        with pytest.raises(RuntimeError, match=message):
            _configure(
                resident_tile_binding,
                model,
                sidecar,
                cuda_runtime_lib=None,
            )
        assert resident_tile_binding.model_stats(model)[
            "turboquant_tile_attention_configured"
        ] is False
    finally:
        resident_tile_binding.close_model(model)


def test_tile_session_request_requires_successful_model_configuration(
    resident_tile_binding: ModuleType,
    tmp_path: Path,
) -> None:
    model = _load_tiny_model(resident_tile_binding, tmp_path)
    try:
        with pytest.raises(RuntimeError, match="before model configuration"):
            resident_tile_binding.create_session(
                model,
                {"seed": 0, "kv_cache": _cache_payload("mse-3.5", "tile-cuda")},
            )

        cpu_payload = _cache_payload("mse-3.5", "cpu")
        cpu_payload.pop("turboquant_attention_backend")
        session = resident_tile_binding.create_session(
            model,
            {"seed": 0, "kv_cache": cpu_payload},
        )
        try:
            resident_tile_binding.prefill(model, session, [1, 2, 3], 0)
            stats = resident_tile_binding.session_stats(model, session)
            assert stats["turboquant_attention_backend"] == "cpu"
            assert stats["turboquant_gpu_launches"] == 0
            assert stats["turboquant_row_uploads"] == 0
            assert stats["turboquant_cpu_compressed_attention_calls"] > 0
        finally:
            resident_tile_binding.close_session(model, session)
    finally:
        resident_tile_binding.close_model(model)


def test_tile_geometry_limits_fail_before_loading_sidecar(
    resident_tile_binding: ModuleType,
    tmp_path: Path,
) -> None:
    context_root = tmp_path / "context"
    context_root.mkdir()
    context_model = _load_tiny_model(
        resident_tile_binding,
        context_root,
        max_seq_len=16385,
    )
    try:
        with pytest.raises(RuntimeError, match="no greater than 16384"):
            _configure(
                resident_tile_binding,
                context_model,
                (tmp_path / "missing.so").resolve(),
            )
    finally:
        resident_tile_binding.close_model(context_model)

    head_root = tmp_path / "head"
    head_root.mkdir()
    head_model = _load_tiny_model(
        resident_tile_binding,
        head_root,
        channels=258,
    )
    try:
        with pytest.raises(RuntimeError, match="head dimension in 2..256"):
            _configure(
                resident_tile_binding,
                head_model,
                (tmp_path / "missing.so").resolve(),
            )
    finally:
        resident_tile_binding.close_model(head_model)


@pytest.fixture(scope="session")
def strict_tile_sidecar(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not LIVE_CUDA:
        pytest.skip(
            "set NFN_NATIVE_TURBOQUANT_CUDA_TEST=1 for live resident Tile-CUDA coverage"
        )
    prebuilt = os.environ.get("NFN_NATIVE_TURBOQUANT_TILE_OPS_LIB")
    if prebuilt:
        resolved = Path(prebuilt).expanduser().resolve()
        if not resolved.is_file():
            pytest.fail(
                "NFN_NATIVE_TURBOQUANT_TILE_OPS_LIB is not a regular file: "
                f"{resolved}"
            )
        return resolved
    output_dir = tmp_path_factory.mktemp("native-resident-tile-sidecar")
    default_path = output_dir / "libnfn_native_train_tile_ops.so"
    strict_path = output_dir / "libnfn_native_train_tile_ops_strict.so"
    environment = os.environ.copy()
    environment["NFN_NATIVE_BUILD_STRICT_TILE_OPS"] = "1"
    subprocess.run(
        ["bash", str(ROOT / "tools" / "build_native_train_tile_ops.sh"), str(default_path)],
        cwd=ROOT,
        env=environment,
        check=True,
        text=True,
    )
    assert strict_path.is_file()
    return strict_path.resolve()


@pytest.mark.parametrize("profile", ["mse-3.5", "qjl-3.5"])
def test_live_tile_attention_matches_cpu_and_preserves_session_lifecycle(
    resident_tile_binding: ModuleType,
    strict_tile_sidecar: Path,
    tmp_path: Path,
    profile: str,
) -> None:
    model = _load_tiny_model(resident_tile_binding, tmp_path)
    configured = _configure(resident_tile_binding, model, strict_tile_sidecar)
    assert configured["configured"] is True
    assert configured["backend"] == "tile-cuda"
    assert Path(configured["tile_ops_lib"]) == strict_tile_sidecar
    assert configured["device"] == 0

    cpu = resident_tile_binding.create_session(
        model,
        {"seed": 7, "kv_cache": _cache_payload(profile, "cpu")},
    )
    tile = resident_tile_binding.create_session(
        model,
        {"seed": 7, "kv_cache": _cache_payload(profile, "tile-cuda")},
    )
    repeat = resident_tile_binding.create_session(
        model,
        {"seed": 7, "kv_cache": _cache_payload(profile, "tile-cuda")},
    )
    strict_decode = {
        "temperature": 0.0,
        "top_k": None,
        "top_p": 1.0,
        "seed": 7,
        "stop_token_ids": [],
        "strict_model_compute": True,
    }
    try:
        sessions = (cpu, tile, repeat)
        prompt = [1, 2, 3]
        for session in sessions:
            resident_tile_binding.prefill(model, session, prompt, 0)
        assert resident_tile_binding.current_logits(model, tile) == pytest.approx(
            resident_tile_binding.current_logits(model, cpu), abs=1.0e-4, rel=1.0e-4
        )
        assert resident_tile_binding.current_logits(
            model, repeat
        ) == resident_tile_binding.current_logits(model, tile)

        tile_before_rejected_fork = resident_tile_binding.session_stats(model, tile)
        open_sessions_before_rejected_fork = resident_tile_binding.model_stats(model)[
            "open_sessions"
        ]
        with pytest.raises(RuntimeError, match="configured for Tile-CUDA"):
            resident_tile_binding.fork_session(
                model,
                tile,
                {"token_count": len(prompt), "seed": 17},
            )
        tile_after_rejected_fork = resident_tile_binding.session_stats(model, tile)
        assert resident_tile_binding.model_stats(model)["open_sessions"] == (
            open_sessions_before_rejected_fork
        )
        for key in (
            "turboquant_gpu_launches",
            "turboquant_row_uploads",
            "turboquant_h2d_bytes",
            "turboquant_d2h_bytes",
            "prefix_cow_detach_count",
            "prefix_cow_detached_capacity_bytes",
        ):
            assert tile_after_rejected_fork[key] == tile_before_rejected_fork[key]

        for session in sessions:
            resident_tile_binding.truncate_session(model, session, 1)
            resident_tile_binding.prefill(model, session, prompt[1:], 1)
        assert resident_tile_binding.current_logits(model, tile) == pytest.approx(
            resident_tile_binding.current_logits(model, cpu), abs=1.0e-4, rel=1.0e-4
        )
        assert resident_tile_binding.current_logits(
            model, repeat
        ) == resident_tile_binding.current_logits(model, tile)

        for session in sessions:
            resident_tile_binding.reset_session(model, session)
            resident_tile_binding.prefill(model, session, prompt, 0)
        cpu_decode = resident_tile_binding.decode_one(model, cpu, strict_decode)
        tile_decode = resident_tile_binding.decode_one(model, tile, strict_decode)
        repeat_decode = resident_tile_binding.decode_one(model, repeat, strict_decode)
        assert tile_decode["token_id"] == cpu_decode["token_id"]
        assert tile_decode["selected_logit"] == pytest.approx(
            cpu_decode["selected_logit"], abs=1.0e-4, rel=1.0e-4
        )
        assert repeat_decode == tile_decode
        assert resident_tile_binding.current_logits(
            model, repeat
        ) == resident_tile_binding.current_logits(model, tile)

        cpu_stats = resident_tile_binding.session_stats(model, cpu)
        tile_stats = resident_tile_binding.session_stats(model, tile)
        assert cpu_stats["turboquant_attention_backend"] == "cpu"
        assert cpu_stats["turboquant_gpu_launches"] == 0
        assert cpu_stats["turboquant_cpu_compressed_attention_calls"] > 0
        assert tile_stats["turboquant_attention_backend"] == "tile-cuda"
        assert Path(tile_stats["turboquant_tile_ops_lib"]) == strict_tile_sidecar
        assert tile_stats["turboquant_cuda_device"] == 0
        assert tile_stats["turboquant_gpu_launches"] == 9
        assert tile_stats["turboquant_row_uploads"] == 9
        assert tile_stats["turboquant_h2d_bytes"] > 0
        assert tile_stats["turboquant_d2h_bytes"] > 0
        assert tile_stats["turboquant_cpu_compressed_attention_calls"] == 0
    finally:
        resident_tile_binding.close_session(model, cpu)
        resident_tile_binding.close_session(model, tile)
        resident_tile_binding.close_session(model, repeat)
        resident_tile_binding.close_model(model)


@pytest.mark.parametrize("profile", ["mse-3.5", "qjl-3.5"])
def test_live_public_sdk_uses_tile_attention_without_cpu_compressed_calls(
    resident_tile_binding: ModuleType,
    strict_tile_sidecar: Path,
    tmp_path: Path,
    profile: str,
) -> None:
    checkpoint = tmp_path / "model.bin"
    _write_tiny_dense_v5(
        checkpoint,
        nontrivial=True,
        max_seq_len=8,
        vocab_size=8,
        num_layers=1,
        num_heads=1,
        channels=8,
    )
    manifest = _manifest(checkpoint, turboquant=True)
    manifest["kernel_abi"]["turboquant_tile_attention"] = {
        "symbol": "nfn_native_tile_turboquant_attention_forward_v1",
        "version": 1,
        "status": "ready",
        "backend": "tile-cuda-hybrid",
    }
    manifest["capabilities"]["turboquant_tile_attention"] = True
    (tmp_path / "native-execution-manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    cache = KVCacheConfig(
        mode="turboquant",
        turboquant_profile=profile,
        turboquant_attention_backend="tile-cuda",
        tile_ops_lib=str(strict_tile_sidecar),
    )

    with NativeInferenceModel.load(
        tmp_path,
        binding=resident_tile_binding,
        kv_cache=cache,
    ) as model:
        assert model.capabilities.turboquant_tile_attention is True
        with model.create_session(seed=11) as session:
            session.prefill([1, 2, 3])
            session.truncate(1)
            session.prefill([1, 2, 3])
            session.reset()
            session.prefill([1, 2, 3])
            result = session.decode(
                GenerationConfig(
                    max_new_tokens=1,
                    temperature=0.0,
                    top_k=None,
                    top_p=1.0,
                    seed=11,
                )
            )
            assert len(result.events) == 1
            stats = session.stats()
            assert stats["turboquant_attention_backend"] == "tile-cuda"
            assert stats["turboquant_gpu_launches"] > 0
            assert stats["turboquant_row_uploads"] > 0
            assert stats["turboquant_h2d_bytes"] > 0
            assert stats["turboquant_d2h_bytes"] > 0
            assert stats["turboquant_cpu_compressed_attention_calls"] == 0
