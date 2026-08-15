from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "qualify_muse_glimmer_gpu.py"
SPEC = importlib.util.spec_from_file_location("qualify_muse_glimmer_gpu", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
qualification = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = qualification
SPEC.loader.exec_module(qualification)


class _FakeProbe:
    def __init__(self, _runtime: str, device: int) -> None:
        self.device = device
        self.total_bytes = 24 * qualification.GIB
        self.baseline_free_bytes = 23 * qualification.GIB
        self.samples: list[str] = []

    def checkpoint(self, label: str) -> None:
        self.samples.append(label)

    def hardware(self) -> dict[str, object]:
        return {
            "cuda_runtime_lib": "/real/libcudart.so",
            "cuda_device": self.device,
            "cuda_device_count": 1,
            "compute_capability": "8.9",
            "cuda_runtime_version": 13000,
            "cuda_driver_version": 13000,
            "total_bytes": self.total_bytes,
            "baseline_free_bytes": self.baseline_free_bytes,
        }

    def memory_result(self) -> dict[str, object]:
        return {
            "provider": "cudaMemGetInfo",
            "scope": "sampled-device-global-baseline-subtracted",
            "peak_sampled_delta_bytes": 18 * qualification.GIB,
            "minimum_free_bytes": 5 * qualification.GIB,
            "samples": list(self.samples),
        }


class _FakeSession:
    def __init__(self) -> None:
        self.tokens: list[int] = []

    def __enter__(self) -> "_FakeSession":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def prefill(self, tokens: tuple[int, ...]) -> None:
        self.tokens.extend(tokens)

    def decode(self, generation: object) -> object:
        count = int(getattr(generation, "max_new_tokens"))
        self.tokens.extend([2] * count)
        return SimpleNamespace(
            token_ids=tuple([2] * count),
            speculative_proposed_tokens=count * 2,
            speculative_accepted_tokens=count,
            speculative_rejected_tokens=count,
            speculative_target_rows=count + 1,
            speculative_assistant_blocks=1,
        )

    def stats(self) -> dict[str, object]:
        return {
            "effective_cache": "full",
            "token_count": len(self.tokens),
            "cache_bytes": 4096,
            "cache_capacity_bytes": 8192,
        }


class _FakeModel:
    def __init__(self, config: object) -> None:
        self.config = config
        self.computed = False

    def __enter__(self) -> "_FakeModel":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def create_session(self, *, seed: int) -> _FakeSession:
        assert isinstance(seed, int)
        self.computed = True
        return _FakeSession()

    def encode_media(
        self,
        patches: tuple[tuple[float, ...], ...],
        grid: tuple[tuple[int, int, int], ...],
    ) -> tuple[tuple[float, ...], ...]:
        assert patches and grid == ((1, 2, 2),)
        self.computed = True
        return (tuple(0.25 for _ in range(6_656)),)

    def stats(self) -> dict[str, object]:
        request = getattr(self.config, "weight_precision")
        assert request in {"auto", "k-quant-17gb"}
        profile = "k-quant-17gb"
        return {
            **qualification.FULL_GEOMETRY,
            "parameter_count": qualification.FULL_TEXT_PARAMETER_COUNT,
            "resident_weight_bytes": 16_756_683_904,
            "whole_model_cuda": True,
            "cuda_model_compute_only": True,
            "cpu_model_compute_rows": 0,
            "requested_weight_precision": request,
            "effective_weight_precision": profile,
            "weight_precision_selection": (
                "auto-vram" if request == "auto" else "explicit"
            ),
            "selected_artifact_sha256": qualification.CANONICAL_KQUANT_SHA256[profile],
            "cuda_resident_weight_bytes": 16_000_000_000,
            "cuda_workspace_bytes": 1024,
            "cuda_kernel_launches": 8 if self.computed else 0,
            "cuda_device": 0,
            "cuda_tile_ops_lib": str(Path("/tmp/strict-tile.so").resolve()),
            "dflash_loaded": True,
            "dflash_cuda": True,
            "dflash_cuda_resident_weight_bytes": 1_500_000_000,
            "dflash_cuda_workspace_bytes": 1024,
            "dflash_cuda_kernel_launches": 4 if self.computed else 0,
            "vision_loaded": True,
            "vision_cuda": True,
            "vision_resident_weight_bytes": 1_400_000_000,
            "effective_speculative_decoding": "dflash",
        }


class _FakeModelLoader:
    @staticmethod
    def load(
        _artifact: Path,
        *,
        binding: object,
        kv_cache: object,
        load_config: object,
    ) -> _FakeModel:
        assert binding is not None and getattr(kv_cache, "mode") == "full"
        return _FakeModel(load_config)


def _artifact(tmp_path: Path) -> Path:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "native-execution-manifest.json").write_text(
        json.dumps(
            {
                "schema": "neuralfn.native_execution_manifest",
                "version": 1,
                "source_graph": {
                    "repository": "meta-models/Muse-Glimmer-30B-GGUF",
                    "revision": qualification.GGUF_ARTIFACT_REVISION,
                },
                "model": {"family": "muse_glimmer"},
                "checkpoint_variants": {
                    "k-quant-17gb": {
                        "target_sha256": qualification.CANONICAL_KQUANT_SHA256[
                            "k-quant-17gb"
                        ],
                        "target_nbytes": 16_756_683_904,
                    }
                },
                "companion_checkpoints": {
                    "dflash": {"capabilities": {"resident_cuda": True}},
                    "mmproj": {"capabilities": {"resident_cuda": True}},
                },
            }
        ),
        encoding="utf-8",
    )
    return artifact


def test_real_gpu_worker_contract_rejects_fallback_and_records_all_paths(
    tmp_path: Path,
) -> None:
    source_sha = qualification._source_tree_proof()[0]
    config = qualification.RunConfig(
        artifact=str(_artifact(tmp_path)),
        profile="k-quant-17gb",
        gpu_class="24",
        cuda_runtime_lib="/real/libcudart.so",
        cuda_device=0,
        binding_lib="/tmp/binding.so",
        tile_ops_lib="/tmp/strict-tile.so",
        contexts=(2, 8),
        decode_tokens=2,
        warmups=1,
        repetitions=2,
        prompt_token_id=1,
        stop_token_id=12,
        companions=("dflash", "mmproj"),
        require_dflash=True,
        run_vision=True,
        vision_patch_width=588,
        vision_grid=(1, 2, 2),
        source_tree_sha256=source_sha,
        progress_path=str(tmp_path / "worker-progress.json"),
    )
    payload = qualification._run_worker(
        config,
        model_loader=_FakeModelLoader,
        binding_loader=lambda _path: object(),
        probe_factory=_FakeProbe,
    )
    assert payload["status"] == "worker-complete"
    assert payload["model_stats"]["cpu_model_compute_rows"] == 0
    assert payload["model_stats"]["dflash_cuda"] is True
    assert payload["model_stats"]["vision_cuda"] is True
    assert payload["contexts"][1]["summary"]["dflash"]["proposed_tokens"] > 0
    assert payload["vision"]["output_width"] == 6_656
    assert payload["memory"]["peak_sampled_delta_bytes"] > 0
    progress = json.loads((tmp_path / "worker-progress.json").read_text(encoding="utf-8"))
    assert progress["stage"] == "worker_complete"
    assert progress["source_tree_sha256"] == source_sha

    bad_stats = _FakeModel(SimpleNamespace(weight_precision="auto")).stats()
    bad_stats["cpu_model_compute_rows"] = 1
    with pytest.raises(qualification.QualificationError, match="CPU model-compute"):
        qualification._validate_full_size_stats(bad_stats, config, after_compute=False)


def _qualified_result(gpu_class: str, source_sha: str) -> dict[str, object]:
    profile = qualification.PROFILE_BY_CLASS[gpu_class]
    selected_sha = qualification.CANONICAL_KQUANT_SHA256.get(profile, "f" * 64)
    # The 24-GB profile is deliberately qualified on the same larger physical
    # GPU as the 32-GB profile. Device capacity is a minimum, not an identity.
    total = {"24": 32, "32": 32, "80": 80}[gpu_class] * qualification.GIB
    request = qualification._weight_precision_request_for_tier(total, gpu_class)
    resident = qualification.MINIMUM_CUDA_WEIGHT_BYTES[profile]
    return {
        "schema": qualification.SCHEMA,
        "version": qualification.VERSION,
        "status": "qualified",
        "release_qualified": True,
        "profile": profile,
        "gpu_class": gpu_class,
        "profile_tier": {
            "minimum_total_vram_bytes": qualification.GPU_CLASS_MINIMUM_BYTES[
                gpu_class
            ],
            "physical_total_vram_bytes": total,
            "larger_device": request != "auto",
        },
        "source_tree_sha256": source_sha,
        "hardware": {"total_bytes": total},
        "build": {
            "source_tree_sha256": source_sha,
            "strict_inference_tile_ops": {"sha256": "c" * 64},
        },
        "compute_sanitizer": {
            "status": "passed",
            "zero_error_summary": True,
            "scope": "real-device-kernel-probe-before-full-artifact-benchmark",
            "source_tree_sha256": source_sha,
            "benchmark_artifact_sha256": selected_sha,
            "kernels": list(qualification.KERNEL_PROBE_NAMES),
            "probe": {"strict_tile_sha256": "c" * 64},
            "tools": {
                tool: {"status": "passed", "zero_error_summary": True}
                for tool in qualification.SANITIZER_TOOLS
            },
        },
        "artifact": {"selected_artifact_sha256": selected_sha},
        "load": {
            "requested_weight_precision": request,
            "effective_weight_precision": profile,
            "weight_precision_selection": (
                "auto-vram" if request == "auto" else "explicit"
            ),
            "speculative_decoding": "dflash",
        },
        "model_stats": {
            **qualification.FULL_GEOMETRY,
            "cuda_resident_weight_bytes": resident,
            "whole_model_cuda": True,
            "vision_cuda": True,
            "dflash_loaded": True,
            "dflash_cuda": True,
            "cpu_model_compute_rows": 0,
        },
        "contexts": [{"prompt_context_tokens": 8192}],
        "vision": {"output_width": 6656},
        "memory": {"peak_sampled_delta_bytes": resident},
    }


def test_matrix_verifier_requires_all_classes_same_source_and_sanitizer(
    tmp_path: Path,
) -> None:
    source_sha = "a" * 64
    paths: list[Path] = []
    for gpu_class in ("24", "32", "80"):
        path = tmp_path / f"{gpu_class}.json"
        path.write_text(
            json.dumps(_qualified_result(gpu_class, source_sha)), encoding="utf-8"
        )
        paths.append(path)
    result = qualification._verify_matrix(
        SimpleNamespace(result=paths, minimum_long_context=8192)
    )
    assert result["status"] == "qualified"
    assert set(result["results"]) == {"24", "32", "80"}

    corrupted = deepcopy(_qualified_result("32", source_sha))
    corrupted["compute_sanitizer"] = {"status": "skipped"}
    paths[1].write_text(json.dumps(corrupted), encoding="utf-8")
    with pytest.raises(qualification.QualificationError, match="sanitizer"):
        qualification._verify_matrix(
            SimpleNamespace(result=paths, minimum_long_context=8192)
        )


def test_nvcc_build_contract_keeps_strict_architecture_generic() -> None:
    script = (ROOT / "tools/build_native_train_tile_ops.sh").read_text(
        encoding="utf-8"
    )
    assert 'STRICT_CUDA_ARCH="${NFN_TILE_CUDA_STRICT_ARCH:-${CUDA_ARCH%a}}"' in script
    assert script.count("-enable-tile") == 2


def test_vision_grid_allows_equal_height_and_width() -> None:
    assert qualification._parse_csv_ints(
        "1,2,2", "--vision-grid", unique=False
    ) == (1, 2, 2)
    with pytest.raises(qualification.QualificationError, match="duplicates"):
        qualification._parse_csv_ints("128,128", "--contexts")


def test_larger_gpu_can_qualify_lower_memory_profile() -> None:
    physical_total = 32 * qualification.GIB
    qualification._validate_gpu_class(physical_total, "24")
    assert qualification._weight_precision_request_for_tier(
        physical_total, "24"
    ) == "k-quant-17gb"
    assert qualification._weight_precision_request_for_tier(
        physical_total, "32"
    ) == "auto"
    with pytest.raises(qualification.QualificationError, match="at least"):
        qualification._validate_gpu_class(23_999_999_999, "24")


def test_full_size_gate_uses_exact_pinned_text_parameter_count() -> None:
    config = SimpleNamespace(
        profile="k-quant-17gb",
        cuda_device=0,
        require_dflash=False,
        run_vision=False,
        tile_ops_lib="/tmp/strict-tile.so",
        weight_precision_request="auto",
    )
    stats = _FakeModel(SimpleNamespace(weight_precision="auto")).stats()
    qualification._validate_full_size_stats(stats, config, after_compute=False)
    stats["parameter_count"] += 1
    with pytest.raises(qualification.QualificationError, match="parameter count mismatch"):
        qualification._validate_full_size_stats(stats, config, after_compute=False)


def test_parent_pairs_sanitized_kernel_probe_with_full_timing_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_sha = "a" * 64
    artifact_sha = qualification.CANONICAL_KQUANT_SHA256["k-quant-17gb"]
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    config = qualification.RunConfig(
        artifact=str(artifact),
        profile="k-quant-17gb",
        gpu_class="24",
        cuda_runtime_lib="/real/libcudart.so",
        cuda_device=0,
        binding_lib="/tmp/binding.so",
        tile_ops_lib="/tmp/strict.so",
        contexts=(128, 8192),
        decode_tokens=16,
        warmups=1,
        repetitions=3,
        prompt_token_id=200000,
        stop_token_id=200018,
        companions=("dflash", "mmproj"),
        require_dflash=True,
        run_vision=True,
        vision_patch_width=588,
        vision_grid=(1, 2, 2),
        source_tree_sha256=source_sha,
    )
    build = {
        "source_tree_sha256": source_sha,
        "strict_inference_tile_ops": {"sha256": "c" * 64},
    }
    monkeypatch.setattr(qualification, "_validate_run_args", lambda _args: None)
    monkeypatch.setattr(
        qualification,
        "CudaProbe",
        lambda _runtime, device: _FakeProbe(_runtime, device),
    )
    monkeypatch.setattr(
        qualification,
        "_prepare_build",
        lambda _args, _probe: (config, build),
    )
    sanitizer = tmp_path / "compute-sanitizer"
    sanitizer.write_text("fixture", encoding="utf-8")
    monkeypatch.setattr(
        qualification,
        "_resolve_executable",
        lambda _requested, _default: sanitizer,
    )
    probe = {
        "path": str(tmp_path / "probe"),
        "sha256": "d" * 64,
        "strict_tile_path": config.tile_ops_lib,
        "strict_tile_sha256": "c" * 64,
    }
    monkeypatch.setattr(
        qualification,
        "_build_cuda_kernel_probe",
        lambda _build, _tile, _runtime, _directory: probe,
    )
    sanitizer_calls: list[tuple[dict[str, object], Path, int]] = []

    def fake_sanitizers(probe_arg, sanitizer_arg, *, cuda_device, environment):
        assert environment
        sanitizer_calls.append((dict(probe_arg), sanitizer_arg, cuda_device))
        return {
            "mode": "required",
            "status": "passed",
            "scope": "real-device-kernel-probe-before-full-artifact-benchmark",
            "path": str(sanitizer_arg),
            "cuda_device": cuda_device,
            "probe": dict(probe_arg),
            "kernels": list(qualification.KERNEL_PROBE_NAMES),
            "tools": {
                tool: {"status": "passed", "zero_error_summary": True}
                for tool in qualification.SANITIZER_TOOLS
            },
            "zero_error_summary": True,
        }

    monkeypatch.setattr(
        qualification, "_run_kernel_probe_sanitizers", fake_sanitizers
    )
    worker_configs: list[qualification.RunConfig] = []

    def fake_worker(config_arg, _config_path, *, environment, sanitizer_path=None):
        assert environment and sanitizer_path is None
        worker_configs.append(config_arg)
        payload = {
            "schema": qualification.SCHEMA,
            "version": qualification.VERSION,
            "status": "worker-complete",
            "profile": "k-quant-17gb",
            "gpu_class": "24",
            "artifact": {
                "manifest_sha256": "b" * 64,
                "selected_artifact_sha256": artifact_sha,
            },
            "model_stats": {
                **qualification.FULL_GEOMETRY,
                "whole_model_cuda": True,
                "cuda_model_compute_only": True,
                "dflash_cuda": True,
                "vision_cuda": True,
                "cpu_model_compute_rows": 0,
            },
            "contexts": [
                {
                    "prompt_context_tokens": context,
                    "decode_tokens": config_arg.decode_tokens,
                }
                for context in config_arg.contexts
            ],
            "source_tree_sha256": source_sha,
        }
        return qualification.subprocess.CompletedProcess(
            ["worker"],
            0,
            stdout=json.dumps(payload) + "\n",
            stderr="",
        )

    monkeypatch.setattr(qualification, "_invoke_worker", fake_worker)
    args = SimpleNamespace(
        artifact=artifact,
        profile="k-quant-17gb",
        gpu_class="24",
        cuda_runtime_lib="/real/libcudart.so",
        cuda_device=0,
        build_dir=tmp_path,
        compute_sanitizer="required",
        compute_sanitizer_bin=str(sanitizer),
    )
    result = qualification._run_parent(args)
    assert sanitizer_calls == [(probe, sanitizer, 0)]
    assert len(worker_configs) == 1
    assert worker_configs[0].contexts == (128, 8192)
    assert worker_configs[0].decode_tokens == 16
    assert worker_configs[0].warmups == 1
    assert worker_configs[0].repetitions == 3
    assert result["contexts"] == [
        {"prompt_context_tokens": 128, "decode_tokens": 16},
        {"prompt_context_tokens": 8192, "decode_tokens": 16},
    ]
    assert result["compute_sanitizer"]["tools"]["memcheck"]["status"] == "passed"
    assert result["compute_sanitizer"]["benchmark_artifact_sha256"] == artifact_sha
    assert result["compute_sanitizer"]["source_tree_sha256"] == source_sha
    assert result["release_qualified"] is True


def test_compiler_only_build_contract_is_explicitly_not_release_qualified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert qualification._validated_cuda_arch("sm_80") == "sm_80"
    assert qualification._validated_cuda_arch("sm_120") == "sm_120"
    for invalid in ("sm_75", "compute_90", "native", "sm_90a"):
        with pytest.raises(qualification.QualificationError, match="sm_80-or-newer"):
            qualification._validated_cuda_arch(invalid)

    build = {
        "cuda_arch": "sm_90",
        "source_tree_sha256": "a" * 64,
        "abi_versions": {
            "strict_math": 1,
            "packed_weight": 1,
            "glimmer_inference": 1,
            "glimmer_vision": 1,
            "glimmer_training": 1,
        },
    }
    monkeypatch.setattr(
        qualification,
        "_build_native_binaries",
        lambda _args, arch: (dict(build, cuda_arch=arch), Path("binding"), Path("strict")),
    )
    payload = qualification._run_build_only(
        SimpleNamespace(cuda_arch="sm_90", nvcc=None, build_dir=Path("build"))
    )
    assert payload["status"] == "source-built"
    assert payload["release_qualified"] is False
    assert payload["build"]["cuda_arch"] == "sm_90"
