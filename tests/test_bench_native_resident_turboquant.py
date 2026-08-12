from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "bench_native_resident_turboquant.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("bench_native_resident_turboquant_test", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench = _load_tool()


def _paths(tmp_path: Path, *, tile: bool = False) -> tuple[Path, Path, Path | None]:
    artifact = tmp_path / "artifact"
    artifact.mkdir(parents=True)
    (artifact / "native-execution-manifest.json").write_text("{}", encoding="utf-8")
    binding = tmp_path / "_native_inference.so"
    binding.write_bytes(b"binding-fixture")
    tile_lib = None
    if tile:
        tile_lib = tmp_path / "libtile.so"
        tile_lib.write_bytes(b"tile-fixture")
    return artifact, binding, tile_lib


def _config(
    tmp_path: Path,
    *,
    mode: str | None = None,
    tile: bool = False,
    runtime: str | None = None,
) -> Any:
    artifact, binding, tile_lib = _paths(tmp_path, tile=tile)
    return bench.Config(
        artifact=artifact,
        binding_lib=binding,
        tile_ops_lib=tile_lib,
        cuda_runtime_lib=runtime,
        cuda_device=0,
        contexts=(6,),
        decode_tokens=2,
        quality_window=2,
        warmups=1,
        repetitions=2,
        tokens=(0, 1, 2, 3, 0, 1),
        json_out=None,
        worker=mode is not None,
        worker_mode=mode,
        worker_pass=None,
    )


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 0.01
        return self.value


class FakeSession:
    def __init__(self, model: "FakeModel") -> None:
        self.model = model
        self.tokens: list[int] = []
        self.truncations: list[int] = []

    def __enter__(self) -> "FakeSession":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        return None

    def prefill(self, token_ids) -> dict[str, int]:
        self.tokens = list(token_ids)
        self.model.prefill_calls += 1
        return {
            "prefix_tokens": len(self.tokens),
            "prefix_reused": 0,
            "prefilled_tokens": len(self.tokens),
        }

    def truncate(self, token_count: int) -> None:
        del self.tokens[token_count:]
        self.truncations.append(token_count)
        self.model.truncations.append(token_count)

    def current_logits(self) -> tuple[float, ...]:
        assert self.tokens
        self.model.current_logits_calls += 1
        target = (self.tokens[-1] + 1) % 4
        return tuple(2.0 if token == target else -2.0 for token in range(4))

    def decode(self, generation) -> Any:
        generated = []
        for _ in range(generation.max_new_tokens):
            logits = self.current_logits()
            token = max(range(len(logits)), key=lambda index: logits[index])
            self.tokens.append(token)
            generated.append(token)
        return SimpleNamespace(token_ids=tuple(generated))

    def stats(self) -> dict[str, Any]:
        count = len(self.tokens)
        turbo = self.model.cache.mode == "turboquant"
        tile = self.model.cache.turboquant_attention_backend == "tile-cuda"
        return {
            "token_count": count,
            "effective_cache": self.model.cache.mode,
            "cache_bytes": count * (40 if turbo else 100),
            "uncompressed_cache_bytes": count * 100,
            "cache_capacity_bytes": 64 * (40 if turbo else 100),
            "turboquant_attention_backend": "tile-cuda" if tile else "cpu",
            "turboquant_gpu_launches": count if tile else 0,
            "turboquant_row_uploads": count if tile else 0,
            "turboquant_h2d_bytes": count * 8 if tile else 0,
            "turboquant_d2h_bytes": count * 4 if tile else 0,
            "turboquant_cpu_compressed_attention_calls": count if turbo and not tile else 0,
        }


class FakeModel:
    def __init__(self, cache) -> None:
        self.cache = cache
        self.prefill_calls = 0
        self.current_logits_calls = 0
        self.truncations: list[int] = []

    def __enter__(self) -> "FakeModel":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        return None

    def stats(self) -> dict[str, Any]:
        return {"max_seq_len": 64, "vocab_size": 4}

    def create_session(self, *, seed: int = 0) -> FakeSession:
        assert isinstance(seed, int)
        return FakeSession(self)


class FakeModelLoader:
    loaded: list[FakeModel] = []

    @classmethod
    def load(cls, _artifact, *, binding, kv_cache):
        assert binding == "binding"
        model = FakeModel(kv_cache)
        cls.loaded.append(model)
        return model


class FakeTracker:
    def __init__(self, _runtime: str, _device: int, *, peak: int = 4096) -> None:
        self.peak = peak
        self.labels: list[str] = []

    def checkpoint(self, label: str) -> None:
        self.labels.append(label)

    def result(self, *, required: bool) -> dict[str, Any]:
        if required and self.peak <= 0:
            raise bench.BenchmarkError("VRAM unavailable")
        return {
            "status": "measured",
            "required": required,
            "provider": "fake-cudaMemGetInfo",
            "scope": "sampled-device-global-baseline-subtracted",
            "is_process_allocator_high_water_mark": False,
            "peak_sampled_delta_bytes": self.peak,
        }


def test_cli_defaults_and_tile_runtime_gate(tmp_path: Path) -> None:
    artifact, binding, tile = _paths(tmp_path, tile=True)
    parser = bench._parser()
    args = parser.parse_args(["--artifact", str(artifact), "--binding-lib", str(binding)])
    config, corpus_kind = bench._config_from_args(args)
    assert config.contexts == (1024, 4096, 16384)
    assert config.decode_tokens == 16
    assert config.quality_window == 128
    assert corpus_kind == "synthetic-repeated-token-0"

    args = parser.parse_args(
        [
            "--artifact",
            str(artifact),
            "--binding-lib",
            str(binding),
            "--tile-ops-lib",
            str(tile),
        ]
    )
    with pytest.raises(bench.BenchmarkError, match="same explicit runtime"):
        bench._config_from_args(args)


def test_timing_and_backward_truncate_quality_use_public_session_path(tmp_path: Path) -> None:
    FakeModelLoader.loaded.clear()
    config = _config(tmp_path)
    timing = bench._run_worker_timing(
        config,
        bench.MODES["full"],
        6,
        binding_loader=lambda _path: "binding",
        model_loader=FakeModelLoader,
        clock=FakeClock(),
    )
    assert timing["pass"] == "timing"
    assert timing["timing"]["ttft_seconds"]["count"] == 2
    assert timing["timing"]["decode_tokens_per_second"]["mean"] == pytest.approx(200.0)
    assert timing["cache"]["live_cache_bytes"]["samples"] == [600, 600]
    assert timing["cache"]["live_uncompressed_equivalent_bytes"]["samples"] == [600, 600]
    assert timing["cache"]["allocated_cache_capacity_bytes"]["samples"] == [6400, 6400]
    assert timing["free_running_greedy"]["token_ids"] == [0, 1]

    quality = bench._run_worker_quality_vram(
        config,
        bench.MODES["full"],
        6,
        binding_loader=lambda _path: "binding",
        model_loader=FakeModelLoader,
    )
    assert quality["pass"] == "quality-vram"
    assert quality["quality"]["teacher_forced_greedy_token_ids"] == [0, 1]
    assert quality["quality"]["perplexity"] > 1.0
    quality_model = FakeModelLoader.loaded[-1]
    assert quality_model.truncations == [5, 4]
    assert quality_model.current_logits_calls == 2
    assert quality["vram"]["status"] == "not-applicable-cpu-mode"


def test_tile_vram_is_required_and_cpu_runtime_delta_can_be_zero(tmp_path: Path) -> None:
    tile_config = _config(tmp_path / "tile", tile=True, runtime="libcudart-test.so")
    with pytest.raises(bench.BenchmarkError, match="VRAM unavailable"):
        bench._run_worker_quality_vram(
            tile_config,
            bench.MODES["mse-tile"],
            6,
            binding_loader=lambda _path: "binding",
            model_loader=FakeModelLoader,
            tracker_factory=lambda runtime, device: FakeTracker(runtime, device, peak=0),
        )

    cpu_config = _config(tmp_path / "cpu", runtime="libcudart-test.so")
    result = bench._run_worker_quality_vram(
        cpu_config,
        bench.MODES["mse-cpu"],
        6,
        binding_loader=lambda _path: "binding",
        model_loader=FakeModelLoader,
        tracker_factory=lambda runtime, device: FakeTracker(runtime, device, peak=0),
    )
    assert result["vram"]["required"] is False
    assert result["vram"]["peak_sampled_delta_bytes"] == 0


def test_cuda_mem_info_tracker_reports_sampled_device_global_delta(monkeypatch) -> None:
    samples = iter(((1000, 2000), (760, 2000), (900, 2000)))

    class Function:
        def __init__(self, callback) -> None:
            self.callback = callback
            self.argtypes = None
            self.restype = None

        def __call__(self, *args):
            return self.callback(*args)

    class Library:
        cudaSetDevice = Function(lambda _device: 0)

        @staticmethod
        def _mem(free, total):
            free_value, total_value = next(samples)
            free._obj.value = free_value
            total._obj.value = total_value
            return 0

        cudaMemGetInfo = Function(_mem)

    monkeypatch.setattr(bench.ctypes, "CDLL", lambda _runtime: Library())
    tracker = bench.CudaMemInfoTracker("libcudart-fixture.so", 0)
    tracker.checkpoint("session-live")
    tracker.checkpoint("decode-complete")
    result = tracker.result(required=True)
    assert result["scope"] == "sampled-device-global-baseline-subtracted"
    assert result["is_process_allocator_high_water_mark"] is False
    assert result["peak_sampled_delta_bytes"] == 240
    assert result["sample_count"] == 3


def _mock_cell(mode: str, context: int, worker_pass: str) -> dict[str, Any]:
    mode_payload = {
        "name": mode,
        "cache_mode": "full" if mode == "full" else "turboquant",
        "turboquant_profile": "mse-3.5" if mode.startswith("mse") else (
            "qjl-3.5" if mode.startswith("qjl") else None
        ),
        "attention_backend": "tile-cuda" if mode.endswith("-tile") else "cpu",
    }
    sequences = {
        "full": [1, 2],
        "mse-cpu": [1, 3],
        "qjl-cpu": [2, 2],
        "mse-tile": [1, 3],
        "qjl-tile": [2, 3],
    }
    perplexities = {
        "full": 2.0,
        "mse-cpu": 2.2,
        "qjl-cpu": 1.8,
        "mse-tile": 2.25,
        "qjl-tile": 1.9,
    }
    base = {
        "status": "complete",
        "pass": worker_pass,
        "mode": mode_payload,
        "context_tokens": context,
    }
    if worker_pass == "timing":
        return {
            **base,
            "timing": {},
            "cache": {},
            "transfers": {},
            "free_running_greedy": {
                "token_ids": sequences[mode],
                "agreement_vs_full": None,
                "agreement_vs_cpu_same_profile": None,
            },
        }
    return {
        **base,
        "quality": {
            "tokens_scored": 2,
            "negative_log_likelihood": 1.0,
            "mean_negative_log_likelihood": perplexities[mode] / 10.0,
            "perplexity": perplexities[mode],
            "perplexity_delta_vs_full": None,
            "negative_log_likelihood_delta_vs_full": None,
            "mean_nll_delta_vs_full": None,
            "teacher_forced_greedy_token_ids": sequences[mode],
            "teacher_forced_greedy_agreement_vs_full": None,
            "teacher_forced_greedy_agreement_vs_cpu_same_profile": None,
        },
        "vram": {
            "status": "measured",
            "scope": "sampled-device-global-baseline-subtracted",
            "peak_sampled_delta_bytes": 123 if mode.endswith("-tile") else 0,
        },
    }


def test_parent_isolates_both_passes_and_attaches_quality_references(tmp_path: Path) -> None:
    config = _config(tmp_path, tile=True, runtime="libcudart-test.so")
    calls: list[list[str]] = []

    def runner(command, **kwargs):
        assert kwargs["cwd"] == ROOT
        calls.append(command)
        mode = command[command.index("--worker-mode") + 1]
        worker_pass = command[command.index("--worker-pass") + 1]
        context = int(command[command.index("--contexts") + 1])
        payload = _mock_cell(mode, context, worker_pass)
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    payload = bench._run_parent(config, "provided-token-corpus", child_runner=runner)
    assert payload["schema"] == bench.SCHEMA
    assert payload["speedup_claimed"] is False
    assert len(calls) == 10
    assert all("--worker" in command and "--worker-pass" in command for command in calls)
    cells = {
        mode["name"]: mode["cells"][0]
        for mode in payload["modes"]
    }
    mse_tile = cells["mse-tile"]
    assert mse_tile["quality"]["perplexity_delta_vs_full"] == {
        "signed": pytest.approx(0.25),
        "absolute": pytest.approx(0.25),
        "relative": pytest.approx(0.125),
    }
    assert mse_tile["quality"]["teacher_forced_greedy_agreement_vs_full"]["rate"] == 0.5
    assert mse_tile["quality"]["teacher_forced_greedy_agreement_vs_cpu_same_profile"]["rate"] == 1.0
    assert mse_tile["free_running_greedy"]["agreement_vs_cpu_same_profile"]["rate"] == 1.0
    assert mse_tile["worker_isolation"] == {
        "timing_worker": True,
        "quality_vram_worker": True,
    }


def test_main_emits_json_failure_before_tile_worker_without_runtime(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifact, binding, tile = _paths(tmp_path, tile=True)
    result = bench.main(
        [
            "--artifact",
            str(artifact),
            "--binding-lib",
            str(binding),
            "--tile-ops-lib",
            str(tile),
            "--contexts",
            "4",
            "--decode-tokens",
            "1",
            "--quality-window",
            "2",
        ]
    )
    assert result == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "failed"
    assert payload["speedup_claimed"] is False
    assert "same explicit runtime" in payload["error"]["message"]
