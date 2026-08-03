from __future__ import annotations

import importlib
import os
import sys
import threading
import time
from types import SimpleNamespace

import pytest


def _fake_legacy_precision_torch() -> SimpleNamespace:
    state = {
        "deterministic": False,
        "warn_only": True,
        "matmul_precision": "medium",
        "math_reduction": True,
        "flash": True,
        "mem_efficient": True,
        "math_sdp": False,
    }

    def use_deterministic_algorithms(enabled: bool, *, warn_only: bool = False) -> None:
        state["deterministic"] = enabled
        state["warn_only"] = warn_only

    def flag_getter(name: str):
        return lambda: state[name]

    def flag_setter(name: str):
        return lambda enabled: state.__setitem__(name, enabled)

    matmul = SimpleNamespace(
        allow_tf32=True,
        allow_fp16_reduced_precision_reduction=True,
        allow_bf16_reduced_precision_reduction=True,
        allow_fp16_accumulation=True,
    )
    cuda_backend = SimpleNamespace(
        matmul=matmul,
        fp16_bf16_reduction_math_sdp_allowed=flag_getter("math_reduction"),
        allow_fp16_bf16_reduction_math_sdp=flag_setter("math_reduction"),
        flash_sdp_enabled=flag_getter("flash"),
        enable_flash_sdp=flag_setter("flash"),
        mem_efficient_sdp_enabled=flag_getter("mem_efficient"),
        enable_mem_efficient_sdp=flag_setter("mem_efficient"),
        math_sdp_enabled=flag_getter("math_sdp"),
        enable_math_sdp=flag_setter("math_sdp"),
    )
    cudnn_backend = SimpleNamespace(
        deterministic=False,
        benchmark=True,
        allow_tf32=True,
    )
    return SimpleNamespace(
        _test_state=state,
        backends=SimpleNamespace(cuda=cuda_backend, cudnn=cudnn_backend),
        are_deterministic_algorithms_enabled=flag_getter("deterministic"),
        is_deterministic_algorithms_warn_only_enabled=flag_getter("warn_only"),
        use_deterministic_algorithms=use_deterministic_algorithms,
        get_float32_matmul_precision=lambda: state["matmul_precision"],
        set_float32_matmul_precision=lambda value: state.__setitem__("matmul_precision", value),
    )


def test_inference_policy_import_is_torch_free() -> None:
    previous = sys.modules.pop("neuralfn.inference_policy", None)
    torch_before = sys.modules.get("torch")
    try:
        module = importlib.import_module("neuralfn.inference_policy")
        assert module is not None
        assert sys.modules.get("torch") is torch_before
    finally:
        if previous is not None:
            sys.modules["neuralfn.inference_policy"] = previous


def test_prepare_environment_overrides_incompatible_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    from neuralfn.inference_policy import (
        DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG,
        prepare_inference_process_environment,
    )

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False)),
    )
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    prepare_inference_process_environment()
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG


def test_prepare_environment_fails_if_cuda_was_initialized(monkeypatch: pytest.MonkeyPatch) -> None:
    from neuralfn.inference_policy import prepare_inference_process_environment

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with pytest.raises(RuntimeError, match="after CUDA initialization"):
        prepare_inference_process_environment()
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"


@pytest.mark.parametrize("value", [0, 0.0, -0.0, "0"])
def test_exact_zero_activates_strict(value: object) -> None:
    from neuralfn.inference_policy import temperature_uses_strict_inference

    assert temperature_uses_strict_inference(value)


@pytest.mark.parametrize("value", [1.0e-50, 5.0e-324, "1e-50"])
def test_every_positive_temperature_remains_standard(value: object) -> None:
    from neuralfn.inference_policy import temperature_uses_strict_inference

    assert not temperature_uses_strict_inference(value)


@pytest.mark.parametrize("value", [-0.01, float("nan"), float("inf"), float("-inf"), "nope", None])
def test_invalid_temperature_is_rejected(value: object) -> None:
    from neuralfn.inference_policy import validate_inference_temperature

    with pytest.raises(ValueError, match="finite number"):
        validate_inference_temperature(value)


def test_compute_policy_payload_has_stable_shape() -> None:
    from neuralfn.inference_policy import STRICT_INFERENCE_POLICY, compute_policy_payload

    assert compute_policy_payload(STRICT_INFERENCE_POLICY, backend="torch_math") == {
        "version": 1,
        "mode": "strict",
        "trigger": "temperature_zero",
        "backend": "torch_math",
        "deterministic_algorithms": True,
        "autocast_disabled": True,
        "tf32_disabled": True,
        "reduced_precision_reductions_disabled": True,
        "fast_math_disabled": True,
    }


def test_strict_execution_requires_early_workspace_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    from neuralfn.inference_policy import inference_execution

    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(RuntimeError, match="prepare_inference_process_environment"):
        with inference_execution(0, torch_module=object()):
            pass


def test_exceptional_strict_execution_synchronizes_then_restores_before_unlock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import neuralfn.inference_policy as policy_module

    events: list[str] = []

    class FakeStrictSettings:
        def __init__(self, _torch_module: object) -> None:
            pass

        def apply(self) -> None:
            events.append("apply")

        def synchronize(self) -> None:
            assert policy_module.INFERENCE_CONCURRENCY_GATE._writer
            events.append("synchronize")
            raise RuntimeError("sync failed")

        def restore(self) -> None:
            assert policy_module.INFERENCE_CONCURRENCY_GATE._writer
            events.append("restore")

    monkeypatch.setattr(policy_module, "_TorchStrictSettings", FakeStrictSettings)
    monkeypatch.setenv(
        "CUBLAS_WORKSPACE_CONFIG",
        policy_module.DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG,
    )
    generation_error = RuntimeError("generation failed")
    with pytest.raises(RuntimeError) as captured:
        with policy_module.inference_execution(0, torch_module=object()):
            events.append("body")
            raise generation_error

    assert captured.value is generation_error
    assert events == ["apply", "body", "synchronize", "restore"]
    assert not policy_module.INFERENCE_CONCURRENCY_GATE._writer
    assert any("sync failed" in note for note in getattr(generation_error, "__notes__", ()))


def test_writer_preferring_gate_blocks_new_readers() -> None:
    from neuralfn.inference_policy import InferenceConcurrencyGate

    gate = InferenceConcurrencyGate()
    reader_one_entered = threading.Event()
    release_reader_one = threading.Event()
    writer_waiting = threading.Event()
    writer_entered = threading.Event()
    release_writer = threading.Event()
    reader_two_entered = threading.Event()
    order: list[str] = []

    def reader_one() -> None:
        with gate.shared():
            order.append("reader_one")
            reader_one_entered.set()
            release_reader_one.wait(timeout=2)

    def writer() -> None:
        writer_waiting.set()
        with gate.exclusive():
            order.append("writer")
            writer_entered.set()
            release_writer.wait(timeout=2)

    def reader_two() -> None:
        with gate.shared():
            order.append("reader_two")
            reader_two_entered.set()

    threads = [threading.Thread(target=reader_one), threading.Thread(target=writer), threading.Thread(target=reader_two)]
    threads[0].start()
    assert reader_one_entered.wait(timeout=2)
    threads[1].start()
    assert writer_waiting.wait(timeout=2)
    for _ in range(100):
        if gate._waiting_writers:  # targeted white-box assertion for ordering
            break
        time.sleep(0.001)
    threads[2].start()
    assert not reader_two_entered.wait(timeout=0.05)
    release_reader_one.set()
    assert writer_entered.wait(timeout=2)
    assert not reader_two_entered.is_set()
    release_writer.set()
    assert reader_two_entered.wait(timeout=2)
    for thread in threads:
        thread.join(timeout=2)
    assert order == ["reader_one", "writer", "reader_two"]


def test_strict_execution_sets_and_restores_torch_flags() -> None:
    torch = pytest.importorskip("torch")
    from neuralfn.inference_policy import inference_execution, prepare_inference_process_environment

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    prepare_inference_process_environment()

    precision_targets = {
        "global_fp32": torch.backends,
        "matmul_fp32": torch.backends.cuda.matmul,
        "cudnn_fp32": torch.backends.cudnn,
        "cudnn_conv_fp32": getattr(torch.backends.cudnn, "conv", None),
        "cudnn_rnn_fp32": getattr(torch.backends.cudnn, "rnn", None),
    }
    has_new_fp32_api = all(
        target is not None and hasattr(target, "fp32_precision")
        for target in precision_targets.values()
    )

    before = {
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "warn": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "fp16_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        "bf16_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "flash": torch.backends.cuda.flash_sdp_enabled(),
        "mem": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math": torch.backends.cuda.math_sdp_enabled(),
    }
    if has_new_fp32_api:
        before.update(
            (name, getattr(target, "fp32_precision"))
            for name, target in precision_targets.items()
        )
    else:
        before.update(
            cudnn_tf32=torch.backends.cudnn.allow_tf32,
            matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
            precision=torch.get_float32_matmul_precision(),
        )
    with inference_execution(0, torch_module=torch) as policy:
        assert policy.strict
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        assert torch.backends.cudnn.deterministic
        assert not torch.backends.cudnn.benchmark
        if has_new_fp32_api:
            assert all(
                getattr(target, "fp32_precision") == "ieee"
                for target in precision_targets.values()
            )
        else:
            assert not torch.backends.cudnn.allow_tf32
            assert not torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        assert not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        if not has_new_fp32_api:
            assert torch.get_float32_matmul_precision() == "highest"
        assert not torch.backends.cuda.flash_sdp_enabled()
        assert not torch.backends.cuda.mem_efficient_sdp_enabled()
        assert torch.backends.cuda.math_sdp_enabled()

    after = {
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "warn": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "fp16_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        "bf16_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "flash": torch.backends.cuda.flash_sdp_enabled(),
        "mem": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math": torch.backends.cuda.math_sdp_enabled(),
    }
    if has_new_fp32_api:
        after.update(
            (name, getattr(target, "fp32_precision"))
            for name, target in precision_targets.items()
        )
    else:
        after.update(
            cudnn_tf32=torch.backends.cudnn.allow_tf32,
            matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
            precision=torch.get_float32_matmul_precision(),
        )
    assert after == before


def test_strict_settings_support_and_restore_legacy_precision_api() -> None:
    from neuralfn.inference_policy import _TorchStrictSettings

    torch = _fake_legacy_precision_torch()
    settings = _TorchStrictSettings(torch)
    settings.apply()
    assert not settings._uses_fp32_precision_api
    assert torch.backends.cudnn.allow_tf32 is False
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.get_float32_matmul_precision() == "highest"
    settings.restore()
    assert torch.backends.cudnn.allow_tf32 is True
    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.get_float32_matmul_precision() == "medium"
    assert torch.are_deterministic_algorithms_enabled() is False
    assert torch.is_deterministic_algorithms_warn_only_enabled() is True


def test_strict_settings_fail_closed_and_restore_partial_precision_api() -> None:
    from neuralfn.inference_policy import _TorchStrictSettings

    torch = _fake_legacy_precision_torch()
    torch.backends.fp32_precision = "none"
    settings = _TorchStrictSettings(torch)
    with pytest.raises(RuntimeError, match="incomplete PyTorch FP32-precision API"):
        settings.apply()
    settings.restore()
    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.benchmark is True
    assert torch.are_deterministic_algorithms_enabled() is False
    assert torch.is_deterministic_algorithms_warn_only_enabled() is True


def test_strict_graph_rejects_explicit_lossy_compute() -> None:
    from types import SimpleNamespace

    from neuralfn.inference_policy import validate_strict_graph_support

    graph = SimpleNamespace(
        torch_config={"template_spec": {"template": {"compression": "fp8_e4m3"}}},
        nodes={},
        variant_library={},
    )
    with pytest.raises(RuntimeError, match="fp8_e4m3"):
        validate_strict_graph_support(graph)

    tile_graph = SimpleNamespace(
        torch_config={"tile_cuda_activation_dtype": "nvfp4"},
        nodes={},
        variant_library={},
    )
    with pytest.raises(RuntimeError, match="low-precision Tile activations"):
        validate_strict_graph_support(tile_graph)

    stochastic_graph = SimpleNamespace(
        torch_config={},
        nodes={
            "mask": SimpleNamespace(
                neuron_def=SimpleNamespace(
                    module_type="jepa_mask",
                    module_config={},
                    subgraph=None,
                )
            )
        },
        variant_library={},
    )
    with pytest.raises(RuntimeError, match="stateful stochastic module 'jepa_mask'"):
        validate_strict_graph_support(stochastic_graph)


def test_strict_compiled_graph_rejects_low_precision_parameters_and_buffers() -> None:
    torch = pytest.importorskip("torch")
    from neuralfn.inference_policy import validate_strict_compiled_graph

    compiled = SimpleNamespace(
        named_parameters=lambda: (("weight", torch.ones(2, dtype=torch.float32)),),
        named_buffers=lambda: (("cache", torch.ones(2, dtype=torch.bfloat16)),),
    )
    with pytest.raises(RuntimeError, match="buffer cache"):
        validate_strict_compiled_graph(compiled)


def test_strict_verification_covers_optional_precision_controls() -> None:
    torch = pytest.importorskip("torch")
    from neuralfn.inference_policy import _TorchStrictSettings

    settings = _TorchStrictSettings(torch)
    settings.apply()
    try:
        if settings._uses_fp32_precision_api:
            torch.backends.cuda.matmul.fp32_precision = "tf32"
            with pytest.raises(RuntimeError, match="CUDA matmul IEEE FP32 precision"):
                settings.verify()
            torch.backends.cuda.matmul.fp32_precision = "ieee"

        torch.backends.cuda.matmul.allow_fp16_accumulation = True
        with pytest.raises(RuntimeError, match="FP16 accumulation"):
            settings.verify()
        torch.backends.cuda.matmul.allow_fp16_accumulation = False

        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
        with pytest.raises(RuntimeError, match="math SDPA reductions"):
            settings.verify()
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)

        torch.backends.cuda.enable_cudnn_sdp(True)
        with pytest.raises(RuntimeError, match="cuDNN SDPA"):
            settings.verify()
        torch.backends.cuda.enable_cudnn_sdp(False)
    finally:
        settings.restore()
