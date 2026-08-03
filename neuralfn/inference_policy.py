"""Process-wide inference policy for deterministic temperature-zero execution.

This module deliberately has no PyTorch dependency at import time.  Entry
points must call :func:`prepare_inference_process_environment` before their
first import/use of CUDA; PyTorch itself is supplied lazily to the execution
context once generation is ready to start.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
import os
import sys
import threading
from typing import Any, Callable, Iterator


DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def prepare_inference_process_environment() -> None:
    """Arm process environment required by a later strict CUDA generation.

    ``CUBLAS_WORKSPACE_CONFIG`` is consumed when CUDA/cuBLAS is initialized,
    so setting it only after an interactive ``/temp 0`` command is too late.
    This call is safe and idempotent and intentionally does not import torch.
    """

    current = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    loaded_torch = sys.modules.get("torch")
    loaded_cuda = getattr(loaded_torch, "cuda", None)
    is_initialized = getattr(loaded_cuda, "is_initialized", None)
    if callable(is_initialized) and is_initialized() and current != DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(
            "Strict inference environment was prepared after CUDA initialization. Restart the process so "
            f"CUBLAS_WORKSPACE_CONFIG={DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG!r} is set before CUDA starts."
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG


def validate_inference_temperature(value: Any) -> float:
    """Return a finite, nonnegative inference temperature.

    Exact zero (including ``-0.0``) is the only strict-compute trigger.
    Negative and non-finite values are rejected rather than being treated as
    aliases for greedy decoding.
    """

    try:
        temperature = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("temperature must be a finite number greater than or equal to 0.") from exc
    if not math.isfinite(temperature) or temperature < 0.0:
        raise ValueError("temperature must be a finite number greater than or equal to 0.")
    return temperature


def temperature_uses_strict_inference(value: Any) -> bool:
    return validate_inference_temperature(value) == 0.0


class InferenceConcurrencyGate:
    """Writer-preferring process gate for process-global CUDA/Torch flags.

    Standard generations are readers and may overlap.  Strict generations
    are writers because they temporarily change process-global backend flags.
    Once a writer is waiting, new readers wait too, preventing strict requests
    from starving under sustained standard traffic.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition(threading.Lock())
        self._readers = 0
        self._writer = False
        self._waiting_writers = 0

    @contextmanager
    def shared(self) -> Iterator[None]:
        with self._condition:
            while self._writer or self._waiting_writers:
                self._condition.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._condition:
                self._readers -= 1
                if self._readers == 0:
                    self._condition.notify_all()

    @contextmanager
    def exclusive(self) -> Iterator[None]:
        with self._condition:
            self._waiting_writers += 1
            try:
                while self._writer or self._readers:
                    self._condition.wait()
                self._writer = True
            finally:
                self._waiting_writers -= 1
        try:
            yield
        finally:
            with self._condition:
                self._writer = False
                self._condition.notify_all()


INFERENCE_CONCURRENCY_GATE = InferenceConcurrencyGate()


@dataclass(frozen=True)
class InferenceComputePolicy:
    mode: str
    trigger: str | None
    deterministic_algorithms: bool
    autocast_disabled: bool
    tf32_disabled: bool
    reduced_precision_reductions_disabled: bool
    fast_math_disabled: bool

    @property
    def strict(self) -> bool:
        return self.mode == "strict"


STANDARD_INFERENCE_POLICY = InferenceComputePolicy(
    mode="standard",
    trigger=None,
    deterministic_algorithms=False,
    autocast_disabled=False,
    tf32_disabled=False,
    reduced_precision_reductions_disabled=False,
    fast_math_disabled=False,
)

STRICT_INFERENCE_POLICY = InferenceComputePolicy(
    mode="strict",
    trigger="temperature_zero",
    deterministic_algorithms=True,
    autocast_disabled=True,
    tf32_disabled=True,
    reduced_precision_reductions_disabled=True,
    fast_math_disabled=True,
)


def compute_policy_payload(
    policy: InferenceComputePolicy,
    *,
    backend: str,
) -> dict[str, Any]:
    """Return the stable REST/native-facing compute-policy telemetry shape."""

    return {
        "version": 1,
        "mode": policy.mode,
        "trigger": policy.trigger,
        "backend": str(backend),
        "deterministic_algorithms": policy.deterministic_algorithms,
        "autocast_disabled": policy.autocast_disabled,
        "tf32_disabled": policy.tf32_disabled,
        "reduced_precision_reductions_disabled": policy.reduced_precision_reductions_disabled,
        "fast_math_disabled": policy.fast_math_disabled,
    }


def _read_optional_attr(obj: Any, name: str) -> tuple[bool, Any]:
    if obj is None or not hasattr(obj, name):
        return False, None
    return True, getattr(obj, name)


def _call_optional(obj: Any, name: str) -> tuple[bool, Any]:
    function = getattr(obj, name, None) if obj is not None else None
    if not callable(function):
        return False, None
    return True, function()


class _TorchStrictSettings:
    """Snapshot, enforce, verify, and restore strict PyTorch backend flags."""

    def __init__(self, torch_module: Any) -> None:
        self.torch = torch_module
        self._restore_actions: list[tuple[Callable[[Any], Any], Any]] = []
        self._uses_fp32_precision_api = False

    def _set_attr(self, obj: Any, name: str, value: Any, *, required: bool = True) -> None:
        exists, previous = _read_optional_attr(obj, name)
        if not exists:
            if required:
                raise RuntimeError(f"Strict inference requires PyTorch backend control {name!r}.")
            return
        setattr(obj, name, value)
        self._restore_actions.append(
            (lambda old_value, target=obj, attribute=name: setattr(target, attribute, old_value), previous)
        )

    def _set_via_call(
        self,
        getter_owner: Any,
        getter_name: str,
        setter_owner: Any,
        setter_name: str,
        value: Any,
        *,
        required: bool = True,
    ) -> None:
        getter = getattr(getter_owner, getter_name, None)
        setter = getattr(setter_owner, setter_name, None)
        if not callable(getter) or not callable(setter):
            if required:
                raise RuntimeError(f"Strict inference requires PyTorch backend control {setter_name!r}.")
            return
        previous = getter()
        setter(value)
        self._restore_actions.append((setter, previous))

    def apply(self) -> None:
        torch = self.torch
        deterministic_getter = getattr(torch, "are_deterministic_algorithms_enabled", None)
        deterministic_setter = getattr(torch, "use_deterministic_algorithms", None)
        if not callable(deterministic_getter) or not callable(deterministic_setter):
            raise RuntimeError("Strict inference requires torch.use_deterministic_algorithms().")
        previous_deterministic = bool(deterministic_getter())
        warn_getter = getattr(torch, "is_deterministic_algorithms_warn_only_enabled", None)
        previous_warn_only = bool(warn_getter()) if callable(warn_getter) else False
        deterministic_setter(True, warn_only=False)
        self._restore_actions.append(
            (lambda state: deterministic_setter(state[0], warn_only=state[1]), (previous_deterministic, previous_warn_only))
        )

        backends = getattr(torch, "backends", None)
        cuda_backend = getattr(backends, "cuda", None)
        cudnn_backend = getattr(backends, "cudnn", None)
        matmul_backend = getattr(cuda_backend, "matmul", None)
        if cuda_backend is None or cudnn_backend is None or matmul_backend is None:
            raise RuntimeError("Strict inference requires the PyTorch CUDA backend controls.")

        self._set_attr(cudnn_backend, "deterministic", True)
        self._set_attr(cudnn_backend, "benchmark", False)

        # Recent PyTorch releases expose hierarchical FP32-precision controls.
        # The new and legacy TF32 APIs must not be mixed, so use the complete
        # new API when present and otherwise fall back to the legacy flags.
        fp32_precision_targets = (
            (backends, "global"),
            (matmul_backend, "CUDA matmul"),
            (cudnn_backend, "cuDNN"),
            (getattr(cudnn_backend, "conv", None), "cuDNN convolution"),
            (getattr(cudnn_backend, "rnn", None), "cuDNN RNN"),
        )
        fp32_precision_presence = tuple(
            _read_optional_attr(target, "fp32_precision")[0]
            for target, _label in fp32_precision_targets
        )
        if all(fp32_precision_presence):
            self._uses_fp32_precision_api = True
            # Snapshot the entire hierarchy before changing its parent: a
            # parent's value is inherited by children whose stored value is
            # ``"none"``, so snapshotting and setting sequentially would lose
            # the child's real pre-context state.
            previous_fp32_precision = tuple(
                getattr(target, "fp32_precision")
                for target, _label in fp32_precision_targets
            )
            for (target, _label), previous in zip(fp32_precision_targets, previous_fp32_precision):
                setattr(target, "fp32_precision", "ieee")
                self._restore_actions.append(
                    (
                        lambda old_value, target=target: setattr(target, "fp32_precision", old_value),
                        previous,
                    )
                )
        elif any(fp32_precision_presence):
            missing = ", ".join(
                label
                for present, (_target, label) in zip(fp32_precision_presence, fp32_precision_targets)
                if not present
            )
            raise RuntimeError(
                "Strict inference found an incomplete PyTorch FP32-precision API; "
                f"missing controls: {missing}."
            )
        else:
            self._set_attr(cudnn_backend, "allow_tf32", False)
            self._set_attr(matmul_backend, "allow_tf32", False)
            self._set_via_call(
                torch,
                "get_float32_matmul_precision",
                torch,
                "set_float32_matmul_precision",
                "highest",
            )
        self._set_attr(matmul_backend, "allow_fp16_reduced_precision_reduction", False)
        self._set_attr(matmul_backend, "allow_bf16_reduced_precision_reduction", False)
        self._set_attr(matmul_backend, "allow_fp16_accumulation", False, required=False)

        self._set_via_call(
            cuda_backend,
            "fp16_bf16_reduction_math_sdp_allowed",
            cuda_backend,
            "allow_fp16_bf16_reduction_math_sdp",
            False,
            required=False,
        )

        for enabled_name, setter_name, enabled in (
            ("flash_sdp_enabled", "enable_flash_sdp", False),
            ("mem_efficient_sdp_enabled", "enable_mem_efficient_sdp", False),
            ("cudnn_sdp_enabled", "enable_cudnn_sdp", False),
            ("math_sdp_enabled", "enable_math_sdp", True),
        ):
            self._set_via_call(
                cuda_backend,
                enabled_name,
                cuda_backend,
                setter_name,
                enabled,
                required=setter_name != "enable_cudnn_sdp",
            )

        self.verify()

    def verify(self) -> None:
        torch = self.torch
        backends = torch.backends
        cuda_backend = backends.cuda
        matmul_backend = cuda_backend.matmul
        failures: list[str] = []
        if not torch.are_deterministic_algorithms_enabled():
            failures.append("deterministic algorithms")
        warn_getter = getattr(torch, "is_deterministic_algorithms_warn_only_enabled", None)
        if callable(warn_getter) and warn_getter():
            failures.append("deterministic fail-closed mode")
        checks = [
            (backends.cudnn.deterministic is True, "cuDNN deterministic mode"),
            (backends.cudnn.benchmark is False, "cuDNN benchmarking disabled"),
            (
                matmul_backend.allow_fp16_reduced_precision_reduction is False,
                "FP16 reduced-precision reductions disabled",
            ),
            (
                matmul_backend.allow_bf16_reduced_precision_reduction is False,
                "BF16 reduced-precision reductions disabled",
            ),
        ]
        if self._uses_fp32_precision_api:
            checks.extend(
                (getattr(target, "fp32_precision") == "ieee", f"{label} IEEE FP32 precision")
                for target, label in (
                    (backends, "global"),
                    (matmul_backend, "CUDA matmul"),
                    (backends.cudnn, "cuDNN"),
                    (backends.cudnn.conv, "cuDNN convolution"),
                    (backends.cudnn.rnn, "cuDNN RNN"),
                )
            )
        else:
            checks.extend(
                (
                    (backends.cudnn.allow_tf32 is False, "cuDNN TF32 disabled"),
                    (matmul_backend.allow_tf32 is False, "matmul TF32 disabled"),
                    (torch.get_float32_matmul_precision() == "highest", "highest float32 matmul precision"),
                )
            )
        failures.extend(label for passed, label in checks if not passed)
        if (
            hasattr(matmul_backend, "allow_fp16_accumulation")
            and matmul_backend.allow_fp16_accumulation is not False
        ):
            failures.append("FP16 accumulation disabled")
        reduction_getter = getattr(cuda_backend, "fp16_bf16_reduction_math_sdp_allowed", None)
        if callable(reduction_getter) and reduction_getter():
            failures.append("FP16/BF16 math SDPA reductions disabled")
        for enabled_name, expected, label in (
            ("flash_sdp_enabled", False, "Flash SDPA disabled"),
            ("mem_efficient_sdp_enabled", False, "memory-efficient SDPA disabled"),
            ("cudnn_sdp_enabled", False, "cuDNN SDPA disabled"),
            ("math_sdp_enabled", True, "math SDPA enabled"),
        ):
            present, value = _call_optional(cuda_backend, enabled_name)
            if (not present and enabled_name != "cudnn_sdp_enabled") or (
                present and bool(value) is not expected
            ):
                failures.append(label)
        if failures:
            raise RuntimeError("Strict inference could not enforce: " + ", ".join(failures) + ".")

    def synchronize(self) -> None:
        cuda = getattr(self.torch, "cuda", None)
        is_available = getattr(cuda, "is_available", None)
        is_initialized = getattr(cuda, "is_initialized", None)
        synchronize = getattr(cuda, "synchronize", None)
        if (
            callable(is_available)
            and is_available()
            and callable(is_initialized)
            and is_initialized()
            and callable(synchronize)
        ):
            synchronize()

    def restore(self) -> None:
        errors: list[BaseException] = []
        for restore, value in reversed(self._restore_actions):
            try:
                restore(value)
            except BaseException as exc:  # pragma: no cover - catastrophic backend failure
                errors.append(exc)
        if errors:
            raise RuntimeError("Strict inference failed to restore PyTorch backend settings.") from errors[0]


@contextmanager
def inference_execution(
    temperature: Any,
    *,
    torch_module: Any | None = None,
) -> Iterator[InferenceComputePolicy]:
    """Gate and configure one complete generation for ``temperature``.

    The caller must keep every model forward, token selection, and enqueued
    CUDA operation inside this context.  Strict execution requires the caller
    to pass its lazily imported ``torch`` module.
    """

    strict = temperature_uses_strict_inference(temperature)
    if not strict:
        with INFERENCE_CONCURRENCY_GATE.shared():
            yield STANDARD_INFERENCE_POLICY
        return
    if torch_module is None:
        raise RuntimeError("Strict inference requires the active torch module.")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(
            "Strict inference requires prepare_inference_process_environment() before CUDA initialization "
            f"(CUBLAS_WORKSPACE_CONFIG={DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG!r})."
        )

    with INFERENCE_CONCURRENCY_GATE.exclusive():
        settings = _TorchStrictSettings(torch_module)
        active_error: BaseException | None = None
        applied = False
        try:
            try:
                settings.apply()
                applied = True
            except BaseException as exc:
                active_error = exc
                raise
            try:
                yield STRICT_INFERENCE_POLICY
            except BaseException as exc:
                active_error = exc
                raise
            finally:
                if applied:
                    try:
                        settings.synchronize()
                    except BaseException as exc:
                        if active_error is not None:
                            active_error.add_note(f"Strict CUDA synchronization also failed: {exc}")
                        else:
                            active_error = exc
                            raise
        finally:
            try:
                settings.restore()
            except BaseException as exc:
                if active_error is not None:
                    active_error.add_note(f"Strict PyTorch setting restoration also failed: {exc}")
                else:
                    raise


_STRICT_UNSUPPORTED_MODULE_TYPES = frozenset(
    {
        "bitlinear_ternary",
        "fp8_linear",
        "mx_linear",
        "nf4_linear",
        "kv_quant_pack",
        "kv_quant_unpack",
    }
)
_STRICT_UNSUPPORTED_COMPRESSIONS = frozenset(
    {"bitlinear", "ternary", "fp8_e4m3", "fp8_e5m2", "mxfp4", "mxfp8", "qlora"}
)
_STRICT_STATEFUL_STOCHASTIC_MODULE_TYPES = frozenset(
    {"jepa_mask", "mask_scheduler", "random_timesteps"}
)


def validate_strict_graph_support(graph: Any) -> None:
    """Fail closed when a graph explicitly computes in a lossy format."""

    torch_config = dict(getattr(graph, "torch_config", {}) or {})
    tile_activation_dtype = str(torch_config.get("tile_cuda_activation_dtype", "") or "").strip().lower()
    if tile_activation_dtype not in {"", "none", "float32", "fp32"}:
        raise RuntimeError(
            "temperature=0 strict inference does not support explicit low-precision Tile activations "
            f"({tile_activation_dtype!r}); use float32 activations."
        )
    template_spec = dict(torch_config.get("template_spec", {}) or {})
    template = dict(template_spec.get("template", {}) or {})
    compression = str(template.get("compression", "none") or "none").strip().lower()
    if compression in _STRICT_UNSUPPORTED_COMPRESSIONS:
        raise RuntimeError(
            f"temperature=0 strict inference does not support explicit {compression!r} graph compression; "
            "use a float32-capable graph/checkpoint."
        )

    visited: set[int] = set()

    def walk(candidate: Any) -> None:
        identity = id(candidate)
        if identity in visited:
            return
        visited.add(identity)
        for node in dict(getattr(candidate, "nodes", {}) or {}).values():
            neuron_def = getattr(node, "neuron_def", None)
            module_type = str(getattr(neuron_def, "module_type", "") or "").strip().lower()
            if module_type in _STRICT_UNSUPPORTED_MODULE_TYPES:
                raise RuntimeError(
                    f"temperature=0 strict inference does not support lossy module {module_type!r}; "
                    "use a float32-capable graph/checkpoint."
                )
            if module_type in _STRICT_STATEFUL_STOCHASTIC_MODULE_TYPES:
                raise RuntimeError(
                    "temperature=0 strict inference does not support stateful stochastic module "
                    f"{module_type!r}; remove training-time noise/masking from the inference graph."
                )
            module_config = dict(getattr(neuron_def, "module_config", {}) or {})
            compute_dtype = str(module_config.get("compute_dtype", "float32") or "float32").lower()
            if module_type == "nf4_linear" and compute_dtype not in {"float32", "fp32"}:
                raise RuntimeError(
                    "temperature=0 strict inference does not support NF4 low-precision compute; "
                    "use a float32-capable graph/checkpoint."
                )
            subgraph = getattr(neuron_def, "subgraph", None)
            if subgraph is not None:
                walk(subgraph)
        for family in dict(getattr(candidate, "variant_library", {}) or {}).values():
            for variant in dict(family or {}).values():
                if variant is not None:
                    walk(variant)

    walk(graph)


def validate_strict_compiled_graph(compiled: Any) -> None:
    """Reject a supposedly strict model with low-precision floating state."""

    low_precision: list[str] = []
    for collection_name, getter_name in (("parameter", "named_parameters"), ("buffer", "named_buffers")):
        getter = getattr(compiled, getter_name, None)
        if not callable(getter):
            continue
        for name, value in getter():
            is_floating_point = getattr(value, "is_floating_point", None)
            if callable(is_floating_point) and is_floating_point() and str(getattr(value, "dtype", "")) != "torch.float32":
                low_precision.append(f"{collection_name} {name} ({getattr(value, 'dtype', 'unknown')})")
    if low_precision:
        preview = ", ".join(low_precision[:3])
        raise RuntimeError(
            "temperature=0 strict inference requires FP32 floating model state; "
            f"low-precision state remains: {preview}."
        )
