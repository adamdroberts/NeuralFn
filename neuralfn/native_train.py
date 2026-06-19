from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Sequence


DEFAULT_NATIVE_TRAIN_CLI = "build/nfn_native_train"
DEFAULT_NATIVE_GPT_TRAIN_CLI = "build/nfn_gpt_native_train"
DENSE_GPT_MODEL_FAMILIES = frozenset({"gpt", "gpt2", "gpt3", "nanogpt", "nano-gpt"})
NATIVE_TRAIN_BINDING_MODULES = ("neuralfn_native_train", "neuralfn._native_train")


@dataclass(frozen=True)
class NativeTrainRunnerStatus:
    requested: str
    resolved: str
    binding_module: str | None = None
    available: bool = True
    reason: str = ""
    command_resolver_available: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NativeTrainRunConfig:
    """Configuration for the unified native CUDA/C++ training frontend."""

    model_family: str = "gpt"
    args: tuple[str, ...] = ()
    native_train_cli: str | None = None
    cuda_visible_devices: str = "0"
    cuda_device_max_connections: str = "1"

    def argv(self) -> list[str]:
        normalized_family = normalize_native_model_family(self.model_family)
        family_cli = resolve_native_train_family_cli(normalized_family, self.native_train_cli)
        if family_cli is not None:
            return [
                family_cli,
                "--model-family",
                normalized_family,
                *self.args,
            ]
        return [
            resolve_native_train_cli(self.native_train_cli),
            "--base-model",
            normalized_family,
            *self.args,
        ]

    def command(self) -> str:
        return shlex.join(self.argv())

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "model_family": normalize_native_model_family(self.model_family),
            "args": list(self.args),
            "argv": self.argv(),
            "command": self.command(),
        }


def normalize_native_model_family(value: str | None) -> str:
    normalized = str(value or "gpt").strip().lower().replace("_", "-")
    return normalized or "gpt"


def resolve_native_train_cli(value: str | None = None) -> str:
    requested = str(value or "").strip()
    if requested:
        return requested
    env_value = str(os.environ.get("NFN_NATIVE_TRAIN_CLI", "")).strip()
    if env_value:
        return env_value
    repo_root = Path(__file__).resolve().parents[1]
    return str(repo_root / DEFAULT_NATIVE_TRAIN_CLI)


def resolve_native_train_family_cli(model_family: str | None, native_train_cli: str | None = None) -> str | None:
    """Return a direct model-family trainer CLI when the SDK can skip the generic dispatcher."""

    if str(native_train_cli or "").strip() or str(os.environ.get("NFN_NATIVE_TRAIN_CLI", "")).strip():
        return None
    normalized = normalize_native_model_family(model_family)
    if normalized not in DENSE_GPT_MODEL_FAMILIES:
        return None
    env_value = str(os.environ.get("NFN_NATIVE_GPT_CLI", "")).strip()
    if env_value:
        return env_value
    repo_root = Path(__file__).resolve().parents[1]
    family_cli = repo_root / DEFAULT_NATIVE_GPT_TRAIN_CLI
    if family_cli.exists():
        return str(family_cli)
    return None


def resolve_available_native_train_cli_for_status() -> Path:
    cli = Path(resolve_native_train_cli())
    if cli.exists():
        return cli
    family_cli = resolve_native_train_family_cli("gpt")
    if family_cli:
        return Path(family_cli)
    return cli


def build_native_train_run_config(
    model_family: str = "gpt",
    args: Sequence[str] | None = None,
    *,
    native_train_cli: str | None = None,
) -> NativeTrainRunConfig:
    return NativeTrainRunConfig(
        model_family=normalize_native_model_family(model_family),
        args=tuple(str(arg) for arg in (args or ())),
        native_train_cli=native_train_cli,
    )


def _load_native_train_binding():
    binding_enabled = str(os.environ.get("NFN_NATIVE_TRAIN_BINDING", "1")).strip().lower()
    if binding_enabled in {"0", "false", "no", "off"}:
        raise ImportError("native train binding disabled by NFN_NATIVE_TRAIN_BINDING=0")
    errors: list[str] = []
    importlib.invalidate_caches()
    for module_name in NATIVE_TRAIN_BINDING_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            errors.append(f"{module_name}: {exc}")
            continue
        runner = getattr(module, "run_train", None) or getattr(module, "run_native_train", None)
        resolver = getattr(module, "resolve_command", None) or getattr(module, "resolve_native_train_command", None)
        if callable(runner) and callable(resolver):
            return module_name, runner, resolver
        if module_name == "neuralfn._native_train":
            fallback = _load_complete_native_train_package_binding(module)
            if fallback is not None:
                return fallback
        errors.append(
            f"{module_name}: missing run_train(config_dict)/run_native_train(config_dict) "
            "or resolve_command(config_dict)/resolve_native_train_command(config_dict)"
        )
    raise ImportError("; ".join(errors) if errors else "no native train binding modules configured")


def _load_complete_native_train_package_binding(incomplete_module: Any) -> tuple[str, Any, Any] | None:
    """Find a complete neuralfn._native_train extension if a stale one shadows it."""

    try:
        package = importlib.import_module("neuralfn")
    except ImportError:
        return None
    package_paths = [str(path) for path in getattr(package, "__path__", [])]
    if not package_paths:
        return None
    current_file = str(getattr(incomplete_module, "__file__", "") or "")
    original_path = list(getattr(package, "__path__", []))
    original_module = incomplete_module
    module_name = "neuralfn._native_train"
    suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    for package_path in reversed(package_paths):
        try:
            candidates = list(Path(package_path).glob("_native_train*.so"))
        except OSError:
            continue
        if not any(str(candidate).endswith(suffixes) and str(candidate) != current_file for candidate in candidates):
            continue
        try:
            package.__path__ = [package_path, *[path for path in original_path if str(path) != package_path]]
            importlib.invalidate_caches()
            sys.modules.pop(module_name, None)
            module = importlib.import_module(module_name)
            runner = getattr(module, "run_train", None) or getattr(module, "run_native_train", None)
            resolver = getattr(module, "resolve_command", None) or getattr(module, "resolve_native_train_command", None)
            if callable(runner) and callable(resolver):
                return module_name, runner, resolver
        except ImportError:
            continue
        finally:
            package.__path__ = original_path
    sys.modules[module_name] = original_module
    return None


def native_train_runner_status(requested: str = "auto") -> NativeTrainRunnerStatus:
    normalized = str(requested or "auto").strip().lower().replace("_", "-")
    if normalized == "cli":
        normalized = "compiled-cli"
    if normalized not in {"auto", "binding", "compiled-cli", "subprocess"}:
        raise ValueError("native train runner must be one of: auto, binding, compiled-cli, subprocess")
    if normalized in {"compiled-cli", "subprocess"}:
        cli = resolve_available_native_train_cli_for_status()
        return NativeTrainRunnerStatus(
            requested=normalized,
            resolved=normalized,
            available=cli.exists() or normalized == "subprocess",
            reason="" if cli.exists() else f"compiled native train CLI not found: {cli}",
        )
    try:
        module_name, _runner, _resolver = _load_native_train_binding()
    except ImportError as exc:
        if normalized == "binding":
            return NativeTrainRunnerStatus(
                requested=normalized,
                resolved="binding",
                available=False,
                reason=str(exc),
            )
        cli = resolve_available_native_train_cli_for_status()
        return NativeTrainRunnerStatus(
            requested=normalized,
            resolved="compiled-cli",
            available=cli.exists(),
            reason="" if cli.exists() else f"native binding unavailable and compiled native train CLI not found: {exc}",
        )
    return NativeTrainRunnerStatus(
        requested=normalized,
        resolved="binding",
        binding_module=module_name,
        command_resolver_available=True,
    )


def resolve_native_train_binding_command(config: NativeTrainRunConfig) -> list[str]:
    """Return the argv that the compiled native-train binding will spawn."""

    _module_name, _runner, resolver = _load_native_train_binding()
    return [str(item) for item in resolver(config.to_dict())]


def run_native_train(config: NativeTrainRunConfig, *, runner: str = "auto") -> int:
    status = native_train_runner_status(runner)
    if status.resolved == "binding":
        if not status.available:
            raise RuntimeError(f"Native train binding requested but unavailable: {status.reason}")
        _module_name, binding_runner, _resolver = _load_native_train_binding()
        return int(binding_runner(config.to_dict()))
    if not status.available:
        raise RuntimeError(f"Native train CLI requested but unavailable: {status.reason}")
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", config.cuda_visible_devices)
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", config.cuda_device_max_connections)
    env.setdefault("CUDA_MODULE_LOADING", "LAZY")
    proc = subprocess.run(config.argv(), env=env, check=False)
    return int(proc.returncode)


def native_train_model_registry(*, native_train_cli: str | None = None) -> dict[str, Any]:
    command = [
        resolve_native_train_cli(native_train_cli),
        "--list-models",
        "--json",
    ]
    proc = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"native train registry command failed with exit {proc.returncode}: {proc.stderr.strip()}")
    return json.loads(proc.stdout)
