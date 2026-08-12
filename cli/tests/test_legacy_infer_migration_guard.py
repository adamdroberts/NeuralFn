from __future__ import annotations

import importlib.util
from pathlib import Path
import shlex
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
NFN = ROOT / "cli" / "nfn.py"


def _load_cli_module():
    spec = importlib.util.spec_from_file_location("nfn_legacy_infer_guard_test", NFN)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _migration_output_for(source: Path) -> Path:
    return source.with_name(f"{source.stem}-native")


def _graph_migration_command(graph: Path, weights: Path) -> str:
    return shlex.join(
        [
            "nfn",
            "migrate",
            "graph-to-native",
            "--graph",
            str(graph),
            "--weights",
            str(weights),
            "--output-dir",
            str(_migration_output_for(graph)),
        ]
    )


def _parameter_golf_migration_template(checkpoint: Path) -> str:
    return shlex.join(
        [
            "nfn",
            "migrate",
            "graph-to-native",
            "--graph",
            "MATCHING_GRAPH.json",
            "--weights",
            str(checkpoint),
            "--output-dir",
            str(_migration_output_for(checkpoint)),
        ]
    )


def _install_forwarding_stub(monkeypatch: pytest.MonkeyPatch, module, *, result: int = 19):
    calls: list[tuple[list[str], dict[str, bool]]] = []

    def main(argv, **kwargs):
        calls.append((list(argv), dict(kwargs)))
        return result

    monkeypatch.setattr(module, "_load_full_impl", lambda: SimpleNamespace(main=main))
    return calls


def _forbid_legacy_or_native_execution(monkeypatch: pytest.MonkeyPatch, module) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("an unsupported legacy resident request reached a runtime")

    monkeypatch.setattr(module, "_native_serve_main", forbidden)
    monkeypatch.setattr(module, "_load_full_impl", forbidden)


def test_legacy_graph_weights_inference_is_preserved_with_exact_migration_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    graph = tmp_path / "legacy graph.json"
    weights = tmp_path / "legacy weights.pt"
    module = _load_cli_module()
    calls = _install_forwarding_stub(monkeypatch, module)
    argv = [
        "infer",
        "--graph",
        str(graph),
        "--weights",
        str(weights),
        "--prompt",
        "Hello",
    ]

    result = module.main(argv, stdin_isatty=False, stdout_isatty=False)

    assert result == 19
    assert calls == [(argv, {"stdin_isatty": False, "stdout_isatty": False})]
    stderr = capsys.readouterr().err
    assert "deprecated" in stderr.lower()
    assert "legacy graph" in stderr.lower()
    assert _graph_migration_command(graph, weights) in stderr


@pytest.mark.parametrize("artifact_flag", ["--checkpoint", "--weights"])
def test_graphless_parameter_golf_inference_is_preserved_without_guessing_a_graph(
    artifact_flag: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint = tmp_path / "parameter golf.pt"
    tokenizer = tmp_path / "tokenizer.model"
    module = _load_cli_module()
    calls = _install_forwarding_stub(monkeypatch, module)
    argv = [
        "infer",
        artifact_flag,
        str(checkpoint),
        "--checkpoint-tokenizer",
        str(tokenizer),
        "--prompt",
        "Hello",
    ]

    result = module.main(argv, stdin_isatty=False, stdout_isatty=False)

    assert result == 19
    assert calls == [(argv, {"stdin_isatty": False, "stdout_isatty": False})]
    stderr = capsys.readouterr().err
    assert "deprecated" in stderr.lower()
    assert "graphless parameter golf" in stderr.lower()
    assert "matching neuralfn graph" in stderr.lower()
    assert _parameter_golf_migration_template(checkpoint) in stderr


@pytest.mark.parametrize(
    ("unsupported_args", "expected_reason"),
    [
        (["--serve"], "serve"),
        (["--kv-cache", "turboquant"], "turboquant"),
        (["--kv-cache=turboquant"], "turboquant"),
    ],
)
def test_legacy_graph_resident_requests_fail_before_any_runtime_with_exact_migration_command(
    unsupported_args: list[str],
    expected_reason: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    graph = tmp_path / "legacy graph.json"
    weights = tmp_path / "legacy weights.pt"
    module = _load_cli_module()
    _forbid_legacy_or_native_execution(monkeypatch, module)
    argv = [
        "infer",
        "--graph",
        str(graph),
        "--weights",
        str(weights),
        "--prompt",
        "Hello",
        *unsupported_args,
    ]

    result = module.main(argv, stdin_isatty=False, stdout_isatty=False)

    assert result == 2
    stderr = capsys.readouterr().err
    assert expected_reason in stderr.lower()
    assert "legacy graph" in stderr.lower()
    assert _graph_migration_command(graph, weights) in stderr
    assert "does not make legacy weights resident-loadable" in stderr
    assert "compatible resident native dense-v5 checkpoint" in stderr
    assert "Then use the native artifact" not in stderr


@pytest.mark.parametrize(
    ("unsupported_args", "expected_reason"),
    [
        (["--serve"], "serve"),
        (["--kv-cache", "turboquant"], "turboquant"),
        (["--kv-cache=turboquant"], "turboquant"),
    ],
)
def test_graphless_parameter_golf_resident_requests_fail_before_any_runtime_with_graph_prerequisite(
    unsupported_args: list[str],
    expected_reason: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint = tmp_path / "parameter golf.pt"
    module = _load_cli_module()
    _forbid_legacy_or_native_execution(monkeypatch, module)
    argv = [
        "infer",
        "--checkpoint",
        str(checkpoint),
        "--checkpoint-tokenizer",
        str(tmp_path / "tokenizer.model"),
        "--prompt",
        "Hello",
        *unsupported_args,
    ]

    result = module.main(argv, stdin_isatty=False, stdout_isatty=False)

    assert result == 2
    stderr = capsys.readouterr().err
    assert expected_reason in stderr.lower()
    assert "graphless parameter golf" in stderr.lower()
    assert "matching neuralfn graph" in stderr.lower()
    assert _parameter_golf_migration_template(checkpoint) in stderr
    assert "does not make legacy weights resident-loadable" in stderr
    assert "compatible resident native dense-v5 checkpoint" in stderr
    assert "Then use the native artifact" not in stderr
