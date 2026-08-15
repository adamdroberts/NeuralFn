from __future__ import annotations

from contextlib import nullcontext

import pytest

import server.mcp_server as mcp_server


def test_mcp_native_train_returns_server_compatibility_and_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, dict | None]] = []
    preflight = {
        "runtime": "native-cuda",
        "execution_ready": True,
        "compatible": True,
        "issues": [],
        "compatibility_report": {"compatible": True, "graph_fingerprint": "abc"},
        "artifact_metadata": {"materialized": False},
    }
    started = {
        "run_id": "run-1",
        "status": "running",
        "runtime": "native-cuda",
        "compatibility_report": {"compatible": True, "graph_fingerprint": "abc"},
        "artifact_metadata": {
            "materialized": True,
            "manifest_path": "/artifacts/run-1/native-ir/native-execution-manifest.json",
        },
    }

    def fake_request(method: str, endpoint: str, data: dict | None = None, **_kwargs):
        calls.append((method, endpoint, data))
        if endpoint.endswith("/runs/preflight"):
            return preflight
        if endpoint.endswith("/runs/start"):
            return started
        raise AssertionError(endpoint)

    monkeypatch.setattr(mcp_server, "AgentSession", lambda *_args: nullcontext())
    monkeypatch.setattr(mcp_server, "_request", fake_request)

    result = mcp_server.train_start(
        project_id="project-1",
        session_id="session-1",
        runtime="native-cuda",
        dataset_names=["tiny_tokens"],
        epochs=2,
    )

    assert result["run_id"] == "run-1"
    assert result["status"] == "started"
    assert result["run_status"] == "running"
    assert result["runtime"] == "native-cuda"
    assert result["compatibility_report"]["graph_fingerprint"] == "abc"
    assert result["artifact_metadata"]["materialized"] is True
    assert [endpoint.rsplit("/", 1)[-1] for _, endpoint, _ in calls] == ["preflight", "start"]
    assert calls[0][2]["runtime"] == "native-cuda"


def test_mcp_native_train_returns_node_failures_without_starting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    incompatible = {
        "runtime": "native-cuda",
        "execution_ready": False,
        "compatible": False,
        "issues": [
            {
                "path": "root/nodes/model/nodes/future",
                "code": "unsupported_module",
                "message": "No reviewed lowerer.",
            }
        ],
        "compatibility_report": {"compatible": False},
        "artifact_metadata": {"materialized": False},
    }

    def fake_request(_method: str, endpoint: str, _data: dict | None = None, **_kwargs):
        calls.append(endpoint)
        if endpoint.endswith("/runs/preflight"):
            return incompatible
        raise AssertionError("MCP launched a trainer after failed preflight")

    monkeypatch.setattr(mcp_server, "AgentSession", lambda *_args: nullcontext())
    monkeypatch.setattr(mcp_server, "_request", fake_request)

    result = mcp_server.train_start(
        project_id="project-1",
        session_id="session-1",
        runtime="native-cuda",
        dataset_names=["tiny_tokens"],
    )

    assert result["status"] == "incompatible"
    assert result["issues"][0]["path"].endswith("/future")
    assert len(calls) == 1


def test_mcp_rejects_unknown_runtime_before_request() -> None:
    with pytest.raises(ValueError, match="runtime must be"):
        mcp_server.train_start("project-1", "session-1", runtime="native-ish")


def test_mcp_runtime_extension_preserves_existing_positional_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict | None] = []

    def fake_request(_method: str, endpoint: str, data: dict | None = None, **_kwargs):
        calls.append(data)
        if endpoint.endswith("/runs/preflight"):
            return {
                "runtime": "torch",
                "execution_ready": True,
                "compatibility_report": {},
                "artifact_metadata": {},
            }
        return {"run_id": "legacy-positional", "status": "running", "runtime": "torch"}

    monkeypatch.setattr(mcp_server, "AgentSession", lambda *_args: nullcontext())
    monkeypatch.setattr(mcp_server, "_request", fake_request)

    result = mcp_server.train_start(
        "project-1",
        "session-1",
        "torch",
        7,
        0.02,
        [[1.0]],
        [[2.0]],
        ["tiny_tokens"],
    )

    assert result["status"] == "started"
    assert calls[0] == {
        "method": "torch",
        "runtime": None,
        "epochs": 7,
        "learning_rate": 0.02,
        "train_inputs": [[1.0]],
        "train_targets": [[2.0]],
        "dataset_names": ["tiny_tokens"],
    }
