from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from neuralfn.inference_policy import prepare_inference_process_environment
from server.routers import sessions


class _Workspace:
    def __init__(self) -> None:
        self.bundle = SimpleNamespace(
            graph_state=SimpleNamespace(
                graph={
                    "name": "chat-test",
                    "runtime": "torch",
                    "torch_config": {"device": "cpu"},
                    "nodes": {},
                    "edges": {},
                    "input_node_ids": [],
                    "output_node_ids": [],
                }
            )
        )

    def get_session_bundle(self, _db, _user, _project_id, _session_id):
        return self.bundle


def _call_chat(body: dict) -> dict:
    with patch.object(sessions, "get_workspace_service", return_value=_Workspace()):
        return sessions.chat_generate(
            "project",
            "session",
            body,
            auth=SimpleNamespace(user=object()),
            db=object(),
        )


@pytest.mark.parametrize("temperature", [-1, float("nan"), float("inf"), float("-inf")])
def test_chat_generate_rejects_invalid_temperature_before_torch_setup(temperature: float) -> None:
    with patch("neuralfn.torch_backend.CompiledTorchGraph") as compiled:
        with pytest.raises(HTTPException) as exc_info:
            _call_chat({"prompt": "hello", "temperature": temperature})

    assert exc_info.value.status_code == 400
    assert "finite number" in str(exc_info.value.detail)
    compiled.assert_not_called()


@pytest.mark.parametrize("temperature", [0.0, -0.0])
def test_chat_generate_uses_loaded_compiled_model_and_reports_strict_policy(temperature: float) -> None:
    import torch

    prepare_inference_process_environment()
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    compiled_instances = []
    cache_compiled = []
    deterministic_during_step = []

    class FakeCompiled:
        resolved_kernel_backend = "torch"

        def __init__(self, _graph, *, kernel_backend: str):
            assert kernel_backend == "torch"
            self.loaded_states = []
            compiled_instances.append(self)

        def load_state_dict(self, state, *, strict: bool):
            assert strict is False
            self.loaded_states.append(state)

        def to(self, device):
            assert str(device) == "cpu"
            return self

        def train(self, mode: bool):
            assert mode is False
            return self

        def eval(self):
            return self

    class FakeCache:
        def __init__(self, _graph, device=None, *, compiled=None):
            assert device == "cpu"
            assert compiled is compiled_instances[0]
            cache_compiled.append(compiled)

        def step(self, _tokens):
            deterministic_during_step.append(torch.are_deterministic_algorithms_enabled())
            logits = torch.zeros((1, 256), dtype=torch.float32)
            logits[0, 65] = 1.0
            return logits

    def fake_checkpoint(path):
        return ({"source": str(path)}, {})

    with (
        patch("neuralfn.torch_backend.CompiledTorchGraph", FakeCompiled),
        patch("neuralfn.inference.InferenceCache", FakeCache),
        patch("neuralfn.inference.load_pt_checkpoint", side_effect=fake_checkpoint),
    ):
        response = _call_chat(
            {
                "prompt": "hi",
                "max_new_tokens": 1,
                "temperature": temperature,
                "base_checkpoint": "base.pt",
                "adapter_checkpoint": "adapter.pt",
            }
        )

    assert len(compiled_instances) == 1
    assert cache_compiled == compiled_instances
    assert compiled_instances[0].loaded_states == [
        {"source": "base.pt"},
        {"source": "adapter.pt"},
    ]
    assert deterministic_during_step == [True]
    assert response["tokens"] == [104, 105, 65]
    assert response["compute_policy"] == {
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
    assert torch.are_deterministic_algorithms_enabled() is deterministic_before


def test_chat_generate_keeps_positive_top_k_one_on_standard_compute_policy() -> None:
    import torch

    class FakeCompiled:
        resolved_kernel_backend = "tile_cuda"

        def __init__(self, _graph, *, kernel_backend: str):
            assert kernel_backend == "auto"

        def to(self, _device):
            return self

        def train(self, mode: bool):
            assert mode is False
            return self

    class FakeCache:
        def __init__(self, _graph, device=None, *, compiled=None):
            assert device == "cpu"
            assert isinstance(compiled, FakeCompiled)

        def step(self, _tokens):
            logits = torch.zeros((1, 256), dtype=torch.float32)
            logits[0, 66] = 1.0
            return logits

    with (
        patch("neuralfn.torch_backend.CompiledTorchGraph", FakeCompiled),
        patch("neuralfn.inference.InferenceCache", FakeCache),
    ):
        response = _call_chat(
            {
                "prompt": "hi",
                "max_new_tokens": 1,
                "temperature": 0.8,
                "top_k": 1,
            }
        )

    assert response["tokens"] == [104, 105, 66]
    assert response["compute_policy"] == {
        "version": 1,
        "mode": "standard",
        "trigger": None,
        "backend": "tile_cuda",
        "deterministic_algorithms": False,
        "autocast_disabled": True,
        "tf32_disabled": False,
        "reduced_precision_reductions_disabled": False,
        "fast_math_disabled": False,
    }
