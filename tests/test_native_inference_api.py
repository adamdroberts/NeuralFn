from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any, Mapping

import pytest

from neuralfn.native_inference import (
    GenerationConfig,
    GenerationEvent,
    GenerationResult,
    KVCacheConfig,
    LLAMA_SESSION_PREFIX_COW_PROFILE,
    NativeInferenceCapabilities,
    NativeInferenceCapabilityError,
    NativeInferenceClosedError,
    NativeInferenceModel,
    NativeModelLoadConfig,
    SESSION_PREFIX_COW_PROFILE,
    STANDARD_MOE_SESSION_PREFIX_COW_PROFILE,
    CPU_TURBOQUANT_SESSION_PREFIX_COW_PROFILE,
)


def _write_artifact(
    root: Path,
    *,
    native_inference: bool = True,
    resident_inference: bool = True,
    lossless_kv_cache: bool = True,
    turboquant_kv_cache: bool = False,
    turboquant_tile_attention: bool = False,
    structured_output: bool = False,
    function_tools: bool = False,
    session_prefix_cow: bool = False,
    checkpoint_format: str = "neuralfn.native_dense_gpt.v5",
    session_prefix_cow_profile: str = SESSION_PREFIX_COW_PROFILE,
    session_prefix_cow_cpu_turboquant: bool = False,
    session_prefix_cow_cpu_turboquant_profile: str = (
        CPU_TURBOQUANT_SESSION_PREFIX_COW_PROFILE
    ),
    resident_abi_version: int | None = 1,
    resident_abi_status: str = "ready",
    speculative_decoding: bool = False,
) -> Path:
    root.mkdir()
    checkpoint_bytes = b"neuralfn-fake-resident-checkpoint-v1"
    (root / "checkpoint.bin").write_bytes(checkpoint_bytes)
    checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    dflash_bytes = b"neuralfn-fake-dflash-checkpoint-v1"
    if speculative_decoding:
        (root / "dflash.bin").write_bytes(dflash_bytes)
    payload = {
        "schema": "neuralfn.native_execution_manifest",
        "version": 1,
        "source_graph": {"sha256": "fixture"},
        "model": {"family": "fake-test-only"},
        "topology": {"resolved": True},
        "tensors": [],
        "tokenizer": (
            {
                "constrained_decoding": {
                    "version": 1,
                    "profile": "json-schema-ascii-byte-greedy-v1",
                    "token_selection": "current_logits_exact_prefill",
                }
            }
            if structured_output
            else {}
        ),
        "chat_template": {
            "source": "fixture",
            "template": "{messages}",
            "tool_template": (
                {
                    "version": 1,
                    "profile": "responses-forced-function-call-v1",
                }
                if function_tools
                else None
            ),
        },
        "context_limits": {"max_context_tokens": 64, "max_output_tokens": 8},
        "stop_tokens": [99],
        "kernel_abi": {
            "resident_inference": {
                "version": resident_abi_version,
                "status": resident_abi_status,
            },
            "turboquant_cache": {
                "version": 1 if turboquant_kv_cache else None,
                "status": "ready" if turboquant_kv_cache else "not_implemented",
                "backend": "test-double" if turboquant_kv_cache else None,
            },
            "turboquant_tile_attention": {
                "symbol": "nfn_native_tile_turboquant_attention_forward_v1",
                "version": 1 if turboquant_tile_attention else None,
                "status": "ready" if turboquant_tile_attention else "not_implemented",
                "backend": "tile-cuda-hybrid" if turboquant_tile_attention else None,
            },
            "structured_output": {
                "version": 1 if structured_output else None,
                "status": "ready" if structured_output else "not_implemented",
                "profile": (
                    "json-schema-ascii-byte-greedy-v1" if structured_output else None
                ),
                "token_selection": (
                    "current_logits_exact_prefill" if structured_output else None
                ),
            },
            "function_tools": {
                "version": 1 if function_tools else None,
                "status": "ready" if function_tools else "not_implemented",
                "profile": (
                    "responses-forced-function-call-v1" if function_tools else None
                ),
                "structured_output_profile": (
                    "json-schema-ascii-byte-greedy-v1" if function_tools else None
                ),
            },
            "session_prefix_cow": {
                "version": 1 if session_prefix_cow else None,
                "status": "ready" if session_prefix_cow else "not_implemented",
                "profile": (
                    session_prefix_cow_profile
                    if session_prefix_cow
                    else None
                ),
                "operation": "fork_session" if session_prefix_cow else None,
            },
            "session_prefix_cow_cpu_turboquant": {
                "version": 1 if session_prefix_cow_cpu_turboquant else None,
                "status": (
                    "ready"
                    if session_prefix_cow_cpu_turboquant
                    else "not_implemented"
                ),
                "profile": (
                    session_prefix_cow_cpu_turboquant_profile
                    if session_prefix_cow_cpu_turboquant
                    else None
                ),
                "operation": (
                    "fork_session" if session_prefix_cow_cpu_turboquant else None
                ),
                "backend": (
                    "cpu-reference-packed"
                    if session_prefix_cow_cpu_turboquant
                    else None
                ),
            },
            "speculative_decoding": {
                "version": 1 if speculative_decoding else 0,
                "status": "ready" if speculative_decoding else "unavailable",
                "profile": (
                    "muse-glimmer-dflash-block16-v1"
                    if speculative_decoding
                    else None
                ),
                "load_operation": "load_companion" if speculative_decoding else None,
                "decode_operation": (
                    "decode_speculative_block" if speculative_decoding else None
                ),
                "block_size": 16 if speculative_decoding else None,
                "proposal_tokens": 15 if speculative_decoding else None,
            },
        },
        "checkpoint": {
            "format": checkpoint_format,
            "artifact_path": "checkpoint.bin",
            "target_nbytes": len(checkpoint_bytes),
            "target_sha256": checkpoint_sha256,
        },
        "primary_checkpoint_variant": "bf16" if speculative_decoding else None,
        "checkpoint_variants": (
            {
                "bf16": {
                    "format": checkpoint_format,
                    "artifact_path": "checkpoint.bin",
                    "target_nbytes": len(checkpoint_bytes),
                    "target_sha256": checkpoint_sha256,
                    "required_kernel_profile": "fixture-bf16",
                    "resident_weight_bytes": len(checkpoint_bytes),
                    "peak_load_staging_bytes": 0,
                    "max_workspace_bytes": 0,
                    "memory_profile": {
                        "version": 1,
                        "minimum_total_vram_bytes": 1,
                        "backend_fingerprint": "fixture-cpu-v1",
                        "fixed_runtime_bytes": 0,
                        "kv_cache_bytes_per_context_token_per_session": 1,
                        "session_bytes": 1,
                    },
                }
            }
            if speculative_decoding
            else {}
        ),
        "companion_checkpoints": (
            {
                "dflash": {
                    "format": "neuralfn.native_family_muse_glimmer_dflash.bf16.v1",
                    "artifact_path": "dflash.bin",
                    "target_nbytes": len(dflash_bytes),
                    "target_sha256": hashlib.sha256(dflash_bytes).hexdigest(),
                    "resident_weight_bytes": len(dflash_bytes),
                    "target_compatibility": {
                        "allowed_target_checkpoint_sha256": [checkpoint_sha256],
                        "target_layer_ids_zero_based": [1, 13, 25, 37, 49],
                        "block_size": 16,
                        "proposal_tokens": 15,
                        "mask_token_id": 201818,
                        "shared_embedding": True,
                        "shared_lm_head": True,
                    },
                }
            }
            if speculative_decoding
            else {}
        ),
        "session_state_kinds": ["kv"],
        "capabilities": {
            "native_inference": native_inference,
            "resident_inference": resident_inference,
            "lossless_kv_cache": lossless_kv_cache,
            "turboquant_kv_cache": turboquant_kv_cache,
            "turboquant_tile_attention": turboquant_tile_attention,
            "function_tools": function_tools,
            "structured_output": structured_output,
            "session_prefix_cow": session_prefix_cow,
            "session_prefix_cow_cpu_turboquant": session_prefix_cow_cpu_turboquant,
            "speculative_decoding": speculative_decoding,
        },
    }
    if not speculative_decoding:
        payload.pop("primary_checkpoint_variant", None)
        payload.pop("checkpoint_variants", None)
        payload.pop("companion_checkpoints", None)
    (root / "native-execution-manifest.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    return root


class FakeResidentBinding:
    """Stateful in-process test double; it is not a model implementation."""

    def __init__(
        self,
        *,
        turboquant: bool = False,
        tile_attention: bool = False,
        constrained_decoding: bool = False,
        channels: int = 8,
        num_heads: int = 1,
        session_prefix_cow: bool = False,
        session_prefix_cow_cpu_turboquant: bool = False,
        session_prefix_cow_profiles: tuple[str, ...] = (
            SESSION_PREFIX_COW_PROFILE,
            CPU_TURBOQUANT_SESSION_PREFIX_COW_PROFILE,
            LLAMA_SESSION_PREFIX_COW_PROFILE,
            STANDARD_MOE_SESSION_PREFIX_COW_PROFILE,
        ),
        speculative_decoding: bool = False,
    ) -> None:
        self.turboquant = turboquant
        self.tile_attention = tile_attention
        self.constrained_decoding = constrained_decoding
        self.channels = channels
        self.num_heads = num_heads
        self.session_prefix_cow = session_prefix_cow
        self.session_prefix_cow_cpu_turboquant = session_prefix_cow_cpu_turboquant
        self.session_prefix_cow_profiles = session_prefix_cow_profiles
        self.speculative_decoding = speculative_decoding
        self.model_loads = 0
        self.model_closes = 0
        self.session_closes = 0
        self.cancel_calls = 0
        self.calls: list[tuple[Any, ...]] = []
        self.sessions: dict[int, dict[str, Any]] = {}
        self._next_session = 1

    def resident_inference_abi_version(self) -> int:
        return 1

    def resident_inference_capabilities(self) -> Mapping[str, Any]:
        return {
            "native_inference": True,
            "resident_inference": True,
            "lossless_kv_cache": True,
            "turboquant_kv_cache": self.turboquant,
            "turboquant_tile_attention": self.tile_attention,
            "function_tools": False,
            "structured_output": False,
            "current_logits_exact_prefill": self.constrained_decoding,
            "session_prefix_cow": self.session_prefix_cow,
            "session_prefix_cow_cpu_turboquant": self.session_prefix_cow_cpu_turboquant,
            "session_prefix_cow_abi": {
                "version": 1,
                "operation": "fork_session",
                "profiles": list(self.session_prefix_cow_profiles),
            },
            "weight_kernel_profiles": ["fixture-bf16"],
            "speculative_decoding": self.speculative_decoding,
            "dflash_cpu": self.speculative_decoding,
            "dflash_cuda": self.speculative_decoding,
            "speculative_decoding_abi": {
                "version": 1,
                "load_operation": "load_companion",
                "decode_operation": "decode_speculative_block",
                "block_size": 16,
                "proposal_tokens": 15,
            },
        }

    def load_model(self, artifact_root: str, manifest: dict[str, Any]) -> str:
        self.model_loads += 1
        self.calls.append(("load_model", artifact_root, manifest["schema"]))
        return "model-handle"

    def close_model(self, model: str) -> None:
        assert model == "model-handle"
        self.model_closes += 1
        self.calls.append(("close_model",))

    def load_companion(
        self,
        model: str,
        artifact_root: str,
        descriptor: dict[str, Any],
    ) -> Mapping[str, Any]:
        assert model == "model-handle"
        assert descriptor["format"] == (
            "neuralfn.native_family_muse_glimmer_dflash.bf16.v1"
        )
        self.calls.append(("load_companion", artifact_root, descriptor["target_sha256"]))
        return {"loaded": True, "component": "dflash", "whole_model_cuda": True}

    def configure_model_turboquant_attention(
        self,
        model: str,
        config: dict[str, Any],
    ) -> Mapping[str, Any]:
        assert model == "model-handle"
        self.calls.append(("configure_tile", dict(config)))
        return {"configured": True, "backend": "tile-cuda"}

    def create_session(self, model: str, config: dict[str, Any]) -> int:
        assert model == "model-handle"
        session = self._next_session
        self._next_session += 1
        self.sessions[session] = {
            "tokens": [],
            "seed": config["seed"],
            "cache": dict(config["kv_cache"]),
            "cancelled": False,
        }
        self.calls.append(("create_session", session, config))
        return session

    def close_session(self, model: str, session: int) -> None:
        assert model == "model-handle"
        self.session_closes += 1
        self.calls.append(("close_session", session))

    def fork_session(
        self,
        model: str,
        source: int,
        config: dict[str, Any],
    ) -> int:
        assert model == "model-handle"
        target = self._next_session
        self._next_session += 1
        source_state = self.sessions[source]
        token_count = config["token_count"]
        self.sessions[target] = {
            "tokens": list(source_state["tokens"][:token_count]),
            "seed": config["seed"],
            "cache": dict(source_state["cache"]),
            "cancelled": False,
        }
        self.calls.append(("fork_session", source, target, token_count, config["seed"]))
        return target

    def prefill(
        self,
        model: str,
        session: int,
        token_ids: list[int],
        start_position: int,
    ) -> None:
        assert model == "model-handle"
        state = self.sessions[session]
        assert len(state["tokens"]) == start_position
        state["tokens"].extend(token_ids)
        self.calls.append(("prefill", session, tuple(token_ids), start_position))

    def decode_one(
        self,
        model: str,
        session: int,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        assert model == "model-handle"
        state = self.sessions[session]
        # Session ID and position make cross-session leakage immediately visible.
        token_id = session * 100 + len(state["tokens"])
        state["tokens"].append(token_id)  # native commitment precedes return
        self.calls.append(
            (
                "decode_one",
                session,
                token_id,
                config["temperature"],
                config["strict_model_compute"],
            )
        )
        return {"token_id": token_id, "text": f"<{token_id}>", "finish_reason": None}

    def decode_speculative_block(
        self,
        model: str,
        session: int,
        config: dict[str, Any],
    ) -> Mapping[str, Any]:
        assert model == "model-handle"
        state = self.sessions[session]
        count = min(3, config["max_tokens_remaining"])
        rows: list[dict[str, Any]] = []
        for _index in range(count):
            token_id = session * 100 + len(state["tokens"])
            finish = "stop" if token_id in config["stop_token_ids"] else None
            state["tokens"].append(token_id)
            rows.append(
                {
                    "token_id": token_id,
                    "text": f"<{token_id}>",
                    "finish_reason": finish,
                }
            )
            if finish is not None:
                break
        proposed = max(0, len(rows) - 1)
        self.calls.append(
            ("decode_speculative_block", session, tuple(row["token_id"] for row in rows))
        )
        return {
            "tokens": rows,
            "proposed_tokens": proposed,
            "accepted_tokens": proposed,
            "rejected_tokens": 0,
            "target_rows": len(rows),
            "assistant_blocks": 1 if proposed else 0,
        }

    def current_logits(self, model: str, session: int) -> list[float]:
        assert model == "model-handle"
        token_count = len(self.sessions[session]["tokens"])
        self.calls.append(("current_logits", session, token_count))
        return [float(token_count), -float(token_count)]

    def truncate_session(self, model: str, session: int, token_count: int) -> None:
        assert model == "model-handle"
        del self.sessions[session]["tokens"][token_count:]
        self.calls.append(("truncate", session, token_count))

    def reset_session(self, model: str, session: int) -> None:
        assert model == "model-handle"
        self.sessions[session]["tokens"].clear()
        self.sessions[session]["cancelled"] = False
        self.calls.append(("reset", session))

    def cancel_session(self, model: str, session: int) -> None:
        assert model == "model-handle"
        self.sessions[session]["cancelled"] = True
        self.cancel_calls += 1
        self.calls.append(("cancel", session))

    def model_stats(self, model: str) -> Mapping[str, Any]:
        assert model == "model-handle"
        return {
            "model_loads": self.model_loads,
            "channels": self.channels,
            "num_heads": self.num_heads,
        }

    def session_stats(self, model: str, session: int) -> Mapping[str, Any]:
        assert model == "model-handle"
        return {"native_token_count": len(self.sessions[session]["tokens"])}


class CoordinatedResidentBinding(FakeResidentBinding):
    """Fake binding with event-controlled lifecycle return points."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.block_create = False
        self.block_fork = False
        self.block_close_session = False
        self.block_close_session_handle: int | None = None
        self.create_entered = threading.Event()
        self.fork_entered = threading.Event()
        self.close_session_entered = threading.Event()
        self.allow_create = threading.Event()
        self.allow_fork = threading.Event()
        self.allow_close_session = threading.Event()

    @staticmethod
    def _await_release(release: threading.Event, operation: str) -> None:
        if not release.wait(timeout=5):
            raise RuntimeError(f"timed out waiting to release {operation}")

    def create_session(self, model: str, config: dict[str, Any]) -> int:
        session = super().create_session(model, config)
        if self.block_create:
            self.create_entered.set()
            self._await_release(self.allow_create, "create_session")
        return session

    def fork_session(
        self,
        model: str,
        source: int,
        config: dict[str, Any],
    ) -> int:
        session = super().fork_session(model, source, config)
        if self.block_fork:
            self.fork_entered.set()
            self._await_release(self.allow_fork, "fork_session")
        return session

    def close_session(self, model: str, session: int) -> None:
        super().close_session(model, session)
        if self.block_close_session and (
            self.block_close_session_handle is None
            or self.block_close_session_handle == session
        ):
            self.close_session_entered.set()
            self._await_release(self.allow_close_session, "close_session")


def _thread_call(
    operation: Any,
    results: list[Any],
    errors: list[BaseException],
) -> None:
    try:
        results.append(operation())
    except BaseException as exc:
        errors.append(exc)


def _join_thread(thread: threading.Thread) -> None:
    thread.join(timeout=5)
    assert not thread.is_alive(), f"thread {thread.name} did not finish"


def test_model_loads_once_and_keeps_multi_session_state_isolated(tmp_path: Path) -> None:
    artifact = _write_artifact(tmp_path / "artifact")
    binding = FakeResidentBinding()

    with NativeInferenceModel.load(artifact, binding=binding) as model:
        first = model.create_session(seed=11)
        second = model.create_session(seed=22)
        first.prefill([1, 2])
        second.prefill([7])

        assert first.current_logits() == (2.0, -2.0)
        assert model.current_logits(second) == (1.0, -1.0)

        first_result = model.decode(first, GenerationConfig(max_new_tokens=2))
        second_result = second.decode(GenerationConfig(max_new_tokens=1))

        assert binding.model_loads == 1
        assert first_result.token_ids == (102, 103)
        assert second_result.token_ids == (201,)
        assert first.token_ids == (1, 2, 102, 103)
        assert second.token_ids == (7, 201)
        assert binding.sessions[1]["tokens"] == list(first.token_ids)
        assert binding.sessions[2]["tokens"] == list(second.token_ids)
        assert first.stats()["seed"] == 11
        assert second.stats()["seed"] == 22
        assert model.stats()["open_sessions"] == 2

    assert binding.model_closes == 1
    assert binding.session_closes == 2


def test_dflash_required_loads_companion_and_mirrors_atomic_blocks_before_callbacks(
    tmp_path: Path,
) -> None:
    artifact = _write_artifact(
        tmp_path / "artifact",
        speculative_decoding=True,
    )
    binding = FakeResidentBinding(speculative_decoding=True)
    observed_native_lengths: list[int] = []

    with NativeInferenceModel.load(
        artifact,
        binding=binding,
        load_config=NativeModelLoadConfig(
            runtime="cpu",
            speculative_decoding="required",
        ),
    ) as model:
        assert model.capabilities.speculative_decoding is True
        assert model.capabilities.dflash_cpu is True
        assert model.stats()["effective_speculative_decoding"] == "dflash"
        session = model.create_session(seed=7)
        session.prefill([1, 2])
        result = session.decode(
            GenerationConfig(max_new_tokens=5, temperature=0.0),
            on_token=lambda _event: observed_native_lengths.append(
                len(binding.sessions[1]["tokens"])
            ),
        )

        assert result.token_ids == (102, 103, 104, 105, 106)
        assert result.speculative_proposed_tokens == 3
        assert result.speculative_accepted_tokens == 3
        assert result.speculative_rejected_tokens == 0
        assert result.speculative_target_rows == 5
        assert result.speculative_assistant_blocks == 2
        assert observed_native_lengths == [5, 5, 5, 7, 7]
        assert session.token_ids == tuple(binding.sessions[1]["tokens"])

    assert any(call[0] == "load_companion" for call in binding.calls)


def test_dflash_stop_boundary_and_callback_failure_preserve_mirrored_state(
    tmp_path: Path,
) -> None:
    artifact = _write_artifact(
        tmp_path / "artifact",
        speculative_decoding=True,
    )
    binding = FakeResidentBinding(speculative_decoding=True)
    with NativeInferenceModel.load(
        artifact,
        binding=binding,
        load_config=NativeModelLoadConfig(
            runtime="cpu",
            speculative_decoding="required",
        ),
    ) as model:
        stopped = model.create_session()
        stopped.prefill([1, 2])
        result = stopped.decode(
            GenerationConfig(
                max_new_tokens=5,
                temperature=0.0,
                stop_token_ids=(104,),
            )
        )
        assert result.token_ids == (102, 103, 104)
        assert result.finish_reason == "stop"
        assert stopped.token_ids == (1, 2, 102, 103, 104)

        callback_session = model.create_session()
        callback_session.prefill([8])

        def fail_callback(_event: GenerationEvent) -> None:
            raise RuntimeError("callback failed")

        with pytest.raises(RuntimeError, match="callback failed"):
            callback_session.decode(
                GenerationConfig(max_new_tokens=3, temperature=0.0),
                on_token=fail_callback,
            )
        assert callback_session.token_ids == tuple(binding.sessions[2]["tokens"])
        assert callback_session.token_ids == (8, 201, 202, 203)


def test_speculative_off_uses_target_decode_and_required_fails_without_binding_proof(
    tmp_path: Path,
) -> None:
    artifact = _write_artifact(
        tmp_path / "artifact",
        speculative_decoding=True,
    )
    binding = FakeResidentBinding(speculative_decoding=True)
    with NativeInferenceModel.load(
        artifact,
        binding=binding,
        load_config=NativeModelLoadConfig(runtime="cpu", speculative_decoding="off"),
    ) as model:
        session = model.create_session()
        session.prefill([1])
        assert session.decode(
            GenerationConfig(max_new_tokens=1, temperature=0.0)
        ).token_ids == (101,)
    assert not any(call[0] == "load_companion" for call in binding.calls)
    assert any(call[0] == "decode_one" for call in binding.calls)

    with pytest.raises(NativeInferenceCapabilityError, match="feature ABI"):
        NativeInferenceModel.load(
            artifact,
            binding=FakeResidentBinding(speculative_decoding=False),
            load_config=NativeModelLoadConfig(
                runtime="cpu",
                speculative_decoding="required",
            ),
        )


def test_exact_prefix_sync_truncates_suffix_and_rebuilds_after_front_trim(
    tmp_path: Path,
) -> None:
    binding = FakeResidentBinding()
    model = NativeInferenceModel.load(_write_artifact(tmp_path / "artifact"), binding=binding)
    session = model.create_session()

    assert session.prefill([1, 2, 3]) == {
        "prefix_tokens": 3,
        "prefix_reused": 0,
        "prefilled_tokens": 3,
    }
    assert session.prefill([1, 2, 4, 5]) == {
        "prefix_tokens": 4,
        "prefix_reused": 2,
        "prefilled_tokens": 2,
    }
    # [2, 4, 5] looks like an old suffix but its absolute positions changed.
    assert session.prefill([2, 4, 5]) == {
        "prefix_tokens": 3,
        "prefix_reused": 0,
        "prefilled_tokens": 3,
    }
    assert session.prefill([2, 4, 5])["prefix_reused"] == 3
    assert session.token_ids == (2, 4, 5)

    assert ("truncate", 1, 2) in binding.calls
    reset_index = binding.calls.index(("reset", 1))
    rebuilt_index = binding.calls.index(("prefill", 1, (2, 4, 5), 0))
    assert reset_index < rebuilt_index

    session.truncate(2)
    assert session.token_ids == (2, 4)
    session.reset()
    assert session.token_ids == ()
    assert session.cancelled is False
    model.close()


def test_token_callback_observes_committed_native_and_python_state_and_can_cancel(
    tmp_path: Path,
) -> None:
    binding = FakeResidentBinding()
    model = NativeInferenceModel.load(_write_artifact(tmp_path / "artifact"), binding=binding)
    session = model.create_session()
    session.prefill([10])
    observations: list[tuple[GenerationEvent, tuple[int, ...], tuple[int, ...]]] = []

    def after_commit(event: GenerationEvent) -> None:
        observations.append(
            (event, session.token_ids, tuple(binding.sessions[1]["tokens"]))
        )
        session.cancel()

    result = session.decode(GenerationConfig(max_new_tokens=4), on_token=after_commit)

    assert result.token_ids == (101,)
    assert result.finish_reason == "cancelled"
    assert result.cancelled is True
    assert len(observations) == 1
    event, python_tokens, native_tokens = observations[0]
    assert event.committed is True
    assert python_tokens == native_tokens == (10, 101)
    assert binding.cancel_calls == 1

    session.reset()
    resumed = session.decode(GenerationConfig(max_new_tokens=1))
    assert resumed.token_ids == (100,)
    model.close()


@pytest.mark.parametrize("temperature", [0.0, -0.0])
def test_exact_signed_zero_selects_strict_model_compute(temperature: float) -> None:
    config = GenerationConfig(temperature=temperature)
    assert config.strict_model_compute is True
    assert config.to_binding_payload()["strict_model_compute"] is True


def test_positive_subnormal_temperature_remains_standard(tmp_path: Path) -> None:
    binding = FakeResidentBinding()
    model = NativeInferenceModel.load(_write_artifact(tmp_path / "artifact"), binding=binding)
    session = model.create_session()

    config = GenerationConfig(temperature=1.0e-50)
    assert config.temperature > 0.0
    assert config.strict_model_compute is False
    session.decode(config)
    assert ("decode_one", 1, 100, 1.0e-50, False) in binding.calls
    model.close()


@pytest.mark.parametrize("temperature", [-1.0, -1.0e-50, float("nan"), float("inf"), float("-inf")])
def test_negative_and_nonfinite_temperatures_are_rejected(temperature: float) -> None:
    with pytest.raises(ValueError, match="temperature must be a finite number"):
        GenerationConfig(temperature=temperature)


def test_explicit_turboquant_fails_closed_when_either_proof_is_missing(
    tmp_path: Path,
) -> None:
    artifact_without_turboquant = _write_artifact(tmp_path / "artifact-one")
    binding_with_turboquant = FakeResidentBinding(turboquant=True)
    with pytest.raises(NativeInferenceCapabilityError, match="Explicit TurboQuant"):
        NativeInferenceModel.load(
            artifact_without_turboquant,
            binding=binding_with_turboquant,
            kv_cache=KVCacheConfig(mode="turboquant", turboquant_profile="qjl-3.5"),
        )
    assert binding_with_turboquant.model_loads == 0

    artifact_with_turboquant = _write_artifact(
        tmp_path / "artifact-two",
        turboquant_kv_cache=True,
    )
    binding_without_turboquant = FakeResidentBinding(turboquant=False)
    with pytest.raises(NativeInferenceCapabilityError, match="Explicit TurboQuant"):
        NativeInferenceModel.load(
            artifact_with_turboquant,
            binding=binding_without_turboquant,
            kv_cache=KVCacheConfig(mode="turboquant"),
        )
    assert binding_without_turboquant.model_loads == 0


def test_tile_cuda_turboquant_configuration_is_explicit_and_jointly_proven(
    tmp_path: Path,
) -> None:
    sidecar = tmp_path / "libtile-strict.so"
    sidecar.write_bytes(b"test-double")

    with pytest.raises(ValueError, match="requires mode='turboquant'"):
        KVCacheConfig(
            mode="full",
            turboquant_attention_backend="tile-cuda",
            tile_ops_lib=str(sidecar),
        )
    with pytest.raises(ValueError, match="requires tile_ops_lib"):
        KVCacheConfig(
            mode="turboquant",
            turboquant_attention_backend="tile-cuda",
        )
    with pytest.raises(ValueError, match="library/device options require"):
        KVCacheConfig(mode="turboquant", tile_ops_lib=str(sidecar))

    config = KVCacheConfig(
        mode="turboquant",
        turboquant_profile="qjl-3.5",
        turboquant_attention_backend="tile-cuda",
        tile_ops_lib=str(sidecar),
        cuda_runtime_lib="libcudart.so.13",
        cuda_device=2,
    )
    unproven_artifact = _write_artifact(
        tmp_path / "unproven",
        turboquant_kv_cache=True,
    )
    tile_binding = FakeResidentBinding(turboquant=True, tile_attention=True)
    with pytest.raises(NativeInferenceCapabilityError, match="do not both prove"):
        NativeInferenceModel.load(
            unproven_artifact,
            binding=tile_binding,
            kv_cache=config,
        )
    assert tile_binding.model_loads == 0

    artifact = _write_artifact(
        tmp_path / "proven",
        turboquant_kv_cache=True,
        turboquant_tile_attention=True,
    )
    binding = FakeResidentBinding(turboquant=True, tile_attention=True)
    with NativeInferenceModel.load(artifact, binding=binding, kv_cache=config) as model:
        configure = next(call for call in binding.calls if call[0] == "configure_tile")
        assert configure[1] == {
            "backend": "tile-cuda",
            "tile_ops_lib": str(sidecar.resolve()),
            "cuda_runtime_lib": "libcudart.so.13",
            "device": 2,
        }
        with model.create_session() as session:
            payload = binding.sessions[session._handle]["cache"]
            assert payload["effective_mode"] == "turboquant"
            assert payload["turboquant_attention_backend"] == "tile-cuda"
            assert payload["tables"]["dimension"] == 8

    assert sum(call[0] == "configure_tile" for call in binding.calls) == 1


def test_session_prefix_cow_is_jointly_proven_and_rejects_tile_models(
    tmp_path: Path,
) -> None:
    artifact_only = _write_artifact(
        tmp_path / "artifact-only-cow",
        session_prefix_cow=True,
    )
    with NativeInferenceModel.load(
        artifact_only,
        binding=FakeResidentBinding(session_prefix_cow=False),
    ) as model:
        source = model.create_session()
        source.prefill([1])
        assert model.capabilities.session_prefix_cow is False
        with pytest.raises(NativeInferenceCapabilityError, match="do not prove"):
            model.fork_session(source)

    binding_only = FakeResidentBinding(session_prefix_cow=True)
    with NativeInferenceModel.load(
        _write_artifact(tmp_path / "binding-only-cow"),
        binding=binding_only,
    ) as model:
        source = model.create_session()
        source.prefill([1])
        assert model.capabilities.session_prefix_cow is False
        with pytest.raises(NativeInferenceCapabilityError, match="do not prove"):
            model.fork_session(source)

    sidecar = tmp_path / "libtile-prefix-cow-test.so"
    sidecar.write_bytes(b"test-double")
    tile_artifact = _write_artifact(
        tmp_path / "tile-cow",
        turboquant_kv_cache=True,
        turboquant_tile_attention=True,
        session_prefix_cow=True,
        session_prefix_cow_cpu_turboquant=True,
    )
    tile_binding = FakeResidentBinding(
        turboquant=True,
        tile_attention=True,
        session_prefix_cow=True,
        session_prefix_cow_cpu_turboquant=True,
    )
    tile_config = KVCacheConfig(
        mode="turboquant",
        turboquant_attention_backend="tile-cuda",
        tile_ops_lib=str(sidecar),
    )
    with NativeInferenceModel.load(
        tile_artifact,
        binding=tile_binding,
        kv_cache=tile_config,
    ) as model:
        source = model.create_session()
        source.prefill([1])
        assert model.capabilities.session_prefix_cow is True
        assert model.capabilities.session_prefix_cow_cpu_turboquant is True
        assert model.stats()["session_prefix_cow_cpu_turboquant"] is True
        with pytest.raises(NativeInferenceCapabilityError, match="Tile-CUDA"):
            model.fork_session(source)
        assert not any(call[0] == "fork_session" for call in tile_binding.calls)


def test_cpu_turboquant_prefix_cow_is_additive_exact_and_accepts_tile_capable_artifact(
    tmp_path: Path,
) -> None:
    artifact = _write_artifact(
        tmp_path / "cpu-turboquant-cow",
        turboquant_kv_cache=True,
        turboquant_tile_attention=True,
        session_prefix_cow=True,
        session_prefix_cow_cpu_turboquant=True,
    )
    binding = FakeResidentBinding(
        turboquant=True,
        tile_attention=True,
        session_prefix_cow=True,
        session_prefix_cow_cpu_turboquant=True,
    )
    with NativeInferenceModel.load(
        artifact,
        binding=binding,
        kv_cache=KVCacheConfig(
            mode="turboquant",
            turboquant_profile="qjl-3.5",
            turboquant_attention_backend="cpu",
        ),
    ) as model:
        assert model.capabilities.session_prefix_cow is True
        assert model.capabilities.session_prefix_cow_cpu_turboquant is True
        source = model.create_session(seed=3)
        source.prefill([1, 2, 3])
        child = model.fork_session(source, token_count=2, seed=9)
        assert child.token_ids == (1, 2)
        assert child.stats()["effective_cache"] == "turboquant"
        assert not any(call[0] == "configure_tile" for call in binding.calls)

    mismatched = _write_artifact(
        tmp_path / "cpu-turboquant-cow-mismatched",
        turboquant_kv_cache=True,
        session_prefix_cow_cpu_turboquant=True,
        session_prefix_cow_cpu_turboquant_profile="future-packed-profile",
    )
    mismatched_binding = FakeResidentBinding(
        turboquant=True,
        session_prefix_cow_cpu_turboquant=True,
    )
    with NativeInferenceModel.load(
        mismatched,
        binding=mismatched_binding,
        kv_cache=KVCacheConfig(mode="turboquant"),
    ) as model:
        assert model.capabilities.session_prefix_cow_cpu_turboquant is False
        source = model.create_session()
        source.prefill([1])
        with pytest.raises(NativeInferenceCapabilityError, match="packed-TurboQuant"):
            model.fork_session(source)
        assert not any(
            call[0] == "fork_session" for call in mismatched_binding.calls
        )

    unsupported_binding = FakeResidentBinding(
        turboquant=True,
        session_prefix_cow_cpu_turboquant=True,
        session_prefix_cow_profiles=(SESSION_PREFIX_COW_PROFILE,),
    )
    with NativeInferenceModel.load(
        artifact,
        binding=unsupported_binding,
        kv_cache=KVCacheConfig(mode="turboquant"),
    ) as model:
        assert model.capabilities.session_prefix_cow_cpu_turboquant is False


@pytest.mark.parametrize(
    ("checkpoint_format", "profile"),
    (
        (
            "neuralfn.native_family_llama.f32.v1",
            LLAMA_SESSION_PREFIX_COW_PROFILE,
        ),
        (
            "neuralfn.native_family_standard_moe.f32.v1",
            STANDARD_MOE_SESSION_PREFIX_COW_PROFILE,
        ),
    ),
)
def test_family_prefix_cow_requires_exact_artifact_and_binding_profile(
    tmp_path: Path,
    checkpoint_format: str,
    profile: str,
) -> None:
    proven = _write_artifact(
        tmp_path / f"proven-{profile}",
        session_prefix_cow=True,
        checkpoint_format=checkpoint_format,
        session_prefix_cow_profile=profile,
    )
    with NativeInferenceModel.load(
        proven,
        binding=FakeResidentBinding(session_prefix_cow=True),
    ) as model:
        assert model.capabilities.session_prefix_cow is True

    unsupported_binding = FakeResidentBinding(
        session_prefix_cow=True,
        session_prefix_cow_profiles=(SESSION_PREFIX_COW_PROFILE,),
    )
    with NativeInferenceModel.load(proven, binding=unsupported_binding) as model:
        assert model.capabilities.session_prefix_cow is False

    mismatched = _write_artifact(
        tmp_path / f"mismatched-{profile}",
        session_prefix_cow=True,
        checkpoint_format=checkpoint_format,
        session_prefix_cow_profile=SESSION_PREFIX_COW_PROFILE,
    )
    with NativeInferenceModel.load(
        mismatched,
        binding=FakeResidentBinding(session_prefix_cow=True),
    ) as model:
        assert model.capabilities.session_prefix_cow is False


def test_prefix_cow_rejects_an_unrecognized_checkpoint_profile_cluster(
    tmp_path: Path,
) -> None:
    artifact = _write_artifact(
        tmp_path / "unsupported-prefix-cow-format",
        session_prefix_cow=True,
        checkpoint_format="neuralfn.native_future_model.v1",
        session_prefix_cow_profile=LLAMA_SESSION_PREFIX_COW_PROFILE,
    )
    with NativeInferenceModel.load(
        artifact,
        binding=FakeResidentBinding(session_prefix_cow=True),
    ) as model:
        assert model.capabilities.session_prefix_cow is False


def test_sdk_prefix_fork_rejects_cross_model_source(tmp_path: Path) -> None:
    first = NativeInferenceModel.load(
        _write_artifact(
            tmp_path / "cow-first",
            session_prefix_cow=True,
        ),
        binding=FakeResidentBinding(session_prefix_cow=True),
    )
    second = NativeInferenceModel.load(
        _write_artifact(
            tmp_path / "cow-second",
            session_prefix_cow=True,
        ),
        binding=FakeResidentBinding(session_prefix_cow=True),
    )
    try:
        source = first.create_session()
        source.prefill([1])
        with pytest.raises(ValueError, match="does not belong"):
            second.fork_session(source)
    finally:
        second.close()
        first.close()


@pytest.mark.parametrize("defect", ["feature_abi", "configure_callable"])
def test_cpu_default_does_not_advertise_unusable_tile_attention(
    tmp_path: Path,
    defect: str,
) -> None:
    artifact = _write_artifact(
        tmp_path / defect,
        turboquant_kv_cache=True,
        turboquant_tile_attention=True,
    )
    binding = FakeResidentBinding(turboquant=True, tile_attention=True)
    if defect == "feature_abi":
        manifest_path = artifact / "native-execution-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["kernel_abi"]["turboquant_tile_attention"]["status"] = (
            "not_implemented"
        )
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    else:
        binding.configure_model_turboquant_attention = None  # type: ignore[method-assign]

    with NativeInferenceModel.load(artifact, binding=binding) as model:
        assert model.capabilities.turboquant_tile_attention is False


def test_constrained_responses_capabilities_require_artifact_abi_and_binding_primitive(
    tmp_path: Path,
) -> None:
    artifact = _write_artifact(
        tmp_path / "ready",
        structured_output=True,
        function_tools=True,
    )
    with NativeInferenceModel.load(
        artifact,
        binding=FakeResidentBinding(constrained_decoding=True),
        kv_cache=KVCacheConfig(mode="off"),
    ) as model:
        assert model.capabilities.structured_output is True
        assert model.capabilities.function_tools is True

    with NativeInferenceModel.load(
        artifact,
        binding=FakeResidentBinding(constrained_decoding=False),
        kv_cache=KVCacheConfig(mode="off"),
    ) as model:
        assert model.capabilities.structured_output is False
        assert model.capabilities.function_tools is False


def test_constrained_responses_capabilities_fail_closed_on_profile_mismatch(
    tmp_path: Path,
) -> None:
    artifact = _write_artifact(
        tmp_path / "mismatch",
        structured_output=True,
        function_tools=True,
    )
    manifest_path = artifact / "native-execution-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["kernel_abi"]["structured_output"]["profile"] = "future-profile"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with NativeInferenceModel.load(
        artifact,
        binding=FakeResidentBinding(constrained_decoding=True),
        kv_cache=KVCacheConfig(mode="off"),
    ) as model:
        assert model.capabilities.structured_output is False
        assert model.capabilities.function_tools is False


@pytest.mark.parametrize("version_location", ["manifest", "resident", "binding"])
def test_boolean_versions_are_not_accepted_as_integer_abi_versions(
    tmp_path: Path,
    version_location: str,
) -> None:
    artifact = _write_artifact(tmp_path / version_location)
    binding = FakeResidentBinding()
    if version_location in {"manifest", "resident"}:
        manifest_path = artifact / "native-execution-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if version_location == "manifest":
            manifest["version"] = True
        else:
            manifest["kernel_abi"]["resident_inference"]["version"] = True
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    else:
        binding.resident_inference_abi_version = lambda: True  # type: ignore[method-assign]

    with pytest.raises(NativeInferenceCapabilityError, match="version|ABI mismatch"):
        NativeInferenceModel.load(artifact, binding=binding)


def test_turboquant_geometry_is_proved_during_model_load(tmp_path: Path) -> None:
    artifact = _write_artifact(
        tmp_path / "artifact",
        turboquant_kv_cache=True,
    )
    odd_binding = FakeResidentBinding(
        turboquant=True,
        channels=3,
        num_heads=1,
    )
    with pytest.raises(NativeInferenceCapabilityError, match="even attention head"):
        NativeInferenceModel.load(
            artifact,
            binding=odd_binding,
            kv_cache=KVCacheConfig(mode="turboquant"),
        )
    assert odd_binding.model_loads == 1
    assert odd_binding.model_closes == 1

    even_binding = FakeResidentBinding(
        turboquant=True,
        channels=8,
        num_heads=1,
    )
    with NativeInferenceModel.load(
        artifact,
        binding=even_binding,
        kv_cache=KVCacheConfig(mode="turboquant", turboquant_profile="qjl-3.5"),
    ) as model:
        with model.create_session() as session:
            payload = even_binding.sessions[session._handle]["cache"]
            assert payload["effective_mode"] == "turboquant"
            assert payload["turboquant_profile"] == "qjl-3.5"
            assert payload["tables"]["dimension"] == 8


@pytest.mark.parametrize(
    ("artifact_kwargs", "message"),
    [
        ({"resident_inference": False}, "does not prove native resident inference"),
        ({"native_inference": False}, "does not prove native resident inference"),
        ({"resident_abi_version": None}, "ABI is not proven ready"),
        ({"resident_abi_status": "not_implemented"}, "ABI is not proven ready"),
    ],
)
def test_artifact_must_prove_resident_capability_and_abi(
    tmp_path: Path,
    artifact_kwargs: dict[str, Any],
    message: str,
) -> None:
    artifact = _write_artifact(tmp_path / "artifact", **artifact_kwargs)
    binding = FakeResidentBinding()
    with pytest.raises(NativeInferenceCapabilityError, match=message):
        NativeInferenceModel.load(artifact, binding=binding, kv_cache=KVCacheConfig(mode="off"))
    assert binding.model_loads == 0


def test_checkpoint_fingerprint_and_containment_are_verified_before_binding_load(
    tmp_path: Path,
) -> None:
    binding = FakeResidentBinding()
    artifact = _write_artifact(tmp_path / "artifact")
    checkpoint = artifact / "checkpoint.bin"
    checkpoint.write_bytes(b"x" * checkpoint.stat().st_size)
    with pytest.raises(NativeInferenceCapabilityError, match="checksum"):
        NativeInferenceModel.load(
            artifact,
            binding=binding,
            kv_cache=KVCacheConfig(mode="off"),
        )
    assert binding.model_loads == 0

    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    manifest_path = artifact / "native-execution-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checkpoint"].update(
        {
            "artifact_path": "../outside.bin",
            "target_nbytes": outside.stat().st_size,
            "target_sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(NativeInferenceCapabilityError, match="escapes"):
        NativeInferenceModel.load(
            artifact,
            binding=binding,
            kv_cache=KVCacheConfig(mode="off"),
        )
    assert binding.model_loads == 0


def test_missing_binding_fails_without_subprocess_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import neuralfn.native_inference as native_inference

    artifact = _write_artifact(tmp_path / "artifact")

    def missing_binding(name: str):
        error = ModuleNotFoundError(f"No module named {name!r}")
        error.name = name
        raise error

    monkeypatch.setattr(native_inference.importlib, "import_module", missing_binding)
    with pytest.raises(NativeInferenceCapabilityError, match="No subprocess fallback"):
        NativeInferenceModel.load(artifact)


def test_close_includes_create_already_in_flight_before_its_snapshot(
    tmp_path: Path,
) -> None:
    binding = CoordinatedResidentBinding()
    model = NativeInferenceModel.load(
        _write_artifact(tmp_path / "create-before-close"),
        binding=binding,
    )
    binding.block_create = True
    created: list[Any] = []
    create_errors: list[BaseException] = []
    close_errors: list[BaseException] = []
    close_started = threading.Event()

    creator = threading.Thread(
        target=_thread_call,
        args=(model.create_session, created, create_errors),
        name="native-create",
        daemon=True,
    )

    def close_model() -> None:
        close_started.set()
        model.close()

    closer = threading.Thread(
        target=_thread_call,
        args=(close_model, [], close_errors),
        name="native-close-after-create",
        daemon=True,
    )
    creator.start()
    try:
        assert binding.create_entered.wait(timeout=5)
        closer.start()
        assert close_started.wait(timeout=5)
        assert closer.is_alive()
    finally:
        binding.allow_create.set()
    _join_thread(creator)
    _join_thread(closer)

    assert create_errors == []
    assert close_errors == []
    assert len(created) == 1
    assert created[0].closed is True
    assert binding.session_closes == 1
    assert binding.model_closes == 1
    close_session_index = binding.calls.index(("close_session", 1))
    close_model_index = binding.calls.index(("close_model",))
    assert close_session_index < close_model_index

    model.close()
    assert binding.session_closes == 1
    assert binding.model_closes == 1


def test_create_fails_after_close_snapshot_without_allocating_a_handle(
    tmp_path: Path,
) -> None:
    binding = CoordinatedResidentBinding()
    model = NativeInferenceModel.load(
        _write_artifact(tmp_path / "close-before-create"),
        binding=binding,
    )
    existing = model.create_session()
    binding.block_close_session = True
    close_errors: list[BaseException] = []
    closer = threading.Thread(
        target=_thread_call,
        args=(model.close, [], close_errors),
        name="native-close-before-create",
        daemon=True,
    )
    closer.start()
    try:
        assert binding.close_session_entered.wait(timeout=5)
        with pytest.raises(NativeInferenceClosedError):
            model.create_session()
        # A concurrent duplicate close does not become a second model owner.
        model.close()
        assert binding.model_closes == 0
    finally:
        binding.allow_close_session.set()
        _join_thread(closer)

    assert close_errors == []
    assert existing.closed is True
    assert sum(call[0] == "create_session" for call in binding.calls) == 1
    assert binding.session_closes == 1
    assert binding.model_closes == 1


def test_session_operation_fails_after_close_boundary_before_binding_compute(
    tmp_path: Path,
) -> None:
    binding = CoordinatedResidentBinding()
    model = NativeInferenceModel.load(
        _write_artifact(tmp_path / "session-operation-after-close"),
        binding=binding,
    )
    first = model.create_session()
    second = model.create_session()

    class OrderedSessionRegistry:
        def __init__(self, sessions: list[Any]) -> None:
            self._sessions = list(sessions)

        def __iter__(self):
            return iter(tuple(self._sessions))

        def add(self, session: Any) -> None:
            self._sessions.append(session)

        def discard(self, session: Any) -> None:
            if session in self._sessions:
                self._sessions.remove(session)

        def clear(self) -> None:
            self._sessions.clear()

    # Make the close order explicit so teardown blocks on session A while B
    # remains registered but has not yet entered NativeInferenceSession.close.
    model._sessions = OrderedSessionRegistry([first, second])  # type: ignore[assignment]
    binding.block_close_session = True
    binding.block_close_session_handle = first._handle
    close_errors: list[BaseException] = []
    closer = threading.Thread(
        target=_thread_call,
        args=(model.close, [], close_errors),
        name="native-close-before-session-operation",
        daemon=True,
    )
    closer.start()
    duplicate_results: list[Any] = []
    duplicate_errors: list[BaseException] = []
    duplicate: threading.Thread | None = None
    try:
        assert binding.close_session_entered.wait(timeout=5)
        with pytest.raises(NativeInferenceClosedError):
            second.prefill([9])
        assert not any(
            call[0] == "prefill" and call[1] == second._handle
            for call in binding.calls
        )

        # Waiting for the owner here is unsafe in the general case: a
        # duplicate caller may hold a session operation lock needed by it.
        def duplicate_close_with_session_lock() -> None:
            with second._operation_lock:
                model.close()

        duplicate = threading.Thread(
            target=_thread_call,
            args=(
                duplicate_close_with_session_lock,
                duplicate_results,
                duplicate_errors,
            ),
            name="native-duplicate-close",
            daemon=True,
        )
        duplicate.start()
        _join_thread(duplicate)
        assert closer.is_alive()
        assert duplicate_results == [None]
        assert duplicate_errors == []
        assert binding.model_closes == 0
    finally:
        binding.allow_close_session.set()
        if duplicate is not None and duplicate.is_alive():
            _join_thread(duplicate)
        _join_thread(closer)

    assert close_errors == []
    assert first.closed is True
    assert second.closed is True
    closed_handles = [call[1] for call in binding.calls if call[0] == "close_session"]
    assert closed_handles == [first._handle, second._handle]
    assert binding.model_closes == 1


def test_close_includes_fork_already_in_flight_before_its_snapshot(
    tmp_path: Path,
) -> None:
    binding = CoordinatedResidentBinding(session_prefix_cow=True)
    model = NativeInferenceModel.load(
        _write_artifact(
            tmp_path / "fork-before-close",
            session_prefix_cow=True,
        ),
        binding=binding,
    )
    source = model.create_session()
    source.prefill([1, 2, 3])
    binding.block_fork = True
    forked: list[Any] = []
    fork_errors: list[BaseException] = []
    close_errors: list[BaseException] = []
    close_started = threading.Event()
    forker = threading.Thread(
        target=_thread_call,
        args=(lambda: model.fork_session(source), forked, fork_errors),
        name="native-fork",
        daemon=True,
    )

    def close_model() -> None:
        close_started.set()
        model.close()

    closer = threading.Thread(
        target=_thread_call,
        args=(close_model, [], close_errors),
        name="native-close-after-fork",
        daemon=True,
    )
    forker.start()
    try:
        assert binding.fork_entered.wait(timeout=5)
        closer.start()
        assert close_started.wait(timeout=5)
        assert closer.is_alive()
    finally:
        binding.allow_fork.set()
    _join_thread(forker)
    _join_thread(closer)

    assert fork_errors == []
    assert close_errors == []
    assert len(forked) == 1
    assert source.closed is True
    assert forked[0].closed is True
    closed_handles = [call[1] for call in binding.calls if call[0] == "close_session"]
    assert sorted(closed_handles) == [1, 2]
    assert binding.model_closes == 1

    model.close()
    assert binding.session_closes == 2
    assert binding.model_closes == 1


def test_fork_that_reaches_registration_after_close_snapshot_fails_closed(
    tmp_path: Path,
) -> None:
    binding = FakeResidentBinding(session_prefix_cow=True)
    model = NativeInferenceModel.load(
        _write_artifact(
            tmp_path / "close-before-fork-register",
            session_prefix_cow=True,
        ),
        binding=binding,
    )
    source = model.create_session()
    source.prefill([1, 2, 3])
    snapshot_taken = threading.Event()

    class SnapshotSignalingRegistry:
        def __init__(self, sessions: list[Any]) -> None:
            self._sessions = set(sessions)

        def __iter__(self):
            snapshot_taken.set()
            return iter(self._sessions)

        def add(self, session: Any) -> None:
            self._sessions.add(session)

        def discard(self, session: Any) -> None:
            self._sessions.discard(session)

        def clear(self) -> None:
            self._sessions.clear()

    model._sessions = SnapshotSignalingRegistry([source])  # type: ignore[assignment]
    fork_validated = threading.Event()
    allow_fork_to_register = threading.Event()
    original_ensure_usable = source._ensure_usable

    def pause_after_source_validation() -> None:
        original_ensure_usable()
        fork_validated.set()
        if not allow_fork_to_register.wait(timeout=5):
            raise RuntimeError("timed out waiting to continue fork_session")

    source._ensure_usable = pause_after_source_validation  # type: ignore[method-assign]
    forked: list[Any] = []
    fork_errors: list[BaseException] = []
    close_errors: list[BaseException] = []
    forker = threading.Thread(
        target=_thread_call,
        args=(lambda: model.fork_session(source), forked, fork_errors),
        name="native-fork-after-snapshot",
        daemon=True,
    )
    closer = threading.Thread(
        target=_thread_call,
        args=(model.close, [], close_errors),
        name="native-close-before-fork-register",
        daemon=True,
    )
    forker.start()
    assert fork_validated.wait(timeout=5)
    closer.start()
    try:
        assert snapshot_taken.wait(timeout=5)
    finally:
        allow_fork_to_register.set()
    _join_thread(forker)
    _join_thread(closer)

    assert forked == []
    assert len(fork_errors) == 1
    assert isinstance(fork_errors[0], NativeInferenceClosedError)
    assert close_errors == []
    assert sum(call[0] == "fork_session" for call in binding.calls) == 0
    assert binding.session_closes == 1
    assert binding.model_closes == 1


def test_create_closes_native_handle_when_python_session_construction_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import neuralfn.native_inference as native_inference

    binding = FakeResidentBinding()
    model = NativeInferenceModel.load(
        _write_artifact(tmp_path / "constructor-failure"),
        binding=binding,
    )

    def fail_session_construction(**_kwargs: Any) -> Any:
        raise RuntimeError("session construction failed")

    with monkeypatch.context() as patch:
        patch.setattr(native_inference, "NativeInferenceSession", fail_session_construction)
        with pytest.raises(RuntimeError, match="session construction failed"):
            model.create_session()

    assert [call for call in binding.calls if call[0] == "close_session"] == [
        ("close_session", 1)
    ]
    model.close()
    assert binding.session_closes == 1
    assert binding.model_closes == 1


@pytest.mark.parametrize("operation", ["create", "fork"])
def test_new_native_handle_is_closed_when_python_session_registration_fails(
    tmp_path: Path,
    operation: str,
) -> None:
    prefix_cow = operation == "fork"
    binding = FakeResidentBinding(session_prefix_cow=prefix_cow)
    model = NativeInferenceModel.load(
        _write_artifact(
            tmp_path / f"registration-failure-{operation}",
            session_prefix_cow=prefix_cow,
        ),
        binding=binding,
    )
    source = None
    registered: set[Any] = set()
    if operation == "fork":
        source = model.create_session()
        source.prefill([1, 2])
        registered.add(source)

    class RejectingSessionRegistry:
        def __iter__(self):
            return iter(registered)

        def add(self, _session: Any) -> None:
            raise RuntimeError("session registration failed")

        def discard(self, session: Any) -> None:
            registered.discard(session)

        def clear(self) -> None:
            registered.clear()

    model._sessions = RejectingSessionRegistry()  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="session registration failed"):
        if source is None:
            model.create_session()
        else:
            model.fork_session(source)

    target_handle = 1 if source is None else 2
    closed_handles = [call[1] for call in binding.calls if call[0] == "close_session"]
    assert closed_handles.count(target_handle) == 1
    model.close()
    closed_handles = [call[1] for call in binding.calls if call[0] == "close_session"]
    assert closed_handles.count(target_handle) == 1
    if source is not None:
        assert closed_handles.count(1) == 1
    assert binding.model_closes == 1


def test_close_is_idempotent_and_rejects_later_operations(tmp_path: Path) -> None:
    binding = FakeResidentBinding()
    model = NativeInferenceModel.load(_write_artifact(tmp_path / "artifact"), binding=binding)
    session = model.create_session()

    session.close()
    session.close()
    assert binding.session_closes == 1
    with pytest.raises(NativeInferenceClosedError):
        session.prefill([1])

    model.close()
    model.close()
    assert binding.model_closes == 1
    with pytest.raises(NativeInferenceClosedError):
        model.create_session()


def test_required_contract_dataclasses_are_frozen() -> None:
    values = [
        NativeInferenceCapabilities(True, True, True, False),
        GenerationConfig(),
        KVCacheConfig(),
        GenerationEvent(token_id=1, index=0, position=0),
        GenerationResult((), "", "length", 0, 0),
    ]
    for value in values:
        field_name = fields(value)[0].name
        with pytest.raises(FrozenInstanceError):
            setattr(value, field_name, None)


def test_resident_contract_is_available_from_lazy_top_level_exports() -> None:
    import neuralfn

    assert neuralfn.NativeInferenceModel is NativeInferenceModel
    assert neuralfn.NativeInferenceCapabilities is NativeInferenceCapabilities
    assert neuralfn.GenerationConfig is GenerationConfig
    assert neuralfn.GenerationEvent is GenerationEvent
    assert neuralfn.GenerationResult is GenerationResult
    assert neuralfn.KVCacheConfig is KVCacheConfig


def test_import_is_lean_and_module_has_no_subprocess_dependency() -> None:
    script = (
        "import sys; import neuralfn.native_inference; "
        "print(','.join(name for name in ('torch','numpy','networkx','subprocess') "
        "if name in sys.modules))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == ""
