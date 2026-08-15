from __future__ import annotations

import asyncio
import base64
from io import BytesIO
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
from typing import Any, Mapping, Sequence

import anyio
import httpx
import pytest
from PIL import Image

from neuralfn.native_inference import (
    GenerationEvent,
    GenerationResult,
    NativeInferenceCapabilities,
)
from neuralfn.native_serve import (
    BearerAuth,
    BoundedSingleWorkerQueue,
    NativeServeConfig,
    NativeServingConfigurationError,
    NativeServingRuntime,
    _NativeChatRendererAdapter,
    _PlainRolesRenderer,
    _TextCodec,
    create_native_inference_app,
    resolve_bearer_auth,
    run_native_inference_server,
)
from neuralfn.native_chat import MuseGlimmerATEMRenderer
from neuralfn.native_state import NativeStateStore, api_key_fingerprint


class FakeCodec(_TextCodec):
    name = "fake-test-codec"

    def encode(self, text: str) -> tuple[int, ...]:
        assert "<|user|>" in text
        assert text.endswith("<|assistant|>\n")
        return (1, 2, 3)

    def decode(self, token_ids: Sequence[int]) -> str:
        return b"".join(self.token_bytes(token) for token in token_ids).decode("utf-8")

    def token_bytes(self, token_id: int) -> bytes:
        return {10: b"Hello", 11: b"!"}[token_id]


class RecordingCodec(FakeCodec):
    def __init__(self) -> None:
        self.rendered: list[str] = []

    def encode(self, text: str) -> tuple[int, ...]:
        self.rendered.append(text)
        return super().encode(text)


class MediaCodec(FakeCodec):
    def encode(self, text: str) -> tuple[int, ...]:
        assert "<|image_start|>" in text and "<|image_end|>" in text
        return (1, *(200_092 for _ in range(text.count("<|patch|>"))), 3)


class ATEMCodec(_TextCodec):
    name = "atem-test-codec"

    def encode(self, _text: str) -> tuple[int, ...]:
        return (1, 2, 3)

    def decode(self, token_ids: Sequence[int]) -> str:
        return b"".join(self.token_bytes(token) for token in token_ids).decode("utf-8")

    def token_bytes(self, token_id: int) -> bytes:
        return {
            10: b" to=self<|message|>private reasoning<|eom|>",
            11: b"<|start|>assistant to=user<|message|>ready<|eot|>",
        }[token_id]


class FakeSession:
    def __init__(self, model: "FakeModel") -> None:
        self.model = model
        self.tokens: list[int] = []
        self.cancelled = False
        self.closed = False

    def __enter__(self) -> "FakeSession":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def prefill(self, token_ids: Sequence[int]) -> dict[str, int]:
        self.tokens = list(token_ids)
        self.model.prefills.append(tuple(token_ids))
        return {
            "prefix_tokens": len(token_ids),
            "prefix_reused": 0,
            "prefilled_tokens": len(token_ids),
        }

    def prefill_with_embeddings(
        self,
        token_ids: Sequence[int],
        *,
        replacement_positions: Sequence[int],
        replacement_embeddings: Sequence[Sequence[float]],
    ) -> dict[str, int]:
        self.model.embedding_prefills.append(
            (
                tuple(token_ids),
                tuple(replacement_positions),
                tuple(tuple(row) for row in replacement_embeddings),
            )
        )
        return self.prefill(token_ids)

    def decode(self, generation, *, on_token=None) -> GenerationResult:
        self.model.started.set()
        if self.model.release is not None:
            assert self.model.release.wait(timeout=5)
        generated: list[int] = []
        events: list[GenerationEvent] = []
        for index, token_id in enumerate((10, 11)[: generation.max_new_tokens]):
            if self.cancelled:
                break
            self.tokens.append(token_id)
            event = GenerationEvent(
                token_id=token_id,
                index=index,
                position=len(self.tokens) - 1,
            )
            generated.append(token_id)
            events.append(event)
            if on_token is not None:
                on_token(event)
        finish = "cancelled" if self.cancelled else ("stop" if len(generated) == 2 else "length")
        return GenerationResult(
            token_ids=tuple(generated),
            text="",
            finish_reason=finish,
            prompt_tokens=len(self.tokens) - len(generated),
            completion_tokens=len(generated),
            events=tuple(events),
            cancelled=self.cancelled,
        )

    def cancel(self) -> None:
        self.cancelled = True

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.model.session_closes += 1


class FakeModel:
    def __init__(self, *, release: threading.Event | None = None) -> None:
        self.capabilities = NativeInferenceCapabilities(
            native_inference=True,
            resident_inference=True,
            lossless_kv_cache=True,
            turboquant_kv_cache=False,
            session_state_kinds=("token_history",),
        )
        self.release = release
        self.started = threading.Event()
        self.prefills: list[tuple[int, ...]] = []
        self.embedding_prefills: list[
            tuple[tuple[int, ...], tuple[int, ...], tuple[tuple[float, ...], ...]]
        ] = []
        self.session_creates = 0
        self.session_closes = 0
        self.model_closes = 0

    def create_session(self) -> FakeSession:
        self.session_creates += 1
        return FakeSession(self)

    def stats(self) -> dict[str, Any]:
        return {
            "backend": "fake-test-only",
            "weights_load_count": 1,
            "subprocess_spawns": 0,
            "requested_cache": "auto",
            "effective_cache": "full",
        }

    def encode_media(
        self,
        packed_patches: Sequence[Sequence[float]],
        grid_thw: Sequence[Sequence[int]],
    ) -> tuple[tuple[float, ...], ...]:
        rows = sum(int(t) * int(h) * int(w) // 4 for t, h, w in grid_thw)
        return tuple((float(index),) * 8 for index in range(rows))

    def close(self) -> None:
        self.model_closes += 1


class MediaModel(FakeModel):
    def __init__(self) -> None:
        super().__init__()
        self.capabilities = NativeInferenceCapabilities(
            native_inference=True,
            resident_inference=True,
            lossless_kv_cache=True,
            turboquant_kv_cache=False,
            vision=True,
            vision_cpu=True,
            session_state_kinds=("token_history",),
        )

    def stats(self) -> dict[str, Any]:
        return {
            **super().stats(),
            "vision_loaded": True,
            "vision_resident_weight_bytes": 3_843_691_520,
        }


class PrefixSession(FakeSession):
    def __init__(
        self,
        model: "PrefixModel",
        tokens: Sequence[int] = (),
        *,
        cached_tokens: int = 0,
        shared_bytes: int = 0,
    ) -> None:
        super().__init__(model)
        self.tokens = list(tokens)
        self._native_cached_tokens = cached_tokens
        self.shared_bytes = shared_bytes
        self.detached_bytes = 0

    @property
    def token_ids(self) -> tuple[int, ...]:
        return tuple(self.tokens)

    def prefill(self, token_ids: Sequence[int]) -> dict[str, int]:
        requested = tuple(token_ids)
        current = tuple(self.tokens)
        assert requested[: len(current)] == current
        appended = requested[len(current) :]
        if appended and self.shared_bytes:
            self.detached_bytes += 64
            self.shared_bytes = 0
        self.tokens.extend(appended)
        self._native_cached_tokens = len(self.tokens)
        self.model.prefills.append(requested)
        return {
            "prefix_tokens": len(requested),
            "prefix_reused": len(current),
            "prefilled_tokens": len(appended),
        }

    def decode(self, generation, *, on_token=None) -> GenerationResult:
        result = super().decode(generation, on_token=on_token)
        self._native_cached_tokens = len(self.tokens)
        return result

    def stats(self) -> dict[str, int]:
        return {
            "cached_tokens": self._native_cached_tokens,
            "cache_capacity_bytes": 64,
            "prefix_cow_shared_capacity_bytes": self.shared_bytes,
            "prefix_cow_detached_capacity_bytes": self.detached_bytes,
        }


class PrefixModel(FakeModel):
    def __init__(self, *, release: threading.Event | None = None) -> None:
        super().__init__(release=release)
        self.capabilities = NativeInferenceCapabilities(
            native_inference=True,
            resident_inference=True,
            lossless_kv_cache=True,
            turboquant_kv_cache=False,
            session_state_kinds=("token_history",),
            session_prefix_cow=True,
        )
        self.sessions: list[PrefixSession] = []
        self.forks: list[tuple[PrefixSession, int, int]] = []

    def create_session(self, *, seed: int = 0) -> PrefixSession:
        del seed
        self.session_creates += 1
        session = PrefixSession(self)
        self.sessions.append(session)
        return session

    def fork_session(
        self,
        source: PrefixSession,
        *,
        token_count: int | None = None,
        seed: int = 0,
    ) -> PrefixSession:
        assert token_count is not None
        self.forks.append((source, token_count, seed))
        source.shared_bytes = 64
        session = PrefixSession(
            self,
            source.token_ids[:token_count],
            cached_tokens=token_count,
            shared_bytes=64,
        )
        self.sessions.append(session)
        return session


class BranchingCodec(FakeCodec):
    def encode(self, text: str) -> tuple[int, ...]:
        assert "<|user|>" in text
        assert text.endswith("<|assistant|>\n")
        if "Sibling" in text:
            return (1, 2, 3, 5)
        if "Branch" in text:
            return (1, 2, 3, 4)
        return (1, 2, 3)


class PrintableASCIICodec(_TextCodec):
    """Byte-exact test codec with one token for each 7-bit ASCII byte."""

    name = "printable-ascii-test-codec"
    vocab_size = 128

    def __init__(self, *, missing_byte: int | None = None) -> None:
        self.missing_byte = missing_byte
        self.rendered: list[str] = []

    def encode(self, text: str) -> tuple[int, ...]:
        self.rendered.append(text)
        return (1, 2, 3)

    def decode(self, token_ids: Sequence[int]) -> str:
        return b"".join(self.token_bytes(token_id) for token_id in token_ids).decode(
            "ascii"
        )

    def token_bytes(self, token_id: int) -> bytes:
        if token_id == self.missing_byte:
            return b"not-one-byte"
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise KeyError(token_id)
        if not 0 <= token_id < self.vocab_size:
            raise KeyError(token_id)
        return bytes((token_id,))


class ConstrainedSession:
    def __init__(
        self,
        model: "ConstrainedModel",
        *,
        kind: str,
        output: bytes,
    ) -> None:
        self.model = model
        self.kind = kind
        self.output = output
        self.tokens: list[int] = []
        self.prompt_length: int | None = None
        self.current_logits_calls = 0
        self.decode_calls = 0
        self.cancelled = False
        self.closed = False

    def __enter__(self) -> "ConstrainedSession":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    @property
    def token_ids(self) -> tuple[int, ...]:
        return tuple(self.tokens)

    def prefill(self, token_ids: Sequence[int]) -> dict[str, int]:
        requested = tuple(token_ids)
        if self.prompt_length is None:
            self.tokens[:] = requested
            self.prompt_length = len(requested)
            appended = len(requested)
            reused = 0
        else:
            assert requested[:-1] == tuple(self.tokens)
            assert len(requested) == len(self.tokens) + 1
            self.tokens[:] = requested
            appended = 1
            reused = len(requested) - 1
        self.model.prefills.append(requested)
        return {
            "prefix_tokens": len(requested),
            "prefix_reused": reused,
            "prefilled_tokens": appended,
        }

    def current_logits(self) -> tuple[float, ...]:
        assert self.kind == "constrained"
        assert self.prompt_length is not None
        index = len(self.tokens) - self.prompt_length
        assert 0 <= index < len(self.output)
        self.current_logits_calls += 1
        self.model.current_logits_calls += 1
        logits = [-1000.0] * PrintableASCIICodec.vocab_size
        # NUL is the unconstrained global argmax but can never be accepted by
        # the printable-ASCII grammar.
        logits[0] = 1000.0
        logits[self.output[index]] = 100.0
        self.model.unconstrained_argmaxes.append(0)
        return tuple(logits)

    def decode(self, generation, *, on_token=None) -> GenerationResult:
        self.decode_calls += 1
        self.model.decode_calls += 1
        if self.kind != "text":
            raise AssertionError("constrained Responses must not call ordinary decode")
        assert self.prompt_length is not None
        emitted = self.output[: generation.max_new_tokens]
        events: list[GenerationEvent] = []
        for index, token_id in enumerate(emitted):
            self.tokens.append(token_id)
            event = GenerationEvent(
                token_id=token_id,
                index=index,
                position=len(self.tokens) - 1,
                text=chr(token_id),
                finish_reason="stop" if index + 1 == len(self.output) else None,
            )
            events.append(event)
            if on_token is not None:
                on_token(event)
        finish_reason = "stop" if len(emitted) == len(self.output) else "length"
        return GenerationResult(
            token_ids=tuple(emitted),
            text=emitted.decode("ascii"),
            finish_reason=finish_reason,
            prompt_tokens=self.prompt_length,
            completion_tokens=len(emitted),
            events=tuple(events),
            cancelled=False,
        )

    def cancel(self) -> None:
        self.cancelled = True

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.model.session_closes += 1


class ConstrainedModel:
    """Scripted model proving constrained and ordinary paths stay separate."""

    def __init__(
        self,
        *,
        structured_output: bool = True,
        function_tools: bool = True,
    ) -> None:
        self.capabilities = NativeInferenceCapabilities(
            native_inference=True,
            resident_inference=True,
            lossless_kv_cache=True,
            turboquant_kv_cache=False,
            structured_output=structured_output,
            function_tools=function_tools,
            session_state_kinds=("token_history",),
        )
        self.pending_outputs: list[tuple[str, bytes]] = []
        self.sessions: list[ConstrainedSession] = []
        self.prefills: list[tuple[int, ...]] = []
        self.unconstrained_argmaxes: list[int] = []
        self.session_creates = 0
        self.session_closes = 0
        self.model_closes = 0
        self.current_logits_calls = 0
        self.decode_calls = 0
        self.function_executions = 0

    def queue_constrained(self, output: bytes | str) -> None:
        encoded = output.encode("ascii") if isinstance(output, str) else bytes(output)
        assert encoded and all(0x20 <= value <= 0x7E for value in encoded)
        self.pending_outputs.append(("constrained", encoded))

    def queue_text(self, output: str) -> None:
        encoded = output.encode("ascii")
        assert encoded and all(0x20 <= value <= 0x7E for value in encoded)
        self.pending_outputs.append(("text", encoded))

    def create_session(self) -> ConstrainedSession:
        self.session_creates += 1
        assert self.pending_outputs, "test must queue an output before model execution"
        kind, output = self.pending_outputs.pop(0)
        session = ConstrainedSession(self, kind=kind, output=output)
        self.sessions.append(session)
        return session

    def stats(self) -> dict[str, Any]:
        return {
            "backend": "printable-ascii-constrained-test",
            "vocab_size": PrintableASCIICodec.vocab_size,
            "weights_load_count": 1,
            "subprocess_spawns": 0,
            "requested_cache": "auto",
            "effective_cache": "full",
        }

    def close(self) -> None:
        self.model_closes += 1


class FailOnceSession(FakeSession):
    def __init__(self, model: "FailOnceModel", *, should_fail: bool) -> None:
        super().__init__(model)
        self.should_fail = should_fail
        self.poisoned = False

    def decode(self, generation, *, on_token=None) -> GenerationResult:
        if self.should_fail:
            self.poisoned = True
            raise RuntimeError("simulated poisoned native session")
        return super().decode(generation, on_token=on_token)


class FailOnceModel(FakeModel):
    def __init__(self) -> None:
        super().__init__()
        self.fail_next_session = True
        self.sessions: list[FailOnceSession] = []

    def create_session(self) -> FailOnceSession:
        self.session_creates += 1
        session = FailOnceSession(self, should_fail=self.fail_next_session)
        self.fail_next_session = False
        self.sessions.append(session)
        return session


def _runtime(model: FakeModel | None = None, *, context_limit: int = 64) -> NativeServingRuntime:
    return NativeServingRuntime(
        model=model or FakeModel(),  # type: ignore[arg-type]
        manifest={"schema": "neuralfn.native_execution_manifest", "version": 1},
        codec=FakeCodec(),
        renderer=_PlainRolesRenderer(),
        served_model_name="nfn-test",
        context_limit=context_limit,
        max_output_tokens=8,
        created=1_700_000_000,
    )


def _atem_runtime() -> NativeServingRuntime:
    return NativeServingRuntime(
        model=FakeModel(),  # type: ignore[arg-type]
        manifest={"schema": "neuralfn.native_execution_manifest", "version": 1},
        codec=ATEMCodec(),
        renderer=_NativeChatRendererAdapter(
            MuseGlimmerATEMRenderer(current_date="2026-08-15")
        ),
        served_model_name="nfn-test",
        context_limit=64,
        max_output_tokens=8,
        created=1_700_000_000,
    )


def _stateful_runtime(
    path: Path,
    model: FakeModel | None = None,
    *,
    codec: _TextCodec | None = None,
    prefix_cache_capacity: int = 0,
) -> NativeServingRuntime:
    return NativeServingRuntime(
        model=model or FakeModel(),  # type: ignore[arg-type]
        manifest={"schema": "neuralfn.native_execution_manifest", "version": 1},
        codec=codec or FakeCodec(),
        renderer=_PlainRolesRenderer(),
        served_model_name="nfn-test",
        context_limit=64,
        max_output_tokens=8,
        state_store=NativeStateStore(path),
        prefix_cache_capacity=prefix_cache_capacity,
        created=1_700_000_000,
    )


def _constrained_stateful_runtime(
    path: Path,
    *,
    model: ConstrainedModel | None = None,
    codec: PrintableASCIICodec | None = None,
    structured_output: bool = True,
    function_tools: bool = True,
    structured_profile: bool = True,
    tool_template: bool | Mapping[str, Any] = True,
    chat_template_selection: str = "auto",
) -> NativeServingRuntime:
    effective_structured = structured_output and structured_profile
    effective_function_tools = function_tools and effective_structured
    selected_model = model or ConstrainedModel(
        structured_output=effective_structured,
        function_tools=effective_function_tools,
    )
    selected_codec = codec or PrintableASCIICodec()
    chat_template: dict[str, Any] = {"format": "plain_roles"}
    if tool_template is True:
        chat_template["tool_template"] = {
            "version": 1,
            "profile": "responses-forced-function-call-v1",
        }
    elif isinstance(tool_template, Mapping):
        chat_template["tool_template"] = dict(tool_template)
    tokenizer: dict[str, Any] = {}
    if structured_profile:
        tokenizer["constrained_decoding"] = {
            "version": 1,
            "profile": "json-schema-ascii-byte-greedy-v1",
            "token_selection": "current_logits_exact_prefill",
        }
    return NativeServingRuntime(
        model=selected_model,  # type: ignore[arg-type]
        manifest={
            "schema": "neuralfn.native_execution_manifest",
            "version": 1,
            "tokenizer": tokenizer,
            "chat_template": chat_template,
        },
        codec=selected_codec,
        renderer=_PlainRolesRenderer(),
        served_model_name="nfn-test",
        context_limit=512,
        max_output_tokens=256,
        state_store=NativeStateStore(path),
        created=1_700_000_000,
        chat_template_selection=chat_template_selection,
    )


def _native_state_row_counts(path: Path) -> tuple[int, int, int, int]:
    with sqlite3.connect(path) as connection:
        return tuple(
            int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in (
                "responses",
                "response_items",
                "response_events",
                "background_jobs",
            )
        )


def _chat_payload(**updates: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": "nfn-test",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_completion_tokens": 2,
        "temperature": 0.0,
    }
    payload.update(updates)
    return payload


def test_chat_omitted_sampling_fields_use_glimmer_model_card_defaults() -> None:
    runtime = _runtime()
    prepared = runtime.prepare_chat(
        {
            "model": "nfn-test",
            "messages": [{"role": "user", "content": "Hello"}],
        }
    )

    assert prepared.generation.max_new_tokens == runtime.max_output_tokens
    assert prepared.generation.temperature == 1.0
    assert prepared.generation.top_p == 0.95
    assert prepared.generation.top_k == 64


def _structured_text_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": "Answer",
        "schema": {
            "title": "Answer",
            "description": "One bounded answer.",
            "type": "object",
            "properties": {
                "answer": {
                    "title": "Answer",
                    "description": "The printable answer.",
                    "type": "string",
                },
                "count": {"title": "Count", "type": "integer"},
            },
            "required": ["answer", "count"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def _weather_function_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "name": "weather",
        "description": "Read bounded weather data.",
        "parameters": {
            "title": "WeatherArgs",
            "type": "object",
            "properties": {
                "city": {"title": "City", "type": "string"},
                "days": {"title": "Days", "type": "integer"},
            },
            "required": ["city", "days"],
            "additionalProperties": False,
        },
        "strict": True,
    }


async def _with_client(app, scenario) -> None:
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            await scenario(client)


def test_health_models_and_non_streaming_chat_are_openai_shaped() -> None:
    runtime = _runtime()
    app = create_native_inference_app(runtime, queue_capacity=1)

    async def scenario(client: httpx.AsyncClient) -> None:
        health = await client.get("/health")
        assert health.status_code == 200
        assert health.json()["backend"] == "fake-test-only"
        assert health.json()["cache"] == {"requested": "auto", "effective": "full"}
        assert health.json()["queue"]["workers"] == 1

        models = await client.get("/v1/models")
        assert models.json() == {
            "object": "list",
            "data": [
                {
                    "id": "nfn-test",
                    "object": "model",
                    "created": 1_700_000_000,
                    "owned_by": "neuralfn",
                }
            ],
        }
        assert (await client.get("/v1/models/nfn-test")).status_code == 200

        response = await client.post("/v1/chat/completions", json=_chat_payload())
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["id"].startswith("chatcmpl-")
        assert payload["object"] == "chat.completion"
        assert payload["model"] == "nfn-test"
        assert payload["choices"] == [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello!",
                    "refusal": None,
                    "annotations": [],
                },
                "logprobs": None,
                "finish_reason": "stop",
            }
        ]
        assert payload["usage"] == {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        }

    anyio.run(_with_client, app, scenario)
    assert runtime.model.prefills == [(1, 2, 3)]
    assert runtime.model.session_creates == runtime.model.session_closes == 1
    assert runtime.model.model_closes == 1


def test_glimmer_chat_never_returns_private_atem_reasoning() -> None:
    app = create_native_inference_app(_atem_runtime(), queue_capacity=0)

    async def scenario(client: httpx.AsyncClient) -> None:
        completed = await client.post("/v1/chat/completions", json=_chat_payload())
        assert completed.status_code == 200, completed.text
        assert completed.json()["choices"][0]["message"]["content"] == "ready"
        assert "private reasoning" not in completed.text

        truncated = await client.post(
            "/v1/chat/completions",
            json=_chat_payload(max_completion_tokens=1),
        )
        assert truncated.status_code == 200, truncated.text
        assert truncated.json()["choices"][0]["message"]["content"] == ""
        assert "private reasoning" not in truncated.text

    anyio.run(_with_client, app, scenario)


def test_glimmer_stream_buffers_atem_and_emits_only_user_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def connected(_request) -> bool:
        return False

    monkeypatch.setattr("starlette.requests.Request.is_disconnected", connected)
    app = create_native_inference_app(_atem_runtime(), queue_capacity=0)

    async def scenario(client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/v1/chat/completions",
            json=_chat_payload(stream=True),
        )
        assert response.status_code == 200, response.text
        assert "private reasoning" not in response.text
        data_lines = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
        chunks = [json.loads(line) for line in data_lines[:-1]]
        visible = [
            chunk["choices"][0]["delta"].get("content", "")
            for chunk in chunks
            if chunk["choices"]
        ]
        assert "".join(visible) == "ready"
        assert data_lines[-1] == "[DONE]"

    anyio.run(_with_client, app, scenario)


def test_failed_native_session_is_disposed_and_next_request_uses_fresh_session() -> None:
    model = FailOnceModel()
    runtime = _runtime(model)
    app = create_native_inference_app(runtime, queue_capacity=0)

    async def scenario(client: httpx.AsyncClient) -> None:
        failed = await client.post("/v1/chat/completions", json=_chat_payload())
        assert failed.status_code == 500
        assert failed.json() == {
            "error": {
                "message": "Resident native generation failed.",
                "type": "server_error",
                "param": None,
                "code": "generation_failed",
            }
        }
        assert model.session_creates == model.session_closes == 1
        assert model.sessions[0].poisoned is True
        assert model.sessions[0].closed is True

        succeeded = await client.post("/v1/chat/completions", json=_chat_payload())
        assert succeeded.status_code == 200, succeeded.text
        assert succeeded.json()["choices"][0]["message"]["content"] == "Hello!"
        assert model.session_creates == model.session_closes == 2
        assert model.sessions[1] is not model.sessions[0]
        assert model.sessions[1].poisoned is False
        assert model.sessions[1].closed is True
        assert model.stats()["weights_load_count"] == 1
        assert model.stats()["subprocess_spawns"] == 0

    anyio.run(_with_client, app, scenario)
    assert model.model_closes == 1


def test_unexpected_route_failure_returns_normalized_openai_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()

    def fail_model_object() -> dict[str, Any]:
        raise RuntimeError("unexpected route failure")

    monkeypatch.setattr(runtime, "model_object", fail_model_object)
    app = create_native_inference_app(runtime)

    async def scenario(client: httpx.AsyncClient) -> None:
        response = await client.get("/v1/models")
        assert response.status_code == 500
        assert response.json() == {
            "error": {
                "message": (
                    "The server encountered an internal error while processing the request."
                ),
                "type": "server_error",
                "param": None,
                "code": "internal_error",
            }
        }

    anyio.run(_with_client, app, scenario)
    assert runtime.model.session_creates == 0
    assert runtime.model.model_closes == 1


def test_streaming_chat_emits_real_chunks_usage_and_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def connected(_request) -> bool:
        return False

    monkeypatch.setattr("starlette.requests.Request.is_disconnected", connected)
    app = create_native_inference_app(_runtime(), queue_capacity=1)

    async def scenario(client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/v1/chat/completions",
            json=_chat_payload(stream=True, stream_options={"include_usage": True}),
        )
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("text/event-stream")
        data_lines = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
        assert data_lines[-1] == "[DONE]"
        chunks = [json.loads(line) for line in data_lines[:-1]]
        assert all(chunk["object"] == "chat.completion.chunk" for chunk in chunks)
        assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
        assert [
            chunk["choices"][0]["delta"]["content"]
            for chunk in chunks
            if chunk["choices"] and "content" in chunk["choices"][0]["delta"]
            and chunk["choices"][0]["delta"]["content"]
        ] == ["Hello", "!"]
        assert chunks[-2]["choices"][0]["finish_reason"] == "stop"
        assert chunks[-1]["choices"] == []
        assert chunks[-1]["usage"] == {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        }
        assert all("usage" in chunk for chunk in chunks)

        no_usage = await client.post(
            "/v1/chat/completions",
            json=_chat_payload(stream=True),
        )
        no_usage_lines = [
            line[6:] for line in no_usage.text.splitlines() if line.startswith("data: ")
        ]
        no_usage_chunks = [json.loads(line) for line in no_usage_lines[:-1]]
        assert all("usage" not in chunk for chunk in no_usage_chunks)

    anyio.run(_with_client, app, scenario)


def test_stream_disconnect_cancels_and_disposes_the_request_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = threading.Event()
    model = FakeModel(release=release)
    app = create_native_inference_app(_runtime(model), queue_capacity=0)

    async def disconnected(_request) -> bool:
        release.set()
        return True

    monkeypatch.setattr("starlette.requests.Request.is_disconnected", disconnected)

    async def scenario(client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/v1/chat/completions",
            json=_chat_payload(stream=True),
        )
        assert response.status_code == 200
        assert "data: [DONE]" in response.text

    anyio.run(_with_client, app, scenario)
    assert model.session_creates == model.session_closes == 1


def test_bearer_auth_protects_every_route_with_openai_error_envelope() -> None:
    app = create_native_inference_app(
        _runtime(),
        auth=BearerAuth(keys=("secret-key",)),
        queue_capacity=0,
    )

    async def scenario(client: httpx.AsyncClient) -> None:
        for path in ("/health", "/v1/models", "/v1/responses"):
            response = await client.get(path)
            assert response.status_code == 401
            assert response.json()["error"]["code"] == "invalid_api_key"
            assert response.headers["www-authenticate"] == "Bearer"
        authorized = await client.get(
            "/v1/models",
            headers={"Authorization": "Bearer secret-key"},
        )
        assert authorized.status_code == 200

    anyio.run(_with_client, app, scenario)


def test_remote_bind_requires_auth_unless_explicitly_overridden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NFN_INFER_API_KEY", raising=False)
    with pytest.raises(NativeServingConfigurationError, match="Non-loopback binding requires"):
        resolve_bearer_auth(NativeServeConfig(Path("artifact"), host="0.0.0.0"))
    assert not resolve_bearer_auth(
        NativeServeConfig(
            Path("artifact"),
            host="0.0.0.0",
            allow_unauthenticated_remote=True,
        )
    ).enabled
    assert resolve_bearer_auth(
        NativeServeConfig(Path("artifact"), host="::", api_key="secret")
    ).enabled


def test_api_key_file_must_be_private(tmp_path: Path) -> None:
    key_file = tmp_path / "keys"
    key_file.write_text("secret\n", encoding="utf-8")
    key_file.chmod(0o644)
    config = NativeServeConfig(Path("artifact"), api_key_file=key_file)
    with pytest.raises(NativeServingConfigurationError, match="group or other"):
        resolve_bearer_auth(config)
    key_file.chmod(0o600)
    assert resolve_bearer_auth(config).keys == ("secret",)


def test_unsupported_resources_and_features_fail_explicitly() -> None:
    app = create_native_inference_app(_runtime())

    async def scenario(client: httpx.AsyncClient) -> None:
        responses = await client.post("/v1/responses", json={})
        assert responses.status_code == 404
        assert responses.json()["error"]["code"] == "unsupported_resource"
        compaction = await client.post("/v1/responses/compact", json={})
        assert compaction.status_code == 404
        assert compaction.json()["error"]["code"] == "unsupported_resource"

        tools = await client.post(
            "/v1/chat/completions",
            json=_chat_payload(tools=[{"type": "function", "function": {"name": "x"}}]),
        )
        assert tools.status_code == 400
        assert tools.json()["error"] == {
            "message": "Chat Completions field 'tools' is not supported by this resident model.",
            "type": "invalid_request_error",
            "param": "tools",
            "code": "unsupported_feature",
        }

        image = await client.post(
            "/v1/chat/completions",
            json=_chat_payload(
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": "x"}}],
                    }
                ]
            ),
        )
        assert image.status_code == 400
        assert image.json()["error"]["code"] == "unsupported_feature"

        nested_tool = await client.post(
            "/v1/chat/completions",
            json=_chat_payload(
                messages=[
                    {
                        "role": "assistant",
                        "content": "Calling a tool",
                        "tool_calls": [],
                    }
                ]
            ),
        )
        assert nested_tool.status_code == 400
        assert nested_tool.json()["error"] == {
            "message": "messages.0.tool_calls is not supported by this bounded server.",
            "type": "invalid_request_error",
            "param": "messages.0.tool_calls",
            "code": "unsupported_feature",
        }

    anyio.run(_with_client, app, scenario)


def test_stateful_text_responses_crud_input_items_and_token_count(tmp_path: Path) -> None:
    app = create_native_inference_app(_stateful_runtime(tmp_path / "state.sqlite3"))

    async def scenario(client: httpx.AsyncClient) -> None:
        created = await client.post(
            "/v1/responses",
            json={
                "model": "nfn-test",
                "input": "Hello",
                "max_output_tokens": 2,
                "temperature": 0.0,
            },
        )
        assert created.status_code == 200, created.text
        payload = created.json()
        assert payload["id"].startswith("resp_")
        assert payload["object"] == "response"
        assert payload["status"] == "completed"
        assert payload["output"][0]["content"][0]["text"] == "Hello!"
        assert payload["usage"] == {
            "input_tokens": 3,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens": 2,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 5,
        }
        assert "_nfn" not in payload

        retrieved = await client.get(f"/v1/responses/{payload['id']}")
        assert retrieved.status_code == 200
        assert retrieved.json() == payload

        items = await client.get(f"/v1/responses/{payload['id']}/input_items")
        assert items.status_code == 200
        item_payload = items.json()
        assert item_payload["object"] == "list"
        assert item_payload["data"][0]["role"] == "user"
        assert item_payload["data"][0]["content"] == [
            {"type": "input_text", "text": "Hello"}
        ]

        counted = await client.post(
            "/v1/responses/input_tokens",
            json={"model": "nfn-test", "input": "Hello"},
        )
        assert counted.json() == {"object": "response.input_tokens", "input_tokens": 3}

        deleted = await client.delete(f"/v1/responses/{payload['id']}")
        assert deleted.status_code == 200
        assert deleted.json() == {
            "id": payload["id"],
            "object": "response",
            "deleted": True,
        }
        missing = await client.get(f"/v1/responses/{payload['id']}")
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "response_not_found"

    anyio.run(_with_client, app, scenario)


def test_stateful_item_lists_honor_openai_cursor_query_contract(tmp_path: Path) -> None:
    app = create_native_inference_app(_stateful_runtime(tmp_path / "state.sqlite3"))

    async def scenario(client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/v1/responses",
            json={
                "model": "nfn-test",
                "input": [
                    {"role": "user", "content": "First"},
                    {"role": "assistant", "content": "Second"},
                    {"role": "user", "content": "Third"},
                ],
                "max_output_tokens": 2,
            },
        )
        assert response.status_code == 200, response.text
        input_items_url = f"/v1/responses/{response.json()['id']}/input_items"

        first_page = await client.get(input_items_url, params={"limit": 2})
        assert first_page.status_code == 200
        first_payload = first_page.json()
        assert [
            item["content"][0]["text"] for item in first_payload["data"]
        ] == ["Third", "Second"]
        assert first_payload["has_more"] is True
        assert first_payload["first_id"] == first_payload["data"][0]["id"]
        assert first_payload["last_id"] == first_payload["data"][-1]["id"]

        second_page = await client.get(
            input_items_url,
            params={"after": first_payload["last_id"], "limit": 2},
        )
        assert [
            item["content"][0]["text"] for item in second_page.json()["data"]
        ] == ["First"]
        assert second_page.json()["has_more"] is False

        ascending = await client.get(
            input_items_url,
            params={"limit": 100, "order": "asc"},
        )
        assert [
            item["content"][0]["text"] for item in ascending.json()["data"]
        ] == ["First", "Second", "Third"]

        conversation = await client.post(
            "/v1/conversations",
            json={
                "items": [
                    {"role": "user", "content": "One"},
                    {"role": "assistant", "content": "Two"},
                    {"role": "user", "content": "Three"},
                ]
            },
        )
        conversation_items_url = (
            f"/v1/conversations/{conversation.json()['id']}/items"
        )
        conversation_page = await client.get(
            conversation_items_url,
            params={"limit": 2},
        )
        assert [
            item["content"][0]["text"]
            for item in conversation_page.json()["data"]
        ] == ["Three", "Two"]
        assert conversation_page.json()["has_more"] is True

        invalid_limit = await client.get(input_items_url, params={"limit": 0})
        assert invalid_limit.status_code == 400
        assert invalid_limit.json()["error"]["param"] == "limit"
        invalid_order = await client.get(input_items_url, params={"order": "sideways"})
        assert invalid_order.status_code == 400
        assert invalid_order.json()["error"]["param"] == "order"
        invalid_cursor = await client.get(input_items_url, params={"after": "msg_missing"})
        assert invalid_cursor.status_code == 400
        assert invalid_cursor.json()["error"]["code"] == "invalid_cursor"

    anyio.run(_with_client, app, scenario)


def test_previous_response_lineage_replays_text_and_is_api_key_scoped(tmp_path: Path) -> None:
    codec = RecordingCodec()
    app = create_native_inference_app(
        _stateful_runtime(tmp_path / "state.sqlite3", codec=codec),
        auth=BearerAuth(keys=("left-key", "right-key")),
    )
    left = {"Authorization": "Bearer left-key"}
    right = {"Authorization": "Bearer right-key"}

    async def scenario(client: httpx.AsyncClient) -> None:
        root = await client.post(
            "/v1/responses",
            headers=left,
            json={"model": "nfn-test", "input": "First", "max_output_tokens": 2},
        )
        assert root.status_code == 200, root.text
        root_id = root.json()["id"]

        branch = await client.post(
            "/v1/responses",
            headers=left,
            json={
                "model": "nfn-test",
                "input": "Second",
                "previous_response_id": root_id,
                "max_output_tokens": 2,
            },
        )
        assert branch.status_code == 200, branch.text
        assert "First" in codec.rendered[-1]
        assert "Hello!" in codec.rendered[-1]
        assert "Second" in codec.rendered[-1]

        isolated = await client.post(
            "/v1/responses",
            headers=right,
            json={
                "model": "nfn-test",
                "input": "Cross scope",
                "previous_response_id": root_id,
            },
        )
        assert isolated.status_code == 404
        assert isolated.json()["error"]["code"] == "response_not_found"

    anyio.run(_with_client, app, scenario)


def test_conversations_crud_items_and_response_history(tmp_path: Path) -> None:
    codec = RecordingCodec()
    app = create_native_inference_app(
        _stateful_runtime(tmp_path / "state.sqlite3", codec=codec)
    )

    async def scenario(client: httpx.AsyncClient) -> None:
        created = await client.post(
            "/v1/conversations",
            json={
                "metadata": {"purpose": "test"},
                "items": [{"role": "user", "content": "Earlier"}],
            },
        )
        assert created.status_code == 200, created.text
        conversation = created.json()
        conversation_id = conversation["id"]
        assert conversation["object"] == "conversation"

        updated = await client.post(
            f"/v1/conversations/{conversation_id}",
            json={"metadata": {"purpose": "updated"}},
        )
        assert updated.json()["metadata"] == {"purpose": "updated"}

        initial_items = await client.get(f"/v1/conversations/{conversation_id}/items")
        initial_item_id = initial_items.json()["data"][0]["id"]
        retrieved_item = await client.get(
            f"/v1/conversations/{conversation_id}/items/{initial_item_id}"
        )
        assert retrieved_item.json()["content"][0]["text"] == "Earlier"

        added = await client.post(
            f"/v1/conversations/{conversation_id}/items",
            json={"items": [{"role": "assistant", "content": "Context"}]},
        )
        assert added.status_code == 200
        added_id = added.json()["data"][0]["id"]

        response = await client.post(
            "/v1/responses",
            json={
                "model": "nfn-test",
                "conversation": conversation_id,
                "input": "Now",
                "max_output_tokens": 2,
            },
        )
        assert response.status_code == 200, response.text
        assert "Earlier" in codec.rendered[-1]
        assert "Context" in codec.rendered[-1]
        assert "Now" in codec.rendered[-1]

        all_items = await client.get(
            f"/v1/conversations/{conversation_id}/items",
            params={"order": "asc", "limit": 100},
        )
        assert [item["role"] for item in all_items.json()["data"]] == [
            "user",
            "assistant",
            "user",
            "assistant",
        ]

        deleted_item = await client.delete(
            f"/v1/conversations/{conversation_id}/items/{added_id}"
        )
        assert deleted_item.status_code == 200
        assert deleted_item.json()["id"] == conversation_id

        deleted = await client.delete(f"/v1/conversations/{conversation_id}")
        assert deleted.json() == {
            "id": conversation_id,
            "object": "conversation.deleted",
            "deleted": True,
        }
        assert (await client.get(f"/v1/conversations/{conversation_id}")).status_code == 404

    anyio.run(_with_client, app, scenario)


def test_response_compaction_is_lossless_durable_and_api_key_scoped(
    tmp_path: Path,
) -> None:
    codec = RecordingCodec()
    model = FakeModel()
    app = create_native_inference_app(
        _stateful_runtime(tmp_path / "state.sqlite3", model=model, codec=codec),
        auth=BearerAuth(keys=("left-key", "right-key")),
    )
    left = {"Authorization": "Bearer left-key"}
    right = {"Authorization": "Bearer right-key"}

    async def scenario(client: httpx.AsyncClient) -> None:
        compacted = await client.post(
            "/v1/responses/compact",
            headers=left,
            json={
                "model": "nfn-test",
                "input": [
                    {"role": "user", "content": "Earlier"},
                    {"role": "assistant", "content": "Context"},
                ],
            },
        )
        assert compacted.status_code == 200, compacted.text
        payload = compacted.json()
        assert payload["id"].startswith("resp_")
        assert payload["object"] == "response.compaction"
        assert payload["output"][-1]["id"].startswith("cmp_")
        assert payload["output"][-1]["type"] == "compaction"
        assert payload["output"][-1]["encrypted_content"].startswith("nfncmp_")
        assert payload["usage"] == {
            "input_tokens": 3,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 3,
        }
        assert model.session_creates == 0

        resumed = await client.post(
            "/v1/responses",
            headers=left,
            json={
                "model": "nfn-test",
                "input": [
                    *payload["output"],
                    {"role": "user", "content": "Next"},
                ],
                "max_output_tokens": 2,
            },
        )
        assert resumed.status_code == 200, resumed.text
        assert "Earlier" in codec.rendered[-1]
        assert "Context" in codec.rendered[-1]
        assert "Next" in codec.rendered[-1]
        assert codec.rendered[-1].count("Earlier") == 1

        reused = await client.post(
            "/v1/responses",
            headers=left,
            json={
                "model": "nfn-test",
                "input": [
                    *payload["output"],
                    {"role": "user", "content": "Branch"},
                ],
                "max_output_tokens": 2,
            },
        )
        assert reused.status_code == 200, reused.text
        assert "Branch" in codec.rendered[-1]

        mismatched_retained_user = await client.post(
            "/v1/responses",
            headers=left,
            json={
                "model": "nfn-test",
                "input": [
                    {"role": "user", "content": "Tampered"},
                    payload["output"][-1],
                ],
            },
        )
        assert mismatched_retained_user.status_code == 400
        assert (
            mismatched_retained_user.json()["error"]["code"]
            == "invalid_compaction_input"
        )

        isolated = await client.post(
            "/v1/responses",
            headers=right,
            json={
                "model": "nfn-test",
                "input": [payload["output"][-1]],
            },
        )
        assert isolated.status_code == 404
        assert isolated.json()["error"]["code"] == "compaction_not_found"

        oversized_token = await client.post(
            "/v1/responses",
            headers=left,
            json={
                "model": "nfn-test",
                "input": [
                    {
                        "type": "compaction",
                        "encrypted_content": "x" * 256,
                    }
                ],
            },
        )
        assert oversized_token.status_code == 400
        assert oversized_token.json()["error"]["code"] == "invalid_compaction_input"

        unsupported = await client.post(
            "/v1/responses/compact",
            headers=left,
            json={
                "model": "nfn-test",
                "input": "No cache mode",
                "service_tier": "auto",
            },
        )
        assert unsupported.status_code == 400
        assert unsupported.json()["error"]["code"] == "unsupported_feature"

        oversized_body = await client.post(
            "/v1/responses/compact",
            headers=left,
            json={"model": "nfn-test", "input": "x" * (1024 * 1024)},
        )
        assert oversized_body.status_code == 413
        assert oversized_body.json()["error"]["code"] == "request_too_large"

    anyio.run(_with_client, app, scenario)


def test_responses_stream_uses_semantic_terminal_event_without_done_marker(
    tmp_path: Path,
) -> None:
    app = create_native_inference_app(_stateful_runtime(tmp_path / "state.sqlite3"))

    async def scenario(client: httpx.AsyncClient) -> None:
        streamed = await client.post(
            "/v1/responses",
            json={
                "model": "nfn-test",
                "input": "Hello",
                "max_output_tokens": 2,
                "stream": True,
            },
        )
        assert streamed.status_code == 200, streamed.text
        assert streamed.headers["content-type"].startswith("text/event-stream")
        assert "[DONE]" not in streamed.text
        events = [
            json.loads(line[6:])
            for line in streamed.text.splitlines()
            if line.startswith("data: ")
        ]
        assert [event["sequence_number"] for event in events] == list(range(len(events)))
        assert events[0]["type"] == "response.created"
        assert events[-1]["type"] == "response.completed"
        response_id = events[0]["response"]["id"]
        item_ids = {
            event["item_id"]
            for event in events
            if "item_id" in event
        }
        assert item_ids == {events[2]["item"]["id"]}
        assert [
            event["delta"]
            for event in events
            if event["type"] == "response.output_text.delta"
        ] == ["Hello", "!"]
        assert events[-1]["response"]["id"] == response_id
        assert (await client.get(f"/v1/responses/{response_id}")).json()["status"] == "completed"

    anyio.run(_with_client, app, scenario)


def test_background_response_stream_is_durable_resumable_and_query_strict(
    tmp_path: Path,
) -> None:
    app = create_native_inference_app(_stateful_runtime(tmp_path / "state.sqlite3"))

    async def scenario(client: httpx.AsyncClient) -> None:
        created = await client.post(
            "/v1/responses",
            json={
                "model": "nfn-test",
                "input": "Hello",
                "max_output_tokens": 2,
                "background": True,
                "stream": True,
            },
        )
        assert created.status_code == 200, created.text
        assert created.headers["content-type"].startswith("text/event-stream")
        assert "[DONE]" not in created.text
        events = [
            json.loads(line[6:])
            for line in created.text.splitlines()
            if line.startswith("data: ")
        ]
        assert [event["sequence_number"] for event in events] == list(range(len(events)))
        assert events[0]["type"] == "response.created"
        assert events[-1]["type"] == "response.completed"
        assert events[-1]["response"]["status"] == "completed"
        deltas = [
            event for event in events if event["type"] == "response.output_text.delta"
        ]
        assert [event["delta"] for event in deltas] == ["Hello", "!"]
        assert all(event.get("obfuscation") for event in deltas)
        response_id = events[0]["response"]["id"]

        cursor = deltas[0]["sequence_number"]
        default_resumed = await client.get(
            f"/v1/responses/{response_id}",
            params={"stream": "true", "starting_after": str(cursor)},
        )
        default_suffix = [
            json.loads(line[6:])
            for line in default_resumed.text.splitlines()
            if line.startswith("data: ")
        ]
        assert any(
            event.get("obfuscation")
            for event in default_suffix
            if event["type"] == "response.output_text.delta"
        )

        resumed = await client.get(
            f"/v1/responses/{response_id}",
            params=[
                ("stream", "true"),
                ("starting_after", str(cursor)),
                ("include_obfuscation", "false"),
                ("include[]", "message.output_text.logprobs"),
            ],
        )
        assert resumed.status_code == 200, resumed.text
        assert resumed.headers["content-type"].startswith("text/event-stream")
        assert "[DONE]" not in resumed.text
        suffix = [
            json.loads(line[6:])
            for line in resumed.text.splitlines()
            if line.startswith("data: ")
        ]
        assert suffix
        assert all(event["sequence_number"] > cursor for event in suffix)
        assert [event["sequence_number"] for event in suffix] == [
            event["sequence_number"] for event in events if event["sequence_number"] > cursor
        ]
        assert suffix[-1]["type"] == "response.completed"
        assert all("obfuscation" not in event for event in suffix)

        after_terminal = await client.get(
            f"/v1/responses/{response_id}",
            params={
                "stream": "true",
                "starting_after": str(events[-1]["sequence_number"]),
            },
        )
        assert after_terminal.status_code == 200
        assert after_terminal.headers["content-type"].startswith("text/event-stream")
        assert after_terminal.text == ""

        json_retrieval = await client.get(
            f"/v1/responses/{response_id}",
            params=[
                ("stream", "false"),
                ("include[]", "message.output_text.logprobs"),
            ],
        )
        assert json_retrieval.status_code == 200
        assert json_retrieval.headers["content-type"].startswith("application/json")

        included_items = await client.get(
            f"/v1/responses/{response_id}/input_items",
            params=[("include[]", "message.output_text.logprobs")],
        )
        assert included_items.status_code == 200, included_items.text

        non_durable_background = await client.post(
            "/v1/responses",
            json={
                "model": "nfn-test",
                "input": "Cannot resume without storage",
                "background": True,
                "stream": True,
                "store": False,
            },
        )
        assert non_durable_background.status_code == 400
        assert non_durable_background.json()["error"]["param"] == "store"

        invalid_queries = (
            ("?stream=1", "stream"),
            ("?stream=true&stream=false", "stream"),
            ("?starting_after=0", "starting_after"),
            ("?stream=true&starting_after=-1", "starting_after"),
            ("?stream=true&starting_after=9223372036854775808", "starting_after"),
            ("?include_obfuscation=0", "include_obfuscation"),
            ("?include[]=not.real", "include"),
            ("?unknown=true", "unknown"),
        )
        for query, param in invalid_queries:
            invalid = await client.get(f"/v1/responses/{response_id}{query}")
            assert invalid.status_code == 400, (query, invalid.text)
            assert invalid.json()["error"]["param"] == param

    anyio.run(_with_client, app, scenario)


def test_background_stream_disconnect_does_not_cancel_and_replays_after_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "disconnect.sqlite3"
    release = threading.Event()
    model = FakeModel(release=release)
    app = create_native_inference_app(_stateful_runtime(path, model=model))
    captured: dict[str, Any] = {}
    disconnect_checks = 0

    async def disconnected(request) -> bool:
        nonlocal disconnect_checks
        if request.method != "POST":
            return False
        disconnect_checks += 1
        return disconnect_checks > 1

    async def disconnect_scenario(client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/v1/responses",
            json={
                "model": "nfn-test",
                "input": "Keep going",
                "max_output_tokens": 2,
                "background": True,
                "stream": True,
            },
        )
        assert response.status_code == 200, response.text
        partial = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        assert partial and partial[0]["type"] == "response.created"
        assert not any(
            event["type"] in {"response.completed", "response.failed", "response.incomplete"}
            for event in partial
        )
        captured["response_id"] = partial[0]["response"]["id"]
        assert await anyio.to_thread.run_sync(model.started.wait, 1.0)
        live_replay = asyncio.create_task(
            client.get(
                f"/v1/responses/{captured['response_id']}",
                params={
                    "stream": "true",
                    "starting_after": str(partial[-1]["sequence_number"]),
                },
            )
        )
        await anyio.sleep(0.05)
        assert not live_replay.done()
        release.set()
        live_response = await live_replay
        live_events = [
            json.loads(line[6:])
            for line in live_response.text.splitlines()
            if line.startswith("data: ")
        ]
        assert live_events
        assert all(
            event["sequence_number"] > partial[-1]["sequence_number"]
            for event in live_events
        )
        assert live_events[-1]["type"] == "response.completed"
        for _ in range(200):
            stored = app.state.native_runtime.state_store.get_response(
                api_key_fingerprint(None),
                captured["response_id"],
            )
            if stored is not None and stored["status"] in {
                "completed",
                "failed",
                "incomplete",
            }:
                break
            await anyio.sleep(0.01)
        assert stored is not None and stored["status"] == "completed"
        durable = app.state.native_runtime.state_store.list_response_events(
            api_key_fingerprint(None),
            captured["response_id"],
        )
        assert durable[-1]["type"] == "response.completed"
        captured["cursor"] = next(
            event["sequence_number"]
            for event in durable
            if event["type"] == "response.output_text.delta"
        )
        captured["suffix_sequences"] = [
            event["sequence_number"]
            for event in durable
            if event["sequence_number"] > captured["cursor"]
        ]

    with monkeypatch.context() as patcher:
        patcher.setattr("starlette.requests.Request.is_disconnected", disconnected)
        anyio.run(_with_client, app, disconnect_scenario)

    restarted = create_native_inference_app(_stateful_runtime(path))

    async def replay_scenario(client: httpx.AsyncClient) -> None:
        replay = await client.get(
            f"/v1/responses/{captured['response_id']}",
            params={
                "stream": "true",
                "starting_after": str(captured["cursor"]),
            },
        )
        assert replay.status_code == 200, replay.text
        events = [
            json.loads(line[6:])
            for line in replay.text.splitlines()
            if line.startswith("data: ")
        ]
        assert [event["sequence_number"] for event in events] == captured[
            "suffix_sequences"
        ]
        assert events[-1]["type"] == "response.completed"

    anyio.run(_with_client, restarted, replay_scenario)


def test_retrieve_stream_is_scoped_and_rejects_non_stream_backgrounds(
    tmp_path: Path,
) -> None:
    app = create_native_inference_app(
        _stateful_runtime(tmp_path / "scoped.sqlite3"),
        auth=BearerAuth(keys=("left-key", "right-key")),
    )
    left = {"Authorization": "Bearer left-key"}
    right = {"Authorization": "Bearer right-key"}

    async def scenario(client: httpx.AsyncClient) -> None:
        streamed = await client.post(
            "/v1/responses",
            headers=left,
            json={
                "model": "nfn-test",
                "input": "Streamed",
                "max_output_tokens": 2,
                "background": True,
                "stream": True,
            },
        )
        events = [
            json.loads(line[6:])
            for line in streamed.text.splitlines()
            if line.startswith("data: ")
        ]
        streamed_id = events[0]["response"]["id"]
        cross_scope = await client.get(
            f"/v1/responses/{streamed_id}?stream=true",
            headers=right,
        )
        assert cross_scope.status_code == 404
        assert cross_scope.headers["content-type"].startswith("application/json")
        assert cross_scope.json()["error"]["code"] == "response_not_found"

        ordinary_background = await client.post(
            "/v1/responses",
            headers=left,
            json={
                "model": "nfn-test",
                "input": "Not streamed",
                "background": True,
            },
        )
        denied = await client.get(
            f"/v1/responses/{ordinary_background.json()['id']}?stream=true",
            headers=left,
        )
        assert denied.status_code == 400
        assert denied.headers["content-type"].startswith("application/json")
        assert denied.json()["error"]["code"] == "response_not_streamable"

        foreground = await client.post(
            "/v1/responses",
            headers=left,
            json={
                "model": "nfn-test",
                "input": "Foreground",
                "max_output_tokens": 2,
                "stream": True,
            },
        )
        foreground_events = [
            json.loads(line[6:])
            for line in foreground.text.splitlines()
            if line.startswith("data: ")
        ]
        foreground_id = foreground_events[0]["response"]["id"]
        foreground_denied = await client.get(
            f"/v1/responses/{foreground_id}?stream=true",
            headers=left,
        )
        assert foreground_denied.status_code == 400
        assert foreground_denied.json()["error"]["code"] == "response_not_streamable"

    anyio.run(_with_client, app, scenario)


def test_failed_background_stream_persists_exactly_one_semantic_terminal(
    tmp_path: Path,
) -> None:
    model = FailOnceModel()
    app = create_native_inference_app(
        _stateful_runtime(tmp_path / "failed.sqlite3", model=model)
    )

    async def scenario(client: httpx.AsyncClient) -> None:
        failed = await client.post(
            "/v1/responses",
            json={
                "model": "nfn-test",
                "input": "Fail",
                "background": True,
                "stream": True,
            },
        )
        assert failed.status_code == 200, failed.text
        events = [
            json.loads(line[6:])
            for line in failed.text.splitlines()
            if line.startswith("data: ")
        ]
        terminals = [
            event
            for event in events
            if event["type"]
            in {"response.completed", "response.failed", "response.incomplete"}
        ]
        assert len(terminals) == 1
        assert terminals[0]["type"] == "response.failed"
        assert terminals[0]["response"]["status"] == "failed"
        response_id = events[0]["response"]["id"]
        persisted = app.state.native_runtime.state_store.list_response_events(
            api_key_fingerprint(None), response_id
        )
        assert sum(
            event["type"]
            in {"response.completed", "response.failed", "response.incomplete"}
            for event in persisted
        ) == 1

    anyio.run(_with_client, app, scenario)


def test_bounded_structured_response_masks_logits_and_persists_canonical_output(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "structured.sqlite3"
    model = ConstrainedModel()
    codec = PrintableASCIICodec()
    runtime = _constrained_stateful_runtime(state_path, model=model, codec=codec)
    app = create_native_inference_app(runtime)
    text_format = _structured_text_format()
    target = b'{"answer":"ok","count":2}'
    model.queue_constrained(target)

    async def scenario(client: httpx.AsyncClient) -> None:
        created = await client.post(
            "/v1/responses",
            json={
                "model": "nfn-test",
                "input": "Return the bounded answer.",
                "text": {"format": text_format},
                "max_output_tokens": 64,
            },
        )
        assert created.status_code == 200, created.text
        payload = created.json()
        assert payload["status"] == "completed"
        assert payload["incomplete_details"] is None
        assert payload["text"] == {"format": text_format}
        assert payload["temperature"] == 0.0
        assert payload["top_p"] == 1.0
        assert payload["store"] is True
        assert payload["tools"] == []
        assert payload["tool_choice"] == "none"
        assert payload["output"][0]["type"] == "message"
        assert payload["output"][0]["status"] == "completed"
        assert payload["output"][0]["content"][0]["text"] == target.decode("ascii")
        assert json.loads(payload["output"][0]["content"][0]["text"]) == {
            "answer": "ok",
            "count": 2,
        }

        retrieved = await client.get(f"/v1/responses/{payload['id']}")
        assert retrieved.status_code == 200
        assert retrieved.json() == payload

        model.queue_constrained(target)
        incomplete = await client.post(
            "/v1/responses",
            json={
                "model": "nfn-test",
                "input": "Stop at the byte bound.",
                "text": {"format": text_format},
                "max_output_tokens": 7,
            },
        )
        assert incomplete.status_code == 200, incomplete.text
        bounded = incomplete.json()
        assert bounded["status"] == "incomplete"
        assert bounded["incomplete_details"] == {"reason": "max_output_tokens"}
        assert bounded["output"][0]["status"] == "incomplete"
        assert bounded["output"][0]["content"][0]["text"] == target[:7].decode(
            "ascii"
        )
        assert bounded["usage"]["output_tokens"] == 7
        retrieved_bounded = await client.get(f"/v1/responses/{bounded['id']}")
        assert retrieved_bounded.status_code == 200
        assert retrieved_bounded.json() == bounded

    anyio.run(_with_client, app, scenario)

    assert model.session_creates == model.session_closes == 2
    assert model.decode_calls == 0
    assert model.current_logits_calls == len(target) + 7
    assert model.unconstrained_argmaxes == [0] * model.current_logits_calls
    assert len(model.sessions) == 2
    for session, expected in zip(model.sessions, (target, target[:7]), strict=True):
        assert session.prompt_length is not None
        assert bytes(session.tokens[session.prompt_length :]) == expected
        assert session.decode_calls == 0
    assert model.model_closes == 1


def test_forced_function_call_and_client_output_continue_as_ordinary_text(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "function.sqlite3"
    model = ConstrainedModel()
    codec = PrintableASCIICodec()
    runtime = _constrained_stateful_runtime(state_path, model=model, codec=codec)
    app = create_native_inference_app(runtime)
    tool = _weather_function_tool()
    arguments = b'{"city":"Paris","days":2}'
    model.queue_constrained(arguments)
    model.queue_text("Weather accepted")

    async def scenario(client: httpx.AsyncClient) -> None:
        called = await client.post(
            "/v1/responses",
            json={
                "model": "nfn-test",
                "input": "Check Paris.",
                "tools": [tool],
                "tool_choice": {"type": "function", "name": "weather"},
                "parallel_tool_calls": False,
                "max_output_tokens": 64,
            },
        )
        assert called.status_code == 200, called.text
        first = called.json()
        assert first["status"] == "completed"
        assert first["tools"] == [tool]
        assert first["tool_choice"] == {"type": "function", "name": "weather"}
        assert first["parallel_tool_calls"] is False
        assert len(first["output"]) == 1
        call = first["output"][0]
        assert set(call) == {
            "id",
            "type",
            "status",
            "call_id",
            "name",
            "arguments",
        }
        assert call["id"].startswith("fc_")
        assert call["type"] == "function_call"
        assert call["status"] == "completed"
        assert call["call_id"].startswith("call_")
        assert call["name"] == "weather"
        assert call["arguments"] == arguments.decode("ascii")
        assert json.loads(call["arguments"]) == {"city": "Paris", "days": 2}

        continued = await client.post(
            "/v1/responses",
            json={
                "model": "nfn-test",
                "previous_response_id": first["id"],
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": "sunny",
                    }
                ],
                "tools": [],
                "tool_choice": "none",
                "parallel_tool_calls": False,
                "max_output_tokens": 32,
            },
        )
        assert continued.status_code == 200, continued.text
        second = continued.json()
        assert second["status"] == "completed"
        assert second["previous_response_id"] == first["id"]
        assert second["output"][0]["type"] == "message"
        assert second["output"][0]["content"][0]["text"] == "Weather accepted"
        assert second["tools"] == []
        assert second["tool_choice"] == "none"

        input_items = await client.get(f"/v1/responses/{second['id']}/input_items")
        assert input_items.status_code == 200, input_items.text
        items = input_items.json()["data"]
        assert len(items) == 1
        assert set(items[0]) == {"id", "type", "status", "call_id", "output"}
        assert items[0]["id"].startswith("fco_")
        assert items[0]["type"] == "function_call_output"
        assert items[0]["status"] == "completed"
        assert items[0]["call_id"] == call["call_id"]
        assert items[0]["output"] == "sunny"

        rendered = codec.rendered[-1]
        assert "Client-executed function call" in rendered
        assert call["call_id"] in rendered
        assert "Client result for call ID" in rendered
        assert "sunny" in rendered

    anyio.run(_with_client, app, scenario)

    assert model.session_creates == model.session_closes == 2
    assert model.sessions[0].decode_calls == 0
    assert model.sessions[1].decode_calls == 1
    assert model.decode_calls == 1
    assert model.function_executions == 0


def test_responses_prefix_cache_reuses_parent_and_reports_prompt_only_usage(
    tmp_path: Path,
) -> None:
    model = PrefixModel()
    runtime = _stateful_runtime(
        tmp_path / "prefix.sqlite3",
        model=model,
        codec=BranchingCodec(),
        prefix_cache_capacity=4,
    )
    app = create_native_inference_app(runtime)

    async def scenario(client: httpx.AsyncClient) -> None:
        parent_response = await client.post(
            "/v1/responses",
            json={
                "model": "nfn-test",
                "input": "Root",
                "max_output_tokens": 2,
                "temperature": 0.0,
            },
        )
        assert parent_response.status_code == 200, parent_response.text
        parent = parent_response.json()
        assert parent["usage"]["input_tokens_details"] == {
            "cached_tokens": 0,
            "cache_write_tokens": 3,
        }

        branch_response = await client.post(
            "/v1/responses",
            json={
                "model": "nfn-test",
                "previous_response_id": parent["id"],
                "input": "Branch",
                "max_output_tokens": 2,
                "temperature": 0.0,
            },
        )
        assert branch_response.status_code == 200, branch_response.text
        branch = branch_response.json()
        assert branch["usage"]["input_tokens_details"] == {
            "cached_tokens": 3,
            "cache_write_tokens": 1,
        }

        sibling_response = await client.post(
            "/v1/responses",
            json={
                "model": "nfn-test",
                "previous_response_id": parent["id"],
                "input": "Sibling",
                "max_output_tokens": 2,
                "temperature": 0.0,
            },
        )
        assert sibling_response.status_code == 200, sibling_response.text
        assert sibling_response.json()["usage"]["input_tokens_details"] == {
            "cached_tokens": 3,
            "cache_write_tokens": 1,
        }

        transient_response = await client.post(
            "/v1/responses",
            json={
                "model": "nfn-test",
                "previous_response_id": parent["id"],
                "input": "Branch",
                "store": False,
                "max_output_tokens": 2,
                "temperature": 0.0,
            },
        )
        assert transient_response.status_code == 200, transient_response.text
        transient = transient_response.json()
        assert transient["store"] is False
        assert transient["usage"]["input_tokens_details"] == {
            "cached_tokens": 3,
            "cache_write_tokens": 1,
        }

        health = await client.get("/health")
        assert health.status_code == 200
        stats = health.json()["prefix_cache"]
        assert stats["capacity"] == 4
        assert stats["entries"] == 3
        assert stats["hits"] == 3
        assert stats["commits"] == 3
        assert stats["rejections"] == 1
        assert stats["byte_accounting_scope"] == (
            "sum-of-per-session-capacity-observations; shared allocations may be "
            "represented by more than one session"
        )

    anyio.run(_with_client, app, scenario)

    assert len(model.forks) == 3
    assert [fork[1] for fork in model.forks] == [3, 3, 3]
    assert model.session_creates == 1
    assert model.session_closes == len(model.sessions) == 4


def test_response_delete_fences_in_flight_prefix_publish_and_lineage_commit(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "prefix-delete-race.sqlite3"
    model = PrefixModel()
    runtime = _stateful_runtime(
        state_path,
        model=model,
        codec=BranchingCodec(),
        prefix_cache_capacity=2,
    )
    app = create_native_inference_app(runtime)

    async def scenario(client: httpx.AsyncClient) -> None:
        parent_response = await client.post(
            "/v1/responses",
            json={
                "model": "nfn-test",
                "input": "Root",
                "max_output_tokens": 2,
                "temperature": 0.0,
            },
        )
        assert parent_response.status_code == 200
        parent = parent_response.json()

        model.started.clear()
        model.release = threading.Event()
        branch_task = asyncio.create_task(
            client.post(
                "/v1/responses",
                json={
                    "model": "nfn-test",
                    "previous_response_id": parent["id"],
                    "input": "Branch",
                    "max_output_tokens": 2,
                    "temperature": 0.0,
                },
            )
        )
        started = False
        for _ in range(5_000):
            if model.started.is_set():
                started = True
                break
            await asyncio.sleep(0.001)
        if not started:
            model.release.set()
        assert started

        try:
            deleted = await asyncio.wait_for(
                client.delete(f"/v1/responses/{parent['id']}"),
                timeout=5,
            )
        finally:
            # Never strand the resident worker when an interleaving assertion
            # fails; lifespan must still be able to drain the submitted ticket.
            model.release.set()
        assert deleted.status_code == 200, deleted.text
        branch_response = await asyncio.wait_for(branch_task, timeout=5)
        assert branch_response.status_code == 409, branch_response.text
        assert branch_response.json()["error"] == {
            "message": "Previous response lineage changed before completion.",
            "type": "conflict_error",
            "param": "previous_response_id",
            "code": "response_lineage_conflict",
        }

        with sqlite3.connect(state_path) as connection:
            rows = connection.execute(
                "SELECT payload_json FROM responses"
            ).fetchall()
        assert len(rows) == 1
        failed = json.loads(rows[0][0])
        assert failed["status"] == "failed"
        assert failed["output"] == []
        assert failed["error"]["code"] == "response_lineage_conflict"

        cache_stats = (await client.get("/health")).json()["prefix_cache"]
        assert cache_stats["entries"] == 0
        assert cache_stats["active_leases"] == 0
        assert cache_stats["scope_purges"] == 1

    anyio.run(_with_client, app, scenario)

    assert len(model.forks) == 1
    assert model.session_closes == len(model.sessions) == 2


def test_function_call_output_id_errors_do_not_generate_or_mutate_state(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "function-errors.sqlite3"
    model = ConstrainedModel()
    runtime = _constrained_stateful_runtime(state_path, model=model)
    app = create_native_inference_app(runtime)
    tool = _weather_function_tool()
    model.queue_constrained('{"city":"Paris","days":1}')

    async def scenario(client: httpx.AsyncClient) -> None:
        called = await client.post(
            "/v1/responses",
            json={
                "model": "nfn-test",
                "input": "Check Paris.",
                "tools": [tool],
                "tool_choice": {"type": "function", "name": "weather"},
                "max_output_tokens": 64,
            },
        )
        assert called.status_code == 200, called.text
        first = called.json()
        call_id = first["output"][0]["call_id"]

        def continuation(*items: dict[str, Any], previous: str = first["id"]):
            return {
                "model": "nfn-test",
                "previous_response_id": previous,
                "input": list(items),
                "tools": [],
                "tool_choice": "none",
                "parallel_tool_calls": False,
            }

        mismatch_item = {
            "type": "function_call_output",
            "call_id": "call_not_visible",
            "output": "sunny",
        }
        before_mismatch = (_native_state_row_counts(state_path), model.session_creates)
        mismatch = await client.post(
            "/v1/responses",
            json=continuation(mismatch_item),
        )
        assert mismatch.status_code == 400
        assert mismatch.json()["error"]["code"] == "function_call_not_found"
        assert (_native_state_row_counts(state_path), model.session_creates) == before_mismatch

        valid_item = {
            "type": "function_call_output",
            "call_id": call_id,
            "output": "sunny",
        }
        before_duplicate = (_native_state_row_counts(state_path), model.session_creates)
        duplicate = await client.post(
            "/v1/responses",
            json=continuation(valid_item, valid_item),
        )
        assert duplicate.status_code == 400
        assert duplicate.json()["error"]["code"] == "unsupported_feature"
        assert (_native_state_row_counts(state_path), model.session_creates) == before_duplicate

        model.queue_text("accepted")
        resolved = await client.post(
            "/v1/responses",
            json=continuation(valid_item),
        )
        assert resolved.status_code == 200, resolved.text
        second = resolved.json()

        before_resolved = (_native_state_row_counts(state_path), model.session_creates)
        already_resolved = await client.post(
            "/v1/responses",
            json=continuation(valid_item, previous=second["id"]),
        )
        assert already_resolved.status_code == 400
        assert already_resolved.json()["error"]["code"] == "function_call_already_resolved"
        assert (_native_state_row_counts(state_path), model.session_creates) == before_resolved

    anyio.run(_with_client, app, scenario)
    assert model.session_creates == model.session_closes == 2
    assert model.function_executions == 0


def test_constrained_modes_reject_before_generation_or_state_mutation(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "mode-errors.sqlite3"
    model = ConstrainedModel()
    runtime = _constrained_stateful_runtime(state_path, model=model)
    app = create_native_inference_app(runtime)
    text_format = _structured_text_format()
    tool = _weather_function_tool()
    structured = {
        "model": "nfn-test",
        "input": "Return JSON.",
        "text": {"format": text_format},
    }
    forced = {
        "model": "nfn-test",
        "input": "Call weather.",
        "tools": [tool],
        "tool_choice": {"type": "function", "name": "weather"},
    }

    async def scenario(client: httpx.AsyncClient) -> None:
        cases = (
            ("/v1/responses", {**structured, "stream": True}),
            ("/v1/responses", {**structured, "background": True}),
            ("/v1/responses", {**structured, "store": False}),
            ("/v1/responses", {**structured, "temperature": 0.01}),
            ("/v1/responses", {**structured, "top_p": 0.99}),
            ("/v1/responses", {**forced, "tool_choice": "auto"}),
            ("/v1/responses", {**forced, "tool_choice": "required"}),
            ("/v1/responses", {**forced, "parallel_tool_calls": True}),
            (
                "/v1/chat/completions",
                _chat_payload(response_format=text_format),
            ),
            (
                "/v1/chat/completions",
                _chat_payload(tools=[tool]),
            ),
        )
        for endpoint, payload in cases:
            response = await client.post(endpoint, json=payload)
            assert response.status_code == 400, response.text
            assert response.json()["error"]["code"] == "unsupported_feature"

    anyio.run(_with_client, app, scenario)
    assert model.session_creates == 0
    assert model.prefills == []
    assert _native_state_row_counts(state_path) == (0, 0, 0, 0)


def test_chat_image_data_url_uses_vision_rows_and_embedding_prefill() -> None:
    model = MediaModel()
    runtime = NativeServingRuntime(
        model=model,  # type: ignore[arg-type]
        manifest={"schema": "neuralfn.native_execution_manifest", "version": 1},
        codec=MediaCodec(),
        renderer=_PlainRolesRenderer(),
        served_model_name="nfn-test",
        context_limit=64,
        max_output_tokens=8,
        created=1_700_000_000,
    )
    image = Image.new("RGB", (28, 28), (127, 64, 255))
    encoded = BytesIO()
    image.save(encoded, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(encoded.getvalue()).decode()
    prepared = runtime.prepare_chat(
        _chat_payload(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe: "},
                        {"type": "image_url", "image_url": {"url": url}},
                    ],
                }
            ]
        )
    )
    assert prepared.media_batch is not None
    assert prepared.media_batch.grid_thw == ((1, 2, 2),)
    assert prepared.prompt_token_ids == (1, 200_092, 3)
    completed = runtime.complete(prepared)
    assert completed.text == "Hello!"
    assert len(model.embedding_prefills) == 1
    token_ids, positions, embeddings = model.embedding_prefills[0]
    assert token_ids == (1, 200_092, 3)
    assert positions == (1,)
    assert embeddings == ((0.0,) * 8,)
    runtime.close()


@pytest.mark.parametrize(
    ("label", "runtime_options"),
    (
        ("capability", {"structured_output": False, "function_tools": False}),
        ("profile", {"structured_profile": False}),
        ("template-selection", {"chat_template_selection": "plain_roles"}),
    ),
)
def test_structured_output_requires_effective_capability_and_profile(
    tmp_path: Path,
    label: str,
    runtime_options: dict[str, Any],
) -> None:
    state_path = tmp_path / f"missing-{label}.sqlite3"
    runtime = _constrained_stateful_runtime(state_path, **runtime_options)
    model = runtime.model
    assert runtime.capabilities.structured_output is False
    app = create_native_inference_app(runtime)

    async def scenario(client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/v1/responses",
            json={
                "model": "nfn-test",
                "input": "Return JSON.",
                "text": {"format": _structured_text_format()},
            },
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "unsupported_feature"

    anyio.run(_with_client, app, scenario)
    assert model.session_creates == 0
    assert _native_state_row_counts(state_path) == (0, 0, 0, 0)


@pytest.mark.parametrize(
    ("label", "tool_template"),
    (
        ("missing", False),
        (
            "boolean-version",
            {"version": True, "profile": "responses-forced-function-call-v1"},
        ),
        (
            "extra-field",
            {
                "version": 1,
                "profile": "responses-forced-function-call-v1",
                "future": True,
            },
        ),
    ),
)
def test_function_tools_require_exact_manifest_template(
    tmp_path: Path,
    label: str,
    tool_template: bool | Mapping[str, Any],
) -> None:
    state_path = tmp_path / f"tool-template-{label}.sqlite3"
    model = ConstrainedModel()
    runtime = _constrained_stateful_runtime(
        state_path,
        model=model,
        tool_template=tool_template,
    )
    assert runtime.capabilities.structured_output is True
    assert runtime.capabilities.function_tools is False
    app = create_native_inference_app(runtime)

    async def scenario(client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/v1/responses",
            json={
                "model": "nfn-test",
                "input": "Call weather.",
                "tools": [_weather_function_tool()],
                "tool_choice": {"type": "function", "name": "weather"},
            },
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "unsupported_feature"

    anyio.run(_with_client, app, scenario)
    assert model.session_creates == 0
    assert _native_state_row_counts(state_path) == (0, 0, 0, 0)


def test_constrained_runtime_rejects_incomplete_single_byte_inventory() -> None:
    model = ConstrainedModel()
    try:
        with pytest.raises(
            NativeServingConfigurationError,
            match="tokenizer preflight failed:.*0x7e",
        ):
            NativeServingRuntime(
                model=model,  # type: ignore[arg-type]
                manifest={
                    "schema": "neuralfn.native_execution_manifest",
                    "version": 1,
                    "tokenizer": {
                        "constrained_decoding": {
                            "version": 1,
                            "profile": "json-schema-ascii-byte-greedy-v1",
                            "token_selection": "current_logits_exact_prefill",
                        }
                    },
                    "chat_template": {
                        "format": "plain_roles",
                        "tool_template": {
                            "version": 1,
                            "profile": "responses-forced-function-call-v1",
                        },
                    },
                },
                codec=PrintableASCIICodec(missing_byte=ord("~")),
                renderer=_PlainRolesRenderer(),
                served_model_name="nfn-test",
                context_limit=512,
                max_output_tokens=256,
                created=1_700_000_000,
            )
    finally:
        model.close()
    assert model.session_creates == 0


def test_responses_fail_closed_for_tools_structured_and_multimodal_input(
    tmp_path: Path,
) -> None:
    model = FakeModel()
    app = create_native_inference_app(
        _stateful_runtime(tmp_path / "state.sqlite3", model=model)
    )

    async def scenario(client: httpx.AsyncClient) -> None:
        requests = (
            {
                "model": "nfn-test",
                "input": "tool",
                "tools": [{"type": "function", "name": "lookup"}],
            },
            {
                "model": "nfn-test",
                "input": "json",
                "text": {"format": {"type": "json_schema", "name": "answer"}},
            },
            {
                "model": "nfn-test",
                "input": [
                    {
                        "role": "user",
                        "content": [{"type": "input_image", "image_url": "https://x"}],
                    }
                ],
            },
        )
        for payload in requests:
            response = await client.post("/v1/responses", json=payload)
            assert response.status_code == 400
            assert response.json()["error"]["code"] == "unsupported_feature"

    anyio.run(_with_client, app, scenario)
    assert model.session_creates == 0


def test_background_response_cancel_and_restart_safe_queue(tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite3"
    release = threading.Event()
    model = FakeModel(release=release)
    app = create_native_inference_app(_stateful_runtime(path, model=model))

    async def cancel_scenario(client: httpx.AsyncClient) -> None:
        created = await client.post(
            "/v1/responses",
            json={
                "model": "nfn-test",
                "input": "Cancel me",
                "background": True,
                "max_output_tokens": 2,
            },
        )
        assert created.status_code == 200, created.text
        response_id = created.json()["id"]
        assert created.json()["status"] in {"queued", "in_progress"}
        assert await anyio.to_thread.run_sync(model.started.wait, 1.0)
        cancelled = await client.post(f"/v1/responses/{response_id}/cancel")
        assert cancelled.status_code == 200, cancelled.text
        release.set()
        terminal = None
        for _ in range(100):
            terminal = (await client.get(f"/v1/responses/{response_id}")).json()
            if terminal["status"] in {"cancelled", "completed", "failed", "incomplete"}:
                break
            await anyio.sleep(0.01)
        assert terminal is not None
        assert terminal["status"] == "cancelled"

    anyio.run(_with_client, app, cancel_scenario)

    queued_path = tmp_path / "queued.sqlite3"
    first_runtime = _stateful_runtime(queued_path)
    from neuralfn.native_responses import NativeResponsesService

    first_service = NativeResponsesService(first_runtime, first_runtime.state_store)
    prepared = first_service.prepare(
        api_key_fingerprint(None),
        {
            "model": "nfn-test",
            "input": "Survive restart",
            "background": True,
            "max_output_tokens": 2,
        },
    )
    first_service.persist(prepared)
    first_runtime.state_store.close()

    restarted_model = FakeModel()
    restarted_app = create_native_inference_app(
        _stateful_runtime(queued_path, model=restarted_model)
    )

    async def restart_scenario(client: httpx.AsyncClient) -> None:
        terminal = None
        for _ in range(100):
            terminal = (await client.get(f"/v1/responses/{prepared.response_id}")).json()
            if terminal["status"] in {"completed", "failed", "incomplete"}:
                break
            await anyio.sleep(0.01)
        assert terminal is not None
        assert terminal["status"] == "completed"
        assert terminal["output"][0]["content"][0]["text"] == "Hello!"

    anyio.run(_with_client, restarted_app, restart_scenario)
    assert restarted_model.session_creates == 1


def test_context_and_model_validation_are_400_and_404() -> None:
    app = create_native_inference_app(_runtime(context_limit=4))

    async def scenario(client: httpx.AsyncClient) -> None:
        context = await client.post("/v1/chat/completions", json=_chat_payload())
        assert context.status_code == 400
        assert context.json()["error"]["code"] == "context_length_exceeded"
        missing = await client.post(
            "/v1/chat/completions",
            json=_chat_payload(model="missing"),
        )
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "model_not_found"

    anyio.run(_with_client, app, scenario)


def test_single_worker_queue_rejects_immediately_when_saturated() -> None:
    app = create_native_inference_app(_runtime(), queue_capacity=0)

    async def scenario(client: httpx.AsyncClient) -> None:
        held = app.state.generation_queue.reserve()
        assert held is not None
        response = await client.post("/v1/chat/completions", json=_chat_payload())
        assert response.status_code == 429
        assert response.json()["error"]["code"] == "queue_saturated"
        await held.run(lambda: None)

    anyio.run(_with_client, app, scenario)


def test_unused_ticket_release_is_idempotent_and_restores_capacity() -> None:
    queue = BoundedSingleWorkerQueue(1, session_limit=1)
    held = queue.reserve()
    assert held is not None
    rejected, reason = queue.admit()
    assert rejected is None
    assert reason == "session_limit_exceeded"

    assert held.release() is True
    assert held.release() is False
    assert queue.stats()["queued"] == 0
    assert queue.stats()["session_reservations"] == 0

    replacement = queue.reserve()
    assert replacement is not None
    assert anyio.run(replacement.run, lambda: "released") == "released"
    assert queue.stats()["queued"] == 0
    assert queue.stats()["running"] == 0
    assert queue.stats()["session_reservations"] == 0
    queue.close()


def test_cancelled_waiter_cannot_leak_submitted_ticket() -> None:
    queue = BoundedSingleWorkerQueue(0, session_limit=1)

    async def scenario() -> None:
        ticket = queue.reserve()
        assert ticket is not None
        task = asyncio.create_task(ticket.run(lambda: "completed"))
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        for _ in range(100):
            if queue.stats()["session_reservations"] == 0:
                break
            await asyncio.sleep(0.001)
        assert queue.stats()["queued"] == 0
        assert queue.stats()["running"] == 0
        assert queue.stats()["session_reservations"] == 0

    anyio.run(scenario)
    queue.close()


def test_close_revokes_unused_ticket_and_run_after_close_does_not_leak() -> None:
    queue = BoundedSingleWorkerQueue(0, session_limit=1)
    ticket = queue.reserve()
    assert ticket is not None

    queue.close()
    assert queue.stats()["queued"] == 0
    assert queue.stats()["session_reservations"] == 0

    async def scenario() -> None:
        with pytest.raises(RuntimeError, match="already used or released"):
            await ticket.run(lambda: None)

    anyio.run(scenario)
    assert queue.stats()["queued"] == 0
    assert queue.stats()["session_reservations"] == 0


def test_close_drains_submitted_waiter_without_counter_leak() -> None:
    queue = BoundedSingleWorkerQueue(1, session_limit=2)
    started = threading.Event()
    release = threading.Event()

    def first_job() -> str:
        started.set()
        assert release.wait(timeout=5)
        return "first"

    async def scenario() -> None:
        first = queue.reserve()
        second = queue.reserve()
        assert first is not None and second is not None
        first_task = asyncio.create_task(first.run(first_job))
        second_task = asyncio.create_task(second.run(lambda: "second"))
        for _ in range(100):
            if started.is_set():
                break
            await asyncio.sleep(0.001)
        assert started.is_set()
        closing = asyncio.create_task(queue.aclose())
        await asyncio.sleep(0.01)
        assert not closing.done()
        release.set()
        assert await first_task == "first"
        assert await second_task == "second"
        await closing

    anyio.run(scenario)
    assert queue.stats()["queued"] == 0
    assert queue.stats()["running"] == 0
    assert queue.stats()["session_reservations"] == 0


def test_concurrent_close_and_aclose_join_the_same_worker_drain() -> None:
    queue = BoundedSingleWorkerQueue(0, session_limit=1)
    started = threading.Event()
    release = threading.Event()
    close_failures: list[BaseException] = []

    def job() -> str:
        started.set()
        assert release.wait(timeout=5)
        return "completed"

    def close_first() -> None:
        try:
            queue.close()
        except BaseException as exc:
            close_failures.append(exc)

    async def scenario() -> None:
        ticket = queue.reserve()
        assert ticket is not None
        waiter = asyncio.create_task(ticket.run(job))
        for _ in range(100):
            if started.is_set():
                break
            await asyncio.sleep(0.001)
        assert started.is_set()

        first_closer = threading.Thread(target=close_first, name="test-first-close")
        first_closer.start()
        for _ in range(100):
            if queue._closed:
                break
            await asyncio.sleep(0.001)
        assert queue._closed

        concurrent_close = asyncio.create_task(queue.aclose())
        await asyncio.sleep(0.01)
        assert not concurrent_close.done()
        assert first_closer.is_alive()
        assert queue.stats()["session_reservations"] == 1

        release.set()
        assert await waiter == "completed"
        await concurrent_close
        first_closer.join(timeout=5)
        assert not first_closer.is_alive()

    anyio.run(scenario)
    assert close_failures == []
    assert queue.stats()["queued"] == 0
    assert queue.stats()["running"] == 0
    assert queue.stats()["session_reservations"] == 0


def test_resident_worker_cannot_start_a_self_joining_close() -> None:
    queue = BoundedSingleWorkerQueue(0, session_limit=1)

    async def scenario() -> None:
        ticket = queue.reserve()
        assert ticket is not None
        with pytest.raises(RuntimeError, match="cannot be closed from its resident worker"):
            await ticket.run(queue.close)

    anyio.run(scenario)
    assert queue.stats()["session_reservations"] == 0
    replacement = queue.reserve()
    assert replacement is not None
    assert anyio.run(replacement.run, lambda: "still open") == "still open"
    queue.close()


def test_aclose_preserves_cancellation_only_after_worker_drain() -> None:
    queue = BoundedSingleWorkerQueue(0, session_limit=1)
    started = threading.Event()
    release = threading.Event()

    def job() -> None:
        started.set()
        assert release.wait(timeout=5)

    async def scenario() -> None:
        ticket = queue.reserve()
        assert ticket is not None
        waiter = asyncio.create_task(ticket.run(job))
        for _ in range(100):
            if started.is_set():
                break
            await asyncio.sleep(0.001)
        assert started.is_set()

        closing = asyncio.create_task(queue.aclose())
        for _ in range(100):
            if queue._closed:
                break
            await asyncio.sleep(0.001)
        assert queue._closed
        closing.cancel()
        await asyncio.sleep(0.01)
        returned_before_drain = closing.done()

        release.set()
        await waiter
        with pytest.raises(asyncio.CancelledError):
            await closing
        assert returned_before_drain is False

    anyio.run(scenario)
    assert queue.stats()["running"] == 0
    assert queue.stats()["session_reservations"] == 0


@pytest.mark.parametrize("failure_type", [RuntimeError, asyncio.CancelledError])
def test_lifespan_always_drains_queue_and_runtime_after_background_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_type: type[BaseException],
) -> None:
    model = FakeModel()
    runtime = _stateful_runtime(tmp_path / "state.sqlite3", model=model)
    app = create_native_inference_app(runtime)
    polled = threading.Event()

    def fail_background_poll() -> None:
        polled.set()
        raise failure_type("background poll failed")

    assert runtime.state_store is not None
    monkeypatch.setattr(runtime.state_store, "queued_background_jobs", fail_background_poll)

    async def scenario() -> None:
        with pytest.raises(failure_type, match="background poll failed"):
            async with app.router.lifespan_context(app):
                for _ in range(100):
                    if polled.is_set():
                        break
                    await asyncio.sleep(0.001)
                assert polled.is_set()

    anyio.run(scenario)
    assert app.state.generation_queue.reserve() is None
    assert model.model_closes == 1


def test_session_limit_rejects_admission_separately_and_releases() -> None:
    app = create_native_inference_app(
        _runtime(),
        queue_capacity=2,
        session_limit=1,
    )

    async def scenario(client: httpx.AsyncClient) -> None:
        held = app.state.generation_queue.reserve()
        assert held is not None

        saturated = await client.post("/v1/chat/completions", json=_chat_payload())
        assert saturated.status_code == 429
        assert saturated.json()["error"]["code"] == "session_limit_exceeded"

        stats = (await client.get("/health")).json()["queue"]
        assert stats["waiting_capacity"] == 2
        assert stats["session_limit"] == 1
        assert stats["session_reservations"] == 1
        assert stats["queue_rejected"] == 0
        assert stats["session_rejected"] == 1

        await held.run(lambda: None)
        admitted = await client.post("/v1/chat/completions", json=_chat_payload())
        assert admitted.status_code == 200, admitted.text

    anyio.run(_with_client, app, scenario)


def test_stream_setup_failure_releases_ticket_for_next_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    app = create_native_inference_app(runtime, queue_capacity=0, session_limit=1)

    def fail_decoder():
        raise RuntimeError("stream decoder setup failed")

    monkeypatch.setattr(runtime.codec, "incremental_decoder", fail_decoder)

    async def scenario(client: httpx.AsyncClient) -> None:
        await client.post(
            "/v1/chat/completions",
            json=_chat_payload(stream=True),
        )
        stats = (await client.get("/health")).json()["queue"]
        assert stats["queued"] == 0
        assert stats["running"] == 0
        assert stats["session_reservations"] == 0

        admitted = await client.post("/v1/chat/completions", json=_chat_payload())
        assert admitted.status_code == 200, admitted.text

    anyio.run(_with_client, app, scenario)


@pytest.mark.parametrize("session_limit", [0, -1, True, 1.5])
def test_native_serve_config_rejects_invalid_session_limit(session_limit: Any) -> None:
    with pytest.raises(ValueError, match="session_limit must be a positive integer"):
        NativeServeConfig(Path("artifact"), session_limit=session_limit)


def test_native_serve_config_defaults_session_limit_to_total_queue_admission() -> None:
    assert NativeServeConfig(Path("artifact"), queue_capacity=3).session_limit == 4


def test_missing_presentation_metadata_fails_before_resident_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "native-execution-manifest.json").write_text(
        json.dumps(
            {
                "schema": "neuralfn.native_execution_manifest",
                "version": 1,
                "model": {"name": "missing-metadata"},
                "context_limits": {"max_context_tokens": 32},
            }
        ),
        encoding="utf-8",
    )
    loads: list[Path] = []

    def must_not_load(path, **_kwargs):
        loads.append(Path(path))
        raise AssertionError("resident model loaded before serving metadata validation")

    monkeypatch.setattr("neuralfn.native_serve.NativeInferenceModel.load", must_not_load)
    with pytest.raises(NativeServingConfigurationError, match="tokenizer metadata"):
        NativeServingRuntime.load(NativeServeConfig(artifact))
    assert loads == []


def test_artifact_must_prove_serve_capability_before_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "native-execution-manifest.json").write_text(
        json.dumps(
            {
                "schema": "neuralfn.native_execution_manifest",
                "version": 1,
                "model": {"name": "not-serve-ready"},
                "context_limits": {"max_context_tokens": 32},
                "capabilities": {"serve": False},
            }
        ),
        encoding="utf-8",
    )
    loads: list[Path] = []

    monkeypatch.setattr("neuralfn.native_serve._load_text_codec", lambda _manifest: FakeCodec())

    def must_not_load(path, **_kwargs):
        loads.append(Path(path))
        raise AssertionError("resident model loaded before serving capability validation")

    monkeypatch.setattr("neuralfn.native_serve.NativeInferenceModel.load", must_not_load)
    with pytest.raises(NativeServingConfigurationError, match=r"capabilities\.serve=true"):
        NativeServingRuntime.load(
            NativeServeConfig(artifact, chat_template="plain_roles")
        )
    assert loads == []


def test_non_text_generation_artifact_is_excluded_before_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "embedding-artifact"
    artifact.mkdir()
    (artifact / "native-execution-manifest.json").write_text(
        json.dumps(
            {
                "schema": "neuralfn.native_execution_manifest",
                "version": 1,
                "model": {
                    "name": "embedding-model",
                    "family": "embedding",
                    "text_generation": False,
                },
                "context_limits": {"max_context_tokens": 32},
                # Defend against a malformed or stale manifest that overclaims
                # the high-level serving bit.
                "capabilities": {"serve": True},
            }
        ),
        encoding="utf-8",
    )
    loads: list[Path] = []

    monkeypatch.setattr("neuralfn.native_serve._load_text_codec", lambda _manifest: FakeCodec())
    monkeypatch.setattr(
        "neuralfn.native_serve._load_chat_renderer",
        lambda _manifest, _selection: _PlainRolesRenderer(),
    )

    def must_not_load(path, **_kwargs):
        loads.append(Path(path))
        raise AssertionError("non-text artifact reached resident model loading")

    monkeypatch.setattr("neuralfn.native_serve.NativeInferenceModel.load", must_not_load)
    with pytest.raises(NativeServingConfigurationError, match="not a text-generation model"):
        NativeServingRuntime.load(
            NativeServeConfig(artifact, chat_template="plain_roles")
        )
    assert loads == []


def test_server_preparation_completes_before_uvicorn_can_bind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import neuralfn.native_serve as native_serve

    events: list[str] = []
    runtime = _runtime()
    app = object()

    def prepare(config):
        assert config.artifact == Path("artifact")
        events.append("validated-and-loaded")
        return app, runtime, BearerAuth()

    class Uvicorn:
        @staticmethod
        def run(received_app, **kwargs) -> None:
            assert received_app is app
            assert events == ["validated-and-loaded"]
            assert kwargs["workers"] == 1
            events.append("uvicorn-run")

    def import_module(name: str):
        assert name == "uvicorn"
        assert events == ["validated-and-loaded"]
        return Uvicorn

    monkeypatch.setattr(native_serve, "prepare_native_inference_server", prepare)
    monkeypatch.setattr(native_serve.importlib, "import_module", import_module)

    run_native_inference_server(NativeServeConfig(Path("artifact")))

    assert events == ["validated-and-loaded", "uvicorn-run"]


def test_importing_serve_surface_does_not_initialize_editor_or_heavy_stacks() -> None:
    script = (
        "import sys; import neuralfn.native_serve; "
        "print(','.join(name for name in "
        "('torch','numpy','networkx','sqlalchemy','server.app') if name in sys.modules))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                filter(
                    None,
                    (
                        str(Path(__file__).resolve().parents[1]),
                        os.environ.get("PYTHONPATH"),
                    ),
                )
            ),
        },
    )
    assert completed.stdout.strip() == ""
