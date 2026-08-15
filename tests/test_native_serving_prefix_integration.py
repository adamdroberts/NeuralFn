from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import threading
from typing import Any, Mapping, Sequence

import pytest

from neuralfn.native_inference import (
    GenerationEvent,
    GenerationResult,
    KVCacheConfig,
    NativeInferenceCapabilities,
)
from neuralfn.native_responses import NativeResponsesAPIError, NativeResponsesService
from neuralfn.native_serve import (
    NativeServeConfig,
    NativeServingConfigurationError,
    NativeServingRuntime,
    _PlainRolesRenderer,
    _TextCodec,
    create_native_inference_app,
)
from neuralfn.native_state import NativeStateStore, api_key_fingerprint


SCOPE = api_key_fingerprint(None)


class _BranchCodec(_TextCodec):
    name = "serving-prefix-integration-test"

    def encode(self, text: str) -> tuple[int, ...]:
        if "Sibling" in text:
            return (1, 2, 3, 5)
        if "Branch" in text:
            return (1, 2, 3, 4)
        if "Second" in text:
            return (1, 2, 3, 8)
        if "First" in text:
            return (1, 2, 3, 7)
        return (1, 2, 3)

    def decode(self, token_ids: Sequence[int]) -> str:
        assert tuple(token_ids) == (10,)
        return "ok"

    def token_bytes(self, token_id: int) -> bytes:
        assert token_id == 10
        return b"ok"


class _PrefixSession:
    def __init__(
        self,
        model: "_PrefixModel",
        tokens: Sequence[int] = (),
        *,
        cached_tokens: int = 0,
    ) -> None:
        self.model = model
        self.tokens = list(tokens)
        self.cached_tokens = cached_tokens
        self.cancelled = False
        self.closed = False

    def __enter__(self) -> "_PrefixSession":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    @property
    def token_ids(self) -> tuple[int, ...]:
        return tuple(self.tokens)

    def prefill(self, token_ids: Sequence[int]) -> dict[str, int]:
        requested = tuple(token_ids)
        current = tuple(self.tokens)
        assert requested[: len(current)] == current
        appended = requested[len(current) :]
        self.tokens.extend(appended)
        self.cached_tokens = len(self.tokens)
        self.model.prefills.append(requested)
        return {
            "prefix_tokens": len(requested),
            "prefix_reused": len(current),
            "prefilled_tokens": len(appended),
        }

    def decode(self, generation, *, on_token=None) -> GenerationResult:
        self.model.started.set()
        gate = self.model.decode_release
        if gate is not None:
            assert gate.wait(timeout=5), "test decode gate was never released"
        if self.cancelled:
            return GenerationResult(
                token_ids=(),
                text="",
                finish_reason="cancelled",
                prompt_tokens=len(self.tokens),
                completion_tokens=0,
                cancelled=True,
            )
        assert generation.max_new_tokens >= 1
        position = len(self.tokens)
        self.tokens.append(10)
        self.cached_tokens = len(self.tokens)
        event = GenerationEvent(token_id=10, index=0, position=position)
        if on_token is not None:
            on_token(event)
        return GenerationResult(
            token_ids=(10,),
            text="",
            finish_reason="stop",
            prompt_tokens=position,
            completion_tokens=1,
            events=(event,),
        )

    def stats(self) -> dict[str, int]:
        return {
            "cached_tokens": self.cached_tokens,
            "cache_capacity_bytes": 64,
            "prefix_cow_shared_capacity_bytes": 0,
            "prefix_cow_detached_capacity_bytes": 0,
        }

    def cancel(self) -> None:
        self.cancelled = True

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.model.session_closes += 1
            self.model.close_order.append("session")


class _PrefixModel:
    def __init__(
        self,
        *,
        effective_cache: str = "full",
        full_cow: bool = True,
        cpu_turboquant_cow: bool = False,
        turboquant_backend: str = "cpu",
        tile: bool = False,
    ) -> None:
        self.capabilities = NativeInferenceCapabilities(
            native_inference=True,
            resident_inference=True,
            lossless_kv_cache=effective_cache == "full",
            turboquant_kv_cache=effective_cache == "turboquant",
            session_state_kinds=("token_history",),
            session_prefix_cow=full_cow,
            session_prefix_cow_cpu_turboquant=cpu_turboquant_cow,
        )
        self.effective_cache = effective_cache
        self._kv_cache = SimpleNamespace(
            turboquant_attention_backend=turboquant_backend
        )
        self._tile_attention_config = object() if tile else None
        self.started = threading.Event()
        self.decode_release: threading.Event | None = None
        self.prefills: list[tuple[int, ...]] = []
        self.sessions: list[_PrefixSession] = []
        self.forks: list[tuple[_PrefixSession, int, int]] = []
        self.session_creates = 0
        self.session_closes = 0
        self.model_closes = 0
        self.close_order: list[str] = []

    def create_session(self, *, seed: int = 0) -> _PrefixSession:
        del seed
        self.session_creates += 1
        session = _PrefixSession(self)
        self.sessions.append(session)
        return session

    def fork_session(
        self,
        source: _PrefixSession,
        *,
        token_count: int | None = None,
        seed: int = 0,
    ) -> _PrefixSession:
        assert token_count is not None
        self.forks.append((source, token_count, seed))
        session = _PrefixSession(
            self,
            source.token_ids[:token_count],
            cached_tokens=token_count,
        )
        self.sessions.append(session)
        return session

    def stats(self) -> dict[str, Any]:
        return {
            "backend": "prefix-integration-test",
            "effective_cache": self.effective_cache,
            "requested_cache": self.effective_cache,
            "turboquant_attention_backend": (
                self._kv_cache.turboquant_attention_backend
            ),
            "weights_load_count": 1,
            "subprocess_spawns": 0,
        }

    def close(self) -> None:
        self.model_closes += 1
        self.close_order.append("model")


def _runtime(
    path: Path,
    *,
    model: _PrefixModel | None = None,
    capacity: int = 2,
) -> NativeServingRuntime:
    return NativeServingRuntime(
        model=model or _PrefixModel(),  # type: ignore[arg-type]
        manifest={"schema": "neuralfn.native_execution_manifest", "version": 1},
        codec=_BranchCodec(),
        renderer=_PlainRolesRenderer(),
        served_model_name="nfn-test",
        context_limit=64,
        max_output_tokens=8,
        state_store=NativeStateStore(path),
        prefix_cache_capacity=capacity,
        created=1_700_000_000,
    )


def _request(input_text: str, **overrides: Any) -> dict[str, Any]:
    return {
        "model": "nfn-test",
        "input": input_text,
        "max_output_tokens": 1,
        "temperature": 0.0,
        **overrides,
    }


def _resident(
    service: NativeResponsesService,
    payload: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    prepared = service.prepare(SCOPE, payload)
    service.persist(prepared)
    completed, response = service._execute_and_finish_resident(prepared)
    return prepared, response


def _assert_conflict(exc: pytest.ExceptionInfo[NativeResponsesAPIError], code: str) -> None:
    assert exc.value.status_code == 409
    assert exc.value.code == code


def test_prepared_branch_starting_after_parent_purge_cannot_terminal_commit(
    tmp_path: Path,
) -> None:
    model = _PrefixModel()
    runtime = _runtime(tmp_path / "post-purge.sqlite3", model=model)
    service = NativeResponsesService(runtime, runtime.state_store)
    try:
        _parent_prepared, parent = _resident(service, _request("Root"))
        branch = service.prepare(
            SCOPE,
            _request("Branch", previous_response_id=parent["id"]),
        )
        service.persist(branch)

        # This models a request already admitted to the compute queue while a
        # blocker is running: its immutable history snapshot and durable
        # in-progress row exist, but native execution starts only after purge.
        service.delete_response(SCOPE, parent["id"])
        assert runtime._prefix_cache.stats()["entries"] == 0

        with pytest.raises(NativeResponsesAPIError) as exc:
            service._execute_and_finish_resident(branch)
        _assert_conflict(exc, "response_lineage_conflict")

        failed = runtime.state_store.get_response(SCOPE, branch.response_id)
        assert failed is not None
        assert failed["status"] == "failed"
        assert failed["output"] == []
        assert failed["error"]["code"] == "response_lineage_conflict"
        assert runtime.state_store.list_response_items(
            SCOPE, branch.response_id, phase="output"
        ) == ()
        stats = runtime._prefix_cache.stats()
        assert stats["entries"] == 0
        assert stats["active_leases"] == 0
        assert stats["scope_purges"] == 1
        assert model.forks == []
        assert model.session_creates == 2
    finally:
        runtime.close()


def test_conversation_snapshot_cas_has_one_winner_and_one_outputless_loser(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path / "conversation-cas.sqlite3")
    service = NativeResponsesService(runtime, runtime.state_store)
    second_service = NativeResponsesService(runtime, runtime.state_store)
    try:
        assert service._transition_lock is second_service._transition_lock
        assert service._transition_lock is runtime.state_store._responses_transition_lock

        conversation = service.create_conversation(SCOPE, {})
        first = service.prepare(
            SCOPE,
            _request("First", conversation=conversation["id"]),
        )
        second = second_service.prepare(
            SCOPE,
            _request("Second", conversation=conversation["id"]),
        )
        assert first.response["_nfn"]["conversation_revision"] == 0
        assert second.response["_nfn"]["conversation_revision"] == 0
        service.persist(first)
        second_service.persist(second)

        _completed, winner = service._execute_and_finish_resident(first)
        assert winner["status"] == "completed"
        with pytest.raises(NativeResponsesAPIError) as exc:
            second_service._execute_and_finish_resident(second)
        _assert_conflict(exc, "conversation_conflict")

        items, revision = runtime.state_store.conversation_items_snapshot(
            SCOPE, conversation["id"]
        )
        assert revision == 1
        assert [item["role"] for item in items] == ["user", "assistant"]
        loser = runtime.state_store.get_response(SCOPE, second.response_id)
        assert loser is not None
        assert loser["status"] == "failed"
        assert loser["output"] == []
        assert runtime.state_store.list_response_items(
            SCOPE, second.response_id, phase="output"
        ) == ()
        stats = runtime._prefix_cache.stats()
        assert stats["entries"] == 1
        assert stats["conversation_aliases"] == 1
        assert stats["commits"] == 1
        assert stats["active_leases"] == 0
    finally:
        runtime.close()


def test_conversation_deleted_during_generation_fails_without_cache_publish(
    tmp_path: Path,
) -> None:
    model = _PrefixModel()
    runtime = _runtime(tmp_path / "conversation-delete.sqlite3", model=model)
    service = NativeResponsesService(runtime, runtime.state_store)
    release = threading.Event()
    try:
        conversation = service.create_conversation(SCOPE, {})
        prepared = service.prepare(
            SCOPE,
            _request("First", conversation=conversation["id"]),
        )
        service.persist(prepared)
        model.started.clear()
        model.decode_release = release

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(service._execute_and_finish_resident, prepared)
            assert model.started.wait(timeout=5)
            service.delete_conversation(SCOPE, conversation["id"])
            release.set()
            with pytest.raises(NativeResponsesAPIError) as exc:
                future.result(timeout=5)
        _assert_conflict(exc, "conversation_conflict")

        failed = runtime.state_store.get_response(SCOPE, prepared.response_id)
        assert failed is not None
        assert failed["status"] == "failed"
        assert failed["output"] == []
        stats = runtime._prefix_cache.stats()
        assert stats["entries"] == 0
        assert stats["active_leases"] == 0
        assert stats["scope_purges"] == 1
    finally:
        release.set()
        runtime.close()


def test_background_and_public_execute_finish_stay_cold(tmp_path: Path) -> None:
    model = _PrefixModel()
    runtime = _runtime(tmp_path / "cold-paths.sqlite3", model=model)
    service = NativeResponsesService(runtime, runtime.state_store)
    try:
        _prepared, parent = _resident(service, _request("Root"))
        assert runtime._prefix_cache.stats()["entries"] == 1

        queued = service.prepare(
            SCOPE,
            _request(
                "Branch",
                previous_response_id=parent["id"],
                background=True,
            ),
        )
        service.persist(queued)
        claimed = runtime.state_store.claim_next_background_job()
        assert claimed is not None and claimed[0] == SCOPE
        background = service.from_stored_background(*claimed)
        completed = service.execute(background)
        finished = service.finish(background, completed)
        assert finished["status"] == "completed"

        foreground = service.prepare(
            SCOPE,
            _request("Sibling", previous_response_id=parent["id"]),
        )
        service.persist(foreground)
        completed = service.execute(foreground)
        cold = service.finish(foreground, completed)
        assert cold["status"] == "completed"
        assert cold["usage"]["input_tokens_details"] == {
            "cached_tokens": 0,
            "cache_write_tokens": 0,
        }

        stats = runtime._prefix_cache.stats()
        assert stats["entries"] == 1
        assert stats["commits"] == 1
        assert stats["hits"] == 0
        assert stats["active_leases"] == 0
        assert model.forks == []
        assert model.session_creates == 3
        assert model.session_closes == 2
    finally:
        runtime.close()


def test_legacy_background_snapshot_compatibility_is_pre_generation(
    tmp_path: Path,
) -> None:
    model = _PrefixModel()
    runtime = _runtime(tmp_path / "legacy-background.sqlite3", model=model)
    service = NativeResponsesService(runtime, runtime.state_store)
    try:
        def legacy(response: Mapping[str, Any]) -> dict[str, Any]:
            copied = deepcopy(dict(response))
            copied["_nfn"].pop("conversation_revision")
            copied["_nfn"].pop("previous_lineage")
            return copied

        root = service.prepare(
            SCOPE, _request("Root legacy", background=True)
        )
        restored_root = service.from_stored_background(SCOPE, legacy(root.response))
        assert restored_root.response["_nfn"]["conversation_revision"] is None
        assert restored_root.response["_nfn"]["previous_lineage"] == []

        _parent_prepared, parent = _resident(service, _request("Root"))
        previous = service.prepare(
            SCOPE,
            _request(
                "Branch",
                previous_response_id=parent["id"],
                background=True,
            ),
        )
        restored = service.from_stored_background(SCOPE, legacy(previous.response))
        assert restored.response["_nfn"]["previous_lineage"] == [
            {"id": parent["id"], "status": "completed"}
        ]

        conversation = service.create_conversation(SCOPE, {})
        conversation_job = service.prepare(
            SCOPE,
            _request(
                "First",
                conversation=conversation["id"],
                background=True,
            ),
        )
        sessions_before = model.session_creates
        with pytest.raises(NativeResponsesAPIError) as exc:
            service.from_stored_background(
                SCOPE, legacy(conversation_job.response)
            )
        _assert_conflict(exc, "conversation_snapshot_unavailable")
        assert model.session_creates == sessions_before

        partial = deepcopy(previous.response)
        partial["_nfn"].pop("previous_lineage")
        with pytest.raises(RuntimeError, match="partial native snapshot envelope"):
            service.from_stored_background(SCOPE, partial)

        malformed_revision = deepcopy(conversation_job.response)
        malformed_revision["_nfn"]["conversation_revision"] = "zero"
        with pytest.raises(RuntimeError, match="conversation revision"):
            service.from_stored_background(SCOPE, malformed_revision)

        malformed_lineage = deepcopy(previous.response)
        malformed_lineage["_nfn"]["previous_lineage"] = [
            {"id": parent["id"], "status": "failed"}
        ]
        with pytest.raises(RuntimeError, match="lineage"):
            service.from_stored_background(SCOPE, malformed_lineage)
        assert model.session_creates == sessions_before

        service.delete_response(SCOPE, parent["id"])
        with pytest.raises(NativeResponsesAPIError) as exc:
            service.from_stored_background(SCOPE, legacy(previous.response))
        _assert_conflict(exc, "response_lineage_unavailable")
        assert model.session_creates == sessions_before
    finally:
        runtime.close()


def test_runtime_prefix_cache_configuration_and_capability_gates(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="requires state_db"):
        NativeServeConfig(Path("artifact"), prefix_cache_capacity=1)

    tile_cache = KVCacheConfig(
        mode="turboquant",
        turboquant_attention_backend="tile-cuda",
        tile_ops_lib="libtile.so",
    )
    with pytest.raises(ValueError, match="rejects Tile-CUDA"):
        NativeServeConfig(
            Path("artifact"),
            state_db=tmp_path / "config.sqlite3",
            prefix_cache_capacity=1,
            kv_cache=tile_cache,
        )

    missing_state = _PrefixModel()
    with pytest.raises(
        NativeServingConfigurationError, match="requires a private state_db"
    ):
        NativeServingRuntime(
            model=missing_state,  # type: ignore[arg-type]
            manifest={"schema": "neuralfn.native_execution_manifest", "version": 1},
            codec=_BranchCodec(),
            renderer=_PlainRolesRenderer(),
            served_model_name="nfn-test",
            context_limit=64,
            max_output_tokens=8,
            prefix_cache_capacity=1,
        )

    unsupported_state = NativeStateStore(tmp_path / "unsupported.sqlite3")
    try:
        with pytest.raises(
            NativeServingConfigurationError,
            match="jointly proven full-cache session prefix COW",
        ):
            NativeServingRuntime(
                model=_PrefixModel(full_cow=False),  # type: ignore[arg-type]
                manifest={"schema": "neuralfn.native_execution_manifest", "version": 1},
                codec=_BranchCodec(),
                renderer=_PlainRolesRenderer(),
                served_model_name="nfn-test",
                context_limit=64,
                max_output_tokens=8,
                state_store=unsupported_state,
                prefix_cache_capacity=1,
            )
    finally:
        unsupported_state.close()

    tile_state = NativeStateStore(tmp_path / "tile.sqlite3")
    try:
        with pytest.raises(
            NativeServingConfigurationError, match="rejects Tile-CUDA"
        ):
            NativeServingRuntime(
                model=_PrefixModel(tile=True),  # type: ignore[arg-type]
                manifest={"schema": "neuralfn.native_execution_manifest", "version": 1},
                codec=_BranchCodec(),
                renderer=_PlainRolesRenderer(),
                served_model_name="nfn-test",
                context_limit=64,
                max_output_tokens=8,
                state_store=tile_state,
                prefix_cache_capacity=1,
            )
    finally:
        tile_state.close()

    full = _runtime(tmp_path / "full.sqlite3")
    try:
        assert full._prefix_cache.stats()["capacity"] == 2
    finally:
        full.close()

    turboquant_model = _PrefixModel(
        effective_cache="turboquant",
        full_cow=False,
        cpu_turboquant_cow=True,
    )
    turboquant = _runtime(
        tmp_path / "turboquant.sqlite3",
        model=turboquant_model,
    )
    try:
        assert turboquant._prefix_cache.stats()["capacity"] == 2
    finally:
        turboquant.close()


def test_restart_drops_process_cache_but_durable_history_still_continues(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "restart.sqlite3"
    first_model = _PrefixModel()
    first_runtime = _runtime(state_path, model=first_model)
    first_service = NativeResponsesService(first_runtime, first_runtime.state_store)
    _prepared, parent = _resident(first_service, _request("Root"))
    assert first_runtime._prefix_cache.stats()["entries"] == 1
    first_runtime.close()

    second_model = _PrefixModel()
    second_runtime = _runtime(state_path, model=second_model)
    second_service = NativeResponsesService(second_runtime, second_runtime.state_store)
    try:
        assert second_runtime._prefix_cache.stats()["entries"] == 0
        _branch_prepared, branch = _resident(
            second_service,
            _request("Branch", previous_response_id=parent["id"]),
        )
        assert branch["status"] == "completed"
        assert second_model.forks == []
        assert second_model.session_creates == 1
        assert second_runtime._prefix_cache.stats()["entries"] == 1
    finally:
        second_runtime.close()


def test_lifespan_drains_eager_unconsumed_response_sse_before_runtime_close(
    tmp_path: Path,
) -> None:
    model = _PrefixModel()
    runtime = _runtime(tmp_path / "sse-shutdown.sqlite3", model=model)
    app = create_native_inference_app(runtime)
    service = app.state.responses_service
    _prepared, parent = _resident(service, _request("Root"))
    assert runtime._prefix_cache.stats()["entries"] == 1

    release_decode = threading.Event()
    model.started.clear()
    model.decode_release = release_decode

    async def scenario() -> None:
        async with asyncio.timeout(10):
            request_body = json.dumps(
                _request(
                    "Branch",
                    previous_response_id=parent["id"],
                    stream=True,
                )
            ).encode("utf-8")
            request_delivered = False
            response_started = asyncio.Event()
            response_body_attempted = asyncio.Event()
            receive_wait = asyncio.Event()
            body_send_block = asyncio.Event()
            accepted_bodies: list[bytes] = []

            async def receive() -> dict[str, Any]:
                nonlocal request_delivered
                if not request_delivered:
                    request_delivered = True
                    return {
                        "type": "http.request",
                        "body": request_body,
                        "more_body": False,
                    }
                await receive_wait.wait()
                return {"type": "http.disconnect"}

            async def send(message: Mapping[str, Any]) -> None:
                if message["type"] == "http.response.start":
                    response_started.set()
                    return
                assert message["type"] == "http.response.body"
                response_body_attempted.set()
                await body_send_block.wait()
                accepted_bodies.append(bytes(message.get("body", b"")))

            scope = {
                "type": "http",
                "asgi": {"version": "3.0", "spec_version": "2.3"},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/v1/responses",
                "raw_path": b"/v1/responses",
                "query_string": b"",
                "root_path": "",
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(request_body)).encode("ascii")),
                ],
                "client": ("127.0.0.1", 12345),
                "server": ("testserver", 80),
            }

            release_task: asyncio.Task[None] | None = None
            request_task: asyncio.Task[None] | None = None
            print("DBG before lifespan", flush=True)
            async with app.router.lifespan_context(app):
                print("DBG in lifespan", flush=True)
                request_task = asyncio.create_task(app(scope, receive, send))
                try:
                    await asyncio.wait_for(response_started.wait(), timeout=2)
                    print("DBG response start", flush=True)
                    await asyncio.wait_for(
                        response_body_attempted.wait(), timeout=2
                    )
                    print("DBG body attempted", flush=True)
                    started = await asyncio.wait_for(
                        asyncio.to_thread(model.started.wait, 2),
                        timeout=3,
                    )
                    assert started
                    print("DBG model started", flush=True)

                    # The driver and native child lease exist before this
                    # deliberately stalled transport accepts any SSE bytes.
                    assert accepted_bodies == []
                    assert len(app.state.foreground_drivers) == 1
                    assert len(model.forks) == 1
                    live = runtime._prefix_cache.stats()
                    assert live["entries"] == 1
                    assert live["hits"] == 1
                    assert live["active_leases"] == 1
                    assert live["in_flight_forks"] == 0

                    request_task.cancel()
                    print("DBG request cancelled", flush=True)
                    try:
                        await asyncio.wait_for(request_task, timeout=2)
                    except asyncio.CancelledError:
                        pass
                    print("DBG request awaited", flush=True)
                    assert request_task.done()
                    assert accepted_bodies == []
                    assert len(app.state.foreground_drivers) == 1
                    assert runtime._prefix_cache.stats()["active_leases"] == 1

                    async def release_after_shutdown_begins() -> None:
                        await asyncio.sleep(0.05)
                        assert runtime._prefix_cache.closed is False
                        assert model.model_closes == 0
                        assert model.session_closes == 0
                        assert all(not session.closed for session in model.sessions)
                        assert len(app.state.foreground_drivers) == 1
                        assert runtime._prefix_cache.stats()["active_leases"] == 1
                        release_decode.set()

                    release_task = asyncio.create_task(
                        release_after_shutdown_begins()
                    )
                    print("DBG release scheduled", flush=True)
                finally:
                    if request_task is not None and not request_task.done():
                        request_task.cancel()
                        try:
                            await asyncio.wait_for(request_task, timeout=2)
                        except asyncio.CancelledError:
                            pass
                    if release_task is None:
                        release_decode.set()
                print("DBG exiting lifespan body", flush=True)

            print("DBG lifespan exited", flush=True)
            # Starlette/anyio may still own the two transport wait coroutines
            # briefly after the parent ASGI task observes cancellation. Let
            # them settle before asyncio.run performs its global task drain.
            receive_wait.set()
            body_send_block.set()
            await asyncio.sleep(0.01)
            assert release_task is not None
            await asyncio.wait_for(release_task, timeout=2)
            drained = runtime._prefix_cache.stats()
            assert drained["active_leases"] == 0
            assert drained["in_flight_forks"] == 0
            assert drained["entries"] == 0
            assert drained["shutdown"] is True
            assert app.state.foreground_drivers == set()
            assert model.session_closes == 2
            assert model.model_closes == 1
            assert model.close_order == ["session", "session", "model"]
            print(
                "DBG pending",
                [
                    (task.get_name(), repr(task.get_coro()), task.done())
                    for task in asyncio.all_tasks()
                    if task is not asyncio.current_task()
                ],
                flush=True,
            )
            print(
                "DBG threads",
                [(thread.name, thread.daemon) for thread in threading.enumerate()],
                flush=True,
            )
            print("DBG scenario complete", flush=True)

    asyncio.run(scenario())
