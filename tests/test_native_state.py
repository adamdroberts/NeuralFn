from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sqlite3
import stat
import threading

import pytest

from neuralfn.native_state import (
    NATIVE_STATE_SCHEMA_VERSION,
    NativeStateConflictError,
    NativeStateError,
    NativeStateStore,
    api_key_fingerprint,
)


def test_state_database_is_private_wal_scoped_and_copy_on_write(tmp_path: Path) -> None:
    path = tmp_path / "state" / "responses.sqlite3"
    left = api_key_fingerprint("left-secret")
    right = api_key_fingerprint("right-secret")

    with NativeStateStore(path) as store:
        root = store.put_response(
            left,
            {"id": "resp_root", "status": "completed", "output": [{"text": "root"}]},
        )
        branch_a = store.put_response(
            left,
            {
                "id": "resp_a",
                "status": "completed",
                "previous_response_id": root["id"],
                "output": [{"text": "a"}],
            },
        )
        branch_b = store.put_response(
            left,
            {
                "id": "resp_b",
                "status": "completed",
                "previous_response_id": root["id"],
                "output": [{"text": "b"}],
            },
        )
        assert [item["id"] for item in store.response_lineage(left, branch_a["id"])] == [
            "resp_root",
            "resp_a",
        ]
        assert [item["id"] for item in store.response_lineage(left, branch_b["id"])] == [
            "resp_root",
            "resp_b",
        ]
        assert store.get_response(right, "resp_root") is None
        store.put_response(right, {"id": "resp_root", "status": "completed"})
        assert store.get_response(right, "resp_root") is not None
        with pytest.raises(NativeStateConflictError):
            store.put_response(left, {"id": "resp_root", "status": "completed"})

        stats_payload = store.stats()
        assert stats_payload["schema_version"] == NATIVE_STATE_SCHEMA_VERSION
        assert stats_payload["journal_mode"] == "wal"

    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_response_and_conversation_items_are_ordered_and_cascade(tmp_path: Path) -> None:
    scope = api_key_fingerprint(None)
    with NativeStateStore(tmp_path / "state.sqlite3") as store:
        store.put_response(scope, {"id": "resp_1", "status": "completed"})
        store.append_response_item(
            scope,
            "resp_1",
            {"id": "item_in", "type": "message", "content": "hello"},
            phase="input",
        )
        store.append_response_item(
            scope,
            "resp_1",
            {"id": "item_out", "type": "message", "content": "world"},
            phase="output",
        )
        assert [item["id"] for item in store.list_response_items(scope, "resp_1")] == [
            "item_in",
            "item_out",
        ]
        assert [
            item["id"] for item in store.list_response_items(scope, "resp_1", phase="input")
        ] == ["item_in"]

        store.put_conversation(scope, {"id": "conv_1", "metadata": {"purpose": "test"}})
        store.append_conversation_items(
            scope,
            "conv_1",
            (
                {"id": "msg_1", "role": "user", "content": "first"},
                {"id": "msg_2", "role": "assistant", "content": "second"},
            ),
        )
        first_items, first_revision = store.conversation_items_snapshot(scope, "conv_1")
        assert [item["id"] for item in first_items] == ["msg_1", "msg_2"]
        assert first_revision == 1
        store.append_conversation_items(
            scope,
            "conv_1",
            ({"id": "msg_3", "role": "user", "content": "third"},),
        )
        assert [item["id"] for item in store.list_conversation_items(scope, "conv_1")] == [
            "msg_1",
            "msg_2",
            "msg_3",
        ]
        assert store.conversation_items_snapshot(scope, "conv_1")[1] == 2
        assert store.delete_conversation(scope, "conv_1") is True
        assert store.list_conversation_items(scope, "conv_1") == ()
        assert store.delete_response(scope, "resp_1") is True
        assert store.list_response_items(scope, "resp_1") == ()


def test_restart_preserves_queued_jobs_and_fails_only_interrupted_work(tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite3"
    scope = api_key_fingerprint("owner")
    first = NativeStateStore(path)
    first.put_response(
        scope,
        {"id": "resp_running", "status": "queued"},
        background=True,
        enqueue=True,
    )
    first.put_response(
        scope,
        {"id": "resp_queued", "status": "queued"},
        background=True,
        enqueue=True,
    )
    claimed_scope, claimed = first.claim_next_background_job() or (None, None)
    assert claimed_scope == scope
    assert claimed is not None
    assert claimed["id"] in {"resp_queued", "resp_running"}
    running_id = claimed["id"]
    queued_id = "resp_running" if running_id == "resp_queued" else "resp_queued"
    first.close()  # simulate process exit while one job is in progress

    with NativeStateStore(path) as restarted:
        assert restarted.recovered_interrupted_jobs == 1
        failed = restarted.get_response(scope, running_id)
        assert failed is not None
        assert failed["status"] == "failed"
        assert failed["error"]["code"] == "server_restarted"
        queued = restarted.queued_background_jobs()
        assert len(queued) == 1
        assert queued[0][0] == scope
        assert queued[0][1]["id"] == queued_id


def test_state_store_refuses_symlink_and_cancel_is_scoped(tmp_path: Path) -> None:
    target = tmp_path / "target.sqlite3"
    target.write_bytes(b"")
    link = tmp_path / "state.sqlite3"
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable")
    with pytest.raises(NativeStateError, match="symlink"):
        NativeStateStore(link)

    scope = api_key_fingerprint("a")
    other = api_key_fingerprint("b")
    with NativeStateStore(tmp_path / "safe.sqlite3") as store:
        store.put_response(scope, {"id": "resp", "status": "in_progress"})
        assert store.request_cancel(other, "resp") is False
        assert store.request_cancel(scope, "resp") is True
        assert store.is_cancel_requested(scope, "resp") is True


def test_conversation_update_item_lookup_delete_and_queued_cancel(tmp_path: Path) -> None:
    scope = api_key_fingerprint("owner")
    with NativeStateStore(tmp_path / "state.sqlite3") as store:
        store.put_conversation(
            scope,
            {"id": "conv", "object": "conversation", "metadata": {}},
        )
        appended, append_revision = store.append_conversation_items_with_revision(
            scope,
            "conv",
            ({"id": "item", "type": "message", "role": "user", "content": "x"},),
        )
        assert appended[0]["id"] == "item"
        assert append_revision == 1
        assert store.get_conversation_item(scope, "conv", "item")["id"] == "item"
        assert store.update_conversation(
            scope,
            "conv",
            {"metadata": {"updated": "yes"}},
        )["metadata"] == {"updated": "yes"}
        assert store.delete_conversation_item_with_revision(
            scope,
            "conv",
            "item",
        ) == (True, 2)
        assert store.get_conversation_item(scope, "conv", "item") is None

        store.put_response(
            scope,
            {"id": "queued", "status": "queued"},
            background=True,
            enqueue=True,
        )
        assert store.request_cancel(scope, "queued") is True
        assert store.get_response(scope, "queued")["status"] == "cancelled"
        assert store.queued_background_jobs() == ()


def test_response_compaction_reference_is_scope_bound_and_restart_durable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.sqlite3"
    scope = api_key_fingerprint("owner")
    other = api_key_fingerprint("other")
    token = "nfncmp_private-reference"
    payload = {
        "id": "cmp_1",
        "created_at": 1_700_000_000,
        "model": "nfn-test",
        "messages": [{"role": "user", "content": "remember this"}],
    }

    with NativeStateStore(path) as store:
        assert store.put_response_compaction(
            scope,
            payload,
            encrypted_content=token,
        ) == payload
        assert store.get_response_compaction(scope, token) == payload
        assert store.get_response_compaction(other, token) is None
        assert store.stats()["response_compactions"] == 1

    with NativeStateStore(path) as restarted:
        assert restarted.get_response_compaction(scope, token) == payload


def test_v1_state_migrates_to_current_typed_response_item_schema(tmp_path: Path) -> None:
    path = tmp_path / "migrate.sqlite3"
    scope = api_key_fingerprint("owner")
    with NativeStateStore(path) as store:
        store.put_response(scope, {"id": "resp", "status": "in_progress"})

    connection = sqlite3.connect(path)
    try:
        connection.execute("DROP INDEX response_events_one_terminal")
        connection.execute("DROP TABLE response_events")
        connection.execute(
            "UPDATE native_state_meta SET value='1' WHERE key='schema_version'"
        )
        connection.commit()
    finally:
        connection.close()

    with NativeStateStore(path) as migrated:
        assert migrated.stats()["schema_version"] == NATIVE_STATE_SCHEMA_VERSION == 4
        assert migrated.append_response_events(
            scope,
            "resp",
            ({"type": "response.created", "response": {"id": "resp"}},),
        )[0]["sequence_number"] == 0


def test_v2_state_migrates_to_typed_response_item_schema(tmp_path: Path) -> None:
    path = tmp_path / "migrate-v2.sqlite3"
    with NativeStateStore(path):
        pass

    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "UPDATE native_state_meta SET value='2' WHERE key='schema_version'"
        )
        connection.commit()
    finally:
        connection.close()

    with NativeStateStore(path) as migrated:
        assert migrated.stats()["schema_version"] == NATIVE_STATE_SCHEMA_VERSION == 4


def test_v3_state_migrates_existing_conversation_history_to_revision_zero(
    tmp_path: Path,
) -> None:
    path = tmp_path / "migrate-v3.sqlite3"
    scope = api_key_fingerprint("owner")
    with NativeStateStore(path) as store:
        store.put_conversation(scope, {"id": "conv", "metadata": {}})
        store.append_conversation_items(
            scope,
            "conv",
            ({"id": "existing", "type": "message", "content": "history"},),
        )
        assert store.conversation_items_snapshot(scope, "conv")[1] == 1

    connection = sqlite3.connect(path)
    try:
        connection.execute("ALTER TABLE conversations DROP COLUMN items_revision")
        connection.execute(
            "UPDATE native_state_meta SET value='3' WHERE key='schema_version'"
        )
        connection.commit()
    finally:
        connection.close()

    with NativeStateStore(path) as migrated:
        items, revision = migrated.conversation_items_snapshot(scope, "conv")
        assert [item["id"] for item in items] == ["existing"]
        assert revision == 0
        assert migrated.stats()["schema_version"] == NATIVE_STATE_SCHEMA_VERSION == 4
        migrated.append_conversation_items(
            scope,
            "conv",
            ({"id": "new", "type": "message", "content": "after migration"},),
        )
        assert migrated.conversation_items_snapshot(scope, "conv")[1] == 1


def test_state_rejects_a_newer_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "newer.sqlite3"
    with NativeStateStore(path):
        pass
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "UPDATE native_state_meta SET value=? WHERE key='schema_version'",
            (str(NATIVE_STATE_SCHEMA_VERSION + 1),),
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(NativeStateError, match="Unsupported native state schema version"):
        NativeStateStore(path)


def test_foreground_conversation_revision_cas_has_one_winner_and_atomic_output(
    tmp_path: Path,
) -> None:
    path = tmp_path / "foreground-cas.sqlite3"
    scope = api_key_fingerprint("owner")
    with NativeStateStore(path) as store:
        store.put_conversation(scope, {"id": "conv", "metadata": {}})
        for branch in ("a", "b"):
            store.put_response(
                scope,
                {
                    "id": f"resp_{branch}",
                    "status": "in_progress",
                    "conversation": "conv",
                },
            )
        history, prepared_revision = store.conversation_items_snapshot(scope, "conv")
        assert history == ()
        assert prepared_revision == 0

    barrier = threading.Barrier(2)

    def finish(branch: str) -> tuple[str, str, str]:
        output = {
            "id": f"out_{branch}",
            "type": "message",
            "role": "assistant",
            "content": branch,
        }
        input_item = {
            "id": "shared_input",
            "type": "message",
            "role": "user",
            "content": branch,
        }
        with NativeStateStore(path) as branch_store:
            barrier.wait(timeout=5)
            try:
                commit = branch_store.finish_foreground_response(
                    scope,
                    f"resp_{branch}",
                    status="completed",
                    response_patch={"completed_at": 1, "output": [output]},
                    response_items=(output,),
                    conversation_id="conv",
                    conversation_items=(input_item, output),
                    expected_conversation_revision=prepared_revision,
                )
            except NativeStateConflictError as exc:
                return "conflict", branch, exc.code
            assert commit is not None
            stored, committed_revision = commit
            assert committed_revision == prepared_revision + 1
            return "winner", branch, stored["status"]

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(finish, ("a", "b")))
    winners = [result for result in results if result[0] == "winner"]
    conflicts = [result for result in results if result[0] == "conflict"]
    assert len(winners) == 1
    assert len(conflicts) == 1
    assert conflicts[0][2] == "conversation_conflict"

    winner = winners[0][1]
    loser = conflicts[0][1]
    with NativeStateStore(path) as store:
        committed_items, committed_revision = store.conversation_items_snapshot(scope, "conv")
        assert committed_revision == 1
        assert [item["id"] for item in committed_items] == [
            "shared_input",
            f"out_{winner}",
        ]
        assert store.get_response(scope, f"resp_{winner}")["status"] == "completed"
        assert [
            item["id"]
            for item in store.list_response_items(
                scope,
                f"resp_{winner}",
                phase="output",
            )
        ] == [f"out_{winner}"]
        assert store.get_response(scope, f"resp_{loser}")["status"] == "in_progress"
        assert store.list_response_items(scope, f"resp_{loser}", phase="output") == ()


def test_conversation_item_revisions_are_api_key_scope_isolated(tmp_path: Path) -> None:
    left = api_key_fingerprint("left")
    right = api_key_fingerprint("right")
    with NativeStateStore(tmp_path / "revision-scopes.sqlite3") as store:
        for scope in (left, right):
            store.put_conversation(scope, {"id": "conv", "metadata": {}})
        _items, left_revision = store.append_conversation_items_with_revision(
            left,
            "conv",
            ({"id": "left_item", "type": "message", "content": "left"},),
        )
        assert left_revision == 1
        assert store.conversation_items_snapshot(left, "conv")[1] == 1
        assert store.conversation_items_snapshot(right, "conv") == ((), 0)
        with pytest.raises(KeyError):
            store.conversation_items_snapshot(api_key_fingerprint("missing"), "conv")


def test_conversation_snapshot_keeps_items_and_revision_from_one_read_transaction(
    tmp_path: Path,
) -> None:
    path = tmp_path / "snapshot-atomic.sqlite3"
    scope = api_key_fingerprint("owner")
    reader = NativeStateStore(path)
    writer = NativeStateStore(path)
    reader.put_conversation(scope, {"id": "conv", "metadata": {}})
    second_select_started = threading.Event()
    writer_finished = threading.Event()

    def trace(statement: str) -> None:
        if statement.startswith("SELECT payload_json FROM conversation_items"):
            second_select_started.set()
            assert writer_finished.wait(timeout=5)

    def append_while_snapshot_is_open() -> None:
        assert second_select_started.wait(timeout=5)
        writer.append_conversation_items(
            scope,
            "conv",
            ({"id": "new", "type": "message", "content": "new"},),
        )
        writer_finished.set()

    reader._connection.set_trace_callback(trace)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(append_while_snapshot_is_open)
            items, revision = reader.conversation_items_snapshot(scope, "conv")
            future.result(timeout=5)
        assert items == ()
        assert revision == 0
    finally:
        reader._connection.set_trace_callback(None)
        writer.close()
        reader.close()

    with NativeStateStore(path) as reopened:
        items, revision = reopened.conversation_items_snapshot(scope, "conv")
        assert [item["id"] for item in items] == ["new"]
        assert revision == 1


def test_foreground_commit_rolls_back_output_and_revision_on_item_conflict(
    tmp_path: Path,
) -> None:
    path = tmp_path / "foreground-rollback.sqlite3"
    scope = api_key_fingerprint("owner")
    output = {
        "id": "out",
        "type": "message",
        "role": "assistant",
        "content": "result",
    }
    duplicate = {
        "id": "duplicate",
        "type": "message",
        "role": "user",
        "content": "already stored",
    }
    with NativeStateStore(path) as store:
        store.put_conversation(scope, {"id": "conv", "metadata": {}})
        store.append_conversation_items(scope, "conv", (duplicate,))
        store.put_response(
            scope,
            {"id": "resp", "status": "in_progress", "conversation": "conv"},
        )
        _, revision = store.conversation_items_snapshot(scope, "conv")
        assert revision == 1
        with pytest.raises(NativeStateConflictError) as conflict:
            store.finish_foreground_response(
                scope,
                "resp",
                status="completed",
                response_patch={"completed_at": 1, "output": [output]},
                response_items=(output,),
                conversation_id="conv",
                conversation_items=(duplicate, output),
                expected_conversation_revision=revision,
            )
        assert conflict.value.code == "state_conflict"
        assert store.get_response(scope, "resp")["status"] == "in_progress"
        assert store.list_response_items(scope, "resp", phase="output") == ()
        items, unchanged_revision = store.conversation_items_snapshot(scope, "conv")
        assert [item["id"] for item in items] == ["duplicate"]
        assert unchanged_revision == revision


def test_foreground_commit_without_conversation_is_atomic_and_has_no_revision(
    tmp_path: Path,
) -> None:
    scope = api_key_fingerprint("owner")
    output = {"id": "out", "type": "message", "content": "result"}
    with NativeStateStore(tmp_path / "foreground-no-conversation.sqlite3") as store:
        store.put_response(
            scope,
            {
                "id": "resp",
                "status": "in_progress",
                "previous_response_id": "prior",
            },
        )
        commit = store.finish_foreground_response(
            scope,
            "resp",
            status="completed",
            response_patch={"completed_at": 1, "output": [output]},
            response_items=(output,),
        )
        assert commit is not None
        response, conversation_revision = commit
        assert response["status"] == "completed"
        assert conversation_revision is None
        assert store.list_response_items(scope, "resp", phase="output") == (output,)
        late_output = {"id": "late", "type": "message", "content": "late"}
        with pytest.raises(NativeStateConflictError, match="already terminal"):
            store.finish_foreground_response(
                scope,
                "resp",
                status="completed",
                response_patch={"completed_at": 2, "output": [late_output]},
                response_items=(late_output,),
            )
        assert store.list_response_items(scope, "resp", phase="output") == (output,)
        assert store.get_response(scope, "resp")["output"] == [output]


def test_cancelled_foreground_conversation_response_does_not_append_history(
    tmp_path: Path,
) -> None:
    scope = api_key_fingerprint("owner")
    output = {"id": "out", "type": "message", "content": "partial"}
    with NativeStateStore(tmp_path / "foreground-cancelled.sqlite3") as store:
        store.put_conversation(scope, {"id": "conv", "metadata": {}})
        store.put_response(
            scope,
            {"id": "resp", "status": "in_progress", "conversation": "conv"},
        )
        with pytest.raises(ValueError, match="only completed or incomplete"):
            store.finish_foreground_response(
                scope,
                "resp",
                status="cancelled",
                response_patch={"completed_at": 1, "output": [output]},
                response_items=(output,),
                conversation_id="conv",
                conversation_items=(output,),
                expected_conversation_revision=0,
            )
        assert store.get_response(scope, "resp")["status"] == "in_progress"
        assert store.list_response_items(scope, "resp", phase="output") == ()
        assert store.conversation_items_snapshot(scope, "conv") == ((), 0)
        commit = store.finish_foreground_response(
            scope,
            "resp",
            status="cancelled",
            response_patch={"completed_at": 1, "output": [output]},
            response_items=(output,),
        )
        assert commit is not None
        response, conversation_revision = commit
        assert response["status"] == "cancelled"
        assert conversation_revision is None
        assert store.list_response_items(scope, "resp", phase="output") == (output,)
        assert store.conversation_items_snapshot(scope, "conv") == ((), 0)


def test_background_conversation_completion_uses_the_same_revision_cas(
    tmp_path: Path,
) -> None:
    path = tmp_path / "background-cas.sqlite3"
    scope = api_key_fingerprint("owner")
    with NativeStateStore(path) as store:
        store.put_conversation(scope, {"id": "conv", "metadata": {}})
        for branch in ("a", "b"):
            store.put_response(
                scope,
                {
                    "id": f"resp_{branch}",
                    "status": "queued",
                    "conversation": "conv",
                },
                background=True,
                enqueue=True,
            )
        _, prepared_revision = store.conversation_items_snapshot(scope, "conv")

    barrier = threading.Barrier(2)

    def finish(branch: str) -> tuple[str, str, str]:
        output = {
            "id": f"out_{branch}",
            "type": "message",
            "role": "assistant",
            "content": branch,
        }
        conversation_input = {
            "id": f"in_{branch}",
            "type": "message",
            "role": "user",
            "content": branch,
        }
        with NativeStateStore(path) as branch_store:
            barrier.wait(timeout=5)
            try:
                stored = branch_store.finish_background_job(
                    scope,
                    f"resp_{branch}",
                    status="completed",
                    response_patch={"completed_at": 1, "output": [output]},
                    response_item=output,
                    conversation_id="conv",
                    conversation_items=(conversation_input, output),
                    expected_conversation_revision=prepared_revision,
                )
            except NativeStateConflictError as exc:
                return "conflict", branch, exc.code
            assert stored is not None
            return "winner", branch, stored["status"]

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(finish, ("a", "b")))
    winners = [result for result in results if result[0] == "winner"]
    conflicts = [result for result in results if result[0] == "conflict"]
    assert len(winners) == 1
    assert len(conflicts) == 1
    assert conflicts[0][2] == "conversation_conflict"

    winner = winners[0][1]
    loser = conflicts[0][1]
    with NativeStateStore(path) as store:
        items, revision = store.conversation_items_snapshot(scope, "conv")
        assert revision == 1
        assert [item["id"] for item in items] == [f"in_{winner}", f"out_{winner}"]
        assert store.get_response(scope, f"resp_{winner}")["status"] == "completed"
        assert store.get_response(scope, f"resp_{loser}")["status"] == "queued"
        assert store.list_response_items(scope, f"resp_{loser}", phase="output") == ()
        assert [job[1]["id"] for job in store.queued_background_jobs()] == [f"resp_{loser}"]


def test_background_conversation_completion_fails_closed_without_revision(
    tmp_path: Path,
) -> None:
    scope = api_key_fingerprint("owner")
    with NativeStateStore(tmp_path / "background-no-cas.sqlite3") as store:
        store.put_conversation(scope, {"id": "conv", "metadata": {}})
        store.put_response(
            scope,
            {"id": "resp", "status": "queued", "conversation": "conv"},
            background=True,
            enqueue=True,
        )
        output = {"id": "out", "type": "message", "content": "result"}
        with pytest.raises(ValueError, match="revision CAS"):
            store.finish_background_job(
                scope,
                "resp",
                status="completed",
                response_patch={"completed_at": 1, "output": [output]},
                response_item=output,
            )
        assert store.get_response(scope, "resp")["status"] == "queued"
        assert store.list_response_items(scope, "resp", phase="output") == ()
        assert store.conversation_items_snapshot(scope, "conv") == ((), 0)


def test_response_event_sequence_terminal_atomicity_scoping_and_cascade(
    tmp_path: Path,
) -> None:
    left = api_key_fingerprint("left")
    right = api_key_fingerprint("right")
    path = tmp_path / "events.sqlite3"
    stream_internal = {"stream_requested": True}
    with NativeStateStore(path) as store:
        for scope in (left, right):
            store.put_response(
                scope,
                {
                    "id": "resp_shared",
                    "status": "queued",
                    "background": True,
                    "_nfn": stream_internal,
                },
                background=True,
                enqueue=True,
            )
        first = store.append_response_events(
            left,
            "resp_shared",
            (
                {"type": "response.created", "response": {"id": "resp_shared"}},
                {"type": "response.output_text.delta", "delta": "hello"},
            ),
        )
        assert [event["sequence_number"] for event in first] == [0, 1]
        terminal = store.finish_background_job(
            left,
            "resp_shared",
            status="completed",
            response_patch={"completed_at": 1, "output": []},
            response_events=({"type": "response.output_text.done", "text": "hello"},),
        )
        assert terminal is not None and terminal["status"] == "completed"
        events = store.list_response_events(left, "resp_shared")
        assert [event["sequence_number"] for event in events] == [0, 1, 2, 3]
        assert events[-1]["type"] == "response.completed"
        assert store.list_response_events(left, "resp_shared", starting_after=1) == events[2:]
        assert store.list_response_events(right, "resp_shared") == ()
        with pytest.raises(NativeStateConflictError, match="terminal"):
            store.append_response_events(
                left,
                "resp_shared",
                ({"type": "response.output_text.delta", "delta": "late"},),
            )
        assert store.delete_response(left, "resp_shared") is True
        assert store.stats()["response_events"] == 0


def test_streamed_background_cancel_and_restart_recovery_persist_terminal_events(
    tmp_path: Path,
) -> None:
    path = tmp_path / "terminal.sqlite3"
    scope = api_key_fingerprint("owner")
    first = NativeStateStore(path)
    for response_id in ("resp_cancel", "resp_restart"):
        first.put_response(
            scope,
            {
                "id": response_id,
                "status": "queued",
                "background": True,
                "_nfn": {"stream_requested": True},
            },
            background=True,
            enqueue=True,
        )
        first.append_response_events(
            scope,
            response_id,
            ({"type": "response.created", "response": {"id": response_id}},),
        )

    assert first.request_cancel(scope, "resp_cancel") is True
    cancelled_events = first.list_response_events(scope, "resp_cancel")
    assert cancelled_events[-1]["type"] == "response.incomplete"
    assert cancelled_events[-1]["response"]["status"] == "cancelled"
    assert sum(
        event["type"] in {
            "response.completed",
            "response.failed",
            "response.incomplete",
        }
        for event in cancelled_events
    ) == 1

    claimed_scope, claimed = first.claim_next_background_job() or (None, None)
    assert claimed_scope == scope
    assert claimed is not None and claimed["id"] == "resp_restart"
    first.append_response_events(
        scope,
        "resp_restart",
        ({"type": "response.in_progress", "response": claimed},),
    )
    first.close()

    with NativeStateStore(path) as restarted:
        failed = restarted.get_response(scope, "resp_restart")
        assert failed is not None and failed["error"]["code"] == "server_restarted"
        failed_events = restarted.list_response_events(scope, "resp_restart")
        assert failed_events[-1]["type"] == "response.failed"
        assert failed_events[-1]["response"]["error"]["code"] == "server_restarted"
        assert sum(
            event["type"] in {
                "response.completed",
                "response.failed",
                "response.incomplete",
            }
            for event in failed_events
        ) == 1


def test_concurrent_response_event_appenders_allocate_one_contiguous_sequence(
    tmp_path: Path,
) -> None:
    scope = api_key_fingerprint("owner")
    path = tmp_path / "concurrent.sqlite3"
    first = NativeStateStore(path)
    second = NativeStateStore(path)
    try:
        first.put_response(scope, {"id": "resp", "status": "in_progress"})

        def append(index: int) -> int:
            store = first if index % 2 == 0 else second
            return store.append_response_events(
                scope,
                "resp",
                ({"type": "response.output_text.delta", "delta": str(index)},),
            )[0]["sequence_number"]

        with ThreadPoolExecutor(max_workers=8) as executor:
            allocated = list(executor.map(append, range(32)))

        assert sorted(allocated) == list(range(32))
        persisted = first.list_response_events(scope, "resp")
        assert [event["sequence_number"] for event in persisted] == list(range(32))
        assert {event["delta"] for event in persisted} == {str(index) for index in range(32)}
    finally:
        second.close()
        first.close()
