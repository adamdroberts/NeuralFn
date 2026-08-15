"""Versioned local state for the standalone native inference server.

The store is deliberately independent of the editor database.  Records are
partitioned by an API-key fingerprint, SQLite runs in WAL mode, and only token
or JSON history is durable: resident KV/cache buffers never enter this file.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import stat
import threading
import time
from typing import Any, Iterator, Mapping, Sequence


NATIVE_STATE_SCHEMA_VERSION = 4
ANONYMOUS_API_KEY_FINGERPRINT = hashlib.sha256(
    b"neuralfn-native-inference-anonymous-v1"
).hexdigest()
_TERMINAL_RESPONSE_EVENT_TYPES = frozenset(
    {"response.completed", "response.failed", "response.incomplete"}
)
_TERMINAL_RESPONSE_STATUSES = frozenset(
    {"completed", "failed", "incomplete", "cancelled"}
)


class NativeStateError(RuntimeError):
    """Base error for local native inference state."""


class NativeStateConflictError(NativeStateError):
    """Raised for identifier collisions and optimistic-concurrency conflicts."""

    def __init__(self, message: str, *, code: str = "state_conflict") -> None:
        super().__init__(message)
        self.code = code


def api_key_fingerprint(api_key: str | None) -> str:
    """Return a non-reversible stable scope identifier for one Bearer key."""

    if api_key is None:
        return ANONYMOUS_API_KEY_FINGERPRINT
    normalized = str(api_key)
    if not normalized:
        raise ValueError("api_key must not be empty")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _scope(value: str) -> str:
    normalized = str(value).strip().lower()
    if (
        len(normalized) != 64
        or any(character not in "0123456789abcdef" for character in normalized)
    ):
        raise ValueError("scope must be a lowercase SHA-256 fingerprint")
    return normalized


def _identifier(value: Any, *, field: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field} must be a non-empty string")
    if len(normalized) > 255:
        raise ValueError(f"{field} must not exceed 255 characters")
    return normalized


def _conversation_revision(value: Any, *, field: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > 9_223_372_036_854_775_806
    ):
        raise ValueError(f"{field} must be an integer from 0 through 2^63-2")
    return value


def _json_object(value: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    payload = deepcopy(dict(value))
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be finite JSON data") from exc
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):  # defensive: Mapping above should ensure this
        raise TypeError(f"{field} must encode to an object")
    return decoded


def _encode(value: Mapping[str, Any], *, field: str) -> str:
    return json.dumps(
        _json_object(value, field=field),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _decode(value: str) -> dict[str, Any]:
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise NativeStateError("Stored state payload is not a JSON object")
    return payload


def _public_response(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the public response snapshot embedded in durable stream events."""

    public = deepcopy(dict(payload))
    public.pop("_nfn", None)
    public.pop("cancel_requested", None)
    return public


def _stream_was_requested(payload: Mapping[str, Any]) -> bool:
    internal = payload.get("_nfn")
    return bool(
        isinstance(internal, Mapping)
        and internal.get("stream_requested") is True
        and payload.get("background") is True
    )


class NativeStateStore:
    """Thread-safe SQLite state scoped independently from editor persistence."""

    def __init__(self, path: str | Path) -> None:
        requested = Path(path).expanduser()
        if requested.is_symlink():
            raise NativeStateError(f"State database must not be a symlink: {requested}")
        requested.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        resolved = requested.resolve()
        if resolved.exists() and not resolved.is_file():
            raise NativeStateError(f"State database path is not a file: {resolved}")
        self.path = resolved
        self._secure_create_file()
        self._lock = threading.RLock()
        # Every supported Responses service over this store shares one
        # process-local linearization boundary for terminal commits, semantic
        # deletion, and prefix-cache publication.  SQLite remains the durable
        # boundary; this private lock coordinates in-process resident state.
        self._responses_transition_lock = threading.RLock()
        self._connection = sqlite3.connect(
            str(self.path),
            timeout=5.0,
            isolation_level=None,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._closed = False
        with self._lock:
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._connection.execute("PRAGMA busy_timeout=5000")
            journal = self._connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
            if str(journal).lower() != "wal":
                self._connection.close()
                self._closed = True
                raise NativeStateError("Native state database could not enable WAL mode")
            self._connection.execute("PRAGMA synchronous=FULL")
            self._initialize_schema()
            self.recovered_interrupted_jobs = self._recover_interrupted_jobs_locked()

    def _secure_create_file(self) -> None:
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(self.path, flags, stat.S_IRUSR | stat.S_IWUSR)
        except OSError as exc:
            raise NativeStateError(f"Unable to create state database securely: {self.path}") from exc
        try:
            os.fchmod(descriptor, stat.S_IRUSR | stat.S_IWUSR)
        finally:
            os.close(descriptor)

    def _ensure_open(self) -> None:
        if self._closed:
            raise NativeStateError("Native state database is closed")

    @contextmanager
    def _transaction(self, *, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._ensure_open()
            self._connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            try:
                yield self._connection
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
            else:
                self._connection.execute("COMMIT")

    def _initialize_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS native_state_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS responses (
                scope TEXT NOT NULL,
                id TEXT NOT NULL,
                status TEXT NOT NULL,
                background INTEGER NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                previous_response_id TEXT,
                conversation_id TEXT,
                cancel_requested INTEGER NOT NULL DEFAULT 0,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (scope, id)
            );
            CREATE INDEX IF NOT EXISTS responses_scope_created
                ON responses(scope, created_at, id);
            CREATE TABLE IF NOT EXISTS response_items (
                scope TEXT NOT NULL,
                response_id TEXT NOT NULL,
                position INTEGER NOT NULL,
                item_id TEXT NOT NULL,
                phase TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (scope, response_id, position),
                UNIQUE (scope, response_id, item_id),
                FOREIGN KEY (scope, response_id)
                    REFERENCES responses(scope, id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS response_events (
                scope TEXT NOT NULL,
                response_id TEXT NOT NULL,
                sequence_number INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                terminal INTEGER NOT NULL DEFAULT 0,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (scope, response_id, sequence_number),
                FOREIGN KEY (scope, response_id)
                    REFERENCES responses(scope, id) ON DELETE CASCADE,
                CHECK (sequence_number >= 0),
                CHECK (terminal IN (0, 1))
            );
            CREATE UNIQUE INDEX IF NOT EXISTS response_events_one_terminal
                ON response_events(scope, response_id) WHERE terminal=1;
            CREATE TABLE IF NOT EXISTS conversations (
                scope TEXT NOT NULL,
                id TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                items_revision INTEGER NOT NULL DEFAULT 0
                    CHECK (items_revision >= 0),
                payload_json TEXT NOT NULL,
                PRIMARY KEY (scope, id)
            );
            CREATE TABLE IF NOT EXISTS conversation_items (
                scope TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                position INTEGER NOT NULL,
                item_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (scope, conversation_id, position),
                UNIQUE (scope, conversation_id, item_id),
                FOREIGN KEY (scope, conversation_id)
                    REFERENCES conversations(scope, id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS response_compactions (
                scope TEXT NOT NULL,
                id TEXT NOT NULL,
                token_sha256 TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (scope, id),
                UNIQUE (scope, token_sha256)
            );
            CREATE TABLE IF NOT EXISTS background_jobs (
                scope TEXT NOT NULL,
                response_id TEXT NOT NULL,
                status TEXT NOT NULL,
                enqueued_at INTEGER NOT NULL,
                started_at INTEGER,
                updated_at INTEGER NOT NULL,
                error_json TEXT,
                PRIMARY KEY (scope, response_id),
                FOREIGN KEY (scope, response_id)
                    REFERENCES responses(scope, id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS background_jobs_queue
                ON background_jobs(status, enqueued_at, response_id);
            """
        )
        row = self._connection.execute(
            "SELECT value FROM native_state_meta WHERE key='schema_version'"
        ).fetchone()
        if row is None:
            self._connection.execute(
                "INSERT INTO native_state_meta(key, value) VALUES('schema_version', ?)",
                (str(NATIVE_STATE_SCHEMA_VERSION),),
            )
        else:
            try:
                version = int(row["value"])
            except (TypeError, ValueError) as exc:
                raise NativeStateError("Native state schema version is invalid") from exc
            if version in {1, 2, 3}:
                # Version 2 added the scoped, cascading response-event ledger.
                # Version 3 makes typed function-call and function-call-output
                # response items part of the durable semantic contract. Version
                # 4 adds a per-conversation item revision used for snapshot/CAS
                # commits. Existing history becomes revision-zero baseline state.
                columns = {
                    str(column["name"])
                    for column in self._connection.execute(
                        "PRAGMA table_info(conversations)"
                    ).fetchall()
                }
                if "items_revision" not in columns:
                    self._connection.execute(
                        "ALTER TABLE conversations ADD COLUMN items_revision "
                        "INTEGER NOT NULL DEFAULT 0 CHECK (items_revision >= 0)"
                    )
                self._connection.execute(
                    "UPDATE native_state_meta SET value=? WHERE key='schema_version'",
                    (str(NATIVE_STATE_SCHEMA_VERSION),),
                )
            elif version != NATIVE_STATE_SCHEMA_VERSION:
                raise NativeStateError(
                    "Unsupported native state schema version "
                    f"{version}; expected {NATIVE_STATE_SCHEMA_VERSION}"
                )

    def _append_response_events_locked(
        self,
        connection: sqlite3.Connection,
        scope: str,
        response_id: str,
        events: Sequence[Mapping[str, Any]],
    ) -> tuple[dict[str, Any], ...]:
        """Allocate and append response event sequence numbers in one transaction."""

        if not events:
            return ()
        records = tuple(_json_object(event, field="response event") for event in events)
        terminal_indexes: list[int] = []
        for index, record in enumerate(records):
            if "sequence_number" in record:
                raise ValueError("response event sequence_number is allocated by the store")
            event_type = _identifier(record.get("type"), field="response event.type")
            record["type"] = event_type
            if event_type in _TERMINAL_RESPONSE_EVENT_TYPES:
                terminal_indexes.append(index)
        if len(terminal_indexes) > 1 or (
            terminal_indexes and terminal_indexes[0] != len(records) - 1
        ):
            raise ValueError("a terminal response event must be the final appended event")

        parent = connection.execute(
            "SELECT 1 FROM responses WHERE scope=? AND id=?",
            (scope, response_id),
        ).fetchone()
        if parent is None:
            raise KeyError(response_id)
        existing_terminal = connection.execute(
            "SELECT 1 FROM response_events "
            "WHERE scope=? AND response_id=? AND terminal=1",
            (scope, response_id),
        ).fetchone()
        if existing_terminal is not None:
            raise NativeStateConflictError(
                f"Response {response_id!r} already has a terminal stream event"
            )
        start = int(
            connection.execute(
                "SELECT COALESCE(MAX(sequence_number), -1) + 1 "
                "FROM response_events WHERE scope=? AND response_id=?",
                (scope, response_id),
            ).fetchone()[0]
        )
        if start > 9_223_372_036_854_775_807 - (len(records) - 1):
            raise NativeStateError("Response event sequence space is exhausted")
        inserted: list[dict[str, Any]] = []
        for offset, record in enumerate(records):
            sequence_number = start + offset
            stored = {**record, "sequence_number": sequence_number}
            terminal = int(record["type"] in _TERMINAL_RESPONSE_EVENT_TYPES)
            try:
                connection.execute(
                    "INSERT INTO response_events(scope,response_id,sequence_number,"
                    "event_type,terminal,payload_json) VALUES(?,?,?,?,?,?)",
                    (
                        scope,
                        response_id,
                        sequence_number,
                        record["type"],
                        terminal,
                        _encode(stored, field="response event"),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise NativeStateConflictError(
                    f"Response event sequence {sequence_number} already exists"
                ) from exc
            inserted.append(stored)
        return tuple(inserted)

    def _append_response_items_locked(
        self,
        connection: sqlite3.Connection,
        scope: str,
        response_id: str,
        records: Sequence[Mapping[str, Any]],
        *,
        phase: str,
    ) -> None:
        if not records:
            return
        position = int(
            connection.execute(
                "SELECT COALESCE(MAX(position), -1) + 1 FROM response_items "
                "WHERE scope=? AND response_id=?",
                (scope, response_id),
            ).fetchone()[0]
        )
        for offset, record in enumerate(records):
            item_id = _identifier(record.get("id"), field="response item.id")
            try:
                connection.execute(
                    "INSERT INTO response_items(scope,response_id,position,item_id,phase,payload_json) "
                    "VALUES(?,?,?,?,?,?)",
                    (
                        scope,
                        response_id,
                        position + offset,
                        item_id,
                        phase,
                        _encode(record, field="response item"),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise NativeStateConflictError(
                    f"Response item {item_id!r} already exists"
                ) from exc

    def _append_conversation_items_locked(
        self,
        connection: sqlite3.Connection,
        scope: str,
        conversation_id: str,
        records: Sequence[Mapping[str, Any]],
    ) -> None:
        if not records:
            return
        position = int(
            connection.execute(
                "SELECT COALESCE(MAX(position), -1) + 1 FROM conversation_items "
                "WHERE scope=? AND conversation_id=?",
                (scope, conversation_id),
            ).fetchone()[0]
        )
        for offset, record in enumerate(records):
            item_id = _identifier(record.get("id"), field="conversation item.id")
            try:
                connection.execute(
                    "INSERT INTO conversation_items(scope,conversation_id,position,item_id,payload_json) "
                    "VALUES(?,?,?,?,?)",
                    (
                        scope,
                        conversation_id,
                        position + offset,
                        item_id,
                        _encode(record, field="conversation item"),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise NativeStateConflictError(
                    f"Conversation item {item_id!r} already exists"
                ) from exc

    def _reserve_conversation_revision_locked(
        self,
        connection: sqlite3.Connection,
        scope: str,
        conversation_id: str,
        expected_revision: int,
        *,
        updated_at: int,
    ) -> int:
        """Atomically reserve the next item revision or fail with a stable code."""

        cursor = connection.execute(
            "UPDATE conversations SET items_revision=items_revision+1,updated_at=? "
            "WHERE scope=? AND id=? AND items_revision=?",
            (updated_at, scope, conversation_id, expected_revision),
        )
        if cursor.rowcount == 1:
            return expected_revision + 1
        current = connection.execute(
            "SELECT items_revision FROM conversations WHERE scope=? AND id=?",
            (scope, conversation_id),
        ).fetchone()
        if current is None:
            raise KeyError(conversation_id)
        raise NativeStateConflictError(
            f"Conversation {conversation_id!r} changed after revision {expected_revision}",
            code="conversation_conflict",
        )

    def _recover_interrupted_jobs_locked(self) -> int:
        error = {
            "type": "server_error",
            "code": "server_restarted",
            "message": "Background generation was interrupted by a server restart.",
        }
        now = int(time.time())
        with self._transaction(immediate=True) as connection:
            rows = connection.execute(
                "SELECT scope, response_id FROM background_jobs WHERE status='in_progress'"
            ).fetchall()
            encoded_error = _encode(error, field="restart error")
            for row in rows:
                response = connection.execute(
                    "SELECT payload_json FROM responses WHERE scope=? AND id=?",
                    (row["scope"], row["response_id"]),
                ).fetchone()
                if response is not None:
                    payload = _decode(response["payload_json"])
                    payload.update({"status": "failed", "error": error})
                    connection.execute(
                        "UPDATE responses SET status='failed', updated_at=?, payload_json=? "
                        "WHERE scope=? AND id=?",
                        (
                            now,
                            _encode(payload, field="response"),
                            row["scope"],
                            row["response_id"],
                        ),
                    )
                    if _stream_was_requested(payload):
                        self._append_response_events_locked(
                            connection,
                            str(row["scope"]),
                            str(row["response_id"]),
                            (
                                {
                                    "type": "response.failed",
                                    "response": _public_response(payload),
                                },
                            ),
                        )
                connection.execute(
                    "UPDATE background_jobs SET status='failed', updated_at=?, error_json=? "
                    "WHERE scope=? AND response_id=?",
                    (now, encoded_error, row["scope"], row["response_id"]),
                )
            return len(rows)

    def put_response(
        self,
        scope: str,
        payload: Mapping[str, Any],
        *,
        background: bool = False,
        enqueue: bool = False,
        response_events: Sequence[Mapping[str, Any]] = (),
    ) -> dict[str, Any]:
        normalized_scope = _scope(scope)
        record = _json_object(payload, field="response")
        response_id = _identifier(record.get("id"), field="response.id")
        status_value = _identifier(record.get("status", "queued"), field="response.status")
        created_at = record.get("created_at", int(time.time()))
        if isinstance(created_at, bool) or not isinstance(created_at, int) or created_at < 0:
            raise ValueError("response.created_at must be a non-negative integer")
        previous_id = record.get("previous_response_id")
        if previous_id is not None:
            previous_id = _identifier(previous_id, field="response.previous_response_id")
        conversation_id = record.get("conversation") or record.get("conversation_id")
        if isinstance(conversation_id, Mapping):
            conversation_id = conversation_id.get("id")
        if conversation_id is not None:
            conversation_id = _identifier(conversation_id, field="response.conversation_id")
        record.update(
            {
                "id": response_id,
                "status": status_value,
                "background": bool(background),
                "created_at": created_at,
            }
        )
        now = int(time.time())
        with self._transaction(immediate=True) as connection:
            try:
                connection.execute(
                    "INSERT INTO responses(scope,id,status,background,created_at,updated_at,"
                    "previous_response_id,conversation_id,cancel_requested,payload_json) "
                    "VALUES(?,?,?,?,?,?,?,?,0,?)",
                    (
                        normalized_scope,
                        response_id,
                        status_value,
                        int(bool(background)),
                        created_at,
                        now,
                        previous_id,
                        conversation_id,
                        _encode(record, field="response"),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise NativeStateConflictError(f"Response {response_id!r} already exists") from exc
            if enqueue:
                if not background:
                    raise ValueError("Only a background response may be enqueued")
                connection.execute(
                    "INSERT INTO background_jobs(scope,response_id,status,enqueued_at,updated_at) "
                    "VALUES(?,?,'queued',?,?)",
                    (normalized_scope, response_id, now, now),
                )
            if response_events:
                self._append_response_events_locked(
                    connection,
                    normalized_scope,
                    response_id,
                    response_events,
                )
        return deepcopy(record)

    def get_response(self, scope: str, response_id: str) -> dict[str, Any] | None:
        with self._lock:
            self._ensure_open()
            row = self._connection.execute(
                "SELECT payload_json FROM responses WHERE scope=? AND id=?",
                (_scope(scope), _identifier(response_id, field="response_id")),
            ).fetchone()
            return _decode(row["payload_json"]) if row is not None else None

    def append_response_events(
        self,
        scope: str,
        response_id: str,
        events: Sequence[Mapping[str, Any]],
    ) -> tuple[dict[str, Any], ...]:
        """Durably append semantic events with scoped, contiguous sequence numbers."""

        normalized_scope = _scope(scope)
        normalized_id = _identifier(response_id, field="response_id")
        with self._transaction(immediate=True) as connection:
            return self._append_response_events_locked(
                connection,
                normalized_scope,
                normalized_id,
                events,
            )

    def list_response_events(
        self,
        scope: str,
        response_id: str,
        *,
        starting_after: int = -1,
    ) -> tuple[dict[str, Any], ...]:
        """Return persisted semantic events after one sequence cursor."""

        if (
            isinstance(starting_after, bool)
            or not isinstance(starting_after, int)
            or starting_after < -1
            or starting_after > 9_223_372_036_854_775_807
        ):
            raise ValueError("starting_after must be a signed 64-bit integer at least -1")
        normalized_scope = _scope(scope)
        normalized_id = _identifier(response_id, field="response_id")
        with self._lock:
            self._ensure_open()
            parent = self._connection.execute(
                "SELECT 1 FROM responses WHERE scope=? AND id=?",
                (normalized_scope, normalized_id),
            ).fetchone()
            if parent is None:
                raise KeyError(normalized_id)
            rows = self._connection.execute(
                "SELECT payload_json FROM response_events "
                "WHERE scope=? AND response_id=? AND sequence_number>? "
                "ORDER BY sequence_number",
                (normalized_scope, normalized_id, starting_after),
            ).fetchall()
            return tuple(_decode(row["payload_json"]) for row in rows)

    def latest_response_event(
        self,
        scope: str,
        response_id: str,
    ) -> dict[str, Any] | None:
        """Return the highest persisted sequence event for one scoped response."""

        normalized_scope = _scope(scope)
        normalized_id = _identifier(response_id, field="response_id")
        with self._lock:
            self._ensure_open()
            parent = self._connection.execute(
                "SELECT 1 FROM responses WHERE scope=? AND id=?",
                (normalized_scope, normalized_id),
            ).fetchone()
            if parent is None:
                raise KeyError(normalized_id)
            row = self._connection.execute(
                "SELECT payload_json FROM response_events "
                "WHERE scope=? AND response_id=? ORDER BY sequence_number DESC LIMIT 1",
                (normalized_scope, normalized_id),
            ).fetchone()
            return _decode(row["payload_json"]) if row is not None else None

    def update_response(
        self,
        scope: str,
        response_id: str,
        patch: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        normalized_scope = _scope(scope)
        normalized_id = _identifier(response_id, field="response_id")
        changes = _json_object(patch, field="response patch")
        if "id" in changes and changes["id"] != normalized_id:
            raise ValueError("response id is immutable")
        now = int(time.time())
        with self._transaction(immediate=True) as connection:
            row = connection.execute(
                "SELECT payload_json FROM responses WHERE scope=? AND id=?",
                (normalized_scope, normalized_id),
            ).fetchone()
            if row is None:
                return None
            payload = _decode(row["payload_json"])
            payload.update(changes)
            payload["id"] = normalized_id
            status_value = _identifier(payload.get("status", "in_progress"), field="response.status")
            connection.execute(
                "UPDATE responses SET status=?, updated_at=?, payload_json=? "
                "WHERE scope=? AND id=?",
                (
                    status_value,
                    now,
                    _encode(payload, field="response"),
                    normalized_scope,
                    normalized_id,
                ),
            )
            return payload

    def delete_response(self, scope: str, response_id: str) -> bool:
        with self._transaction(immediate=True) as connection:
            cursor = connection.execute(
                "DELETE FROM responses WHERE scope=? AND id=?",
                (_scope(scope), _identifier(response_id, field="response_id")),
            )
            return cursor.rowcount == 1

    def append_response_item(
        self,
        scope: str,
        response_id: str,
        item: Mapping[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        normalized_scope = _scope(scope)
        normalized_response = _identifier(response_id, field="response_id")
        record = _json_object(item, field="response item")
        item_id = _identifier(record.get("id"), field="response item.id")
        phase_value = str(phase).strip().lower()
        if phase_value not in {"input", "output"}:
            raise ValueError("response item phase must be input or output")
        with self._transaction(immediate=True) as connection:
            parent = connection.execute(
                "SELECT 1 FROM responses WHERE scope=? AND id=?",
                (normalized_scope, normalized_response),
            ).fetchone()
            if parent is None:
                raise KeyError(normalized_response)
            position = connection.execute(
                "SELECT COALESCE(MAX(position), -1) + 1 FROM response_items "
                "WHERE scope=? AND response_id=?",
                (normalized_scope, normalized_response),
            ).fetchone()[0]
            try:
                connection.execute(
                    "INSERT INTO response_items(scope,response_id,position,item_id,phase,payload_json) "
                    "VALUES(?,?,?,?,?,?)",
                    (
                        normalized_scope,
                        normalized_response,
                        position,
                        item_id,
                        phase_value,
                        _encode(record, field="response item"),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise NativeStateConflictError(f"Response item {item_id!r} already exists") from exc
        return deepcopy(record)

    def list_response_items(
        self,
        scope: str,
        response_id: str,
        *,
        phase: str | None = None,
    ) -> tuple[dict[str, Any], ...]:
        normalized_scope = _scope(scope)
        normalized_response = _identifier(response_id, field="response_id")
        parameters: list[Any] = [normalized_scope, normalized_response]
        query = (
            "SELECT payload_json FROM response_items WHERE scope=? AND response_id=?"
        )
        if phase is not None:
            phase_value = str(phase).strip().lower()
            if phase_value not in {"input", "output"}:
                raise ValueError("response item phase must be input or output")
            query += " AND phase=?"
            parameters.append(phase_value)
        query += " ORDER BY position"
        with self._lock:
            self._ensure_open()
            rows = self._connection.execute(query, parameters).fetchall()
            return tuple(_decode(row["payload_json"]) for row in rows)

    def finish_foreground_response(
        self,
        scope: str,
        response_id: str,
        *,
        status: str,
        response_patch: Mapping[str, Any],
        response_items: Sequence[Mapping[str, Any]],
        conversation_id: str | None = None,
        conversation_items: Sequence[Mapping[str, Any]] = (),
        expected_conversation_revision: int | None = None,
    ) -> tuple[dict[str, Any], int | None] | None:
        """Atomically publish output and return it with the post-commit revision."""

        normalized_status = str(status).strip().lower()
        if normalized_status not in {"completed", "incomplete", "cancelled"}:
            raise ValueError("foreground output terminal status is invalid")
        normalized_scope = _scope(scope)
        normalized_id = _identifier(response_id, field="response_id")
        patch = _json_object(response_patch, field="response patch")
        if "id" in patch and patch["id"] != normalized_id:
            raise ValueError("response id is immutable")
        output_records = tuple(
            _json_object(item, field="response item") for item in response_items
        )
        if not output_records:
            raise ValueError("foreground response_items must not be empty")
        for record in output_records:
            _identifier(record.get("id"), field="response item.id")
        if "output" in patch and patch["output"] != list(output_records):
            raise ValueError("response patch output must match response_items")

        normalized_conversation: str | None = None
        conversation_records = tuple(
            _json_object(item, field="conversation item") for item in conversation_items
        )
        expected_revision: int | None = None
        if conversation_id is None:
            if conversation_records or expected_conversation_revision is not None:
                raise ValueError(
                    "conversation_id is required for conversation items or revision CAS"
                )
        else:
            normalized_conversation = _identifier(
                conversation_id,
                field="conversation_id",
            )
            if normalized_status not in {"completed", "incomplete"}:
                raise ValueError(
                    "only completed or incomplete foreground responses may append a conversation"
                )
            if not conversation_records:
                raise ValueError("conversation_items must not be empty for a conversation commit")
            if expected_conversation_revision is None:
                raise ValueError("expected_conversation_revision is required")
            expected_revision = _conversation_revision(
                expected_conversation_revision,
                field="expected_conversation_revision",
            )
            conversation_by_id = {
                _identifier(item.get("id"), field="conversation item.id"): item
                for item in conversation_records
            }
            if any(
                conversation_by_id.get(str(item["id"])) != item
                for item in output_records
            ):
                raise ValueError(
                    "conversation_items must include every foreground response output item"
                )

        now = int(time.time())
        with self._transaction(immediate=True) as connection:
            row = connection.execute(
                "SELECT status,background,conversation_id,payload_json FROM responses "
                "WHERE scope=? AND id=?",
                (normalized_scope, normalized_id),
            ).fetchone()
            if row is None:
                return None
            if bool(row["background"]):
                raise ValueError("finish_foreground_response requires a foreground response")
            stored_conversation = (
                str(row["conversation_id"]) if row["conversation_id"] is not None else None
            )
            if (
                stored_conversation is not None
                and normalized_status in {"completed", "incomplete"}
                and normalized_conversation is None
            ):
                raise ValueError(
                    "a conversation-linked foreground completion requires revision CAS"
                )
            if (
                stored_conversation != normalized_conversation
                and not (
                    stored_conversation is not None
                    and normalized_conversation is None
                    and normalized_status == "cancelled"
                )
            ):
                raise ValueError("conversation_id must match the stored response conversation")
            if str(row["status"]).strip().lower() in _TERMINAL_RESPONSE_STATUSES:
                raise NativeStateConflictError(
                    f"Response {normalized_id!r} is already terminal"
                )
            committed_revision: int | None = None
            if normalized_conversation is not None:
                assert expected_revision is not None
                committed_revision = self._reserve_conversation_revision_locked(
                    connection,
                    normalized_scope,
                    normalized_conversation,
                    expected_revision,
                    updated_at=now,
                )
            self._append_response_items_locked(
                connection,
                normalized_scope,
                normalized_id,
                output_records,
                phase="output",
            )
            if normalized_conversation is not None:
                self._append_conversation_items_locked(
                    connection,
                    normalized_scope,
                    normalized_conversation,
                    conversation_records,
                )
            payload = _decode(row["payload_json"])
            payload.update(patch)
            payload.update(
                {
                    "id": normalized_id,
                    "status": normalized_status,
                    "output": [deepcopy(item) for item in output_records],
                }
            )
            connection.execute(
                "UPDATE responses SET status=?,updated_at=?,payload_json=? "
                "WHERE scope=? AND id=?",
                (
                    normalized_status,
                    now,
                    _encode(payload, field="response"),
                    normalized_scope,
                    normalized_id,
                ),
            )
            return payload, committed_revision

    def response_lineage(
        self,
        scope: str,
        response_id: str,
    ) -> tuple[dict[str, Any], ...]:
        """Return an immutable oldest-to-newest previous-response chain."""

        normalized_scope = _scope(scope)
        current = _identifier(response_id, field="response_id")
        reversed_chain: list[dict[str, Any]] = []
        seen: set[str] = set()
        with self._lock:
            self._ensure_open()
            while current:
                if current in seen:
                    raise NativeStateError("Stored previous_response_id chain contains a cycle")
                seen.add(current)
                row = self._connection.execute(
                    "SELECT previous_response_id,payload_json FROM responses "
                    "WHERE scope=? AND id=?",
                    (normalized_scope, current),
                ).fetchone()
                if row is None:
                    raise KeyError(current)
                reversed_chain.append(_decode(row["payload_json"]))
                current = str(row["previous_response_id"] or "")
        return tuple(reversed(reversed_chain))

    def put_conversation(
        self,
        scope: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        normalized_scope = _scope(scope)
        record = _json_object(payload, field="conversation")
        conversation_id = _identifier(record.get("id"), field="conversation.id")
        created_at = record.get("created_at", int(time.time()))
        if isinstance(created_at, bool) or not isinstance(created_at, int) or created_at < 0:
            raise ValueError("conversation.created_at must be a non-negative integer")
        record.update({"id": conversation_id, "created_at": created_at})
        now = int(time.time())
        with self._transaction(immediate=True) as connection:
            try:
                connection.execute(
                    "INSERT INTO conversations(scope,id,created_at,updated_at,payload_json) "
                    "VALUES(?,?,?,?,?)",
                    (
                        normalized_scope,
                        conversation_id,
                        created_at,
                        now,
                        _encode(record, field="conversation"),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise NativeStateConflictError(
                    f"Conversation {conversation_id!r} already exists"
                ) from exc
        return deepcopy(record)

    def get_conversation(self, scope: str, conversation_id: str) -> dict[str, Any] | None:
        with self._lock:
            self._ensure_open()
            row = self._connection.execute(
                "SELECT payload_json FROM conversations WHERE scope=? AND id=?",
                (_scope(scope), _identifier(conversation_id, field="conversation_id")),
            ).fetchone()
            return _decode(row["payload_json"]) if row is not None else None

    def conversation_items_snapshot(
        self,
        scope: str,
        conversation_id: str,
    ) -> tuple[tuple[dict[str, Any], ...], int]:
        """Return one transactionally consistent item history and revision."""

        normalized_scope = _scope(scope)
        normalized_conversation = _identifier(
            conversation_id,
            field="conversation_id",
        )
        with self._transaction() as connection:
            parent = connection.execute(
                "SELECT items_revision FROM conversations WHERE scope=? AND id=?",
                (normalized_scope, normalized_conversation),
            ).fetchone()
            if parent is None:
                raise KeyError(normalized_conversation)
            rows = connection.execute(
                "SELECT payload_json FROM conversation_items "
                "WHERE scope=? AND conversation_id=? ORDER BY position",
                (normalized_scope, normalized_conversation),
            ).fetchall()
            return (
                tuple(_decode(row["payload_json"]) for row in rows),
                int(parent["items_revision"]),
            )

    def update_conversation(
        self,
        scope: str,
        conversation_id: str,
        patch: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        normalized_scope = _scope(scope)
        normalized_id = _identifier(conversation_id, field="conversation_id")
        changes = _json_object(patch, field="conversation patch")
        if "id" in changes and changes["id"] != normalized_id:
            raise ValueError("conversation id is immutable")
        now = int(time.time())
        with self._transaction(immediate=True) as connection:
            row = connection.execute(
                "SELECT payload_json FROM conversations WHERE scope=? AND id=?",
                (normalized_scope, normalized_id),
            ).fetchone()
            if row is None:
                return None
            payload = _decode(row["payload_json"])
            payload.update(changes)
            payload["id"] = normalized_id
            connection.execute(
                "UPDATE conversations SET updated_at=?,payload_json=? "
                "WHERE scope=? AND id=?",
                (
                    now,
                    _encode(payload, field="conversation"),
                    normalized_scope,
                    normalized_id,
                ),
            )
            return payload

    def delete_conversation(self, scope: str, conversation_id: str) -> bool:
        with self._transaction(immediate=True) as connection:
            cursor = connection.execute(
                "DELETE FROM conversations WHERE scope=? AND id=?",
                (_scope(scope), _identifier(conversation_id, field="conversation_id")),
            )
            return cursor.rowcount == 1

    def append_conversation_items(
        self,
        scope: str,
        conversation_id: str,
        items: Sequence[Mapping[str, Any]],
    ) -> tuple[dict[str, Any], ...]:
        records, _revision = self.append_conversation_items_with_revision(
            scope,
            conversation_id,
            items,
        )
        return records

    def append_conversation_items_with_revision(
        self,
        scope: str,
        conversation_id: str,
        items: Sequence[Mapping[str, Any]],
    ) -> tuple[tuple[dict[str, Any], ...], int]:
        """Blind-append items once and return the new monotonic revision."""

        normalized_scope = _scope(scope)
        normalized_conversation = _identifier(conversation_id, field="conversation_id")
        records = tuple(_json_object(item, field="conversation item") for item in items)
        if not records:
            raise ValueError("conversation items must not be empty")
        now = int(time.time())
        with self._transaction(immediate=True) as connection:
            parent = connection.execute(
                "SELECT items_revision FROM conversations WHERE scope=? AND id=?",
                (normalized_scope, normalized_conversation),
            ).fetchone()
            if parent is None:
                raise KeyError(normalized_conversation)
            revision = int(parent["items_revision"])
            if revision < 0 or revision > 9_223_372_036_854_775_806:
                raise NativeStateError("Conversation item revision space is exhausted")
            committed_revision = self._reserve_conversation_revision_locked(
                connection,
                normalized_scope,
                normalized_conversation,
                revision,
                updated_at=now,
            )
            self._append_conversation_items_locked(
                connection,
                normalized_scope,
                normalized_conversation,
                records,
            )
        return tuple(deepcopy(record) for record in records), committed_revision

    def get_conversation_item(
        self,
        scope: str,
        conversation_id: str,
        item_id: str,
    ) -> dict[str, Any] | None:
        with self._lock:
            self._ensure_open()
            row = self._connection.execute(
                "SELECT payload_json FROM conversation_items "
                "WHERE scope=? AND conversation_id=? AND item_id=?",
                (
                    _scope(scope),
                    _identifier(conversation_id, field="conversation_id"),
                    _identifier(item_id, field="item_id"),
                ),
            ).fetchone()
            return _decode(row["payload_json"]) if row is not None else None

    def list_conversation_items(
        self,
        scope: str,
        conversation_id: str,
    ) -> tuple[dict[str, Any], ...]:
        with self._lock:
            self._ensure_open()
            rows = self._connection.execute(
                "SELECT payload_json FROM conversation_items "
                "WHERE scope=? AND conversation_id=? ORDER BY position",
                (_scope(scope), _identifier(conversation_id, field="conversation_id")),
            ).fetchall()
            return tuple(_decode(row["payload_json"]) for row in rows)

    def delete_conversation_item(
        self,
        scope: str,
        conversation_id: str,
        item_id: str,
    ) -> bool:
        deleted, _revision = self.delete_conversation_item_with_revision(
            scope,
            conversation_id,
            item_id,
        )
        return deleted

    def delete_conversation_item_with_revision(
        self,
        scope: str,
        conversation_id: str,
        item_id: str,
    ) -> tuple[bool, int | None]:
        """Delete one item and return its conversation's resulting revision."""

        normalized_scope = _scope(scope)
        normalized_conversation = _identifier(conversation_id, field="conversation_id")
        normalized_item = _identifier(item_id, field="item_id")
        now = int(time.time())
        with self._transaction(immediate=True) as connection:
            parent = connection.execute(
                "SELECT items_revision FROM conversations WHERE scope=? AND id=?",
                (normalized_scope, normalized_conversation),
            ).fetchone()
            if parent is None:
                return False, None
            revision = int(parent["items_revision"])
            if revision < 0 or revision > 9_223_372_036_854_775_806:
                raise NativeStateError("Conversation item revision space is exhausted")
            cursor = connection.execute(
                "DELETE FROM conversation_items "
                "WHERE scope=? AND conversation_id=? AND item_id=?",
                (normalized_scope, normalized_conversation, normalized_item),
            )
            if cursor.rowcount == 1:
                committed_revision = self._reserve_conversation_revision_locked(
                    connection,
                    normalized_scope,
                    normalized_conversation,
                    revision,
                    updated_at=now,
                )
                return True, committed_revision
            return False, revision

    def put_response_compaction(
        self,
        scope: str,
        payload: Mapping[str, Any],
        *,
        encrypted_content: str,
    ) -> dict[str, Any]:
        """Persist one scope-bound compacted-context reference.

        The compaction registry indexes only the token's SHA-256 digest.  If a
        caller later submits that token as a Response input item, the ordinary
        response-item history retains the submitted JSON, including the token.
        """

        normalized_scope = _scope(scope)
        record = _json_object(payload, field="response compaction")
        compaction_id = _identifier(record.get("id"), field="response compaction.id")
        token = _identifier(encrypted_content, field="encrypted_content")
        created_at = record.get("created_at", int(time.time()))
        if isinstance(created_at, bool) or not isinstance(created_at, int) or created_at < 0:
            raise ValueError("response compaction.created_at must be a non-negative integer")
        record.update({"id": compaction_id, "created_at": created_at})
        token_sha256 = hashlib.sha256(token.encode("utf-8")).hexdigest()
        with self._transaction(immediate=True) as connection:
            try:
                connection.execute(
                    "INSERT INTO response_compactions(scope,id,token_sha256,created_at,payload_json) "
                    "VALUES(?,?,?,?,?)",
                    (
                        normalized_scope,
                        compaction_id,
                        token_sha256,
                        created_at,
                        _encode(record, field="response compaction"),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise NativeStateConflictError(
                    f"Response compaction {compaction_id!r} already exists"
                ) from exc
        return deepcopy(record)

    def get_response_compaction(
        self,
        scope: str,
        encrypted_content: str,
    ) -> dict[str, Any] | None:
        normalized_scope = _scope(scope)
        token = _identifier(encrypted_content, field="encrypted_content")
        token_sha256 = hashlib.sha256(token.encode("utf-8")).hexdigest()
        with self._lock:
            self._ensure_open()
            row = self._connection.execute(
                "SELECT payload_json FROM response_compactions "
                "WHERE scope=? AND token_sha256=?",
                (normalized_scope, token_sha256),
            ).fetchone()
            return _decode(row["payload_json"]) if row is not None else None

    def claim_next_background_job(self) -> tuple[str, dict[str, Any]] | None:
        now = int(time.time())
        with self._transaction(immediate=True) as connection:
            row = connection.execute(
                "SELECT scope,response_id FROM background_jobs WHERE status='queued' "
                "ORDER BY enqueued_at,response_id LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                "UPDATE background_jobs SET status='in_progress',started_at=?,updated_at=? "
                "WHERE scope=? AND response_id=? AND status='queued'",
                (now, now, row["scope"], row["response_id"]),
            )
            response = connection.execute(
                "SELECT payload_json FROM responses WHERE scope=? AND id=?",
                (row["scope"], row["response_id"]),
            ).fetchone()
            if response is None:
                raise NativeStateError("Background job references a missing response")
            payload = _decode(response["payload_json"])
            payload["status"] = "in_progress"
            connection.execute(
                "UPDATE responses SET status='in_progress',updated_at=?,payload_json=? "
                "WHERE scope=? AND id=?",
                (now, _encode(payload, field="response"), row["scope"], row["response_id"]),
            )
            return str(row["scope"]), payload

    def queued_background_jobs(self) -> tuple[tuple[str, dict[str, Any]], ...]:
        with self._lock:
            self._ensure_open()
            rows = self._connection.execute(
                "SELECT j.scope,r.payload_json FROM background_jobs j "
                "JOIN responses r ON r.scope=j.scope AND r.id=j.response_id "
                "WHERE j.status='queued' ORDER BY j.enqueued_at,j.response_id"
            ).fetchall()
            return tuple((str(row["scope"]), _decode(row["payload_json"])) for row in rows)

    def request_cancel(self, scope: str, response_id: str) -> bool:
        normalized_scope = _scope(scope)
        normalized_id = _identifier(response_id, field="response_id")
        now = int(time.time())
        with self._transaction(immediate=True) as connection:
            row = connection.execute(
                "SELECT r.payload_json,j.status AS job_status FROM responses r "
                "LEFT JOIN background_jobs j "
                "ON j.scope=r.scope AND j.response_id=r.id "
                "WHERE r.scope=? AND r.id=?",
                (normalized_scope, normalized_id),
            ).fetchone()
            if row is None:
                return False
            payload = _decode(row["payload_json"])
            payload["cancel_requested"] = True
            response_status = str(payload.get("status") or "in_progress")
            if row["job_status"] == "queued":
                response_status = "cancelled"
                payload["status"] = response_status
                payload["completed_at"] = now
            connection.execute(
                "UPDATE responses SET status=?,cancel_requested=1,updated_at=?,payload_json=? "
                "WHERE scope=? AND id=?",
                (
                    response_status,
                    now,
                    _encode(payload, field="response"),
                    normalized_scope,
                    normalized_id,
                ),
            )
            if row["job_status"] == "queued":
                if _stream_was_requested(payload):
                    self._append_response_events_locked(
                        connection,
                        normalized_scope,
                        normalized_id,
                        (
                            {
                                "type": "response.incomplete",
                                "response": _public_response(payload),
                            },
                        ),
                    )
                connection.execute(
                    "UPDATE background_jobs SET status='cancelled',updated_at=? "
                    "WHERE scope=? AND response_id=? AND status='queued'",
                    (now, normalized_scope, normalized_id),
                )
            return True

    def is_cancel_requested(self, scope: str, response_id: str) -> bool:
        with self._lock:
            self._ensure_open()
            row = self._connection.execute(
                "SELECT cancel_requested FROM responses WHERE scope=? AND id=?",
                (_scope(scope), _identifier(response_id, field="response_id")),
            ).fetchone()
            return bool(row["cancel_requested"]) if row is not None else False

    def finish_background_job(
        self,
        scope: str,
        response_id: str,
        *,
        status: str,
        response_patch: Mapping[str, Any],
        error: Mapping[str, Any] | None = None,
        response_item: Mapping[str, Any] | None = None,
        response_events: Sequence[Mapping[str, Any]] = (),
        conversation_id: str | None = None,
        conversation_items: Sequence[Mapping[str, Any]] = (),
        expected_conversation_revision: int | None = None,
    ) -> dict[str, Any] | None:
        normalized_status = str(status).strip().lower()
        if normalized_status not in {"completed", "incomplete", "failed", "cancelled"}:
            raise ValueError("background terminal status is invalid")
        normalized_scope = _scope(scope)
        normalized_id = _identifier(response_id, field="response_id")
        patch = _json_object(response_patch, field="response patch")
        item = (
            _json_object(response_item, field="response item")
            if response_item is not None
            else None
        )
        event_templates = tuple(
            _json_object(event, field="response event") for event in response_events
        )
        normalized_conversation: str | None = None
        conversation_records = tuple(
            _json_object(conversation_item, field="conversation item")
            for conversation_item in conversation_items
        )
        expected_revision: int | None = None
        if conversation_id is None:
            if conversation_records or expected_conversation_revision is not None:
                raise ValueError(
                    "conversation_id is required for conversation items or revision CAS"
                )
        else:
            normalized_conversation = _identifier(
                conversation_id,
                field="conversation_id",
            )
            if normalized_status not in {"completed", "incomplete"}:
                raise ValueError(
                    "only completed or incomplete background responses may append a conversation"
                )
            if item is None:
                raise ValueError("response_item is required for a conversation commit")
            if not conversation_records:
                raise ValueError("conversation_items must not be empty for a conversation commit")
            if expected_conversation_revision is None:
                raise ValueError("expected_conversation_revision is required")
            expected_revision = _conversation_revision(
                expected_conversation_revision,
                field="expected_conversation_revision",
            )
            conversation_by_id = {
                _identifier(record.get("id"), field="conversation item.id"): record
                for record in conversation_records
            }
            item_id = _identifier(item.get("id"), field="response item.id")
            if conversation_by_id.get(item_id) != item:
                raise ValueError("conversation_items must include the response_item")
            if "output" in patch and patch["output"] != [item]:
                raise ValueError("response patch output must match response_item")
        now = int(time.time())
        with self._transaction(immediate=True) as connection:
            row = connection.execute(
                "SELECT status,background,conversation_id,payload_json FROM responses "
                "WHERE scope=? AND id=?",
                (normalized_scope, normalized_id),
            ).fetchone()
            if row is None:
                return None
            if not bool(row["background"]):
                raise ValueError("finish_background_job requires a background response")
            stored_conversation = (
                str(row["conversation_id"]) if row["conversation_id"] is not None else None
            )
            if (
                stored_conversation is not None
                and normalized_status in {"completed", "incomplete"}
                and normalized_conversation is None
            ):
                raise ValueError(
                    "a conversation-linked background completion requires revision CAS"
                )
            if (
                stored_conversation != normalized_conversation
                and not (
                    stored_conversation is not None
                    and normalized_conversation is None
                    and normalized_status in {"failed", "cancelled"}
                )
            ):
                raise ValueError("conversation_id must match the stored response conversation")
            if str(row["status"]).strip().lower() in _TERMINAL_RESPONSE_STATUSES:
                raise NativeStateConflictError(
                    f"Response {normalized_id!r} is already terminal"
                )
            if normalized_conversation is not None:
                assert expected_revision is not None
                self._reserve_conversation_revision_locked(
                    connection,
                    normalized_scope,
                    normalized_conversation,
                    expected_revision,
                    updated_at=now,
                )
            payload = _decode(row["payload_json"])
            payload.update(patch)
            payload["status"] = normalized_status
            if error is not None:
                payload["error"] = _json_object(error, field="background error")
            if item is not None:
                self._append_response_items_locked(
                    connection,
                    normalized_scope,
                    normalized_id,
                    (item,),
                    phase="output",
                )
            if normalized_conversation is not None:
                self._append_conversation_items_locked(
                    connection,
                    normalized_scope,
                    normalized_conversation,
                    conversation_records,
                )
            connection.execute(
                "UPDATE responses SET status=?,updated_at=?,payload_json=? "
                "WHERE scope=? AND id=?",
                (
                    normalized_status,
                    now,
                    _encode(payload, field="response"),
                    normalized_scope,
                    normalized_id,
                ),
            )
            connection.execute(
                "UPDATE background_jobs SET status=?,updated_at=?,error_json=? "
                "WHERE scope=? AND response_id=?",
                (
                    normalized_status,
                    now,
                    _encode(error, field="background error") if error is not None else None,
                    normalized_scope,
                    normalized_id,
                ),
            )
            if _stream_was_requested(payload):
                terminal_type = {
                    "completed": "response.completed",
                    "failed": "response.failed",
                    "incomplete": "response.incomplete",
                    "cancelled": "response.incomplete",
                }[normalized_status]
                event_templates = (
                    *event_templates,
                    {
                        "type": terminal_type,
                        "response": _public_response(payload),
                    },
                )
            if event_templates:
                self._append_response_events_locked(
                    connection,
                    normalized_scope,
                    normalized_id,
                    event_templates,
                )
            return payload

    def stats(self) -> dict[str, Any]:
        with self._lock:
            self._ensure_open()
            journal = self._connection.execute("PRAGMA journal_mode").fetchone()[0]
            counts = {
                table: int(self._connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in (
                    "responses",
                    "response_items",
                    "response_events",
                    "conversations",
                    "conversation_items",
                    "response_compactions",
                    "background_jobs",
                )
            }
            return {
                "schema_version": NATIVE_STATE_SCHEMA_VERSION,
                "journal_mode": str(journal).lower(),
                "path": str(self.path),
                "recovered_interrupted_jobs": self.recovered_interrupted_jobs,
                **counts,
            }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._connection.close()
            self._closed = True

    def __enter__(self) -> "NativeStateStore":
        with self._lock:
            self._ensure_open()
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()


__all__ = [
    "ANONYMOUS_API_KEY_FINGERPRINT",
    "NATIVE_STATE_SCHEMA_VERSION",
    "NativeStateConflictError",
    "NativeStateError",
    "NativeStateStore",
    "api_key_fingerprint",
]
