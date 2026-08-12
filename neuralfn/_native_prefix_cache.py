"""Bounded in-process ownership of sealed native inference prefixes.

This module is deliberately independent of the Responses persistence layer.
The caller must publish a session only *after* the corresponding stored
response (and, when applicable, conversation revision) has committed.  Cache
entries own their sessions exclusively and expose only forked child leases.

The cache is process-local and best-effort.  Durable token history remains the
source of truth; an alias hit is accepted only after an exact token longest-
common-prefix check against the newly prepared prompt.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import threading
from typing import Any, Mapping, Protocol, Sequence


_MAX_CONVERSATION_REVISION = 9_223_372_036_854_775_807
_CACHEABLE_STATUSES = frozenset({"completed", "incomplete"})
_TERMINAL_STATUSES = frozenset({"completed", "incomplete", "failed", "cancelled"})
_BYTE_ACCOUNTING_SCOPE = (
    "sum-of-per-session-capacity-observations; shared allocations may be "
    "represented by more than one session"
)


class NativePrefixCacheClosedError(RuntimeError):
    """Raised when new cache work is requested after shutdown."""


class NativePrefixCacheInvariantError(RuntimeError):
    """Raised when a native fork violates its exact-token contract."""


class _Session(Protocol):
    @property
    def token_ids(self) -> Sequence[int]: ...

    @property
    def cancelled(self) -> bool: ...

    @property
    def closed(self) -> bool: ...

    def stats(self) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


class _ForkableModel(Protocol):
    def fork_session(
        self,
        source: _Session,
        *,
        token_count: int | None = None,
        seed: int = 0,
    ) -> _Session: ...


@dataclass(frozen=True, slots=True)
class NativePrefixCacheUsage:
    """One request's native prefix-cache observations.

    ``cached_tokens`` is never inferred from token ancestry alone.  It is
    bounded by both the exact token LCP selected for a successful native fork
    and the child session's native ``cached_tokens`` statistic.

    Byte fields are capacity observations attributed to this session, not
    unique process-resident allocation sizes.  Native COW allocations can be
    reported by more than one sharing session, which is why the accounting
    scope is carried alongside every value.
    """

    cached_tokens: int = 0
    cache_write_tokens: int = 0
    shared_bytes: int = 0
    private_bytes: int = 0
    detach_bytes: int = 0
    byte_accounting_scope: str = _BYTE_ACCOUNTING_SCOPE

    def input_tokens_details(self) -> dict[str, int]:
        """Return the two fields used by Responses input-token accounting."""

        return {
            "cached_tokens": self.cached_tokens,
            "cache_write_tokens": self.cache_write_tokens,
        }


@dataclass(frozen=True, slots=True)
class NativePrefixCacheCommit:
    """Result of consuming a lease at the persistence/cache boundary."""

    admitted: bool
    reason: str | None
    usage: NativePrefixCacheUsage
    evicted_entries: int = 0


@dataclass(frozen=True, slots=True)
class _SessionSnapshot:
    native_cached_tokens: int | None
    shared_bytes: int
    private_bytes: int
    detach_bytes: int


@dataclass(slots=True)
class _Entry:
    entry_id: int
    scope: str
    session: _Session
    tokens: tuple[int, ...]
    snapshot: _SessionSnapshot
    response_id: str
    conversation_alias: tuple[str, int] | None
    pins: int = 0
    retired: bool = False
    close_scheduled: bool = False


def _identifier(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} must be a non-empty string")
    if len(normalized) > 255:
        raise ValueError(f"{field} must not exceed 255 characters")
    return normalized


def _scope(value: Any) -> str:
    # The durable store validates API-key fingerprints more narrowly.  Keeping
    # this manager's scope opaque makes it usable with isolated runtime/test
    # namespaces while retaining exact (and case-sensitive) separation.
    return _identifier(value, field="scope")


def _revision(value: Any) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > _MAX_CONVERSATION_REVISION
    ):
        raise ValueError("conversation_revision must be an integer from 0 through 2^63-1")
    return value


def _token_ids(value: Sequence[int], *, field: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{field} must be a sequence of non-negative integers")
    normalized: list[int] = []
    for token in value:
        if isinstance(token, bool) or not isinstance(token, int) or token < 0:
            raise ValueError(f"{field} must contain only non-negative integers")
        normalized.append(token)
    return tuple(normalized)


def _non_negative_stat(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _snapshot_session(session: _Session) -> _SessionSnapshot:
    """Read only the native counters that can support honest attribution.

    Observability must not turn a successfully forked inference session into a
    request failure.  Missing, malformed, or unavailable counters therefore
    conservatively contribute zero (or ``None`` for native cached rows).
    """

    try:
        raw = session.stats()
    except BaseException:
        raw = {}
    if not isinstance(raw, Mapping):
        raw = {}
    native_cached_tokens = _non_negative_stat(raw.get("cached_tokens"))
    capacity = _non_negative_stat(raw.get("cache_capacity_bytes"))
    shared = _non_negative_stat(raw.get("prefix_cow_shared_capacity_bytes"))
    detached = _non_negative_stat(raw.get("prefix_cow_detached_capacity_bytes"))
    shared_bytes = 0 if shared is None else shared
    capacity_bytes = 0 if capacity is None else capacity
    # A malformed shared value must never create a negative private count.
    shared_bytes = min(shared_bytes, capacity_bytes) if capacity is not None else 0
    private_bytes = max(0, capacity_bytes - shared_bytes)
    return _SessionSnapshot(
        native_cached_tokens=native_cached_tokens,
        shared_bytes=shared_bytes,
        private_bytes=private_bytes,
        detach_bytes=0 if detached is None else detached,
    )


def _longest_common_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    count = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        count += 1
    return count


class NativePrefixCacheLease:
    """Exclusive ownership of one forked or newly-created request session.

    A lease is single-use.  ``commit`` offers its session for admission only
    when ``stored=True`` and the terminal status is ``completed`` or
    ``incomplete``; capacity or shutdown can still reject that offer.  Every
    other terminal outcome consumes and closes the session.  ``close`` is the
    failure/cancellation path.
    """

    def __init__(
        self,
        *,
        manager: "NativePrefixCache",
        session: _Session,
        scope: str,
        scope_epoch: int,
        prompt_token_ids: tuple[int, ...],
        cached_tokens: int,
        initial_snapshot: _SessionSnapshot,
    ) -> None:
        self._manager = manager
        self._session: _Session | None = session
        self._scope = scope
        self._scope_epoch = scope_epoch
        self._prompt_token_ids = prompt_token_ids
        self._cached_tokens = cached_tokens
        self._initial_snapshot = initial_snapshot
        self._lock = threading.Lock()

    @property
    def session(self) -> _Session:
        with self._lock:
            if self._session is None:
                raise NativePrefixCacheClosedError("Native prefix-cache lease is closed")
            return self._session

    @property
    def cached_tokens(self) -> int:
        return self._cached_tokens

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._session is None

    def usage(self) -> NativePrefixCacheUsage:
        with self._lock:
            session = self._session
        if session is None:
            raise NativePrefixCacheClosedError("Native prefix-cache lease is closed")
        return self._usage_from_snapshot(_snapshot_session(session))

    def _usage_from_snapshot(self, final: _SessionSnapshot) -> NativePrefixCacheUsage:
        # Responses input accounting stops at the prepared-prompt boundary.
        # Native ``cached_tokens`` grows during decode too, so the raw final
        # counter must never be exposed as input cache writes.  Bound observed
        # new cache rows by the exact prompt suffix that was not reused.
        remaining_prompt_tokens = max(
            0, len(self._prompt_token_ids) - self._cached_tokens
        )
        initial_cached = self._initial_snapshot.native_cached_tokens
        final_cached = final.native_cached_tokens
        observed_new_rows = (
            max(0, final_cached - initial_cached)
            if initial_cached is not None and final_cached is not None
            else 0
        )
        cache_write_tokens = min(remaining_prompt_tokens, observed_new_rows)
        return NativePrefixCacheUsage(
            cached_tokens=self._cached_tokens,
            cache_write_tokens=cache_write_tokens,
            shared_bytes=final.shared_bytes,
            private_bytes=final.private_bytes,
            detach_bytes=max(0, final.detach_bytes - self._initial_snapshot.detach_bytes),
        )

    def _take(self) -> _Session:
        with self._lock:
            session = self._session
            if session is None:
                raise NativePrefixCacheClosedError("Native prefix-cache lease is closed")
            self._session = None
            return session

    def commit(
        self,
        *,
        scope: str,
        response_id: str,
        status: str,
        stored: bool,
        conversation_id: str | None = None,
        conversation_revision: int | None = None,
    ) -> NativePrefixCacheCommit:
        """Consume the lease after the caller's durable terminal commit."""

        return self._manager.commit(
            self,
            scope=scope,
            response_id=response_id,
            status=status,
            stored=stored,
            conversation_id=conversation_id,
            conversation_revision=conversation_revision,
        )

    def close(self) -> None:
        """Consume and close an uncommitted request session exactly once."""

        with self._lock:
            session = self._session
            if session is None:
                return
            self._session = None
        final = _snapshot_session(session)
        usage = self._usage_from_snapshot(final)
        self._manager._finish_lease(usage)
        self._manager._close_sessions((session,))

    def __enter__(self) -> "NativePrefixCacheLease":
        # Accessing ``session`` supplies the closed check without exposing the
        # retained cache entry.
        self.session
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()


class NativePrefixCache:
    """Capacity-bounded deterministic LRU of sealed native session prefixes.

    The manager never calls ``fork_session``, ``stats``, or ``close`` while its
    own lock is held.  Cache entries are briefly pinned while a fork is in
    flight; eviction and purge retire such entries immediately but defer their
    close until the pin is released.  The returned child lease is independent
    of that pin, so a capacity-one cache may evict the parent immediately after
    the child fork completes.
    """

    def __init__(self, model: _ForkableModel, *, capacity: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 0:
            raise ValueError("capacity must be a non-negative integer")
        if not callable(getattr(model, "fork_session", None)):
            raise TypeError("model must expose fork_session")
        self._model = model
        self._capacity = capacity
        self._lock = threading.RLock()
        self._entries: OrderedDict[int, _Entry] = OrderedDict()
        self._response_aliases: dict[tuple[str, str], _Entry] = {}
        self._conversation_aliases: dict[tuple[str, str, int], _Entry] = {}
        self._scope_epochs: dict[str, int] = {}
        self._next_entry_id = 1
        self._shutdown = False

        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._scope_purges = 0
        self._commits = 0
        self._rejections = 0
        self._close_errors = 0
        self._active_leases = 0
        self._in_flight_forks = 0
        self._cached_tokens = 0
        self._cache_write_tokens = 0
        self._shared_bytes = 0
        self._private_bytes = 0
        self._detach_bytes = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._shutdown

    def _ensure_open_locked(self) -> None:
        if self._shutdown:
            raise NativePrefixCacheClosedError("Native prefix cache is shut down")

    @staticmethod
    def _lookup_key(
        *,
        scope: str,
        response_id: str | None,
        conversation_id: str | None,
        conversation_revision: int | None,
    ) -> tuple[str, tuple[Any, ...]]:
        has_response = response_id is not None
        has_conversation = conversation_id is not None or conversation_revision is not None
        if has_response == has_conversation:
            raise ValueError(
                "Specify exactly one of response_id or conversation_id with conversation_revision"
            )
        if has_response:
            return "response", (scope, _identifier(response_id, field="response_id"))
        if conversation_id is None or conversation_revision is None:
            raise ValueError(
                "conversation_id and conversation_revision must be specified together"
            )
        return (
            "conversation",
            (
                scope,
                _identifier(conversation_id, field="conversation_id"),
                _revision(conversation_revision),
            ),
        )

    @staticmethod
    def _commit_aliases(
        *,
        scope: str,
        response_id: str,
        conversation_id: str | None,
        conversation_revision: int | None,
    ) -> tuple[str, tuple[str, int] | None]:
        normalized_response = _identifier(response_id, field="response_id")
        if (conversation_id is None) != (conversation_revision is None):
            raise ValueError(
                "conversation_id and conversation_revision must be specified together"
            )
        conversation_alias = None
        if conversation_id is not None:
            conversation_alias = (
                _identifier(conversation_id, field="conversation_id"),
                _revision(conversation_revision),
            )
        return normalized_response, conversation_alias

    def acquire(
        self,
        *,
        scope: str,
        prompt_token_ids: Sequence[int],
        response_id: str | None = None,
        conversation_id: str | None = None,
        conversation_revision: int | None = None,
        seed: int = 0,
    ) -> NativePrefixCacheLease | None:
        """Fork the exact reusable LCP for one scope-local alias.

        Alias ancestry only identifies a candidate.  A zero token LCP is a
        miss, and a partial LCP forks only the exact matching token prefix.
        """

        normalized_scope = _scope(scope)
        prompt = _token_ids(prompt_token_ids, field="prompt_token_ids")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        kind, key = self._lookup_key(
            scope=normalized_scope,
            response_id=response_id,
            conversation_id=conversation_id,
            conversation_revision=conversation_revision,
        )

        with self._lock:
            self._ensure_open_locked()
            scope_epoch = self._scope_epochs.get(normalized_scope, 0)
            entry = (
                self._response_aliases.get(key)
                if kind == "response"
                else self._conversation_aliases.get(key)
            )
            if entry is None or self._capacity == 0:
                self._misses += 1
                return None
            common = _longest_common_prefix(entry.tokens, prompt)
            if common == 0:
                self._misses += 1
                return None
            entry.pins += 1
            self._in_flight_forks += 1

        child: _Session | None = None
        source_corrupt = False
        try:
            try:
                live_tokens = _token_ids(entry.session.token_ids, field="cached session token_ids")
                source_corrupt = (
                    live_tokens != entry.tokens
                    or bool(getattr(entry.session, "closed", False))
                    or bool(getattr(entry.session, "cancelled", False))
                )
            except BaseException:
                source_corrupt = True
            if not source_corrupt:
                child = self._model.fork_session(
                    entry.session,
                    token_count=common,
                    seed=seed,
                )
                child_tokens = _token_ids(child.token_ids, field="forked session token_ids")
                if child_tokens != entry.tokens[:common]:
                    raise NativePrefixCacheInvariantError(
                        "Native session fork did not preserve the selected exact token prefix"
                    )
        except BaseException:
            sessions_to_close: list[_Session] = []
            with self._lock:
                sessions_to_close.extend(self._release_entry_pin_locked(entry))
            if child is not None:
                sessions_to_close.append(child)
            self._close_sessions(sessions_to_close)
            raise

        sessions_to_close = []
        if source_corrupt:
            with self._lock:
                self._misses += 1
                sessions_to_close.extend(self._retire_entry_locked(entry))
                sessions_to_close.extend(self._release_entry_pin_locked(entry))
            self._close_sessions(sessions_to_close)
            return None

        assert child is not None
        initial = _snapshot_session(child)
        native_cached = initial.native_cached_tokens
        reused = min(common, native_cached) if native_cached is not None else 0
        with self._lock:
            if not entry.retired and entry.entry_id in self._entries:
                self._entries.move_to_end(entry.entry_id)
            self._hits += 1
            self._cached_tokens += reused
            self._active_leases += 1
            sessions_to_close.extend(self._release_entry_pin_locked(entry))
        self._close_sessions(sessions_to_close)
        return NativePrefixCacheLease(
            manager=self,
            session=child,
            scope=normalized_scope,
            scope_epoch=scope_epoch,
            prompt_token_ids=prompt,
            cached_tokens=reused,
            initial_snapshot=initial,
        )

    def lease_session(
        self,
        session: _Session,
        *,
        scope: str,
        prompt_token_ids: Sequence[int],
    ) -> NativePrefixCacheLease:
        """Adopt a fresh request session before its exact prepared-prompt prefill."""

        if not callable(getattr(session, "close", None)) or not callable(
            getattr(session, "stats", None)
        ):
            raise TypeError("session must expose stats and close")
        normalized_scope = _scope(scope)
        prompt = _token_ids(prompt_token_ids, field="prompt_token_ids")
        with self._lock:
            self._ensure_open_locked()
            scope_epoch = self._scope_epochs.get(normalized_scope, 0)
            self._active_leases += 1
        return NativePrefixCacheLease(
            manager=self,
            session=session,
            scope=normalized_scope,
            scope_epoch=scope_epoch,
            prompt_token_ids=prompt,
            cached_tokens=0,
            initial_snapshot=_snapshot_session(session),
        )

    def commit(
        self,
        lease: NativePrefixCacheLease,
        *,
        scope: str,
        response_id: str,
        status: str,
        stored: bool,
        conversation_id: str | None = None,
        conversation_revision: int | None = None,
    ) -> NativePrefixCacheCommit:
        """Consume a request lease after durable terminal persistence.

        ``stored`` is an assertion about a completed persistence/CAS operation,
        not merely the request's ``store`` preference.  Only stored completed
        or incomplete responses are eligible for admission.
        """

        if not isinstance(lease, NativePrefixCacheLease) or lease._manager is not self:
            raise ValueError("lease does not belong to this native prefix cache")
        normalized_scope = _scope(scope)
        if lease._scope != normalized_scope:
            raise ValueError("commit scope does not match the lease scope")
        normalized_response, conversation_alias = self._commit_aliases(
            scope=normalized_scope,
            response_id=response_id,
            conversation_id=conversation_id,
            conversation_revision=conversation_revision,
        )
        if not isinstance(stored, bool):
            raise TypeError("stored must be a boolean")
        if not isinstance(status, str) or status not in _TERMINAL_STATUSES:
            raise ValueError(
                "status must be one of completed, incomplete, failed, or cancelled"
            )

        # This early observation avoids needless session inspection for the
        # common stale-lease case.  Admission repeats the comparison under the
        # same lock as purge, closing the race between this check and publish.
        with self._lock:
            scope_was_purged = (
                self._scope_epochs.get(normalized_scope, 0) != lease._scope_epoch
            )

        session = lease._take()
        final = _snapshot_session(session)
        usage = lease._usage_from_snapshot(final)
        try:
            tokens = _token_ids(session.token_ids, field="session token_ids")
        except BaseException:
            tokens = ()
        prompt_matches = (
            len(tokens) >= len(lease._prompt_token_ids)
            and tokens[: len(lease._prompt_token_ids)] == lease._prompt_token_ids
        )
        session_cancelled = bool(getattr(session, "cancelled", False))
        session_closed = bool(getattr(session, "closed", False))

        self._finish_lease(usage)
        if scope_was_purged:
            self._reject_and_close(session)
            return NativePrefixCacheCommit(False, "scope_purged", usage)
        if not stored:
            self._reject_and_close(session)
            return NativePrefixCacheCommit(False, "not_stored", usage)
        if status not in _CACHEABLE_STATUSES:
            self._reject_and_close(session)
            return NativePrefixCacheCommit(False, f"status_{status}", usage)
        if session_cancelled:
            self._reject_and_close(session)
            return NativePrefixCacheCommit(False, "session_cancelled", usage)
        if session_closed:
            self._reject_and_close(session)
            return NativePrefixCacheCommit(False, "session_closed", usage)
        if not tokens:
            self._reject_and_close(session)
            return NativePrefixCacheCommit(False, "empty_session", usage)
        if not prompt_matches:
            self._reject_and_close(session)
            return NativePrefixCacheCommit(False, "prompt_mismatch", usage)

        evicted: list[_Session] = []
        evicted_count = 0
        admitted = False
        rejection_reason: str | None = None
        with self._lock:
            if self._scope_epochs.get(normalized_scope, 0) != lease._scope_epoch:
                rejection_reason = "scope_purged"
            elif self._shutdown:
                rejection_reason = "shutdown"
            elif self._capacity == 0:
                rejection_reason = "capacity_zero"
            else:
                response_key = (normalized_scope, normalized_response)
                conversation_key = (
                    (normalized_scope, conversation_alias[0], conversation_alias[1])
                    if conversation_alias is not None
                    else None
                )
                forced: list[_Entry] = []
                for existing in (
                    self._response_aliases.get(response_key),
                    self._conversation_aliases.get(conversation_key)
                    if conversation_key is not None
                    else None,
                ):
                    if existing is not None and all(
                        existing is not candidate for candidate in forced
                    ):
                        forced.append(existing)

                required = max(0, len(self._entries) - self._capacity + 1)
                selected: list[_Entry] = []
                blocked_collision = any(entry.pins for entry in forced)
                if not blocked_collision:
                    selected.extend(forced)
                    if len(selected) < required:
                        for candidate in self._entries.values():
                            if (
                                any(candidate is chosen for chosen in selected)
                                or candidate.pins
                            ):
                                continue
                            selected.append(candidate)
                            if len(selected) >= required:
                                break
                if blocked_collision or len(selected) < required:
                    rejection_reason = "capacity_leased"
                else:
                    for candidate in selected:
                        evicted.extend(self._retire_entry_locked(candidate))
                    evicted_count = len(selected)
                    self._evictions += evicted_count
                    entry = _Entry(
                        entry_id=self._next_entry_id,
                        scope=normalized_scope,
                        session=session,
                        tokens=tokens,
                        snapshot=final,
                        response_id=normalized_response,
                        conversation_alias=conversation_alias,
                    )
                    self._next_entry_id += 1
                    self._entries[entry.entry_id] = entry
                    self._response_aliases[response_key] = entry
                    if conversation_key is not None:
                        self._conversation_aliases[conversation_key] = entry
                    self._commits += 1
                    admitted = True

        self._close_sessions(evicted)
        if admitted:
            return NativePrefixCacheCommit(True, None, usage, evicted_count)
        self._reject_and_close(session)
        return NativePrefixCacheCommit(False, rejection_reason, usage, evicted_count)

    def _finish_lease(self, usage: NativePrefixCacheUsage) -> None:
        with self._lock:
            if self._active_leases > 0:
                self._active_leases -= 1
            self._cache_write_tokens += usage.cache_write_tokens
            self._shared_bytes += usage.shared_bytes
            self._private_bytes += usage.private_bytes
            self._detach_bytes += usage.detach_bytes

    def _reject_and_close(self, session: _Session) -> None:
        with self._lock:
            self._rejections += 1
        self._close_sessions((session,))

    def _release_entry_pin_locked(self, entry: _Entry) -> list[_Session]:
        if entry.pins <= 0:
            raise NativePrefixCacheInvariantError("Cache entry pin underflow")
        entry.pins -= 1
        self._in_flight_forks -= 1
        if entry.retired and entry.pins == 0 and not entry.close_scheduled:
            entry.close_scheduled = True
            return [entry.session]
        return []

    def _retire_entry_locked(self, entry: _Entry) -> list[_Session]:
        if entry.retired:
            return []
        entry.retired = True
        self._entries.pop(entry.entry_id, None)
        response_key = (entry.scope, entry.response_id)
        if self._response_aliases.get(response_key) is entry:
            self._response_aliases.pop(response_key, None)
        if entry.conversation_alias is not None:
            conversation_key = (
                entry.scope,
                entry.conversation_alias[0],
                entry.conversation_alias[1],
            )
            if self._conversation_aliases.get(conversation_key) is entry:
                self._conversation_aliases.pop(conversation_key, None)
        if entry.pins == 0 and not entry.close_scheduled:
            entry.close_scheduled = True
            return [entry.session]
        return []

    def _close_sessions(self, sessions: Sequence[_Session]) -> None:
        for session in sessions:
            try:
                session.close()
            except BaseException:
                # Cache teardown is best effort and must not reverse a durable
                # response commit.  The exact close attempt is still counted.
                with self._lock:
                    self._close_errors += 1

    def delete_response_alias(self, *, scope: str, response_id: str) -> bool:
        """Delete one response alias, closing its entry when it is the last alias."""

        normalized_scope = _scope(scope)
        normalized_response = _identifier(response_id, field="response_id")
        sessions: list[_Session] = []
        with self._lock:
            entry = self._response_aliases.pop(
                (normalized_scope, normalized_response), None
            )
            if entry is None:
                return False
            entry.response_id = ""
            if entry.conversation_alias is None:
                sessions.extend(self._retire_entry_locked(entry))
        self._close_sessions(sessions)
        return True

    def delete_conversation_alias(
        self,
        *,
        scope: str,
        conversation_id: str,
        conversation_revision: int | None = None,
    ) -> int:
        """Delete one revision alias, or every cached revision for a conversation."""

        normalized_scope = _scope(scope)
        normalized_conversation = _identifier(conversation_id, field="conversation_id")
        normalized_revision = (
            None if conversation_revision is None else _revision(conversation_revision)
        )
        sessions: list[_Session] = []
        removed = 0
        with self._lock:
            keys = [
                key
                for key in self._conversation_aliases
                if key[0] == normalized_scope
                and key[1] == normalized_conversation
                and (normalized_revision is None or key[2] == normalized_revision)
            ]
            for key in sorted(keys, key=lambda item: item[2]):
                entry = self._conversation_aliases.pop(key)
                removed += 1
                entry.conversation_alias = None
                if not entry.response_id:
                    sessions.extend(self._retire_entry_locked(entry))
        self._close_sessions(sessions)
        return removed

    def purge_scope(self, scope: str) -> int:
        """Fence old leases and retire every entry in one API-key scope.

        The epoch advances even when the scope currently has no retained
        entries.  A child or fresh lease admitted before this linearization
        point cannot publish deleted ancestor state afterward.
        """

        normalized_scope = _scope(scope)
        sessions: list[_Session] = []
        with self._lock:
            self._scope_epochs[normalized_scope] = (
                self._scope_epochs.get(normalized_scope, 0) + 1
            )
            self._scope_purges += 1
            entries = [
                entry for entry in self._entries.values() if entry.scope == normalized_scope
            ]
            for entry in entries:
                sessions.extend(self._retire_entry_locked(entry))
        self._close_sessions(sessions)
        return len(entries)

    def shutdown(self) -> None:
        """Reject new work and retire every retained entry exactly once."""

        sessions: list[_Session] = []
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
            for entry in list(self._entries.values()):
                sessions.extend(self._retire_entry_locked(entry))
        self._close_sessions(sessions)

    def stats(self) -> dict[str, Any]:
        """Return counter totals plus sealed-entry snapshot attribution."""

        with self._lock:
            retained_shared = sum(
                entry.snapshot.shared_bytes for entry in self._entries.values()
            )
            retained_private = sum(
                entry.snapshot.private_bytes for entry in self._entries.values()
            )
            retained_detach = sum(
                entry.snapshot.detach_bytes for entry in self._entries.values()
            )
            return {
                "capacity": self._capacity,
                "entries": len(self._entries),
                "response_aliases": len(self._response_aliases),
                "conversation_aliases": len(self._conversation_aliases),
                "active_leases": self._active_leases,
                "in_flight_forks": self._in_flight_forks,
                "shutdown": self._shutdown,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "scope_purges": self._scope_purges,
                "scope_epoch_count": len(self._scope_epochs),
                "commits": self._commits,
                "rejections": self._rejections,
                "close_errors": self._close_errors,
                "cached_tokens": self._cached_tokens,
                "cache_write_tokens": self._cache_write_tokens,
                "shared_bytes": self._shared_bytes,
                "private_bytes": self._private_bytes,
                "detach_bytes": self._detach_bytes,
                "retained_shared_bytes": retained_shared,
                "retained_private_bytes": retained_private,
                "retained_detach_bytes": retained_detach,
                "byte_accounting_scope": _BYTE_ACCOUNTING_SCOPE,
            }

    def __enter__(self) -> "NativePrefixCache":
        with self._lock:
            self._ensure_open_locked()
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.shutdown()
