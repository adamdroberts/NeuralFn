from __future__ import annotations

import threading
from typing import Any, Callable, Mapping, Sequence

import pytest

from neuralfn._native_prefix_cache import (
    NativePrefixCache,
    NativePrefixCacheClosedError,
    NativePrefixCacheInvariantError,
)


class _FakeSession:
    def __init__(
        self,
        model: "_FakeModel",
        tokens: Sequence[int] = (),
        *,
        capacity_bytes: int = 64,
        shared_bytes: int = 0,
    ) -> None:
        self.model = model
        self._tokens = list(tokens)
        self.capacity_bytes = capacity_bytes
        self.shared_bytes = shared_bytes
        self.detached_bytes = 0
        self.cancelled = False
        self.closed = False
        self.close_calls = 0
        self.close_hook: Callable[[], None] | None = None
        self.stats_started: threading.Event | None = None
        self.allow_stats: threading.Event | None = None

    @property
    def token_ids(self) -> tuple[int, ...]:
        return tuple(self._tokens)

    def append(self, token_ids: Sequence[int]) -> None:
        if self.closed:
            raise RuntimeError("session closed")
        if self.shared_bytes:
            self.shared_bytes = 0
            self.detached_bytes += self.capacity_bytes
        self._tokens.extend(token_ids)

    def stats(self) -> Mapping[str, Any]:
        if self.closed:
            raise RuntimeError("session closed")
        if self.stats_started is not None:
            self.stats_started.set()
        if self.allow_stats is not None and not self.allow_stats.wait(timeout=5):
            raise RuntimeError("timed out waiting for stats release")
        return {
            "cached_tokens": len(self._tokens),
            "cache_capacity_bytes": self.capacity_bytes,
            "prefix_cow_shared_capacity_bytes": self.shared_bytes,
            "prefix_cow_detached_capacity_bytes": self.detached_bytes,
        }

    def close(self) -> None:
        self.close_calls += 1
        if self.close_hook is not None:
            self.close_hook()
        self.closed = True


class _FakeModel:
    def __init__(self) -> None:
        self.sessions: list[_FakeSession] = []
        self.forks: list[tuple[_FakeSession, int, int]] = []
        self.fork_started: threading.Event | None = None
        self.allow_fork: threading.Event | None = None
        self.fail_fork = False
        self.corrupt_fork = False

    def session(self, tokens: Sequence[int] = ()) -> _FakeSession:
        session = _FakeSession(self, tokens)
        self.sessions.append(session)
        return session

    def fork_session(
        self,
        source: _FakeSession,
        *,
        token_count: int | None = None,
        seed: int = 0,
    ) -> _FakeSession:
        assert token_count is not None
        self.forks.append((source, token_count, seed))
        if self.fork_started is not None:
            self.fork_started.set()
        if self.allow_fork is not None and not self.allow_fork.wait(timeout=5):
            raise RuntimeError("timed out waiting for fork release")
        if self.fail_fork:
            raise RuntimeError("fork failed")
        source.shared_bytes = source.capacity_bytes
        tokens = source.token_ids[:token_count]
        if self.corrupt_fork:
            tokens = (*tokens, 999)
        child = _FakeSession(
            self,
            tokens,
            capacity_bytes=source.capacity_bytes,
            shared_bytes=source.capacity_bytes,
        )
        self.sessions.append(child)
        return child


def _commit_fresh(
    cache: NativePrefixCache,
    model: _FakeModel,
    *,
    scope: str,
    response_id: str,
    tokens: Sequence[int],
    status: str = "completed",
    stored: bool = True,
    conversation_id: str | None = None,
    conversation_revision: int | None = None,
) -> tuple[_FakeSession, Any]:
    session = model.session()
    lease = cache.lease_session(
        session, scope=scope, prompt_token_ids=tokens
    )
    session.append(tokens)
    result = lease.commit(
        scope=scope,
        response_id=response_id,
        status=status,
        stored=stored,
        conversation_id=conversation_id,
        conversation_revision=conversation_revision,
    )
    return session, result


def test_exact_lcp_forks_only_verified_tokens_across_bpe_boundary_mismatch() -> None:
    model = _FakeModel()
    cache = NativePrefixCache(model, capacity=2)
    parent, committed = _commit_fresh(
        cache,
        model,
        scope="scope-a",
        response_id="resp-parent",
        tokens=(10, 20, 30),
    )
    assert committed.admitted is True

    lease = cache.acquire(
        scope="scope-a",
        response_id="resp-parent",
        prompt_token_ids=(10, 21, 99),
        seed=7,
    )
    assert lease is not None
    assert lease.session.token_ids == (10,)
    assert lease.cached_tokens == 1
    assert model.forks[-1] == (parent, 1, 7)
    lease.close()

    assert cache.acquire(
        scope="scope-a",
        response_id="resp-parent",
        prompt_token_ids=(11, 20, 30),
    ) is None
    assert len(model.forks) == 1
    assert cache.stats()["hits"] == 1
    assert cache.stats()["misses"] == 1


def test_response_and_conversation_aliases_are_scope_and_revision_isolated() -> None:
    model = _FakeModel()
    cache = NativePrefixCache(model, capacity=3)
    _commit_fresh(
        cache,
        model,
        scope="scope-a",
        response_id="same-response",
        tokens=(1, 2),
        conversation_id="conv",
        conversation_revision=4,
    )

    assert cache.acquire(
        scope="scope-b",
        response_id="same-response",
        prompt_token_ids=(1, 2, 3),
    ) is None
    assert cache.acquire(
        scope="scope-a",
        conversation_id="conv",
        conversation_revision=3,
        prompt_token_ids=(1, 2, 3),
    ) is None
    conversation_lease = cache.acquire(
        scope="scope-a",
        conversation_id="conv",
        conversation_revision=4,
        prompt_token_ids=(1, 2, 3),
    )
    assert conversation_lease is not None
    assert conversation_lease.cached_tokens == 2
    conversation_lease.close()

    _commit_fresh(
        cache,
        model,
        scope="scope-b",
        response_id="same-response",
        tokens=(8, 9),
    )
    scope_a = cache.acquire(
        scope="scope-a",
        response_id="same-response",
        prompt_token_ids=(1, 2, 4),
    )
    scope_b = cache.acquire(
        scope="scope-b",
        response_id="same-response",
        prompt_token_ids=(8, 9, 4),
    )
    assert scope_a is not None and scope_a.session.token_ids == (1, 2)
    assert scope_b is not None and scope_b.session.token_ids == (8, 9)
    scope_a.close()
    scope_b.close()


def test_conversation_alias_accepts_post_commit_signed_int_max_revision() -> None:
    model = _FakeModel()
    cache = NativePrefixCache(model, capacity=1)
    maximum_revision = 2**63 - 1
    _commit_fresh(
        cache,
        model,
        scope="scope",
        response_id="response",
        tokens=(1, 2),
        conversation_id="conversation",
        conversation_revision=maximum_revision,
    )

    lease = cache.acquire(
        scope="scope",
        conversation_id="conversation",
        conversation_revision=maximum_revision,
        prompt_token_ids=(1, 2, 3),
    )
    assert lease is not None
    lease.close()
    with pytest.raises(ValueError, match=r"2\^63-1"):
        cache.acquire(
            scope="scope",
            conversation_id="conversation",
            conversation_revision=2**63,
            prompt_token_ids=(1, 2, 3),
        )


def test_divergent_branches_never_mutate_the_sealed_parent() -> None:
    model = _FakeModel()
    cache = NativePrefixCache(model, capacity=4)
    parent, _ = _commit_fresh(
        cache,
        model,
        scope="scope",
        response_id="root",
        tokens=(1, 2, 3),
    )

    left = cache.acquire(
        scope="scope", response_id="root", prompt_token_ids=(1, 2, 3, 4)
    )
    right = cache.acquire(
        scope="scope", response_id="root", prompt_token_ids=(1, 2, 3, 8)
    )
    assert left is not None and right is not None
    left.session.append((4, 5))
    right.session.append((8, 9))
    left_result = left.commit(
        scope="scope", response_id="left", status="completed", stored=True
    )
    right_result = right.commit(
        scope="scope", response_id="right", status="incomplete", stored=True
    )
    assert left_result.admitted is True
    assert right_result.admitted is True
    assert parent.token_ids == (1, 2, 3)

    left_branch = cache.acquire(
        scope="scope", response_id="left", prompt_token_ids=(1, 2, 3, 4, 5, 6)
    )
    right_branch = cache.acquire(
        scope="scope", response_id="right", prompt_token_ids=(1, 2, 3, 8, 9, 7)
    )
    assert left_branch is not None and left_branch.session.token_ids == (1, 2, 3, 4, 5)
    assert right_branch is not None and right_branch.session.token_ids == (1, 2, 3, 8, 9)
    left_branch.close()
    right_branch.close()


def test_deterministic_lru_touch_evicts_oldest_unpinned_entry() -> None:
    model = _FakeModel()
    cache = NativePrefixCache(model, capacity=2)
    first, _ = _commit_fresh(
        cache, model, scope="scope", response_id="first", tokens=(1,)
    )
    second, _ = _commit_fresh(
        cache, model, scope="scope", response_id="second", tokens=(2,)
    )

    touch = cache.acquire(
        scope="scope", response_id="first", prompt_token_ids=(1, 9)
    )
    assert touch is not None
    touch.close()
    third, result = _commit_fresh(
        cache, model, scope="scope", response_id="third", tokens=(3,)
    )

    assert result.admitted is True
    assert result.evicted_entries == 1
    assert first.closed is False
    assert second.closed is True and second.close_calls == 1
    assert third.closed is False
    assert cache.acquire(
        scope="scope", response_id="second", prompt_token_ids=(2, 4)
    ) is None
    assert cache.stats()["evictions"] == 1


def test_capacity_one_can_evict_parent_after_child_fork() -> None:
    model = _FakeModel()
    cache = NativePrefixCache(model, capacity=1)
    parent, _ = _commit_fresh(
        cache, model, scope="scope", response_id="parent", tokens=(1, 2)
    )
    child = cache.acquire(
        scope="scope", response_id="parent", prompt_token_ids=(1, 2, 3)
    )
    assert child is not None
    child.session.append((3,))
    result = child.commit(
        scope="scope", response_id="child", status="completed", stored=True
    )

    assert result.admitted is True
    assert result.evicted_entries == 1
    assert parent.closed is True and parent.close_calls == 1
    assert cache.acquire(
        scope="scope", response_id="parent", prompt_token_ids=(1, 2, 4)
    ) is None
    survivor = cache.acquire(
        scope="scope", response_id="child", prompt_token_ids=(1, 2, 3, 4)
    )
    assert survivor is not None
    survivor.close()


def test_eviction_skips_entry_pinned_by_in_flight_fork() -> None:
    model = _FakeModel()
    cache = NativePrefixCache(model, capacity=1)
    parent, _ = _commit_fresh(
        cache, model, scope="scope", response_id="parent", tokens=(1, 2)
    )
    model.fork_started = threading.Event()
    model.allow_fork = threading.Event()
    acquired: list[Any] = []
    errors: list[BaseException] = []

    def acquire() -> None:
        try:
            acquired.append(
                cache.acquire(
                    scope="scope",
                    response_id="parent",
                    prompt_token_ids=(1, 2, 3),
                )
            )
        except BaseException as exc:  # pragma: no cover - assertion aid
            errors.append(exc)

    worker = threading.Thread(target=acquire)
    worker.start()
    assert model.fork_started.wait(timeout=2)

    incoming, rejected = _commit_fresh(
        cache, model, scope="scope", response_id="incoming", tokens=(8,)
    )
    assert rejected.admitted is False
    assert rejected.reason == "capacity_leased"
    assert incoming.closed is True
    assert parent.closed is False

    model.allow_fork.set()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert errors == []
    assert len(acquired) == 1 and acquired[0] is not None
    acquired[0].close()
    assert cache.stats()["entries"] == 1


def test_purge_and_alias_deletion_close_retained_parent_not_leased_child() -> None:
    model = _FakeModel()
    cache = NativePrefixCache(model, capacity=2)
    parent, _ = _commit_fresh(
        cache,
        model,
        scope="scope",
        response_id="response",
        tokens=(1, 2),
        conversation_id="conversation",
        conversation_revision=5,
    )
    child = cache.acquire(
        scope="scope", response_id="response", prompt_token_ids=(1, 2, 3)
    )
    assert child is not None

    assert cache.delete_response_alias(scope="scope", response_id="response") is True
    assert parent.closed is False  # the exact conversation revision still aliases it
    assert cache.delete_conversation_alias(
        scope="scope", conversation_id="conversation", conversation_revision=5
    ) == 1
    assert parent.closed is True and parent.close_calls == 1
    assert child.session.closed is False
    child.session.append((3,))
    child.close()
    assert child.closed is True

    retained, _ = _commit_fresh(
        cache, model, scope="scope", response_id="other", tokens=(9,)
    )
    assert cache.purge_scope("scope") == 1
    assert retained.closed is True and retained.close_calls == 1
    assert cache.purge_scope("scope") == 0


def test_acquired_lease_cannot_republish_after_scope_purge() -> None:
    model = _FakeModel()
    cache = NativePrefixCache(model, capacity=2)
    parent, _ = _commit_fresh(
        cache, model, scope="scope", response_id="parent", tokens=(1, 2)
    )
    child = cache.acquire(
        scope="scope", response_id="parent", prompt_token_ids=(1, 2, 3)
    )
    assert child is not None

    assert cache.purge_scope("scope") == 1
    assert parent.closed is True
    child.session.append((3, 4))
    result = child.commit(
        scope="scope", response_id="descendant", status="completed", stored=True
    )

    assert result.admitted is False
    assert result.reason == "scope_purged"
    assert child.closed is True
    assert cache.stats()["entries"] == 0


def test_zero_entry_purge_fences_fresh_lease_but_not_new_generation() -> None:
    model = _FakeModel()
    cache = NativePrefixCache(model, capacity=1)
    stale_session = model.session()
    stale = cache.lease_session(
        stale_session, scope="scope", prompt_token_ids=(1, 2)
    )

    assert cache.purge_scope("scope") == 0
    stale_session.append((1, 2, 3))
    rejected = stale.commit(
        scope="scope", response_id="stale", status="completed", stored=True
    )
    assert rejected.admitted is False
    assert rejected.reason == "scope_purged"
    assert stale_session.closed is True

    current_session = model.session()
    current = cache.lease_session(
        current_session, scope="scope", prompt_token_ids=(4, 5)
    )
    current_session.append((4, 5, 6))
    admitted = current.commit(
        scope="scope", response_id="current", status="completed", stored=True
    )
    assert admitted.admitted is True
    assert cache.stats()["scope_purges"] == 1
    assert cache.stats()["scope_epoch_count"] == 1


def test_scope_purge_does_not_fence_other_scope_leases() -> None:
    model = _FakeModel()
    cache = NativePrefixCache(model, capacity=2)
    session_a = model.session()
    session_b = model.session()
    lease_a = cache.lease_session(
        session_a, scope="scope-a", prompt_token_ids=(1,)
    )
    lease_b = cache.lease_session(
        session_b, scope="scope-b", prompt_token_ids=(2,)
    )
    session_a.append((1,))
    session_b.append((2,))

    assert cache.purge_scope("scope-a") == 0
    result_a = lease_a.commit(
        scope="scope-a", response_id="a", status="completed", stored=True
    )
    result_b = lease_b.commit(
        scope="scope-b", response_id="b", status="completed", stored=True
    )
    assert result_a.admitted is False
    assert result_a.reason == "scope_purged"
    assert result_b.admitted is True
    assert session_a.closed is True
    assert session_b.closed is False


def test_concurrent_purge_linearizes_before_blocked_commit_admission() -> None:
    model = _FakeModel()
    cache = NativePrefixCache(model, capacity=1)
    session = model.session()
    lease = cache.lease_session(
        session, scope="scope", prompt_token_ids=(1, 2)
    )
    session.append((1, 2, 3))
    session.stats_started = threading.Event()
    session.allow_stats = threading.Event()
    results: list[Any] = []
    errors: list[BaseException] = []

    def commit() -> None:
        try:
            results.append(
                lease.commit(
                    scope="scope",
                    response_id="response",
                    status="completed",
                    stored=True,
                )
            )
        except BaseException as exc:  # pragma: no cover - assertion aid
            errors.append(exc)

    worker = threading.Thread(target=commit)
    worker.start()
    assert session.stats_started.wait(timeout=2)
    assert cache.purge_scope("scope") == 0
    session.allow_stats.set()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert errors == []
    assert len(results) == 1
    assert results[0].admitted is False
    assert results[0].reason == "scope_purged"
    assert session.closed is True
    assert cache.stats()["entries"] == 0


@pytest.mark.parametrize(
    ("stored", "status", "cancelled", "reason"),
    [
        (False, "completed", False, "not_stored"),
        (True, "failed", False, "status_failed"),
        (True, "cancelled", True, "status_cancelled"),
        (True, "incomplete", True, "session_cancelled"),
    ],
)
def test_unstored_failed_and_cancelled_sessions_never_enter(
    stored: bool,
    status: str,
    cancelled: bool,
    reason: str,
) -> None:
    model = _FakeModel()
    cache = NativePrefixCache(model, capacity=2)
    session = model.session()
    lease = cache.lease_session(
        session, scope="scope", prompt_token_ids=(1, 2)
    )
    session.append((1, 2))
    session.cancelled = cancelled
    result = lease.commit(
        scope="scope",
        response_id="response",
        status=status,
        stored=stored,
    )

    assert result.admitted is False
    assert result.reason == reason
    assert session.closed is True and session.close_calls == 1
    assert cache.stats()["entries"] == 0
    assert cache.acquire(
        scope="scope", response_id="response", prompt_token_ids=(1, 2, 3)
    ) is None


def test_metrics_exclude_decoded_output_rows_from_input_cache_writes() -> None:
    model = _FakeModel()
    cache = NativePrefixCache(model, capacity=2)
    _, parent_result = _commit_fresh(
        cache, model, scope="scope", response_id="parent", tokens=(1, 2, 3)
    )
    assert parent_result.usage.cached_tokens == 0
    assert parent_result.usage.cache_write_tokens == 3
    assert parent_result.usage.private_bytes == 64

    child = cache.acquire(
        scope="scope", response_id="parent", prompt_token_ids=(1, 2, 3, 4, 5)
    )
    assert child is not None
    initial = child.usage()
    assert initial.cached_tokens == 3
    assert initial.cache_write_tokens == 0
    assert initial.shared_bytes == 64
    assert initial.private_bytes == 0
    # Two prompt rows followed by two decoded output rows.  The native session
    # counter reaches seven, but Responses input cache writes stop at five.
    child.session.append((4, 5, 90, 91))
    result = child.commit(
        scope="scope", response_id="child", status="completed", stored=True
    )

    assert result.usage.input_tokens_details() == {
        "cached_tokens": 3,
        "cache_write_tokens": 2,
    }
    assert result.usage.shared_bytes == 0
    assert result.usage.private_bytes == 64
    assert result.usage.detach_bytes == 64
    stats = cache.stats()
    assert stats["cached_tokens"] == 3
    assert stats["cache_write_tokens"] == 5
    assert stats["private_bytes"] == 128
    assert stats["detach_bytes"] == 64
    assert "per-session-capacity" in stats["byte_accounting_scope"]


def test_shutdown_is_idempotent_rejects_new_work_and_preserves_child_lease() -> None:
    model = _FakeModel()
    cache = NativePrefixCache(model, capacity=2)
    first, _ = _commit_fresh(
        cache, model, scope="scope", response_id="first", tokens=(1, 2)
    )
    second, _ = _commit_fresh(
        cache, model, scope="scope", response_id="second", tokens=(3, 4)
    )
    child = cache.acquire(
        scope="scope", response_id="first", prompt_token_ids=(1, 2, 5)
    )
    assert child is not None

    cache.shutdown()
    cache.shutdown()
    assert first.close_calls == 1
    assert second.close_calls == 1
    assert child.session.closed is False
    assert cache.stats()["shutdown"] is True
    assert cache.stats()["entries"] == 0

    with pytest.raises(NativePrefixCacheClosedError):
        cache.acquire(
            scope="scope", response_id="first", prompt_token_ids=(1, 2, 5)
        )
    fresh = model.session()
    with pytest.raises(NativePrefixCacheClosedError):
        cache.lease_session(fresh, scope="scope", prompt_token_ids=(1,))
    assert fresh.closed is False  # rejected before ownership transfer

    child.session.append((5,))
    rejected = child.commit(
        scope="scope", response_id="late", status="completed", stored=True
    )
    assert rejected.admitted is False
    assert rejected.reason == "shutdown"
    assert child.closed is True


def test_mutated_sealed_source_is_retired_as_miss() -> None:
    model = _FakeModel()
    cache = NativePrefixCache(model, capacity=1)
    parent, _ = _commit_fresh(
        cache, model, scope="scope", response_id="response", tokens=(1, 2)
    )
    # This deliberately violates the ownership contract.  The manager checks
    # the live handle against its immutable seal before asking native code to
    # fork, so retained-token ancestry cannot be misreported as KV reuse.
    parent.append((3,))

    assert cache.acquire(
        scope="scope", response_id="response", prompt_token_ids=(1, 2, 4)
    ) is None
    assert model.forks == []
    assert parent.closed is True and parent.close_calls == 1
    assert cache.stats()["entries"] == 0


def test_fork_contract_violation_closes_child_and_keeps_parent_entry() -> None:
    model = _FakeModel()
    cache = NativePrefixCache(model, capacity=1)
    parent, _ = _commit_fresh(
        cache, model, scope="scope", response_id="response", tokens=(1, 2)
    )
    model.corrupt_fork = True

    with pytest.raises(NativePrefixCacheInvariantError):
        cache.acquire(
            scope="scope", response_id="response", prompt_token_ids=(1, 2, 3)
        )
    assert parent.closed is False
    assert model.sessions[-1].closed is True
    assert cache.stats()["entries"] == 1


def test_close_callbacks_run_outside_manager_lock_and_once() -> None:
    model = _FakeModel()
    cache = NativePrefixCache(model, capacity=1)
    parent, _ = _commit_fresh(
        cache, model, scope="scope", response_id="response", tokens=(1,)
    )
    observed: list[int] = []
    parent.close_hook = lambda: observed.append(cache.stats()["entries"])

    assert cache.purge_scope("scope") == 1
    assert observed == [0]
    assert parent.close_calls == 1
    cache.shutdown()
    assert parent.close_calls == 1


def test_invalid_commit_does_not_consume_lease() -> None:
    model = _FakeModel()
    cache = NativePrefixCache(model, capacity=1)
    session = model.session()
    lease = cache.lease_session(session, scope="scope", prompt_token_ids=(1,))
    session.append((1,))

    with pytest.raises(ValueError, match="lease scope"):
        lease.commit(
            scope="different-scope",
            response_id="response",
            status="completed",
            stored=True,
        )
    assert lease.closed is False
    with pytest.raises(ValueError, match="status"):
        lease.commit(
            scope="scope", response_id="response", status="running", stored=True
        )
    assert lease.closed is False
    lease.close()
    assert session.close_calls == 1
