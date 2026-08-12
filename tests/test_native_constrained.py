from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import json
import math
from typing import Any, Callable

import pytest

from neuralfn.native_constrained import (
    JSON_SCHEMA_ASCII_BYTE_GREEDY_PROFILE,
    MAX_ENUM_VALUES,
    MAX_OUTPUT_BYTES,
    MAX_PROPERTIES,
    MAX_PROPERTY_NAME_BYTES,
    MAX_SCHEMA_BYTES,
    CompiledJSONSchema,
    NativeConstrainedCapabilityError,
    NativeConstrainedInvariantError,
    NativeConstrainedSchemaError,
    compile_json_schema_ascii_byte_greedy,
    compile_single_byte_token_inventory,
    generate_json_schema_ascii_byte_greedy,
)


def _text_format(
    properties: Mapping[str, Mapping[str, Any]],
    *,
    required: Sequence[str] | None = None,
) -> dict[str, Any]:
    property_dict = {name: dict(spec) for name, spec in properties.items()}
    return {
        "type": "json_schema",
        "name": "bounded_result",
        "schema": {
            "type": "object",
            "properties": property_dict,
            "required": list(required if required is not None else property_dict),
            "additionalProperties": False,
        },
        "strict": True,
    }


def _compile_all_scalars() -> CompiledJSONSchema:
    return compile_json_schema_ascii_byte_greedy(
        _text_format(
            {
                "answer": {"type": "string"},
                "count": {"type": "integer"},
                "ratio": {"type": "number"},
                "ready": {"type": "boolean"},
                "kind": {"type": "string", "enum": ["a", "b"]},
            }
        )
    )


class ByteCodec:
    """Test codec with one canonical token for every raw byte."""

    def __init__(
        self,
        *,
        vocab_size: int = 128,
        replacements: Mapping[int, bytes] | None = None,
        fail_token: int | None = None,
    ) -> None:
        self.vocab_size = vocab_size
        self.replacements = dict(replacements or {})
        self.fail_token = fail_token
        self.calls: list[int] = []

    def token_bytes(self, token_id: int) -> bytes:
        self.calls.append(token_id)
        if token_id == self.fail_token:
            raise KeyError(token_id)
        if token_id in self.replacements:
            return self.replacements[token_id]
        if 0 <= token_id < 128:
            return bytes((token_id,))
        return b"multi-byte"


class GreedySession:
    def __init__(
        self,
        codec: ByteCodec,
        *,
        target: bytes | None = None,
        logit_factory: Callable[[int, int], list[float]] | None = None,
    ) -> None:
        self.codec = codec
        self.tokens = [1]
        self.target = target
        self.logit_factory = logit_factory
        self.logit_calls = 0
        self.prefill_calls: list[tuple[int, ...]] = []
        self.decode_calls = 0

    @property
    def token_ids(self) -> tuple[int, ...]:
        return tuple(self.tokens)

    def current_logits(self) -> list[float]:
        index = len(self.tokens) - 1
        self.logit_calls += 1
        if self.logit_factory is not None:
            return self.logit_factory(index, self.codec.vocab_size)
        assert self.target is not None and index < len(self.target)
        logits = [-1000.0] * self.codec.vocab_size
        # This is the unconstrained global argmax, but a newline is never in the
        # profile's printable-byte grammar.
        logits[ord("\n")] = 1000.0
        logits[self.target[index]] = 100.0
        return logits

    def prefill(self, token_ids: Sequence[int]) -> dict[str, int]:
        requested = tuple(token_ids)
        assert requested[:-1] == tuple(self.tokens)
        assert len(requested) == len(self.tokens) + 1
        self.tokens[:] = requested
        self.prefill_calls.append(requested)
        return {
            "prefix_tokens": len(requested),
            "prefix_reused": len(requested) - 1,
            "prefilled_tokens": 1,
        }

    def decode(self, *_args: Any, **_kwargs: Any) -> None:
        self.decode_calls += 1
        raise AssertionError("ordinary decode must not be called")


def _feed(compiled: CompiledJSONSchema, payload: bytes) -> None:
    grammar = compiled.new_grammar()
    for value in payload:
        assert value in grammar.allowed_bytes()
        grammar.advance(value)
    assert grammar.prefix == payload
    assert grammar.accepting
    assert grammar.allowed_bytes() == frozenset()


def test_compile_accepts_exact_flat_strict_scalar_and_enum_profile() -> None:
    compiled = _compile_all_scalars()

    assert compiled.profile == JSON_SCHEMA_ASCII_BYTE_GREEDY_PROFILE
    assert compiled.name == "bounded_result"
    assert compiled.property_names == ("answer", "count", "ratio", "ready", "kind")
    assert len(compiled.canonical_schema_json.encode("ascii")) <= MAX_SCHEMA_BYTES
    assert compiled.format_payload() == _text_format(
        {
            "answer": {"type": "string"},
            "count": {"type": "integer"},
            "ratio": {"type": "number"},
            "ready": {"type": "boolean"},
            "kind": {"type": "string", "enum": ["a", "b"]},
        }
    )
    compiled.validate_instance(
        {"answer": "ok", "count": -12, "ratio": 150.0, "ready": True, "kind": "a"}
    )


@pytest.mark.parametrize(
    ("mutate", "match"),
    (
        (lambda value: value.pop("name"), "missing required field 'name'"),
        (lambda value: value.__setitem__("description", "x"), "field 'description'"),
        (lambda value: value.__setitem__("type", "text"), "must be 'json_schema'"),
        (lambda value: value.__setitem__("strict", False), "strict must be true"),
        (lambda value: value.__setitem__("name", "not valid"), "must match"),
        (lambda value: value["schema"].__setitem__("type", "array"), "root type"),
        (
            lambda value: value["schema"].__setitem__("additionalProperties", True),
            "additionalProperties must be false",
        ),
        (lambda value: value["schema"].__setitem__("default", {}), "keyword 'default'"),
        (
            lambda value: value["schema"]["properties"]["value"].__setitem__(
                "type", "array"
            ),
            "type must be one of",
        ),
        (
            lambda value: value["schema"]["properties"]["value"].__setitem__(
                "minimum", 0
            ),
            "keyword 'minimum'",
        ),
        (lambda value: value["schema"].__setitem__("required", []), "every property"),
        (
            lambda value: value["schema"].__setitem__("required", ["value", "value"]),
            "must not contain duplicates",
        ),
    ),
)
def test_compile_rejects_format_and_schema_outside_profile(
    mutate: Callable[[dict[str, Any]], Any],
    match: str,
) -> None:
    payload = _text_format({"value": {"type": "string"}})
    mutate(payload)

    with pytest.raises(NativeConstrainedSchemaError, match=match):
        compile_json_schema_ascii_byte_greedy(payload)


def test_compile_rejects_non_string_schema_object_keys_cleanly() -> None:
    payload = _text_format({"value": {"type": "string"}})
    payload["schema"][3] = "invalid"

    with pytest.raises(NativeConstrainedSchemaError, match="field names must be strings"):
        compile_json_schema_ascii_byte_greedy(payload)


def test_compile_rejects_profile_size_and_ascii_limit_violations() -> None:
    too_many = {
        f"p{index}": {"type": "boolean"} for index in range(MAX_PROPERTIES + 1)
    }
    with pytest.raises(NativeConstrainedSchemaError, match="between 1 and 32"):
        compile_json_schema_ascii_byte_greedy(_text_format(too_many))

    long_name = "p" * (MAX_PROPERTY_NAME_BYTES + 1)
    with pytest.raises(NativeConstrainedSchemaError, match="1--64 printable ASCII"):
        compile_json_schema_ascii_byte_greedy(
            _text_format({long_name: {"type": "string"}})
        )

    with pytest.raises(NativeConstrainedSchemaError, match="printable ASCII strings"):
        compile_json_schema_ascii_byte_greedy(
            _text_format({"value": {"type": "string", "enum": ["café"]}})
        )

    too_many_enum_values = list(range(MAX_ENUM_VALUES + 1))
    with pytest.raises(NativeConstrainedSchemaError, match="between 1 and 64 values"):
        compile_json_schema_ascii_byte_greedy(
            _text_format(
                {"value": {"type": "integer", "enum": too_many_enum_values}}
            )
        )

    oversized = _text_format(
        {"value": {"type": "string", "enum": ["x" * MAX_SCHEMA_BYTES]}}
    )
    with pytest.raises(NativeConstrainedSchemaError, match="must not exceed"):
        compile_json_schema_ascii_byte_greedy(oversized)


def test_compile_accepts_and_preserves_pydantic_title_description_annotations() -> None:
    # This is the basic shape emitted by Pydantic for an extra-forbid model and
    # passed by the typed SDK's strict JSON-schema helper.
    payload = {
        "type": "json_schema",
        "name": "Answer",
        "strict": True,
        "schema": {
            "title": "Answer",
            "description": "One bounded result.",
            "type": "object",
            "properties": {
                "answer": {
                    "title": "Answer",
                    "description": "Final answer",
                    "type": "string",
                },
                "count": {"title": "Count", "type": "integer"},
            },
            "required": ["answer", "count"],
            "additionalProperties": False,
        },
    }

    compiled = compile_json_schema_ascii_byte_greedy(payload)

    assert compiled.format_payload() == payload
    _feed(compiled, b'{"answer":"yes","count":2}')


@pytest.mark.parametrize(
    "path",
    (
        ("schema", "title"),
        ("schema", "description"),
        ("schema", "properties", "value", "title"),
        ("schema", "properties", "value", "description"),
    ),
)
def test_compile_requires_annotation_values_to_be_strings(path: tuple[str, ...]) -> None:
    payload = _text_format({"value": {"type": "string"}})
    cursor: dict[str, Any] = payload
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = 3

    with pytest.raises(NativeConstrainedSchemaError, match="must be a string"):
        compile_json_schema_ascii_byte_greedy(payload)


@pytest.mark.parametrize(
    ("value_type", "enum", "match"),
    (
        ("string", [], "between 1 and 64"),
        ("string", ["a", "a"], "must be unique"),
        ("integer", [1, True], "must contain integers"),
        ("number", [1, 1.0], "must be unique"),
        ("number", [math.inf], "finite JSON numbers"),
        ("boolean", [True, 1], "must contain booleans"),
    ),
)
def test_compile_rejects_invalid_or_non_homogeneous_enums(
    value_type: str,
    enum: list[Any],
    match: str,
) -> None:
    with pytest.raises(NativeConstrainedSchemaError, match=match):
        compile_json_schema_ascii_byte_greedy(
            _text_format({"value": {"type": value_type, "enum": enum}})
        )


def test_byte_grammar_accepts_scalar_forms_escapes_and_enum_branch() -> None:
    compiled = _compile_all_scalars()
    payload = b'{"answer":"a\\\"b\\\\c","count":-12,"ratio":1.5e2,"ready":true,"kind":"b"}'

    _feed(compiled, payload)
    parsed = json.loads(payload)
    compiled.validate_instance(parsed)


def test_byte_grammar_rejects_wrong_order_and_invalid_numeric_prefixes() -> None:
    compiled = compile_json_schema_ascii_byte_greedy(
        _text_format({"integer": {"type": "integer"}, "number": {"type": "number"}})
    )
    grammar = compiled.new_grammar()
    with pytest.raises(NativeConstrainedInvariantError, match="not a viable"):
        grammar.advance(ord("["))

    integer_prefix = b'{"integer":-'
    grammar = compiled.new_grammar()
    for value in integer_prefix:
        grammar.advance(value)
    assert ord(",") not in grammar.allowed_bytes()
    assert set(range(ord("0"), ord("9") + 1)) <= set(grammar.allowed_bytes())

    zero_prefix = b'{"integer":0'
    grammar = compiled.new_grammar()
    for value in zero_prefix:
        grammar.advance(value)
    assert ord("1") not in grammar.allowed_bytes()
    assert ord(",") in grammar.allowed_bytes()


def test_inventory_requires_exact_complete_printable_ascii_tokens() -> None:
    codec = ByteCodec()
    inventory = compile_single_byte_token_inventory(codec, codec.vocab_size)
    assert inventory.vocab_size == 128
    assert inventory.token_ids(ord("A")) == (ord("A"),)
    assert codec.calls == list(range(128))

    missing = ByteCodec(replacements={ord("~"): b"not-one-byte"})
    with pytest.raises(NativeConstrainedCapabilityError, match="0x7e"):
        compile_single_byte_token_inventory(missing, missing.vocab_size)

    opaque = ByteCodec(fail_token=7)
    with pytest.raises(NativeConstrainedCapabilityError, match="token 7"):
        compile_single_byte_token_inventory(opaque, opaque.vocab_size)


def test_generator_selects_highest_allowed_bytes_commits_then_emits() -> None:
    compiled = _compile_all_scalars()
    target = b'{"answer":"ok","count":-12,"ratio":1.5e2,"ready":true,"kind":"a"}'
    codec = ByteCodec()
    session = GreedySession(codec, target=target)
    callbacks: list[tuple[int, tuple[int, ...], str | None]] = []

    def committed(event) -> None:
        callbacks.append((event.token_id, session.token_ids, event.finish_reason))

    result = generate_json_schema_ascii_byte_greedy(
        session,
        codec,
        compiled,
        max_new_tokens=len(target) + 10,
        on_token=committed,
    )

    assert result.text == target.decode("ascii")
    assert result.token_ids == tuple(target)
    assert result.finish_reason == "stop"
    assert result.prompt_tokens == 1
    assert result.completion_tokens == len(target)
    assert result.events[-1].finish_reason == "stop"
    assert all(event.finish_reason is None for event in result.events[:-1])
    assert session.decode_calls == 0
    assert len(session.prefill_calls) == len(target)
    assert session.logit_calls == len(target)
    assert all(tokens[-1] == token_id for token_id, tokens, _finish in callbacks)
    assert callbacks[-1][2] == "stop"


def test_generator_tie_breaks_duplicate_byte_tokens_by_lowest_id() -> None:
    compiled = compile_json_schema_ascii_byte_greedy(
        _text_format({"ok": {"type": "boolean"}})
    )
    codec = ByteCodec(replacements={20: b"{"})

    def logits(_index: int, vocab_size: int) -> list[float]:
        values = [-1.0] * vocab_size
        values[20] = 10.0
        values[ord("{")] = 10.0
        return values

    session = GreedySession(codec, logit_factory=logits)
    result = generate_json_schema_ascii_byte_greedy(
        session,
        codec,
        compiled,
        max_new_tokens=1,
    )

    assert result.finish_reason == "length"
    assert result.text == "{"
    assert result.token_ids == (20,)


def test_generator_returns_incomplete_partial_prefix_at_token_bound() -> None:
    compiled = compile_json_schema_ascii_byte_greedy(
        _text_format({"value": {"type": "string"}})
    )
    codec = ByteCodec()

    def prefer_unclosed_string(_index: int, vocab_size: int) -> list[float]:
        logits = [-1000.0] * vocab_size
        logits[ord("a")] = 100.0
        return logits

    session = GreedySession(codec, logit_factory=prefer_unclosed_string)
    result = generate_json_schema_ascii_byte_greedy(
        session,
        codec,
        compiled,
        max_new_tokens=20,
    )

    assert result.finish_reason == "length"
    assert result.completion_tokens == 20
    assert not result.text.endswith("}")
    assert session.decode_calls == 0


def test_generator_runs_independent_final_json_schema_invariant(monkeypatch) -> None:
    compiled = compile_json_schema_ascii_byte_greedy(
        _text_format({"ok": {"type": "boolean"}})
    )
    target = b'{"ok":true}'
    codec = ByteCodec()
    session = GreedySession(codec, target=target)
    validated: list[Any] = []
    original = CompiledJSONSchema.validate_instance

    def recording_validate(self: CompiledJSONSchema, value: Any) -> None:
        validated.append(value)
        original(self, value)

    monkeypatch.setattr(CompiledJSONSchema, "validate_instance", recording_validate)
    result = generate_json_schema_ascii_byte_greedy(
        session,
        codec,
        compiled,
        max_new_tokens=len(target),
    )

    assert result.finish_reason == "stop"
    assert validated == [{"ok": True}]


def test_generator_fails_closed_before_commit_on_missing_primitives_or_inventory() -> None:
    compiled = compile_json_schema_ascii_byte_greedy(
        _text_format({"ok": {"type": "boolean"}})
    )
    codec = ByteCodec(replacements={ord("~"): b"two"})
    session = GreedySession(codec, target=b'{"ok":true}')

    with pytest.raises(NativeConstrainedCapabilityError, match="standalone tokens"):
        generate_json_schema_ascii_byte_greedy(
            session,
            codec,
            compiled,
            max_new_tokens=20,
        )
    assert session.prefill_calls == []

    class MissingLogits:
        token_ids = (1,)

        def prefill(self, _token_ids: Sequence[int]) -> None:
            raise AssertionError("must not be reached")

    with pytest.raises(NativeConstrainedCapabilityError, match="current_logits"):
        generate_json_schema_ascii_byte_greedy(
            MissingLogits(),
            ByteCodec(),
            compiled,
            max_new_tokens=20,
        )


def test_generator_detects_logit_width_and_exact_prefill_contract_violations() -> None:
    compiled = compile_json_schema_ascii_byte_greedy(
        _text_format({"ok": {"type": "boolean"}})
    )
    codec = ByteCodec()

    class ChangingLogits(GreedySession):
        def current_logits(self) -> list[float]:
            values = super().current_logits()
            return values if self.logit_calls == 1 else values[:-1]

    changing = ChangingLogits(codec, target=b'{"ok":true}')
    with pytest.raises(NativeConstrainedInvariantError, match="vocabulary changed"):
        generate_json_schema_ascii_byte_greedy(
            changing,
            codec,
            compiled,
            max_new_tokens=20,
        )

    class NonCommitting(GreedySession):
        def prefill(self, token_ids: Sequence[int]) -> None:
            self.prefill_calls.append(tuple(token_ids))

    non_committing = NonCommitting(codec, target=b'{"ok":true}')
    with pytest.raises(NativeConstrainedInvariantError, match="did not commit exactly"):
        generate_json_schema_ascii_byte_greedy(
            non_committing,
            codec,
            compiled,
            max_new_tokens=20,
        )

    class ExternallyMutated(GreedySession):
        def current_logits(self) -> list[float]:
            values = super().current_logits()
            if self.logit_calls == 2:
                self.tokens.append(7)
            return values

    mutated = ExternallyMutated(codec, target=b'{"ok":true}')
    with pytest.raises(NativeConstrainedInvariantError, match="changed outside"):
        generate_json_schema_ascii_byte_greedy(
            mutated,
            codec,
            compiled,
            max_new_tokens=20,
        )


@pytest.mark.parametrize("value", (0, MAX_OUTPUT_BYTES + 1, True, 1.5))
def test_generator_rejects_invalid_token_bounds(value: Any) -> None:
    compiled = compile_json_schema_ascii_byte_greedy(
        _text_format({"ok": {"type": "boolean"}})
    )
    with pytest.raises((TypeError, ValueError), match="max_new_tokens"):
        generate_json_schema_ascii_byte_greedy(
            object(),
            ByteCodec(),
            compiled,
            max_new_tokens=value,
        )


def test_independent_validator_rejects_wrong_order_type_enum_and_extra_fields() -> None:
    compiled = compile_json_schema_ascii_byte_greedy(
        _text_format(
            {
                "kind": {"type": "string", "enum": ["a", "b"]},
                "count": {"type": "integer"},
            }
        )
    )
    invalid_values = (
        {"count": 1, "kind": "a"},
        {"kind": "a", "count": True},
        {"kind": "c", "count": 1},
        {"kind": "a", "count": 1, "extra": False},
    )
    for value in invalid_values:
        with pytest.raises(NativeConstrainedInvariantError):
            compiled.validate_instance(copy.deepcopy(value))
