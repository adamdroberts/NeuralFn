"""Bounded byte-exact constrained generation for resident native sessions.

This module intentionally implements one small profile rather than a general
JSON Schema engine.  It compiles strict, flat object schemas and constrains
greedy generation before each token is committed.  Only tokenizer IDs whose
exact payload is one printable ASCII byte participate in generation.

The resident ABI remains unchanged: logits are read with ``current_logits``
and the selected token is committed through exact-prefix ``prefill``.  Calling
ordinary ``decode`` would commit an unconstrained token and is never permitted
by this engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import json
import math
import re
from typing import Any, Callable, Mapping, Sequence

from .native_inference import GenerationEvent, GenerationResult


JSON_SCHEMA_ASCII_BYTE_GREEDY_PROFILE = "json-schema-ascii-byte-greedy-v1"
MAX_SCHEMA_BYTES = 32 * 1024
MAX_PROPERTIES = 32
MAX_PROPERTY_NAME_BYTES = 64
MAX_ENUM_VALUES = 64
MAX_OUTPUT_BYTES = 4 * 1024

_FORMAT_FIELDS = frozenset({"type", "name", "schema", "strict"})
_SCHEMA_REQUIRED_FIELDS = frozenset(
    {"type", "properties", "required", "additionalProperties"}
)
_SCHEMA_FIELDS = _SCHEMA_REQUIRED_FIELDS | {"title", "description"}
_PROPERTY_FIELDS = frozenset({"type", "enum", "title", "description"})
_SCALAR_TYPES = frozenset({"string", "integer", "number", "boolean"})
_FORMAT_NAME_RE = re.compile(r"[A-Za-z0-9_-]{1,64}\Z", re.ASCII)
_PRINTABLE_ASCII = frozenset(range(0x20, 0x7F))
_STRING_BODY_ASCII = _PRINTABLE_ASCII - {ord('"'), ord("\\")}


class NativeConstrainedError(RuntimeError):
    """Base error for the bounded constrained-generation engine."""


class NativeConstrainedSchemaError(ValueError):
    """Raised when a requested schema is outside the supported profile."""


class NativeConstrainedCapabilityError(NativeConstrainedError):
    """Raised when a session/tokenizer cannot prove the profile primitives."""


class NativeConstrainedInvariantError(NativeConstrainedError):
    """Raised when committed state or a completed output violates the contract."""


def _schema_error(message: str) -> NativeConstrainedSchemaError:
    return NativeConstrainedSchemaError(
        f"{JSON_SCHEMA_ASCII_BYTE_GREEDY_PROFILE}: {message}"
    )


def _is_printable_ascii(value: str) -> bool:
    return all(0x20 <= ord(character) <= 0x7E for character in value)


def _field_names(mapping: Mapping[Any, Any], *, context: str) -> set[str]:
    fields = set(mapping)
    if any(not isinstance(field, str) for field in fields):
        raise _schema_error(f"{context} field names must be strings")
    return fields


def _json_literal(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise _schema_error("schema values must be finite JSON values") from exc
    return encoded.encode("ascii")


def _number_key(value: int | float | Decimal) -> Decimal:
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _enum_key(value_type: str, value: Any) -> tuple[str, Any]:
    if value_type == "number":
        return (value_type, _number_key(value))
    return (value_type, value)


def _validate_scalar_value(value_type: str, value: Any, *, context: str) -> None:
    if value_type == "string":
        if not isinstance(value, str) or not _is_printable_ascii(value):
            raise _schema_error(f"{context} must contain printable ASCII strings")
        return
    if value_type == "boolean":
        if not isinstance(value, bool):
            raise _schema_error(f"{context} must contain booleans")
        return
    if value_type == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise _schema_error(f"{context} must contain integers")
        return
    if value_type == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _schema_error(f"{context} must contain JSON numbers")
        if isinstance(value, float) and not math.isfinite(value):
            raise _schema_error(f"{context} must contain finite JSON numbers")
        return
    raise AssertionError(f"unexpected scalar type {value_type!r}")


@dataclass(frozen=True, slots=True)
class _PropertySpec:
    name: str
    value_type: str
    enum_values: tuple[Any, ...] | None
    enum_literals: tuple[bytes, ...] | None


@dataclass(frozen=True, slots=True)
class CompiledJSONSchema:
    """Validated immutable form of the bounded structured-output profile."""

    name: str
    properties: tuple[_PropertySpec, ...]
    canonical_schema_json: str
    profile: str = JSON_SCHEMA_ASCII_BYTE_GREEDY_PROFILE

    @property
    def property_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.properties)

    def format_payload(self) -> dict[str, Any]:
        """Return the canonical Responses ``text.format`` payload."""

        return {
            "type": "json_schema",
            "name": self.name,
            "schema": json.loads(self.canonical_schema_json),
            "strict": True,
        }

    def new_grammar(self) -> "JSONSchemaByteGrammar":
        return JSONSchemaByteGrammar(self)

    def validate_instance(self, value: Any) -> None:
        """Independently validate one decoded instance against this profile."""

        if not isinstance(value, Mapping):
            raise NativeConstrainedInvariantError("structured output is not an object")
        actual_names = tuple(value.keys())
        if actual_names != self.property_names:
            raise NativeConstrainedInvariantError(
                "structured output properties or property order do not match the schema"
            )
        for spec in self.properties:
            candidate = value[spec.name]
            if spec.value_type == "string":
                valid_type = isinstance(candidate, str) and _is_printable_ascii(candidate)
            elif spec.value_type == "boolean":
                valid_type = isinstance(candidate, bool)
            elif spec.value_type == "integer":
                valid_type = isinstance(candidate, int) and not isinstance(candidate, bool)
            else:
                valid_type = (
                    not isinstance(candidate, bool)
                    and isinstance(candidate, (int, float, Decimal))
                    and not (
                        isinstance(candidate, float) and not math.isfinite(candidate)
                    )
                    and not (
                        isinstance(candidate, Decimal) and not candidate.is_finite()
                    )
                )
            if not valid_type:
                raise NativeConstrainedInvariantError(
                    f"structured output property {spec.name!r} is not a {spec.value_type}"
                )
            if spec.enum_values is not None:
                expected = {
                    _enum_key(spec.value_type, enum_value)
                    for enum_value in spec.enum_values
                }
                if _enum_key(spec.value_type, candidate) not in expected:
                    raise NativeConstrainedInvariantError(
                        f"structured output property {spec.name!r} is outside its enum"
                    )


def compile_json_schema_ascii_byte_greedy(
    text_format: Mapping[str, Any],
) -> CompiledJSONSchema:
    """Validate and compile one Responses ``text.format`` object.

    The accepted schema is exactly a strict flat object with 1--32 required
    scalar properties.  Scalar types are string, integer, number, and boolean;
    each property may instead be narrowed by a finite homogeneous enum.
    """

    if not isinstance(text_format, Mapping):
        raise _schema_error("text.format must be an object")
    fields = _field_names(text_format, context="text.format")
    missing = sorted(_FORMAT_FIELDS - fields)
    unknown = sorted(fields - _FORMAT_FIELDS)
    if missing:
        raise _schema_error(f"text.format is missing required field {missing[0]!r}")
    if unknown:
        raise _schema_error(f"text.format field {unknown[0]!r} is not supported")
    if text_format.get("type") != "json_schema":
        raise _schema_error("text.format.type must be 'json_schema'")
    if text_format.get("strict") is not True:
        raise _schema_error("text.format.strict must be true")
    name = text_format.get("name")
    if not isinstance(name, str) or _FORMAT_NAME_RE.fullmatch(name) is None:
        raise _schema_error(
            "text.format.name must match [A-Za-z0-9_-]{1,64}"
        )

    schema = text_format.get("schema")
    if not isinstance(schema, Mapping):
        raise _schema_error("text.format.schema must be an object")
    schema_fields = _field_names(schema, context="schema")
    missing_schema = sorted(_SCHEMA_REQUIRED_FIELDS - schema_fields)
    unknown_schema = sorted(schema_fields - _SCHEMA_FIELDS)
    if missing_schema:
        raise _schema_error(
            f"schema is missing required field {missing_schema[0]!r}"
        )
    if unknown_schema:
        raise _schema_error(f"schema keyword {unknown_schema[0]!r} is not supported")
    if schema.get("type") != "object":
        raise _schema_error("schema root type must be 'object'")
    if schema.get("additionalProperties") is not False:
        raise _schema_error("schema.additionalProperties must be false")
    for annotation in ("title", "description"):
        if annotation in schema and not isinstance(schema[annotation], str):
            raise _schema_error(f"schema.{annotation} must be a string")

    raw_properties = schema.get("properties")
    if not isinstance(raw_properties, Mapping):
        raise _schema_error("schema.properties must be an object")
    if not 1 <= len(raw_properties) <= MAX_PROPERTIES:
        raise _schema_error(
            f"schema.properties must contain between 1 and {MAX_PROPERTIES} properties"
        )
    property_names: list[str] = []
    property_specs: list[_PropertySpec] = []
    for property_name, raw_spec in raw_properties.items():
        if not isinstance(property_name, str):
            raise _schema_error("schema property names must be strings")
        encoded_name = (
            property_name.encode("ascii", errors="strict")
            if _is_printable_ascii(property_name)
            else b""
        )
        if not encoded_name or len(encoded_name) > MAX_PROPERTY_NAME_BYTES:
            raise _schema_error(
                "schema property names must be 1--64 printable ASCII bytes"
            )
        property_names.append(property_name)
        if not isinstance(raw_spec, Mapping):
            raise _schema_error(f"property {property_name!r} schema must be an object")
        spec_fields = _field_names(
            raw_spec,
            context=f"property {property_name!r} schema",
        )
        missing_spec = {"type"} - spec_fields
        unknown_spec = spec_fields - _PROPERTY_FIELDS
        if missing_spec:
            raise _schema_error(f"property {property_name!r} is missing 'type'")
        if unknown_spec:
            keyword = sorted(unknown_spec)[0]
            raise _schema_error(
                f"property {property_name!r} keyword {keyword!r} is not supported"
            )
        for annotation in ("title", "description"):
            if annotation in raw_spec and not isinstance(raw_spec[annotation], str):
                raise _schema_error(
                    f"property {property_name!r} {annotation} must be a string"
                )
        value_type = raw_spec.get("type")
        if not isinstance(value_type, str) or value_type not in _SCALAR_TYPES:
            raise _schema_error(
                f"property {property_name!r} type must be one of: "
                "boolean, integer, number, string"
            )
        enum_values: tuple[Any, ...] | None = None
        enum_literals: tuple[bytes, ...] | None = None
        if "enum" in raw_spec:
            raw_enum = raw_spec["enum"]
            if not isinstance(raw_enum, list) or not 1 <= len(raw_enum) <= MAX_ENUM_VALUES:
                raise _schema_error(
                    f"property {property_name!r} enum must contain between 1 and "
                    f"{MAX_ENUM_VALUES} values"
                )
            seen: set[tuple[str, Any]] = set()
            values: list[Any] = []
            literals: list[bytes] = []
            for enum_value in raw_enum:
                _validate_scalar_value(
                    value_type,
                    enum_value,
                    context=f"property {property_name!r} enum",
                )
                key = _enum_key(value_type, enum_value)
                if key in seen:
                    raise _schema_error(
                        f"property {property_name!r} enum values must be unique"
                    )
                seen.add(key)
                values.append(enum_value)
                literals.append(_json_literal(enum_value))
            enum_values = tuple(values)
            enum_literals = tuple(literals)
        property_specs.append(
            _PropertySpec(
                name=property_name,
                value_type=value_type,
                enum_values=enum_values,
                enum_literals=enum_literals,
            )
        )

    raw_required = schema.get("required")
    if not isinstance(raw_required, list) or any(
        not isinstance(required_name, str) for required_name in raw_required
    ):
        raise _schema_error("schema.required must be an array of property names")
    if len(raw_required) != len(set(raw_required)):
        raise _schema_error("schema.required must not contain duplicates")
    if set(raw_required) != set(property_names):
        raise _schema_error("schema.required must contain every property exactly once")

    try:
        canonical_schema_json = json.dumps(
            schema,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise _schema_error("schema must contain only finite JSON values") from exc
    if len(canonical_schema_json.encode("ascii")) > MAX_SCHEMA_BYTES:
        raise _schema_error(
            f"canonical schema must not exceed {MAX_SCHEMA_BYTES} bytes"
        )
    return CompiledJSONSchema(
        name=name,
        properties=tuple(property_specs),
        canonical_schema_json=canonical_schema_json,
    )


class _ValueState:
    @property
    def can_finish(self) -> bool:
        raise NotImplementedError

    def allowed_bytes(self) -> frozenset[int]:
        raise NotImplementedError

    def advance(self, value: int) -> None:
        raise NotImplementedError


class _EnumState(_ValueState):
    def __init__(self, literals: tuple[bytes, ...]) -> None:
        self._literals = literals
        self._prefix = b""

    @property
    def can_finish(self) -> bool:
        return self._prefix in self._literals

    def allowed_bytes(self) -> frozenset[int]:
        offset = len(self._prefix)
        return frozenset(
            literal[offset]
            for literal in self._literals
            if len(literal) > offset and literal.startswith(self._prefix)
        )

    def advance(self, value: int) -> None:
        candidate = self._prefix + bytes((value,))
        if not any(literal.startswith(candidate) for literal in self._literals):
            raise NativeConstrainedInvariantError("byte is not a viable enum prefix")
        self._prefix = candidate


class _StringState(_ValueState):
    _START = 0
    _BODY = 1
    _ESCAPE = 2
    _DONE = 3

    def __init__(self) -> None:
        self._state = self._START

    @property
    def can_finish(self) -> bool:
        return self._state == self._DONE

    def allowed_bytes(self) -> frozenset[int]:
        if self._state == self._START:
            return frozenset({ord('"')})
        if self._state == self._BODY:
            return _STRING_BODY_ASCII | {ord('"'), ord("\\")}
        if self._state == self._ESCAPE:
            return frozenset({ord('"'), ord("\\")})
        return frozenset()

    def advance(self, value: int) -> None:
        if value not in self.allowed_bytes():
            raise NativeConstrainedInvariantError("byte is not a viable string prefix")
        if self._state == self._START:
            self._state = self._BODY
        elif self._state == self._BODY:
            if value == ord('"'):
                self._state = self._DONE
            elif value == ord("\\"):
                self._state = self._ESCAPE
        else:
            self._state = self._BODY


class _BooleanState(_EnumState):
    def __init__(self) -> None:
        super().__init__((b"true", b"false"))


class _IntegerState(_ValueState):
    _START = 0
    _MINUS = 1
    _ZERO = 2
    _DIGITS = 3

    def __init__(self) -> None:
        self._state = self._START

    @property
    def can_finish(self) -> bool:
        return self._state in {self._ZERO, self._DIGITS}

    def allowed_bytes(self) -> frozenset[int]:
        if self._state == self._START:
            return frozenset({ord("-")} | set(range(ord("0"), ord("9") + 1)))
        if self._state == self._MINUS:
            return frozenset(range(ord("0"), ord("9") + 1))
        if self._state == self._DIGITS:
            return frozenset(range(ord("0"), ord("9") + 1))
        return frozenset()

    def advance(self, value: int) -> None:
        if value not in self.allowed_bytes():
            raise NativeConstrainedInvariantError("byte is not a viable integer prefix")
        if self._state == self._START and value == ord("-"):
            self._state = self._MINUS
        elif value == ord("0") and self._state in {self._START, self._MINUS}:
            self._state = self._ZERO
        else:
            self._state = self._DIGITS


class _NumberState(_ValueState):
    _START = 0
    _MINUS = 1
    _ZERO = 2
    _INTEGER = 3
    _DOT = 4
    _FRACTION = 5
    _EXPONENT = 6
    _EXPONENT_SIGN = 7
    _EXPONENT_DIGITS = 8

    def __init__(self) -> None:
        self._state = self._START

    @property
    def can_finish(self) -> bool:
        return self._state in {
            self._ZERO,
            self._INTEGER,
            self._FRACTION,
            self._EXPONENT_DIGITS,
        }

    def allowed_bytes(self) -> frozenset[int]:
        digits = set(range(ord("0"), ord("9") + 1))
        if self._state == self._START:
            return frozenset({ord("-")} | digits)
        if self._state == self._MINUS:
            return frozenset(digits)
        if self._state == self._ZERO:
            return frozenset({ord("."), ord("e"), ord("E")})
        if self._state == self._INTEGER:
            return frozenset(digits | {ord("."), ord("e"), ord("E")})
        if self._state == self._DOT:
            return frozenset(digits)
        if self._state == self._FRACTION:
            return frozenset(digits | {ord("e"), ord("E")})
        if self._state == self._EXPONENT:
            return frozenset(digits | {ord("+"), ord("-")})
        if self._state == self._EXPONENT_SIGN:
            return frozenset(digits)
        if self._state == self._EXPONENT_DIGITS:
            return frozenset(digits)
        raise AssertionError(f"unexpected number state {self._state}")

    def advance(self, value: int) -> None:
        if value not in self.allowed_bytes():
            raise NativeConstrainedInvariantError("byte is not a viable number prefix")
        if self._state == self._START:
            if value == ord("-"):
                self._state = self._MINUS
            elif value == ord("0"):
                self._state = self._ZERO
            else:
                self._state = self._INTEGER
        elif self._state == self._MINUS:
            self._state = self._ZERO if value == ord("0") else self._INTEGER
        elif self._state in {self._ZERO, self._INTEGER}:
            if value == ord("."):
                self._state = self._DOT
            elif value in {ord("e"), ord("E")}:
                self._state = self._EXPONENT
            else:
                self._state = self._INTEGER
        elif self._state == self._DOT:
            self._state = self._FRACTION
        elif self._state == self._FRACTION:
            self._state = (
                self._EXPONENT
                if value in {ord("e"), ord("E")}
                else self._FRACTION
            )
        elif self._state == self._EXPONENT:
            self._state = (
                self._EXPONENT_SIGN
                if value in {ord("+"), ord("-")}
                else self._EXPONENT_DIGITS
            )
        else:
            self._state = self._EXPONENT_DIGITS


def _value_state(spec: _PropertySpec) -> _ValueState:
    if spec.enum_literals is not None:
        return _EnumState(spec.enum_literals)
    if spec.value_type == "string":
        return _StringState()
    if spec.value_type == "boolean":
        return _BooleanState()
    if spec.value_type == "integer":
        return _IntegerState()
    if spec.value_type == "number":
        return _NumberState()
    raise AssertionError(f"unexpected scalar type {spec.value_type!r}")


class JSONSchemaByteGrammar:
    """Incremental viable-prefix grammar for a compiled bounded schema."""

    def __init__(self, compiled: CompiledJSONSchema) -> None:
        if not isinstance(compiled, CompiledJSONSchema):
            raise TypeError("compiled must be a CompiledJSONSchema")
        components: list[bytes | _PropertySpec] = []
        for index, spec in enumerate(compiled.properties):
            separator = b"{" if index == 0 else b","
            property_literal = _json_literal(spec.name)
            components.extend((separator + property_literal + b":", spec))
        components.append(b"}")
        self._components = tuple(components)
        self._component_index = 0
        self._literal_offset = 0
        self._active_value: _ValueState | None = None
        self._prefix = bytearray()

    @property
    def prefix(self) -> bytes:
        return bytes(self._prefix)

    @property
    def accepting(self) -> bool:
        return self._component_index == len(self._components)

    def _component(self) -> bytes | _PropertySpec | None:
        if self.accepting:
            return None
        return self._components[self._component_index]

    def _ensure_value(self, spec: _PropertySpec) -> _ValueState:
        if self._active_value is None:
            self._active_value = _value_state(spec)
        return self._active_value

    def allowed_bytes(self) -> frozenset[int]:
        component = self._component()
        if component is None:
            return frozenset()
        if isinstance(component, bytes):
            return frozenset({component[self._literal_offset]})
        state = self._ensure_value(component)
        allowed = set(state.allowed_bytes())
        if state.can_finish:
            next_component = self._components[self._component_index + 1]
            if not isinstance(next_component, bytes) or not next_component:
                raise NativeConstrainedInvariantError(
                    "compiled grammar is missing a scalar delimiter"
                )
            allowed.add(next_component[0])
        return frozenset(allowed)

    def advance(self, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255:
            raise TypeError("grammar input must be one byte expressed as an integer")
        if self.accepting:
            raise NativeConstrainedInvariantError(
                "cannot append bytes after the grammar has accepted"
            )
        if value not in self.allowed_bytes():
            raise NativeConstrainedInvariantError(
                f"byte 0x{value:02x} is not a viable structured-output prefix"
            )
        while True:
            component = self._component()
            if isinstance(component, bytes):
                if component[self._literal_offset] != value:
                    raise NativeConstrainedInvariantError(
                        "byte does not match the compiled object structure"
                    )
                self._literal_offset += 1
                if self._literal_offset == len(component):
                    self._component_index += 1
                    self._literal_offset = 0
                break
            if component is None:
                raise NativeConstrainedInvariantError(
                    "cannot append bytes after the grammar has accepted"
                )
            state = self._ensure_value(component)
            if value in state.allowed_bytes():
                state.advance(value)
                break
            if not state.can_finish:
                raise NativeConstrainedInvariantError(
                    "scalar prefix cannot terminate before this byte"
                )
            self._component_index += 1
            self._active_value = None
        self._prefix.append(value)


@dataclass(frozen=True, slots=True)
class SingleByteTokenInventory:
    """Exact printable-byte to tokenizer-ID mapping for one vocabulary."""

    vocab_size: int
    token_ids_by_byte: tuple[tuple[int, ...], ...]

    def token_ids(self, value: int) -> tuple[int, ...]:
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255:
            raise TypeError("value must be one byte expressed as an integer")
        return self.token_ids_by_byte[value]


def compile_single_byte_token_inventory(
    codec: Any,
    vocab_size: int,
) -> SingleByteTokenInventory:
    """Preflight exact token bytes and require complete printable ASCII coverage."""

    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size <= 0:
        raise NativeConstrainedCapabilityError("logits vocabulary must be a positive integer")
    token_bytes = getattr(codec, "token_bytes", None)
    if not callable(token_bytes):
        raise NativeConstrainedCapabilityError("codec does not expose exact token_bytes")
    by_byte: list[list[int]] = [[] for _ in range(256)]
    for token_id in range(vocab_size):
        try:
            raw = token_bytes(token_id)
        except Exception as exc:
            raise NativeConstrainedCapabilityError(
                f"codec cannot resolve exact bytes for vocabulary token {token_id}"
            ) from exc
        if not isinstance(raw, bytes):
            raise NativeConstrainedCapabilityError(
                f"codec token_bytes({token_id}) did not return bytes"
            )
        if len(raw) == 1:
            by_byte[raw[0]].append(token_id)
    missing = sorted(value for value in _PRINTABLE_ASCII if not by_byte[value])
    if missing:
        rendered = ", ".join(f"0x{value:02x}" for value in missing[:8])
        suffix = "..." if len(missing) > 8 else ""
        raise NativeConstrainedCapabilityError(
            "codec lacks standalone tokens for required printable ASCII bytes: "
            f"{rendered}{suffix}"
        )
    return SingleByteTokenInventory(
        vocab_size=vocab_size,
        token_ids_by_byte=tuple(tuple(token_ids) for token_ids in by_byte),
    )


def _normalized_session_tokens(session: Any) -> tuple[int, ...]:
    try:
        raw = session.token_ids
    except Exception as exc:
        raise NativeConstrainedCapabilityError(
            "session does not expose readable token_ids"
        ) from exc
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise NativeConstrainedCapabilityError("session token_ids must be a sequence")
    normalized: list[int] = []
    for token_id in raw:
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
            raise NativeConstrainedInvariantError("session token_ids contains an invalid token")
        normalized.append(token_id)
    return tuple(normalized)


def _normalized_logits(raw: Any, *, expected_size: int | None = None) -> tuple[float, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise NativeConstrainedCapabilityError(
            "session current_logits must return a non-empty numeric sequence"
        )
    logits: list[float] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise NativeConstrainedCapabilityError(
                "session current_logits returned a non-numeric value"
            )
        normalized = float(value)
        if not math.isfinite(normalized):
            raise NativeConstrainedCapabilityError(
                "session current_logits returned a non-finite value"
            )
        logits.append(normalized)
    if expected_size is not None and len(logits) != expected_size:
        raise NativeConstrainedInvariantError(
            "session logits vocabulary changed during constrained generation"
        )
    return tuple(logits)


def _parse_completed_output(text: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-JSON numeric constant {value!r}")

    return json.loads(
        text,
        parse_float=Decimal,
        parse_int=int,
        parse_constant=reject_constant,
    )


def generate_json_schema_ascii_byte_greedy(
    session: Any,
    codec: Any,
    compiled: CompiledJSONSchema,
    *,
    max_new_tokens: int,
    on_token: Callable[[GenerationEvent], None] | None = None,
) -> GenerationResult:
    """Generate one schema-constrained object using greedy allowed-token selection."""

    if not isinstance(compiled, CompiledJSONSchema):
        raise TypeError("compiled must be a CompiledJSONSchema")
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or not 1 <= max_new_tokens <= MAX_OUTPUT_BYTES
    ):
        raise ValueError(
            f"max_new_tokens must be between 1 and {MAX_OUTPUT_BYTES}"
        )
    if on_token is not None and not callable(on_token):
        raise TypeError("on_token must be callable or None")
    current_logits = getattr(session, "current_logits", None)
    prefill = getattr(session, "prefill", None)
    if not callable(current_logits) or not callable(prefill):
        raise NativeConstrainedCapabilityError(
            "session must expose current_logits and exact-prefix prefill"
        )
    initial_tokens = _normalized_session_tokens(session)
    if not initial_tokens:
        raise NativeConstrainedCapabilityError(
            "constrained generation requires a non-empty prefilled token history"
        )

    first_logits = _normalized_logits(current_logits())
    inventory = compile_single_byte_token_inventory(codec, len(first_logits))
    grammar = compiled.new_grammar()
    events: list[GenerationEvent] = []
    generated_token_ids: list[int] = []
    generated_bytes = bytearray()
    logits = first_logits

    for index in range(max_new_tokens):
        if index:
            logits = _normalized_logits(
                current_logits(),
                expected_size=inventory.vocab_size,
            )
        allowed_candidates = tuple(
            (token_id, value)
            for value in grammar.allowed_bytes()
            for token_id in inventory.token_ids(value)
        )
        if not allowed_candidates:
            raise NativeConstrainedInvariantError(
                "viable grammar state has no corresponding single-byte token"
            )
        selected_token, selected_value = min(
            allowed_candidates,
            key=lambda candidate: (-logits[candidate[0]], candidate[0]),
        )
        before = _normalized_session_tokens(session)
        synchronized = initial_tokens + tuple(generated_token_ids)
        if before != synchronized:
            raise NativeConstrainedInvariantError(
                "session token history changed outside constrained generation"
            )
        expected = before + (selected_token,)
        # Advance the local grammar before mutating resident state.  The token
        # came from allowed_candidates, so any failure here is an internal bug
        # and must not leave an invalid token committed.
        grammar.advance(selected_value)
        prefill(expected)
        after = _normalized_session_tokens(session)
        if after != expected:
            raise NativeConstrainedInvariantError(
                "exact-prefix prefill did not commit exactly the selected token"
            )
        generated_token_ids.append(selected_token)
        generated_bytes.append(selected_value)
        accepted = grammar.accepting
        event = GenerationEvent(
            token_id=selected_token,
            index=index,
            position=len(after) - 1,
            text=chr(selected_value),
            finish_reason="stop" if accepted else None,
        )
        events.append(event)
        if on_token is not None:
            on_token(event)
        if accepted:
            break

    text = generated_bytes.decode("ascii")
    if grammar.accepting:
        try:
            instance = _parse_completed_output(text)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise NativeConstrainedInvariantError(
                "accepted byte grammar did not produce parseable JSON"
            ) from exc
        compiled.validate_instance(instance)
        finish_reason = "stop"
    else:
        finish_reason = "length"
    return GenerationResult(
        token_ids=tuple(generated_token_ids),
        text=text,
        finish_reason=finish_reason,
        prompt_tokens=len(initial_tokens),
        completion_tokens=len(generated_token_ids),
        events=tuple(events),
        cancelled=False,
    )
