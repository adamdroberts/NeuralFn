"""Bounded Responses and Conversations service for native inference.

This module owns the durable OpenAI-shaped state contract without importing
FastAPI or the editor server. It supports text plus one fail-closed buffered
JSON-schema/function-item profile; all other tool and multimodal forms remain
unavailable.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import secrets
import threading
import time
from typing import Any, Callable, Mapping, Sequence

from ._native_prefix_cache import (
    NativePrefixCacheLease,
    NativePrefixCacheUsage,
)
from .native_chat import NativeChatMessage
from .native_constrained import (
    CompiledJSONSchema,
    MAX_OUTPUT_BYTES,
    NativeConstrainedSchemaError,
    compile_json_schema_ascii_byte_greedy,
    generate_json_schema_ascii_byte_greedy,
)
from .native_inference import GenerationConfig, GenerationEvent, GenerationResult
from .native_state import NativeStateConflictError, NativeStateStore


_TEXT_ROLES = frozenset({"developer", "system", "user", "assistant"})
_SUPPORTED_RESPONSE_FIELDS = frozenset(
    {
        "background",
        "conversation",
        "input",
        "instructions",
        "max_output_tokens",
        "metadata",
        "model",
        "parallel_tool_calls",
        "previous_response_id",
        "store",
        "stream",
        "temperature",
        "text",
        "tool_choice",
        "tools",
        "top_p",
        "truncation",
    }
)
_SUPPORTED_INPUT_TOKEN_FIELDS = frozenset(
    {
        "conversation",
        "input",
        "instructions",
        "model",
        "parallel_tool_calls",
        "previous_response_id",
        "text",
        "tool_choice",
        "tools",
        "truncation",
    }
)
_SUPPORTED_COMPACT_FIELDS = frozenset(
    {
        "input",
        "instructions",
        "model",
        "previous_response_id",
        "prompt_cache_key",
        "prompt_cache_options",
        "prompt_cache_retention",
        "service_tier",
    }
)
_MESSAGE_FIELDS = frozenset({"content", "id", "name", "role", "status", "type"})
_COMPACTION_ITEM_FIELDS = frozenset({"created_by", "encrypted_content", "id", "type"})
_CONTENT_FIELDS = frozenset({"annotations", "logprobs", "text", "type"})
_FUNCTION_TOOL_FIELDS = frozenset(
    {"description", "name", "parameters", "strict", "type"}
)
_FUNCTION_CALL_FIELDS = frozenset(
    {"arguments", "call_id", "id", "name", "status", "type"}
)
_FUNCTION_CALL_OUTPUT_FIELDS = frozenset(
    {"call_id", "id", "output", "status", "type"}
)
_FUNCTION_TOOL_TEMPLATE = {
    "version": 1,
    "profile": "responses-forced-function-call-v1",
}
_TERMINAL_RESPONSE_STATUSES = frozenset(
    {"cancelled", "completed", "failed", "incomplete"}
)


class NativeResponsesAPIError(Exception):
    """HTTP-shaped validation or state error raised by the service layer."""

    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        error_type: str = "invalid_request_error",
        param: str | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = int(status_code)
        self.message = str(message)
        self.error_type = str(error_type)
        self.param = param
        self.code = code

    def payload(self) -> dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        }


@dataclass(frozen=True, slots=True)
class PreparedNativeResponse:
    scope: str
    response_id: str
    output_item_id: str
    created_at: int
    prompt_token_ids: tuple[int, ...]
    generation: GenerationConfig
    input_items: tuple[dict[str, Any], ...]
    response: dict[str, Any]
    conversation_id: str | None
    store: bool
    background: bool
    stream: bool
    constrained_schema: CompiledJSONSchema | None
    output_kind: str
    function_name: str | None
    tool_call_id: str | None


@dataclass(frozen=True, slots=True)
class CompletedNativeResponse:
    native: GenerationResult
    text: str


@dataclass(slots=True)
class _ResidentResponseExecution:
    """Private foreground execution ownership retained through persistence."""

    completed: CompletedNativeResponse
    lease: NativePrefixCacheLease | None
    usage: NativePrefixCacheUsage
    consumed: bool = False


@dataclass(frozen=True, slots=True)
class _FinishedNativeResponse:
    response: dict[str, Any]
    status: str
    conversation_revision: int | None


@dataclass(frozen=True, slots=True)
class _ResponseOutputPlan:
    kind: str
    compiled_schema: CompiledJSONSchema | None
    tools: tuple[dict[str, Any], ...]
    tool_choice: str | dict[str, str]
    function_name: str | None = None


@dataclass(frozen=True, slots=True)
class _HistoryContext:
    messages: tuple[NativeChatMessage, ...]
    unresolved_calls: tuple[dict[str, Any], ...]
    all_call_ids: frozenset[str]
    resolved_call_ids: frozenset[str]
    has_tool_items: bool
    has_compaction: bool
    conversation_revision: int | None
    previous_lineage: tuple[dict[str, str], ...]


def _resource_id(prefix: str) -> str:
    suffix = secrets.token_urlsafe(18).replace("-", "").replace("_", "")
    return prefix + suffix


def _non_empty_string(value: Any, *, param: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise NativeResponsesAPIError(400, f"'{param}' must be a non-empty string.", param=param)
    return value.strip()


def _metadata(value: Any, *, param: str = "metadata") -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise NativeResponsesAPIError(400, f"'{param}' must be an object.", param=param)
    if len(value) > 16:
        raise NativeResponsesAPIError(
            400,
            f"'{param}' may contain at most 16 entries.",
            param=param,
        )
    normalized: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key or len(key) > 64:
            raise NativeResponsesAPIError(
                400,
                f"'{param}' keys must be non-empty strings of at most 64 characters.",
                param=param,
            )
        if not isinstance(item, str) or len(item) > 512:
            raise NativeResponsesAPIError(
                400,
                f"'{param}.{key}' must be a string of at most 512 characters.",
                param=f"{param}.{key}",
            )
        normalized[key] = item
    return normalized


def _public_response(payload: Mapping[str, Any]) -> dict[str, Any]:
    public = deepcopy(dict(payload))
    public.pop("_nfn", None)
    public.pop("cancel_requested", None)
    return public


def _list_payload(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    data = [deepcopy(dict(item)) for item in items]
    return {
        "object": "list",
        "data": data,
        "first_id": str(data[0].get("id") or "") if data else "",
        "last_id": str(data[-1].get("id") or "") if data else "",
        "has_more": False,
    }


def _paginated_list_payload(
    items: Sequence[Mapping[str, Any]],
    *,
    after: str | None,
    limit: int,
    order: str,
) -> dict[str, Any]:
    if after is not None:
        after = _non_empty_string(after, param="after")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise NativeResponsesAPIError(
            400,
            "'limit' must be an integer between 1 and 100.",
            param="limit",
        )
    if order not in {"asc", "desc"}:
        raise NativeResponsesAPIError(
            400,
            "'order' must be 'asc' or 'desc'.",
            param="order",
        )

    ordered = [deepcopy(dict(item)) for item in items]
    if order == "desc":
        ordered.reverse()
    if after is not None:
        try:
            cursor_index = next(
                index for index, item in enumerate(ordered) if item.get("id") == after
            )
        except StopIteration as exc:
            raise NativeResponsesAPIError(
                400,
                f"Pagination cursor {after!r} was not found.",
                param="after",
                code="invalid_cursor",
            ) from exc
        ordered = ordered[cursor_index + 1 :]

    has_more = len(ordered) > limit
    page = ordered[:limit]
    return {
        "object": "list",
        "data": page,
        "first_id": str(page[0].get("id") or "") if page else "",
        "last_id": str(page[-1].get("id") or "") if page else "",
        "has_more": has_more,
    }


def _normalize_content(raw: Any, *, param: str) -> tuple[str, list[dict[str, Any]]]:
    if isinstance(raw, str):
        return raw, [{"type": "input_text", "text": raw}]
    if not isinstance(raw, list) or not raw:
        raise NativeResponsesAPIError(
            400,
            f"'{param}' must be text or a non-empty array of text parts.",
            param=param,
        )
    text_parts: list[str] = []
    normalized: list[dict[str, Any]] = []
    for index, part in enumerate(raw):
        part_param = f"{param}.{index}"
        if not isinstance(part, Mapping):
            raise NativeResponsesAPIError(400, f"'{part_param}' must be an object.", param=part_param)
        unknown = sorted(set(part) - _CONTENT_FIELDS)
        if unknown:
            field = unknown[0]
            raise NativeResponsesAPIError(
                400,
                f"'{part_param}.{field}' is not supported by this text-only model.",
                param=f"{part_param}.{field}",
                code="unsupported_feature",
            )
        part_type = part.get("type")
        if part_type not in {"input_text", "output_text", "text"}:
            raise NativeResponsesAPIError(
                400,
                f"'{part_param}' is not a supported text content part.",
                param=part_param,
                code="unsupported_feature",
            )
        text = part.get("text")
        if not isinstance(text, str):
            raise NativeResponsesAPIError(
                400,
                f"'{part_param}.text' must be a string.",
                param=f"{part_param}.text",
            )
        text_parts.append(text)
        normalized.append({"type": "input_text", "text": text})
    return "".join(text_parts), normalized


def _normalize_message(raw: Any, *, param: str) -> tuple[dict[str, Any], NativeChatMessage]:
    if not isinstance(raw, Mapping):
        raise NativeResponsesAPIError(400, f"'{param}' must be an object.", param=param)
    unknown = sorted(set(raw) - _MESSAGE_FIELDS)
    if unknown:
        field = unknown[0]
        raise NativeResponsesAPIError(
            400,
            f"'{param}.{field}' is not supported by this text-only model.",
            param=f"{param}.{field}",
            code="unsupported_feature",
        )
    item_type = raw.get("type", "message")
    if item_type != "message":
        raise NativeResponsesAPIError(
            400,
            f"'{param}.type' {item_type!r} is not supported by this text-only model.",
            param=f"{param}.type",
            code="unsupported_feature",
        )
    role = raw.get("role")
    if role not in _TEXT_ROLES:
        raise NativeResponsesAPIError(
            400,
            f"'{param}.role' {role!r} is not supported; tool and multimodal items are unavailable.",
            param=f"{param}.role",
            code="unsupported_feature",
        )
    text, content = _normalize_content(raw.get("content"), param=f"{param}.content")
    if role == "assistant":
        content = [
            {
                "type": "output_text",
                "text": str(part["text"]),
                "annotations": [],
                "logprobs": [],
            }
            for part in content
        ]
    name = raw.get("name")
    if name is not None and (not isinstance(name, str) or not name):
        raise NativeResponsesAPIError(
            400,
            f"'{param}.name' must be a non-empty string.",
            param=f"{param}.name",
        )
    item_id = raw.get("id")
    if item_id is None:
        item_id = _resource_id("msg_")
    else:
        item_id = _non_empty_string(item_id, param=f"{param}.id")
    item: dict[str, Any] = {
        "id": item_id,
        "type": "message",
        "role": role,
        "content": content,
        "status": "completed",
    }
    if name is not None:
        item["name"] = name
    return item, NativeChatMessage(role=role, content=text, name=name)


def _invalid_stored_state(message: str) -> NativeResponsesAPIError:
    return NativeResponsesAPIError(
        500,
        message,
        error_type="server_error",
        code="invalid_state",
    )


def _message_from_stored_item(item: Mapping[str, Any]) -> NativeChatMessage:
    unknown = sorted(set(item) - _MESSAGE_FIELDS)
    if unknown:
        raise _invalid_stored_state(
            f"Stored response message field {unknown[0]!r} is invalid."
        )
    if item.get("type") != "message" or item.get("role") not in _TEXT_ROLES:
        raise _invalid_stored_state("Stored response message type or role is invalid.")
    if not isinstance(item.get("id"), str) or not item["id"]:
        raise _invalid_stored_state("Stored response message ID is invalid.")
    if item.get("status") not in {"completed", "incomplete"}:
        raise _invalid_stored_state("Stored response message status is invalid.")
    content = item.get("content")
    if not isinstance(content, list) or not content:
        raise _invalid_stored_state("Stored response message content is invalid.")
    pieces: list[str] = []
    for part in content:
        if not isinstance(part, Mapping):
            raise _invalid_stored_state("Stored response content part is invalid.")
        unknown_part = sorted(set(part) - _CONTENT_FIELDS)
        if unknown_part or part.get("type") not in {
            "input_text",
            "output_text",
            "text",
        }:
            raise _invalid_stored_state("Stored response content part is invalid.")
        text = part.get("text")
        if not isinstance(text, str):
            raise _invalid_stored_state("Stored response content text is invalid.")
        pieces.append(text)
    name = item.get("name")
    if name is not None and (not isinstance(name, str) or not name):
        raise _invalid_stored_state("Stored response message name is invalid.")
    return NativeChatMessage(
        role=str(item["role"]),
        content="".join(pieces),
        name=name,
    )


def _stored_function_call(item: Mapping[str, Any]) -> dict[str, Any]:
    if set(item) != _FUNCTION_CALL_FIELDS:
        raise _invalid_stored_state("Stored function_call fields are invalid.")
    if item.get("type") != "function_call":
        raise _invalid_stored_state("Stored function_call type is invalid.")
    for field in ("id", "call_id", "name"):
        if not isinstance(item.get(field), str) or not item[field]:
            raise _invalid_stored_state(f"Stored function_call {field} is invalid.")
    arguments = item.get("arguments")
    if not isinstance(arguments, str):
        raise _invalid_stored_state("Stored function_call arguments are invalid.")
    status = item.get("status")
    if status not in {"completed", "incomplete"}:
        raise _invalid_stored_state("Stored function_call status is invalid.")
    if status == "completed":
        def reject_constant(value: str) -> None:
            raise ValueError(f"non-JSON constant {value!r}")

        try:
            parsed = json.loads(
                arguments,
                parse_constant=reject_constant,
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise _invalid_stored_state(
                "Stored completed function_call arguments are not valid JSON."
            ) from exc
        if not isinstance(parsed, Mapping):
            raise _invalid_stored_state(
                "Stored completed function_call arguments are not a JSON object."
            )
    return deepcopy(dict(item))


def _stored_function_call_output(item: Mapping[str, Any]) -> dict[str, Any]:
    if set(item) != _FUNCTION_CALL_OUTPUT_FIELDS:
        raise _invalid_stored_state("Stored function_call_output fields are invalid.")
    if item.get("type") != "function_call_output":
        raise _invalid_stored_state("Stored function_call_output type is invalid.")
    for field in ("id", "call_id"):
        if not isinstance(item.get(field), str) or not item[field]:
            raise _invalid_stored_state(
                f"Stored function_call_output {field} is invalid."
            )
    if item.get("status") != "completed" or not isinstance(item.get("output"), str):
        raise _invalid_stored_state("Stored function_call_output payload is invalid.")
    return deepcopy(dict(item))


def _function_call_history_message(item: Mapping[str, Any]) -> NativeChatMessage:
    content = (
        "Client-executed function call "
        + json.dumps(str(item["name"]), ensure_ascii=True)
        + " with call ID "
        + json.dumps(str(item["call_id"]), ensure_ascii=True)
        + " and arguments "
        + str(item["arguments"])
    )
    return NativeChatMessage(role="assistant", content=content)


def _function_output_history_message(
    item: Mapping[str, Any],
    *,
    function_name: str,
) -> NativeChatMessage:
    content = (
        "Client result for call ID "
        + json.dumps(str(item["call_id"]), ensure_ascii=True)
        + ": "
        + str(item["output"])
    )
    return NativeChatMessage(
        role="tool",
        content=content,
        name=function_name,
        tool_call_id=str(item["call_id"]),
    )


def _stored_message(message: NativeChatMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.name is not None:
        payload["name"] = message.name
    return payload


class NativeResponsesService:
    """Bounded text and buffered constrained Responses over one resident runtime."""

    def __init__(self, runtime: Any, state_store: NativeStateStore) -> None:
        self.runtime = runtime
        self.state = state_store
        # Durable terminal transitions/cache publication and destructive
        # mutations share one linearization boundary.  The cache manager's
        # per-scope epoch supplies the second fence for leases already running
        # when a deletion wins this lock.
        state_lock = getattr(state_store, "_responses_transition_lock", None)
        runtime_lock = getattr(runtime, "_responses_transition_lock", None)
        self._transition_lock = (
            state_lock
            if state_lock is not None
            else runtime_lock if runtime_lock is not None else threading.RLock()
        )

    @property
    def _prefix_cache(self) -> Any | None:
        return getattr(self.runtime, "_prefix_cache", None)

    @staticmethod
    def _internal_request_state(
        prepared: PreparedNativeResponse,
    ) -> Mapping[str, Any]:
        internal = prepared.response.get("_nfn")
        if not isinstance(internal, Mapping):
            raise RuntimeError("Prepared response is missing native request state")
        return internal

    def _previous_response_id(
        self,
        prepared: PreparedNativeResponse,
    ) -> str | None:
        raw = self._internal_request_state(prepared).get("previous_response_id")
        if raw is None:
            return None
        if not isinstance(raw, str) or not raw:
            raise RuntimeError("Prepared previous response ID is invalid")
        return raw

    def _conversation_revision(
        self,
        prepared: PreparedNativeResponse,
    ) -> int | None:
        raw = self._internal_request_state(prepared).get("conversation_revision")
        if prepared.conversation_id is None:
            if raw is not None:
                raise RuntimeError(
                    "Prepared response has a conversation revision without a conversation"
                )
            return None
        if (
            isinstance(raw, bool)
            or not isinstance(raw, int)
            or raw < 0
            or raw > 9_223_372_036_854_775_806
        ):
            raise RuntimeError("Prepared conversation revision is invalid")
        return raw

    def _previous_lineage_signature(
        self,
        prepared: PreparedNativeResponse,
    ) -> tuple[tuple[str, str], ...]:
        raw = self._internal_request_state(prepared).get("previous_lineage")
        if not isinstance(raw, list):
            raise RuntimeError("Prepared previous-response lineage is invalid")
        signature: list[tuple[str, str]] = []
        for item in raw:
            if not isinstance(item, Mapping) or set(item) != {"id", "status"}:
                raise RuntimeError("Prepared previous-response lineage is invalid")
            response_id = item.get("id")
            status = item.get("status")
            if (
                not isinstance(response_id, str)
                or not response_id
                or status not in {"completed", "incomplete"}
            ):
                raise RuntimeError("Prepared previous-response lineage is invalid")
            signature.append((response_id, status))
        previous_response_id = self._previous_response_id(prepared)
        if (previous_response_id is None) != (not signature):
            raise RuntimeError("Prepared previous-response lineage is inconsistent")
        if previous_response_id is not None and signature[-1][0] != previous_response_id:
            raise RuntimeError("Prepared previous-response lineage tail is inconsistent")
        return tuple(signature)

    def _validate_previous_lineage_locked(
        self,
        prepared: PreparedNativeResponse,
    ) -> None:
        expected = self._previous_lineage_signature(prepared)
        if not expected:
            return
        previous_response_id = self._previous_response_id(prepared)
        assert previous_response_id is not None
        try:
            current = self.state.response_lineage(
                prepared.scope,
                previous_response_id,
            )
        except KeyError as exc:
            raise NativeStateConflictError(
                "Previous response lineage changed after request preparation",
                code="response_lineage_conflict",
            ) from exc
        actual = tuple(
            (str(response.get("id") or ""), str(response.get("status") or ""))
            for response in current
        )
        if actual != expected:
            raise NativeStateConflictError(
                "Previous response lineage changed after request preparation",
                code="response_lineage_conflict",
            )

    def _purge_scope_locked(self, scope: str) -> None:
        cache = self._prefix_cache
        if cache is not None:
            cache.purge_scope(scope)

    def _require_model(self, payload: Mapping[str, Any]) -> str:
        model = payload.get("model")
        if not isinstance(model, str) or not model:
            raise NativeResponsesAPIError(400, "'model' is required.", param="model", code="invalid_model")
        if model != self.runtime.served_model_name:
            raise NativeResponsesAPIError(
                404,
                f"The model {model!r} does not exist.",
                param="model",
                code="model_not_found",
            )
        return model

    def _runtime_capability(self, name: str) -> bool:
        capabilities = getattr(self.runtime, "capabilities", None)
        if isinstance(capabilities, Mapping):
            return capabilities.get(name) is True
        return getattr(capabilities, name, False) is True

    def _has_exact_function_tool_template(self) -> bool:
        manifest = getattr(self.runtime, "manifest", None)
        chat_template = (
            manifest.get("chat_template") if isinstance(manifest, Mapping) else None
        )
        tool_template = (
            chat_template.get("tool_template")
            if isinstance(chat_template, Mapping)
            else None
        )
        return bool(
            isinstance(tool_template, Mapping)
            and set(tool_template) == set(_FUNCTION_TOOL_TEMPLATE)
            and type(tool_template.get("version")) is int
            and tool_template.get("version") == 1
            and tool_template.get("profile")
            == _FUNCTION_TOOL_TEMPLATE["profile"]
        )

    @staticmethod
    def _compile_schema(
        text_format: Mapping[str, Any],
        *,
        param: str,
    ) -> CompiledJSONSchema:
        try:
            return compile_json_schema_ascii_byte_greedy(text_format)
        except NativeConstrainedSchemaError as exc:
            raise NativeResponsesAPIError(
                400,
                str(exc),
                param=param,
                code="invalid_json_schema",
            ) from exc

    def _normalize_function_tool(
        self,
        raw: Any,
        *,
        param: str,
    ) -> tuple[dict[str, Any], CompiledJSONSchema]:
        if not isinstance(raw, Mapping):
            raise NativeResponsesAPIError(400, f"'{param}' must be an object.", param=param)
        unknown = sorted(set(raw) - _FUNCTION_TOOL_FIELDS)
        if unknown:
            field = unknown[0]
            raise NativeResponsesAPIError(
                400,
                f"'{param}.{field}' is not supported for a function tool.",
                param=f"{param}.{field}",
                code="unsupported_feature",
            )
        if raw.get("type") != "function":
            raise NativeResponsesAPIError(
                400,
                "Only flat function tools are supported by this resident model.",
                param=f"{param}.type",
                code="unsupported_feature",
            )
        if raw.get("strict") is not True:
            raise NativeResponsesAPIError(
                400,
                "Resident function tools require 'strict: true'.",
                param=f"{param}.strict",
                code="unsupported_feature",
            )
        description = raw.get("description")
        if description is not None and not isinstance(description, str):
            raise NativeResponsesAPIError(
                400,
                f"'{param}.description' must be a string or null.",
                param=f"{param}.description",
            )
        wrapped_format = {
            "type": "json_schema",
            "name": raw.get("name"),
            "schema": raw.get("parameters"),
            "strict": True,
        }
        compiled = self._compile_schema(
            wrapped_format,
            param=f"{param}.parameters",
        )
        canonical: dict[str, Any] = {
            "type": "function",
            "name": compiled.name,
            "parameters": json.loads(compiled.canonical_schema_json),
            "strict": True,
        }
        if "description" in raw:
            canonical["description"] = description
        return canonical, compiled

    def _validate_capability_fields(
        self,
        payload: Mapping[str, Any],
        *,
        supported_fields: frozenset[str],
        allow_constrained: bool = False,
    ) -> _ResponseOutputPlan:
        unknown = sorted(set(payload) - supported_fields)
        if unknown:
            field = unknown[0]
            raise NativeResponsesAPIError(
                400,
                f"Responses field {field!r} is not supported by this resident model.",
                param=field,
                code="unsupported_feature",
            )
        parallel = payload.get("parallel_tool_calls", False)
        if not isinstance(parallel, bool):
            raise NativeResponsesAPIError(
                400,
                "'parallel_tool_calls' must be a boolean.",
                param="parallel_tool_calls",
            )
        if parallel:
            raise NativeResponsesAPIError(
                400,
                "Parallel tool calls are not supported by this resident text model.",
                param="parallel_tool_calls",
                code="unsupported_feature",
            )

        text = payload.get("text")
        compiled_text: CompiledJSONSchema | None = None
        if text is not None:
            if not isinstance(text, Mapping):
                raise NativeResponsesAPIError(400, "'text' must be an object.", param="text")
            unknown_text = sorted(set(text) - {"format"})
            if unknown_text:
                field = unknown_text[0]
                raise NativeResponsesAPIError(
                    400,
                    f"'text.{field}' is not supported by this resident model.",
                    param=f"text.{field}",
                    code="unsupported_feature",
                )
            text_format = text.get("format")
            if text_format is not None:
                if not isinstance(text_format, Mapping):
                    raise NativeResponsesAPIError(
                        400,
                        "'text.format' must be an object.",
                        param="text.format",
                    )
                if text_format.get("type") == "text" and set(text_format) == {"type"}:
                    pass
                elif text_format.get("type") == "json_schema":
                    if not allow_constrained or not self._runtime_capability(
                        "structured_output"
                    ):
                        raise NativeResponsesAPIError(
                            400,
                            "Structured output is not supported by this resident text model.",
                            param="text.format",
                            code="unsupported_feature",
                        )
                    compiled_text = self._compile_schema(
                        text_format,
                        param="text.format",
                    )
                else:
                    raise NativeResponsesAPIError(
                        400,
                        "Structured output is not supported by this resident text model.",
                        param="text.format",
                        code="unsupported_feature",
                    )

        tools = payload.get("tools")
        if tools is None:
            tools = []
        if not isinstance(tools, list):
            raise NativeResponsesAPIError(400, "'tools' must be an array.", param="tools")
        tool_choice = payload.get("tool_choice")
        if not tools:
            if tool_choice not in (None, "none"):
                raise NativeResponsesAPIError(
                    400,
                    "Tool selection is unavailable without a supported function tool.",
                    param="tool_choice",
                    code="unsupported_feature",
                )
            if compiled_text is not None:
                return _ResponseOutputPlan(
                    kind="json_schema",
                    compiled_schema=compiled_text,
                    tools=(),
                    tool_choice="none",
                )
            return _ResponseOutputPlan(
                kind="text",
                compiled_schema=None,
                tools=(),
                tool_choice="none",
            )

        if compiled_text is not None:
            raise NativeResponsesAPIError(
                400,
                "Function tools and structured text output cannot be requested together.",
                param="text.format",
                code="unsupported_feature",
            )
        if not allow_constrained or not self._runtime_capability("function_tools"):
            raise NativeResponsesAPIError(
                400,
                "Function and hosted tools are not supported by this resident text model.",
                param="tools",
                code="unsupported_feature",
            )
        if not self._has_exact_function_tool_template():
            raise NativeResponsesAPIError(
                400,
                "The resident artifact does not declare the required function-tool template.",
                param="tools",
                code="unsupported_feature",
            )
        if len(tools) != 1:
            raise NativeResponsesAPIError(
                400,
                "Exactly one forced function tool is supported.",
                param="tools",
                code="unsupported_feature",
            )
        canonical_tool, compiled_tool = self._normalize_function_tool(
            tools[0],
            param="tools.0",
        )
        if not isinstance(tool_choice, Mapping):
            raise NativeResponsesAPIError(
                400,
                "A non-empty tool list requires one forced function 'tool_choice'.",
                param="tool_choice",
                code="unsupported_feature",
            )
        if set(tool_choice) != {"type", "name"}:
            raise NativeResponsesAPIError(
                400,
                "Forced function 'tool_choice' must contain exactly 'type' and 'name'.",
                param="tool_choice",
                code="unsupported_feature",
            )
        if (
            tool_choice.get("type") != "function"
            or tool_choice.get("name") != canonical_tool["name"]
        ):
            raise NativeResponsesAPIError(
                400,
                "Forced function 'tool_choice' must select the only supplied function.",
                param="tool_choice",
                code="unsupported_feature",
            )
        canonical_choice = {
            "type": "function",
            "name": str(canonical_tool["name"]),
        }
        return _ResponseOutputPlan(
            kind="function_call",
            compiled_schema=compiled_tool,
            tools=(canonical_tool,),
            tool_choice=canonical_choice,
            function_name=str(canonical_tool["name"]),
        )

    def _conversation_id(self, raw: Any) -> str | None:
        if raw is None:
            return None
        if isinstance(raw, str):
            return _non_empty_string(raw, param="conversation")
        if isinstance(raw, Mapping) and set(raw) == {"id"}:
            return _non_empty_string(raw.get("id"), param="conversation.id")
        raise NativeResponsesAPIError(
            400,
            "'conversation' must be a conversation ID or an object containing only 'id'.",
            param="conversation",
        )

    def _compaction_messages(
        self,
        scope: str,
        encrypted_content: Any,
        *,
        param: str,
    ) -> tuple[NativeChatMessage, ...]:
        token = _non_empty_string(encrypted_content, param=param)
        compacted = self.state.get_response_compaction(scope, token)
        if compacted is None:
            raise NativeResponsesAPIError(
                404,
                "The compacted response context was not found in this API-key scope.",
                param=param,
                code="compaction_not_found",
            )
        raw_messages = compacted.get("messages")
        if not isinstance(raw_messages, list):
            raise NativeResponsesAPIError(
                500,
                "Stored compacted response context is invalid.",
                error_type="server_error",
                code="invalid_state",
            )
        messages: list[NativeChatMessage] = []
        for raw in raw_messages:
            if not isinstance(raw, Mapping):
                raise NativeResponsesAPIError(
                    500,
                    "Stored compacted response context is invalid.",
                    error_type="server_error",
                    code="invalid_state",
                )
            role = raw.get("role")
            content = raw.get("content")
            name = raw.get("name")
            if (
                role not in _TEXT_ROLES
                or not isinstance(content, str)
                or (name is not None and (not isinstance(name, str) or not name))
            ):
                raise NativeResponsesAPIError(
                    500,
                    "Stored compacted response context is invalid.",
                    error_type="server_error",
                    code="invalid_state",
                )
            messages.append(NativeChatMessage(role=role, content=content, name=name))
        return tuple(messages)

    def _normalize_compaction_item(
        self,
        scope: str,
        raw: Mapping[str, Any],
        *,
        param: str,
    ) -> tuple[dict[str, Any], tuple[NativeChatMessage, ...]]:
        unknown = sorted(set(raw) - _COMPACTION_ITEM_FIELDS)
        if unknown:
            field = unknown[0]
            raise NativeResponsesAPIError(
                400,
                f"'{param}.{field}' is not supported for a compaction item.",
                param=f"{param}.{field}",
                code="unsupported_feature",
            )
        if raw.get("type") != "compaction":
            raise NativeResponsesAPIError(
                400,
                f"'{param}.type' must be 'compaction'.",
                param=f"{param}.type",
            )
        token = _non_empty_string(
            raw.get("encrypted_content"),
            param=f"{param}.encrypted_content",
        )
        if len(token) > 255:
            raise NativeResponsesAPIError(
                400,
                f"'{param}.encrypted_content' must not exceed 255 characters.",
                param=f"{param}.encrypted_content",
                code="invalid_compaction_input",
            )
        item_id = raw.get("id")
        if item_id is None:
            item_id = _resource_id("cmp_")
        else:
            item_id = _non_empty_string(item_id, param=f"{param}.id")
        created_by = raw.get("created_by")
        if created_by is not None:
            created_by = _non_empty_string(created_by, param=f"{param}.created_by")
        item = {
            "id": item_id,
            "type": "compaction",
            "encrypted_content": token,
            "created_by": created_by,
        }
        return item, self._compaction_messages(
            scope,
            token,
            param=f"{param}.encrypted_content",
        )

    def _normalize_response_input(
        self,
        scope: str,
        raw: Any,
        *,
        history: _HistoryContext | None = None,
        allow_function_output: bool = False,
    ) -> tuple[
        tuple[dict[str, Any], ...],
        tuple[NativeChatMessage, ...],
        bool,
    ]:
        context = history or _HistoryContext(
            (),
            (),
            frozenset(),
            frozenset(),
            False,
            False,
            None,
            (),
        )
        if raw is None:
            if context.unresolved_calls and allow_function_output:
                raise NativeResponsesAPIError(
                    400,
                    "The previous response has an unresolved function call; submit its output.",
                    param="input",
                    code="function_call_output_required",
                )
            return (), (), False
        values: list[Any]
        if isinstance(raw, str):
            values = [{"role": "user", "content": raw}]
        elif isinstance(raw, list):
            values = raw
        else:
            raise NativeResponsesAPIError(
                400,
                "'input' must be text or an array of text message or compaction items.",
                param="input",
            )

        function_outputs = [
            (index, value)
            for index, value in enumerate(values)
            if isinstance(value, Mapping)
            and value.get("type") == "function_call_output"
        ]
        if function_outputs:
            if not allow_function_output:
                raise NativeResponsesAPIError(
                    400,
                    "Function call output items are not supported by this operation.",
                    param=f"input.{function_outputs[0][0]}",
                    code="unsupported_feature",
                )
            if len(values) != 1 or len(function_outputs) != 1:
                raise NativeResponsesAPIError(
                    400,
                    "A function_call_output must be the only item in 'input'.",
                    param="input",
                    code="unsupported_feature",
                )
            index, value = function_outputs[0]
            param = f"input.{index}"
            if set(value) != {"type", "call_id", "output"}:
                raise NativeResponsesAPIError(
                    400,
                    f"'{param}' must contain exactly 'type', 'call_id', and 'output'.",
                    param=param,
                    code="unsupported_feature",
                )
            call_id = _non_empty_string(
                value.get("call_id"),
                param=f"{param}.call_id",
            )
            output = value.get("output")
            if not isinstance(output, str):
                raise NativeResponsesAPIError(
                    400,
                    f"'{param}.output' must be a string.",
                    param=f"{param}.output",
                )
            if call_id in context.resolved_call_ids:
                raise NativeResponsesAPIError(
                    400,
                    f"Function call {call_id!r} already has an output.",
                    param=f"{param}.call_id",
                    code="function_call_already_resolved",
                )
            unresolved = {str(call["call_id"]): call for call in context.unresolved_calls}
            call = unresolved.get(call_id)
            if call is None:
                message = (
                    f"Function call {call_id!r} is not a completed unresolved call "
                    "in the previous-response lineage."
                    if call_id in context.all_call_ids
                    else f"Function call {call_id!r} is not visible in the previous-response lineage."
                )
                raise NativeResponsesAPIError(
                    400,
                    message,
                    param=f"{param}.call_id",
                    code="function_call_not_found",
                )
            if len(unresolved) != 1:
                raise NativeResponsesAPIError(
                    400,
                    "The previous-response lineage must expose exactly one unresolved function call.",
                    param=f"{param}.call_id",
                    code="ambiguous_function_call",
                )
            item = {
                "id": _resource_id("fco_"),
                "type": "function_call_output",
                "call_id": call_id,
                "output": output,
                "status": "completed",
            }
            message = _function_output_history_message(
                item,
                function_name=str(call["name"]),
            )
            return (item,), (message,), True

        if context.unresolved_calls and allow_function_output:
            raise NativeResponsesAPIError(
                400,
                "The previous response has an unresolved function call; submit its output as the only input item.",
                param="input",
                code="function_call_output_required",
            )

        items: list[dict[str, Any]] = []
        ordinary_messages: list[NativeChatMessage | None] = []
        compaction_index: int | None = None
        compacted_messages: tuple[NativeChatMessage, ...] = ()
        for index, value in enumerate(values):
            param = f"input.{index}"
            if isinstance(value, Mapping) and value.get("type") == "compaction":
                if compaction_index is not None:
                    raise NativeResponsesAPIError(
                        400,
                        "Only one compaction item may appear in a response input.",
                        param=param,
                    )
                item, compacted_messages = self._normalize_compaction_item(
                    scope,
                    value,
                    param=param,
                )
                compaction_index = index
                ordinary_messages.append(None)
            else:
                item, message = _normalize_message(value, param=param)
                ordinary_messages.append(message)
            items.append(item)

        if compaction_index is None:
            messages = [message for message in ordinary_messages if message is not None]
        else:
            retained_users: list[NativeChatMessage] = []
            for index, message in enumerate(ordinary_messages[:compaction_index]):
                if message is not None and message.role != "user":
                    raise NativeResponsesAPIError(
                        400,
                        "Only retained user messages may precede a compaction item.",
                        param=f"input.{index}",
                    )
                if message is not None:
                    retained_users.append(message)
            expected_users = [
                message for message in compacted_messages if message.role == "user"
            ]
            if retained_users and retained_users != expected_users:
                raise NativeResponsesAPIError(
                    400,
                    "Messages before a compaction item must exactly match its retained user messages.",
                    param="input",
                    code="invalid_compaction_input",
                )
            messages = list(compacted_messages)
            messages.extend(
                message
                for message in ordinary_messages[compaction_index + 1 :]
                if message is not None
            )
        if compaction_index is not None and context.has_tool_items:
            raise NativeResponsesAPIError(
                400,
                "Response compaction is not supported for function-tool history.",
                param="input",
                code="unsupported_feature",
            )
        return tuple(items), tuple(messages), False

    def _history_from_items(
        self,
        scope: str,
        items: Sequence[Mapping[str, Any]],
        *,
        calls_by_id: dict[str, dict[str, Any]],
        resolved_call_ids: set[str],
        allow_tool_items: bool,
    ) -> tuple[tuple[NativeChatMessage, ...], bool, bool]:
        compaction_index: int | None = None
        compacted_messages: tuple[NativeChatMessage, ...] = ()
        has_tool_items = False
        for index, item in enumerate(items):
            if not isinstance(item, Mapping):
                raise _invalid_stored_state("Stored response item is not an object.")
            item_type = item.get("type")
            if item_type == "compaction":
                if compaction_index is not None:
                    raise _invalid_stored_state(
                        "Stored response contains multiple compaction items."
                    )
                if set(item) != _COMPACTION_ITEM_FIELDS:
                    raise _invalid_stored_state("Stored compaction item fields are invalid.")
                if not isinstance(item.get("id"), str) or not item["id"]:
                    raise _invalid_stored_state("Stored compaction item ID is invalid.")
                created_by = item.get("created_by")
                if created_by is not None and (
                    not isinstance(created_by, str) or not created_by
                ):
                    raise _invalid_stored_state(
                        "Stored compaction item created_by is invalid."
                    )
                compaction_index = index
                compacted_messages = self._compaction_messages(
                    scope,
                    item.get("encrypted_content"),
                    param="input.encrypted_content",
                )
            elif item_type in {"function_call", "function_call_output"}:
                has_tool_items = True
                if not allow_tool_items:
                    raise NativeResponsesAPIError(
                        400,
                        "Function-tool history is not supported by this operation.",
                        code="unsupported_feature",
                    )
            elif item_type != "message":
                raise _invalid_stored_state(
                    f"Stored response item type {item_type!r} is invalid."
                )
            else:
                # Validate even retained messages that a following compaction
                # supersedes; corrupt durable state must never be silently skipped.
                _message_from_stored_item(item)
        if compaction_index is not None and (
            has_tool_items or calls_by_id or resolved_call_ids
        ):
            raise NativeResponsesAPIError(
                400,
                "Response compaction is not supported for function-tool history.",
                code="unsupported_feature",
            )
        start = (compaction_index + 1) if compaction_index is not None else 0
        messages = list(compacted_messages)
        for item in items[start:]:
            item_type = item.get("type")
            if item_type == "message":
                messages.append(_message_from_stored_item(item))
            elif item_type == "function_call":
                call = _stored_function_call(item)
                call_id = str(call["call_id"])
                if call_id in calls_by_id:
                    raise _invalid_stored_state(
                        f"Stored function call ID {call_id!r} is duplicated."
                    )
                calls_by_id[call_id] = call
                messages.append(_function_call_history_message(call))
            elif item_type == "function_call_output":
                output = _stored_function_call_output(item)
                call_id = str(output["call_id"])
                call = calls_by_id.get(call_id)
                if call is None or call.get("status") != "completed":
                    raise _invalid_stored_state(
                        f"Stored function output {call_id!r} has no completed call."
                    )
                if call_id in resolved_call_ids:
                    raise _invalid_stored_state(
                        f"Stored function output {call_id!r} is duplicated."
                    )
                resolved_call_ids.add(call_id)
                messages.append(
                    _function_output_history_message(
                        output,
                        function_name=str(call["name"]),
                    )
                )
        return tuple(messages), has_tool_items, compaction_index is not None

    def _history_messages(
        self,
        scope: str,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        allow_tool_items: bool = False,
    ) -> _HistoryContext:
        history: list[NativeChatMessage] = []
        calls_by_id: dict[str, dict[str, Any]] = {}
        resolved_call_ids: set[str] = set()
        has_tool_items = False
        has_compaction = False
        conversation_revision: int | None = None
        previous_lineage: list[dict[str, str]] = []
        if previous_response_id is not None:
            previous = self.state.get_response(scope, previous_response_id)
            if previous is None:
                raise NativeResponsesAPIError(
                    404,
                    f"Response {previous_response_id!r} was not found.",
                    param="previous_response_id",
                    code="response_not_found",
                )
            if previous.get("status") not in {"completed", "incomplete"}:
                raise NativeResponsesAPIError(
                    400,
                    "'previous_response_id' must refer to a completed or incomplete response.",
                    param="previous_response_id",
                    code="previous_response_not_ready",
                )
            try:
                lineage = self.state.response_lineage(scope, previous_response_id)
            except KeyError as exc:
                raise NativeResponsesAPIError(
                    404,
                    "The previous response lineage is unavailable in this API-key scope.",
                    param="previous_response_id",
                    code="response_not_found",
                ) from exc
            for response in lineage:
                lineage_id = str(response["id"])
                lineage_status = str(response.get("status") or "")
                previous_lineage.append(
                    {"id": lineage_id, "status": lineage_status}
                )
                chunk, chunk_tools, chunk_compaction = self._history_from_items(
                    scope,
                    self.state.list_response_items(scope, lineage_id),
                    calls_by_id=calls_by_id,
                    resolved_call_ids=resolved_call_ids,
                    allow_tool_items=allow_tool_items,
                )
                history.extend(chunk)
                has_tool_items = has_tool_items or chunk_tools
                has_compaction = has_compaction or chunk_compaction
        if conversation_id is not None:
            try:
                conversation_items, conversation_revision = (
                    self.state.conversation_items_snapshot(scope, conversation_id)
                )
            except KeyError:
                raise NativeResponsesAPIError(
                    404,
                    f"Conversation {conversation_id!r} was not found.",
                    param="conversation",
                    code="conversation_not_found",
                ) from None
            chunk, chunk_tools, chunk_compaction = self._history_from_items(
                scope,
                conversation_items,
                calls_by_id=calls_by_id,
                resolved_call_ids=resolved_call_ids,
                allow_tool_items=False,
            )
            history.extend(chunk)
            has_tool_items = has_tool_items or chunk_tools
            has_compaction = has_compaction or chunk_compaction
        unresolved_calls = tuple(
            call
            for call_id, call in calls_by_id.items()
            if call.get("status") == "completed" and call_id not in resolved_call_ids
        )
        return _HistoryContext(
            messages=tuple(history),
            unresolved_calls=unresolved_calls,
            all_call_ids=frozenset(calls_by_id),
            resolved_call_ids=frozenset(resolved_call_ids),
            has_tool_items=has_tool_items,
            has_compaction=has_compaction,
            conversation_revision=conversation_revision,
            previous_lineage=tuple(previous_lineage),
        )

    def _render_prompt(
        self,
        messages: Sequence[NativeChatMessage],
        *,
        preserve_tail: int,
        reserve_output_tokens: int,
        truncation: str,
    ) -> tuple[int, ...]:
        retained = list(messages)

        def encode() -> tuple[int, ...]:
            try:
                return tuple(self.runtime.codec.encode(self.runtime.renderer.render(retained)))
            except Exception as exc:
                raise NativeResponsesAPIError(
                    400,
                    f"Unable to tokenize rendered response input: {exc}",
                    param="input",
                    code="invalid_prompt",
                ) from exc

        token_ids = encode()
        if len(token_ids) + reserve_output_tokens <= self.runtime.context_limit:
            return token_ids
        if truncation != "auto":
            raise NativeResponsesAPIError(
                400,
                "This model's maximum context length is "
                f"{self.runtime.context_limit} tokens, but the request uses "
                f"{len(token_ids)} input tokens plus {reserve_output_tokens} reserved output tokens.",
                param="input",
                code="context_length_exceeded",
            )
        fixed_prefix = 1 if retained and retained[0].role in {"developer", "system"} else 0
        required_tail = max(1, preserve_tail)
        while len(retained) > fixed_prefix + required_tail:
            del retained[fixed_prefix]
            token_ids = encode()
            if len(token_ids) + reserve_output_tokens <= self.runtime.context_limit:
                return token_ids
        raise NativeResponsesAPIError(
            400,
            "The newest response input cannot fit the model context after automatic truncation.",
            param="input",
            code="context_length_exceeded",
        )

    @staticmethod
    def _constrained_developer_prompt(plan: _ResponseOutputPlan) -> str | None:
        compiled = plan.compiled_schema
        if compiled is None:
            return None
        if plan.kind == "function_call":
            canonical_tool = json.dumps(
                plan.tools[0],
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            return (
                "Produce exactly one client-executed Responses function call. "
                "The server must not execute the function. Output only the JSON object "
                "for the function arguments, with no prose or Markdown. Canonical tool: "
                + canonical_tool
            )
        if plan.kind == "json_schema":
            return (
                "Output exactly one JSON object matching the requested strict Responses "
                "text.format schema, with no prose or Markdown. Schema name: "
                + json.dumps(compiled.name, ensure_ascii=True)
                + ". Strict schema: "
                + compiled.canonical_schema_json
            )
        raise AssertionError(f"unexpected constrained output kind {plan.kind!r}")

    def _common_input(
        self,
        scope: str,
        payload: Mapping[str, Any],
        *,
        supported_fields: frozenset[str],
        reserve_output_tokens: int,
        allow_constrained: bool = False,
    ) -> tuple[
        str,
        tuple[dict[str, Any], ...],
        tuple[int, ...],
        str | None,
        str | None,
        str | None,
        str,
        _ResponseOutputPlan,
        bool,
        int | None,
        tuple[dict[str, str], ...],
    ]:
        output_plan = self._validate_capability_fields(
            payload,
            supported_fields=supported_fields,
            allow_constrained=allow_constrained,
        )
        model = self._require_model(payload)
        previous_response_id = payload.get("previous_response_id")
        if previous_response_id is not None:
            previous_response_id = _non_empty_string(
                previous_response_id,
                param="previous_response_id",
            )
        conversation_id = self._conversation_id(payload.get("conversation"))
        if previous_response_id is not None and conversation_id is not None:
            raise NativeResponsesAPIError(
                400,
                "'previous_response_id' and 'conversation' cannot be used together.",
                param="previous_response_id",
            )
        instructions = payload.get("instructions")
        if instructions is not None and not isinstance(instructions, str):
            raise NativeResponsesAPIError(
                400,
                "'instructions' must be a string for this text-only server.",
                param="instructions",
                code="unsupported_feature",
            )
        history = self._history_messages(
            scope,
            previous_response_id=previous_response_id,
            conversation_id=conversation_id,
            allow_tool_items=allow_constrained,
        )
        if history.conversation_revision == 9_223_372_036_854_775_807:
            raise NativeResponsesAPIError(
                409,
                "Conversation item revision space is exhausted.",
                param="conversation",
                error_type="conflict_error",
                code="conversation_revision_exhausted",
            )
        input_items, input_messages, has_function_output = self._normalize_response_input(
            scope,
            payload.get("input"),
            history=history,
            allow_function_output=allow_constrained,
        )
        if output_plan.kind == "function_call" and conversation_id is not None:
            raise NativeResponsesAPIError(
                400,
                "Function tools are not supported with Conversations.",
                param="conversation",
                code="unsupported_feature",
            )
        if output_plan.kind != "text" and has_function_output:
            raise NativeResponsesAPIError(
                400,
                "A function_call_output continuation cannot request another constrained output.",
                param="input",
                code="unsupported_feature",
            )
        if output_plan.kind == "function_call" and (
            history.has_compaction
            or any(item.get("type") == "compaction" for item in input_items)
        ):
            raise NativeResponsesAPIError(
                400,
                "Function tools are not supported with response compaction.",
                param="input",
                code="unsupported_feature",
            )
        messages: list[NativeChatMessage] = []
        developer_parts: list[str] = []
        if instructions:
            developer_parts.append(instructions)
        constrained_prompt = self._constrained_developer_prompt(output_plan)
        if constrained_prompt is not None:
            developer_parts.append(constrained_prompt)
        if developer_parts:
            messages.append(
                NativeChatMessage(
                    role="developer",
                    content="\n\n".join(developer_parts),
                )
            )
        messages.extend(history.messages)
        messages.extend(input_messages)
        if not messages:
            raise NativeResponsesAPIError(
                400,
                "'input' is required when no previous response or conversation supplies context.",
                param="input",
                code="invalid_input",
            )
        truncation = payload.get("truncation", "disabled")
        if truncation not in {"auto", "disabled"}:
            raise NativeResponsesAPIError(
                400,
                "'truncation' must be 'auto' or 'disabled'.",
                param="truncation",
            )
        has_tool_context = history.has_tool_items or has_function_output
        if (output_plan.kind == "function_call" or has_tool_context) and truncation != "disabled":
            raise NativeResponsesAPIError(
                400,
                "Automatic truncation is not supported for function-tool requests or history.",
                param="truncation",
                code="unsupported_feature",
            )
        prompt_token_ids = self._render_prompt(
            messages,
            preserve_tail=len(input_messages),
            reserve_output_tokens=reserve_output_tokens,
            truncation=truncation,
        )
        return (
            model,
            input_items,
            prompt_token_ids,
            instructions,
            previous_response_id,
            conversation_id,
            truncation,
            output_plan,
            has_tool_context,
            history.conversation_revision,
            history.previous_lineage,
        )

    def prepare(self, scope: str, payload: Mapping[str, Any]) -> PreparedNativeResponse:
        raw_text = payload.get("text")
        raw_text_format = (
            raw_text.get("format") if isinstance(raw_text, Mapping) else None
        )
        requests_constrained = bool(
            isinstance(payload.get("tools"), list)
            and payload.get("tools")
        ) or bool(
            isinstance(raw_text_format, Mapping)
            and raw_text_format.get("type") == "json_schema"
        )
        default_max_output_tokens = min(
            256 if requests_constrained else 16,
            self.runtime.max_output_tokens,
        )
        max_output_tokens = payload.get(
            "max_output_tokens",
            default_max_output_tokens,
        )
        if (
            isinstance(max_output_tokens, bool)
            or not isinstance(max_output_tokens, int)
            or max_output_tokens <= 0
        ):
            raise NativeResponsesAPIError(
                400,
                "'max_output_tokens' must be a positive integer.",
                param="max_output_tokens",
            )
        if max_output_tokens > self.runtime.max_output_tokens:
            raise NativeResponsesAPIError(
                400,
                f"'max_output_tokens' cannot exceed {self.runtime.max_output_tokens}.",
                param="max_output_tokens",
                code="max_tokens_exceeded",
            )
        (
            model,
            input_items,
            prompt_token_ids,
            instructions,
            previous_response_id,
            conversation_id,
            truncation,
            output_plan,
            has_tool_context,
            conversation_revision,
            previous_lineage,
        ) = self._common_input(
            scope,
            payload,
            supported_fields=_SUPPORTED_RESPONSE_FIELDS,
            reserve_output_tokens=max_output_tokens,
            allow_constrained=True,
        )
        stream = payload.get("stream", False)
        background = payload.get("background", False)
        store = payload.get("store", True)
        for name, value in (("stream", stream), ("background", background), ("store", store)):
            if not isinstance(value, bool):
                raise NativeResponsesAPIError(400, f"'{name}' must be a boolean.", param=name)
        if background and not store:
            raise NativeResponsesAPIError(
                400,
                "Background responses require 'store: true'.",
                param="store",
            )
        if conversation_id is not None and not store:
            raise NativeResponsesAPIError(
                400,
                "Conversation responses require 'store: true'.",
                param="store",
            )
        buffered_constrained = output_plan.kind != "text"
        if buffered_constrained or has_tool_context:
            if stream:
                raise NativeResponsesAPIError(
                    400,
                    "Function items and constrained output require 'stream: false'.",
                    param="stream",
                    code="unsupported_feature",
                )
            if background:
                raise NativeResponsesAPIError(
                    400,
                    "Function items and constrained output require 'background: false'.",
                    param="background",
                    code="unsupported_feature",
                )
            if not store:
                raise NativeResponsesAPIError(
                    400,
                    "Function items and constrained output require 'store: true'.",
                    param="store",
                    code="unsupported_feature",
                )
        if buffered_constrained and max_output_tokens > MAX_OUTPUT_BYTES:
            raise NativeResponsesAPIError(
                400,
                f"Constrained 'max_output_tokens' cannot exceed {MAX_OUTPUT_BYTES}.",
                param="max_output_tokens",
                code="max_tokens_exceeded",
            )
        temperature = payload.get("temperature", 0.0 if buffered_constrained else 0.8)
        top_p = payload.get("top_p", 1.0)
        if buffered_constrained and (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or float(temperature) != 0.0
        ):
            raise NativeResponsesAPIError(
                400,
                "Constrained output requires 'temperature: 0'.",
                param="temperature",
                code="unsupported_feature",
            )
        if buffered_constrained and (
            isinstance(top_p, bool)
            or not isinstance(top_p, (int, float))
            or float(top_p) != 1.0
        ):
            raise NativeResponsesAPIError(
                400,
                "Constrained output requires 'top_p: 1'.",
                param="top_p",
                code="unsupported_feature",
            )
        try:
            generation = GenerationConfig(
                max_new_tokens=max_output_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_token_ids=self._manifest_stop_token_ids(),
            )
        except (TypeError, ValueError) as exc:
            raise NativeResponsesAPIError(400, str(exc), code="invalid_parameter") from exc
        response_id = _resource_id("resp_")
        output_item_id = _resource_id(
            "fc_" if output_plan.kind == "function_call" else "msg_"
        )
        tool_call_id = (
            _resource_id("call_") if output_plan.kind == "function_call" else None
        )
        created_at = int(time.time())
        metadata = _metadata(payload.get("metadata"))
        status = "queued" if background else "in_progress"
        internal = {
            "prompt_token_ids": list(prompt_token_ids),
            "generation": {
                "max_new_tokens": generation.max_new_tokens,
                "temperature": generation.temperature,
                "top_k": generation.top_k,
                "top_p": generation.top_p,
                "seed": generation.seed,
                "stop_token_ids": list(generation.stop_token_ids),
            },
            "input_items": [deepcopy(item) for item in input_items],
            "conversation_id": conversation_id,
            "conversation_revision": conversation_revision,
            "previous_response_id": previous_response_id,
            "previous_lineage": [deepcopy(item) for item in previous_lineage],
            "output_item_id": output_item_id,
            "stream_requested": stream,
        }
        if tool_call_id is not None:
            internal["tool_call_id"] = tool_call_id
        response = {
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "status": status,
            "background": background,
            "completed_at": None,
            "conversation": ({"id": conversation_id} if conversation_id is not None else None),
            "error": None,
            "incomplete_details": None,
            "instructions": instructions,
            "max_output_tokens": max_output_tokens,
            "metadata": metadata,
            "model": model,
            "output": [],
            "parallel_tool_calls": False,
            "previous_response_id": previous_response_id,
            "store": store,
            "temperature": generation.temperature,
            "text": {
                "format": (
                    output_plan.compiled_schema.format_payload()
                    if output_plan.kind == "json_schema"
                    and output_plan.compiled_schema is not None
                    else {"type": "text"}
                )
            },
            "tool_choice": deepcopy(output_plan.tool_choice),
            "tools": [deepcopy(tool) for tool in output_plan.tools],
            "top_p": generation.top_p,
            "truncation": truncation,
            "usage": None,
            "_nfn": internal,
        }
        return PreparedNativeResponse(
            scope=scope,
            response_id=response_id,
            output_item_id=output_item_id,
            created_at=created_at,
            prompt_token_ids=prompt_token_ids,
            generation=generation,
            input_items=input_items,
            response=response,
            conversation_id=conversation_id,
            store=store,
            background=background,
            stream=stream,
            constrained_schema=output_plan.compiled_schema,
            output_kind=output_plan.kind,
            function_name=output_plan.function_name,
            tool_call_id=tool_call_id,
        )

    def _manifest_stop_token_ids(self) -> tuple[int, ...]:
        generation = self.runtime.manifest.get("generation")
        raw = generation.get("stop_token_ids") if isinstance(generation, Mapping) else None
        if raw is None:
            return ()
        if not isinstance(raw, list) or any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in raw
        ):
            raise NativeResponsesAPIError(
                500,
                "Artifact generation.stop_token_ids is invalid.",
                error_type="server_error",
                code="invalid_artifact",
            )
        return tuple(raw)

    def persist(self, prepared: PreparedNativeResponse) -> dict[str, Any]:
        if not prepared.store:
            return _public_response(prepared.response)
        response_created = False
        try:
            initial_events: tuple[dict[str, Any], ...] = ()
            if prepared.background and prepared.stream:
                initial_events = (
                    {
                        "type": "response.created",
                        "response": _public_response(prepared.response),
                    },
                )
            stored = self.state.put_response(
                prepared.scope,
                prepared.response,
                background=prepared.background,
                enqueue=prepared.background,
                response_events=initial_events,
            )
            response_created = True
            for item in prepared.input_items:
                self.state.append_response_item(
                    prepared.scope,
                    prepared.response_id,
                    item,
                    phase="input",
                )
        except NativeStateConflictError as exc:
            if response_created:
                with self._transition_lock:
                    self.state.delete_response(
                        prepared.scope,
                        prepared.response_id,
                    )
                    self._purge_scope_locked(prepared.scope)
            raise NativeResponsesAPIError(
                409,
                "A generated response or item ID already exists.",
                error_type="conflict_error",
                code="state_conflict",
            ) from exc
        return _public_response(stored)

    def from_stored_background(
        self,
        scope: str,
        response: Mapping[str, Any],
    ) -> PreparedNativeResponse:
        stored_response = deepcopy(dict(response))
        raw_internal = stored_response.get("_nfn")
        if not isinstance(raw_internal, Mapping):
            raise RuntimeError("Stored background response is missing native request state")
        internal = deepcopy(dict(raw_internal))
        has_conversation_snapshot = "conversation_revision" in internal
        has_lineage_snapshot = "previous_lineage" in internal
        if has_conversation_snapshot != has_lineage_snapshot:
            raise RuntimeError(
                "Stored background response has a partial native snapshot envelope"
            )
        previous_response_id = stored_response.get("previous_response_id")
        if previous_response_id is not None and (
            not isinstance(previous_response_id, str) or not previous_response_id
        ):
            raise RuntimeError("Stored background previous response ID is invalid")
        conversation_id = internal.get("conversation_id")
        public_conversation = stored_response.get("conversation")
        if conversation_id is None:
            if public_conversation is not None:
                raise RuntimeError(
                    "Stored background conversation identity is inconsistent"
                )
        elif (
            not isinstance(conversation_id, str)
            or not conversation_id.strip()
            or conversation_id != conversation_id.strip()
            or not isinstance(public_conversation, Mapping)
            or set(public_conversation) != {"id"}
            or public_conversation.get("id") != conversation_id
        ):
            raise RuntimeError(
                "Stored background conversation identity is inconsistent"
            )
        if not has_conversation_snapshot:
            if conversation_id is not None:
                raise NativeResponsesAPIError(
                    409,
                    "Legacy background conversation snapshot is unavailable.",
                    error_type="conflict_error",
                    code="conversation_snapshot_unavailable",
                )
            lineage_snapshot: list[dict[str, str]] = []
            if previous_response_id is not None:
                try:
                    lineage = self.state.response_lineage(
                        scope,
                        previous_response_id,
                    )
                except KeyError as exc:
                    raise NativeResponsesAPIError(
                        409,
                        "Legacy background previous-response lineage is unavailable.",
                        error_type="conflict_error",
                        code="response_lineage_unavailable",
                    ) from exc
                for ancestor in lineage:
                    status = str(ancestor.get("status") or "")
                    if status not in {"completed", "incomplete"}:
                        raise NativeResponsesAPIError(
                            409,
                            "Legacy background previous-response lineage is not ready.",
                            error_type="conflict_error",
                            code="response_lineage_unavailable",
                        )
                    lineage_snapshot.append(
                        {"id": str(ancestor["id"]), "status": status}
                    )
            internal["conversation_revision"] = None
            internal["previous_lineage"] = lineage_snapshot
        internal["previous_response_id"] = previous_response_id
        stored_response["_nfn"] = internal
        response = stored_response
        text = response.get("text")
        text_format = text.get("format") if isinstance(text, Mapping) else None
        if (
            internal.get("tool_call_id") is not None
            or response.get("tools") not in (None, [])
            or response.get("tool_choice") not in (None, "none")
            or not isinstance(text_format, Mapping)
            or dict(text_format) != {"type": "text"}
        ):
            raise RuntimeError(
                "Stored background responses cannot contain constrained output or function tools"
            )
        generation_raw = internal.get("generation")
        if not isinstance(generation_raw, Mapping):
            raise RuntimeError("Stored background response generation state is invalid")
        generation = GenerationConfig(
            max_new_tokens=generation_raw.get("max_new_tokens"),
            temperature=generation_raw.get("temperature"),
            top_k=generation_raw.get("top_k"),
            top_p=generation_raw.get("top_p"),
            seed=generation_raw.get("seed"),
            stop_token_ids=tuple(generation_raw.get("stop_token_ids") or ()),
        )
        token_ids = tuple(internal.get("prompt_token_ids") or ())
        if not token_ids or any(isinstance(token, bool) or not isinstance(token, int) for token in token_ids):
            raise RuntimeError("Stored background response prompt tokens are invalid")
        raw_items = internal.get("input_items") or ()
        if not isinstance(raw_items, list) or any(not isinstance(item, Mapping) for item in raw_items):
            raise RuntimeError("Stored background response input items are invalid")
        if any(
            item.get("type") in {"function_call", "function_call_output"}
            for item in raw_items
        ):
            raise RuntimeError("Stored background responses cannot contain function items")
        output_item_id = _non_empty_string(internal.get("output_item_id"), param="output_item_id")
        prepared = PreparedNativeResponse(
            scope=scope,
            response_id=str(response["id"]),
            output_item_id=output_item_id,
            created_at=int(response["created_at"]),
            prompt_token_ids=token_ids,
            generation=generation,
            input_items=tuple(deepcopy(dict(item)) for item in raw_items),
            response=deepcopy(dict(response)),
            conversation_id=conversation_id,
            store=True,
            background=True,
            stream=internal.get("stream_requested") is True,
            constrained_schema=None,
            output_kind="text",
            function_name=None,
            tool_call_id=None,
        )
        # A v4 snapshot envelope is a pre-generation contract.  Validate every
        # revision/lineage field at claim time so corrupt durable state never
        # reaches native session creation and merely fails later at terminal
        # persistence.
        self._conversation_revision(prepared)
        self._previous_lineage_signature(prepared)
        return prepared

    def begin_background_stream(self, prepared: PreparedNativeResponse) -> None:
        """Persist the semantic prelude when a queued streamed job starts."""

        if not (prepared.background and prepared.stream and prepared.store):
            return
        initial_response = _public_response(prepared.response)
        pending_item = {
            "id": prepared.output_item_id,
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": [],
        }
        empty_part = {
            "type": "output_text",
            "text": "",
            "annotations": [],
            "logprobs": [],
        }
        self.state.append_response_events(
            prepared.scope,
            prepared.response_id,
            (
                {"type": "response.in_progress", "response": initial_response},
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": pending_item,
                },
                {
                    "type": "response.content_part.added",
                    "item_id": prepared.output_item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "part": empty_part,
                },
            ),
        )

    def append_background_stream_delta(
        self,
        prepared: PreparedNativeResponse,
        fragment: str,
    ) -> None:
        """Persist one decoded text delta before native generation advances."""

        if not fragment or not (prepared.background and prepared.stream and prepared.store):
            return
        self.state.append_response_events(
            prepared.scope,
            prepared.response_id,
            (
                {
                    "type": "response.output_text.delta",
                    "item_id": prepared.output_item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": fragment,
                    "logprobs": [],
                    # Background stream events are replayable byte-for-byte, so
                    # generate transport padding once and persist it with the
                    # semantic event. Retrieval may explicitly omit this field.
                    "obfuscation": secrets.token_urlsafe(18),
                },
            ),
        )

    def require_replayable_response(
        self,
        scope: str,
        response_id: str,
    ) -> dict[str, Any]:
        """Validate that a response was created as a resumable background stream."""

        response = self.state.get_response(scope, response_id)
        if response is None:
            raise NativeResponsesAPIError(
                404,
                f"Response {response_id!r} was not found.",
                code="response_not_found",
            )
        internal = response.get("_nfn")
        if not (
            response.get("background") is True
            and isinstance(internal, Mapping)
            and internal.get("stream_requested") is True
        ):
            raise NativeResponsesAPIError(
                400,
                "Only background responses originally created with 'stream: true' "
                "can be retrieved as a stream.",
                param="stream",
                code="response_not_streamable",
            )
        return _public_response(response)

    def execute(
        self,
        prepared: PreparedNativeResponse,
        *,
        on_token: Callable[[GenerationEvent], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> CompletedNativeResponse:
        """Execute without retaining a resident cache lease.

        Background jobs deliberately use this cold path.  The standalone app
        uses :meth:`_execute_and_finish_resident` for foreground Responses so
        native ownership never crosses the synchronous queue-worker boundary.
        """

        with self.runtime.model.create_session() as session:
            return self._generate_on_session(
                prepared,
                session,
                on_token=on_token,
                cancel_event=cancel_event,
            )

    def _generate_on_session(
        self,
        prepared: PreparedNativeResponse,
        session: Any,
        *,
        on_token: Callable[[GenerationEvent], None] | None,
        cancel_event: threading.Event | None,
    ) -> CompletedNativeResponse:
        session.prefill(prepared.prompt_token_ids)

        def committed(event: GenerationEvent) -> None:
            if on_token is not None:
                on_token(event)
            if cancel_event is not None and cancel_event.is_set():
                session.cancel()

        if cancel_event is not None and cancel_event.is_set():
            session.cancel()
        if prepared.constrained_schema is not None:
            result = generate_json_schema_ascii_byte_greedy(
                session,
                self.runtime.codec,
                prepared.constrained_schema,
                max_new_tokens=prepared.generation.max_new_tokens,
                on_token=committed if on_token is not None else None,
            )
        else:
            result = session.decode(prepared.generation, on_token=committed)
        return CompletedNativeResponse(
            native=result,
            text=(
                result.text
                if prepared.constrained_schema is not None
                else self.runtime.codec.decode(result.token_ids)
            ),
        )

    def _execute_resident(
        self,
        prepared: PreparedNativeResponse,
        *,
        on_token: Callable[[GenerationEvent], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> _ResidentResponseExecution:
        """Execute a foreground response while privately retaining ownership."""

        if prepared.background:
            raise ValueError("Background responses use the cold execution path")
        cache = self._prefix_cache
        if cache is None:
            return _ResidentResponseExecution(
                completed=self.execute(
                    prepared,
                    on_token=on_token,
                    cancel_event=cancel_event,
                ),
                lease=None,
                usage=NativePrefixCacheUsage(),
            )

        seed = prepared.generation.seed
        lease: NativePrefixCacheLease | None = None
        previous_response_id = self._previous_response_id(prepared)
        conversation_revision = self._conversation_revision(prepared)
        if previous_response_id is not None:
            lease = cache.acquire(
                scope=prepared.scope,
                response_id=previous_response_id,
                prompt_token_ids=prepared.prompt_token_ids,
                seed=0 if seed is None else seed,
            )
        elif prepared.conversation_id is not None:
            assert conversation_revision is not None
            lease = cache.acquire(
                scope=prepared.scope,
                conversation_id=prepared.conversation_id,
                conversation_revision=conversation_revision,
                prompt_token_ids=prepared.prompt_token_ids,
                seed=0 if seed is None else seed,
            )

        if lease is None:
            session = self.runtime.model.create_session(
                seed=0 if seed is None else seed
            )
            try:
                lease = cache.lease_session(
                    session,
                    scope=prepared.scope,
                    prompt_token_ids=prepared.prompt_token_ids,
                )
            except BaseException:
                session.close()
                raise

        try:
            completed = self._generate_on_session(
                prepared,
                lease.session,
                on_token=on_token,
                cancel_event=cancel_event,
            )
            usage = lease.usage()
        except BaseException:
            lease.close()
            raise
        return _ResidentResponseExecution(
            completed=completed,
            lease=lease,
            usage=usage,
        )

    def _usage(
        self,
        prepared: PreparedNativeResponse,
        completed: CompletedNativeResponse,
        *,
        prefix_usage: NativePrefixCacheUsage | None = None,
    ) -> dict[str, Any]:
        input_tokens = len(prepared.prompt_token_ids)
        output_tokens = completed.native.completion_tokens
        details = (
            NativePrefixCacheUsage()
            if prefix_usage is None
            else prefix_usage
        ).input_tokens_details()
        return {
            "input_tokens": input_tokens,
            "input_tokens_details": details,
            "output_tokens": output_tokens,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": input_tokens + output_tokens,
        }

    def output_item(
        self,
        prepared: PreparedNativeResponse,
        text: str,
        *,
        status: str = "completed",
    ) -> dict[str, Any]:
        if prepared.output_kind == "function_call":
            if prepared.function_name is None or prepared.tool_call_id is None:
                raise RuntimeError("Prepared function call is missing stable identifiers")
            return {
                "id": prepared.output_item_id,
                "type": "function_call",
                "status": status,
                "call_id": prepared.tool_call_id,
                "name": prepared.function_name,
                "arguments": text,
            }
        return {
            "id": prepared.output_item_id,
            "type": "message",
            "status": status,
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": text,
                    "annotations": [],
                    "logprobs": [],
                }
            ],
        }

    def finish(
        self,
        prepared: PreparedNativeResponse,
        completed: CompletedNativeResponse,
    ) -> dict[str, Any]:
        """Finish a cold/public execution without resident cache admission."""

        with self._transition_lock:
            return self._finish_locked(
                prepared,
                completed,
                prefix_usage=NativePrefixCacheUsage(),
            ).response

    def _finish_resident(
        self,
        prepared: PreparedNativeResponse,
        execution: _ResidentResponseExecution,
    ) -> dict[str, Any]:
        """Atomically persist, then publish one private resident cache lease."""

        if not isinstance(execution, _ResidentResponseExecution):
            raise TypeError("execution must come from execute_resident")
        if execution.consumed:
            raise RuntimeError("Resident response execution is already consumed")
        execution.consumed = True
        lease = execution.lease
        with self._transition_lock:
            try:
                finished = self._finish_locked(
                    prepared,
                    execution.completed,
                    prefix_usage=execution.usage,
                )
            except BaseException:
                if lease is not None:
                    lease.close()
                raise
            if lease is not None:
                try:
                    lease.commit(
                        scope=prepared.scope,
                        response_id=prepared.response_id,
                        status=finished.status,
                        stored=prepared.store,
                        conversation_id=(
                            prepared.conversation_id
                            if finished.conversation_revision is not None
                            else None
                        ),
                        conversation_revision=finished.conversation_revision,
                    )
                except BaseException:
                    # Prefix reuse is best-effort and must never reverse a
                    # durable response/conversation commit.
                    lease.close()
            return finished.response

    def _execute_and_finish_resident(
        self,
        prepared: PreparedNativeResponse,
        *,
        on_token: Callable[[GenerationEvent], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> tuple[CompletedNativeResponse, dict[str, Any]]:
        """Own generation, durable finish, and lease disposition synchronously.

        The compute-queue callable uses this boundary so cancellation of its
        async waiter can never strand a returned live native session lease.
        """

        try:
            execution = self._execute_resident(
                prepared,
                on_token=on_token,
                cancel_event=cancel_event,
            )
        except BaseException:
            try:
                self.fail(prepared)
            except BaseException:
                pass
            raise
        try:
            response = self._finish_resident(prepared, execution)
        except NativeResponsesAPIError:
            # Lineage/CAS failures terminalize the durable response with their
            # stable conflict code before surfacing the HTTP-shaped error.
            raise
        except BaseException:
            try:
                self.fail(prepared)
            except BaseException:
                pass
            raise
        return execution.completed, response

    def _finish_locked(
        self,
        prepared: PreparedNativeResponse,
        completed: CompletedNativeResponse,
        *,
        prefix_usage: NativePrefixCacheUsage,
    ) -> _FinishedNativeResponse:
        try:
            self._validate_previous_lineage_locked(prepared)
        except NativeStateConflictError as exc:
            self._fail_locked(
                prepared,
                code=exc.code,
                message="Previous response lineage changed before completion.",
            )
            raise NativeResponsesAPIError(
                409,
                "Previous response lineage changed before completion.",
                error_type="conflict_error",
                param="previous_response_id",
                code=exc.code,
            ) from exc

        finish_reason = completed.native.finish_reason
        cancel_requested = prepared.background and self.state.is_cancel_requested(
            prepared.scope,
            prepared.response_id,
        )
        if cancel_requested or completed.native.cancelled or finish_reason == "cancelled":
            status = "cancelled"
            incomplete_details: dict[str, Any] | None = None
        elif finish_reason == "length":
            status = "incomplete"
            incomplete_details = {"reason": "max_output_tokens"}
        else:
            status = "completed"
            incomplete_details = None
        item_status = "incomplete" if status in {"cancelled", "incomplete"} else "completed"
        output_item = self.output_item(prepared, completed.text, status=item_status)
        patch = {
            "status": status,
            "completed_at": int(time.time()),
            "error": None,
            "incomplete_details": incomplete_details,
            "output": [output_item],
            "usage": self._usage(
                prepared,
                completed,
                prefix_usage=prefix_usage,
            ),
        }
        committed_revision: int | None = None
        expected_revision = self._conversation_revision(prepared)
        commits_conversation = (
            prepared.conversation_id is not None
            and status in {"completed", "incomplete"}
        )
        conversation_items = (
            (*prepared.input_items, output_item) if commits_conversation else ()
        )
        if prepared.store:
            try:
                if prepared.background:
                    stream_events: tuple[dict[str, Any], ...] = ()
                    if prepared.stream:
                        output_part = output_item["content"][0]
                        stream_events = (
                            {
                                "type": "response.output_text.done",
                                "item_id": prepared.output_item_id,
                                "output_index": 0,
                                "content_index": 0,
                                "text": completed.text,
                                "logprobs": [],
                            },
                            {
                                "type": "response.content_part.done",
                                "item_id": prepared.output_item_id,
                                "output_index": 0,
                                "content_index": 0,
                                "part": output_part,
                            },
                            {
                                "type": "response.output_item.done",
                                "output_index": 0,
                                "item": output_item,
                            },
                        )
                    stored = self.state.finish_background_job(
                        prepared.scope,
                        prepared.response_id,
                        status=status,
                        response_patch=patch,
                        response_item=output_item,
                        response_events=stream_events,
                        conversation_id=(
                            prepared.conversation_id if commits_conversation else None
                        ),
                        conversation_items=conversation_items,
                        expected_conversation_revision=(
                            expected_revision if commits_conversation else None
                        ),
                    )
                    if commits_conversation:
                        assert expected_revision is not None
                        committed_revision = expected_revision + 1
                else:
                    commit = self.state.finish_foreground_response(
                        prepared.scope,
                        prepared.response_id,
                        status=status,
                        response_patch=patch,
                        response_items=(output_item,),
                        conversation_id=(
                            prepared.conversation_id if commits_conversation else None
                        ),
                        conversation_items=conversation_items,
                        expected_conversation_revision=(
                            expected_revision if commits_conversation else None
                        ),
                    )
                    if commit is None:
                        stored = None
                    else:
                        stored, committed_revision = commit
            except NativeStateConflictError as exc:
                if exc.code != "conversation_conflict":
                    raise
                self._fail_locked(
                    prepared,
                    code=exc.code,
                    message="Conversation changed before this response completed.",
                )
                raise NativeResponsesAPIError(
                    409,
                    "Conversation changed before this response completed.",
                    error_type="conflict_error",
                    param="conversation",
                    code=exc.code,
                ) from exc
            except KeyError as exc:
                if not commits_conversation:
                    raise
                self._fail_locked(
                    prepared,
                    code="conversation_conflict",
                    message="Conversation changed before this response completed.",
                )
                raise NativeResponsesAPIError(
                    409,
                    "Conversation changed before this response completed.",
                    error_type="conflict_error",
                    param="conversation",
                    code="conversation_conflict",
                ) from exc
            if stored is None:
                raise RuntimeError("Stored response disappeared during generation")
            response = stored
        else:
            response = deepcopy(prepared.response)
            response.update(patch)
        if commits_conversation:
            assert prepared.conversation_id is not None
            assert expected_revision is not None
            assert committed_revision is not None
            cache = self._prefix_cache
            if cache is not None:
                if prepared.background:
                    # Background work intentionally stays cold.  Its history
                    # mutation fences every previously prepared foreground
                    # lease in this API-key scope rather than publishing a new
                    # resident alias.
                    cache.purge_scope(prepared.scope)
                else:
                    cache.delete_conversation_alias(
                        scope=prepared.scope,
                        conversation_id=prepared.conversation_id,
                        conversation_revision=expected_revision,
                    )
        return _FinishedNativeResponse(
            response=_public_response(response),
            status=status,
            conversation_revision=committed_revision,
        )

    def _fail_locked(
        self,
        prepared: PreparedNativeResponse,
        *,
        code: str,
        message: str,
    ) -> dict[str, Any]:
        error = {"code": code, "message": message}
        patch = {
            "status": "failed",
            "completed_at": int(time.time()),
            "error": error,
            "incomplete_details": None,
            "output": [],
        }
        if prepared.store:
            if prepared.background:
                stored = self.state.finish_background_job(
                    prepared.scope,
                    prepared.response_id,
                    status="failed",
                    response_patch=patch,
                    error=error,
                )
            else:
                stored = self.state.update_response(
                    prepared.scope,
                    prepared.response_id,
                    patch,
                )
            if stored is None:
                raise RuntimeError("Stored response disappeared during failure handling")
            return _public_response(stored)
        response = deepcopy(prepared.response)
        response.update(patch)
        return _public_response(response)

    def fail(
        self,
        prepared: PreparedNativeResponse,
        *,
        code: str = "generation_failed",
        message: str = "Resident native generation failed.",
    ) -> dict[str, Any]:
        with self._transition_lock:
            return self._fail_locked(
                prepared,
                code=code,
                message=message,
            )

    def retrieve_response(self, scope: str, response_id: str) -> dict[str, Any]:
        response = self.state.get_response(scope, response_id)
        if response is None:
            raise NativeResponsesAPIError(
                404,
                f"Response {response_id!r} was not found.",
                code="response_not_found",
            )
        return _public_response(response)

    def delete_response(self, scope: str, response_id: str) -> dict[str, Any]:
        with self._transition_lock:
            response = self.state.get_response(scope, response_id)
            if response is None:
                raise NativeResponsesAPIError(
                    404,
                    f"Response {response_id!r} was not found.",
                    code="response_not_found",
                )
            if response.get("status") not in _TERMINAL_RESPONSE_STATUSES:
                raise NativeResponsesAPIError(
                    409,
                    "An active response must be cancelled before it can be deleted.",
                    code="response_active",
                )
            if not self.state.delete_response(scope, response_id):
                raise NativeResponsesAPIError(
                    404,
                    f"Response {response_id!r} was not found.",
                    code="response_not_found",
                )
            self._purge_scope_locked(scope)
        return {"id": response_id, "object": "response", "deleted": True}

    def response_input_items(
        self,
        scope: str,
        response_id: str,
        *,
        after: str | None = None,
        limit: int = 20,
        order: str = "desc",
    ) -> dict[str, Any]:
        if self.state.get_response(scope, response_id) is None:
            raise NativeResponsesAPIError(
                404,
                f"Response {response_id!r} was not found.",
                code="response_not_found",
            )
        return _paginated_list_payload(
            self.state.list_response_items(scope, response_id, phase="input"),
            after=after,
            limit=limit,
            order=order,
        )

    def count_input_tokens(self, scope: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        _, _, token_ids, _, _, _, _, _, _, _, _ = self._common_input(
            scope,
            payload,
            supported_fields=_SUPPORTED_INPUT_TOKEN_FIELDS,
            reserve_output_tokens=0,
        )
        return {"object": "response.input_tokens", "input_tokens": len(token_ids)}

    def compact_response(self, scope: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Create a durable, lossless local compaction in the OpenAI shape.

        The native server does not run a summarizer here.  Instead, the
        ``encrypted_content`` value is an unguessable, scope-bound reference to
        context held in the private state database.  Passing the returned
        output items to a later Responses request restores that exact context.
        """

        self._validate_capability_fields(
            payload,
            supported_fields=_SUPPORTED_COMPACT_FIELDS,
        )
        model = self._require_model(payload)
        for field in (
            "prompt_cache_key",
            "prompt_cache_options",
            "prompt_cache_retention",
            "service_tier",
        ):
            if payload.get(field) is not None:
                raise NativeResponsesAPIError(
                    400,
                    f"'{field}' is not supported by local response compaction.",
                    param=field,
                    code="unsupported_feature",
                )
        previous_response_id = payload.get("previous_response_id")
        if previous_response_id is not None:
            previous_response_id = _non_empty_string(
                previous_response_id,
                param="previous_response_id",
            )
        _input_items, input_messages, _has_function_output = self._normalize_response_input(
            scope,
            payload.get("input"),
        )
        instructions = payload.get("instructions")
        if instructions is not None and not isinstance(instructions, str):
            raise NativeResponsesAPIError(
                400,
                "'instructions' must be a string for this text-only server.",
                param="instructions",
                code="unsupported_feature",
            )
        messages: list[NativeChatMessage] = []
        if instructions:
            messages.append(NativeChatMessage(role="developer", content=instructions))
        history = self._history_messages(
            scope,
            previous_response_id=previous_response_id,
            conversation_id=None,
        )
        messages.extend(history.messages)
        messages.extend(input_messages)
        if not messages:
            raise NativeResponsesAPIError(
                400,
                "'input' is required when no previous response supplies context.",
                param="input",
                code="invalid_input",
            )
        prompt_token_ids = self._render_prompt(
            messages,
            preserve_tail=len(input_messages),
            reserve_output_tokens=0,
            truncation="disabled",
        )

        compaction_id = _resource_id("cmp_")
        created_at = int(time.time())
        encrypted_content = "nfncmp_" + secrets.token_urlsafe(32)
        try:
            self.state.put_response_compaction(
                scope,
                {
                    "id": compaction_id,
                    "created_at": created_at,
                    "model": model,
                    "messages": [_stored_message(message) for message in messages],
                },
                encrypted_content=encrypted_content,
            )
        except NativeStateConflictError as exc:
            raise NativeResponsesAPIError(
                409,
                "A generated response compaction ID already exists.",
                error_type="conflict_error",
                code="state_conflict",
            ) from exc

        retained_user_items: list[dict[str, Any]] = []
        for message in messages:
            if message.role != "user":
                continue
            item, _ = _normalize_message(
                _stored_message(message),
                param="input",
            )
            retained_user_items.append(item)
        compaction_item = {
            "id": compaction_id,
            "type": "compaction",
            "encrypted_content": encrypted_content,
            "created_by": "neuralfn",
        }
        return {
            "id": _resource_id("resp_"),
            "created_at": created_at,
            "object": "response.compaction",
            "output": [*retained_user_items, compaction_item],
            "usage": {
                "input_tokens": len(prompt_token_ids),
                "input_tokens_details": {
                    "cached_tokens": 0,
                    "cache_write_tokens": 0,
                },
                "output_tokens": 0,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": len(prompt_token_ids),
            },
        }

    def cancel_response(self, scope: str, response_id: str) -> dict[str, Any]:
        response = self.state.get_response(scope, response_id)
        if response is None:
            raise NativeResponsesAPIError(
                404,
                f"Response {response_id!r} was not found.",
                code="response_not_found",
            )
        if not response.get("background"):
            raise NativeResponsesAPIError(
                400,
                "Only background responses can be cancelled through this endpoint.",
                code="invalid_response_status",
            )
        if response.get("status") in _TERMINAL_RESPONSE_STATUSES:
            raise NativeResponsesAPIError(
                400,
                f"Response {response_id!r} is already {response.get('status')}.",
                code="invalid_response_status",
            )
        if not self.state.request_cancel(scope, response_id):
            raise NativeResponsesAPIError(404, "Response was not found.", code="response_not_found")
        return self.retrieve_response(scope, response_id)

    def create_conversation(self, scope: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        unknown = sorted(set(payload) - {"items", "metadata"})
        if unknown:
            field = unknown[0]
            raise NativeResponsesAPIError(
                400,
                f"Conversation field {field!r} is not supported.",
                param=field,
                code="unsupported_feature",
            )
        raw_items = payload.get("items", [])
        if not isinstance(raw_items, list):
            raise NativeResponsesAPIError(400, "'items' must be an array.", param="items")
        if len(raw_items) > 20:
            raise NativeResponsesAPIError(400, "At most 20 conversation items may be added at once.", param="items")
        items: list[dict[str, Any]] = []
        for index, raw in enumerate(raw_items):
            item, _message = _normalize_message(raw, param=f"items.{index}")
            items.append(item)
        conversation = {
            "id": _resource_id("conv_"),
            "object": "conversation",
            "created_at": int(time.time()),
            "metadata": _metadata(payload.get("metadata")),
        }
        conversation_created = False
        try:
            stored = self.state.put_conversation(scope, conversation)
            conversation_created = True
            if items:
                with self._transition_lock:
                    self.state.append_conversation_items_with_revision(
                        scope,
                        conversation["id"],
                        items,
                    )
        except NativeStateConflictError as exc:
            if conversation_created:
                with self._transition_lock:
                    self.state.delete_conversation(scope, conversation["id"])
                    self._purge_scope_locked(scope)
            raise NativeResponsesAPIError(
                409,
                "A generated conversation or item ID already exists.",
                error_type="conflict_error",
                code="state_conflict",
            ) from exc
        return stored

    def retrieve_conversation(self, scope: str, conversation_id: str) -> dict[str, Any]:
        conversation = self.state.get_conversation(scope, conversation_id)
        if conversation is None:
            raise NativeResponsesAPIError(
                404,
                f"Conversation {conversation_id!r} was not found.",
                code="conversation_not_found",
            )
        return conversation

    def update_conversation(
        self,
        scope: str,
        conversation_id: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        if set(payload) != {"metadata"}:
            raise NativeResponsesAPIError(
                400,
                "Conversation updates require exactly the 'metadata' field.",
                param="metadata",
            )
        updated = self.state.update_conversation(
            scope,
            conversation_id,
            {"metadata": _metadata(payload.get("metadata"))},
        )
        if updated is None:
            raise NativeResponsesAPIError(
                404,
                f"Conversation {conversation_id!r} was not found.",
                code="conversation_not_found",
            )
        return updated

    def delete_conversation(self, scope: str, conversation_id: str) -> dict[str, Any]:
        with self._transition_lock:
            if not self.state.delete_conversation(scope, conversation_id):
                raise NativeResponsesAPIError(
                    404,
                    f"Conversation {conversation_id!r} was not found.",
                    code="conversation_not_found",
                )
            self._purge_scope_locked(scope)
        return {"id": conversation_id, "object": "conversation.deleted", "deleted": True}

    def create_conversation_items(
        self,
        scope: str,
        conversation_id: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        if set(payload) != {"items"}:
            raise NativeResponsesAPIError(400, "Request body must contain only 'items'.", param="items")
        raw_items = payload.get("items")
        if not isinstance(raw_items, list) or not raw_items:
            raise NativeResponsesAPIError(400, "'items' must be a non-empty array.", param="items")
        if len(raw_items) > 20:
            raise NativeResponsesAPIError(400, "At most 20 conversation items may be added at once.", param="items")
        items = [
            _normalize_message(raw, param=f"items.{index}")[0]
            for index, raw in enumerate(raw_items)
        ]
        try:
            with self._transition_lock:
                stored, _committed_revision = (
                    self.state.append_conversation_items_with_revision(
                        scope,
                        conversation_id,
                        items,
                    )
                )
                cache = self._prefix_cache
                if cache is not None:
                    cache.purge_scope(scope)
        except NativeStateConflictError as exc:
            raise NativeResponsesAPIError(
                409,
                "A conversation item ID already exists.",
                error_type="conflict_error",
                code="state_conflict",
            ) from exc
        except KeyError as exc:
            raise NativeResponsesAPIError(
                404,
                f"Conversation {conversation_id!r} was not found.",
                code="conversation_not_found",
            ) from exc
        return _list_payload(stored)

    def list_conversation_items(
        self,
        scope: str,
        conversation_id: str,
        *,
        after: str | None = None,
        limit: int = 20,
        order: str = "desc",
    ) -> dict[str, Any]:
        self.retrieve_conversation(scope, conversation_id)
        return _paginated_list_payload(
            self.state.list_conversation_items(scope, conversation_id),
            after=after,
            limit=limit,
            order=order,
        )

    def retrieve_conversation_item(
        self,
        scope: str,
        conversation_id: str,
        item_id: str,
    ) -> dict[str, Any]:
        self.retrieve_conversation(scope, conversation_id)
        item = self.state.get_conversation_item(scope, conversation_id, item_id)
        if item is None:
            raise NativeResponsesAPIError(
                404,
                f"Conversation item {item_id!r} was not found.",
                code="conversation_item_not_found",
            )
        return item

    def delete_conversation_item(
        self,
        scope: str,
        conversation_id: str,
        item_id: str,
    ) -> dict[str, Any]:
        with self._transition_lock:
            self.retrieve_conversation(scope, conversation_id)
            if not self.state.delete_conversation_item(
                scope,
                conversation_id,
                item_id,
            ):
                raise NativeResponsesAPIError(
                    404,
                    f"Conversation item {item_id!r} was not found.",
                    code="conversation_item_not_found",
                )
            self._purge_scope_locked(scope)
        return self.retrieve_conversation(scope, conversation_id)


__all__ = [
    "CompletedNativeResponse",
    "NativeResponsesAPIError",
    "NativeResponsesService",
    "PreparedNativeResponse",
]
