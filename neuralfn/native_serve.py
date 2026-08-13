"""Standalone, inference-only OpenAI-compatible serving for native artifacts.

This module deliberately does not import ``server.app`` or initialize editor
authentication, databases, persistence workers, Torch, NumPy, or NetworkX.
It serves one already-proven resident native model through a bounded
single-worker compute queue.  Text Chat Completions are always available and
jointly proven Muse Glimmer artifacts may also accept bounded image data URLs;
opt-in local state additionally enables text Responses and Conversations.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
import hmac
import importlib
import ipaddress
import json
import os
from pathlib import Path
import secrets
import stat
import threading
import time
from typing import Any, Callable, Coroutine, Mapping, Sequence

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from starlette.exceptions import HTTPException as StarletteHTTPException
except ImportError as exc:  # pragma: no cover - exercised by lean-install probe
    raise ImportError(
        "Native inference serving requires the optional dependencies from "
        "`pip install -e '.[serve]'`."
    ) from exc

from .native_inference import (
    GenerationConfig,
    GenerationEvent,
    GenerationResult,
    KVCacheConfig,
    NativeInferenceCapabilities,
    NativeInferenceCapabilityError,
    NativeInferenceModel,
    NativeModelLoadConfig,
)
from .native_chat import (
    NativeChatConfigurationError,
    NativeChatMessage,
    NativeTextCodec,
    load_native_text_codec,
    resolve_native_chat_renderer,
)
from ._native_prefix_cache import NativePrefixCache
from .native_constrained import (
    NativeConstrainedError,
    compile_single_byte_token_inventory,
)
from .native_state import NativeStateStore, api_key_fingerprint
from .native_responses import (
    NativeResponsesAPIError,
    NativeResponsesService,
    PreparedNativeResponse,
)


_MANIFEST_SCHEMA = "neuralfn.native_execution_manifest"
_MANIFEST_VERSION = 1
_TEXT_ROLES = frozenset({"developer", "system", "user", "assistant"})
_SUPPORTED_MESSAGE_FIELDS = frozenset({"content", "name", "role"})
_SUPPORTED_TEXT_PART_FIELDS = frozenset({"text", "type"})
_SUPPORTED_IMAGE_PART_FIELDS = frozenset({"image_url", "type"})
_SUPPORTED_IMAGE_URL_FIELDS = frozenset({"detail", "url"})
_SUPPORTED_CHAT_FIELDS = frozenset(
    {
        "max_completion_tokens",
        "max_tokens",
        "messages",
        "model",
        "n",
        "seed",
        "stream",
        "stream_options",
        "temperature",
        "top_p",
    }
)
_MAX_STATEFUL_REQUEST_BYTES = 1024 * 1024
_RESPONSE_INCLUDE_VALUES = frozenset(
    {
        "file_search_call.results",
        "web_search_call.results",
        "web_search_call.action.sources",
        "message.input_image.image_url",
        "computer_call_output.output.image_url",
        "code_interpreter_call.outputs",
        "reasoning.encrypted_content",
        "message.output_text.logprobs",
    }
)
_TERMINAL_RESPONSE_STREAM_EVENTS = frozenset(
    {"response.completed", "response.failed", "response.incomplete"}
)
_IMAGE_MARKER_PREFIX = "\x00neuralfn_muse_glimmer_image_"


@dataclass(frozen=True, slots=True)
class _ResponseRetrieveQuery:
    stream: bool
    starting_after: int | None
    include: tuple[str, ...]
    include_obfuscation: bool


class OpenAIAPIError(Exception):
    """An HTTP failure rendered with the OpenAI error envelope."""

    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        error_type: str = "invalid_request_error",
        param: str | None = None,
        code: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = int(status_code)
        self.message = str(message)
        self.error_type = str(error_type)
        self.param = param
        self.code = code
        self.headers = dict(headers or {})

    def payload(self) -> dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        }


class NativeServingConfigurationError(RuntimeError):
    """A startup-time validation failure that must occur before socket bind."""


@dataclass(frozen=True, slots=True)
class NativeServeConfig:
    artifact: Path
    host: str = "127.0.0.1"
    port: int = 8000
    served_model_name: str | None = None
    queue_capacity: int = 8
    session_limit: int | None = None
    max_output_tokens: int = 256
    kv_cache: KVCacheConfig = field(default_factory=lambda: KVCacheConfig(mode="auto"))
    model_load: NativeModelLoadConfig = field(default_factory=NativeModelLoadConfig)
    chat_template: str = "auto"
    api_key_file: Path | None = None
    state_db: Path | None = None
    api_key: str | None = field(default=None, repr=False)
    allow_unauthenticated_remote: bool = False
    log_level: str = "info"
    prefix_cache_capacity: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact", Path(self.artifact).expanduser())
        host = str(self.host).strip()
        if not host:
            raise ValueError("host must not be empty")
        object.__setattr__(self, "host", host)
        if isinstance(self.port, bool) or not isinstance(self.port, int) or not 1 <= self.port <= 65535:
            raise ValueError("port must be an integer between 1 and 65535")
        if (
            isinstance(self.queue_capacity, bool)
            or not isinstance(self.queue_capacity, int)
            or self.queue_capacity < 0
        ):
            raise ValueError("queue_capacity must be a non-negative integer")
        if self.session_limit is None:
            # Preserve the queue contract by default: one running request plus
            # every configured waiter may hold a request-session reservation.
            object.__setattr__(self, "session_limit", self.queue_capacity + 1)
        elif (
            isinstance(self.session_limit, bool)
            or not isinstance(self.session_limit, int)
            or self.session_limit <= 0
        ):
            raise ValueError("session_limit must be a positive integer")
        if (
            isinstance(self.max_output_tokens, bool)
            or not isinstance(self.max_output_tokens, int)
            or self.max_output_tokens <= 0
        ):
            raise ValueError("max_output_tokens must be a positive integer")
        if (
            isinstance(self.prefix_cache_capacity, bool)
            or not isinstance(self.prefix_cache_capacity, int)
            or self.prefix_cache_capacity < 0
        ):
            raise ValueError("prefix_cache_capacity must be a non-negative integer")
        if self.served_model_name is not None and not str(self.served_model_name).strip():
            raise ValueError("served_model_name must not be empty")
        object.__setattr__(self, "chat_template", str(self.chat_template).strip() or "auto")
        if self.api_key_file is not None:
            object.__setattr__(self, "api_key_file", Path(self.api_key_file).expanduser())
        if self.state_db is not None:
            object.__setattr__(self, "state_db", Path(self.state_db).expanduser())
        if self.prefix_cache_capacity and self.state_db is None:
            raise ValueError("prefix_cache_capacity requires state_db")
        if (
            self.prefix_cache_capacity
            and self.kv_cache.turboquant_attention_backend == "tile-cuda"
        ):
            raise ValueError(
                "prefix_cache_capacity rejects Tile-CUDA TurboQuant attention"
            )


@dataclass(frozen=True, slots=True)
class BearerAuth:
    keys: tuple[str, ...] = field(default=(), repr=False)

    @property
    def enabled(self) -> bool:
        return bool(self.keys)

    def require(self, authorization: str | None) -> str:
        if not self.enabled:
            return api_key_fingerprint(None)
        scheme, separator, supplied = str(authorization or "").partition(" ")
        valid = separator == " " and scheme.lower() == "bearer" and bool(supplied)
        if valid:
            valid = any(hmac.compare_digest(supplied, expected) for expected in self.keys)
        if not valid:
            raise OpenAIAPIError(
                401,
                "Incorrect API key provided.",
                error_type="invalid_request_error",
                code="invalid_api_key",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return api_key_fingerprint(supplied)


def _is_loopback_host(host: str) -> bool:
    normalized = host.strip().lower()
    if normalized in {"localhost", "ip6-localhost"}:
        return True
    if normalized.startswith("[") and normalized.endswith("]"):
        normalized = normalized[1:-1]
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _read_api_key_file(path: Path) -> tuple[str, ...]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise NativeServingConfigurationError(f"API key file does not exist: {resolved}")
    mode = stat.S_IMODE(resolved.stat().st_mode)
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise NativeServingConfigurationError(
            f"API key file must not be accessible by group or other users: {resolved}"
        )
    keys = tuple(
        line.strip()
        for line in resolved.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not keys:
        raise NativeServingConfigurationError(f"API key file contains no keys: {resolved}")
    if len(set(keys)) != len(keys):
        raise NativeServingConfigurationError(f"API key file contains duplicate keys: {resolved}")
    return keys


def resolve_bearer_auth(config: NativeServeConfig) -> BearerAuth:
    environment_key = os.environ.get("NFN_INFER_API_KEY")
    direct_key = config.api_key if config.api_key is not None else environment_key
    if direct_key is not None:
        direct_key = str(direct_key).strip()
        if not direct_key:
            raise NativeServingConfigurationError("NFN_INFER_API_KEY must not be empty")
    if direct_key and config.api_key_file is not None:
        raise NativeServingConfigurationError(
            "Configure either NFN_INFER_API_KEY or --api-key-file, not both"
        )
    keys = _read_api_key_file(config.api_key_file) if config.api_key_file else ()
    if direct_key:
        keys = (direct_key,)
    if not _is_loopback_host(config.host) and not keys and not config.allow_unauthenticated_remote:
        raise NativeServingConfigurationError(
            "Non-loopback binding requires NFN_INFER_API_KEY or --api-key-file. "
            "Use --allow-unauthenticated-remote only after accepting the exposure risk."
        )
    return BearerAuth(keys=keys)


class _IncrementalTokenDecoder:
    def __init__(self, codec: "_TextCodec") -> None:
        import codecs

        self._codec = codec
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def push(self, token_id: int) -> str:
        return self._decoder.decode(self._codec.token_bytes(token_id), final=False)

    def finish(self) -> str:
        return self._decoder.decode(b"", final=True)


class _TextCodec:
    name: str

    def encode(self, text: str) -> tuple[int, ...]:
        raise NotImplementedError

    def decode(self, token_ids: Sequence[int]) -> str:
        raise NotImplementedError

    def token_bytes(self, token_id: int) -> bytes:
        raise NotImplementedError

    def incremental_decoder(self) -> _IncrementalTokenDecoder:
        return _IncrementalTokenDecoder(self)


class _TiktokenCodec(_TextCodec):
    def __init__(self, encoding_name: str) -> None:
        try:
            tiktoken = importlib.import_module("tiktoken")
            encoding = tiktoken.get_encoding(encoding_name)
        except (ImportError, ValueError) as exc:
            raise NativeServingConfigurationError(
                f"Unable to load artifact tiktoken encoding {encoding_name!r}; "
                "install the [serve] extra and verify tokenizer metadata"
            ) from exc
        self.name = encoding_name
        self._encoding = encoding

    def encode(self, text: str) -> tuple[int, ...]:
        return tuple(self._encoding.encode(text, allowed_special=set(), disallowed_special=()))

    def decode(self, token_ids: Sequence[int]) -> str:
        return self._encoding.decode(list(token_ids), errors="replace")

    def token_bytes(self, token_id: int) -> bytes:
        try:
            return self._encoding.decode_single_token_bytes(token_id)
        except KeyError as exc:
            raise RuntimeError(f"Native binding produced unknown token id {token_id}") from exc


def _load_text_codec(manifest: Mapping[str, Any]) -> _TextCodec:
    artifact_root = manifest.get("__artifact_root__")
    try:
        return load_native_text_codec(
            manifest,
            artifact_root=Path(str(artifact_root)) if artifact_root else None,
        )  # type: ignore[return-value]
    except NativeChatConfigurationError as exc:
        raise NativeServingConfigurationError(str(exc)) from exc


@dataclass(frozen=True, slots=True)
class _ChatMessage:
    role: str
    content: str
    name: str | None = None
    image_sources: tuple[str, ...] = ()


class _ChatRenderer:
    name: str

    def render(self, messages: Sequence[_ChatMessage]) -> str:
        raise NotImplementedError


def _plain_role_transcript(messages: Sequence[_ChatMessage]) -> str:
    chunks: list[str] = []
    for message in messages:
        label = message.role if not message.name else f"{message.role}:{message.name}"
        chunks.append(f"<|{label}|>\n{message.content}\n")
    chunks.append("<|assistant|>\n")
    return "".join(chunks)


class _PlainRolesRenderer(_ChatRenderer):
    name = "plain_roles"

    def render(self, messages: Sequence[_ChatMessage]) -> str:
        return _plain_role_transcript(messages)


class _PlaceholderRenderer(_ChatRenderer):
    def __init__(self, template: str, *, name: str) -> None:
        if "{messages}" not in template:
            raise NativeServingConfigurationError(
                f"Chat template {name!r} must contain the literal {{messages}} placeholder"
            )
        self.name = name
        self._template = template

    def render(self, messages: Sequence[_ChatMessage]) -> str:
        return self._template.replace("{messages}", _plain_role_transcript(messages))


class _NativeChatRendererAdapter(_ChatRenderer):
    def __init__(self, renderer: Any) -> None:
        self.name = str(renderer.name)
        self._renderer = renderer

    def render(self, messages: Sequence[_ChatMessage]) -> str:
        converted = tuple(
            NativeChatMessage(role=message.role, content=message.content, name=message.name)
            for message in messages
        )
        return self._renderer.render(converted, include_assistant_prompt=True)


def _load_chat_renderer(manifest: Mapping[str, Any], selection: str) -> _ChatRenderer:
    artifact_root = manifest.get("__artifact_root__")
    try:
        resolution = resolve_native_chat_renderer(
            manifest,
            selection,
            allow_auto_fallback=False,
            artifact_root=Path(str(artifact_root)) if artifact_root else None,
        )
    except NativeChatConfigurationError as exc:
        raise NativeServingConfigurationError(str(exc)) from exc
    return _NativeChatRendererAdapter(resolution.renderer)


def _read_manifest(artifact: Path) -> tuple[Path, dict[str, Any]]:
    resolved = artifact.expanduser().resolve()
    manifest_path = (
        resolved / "native-execution-manifest.json" if resolved.is_dir() else resolved
    )
    if not manifest_path.is_file():
        raise NativeServingConfigurationError(
            f"Native Execution manifest does not exist: {manifest_path}"
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise NativeServingConfigurationError(
            f"Native Execution manifest is not valid JSON: {manifest_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise NativeServingConfigurationError("Native Execution manifest root must be an object")
    if payload.get("schema") != _MANIFEST_SCHEMA or payload.get("version") != _MANIFEST_VERSION:
        raise NativeServingConfigurationError(
            "Serving requires NeuralFn Native Execution manifest schema/version 1"
        )
    return manifest_path, payload


def _context_limit(manifest: Mapping[str, Any]) -> int:
    raw = manifest.get("context_limits")
    value = raw.get("max_context_tokens") if isinstance(raw, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise NativeServingConfigurationError(
            "Serving requires a positive context_limits.max_context_tokens in the manifest"
        )
    return value


def _require_serving_capability(manifest: Mapping[str, Any]) -> None:
    capabilities = manifest.get("capabilities")
    if not isinstance(capabilities, Mapping) or capabilities.get("serve") is not True:
        raise NativeServingConfigurationError(
            "Artifact does not prove capabilities.serve=true; rebuild or migrate it "
            "through a verified resident-serving adapter before using --serve"
        )
    model = manifest.get("model")
    if not isinstance(model, Mapping) or model.get("text_generation") is not True:
        raise NativeServingConfigurationError(
            "Artifact is not a text-generation model and cannot appear in the "
            "chat/model-serving catalog"
        )


@dataclass(frozen=True, slots=True)
class _PreparedChat:
    model: str
    messages: tuple[_ChatMessage, ...]
    prompt: str
    prompt_token_ids: tuple[int, ...]
    generation: GenerationConfig
    stream: bool
    include_usage: bool
    media_batch: Any | None = None


@dataclass(frozen=True, slots=True)
class _CompletedChat:
    native: GenerationResult
    text: str


class NativeServingRuntime:
    """One loaded resident model plus validated text presentation metadata."""

    def __init__(
        self,
        *,
        model: NativeInferenceModel,
        manifest: dict[str, Any],
        codec: _TextCodec,
        renderer: _ChatRenderer,
        served_model_name: str,
        context_limit: int,
        max_output_tokens: int,
        state_store: NativeStateStore | None = None,
        prefix_cache_capacity: int = 0,
        created: int | None = None,
        chat_template_selection: str = "auto",
    ) -> None:
        self.model = model
        self.manifest = manifest
        self.codec = codec
        self.renderer = renderer
        self.served_model_name = served_model_name
        self.context_limit = context_limit
        self.max_output_tokens = max_output_tokens
        self.state_store = state_store
        self._responses_transition_lock = (
            state_store._responses_transition_lock
            if state_store is not None
            else threading.RLock()
        )
        if (
            isinstance(prefix_cache_capacity, bool)
            or not isinstance(prefix_cache_capacity, int)
            or prefix_cache_capacity < 0
        ):
            raise ValueError("prefix_cache_capacity must be a non-negative integer")
        self.created = int(time.time()) if created is None else int(created)
        self.chat_template_selection = str(chat_template_selection).strip() or "auto"
        effective_capabilities = model.capabilities
        if effective_capabilities.function_tools:
            chat_metadata = manifest.get("chat_template")
            tool_template = (
                chat_metadata.get("tool_template")
                if isinstance(chat_metadata, Mapping)
                else None
            )
            if not (
                isinstance(tool_template, Mapping)
                and set(tool_template) == {"version", "profile"}
                and type(tool_template.get("version")) is int
                and tool_template.get("version") == 1
                and tool_template.get("profile")
                == "responses-forced-function-call-v1"
            ):
                effective_capabilities = replace(
                    effective_capabilities,
                    function_tools=False,
                )
        if (
            effective_capabilities.structured_output
            and self.chat_template_selection.lower() != "auto"
        ):
            effective_capabilities = replace(
                effective_capabilities,
                function_tools=False,
                structured_output=False,
            )
        if effective_capabilities.structured_output:
            stats = model.stats()
            vocab_size = stats.get("vocab_size") if isinstance(stats, Mapping) else None
            if (
                isinstance(vocab_size, bool)
                or not isinstance(vocab_size, int)
                or vocab_size <= 0
            ):
                raise NativeServingConfigurationError(
                    "Constrained Responses requires an exact positive model vocab_size"
                )
            try:
                compile_single_byte_token_inventory(codec, vocab_size)
            except NativeConstrainedError as exc:
                raise NativeServingConfigurationError(
                    "Constrained Responses tokenizer preflight failed: " + str(exc)
                ) from exc
        self._capabilities = effective_capabilities
        self._prefix_cache: NativePrefixCache | None = None
        if prefix_cache_capacity:
            if state_store is None:
                raise NativeServingConfigurationError(
                    "Resident prefix caching requires a private state_db"
                )
            stats = model.stats()
            if not isinstance(stats, Mapping):
                raise NativeServingConfigurationError(
                    "Resident prefix caching requires exact model cache statistics"
                )
            effective_cache = stats.get("effective_cache")
            tile_config = getattr(model, "_tile_attention_config", None)
            if tile_config is not None:
                raise NativeServingConfigurationError(
                    "Resident prefix caching rejects Tile-CUDA TurboQuant attention"
                )
            if effective_cache == "full":
                cache_ready = effective_capabilities.session_prefix_cow is True
            elif effective_cache == "turboquant":
                kv_config = getattr(model, "_kv_cache", None)
                backend = getattr(
                    kv_config,
                    "turboquant_attention_backend",
                    stats.get("turboquant_attention_backend"),
                )
                cache_ready = bool(
                    backend == "cpu"
                    and effective_capabilities.session_prefix_cow_cpu_turboquant
                    is True
                )
            else:
                cache_ready = False
            if not cache_ready:
                raise NativeServingConfigurationError(
                    "Resident prefix caching requires jointly proven full-cache "
                    "session prefix COW or dense CPU TurboQuant session prefix COW"
                )
            self._prefix_cache = NativePrefixCache(
                model,
                capacity=prefix_cache_capacity,
            )

    def _prefix_cache_stats(self) -> dict[str, Any] | None:
        cache = self._prefix_cache
        return None if cache is None else cache.stats()

    @classmethod
    def load(
        cls,
        config: NativeServeConfig,
        *,
        binding: Any | None = None,
    ) -> "NativeServingRuntime":
        if config.prefix_cache_capacity and config.state_db is None:
            raise NativeServingConfigurationError(
                "--prefix-cache-capacity requires --state-db"
            )
        if (
            config.prefix_cache_capacity
            and config.kv_cache.turboquant_attention_backend == "tile-cuda"
        ):
            raise NativeServingConfigurationError(
                "Resident prefix caching rejects Tile-CUDA TurboQuant attention"
            )
        manifest_path, manifest = _read_manifest(config.artifact)
        presentation_manifest = dict(manifest)
        presentation_manifest["__artifact_root__"] = str(manifest_path.parent)
        codec = _load_text_codec(presentation_manifest)
        renderer = _load_chat_renderer(presentation_manifest, config.chat_template)
        context_limit = _context_limit(manifest)
        _require_serving_capability(manifest)
        model_metadata = manifest.get("model")
        if not isinstance(model_metadata, Mapping):
            raise NativeServingConfigurationError("Serving requires model metadata in the manifest")
        name = str(
            config.served_model_name
            or model_metadata.get("name")
            or model_metadata.get("family")
            or config.artifact.stem
        ).strip()
        if not name:
            raise NativeServingConfigurationError("Unable to derive a served model name")
        state_store = NativeStateStore(config.state_db) if config.state_db is not None else None
        try:
            model = NativeInferenceModel.load(
                config.artifact,
                binding=binding,
                kv_cache=config.kv_cache,
                load_config=config.model_load,
            )
        except BaseException:
            if state_store is not None:
                state_store.close()
            raise
        try:
            return cls(
                model=model,
                manifest=manifest,
                codec=codec,
                renderer=renderer,
                served_model_name=name,
                context_limit=context_limit,
                max_output_tokens=config.max_output_tokens,
                state_store=state_store,
                prefix_cache_capacity=config.prefix_cache_capacity,
                chat_template_selection=config.chat_template,
            )
        except BaseException:
            model.close()
            if state_store is not None:
                state_store.close()
            raise

    @property
    def capabilities(self) -> NativeInferenceCapabilities:
        return self._capabilities

    def model_object(self) -> dict[str, Any]:
        return {
            "id": self.served_model_name,
            "object": "model",
            "created": self.created,
            "owned_by": "neuralfn",
        }

    def prepare_chat(self, payload: Mapping[str, Any]) -> _PreparedChat:
        model = payload.get("model")
        if not isinstance(model, str) or not model:
            raise OpenAIAPIError(400, "'model' is required.", param="model", code="invalid_model")
        if model != self.served_model_name:
            raise OpenAIAPIError(
                404,
                f"The model {model!r} does not exist.",
                param="model",
                code="model_not_found",
            )
        unsupported_fields = sorted(set(payload) - _SUPPORTED_CHAT_FIELDS)
        if unsupported_fields:
            field_name = unsupported_fields[0]
            raise OpenAIAPIError(
                400,
                f"Chat Completions field {field_name!r} is not supported by this resident model.",
                param=field_name,
                code="unsupported_feature",
            )
        n = payload.get("n", 1)
        if isinstance(n, bool) or not isinstance(n, int) or n != 1:
            raise OpenAIAPIError(
                400,
                "This server supports exactly one choice (n=1).",
                param="n",
                code="unsupported_feature",
            )
        messages = _parse_messages(payload.get("messages"))
        media_batch = None
        image_sources = tuple(
            source for message in messages for source in message.image_sources
        )
        if image_sources:
            if not self._capabilities.vision:
                raise OpenAIAPIError(
                    400,
                    "This resident artifact does not have a loaded, jointly proven Muse Glimmer image encoder.",
                    param="messages",
                    code="unsupported_feature",
                )
            try:
                from .native_glimmer_media import (
                    NativeMuseGlimmerMediaError,
                    prepare_images_for_model,
                )

                media_batch = prepare_images_for_model(self.model, image_sources)
            except (NativeMuseGlimmerMediaError, OSError) as exc:
                raise OpenAIAPIError(
                    400,
                    f"Unable to preprocess Muse Glimmer image input: {exc}",
                    param="messages",
                    code="invalid_image",
                ) from exc
            fragments = iter(media_batch.prompt_fragments)
            rendered_messages: list[_ChatMessage] = []
            for message in messages:
                content = message.content
                for local_index in range(len(message.image_sources)):
                    marker = f"{_IMAGE_MARKER_PREFIX}{local_index}\x00"
                    if content.count(marker) != 1:
                        raise OpenAIAPIError(
                            400,
                            "Image content marker accounting is inconsistent.",
                            param="messages",
                            code="invalid_image",
                        )
                    content = content.replace(marker, next(fragments), 1)
                rendered_messages.append(
                    _ChatMessage(role=message.role, content=content, name=message.name)
                )
            messages = tuple(rendered_messages)

        stream = payload.get("stream", False)
        if not isinstance(stream, bool):
            raise OpenAIAPIError(400, "'stream' must be a boolean.", param="stream")
        include_usage = False
        stream_options = payload.get("stream_options")
        if stream_options is not None:
            if not stream:
                raise OpenAIAPIError(
                    400,
                    "'stream_options' may only be set when stream=true.",
                    param="stream_options",
                )
            if not isinstance(stream_options, Mapping):
                raise OpenAIAPIError(400, "'stream_options' must be an object.", param="stream_options")
            unknown = set(stream_options) - {"include_usage"}
            if unknown:
                field_name = sorted(unknown)[0]
                raise OpenAIAPIError(
                    400,
                    f"stream_options.{field_name} is not supported.",
                    param=f"stream_options.{field_name}",
                    code="unsupported_feature",
                )
            include_usage = stream_options.get("include_usage", False)
            if not isinstance(include_usage, bool):
                raise OpenAIAPIError(
                    400,
                    "stream_options.include_usage must be a boolean.",
                    param="stream_options.include_usage",
                )

        max_new_tokens = payload.get("max_completion_tokens")
        legacy_max = payload.get("max_tokens")
        if max_new_tokens is not None and legacy_max is not None:
            raise OpenAIAPIError(
                400,
                "Specify only one of 'max_completion_tokens' or 'max_tokens'.",
                param="max_completion_tokens",
            )
        if max_new_tokens is None:
            max_new_tokens = legacy_max
        if max_new_tokens is None:
            max_new_tokens = min(16, self.max_output_tokens)
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or max_new_tokens <= 0
        ):
            raise OpenAIAPIError(
                400,
                "Maximum completion tokens must be a positive integer.",
                param="max_completion_tokens",
            )
        if max_new_tokens > self.max_output_tokens:
            raise OpenAIAPIError(
                400,
                f"Maximum completion tokens cannot exceed {self.max_output_tokens}.",
                param="max_completion_tokens",
                code="max_tokens_exceeded",
            )
        try:
            generation = GenerationConfig(
                max_new_tokens=max_new_tokens,
                temperature=payload.get("temperature", 0.8),
                top_p=payload.get("top_p", 1.0),
                seed=payload.get("seed"),
            )
        except (TypeError, ValueError) as exc:
            raise OpenAIAPIError(400, str(exc), code="invalid_parameter") from exc

        prompt = self.renderer.render(messages)
        try:
            prompt_token_ids = self.codec.encode(prompt)
        except Exception as exc:
            raise OpenAIAPIError(
                400,
                f"Unable to tokenize rendered chat input: {exc}",
                param="messages",
                code="invalid_prompt",
            ) from exc
        if not prompt_token_ids:
            raise OpenAIAPIError(400, "Rendered chat input is empty.", param="messages")
        if len(prompt_token_ids) + max_new_tokens > self.context_limit:
            raise OpenAIAPIError(
                400,
                "This model's maximum context length is "
                f"{self.context_limit} tokens, but the request uses {len(prompt_token_ids)} prompt "
                f"tokens plus {max_new_tokens} requested completion tokens.",
                param="messages",
                code="context_length_exceeded",
            )
        return _PreparedChat(
            model=model,
            messages=messages,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            generation=generation,
            stream=stream,
            include_usage=include_usage,
            media_batch=media_batch,
        )

    def complete(
        self,
        request: _PreparedChat,
        *,
        on_token: Callable[[GenerationEvent], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> _CompletedChat:
        with self.model.create_session() as session:
            if request.media_batch is None:
                session.prefill(request.prompt_token_ids)
            else:
                embeddings = self.model.encode_media(
                    request.media_batch.packed_patches,
                    request.media_batch.grid_thw,
                )
                positions = request.media_batch.replacement_positions(
                    request.prompt_token_ids
                )
                if len(embeddings) != len(positions):
                    raise NativeInferenceCapabilityError(
                        "Resident image rows do not match rendered patch placeholders"
                    )
                session.prefill_with_embeddings(
                    request.prompt_token_ids,
                    replacement_positions=positions,
                    replacement_embeddings=embeddings,
                )

            def committed(event: GenerationEvent) -> None:
                if on_token is not None:
                    on_token(event)
                if cancel_event is not None and cancel_event.is_set():
                    session.cancel()

            if cancel_event is not None and cancel_event.is_set():
                session.cancel()
            result = session.decode(request.generation, on_token=committed)
            return _CompletedChat(native=result, text=self.codec.decode(result.token_ids))

    def close(self) -> None:
        first_error: BaseException | None = None
        if self._prefix_cache is not None:
            try:
                cache_stats = self._prefix_cache.stats()
                if (
                    cache_stats.get("active_leases") != 0
                    or cache_stats.get("in_flight_forks") != 0
                ):
                    first_error = RuntimeError(
                        "Native prefix cache still owns active leases or forks at shutdown"
                    )
                self._prefix_cache.shutdown()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        try:
            self.model.close()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
        if self.state_store is not None:
            try:
                self.state_store.close()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error


def _parse_messages(raw: Any) -> tuple[_ChatMessage, ...]:
    if not isinstance(raw, list) or not raw:
        raise OpenAIAPIError(
            400,
            "'messages' must be a non-empty array.",
            param="messages",
            code="invalid_messages",
        )
    messages: list[_ChatMessage] = []
    for index, item in enumerate(raw):
        param = f"messages.{index}"
        if not isinstance(item, Mapping):
            raise OpenAIAPIError(400, f"{param} must be an object.", param=param)
        unknown_message_fields = sorted(set(item) - _SUPPORTED_MESSAGE_FIELDS)
        if unknown_message_fields:
            field_name = unknown_message_fields[0]
            raise OpenAIAPIError(
                400,
                f"{param}.{field_name} is not supported by this bounded server.",
                param=f"{param}.{field_name}",
                code="unsupported_feature",
            )
        role = item.get("role")
        if role not in _TEXT_ROLES:
            raise OpenAIAPIError(
                400,
                f"{param}.role {role!r} is not supported; tool roles are unavailable.",
                param=f"{param}.role",
                code="unsupported_feature",
            )
        content = item.get("content")
        image_sources: list[str] = []
        if isinstance(content, str):
            if _IMAGE_MARKER_PREFIX in content:
                raise OpenAIAPIError(400, f"{param}.content contains a reserved marker.", param=param)
            text = content
        elif isinstance(content, list) and content:
            parts: list[str] = []
            for part_index, part in enumerate(content):
                part_param = f"{param}.content.{part_index}"
                if not isinstance(part, Mapping):
                    raise OpenAIAPIError(
                        400,
                        f"{part_param} must be an object.",
                        param=part_param,
                    )
                part_type = part.get("type")
                if part_type == "text":
                    unknown_part_fields = sorted(set(part) - _SUPPORTED_TEXT_PART_FIELDS)
                    if unknown_part_fields:
                        field_name = unknown_part_fields[0]
                        raise OpenAIAPIError(
                            400,
                            f"{part_param}.{field_name} is not supported.",
                            param=f"{part_param}.{field_name}",
                            code="unsupported_feature",
                        )
                    text_part = part.get("text")
                    if not isinstance(text_part, str):
                        raise OpenAIAPIError(
                            400,
                            f"{part_param}.text must be a string.",
                            param=part_param,
                        )
                    if _IMAGE_MARKER_PREFIX in text_part:
                        raise OpenAIAPIError(
                            400,
                            f"{part_param}.text contains a reserved marker.",
                            param=part_param,
                        )
                    parts.append(text_part)
                elif part_type == "image_url":
                    if role != "user":
                        raise OpenAIAPIError(
                            400,
                            f"{part_param} is only supported on user messages.",
                            param=part_param,
                            code="unsupported_feature",
                        )
                    unknown_part_fields = sorted(set(part) - _SUPPORTED_IMAGE_PART_FIELDS)
                    image_url = part.get("image_url")
                    if unknown_part_fields or not isinstance(image_url, Mapping):
                        raise OpenAIAPIError(
                            400,
                            f"{part_param} has an invalid image_url contract.",
                            param=part_param,
                        )
                    unknown_url_fields = sorted(
                        set(image_url) - _SUPPORTED_IMAGE_URL_FIELDS
                    )
                    url = image_url.get("url")
                    detail = image_url.get("detail", "auto")
                    if (
                        unknown_url_fields
                        or not isinstance(url, str)
                        or not url
                        or detail not in {"auto", "high"}
                    ):
                        raise OpenAIAPIError(
                            400,
                            f"{part_param}.image_url must provide a data URL and auto/high detail.",
                            param=f"{part_param}.image_url",
                        )
                    parts.append(
                        f"{_IMAGE_MARKER_PREFIX}{len(image_sources)}\x00"
                    )
                    image_sources.append(url)
                else:
                    raise OpenAIAPIError(
                        400,
                        f"{part_param} is not a supported text/image content part.",
                        param=part_param,
                        code="unsupported_feature",
                    )
            text = "".join(parts)
        else:
            raise OpenAIAPIError(
                400,
                f"{param}.content must be text.",
                param=f"{param}.content",
                code="unsupported_feature",
            )
        name = item.get("name")
        if name is not None and (not isinstance(name, str) or not name):
            raise OpenAIAPIError(400, f"{param}.name must be a non-empty string.", param=f"{param}.name")
        messages.append(
            _ChatMessage(
                role=role,
                content=text,
                name=name,
                image_sources=tuple(image_sources),
            )
        )
    return tuple(messages)


class _QueueTicket:
    def __init__(self, queue: "BoundedSingleWorkerQueue") -> None:
        self._queue = queue
        self._state = "reserved"

    def run(self, function: Callable[[], Any]) -> Coroutine[Any, Any, Any]:
        # Submit synchronously before returning the waiter coroutine.  A caller
        # may cancel that coroutine before its first event-loop turn, but the
        # worker still owns and eventually releases the reservation.
        return self._queue._submit_reserved(self, function)

    def release(self) -> bool:
        """Idempotently refund this ticket if it has not been submitted."""

        return self._queue._release_unused(self)


class BoundedSingleWorkerQueue:
    """One compute worker with an explicit non-blocking admission bound."""

    def __init__(
        self,
        waiting_capacity: int,
        *,
        session_limit: int | None = None,
    ) -> None:
        if (
            isinstance(waiting_capacity, bool)
            or not isinstance(waiting_capacity, int)
            or waiting_capacity < 0
        ):
            raise ValueError("waiting_capacity must be a non-negative integer")
        if session_limit is None:
            session_limit = waiting_capacity + 1
        elif (
            isinstance(session_limit, bool)
            or not isinstance(session_limit, int)
            or session_limit <= 0
        ):
            raise ValueError("session_limit must be a positive integer")
        self.waiting_capacity = waiting_capacity
        self.session_limit = session_limit
        self._slots = threading.BoundedSemaphore(waiting_capacity + 1)
        self._session_slots = threading.BoundedSemaphore(session_limit)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nfn-resident")
        self._lock = threading.Lock()
        self._queued = 0
        self._running = 0
        self._session_reservations = 0
        self._accepted = 0
        self._rejected = 0
        self._queue_rejected = 0
        self._session_rejected = 0
        self._closed = False
        self._close_complete = threading.Event()
        self._close_failure: BaseException | None = None
        self._active_worker_thread_id: int | None = None
        self._unused_tickets: set[_QueueTicket] = set()

    def admit(self) -> tuple[_QueueTicket | None, str | None]:
        with self._lock:
            if self._closed:
                return None, "queue_saturated"
        if not self._slots.acquire(blocking=False):
            with self._lock:
                self._rejected += 1
                self._queue_rejected += 1
            return None, "queue_saturated"
        if not self._session_slots.acquire(blocking=False):
            self._slots.release()
            with self._lock:
                self._rejected += 1
                self._session_rejected += 1
            return None, "session_limit_exceeded"
        with self._lock:
            if self._closed:
                self._session_slots.release()
                self._slots.release()
                return None, "queue_saturated"
            self._queued += 1
            self._session_reservations += 1
            self._accepted += 1
            ticket = _QueueTicket(self)
            self._unused_tickets.add(ticket)
        return ticket, None

    def reserve(self) -> _QueueTicket | None:
        ticket, _reason = self.admit()
        return ticket

    def _release_reserved_locked(self, ticket: _QueueTicket) -> None:
        ticket._state = "released"
        self._unused_tickets.discard(ticket)
        self._queued -= 1
        self._session_reservations -= 1
        self._session_slots.release()
        self._slots.release()

    def _release_unused(self, ticket: _QueueTicket) -> bool:
        with self._lock:
            if ticket._state != "reserved":
                return False
            self._release_reserved_locked(ticket)
            return True

    def _submit_reserved(
        self,
        ticket: _QueueTicket,
        function: Callable[[], Any],
    ) -> Coroutine[Any, Any, Any]:
        def execute() -> tuple[bool, Any]:
            with self._lock:
                ticket._state = "running"
                self._queued -= 1
                self._running += 1
                self._active_worker_thread_id = threading.get_ident()
            try:
                try:
                    return True, function()
                except BaseException as exc:
                    # Keep worker failures out of the concurrent future itself.
                    # Some supported Python runtimes can otherwise leave the next
                    # submission completed in the worker but unwoken in asyncio.
                    return False, exc
            finally:
                with self._lock:
                    ticket._state = "released"
                    self._running -= 1
                    self._session_reservations -= 1
                    self._active_worker_thread_id = None
                self._session_slots.release()
                self._slots.release()

        with self._lock:
            if ticket._state != "reserved":
                raise RuntimeError("Generation queue ticket was already used or released")
            if self._closed:
                self._release_reserved_locked(ticket)
                raise RuntimeError("Generation queue is closed")
            try:
                worker_future = self._executor.submit(execute)
            except BaseException:
                self._release_reserved_locked(ticket)
                raise
            ticket._state = "submitted"
            self._unused_tickets.discard(ticket)

        async def wait_for_worker() -> Any:
            # Poll the thread-safe future instead of depending on repeated
            # call_soon_threadsafe wakeups from the resident worker. Python 3.13
            # runtimes in the supported serving environment can lose the wakeup
            # for a later submission even though its native work has completed.
            while not worker_future.done():
                await asyncio.sleep(0.001)
            succeeded, result = worker_future.result()
            if succeeded:
                return result
            if not isinstance(result, BaseException):  # pragma: no cover - defensive invariant
                raise RuntimeError("Generation worker returned an invalid failure outcome")
            raise result

        return wait_for_worker()

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "workers": 1,
                "waiting_capacity": self.waiting_capacity,
                "session_limit": self.session_limit,
                "session_reservations": self._session_reservations,
                "queued": self._queued,
                "running": self._running,
                "accepted": self._accepted,
                "rejected": self._rejected,
                "queue_rejected": self._queue_rejected,
                "session_rejected": self._session_rejected,
            }

    def close(self) -> None:
        owns_close = False
        with self._lock:
            if self._active_worker_thread_id == threading.get_ident():
                raise RuntimeError("Generation queue cannot be closed from its resident worker")
            if not self._closed:
                self._closed = True
                owns_close = True
                for ticket in tuple(self._unused_tickets):
                    self._release_reserved_locked(ticket)

        if not owns_close:
            # `_closed` rejects new admission, but it does not mean the worker
            # has drained. Every concurrent closer joins the first close so a
            # successful return always means submitted work has finished.
            self._close_complete.wait()
            with self._lock:
                failure = self._close_failure
            if failure is not None:
                raise failure
            return

        try:
            # Submitted work owns its reservation and must reach execute()'s
            # finally block. Cancelling queued futures would bypass that cleanup.
            self._executor.shutdown(wait=True, cancel_futures=False)
        except BaseException as exc:
            with self._lock:
                self._close_failure = exc
            raise
        finally:
            self._close_complete.set()

    async def aclose(self) -> None:
        """Close off-loop without relying on a cross-thread asyncio wakeup."""

        failures: list[BaseException] = []
        cancellation: asyncio.CancelledError | None = None

        def close_in_thread() -> None:
            try:
                self.close()
            except BaseException as exc:  # pragma: no cover - defensive propagation
                failures.append(exc)

        closer = threading.Thread(
            target=close_in_thread,
            name="nfn-resident-close",
        )
        closer.start()
        # Poll for the same Python 3.13 serving environment reason used by the
        # worker waiter above: a completed thread-safe future can lose its loop
        # wakeup even though the underlying work is finished.
        while closer.is_alive():
            try:
                await asyncio.sleep(0.001)
            except asyncio.CancelledError as exc:
                # Shutdown is a resource-safety boundary. Finish joining the
                # one close operation, then preserve the caller's cancellation.
                cancellation = exc
        closer.join()
        if cancellation is not None:
            raise cancellation
        if failures:
            raise failures[0]


class _TicketStreamingResponse(StreamingResponse):
    """Release an unsubmitted ticket even if streaming never begins."""

    def __init__(self, content: Any, *, ticket: _QueueTicket, **kwargs: Any) -> None:
        super().__init__(content, **kwargs)
        self._ticket = ticket

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._ticket.release()


def _completion_id() -> str:
    return "chatcmpl-" + secrets.token_urlsafe(18).replace("-", "").replace("_", "")


def _usage(prepared: _PreparedChat, result: GenerationResult) -> dict[str, int]:
    prompt_tokens = len(prepared.prompt_token_ids)
    completion_tokens = result.completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _chat_completion_payload(
    runtime: NativeServingRuntime,
    prepared: _PreparedChat,
    completed: _CompletedChat,
    *,
    completion_id: str,
    created: int,
) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": runtime.served_model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": completed.text,
                    "refusal": None,
                    "annotations": [],
                },
                "logprobs": None,
                "finish_reason": completed.native.finish_reason,
            }
        ],
        "usage": _usage(prepared, completed.native),
        "system_fingerprint": "nfn-resident-abi1",
    }


def _sse(payload: Mapping[str, Any]) -> str:
    return "data: " + json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n\n"


def _stream_chunk(
    runtime: NativeServingRuntime,
    *,
    completion_id: str,
    created: int,
    delta: Mapping[str, Any],
    finish_reason: str | None,
    usage: Mapping[str, int] | None = None,
    choices: bool = True,
    include_usage: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": runtime.served_model_name,
        "system_fingerprint": "nfn-resident-abi1",
        "choices": (
            [
                {
                    "index": 0,
                    "delta": dict(delta),
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ]
            if choices
            else []
        ),
    }
    if include_usage:
        payload["usage"] = dict(usage) if usage is not None else None
    return payload


def _response_stream_event(
    event_type: str,
    sequence_number: int,
    **payload: Any,
) -> dict[str, Any]:
    return {
        "type": event_type,
        "sequence_number": sequence_number,
        **payload,
    }


async def _json_request_object(request: Request) -> Mapping[str, Any]:
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared_bytes = int(content_length)
        except ValueError:
            declared_bytes = 0
        if declared_bytes > _MAX_STATEFUL_REQUEST_BYTES:
            raise OpenAIAPIError(
                413,
                "Stateful request body exceeds the 1 MiB limit.",
                code="request_too_large",
            )
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > _MAX_STATEFUL_REQUEST_BYTES:
            raise OpenAIAPIError(
                413,
                "Stateful request body exceeds the 1 MiB limit.",
                code="request_too_large",
            )
        body.extend(chunk)
    try:
        payload = json.loads(bytes(body))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise OpenAIAPIError(400, "Request body must be valid JSON.", code="invalid_json") from exc
    if not isinstance(payload, Mapping):
        raise OpenAIAPIError(400, "Request body must be a JSON object.", code="invalid_json")
    return payload


def _response_include_query(request: Request) -> tuple[str, ...]:
    values = (
        *request.query_params.getlist("include"),
        *request.query_params.getlist("include[]"),
    )
    for value in values:
        if value not in _RESPONSE_INCLUDE_VALUES:
            raise NativeResponsesAPIError(
                400,
                f"'include' value {value!r} is not supported.",
                param="include",
                code="invalid_parameter",
            )
    return tuple(values)


def _strict_query_boolean(request: Request, field: str, *, default: bool) -> bool:
    values = request.query_params.getlist(field)
    if len(values) > 1:
        raise NativeResponsesAPIError(
            400,
            f"Query parameter {field!r} may be supplied only once.",
            param=field,
        )
    if not values:
        return default
    if values[0] not in {"true", "false"}:
        raise NativeResponsesAPIError(
            400,
            f"'{field}' must be 'true' or 'false'.",
            param=field,
        )
    return values[0] == "true"


def _response_retrieve_query(request: Request) -> _ResponseRetrieveQuery:
    supported = {
        "include",
        "include[]",
        "include_obfuscation",
        "starting_after",
        "stream",
    }
    unknown = sorted(set(request.query_params) - supported)
    if unknown:
        field = unknown[0]
        raise NativeResponsesAPIError(
            400,
            f"Query parameter {field!r} is not supported by this resident text model.",
            param=field,
            code="unsupported_feature",
        )
    stream = _strict_query_boolean(request, "stream", default=False)
    include_obfuscation = _strict_query_boolean(
        request,
        "include_obfuscation",
        default=True,
    )
    starting_values = request.query_params.getlist("starting_after")
    if len(starting_values) > 1:
        raise NativeResponsesAPIError(
            400,
            "Query parameter 'starting_after' may be supplied only once.",
            param="starting_after",
        )
    starting_after: int | None = None
    if starting_values:
        try:
            starting_after = int(starting_values[0])
        except (TypeError, ValueError) as exc:
            raise NativeResponsesAPIError(
                400,
                "'starting_after' must be a non-negative integer.",
                param="starting_after",
            ) from exc
        if (
            starting_after < 0
            or starting_after > 9_223_372_036_854_775_807
            or str(starting_after) != starting_values[0]
        ):
            raise NativeResponsesAPIError(
                400,
                "'starting_after' must be a non-negative integer.",
                param="starting_after",
            )
        if not stream:
            raise NativeResponsesAPIError(
                400,
                "'starting_after' requires 'stream: true'.",
                param="starting_after",
                code="invalid_parameter",
            )
    return _ResponseRetrieveQuery(
        stream=stream,
        starting_after=starting_after,
        include=_response_include_query(request),
        include_obfuscation=include_obfuscation,
    )


def _cursor_list_query(
    request: Request,
    *,
    allow_include: bool = False,
) -> tuple[str | None, int, str]:
    supported = {"after", "limit", "order"}
    if allow_include:
        supported.update({"include", "include[]"})
    unknown = sorted(set(request.query_params) - supported)
    if unknown:
        field = unknown[0]
        raise NativeResponsesAPIError(
            400,
            f"Query parameter {field!r} is not supported by this resident text model.",
            param=field,
            code="unsupported_feature",
        )
    for field in supported:
        if field in {"include", "include[]"}:
            continue
        if len(request.query_params.getlist(field)) > 1:
            raise NativeResponsesAPIError(
                400,
                f"Query parameter {field!r} may be supplied only once.",
                param=field,
            )
    if allow_include:
        _response_include_query(request)
    raw_limit = request.query_params.get("limit", "20")
    try:
        limit = int(raw_limit)
    except (TypeError, ValueError) as exc:
        raise NativeResponsesAPIError(
            400,
            "'limit' must be an integer between 1 and 100.",
            param="limit",
        ) from exc
    return request.query_params.get("after"), limit, request.query_params.get("order", "desc")


def create_native_inference_app(
    runtime: NativeServingRuntime,
    *,
    auth: BearerAuth | None = None,
    queue_capacity: int = 8,
    session_limit: int | None = None,
) -> FastAPI:
    """Create an isolated ASGI app around an already-loaded resident model."""

    bearer = auth or BearerAuth()
    generation_queue = BoundedSingleWorkerQueue(
        queue_capacity,
        session_limit=session_limit,
    )
    responses_service = (
        NativeResponsesService(runtime, runtime.state_store)
        if runtime.state_store is not None
        else None
    )
    background_stop = asyncio.Event()
    background_wakeup = asyncio.Event()
    foreground_drivers: set[asyncio.Task[Any]] = set()
    foreground_accepting = True

    def require_foreground_accepting() -> None:
        if not foreground_accepting:
            raise RuntimeError("Native inference app is shutting down")

    def track_foreground_driver(task: asyncio.Task[Any]) -> asyncio.Task[Any]:
        if not foreground_accepting:
            task.cancel()
            raise RuntimeError("Native inference app is shutting down")
        foreground_drivers.add(task)

        def settled(done: asyncio.Task[Any]) -> None:
            foreground_drivers.discard(done)
            if not done.cancelled():
                # Consume a detached transport waiter's exception while the
                # durable worker outcome remains independently owned.
                done.exception()

        task.add_done_callback(settled)
        return task

    async def drain_foreground_drivers() -> None:
        while foreground_drivers:
            await asyncio.gather(
                *tuple(foreground_drivers),
                return_exceptions=True,
            )

    async def background_loop() -> None:
        if responses_service is None:
            return
        state = responses_service.state
        while not background_stop.is_set():
            if not state.queued_background_jobs():
                background_wakeup.clear()
                try:
                    await asyncio.wait_for(background_wakeup.wait(), timeout=0.1)
                except asyncio.TimeoutError:
                    pass
                continue
            ticket = generation_queue.reserve()
            if ticket is None:
                await asyncio.sleep(0.025)
                continue
            try:
                claimed = state.claim_next_background_job()
                if claimed is None:
                    continue
                scope, stored_response = claimed
                try:
                    prepared = responses_service.from_stored_background(scope, stored_response)
                except NativeResponsesAPIError as exc:
                    error = {
                        "code": exc.code or "invalid_background_state",
                        "message": exc.message,
                    }
                    state.finish_background_job(
                        scope,
                        str(stored_response["id"]),
                        status="failed",
                        response_patch={"completed_at": int(time.time()), "output": []},
                        error=error,
                    )
                    continue
                except Exception:
                    error = {
                        "code": "invalid_background_state",
                        "message": "Stored background request state is invalid.",
                    }
                    state.finish_background_job(
                        scope,
                        str(stored_response["id"]),
                        status="failed",
                        response_patch={"completed_at": int(time.time()), "output": []},
                        error=error,
                    )
                    continue

                cancel_event = threading.Event()
                decoder = runtime.codec.incremental_decoder() if prepared.stream else None

                def committed(event: GenerationEvent) -> None:
                    if decoder is None:
                        return
                    fragment = decoder.push(event.token_id)
                    if fragment:
                        responses_service.append_background_stream_delta(
                            prepared,
                            fragment,
                        )

                try:
                    if prepared.stream:
                        responses_service.begin_background_stream(prepared)
                    run_task = asyncio.create_task(
                        ticket.run(
                            lambda: responses_service.execute(
                                prepared,
                                on_token=committed if prepared.stream else None,
                                cancel_event=cancel_event,
                            )
                        )
                    )
                    while not run_task.done():
                        if state.is_cancel_requested(
                            scope,
                            prepared.response_id,
                        ):
                            cancel_event.set()
                        await asyncio.sleep(0.025)
                    completed = await run_task
                    if decoder is not None:
                        tail = decoder.finish()
                        if tail:
                            responses_service.append_background_stream_delta(
                                prepared,
                                tail,
                            )
                    responses_service.finish(prepared, completed)
                except NativeResponsesAPIError:
                    # Lineage/conversation conflicts are already persisted as
                    # response.failed by the service's terminal boundary.
                    pass
                except Exception:
                    try:
                        responses_service.fail(prepared)
                    except Exception:
                        # The durable store may itself be unavailable.  The app-level
                        # exception handler cannot help a detached background task.
                        pass
            finally:
                ticket.release()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        nonlocal foreground_accepting
        background_task: asyncio.Task[None] | None = None
        if responses_service is not None:
            background_wakeup.set()
            background_task = asyncio.create_task(background_loop())
        try:
            yield
        finally:
            # No new submitted waiter may cross the empty-set observation in
            # drain_foreground_drivers and race prefix/model teardown.
            foreground_accepting = False
            background_stop.set()
            background_wakeup.set()
            try:
                if background_task is not None:
                    await background_task
            finally:
                try:
                    await drain_foreground_drivers()
                finally:
                    try:
                        await generation_queue.aclose()
                    finally:
                        runtime.close()

    app = FastAPI(
        title="NeuralFn Native Inference",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.native_runtime = runtime
    app.state.generation_queue = generation_queue
    app.state.bearer_auth = bearer
    app.state.responses_service = responses_service
    app.state.foreground_drivers = foreground_drivers

    async def authorize(request: Request) -> str:
        return bearer.require(request.headers.get("authorization"))

    def admit_generation() -> _QueueTicket:
        ticket, rejection = generation_queue.admit()
        if ticket is not None:
            return ticket
        if rejection == "session_limit_exceeded":
            raise OpenAIAPIError(
                429,
                "The resident model request-session limit is reached. "
                "Retry after another request completes.",
                error_type="rate_limit_error",
                code="session_limit_exceeded",
            )
        raise OpenAIAPIError(
            429,
            "The resident model compute queue is full. Retry after another request completes.",
            error_type="rate_limit_error",
            code="queue_saturated",
        )

    @app.exception_handler(OpenAIAPIError)
    async def openai_error_handler(_request: Request, exc: OpenAIAPIError) -> JSONResponse:
        return JSONResponse(exc.payload(), status_code=exc.status_code, headers=exc.headers)

    @app.exception_handler(NativeResponsesAPIError)
    async def responses_error_handler(
        _request: Request,
        exc: NativeResponsesAPIError,
    ) -> JSONResponse:
        return JSONResponse(exc.payload(), status_code=exc.status_code)

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
        if exc.status_code == 404:
            error = OpenAIAPIError(404, "Resource not found.", code="not_found")
        else:
            error = OpenAIAPIError(exc.status_code, str(exc.detail), code="http_error")
        return JSONResponse(error.payload(), status_code=error.status_code)

    @app.exception_handler(Exception)
    async def internal_error_handler(_request: Request, _exc: Exception) -> JSONResponse:
        error = OpenAIAPIError(
            500,
            "The server encountered an internal error while processing the request.",
            error_type="server_error",
            code="internal_error",
        )
        return JSONResponse(error.payload(), status_code=500)

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        await authorize(request)
        model_stats = runtime.model.stats()
        payload = {
            "status": "ok",
            "model": runtime.served_model_name,
            "backend": model_stats.get("backend", "resident-native"),
            "cache": {
                "requested": model_stats.get("requested_cache"),
                "effective": model_stats.get("effective_cache"),
            },
            "capabilities": runtime.capabilities.to_dict(),
            "queue": generation_queue.stats(),
        }
        if runtime.state_store is not None:
            payload["state"] = runtime.state_store.stats()
        prefix_cache_stats = runtime._prefix_cache_stats()
        if prefix_cache_stats is not None:
            payload["prefix_cache"] = prefix_cache_stats
        return payload

    @app.get("/v1/models")
    async def list_models(request: Request) -> dict[str, Any]:
        await authorize(request)
        return {"object": "list", "data": [runtime.model_object()]}

    @app.get("/v1/models/{model_id}")
    async def retrieve_model(model_id: str, request: Request) -> dict[str, Any]:
        await authorize(request)
        if model_id != runtime.served_model_name:
            raise OpenAIAPIError(
                404,
                f"The model {model_id!r} does not exist.",
                param="model",
                code="model_not_found",
            )
        return runtime.model_object()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        await authorize(request)
        try:
            payload = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise OpenAIAPIError(400, "Request body must be valid JSON.", code="invalid_json") from exc
        if not isinstance(payload, Mapping):
            raise OpenAIAPIError(400, "Request body must be a JSON object.", code="invalid_json")
        prepared = runtime.prepare_chat(payload)
        ticket = admit_generation()
        try:
            completion_id = _completion_id()
            created = int(time.time())
        except BaseException:
            ticket.release()
            raise

        if not prepared.stream:
            try:
                completed = await ticket.run(lambda: runtime.complete(prepared))
            except OpenAIAPIError:
                raise
            except Exception as exc:
                raise OpenAIAPIError(
                    500,
                    "Resident native generation failed.",
                    error_type="server_error",
                    code="generation_failed",
                ) from exc
            finally:
                ticket.release()
            return _chat_completion_payload(
                runtime,
                prepared,
                completed,
                completion_id=completion_id,
                created=created,
            )

        async def stream_events():
            loop = asyncio.get_running_loop()
            events: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
            cancel_event = threading.Event()
            decoder = runtime.codec.incremental_decoder()

            def committed(event: GenerationEvent) -> None:
                fragment = decoder.push(event.token_id)
                if fragment:
                    queued = asyncio.run_coroutine_threadsafe(
                        events.put(("token", fragment)),
                        loop,
                    )
                    queued.result()

            async def drive() -> None:
                try:
                    completed = await ticket.run(
                        lambda: runtime.complete(
                            prepared,
                            on_token=committed,
                            cancel_event=cancel_event,
                        )
                    )
                    tail = decoder.finish()
                    if tail:
                        await events.put(("token", tail))
                    await events.put(("done", completed))
                except Exception as exc:
                    await events.put(("error", exc))
                finally:
                    ticket.release()

            require_foreground_accepting()
            driver = track_foreground_driver(asyncio.create_task(drive()))
            try:
                yield _sse(
                    _stream_chunk(
                        runtime,
                        completion_id=completion_id,
                        created=created,
                        delta={"role": "assistant", "content": ""},
                        finish_reason=None,
                        include_usage=prepared.include_usage,
                    )
                )
                while True:
                    if await request.is_disconnected():
                        cancel_event.set()
                    try:
                        kind, value = await asyncio.wait_for(events.get(), timeout=0.1)
                    except asyncio.TimeoutError:
                        continue
                    if kind == "token":
                        yield _sse(
                            _stream_chunk(
                                runtime,
                                completion_id=completion_id,
                                created=created,
                                delta={"content": value},
                                finish_reason=None,
                                include_usage=prepared.include_usage,
                            )
                        )
                        continue
                    if kind == "error":
                        error = OpenAIAPIError(
                            500,
                            "Resident native generation failed during streaming.",
                            error_type="server_error",
                            code="generation_failed",
                        )
                        yield _sse(error.payload())
                        yield "data: [DONE]\n\n"
                        break
                    completed = value
                    yield _sse(
                        _stream_chunk(
                            runtime,
                            completion_id=completion_id,
                            created=created,
                            delta={},
                            finish_reason=completed.native.finish_reason,
                            include_usage=prepared.include_usage,
                        )
                    )
                    if prepared.include_usage:
                        yield _sse(
                            _stream_chunk(
                                runtime,
                                completion_id=completion_id,
                                created=created,
                                delta={},
                                finish_reason=None,
                                usage=_usage(prepared, completed.native),
                                choices=False,
                                include_usage=True,
                            )
                        )
                    yield "data: [DONE]\n\n"
                    break
            finally:
                cancel_event.set()
                ticket.release()
                # Lifespan owns and drains the registered driver even when the
                # transport disappears before native generation settles.

        try:
            return _TicketStreamingResponse(
                stream_events(),
                ticket=ticket,
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        except BaseException:
            ticket.release()
            raise

    if responses_service is not None:

        def start_response_driver(
            prepared: PreparedNativeResponse,
            initial_response: Mapping[str, Any],
            ticket: _QueueTicket,
        ) -> tuple[asyncio.Queue[tuple[str, Any]], threading.Event]:
            """Eagerly register durable foreground work before returning SSE."""

            loop = asyncio.get_running_loop()
            events: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
            cancel_event = threading.Event()
            decoder = runtime.codec.incremental_decoder()

            def committed(event: GenerationEvent) -> None:
                fragment = decoder.push(event.token_id)
                if fragment:
                    queued = asyncio.run_coroutine_threadsafe(
                        events.put(("token", fragment)),
                        loop,
                    )
                    queued.result()

            async def drive() -> None:
                try:
                    completed, response = await ticket.run(
                        lambda: responses_service._execute_and_finish_resident(
                            prepared,
                            on_token=committed,
                            cancel_event=cancel_event,
                        )
                    )
                    tail = decoder.finish()
                    if tail:
                        await events.put(("token", tail))
                    await events.put(("done", (completed, response)))
                except NativeResponsesAPIError as exc:
                    try:
                        failed = responses_service.retrieve_response(
                            prepared.scope,
                            prepared.response_id,
                        )
                    except Exception:
                        failed = {
                            **dict(initial_response),
                            "status": "failed",
                            "error": {
                                "code": exc.code,
                                "message": exc.message,
                            },
                        }
                    await events.put(("failed", failed))
                except Exception:
                    try:
                        failed = responses_service.retrieve_response(
                            prepared.scope,
                            prepared.response_id,
                        )
                    except Exception:
                        failed = {
                            **dict(initial_response),
                            "status": "failed",
                            "error": {
                                "code": "generation_failed",
                                "message": "Resident native generation failed.",
                            },
                        }
                    await events.put(("failed", failed))
                finally:
                    ticket.release()

            require_foreground_accepting()
            track_foreground_driver(asyncio.create_task(drive()))
            return events, cancel_event

        async def response_sse(
            request: Request,
            prepared: PreparedNativeResponse,
            initial_response: Mapping[str, Any],
            ticket: _QueueTicket,
            events: asyncio.Queue[tuple[str, Any]],
            cancel_event: threading.Event,
        ):
            sequence = 0

            def semantic(event_type: str, **payload: Any) -> str:
                nonlocal sequence
                event = _response_stream_event(event_type, sequence, **payload)
                sequence += 1
                return _sse(event)

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
            try:
                yield semantic("response.created", response=dict(initial_response))
                yield semantic("response.in_progress", response=dict(initial_response))
                yield semantic(
                    "response.output_item.added",
                    output_index=0,
                    item=pending_item,
                )
                yield semantic(
                    "response.content_part.added",
                    item_id=prepared.output_item_id,
                    output_index=0,
                    content_index=0,
                    part=empty_part,
                )
                while True:
                    if await request.is_disconnected():
                        cancel_event.set()
                    try:
                        kind, value = await asyncio.wait_for(events.get(), timeout=0.1)
                    except asyncio.TimeoutError:
                        continue
                    if kind == "token":
                        yield semantic(
                            "response.output_text.delta",
                            item_id=prepared.output_item_id,
                            output_index=0,
                            content_index=0,
                            delta=value,
                            logprobs=[],
                        )
                        continue
                    if kind == "failed":
                        yield semantic("response.failed", response=value)
                        break
                    completed, final_response = value
                    output_item = final_response["output"][0]
                    output_part = output_item["content"][0]
                    yield semantic(
                        "response.output_text.done",
                        item_id=prepared.output_item_id,
                        output_index=0,
                        content_index=0,
                        text=completed.text,
                        logprobs=[],
                    )
                    yield semantic(
                        "response.content_part.done",
                        item_id=prepared.output_item_id,
                        output_index=0,
                        content_index=0,
                        part=output_part,
                    )
                    yield semantic(
                        "response.output_item.done",
                        output_index=0,
                        item=output_item,
                    )
                    terminal_type = {
                        "completed": "response.completed",
                        "failed": "response.failed",
                        "incomplete": "response.incomplete",
                        "cancelled": "response.incomplete",
                    }.get(final_response.get("status"), "response.failed")
                    yield semantic(terminal_type, response=final_response)
                    break
            finally:
                cancel_event.set()
                ticket.release()
                # Registered driver ownership survives transport cancellation;
                # lifespan drains it before prefix/model teardown.

        async def replay_response_sse(
            request: Request,
            *,
            scope: str,
            response_id: str,
            starting_after: int | None,
            include_obfuscation: bool,
        ):
            """Tail one durable background stream without owning its generation."""

            cursor = -1 if starting_after is None else starting_after
            terminal_statuses = {"cancelled", "completed", "failed", "incomplete"}
            while True:
                if await request.is_disconnected():
                    return
                try:
                    persisted = responses_service.state.list_response_events(
                        scope,
                        response_id,
                        starting_after=cursor,
                    )
                except KeyError:
                    return
                for event in persisted:
                    cursor = int(event["sequence_number"])
                    outgoing = dict(event)
                    if not include_obfuscation:
                        outgoing.pop("obfuscation", None)
                    yield _sse(outgoing)
                    if outgoing.get("type") in _TERMINAL_RESPONSE_STREAM_EVENTS:
                        return
                response = responses_service.state.get_response(scope, response_id)
                if response is None:
                    return
                if response.get("status") in terminal_statuses:
                    latest = responses_service.state.latest_response_event(
                        scope,
                        response_id,
                    )
                    if (
                        latest is None
                        or latest.get("type") not in _TERMINAL_RESPONSE_STREAM_EVENTS
                    ):
                        raise RuntimeError(
                            "A terminal streamed response is missing its durable terminal event"
                        )
                    # A cursor at or beyond the durable terminal event correctly
                    # yields an empty suffix and closes immediately. Otherwise,
                    # loop once more so a terminal commit that raced the first
                    # event query is never skipped.
                    if int(latest["sequence_number"]) <= cursor:
                        return
                    continue
                await asyncio.sleep(0.025)

        @app.post("/v1/responses")
        async def create_response(request: Request):
            scope = await authorize(request)
            payload = await _json_request_object(request)
            prepared = responses_service.prepare(scope, payload)
            if prepared.background:
                response = responses_service.persist(prepared)
                background_wakeup.set()
                if prepared.stream:
                    return StreamingResponse(
                        replay_response_sse(
                            request,
                            scope=scope,
                            response_id=prepared.response_id,
                            starting_after=None,
                            include_obfuscation=True,
                        ),
                        media_type="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                    )
                return response
            # This fence runs before admission and persistence.  There is no
            # await between it and eager driver registration, so lifespan
            # cannot observe an empty registry after a durable in_progress row
            # has been created.
            require_foreground_accepting()
            ticket = admit_generation()
            try:
                response = responses_service.persist(prepared)
            except BaseException:
                ticket.release()
                raise
            if prepared.stream:
                try:
                    events, cancel_event = start_response_driver(
                        prepared,
                        response,
                        ticket,
                    )
                    return _TicketStreamingResponse(
                        response_sse(
                            request,
                            prepared,
                            response,
                            ticket,
                            events,
                            cancel_event,
                        ),
                        ticket=ticket,
                        media_type="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                    )
                except BaseException:
                    ticket.release()
                    try:
                        responses_service.fail(
                            prepared,
                            code="server_shutdown",
                            message="Native inference app is shutting down.",
                        )
                    except BaseException:
                        pass
                    raise
            try:
                driver = track_foreground_driver(
                    asyncio.create_task(
                        ticket.run(
                            lambda: responses_service._execute_and_finish_resident(
                                prepared
                            )
                        )
                    )
                )
            except BaseException:
                ticket.release()
                try:
                    responses_service.fail(
                        prepared,
                        code="server_shutdown",
                        message="Native inference app is shutting down.",
                    )
                except BaseException:
                    pass
                raise
            try:
                _completed, final_response = await asyncio.shield(driver)
                return final_response
            except NativeResponsesAPIError:
                raise
            except asyncio.CancelledError:
                # The registered driver retains the submitted worker outcome;
                # lifespan drains it before cache/model shutdown.
                raise
            except Exception as exc:
                raise OpenAIAPIError(
                    500,
                    "Resident native generation failed.",
                    error_type="server_error",
                    code="generation_failed",
                ) from exc

        @app.get("/v1/responses/{response_id}")
        async def retrieve_response(response_id: str, request: Request):
            scope = await authorize(request)
            query = _response_retrieve_query(request)
            if not query.stream:
                return responses_service.retrieve_response(scope, response_id)
            responses_service.require_replayable_response(scope, response_id)
            return StreamingResponse(
                replay_response_sse(
                    request,
                    scope=scope,
                    response_id=response_id,
                    starting_after=query.starting_after,
                    include_obfuscation=query.include_obfuscation,
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        @app.delete("/v1/responses/{response_id}")
        async def delete_response(response_id: str, request: Request) -> dict[str, Any]:
            scope = await authorize(request)
            return responses_service.delete_response(scope, response_id)

        @app.get("/v1/responses/{response_id}/input_items")
        async def list_response_input_items(
            response_id: str,
            request: Request,
        ) -> dict[str, Any]:
            scope = await authorize(request)
            after, limit, order = _cursor_list_query(request, allow_include=True)
            return responses_service.response_input_items(
                scope,
                response_id,
                after=after,
                limit=limit,
                order=order,
            )

        @app.post("/v1/responses/input_tokens")
        async def count_response_input_tokens(request: Request) -> dict[str, Any]:
            scope = await authorize(request)
            payload = await _json_request_object(request)
            return responses_service.count_input_tokens(scope, payload)

        @app.post("/v1/responses/compact")
        async def compact_response(request: Request) -> dict[str, Any]:
            scope = await authorize(request)
            payload = await _json_request_object(request)
            return responses_service.compact_response(scope, payload)

        @app.post("/v1/responses/{response_id}/cancel")
        async def cancel_response(response_id: str, request: Request) -> dict[str, Any]:
            scope = await authorize(request)
            return responses_service.cancel_response(scope, response_id)

        @app.post("/v1/conversations")
        async def create_conversation(request: Request) -> dict[str, Any]:
            scope = await authorize(request)
            payload = await _json_request_object(request)
            return responses_service.create_conversation(scope, payload)

        @app.get("/v1/conversations/{conversation_id}")
        async def retrieve_conversation(
            conversation_id: str,
            request: Request,
        ) -> dict[str, Any]:
            scope = await authorize(request)
            return responses_service.retrieve_conversation(scope, conversation_id)

        @app.post("/v1/conversations/{conversation_id}")
        async def update_conversation(
            conversation_id: str,
            request: Request,
        ) -> dict[str, Any]:
            scope = await authorize(request)
            payload = await _json_request_object(request)
            return responses_service.update_conversation(scope, conversation_id, payload)

        @app.delete("/v1/conversations/{conversation_id}")
        async def delete_conversation(
            conversation_id: str,
            request: Request,
        ) -> dict[str, Any]:
            scope = await authorize(request)
            return responses_service.delete_conversation(scope, conversation_id)

        @app.post("/v1/conversations/{conversation_id}/items")
        async def create_conversation_items(
            conversation_id: str,
            request: Request,
        ) -> dict[str, Any]:
            scope = await authorize(request)
            payload = await _json_request_object(request)
            return responses_service.create_conversation_items(scope, conversation_id, payload)

        @app.get("/v1/conversations/{conversation_id}/items")
        async def list_conversation_items(
            conversation_id: str,
            request: Request,
        ) -> dict[str, Any]:
            scope = await authorize(request)
            after, limit, order = _cursor_list_query(request)
            return responses_service.list_conversation_items(
                scope,
                conversation_id,
                after=after,
                limit=limit,
                order=order,
            )

        @app.get("/v1/conversations/{conversation_id}/items/{item_id}")
        async def retrieve_conversation_item(
            conversation_id: str,
            item_id: str,
            request: Request,
        ) -> dict[str, Any]:
            scope = await authorize(request)
            return responses_service.retrieve_conversation_item(
                scope,
                conversation_id,
                item_id,
            )

        @app.delete("/v1/conversations/{conversation_id}/items/{item_id}")
        async def delete_conversation_item(
            conversation_id: str,
            item_id: str,
            request: Request,
        ) -> dict[str, Any]:
            scope = await authorize(request)
            return responses_service.delete_conversation_item(
                scope,
                conversation_id,
                item_id,
            )

    @app.api_route("/v1/{unsupported_path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    async def unsupported_resource(unsupported_path: str, request: Request):
        await authorize(request)
        raise OpenAIAPIError(
            404,
            f"The /v1/{unsupported_path} resource is not implemented by this serving milestone.",
            code="unsupported_resource",
        )

    return app


def prepare_native_inference_server(
    config: NativeServeConfig,
    *,
    binding: Any | None = None,
) -> tuple[FastAPI, NativeServingRuntime, BearerAuth]:
    """Validate auth/artifact/runtime and load the model before socket bind."""

    auth = resolve_bearer_auth(config)
    runtime = NativeServingRuntime.load(config, binding=binding)
    try:
        app = create_native_inference_app(
            runtime,
            auth=auth,
            queue_capacity=config.queue_capacity,
            session_limit=config.session_limit,
        )
    except BaseException:
        runtime.close()
        raise
    return app, runtime, auth


def run_native_inference_server(config: NativeServeConfig) -> None:
    """Synchronously validate everything, then let Uvicorn open the socket."""

    app, runtime, _auth = prepare_native_inference_server(config)
    try:
        uvicorn = importlib.import_module("uvicorn")
        uvicorn.run(
            app,
            host=config.host,
            port=config.port,
            workers=1,
            log_level=config.log_level,
        )
    finally:
        # Uvicorn normally drives the app lifespan. Closing again is idempotent
        # and also covers import/startup failures before lifespan begins.
        queue = getattr(getattr(app, "state", None), "generation_queue", None)
        if queue is not None:
            queue.close()
        runtime.close()


__all__ = [
    "BearerAuth",
    "BoundedSingleWorkerQueue",
    "NativeServeConfig",
    "NativeServingConfigurationError",
    "NativeServingRuntime",
    "OpenAIAPIError",
    "create_native_inference_app",
    "prepare_native_inference_server",
    "resolve_bearer_auth",
    "run_native_inference_server",
]
