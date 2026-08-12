"""Dependency-light text presentation helpers for resident native inference.

The native CLI and standalone server must be able to render/tokenize chat
without importing the graph runtime, Torch, NumPy, or the editor backend.  This
module owns that small boundary; it does not load or execute a model.
"""

from __future__ import annotations

import codecs
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


NATIVE_MANIFEST_SCHEMA = "neuralfn.native_execution_manifest"
NATIVE_MANIFEST_VERSION = 1
NATIVE_MANIFEST_FILENAME = "native-execution-manifest.json"
NATIVE_CHAT_ROLES = frozenset({"developer", "system", "user", "assistant", "tool"})
_MAX_TEMPLATE_BYTES = 1024 * 1024


class NativeChatConfigurationError(ValueError):
    """Invalid native manifest presentation metadata or chat configuration."""


@dataclass(frozen=True, slots=True)
class NativeChatMessage:
    role: str
    content: str
    name: str | None = None
    tool_call_id: str | None = None

    def __post_init__(self) -> None:
        role = str(self.role).strip().lower()
        if role not in NATIVE_CHAT_ROLES:
            raise NativeChatConfigurationError(
                f"Unsupported native chat role {self.role!r}; expected one of "
                + ", ".join(sorted(NATIVE_CHAT_ROLES))
            )
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "content", str(self.content))
        if self.name is not None:
            name = str(self.name).strip()
            if not name:
                raise NativeChatConfigurationError("Chat message name must not be empty")
            object.__setattr__(self, "name", name)
        if self.tool_call_id is not None:
            object.__setattr__(self, "tool_call_id", str(self.tool_call_id))


class NativeTextCodec:
    name: str

    def encode(self, text: str) -> tuple[int, ...]:
        raise NotImplementedError

    def decode(self, token_ids: Sequence[int]) -> str:
        raise NotImplementedError

    def token_bytes(self, token_id: int) -> bytes:
        raise NotImplementedError

    def incremental_decoder(self) -> "IncrementalTokenDecoder":
        return IncrementalTokenDecoder(self)


class TokenIdTextCodec(NativeTextCodec):
    """Tokenizer-free presentation for explicit raw token-ID CLI requests."""

    name = "token_ids"

    def encode(self, text: str) -> tuple[int, ...]:
        raise NativeChatConfigurationError(
            "The token-ID codec cannot encode text; use --prompt-tokens."
        )

    def decode(self, token_ids: Sequence[int]) -> str:
        return ",".join(str(int(token_id)) for token_id in token_ids)

    def token_bytes(self, token_id: int) -> bytes:
        return str(int(token_id)).encode("ascii")


class IncrementalTokenDecoder:
    def __init__(self, codec: NativeTextCodec) -> None:
        self._codec = codec
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def push(self, token_id: int) -> str:
        return self._decoder.decode(self._codec.token_bytes(token_id), final=False)

    def finish(self) -> str:
        return self._decoder.decode(b"", final=True)


class TiktokenTextCodec(NativeTextCodec):
    def __init__(self, encoding_name: str) -> None:
        try:
            tiktoken = importlib.import_module("tiktoken")
            encoding = tiktoken.get_encoding(encoding_name)
        except (ImportError, ValueError) as exc:
            raise NativeChatConfigurationError(
                f"Unable to load artifact tiktoken encoding {encoding_name!r}; "
                "install tiktoken and verify tokenizer metadata"
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


class NativeChatRenderer:
    name: str

    def render(
        self,
        messages: Sequence[NativeChatMessage],
        *,
        include_assistant_prompt: bool,
    ) -> str:
        raise NotImplementedError


def _plain_role_transcript(
    messages: Sequence[NativeChatMessage],
    *,
    include_assistant_prompt: bool,
) -> str:
    chunks: list[str] = []
    for message in messages:
        label = message.role if message.name is None else f"{message.role}:{message.name}"
        chunks.append(f"<|{label}|>\n{message.content}\n")
    if include_assistant_prompt and (not messages or messages[-1].role != "assistant"):
        chunks.append("<|assistant|>\n")
    return "".join(chunks)


class PlainRolesRenderer(NativeChatRenderer):
    name = "plain_roles"

    def render(
        self,
        messages: Sequence[NativeChatMessage],
        *,
        include_assistant_prompt: bool,
    ) -> str:
        return _plain_role_transcript(
            messages,
            include_assistant_prompt=include_assistant_prompt,
        )


class PlaceholderChatRenderer(NativeChatRenderer):
    def __init__(self, template: str, *, name: str) -> None:
        if "{{messages}}" not in template and "{messages}" not in template:
            raise NativeChatConfigurationError(
                f"Chat template {name!r} must contain {{messages}} or {{{{messages}}}}"
            )
        self.name = name
        self._template = template

    def render(
        self,
        messages: Sequence[NativeChatMessage],
        *,
        include_assistant_prompt: bool,
    ) -> str:
        body = _plain_role_transcript(messages, include_assistant_prompt=False)
        assistant = "<|assistant|>\n" if include_assistant_prompt else ""
        had_assistant_marker = (
            "{{assistant_prompt}}" in self._template
            or "{assistant_prompt}" in self._template
        )
        rendered = re.sub(
            r"\{\{messages\}\}|\{messages\}|"
            r"\{\{assistant_prompt\}\}|\{assistant_prompt\}",
            lambda match: body if "messages" in match.group(0) else assistant,
            self._template,
        )
        if assistant and not had_assistant_marker:
            rendered = rendered.rstrip() + "\n" + assistant
        return rendered


@dataclass(frozen=True, slots=True)
class NativeChatRendererResolution:
    renderer: NativeChatRenderer
    warning: str | None = None


@dataclass(frozen=True, slots=True)
class NativeChatPrompt:
    text: str
    token_ids: tuple[int, ...]
    messages: tuple[NativeChatMessage, ...]
    dropped_groups: int


def read_native_execution_manifest(
    artifact: str | Path,
) -> tuple[Path, Path, dict[str, Any]]:
    requested = Path(artifact).expanduser().resolve()
    manifest_path = requested / NATIVE_MANIFEST_FILENAME if requested.is_dir() else requested
    if not manifest_path.is_file():
        raise NativeChatConfigurationError(
            f"Native Execution manifest does not exist: {manifest_path}"
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise NativeChatConfigurationError(
            f"Native Execution manifest is not valid JSON: {manifest_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise NativeChatConfigurationError("Native Execution manifest root must be an object")
    if (
        payload.get("schema") != NATIVE_MANIFEST_SCHEMA
        or payload.get("version") != NATIVE_MANIFEST_VERSION
    ):
        raise NativeChatConfigurationError(
            "Native inference requires NeuralFn Native Execution manifest schema/version 1"
        )
    return manifest_path.parent, manifest_path, payload


def load_native_text_codec(manifest: Mapping[str, Any]) -> NativeTextCodec:
    raw = manifest.get("tokenizer")
    if not isinstance(raw, Mapping):
        raise NativeChatConfigurationError(
            "Native text inference requires tokenizer metadata in the manifest"
        )
    family = str(raw.get("family") or raw.get("tokenizer_family") or "").strip().lower()
    encoding_name = str(raw.get("encoding_name") or raw.get("tokenizer_name") or "").strip()
    tokenization = str(raw.get("tokenization") or "").strip().lower()
    if encoding_name and (
        family in {"", "tiktoken", "tiktoken_bpe", "gpt2_bpe"}
        or tokenization in {"", "gpt2_bpe", "tiktoken", "tiktoken_bpe"}
    ):
        return TiktokenTextCodec(encoding_name)
    raise NativeChatConfigurationError(
        "Native text inference currently supports artifact-declared tiktoken encodings only"
    )


def _manifest_template(manifest: Mapping[str, Any]) -> str | None:
    metadata = manifest.get("chat_template")
    if not isinstance(metadata, Mapping):
        return None
    format_name = str(metadata.get("format") or "").strip().lower()
    template = metadata.get("template")
    if format_name == "plain_roles":
        return "plain_roles"
    if isinstance(template, str) and template.strip():
        return template
    return None


def resolve_native_chat_renderer(
    manifest: Mapping[str, Any],
    selection: str,
    *,
    allow_auto_fallback: bool,
) -> NativeChatRendererResolution:
    requested = str(selection or "auto").strip()
    if requested.lower() == "plain_roles":
        return NativeChatRendererResolution(PlainRolesRenderer())
    if requested.lower() != "auto":
        path = Path(requested).expanduser().resolve()
        if not path.is_file():
            raise NativeChatConfigurationError(f"Chat template file does not exist: {path}")
        if path.stat().st_size > _MAX_TEMPLATE_BYTES:
            raise NativeChatConfigurationError(
                f"Chat template file exceeds the 1 MiB CLI limit: {path}"
            )
        return NativeChatRendererResolution(
            PlaceholderChatRenderer(path.read_text(encoding="utf-8"), name=str(path))
        )

    template = _manifest_template(manifest)
    if template is not None and template.strip().lower() == "plain_roles":
        return NativeChatRendererResolution(PlainRolesRenderer())
    if isinstance(template, str) and (
        "{{messages}}" in template or "{messages}" in template
    ):
        return NativeChatRendererResolution(
            PlaceholderChatRenderer(template, name="artifact")
        )
    if allow_auto_fallback:
        return NativeChatRendererResolution(
            PlainRolesRenderer(),
            "The artifact has no chat template supported by the lean native CLI; "
            "using plain_roles for this process. Pass --chat-template PATH to select one.",
        )
    raise NativeChatConfigurationError(
        "Artifact has no supported chat template. Select --chat-template plain_roles "
        "or a PATH containing {messages}."
    )


def native_context_limit(manifest: Mapping[str, Any]) -> int:
    raw = manifest.get("context_limits")
    value = raw.get("max_context_tokens") if isinstance(raw, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise NativeChatConfigurationError(
            "Native text inference requires a positive context_limits.max_context_tokens"
        )
    return value


def native_output_limit(manifest: Mapping[str, Any]) -> int | None:
    raw = manifest.get("context_limits")
    value = raw.get("max_output_tokens") if isinstance(raw, Mapping) else None
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise NativeChatConfigurationError(
            "context_limits.max_output_tokens must be a positive integer or null"
        )
    return value


def native_stop_token_ids(manifest: Mapping[str, Any]) -> tuple[int, ...]:
    raw = manifest.get("stop_tokens", ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise NativeChatConfigurationError("Manifest stop_tokens must be an array")
    values: list[int] = []
    for index, value in enumerate(raw):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise NativeChatConfigurationError(
                f"Manifest stop_tokens[{index}] must be a non-negative integer"
            )
        if value not in values:
            values.append(value)
    return tuple(values)


def _chat_groups(
    messages: Sequence[NativeChatMessage],
) -> tuple[list[NativeChatMessage], list[list[NativeChatMessage]]]:
    leading: list[NativeChatMessage] = []
    index = 0
    while index < len(messages) and messages[index].role in {"developer", "system"}:
        leading.append(messages[index])
        index += 1
    groups: list[list[NativeChatMessage]] = []
    current: list[NativeChatMessage] = []
    for message in messages[index:]:
        if message.role == "user" and current:
            groups.append(current)
            current = []
        current.append(message)
    if current:
        groups.append(current)
    return leading, groups


def resolve_native_chat_prompt(
    *,
    codec: NativeTextCodec,
    renderer: NativeChatRenderer,
    mode: str,
    history: Sequence[NativeChatMessage],
    draft: str,
    context_limit: int,
    reserved_output_tokens: int,
) -> NativeChatPrompt:
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in {"stateless", "transcript"}:
        raise NativeChatConfigurationError(
            f"Unsupported native chat mode {mode!r}; expected stateless or transcript"
        )
    if isinstance(reserved_output_tokens, bool) or reserved_output_tokens < 0:
        raise NativeChatConfigurationError("reserved_output_tokens must be non-negative")
    prompt_budget = context_limit - reserved_output_tokens
    if prompt_budget <= 0:
        raise NativeChatConfigurationError(
            f"Generation reserves {reserved_output_tokens} output tokens but the context "
            f"window is {context_limit}; reduce --max-new-tokens."
        )
    normalized_history = list(history)
    if not all(isinstance(message, NativeChatMessage) for message in normalized_history):
        raise TypeError("history must contain NativeChatMessage values")
    leading, groups = _chat_groups(normalized_history)
    newest = NativeChatMessage("user", draft)
    if normalized_mode == "stateless":
        messages = (*leading, newest)
        text = renderer.render(messages, include_assistant_prompt=True)
        token_ids = codec.encode(text)
        if len(token_ids) > prompt_budget:
            raise NativeChatConfigurationError(
                f"The stateless prompt uses {len(token_ids)} tokens but only {prompt_budget} "
                f"remain after reserving {reserved_output_tokens} output tokens."
            )
        return NativeChatPrompt(text, token_ids, messages, 0)

    groups.append([newest])
    dropped = 0
    while True:
        messages = tuple([*leading, *(message for group in groups for message in group)])
        text = renderer.render(messages, include_assistant_prompt=True)
        token_ids = codec.encode(text)
        if len(token_ids) <= prompt_budget:
            return NativeChatPrompt(text, token_ids, messages, dropped)
        if len(groups) <= 1:
            raise NativeChatConfigurationError(
                "The leading instructions and newest user/tool turn do not fit the "
                f"{prompt_budget}-token prompt budget after reserving "
                f"{reserved_output_tokens} output tokens."
            )
        groups.pop(0)
        dropped += 1


def native_text_stop_delimiters(
    manifest: Mapping[str, Any],
    renderer: NativeChatRenderer,
) -> tuple[str, ...]:
    values: list[str] = []

    def collect(raw: Any) -> None:
        if isinstance(raw, str) and raw:
            values.append(raw)
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            values.extend(item for item in raw if isinstance(item, str) and item)

    for mapping in (
        manifest,
        manifest.get("tokenizer") if isinstance(manifest.get("tokenizer"), Mapping) else {},
        manifest.get("chat_template")
        if isinstance(manifest.get("chat_template"), Mapping)
        else {},
    ):
        for key in ("stop_strings", "stop_sequences", "role_delimiters", "eos_token"):
            collect(mapping.get(key))
    if renderer.name == "plain_roles":
        values.extend(("<|developer", "<|system", "<|user", "<|tool"))
    return tuple(dict.fromkeys(values))


def strip_native_text_delimiters(text: str, delimiters: Sequence[str]) -> str:
    end = len(text)
    for delimiter in delimiters:
        index = text.find(delimiter)
        if index >= 0:
            end = min(end, index)
    return text[:end].rstrip()


__all__ = [
    "IncrementalTokenDecoder",
    "NATIVE_CHAT_ROLES",
    "NATIVE_MANIFEST_FILENAME",
    "NATIVE_MANIFEST_SCHEMA",
    "NATIVE_MANIFEST_VERSION",
    "NativeChatConfigurationError",
    "NativeChatMessage",
    "NativeChatPrompt",
    "NativeChatRenderer",
    "NativeChatRendererResolution",
    "NativeTextCodec",
    "PlaceholderChatRenderer",
    "PlainRolesRenderer",
    "TiktokenTextCodec",
    "load_native_text_codec",
    "native_context_limit",
    "native_output_limit",
    "native_stop_token_ids",
    "native_text_stop_delimiters",
    "read_native_execution_manifest",
    "resolve_native_chat_prompt",
    "resolve_native_chat_renderer",
    "strip_native_text_delimiters",
]
