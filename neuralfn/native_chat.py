"""Dependency-light text presentation helpers for resident native inference.

The native CLI and standalone server must be able to render/tokenize chat
without importing the graph runtime, Torch, NumPy, or the editor backend.  This
module owns that small boundary; it does not load or execute a model.
"""

from __future__ import annotations

import codecs
from dataclasses import dataclass
from datetime import date
import hashlib
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
MUSE_GLIMMER_ATEM_PROFILE = "muse_glimmer_atem_v1"
MUSE_GLIMMER_ATEM_TEMPLATE_SHA256 = (
    "cfc67e5f349f37690dfd31ed1f18bc4442a9dd32fe39a648f993cb4eb3cae678"
)
MUSE_GLIMMER_TOKENIZER_SHA256 = (
    "c9dbee66967b58f31a7c27f723c3760da3526ccd0427578e8905b0abb0031c4d"
)
MUSE_GLIMMER_TOKENIZER_CONFIG_SHA256 = (
    "781e6c74f571642c71202167b67d9255b28cc439bdda1582ff31346182f5a9c5"
)
MUSE_GLIMMER_ADDED_TOKENS_SHA256 = (
    "6b89e78e0ac391500aa191fae2ec274aaa9453498e273ce6f0e18253abffa5ca"
)
MUSE_GLIMMER_ARTIFACT_REVISION = "a4e59da52a7bc87ae7251dd5545c0dd437c44b68"
MUSE_GLIMMER_VOCAB_SIZE = 202_048
MUSE_GLIMMER_SPECIAL_TOKEN_IDS = {
    "bos": 200_000,
    "eos": 200_001,
    "eom": 200_007,
    "eot": 200_008,
    "pad": 200_018,
    "start": 200_022,
    "message": 200_023,
    "image_start": 200_080,
    "image_end": 200_081,
    "video_start": 200_082,
    "video_end": 200_083,
    "video_frame_separator": 200_087,
    "image": 200_090,
    "video": 200_091,
    "patch": 200_092,
    "dflash_mask": 201_818,
}
_MUSE_GLIMMER_SPECIAL_TOKEN_CONTENT = {
    "bos": "<|begin_of_text|>",
    "eos": "<|end_of_text|>",
    "eom": "<|eom|>",
    "eot": "<|eot|>",
    "pad": "<|finetune_right_pad|>",
    "start": "<|start|>",
    "message": "<|message|>",
    "image_start": "<|image_start|>",
    "image_end": "<|image_end|>",
    "video_start": "<|vid_start|>",
    "video_end": "<|vid_end|>",
    "video_frame_separator": "<|vid_frame_separator|>",
    "image": "<|image|>",
    "video": "<|video|>",
    "patch": "<|patch|>",
    "dflash_mask": "<|reserved_special_token_1818|>",
}
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _contained_artifact_file(root: Path, raw_path: Any, *, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise NativeChatConfigurationError(f"{label} requires a non-empty artifact_path")
    relative = Path(raw_path)
    if relative.is_absolute():
        raise NativeChatConfigurationError(f"{label} artifact_path must be relative")
    resolved_root = root.expanduser().resolve()
    resolved = (resolved_root / relative).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise NativeChatConfigurationError(
            f"{label} artifact_path escapes the Native Execution artifact"
        ) from exc
    if not resolved.is_file():
        raise NativeChatConfigurationError(f"{label} file does not exist: {resolved}")
    return resolved


def _byte_level_decoder() -> dict[str, int]:
    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values += list(range(0xA1, 0xAC + 1))
    byte_values += list(range(0xAE, 0xFF + 1))
    unicode_values = list(byte_values)
    extra = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            unicode_values.append(256 + extra)
            extra += 1
    return {chr(codepoint): value for value, codepoint in zip(byte_values, unicode_values)}


_BYTE_LEVEL_DECODE = _byte_level_decoder()


class HuggingFaceTokenizerJSONCodec(NativeTextCodec):
    """Strict artifact-contained Hugging Face Tokenizers codec.

    Encoding deliberately disables the tokenizer post-processor. Chat renderers
    own BOS insertion, matching ``apply_chat_template(..., tokenize=True)`` and
    avoiding a duplicated ``<|begin_of_text|>`` token.
    """

    def __init__(
        self,
        tokenizer_path: Path,
        *,
        expected_sha256: str,
        expected_vocab_size: int,
        required_special_token_ids: Mapping[str, int] | None = None,
    ) -> None:
        normalized_sha = str(expected_sha256).strip().lower()
        if _SHA256_PATTERN.fullmatch(normalized_sha) is None:
            raise NativeChatConfigurationError(
                "tokenizer.sha256 must be a lowercase 64-character SHA-256 digest"
            )
        if (
            isinstance(expected_vocab_size, bool)
            or not isinstance(expected_vocab_size, int)
            or expected_vocab_size <= 0
        ):
            raise NativeChatConfigurationError("tokenizer.vocab_size must be positive")
        actual_sha = _sha256_file(tokenizer_path)
        if actual_sha != normalized_sha:
            raise NativeChatConfigurationError(
                f"Tokenizer SHA-256 mismatch for {tokenizer_path}: expected "
                f"{normalized_sha}, got {actual_sha}"
            )
        try:
            payload = json.loads(tokenizer_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise NativeChatConfigurationError(
                f"Tokenizer file is not valid UTF-8 JSON: {tokenizer_path}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise NativeChatConfigurationError("tokenizer.json root must be an object")
        model = payload.get("model")
        if not isinstance(model, Mapping) or model.get("type") != "BPE":
            raise NativeChatConfigurationError(
                "Native tokenizer.json support currently requires a BPE model"
            )
        vocab = model.get("vocab")
        if not isinstance(vocab, Mapping):
            raise NativeChatConfigurationError("tokenizer.json model.vocab must be an object")
        added = payload.get("added_tokens", ())
        if not isinstance(added, Sequence) or isinstance(added, (str, bytes)):
            raise NativeChatConfigurationError("tokenizer.json added_tokens must be an array")
        added_by_id: dict[int, tuple[str, bool]] = {}
        added_by_content: dict[str, int] = {}
        for index, item in enumerate(added):
            if not isinstance(item, Mapping):
                raise NativeChatConfigurationError(
                    f"tokenizer.json added_tokens[{index}] must be an object"
                )
            token_id = item.get("id")
            content = item.get("content")
            if (
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
                or not isinstance(content, str)
            ):
                raise NativeChatConfigurationError(
                    f"tokenizer.json added_tokens[{index}] has an invalid id/content"
                )
            if token_id in added_by_id or content in added_by_content:
                raise NativeChatConfigurationError(
                    "tokenizer.json contains duplicate added-token ids or contents"
                )
            added_by_id[token_id] = (content, bool(item.get("special", False)))
            added_by_content[content] = token_id
        model_tokens: dict[int, str] = {}
        for token, raw_id in vocab.items():
            if not isinstance(token, str) or isinstance(raw_id, bool) or not isinstance(raw_id, int):
                raise NativeChatConfigurationError("tokenizer.json model.vocab is malformed")
            overlapping_added = added_by_id.get(raw_id)
            if overlapping_added is not None and overlapping_added[0] == token:
                continue
            if raw_id < 0 or raw_id in model_tokens or overlapping_added is not None:
                raise NativeChatConfigurationError(
                    "tokenizer.json contains duplicate or negative vocabulary ids"
                )
            model_tokens[raw_id] = token
        try:
            tokenizers = importlib.import_module("tokenizers")
        except ImportError as exc:
            raise NativeChatConfigurationError(
                "tokenizer.json inference requires `pip install neuralfn[serve]` "
                "or tokenizers>=0.19"
            ) from exc
        try:
            tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))
        except Exception as exc:
            raise NativeChatConfigurationError(
                f"Unable to load tokenizer.json with Hugging Face Tokenizers: {exc}"
            ) from exc
        actual_vocab_size = int(tokenizer.get_vocab_size(with_added_tokens=True))
        if actual_vocab_size != expected_vocab_size:
            raise NativeChatConfigurationError(
                f"Tokenizer vocabulary mismatch: manifest declares {expected_vocab_size}, "
                f"tokenizer.json contains {actual_vocab_size} tokens"
            )
        required = dict(required_special_token_ids or {})
        for name, expected_id in required.items():
            expected_content = _MUSE_GLIMMER_SPECIAL_TOKEN_CONTENT.get(name)
            actual_id = added_by_content.get(expected_content) if expected_content else None
            if actual_id != expected_id:
                raise NativeChatConfigurationError(
                    f"Tokenizer special token {name!r} must be id {expected_id}, got {actual_id}"
                )
        canonical_added = json.dumps(
            list(added), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        self.name = f"tokenizer_json:{tokenizer_path.name}"
        self.path = tokenizer_path
        self.sha256 = actual_sha
        self.vocab_size = actual_vocab_size
        self.added_tokens_sha256 = hashlib.sha256(canonical_added).hexdigest()
        self.special_token_ids = required
        self._tokenizer = tokenizer
        self._model_tokens = model_tokens
        self._added_by_id = added_by_id

    def encode(self, text: str) -> tuple[int, ...]:
        try:
            return tuple(self._tokenizer.encode(str(text), add_special_tokens=False).ids)
        except Exception as exc:
            raise NativeChatConfigurationError(f"Unable to encode text: {exc}") from exc

    def decode(self, token_ids: Sequence[int]) -> str:
        values = [int(token_id) for token_id in token_ids]
        try:
            return self._tokenizer.decode(values, skip_special_tokens=False)
        except Exception as exc:
            raise RuntimeError(f"Unable to decode native token ids: {exc}") from exc

    def token_bytes(self, token_id: int) -> bytes:
        normalized = int(token_id)
        added = self._added_by_id.get(normalized)
        if added is not None:
            return added[0].encode("utf-8")
        token = self._model_tokens.get(normalized)
        if token is None:
            raise RuntimeError(f"Native binding produced unknown token id {normalized}")
        try:
            return bytes(_BYTE_LEVEL_DECODE[character] for character in token)
        except KeyError as exc:
            raise RuntimeError(
                f"Tokenizer token {normalized} is not ByteLevel encoded"
            ) from exc


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


class MuseGlimmerATEMRenderer(NativeChatRenderer):
    """Reviewed text-only implementation of Muse Glimmer's ATEM template.

    This intentionally implements only the proven system/user/assistant string
    subset. It never evaluates artifact Jinja and fails closed for tool roles,
    named messages, media parts, or the richer reasoning/tool-call structures.
    """

    name = MUSE_GLIMMER_ATEM_PROFILE

    def __init__(
        self,
        *,
        reasoning_strength: str = "high",
        knowledge_cutoff: str = "2026-01-04",
        current_date: str | None = None,
    ) -> None:
        strength = str(reasoning_strength).strip()
        cutoff = str(knowledge_cutoff).strip()
        rendered_date = date.today().isoformat() if current_date is None else str(current_date).strip()
        if not strength or any(marker in strength for marker in ("<|", "\n", "\r")):
            raise NativeChatConfigurationError("ATEM reasoning_strength is invalid")
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", cutoff) is None:
            raise NativeChatConfigurationError("ATEM knowledge_cutoff must be YYYY-MM-DD")
        if rendered_date and re.fullmatch(r"\d{4}-\d{2}-\d{2}", rendered_date) is None:
            raise NativeChatConfigurationError("ATEM current_date must be YYYY-MM-DD")
        self.reasoning_strength = strength
        self.knowledge_cutoff = cutoff
        self.current_date = rendered_date

    @staticmethod
    def _system_text(content: str) -> str:
        return (
            content.replace("Reasoning effort", "Reasoning strength")
            .replace("Reasoning Effort", "Reasoning Strength")
            .replace("reasoning effort", "reasoning strength")
            .replace("REASONING EFFORT", "REASONING STRENGTH")
        )

    def _system_suffix(self, *, include_reasoning: bool) -> str:
        parts: list[str] = []
        if include_reasoning:
            parts.append(f"Reasoning strength: {self.reasoning_strength}.")
        parts.append('# Valid recipients: "self", "user".')
        return "\n\n".join(parts)

    def render(
        self,
        messages: Sequence[NativeChatMessage],
        *,
        include_assistant_prompt: bool,
    ) -> str:
        normalized = tuple(messages)
        if not all(isinstance(message, NativeChatMessage) for message in normalized):
            raise TypeError("ATEM messages must contain NativeChatMessage values")
        for index, message in enumerate(normalized):
            if message.role not in {"system", "user", "assistant"}:
                raise NativeChatConfigurationError(
                    "Muse Glimmer ATEM text mode supports system, user, and assistant "
                    f"messages only; messages[{index}] uses {message.role!r}"
                )
            if message.name is not None or message.tool_call_id is not None:
                raise NativeChatConfigurationError(
                    "Muse Glimmer ATEM text mode does not support named/tool messages"
                )

        chunks = ["<|begin_of_text|>"]
        has_system = any(message.role == "system" for message in normalized)
        if not has_system:
            default_system = (
                "<|start|>system<|message|>You are a helpful AI assistant."
                f"\nKnowledge cutoff: {self.knowledge_cutoff}."
            )
            if self.current_date:
                default_system += f"\nCurrent date: {self.current_date}."
            chunks.append(
                default_system
                + "\n\n"
                + self._system_suffix(include_reasoning=True)
                + "<|eot|>"
            )

        for message in normalized:
            if message.role == "system":
                content = self._system_text(message.content)
                chunks.append("<|start|>system<|message|>" + content)
                chunks.append("\n\n")
                chunks.append(
                    self._system_suffix(
                        include_reasoning="reasoning strength" not in content.lower()
                    )
                )
                chunks.append("<|eot|>")
            elif message.role == "user":
                chunks.append(
                    "<|start|>user<|message|>" + message.content + "<|eot|>"
                )
            else:
                chunks.append(
                    "<|start|>assistant to=user<|message|>"
                    + message.content
                    + "<|eot|>"
                )
        if include_assistant_prompt:
            chunks.append("<|start|>assistant")
        return "".join(chunks)


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


def _is_muse_glimmer_manifest(manifest: Mapping[str, Any]) -> bool:
    model = manifest.get("model")
    if not isinstance(model, Mapping):
        return False
    values = (model.get("family"), model.get("model_type"), model.get("architecture"))
    return any(
        str(value or "").strip().lower().replace("-", "_") in {
            "muse_glimmer",
            "muse_glimmer_for_conditional_generation",
        }
        for value in values
    )


def _validate_muse_glimmer_tokenizer_config(
    root: Path,
    metadata: Mapping[str, Any],
    codec: HuggingFaceTokenizerJSONCodec,
) -> None:
    revision = str(metadata.get("revision") or metadata.get("artifact_revision") or "")
    if revision != MUSE_GLIMMER_ARTIFACT_REVISION:
        raise NativeChatConfigurationError(
            "Muse Glimmer tokenizer metadata must pin artifact revision "
            + MUSE_GLIMMER_ARTIFACT_REVISION
        )
    declared_added_sha = str(metadata.get("added_tokens_sha256") or "").lower()
    if (
        declared_added_sha != MUSE_GLIMMER_ADDED_TOKENS_SHA256
        or codec.added_tokens_sha256 != declared_added_sha
    ):
        raise NativeChatConfigurationError(
            "Muse Glimmer added-token table does not match the pinned artifact revision"
        )
    config_path = _contained_artifact_file(
        root,
        metadata.get("config_artifact_path")
        or metadata.get("tokenizer_config_artifact_path"),
        label="tokenizer_config.json",
    )
    declared_config_sha = str(
        metadata.get("config_sha256") or metadata.get("tokenizer_config_sha256") or ""
    ).lower()
    actual_config_sha = _sha256_file(config_path)
    if (
        declared_config_sha != MUSE_GLIMMER_TOKENIZER_CONFIG_SHA256
        or actual_config_sha != declared_config_sha
    ):
        raise NativeChatConfigurationError(
            "Muse Glimmer tokenizer_config.json does not match the pinned artifact revision"
        )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NativeChatConfigurationError("tokenizer_config.json is invalid") from exc
    if not isinstance(config, Mapping):
        raise NativeChatConfigurationError("tokenizer_config.json root must be an object")
    expected = {
        "backend": "tokenizers",
        "bos_token": "<|begin_of_text|>",
        "eos_token": "<|end_of_text|>",
        "pad_token": "<|finetune_right_pad|>",
        "model_max_length": 131_072,
        "processor_class": "MuseGlimmerProcessor",
        "tokenizer_class": "TokenizersBackend",
    }
    mismatches = [
        key for key, expected_value in expected.items() if config.get(key) != expected_value
    ]
    extras = config.get("extra_special_tokens")
    if not isinstance(extras, Sequence) or isinstance(extras, (str, bytes)) or len(extras) != 2_048:
        mismatches.append("extra_special_tokens")
    else:
        for name, token_id in MUSE_GLIMMER_SPECIAL_TOKEN_IDS.items():
            expected_content = _MUSE_GLIMMER_SPECIAL_TOKEN_CONTENT[name]
            offset = token_id - 200_000
            if offset < 0 or offset >= len(extras) or extras[offset] != expected_content:
                mismatches.append(f"extra_special_tokens[{offset}]")
    if mismatches:
        raise NativeChatConfigurationError(
            "Muse Glimmer tokenizer_config.json contract mismatch: "
            + ", ".join(mismatches)
        )


def load_native_text_codec(
    manifest: Mapping[str, Any],
    *,
    artifact_root: str | Path | None = None,
) -> NativeTextCodec:
    raw = manifest.get("tokenizer")
    if not isinstance(raw, Mapping):
        raise NativeChatConfigurationError(
            "Native text inference requires tokenizer metadata in the manifest"
        )
    family = str(raw.get("family") or raw.get("tokenizer_family") or "").strip().lower()
    encoding_name = str(raw.get("encoding_name") or raw.get("tokenizer_name") or "").strip()
    tokenization = str(raw.get("tokenization") or "").strip().lower()
    backend = str(raw.get("backend") or "").strip().lower()
    tokenizer_path = raw.get("artifact_path") or raw.get("tokenizer_json")
    if tokenizer_path and (
        family in {"hf_tokenizer_json", "huggingface", "tokenizers", "tokenizers_bpe"}
        or backend in {"huggingface", "tokenizers"}
        or tokenization in {"hf_tokenizer_json", "tokenizers", "tokenizers_bpe"}
    ):
        if artifact_root is None:
            raise NativeChatConfigurationError(
                "tokenizer.json loading requires the Native Execution artifact root"
            )
        expected_sha = raw.get("sha256") or raw.get("tokenizer_sha256")
        expected_vocab = raw.get("vocab_size") or raw.get("tokenizer_vocab_size")
        path = _contained_artifact_file(
            Path(artifact_root), tokenizer_path, label="tokenizer.json"
        )
        required = MUSE_GLIMMER_SPECIAL_TOKEN_IDS if _is_muse_glimmer_manifest(manifest) else None
        codec = HuggingFaceTokenizerJSONCodec(
            path,
            expected_sha256=str(expected_sha or ""),
            expected_vocab_size=expected_vocab,
            required_special_token_ids=required,
        )
        if _is_muse_glimmer_manifest(manifest):
            if codec.sha256 != MUSE_GLIMMER_TOKENIZER_SHA256:
                raise NativeChatConfigurationError(
                    "Muse Glimmer tokenizer.json does not match the pinned artifact revision"
                )
            if codec.vocab_size != MUSE_GLIMMER_VOCAB_SIZE:
                raise NativeChatConfigurationError(
                    "Muse Glimmer tokenizer vocabulary must contain 202048 tokens"
                )
            _validate_muse_glimmer_tokenizer_config(Path(artifact_root), raw, codec)
        return codec
    if encoding_name and (
        family in {"", "tiktoken", "tiktoken_bpe", "gpt2_bpe"}
        or tokenization in {"", "gpt2_bpe", "tiktoken", "tiktoken_bpe"}
    ):
        return TiktokenTextCodec(encoding_name)
    raise NativeChatConfigurationError(
        "Native text inference requires an authenticated artifact tokenizer.json or an "
        "artifact-declared tiktoken encoding"
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


def _validated_muse_glimmer_atem_renderer(
    manifest: Mapping[str, Any],
    *,
    artifact_root: str | Path | None,
) -> MuseGlimmerATEMRenderer:
    metadata = manifest.get("chat_template")
    if not isinstance(metadata, Mapping):
        raise NativeChatConfigurationError(
            "Muse Glimmer requires authenticated muse_glimmer_atem_v1 chat metadata"
        )
    format_name = str(metadata.get("format") or metadata.get("profile") or "").strip().lower()
    if format_name not in {
        MUSE_GLIMMER_ATEM_PROFILE,
        "muse-glimmer-atem-v1",
    }:
        raise NativeChatConfigurationError(
            "Muse Glimmer requires chat_template.format=muse_glimmer_atem_v1; "
            "plain_roles and arbitrary Jinja are not compatible"
        )
    declared_sha = str(metadata.get("sha256") or metadata.get("template_sha256") or "").lower()
    template = metadata.get("template")
    if isinstance(template, str) and template:
        actual_sha = hashlib.sha256(template.encode("utf-8")).hexdigest()
    elif metadata.get("artifact_path") is not None:
        if artifact_root is None:
            raise NativeChatConfigurationError(
                "ATEM template validation requires the Native Execution artifact root"
            )
        template_path = _contained_artifact_file(
            Path(artifact_root), metadata.get("artifact_path"), label="ATEM template"
        )
        if template_path.stat().st_size > _MAX_TEMPLATE_BYTES:
            raise NativeChatConfigurationError("ATEM template exceeds the 1 MiB limit")
        actual_sha = _sha256_file(template_path)
    else:
        raise NativeChatConfigurationError(
            "Muse Glimmer ATEM metadata requires an inline template or contained artifact_path"
        )
    if _SHA256_PATTERN.fullmatch(declared_sha) is None or actual_sha != declared_sha:
        raise NativeChatConfigurationError(
            "ATEM template metadata must carry the exact SHA-256 of its template asset"
        )
    if declared_sha != MUSE_GLIMMER_ATEM_TEMPLATE_SHA256:
        raise NativeChatConfigurationError(
            "Muse Glimmer ATEM template does not match the pinned reviewed revision"
        )
    defaults = metadata.get("defaults")
    if defaults is None:
        defaults = {}
    if not isinstance(defaults, Mapping):
        raise NativeChatConfigurationError("chat_template.defaults must be an object")
    allowed_defaults = {"reasoning_strength", "knowledge_cutoff", "current_date"}
    unknown = set(defaults) - allowed_defaults
    if unknown:
        raise NativeChatConfigurationError(
            "Unsupported Muse Glimmer ATEM defaults: " + ", ".join(sorted(unknown))
        )
    return MuseGlimmerATEMRenderer(
        reasoning_strength=str(defaults.get("reasoning_strength") or "high"),
        knowledge_cutoff=str(defaults.get("knowledge_cutoff") or "2026-01-04"),
        current_date=(
            str(defaults["current_date"])
            if defaults.get("current_date") not in (None, "")
            else None
        ),
    )


def resolve_native_chat_renderer(
    manifest: Mapping[str, Any],
    selection: str,
    *,
    allow_auto_fallback: bool,
    artifact_root: str | Path | None = None,
) -> NativeChatRendererResolution:
    requested = str(selection or "auto").strip()
    if _is_muse_glimmer_manifest(manifest):
        if requested.lower() not in {
            "auto",
            MUSE_GLIMMER_ATEM_PROFILE,
            "muse-glimmer-atem-v1",
        }:
            raise NativeChatConfigurationError(
                "Muse Glimmer only supports the reviewed muse_glimmer_atem_v1 renderer"
            )
        return NativeChatRendererResolution(
            _validated_muse_glimmer_atem_renderer(
                manifest,
                artifact_root=artifact_root,
            )
        )
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
    if _is_muse_glimmer_manifest(manifest):
        expected = (MUSE_GLIMMER_SPECIAL_TOKEN_IDS["eos"], MUSE_GLIMMER_SPECIAL_TOKEN_IDS["eot"])
        if tuple(values) != expected:
            raise NativeChatConfigurationError(
                "Muse Glimmer stop_tokens must be [200001, 200008]; <|eom|> (200007) "
                "is a message boundary, not a generation stop"
            )
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
    "HuggingFaceTokenizerJSONCodec",
    "IncrementalTokenDecoder",
    "MUSE_GLIMMER_ATEM_PROFILE",
    "MUSE_GLIMMER_ATEM_TEMPLATE_SHA256",
    "MUSE_GLIMMER_ADDED_TOKENS_SHA256",
    "MUSE_GLIMMER_ARTIFACT_REVISION",
    "MUSE_GLIMMER_SPECIAL_TOKEN_IDS",
    "MUSE_GLIMMER_TOKENIZER_CONFIG_SHA256",
    "MUSE_GLIMMER_TOKENIZER_SHA256",
    "MUSE_GLIMMER_VOCAB_SIZE",
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
    "MuseGlimmerATEMRenderer",
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
