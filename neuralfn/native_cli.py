"""Process-local CLI workflow for resident Native Execution artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
from typing import Any, Callable, Sequence, TextIO

from .native_chat import (
    NativeChatConfigurationError,
    NativeChatMessage,
    NativeTextCodec,
    TokenIdTextCodec,
    load_native_text_codec,
    native_context_limit,
    native_output_limit,
    native_stop_token_ids,
    native_text_stop_delimiters,
    read_native_execution_manifest,
    resolve_native_chat_prompt,
    resolve_native_chat_renderer,
    strip_native_text_delimiters,
)
from .native_inference import (
    GenerationConfig,
    KVCacheConfig,
    NativeInferenceModel,
    NativeModelLoadConfig,
)


@dataclass(frozen=True, slots=True)
class NativeArtifactCLIConfig:
    artifact: Path
    prompt: str = ""
    prompt_token_ids: tuple[int, ...] = ()
    chat_mode: str | None = None
    system_prompt: str = ""
    chat_template: str = "auto"
    max_new_tokens: int = 64
    temperature: float = 0.8
    top_k: int | None = 32
    top_p: float = 1.0
    seed: int = 1337
    kv_cache: KVCacheConfig = field(default_factory=KVCacheConfig)
    model_load: NativeModelLoadConfig = field(default_factory=NativeModelLoadConfig)
    native_info: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact", Path(self.artifact).expanduser())
        object.__setattr__(self, "prompt", str(self.prompt))
        object.__setattr__(self, "system_prompt", str(self.system_prompt).strip())
        template = str(self.chat_template or "auto").strip()
        object.__setattr__(self, "chat_template", template or "auto")
        if self.chat_mode is not None:
            mode = str(self.chat_mode).strip().lower()
            if mode not in {"stateless", "transcript"}:
                raise ValueError("chat_mode must be stateless, transcript, or None")
            object.__setattr__(self, "chat_mode", mode)
        if isinstance(self.max_new_tokens, bool) or not isinstance(self.max_new_tokens, int):
            raise TypeError("max_new_tokens must be an integer")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        normalized_tokens: list[int] = []
        for index, token_id in enumerate(self.prompt_token_ids):
            if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
                raise ValueError(f"prompt_token_ids[{index}] must be a non-negative integer")
            normalized_tokens.append(token_id)
        object.__setattr__(self, "prompt_token_ids", tuple(normalized_tokens))
        if self.prompt and self.prompt_token_ids:
            raise ValueError("Specify either --prompt or --prompt-tokens, not both")


def parse_native_prompt_token_ids(raw: str) -> tuple[int, ...]:
    text = str(raw).strip()
    if not text:
        return ()
    values: list[int] = []
    for index, item in enumerate(text.split(",")):
        item = item.strip()
        if not item:
            raise ValueError(f"Prompt token position {index} is empty")
        try:
            value = int(item)
        except ValueError as exc:
            raise ValueError(f"Prompt token {item!r} is not an integer") from exc
        if value < 0:
            raise ValueError("Prompt token ids must be non-negative")
        values.append(value)
    return tuple(values)


def _model_name(manifest: dict[str, Any], artifact: Path) -> str:
    model = manifest.get("model")
    if isinstance(model, dict):
        value = model.get("name") or model.get("family")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return artifact.stem


def run_native_artifact_cli(
    config: NativeArtifactCLIConfig,
    *,
    interactive: bool,
    binding: Any | None = None,
    codec: NativeTextCodec | None = None,
    input_fn: Callable[[str], str] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run one resident model/session for a one-shot or interactive process."""

    output = stdout or sys.stdout
    errors = stderr or sys.stderr
    read_line = input_fn or input
    artifact_root, _manifest_path, manifest = read_native_execution_manifest(config.artifact)
    if codec is not None:
        text_codec = codec
    elif config.prompt_token_ids and not interactive:
        try:
            text_codec = load_native_text_codec(manifest, artifact_root=artifact_root)
        except NativeChatConfigurationError:
            text_codec = TokenIdTextCodec()
            print(
                "warning: artifact text tokenizer is unavailable; rendering generated "
                "token IDs because --prompt-tokens was used",
                file=errors,
            )
    else:
        text_codec = load_native_text_codec(manifest, artifact_root=artifact_root)
    renderer_resolution = resolve_native_chat_renderer(
        manifest,
        config.chat_template,
        allow_auto_fallback=True,
        artifact_root=artifact_root,
    )
    renderer = renderer_resolution.renderer
    if renderer_resolution.warning:
        print(f"warning: {renderer_resolution.warning}", file=errors)
    context_limit = native_context_limit(manifest)
    output_limit = native_output_limit(manifest)
    if output_limit is not None and config.max_new_tokens > output_limit:
        raise NativeChatConfigurationError(
            f"--max-new-tokens {config.max_new_tokens} exceeds the artifact output limit "
            f"of {output_limit}"
        )
    stop_token_ids = native_stop_token_ids(manifest)
    delimiters = native_text_stop_delimiters(manifest, renderer)
    mode = config.chat_mode or ("transcript" if interactive else "stateless")
    generation = GenerationConfig(
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
        seed=config.seed,
        stop_token_ids=stop_token_ids,
    )
    history: list[NativeChatMessage] = []
    if config.system_prompt:
        history.append(NativeChatMessage("system", config.system_prompt))

    with NativeInferenceModel.load(
        config.artifact,
        binding=binding,
        kv_cache=config.kv_cache,
        load_config=config.model_load,
    ) as model:
        with model.create_session(seed=config.seed) as session:
            if config.native_info:
                print(
                    json.dumps(
                        {
                            "model": _model_name(manifest, config.artifact),
                            "manifest": str(model.manifest_path),
                            "stats": model.stats(),
                        },
                        sort_keys=True,
                    ),
                    file=output,
                )

            if interactive:
                stats = model.stats()
                print(
                    "Native resident inference ready: "
                    f"{_model_name(manifest, config.artifact)} "
                    f"(mode={mode}, cache={stats.get('effective_cache', 'unknown')}).",
                    file=output,
                )
                print("Commands: /mode stateless|transcript, /reset, /help, /exit", file=output)

            def decode_prefilled(prompt_token_ids: Sequence[int]) -> str:
                if len(prompt_token_ids) + config.max_new_tokens > context_limit:
                    raise NativeChatConfigurationError(
                        f"Prompt uses {len(prompt_token_ids)} tokens plus "
                        f"{config.max_new_tokens} reserved output tokens, exceeding the "
                        f"{context_limit}-token context window."
                    )
                session.prefill(prompt_token_ids)
                result = session.decode(generation)
                decoded = text_codec.decode(result.token_ids)
                return strip_native_text_delimiters(decoded, delimiters)

            def respond(user_text: str) -> None:
                nonlocal history
                prepared = resolve_native_chat_prompt(
                    codec=text_codec,
                    renderer=renderer,
                    mode=mode,
                    history=history,
                    draft=user_text,
                    context_limit=context_limit,
                    reserved_output_tokens=config.max_new_tokens,
                )
                response = decode_prefilled(prepared.token_ids)
                if prepared.dropped_groups:
                    print(
                        f"warning: trimmed {prepared.dropped_groups} oldest conversation "
                        f"group{'s' if prepared.dropped_groups != 1 else ''} to fit context",
                        file=errors,
                    )
                print(response, file=output)
                if mode == "transcript":
                    history.append(NativeChatMessage("user", user_text))
                    history.append(NativeChatMessage("assistant", response))

            if config.prompt_token_ids:
                print(decode_prefilled(config.prompt_token_ids), file=output)
            elif config.prompt:
                respond(config.prompt)
            elif not interactive and not config.native_info:
                raise NativeChatConfigurationError(
                    "Non-interactive native inference requires --prompt or --prompt-tokens"
                )

            if not interactive:
                return 0

            while True:
                try:
                    raw = read_line("nfn> ")
                except EOFError:
                    return 0
                message = raw.strip()
                if not message:
                    continue
                if message in {"/exit", "/quit"}:
                    return 0
                if message == "/help":
                    print(
                        "/mode stateless|transcript  /reset  /help  /exit",
                        file=output,
                    )
                    continue
                if message.startswith("/mode "):
                    requested = message.split(None, 1)[1].strip().lower()
                    if requested not in {"stateless", "transcript"}:
                        print("warning: usage: /mode stateless|transcript", file=errors)
                        continue
                    mode = requested
                    print(f"Mode: {mode}", file=output)
                    continue
                if message == "/reset":
                    history.clear()
                    if config.system_prompt:
                        history.append(NativeChatMessage("system", config.system_prompt))
                    session.reset()
                    print("Transcript and resident session reset.", file=output)
                    continue
                if message.startswith("/"):
                    print(f"warning: unknown command {message.split()[0]!r}", file=errors)
                    continue
                respond(message)


__all__ = [
    "NativeArtifactCLIConfig",
    "parse_native_prompt_token_ids",
    "run_native_artifact_cli",
]
