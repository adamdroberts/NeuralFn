"""Process-local CLI workflow for resident Native Execution artifacts."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, ContextManager, Protocol, Sequence, TextIO

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
    parse_native_assistant_response,
    read_native_execution_manifest,
    resolve_native_chat_prompt,
    resolve_native_chat_renderer,
)
from .native_inference import (
    GenerationConfig,
    KVCacheConfig,
    NativeInferenceModel,
    NativeModelLoadConfig,
)


class NativeArtifactCLIUI(Protocol):
    """Optional interactive presentation adapter for the lean native driver."""

    def handle(self, event: str, **payload: Any) -> None: ...

    def read_line(self, prompt: str) -> str: ...

    def progress(self, label: str) -> ContextManager[Any]: ...


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
    interactive_ui: NativeArtifactCLIUI | None = None,
) -> int:
    """Run one resident model/session for a one-shot or interactive process."""

    output = stdout or sys.stdout
    errors = stderr or sys.stderr
    if interactive_ui is not None and not interactive:
        raise ValueError("interactive_ui requires interactive=True")
    read_line = input_fn or (
        interactive_ui.read_line if interactive_ui is not None else input
    )

    def emit(event: str, **payload: Any) -> None:
        if interactive_ui is not None:
            interactive_ui.handle(event, **payload)

    def warn(message: str) -> None:
        if interactive_ui is not None:
            emit("warning", message=message)
        else:
            print(f"warning: {message}", file=errors)

    artifact_root, _manifest_path, manifest = read_native_execution_manifest(config.artifact)
    if codec is not None:
        text_codec = codec
    elif config.prompt_token_ids and not interactive:
        try:
            text_codec = load_native_text_codec(manifest, artifact_root=artifact_root)
        except NativeChatConfigurationError:
            text_codec = TokenIdTextCodec()
            warn(
                "artifact text tokenizer is unavailable; rendering generated "
                "token IDs because --prompt-tokens was used"
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
        warn(renderer_resolution.warning)
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
                ready_payload = {
                    "model_name": _model_name(manifest, config.artifact),
                    "artifact": config.artifact,
                    "mode": mode,
                    "stats": stats,
                    "context_limit": context_limit,
                    "renderer_name": renderer.name,
                    "config": config,
                }
                if interactive_ui is not None:
                    emit("ready", **ready_payload)
                else:
                    print(
                        "Native resident inference ready: "
                        f"{ready_payload['model_name']} "
                        f"(mode={mode}, cache={stats.get('effective_cache', 'unknown')}).",
                        file=output,
                    )
                    print(
                        "Commands: /mode stateless|transcript, /show, /reset, "
                        "/clear, /help, /exit",
                        file=output,
                    )

            def decode_prefilled(prompt_token_ids: Sequence[int]):
                if len(prompt_token_ids) + config.max_new_tokens > context_limit:
                    raise NativeChatConfigurationError(
                        f"Prompt uses {len(prompt_token_ids)} tokens plus "
                        f"{config.max_new_tokens} reserved output tokens, exceeding the "
                        f"{context_limit}-token context window."
                    )
                progress = (
                    interactive_ui.progress("thinking")
                    if interactive_ui is not None
                    else nullcontext()
                )
                with progress:
                    prefill_started = time.perf_counter()
                    prefill_stats = session.prefill(prompt_token_ids)
                    decode_started = time.perf_counter()
                    result = session.decode(generation)
                    finished = time.perf_counter()
                decoded = text_codec.decode(result.token_ids)
                response = parse_native_assistant_response(
                    decoded,
                    renderer,
                    delimiters=delimiters,
                )
                return (
                    response,
                    result,
                    dict(prefill_stats) if isinstance(prefill_stats, dict) else {},
                    decode_started - prefill_started,
                    finished - decode_started,
                )

            turn_index = 0
            def respond(user_text: str) -> None:
                nonlocal history, turn_index
                prepared = resolve_native_chat_prompt(
                    codec=text_codec,
                    renderer=renderer,
                    mode=mode,
                    history=history,
                    draft=user_text,
                    context_limit=context_limit,
                    reserved_output_tokens=config.max_new_tokens,
                )
                response, result, prefill_stats, prefill_seconds, decode_seconds = (
                    decode_prefilled(prepared.token_ids)
                )
                turn_index += 1
                if prepared.dropped_groups:
                    warn(
                        f"trimmed {prepared.dropped_groups} oldest conversation "
                        f"group{'s' if prepared.dropped_groups != 1 else ''} to fit context"
                    )
                if interactive_ui is not None:
                    emit(
                        "turn",
                        index=turn_index,
                        user_text=user_text,
                        response=response,
                        result=result,
                        prefill_stats=prefill_stats,
                        prefill_seconds=prefill_seconds,
                        decode_seconds=decode_seconds,
                    )
                else:
                    print(response.visible_text, file=output)
                if response.used_channel_protocol and not response.visible_text:
                    warn(
                        "Muse Glimmer did not finish a to=user answer; private to=self "
                        "content was hidden and was not added to transcript history. "
                        "Increase --max-new-tokens if generation ended at the token limit."
                    )
                elif (
                    response.used_channel_protocol
                    and not response.final_channel_complete
                ):
                    warn(
                        "Muse Glimmer's to=user answer ended before the ATEM end-of-turn marker"
                    )
                if mode == "transcript" and response.visible_text:
                    history.append(NativeChatMessage("user", user_text))
                    history.append(
                        NativeChatMessage("assistant", response.visible_text)
                    )

            if config.prompt_token_ids:
                response, result, prefill_stats, prefill_seconds, decode_seconds = (
                    decode_prefilled(config.prompt_token_ids)
                )
                if interactive_ui is not None:
                    turn_index += 1
                    emit(
                        "turn",
                        index=turn_index,
                        user_text=f"[tokens] {','.join(map(str, config.prompt_token_ids))}",
                        response=response,
                        result=result,
                        prefill_stats=prefill_stats,
                        prefill_seconds=prefill_seconds,
                        decode_seconds=decode_seconds,
                    )
                else:
                    print(response.visible_text, file=output)
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
                    emit("goodbye")
                    return 0
                if message == "/help":
                    if interactive_ui is not None:
                        emit("help")
                    else:
                        print(
                            "/mode stateless|transcript  /show  /reset  /clear  "
                            "/help  /exit",
                            file=output,
                        )
                    continue
                if message.startswith("/mode "):
                    requested = message.split(None, 1)[1].strip().lower()
                    if requested not in {"stateless", "transcript"}:
                        warn("usage: /mode stateless|transcript")
                        continue
                    mode = requested
                    if interactive_ui is not None:
                        emit("mode", mode=mode)
                    else:
                        print(f"Mode: {mode}", file=output)
                    continue
                if message in {"/show", "/stats"}:
                    if interactive_ui is not None:
                        emit(
                            "show",
                            mode=mode,
                            stats=model.stats(),
                            history_messages=len(history),
                            config=config,
                        )
                    else:
                        print(json.dumps(model.stats(), sort_keys=True), file=output)
                    continue
                if message == "/reset":
                    history.clear()
                    if config.system_prompt:
                        history.append(NativeChatMessage("system", config.system_prompt))
                    session.reset()
                    turn_index = 0
                    if interactive_ui is not None:
                        emit("reset")
                    else:
                        print("Transcript and resident session reset.", file=output)
                    continue
                if message == "/clear":
                    if interactive_ui is not None:
                        emit("clear")
                    else:
                        print("\033[2J\033[H", end="", file=output)
                    continue
                if message.startswith("/"):
                    warn(f"unknown command {message.split()[0]!r}; try /help")
                    continue
                respond(message)


__all__ = [
    "NativeArtifactCLIConfig",
    "NativeArtifactCLIUI",
    "parse_native_prompt_token_ids",
    "run_native_artifact_cli",
]
