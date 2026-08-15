from __future__ import annotations

import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

import pytest

from neuralfn.native_chat import (
    NativeChatConfigurationError,
    NativeChatMessage,
    NativeChatRendererResolution,
    NativeTextCodec,
    MuseGlimmerATEMRenderer,
    PlainRolesRenderer,
    resolve_native_chat_prompt,
    resolve_native_chat_renderer,
)
from neuralfn.native_cli import NativeArtifactCLIConfig, run_native_artifact_cli
from neuralfn.native_inference import (
    GenerationEvent,
    GenerationResult,
    KVCacheConfig,
    NativeInferenceCapabilities,
)
from neuralfn.native_registry import native_graph_training_adapters
import neuralfn.native_cli as native_cli


ROOT = Path(__file__).resolve().parents[2]
NFN = ROOT / "cli" / "nfn.py"
READY_NATIVE_TEXT_ADAPTERS = tuple(
    (adapter.selector, adapter.family)
    for adapter in native_graph_training_adapters()
    if adapter.architecture_persistence_proven
)
EXPECTED_READY_NATIVE_TEXT_ADAPTERS = (
    ("gpt2", "gpt2"),
    ("gpt2_megakernel", "gpt2"),
    ("gpt2_moa", "gpt2"),
    ("gpt2_qknorm", "gpt2"),
    ("gpt2_softcap", "gpt2"),
    ("gpt2_stable", "gpt2"),
    ("gpt2_zloss", "gpt2"),
    ("llama", "llama"),
    ("llama_fast", "llama"),
    ("moe", "mixllama"),
    ("mixllama", "mixllama"),
    ("mixllama_fast", "mixllama"),
    ("muse_glimmer", "muse-glimmer"),
)


class CharacterCodec(NativeTextCodec):
    name = "character-test-codec"

    def encode(self, text: str) -> tuple[int, ...]:
        return tuple(ord(character) for character in text)

    def decode(self, token_ids: Sequence[int]) -> str:
        return "".join(chr(int(token_id)) for token_id in token_ids)

    def token_bytes(self, token_id: int) -> bytes:
        return chr(token_id).encode("utf-8")


class FakeSession:
    def __init__(self, model: "FakeModel") -> None:
        self.model = model
        self.tokens: list[int] = []
        self.prefills: list[tuple[tuple[int, ...], int]] = []
        self.generations = []
        self.reset_count = 0
        self.closed = False

    def __enter__(self) -> "FakeSession":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def prefill(self, token_ids: Sequence[int]) -> dict[str, int]:
        target = tuple(token_ids)
        common = 0
        while common < min(len(self.tokens), len(target)) and self.tokens[common] == target[common]:
            common += 1
        self.tokens = list(target)
        self.prefills.append((target, common))
        return {
            "prefix_tokens": len(target),
            "prefix_reused": common,
            "prefilled_tokens": len(target) - common,
        }

    def decode(self, generation, *, on_token=None) -> GenerationResult:
        self.generations.append(generation)
        text = self.model.outputs.pop(0) if self.model.outputs else ""
        token_ids = tuple(ord(character) for character in text[: generation.max_new_tokens])
        prompt_tokens = len(self.tokens)
        events = []
        for index, token_id in enumerate(token_ids):
            self.tokens.append(token_id)
            event = GenerationEvent(
                token_id=token_id,
                index=index,
                position=len(self.tokens) - 1,
            )
            events.append(event)
            if on_token is not None:
                on_token(event)
        return GenerationResult(
            token_ids=token_ids,
            text=text,
            finish_reason="stop",
            prompt_tokens=prompt_tokens,
            completion_tokens=len(token_ids),
            events=tuple(events),
        )

    def reset(self) -> None:
        self.tokens.clear()
        self.reset_count += 1

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.model.session_closes += 1


class FakeModel:
    def __init__(self, manifest_path: Path, outputs: Sequence[str]) -> None:
        self.manifest_path = manifest_path
        self.outputs = list(outputs)
        self.capabilities = NativeInferenceCapabilities(
            native_inference=True,
            resident_inference=True,
            lossless_kv_cache=True,
            turboquant_kv_cache=False,
        )
        self.session = FakeSession(self)
        self.session_creates = 0
        self.session_closes = 0
        self.model_closes = 0
        self.load_kwargs: dict[str, Any] = {}

    def __enter__(self) -> "FakeModel":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def create_session(self, *, seed: int = 0) -> FakeSession:
        assert seed == 17
        self.session_creates += 1
        return self.session

    def stats(self) -> dict[str, Any]:
        return {
            "backend": "fake-resident-test",
            "requested_cache": self.load_kwargs["kv_cache"].mode,
            "effective_cache": self.load_kwargs["kv_cache"].mode,
        }

    def close(self) -> None:
        self.model_closes += 1


def _manifest_payload(
    *,
    template: Any = "plain_roles",
    context: int = 512,
    model_name: str = "native-cli-test",
    family: str = "gpt2",
) -> dict[str, Any]:
    return {
        "schema": "neuralfn.native_execution_manifest",
        "version": 1,
        "model": {"name": model_name, "family": family},
        "tokenizer": {
            "family": "tiktoken",
            "encoding_name": "fixture",
            "role_delimiters": ["<STOP>"],
        },
        "chat_template": {"source": "artifact", "template": template},
        "context_limits": {"max_context_tokens": context, "max_output_tokens": 128},
        "stop_tokens": [999],
        "checkpoint": {"artifact_path": "model.bin"},
        "capabilities": {
            "native_inference": True,
            "resident_inference": True,
            "lossless_kv_cache": True,
        },
        "kernel_abi": {"resident_inference": {"version": 1, "status": "ready"}},
    }


def _artifact(
    tmp_path: Path,
    *,
    template: Any = "plain_roles",
    context: int = 512,
    model_name: str = "native-cli-test",
    family: str = "gpt2",
) -> Path:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "native-execution-manifest.json").write_text(
        json.dumps(
            _manifest_payload(
                template=template,
                context=context,
                model_name=model_name,
                family=family,
            )
        ),
        encoding="utf-8",
    )
    (artifact / "model.bin").write_bytes(b"native-checkpoint-placeholder")
    return artifact


def _install_fake_loader(monkeypatch: pytest.MonkeyPatch, model: FakeModel) -> None:
    class Loader:
        @staticmethod
        def load(_artifact, **kwargs):
            model.load_kwargs = kwargs
            return model

    monkeypatch.setattr(native_cli, "NativeInferenceModel", Loader)


def _load_nfn_module():
    spec = importlib.util.spec_from_file_location("nfn_native_ir_infer_test", NFN)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_artifact_dispatch_precedes_legacy_bin_detection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _artifact(tmp_path)
    module = _load_nfn_module()
    calls = []
    manifest_path = artifact / "native-execution-manifest.json"
    assert module._resolve_native_ir_manifest(
        ["infer", "--checkpoint", str(artifact)]
    ) == manifest_path
    assert module._resolve_native_ir_manifest(
        ["infer", "--checkpoint", str(manifest_path)]
    ) == manifest_path
    assert module._resolve_native_ir_manifest(
        ["infer", "--checkpoint", str(artifact / "model.bin")]
    ) == manifest_path

    def resident(argv, **kwargs):
        calls.append((argv, kwargs))
        return 23

    monkeypatch.setattr(module, "_native_ir_infer_main", resident)
    monkeypatch.setattr(
        module,
        "_is_lightweight_native_gpt_infer",
        lambda _argv: (_ for _ in ()).throw(AssertionError("legacy .bin route ran first")),
    )

    result = module.main(
        ["infer", "--checkpoint", str(artifact / "model.bin"), "--prompt", "hello"],
        stdin_isatty=False,
        stdout_isatty=False,
    )

    assert result == 23
    assert calls[0][1] == {"stdin_isatty": False, "stdout_isatty": False}


def test_checkpoint_file_dispatch_requires_exact_contained_manifest_binding(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path)
    manifest_path = artifact / "native-execution-manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["checkpoint"]["artifact_path"] = "different.bin"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    (artifact / "different.bin").write_bytes(b"different-native-checkpoint")
    module = _load_nfn_module()

    assert module._resolve_native_ir_manifest(
        ["infer", "--checkpoint", str(artifact / "model.bin")]
    ) is None
    assert module._resolve_native_ir_manifest(
        ["infer", "--checkpoint", str(artifact / "different.bin")]
    ) == manifest_path

    payload["checkpoint"]["artifact_path"] = str((artifact / "model.bin").resolve())
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    assert module._resolve_native_ir_manifest(
        ["infer", "--checkpoint", str(artifact / "model.bin")]
    ) is None


def test_checkpoint_file_dispatch_rejects_symlink_escape(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    checkpoint = artifact / "model.bin"
    checkpoint.unlink()
    outside = tmp_path / "outside-model.bin"
    outside.write_bytes(b"outside-native-checkpoint")
    checkpoint.symlink_to(outside)
    module = _load_nfn_module()

    assert module._resolve_native_ir_manifest(
        ["infer", "--checkpoint", str(checkpoint)]
    ) is None


def test_non_tty_native_artifact_inference_is_one_resident_one_shot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _artifact(tmp_path)
    model = FakeModel(artifact / "native-execution-manifest.json", ["answer<STOP>ignored"])
    _install_fake_loader(monkeypatch, model)
    stdout = io.StringIO()
    stderr = io.StringIO()

    result = run_native_artifact_cli(
        NativeArtifactCLIConfig(
            artifact=artifact,
            prompt="hello",
            max_new_tokens=64,
            seed=17,
            kv_cache=KVCacheConfig(mode="full", turboquant_profile="qjl-3.5"),
        ),
        interactive=False,
        codec=CharacterCodec(),
        stdout=stdout,
        stderr=stderr,
    )

    assert result == 0
    assert stdout.getvalue() == "answer\n"
    assert stderr.getvalue() == ""
    assert model.session_creates == model.session_closes == 1
    assert model.model_closes == 1
    assert model.load_kwargs["kv_cache"].mode == "full"
    prompt, reused = model.session.prefills[0]
    assert reused == 0
    assert "<|user|>\nhello" in CharacterCodec().decode(prompt)
    assert model.session.generations[0].stop_token_ids == (999,)


def test_raw_prompt_tokens_bypass_chat_rendering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _artifact(tmp_path)
    model = FakeModel(artifact / "native-execution-manifest.json", ["X"])
    _install_fake_loader(monkeypatch, model)
    stdout = io.StringIO()

    result = run_native_artifact_cli(
        NativeArtifactCLIConfig(
            artifact=artifact,
            prompt_token_ids=(7, 8, 9),
            max_new_tokens=1,
            seed=17,
        ),
        interactive=False,
        codec=CharacterCodec(),
        stdout=stdout,
        stderr=io.StringIO(),
    )

    assert result == 0
    assert model.session.prefills == [((7, 8, 9), 0)]
    assert stdout.getvalue() == "X\n"


def test_ready_native_text_adapter_matrix_is_explicit_and_excludes_unproved_diff() -> None:
    assert READY_NATIVE_TEXT_ADAPTERS == EXPECTED_READY_NATIVE_TEXT_ADAPTERS
    assert all(selector != "gpt2_diff" for selector, _family in READY_NATIVE_TEXT_ADAPTERS)


@pytest.mark.parametrize(
    ("selector", "family"),
    READY_NATIVE_TEXT_ADAPTERS,
    ids=[selector for selector, _family in READY_NATIVE_TEXT_ADAPTERS],
)
def test_shared_resident_transcript_driver_contract_for_every_ready_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selector: str,
    family: str,
) -> None:
    artifact = _artifact(
        tmp_path,
        context=1024,
        model_name=selector,
        family=family,
    )
    muse_outputs = [
        f" to=user<|message|>{letter}<|eot|>" for letter in "ABCDE"
    ]
    outputs = muse_outputs if selector == "muse_glimmer" else [
        "A<STOP>ignored", "B", "C", "D", "E"
    ]
    model = FakeModel(artifact / "native-execution-manifest.json", outputs)
    _install_fake_loader(monkeypatch, model)
    if selector == "muse_glimmer":
        manifest_path = artifact / "native-execution-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["stop_tokens"] = [200_001, 200_008]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        monkeypatch.setattr(
            native_cli,
            "resolve_native_chat_renderer",
            lambda *_args, **_kwargs: NativeChatRendererResolution(
                MuseGlimmerATEMRenderer(current_date="2026-08-12")
            ),
        )
    commands = iter(
        [
            "second",
            "/mode stateless",
            "third",
            "/mode transcript",
            "fourth",
            "/reset",
            "after",
            "/exit",
        ]
    )
    stdout = io.StringIO()

    result = run_native_artifact_cli(
        NativeArtifactCLIConfig(
            artifact=artifact,
            prompt="first",
            system_prompt="keep this",
            max_new_tokens=64 if selector == "muse_glimmer" else 8,
            seed=17,
        ),
        interactive=True,
        codec=CharacterCodec(),
        input_fn=lambda _prompt: next(commands),
        stdout=stdout,
        stderr=io.StringIO(),
    )

    assert result == 0
    assert model.session_creates == model.session_closes == 1
    assert len(model.session.prefills) == 5
    first_tokens, first_reuse = model.session.prefills[0]
    second_tokens, second_reuse = model.session.prefills[1]
    third_tokens, _third_reuse = model.session.prefills[2]
    fourth_tokens, _fourth_reuse = model.session.prefills[3]
    after_tokens, after_reuse = model.session.prefills[4]
    assert first_reuse == 0
    expected_first_completion = (
        len(muse_outputs[0]) if selector == "muse_glimmer" else 1
    )
    assert second_reuse == len(first_tokens) + expected_first_completion
    expected_completion_tokens = (
        tuple(map(ord, muse_outputs[0]))
        if selector == "muse_glimmer"
        else (ord("A"),)
    )
    assert second_tokens[:second_reuse] == (*first_tokens, *expected_completion_tokens)
    assert "ignored" not in CharacterCodec().decode(second_tokens)
    third_text = CharacterCodec().decode(third_tokens)
    assert "keep this" in third_text and "third" in third_text and "first" not in third_text
    fourth_text = CharacterCodec().decode(fourth_tokens)
    assert "first" in fourth_text and "second" in fourth_text and "fourth" in fourth_text
    assert "third" not in fourth_text
    assert model.session.reset_count == 1
    assert after_reuse == 0
    after_text = CharacterCodec().decode(after_tokens)
    assert "keep this" in after_text and "after" in after_text and "first" not in after_text
    rendered_output = stdout.getvalue()
    assert selector in rendered_output
    assert "mode=transcript" in rendered_output
    assert "Mode: stateless" in rendered_output
    assert "Transcript and resident session reset." in rendered_output


def test_native_prompt_trimming_preserves_instructions_and_newest_group() -> None:
    codec = CharacterCodec()
    renderer = PlainRolesRenderer()
    history = [
        NativeChatMessage("system", "keep this"),
        NativeChatMessage("user", "a" * 30),
        NativeChatMessage("assistant", "b" * 30),
        NativeChatMessage("user", "c" * 8),
        NativeChatMessage("assistant", "d" * 8),
        NativeChatMessage("tool", "tool-result", tool_call_id="call-1"),
    ]

    prepared = resolve_native_chat_prompt(
        codec=codec,
        renderer=renderer,
        mode="transcript",
        history=history,
        draft="newest",
        context_limit=145,
        reserved_output_tokens=20,
    )

    assert prepared.dropped_groups >= 1
    assert "keep this" in prepared.text
    assert "newest" in prepared.text
    assert "tool-result" in prepared.text
    assert "a" * 30 not in prepared.text
    assert len(prepared.token_ids) <= 125


def test_native_prompt_fails_when_mandatory_remainder_exceeds_budget() -> None:
    with pytest.raises(
        NativeChatConfigurationError,
        match="leading instructions and newest user/tool turn",
    ):
        resolve_native_chat_prompt(
            codec=CharacterCodec(),
            renderer=PlainRolesRenderer(),
            mode="transcript",
            history=[NativeChatMessage("system", "s" * 30)],
            draft="u" * 30,
            context_limit=80,
            reserved_output_tokens=16,
        )


def test_auto_template_falls_back_with_warning_and_explicit_template_is_data(
    tmp_path: Path,
) -> None:
    manifest = _manifest_payload(template=None)
    resolution = resolve_native_chat_renderer(
        manifest,
        "auto",
        allow_auto_fallback=True,
    )
    assert resolution.renderer.name == "plain_roles"
    assert resolution.warning and "using plain_roles" in resolution.warning

    template = tmp_path / "chat.txt"
    template.write_text("BEGIN\n{{messages}}\nNEXT={{assistant_prompt}}\nEND", encoding="utf-8")
    explicit = resolve_native_chat_renderer(
        manifest,
        str(template),
        allow_auto_fallback=True,
    ).renderer
    assert explicit.render(
        [NativeChatMessage("user", "hello {messages} {assistant_prompt}")],
        include_assistant_prompt=True,
    ) == (
        "BEGIN\n<|user|>\nhello {messages} {assistant_prompt}\n\n"
        "NEXT=<|assistant|>\n\nEND"
    )


@pytest.mark.parametrize("tile_flag", ["--tile-ops-lib", "--strict-tile-ops-lib"])
def test_native_cli_flag_plumbing_uses_tty_default_and_cache_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tile_flag: str,
) -> None:
    artifact = _artifact(tmp_path)
    tile_sidecar = tmp_path / "libtile-strict.so"
    tile_sidecar.write_bytes(b"fixture")
    module = _load_nfn_module()
    captured = []
    monkeypatch.setattr(
        "neuralfn.native_cli.run_native_artifact_cli",
        lambda config, **kwargs: captured.append((config, kwargs)) or 0,
    )

    result = module._native_ir_infer_main(
        [
            "infer",
            "--checkpoint",
            str(artifact / "model.bin"),
            "--prompt",
            "hello",
            "--system-prompt",
            "rules",
            "--chat-template",
            "plain_roles",
            "--kv-cache",
            "turboquant",
            "--turboquant-profile",
            "qjl-3.5",
            "--turboquant-attention-backend",
            "tile-cuda",
            tile_flag,
            str(tile_sidecar),
            "--cuda-runtime-lib",
            "libcudart.so.13",
            "--cuda-device",
            "2",
            "--max-new-tokens",
            "12",
            "--seed",
            "17",
        ],
        stdin_isatty=True,
        stdout_isatty=True,
    )

    assert result == 0
    config, kwargs = captured[0]
    assert kwargs["interactive"] is True
    assert kwargs["interactive_ui"].__class__.__name__ == "RichNativeInferenceUI"
    assert config.artifact == artifact / "native-execution-manifest.json"
    assert config.chat_mode is None
    assert config.system_prompt == "rules"
    assert config.kv_cache.mode == "turboquant"
    assert config.kv_cache.turboquant_profile == "qjl-3.5"
    assert config.kv_cache.turboquant_attention_backend == "tile-cuda"
    assert config.kv_cache.tile_ops_lib == str(tile_sidecar)
    assert config.kv_cache.cuda_runtime_lib == "libcudart.so.13"
    assert config.kv_cache.cuda_device == 2
    assert config.model_load.tile_ops_lib == str(tile_sidecar)
    assert config.model_load.cuda_runtime_lib == "libcudart.so.13"
    assert config.model_load.cuda_device == 2
    assert config.max_new_tokens == 12


def test_native_cli_default_allows_atem_reasoning_to_reach_final_channel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _artifact(tmp_path)
    module = _load_nfn_module()
    captured = []
    monkeypatch.setattr(
        "neuralfn.native_cli.run_native_artifact_cli",
        lambda config, **kwargs: captured.append((config, kwargs)) or 0,
    )

    assert module._native_ir_infer_main(
        ["infer", "--checkpoint", str(artifact), "--prompt", "hello"],
        stdin_isatty=False,
        stdout_isatty=False,
    ) == 0

    config, kwargs = captured[0]
    assert config.max_new_tokens == 512
    assert kwargs == {"interactive": False, "interactive_ui": None}


def test_glimmer_transcript_retains_only_user_directed_atem_answer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _artifact(tmp_path, family="muse_glimmer", context=2048)
    manifest_path = artifact / "native-execution-manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["stop_tokens"] = [200_001, 200_008]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    outputs = [
        " to=self<|message|>private reasoning<|eom|>"
        "<|start|>assistant to=user<|message|>Visible A<|eot|>",
        " to=user<|message|>Visible B<|eot|>",
    ]
    model = FakeModel(manifest_path, outputs)
    _install_fake_loader(monkeypatch, model)
    monkeypatch.setattr(
        native_cli,
        "resolve_native_chat_renderer",
        lambda *_args, **_kwargs: NativeChatRendererResolution(
            MuseGlimmerATEMRenderer(current_date="2026-08-12")
        ),
    )
    commands = iter(["second", "/exit"])
    stdout = io.StringIO()

    assert run_native_artifact_cli(
        NativeArtifactCLIConfig(
            artifact=artifact,
            prompt="first",
            max_new_tokens=128,
            seed=17,
        ),
        interactive=True,
        codec=CharacterCodec(),
        input_fn=lambda _prompt: next(commands),
        stdout=stdout,
        stderr=io.StringIO(),
    ) == 0

    rendered = stdout.getvalue()
    assert "Visible A" in rendered and "Visible B" in rendered
    assert "private reasoning" not in rendered
    assert "to=self" not in rendered and "to=user" not in rendered
    second_prompt = CharacterCodec().decode(model.session.prefills[1][0])
    assert "Visible A" in second_prompt
    assert "private reasoning" not in second_prompt


def test_native_cli_modules_are_lean_imports() -> None:
    script = (
        "import sys; import neuralfn.native_chat, neuralfn.native_cli; "
        "print(','.join(name for name in "
        "('torch','numpy','networkx','sqlalchemy','server.app') if name in sys.modules))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    assert completed.stdout.strip() == ""
