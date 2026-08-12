from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


CLI_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CLI_ROOT.parent
SCRIPTS_ROOT = CLI_ROOT / "scripts"
for candidate in (CLI_ROOT, REPO_ROOT, SCRIPTS_ROOT):
    value = str(candidate)
    if value not in sys.path:
        sys.path.insert(0, value)

import nfn_impl
from neuralfn.config import SHIPPED_GPT_TEMPLATE_PRESETS


class CharacterTokenizer:
    def encode(self, text: str, out_type=int):
        return [ord(character) for character in text]

    def decode(self, token_ids):
        return "".join(chr(int(token_id)) for token_id in token_ids)


def _context(
    *,
    context_window: int = 128,
    chat_template: str = "plain_roles",
    graph_config: dict | None = None,
    tokenizer=None,
    generation_backend: str = "graph",
    chat_mode: str | None = "transcript",
    system_prompt: str | None = None,
):
    return nfn_impl.InferRuntimeContext(
        args=argparse.Namespace(
            chat_template=chat_template,
            chat_mode=chat_mode,
            system_prompt=system_prompt,
            max_new_tokens=16,
            seed=7,
        ),
        graph_path=Path("/tmp/model.json"),
        resolved_weights_path=Path("/tmp/model.pt"),
        graph=SimpleNamespace(
            nodes=(
                {"semantic_data_source": object()}
                if generation_backend == "semantic_graph"
                else {}
            ),
            torch_config=graph_config or {},
        ),
        compiled=object(),
        state_dict={},
        tokenizer=tokenizer or CharacterTokenizer(),
        tokenizer_path=None,
        tokenizer_name="fixture",
        raw_text_encoding_name="fixture",
        dataset_alias="fixture",
        device=SimpleNamespace(type="cpu"),
        generator=object(),
        amp_dtype=object(),
        amp_name="float32",
        context_window=context_window,
        generation_backend=(
            "graph" if generation_backend == "semantic_graph" else generation_backend
        ),
    )


LEGACY_TEXT_BACKENDS = ("graph", "semantic_graph", "parameter_golf")


def test_shipped_graph_text_preset_catalog_is_bound_to_covered_repl_driver() -> None:
    # Preset-specific model execution branches only after the common transcript
    # loop has rendered the prompt. Keep the complete authored catalog tied to
    # the ordinary/semantic graph paths exercised below.
    assert len(SHIPPED_GPT_TEMPLATE_PRESETS) == 66
    assert len(set(SHIPPED_GPT_TEMPLATE_PRESETS)) == 66


def test_interactive_default_is_transcript_and_stateless_flag_is_retained() -> None:
    parser = nfn_impl.build_command_parser("infer", style="long")
    defaults = parser.parse_args([])
    nfn_impl.ensure_infer_defaults(defaults, interactive=True)
    assert defaults.chat_mode == "transcript"
    assert defaults.chat_template == "auto"

    explicit = parser.parse_args(["--chat-mode", "stateless"])
    nfn_impl.ensure_infer_defaults(explicit, interactive=True)
    assert explicit.chat_mode == "stateless"


@pytest.mark.parametrize("generation_backend", LEGACY_TEXT_BACKENDS)
def test_interactive_session_retains_initial_turn_and_honors_mode_reset_and_stops(
    generation_backend: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(
        context_window=1024,
        graph_config={"role_delimiters": ["<STOP>"]},
        generation_backend=generation_backend,
        chat_mode=None,
        system_prompt="keep this",
    )
    if generation_backend == "semantic_graph":
        assert nfn_impl.infer_graph_uses_semantics(context)
    elif generation_backend == "graph":
        assert not nfn_impl.infer_graph_uses_semantics(context)
    else:
        assert context.generation_backend == "parameter_golf"
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
    generated = iter(
        [
            "A<STOP>ignored",
            "B",
            "C",
            "D",
            "E",
        ]
    )
    prompts: list[str] = []

    class FakeConsole:
        def print(self, *_args, **_kwargs) -> None:
            pass

        def clear(self) -> None:
            pass

    def fake_generation(
        _console,
        _context,
        *,
        prompt_ids,
        prompt_text,
        settings,
    ):
        del prompt_ids, settings
        prompts.append(prompt_text)
        response = next(generated)
        return {
            "generated_text": response,
            "generated_token_ids": [ord(response[0])],
        }, {}

    monkeypatch.setattr(nfn_impl, "make_infer_console", FakeConsole)
    monkeypatch.setattr(
        nfn_impl,
        "read_infer_chat_line",
        lambda *_args, **_kwargs: next(commands),
    )
    monkeypatch.setattr(
        nfn_impl,
        "run_infer_generation_with_spinner",
        fake_generation,
    )

    assert nfn_impl.run_infer_chat_session(
        context,
        initial_prompt_text="first",
    ) == 0

    assert prompts[0] == "System: keep this\nUser: first\nAssistant:"
    assert "User: first\nAssistant: A\nUser: second" in prompts[1]
    assert "ignored" not in prompts[1]
    assert prompts[2] == "System: keep this\nUser: third\nAssistant:"
    assert "User: first\nAssistant: A\nUser: second\nAssistant: B" in prompts[3]
    assert "third" not in prompts[3]
    assert prompts[4] == "System: keep this\nUser: after\nAssistant:"


def test_role_messages_cover_instructions_assistant_and_tool_items() -> None:
    rendered = nfn_impl.render_plain_roles_chat(
        [
            nfn_impl.InferChatMessage("developer", "follow policy"),
            nfn_impl.InferChatMessage("system", "be concise"),
            nfn_impl.InferChatMessage("user", "weather?"),
            nfn_impl.InferChatMessage("assistant", "calling tool"),
            nfn_impl.InferChatMessage("tool", "sunny", tool_call_id="call_1"),
        ],
        include_assistant_prompt=True,
    )
    assert rendered == (
        "Developer: follow policy\n"
        "System: be concise\n"
        "User: weather?\n"
        "Assistant: calling tool\n"
        "Tool: sunny\n"
        "Assistant:"
    )


@pytest.mark.parametrize("generation_backend", LEGACY_TEXT_BACKENDS)
def test_prompt_reserves_output_and_drops_oldest_complete_groups(
    generation_backend: str,
) -> None:
    context = _context(context_window=130, generation_backend=generation_backend)
    history = [
        nfn_impl.InferChatMessage("system", "keep this"),
        nfn_impl.InferChatMessage("user", "a" * 20),
        nfn_impl.InferChatMessage("assistant", "b" * 20),
        nfn_impl.InferChatMessage("user", "c" * 8),
        nfn_impl.InferChatMessage("assistant", "d" * 8),
        nfn_impl.InferChatMessage("tool", "tool-result", tool_call_id="call-1"),
    ]

    prompt, token_ids, dropped = nfn_impl.resolve_infer_chat_prompt(
        context,
        mode="transcript",
        history=history,
        draft="newest",
        include_assistant_prompt=True,
        reserved_output_tokens=20,
    )

    assert dropped >= 1
    assert "System: keep this" in prompt
    assert "User: newest" in prompt
    assert "Tool: tool-result" in prompt
    assert "a" * 20 not in prompt
    assert len(token_ids) <= 110


@pytest.mark.parametrize("generation_backend", LEGACY_TEXT_BACKENDS)
def test_prompt_fails_when_instructions_and_newest_turn_exceed_budget(
    generation_backend: str,
) -> None:
    context = _context(context_window=32, generation_backend=generation_backend)
    with pytest.raises(ValueError, match="leading instructions and newest user/tool turn"):
        nfn_impl.resolve_infer_chat_prompt(
            context,
            mode="transcript",
            history=[nfn_impl.InferChatMessage("system", "s" * 20)],
            draft="u" * 20,
            include_assistant_prompt=True,
            reserved_output_tokens=8,
        )


def test_explicit_template_path_is_data_only_and_supports_markers(tmp_path: Path) -> None:
    template = tmp_path / "chat.txt"
    template.write_text("BEGIN\n{{messages}}\nNEXT={{assistant_prompt}}\nEND", encoding="utf-8")
    context = _context(chat_template=str(template))

    rendered = nfn_impl.render_infer_chat_messages(
        context,
        [nfn_impl.InferChatMessage("user", "hello")],
        include_assistant_prompt=True,
    )
    assert rendered == "BEGIN\nUser: hello\nNEXT=Assistant:\nEND"


def test_auto_prefers_tokenizer_chat_template_and_passes_artifact_source() -> None:
    class TemplateTokenizer(CharacterTokenizer):
        def __init__(self) -> None:
            self.calls = []

        def apply_chat_template(self, conversation, **kwargs):
            self.calls.append((conversation, kwargs))
            return "TOKENIZER_RENDERED"

    tokenizer = TemplateTokenizer()
    context = _context(
        chat_template="auto",
        graph_config={"tokenizer_manifest": {"chat_template": "artifact-jinja"}},
        tokenizer=tokenizer,
    )
    rendered = nfn_impl.render_infer_chat_messages(
        context,
        [nfn_impl.InferChatMessage("system", "rules"), nfn_impl.InferChatMessage("user", "hi")],
        include_assistant_prompt=True,
    )
    assert rendered == "TOKENIZER_RENDERED"
    assert tokenizer.calls[0][1]["chat_template"] == "artifact-jinja"
    assert tokenizer.calls[0][1]["add_generation_prompt"] is True


def test_auto_reports_plain_roles_fallback_and_stop_delimiters_are_stripped() -> None:
    context = _context(chat_template="auto")
    kind, source, warning = nfn_impl.resolve_infer_chat_template(context)
    assert kind == "plain_roles"
    assert source is None
    assert "using plain_roles" in warning
    delimiters = nfn_impl.infer_text_stop_delimiters(context)
    assert nfn_impl.strip_infer_text_delimiters(
        "answer\nUser: injected next turn",
        delimiters,
    ) == "answer"


def test_noninteractive_default_remains_stateless() -> None:
    args = nfn_impl.build_command_parser("infer", style="long").parse_args([])
    nfn_impl.ensure_infer_defaults(args, interactive=False)
    assert args.chat_mode == "stateless"
