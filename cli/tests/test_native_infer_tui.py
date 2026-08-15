from __future__ import annotations

import io
from pathlib import Path

from rich.console import Console

from neuralfn.native_infer_tui import NATIVE_INFER_THEME, RichNativeInferenceUI
from neuralfn.native_chat import NativeAssistantResponse
from neuralfn.native_cli import NativeArtifactCLIConfig
from neuralfn.native_inference import GenerationResult, NativeModelLoadConfig


def _console(buffer: io.StringIO) -> Console:
    return Console(
        theme=NATIVE_INFER_THEME,
        file=buffer,
        force_terminal=True,
        color_system="truecolor",
        width=120,
        emoji=False,
        highlight=False,
    )


def test_rich_native_tui_renders_banner_turn_metrics_and_hidden_reasoning() -> None:
    output = io.StringIO()
    ui = RichNativeInferenceUI(_console(output))
    config = NativeArtifactCLIConfig(
        artifact=Path("/models/glimmer"),
        max_new_tokens=512,
        model_load=NativeModelLoadConfig(
            runtime="native-cuda",
            weight_precision="k-quant-17gb",
            speculative_decoding="auto",
            companion_checkpoints=("dflash",),
        ),
    )
    ui.handle(
        "ready",
        model_name="Muse Glimmer 30B",
        artifact=config.artifact,
        mode="transcript",
        stats={
            "backend": "native-cuda",
            "effective_weight_precision": "k-quant-17gb",
            "effective_cache": "full",
            "effective_speculative_decoding": "dflash",
        },
        context_limit=131_072,
        renderer_name="muse_glimmer_atem_v1",
        config=config,
    )
    ui.handle(
        "turn",
        index=1,
        user_text="How are you?",
        response=NativeAssistantResponse(
            visible_text="I am doing well.",
            reasoning_text="private",
            used_channel_protocol=True,
            final_channel_complete=True,
        ),
        result=GenerationResult(
            token_ids=(1, 2, 3),
            text="",
            finish_reason="stop",
            prompt_tokens=20,
            completion_tokens=3,
            speculative_proposed_tokens=15,
            speculative_accepted_tokens=12,
        ),
        prefill_stats={"prefilled_tokens": 20, "prefix_reused": 10},
        prefill_seconds=0.25,
        decode_seconds=0.1,
    )

    rendered = output.getvalue()
    assert "NeuralFn Native Chat" in rendered
    assert "multi-turn transcript" in rendered
    assert "Muse Glimmer 30B" in rendered
    assert "How are you?" in rendered
    assert "I am doing well." in rendered
    assert "30.0 tok/s" in rendered
    assert "DFlash 12/15 accepted" in rendered
    assert "ATEM reasoning hidden" in rendered
    assert "private" not in rendered
    assert "\x1b[" in rendered


def test_rich_native_tui_explains_missing_final_without_showing_raw_text() -> None:
    output = io.StringIO()
    ui = RichNativeInferenceUI(_console(output))
    ui.handle(
        "turn",
        index=2,
        user_text="hello",
        response=NativeAssistantResponse(
            visible_text="",
            reasoning_text="do not display",
            raw_text=" to=self<|message|>do not display",
            used_channel_protocol=True,
            final_channel_complete=False,
        ),
        result=GenerationResult(
            token_ids=(),
            text="",
            finish_reason="length",
            prompt_tokens=10,
            completion_tokens=0,
        ),
        prefill_stats={},
        prefill_seconds=0.01,
        decode_seconds=0.02,
    )

    rendered = output.getvalue()
    assert "No user-directed answer was completed" in rendered
    assert "do not display" not in rendered
    assert "to=self" not in rendered


def test_rich_native_tui_show_reports_device_argmax_telemetry() -> None:
    output = io.StringIO()
    ui = RichNativeInferenceUI(_console(output))
    config = NativeArtifactCLIConfig(
        artifact=Path("/models/glimmer"),
        model_load=NativeModelLoadConfig(runtime="native-cuda"),
    )

    ui.handle(
        "show",
        mode="transcript",
        stats={
            "cuda_device_argmax_calls": 3,
            "cuda_device_argmax_rows": 31,
            "cpu_model_compute_rows": 0,
        },
        history_messages=5,
        config=config,
    )

    rendered = output.getvalue()
    assert "Device argmax calls" in rendered
    assert "Device argmax rows" in rendered
    assert "31" in rendered
