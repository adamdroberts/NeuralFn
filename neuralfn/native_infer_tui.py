"""Rich terminal presentation for the dependency-light native inference driver.

This module lives in the main ``neuralfn`` package so an editable CLI install
can discover it immediately.  Keeping the presentation behind a lazy import
preserves the dependency-light non-interactive/native paths.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

from rich.align import Align
from rich.box import HEAVY, ROUNDED
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text
from rich.theme import Theme


NATIVE_INFER_THEME = Theme(
    {
        "infer.user": "bold bright_cyan",
        "infer.assistant": "bold bright_magenta",
        "infer.system": "dim italic",
        "infer.banner": "bold white on #2a004d",
        "infer.accent": "bright_yellow",
        "infer.metric": "bright_green",
        "infer.error": "bold red",
    }
)


def make_native_infer_console(*, file: Any | None = None) -> Console:
    return Console(
        theme=NATIVE_INFER_THEME,
        emoji=True,
        highlight=False,
        soft_wrap=False,
        file=file,
    )


def _human_bytes(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "unknown"
    amount = float(value)
    for suffix in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(amount) < 1024.0 or suffix == "TiB":
            return f"{amount:.2f} {suffix}" if suffix != "B" else f"{int(amount)} B"
        amount /= 1024.0
    return f"{amount:.2f} TiB"


def _value(stats: Mapping[str, Any], *keys: str, default: str = "unknown") -> str:
    for key in keys:
        raw = stats.get(key)
        if raw not in (None, ""):
            return str(raw)
    return default


class RichNativeInferenceUI:
    """Colorful multi-turn terminal UI consumed by ``run_native_artifact_cli``."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or make_native_infer_console()
        self._ready: dict[str, Any] | None = None

    def read_line(self, prompt: str) -> str:
        del prompt
        return self.console.input("[infer.user]:bust_in_silhouette: You › [/]")

    @contextmanager
    def progress(self, label: str) -> Iterator[None]:
        spinner = Spinner(
            "dots",
            text=Text.from_markup(
                f":brain: [infer.assistant]{escape(label)}...[/]",
                emoji=True,
            ),
        )
        with Live(spinner, console=self.console, refresh_per_second=12, transient=True):
            yield

    def handle(self, event: str, **payload: Any) -> None:
        handler = getattr(self, f"_on_{event}", None)
        if handler is None:
            raise ValueError(f"Unsupported native inference UI event: {event}")
        handler(**payload)

    def _on_ready(self, **payload: Any) -> None:
        self._ready = dict(payload)
        self._render_banner()
        self.console.print(
            ":white_check_mark: [infer.system]Ready. This is a process-local "
            "multi-turn transcript; type [/][infer.accent]/help[/]"
            "[infer.system] for commands.[/]"
        )

    def _render_banner(self) -> None:
        if self._ready is None:
            return
        payload = self._ready
        stats = payload.get("stats")
        stats = stats if isinstance(stats, Mapping) else {}
        config = payload["config"]
        runtime = _value(stats, "runtime", "backend")
        precision = _value(
            stats,
            "effective_weight_precision",
            default=str(config.model_load.weight_precision),
        )
        cache = _value(stats, "effective_cache", default=str(config.kv_cache.mode))
        speculation = _value(
            stats,
            "effective_speculative_decoding",
            default=str(config.model_load.speculative_decoding),
        )
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="infer.accent", justify="right", no_wrap=True)
        grid.add_column(overflow="fold")
        grid.add_row(":brain: Model", escape(str(payload["model_name"])))
        grid.add_row(
            ":open_file_folder: Artifact",
            f"[dim]{escape(str(Path(payload['artifact'])))}[/dim]",
        )
        grid.add_row(":gear: Runtime", f"{escape(runtime)}  [dim]({escape(precision)})[/dim]")
        grid.add_row(":floppy_disk: Cache", escape(cache))
        grid.add_row(":zap: Speculation", escape(speculation))
        grid.add_row(":abc: Chat", escape(str(payload["renderer_name"])))
        grid.add_row(":straight_ruler: Context", f"{int(payload['context_limit']):,} tokens")
        grid.add_row(":compass: Mode", f"[infer.accent]{escape(str(payload['mode']))}[/]")
        grid.add_row(
            ":control_knobs: Sampling",
            f"top_k={config.top_k}  top_p={config.top_p:g}  "
            f"temp={config.temperature:g}  max_new={config.max_new_tokens}",
        )
        self.console.print(
            Panel(
                Align.center(grid),
                title="[infer.banner] :sparkles:  NeuralFn Native Chat  :sparkles: [/]",
                subtitle=(
                    "[infer.system]ATEM-safe output · /show runtime stats · /help commands[/]"
                ),
                box=HEAVY,
                border_style="bright_magenta",
                padding=(1, 2),
            )
        )

    def _on_turn(
        self,
        *,
        index: int,
        user_text: str,
        response: Any,
        result: Any,
        prefill_stats: Mapping[str, Any],
        prefill_seconds: float,
        decode_seconds: float,
    ) -> None:
        self.console.print(
            Panel(
                Text(str(user_text)),
                title=f":bust_in_silhouette: You  [dim]#{index}[/]",
                title_align="left",
                border_style="infer.user",
                box=ROUNDED,
                padding=(0, 1),
            )
        )
        visible = str(response.visible_text)
        if visible:
            body: Any = Markdown(visible, code_theme="monokai")
        else:
            body = Text(
                "No user-directed answer was completed. Private ATEM reasoning "
                "was hidden and will not be stored.",
                style="infer.error",
            )
        completion_tokens = int(getattr(result, "completion_tokens", 0) or 0)
        decode_rate = completion_tokens / decode_seconds if decode_seconds > 0 else 0.0
        reused = int(prefill_stats.get("prefix_reused", 0) or 0)
        prefilled = int(prefill_stats.get("prefilled_tokens", 0) or 0)
        proposed = int(getattr(result, "speculative_proposed_tokens", 0) or 0)
        accepted = int(getattr(result, "speculative_accepted_tokens", 0) or 0)
        speculative = ""
        if proposed:
            speculative = f" · DFlash {accepted}/{proposed} accepted"
        hidden = " · ATEM reasoning hidden" if response.reasoning_text else ""
        subtitle = (
            f"[infer.metric]{decode_rate:.1f} tok/s[/] · decode {decode_seconds:.2f}s · "
            f"prefill {prefilled} new/{reused} reused in {prefill_seconds:.2f}s"
            f"{speculative}{hidden}"
        )
        self.console.print(
            Panel(
                body,
                title=f":robot: Assistant  [dim]#{index}[/]",
                title_align="left",
                subtitle=subtitle,
                subtitle_align="right",
                border_style="infer.assistant",
                box=ROUNDED,
                padding=(0, 1),
            )
        )

    def _on_warning(self, *, message: str) -> None:
        self.console.print(
            Panel(
                Text(str(message)),
                title=":warning: Warning",
                title_align="left",
                border_style="infer.error",
                box=ROUNDED,
                padding=(0, 1),
            )
        )

    def _on_help(self) -> None:
        table = Table(title=":keyboard: Native chat commands", box=ROUNDED)
        table.add_column("Command", style="infer.accent", no_wrap=True)
        table.add_column("Effect")
        table.add_row("/mode transcript", "Retain user/assistant turns and reuse prefix cache")
        table.add_row("/mode stateless", "Keep system prompt but make turns independent")
        table.add_row("/show or /stats", "Show resident CUDA/cache/speculation state")
        table.add_row("/reset", "Clear transcript and reset resident cache state")
        table.add_row("/clear", "Clear and redraw the native chat banner")
        table.add_row("/exit or /quit", "Close the resident session and model")
        self.console.print(table)

    def _on_mode(self, *, mode: str) -> None:
        if self._ready is not None:
            self._ready["mode"] = mode
        self.console.print(
            f":compass: [infer.system]Switched to[/] [infer.accent]{escape(mode)}[/]"
            " [infer.system]mode.[/]"
        )

    def _on_show(
        self,
        *,
        mode: str,
        stats: Mapping[str, Any],
        history_messages: int,
        config: Any,
    ) -> None:
        table = Table(title=":bar_chart: Native resident state", box=ROUNDED)
        table.add_column("Field", style="infer.accent", no_wrap=True)
        table.add_column("Value", overflow="fold")
        rows = (
            ("mode", mode),
            ("history messages", history_messages),
            ("runtime", _value(stats, "runtime", "backend")),
            ("weight precision", _value(stats, "effective_weight_precision")),
            ("cache", _value(stats, "effective_cache")),
            ("speculation", _value(stats, "effective_speculative_decoding")),
            ("DFlash loaded", stats.get("dflash_loaded", False)),
            ("CUDA device", stats.get("cuda_device", config.model_load.cuda_device)),
            ("CUDA weights", _human_bytes(stats.get("cuda_resident_weight_bytes"))),
            ("DFlash weights", _human_bytes(stats.get("dflash_cuda_resident_weight_bytes"))),
            ("CUDA workspace", _human_bytes(stats.get("cuda_workspace_bytes"))),
            ("CUDA launches", stats.get("cuda_kernel_launches", "unknown")),
            (
                "Q8 activation quantizations",
                stats.get("cuda_q8_activation_quantizations", "unknown"),
            ),
            ("Q8 packed linears", stats.get("cuda_q8_packed_linears", "unknown")),
            ("Device argmax calls", stats.get("cuda_device_argmax_calls", "unknown")),
            ("Device argmax rows", stats.get("cuda_device_argmax_rows", "unknown")),
            ("CPU model rows", stats.get("cpu_model_compute_rows", "unknown")),
        )
        for key, value in rows:
            table.add_row(str(key), escape(str(value)))
        self.console.print(table)

    def _on_reset(self) -> None:
        self.console.print(
            ":broom: [infer.system]Transcript and resident session reset; the configured "
            "system prompt remains active.[/]"
        )

    def _on_clear(self) -> None:
        self.console.clear()
        self._render_banner()

    def _on_goodbye(self) -> None:
        self.console.print(":wave: [infer.system]Resident session closed. Bye.[/]")


__all__ = [
    "NATIVE_INFER_THEME",
    "RichNativeInferenceUI",
    "make_native_infer_console",
]
