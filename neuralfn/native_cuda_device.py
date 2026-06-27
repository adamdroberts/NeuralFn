from __future__ import annotations

import subprocess


_AUTO_CUDA_VISIBLE_DEVICE_VALUES = {"auto", "dedicated", "dedicated-auto"}


def resolve_cuda_visible_devices_value(requested: str | None) -> str:
    """Resolve NeuralFn CUDA-visible-device aliases to CUDA runtime values."""

    value = str(requested or "").strip()
    normalized = value.lower()
    if normalized in {"", "none", "off"}:
        return ""
    if normalized not in _AUTO_CUDA_VISIBLE_DEVICE_VALUES:
        return value
    return _select_display_disabled_cuda_device()


def _select_display_disabled_cuda_device() -> str:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,display_active,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "0"

    first_index = ""
    best_index = ""
    best_util: int | None = None
    for raw_line in proc.stdout.splitlines():
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) < 3 or not parts[0]:
            continue
        index, display, util_text = parts[:3]
        if not first_index:
            first_index = index
        try:
            util = int(util_text)
        except ValueError:
            util = 0
        if display == "Disabled" and (best_util is None or util < best_util):
            best_index = index
            best_util = util
    if best_index:
        return best_index
    return first_index or "0"
