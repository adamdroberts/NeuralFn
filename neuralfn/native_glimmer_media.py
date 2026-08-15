"""Torch-free, pinned Muse Glimmer image and decoded-video preprocessing.

The implementation mirrors the official processor revision bound by
``MAIN_PROCESSOR_CONFIG_SHA256``: aspect-ratio preserving LANCZOS resize,
0.5/0.5 RGB normalization, temporal-2 patch packing, and 2x2 merged
placeholder accounting.  External HTTP fetching and container decoding are
intentionally absent; image callers may provide bytes, paths, data URLs, or
Pillow objects, while video callers provide already-decoded frame sequences.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from io import BytesIO
import itertools
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .native_chat import MUSE_GLIMMER_SPECIAL_TOKEN_IDS


PATCH_SIZE = 14
TEMPORAL_PATCH_SIZE = 2
MERGE_SIZE = 2
MAX_IMAGE_TOKENS = 4_096
MAX_VIDEO_FRAME_TOKENS = 144
DEFAULT_VIDEO_NUM_FRAMES = 96
DEFAULT_VIDEO_FPS = 2.0
IMAGE_MEAN = (0.5, 0.5, 0.5)
IMAGE_STD = (0.5, 0.5, 0.5)
MAX_ENCODED_IMAGE_BYTES = 32 * 1024 * 1024
MAX_SOURCE_PIXELS = 100_000_000


class NativeMuseGlimmerMediaError(ValueError):
    """Media cannot satisfy the pinned native processor contract."""


@dataclass(frozen=True, slots=True)
class NativeMuseGlimmerMediaBatch:
    packed_patches: tuple[tuple[float, ...], ...]
    grid_thw: tuple[tuple[int, int, int], ...]
    merged_token_counts: tuple[int, ...]
    collapsed_temporal_weights: bool
    modality: str = "image"
    video_group_timestamps: tuple[tuple[float, ...], ...] = ()

    @property
    def patch_width(self) -> int:
        return 3 * PATCH_SIZE * PATCH_SIZE * (
            1 if self.collapsed_temporal_weights else TEMPORAL_PATCH_SIZE
        )

    @property
    def prompt_fragments(self) -> tuple[str, ...]:
        if self.modality == "image":
            return tuple(
                "<|image_start|>" + "<|patch|>" * count + "<|image_end|>"
                for count in self.merged_token_counts
            )
        if self.modality != "video":
            raise NativeMuseGlimmerMediaError(
                f"Unsupported Muse Glimmer media modality {self.modality!r}"
            )
        if len(self.video_group_timestamps) != len(self.grid_thw):
            raise NativeMuseGlimmerMediaError(
                "Video timestamp groups do not match the packed video count"
            )
        fragments: list[str] = []
        for grid, timestamps in zip(self.grid_thw, self.video_group_timestamps):
            grid_t, grid_h, grid_w = grid
            if len(timestamps) != grid_t:
                raise NativeMuseGlimmerMediaError(
                    "Video timestamp count does not match temporal patch groups"
                )
            tokens_per_group = grid_h * grid_w // (MERGE_SIZE * MERGE_SIZE)
            fragment = "<|vid_start|>"
            for group, timestamp in enumerate(timestamps):
                fragment += f"Time: {timestamp:.1f}s"
                fragment += "<|video|>" * tokens_per_group
                fragment += (
                    "<|vid_frame_separator|>"
                    if group < grid_t - 1
                    else "<|vid_end|>"
                )
            fragments.append(fragment)
        return tuple(fragments)

    def replacement_positions(self, prompt_token_ids: Sequence[int]) -> tuple[int, ...]:
        placeholder = "patch" if self.modality == "image" else "video"
        positions = tuple(
            index
            for index, token in enumerate(prompt_token_ids)
            if int(token) == MUSE_GLIMMER_SPECIAL_TOKEN_IDS[placeholder]
        )
        expected = sum(self.merged_token_counts)
        if len(positions) != expected:
            raise NativeMuseGlimmerMediaError(
                f"Prompt contains {len(positions)} {placeholder} placeholders; expected {expected}"
            )
        return positions


@dataclass(frozen=True, slots=True)
class NativeMuseGlimmerEncodedMedia:
    batch: NativeMuseGlimmerMediaBatch
    embeddings: tuple[tuple[float, ...], ...]

    def replacement_positions(self, prompt_token_ids: Sequence[int]) -> tuple[int, ...]:
        positions = self.batch.replacement_positions(prompt_token_ids)
        if len(positions) != len(self.embeddings):
            raise NativeMuseGlimmerMediaError(
                "Vision output rows do not match the prompt placeholder count"
            )
        return positions


def smart_resize(
    height: int,
    width: int,
    *,
    patch_size: int = PATCH_SIZE * MERGE_SIZE,
    max_tokens: int = MAX_IMAGE_TOKENS,
) -> tuple[int, int]:
    """Exact integer-grid selection used by the pinned official processor."""

    if (
        isinstance(height, bool)
        or isinstance(width, bool)
        or isinstance(patch_size, bool)
        or isinstance(max_tokens, bool)
        or not all(isinstance(value, int) for value in (height, width, patch_size, max_tokens))
        or min(height, width, patch_size, max_tokens) <= 0
    ):
        raise NativeMuseGlimmerMediaError(
            "height, width, patch_size, and max_tokens must be positive integers"
        )
    ideal_height = height / patch_size
    ideal_width = width / patch_size
    ratio = ideal_width / ideal_height if ideal_height > 0 else 1.0
    if ideal_height * ideal_width > max_tokens:
        ideal_height = math.sqrt(max_tokens / ratio)
        ideal_width = ideal_height * ratio
    candidates = set(
        itertools.product(
            (math.floor(ideal_height), math.ceil(ideal_height)),
            (math.floor(ideal_width), math.ceil(ideal_width)),
        )
    )
    valid = [
        (grid_h, grid_w)
        for grid_h, grid_w in candidates
        if grid_h >= 1 and grid_w >= 1 and grid_h * grid_w <= max_tokens
    ]
    if not valid:
        valid = [(max(1, round(ideal_height)), max(1, round(ideal_width)))]
    grid_h, grid_w = min(
        valid,
        key=lambda grid: abs(grid[0] / grid[1] - height / width),
    )
    return grid_h * patch_size, grid_w * patch_size


def _pillow() -> tuple[Any, Any]:
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:  # pragma: no cover - lean-install error path
        raise NativeMuseGlimmerMediaError(
            "Raw Muse Glimmer images require Pillow; install `neuralfn[vision]` "
            "or `neuralfn[serve]`."
        ) from exc
    return Image, ImageOps


def _decode_data_url(value: str) -> bytes:
    header, separator, payload = value.partition(",")
    if not separator or not header.startswith("data:image/") or ";base64" not in header:
        raise NativeMuseGlimmerMediaError(
            "Only base64 data:image/* URLs are accepted; external URL fetching is disabled"
        )
    try:
        decoded = base64.b64decode(payload, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise NativeMuseGlimmerMediaError("Image data URL has invalid base64") from exc
    if not decoded or len(decoded) > MAX_ENCODED_IMAGE_BYTES:
        raise NativeMuseGlimmerMediaError("Encoded image exceeds the 32 MiB limit")
    return decoded


def _load_image(source: Any) -> Any:
    Image, ImageOps = _pillow()
    if isinstance(source, Image.Image):
        image = source.copy()
    elif isinstance(source, (bytes, bytearray, memoryview)):
        payload = bytes(source)
        if not payload or len(payload) > MAX_ENCODED_IMAGE_BYTES:
            raise NativeMuseGlimmerMediaError("Encoded image exceeds the 32 MiB limit")
        image = Image.open(BytesIO(payload))
    elif isinstance(source, Path):
        path = source.expanduser().resolve()
        if not path.is_file() or path.stat().st_size > MAX_ENCODED_IMAGE_BYTES:
            raise NativeMuseGlimmerMediaError("Image path is missing or exceeds 32 MiB")
        image = Image.open(path)
    elif isinstance(source, str):
        if source.startswith("data:"):
            image = Image.open(BytesIO(_decode_data_url(source)))
        elif "://" in source:
            raise NativeMuseGlimmerMediaError(
                "External image URLs are disabled; use a data URL or decoded bytes"
            )
        else:
            return _load_image(Path(source))
    elif isinstance(source, Mapping) and isinstance(source.get("image_url"), str):
        return _load_image(source["image_url"])
    else:
        raise NativeMuseGlimmerMediaError(
            "Image must be Pillow.Image, bytes, a local path, or a base64 data URL"
        )
    try:
        width, height = image.size
        if width <= 0 or height <= 0 or width * height > MAX_SOURCE_PIXELS:
            raise NativeMuseGlimmerMediaError("Image dimensions exceed the native admission limit")
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.load()
        return image
    except NativeMuseGlimmerMediaError:
        image.close()
        raise
    except Exception as exc:
        image.close()
        raise NativeMuseGlimmerMediaError("Image decoding failed") from exc


def prepare_images(
    images: Sequence[Any],
    *,
    collapsed_temporal_weights: bool = False,
    max_image_tokens: int = MAX_IMAGE_TOKENS,
) -> NativeMuseGlimmerMediaBatch:
    """Decode, resize, normalize, and patchify one or more still images."""

    if isinstance(images, (str, bytes, bytearray, memoryview, Path)):
        raise NativeMuseGlimmerMediaError("images must be a nonempty sequence")
    sources = tuple(images)
    if not sources:
        raise NativeMuseGlimmerMediaError("images must be a nonempty sequence")
    if not isinstance(collapsed_temporal_weights, bool):
        raise NativeMuseGlimmerMediaError("collapsed_temporal_weights must be boolean")
    Image, _ImageOps = _pillow()
    patches: list[tuple[float, ...]] = []
    grids: list[tuple[int, int, int]] = []
    merged_counts: list[int] = []
    for source in sources:
        image = _load_image(source)
        try:
            target_h, target_w = smart_resize(
                image.height,
                image.width,
                max_tokens=max_image_tokens,
            )
            resized = image.resize((target_w, target_h), resample=Image.Resampling.LANCZOS)
            pixels = resized.tobytes()
        finally:
            image.close()
        grid_h = target_h // PATCH_SIZE
        grid_w = target_w // PATCH_SIZE
        grids.append((1, grid_h, grid_w))
        if grid_h % MERGE_SIZE or grid_w % MERGE_SIZE:
            raise NativeMuseGlimmerMediaError("Resized patch grid is not 2x2 mergeable")
        merged_counts.append(grid_h * grid_w // (MERGE_SIZE * MERGE_SIZE))
        for patch_h in range(grid_h):
            for patch_w in range(grid_w):
                spatial: list[float] = []
                for channel in range(3):
                    for local_h in range(PATCH_SIZE):
                        pixel_h = patch_h * PATCH_SIZE + local_h
                        row_offset = pixel_h * target_w * 3
                        for local_w in range(PATCH_SIZE):
                            pixel_w = patch_w * PATCH_SIZE + local_w
                            value = pixels[row_offset + pixel_w * 3 + channel]
                            spatial.append(value * (2.0 / 255.0) - 1.0)
                patches.append(
                    tuple(spatial)
                    if collapsed_temporal_weights
                    else tuple(spatial + spatial)
                )
    return NativeMuseGlimmerMediaBatch(
        packed_patches=tuple(patches),
        grid_thw=tuple(grids),
        merged_token_counts=tuple(merged_counts),
        collapsed_temporal_weights=collapsed_temporal_weights,
        modality="image",
    )


def sample_video_frame_indices(
    total_num_frames: int,
    source_fps: float,
    *,
    num_frames: int = DEFAULT_VIDEO_NUM_FRAMES,
    fps: float = DEFAULT_VIDEO_FPS,
) -> tuple[int, ...]:
    """Mirror the pinned processor's uniform frame-index selection.

    Container decoding remains the caller's responsibility.  The returned
    indexes can be applied to decoded frames and their source timestamps before
    calling :func:`prepare_videos`.
    """

    if (
        isinstance(total_num_frames, bool)
        or not isinstance(total_num_frames, int)
        or total_num_frames <= 0
        or isinstance(num_frames, bool)
        or not isinstance(num_frames, int)
        or num_frames <= 0
    ):
        raise NativeMuseGlimmerMediaError(
            "total_num_frames and num_frames must be positive integers"
        )
    try:
        source_fps_value = float(source_fps)
        fps_value = float(fps)
    except (TypeError, ValueError) as exc:
        raise NativeMuseGlimmerMediaError("source_fps and fps must be finite") from exc
    if (
        not math.isfinite(source_fps_value)
        or not math.isfinite(fps_value)
        or source_fps_value <= 0.0
        or fps_value <= 0.0
    ):
        raise NativeMuseGlimmerMediaError("source_fps and fps must be positive and finite")
    selected = min(
        int(total_num_frames * fps_value / source_fps_value),
        num_frames,
        total_num_frames,
    )
    selected = max(TEMPORAL_PATCH_SIZE, (selected // TEMPORAL_PATCH_SIZE) * TEMPORAL_PATCH_SIZE)
    selected = min(selected, total_num_frames)
    if selected == 1:
        return (0,)
    # torch.linspace(0, total_num_frames - 1, selected).long() for the
    # non-negative integer range used by the pinned reference.
    return tuple(
        int(index * (total_num_frames - 1) / (selected - 1))
        for index in range(selected)
    )


def _validate_video_timestamps(
    timestamps: Sequence[float] | None,
    *,
    frame_count: int,
    sampled_fps: float,
) -> tuple[float, ...]:
    if timestamps is None:
        try:
            sampled_fps_value = float(sampled_fps)
        except (TypeError, ValueError) as exc:
            raise NativeMuseGlimmerMediaError("sampled_fps must be finite") from exc
        if not math.isfinite(sampled_fps_value) or sampled_fps_value <= 0.0:
            raise NativeMuseGlimmerMediaError("sampled_fps must be positive and finite")
        return tuple(index / sampled_fps_value for index in range(frame_count))
    if isinstance(timestamps, (str, bytes, bytearray)):
        raise NativeMuseGlimmerMediaError("video timestamps must be a numeric sequence")
    values: list[float] = []
    try:
        for value in timestamps:
            converted = float(value)
            if not math.isfinite(converted) or converted < 0.0:
                raise NativeMuseGlimmerMediaError(
                    "video timestamps must be finite and non-negative"
                )
            values.append(converted)
    except (TypeError, ValueError) as exc:
        raise NativeMuseGlimmerMediaError(
            "video timestamps must be a numeric sequence"
        ) from exc
    if len(values) != frame_count:
        raise NativeMuseGlimmerMediaError(
            "video timestamp count must equal the decoded frame count"
        )
    if any(right < left for left, right in zip(values, values[1:])):
        raise NativeMuseGlimmerMediaError("video timestamps must be nondecreasing")
    return tuple(values)


def prepare_videos(
    videos: Sequence[Sequence[Any]],
    *,
    frame_timestamps: Sequence[Sequence[float]] | None = None,
    sampled_fps: float = DEFAULT_VIDEO_FPS,
    collapsed_temporal_weights: bool = False,
    max_video_frame_tokens: int = MAX_VIDEO_FRAME_TOKENS,
) -> NativeMuseGlimmerMediaBatch:
    """Resize, normalize, and temporal-patch decoded video frame sequences.

    ``videos`` contains frames after the caller applies the pinned sampling
    indexes.  When exact source timestamps are available, pass one timestamp
    sequence per video.  Otherwise the pinned 2 FPS default is used.  The
    released quantized ``mmproj`` collapses the two temporal patch weights and
    therefore cannot faithfully encode distinct video frames.
    """

    if isinstance(videos, (str, bytes, bytearray, memoryview, Path)):
        raise NativeMuseGlimmerMediaError("videos must be a nonempty sequence of frame sequences")
    video_sources = tuple(videos)
    if not video_sources:
        raise NativeMuseGlimmerMediaError("videos must be a nonempty sequence of frame sequences")
    if not isinstance(collapsed_temporal_weights, bool):
        raise NativeMuseGlimmerMediaError("collapsed_temporal_weights must be boolean")
    if collapsed_temporal_weights:
        raise NativeMuseGlimmerMediaError(
            "The quantized mmproj collapses temporal patch weights and does not support video"
        )
    if (
        isinstance(max_video_frame_tokens, bool)
        or not isinstance(max_video_frame_tokens, int)
        or max_video_frame_tokens <= 0
    ):
        raise NativeMuseGlimmerMediaError(
            "max_video_frame_tokens must be a positive integer"
        )
    timestamp_sources: tuple[Sequence[float] | None, ...]
    if frame_timestamps is None:
        timestamp_sources = (None,) * len(video_sources)
    else:
        if isinstance(frame_timestamps, (str, bytes, bytearray)):
            raise NativeMuseGlimmerMediaError(
                "frame_timestamps must contain one sequence per video"
            )
        timestamp_sources = tuple(frame_timestamps)
        if len(timestamp_sources) != len(video_sources):
            raise NativeMuseGlimmerMediaError(
                "frame_timestamps must contain one sequence per video"
            )

    Image, _ImageOps = _pillow()
    patches: list[tuple[float, ...]] = []
    grids: list[tuple[int, int, int]] = []
    merged_counts: list[int] = []
    group_timestamps: list[tuple[float, ...]] = []
    for video_index, (source_frames, source_timestamps) in enumerate(
        zip(video_sources, timestamp_sources)
    ):
        if isinstance(source_frames, (str, bytes, bytearray, memoryview, Path)):
            raise NativeMuseGlimmerMediaError(
                f"videos[{video_index}] must be a nonempty decoded-frame sequence"
            )
        frame_sources = tuple(source_frames)
        if not frame_sources:
            raise NativeMuseGlimmerMediaError(
                f"videos[{video_index}] must be a nonempty decoded-frame sequence"
            )
        if len(frame_sources) > DEFAULT_VIDEO_NUM_FRAMES:
            raise NativeMuseGlimmerMediaError(
                f"videos[{video_index}] exceeds the pinned {DEFAULT_VIDEO_NUM_FRAMES}-frame cap; "
                "apply sample_video_frame_indices first"
            )
        timestamps = _validate_video_timestamps(
            source_timestamps,
            frame_count=len(frame_sources),
            sampled_fps=sampled_fps,
        )
        decoded: list[Any] = []
        try:
            for frame in frame_sources:
                decoded.append(_load_image(frame))
            source_size = decoded[0].size
            if any(frame.size != source_size for frame in decoded[1:]):
                raise NativeMuseGlimmerMediaError(
                    f"videos[{video_index}] frames must have identical dimensions"
                )
            target_h, target_w = smart_resize(
                decoded[0].height,
                decoded[0].width,
                max_tokens=max_video_frame_tokens,
            )
            resized_pixels: list[bytes] = []
            for frame in decoded:
                resized = frame.resize(
                    (target_w, target_h),
                    resample=Image.Resampling.LANCZOS,
                )
                try:
                    resized_pixels.append(resized.tobytes())
                finally:
                    resized.close()
        finally:
            for frame in decoded:
                frame.close()

        if len(resized_pixels) % TEMPORAL_PATCH_SIZE:
            resized_pixels.append(resized_pixels[-1])
        grid_t = len(resized_pixels) // TEMPORAL_PATCH_SIZE
        grid_h = target_h // PATCH_SIZE
        grid_w = target_w // PATCH_SIZE
        if grid_h % MERGE_SIZE or grid_w % MERGE_SIZE:
            raise NativeMuseGlimmerMediaError("Resized patch grid is not 2x2 mergeable")
        grids.append((grid_t, grid_h, grid_w))
        tokens_per_group = grid_h * grid_w // (MERGE_SIZE * MERGE_SIZE)
        merged_counts.append(grid_t * tokens_per_group)
        # The official prompt uses the timestamp of the first source frame in
        # each temporal pair and pads only the pixel tensor, not timestamps.
        temporal_timestamps = list(timestamps[::TEMPORAL_PATCH_SIZE])[:grid_t]
        while len(temporal_timestamps) < grid_t:
            temporal_timestamps.append(temporal_timestamps[-1] if temporal_timestamps else 0.0)
        group_timestamps.append(tuple(temporal_timestamps))

        for temporal_group in range(grid_t):
            group_frames = resized_pixels[
                temporal_group * TEMPORAL_PATCH_SIZE :
                (temporal_group + 1) * TEMPORAL_PATCH_SIZE
            ]
            for patch_h in range(grid_h):
                for patch_w in range(grid_w):
                    packed: list[float] = []
                    # Exact reference layout: temporal, channel, patch_h,
                    # patch_w (not channel, temporal).
                    for frame_pixels in group_frames:
                        for channel in range(3):
                            for local_h in range(PATCH_SIZE):
                                pixel_h = patch_h * PATCH_SIZE + local_h
                                row_offset = pixel_h * target_w * 3
                                for local_w in range(PATCH_SIZE):
                                    pixel_w = patch_w * PATCH_SIZE + local_w
                                    value = frame_pixels[row_offset + pixel_w * 3 + channel]
                                    packed.append(value * (2.0 / 255.0) - 1.0)
                    patches.append(tuple(packed))

    return NativeMuseGlimmerMediaBatch(
        packed_patches=tuple(patches),
        grid_thw=tuple(grids),
        merged_token_counts=tuple(merged_counts),
        collapsed_temporal_weights=False,
        modality="video",
        video_group_timestamps=tuple(group_timestamps),
    )


def prepare_and_encode_images(
    model: Any,
    images: Sequence[Any],
    *,
    max_image_tokens: int = MAX_IMAGE_TOKENS,
) -> NativeMuseGlimmerEncodedMedia:
    """Select the loaded BF16/mmproj patch ABI and run the resident encoder."""

    batch = prepare_images_for_model(
        model,
        images,
        max_image_tokens=max_image_tokens,
    )
    embeddings = model.encode_media(batch.packed_patches, batch.grid_thw)
    if len(embeddings) != sum(batch.merged_token_counts):
        raise NativeMuseGlimmerMediaError(
            "Resident vision output does not match the placeholder geometry"
        )
    return NativeMuseGlimmerEncodedMedia(batch=batch, embeddings=embeddings)


def prepare_and_encode_videos(
    model: Any,
    videos: Sequence[Sequence[Any]],
    *,
    frame_timestamps: Sequence[Sequence[float]] | None = None,
    sampled_fps: float = DEFAULT_VIDEO_FPS,
    max_video_frame_tokens: int = MAX_VIDEO_FRAME_TOKENS,
) -> NativeMuseGlimmerEncodedMedia:
    """Run pinned decoded-video preprocessing and the resident vision tower."""

    batch = prepare_videos_for_model(
        model,
        videos,
        frame_timestamps=frame_timestamps,
        sampled_fps=sampled_fps,
        max_video_frame_tokens=max_video_frame_tokens,
    )
    embeddings = model.encode_media(batch.packed_patches, batch.grid_thw)
    if len(embeddings) != sum(batch.merged_token_counts):
        raise NativeMuseGlimmerMediaError(
            "Resident vision output does not match the video placeholder geometry"
        )
    return NativeMuseGlimmerEncodedMedia(batch=batch, embeddings=embeddings)


def prepare_images_for_model(
    model: Any,
    images: Sequence[Any],
    *,
    max_image_tokens: int = MAX_IMAGE_TOKENS,
) -> NativeMuseGlimmerMediaBatch:
    """Select the loaded BF16/mmproj patch ABI without running model compute."""

    stats = model.stats()
    if not isinstance(stats, Mapping) or stats.get("vision_loaded") is not True:
        raise NativeMuseGlimmerMediaError(
            "The resident model has no loaded Muse Glimmer vision weights"
        )
    vision_bytes = stats.get("vision_resident_weight_bytes")
    if vision_bytes == 1_400_328_928:
        collapsed = True
    elif vision_bytes == 3_843_691_520:
        collapsed = False
    else:
        raise NativeMuseGlimmerMediaError(
            "The loaded vision payload does not expose a canonical patch ABI"
        )
    return prepare_images(
        images,
        collapsed_temporal_weights=collapsed,
        max_image_tokens=max_image_tokens,
    )


def prepare_videos_for_model(
    model: Any,
    videos: Sequence[Sequence[Any]],
    *,
    frame_timestamps: Sequence[Sequence[float]] | None = None,
    sampled_fps: float = DEFAULT_VIDEO_FPS,
    max_video_frame_tokens: int = MAX_VIDEO_FRAME_TOKENS,
) -> NativeMuseGlimmerMediaBatch:
    """Select the full-BF16 temporal patch ABI without running model compute."""

    stats = model.stats()
    if (
        not isinstance(stats, Mapping)
        or stats.get("vision_loaded") is not True
        or stats.get("video") is not True
    ):
        raise NativeMuseGlimmerMediaError(
            "The resident model does not prove Muse Glimmer video support"
        )
    vision_bytes = stats.get("vision_resident_weight_bytes")
    if vision_bytes == 1_400_328_928:
        raise NativeMuseGlimmerMediaError(
            "The quantized mmproj collapses temporal patch weights and does not support video"
        )
    if vision_bytes != 3_843_691_520:
        raise NativeMuseGlimmerMediaError(
            "The loaded vision payload does not expose the full temporal patch ABI"
        )
    return prepare_videos(
        videos,
        frame_timestamps=frame_timestamps,
        sampled_fps=sampled_fps,
        collapsed_temporal_weights=False,
        max_video_frame_tokens=max_video_frame_tokens,
    )


__all__ = [
    "DEFAULT_VIDEO_FPS",
    "DEFAULT_VIDEO_NUM_FRAMES",
    "MAX_IMAGE_TOKENS",
    "MAX_VIDEO_FRAME_TOKENS",
    "NativeMuseGlimmerEncodedMedia",
    "NativeMuseGlimmerMediaBatch",
    "NativeMuseGlimmerMediaError",
    "prepare_and_encode_images",
    "prepare_and_encode_videos",
    "prepare_images",
    "prepare_images_for_model",
    "prepare_videos",
    "prepare_videos_for_model",
    "sample_video_frame_indices",
    "smart_resize",
]
