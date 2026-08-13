from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from neuralfn.native_glimmer_media import (
    NativeMuseGlimmerMediaError,
    prepare_images,
    prepare_videos,
    prepare_videos_for_model,
    sample_video_frame_indices,
    smart_resize,
)


def _image(width: int = 56, height: int = 28) -> Image.Image:
    image = Image.new("RGB", (width, height))
    image.putdata(
        [
            ((x * 17 + y * 3) % 256, (x * 5 + y * 19) % 256, (x + y * 11) % 256)
            for y in range(height)
            for x in range(width)
        ]
    )
    return image


def test_smart_resize_matches_pinned_grid_policy() -> None:
    assert smart_resize(28, 56) == (28, 56)
    assert smart_resize(1, 10_000)[0] >= 28
    height, width = smart_resize(4_000, 6_000)
    assert height % 28 == 0 and width % 28 == 0
    assert (height // 28) * (width // 28) <= 4_096
    with pytest.raises(NativeMuseGlimmerMediaError):
        smart_resize(0, 10)


def test_image_patch_layout_matches_temporal_channel_contract() -> None:
    image = _image()
    full = prepare_images([image], collapsed_temporal_weights=False)
    packed = prepare_images([image], collapsed_temporal_weights=True)
    assert full.grid_thw == ((1, 2, 4),)
    assert full.merged_token_counts == (2,)
    assert len(full.packed_patches) == 8
    assert full.patch_width == 1_176
    assert packed.patch_width == 588
    assert full.packed_patches[0][:588] == packed.packed_patches[0]
    assert full.packed_patches[0][588:] == packed.packed_patches[0]
    # channel-major within each patch: R[0,0], R[0,1], then G after 14*14.
    first = full.packed_patches[0]
    assert first[0] == pytest.approx(-1.0)
    assert first[1] == pytest.approx(17 * 2 / 255 - 1)
    assert first[14 * 14] == pytest.approx(-1.0)
    assert first[2 * 14 * 14] == pytest.approx(-1.0)
    assert full.prompt_fragments == (
        "<|image_start|><|patch|><|patch|><|image_end|>",
    )
    assert full.replacement_positions([1, 200_092, 2, 200_092, 3]) == (1, 3)


def test_data_url_decode_and_placeholder_mismatch_are_fail_closed() -> None:
    image = _image(28, 28)
    stream = BytesIO()
    image.save(stream, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()
    batch = prepare_images([url], collapsed_temporal_weights=True)
    assert batch.grid_thw == ((1, 2, 2),)
    assert batch.merged_token_counts == (1,)
    with pytest.raises(NativeMuseGlimmerMediaError, match="External image URLs"):
        prepare_images(["https://example.com/image.png"])
    with pytest.raises(NativeMuseGlimmerMediaError, match="placeholders"):
        batch.replacement_positions([200_092, 200_092])


def test_path_and_sequence_contract(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    _image(28, 28).save(path)
    assert prepare_images([path]).grid_thw == ((1, 2, 2),)
    with pytest.raises(NativeMuseGlimmerMediaError, match="sequence"):
        prepare_images(path)  # type: ignore[arg-type]


def test_video_patch_layout_prompt_and_odd_frame_padding_match_reference() -> None:
    frames = [
        Image.new("RGB", (28, 28), (255, 0, 0)),
        Image.new("RGB", (28, 28), (0, 255, 0)),
        Image.new("RGB", (28, 28), (0, 0, 255)),
    ]
    batch = prepare_videos(
        [frames],
        frame_timestamps=[(0.0, 0.5, 1.0)],
    )
    assert batch.modality == "video"
    assert batch.grid_thw == ((2, 2, 2),)
    assert batch.merged_token_counts == (2,)
    assert batch.video_group_timestamps == ((0.0, 1.0),)
    assert len(batch.packed_patches) == 8
    assert batch.patch_width == 1_176

    # Exact upstream flattening order is temporal, channel, patch_h, patch_w.
    first = batch.packed_patches[0]
    plane = 14 * 14
    assert first[0:plane] == pytest.approx((1.0,) * plane)
    assert first[plane : 3 * plane] == pytest.approx((-1.0,) * (2 * plane))
    assert first[3 * plane : 4 * plane] == pytest.approx((-1.0,) * plane)
    assert first[4 * plane : 5 * plane] == pytest.approx((1.0,) * plane)
    assert first[5 * plane : 6 * plane] == pytest.approx((-1.0,) * plane)
    # The padded fourth frame repeats the third frame exactly.
    assert batch.packed_patches[4][:588] == batch.packed_patches[4][588:]
    assert batch.prompt_fragments == (
        "<|vid_start|>Time: 0.0s<|video|><|vid_frame_separator|>"
        "Time: 1.0s<|video|><|vid_end|>",
    )
    assert batch.replacement_positions([3, 200_091, 4, 200_091]) == (1, 3)
    with pytest.raises(NativeMuseGlimmerMediaError, match="video placeholders"):
        batch.replacement_positions([200_092, 200_092])


def test_video_sampling_and_admission_are_fail_closed() -> None:
    indexes = sample_video_frame_indices(240, 24.0)
    assert len(indexes) == 20
    assert indexes[0] == 0 and indexes[-1] == 239
    assert indexes == tuple(sorted(indexes))
    assert sample_video_frame_indices(1, 24.0) == (0,)
    with pytest.raises(NativeMuseGlimmerMediaError, match="positive and finite"):
        sample_video_frame_indices(10, 0.0)

    frame = Image.new("RGB", (28, 28), (0, 0, 0))
    with pytest.raises(NativeMuseGlimmerMediaError, match="collapses temporal"):
        prepare_videos([[frame]], collapsed_temporal_weights=True)
    with pytest.raises(NativeMuseGlimmerMediaError, match="timestamp count"):
        prepare_videos([[frame, frame]], frame_timestamps=[(0.0,)])
    with pytest.raises(NativeMuseGlimmerMediaError, match="96-frame cap"):
        prepare_videos([[frame] * 97])


def test_video_model_selection_requires_full_temporal_payload() -> None:
    frame = Image.new("RGB", (28, 28), (0, 0, 0))

    class Model:
        def __init__(self, weight_bytes: int, video: bool = True) -> None:
            self.weight_bytes = weight_bytes
            self.video = video

        def stats(self) -> dict[str, object]:
            return {
                "vision_loaded": True,
                "video": self.video,
                "vision_resident_weight_bytes": self.weight_bytes,
            }

    batch = prepare_videos_for_model(Model(3_843_691_520), [[frame]])
    assert batch.grid_thw == ((1, 2, 2),)
    with pytest.raises(NativeMuseGlimmerMediaError, match="does not support video"):
        prepare_videos_for_model(Model(1_400_328_928), [[frame]])
    with pytest.raises(NativeMuseGlimmerMediaError, match="does not prove"):
        prepare_videos_for_model(Model(3_843_691_520, video=False), [[frame]])
