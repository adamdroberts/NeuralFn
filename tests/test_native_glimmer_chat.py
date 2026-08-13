from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from neuralfn.native_chat import (
    MUSE_GLIMMER_ATEM_PROFILE,
    MUSE_GLIMMER_ATEM_TEMPLATE_SHA256,
    MUSE_GLIMMER_SPECIAL_TOKEN_IDS,
    MUSE_GLIMMER_TOKENIZER_SHA256,
    MUSE_GLIMMER_VOCAB_SIZE,
    MuseGlimmerATEMRenderer,
    NativeChatConfigurationError,
    NativeChatMessage,
    load_native_text_codec,
    native_stop_token_ids,
    resolve_native_chat_renderer,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "tests" / "fixtures" / "muse_glimmer" / "reference.json"


def _reference() -> dict[str, object]:
    return json.loads(REFERENCE.read_text(encoding="utf-8"))


def _muse_manifest(*, template: str = "fixture-template") -> dict[str, object]:
    template_sha = hashlib.sha256(template.encode("utf-8")).hexdigest()
    return {
        "model": {"family": "muse_glimmer"},
        "chat_template": {
            "format": MUSE_GLIMMER_ATEM_PROFILE,
            "template": template,
            "sha256": template_sha,
            "defaults": {"current_date": "2026-08-12"},
        },
        "stop_tokens": [200_001, 200_008],
    }


def test_pinned_tokenizer_and_atem_contract_matches_reference_fixture() -> None:
    contract = _reference()["tokenizer_contract"]
    assert isinstance(contract, dict)
    assert contract["tokenizer_json_sha256"] == MUSE_GLIMMER_TOKENIZER_SHA256
    assert contract["atem_template_sha256"] == MUSE_GLIMMER_ATEM_TEMPLATE_SHA256
    assert contract["vocab_size"] == MUSE_GLIMMER_VOCAB_SIZE
    assert contract["special_token_ids"] == MUSE_GLIMMER_SPECIAL_TOKEN_IDS
    assert contract["stop_token_ids"] == [200_001, 200_008]


def test_reviewed_atem_renderer_matches_official_text_goldens() -> None:
    fixture = _reference()
    goldens = fixture["chat_goldens"]
    assert isinstance(goldens, list)
    for golden in goldens:
        assert isinstance(golden, dict)
        renderer = MuseGlimmerATEMRenderer(current_date=str(golden["current_date"]))
        messages = tuple(
            NativeChatMessage(str(message["role"]), str(message["content"]))
            for message in golden["messages"]
        )
        rendered = renderer.render(
            messages,
            include_assistant_prompt=bool(golden["add_generation_prompt"]),
        )
        assert rendered == golden["rendered"], golden["name"]
        token_ids = golden["token_ids"]
        assert isinstance(token_ids, list) and token_ids[0] == 200_000
        assert all(isinstance(token_id, int) and 0 <= token_id < 202_048 for token_id in token_ids)


@pytest.mark.parametrize(
    "message",
    [
        NativeChatMessage("developer", "unsupported"),
        NativeChatMessage("tool", "unsupported"),
        NativeChatMessage("user", "unsupported", name="named"),
    ],
)
def test_reviewed_atem_renderer_rejects_unimplemented_constructs(
    message: NativeChatMessage,
) -> None:
    with pytest.raises(NativeChatConfigurationError, match="supports|does not support"):
        MuseGlimmerATEMRenderer(current_date="2026-08-12").render(
            [message], include_assistant_prompt=True
        )


def test_glimmer_renderer_resolution_never_falls_back_to_plain_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _muse_manifest()
    declared = manifest["chat_template"]
    assert isinstance(declared, dict)
    monkeypatch.setattr(
        "neuralfn.native_chat.MUSE_GLIMMER_ATEM_TEMPLATE_SHA256",
        declared["sha256"],
    )
    resolution = resolve_native_chat_renderer(
        manifest,
        "auto",
        allow_auto_fallback=True,
    )
    assert resolution.renderer.name == MUSE_GLIMMER_ATEM_PROFILE
    assert resolution.warning is None

    with pytest.raises(NativeChatConfigurationError, match="only supports"):
        resolve_native_chat_renderer(
            manifest,
            "plain_roles",
            allow_auto_fallback=True,
        )


def test_glimmer_stop_contract_excludes_eom() -> None:
    manifest = _muse_manifest()
    assert native_stop_token_ids(manifest) == (200_001, 200_008)
    manifest["stop_tokens"] = [200_001, 200_007, 200_008]
    with pytest.raises(NativeChatConfigurationError, match="message boundary"):
        native_stop_token_ids(manifest)


def _write_tiny_bytelevel_tokenizer(path: Path) -> int:
    tokenizers = pytest.importorskip("tokenizers")
    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(
        add_prefix_space=False,
        use_regex=False,
    )
    tokenizer.decoder = tokenizers.decoders.ByteLevel()
    trainer = tokenizers.trainers.BpeTrainer(
        vocab_size=320,
        initial_alphabet=tokenizers.pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<unk>", "<|begin_of_text|>", "<|eot|>"],
    )
    tokenizer.train_from_iterator(
        ["café 🦋", "hello tokenizer", "<|begin_of_text|>hello<|eot|>"],
        trainer=trainer,
    )
    tokenizer.save(str(path))
    return int(tokenizer.get_vocab_size(with_added_tokens=True))


def test_tokenizer_json_codec_is_authenticated_byte_safe_and_contained(tmp_path: Path) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    vocab_size = _write_tiny_bytelevel_tokenizer(tokenizer_path)
    digest = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    manifest = {
        "model": {"family": "fixture"},
        "tokenizer": {
            "family": "tokenizers",
            "backend": "tokenizers",
            "artifact_path": "tokenizer.json",
            "sha256": digest,
            "vocab_size": vocab_size,
        },
    }
    codec = load_native_text_codec(manifest, artifact_root=tmp_path)
    text = "café 🦋"
    token_ids = codec.encode(text)
    assert token_ids
    assert codec.decode(token_ids) == text
    incremental = codec.incremental_decoder()
    assert "".join(incremental.push(token_id) for token_id in token_ids) + incremental.finish() == text

    manifest["tokenizer"]["sha256"] = "0" * 64
    with pytest.raises(NativeChatConfigurationError, match="SHA-256 mismatch"):
        load_native_text_codec(manifest, artifact_root=tmp_path)

    manifest["tokenizer"]["sha256"] = digest
    manifest["tokenizer"]["artifact_path"] = "../tokenizer.json"
    with pytest.raises(NativeChatConfigurationError, match="escapes"):
        load_native_text_codec(manifest, artifact_root=tmp_path)
