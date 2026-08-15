from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from neuralfn.config import (
    MuseGlimmerDFlashDistillationSpec,
    build_muse_glimmer_spec,
)
from neuralfn.torch_backend import (
    DFlashDistillationTrainer,
    TorchTrainConfig,
    dflash_dpace_position_weights,
)
from neuralfn.torch_templates import (
    build_muse_glimmer_dflash_distillation_graph,
)


class _TinyFrozenTarget(nn.Module):
    def __init__(self, vocab_size: int = 17, hidden_size: int = 8) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(4)
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = True,
        use_cache: bool = False,
        return_dict: bool = True,
    ) -> SimpleNamespace:
        del attention_mask, output_hidden_states, use_cache, return_dict
        hidden = self.embedding(input_ids)
        hidden_states = [hidden]
        for layer in self.layers:
            hidden = torch.tanh(layer(hidden))
            hidden_states.append(hidden)
        return SimpleNamespace(
            logits=self.lm_head(hidden), hidden_states=tuple(hidden_states)
        )


def _recipe(*, seed: int = 77) -> MuseGlimmerDFlashDistillationSpec:
    return MuseGlimmerDFlashDistillationSpec(
        target_checkpoint_sha256="1" * 64,
        target_config_sha256="2" * 64,
        tokenizer_sha256="3" * 64,
        chat_template_sha256="4" * 64,
        num_anchors=2,
        seed=seed,
    )


def _graph(recipe: MuseGlimmerDFlashDistillationSpec):
    target_spec = build_muse_glimmer_spec(
        num_layers=4,
        model_dim=8,
        num_heads=2,
        num_kv_heads=1,
        head_dim=2,
        attention_inner_dim=4,
        intermediate_size=16,
        vocab_size=17,
        window_size=4,
        max_position_embeddings=32,
    )
    return build_muse_glimmer_dflash_distillation_graph(
        "tiny_dflash_distillation",
        target_spec,
        distillation_spec=recipe,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=2,
        intermediate_size=16,
        block_size=4,
        mask_token_id=16,
        window_size=4,
        target_layer_ids=(0, 2),
    )


def _dataset() -> torch.Tensor:
    return torch.tensor(
        [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [2, 3, 4, 5, 6, 7, 8, 9],
            [3, 4, 5, 6, 7, 8, 9, 10],
            [4, 5, 6, 7, 8, 9, 10, 11],
        ],
        dtype=torch.long,
    )


def _make_trainer(
    *,
    torch_seed: int,
    max_steps: int,
    target_state: dict[str, torch.Tensor] | None = None,
) -> tuple[DFlashDistillationTrainer, _TinyFrozenTarget]:
    torch.manual_seed(torch_seed)
    target = _TinyFrozenTarget()
    if target_state is not None:
        target.load_state_dict(target_state, strict=True)
    recipe = _recipe()
    trainer = DFlashDistillationTrainer(
        _graph(recipe),
        target,
        recipe,
        TorchTrainConfig(
            device="cpu",
            batch_size=2,
            max_steps=max_steps,
            learning_rate=1.0e-3,
            weight_decay=0.0,
            amp_dtype="float32",
        ),
    )
    return trainer, target


def test_dpace_position_weights_match_suffix_sum_and_mask() -> None:
    confidence = torch.tensor([[0.5, 0.25, 0.8]], requires_grad=True)
    valid = torch.tensor([[1.0, 0.0, 1.0]])
    result = dflash_dpace_position_weights(confidence, 0.5, valid)
    # smoothed valid positions are 0.75 and 0.9. The invalid middle slot is
    # a multiplicative no-op and contributes no suffix value.
    expected = torch.tensor([[0.75 + 0.75 * 0.9, 0.75 * 0.9, 0.75 * 0.9]])
    expected[:, 1] = 0.75 * 0.9
    assert torch.allclose(result, expected)
    assert result.requires_grad is False
    with pytest.raises(ValueError, match="alpha"):
        dflash_dpace_position_weights(confidence, 0.0)


def test_dflash_distillation_freezes_target_and_resumes_bit_exactly(
    tmp_path: Path,
) -> None:
    continuous, continuous_target = _make_trainer(torch_seed=19, max_steps=2)
    target_state = {
        key: value.detach().clone()
        for key, value in continuous_target.state_dict().items()
    }
    continuous_losses = continuous.train(_dataset())
    assert len(continuous_losses) == 2
    assert all(torch.isfinite(torch.tensor(continuous_losses)))
    assert all(parameter.requires_grad is False for parameter in continuous_target.parameters())
    for key, value in continuous_target.state_dict().items():
        assert torch.equal(value, target_state[key])

    first, _ = _make_trainer(
        torch_seed=19, max_steps=1, target_state=target_state
    )
    assert len(first.train(_dataset())) == 1
    checkpoint = tmp_path / "dflash-step-1.pt"
    digest = first.save_training_checkpoint(checkpoint)
    assert len(digest) == 64
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["format"] == "neuralfn.muse_glimmer_dflash_distillation.v1"
    assert payload["recipe"]["target_checkpoint_sha256"] == "1" * 64
    assert payload["global_step"] == 1

    resumed, resumed_target = _make_trainer(
        torch_seed=999, max_steps=2, target_state=target_state
    )
    resumed_losses = resumed.train(_dataset(), resume_from_checkpoint=checkpoint)
    assert resumed_losses == continuous_losses
    assert resumed.last_global_step == 2
    assert continuous.last_compiled_graph is not None
    assert resumed.last_compiled_graph is not None
    for key, expected in continuous.last_compiled_graph.state_dict().items():
        assert torch.equal(resumed.last_compiled_graph.state_dict()[key], expected)
    for key, value in resumed_target.state_dict().items():
        assert torch.equal(value, target_state[key])


def test_dflash_distillation_masks_padding_and_reports_online_acceptance() -> None:
    trainer, _target = _make_trainer(torch_seed=29, max_steps=1)
    data = _dataset()
    attention = torch.ones_like(data)
    attention[:, -1] = 0
    loss_mask = attention.float()
    losses = trainer.train(
        data,
        attention_mask=attention,
        loss_mask=loss_mask,
        labels=data.masked_fill(attention == 0, -100),
    )
    assert len(losses) == 1
    assert trainer.metrics_history[0].valid_blocks > 0
    stats = trainer.evaluate_greedy_acceptance(
        data[:1, :3], max_new_tokens=6, eos_token_ids=()
    )
    assert len(stats["token_ids"]) == 6
    assert stats["blocks"] > 0
    assert 0 <= stats["accepted_tokens"] <= stats["proposed_tokens"]
    assert 0.0 <= stats["acceptance_rate"] <= 1.0


def test_dflash_distillation_fails_closed_on_missing_lineage_or_plain_graph() -> None:
    target = _TinyFrozenTarget()
    incomplete = MuseGlimmerDFlashDistillationSpec()
    with pytest.raises(ValueError, match="lineage"):
        DFlashDistillationTrainer(
            _graph(incomplete),
            target,
            incomplete,
            TorchTrainConfig(device="cpu"),
        )

    recipe = _recipe()
    plain = _graph(recipe)
    plain.torch_config["dflash_spec"].pop("training_attention_mask")
    with pytest.raises(ValueError, match="distillation graph"):
        DFlashDistillationTrainer(
            plain,
            _TinyFrozenTarget(),
            recipe,
            TorchTrainConfig(device="cpu"),
        )


def test_native_assistant_export_rejects_nonproduction_geometry() -> None:
    trainer, _target = _make_trainer(torch_seed=31, max_steps=1)
    trainer.train(_dataset())
    with pytest.raises(ValueError, match="production assistant geometry"):
        trainer.export_native_assistant("unused")
