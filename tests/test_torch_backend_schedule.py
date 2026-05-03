from unittest.mock import patch

import pytest

from neuralfn import TorchTrainConfig, TorchTrainer, build_gpt_root_graph
from neuralfn.config import build_gpt2_spec


def _make_gpt_graph():
    spec = build_gpt2_spec(
        vocab_size=16,
        num_layers=2,
        model_dim=16,
        num_heads=4,
    )
    graph = build_gpt_root_graph(name="schedule_test", model_spec=spec)
    graph.torch_config = {"device": "cpu", "amp_dtype": "bfloat16"}
    return graph


def _toy_text_dataset() -> tuple[list[list[int]], list[list[int]]]:
    inputs = [
        [0, 1, 2, 3],
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7],
    ]
    targets = [
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7],
        [5, 6, 7, 8],
    ]
    return inputs, targets


class _PerfCounter:
    def __init__(self, *, delta: float) -> None:
        self.current = -delta
        self.delta = delta

    def __call__(self) -> float:
        self.current += self.delta
        return self.current


def test_lr_warmdown_uses_fractional_tail_span_even_with_wallclock_cap() -> None:
    graph = _make_gpt_graph()
    train_inputs, train_targets = _toy_text_dataset()
    learning_rates: list[float] = []
    trainer = TorchTrainer(
        graph,
        TorchTrainConfig(
            epochs=1,
            batch_size=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            max_steps=5,
            warmdown_fraction=0.75,
            max_wallclock_seconds=999.0,
            device="cpu",
        ),
    )

    losses = trainer.train(
        train_inputs,
        train_targets,
        on_step=lambda info: learning_rates.append(float(info["learning_rates"][0])),
    )

    assert len(losses) == 1
    assert len(learning_rates) == 5
    assert learning_rates[0] == pytest.approx(1e-3)
    assert learning_rates[1] == pytest.approx(1e-3)
    assert learning_rates[2] < learning_rates[1]
    assert learning_rates[3] < learning_rates[2]
    assert learning_rates[4] < learning_rates[3]


def test_wallclock_cap_stops_early_without_triggering_early_lr_decay() -> None:
    graph = _make_gpt_graph()
    train_inputs, train_targets = _toy_text_dataset()
    learning_rates: list[float] = []
    trainer = TorchTrainer(
        graph,
        TorchTrainConfig(
            epochs=1,
            batch_size=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            max_steps=5,
            warmdown_fraction=0.75,
            max_wallclock_seconds=0.11,
            device="cpu",
        ),
    )

    with patch("neuralfn.torch_backend.time.perf_counter", side_effect=_PerfCounter(delta=0.06)):
        losses = trainer.train(
            train_inputs,
            train_targets,
            on_step=lambda info: learning_rates.append(float(info["learning_rates"][0])),
        )

    assert len(losses) == 1
    assert len(learning_rates) == 1
    assert learning_rates[0] == pytest.approx(1e-3)


def test_cosine_lr_decay_reaches_min_lr_and_stays_there() -> None:
    graph = _make_gpt_graph()
    train_inputs, train_targets = _toy_text_dataset()
    learning_rates: list[float] = []
    trainer = TorchTrainer(
        graph,
        TorchTrainConfig(
            epochs=2,
            batch_size=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            max_steps=6,
            lr_decay_iters=4,
            min_lr=1e-4,
            device="cpu",
        ),
    )

    losses = trainer.train(
        train_inputs,
        train_targets,
        on_step=lambda info: learning_rates.append(float(info["learning_rates"][0])),
    )

    assert len(losses) >= 1
    assert len(learning_rates) == 6
    assert learning_rates[0] == pytest.approx(1e-3)
    assert learning_rates[1] < learning_rates[0]
    assert learning_rates[2] < learning_rates[1]
    assert learning_rates[3] < learning_rates[2]
    assert learning_rates[4] == pytest.approx(1e-4)
    assert learning_rates[5] == pytest.approx(1e-4)


def test_fractional_warmdown_still_applies_when_cosine_decay_is_unset() -> None:
    graph = _make_gpt_graph()
    train_inputs, train_targets = _toy_text_dataset()
    learning_rates: list[float] = []
    trainer = TorchTrainer(
        graph,
        TorchTrainConfig(
            epochs=1,
            batch_size=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            max_steps=5,
            warmdown_fraction=0.4,
            lr_decay_iters=None,
            min_lr=1e-5,
            device="cpu",
        ),
    )

    losses = trainer.train(
        train_inputs,
        train_targets,
        on_step=lambda info: learning_rates.append(float(info["learning_rates"][0])),
    )

    assert len(losses) == 1
    assert len(learning_rates) == 5
    assert learning_rates[0] == pytest.approx(1e-3)
    assert learning_rates[1] == pytest.approx(1e-3)
    assert learning_rates[2] == pytest.approx(1e-3)
    assert learning_rates[3] == pytest.approx(1e-3)
    assert learning_rates[4] < learning_rates[3]


def test_invalid_warmdown_fraction_is_rejected() -> None:
    graph = _make_gpt_graph()
    with pytest.raises(ValueError, match="warmdown_fraction must be within \\[0.0, 1.0\\]"):
        TorchTrainer(
            graph,
            TorchTrainConfig(
                epochs=1,
                batch_size=1,
                learning_rate=1e-3,
                weight_decay=0.0,
                warmdown_fraction=1.1,
                device="cpu",
            ),
        )


def test_respect_epoch_boundaries_uses_short_tail_step_without_wrapping_loader() -> None:
    graph = _make_gpt_graph()
    train_inputs, train_targets = _toy_text_dataset()
    step_infos: list[dict[str, object]] = []
    trainer = TorchTrainer(
        graph,
        TorchTrainConfig(
            epochs=1,
            batch_size=2,
            learning_rate=1e-3,
            weight_decay=0.0,
            max_steps=2,
            train_batch_tokens=16,
            device="cpu",
            respect_epoch_boundaries=True,
        ),
    )

    losses = trainer.train(
        train_inputs,
        train_targets,
        on_step=lambda info: step_infos.append(dict(info)),
    )

    assert len(losses) == 1
    assert len(step_infos) == 2
    assert [int(info["grad_accum_steps"]) for info in step_infos] == [2, 2]
    assert [int(info.get("actual_grad_accum_steps", info["grad_accum_steps"])) for info in step_infos] == [2, 1]
    assert graph.torch_config["drop_last"] is False
    assert graph.torch_config["respect_epoch_boundaries"] is True


def test_default_schedule_still_wraps_loader_when_epoch_boundaries_are_disabled() -> None:
    graph = _make_gpt_graph()
    train_inputs, train_targets = _toy_text_dataset()
    step_infos: list[dict[str, object]] = []
    trainer = TorchTrainer(
        graph,
        TorchTrainConfig(
            epochs=1,
            batch_size=2,
            learning_rate=1e-3,
            weight_decay=0.0,
            max_steps=2,
            train_batch_tokens=16,
            device="cpu",
            respect_epoch_boundaries=False,
        ),
    )

    losses = trainer.train(
        train_inputs,
        train_targets,
        on_step=lambda info: step_infos.append(dict(info)),
    )

    assert len(losses) == 1
    assert len(step_infos) == 2
    assert [int(info["grad_accum_steps"]) for info in step_infos] == [2, 2]
    assert [int(info.get("actual_grad_accum_steps", info["grad_accum_steps"])) for info in step_infos] == [2, 2]
    assert graph.torch_config["drop_last"] is False
    assert graph.torch_config["respect_epoch_boundaries"] is False
