from __future__ import annotations

import math
from unittest.mock import patch

from neuralfn import TorchTrainConfig, TorchTrainer, build_gpt_root_graph
from neuralfn.config import build_gpt2_spec


def _make_gpt_graph():
    spec = build_gpt2_spec(
        vocab_size=16,
        num_layers=1,
        model_dim=8,
        num_heads=2,
        num_kv_heads=2,
    )
    graph = build_gpt_root_graph(name="evolutionary_test", model_spec=spec)
    graph.torch_config = {"device": "cpu", "amp_dtype": "bfloat16"}
    return graph


def _toy_text_dataset() -> tuple[list[list[int]], list[list[int]]]:
    inputs = [
        [0, 1, 2, 3],
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
    ]
    targets = [
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7],
    ]
    return inputs, targets


class _PerfCounter:
    def __init__(self, *, delta: float) -> None:
        self.current = -delta
        self.delta = delta

    def __call__(self) -> float:
        self.current += self.delta
        return self.current


def test_evolutionary_torch_training_returns_finite_losses_and_metadata() -> None:
    graph = _make_gpt_graph()
    train_inputs, train_targets = _toy_text_dataset()
    step_infos: list[dict[str, object]] = []
    trainer = TorchTrainer(
        graph,
        TorchTrainConfig(
            epochs=1,
            batch_size=1,
            max_steps=2,
            train_batch_tokens=4,
            device="cpu",
            evolutionary=True,
            evo_population_size=4,
            evo_mutation_rate=0.2,
            evo_mutation_scale=0.05,
            evo_crossover_rate=0.4,
            evo_tournament_size=2,
            evo_elite_count=1,
            evo_seed=7,
        ),
    )

    losses = trainer.train(train_inputs, train_targets, on_step=step_infos.append)

    assert len(losses) == 1
    assert math.isfinite(float(losses[0]))
    assert len(step_infos) == 2
    assert step_infos[0]["optimization_method"] == "evolutionary"
    assert step_infos[0]["learning_rates"] == []
    assert graph.torch_config["optimization_method"] == "evolutionary"
    assert graph.torch_config["evolutionary"] == {
        "population_size": 4,
        "mutation_rate": 0.2,
        "mutation_scale": 0.05,
        "crossover_rate": 0.4,
        "tournament_size": 2,
        "elite_count": 1,
        "seed": 7,
    }


def test_evolutionary_wallclock_limit_stops_after_current_generation() -> None:
    graph = _make_gpt_graph()
    train_inputs, train_targets = _toy_text_dataset()
    step_infos: list[dict[str, object]] = []
    trainer = TorchTrainer(
        graph,
        TorchTrainConfig(
            epochs=2,
            batch_size=1,
            max_steps=5,
            train_batch_tokens=4,
            max_wallclock_seconds=0.11,
            device="cpu",
            evolutionary=True,
            evo_population_size=4,
            evo_seed=11,
        ),
    )

    with patch("neuralfn.torch_backend.time.perf_counter", side_effect=_PerfCounter(delta=0.06)):
        losses = trainer.train(train_inputs, train_targets, on_step=step_infos.append)

    assert len(losses) == 1
    assert len(step_infos) == 1
    assert step_infos[0]["step"] == 1


def test_evolutionary_stop_exits_cleanly_after_first_generation() -> None:
    graph = _make_gpt_graph()
    train_inputs, train_targets = _toy_text_dataset()
    step_infos: list[dict[str, object]] = []
    trainer = TorchTrainer(
        graph,
        TorchTrainConfig(
            epochs=2,
            batch_size=1,
            max_steps=5,
            train_batch_tokens=4,
            device="cpu",
            evolutionary=True,
            evo_population_size=4,
            evo_seed=19,
        ),
    )

    def on_step(info: dict[str, object]) -> None:
        step_infos.append(dict(info))
        trainer.stop()

    losses = trainer.train(train_inputs, train_targets, on_step=on_step)

    assert len(losses) == 1
    assert len(step_infos) == 1
    assert step_infos[0]["step"] == 1


def test_gradient_mode_still_records_gradient_descent_metadata() -> None:
    graph = _make_gpt_graph()
    train_inputs, train_targets = _toy_text_dataset()
    trainer = TorchTrainer(
        graph,
        TorchTrainConfig(
            epochs=1,
            batch_size=1,
            max_steps=1,
            train_batch_tokens=4,
            device="cpu",
        ),
    )

    losses = trainer.train(train_inputs, train_targets)

    assert len(losses) == 1
    assert math.isfinite(float(losses[0]))
    assert graph.torch_config["optimization_method"] == "gradient_descent"
    assert "evolutionary" not in graph.torch_config
