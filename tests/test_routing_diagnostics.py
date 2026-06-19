from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from neuralfn import TorchTrainConfig, TorchTrainer, build_gpt_root_graph
from neuralfn.semantic import EXPERT_TO_DIMENSION, NUM_SEMANTIC_DIMS, NUM_VOCAB_DIMS, SEMANTIC_IGNORE_INDEX
from neuralfn.torch_backend import SemanticHashRouterStage, TopKRouteStage
from neuralfn.torch_templates import build_model_spec_from_config

HARNESS_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "cli" / "scripts"


def _load_harness_module(module_name: str, file_name: str):
    module_path = HARNESS_SCRIPTS_DIR / file_name
    scripts_dir = str(HARNESS_SCRIPTS_DIR)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load harness module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_train_jepa_module():
    return _load_harness_module("train_jepa_semantic_routing_diag", "train_jepa_semantic.py")


def _load_train_mixllama_module():
    _load_train_jepa_module()
    return _load_harness_module("train_mixllama_fast_routing_diag", "train_mixllama_fast.py")


def _load_train_semantic_router_module():
    _load_train_jepa_module()
    return _load_harness_module("train_semantic_router_moe_routing_diag", "train_semantic_router_moe.py")


def _cpu_graph(graph):
    template_spec = dict(graph.torch_config.get("template_spec", {}))
    template = dict(template_spec.get("template", {}))
    template["runtime"] = "eager"
    template_spec["template"] = template
    graph.torch_config = {
        **graph.torch_config,
        "device": "cpu",
        "amp_dtype": "bfloat16",
        "template_spec": template_spec,
    }
    return graph


def _toy_text_inputs() -> tuple[list[list[int]], list[list[int]]]:
    inputs = [
        [0, 1, 2, 3],
        [1, 2, 3, 4],
    ]
    targets = [
        [1, 2, 3, 4],
        [2, 3, 4, 5],
    ]
    return inputs, targets


def _toy_semantic_targets() -> torch.Tensor:
    return torch.full((2, 9), SEMANTIC_IGNORE_INDEX, dtype=torch.long)


def _make_graph(preset: str):
    config: dict[str, int | str] = {
        "preset": preset,
        "vocab_size": 32,
        "num_layers": 1,
        "model_dim": 32,
        "num_heads": 4,
        "num_kv_heads": 4,
        "multiple_of": 16,
    }
    if preset == "mixllama_fast":
        config["experts"] = 8
        config["top_k"] = 2
    elif preset == "semantic_router_moe":
        config["experts"] = NUM_VOCAB_DIMS
        config["top_k"] = 2
    spec = build_model_spec_from_config(config, preview_defaults=True)
    return _cpu_graph(build_gpt_root_graph(name=f"{preset}_routing_diag", model_spec=spec))


def _sample_routing_stats() -> dict[str, object]:
    return {
        "num_experts": 8,
        "route_rows": 10,
        "selection_counts": [4, 0, 6, 0, 0, 0, 0, 0],
        "selection_shares": [0.4, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0],
        "weight_mass": [4.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "weight_mass_shares": [0.4, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0],
        "active_experts": [0, 2],
        "active_expert_count": 2,
        "mean_router_entropy": 1.23,
        "mean_router_entropy_norm": 0.82,
        "mean_topk_entropy": 0.47,
        "mean_topk_entropy_norm": 0.91,
    }


def test_topk_route_stage_reports_routing_stats() -> None:
    stage = TopKRouteStage(top_k=2, experts=8)
    logits = torch.randn(4, 3, 8)
    weights, indices = stage(logits)
    stats = stage.last_routing_stats

    assert weights.shape == (4, 3, 2)
    assert indices.shape == (4, 3, 2)
    assert stats is not None
    assert stats["num_experts"] == 8
    assert stats["route_rows"] == 12
    assert len(stats["selection_counts"]) == 8
    assert len(stats["weight_mass"]) == 8
    assert stats["active_expert_count"] >= 1
    assert 0.0 <= float(stats["mean_router_entropy_norm"]) <= 1.0
    assert 0.0 <= float(stats["mean_topk_entropy_norm"]) <= 1.0
    assert sum(stats["selection_counts"]) == 24
    assert sum(stats["weight_mass"]) == pytest.approx(12.0, abs=1e-5)


def test_semantic_hash_router_stage_reports_routing_stats() -> None:
    stage = SemanticHashRouterStage(
        n_experts=NUM_VOCAB_DIMS,
        semantic_dim=NUM_SEMANTIC_DIMS,
        top_k=2,
        tables=4,
        n_buckets=64,
    )
    sem_vec = torch.randn(16, NUM_SEMANTIC_DIMS)
    bucket_indices = torch.randint(0, 64, (16, 4))
    topic_logits = torch.randn(16, NUM_VOCAB_DIMS, 64)
    sem_targets = torch.full((16, NUM_SEMANTIC_DIMS), SEMANTIC_IGNORE_INDEX, dtype=torch.long)

    weights, indices = stage(sem_vec, bucket_indices, topic_logits, sem_targets)
    stats = stage.last_routing_stats

    assert weights.shape == (16, 2)
    assert indices.shape == (16, 2)
    assert stats is not None
    assert stats["num_experts"] == NUM_VOCAB_DIMS
    assert stats["route_rows"] == 16
    assert stats["active_expert_count"] >= 1
    assert 0.0 <= float(stats["mean_router_entropy_norm"]) <= 1.0
    assert 0.0 <= float(stats["mean_topk_entropy_norm"]) <= 1.0
    assert sum(stats["selection_counts"]) == 32
    assert sum(stats["weight_mass"]) == pytest.approx(16.0, abs=1e-5)


def test_routing_stats_accumulate_across_microbatches() -> None:
    first = {
        "num_experts": 8,
        "route_rows": 2,
        "selection_counts": [4, 0, 0, 0, 0, 0, 0, 0],
        "weight_mass": [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "mean_router_entropy": 0.2,
        "mean_router_entropy_norm": 0.1,
        "mean_topk_entropy": 0.4,
        "mean_topk_entropy_norm": 0.3,
    }
    second = {
        "num_experts": 8,
        "route_rows": 2,
        "selection_counts": [0, 4, 0, 0, 0, 0, 0, 0],
        "weight_mass": [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "mean_router_entropy": 0.6,
        "mean_router_entropy_norm": 0.5,
        "mean_topk_entropy": 0.8,
        "mean_topk_entropy_norm": 0.7,
    }

    acc = TorchTrainer._accumulate_routing_stats(None, first)
    acc = TorchTrainer._accumulate_routing_stats(acc, second)
    stats = TorchTrainer._finalize_routing_stats(acc)

    assert stats is not None
    assert stats["route_rows"] == 4
    assert stats["active_expert_count"] == 2
    assert stats["selection_counts"][:2] == [4, 4]
    assert stats["weight_mass_shares"][:2] == pytest.approx([0.5, 0.5])
    assert stats["mean_router_entropy"] == pytest.approx(0.4)
    assert stats["mean_topk_entropy_norm"] == pytest.approx(0.5)


def test_routing_stats_omit_mixed_expert_widths() -> None:
    first = {
        "num_experts": 8,
        "route_rows": 2,
        "selection_counts": [4, 0, 0, 0, 0, 0, 0, 0],
        "weight_mass": [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "mean_router_entropy": 0.2,
        "mean_router_entropy_norm": 0.1,
        "mean_topk_entropy": 0.4,
        "mean_topk_entropy_norm": 0.3,
    }
    second = {
        "num_experts": 4,
        "route_rows": 2,
        "selection_counts": [2, 2, 0, 0],
        "weight_mass": [1.0, 1.0, 0.0, 0.0],
        "mean_router_entropy": 0.6,
        "mean_router_entropy_norm": 0.5,
        "mean_topk_entropy": 0.8,
        "mean_topk_entropy_norm": 0.7,
    }

    acc = TorchTrainer._accumulate_routing_stats(None, first)
    acc = TorchTrainer._accumulate_routing_stats(acc, second)
    assert TorchTrainer._finalize_routing_stats(acc) is None


def test_torch_trainer_on_step_includes_routing_stats_for_mixllama() -> None:
    graph = _make_graph("mixllama_fast")
    train_inputs, train_targets = _toy_text_inputs()
    step_infos: list[dict[str, object]] = []
    trainer = TorchTrainer(
        graph,
        TorchTrainConfig(
            epochs=1,
            batch_size=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            max_steps=1,
            train_batch_tokens=8,
            device="cpu",
        ),
    )

    trainer.train(train_inputs, train_targets, on_step=step_infos.append)

    assert len(step_infos) == 1
    routing_stats = step_infos[0].get("routing_stats")
    assert routing_stats is not None
    assert routing_stats["num_experts"] == 8
    assert routing_stats["route_rows"] == 8
    assert sum(routing_stats["selection_counts"]) == 16


def test_torch_trainer_on_step_includes_routing_stats_for_semantic_router(monkeypatch: pytest.MonkeyPatch) -> None:
    graph = _make_graph("semantic_router_moe")
    text_dataset = torch.utils.data.TensorDataset(
        torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
        torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]], dtype=torch.long),
    )
    semantic_dataset = _toy_semantic_targets()
    step_infos: list[dict[str, object]] = []
    trainer = TorchTrainer(
        graph,
        TorchTrainConfig(
            epochs=1,
            batch_size=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            max_steps=1,
            train_batch_tokens=8,
            device="cpu",
        ),
    )

    def fake_load_dataset_for_graph(cls, _graph, _dataset_names, *, seq_len):
        del seq_len
        return text_dataset

    monkeypatch.setattr(TorchTrainer, "_load_dataset_for_graph", classmethod(fake_load_dataset_for_graph))
    monkeypatch.setattr(TorchTrainer, "_load_semantic_tensors", classmethod(lambda cls, _graph, active_dims=2: {"sem_targets": semantic_dataset}))

    trainer.train([], [], on_step=step_infos.append)

    assert len(step_infos) == 1
    routing_stats = step_infos[0].get("routing_stats")
    assert routing_stats is not None
    assert routing_stats["num_experts"] == NUM_VOCAB_DIMS
    assert routing_stats["route_rows"] >= 1
    assert sum(routing_stats["selection_counts"]) == routing_stats["route_rows"] * 2


def test_torch_trainer_on_step_omits_routing_stats_for_dense_llama() -> None:
    graph = _make_graph("llama_fast")
    train_inputs, train_targets = _toy_text_inputs()
    step_infos: list[dict[str, object]] = []
    trainer = TorchTrainer(
        graph,
        TorchTrainConfig(
            epochs=1,
            batch_size=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            max_steps=1,
            train_batch_tokens=8,
            device="cpu",
        ),
    )

    trainer.train(train_inputs, train_targets, on_step=step_infos.append)

    assert len(step_infos) == 1
    assert "routing_stats" not in step_infos[0]


def test_train_jepa_progress_logger_renders_semantic_route_preview(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_train_jepa_module()
    messages: list[str] = []
    monkeypatch.setattr(module, "log_stage", messages.append)
    on_step, _ = module.build_progress_logger(train_log_every=1, resolved_epochs=1, max_steps=10)

    on_step(
        {
            "phase": "train",
            "step": 1,
            "max_steps": 10,
            "epoch": 1,
            "max_epochs": 1,
            "epoch_step": 1,
            "steps_per_epoch": 1,
            "loss": 1.0,
            "elapsed_seconds": 1.0,
            "learning_rates": [1e-3],
            "routing_stats": _sample_routing_stats(),
        }
    )

    assert len(messages) == 1
    assert "route=active 2/8" in messages[0]
    assert "entropy=0.82" in messages[0]
    assert f"{EXPERT_TO_DIMENSION[2]}:60%" in messages[0]


def test_mixllama_progress_logger_renders_numeric_route_preview(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_train_mixllama_module()
    messages: list[str] = []
    monkeypatch.setattr(module, "log_stage", messages.append)
    on_step, _ = module.build_progress_logger(train_log_every=1, resolved_epochs=1, max_steps=10)

    on_step(
        {
            "phase": "train",
            "step": 1,
            "max_steps": 10,
            "epoch": 1,
            "max_epochs": 1,
            "epoch_step": 1,
            "steps_per_epoch": 1,
            "loss": 1.0,
            "elapsed_seconds": 1.0,
            "learning_rates": [1e-3],
            "routing_stats": _sample_routing_stats(),
        }
    )

    assert len(messages) == 1
    assert "usage=[2:60%,0:40%]" in messages[0]
    assert "topk_h=0.91" in messages[0]


def test_semantic_router_progress_logger_logs_routing_during_warmup(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_train_semantic_router_module()
    messages: list[str] = []
    monkeypatch.setattr(module, "log_stage", messages.append)
    on_step, _ = module.build_progress_logger(train_log_every=1, resolved_epochs=1, max_steps=10)

    on_step(
        {
            "phase": "warmup",
            "step": 1,
            "warmup_steps": 4,
            "loss": 1.0,
            "elapsed_seconds": 1.0,
            "routing_stats": _sample_routing_stats(),
        }
    )

    assert len(messages) == 1
    assert "Warmup step 1/4" in messages[0]
    assert "route=active 2/8" in messages[0]
    assert f"{EXPERT_TO_DIMENSION[2]}:60%" in messages[0]


def test_progress_logger_omits_route_suffix_when_stats_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_train_jepa_module()
    messages: list[str] = []
    monkeypatch.setattr(module, "log_stage", messages.append)
    on_step, _ = module.build_progress_logger(train_log_every=1, resolved_epochs=1, max_steps=10)

    on_step(
        {
            "phase": "train",
            "step": 1,
            "max_steps": 10,
            "epoch": 1,
            "max_epochs": 1,
            "epoch_step": 1,
            "steps_per_epoch": 1,
            "loss": 1.0,
            "elapsed_seconds": 1.0,
            "learning_rates": [1e-3],
        }
    )

    assert len(messages) == 1
    assert "route=" not in messages[0]
