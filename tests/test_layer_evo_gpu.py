from __future__ import annotations

import math
import os

import pytest
import torch
from torch.nn.utils import parameters_to_vector

from neuralfn import TorchTrainConfig, TorchTrainer, build_gpt_root_graph
from neuralfn.config import build_gpt2_evo_spec
from neuralfn.tile_cuda import TileCudaConfig, load_tile_cuda_extension
from neuralfn.torch_backend import CompiledTorchGraph

LAYER_EVO_KNOBS = (
    "layer_evo_enabled",
    "layer_evo_index",
    "layer_evo_fraction",
    "layer_evo_population",
    "layer_evo_mutation_scale",
    "layer_evo_seed",
)


def _require_tile_cuda_gpu() -> None:
    if os.environ.get("NFN_TILE_CUDA_TEST") != "1":
        pytest.skip("set NFN_TILE_CUDA_TEST=1 to run GPU layer-evo coverage")
    if not torch.cuda.is_available():
        pytest.skip("torch.cuda is not available")
    config = TileCudaConfig(backend="tile_cuda", strict=True, build_enabled=True)
    if load_tile_cuda_extension(config) is None:
        pytest.skip("CUDA Tile extension could not be built or loaded in this environment")


def _make_tiny_evo_graph():
    spec = build_gpt2_evo_spec(
        vocab_size=64,
        num_layers=3,
        model_dim=32,
        num_heads=2,
        layer_evo_index=1,
        layer_evo_fraction=0.25,
        layer_evo_population=3,
        layer_evo_mutation_scale=0.02,
        layer_evo_seed=7,
    )
    graph = build_gpt_root_graph(name="gpt2_evo_test", model_spec=spec)
    graph.torch_config = {**graph.torch_config, "device": "cuda", "amp_dtype": "float32"}
    return graph


def _compile_tile_cuda(graph) -> CompiledTorchGraph:
    compiled = CompiledTorchGraph(graph, kernel_backend="tile_cuda", tile_cuda_strict=True)
    assert compiled.resolved_kernel_backend == "tile_cuda"
    return compiled.to("cuda")


def _train_rows() -> tuple[list[list[int]], list[list[int]]]:
    train_inputs = [[(seed + offset) % 64 for offset in range(8)] for seed in range(8)]
    train_targets = [[(seed + offset + 1) % 64 for offset in range(8)] for seed in range(8)]
    return train_inputs, train_targets


def test_layer_evo_knobs_and_param_selection_tile_cuda() -> None:
    _require_tile_cuda_gpu()
    graph = _make_tiny_evo_graph()

    template_spec = graph.torch_config["template_spec"]
    for knob in LAYER_EVO_KNOBS:
        assert knob in template_spec
    config = TorchTrainer._layer_evo_config(template_spec)
    assert config is not None
    assert config["layer_index"] == 1
    assert config["interval"] == 4
    assert config["population"] == 3
    assert config["mutation_scale"] == pytest.approx(0.02)
    assert config["seed"] == 7

    defaulted = TorchTrainer._layer_evo_config({**template_spec, "layer_evo_index": None})
    assert defaulted is not None
    assert defaulted["layer_index"] == template_spec["num_layers"] // 2

    assert TorchTrainer._layer_evo_config({**template_spec, "layer_evo_enabled": False}) is None

    compiled = _compile_tile_cuda(graph)
    params = TorchTrainer._layer_evo_parameters(compiled, 1)
    assert params
    selected_ids = {id(param) for param in params}
    selected_names = [
        name for name, param in compiled.named_parameters() if id(param) in selected_ids
    ]
    assert all("node_modules.block_1." in name for name in selected_names)
    assert any(name.endswith("q_proj.proj.weight") for name in selected_names)
    for name, param in compiled.named_parameters():
        if "node_modules.block_0." in name or "node_modules.block_2." in name:
            assert id(param) not in selected_ids


def test_layer_evo_excluded_from_optimizer_groups_tile_cuda() -> None:
    _require_tile_cuda_gpu()

    for profile in ("parameter_golf", "adamw"):
        graph = _make_tiny_evo_graph()
        compiled = _compile_tile_cuda(graph)
        evo_params = TorchTrainer._layer_evo_parameters(compiled, 1)
        assert evo_params
        for param in evo_params:
            param.requires_grad_(False)
        evo_ids = {id(param) for param in evo_params}

        config = TorchTrainConfig(device="cuda", optimizer_profile=profile)
        optimizers = TorchTrainer._build_optimizers(compiled, config)
        optimized_ids = {
            id(param)
            for optimizer in optimizers
            for group in optimizer.param_groups
            for param in group["params"]
        }
        assert not (evo_ids & optimized_ids), f"evo params leaked into {profile} param groups"
        for name, param in compiled.named_parameters():
            if id(param) in evo_ids:
                assert param.requires_grad is False
            else:
                assert id(param) in optimized_ids, f"{name} missing from {profile} param groups"


def test_layer_evo_runs_at_interval_and_never_regresses_tile_cuda() -> None:
    _require_tile_cuda_gpu()
    graph = _make_tiny_evo_graph()
    train_inputs, train_targets = _train_rows()

    step_infos: list[dict] = []
    trainer = TorchTrainer(
        graph,
        TorchTrainConfig(
            epochs=2,
            batch_size=2,
            learning_rate=1e-3,
            weight_decay=0.0,
            optimizer_profile="adamw",
            device="cuda",
            kernel_backend="tile_cuda",
            tile_cuda_strict=True,
            amp_dtype="float32",
        ),
    )
    losses = trainer.train(train_inputs, train_targets, on_step=step_infos.append)
    assert losses
    assert all(math.isfinite(float(loss)) for loss in losses)

    evo_steps = sorted(info["step"] for info in step_infos if "layer_evo" in info)
    expected_steps = [step for step in range(1, len(step_infos) + 1) if step % 4 == 0]
    assert evo_steps == expected_steps
    for info in step_infos:
        if "layer_evo" not in info:
            continue
        evo = info["layer_evo"]
        assert evo["layer_index"] == 1
        assert evo["candidate_count"] == 3
        assert math.isfinite(float(evo["best_loss"]))

    # Elite preservation: the adopted candidate can never score worse than the
    # current weights because they are evaluated as candidate 0.
    compiled = _compile_tile_cuda(_make_tiny_evo_graph())
    evo_params = TorchTrainer._layer_evo_parameters(compiled, 1)
    tokens = torch.tensor(train_inputs[:4], dtype=torch.long, device="cuda")
    targets = torch.tensor(train_targets[:4], dtype=torch.long, device="cuda")
    macro_batches = [(tokens, targets)]
    base_loss, _rows, _stats = TorchTrainer._evaluate_evolutionary_candidate(
        compiled,
        parameters_to_vector(evo_params).detach().cpu(),
        evo_params,
        macro_batches,
        graph_name="gpt2_evo_test",
        device=torch.device("cuda"),
        amp_dtype=torch.float32,
        use_amp=False,
        template_runtime="eager",
        routing_modules=[],
    )
    info = TorchTrainer._run_route_evolution(
        compiled,
        evo_params,
        macro_batches,
        graph_name="gpt2_evo_test",
        device=torch.device("cuda"),
        amp_dtype=torch.float32,
        use_amp=False,
        template_runtime="eager",
        routing_modules=[],
        config={"population": 3, "mutation_scale": 0.02, "seed": 7},
        step=1,
    )
    assert info is not None
    assert info["candidate_count"] == 3
    assert info["best_loss"] <= base_loss + 1e-6
