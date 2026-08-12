from __future__ import annotations

import torch
import torch.nn.functional as F

from neuralfn.tile_cuda.config import TileCudaConfig
from neuralfn.tile_cuda.modules import build_tile_module
from neuralfn.torch_backend import CompiledTorchGraph, RMSNormStage, SwiGLUStage, build_module
from neuralfn.torch_templates import build_gpt_root_graph, build_model_spec_from_config


def _compiled_llama(*, mlp_multiplier: float = 8.0 / 3.0) -> CompiledTorchGraph:
    spec = build_model_spec_from_config(
        {
            "preset": "llama",
            "num_layers": 1,
            "model_dim": 16,
            "num_heads": 4,
            "num_kv_heads": 2,
            "multiple_of": 8,
            "mlp_multiplier": mlp_multiplier,
            "vocab_size": 32,
        },
        preview_defaults=True,
    )
    graph = build_gpt_root_graph(name="llama_backend_semantics", model_spec=spec)
    return CompiledTorchGraph(graph, kernel_backend="torch")


def test_legacy_rms_norm_without_model_dim_remains_parameter_free() -> None:
    config = {"eps": 1.0e-6}
    torch_stage = build_module("rms_norm", config)
    tile_stage = build_tile_module(
        "rms_norm",
        config,
        TileCudaConfig(backend="tile_cuda", strict=False),
    )
    assert isinstance(torch_stage, RMSNormStage)
    assert tile_stage is not None
    assert torch_stage.weight is None
    assert getattr(tile_stage, "weight") is None
    assert tuple(torch_stage.state_dict()) == ()
    assert tuple(tile_stage.state_dict()) == ()

    x = torch.linspace(-2.0, 2.0, 2 * 3 * 5).reshape(2, 3, 5)
    expected = F.rms_norm(x, (x.size(-1),), eps=config["eps"])
    torch.testing.assert_close(torch_stage(x), expected)
    torch.testing.assert_close(tile_stage(x), expected)


def test_llama_template_rms_norm_has_affine_state_and_forward() -> None:
    compiled = _compiled_llama()
    model = compiled.node_modules["model"]
    block = model.node_modules["block_0"]
    norms = (
        model.node_modules["final_norm"],
        block.node_modules["attn_norm"],
        block.node_modules["mlp_norm"],
    )

    assert all(isinstance(norm, RMSNormStage) for norm in norms)
    for norm in norms:
        assert norm.weight is not None
        assert tuple(norm.weight.shape) == (16,)
        assert norm.weight.requires_grad
        torch.testing.assert_close(norm.weight, torch.ones(16))

    state = compiled.state_dict()
    assert {
        name for name in state if name.endswith("norm.weight")
    } == {
        "node_modules.model.node_modules.block_0.node_modules.attn_norm.weight",
        "node_modules.model.node_modules.block_0.node_modules.mlp_norm.weight",
        "node_modules.model.node_modules.final_norm.weight",
    }

    norm = block.node_modules["attn_norm"]
    weight = torch.linspace(0.5, 1.5, 16)
    with torch.no_grad():
        norm.weight.copy_(weight)
    x = torch.linspace(-1.5, 1.5, 2 * 4 * 16).reshape(2, 4, 16)
    expected = F.rms_norm(x, (16,), weight=weight, eps=1.0e-6)
    torch.testing.assert_close(norm(x), expected)


def test_affine_rms_norm_tile_cpu_reference_and_strict_modes_match_torch() -> None:
    config = {"model_dim": 8, "eps": 1.0e-6}
    tile_results: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for strict in (False, True):
        torch_stage = build_module("rms_norm", config)
        tile_stage = build_tile_module(
            "rms_norm",
            config,
            TileCudaConfig(backend="tile_cuda", strict=strict),
        )
        assert isinstance(torch_stage, RMSNormStage)
        assert tile_stage is not None
        with torch.no_grad():
            torch_stage.weight.copy_(torch.linspace(0.6, 1.4, 8))
        tile_stage.load_state_dict(torch_stage.state_dict())

        torch_input = torch.linspace(-2.0, 2.0, 2 * 3 * 8).reshape(2, 3, 8).requires_grad_()
        tile_input = torch_input.detach().clone().requires_grad_()
        torch_output = torch_stage(torch_input)
        tile_output = tile_stage(tile_input)
        torch.testing.assert_close(tile_output, torch_output)

        torch_output.square().mean().backward()
        tile_output.square().mean().backward()
        torch.testing.assert_close(tile_input.grad, torch_input.grad)
        torch.testing.assert_close(tile_stage.weight.grad, torch_stage.weight.grad)
        tile_results.append(
            (
                tile_output.detach(),
                tile_input.grad.detach(),
                tile_stage.weight.grad.detach(),
            )
        )

    for strict_result, reference_result in zip(tile_results[1], tile_results[0], strict=True):
        torch.testing.assert_close(strict_result, reference_result)


def test_llama_custom_mlp_multiplier_controls_swiglu_hidden_shape() -> None:
    compiled = _compiled_llama(mlp_multiplier=2.5)
    model = compiled.node_modules["model"]
    block = model.node_modules["block_0"]
    mlp = block.node_modules["mlp"]
    swiglu = mlp.node_modules["swiglu"]
    assert isinstance(swiglu, SwiGLUStage)

    # int(16 * 2.5) = 40, already aligned to multiple_of=8. The former
    # hard-coded 8/3 rule produced 48 and ignored the serialized multiplier.
    assert tuple(swiglu.w1.weight.shape) == (40, 16)
    assert tuple(swiglu.w3.weight.shape) == (40, 16)
    assert tuple(swiglu.w2.weight.shape) == (16, 40)

    config = {"model_dim": 16, "mlp_mult": 2.5, "multiple_of": 8}
    tile_stage = build_tile_module(
        "swiglu",
        config,
        TileCudaConfig(backend="tile_cuda", strict=False),
    )
    assert tile_stage is not None
    tile_stage.load_state_dict(swiglu.state_dict())
    x = torch.linspace(-1.0, 1.0, 2 * 3 * 16).reshape(2, 3, 16)
    torch.testing.assert_close(tile_stage(x), swiglu(x))
