from __future__ import annotations

import pytest
import torch

from neuralfn import BuiltinNeurons, Edge, NeuronGraph, NeuronInstance
from neuralfn.neuron import module_neuron
from neuralfn.port import Port
from neuralfn.tile_cuda import TileCudaConfig, build_tile_function_module, load_tile_cuda_extension
from neuralfn.tile_cuda.autograd import BINARY_OPS, BINARY_PAIR_OPS, UNARY_OPS
from neuralfn.tile_cuda.modules import tile_function_reference
from neuralfn.torch_backend import CompiledTorchGraph


def _input_tensor(device: str) -> torch.Tensor:
    return torch.linspace(-4.0, 4.0, 2051, dtype=torch.float32, device=device, requires_grad=True)


@pytest.mark.parametrize("name", sorted(UNARY_OPS))
def test_tile_cuda_unary_cpu_fallback_matches_torch_forward_and_backward(name: str) -> None:
    module = build_tile_function_module(name, TileCudaConfig(backend="tile_cuda", strict=False))
    assert module is not None
    x = _input_tensor("cpu")
    expected_x = x.detach().clone().requires_grad_(True)

    actual = module(x)
    expected = tile_function_reference(name, expected_x)
    torch.testing.assert_close(actual, expected)

    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(x.grad, expected_x.grad)


@pytest.mark.parametrize("name", sorted(BINARY_OPS))
def test_tile_cuda_binary_cpu_fallback_matches_torch_forward_and_backward(name: str) -> None:
    module = build_tile_function_module(name, TileCudaConfig(backend="tile_cuda", strict=False))
    assert module is not None
    lhs = _input_tensor("cpu")
    rhs = torch.linspace(1.0, -1.0, 2051, dtype=torch.float32, requires_grad=True)
    expected_lhs = lhs.detach().clone().requires_grad_(True)
    expected_rhs = rhs.detach().clone().requires_grad_(True)

    actual = module(lhs, rhs)
    expected = tile_function_reference(name, expected_lhs, expected_rhs)
    torch.testing.assert_close(actual, expected)

    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(lhs.grad, expected_lhs.grad)
    torch.testing.assert_close(rhs.grad, expected_rhs.grad)


@pytest.mark.parametrize("name", sorted(BINARY_PAIR_OPS))
def test_tile_cuda_binary_pair_cpu_fallback_matches_torch_forward_and_backward(name: str) -> None:
    module = build_tile_function_module(name, TileCudaConfig(backend="tile_cuda", strict=False))
    assert module is not None
    lhs = _input_tensor("cpu")
    rhs = torch.linspace(1.0, -1.0, 2051, dtype=torch.float32, requires_grad=True)
    expected_lhs = lhs.detach().clone().requires_grad_(True)
    expected_rhs = rhs.detach().clone().requires_grad_(True)

    actual0, actual1 = module(lhs, rhs)
    expected0, expected1 = tile_function_reference(name, expected_lhs, expected_rhs)
    torch.testing.assert_close(actual0, expected0)
    torch.testing.assert_close(actual1, expected1)

    (actual0 + actual1.square()).sum().backward()
    (expected0 + expected1.square()).sum().backward()
    torch.testing.assert_close(lhs.grad, expected_lhs.grad)
    torch.testing.assert_close(rhs.grad, expected_rhs.grad)


def _require_cuda_tile_extension() -> TileCudaConfig:
    if not torch.cuda.is_available():
        pytest.skip("torch.cuda is not available")
    config = TileCudaConfig(backend="tile_cuda", strict=True, build_enabled=True)
    ext = load_tile_cuda_extension(config)
    if ext is None:
        pytest.skip("CUDA Tile extension could not be built or loaded in this environment")
    return config


@pytest.mark.parametrize("name", sorted(UNARY_OPS))
def test_tile_cuda_unary_gpu_kernel_matches_torch_forward_and_backward(name: str) -> None:
    config = _require_cuda_tile_extension()
    module = build_tile_function_module(name, config)
    assert module is not None
    x = _input_tensor("cuda")
    expected_x = x.detach().clone().requires_grad_(True)

    actual = module(x)
    expected = tile_function_reference(name, expected_x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(x.grad, expected_x.grad, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("name", sorted(BINARY_OPS))
def test_tile_cuda_binary_gpu_kernel_matches_torch_forward_and_backward(name: str) -> None:
    config = _require_cuda_tile_extension()
    module = build_tile_function_module(name, config)
    assert module is not None
    lhs = _input_tensor("cuda")
    rhs = torch.linspace(1.0, -1.0, 2051, dtype=torch.float32, device="cuda", requires_grad=True)
    expected_lhs = lhs.detach().clone().requires_grad_(True)
    expected_rhs = rhs.detach().clone().requires_grad_(True)

    actual = module(lhs, rhs)
    expected = tile_function_reference(name, expected_lhs, expected_rhs)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(lhs.grad, expected_lhs.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(rhs.grad, expected_rhs.grad, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("name", sorted(BINARY_PAIR_OPS))
def test_tile_cuda_binary_pair_gpu_kernel_matches_torch_forward_and_backward(name: str) -> None:
    config = _require_cuda_tile_extension()
    module = build_tile_function_module(name, config)
    assert module is not None
    lhs = _input_tensor("cuda")
    rhs = torch.linspace(1.0, -1.0, 2051, dtype=torch.float32, device="cuda", requires_grad=True)
    expected_lhs = lhs.detach().clone().requires_grad_(True)
    expected_rhs = rhs.detach().clone().requires_grad_(True)

    actual0, actual1 = module(lhs, rhs)
    expected0, expected1 = tile_function_reference(name, expected_lhs, expected_rhs)
    torch.testing.assert_close(actual0, expected0, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual1, expected1, rtol=1e-5, atol=1e-6)

    (actual0 + actual1.square()).sum().backward()
    (expected0 + expected1.square()).sum().backward()
    torch.testing.assert_close(lhs.grad, expected_lhs.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(rhs.grad, expected_rhs.grad, rtol=1e-5, atol=1e-6)


def test_compiled_graph_tile_strict_rejects_uncovered_module(monkeypatch: pytest.MonkeyPatch) -> None:
    from neuralfn.tile_cuda import runtime

    monkeypatch.setattr(runtime, "resolve_backend", lambda _config: "tile_cuda")
    uncovered = module_neuron(
        name="uncovered_tile_test_module",
        module_type="uncovered_tile_test_module",
        input_ports=[Port("x", range=(-1.0, 1.0), precision=0.001, dtype="tensor")],
        output_ports=[Port("y", range=(-1.0, 1.0), precision=0.001, dtype="tensor")],
    )
    graph = NeuronGraph(name="strict-uncovered")
    graph.add_node(NeuronInstance(BuiltinNeurons.input_node, instance_id="in"))
    graph.add_node(NeuronInstance(uncovered, instance_id="uncovered"))
    graph.add_node(NeuronInstance(BuiltinNeurons.output_node, instance_id="out"))
    graph.add_edge(Edge(src_node="in", src_port=0, dst_node="uncovered", dst_port=0))
    graph.add_edge(Edge(src_node="uncovered", src_port=0, dst_node="out", dst_port=0))
    graph.input_node_ids = ["in"]
    graph.output_node_ids = ["out"]

    with pytest.raises(RuntimeError, match="uncovered graph nodes"):
        CompiledTorchGraph(graph, kernel_backend="tile_cuda", tile_cuda_strict=True)


def test_compiled_graph_tile_strict_accepts_covered_scalar_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    from neuralfn.tile_cuda import runtime

    monkeypatch.setattr(runtime, "resolve_backend", lambda _config: "tile_cuda")
    graph = NeuronGraph(name="strict-covered")
    graph.add_node(NeuronInstance(BuiltinNeurons.input_node, instance_id="x"))
    graph.add_node(NeuronInstance(BuiltinNeurons.input_node, instance_id="y"))
    graph.add_node(NeuronInstance(BuiltinNeurons.add, instance_id="add"))
    graph.add_node(NeuronInstance(BuiltinNeurons.output_node, instance_id="out"))
    graph.add_edge(Edge(src_node="x", src_port=0, dst_node="add", dst_port=0))
    graph.add_edge(Edge(src_node="y", src_port=0, dst_node="add", dst_port=1))
    graph.add_edge(Edge(src_node="add", src_port=0, dst_node="out", dst_port=0))
    graph.input_node_ids = ["x", "y"]
    graph.output_node_ids = ["out"]

    compiled = CompiledTorchGraph(graph, kernel_backend="tile_cuda", tile_cuda_strict=True)
    out, = compiled(torch.tensor([1.0]), torch.tensor([2.0]))
    torch.testing.assert_close(out, torch.tensor([3.0]))
