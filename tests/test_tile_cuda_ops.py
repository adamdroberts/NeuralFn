from __future__ import annotations

import pytest
import torch

from neuralfn import BuiltinNeurons, Edge, NeuronGraph, NeuronInstance
from neuralfn.neuron import module_neuron
from neuralfn.port import Port
from neuralfn.tile_cuda import TileCudaConfig, build_tile_function_module, build_tile_module, load_tile_cuda_extension
from neuralfn.tile_cuda.autograd import BINARY_OPS, BINARY_PAIR_OPS, UNARY_OPS
from neuralfn.tile_cuda.modules import tile_function_reference
from neuralfn.torch_backend import CompiledTorchGraph


def _input_tensor(device: str) -> torch.Tensor:
    return torch.linspace(-4.0, 4.0, 2051, dtype=torch.float32, device=device, requires_grad=True)


def _input_tensor_fp16(device: str) -> torch.Tensor:
    return torch.linspace(-3.0, 3.0, 1024, dtype=torch.float16, device=device, requires_grad=True)


def _input_tensor_fp8(device: str, dtype: torch.dtype) -> torch.Tensor:
    return torch.linspace(-3.0, 3.0, 1024, dtype=torch.float32, device=device).to(dtype=dtype).requires_grad_(True)


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


def _require_cuda_for_strict_contract() -> TileCudaConfig:
    if not torch.cuda.is_available():
        pytest.skip("torch.cuda is not available")
    return TileCudaConfig(backend="tile_cuda", strict=True, build_enabled=False)


def test_tile_cuda_function_strict_dtype_error_names_requested_and_supported_dtypes() -> None:
    config = _require_cuda_for_strict_contract()
    module = build_tile_function_module("relu", config)
    assert module is not None
    x = torch.linspace(-1.0, 1.0, 17, device="cuda", dtype=torch.bfloat16)

    with pytest.raises(RuntimeError) as exc_info:
        module(x)

    message = str(exc_info.value)
    assert "dtype=bfloat16" in message
    assert "supported dtypes {float32, float16, float8_e4m3fn, float8_e5m2}" in message


def test_tile_cuda_module_strict_dtype_error_names_requested_and_supported_dtypes() -> None:
    config = _require_cuda_for_strict_contract()
    module = build_tile_module("loss_scale", {"coef": 0.5}, config)
    assert module is not None
    x = torch.linspace(-1.0, 1.0, 17, device="cuda", dtype=torch.bfloat16)

    with pytest.raises(RuntimeError) as exc_info:
        module(x)

    message = str(exc_info.value)
    assert "dtype=bfloat16" in message
    assert "supported dtypes {float32, float16, float8_e4m3fn, float8_e5m2}" in message


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


@pytest.mark.parametrize("name", sorted(UNARY_OPS))
def test_tile_cuda_unary_gpu_kernel_supports_fp16_forward_and_backward(name: str) -> None:
    config = _require_cuda_tile_extension()
    module = build_tile_function_module(name, config)
    assert module is not None
    x = _input_tensor_fp16("cuda")
    expected_x = x.detach().clone().requires_grad_(True)

    actual = module(x)
    expected = tile_function_reference(name, expected_x.to(torch.float32)).to(torch.float16)
    assert actual.dtype == torch.float16
    torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)

    actual.float().sum().backward()
    expected.float().sum().backward()
    torch.testing.assert_close(x.grad, expected_x.grad, rtol=5e-2, atol=5e-2)


@pytest.mark.parametrize("name", sorted(UNARY_OPS))
@pytest.mark.parametrize("dtype", (torch.float8_e4m3fn, torch.float8_e5m2))
def test_tile_cuda_unary_gpu_kernel_supports_fp8_forward_and_backward(name: str, dtype: torch.dtype) -> None:
    config = _require_cuda_tile_extension()
    module = build_tile_function_module(name, config)
    assert module is not None
    x = _input_tensor_fp8("cuda", dtype)
    expected_x = x.detach().clone().requires_grad_(True)

    actual = module(x)
    expected = tile_function_reference(name, expected_x.to(torch.float32)).to(dtype)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual.to(torch.float32), expected.to(torch.float32), rtol=2e-1, atol=2e-1)

    actual.float().sum().backward()
    expected.float().sum().backward()
    torch.testing.assert_close(x.grad.to(torch.float32), expected_x.grad.to(torch.float32), rtol=3e-1, atol=3e-1)


@pytest.mark.parametrize("name", sorted(BINARY_OPS))
def test_tile_cuda_binary_gpu_kernel_supports_fp16_forward_and_backward(name: str) -> None:
    config = _require_cuda_tile_extension()
    module = build_tile_function_module(name, config)
    assert module is not None
    lhs = _input_tensor_fp16("cuda")
    rhs = torch.linspace(1.0, -1.0, 1024, dtype=torch.float16, device="cuda", requires_grad=True)
    expected_lhs = lhs.detach().clone().requires_grad_(True)
    expected_rhs = rhs.detach().clone().requires_grad_(True)

    actual = module(lhs, rhs)
    expected = tile_function_reference(name, expected_lhs.to(torch.float32), expected_rhs.to(torch.float32)).to(torch.float16)
    assert actual.dtype == torch.float16
    torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)

    actual.float().sum().backward()
    expected.float().sum().backward()
    torch.testing.assert_close(lhs.grad, expected_lhs.grad, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(rhs.grad, expected_rhs.grad, rtol=5e-2, atol=5e-2)


@pytest.mark.parametrize("name", sorted(BINARY_OPS))
@pytest.mark.parametrize("dtype", (torch.float8_e4m3fn, torch.float8_e5m2))
def test_tile_cuda_binary_gpu_kernel_supports_fp8_forward_and_backward(name: str, dtype: torch.dtype) -> None:
    config = _require_cuda_tile_extension()
    module = build_tile_function_module(name, config)
    assert module is not None
    lhs = _input_tensor_fp8("cuda", dtype)
    rhs = torch.linspace(1.0, -1.0, 1024, dtype=torch.float32, device="cuda").to(dtype).requires_grad_(True)
    expected_lhs = lhs.detach().clone().requires_grad_(True)
    expected_rhs = rhs.detach().clone().requires_grad_(True)

    actual = module(lhs, rhs)
    expected = tile_function_reference(name, expected_lhs.to(torch.float32), expected_rhs.to(torch.float32)).to(dtype)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual.to(torch.float32), expected.to(torch.float32), rtol=2e-1, atol=2e-1)

    actual.float().sum().backward()
    expected.float().sum().backward()
    torch.testing.assert_close(lhs.grad.to(torch.float32), expected_lhs.grad.to(torch.float32), rtol=3e-1, atol=3e-1)
    torch.testing.assert_close(rhs.grad.to(torch.float32), expected_rhs.grad.to(torch.float32), rtol=3e-1, atol=3e-1)


@pytest.mark.parametrize("name", sorted(BINARY_PAIR_OPS))
def test_tile_cuda_binary_pair_gpu_kernel_supports_fp16_forward_and_backward(name: str) -> None:
    config = _require_cuda_tile_extension()
    module = build_tile_function_module(name, config)
    assert module is not None
    lhs = _input_tensor_fp16("cuda")
    rhs = torch.linspace(1.0, -1.0, 1024, dtype=torch.float16, device="cuda", requires_grad=True)
    expected_lhs = lhs.detach().clone().requires_grad_(True)
    expected_rhs = rhs.detach().clone().requires_grad_(True)

    actual0, actual1 = module(lhs, rhs)
    expected0, expected1 = tile_function_reference(name, expected_lhs.to(torch.float32), expected_rhs.to(torch.float32))
    expected0 = expected0.to(torch.float16)
    expected1 = expected1.to(torch.float16)
    assert actual0.dtype == torch.float16
    assert actual1.dtype == torch.float16
    torch.testing.assert_close(actual0, expected0, rtol=3e-3, atol=3e-3)
    torch.testing.assert_close(actual1, expected1, rtol=3e-3, atol=3e-3)

    (actual0.float() + actual1.float().square()).sum().backward()
    (expected0.float() + expected1.float().square()).sum().backward()
    torch.testing.assert_close(lhs.grad, expected_lhs.grad, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(rhs.grad, expected_rhs.grad, rtol=5e-2, atol=5e-2)


@pytest.mark.parametrize("name", sorted(BINARY_PAIR_OPS))
@pytest.mark.parametrize("dtype", (torch.float8_e4m3fn, torch.float8_e5m2))
def test_tile_cuda_binary_pair_gpu_kernel_supports_fp8_forward_and_backward(name: str, dtype: torch.dtype) -> None:
    config = _require_cuda_tile_extension()
    module = build_tile_function_module(name, config)
    assert module is not None
    lhs = _input_tensor_fp8("cuda", dtype)
    rhs = torch.linspace(1.0, -1.0, 1024, dtype=torch.float32, device="cuda").to(dtype).requires_grad_(True)
    expected_lhs = lhs.detach().clone().requires_grad_(True)
    expected_rhs = rhs.detach().clone().requires_grad_(True)

    actual0, actual1 = module(lhs, rhs)
    expected0, expected1 = tile_function_reference(name, expected_lhs.to(torch.float32), expected_rhs.to(torch.float32))
    expected0 = expected0.to(dtype)
    expected1 = expected1.to(dtype)
    assert actual0.dtype == dtype
    assert actual1.dtype == dtype
    torch.testing.assert_close(actual0.to(torch.float32), expected0.to(torch.float32), rtol=2e-1, atol=2e-1)
    torch.testing.assert_close(actual1.to(torch.float32), expected1.to(torch.float32), rtol=2e-1, atol=2e-1)

    (actual0.float() + actual1.float().square()).sum().backward()
    (expected0.float() + expected1.float().square()).sum().backward()
    torch.testing.assert_close(lhs.grad.to(torch.float32), expected_lhs.grad.to(torch.float32), rtol=3e-1, atol=3e-1)
    torch.testing.assert_close(rhs.grad.to(torch.float32), expected_rhs.grad.to(torch.float32), rtol=3e-1, atol=3e-1)


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
