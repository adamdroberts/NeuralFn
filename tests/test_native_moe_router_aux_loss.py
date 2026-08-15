from __future__ import annotations

import math
from pathlib import Path

import pytest

from tools.native_moe_router_aux_oracle import (
    accumulate_router_aux_gradient,
    standard_moe_router_aux,
    weighted_multi_layer_router_aux_loss,
)


ROOT = Path(__file__).resolve().parents[1]


def test_zero_coefficient_is_an_exact_gradient_noop() -> None:
    result = standard_moe_router_aux(
        [[4.0, -1.0, 0.25], [-2.0, 3.0, 1.0], [0.5, 0.1, -0.4]],
        0.0,
    )

    assert result.weighted_loss == 0.0
    assert result.raw_loss > 0.0
    assert result.gradient == ((0.0, 0.0, 0.0),) * 3


def test_multirow_multiexpert_loss_matches_shipped_graph_formula() -> None:
    logits = [[2.0, 0.0, -1.0], [0.5, 1.5, -0.5], [-1.0, 0.25, 2.25], [0.0, 0.0, 0.0]]
    coefficient = 0.17
    result = standard_moe_router_aux(logits, coefficient)

    expected_density = tuple(
        sum(row[expert] for row in result.probabilities) / len(result.probabilities)
        for expert in range(3)
    )
    expected_raw_loss = 3.0 * sum(value * value for value in expected_density)
    assert result.density == pytest.approx(expected_density)
    assert sum(result.density) == pytest.approx(1.0)
    assert result.raw_loss == pytest.approx(expected_raw_loss)
    assert result.weighted_loss == pytest.approx(coefficient * expected_raw_loss)
    assert all(sum(row) == pytest.approx(0.0, abs=1.0e-14) for row in result.gradient)


def test_multi_layer_loss_sums_raw_layers_then_applies_coefficient_once() -> None:
    layers = [
        [[2.0, -1.0], [0.25, 0.75]],
        [[-0.5, 0.0, 1.5], [2.0, -1.0, 0.5], [0.1, 0.2, 0.3]],
    ]
    coefficient = 0.031
    expected = coefficient * sum(standard_moe_router_aux(layer, 1.0).raw_loss for layer in layers)
    assert weighted_multi_layer_router_aux_loss(layers, coefficient) == pytest.approx(expected)


def test_exact_all_expert_gradient_matches_finite_difference() -> None:
    logits = [[1.2, -0.7, 0.3, 2.1], [-0.4, 0.8, 1.4, -1.1], [0.2, 0.0, -0.9, 0.6]]
    coefficient = 0.073
    analytic = standard_moe_router_aux(logits, coefficient).gradient
    epsilon = 1.0e-6

    for row in range(len(logits)):
        for expert in range(len(logits[row])):
            plus = [values[:] for values in logits]
            minus = [values[:] for values in logits]
            plus[row][expert] += epsilon
            minus[row][expert] -= epsilon
            numeric = (
                standard_moe_router_aux(plus, coefficient).weighted_loss
                - standard_moe_router_aux(minus, coefficient).weighted_loss
            ) / (2.0 * epsilon)
            assert analytic[row][expert] == pytest.approx(numeric, rel=2.0e-6, abs=2.0e-8)


def test_aux_gradient_accumulates_with_expert_route_gradient() -> None:
    base = ((0.5, -0.2, 0.7), (-0.1, 0.3, -0.4))
    auxiliary = standard_moe_router_aux([[1.0, 0.0, -1.0], [-0.5, 1.5, 0.25]], 0.11).gradient
    accumulated = accumulate_router_aux_gradient(base, auxiliary)

    for row in range(2):
        for expert in range(3):
            assert accumulated[row][expert] == pytest.approx(base[row][expert] + auxiliary[row][expert])
            assert accumulated[row][expert] - auxiliary[row][expert] == pytest.approx(base[row][expert])


@pytest.mark.parametrize("coefficient", [-0.1, math.inf, -math.inf, math.nan])
def test_invalid_coefficient_fails_closed(coefficient: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        standard_moe_router_aux([[0.0, 1.0]], coefficient)


@pytest.mark.parametrize(
    "logits",
    [[], [[]], [[0.0, 1.0], [2.0]], [[0.0, math.inf]], [[0.0, math.nan]]],
)
def test_invalid_logits_fail_closed(logits: list[list[float]]) -> None:
    with pytest.raises(ValueError):
        standard_moe_router_aux(logits, 0.01)


def test_native_tile_abi_implements_loss_and_accumulating_jacobian_gradient() -> None:
    header = (ROOT / "neuralfn/csrc/native_train/tile_ops.h").read_text(encoding="utf-8")
    wrapper_source = (ROOT / "neuralfn/csrc/native_train/tile_ops.cu").read_text(encoding="utf-8")
    kernel_source = (ROOT / "neuralfn/csrc/tile_cuda/kernels.cu").read_text(encoding="utf-8")
    symbol = "nfn_native_tile_moe_router_aux_loss_backward_float32"

    assert symbol in header
    wrapper_start = wrapper_source.index(f"int {symbol}(")
    wrapper_end = wrapper_source.index("\n}\n", wrapper_start) + 3
    wrapper = wrapper_source[wrapper_start:wrapper_end]
    assert "!std::isfinite(coefficient) || coefficient < 0.0f" in wrapper
    assert wrapper.index("if (coefficient == 0.0f)") < wrapper.index("router_logits == nullptr")
    assert "launch_moe_router_aux_loss_backward_float32" in wrapper

    compact_kernel = " ".join(kernel_source.split())
    assert "moe_router_aux_density_float32_kernel" in kernel_source
    assert "moe_router_aux_loss_accumulate_float32_kernel" in kernel_source
    assert "moe_router_aux_backward_float32_kernel" in kernel_source
    assert "row_max = fmaxf(row_max, router_logits[base + col])" in compact_kernel
    assert "probability_sum / static_cast<float>(rows)" in compact_kernel
    assert "*weighted_loss_accumulator += coefficient * static_cast<float>(experts) * squared_density_sum" in compact_kernel
    assert "2.0f * coefficient * static_cast<float>(experts) / static_cast<float>(rows)" in compact_kernel
    assert "grad_router_logits[base + expert] +=" in compact_kernel
    launch = kernel_source.index("void launch_moe_router_aux_loss_backward_float32(")
    density_launch = kernel_source.index("moe_router_aux_density_float32_kernel<<<", launch)
    loss_launch = kernel_source.index("moe_router_aux_loss_accumulate_float32_kernel<<<", launch)
    backward_launch = kernel_source.index("moe_router_aux_backward_float32_kernel<<<", launch)
    assert density_launch < loss_launch < backward_launch


def test_full_family_path_loads_symbol_fail_closed_and_accumulates_before_router_backward() -> None:
    source = (ROOT / "neuralfn/csrc/native_train/missing_native_train.cpp").read_text(encoding="utf-8")
    symbol = "nfn_native_tile_moe_router_aux_loss_backward_float32"
    assert symbol in source
    assert "api->moe_router_aux_loss_backward == nullptr" in source
    assert "const std::int64_t rows = batch_size * seq_len;" in source
    assert source.count("api.moe_router_aux_loss_backward(") == 1
    assert 'allocate_float(&device_router_aux_loss_total, 1, "standard MoE router auxiliary loss total")' in source

    prepare_start = source.index("auto prepare_moe_route_gradients =")
    reverse_loop = source.index("for (std::int64_t reverse = layers_count - 1;", prepare_start)
    router_weight_backward = source.index('"full MoE router weight backward"', prepare_start)
    path = source[prepare_start:router_weight_backward]
    selected_backward = path.index('"full MoE selected route backward"')
    aux_backward = path.index('"full standard MoE all-expert router auxiliary loss/backward"')
    router_input_backward = path.index('"full MoE router input backward"')
    assert selected_backward < aux_backward < router_input_backward
    assert prepare_start < reverse_loop < router_weight_backward
    assert "prepare_moe_route_gradients(layer)" in path
    assert "const float route_scale = standard_router_aux\n                ? 0.0f" in path
    assert "layer.grad_route_logits" in path[path.index("api.moe_router_aux_loss_backward("):]

    assert "kNativeFamilyLossReportRouterAux = 6" in source
    assert "kNativeFamilyLossReportScalarCount = 7" in source
    assert "device_router_aux_loss_total" in source
    assert '"zero full-family loss reporting scalar vector"' in source
    assert "host_loss_reporting_totals[kNativeFamilyLossReportRouterAux]" in source


def test_cli_plan_provenance_and_build_symbol_contract_are_explicit() -> None:
    source = (ROOT / "neuralfn/csrc/native_train/missing_native_train.cpp").read_text(encoding="utf-8")
    build_script = (ROOT / "tools/build_native_missing_trainers.sh").read_text(encoding="utf-8")
    symbol = "nfn_native_tile_moe_router_aux_loss_backward_float32"

    assert "--router-aux-loss-coef" in source
    assert "--native-cuda-router-aux-loss-coef" in source
    assert "--router-aux-loss-coef must be finite and non-negative" in source
    assert "--router-aux-loss-coef must be representable as a nonzero finite float32 when positive" in source
    assert "static_cast<float>(cfg.router_aux_loss_coef)" in source
    assert "--router-aux-loss-coef is only supported by standard softmax-MoE native profiles" in source
    assert "!native_family_uses_standard_moe_router_aux_loss(cfg)" in source
    assert "router_aux_loss_coef_explicit = true" in source
    assert "if (!cfg.router_aux_loss_coef_explicit)" in source
    assert "!native_family_uses_auxfree_moe_balance(full_identity)" in source
    assert "legacy_standard_moe ? 0.01 : 0.0" in source
    assert '\\"router_aux_loss_coef\\"' in source
    assert "experts*sum(mean_over_tokens(softmax(router_logits))^2)" in source
    assert "last_production_losses.router" in source

    required_condition = (
        'if [[ "${model}" == "mixllama" || "${model}" == "moe-jepa-evo" '
        '|| "${model}" == "jamba" ]]'
    )
    assert required_condition in build_script
    assert f'symbols="${{symbols}},{symbol}"' in build_script


def test_production_checkpoint_writer_emits_strict_graph_bound_standard_moe_contract() -> None:
    source = (ROOT / "neuralfn/csrc/native_train/missing_native_train.cpp").read_text(
        encoding="utf-8"
    )

    assert "native_family_standard_moe_inference_candidate" in source
    candidate_start = source.index("bool native_family_standard_moe_inference_candidate(")
    candidate_end = source.index("\n}\n", candidate_start)
    candidate = source[candidate_start:candidate_end]
    assert "native_full_moe_geometry_build_enabled()" in candidate
    assert "native_full_geometry_build_enabled()" not in candidate
    assert "build_native_family_standard_moe_inference_tensors" in source
    assert "write_native_family_standard_moe_inference_contract_json" in source
    assert "neuralfn.native_family_standard_moe.inference_checkpoint" in source
    assert "neuralfn.native_family_standard_moe.f32.v1" in source
    assert 'preset == "mixllama" || preset == "mixllama-fast"' in source
    assert "!cfg.graph_file.empty() && cfg.layers_per_expert == 1" in source
    assert "validate_native_family_graph_provenance(cfg, error)" in source
    assert '{2, experts, model_dim, hidden_dim}' in source
    assert '{experts, hidden_dim, model_dim}' in source
    assert "out << std::setprecision(17);" in source
