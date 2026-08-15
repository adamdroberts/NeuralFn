"""Per-Stage parity / correctness tests for the frontier-template kernels.

These complement the build+forward coverage in test_template_presets.py by
checking the numerical behaviour of each new PyTorch-reference Stage (the
contract each will preserve when later repointed at an llm.kittens kernel).
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from neuralfn.torch_backend import (
    DifferentialAttentionStage,
    DyTStage,
    GLUStage,
    GroupNormStage,
    ManifoldHyperConnectionStage,
    MLAStage,
    QKNormStage,
    Rotary,
    build_module,
    _sparse_attn_mask,
)


_SPARSE_REFERENCE_CASES = (
    (
        "sliding_window_attention",
        {"window_size": 4, "is_causal": True, "dropout_p": 0.0},
        {"window": 4, "num_sinks": 0, "block": None, "compress_stride": None},
    ),
    (
        "block_sparse_attention",
        {"sparse_block_size": 4, "num_sinks": 2, "is_causal": True, "dropout_p": 0.0},
        {"window": None, "num_sinks": 2, "block": 4, "compress_stride": None},
    ),
    (
        "streaming_attention_sinks",
        {"window_size": 4, "num_sinks": 2, "is_causal": True, "dropout_p": 0.0},
        {"window": 4, "num_sinks": 2, "block": None, "compress_stride": None},
    ),
    (
        "native_sparse_attention",
        {
            "window_size": 4,
            "num_sinks": 2,
            "compress_stride": 3,
            "is_causal": True,
            "dropout_p": 0.0,
        },
        {"window": 4, "num_sinks": 2, "block": None, "compress_stride": 3},
    ),
)


def _right_aligned_allowed_key_indices(
    key_count: int,
    mask_kwargs: dict[str, int | None],
) -> tuple[int, ...]:
    """Return visible keys for one query aligned to the end of its key prefix."""

    mask = _sparse_attn_mask(
        1,
        key_count,
        is_causal=True,
        device="cpu",
        dtype=torch.float32,
        **mask_kwargs,
    )
    return tuple(torch.nonzero(torch.isfinite(mask[0]), as_tuple=False).flatten().tolist())


def test_g1_norm_and_gate_shapes_and_grads() -> None:
    x = torch.randn(2, 6, 16, requires_grad=True)
    for name, cfg in [
        ("geglu", {"model_dim": 16, "mlp_mult": 4, "multiple_of": 8}),
        ("reglu", {"model_dim": 16, "mlp_mult": 4, "multiple_of": 8}),
        ("solu", {"model_dim": 16, "mlp_mult": 4, "multiple_of": 8}),
        ("dyt", {"model_dim": 16, "alpha_init": 0.7}),
        ("group_norm", {"model_dim": 16, "num_groups": 4}),
    ]:
        y = build_module(name, cfg)(x)
        assert y.shape == x.shape
        y.sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        x.grad = None


def test_dyt_matches_formula() -> None:
    stage = DyTStage(model_dim=8, alpha_init=0.5)
    x = torch.randn(3, 4, 8)
    expected = stage.weight * torch.tanh(stage.alpha * x) + stage.bias
    assert torch.allclose(stage(x), expected, atol=1e-6)


def test_qk_norm_is_unit_rms_over_head_dim() -> None:
    q = torch.randn(2, 4, 5, 8) * 3.0
    k = torch.randn(2, 2, 5, 8) * 3.0
    qn, kn = QKNormStage()(q, k)
    # RMS over the last dim should be ~1
    assert torch.allclose(qn.pow(2).mean(-1).sqrt(), torch.ones(2, 4, 5), atol=1e-3)
    assert torch.allclose(kn.pow(2).mean(-1).sqrt(), torch.ones(2, 2, 5), atol=1e-3)


def test_sparse_masks_have_no_dead_rows_and_respect_causality() -> None:
    for kw in [
        dict(window=4, num_sinks=0, block=None, compress_stride=None),
        dict(window=None, num_sinks=2, block=3, compress_stride=None),
        dict(window=4, num_sinks=2, block=None, compress_stride=4),
    ]:
        m = _sparse_attn_mask(12, 12, is_causal=True, device="cpu", dtype=torch.float32, **kw)
        # every query can attend to at least itself
        assert torch.isfinite(m).any(dim=1).all()
        # strictly-future positions are always masked
        i = torch.arange(12).unsqueeze(1)
        j = torch.arange(12).unsqueeze(0)
        assert torch.isinf(m[(j > i)]).all()


def test_sliding_window_matches_manual_masked_sdpa() -> None:
    torch.manual_seed(0)
    q = torch.randn(1, 2, 8, 8)
    k = torch.randn(1, 2, 8, 8)
    v = torch.randn(1, 2, 8, 8)
    window = 3
    stage = build_module("sliding_window_attention", {"window_size": window})
    got = stage(q, k, v)
    # manual reference
    i = torch.arange(8).unsqueeze(1)
    j = torch.arange(8).unsqueeze(0)
    allowed = (j <= i) & (j > i - window)
    mask = torch.zeros(8, 8).masked_fill(~allowed, float("-inf"))
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
    assert torch.allclose(got, ref, atol=1e-5)


@pytest.mark.parametrize(
    ("module_type", "module_config", "mask_kwargs"),
    _SPARSE_REFERENCE_CASES,
    ids=("sliding-window", "block-sparse", "streaming-sinks", "nsa"),
)
def test_right_aligned_sparse_gqa_chunks_match_full_prefix_recompute_slices(
    module_type: str,
    module_config: dict[str, object],
    mask_kwargs: dict[str, int | None],
) -> None:
    del mask_kwargs  # Geometry is included in the parametrization for the companion tests.
    torch.manual_seed(7)
    token_count = 19
    query = torch.randn(1, 4, token_count, 4)
    key = torch.randn(1, 2, token_count, 4)
    value = torch.randn(1, 2, token_count, 4)
    stage = build_module(module_type, module_config).eval()

    # These are cache-shaped inputs, not a resident cache: each query chunk is
    # right-aligned to the complete K/V prefix available at that point.
    for _region, start, end in (
        ("left", 0, 3),
        ("middle", 5, 10),
        ("right", 15, 19),
    ):
        recomputed_prefix = stage(
            query[:, :, :end],
            key[:, :, :end],
            value[:, :, :end],
        )
        chunk = stage(
            query[:, :, start:end],
            key[:, :, :end],
            value[:, :, :end],
        )
        torch.testing.assert_close(
            chunk,
            recomputed_prefix[:, :, start:end],
            rtol=1.0e-5,
            atol=1.0e-6,
        )


def test_sparse_masks_select_exact_window_block_and_partial_boundary_keys() -> None:
    window = {"window": 4, "num_sinks": 0, "block": None, "compress_stride": None}
    block = {"window": None, "num_sinks": 2, "block": 4, "compress_stride": None}
    streaming = {"window": 4, "num_sinks": 2, "block": None, "compress_stride": None}
    nsa = {"window": 4, "num_sinks": 2, "block": None, "compress_stride": 3}
    cases = (
        ("window-filled", 4, window, (0, 1, 2, 3)),
        ("window-first-shift", 5, window, (1, 2, 3, 4)),
        ("block-last-row", 8, block, (0, 1, 4, 5, 6, 7)),
        ("block-first-next-row", 9, block, (0, 1, 8)),
        ("block-partial-right-edge", 19, block, (0, 1, 16, 17, 18)),
        ("streaming-first-old-non-sink-exclusion", 7, streaming, (0, 1, 3, 4, 5, 6)),
        ("nsa-window-sink-stride-union", 11, nsa, (0, 1, 3, 6, 7, 8, 9, 10)),
    )

    for boundary, key_count, mask_kwargs, expected in cases:
        assert _right_aligned_allowed_key_indices(key_count, mask_kwargs) == expected, boundary


@pytest.mark.parametrize(
    ("module_type", "module_config", "mask_kwargs"),
    _SPARSE_REFERENCE_CASES,
    ids=("sliding-window", "block-sparse", "streaming-sinks", "nsa"),
)
def test_sparse_attention_ignores_excluded_old_kv_and_uses_allowed_values(
    module_type: str,
    module_config: dict[str, object],
    mask_kwargs: dict[str, int | None],
) -> None:
    token_count = 19
    query = torch.ones(1, 4, 1, 4)
    key = torch.zeros(1, 2, token_count, 4)
    value = torch.linspace(-1.0, 1.0, 2 * token_count * 4).reshape(1, 2, token_count, 4)
    stage = build_module(module_type, module_config).eval()
    baseline = stage(query, key, value)

    allowed = _right_aligned_allowed_key_indices(token_count, mask_kwargs)
    excluded = tuple(index for index in range(token_count) if index not in set(allowed))
    assert excluded

    excluded_key = key.clone()
    excluded_value = value.clone()
    excluded_key[:, :, excluded, :] += 100.0
    excluded_value[:, :, excluded, :] -= 100.0
    torch.testing.assert_close(
        stage(query, excluded_key, excluded_value),
        baseline,
        rtol=0.0,
        atol=0.0,
    )

    allowed_value = value.clone()
    allowed_value[:, :, allowed[-1], :] += 10.0
    sensitive = stage(query, key, allowed_value)
    assert torch.max(torch.abs(sensitive - baseline)).item() > 1.0e-3


def test_sparse_mask_long_history_cardinality_bounds_and_nsa_growth() -> None:
    window_size = 128
    num_sinks = 4
    block_size = 64
    compress_stride = 16
    nsa_counts: list[int] = []

    for history in (257, 769):
        recent = set(range(history - window_size, history))
        sinks = set(range(num_sinks))
        block_start = ((history - 1) // block_size) * block_size
        current_block = set(range(block_start, history))
        compressed = set(range(0, history, compress_stride))

        sliding_allowed = set(_right_aligned_allowed_key_indices(
            history,
            {"window": window_size, "num_sinks": 0, "block": None, "compress_stride": None},
        ))
        streaming_allowed = set(_right_aligned_allowed_key_indices(
            history,
            {"window": window_size, "num_sinks": num_sinks, "block": None, "compress_stride": None},
        ))
        block_allowed = set(_right_aligned_allowed_key_indices(
            history,
            {"window": None, "num_sinks": num_sinks, "block": block_size, "compress_stride": None},
        ))
        nsa_allowed = set(_right_aligned_allowed_key_indices(
            history,
            {
                "window": window_size,
                "num_sinks": num_sinks,
                "block": None,
                "compress_stride": compress_stride,
            },
        ))

        assert sliding_allowed == recent
        assert streaming_allowed == recent | sinks
        assert block_allowed == current_block | sinks
        assert nsa_allowed == recent | sinks | compressed
        assert len(sliding_allowed) == window_size
        assert len(streaming_allowed) <= window_size + num_sinks
        assert len(block_allowed) <= block_size + num_sinks
        nsa_counts.append(len(nsa_allowed))

    # NSA's strided-history rule deliberately grows its visible-key set as the
    # prefix grows. This is a logical mask bound, not K/V storage or eviction.
    assert nsa_counts == [140, 172]
    assert nsa_counts[1] > nsa_counts[0]


def test_differential_attention_shape_grad_and_even_head_dim() -> None:
    q = torch.randn(2, 4, 6, 8, requires_grad=True)
    k = torch.randn(2, 4, 6, 8)
    v = torch.randn(2, 4, 6, 8)
    stage = DifferentialAttentionStage(lambda_init=0.8)
    out = stage(q, k, v)
    assert out.shape == (2, 4, 6, 8)
    out.sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    # odd head_dim must raise
    try:
        stage(torch.randn(1, 1, 4, 7), torch.randn(1, 1, 4, 7), torch.randn(1, 1, 4, 7))
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_fp8_linear_roundtrip_bounded() -> None:
    stage = build_module("fp8_linear", {"input_dim": 64, "output_dim": 32, "fp8_format": "e4m3"})
    rel = (stage.weight - stage._quant_weight(stage.weight)).norm() / stage.weight.norm()
    assert rel < 0.1  # E4M3 weight-only quant on init weights
    x = torch.randn(2, 5, 64, requires_grad=True)
    y = stage(x)
    y.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_mxfp4_linear_uses_fp4_grid_and_flows_grads() -> None:
    stage = build_module("mx_linear", {"input_dim": 64, "output_dim": 32, "mx_format": "mxfp4", "mx_block_size": 32})
    x = torch.randn(2, 5, 64, requires_grad=True)
    y = stage(x)
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_auxfree_bias_reduces_load_imbalance() -> None:
    torch.manual_seed(0)
    experts, top_k = 8, 2
    stage = build_module("auxfree_load_balancing", {"experts": experts, "top_k": top_k, "bias_lr": 0.1})
    stage.train()
    # varied per-token logits with expert 0 systematically over-preferred
    logits = torch.randn(512, experts)
    logits[:, 0] += 3.0

    def load_entropy(lg: torch.Tensor) -> float:
        _, idx = torch.topk(lg, top_k, dim=-1)
        counts = torch.bincount(idx.reshape(-1), minlength=experts).float()
        p = counts / counts.sum()
        p = p[p > 0]
        return float(-(p * p.log()).sum())

    before = load_entropy(logits)
    for _ in range(300):
        stage(logits)
    # the over-loaded expert 0 should receive the lowest (most negative) bias
    assert stage.expert_bias.argmin().item() == 0
    # applying the learned bias spreads the load (higher routing entropy)
    assert load_entropy(logits + stage.expert_bias) > before


def test_mhc_is_non_expansive_and_starts_near_residual() -> None:
    stage = ManifoldHyperConnectionStage(dim=16, beta_init=0.05)
    beta = torch.sigmoid(stage.beta_logit)
    coeff_norm = (beta ** 2 + (1 - beta ** 2)).sqrt()
    assert torch.allclose(coeff_norm, torch.ones_like(coeff_norm), atol=1e-5)
    r = torch.randn(2, 4, 16)
    d = torch.randn(2, 4, 16)
    out = stage(r, d)
    # small beta -> output dominated by residual
    assert (out - r).norm() < (out - d).norm()


def test_mla_forward_backward_and_output_dim() -> None:
    stage = MLAStage(model_dim=32, num_heads=4)
    x = torch.randn(2, 8, 32, requires_grad=True)
    y = stage(x)
    assert y.shape == (2, 8, 32)
    y.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_rope_scaling_variants_finite_and_shaped() -> None:
    for scaling in [
        None,
        {"type": "linear", "factor": 4.0},
        {"type": "ntk", "factor": 4.0},
        {"type": "yarn", "factor": 4.0, "original_max_position": 2048},
    ]:
        rot = Rotary(16, 10000.0, scaling=scaling)
        cos, sin = rot(8, torch.device("cpu"), torch.float32)
        assert cos.shape == (1, 1, 8, 8) and sin.shape == (1, 1, 8, 8)
        assert torch.isfinite(cos).all() and torch.isfinite(sin).all()
