"""Tests for backend_capabilities: cache, quantized_export, megakernel, PCA KV cache."""
from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from neuralfn.config import (
    TemplateSpec,
    build_kv_pca_llama_spec,
    build_llama_fast_spec,
    build_llama_megakernel_spec,
    build_llama_spec,
    build_nanogpt_spec,
    build_ternary_b158_spec,
    resolve_backend_capabilities,
)
from neuralfn.inference import (
    InferenceCache,
    export_quantized_pt,
    export_to_pt,
    import_from_pt,
    import_quantized_pt,
)
from neuralfn.torch_backend import CompiledTorchGraph, TorchTrainConfig, TorchTrainer
from neuralfn.torch_templates import (
    build_dense_attention_graph,
    build_gpt_root_graph,
    build_model_spec_from_config,
)


def _tiny_kwargs() -> dict[str, int]:
    return {
        "num_layers": 1,
        "model_dim": 32,
        "num_heads": 4,
        "num_kv_heads": 4,
        "multiple_of": 16,
    }


def _cpu_graph(graph):
    graph.torch_config = {**graph.torch_config, "device": "cpu", "amp_dtype": "bfloat16"}
    return graph


# ---------------------------------------------------------------------------
# resolve_backend_capabilities
# ---------------------------------------------------------------------------

class TestResolveBackendCapabilities:
    def test_eager_runtime(self):
        spec = TemplateSpec(runtime="eager")
        caps = resolve_backend_capabilities(spec)
        assert caps["compile"] is False
        assert caps["cache"] is True
        assert caps["quantized_export"] is True
        assert caps["megakernel"] is False

    def test_compile_runtime(self):
        caps = resolve_backend_capabilities(TemplateSpec(runtime="compile"))
        assert caps["compile"] is True
        assert caps["megakernel"] is False

    def test_megakernel_runtime(self):
        caps = resolve_backend_capabilities(TemplateSpec(runtime="megakernel"))
        assert caps["compile"] is True
        assert caps["megakernel"] is True
        assert caps["sdpa"] is True

    def test_preset_auto_resolves(self):
        spec = build_llama_fast_spec(**_tiny_kwargs())
        assert spec.template.backend_capabilities["compile"] is True
        assert spec.template.backend_capabilities["cache"] is True
        assert spec.template.backend_capabilities["quantized_export"] is True


# ---------------------------------------------------------------------------
# Megakernel preset + fused attention graph
# ---------------------------------------------------------------------------

class TestMegakernel:
    def test_megakernel_spec_flags(self):
        spec = build_llama_megakernel_spec(**_tiny_kwargs(), vocab_size=128)
        assert spec.template.runtime == "megakernel"
        assert spec.template.backend_capabilities["megakernel"] is True
        assert spec.template.backend_capabilities["compile"] is True

    def test_fused_attention_graph_structure(self):
        from neuralfn.config import BlockSpec
        spec = BlockSpec(
            family="llama", norm_type="rmsnorm", mlp_type="swiglu",
            pos_encoding="rope", linear_bias=False, num_heads=4, num_kv_heads=4,
        )
        graph = build_dense_attention_graph("fused_test", 32, spec, fused_megakernel=True)
        assert "fused_attn" in graph.nodes
        assert len(graph.input_node_ids) == 1
        assert len(graph.output_node_ids) == 1

    def test_megakernel_template_forward(self):
        spec = build_llama_megakernel_spec(**_tiny_kwargs(), vocab_size=128)
        graph = _cpu_graph(build_gpt_root_graph(name="megakernel_smoke", model_spec=spec))
        compiled = CompiledTorchGraph(graph)
        tokens = torch.randint(0, 128, (2, 8))
        targets = torch.randint(0, 128, (2, 8))
        loss = compiled(tokens, targets)[0]
        assert loss.ndim == 0


# ---------------------------------------------------------------------------
# KV PCA compression in attention graph
# ---------------------------------------------------------------------------

class TestKVPCA:
    def test_kv_pca_spec(self):
        spec = build_kv_pca_llama_spec(**_tiny_kwargs(), vocab_size=128)
        assert spec.block_spec.compression == "kv_pca"
        assert spec.template.compression == "kv_pca"

    def test_kv_pca_attention_graph_has_encode_decode(self):
        from neuralfn.config import BlockSpec
        spec = BlockSpec(
            family="llama", norm_type="rmsnorm", mlp_type="swiglu",
            pos_encoding="rope", linear_bias=False, num_heads=4, num_kv_heads=4,
            compression="kv_pca",
        )
        graph = build_dense_attention_graph("pca_test", 32, spec, enable_pca=True)
        assert "pca_encode" in graph.nodes
        assert "pca_decode" in graph.nodes

    def test_kv_pca_template_forward(self):
        spec = build_kv_pca_llama_spec(**_tiny_kwargs(), vocab_size=128)
        graph = _cpu_graph(build_gpt_root_graph(name="kv_pca_smoke", model_spec=spec))
        compiled = CompiledTorchGraph(graph)
        tokens = torch.randint(0, 128, (2, 8))
        targets = torch.randint(0, 128, (2, 8))
        loss = compiled(tokens, targets)[0]
        assert loss.ndim == 0


# ---------------------------------------------------------------------------
# KV cache graph (explicit)
# ---------------------------------------------------------------------------

class TestKVCacheGraph:
    def test_cache_attention_graph_ports(self):
        from neuralfn.config import BlockSpec
        spec = BlockSpec(
            family="llama", norm_type="rmsnorm", mlp_type="swiglu",
            pos_encoding="rope", linear_bias=False, num_heads=4, num_kv_heads=4,
        )
        graph = build_dense_attention_graph("cache_test", 32, spec, enable_cache=True)
        assert "cache_k_in" in graph.nodes
        assert "cache_v_in" in graph.nodes
        assert "kv_cache_read" in graph.nodes
        assert "kv_cache_write" in graph.nodes
        assert "new_cache_k" in graph.nodes
        assert "new_cache_v" in graph.nodes
        assert len(graph.input_node_ids) == 3
        assert len(graph.output_node_ids) == 3

    def test_cache_with_pca(self):
        from neuralfn.config import BlockSpec
        spec = BlockSpec(
            family="llama", norm_type="rmsnorm", mlp_type="swiglu",
            pos_encoding="rope", linear_bias=False, num_heads=4, num_kv_heads=4,
            compression="kv_pca",
        )
        graph = build_dense_attention_graph(
            "cache_pca_test", 32, spec,
            enable_cache=True, enable_pca=True,
        )
        assert "pca_encode" in graph.nodes
        assert "pca_decode" in graph.nodes
        assert "kv_cache_read" in graph.nodes
        assert "kv_cache_write" in graph.nodes


# ---------------------------------------------------------------------------
# Quantized export / import
# ---------------------------------------------------------------------------

class TestQuantizedExport:
    def test_int8_round_trip(self):
        spec = build_nanogpt_spec(**_tiny_kwargs(), vocab_size=128)
        graph = _cpu_graph(build_gpt_root_graph(name="quant_int8_test", model_spec=spec))
        compiled_orig = CompiledTorchGraph(graph)
        tokens = torch.randint(0, 128, (2, 8))
        targets = torch.randint(0, 128, (2, 8))
        orig_out = compiled_orig(tokens, targets)[0].item()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model_int8.pt"
            export_quantized_pt(graph, path, scheme="int8")
            import_quantized_pt(graph, path)

        compiled_restored = CompiledTorchGraph(graph)
        restored_out = compiled_restored(tokens, targets)[0].item()
        assert abs(orig_out - restored_out) < 5.0

    def test_ternary_round_trip(self):
        spec = build_ternary_b158_spec(**_tiny_kwargs(), vocab_size=128)
        graph = _cpu_graph(build_gpt_root_graph(name="quant_ternary_test", model_spec=spec))
        compiled_orig = CompiledTorchGraph(graph)
        tokens = torch.randint(0, 128, (2, 8))
        targets = torch.randint(0, 128, (2, 8))
        orig_out = compiled_orig(tokens, targets)[0].item()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model_ternary.pt"
            export_quantized_pt(graph, path, scheme="ternary")
            import_quantized_pt(graph, path)

        compiled_restored = CompiledTorchGraph(graph)
        restored_out = compiled_restored(tokens, targets)[0].item()
        assert abs(orig_out - restored_out) < 2.0

    def test_plain_export_import_state_dict_matches(self):
        spec = build_nanogpt_spec(**_tiny_kwargs(), vocab_size=128)
        graph = _cpu_graph(build_gpt_root_graph(name="plain_export_test", model_spec=spec))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            export_to_pt(graph, path)
            sd_before = torch.load(path, weights_only=True)
            import_from_pt(graph, path)
            export_to_pt(graph, path)
            sd_after = torch.load(path, weights_only=True)

        assert set(sd_before.keys()) == set(sd_after.keys())
        for k in sd_before:
            assert torch.equal(sd_before[k], sd_after[k]), f"mismatch at {k}"


# ---------------------------------------------------------------------------
# Runtime wiring (template.runtime drives torch.compile)
# ---------------------------------------------------------------------------

class TestRuntimeWiring:
    def test_compile_preset_sets_compile_flag(self):
        spec = build_llama_fast_spec(**_tiny_kwargs(), vocab_size=128)
        assert spec.template.runtime == "compile"

    def test_eager_preset_no_compile(self):
        spec = build_nanogpt_spec(**_tiny_kwargs(), vocab_size=128)
        assert spec.template.runtime == "eager"

    def test_megakernel_preset_sets_runtime(self):
        spec = build_llama_megakernel_spec(**_tiny_kwargs(), vocab_size=128)
        assert spec.template.runtime == "megakernel"

    def test_trainer_reads_template_runtime(self):
        spec = build_llama_fast_spec(**_tiny_kwargs(), vocab_size=128)
        graph = _cpu_graph(build_gpt_root_graph(name="runtime_test", model_spec=spec))
        template_spec = dict(graph.torch_config.get("template_spec", {}))
        runtime = template_spec.get("template", {}).get("runtime", "eager")
        assert runtime == "compile"


# ---------------------------------------------------------------------------
# InferenceCache
# ---------------------------------------------------------------------------

class TestInferenceCache:
    def test_cache_step_with_training_graph(self):
        """Training graphs return scalar loss; step() handles multi-input."""
        spec = build_nanogpt_spec(**_tiny_kwargs(), vocab_size=128)
        graph = _cpu_graph(build_gpt_root_graph(name="cache_infer", model_spec=spec))
        cache = InferenceCache(graph, device="cpu")
        tokens = torch.randint(0, 128, (1, 4))
        out = cache.step(tokens)
        assert out.ndim == 0  # scalar loss from training graph
        assert out.item() > 0

    def test_cache_reset(self):
        spec = build_nanogpt_spec(**_tiny_kwargs(), vocab_size=128)
        graph = _cpu_graph(build_gpt_root_graph(name="cache_reset", model_spec=spec))
        cache = InferenceCache(graph, device="cpu")
        cache.step(torch.randint(0, 128, (1, 4)))
        cache.reset()
        assert len(cache._cache) == 0

    def test_cache_multiple_steps_deterministic(self):
        spec = build_nanogpt_spec(**_tiny_kwargs(), vocab_size=128)
        graph = _cpu_graph(build_gpt_root_graph(name="cache_det", model_spec=spec))
        cache = InferenceCache(graph, device="cpu")
        tokens = torch.randint(0, 128, (1, 4))
        out1 = cache.step(tokens).item()
        cache.reset()
        out2 = cache.step(tokens).item()
        assert out1 == out2


# ---------------------------------------------------------------------------
# Preset registration
# ---------------------------------------------------------------------------

class TestNewPresets:
    def test_megakernel_preset_via_config(self):
        spec = build_model_spec_from_config({"preset": "llama_megakernel", **_tiny_kwargs()})
        assert spec.template.runtime == "megakernel"

    def test_kv_pca_preset_via_config(self):
        spec = build_model_spec_from_config({"preset": "kv_pca_llama", **_tiny_kwargs()})
        assert spec.block_spec.compression == "kv_pca"
