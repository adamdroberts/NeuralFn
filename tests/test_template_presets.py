from __future__ import annotations

import ast
from copy import deepcopy
import inspect
import importlib.util
import json
import os
import subprocess
import textwrap
import uuid
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from neuralfn.graph import Edge, NeuronGraph, NeuronInstance
from neuralfn.neuron import neuron_from_source, subgraph_neuron
from neuralfn.port import Port
from neuralfn.config import (
    MODERN_BASE_PRESETS,
    SHIPPED_GPT_TEMPLATE_BASE_PRESETS,
    SHIPPED_GPT_TEMPLATE_PRESETS,
    build_hnet_lm_spec,
    build_llm_jepa_spec,
    build_muse_glimmer_spec,
    build_ttt_llama_spec,
    build_universal_llama_spec,
)
import neuralfn.torch_templates as torch_templates
from neuralfn.torch_backend import (
    CompiledTorchGraph,
    DFlashAttentionStage,
    JEPAMaskStage,
    RMSNormStage,
    TensorScaleStage,
    TorchTrainConfig,
    TorchTrainer,
    MuseGlimmerVisionTowerStage,
)
from neuralfn.torch_templates import build_gpt_root_graph, build_gpt_template_payload, build_model_spec_from_config, make_terminal_def
import server.dataset_manager as dataset_manager
from server.dataset_manager import load_dataset_bytes
from server.models import ExecuteRequest, GPTTemplateRequest, LoadDatasetRequest
from server.services.graph_ops import apply_gpt_template, load_dataset_source_into_graph, trace_torch_graph
from neuralfn.native_gpt import build_native_gpt_compiled_cli_run_config
from neuralfn.native_gpt2 import build_native_gpt2_compiled_cli_run_config


ROOT = Path(__file__).resolve().parents[1]

PRESETS = list(SHIPPED_GPT_TEMPLATE_PRESETS)


def _builder_dispatch_presets() -> set[str]:
    source = textwrap.dedent(inspect.getsource(torch_templates.build_model_spec_from_config))
    tree = ast.parse(source)
    presets: set[str] = set()

    def collect_strings(node: ast.AST) -> set[str]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return {node.value}
        if isinstance(node, (ast.Set, ast.Tuple, ast.List)):
            values: set[str] = set()
            for elt in node.elts:
                values.update(collect_strings(elt))
            return values
        return set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        if not isinstance(node.left, ast.Name) or node.left.id != "preset":
            continue
        for op, comparator in zip(node.ops, node.comparators):
            if isinstance(op, (ast.Eq, ast.In)):
                presets.update(collect_strings(comparator))
    return presets


def test_shipped_gpt_template_catalog_matches_builder_dispatch() -> None:
    assert set(SHIPPED_GPT_TEMPLATE_BASE_PRESETS) == _builder_dispatch_presets()
    assert set(SHIPPED_GPT_TEMPLATE_PRESETS) == {
        *SHIPPED_GPT_TEMPLATE_BASE_PRESETS,
        *(f"{preset}_modern" for preset in MODERN_BASE_PRESETS),
    }


def test_native_gpt_template_catalog_header_matches_python_catalog() -> None:
    header = (ROOT / "neuralfn" / "csrc" / "native_train" / "shipped_gpt_template_presets.h").read_text(
        encoding="utf-8"
    )
    for preset in SHIPPED_GPT_TEMPLATE_PRESETS:
        assert f'"{preset}"' in header
    assert header.count('",') == len(SHIPPED_GPT_TEMPLATE_PRESETS)


def _cpu_graph(graph):
    graph.torch_config = {
        **graph.torch_config,
        "device": "cpu",
        "amp_dtype": "bfloat16",
    }
    return graph


def _tiny_kwargs() -> dict[str, int]:
    return {
        "num_layers": 1,
        "model_dim": 32,
        "num_heads": 4,
        "num_kv_heads": 4,
        "multiple_of": 16,
    }


def _module_configs(value: object, module_type: str) -> list[dict[str, object]]:
    configs: list[dict[str, object]] = []
    if isinstance(value, dict):
        if value.get("module_type") == module_type:
            config = value.get("module_config")
            assert isinstance(config, dict)
            configs.append(config)
        for child in value.values():
            configs.extend(_module_configs(child, module_type))
    elif isinstance(value, list):
        for child in value:
            configs.extend(_module_configs(child, module_type))
    return configs


def _make_terminal_def(role: str, port_name: str):
    source = f"def {role}(x):\n    return x\n"
    ports = [Port(port_name, range=(-1_000_000.0, 1_000_000.0), precision=0.001, dtype="float")]
    return neuron_from_source(source, role, ports, ports)


def _make_variant_graph(name: str) -> NeuronGraph:
    graph = NeuronGraph(name=name)
    graph.add_node(NeuronInstance(_make_terminal_def("input", "x"), instance_id="x_in", position=(0, 0)))
    graph.add_node(NeuronInstance(_make_terminal_def("output", "x"), instance_id="x_out", position=(200, 0)))
    graph.add_edge(Edge(src_node="x_in", src_port=0, dst_node="x_out", dst_port=0))
    graph.input_node_ids = ["x_in"]
    graph.output_node_ids = ["x_out"]
    return graph


def _make_alias_root(link_family: str, available_family: str) -> NeuronGraph:
    variant_graph = _make_variant_graph(f"{available_family}_default")
    root = NeuronGraph(name=f"{link_family}_root", variant_library={available_family: {"default": variant_graph}})
    root.add_node(NeuronInstance(_make_terminal_def("input", "x"), instance_id="x_in", position=(0, 0)))
    root.add_node(
        NeuronInstance(
            subgraph_neuron(
                variant_graph,
                name="block",
                input_aliases=["x"],
                output_aliases=["x"],
                variant_ref={"family": link_family, "version": "default"},
            ),
            instance_id="block",
            position=(200, 0),
        )
    )
    root.add_node(NeuronInstance(_make_terminal_def("output", "x"), instance_id="x_out", position=(400, 0)))
    root.add_edge(Edge(src_node="x_in", src_port=0, dst_node="block", dst_port=0))
    root.add_edge(Edge(src_node="block", src_port=0, dst_node="x_out", dst_port=0))
    root.input_node_ids = ["x_in"]
    root.output_node_ids = ["x_out"]
    return root


def _load_train_gpt2_script_module():
    script = ROOT / "cli" / "scripts" / "train_gpt2.py"
    spec = importlib.util.spec_from_file_location("train_gpt2_template_pass_through_test", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_train_gpt_script_module():
    script = ROOT / "cli" / "scripts" / "train_gpt.py"
    spec = importlib.util.spec_from_file_location("train_gpt_template_pass_through_test", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_gpt2_compiled_cli_accepts_every_gpt_template_name() -> None:
    for preset in PRESETS:
        config = build_native_gpt2_compiled_cli_run_config(
            dataset_alias="/tmp/native-cache",
            executable="/bin/echo",
            output_dir=Path("/tmp/native-output"),
            eval_every_steps=1000,
            sample_every_steps=20000,
            generate_tokens=144,
            checkpoint_every_steps=200,
            batch_size=64,
            seq_len=1024,
            train_batch_tokens=524288,
            learning_rate=0.0006,
            min_lr=None,
            warmup_steps=60,
            weight_decay=0.1,
            max_steps=20000,
            num_layers=12,
            activation="gelu",
            template_name=preset,
        )
        argv = config.compiled_cli_argv(cli="/tmp/nfn_gpt2_native_train")
        assert "--template-name" in argv
        assert argv[argv.index("--template-name") + 1] == preset


def test_native_gpt_compiled_cli_alias_accepts_every_gpt_template_name() -> None:
    for preset in PRESETS:
        config = build_native_gpt_compiled_cli_run_config(
            dataset_alias="/tmp/native-cache",
            executable="/bin/echo",
            output_dir=Path("/tmp/native-output"),
            eval_every_steps=1000,
            sample_every_steps=20000,
            generate_tokens=144,
            checkpoint_every_steps=200,
            batch_size=64,
            seq_len=1024,
            train_batch_tokens=524288,
            learning_rate=0.0006,
            min_lr=None,
            warmup_steps=60,
            weight_decay=0.1,
            max_steps=20000,
            num_layers=12,
            activation="gelu",
            template_name=preset,
        )
        argv = config.compiled_cli_argv(cli="/tmp/nfn_gpt2_native_train")
        assert "--template-name" in argv
        assert argv[argv.index("--template-name") + 1] == preset


def test_native_gpt_compiled_cli_serializes_strict_lm_head_requirement() -> None:
    config = build_native_gpt_compiled_cli_run_config(
        dataset_alias="/tmp/native-cache",
        executable="/bin/echo",
        output_dir=Path("/tmp/native-output"),
        eval_every_steps=1000,
        sample_every_steps=20000,
        generate_tokens=144,
        checkpoint_every_steps=200,
        batch_size=64,
        seq_len=1024,
        train_batch_tokens=524288,
        learning_rate=0.0006,
        min_lr=None,
        warmup_steps=60,
        weight_decay=0.1,
        max_steps=20000,
        num_layers=12,
        activation="gelu",
        require_cooperative_lm_head_backward=True,
    )
    argv = config.compiled_cli_argv(cli="/tmp/nfn_gpt2_native_train")
    assert "--require-cooperative-lm-head-backward" in argv


def test_train_gpt2_fast_path_accepts_every_gpt_template_name() -> None:
    module = _load_train_gpt2_script_module()
    for preset in PRESETS:
        for selector in (
            ["--template-name", preset],
            [f"--template-name={preset}"],
            ["--template", preset],
            [f"--template={preset}"],
            ["--preset", preset],
            [f"--preset={preset}"],
        ):
            argv = module._fast_compiled_cli_argv(
                ["--dataset-alias", "/tmp/native-cache", "--native-cuda-dry-run", *selector]
            )
            assert argv is not None
            assert "--template-name" in argv
            assert argv[argv.index("--template-name") + 1] == preset


def test_train_gpt_fast_path_accepts_every_gpt_template_name() -> None:
    module = _load_train_gpt_script_module()
    for preset in PRESETS:
        argv = module._fast_compiled_cli_argv(
            ["--dataset-alias", "/tmp/native-cache", "--native-cuda-dry-run", "--template-name", preset]
        )
        assert argv is not None
        assert "--template-name" in argv
        assert argv[argv.index("--template-name") + 1] == preset
        if "--model-family" in argv:
            assert argv[argv.index("--model-family") + 1] == "gpt"
        else:
            assert Path(argv[0]).name.startswith("nfn_")


def test_compiled_gpt_launcher_accepts_every_shipped_template_name(tmp_path: Path) -> None:
    launcher = tmp_path / "nfn_train_gpt"
    build = subprocess.run(
        ["bash", str(ROOT / "tools" / "build_train_gpt_cli.sh"), str(launcher)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert build.returncode == 0, build.stderr

    fake_native = tmp_path / "fake-native"
    observed = tmp_path / "observed-argv.txt"
    fake_native.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\" > \"${NFN_TEST_NATIVE_GPT_ARGV}\"\n",
        encoding="utf-8",
    )
    fake_native.chmod(0o755)
    env = {
        **os.environ,
        "NFN_NATIVE_GPT_TRAIN_BIN": str(fake_native),
        "NFN_TEST_NATIVE_GPT_ARGV": str(observed),
        "CUDA_VISIBLE_DEVICES": "",
    }
    for preset in PRESETS:
        proc = subprocess.run(
            [str(launcher), "--base-model", preset, "--dataset-alias", "/tmp/native-cache", "--dry-run"],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert proc.returncode == 0, f"{preset}: {proc.stderr}"
        argv = observed.read_text(encoding="utf-8").splitlines()
        assert argv[argv.index("--template-name") + 1] == preset
        if "--model-family" in argv:
            expected_family = preset if preset in {"gpt2", "nanogpt"} else "gpt"
            assert argv[argv.index("--model-family") + 1] == expected_family
        else:
            assert Path(argv[0]).name.startswith("nfn_")


def test_train_gpt_fast_path_treats_auto_runner_as_compiled_cli() -> None:
    module = _load_train_gpt_script_module()
    argv = module._fast_compiled_cli_argv(
        [
            "--dataset-alias",
            "/tmp/native-cache",
            "--native-cuda-dry-run",
            "--native-cuda-runner",
            "auto",
        ]
    )
    assert argv is not None
    assert "--model-family" in argv
    assert argv[argv.index("--model-family") + 1] == "gpt"
    assert "--native-cuda-runner" not in argv


def test_train_gpt_fast_path_forwards_strict_lm_head_requirement() -> None:
    module = _load_train_gpt_script_module()
    for flag in ("--require-cooperative-lm-head-backward", "--native-cuda-require-cooperative-lm-head-backward"):
        argv = module._fast_compiled_cli_argv(["--dataset-alias", "/tmp/native-cache", "--native-cuda-dry-run", flag])
        assert argv is not None
        assert "--require-cooperative-lm-head-backward" in argv


def test_native_gpt2_compiled_cli_accepts_custom_graph_file() -> None:
    config = build_native_gpt2_compiled_cli_run_config(
        dataset_alias="/tmp/native-cache",
        executable="/bin/echo",
        output_dir=Path("/tmp/native-output"),
        eval_every_steps=1000,
        sample_every_steps=20000,
        generate_tokens=144,
        checkpoint_every_steps=200,
        batch_size=64,
        seq_len=1024,
        train_batch_tokens=524288,
        learning_rate=0.0006,
        min_lr=None,
        warmup_steps=60,
        weight_decay=0.1,
        max_steps=20000,
        num_layers=12,
        activation="gelu",
        graph_file="/tmp/custom-graph.json",
    )
    argv = config.compiled_cli_argv(cli="/tmp/nfn_gpt2_native_train")
    assert "--graph-file" in argv
    assert argv[argv.index("--graph-file") + 1] == "/tmp/custom-graph.json"


def test_train_gpt2_fast_path_accepts_custom_graph_file() -> None:
    module = _load_train_gpt2_script_module()
    for selector in (["--graph-file", "/tmp/custom-graph.json"], ["--graph-file=/tmp/custom-graph.json"]):
        argv = module._fast_compiled_cli_argv(["--dataset-alias", "/tmp/native-cache", "--native-cuda-dry-run", *selector])
        assert argv is not None
        assert "--graph-file" in argv
        assert argv[argv.index("--graph-file") + 1] == "/tmp/custom-graph.json"


def test_train_gpt_fast_path_accepts_custom_graph_file() -> None:
    module = _load_train_gpt_script_module()
    argv = module._fast_compiled_cli_argv(
        ["--dataset-alias", "/tmp/native-cache", "--native-cuda-dry-run", "--graph-file", "/tmp/custom-graph.json"]
    )
    assert argv is not None
    assert "--model-family" in argv
    assert argv[argv.index("--model-family") + 1] == "gpt"
    assert "--graph-file" in argv
    assert argv[argv.index("--graph-file") + 1] == "/tmp/custom-graph.json"


def test_build_gpt_template_payload_supports_all_presets() -> None:
    for preset in PRESETS:
        payload = build_gpt_template_payload(name=f"{preset}_payload", config={"preset": preset})
        assert payload["node_def"]["kind"] == "subgraph"
        assert isinstance(payload["variant_library"], dict)
        assert payload["graph_settings"]["torch_config"]["template_spec"]["template"]


def test_all_moe_preset_dispatch_configs_preserve_fractional_geometry() -> None:
    affected_presets: set[str] = set()
    config_override = {
        **_tiny_kwargs(),
        "mlp_multiplier": 2.5,
        "multiple_of": 24,
    }
    for preset in PRESETS:
        config = {"preset": preset, **config_override}
        spec = build_model_spec_from_config(config, preview_defaults=True)
        if spec.block_spec.mlp_type != "moe":
            continue

        affected_presets.add(preset)
        assert spec.block_spec.mlp_multiplier == 2.5
        assert spec.block_spec.multiple_of == 24
        payload = build_gpt_template_payload(name=f"{preset}_expert_geometry", config=config)
        dispatch_configs = _module_configs(payload, "expert_dispatch")
        assert dispatch_configs, preset
        for dispatch_config in dispatch_configs:
            assert dispatch_config["mlp_mult"] == 2.5, preset
            assert dispatch_config["multiple_of"] == 24, preset

    assert affected_presets


def test_standard_moe_presets_keep_unaligned_fractional_default() -> None:
    for preset in ("mixllama", "moe", "mixllama_fast"):
        config = {
            "preset": preset,
            "num_layers": 1,
            "model_dim": 5,
            "num_heads": 1,
            "num_kv_heads": 1,
        }
        spec = build_model_spec_from_config(config, preview_defaults=True)
        assert spec.block_spec.mlp_multiplier == 8.0 / 3.0
        assert spec.block_spec.multiple_of is None
        payload = build_gpt_template_payload(name=f"{preset}_default_expert_geometry", config=config)
        dispatch_configs = _module_configs(payload, "expert_dispatch")
        assert dispatch_configs, preset
        for dispatch_config in dispatch_configs:
            assert dispatch_config["mlp_mult"] == 8.0 / 3.0, preset
            assert dispatch_config["multiple_of"] is None, preset


def test_root_graph_defaults_to_float32_amp() -> None:
    graph = build_gpt_root_graph(name="float32_default")
    assert graph.torch_config["amp_dtype"] == "float32"


def test_muse_glimmer_production_spec_is_exact_and_capability_scoped() -> None:
    spec = build_muse_glimmer_spec()

    assert (spec.model_dim, spec.num_layers, spec.vocab_size) == (6_656, 52, 202_048)
    assert spec.tie_embeddings is False
    assert spec.max_position_embeddings == 131_072
    assert spec.output_multiplier == 0.19611613513818404
    assert spec.logit_softcap == 20.0
    block = spec.block_spec
    assert (block.num_heads, block.num_kv_heads, block.head_dim) == (32, 2, 128)
    assert block.attention_inner_dim == 4_096
    assert block.intermediate_size == 19_968
    assert [entry.kind for entry in block.layer_attention_pattern] == ["local", "local", "local", "full"]
    assert [entry.pos_encoding for entry in block.layer_attention_pattern] == ["rope", "rope", "rope", "none"]
    assert [entry.window_size for entry in block.layer_attention_pattern] == [2_048, 2_048, 2_048, None]
    assert block.qk_norm_kind == "weightless_rms"
    assert block.qk_norm_eps == 1e-5
    assert block.q_scale_factor == 3.87
    assert block.attention_gate == "sigmoid"
    assert block.norm_layout == "sandwich"
    assert block.centered_rms_norm is True
    assert (block.norm_eps, block.post_norm_eps) == (1e-5, 1e-8)
    assert block.embedding_norm_eps == 1e-5
    for capability in (
        "native_train",
        "native_inference",
        "whole_model_cuda",
        "k_quant",
        "speculative_decoding",
    ):
        assert spec.template.backend_capabilities[capability] is True
    assert spec.template.backend_capabilities["multimodal"] is True
    assert spec.vision is not None
    assert (
        spec.vision.num_hidden_layers,
        spec.vision.hidden_size,
        spec.vision.intermediate_size,
        spec.vision.num_attention_heads,
    ) == (50, 1_536, 8_960, 16)
    assert (spec.vision.patch_temporal, spec.vision.patch_size, spec.vision.merge_size) == (2, 14, 2)
    assert (spec.vision.image_token_id, spec.vision.video_token_id) == (200_092, 200_091)
    assert spec.dflash is not None
    assert spec.dflash.target_layer_ids == (1, 13, 25, 37, 49)


def test_muse_glimmer_graph_preserves_nonsquare_schedule_gate_and_sandwich_norms() -> None:
    spec = build_muse_glimmer_spec(
        num_layers=4,
        model_dim=32,
        num_heads=4,
        num_kv_heads=2,
        head_dim=4,
        attention_inner_dim=16,
        intermediate_size=96,
        vocab_size=128,
        window_size=8,
    )
    graph = build_gpt_root_graph(name="muse_glimmer_contract", model_spec=spec)
    assert set(graph.variant_library) == {
        "muse_glimmer_attention",
        "muse_glimmer_mlp",
        "muse_glimmer_block",
    }

    local = graph.variant_library["muse_glimmer_attention"]["local"]
    global_attention = graph.variant_library["muse_glimmer_attention"]["global"]
    assert local.nodes["q_proj"].neuron_def.module_config["output_dim"] == 16
    assert local.nodes["k_proj"].neuron_def.module_config["output_dim"] == 8
    assert local.nodes["v_proj"].neuron_def.module_config["output_dim"] == 8
    assert local.nodes["o_proj"].neuron_def.module_config == {
        "input_dim": 16,
        "output_dim": 32,
        "bias": False,
    }
    assert local.nodes["q_scale"].neuron_def.module_config["scale"] == 3.87
    assert local.nodes["qk_norm"].neuron_def.module_config == {
        "eps": 1e-5,
        "force_float32": True,
    }
    assert local.nodes["sdpa"].neuron_def.module_type == "sliding_window_attention"
    assert local.nodes["sdpa"].neuron_def.module_config["window_size"] == 8
    assert "rope" in local.nodes
    assert local.nodes["rope"].neuron_def.module_config["convention"] == "hf"
    assert local.nodes["attn_gate_proj"].neuron_def.module_config["output_dim"] == 16
    assert {"gate_sigmoid", "gate_mul"}.issubset(local.nodes)
    assert global_attention.nodes["sdpa"].neuron_def.module_type == "scaled_dot_product_attention"
    assert "rope" not in global_attention.nodes

    block = graph.variant_library["muse_glimmer_block"]["local"]
    assert block.nodes["self_attn"].neuron_def.variant_ref == {
        "family": "muse_glimmer_attention",
        "version": "local",
    }
    for node_id, eps in (
        ("input_layernorm", 1e-5),
        ("post_attention_layernorm", 1e-8),
        ("pre_feedforward_layernorm", 1e-5),
        ("post_feedforward_layernorm", 1e-8),
    ):
        config = block.nodes[node_id].neuron_def.module_config
        assert config["centered"] is True
        assert config["force_float32"] is True
        assert config["eps"] == eps

    mlp = graph.variant_library["muse_glimmer_mlp"]["default"]
    assert {"gate_proj", "up_proj", "silu", "swiglu_mul", "down_proj"}.issubset(mlp.nodes)
    assert mlp.nodes["gate_proj"].neuron_def.module_config["output_dim"] == 96
    assert mlp.nodes["down_proj"].neuron_def.module_config["input_dim"] == 96

    stage = graph.nodes["model"].neuron_def.subgraph
    assert stage is not None
    decoder = stage.nodes["decoder"].neuron_def.subgraph
    assert decoder is not None
    body = decoder.nodes["body"].neuron_def.subgraph
    assert body is not None
    assert [body.nodes[f"block_{idx}"].neuron_def.variant_ref["version"] for idx in range(4)] == [
        "local",
        "local",
        "local",
        "global",
    ]
    assert body.nodes["embedding_norm"].neuron_def.module_config.get("model_dim") is None
    assert body.nodes["final_norm"].neuron_def.module_config["centered"] is False
    assert decoder.nodes["lm_head"].neuron_def.module_type == "lm_head"
    assert decoder.nodes["output_multiplier"].neuron_def.module_config["scale"] == 0.19611613513818404
    assert decoder.edges["e_head_multiplier"].dst_node == "output_multiplier"
    assert decoder.edges["e_multiplier_softcap"].src_node == "output_multiplier"


def test_muse_glimmer_dflash_companion_graph_contract_and_backward() -> None:
    target = build_muse_glimmer_spec(
        num_layers=4,
        model_dim=8,
        num_heads=2,
        num_kv_heads=1,
        head_dim=2,
        attention_inner_dim=4,
        intermediate_size=16,
        vocab_size=13,
        window_size=3,
        max_position_embeddings=12,
    )
    graph = torch_templates.build_muse_glimmer_assistant_graph(
        "muse_glimmer_dflash_contract",
        target,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=2,
        intermediate_size=16,
        block_size=4,
        mask_token_id=12,
        window_size=3,
        target_layer_ids=(0, 2),
    )
    assert graph.input_node_ids == [
        "target_taps_in",
        "noise_embeddings_in",
        "context_positions_in",
        "block_positions_in",
    ]
    assert set(graph.variant_library) == {
        "muse_glimmer_dflash_attention",
        "muse_glimmer_dflash_mlp",
        "muse_glimmer_dflash_block",
    }
    assert graph.torch_config["dflash_spec"] == {
        "block_size": 4,
        "proposal_tokens": 3,
        "mask_token_id": 12,
        "target_layer_ids": [0, 2],
        "shared_target_embedding": True,
        "shared_target_lm_head": True,
    }
    attention = graph.variant_library["muse_glimmer_dflash_attention"]["default"]
    core = attention.nodes["dflash_attention"].neuron_def
    assert core.module_type == "dflash_attention"
    assert core.module_config == {
        "model_dim": 8,
        "num_heads": 2,
        "num_kv_heads": 1,
        "head_dim": 2,
        "window_size": 3,
        "rope_base": 500000.0,
        "norm_eps": 1e-5,
        "convention": "hf",
        "bias": False,
        "dropout_p": 0.0,
    }
    block = graph.variant_library["muse_glimmer_dflash_block"]["default"]
    assert block.nodes["input_layernorm"].neuron_def.module_config["centered"] is False
    assert block.nodes["post_attention_layernorm"].neuron_def.module_config["centered"] is False

    torch.manual_seed(11)
    compiled = CompiledTorchGraph(graph)
    taps = torch.randn(1, 3, 16)
    raw_embeddings = torch.randn(1, 4, 8)
    context_positions = torch.arange(3).unsqueeze(0)
    block_positions = torch.arange(3, 7).unsqueeze(0)
    hidden = compiled(
        taps, raw_embeddings, context_positions, block_positions
    )[0]
    assert hidden.shape == (1, 4, 8)
    hidden.square().mean().backward()
    assert all(parameter.grad is not None for parameter in compiled.parameters())


def _tiny_multimodal_glimmer_spec():
    return build_muse_glimmer_spec(
        num_layers=4,
        model_dim=8,
        num_heads=2,
        num_kv_heads=1,
        head_dim=2,
        attention_inner_dim=4,
        intermediate_size=16,
        vocab_size=32,
        max_position_embeddings=32,
        enable_dflash=False,
        enable_vision=True,
        vision_num_hidden_layers=2,
        vision_hidden_size=8,
        vision_intermediate_size=16,
        vision_num_attention_heads=2,
        vision_patch_size=2,
        vision_patch_temporal=1,
        vision_merge_size=2,
        vision_pos_emb_height=2,
        vision_pos_emb_width=2,
        projector_hidden_size=6,
        image_token_id=30,
        video_token_id=31,
    )


def test_muse_glimmer_vision_and_media_fusion_graphs_compile_and_backward() -> None:
    spec = _tiny_multimodal_glimmer_spec()
    graph = torch_templates.build_muse_glimmer_vision_graph("glimmer_vision", spec)
    tower = graph.nodes["vision_tower"].neuron_def
    assert tower.module_type == "muse_glimmer_vision_tower"
    assert tower.module_config == {
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_heads": 2,
        "num_layers": 2,
        "patch_size": 2,
        "patch_temporal": 1,
        "merge_size": 2,
        "pos_emb_height": 2,
        "pos_emb_width": 2,
        "rope_theta": 10000.0,
        "eps": 1e-5,
    }
    compiled = CompiledTorchGraph(graph)
    patches = torch.randn(4, 12, requires_grad=True)
    grid = torch.tensor([[1, 2, 2]])
    features = compiled(patches, grid)[0]
    assert features.shape == (1, 8)
    features.square().mean().backward()
    assert patches.grad is not None
    assert all(parameter.grad is not None for parameter in compiled.parameters())

    fusion = CompiledTorchGraph(
        torch_templates.build_muse_glimmer_media_fusion_graph("fusion", spec)
    )
    embeddings = torch.zeros(1, 4, 8)
    token_ids = torch.tensor([[1, 30, 2, 31]])
    image = torch.full((1, 8), 2.0)
    video = torch.full((1, 8), 3.0)
    fused = fusion(embeddings, token_ids, image, video)[0]
    assert torch.equal(fused[0, 1], image[0])
    assert torch.equal(fused[0, 3], video[0])
    with pytest.raises(ValueError, match="placeholders"):
        fusion(embeddings, token_ids, image[:0], video)


def test_muse_glimmer_vision_tower_matches_installed_transformers_oracle() -> None:
    if importlib.util.find_spec("transformers.models.muse_glimmer") is None:
        pytest.skip("installed Transformers has no Muse Glimmer oracle")
    from transformers.models.muse_glimmer.configuration_muse_glimmer import (
        MuseGlimmerVisionConfig,
    )
    from transformers.models.muse_glimmer.modeling_muse_glimmer import (
        MuseGlimmerVisionModel,
    )

    torch.manual_seed(4)
    config = MuseGlimmerVisionConfig(
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_hidden_layers=2,
        patch_size=2,
        patch_temporal=1,
        merge_size=2,
        pos_emb_height=2,
        pos_emb_width=2,
        max_position_embeddings=4,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        layer_types=["window_attention", "full_attention"],
    )
    oracle = MuseGlimmerVisionModel(config).eval()
    stage = MuseGlimmerVisionTowerStage(
        hidden_size=8,
        intermediate_size=16,
        num_heads=2,
        num_layers=2,
        patch_size=2,
        patch_temporal=1,
        merge_size=2,
        pos_emb_height=2,
        pos_emb_width=2,
    ).eval()
    with torch.no_grad():
        stage.patch_embedding.weight.copy_(oracle.patch_embedder.patch_embedding.weight)
        stage.position_embedding.weight.copy_(oracle.patch_embedder.position_embedding_table.weight)
        stage.pre_norm.load_state_dict(oracle.ln_pre.state_dict())
        stage.post_norm.load_state_dict(oracle.ln_post.state_dict())
        for ours, theirs in zip(stage.layers, oracle.layers):
            ours.norm1.load_state_dict(theirs.norm1.state_dict())
            ours.norm2.load_state_dict(theirs.norm2.state_dict())
            ours.attn.load_state_dict(theirs.attn.state_dict())
            ours.fc1.load_state_dict(theirs.mlp.fc1.state_dict())
            ours.fc2.load_state_dict(theirs.mlp.fc2.state_dict())
    patches = torch.randn(4, 12)
    grid = torch.tensor([[1, 2, 2]])
    with torch.no_grad():
        expected = oracle(patches, grid).last_hidden_state
        actual = stage(patches, grid)
    assert torch.equal(actual, expected)


def test_dflash_attention_matches_explicit_positioned_bidirectional_formula() -> None:
    torch.manual_seed(23)
    stage = DFlashAttentionStage(
        model_dim=8,
        num_heads=2,
        num_kv_heads=1,
        head_dim=2,
        window_size=3,
        rope_base=500000.0,
        norm_eps=1e-5,
        convention="hf",
    )
    block = torch.randn(1, 4, 8)
    context = torch.randn(1, 3, 8)
    context_positions = torch.arange(3).unsqueeze(0)
    block_positions = torch.arange(3, 7).unsqueeze(0)

    def heads(value: torch.Tensor, count: int) -> torch.Tensor:
        return value.view(1, value.shape[1], count, 2).transpose(1, 2)

    def norm(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + 1e-5) * weight

    def rope(value: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        inverse = 1.0 / (
            500000.0 ** (torch.arange(0, 2, 2, dtype=torch.float32) / 2)
        )
        angle = positions.float().unsqueeze(-1) * inverse
        cosine = torch.cat((angle.cos(), angle.cos()), dim=-1).unsqueeze(1)
        sine = torch.cat((angle.sin(), angle.sin()), dim=-1).unsqueeze(1)
        first, second = value.chunk(2, dim=-1)
        return value * cosine + torch.cat((-second, first), dim=-1) * sine

    q = rope(norm(heads(stage.q_proj(block), 2), stage.q_norm), block_positions)
    context_k = rope(
        norm(heads(stage.k_proj(context), 1), stage.k_norm), context_positions
    )
    block_k = rope(
        norm(heads(stage.k_proj(block), 1), stage.k_norm), block_positions
    )
    key = torch.cat((context_k, block_k), dim=-2)
    value = torch.cat(
        (heads(stage.v_proj(context), 1), heads(stage.v_proj(block), 1)), dim=-2
    )
    key_positions = torch.cat((context_positions, block_positions), dim=-1)
    distance = key_positions.unsqueeze(1) - block_positions.unsqueeze(-1)
    allowed = (distance >= -3) & (distance <= 3)
    mask = torch.zeros((1, 1, 4, 7)).masked_fill(
        ~allowed.unsqueeze(1), float("-inf")
    )
    expected = F.scaled_dot_product_attention(
        q, key, value, attn_mask=mask, is_causal=False, enable_gqa=True
    ).transpose(1, 2).contiguous().view(1, 4, 4)
    assert torch.allclose(
        stage(block, context, context_positions, block_positions),
        expected,
        atol=2e-6,
        rtol=2e-5,
    )


def test_muse_glimmer_centered_rms_and_tensor_scale_reference_math() -> None:
    x = torch.tensor([[[1.0, -2.0, 3.0, -4.0]]], dtype=torch.bfloat16)
    norm = RMSNormStage(1e-5, 4, centered=True, force_float32=True)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([0.0, 0.25, -0.5, 1.0]))
    expected = x.float() * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + 1e-5)
    expected = expected * torch.tensor([1.0, 1.25, 0.5, 2.0])
    actual = norm(x)
    assert actual.dtype == x.dtype
    assert torch.allclose(actual.float(), expected, atol=1e-2, rtol=1e-2)
    assert torch.equal(TensorScaleStage(0.5)(x), x * 0.5)


def _muse_rms(
    x: torch.Tensor,
    *,
    eps: float,
    weight: torch.Tensor | None = None,
    centered: bool = False,
) -> torch.Tensor:
    dtype = x.dtype
    out = x.float() * torch.pow(x.float().square().mean(dim=-1, keepdim=True) + eps, -0.5)
    if weight is not None:
        out = out * ((1.0 + weight.float()) if centered else weight.float())
    return out.to(dtype)


def _muse_hf_rope(x: torch.Tensor, *, theta: float) -> torch.Tensor:
    head_dim = x.shape[-1]
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=x.device) / head_dim)
    )
    freqs = torch.outer(torch.arange(x.shape[-2], dtype=torch.float32, device=x.device), inv_freq)
    cos = freqs.cos()[None, None, :, :].to(x.dtype)
    sin = freqs.sin()[None, None, :, :].to(x.dtype)
    first, second = x.chunk(2, dim=-1)
    return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)


def _muse_glimmer_pinned_text_oracle(
    params: dict[str, torch.nn.Parameter],
    input_ids: torch.Tensor,
    config: dict[str, object],
) -> torch.Tensor:
    """Direct formula transcribed from the immutable Transformers fixture.

    This intentionally bypasses NeuralFn stages and graph execution so the
    template test detects operation-order, mask, norm, and RoPE regressions.
    """

    prefix = "node_modules.body.node_modules."
    hidden_size = int(config["hidden_size"])
    num_heads = int(config["num_attention_heads"])
    num_kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])
    rms_eps = float(config["rms_norm_eps"])
    post_eps = float(config["post_norm_eps"])
    q_scale = float(config["qk_scale_factor"])
    window = int(config["sliding_window"])
    layer_types = list(config["layer_types"])
    layer_rope_theta = list(config["layer_rope_theta"])

    hidden = F.embedding(input_ids, params[prefix + "token_embed.embedding.weight"])
    hidden = _muse_rms(hidden, eps=rms_eps)
    batch, seq_len = input_ids.shape
    row = torch.arange(seq_len, device=input_ids.device).unsqueeze(1)
    col = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

    for layer_index, layer_type in enumerate(layer_types):
        layer = prefix + f"block_{layer_index}.node_modules."
        residual = hidden
        normed = _muse_rms(
            hidden,
            eps=rms_eps,
            weight=params[layer + "input_layernorm.weight"],
            centered=True,
        )
        attn = layer + "self_attn.node_modules."
        q = F.linear(normed, params[attn + "q_proj.proj.weight"])
        k = F.linear(normed, params[attn + "k_proj.proj.weight"])
        v = F.linear(normed, params[attn + "v_proj.proj.weight"])
        q = q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        q = _muse_rms(q, eps=rms_eps) * q_scale
        k = _muse_rms(k, eps=rms_eps)
        theta = float(layer_rope_theta[layer_index])
        if theta:
            q = _muse_hf_rope(q, theta=theta)
            k = _muse_hf_rope(k, theta=theta)
        repeats = num_heads // num_kv_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)
        allowed = col <= row
        if layer_type == "sliding_attention":
            allowed &= col > row - window
        mask = torch.zeros(seq_len, seq_len, device=hidden.device, dtype=hidden.dtype)
        mask = mask.masked_fill(~allowed, float("-inf"))
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, num_heads * head_dim)
        gate = torch.sigmoid(F.linear(normed, params[attn + "attn_gate_proj.proj.weight"]))
        attn_out = F.linear(attn_out * gate, params[attn + "o_proj.proj.weight"])
        attn_out = _muse_rms(
            attn_out,
            eps=post_eps,
            weight=params[layer + "post_attention_layernorm.weight"],
            centered=True,
        )
        hidden = residual + attn_out

        residual = hidden
        normed = _muse_rms(
            hidden,
            eps=rms_eps,
            weight=params[layer + "pre_feedforward_layernorm.weight"],
            centered=True,
        )
        mlp = layer + "mlp.node_modules."
        gated = F.silu(F.linear(normed, params[mlp + "gate_proj.proj.weight"]))
        gated = gated * F.linear(normed, params[mlp + "up_proj.proj.weight"])
        mlp_out = F.linear(gated, params[mlp + "down_proj.proj.weight"])
        mlp_out = _muse_rms(
            mlp_out,
            eps=post_eps,
            weight=params[layer + "post_feedforward_layernorm.weight"],
            centered=True,
        )
        hidden = residual + mlp_out

    hidden = _muse_rms(hidden, eps=rms_eps, weight=params[prefix + "final_norm.weight"])
    logits = F.linear(hidden, params["node_modules.lm_head.proj.weight"])
    logits = logits * float(config["output_multiplier"])
    cap = float(config["final_logit_softcapping"])
    return cap * torch.tanh(logits / cap)


def test_muse_glimmer_tiny_forward_and_backward_match_pinned_formula() -> None:
    fixture = json.loads((ROOT / "tests" / "fixtures" / "muse_glimmer" / "reference.json").read_text())
    assert fixture["schema"] == "neuralfn.muse_glimmer.reference.v1"
    assert fixture["transformers_revision"] == "d1123114da1ab4395198146f4f84dae7fe8b693e"
    config = fixture["tiny_text_config"]
    torch.manual_seed(int(fixture["seed"]))
    spec = build_muse_glimmer_spec(
        num_layers=int(config["num_hidden_layers"]),
        model_dim=int(config["hidden_size"]),
        num_heads=int(config["num_attention_heads"]),
        num_kv_heads=int(config["num_key_value_heads"]),
        head_dim=int(config["head_dim"]),
        attention_inner_dim=int(config["num_attention_heads"]) * int(config["head_dim"]),
        intermediate_size=int(config["intermediate_size"]),
        vocab_size=int(config["vocab_size"]),
        window_size=int(config["sliding_window"]),
        max_position_embeddings=int(config["max_position_embeddings"]),
    )
    graph = torch_templates.build_muse_glimmer_logits_stage_graph("muse_glimmer_parity", spec)
    graph.torch_config = {**graph.torch_config, "device": "cpu", "amp_dtype": "float32"}
    compiled = CompiledTorchGraph(graph)
    reference = CompiledTorchGraph(deepcopy(graph))
    reference.load_state_dict(compiled.state_dict())
    input_ids = torch.tensor(fixture["input_ids"], dtype=torch.long)

    actual = compiled(input_ids)[0]
    expected = _muse_glimmer_pinned_text_oracle(dict(reference.named_parameters()), input_ids, config)
    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-5)

    probe = torch.linspace(-0.5, 0.5, actual.numel(), dtype=actual.dtype).view_as(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    actual_params = dict(compiled.named_parameters())
    reference_params = dict(reference.named_parameters())
    assert actual_params.keys() == reference_params.keys()
    for name in actual_params:
        assert actual_params[name].grad is not None, name
        assert reference_params[name].grad is not None, name
        assert torch.allclose(
            actual_params[name].grad,
            reference_params[name].grad,
            atol=3e-6,
            rtol=3e-5,
        ), name


def test_template_terminals_only_quantize_discrete_token_ports() -> None:
    tensor_terminal = make_terminal_def(role="input", port_name="x", dtype="tensor")
    token_terminal = make_terminal_def(role="input", port_name="tokens", dtype="tokens")

    assert tensor_terminal.input_ports[0].precision is None
    assert tensor_terminal.output_ports[0].precision is None
    assert token_terminal.input_ports[0].precision == 1.0
    assert token_terminal.output_ports[0].precision == 1.0


def test_reported_presets_resolve_variant_libraries() -> None:
    for preset in PRESETS:
        spec = build_model_spec_from_config({"preset": preset, **_tiny_kwargs()}, preview_defaults=True)
        graph = build_gpt_root_graph(name=f"{preset}_resolve", model_spec=spec)
        graph.resolve_variant_library()


def test_all_ordered_preset_pairs_preserve_inline_variant_fallback() -> None:
    """Loading a second preset may overwrite shared variant families safely."""

    graphs: dict[str, NeuronGraph] = {}
    for preset in PRESETS:
        spec = build_model_spec_from_config({"preset": preset, **_tiny_kwargs()}, preview_defaults=True)
        graphs[preset] = build_gpt_root_graph(name=f"{preset}_pair", model_spec=spec)

    checked = 0
    for left_name, left in graphs.items():
        for right_name, right in graphs.items():
            candidate = deepcopy(left)
            # This mirrors mergeVariantLibrary: families present in the newly
            # loaded preset replace old entries, while unrelated families stay.
            candidate.variant_library.update(deepcopy(right.variant_library))
            try:
                candidate.resolve_variant_library()
                candidate.validate()
            except Exception as exc:
                raise AssertionError(
                    f"variant overwrite failed for {left_name} then {right_name}: {exc}"
                ) from exc
            checked += 1

    assert checked == len(PRESETS) ** 2


def test_all_presets_compile_and_forward() -> None:
    """Every shipped preset must build, resolve variants, compile, and run a forward pass."""
    for preset in PRESETS:
        spec = build_model_spec_from_config(
            {"preset": preset, "vocab_size": 128, **_tiny_kwargs()}, preview_defaults=True,
        )
        graph = _cpu_graph(build_gpt_root_graph(name=f"{preset}_fwd", model_spec=spec))
        compiled = CompiledTorchGraph(graph)
        batch = 2
        seq = 8
        roles = []
        for nid in graph.input_node_ids:
            roles.extend(p.name for p in graph.nodes[nid].neuron_def.output_ports)
        inputs = tuple(torch.randint(0, 128, (batch, seq)) for _ in roles)
        outputs = compiled(*inputs)
        assert len(outputs) >= 1, f"{preset}: expected at least 1 output"


def test_seq2seq_blocks_reference_exported_variant_families() -> None:
    spec = build_model_spec_from_config({"preset": "seq2seq", **_tiny_kwargs()}, preview_defaults=True)
    graph = build_gpt_root_graph(name="seq2seq_refs", model_spec=spec)
    enc_block_graph = graph.variant_library["enc_block"]["default"]
    dec_block_graph = graph.variant_library["dec_block"]["default"]

    assert enc_block_graph.nodes["attention"].neuron_def.variant_ref == {"family": "enc_attention", "version": "default"}
    assert enc_block_graph.nodes["mlp"].neuron_def.variant_ref == {"family": "mlp_dense", "version": "default"}
    assert dec_block_graph.nodes["attention"].neuron_def.variant_ref == {"family": "dec_attention", "version": "default"}
    assert dec_block_graph.nodes["cross_attn"].neuron_def.variant_ref == {"family": "cross_attention", "version": "default"}
    assert dec_block_graph.nodes["mlp"].neuron_def.variant_ref == {"family": "mlp_moe", "version": "default"}


def test_legacy_variant_family_aliases_resolve_saved_graphs() -> None:
    for link_family, available_family in [
        ("attn_block", "transformer_block"),
        ("transformer_block", "attn_block"),
        ("mixllama", "attn_block"),
    ]:
        graph = _make_alias_root(link_family, available_family)
        graph.resolve_variant_library()
        assert graph.nodes["block"].neuron_def.subgraph is not None
        assert graph.nodes["block"].neuron_def.subgraph.name == f"{available_family}_default"


def test_apply_gpt_template_supports_all_presets() -> None:
    for preset in PRESETS:
        graph = apply_gpt_template(GPTTemplateRequest(name=f"{preset}_graph", config={"preset": preset}))
        assert "model" in graph.nodes
        assert graph.output_node_ids == ["loss_out"]


def test_jepa_semantic_hybrid_dataset_backed_trace_preview_smoke() -> None:
    spec = build_model_spec_from_config(
        {"preset": "jepa_semantic_hybrid", "vocab_size": 256, **_tiny_kwargs()},
        preview_defaults=True,
    )
    graph = _cpu_graph(build_gpt_root_graph(name="jsh_trace_smoke", model_spec=spec))
    response = trace_torch_graph(graph, ExecuteRequest())
    assert response["source"] == "dataset"
    assert response["trace"]


def test_nonsemantic_jepa_evo_presets_do_not_use_semantic_router() -> None:
    for preset, expected_sparsity in [
        ("dense_jepa_evo", "dense"),
        ("moe_jepa_evo", "moe"),
    ]:
        spec = build_model_spec_from_config(
            {"preset": preset, "vocab_size": 128, **_tiny_kwargs()},
            preview_defaults=True,
        )
        graph = build_gpt_root_graph(name=f"{preset}_no_semantic_router", model_spec=spec)
        assert spec.template.objective == "ar_jepa"
        assert spec.template.sparsity == expected_sparsity
        assert graph.input_node_ids == ["tokens_in", "targets_in"]

        module_types = {
            getattr(node.neuron_def, "module_type", "")
            for node in graph.nodes.values()
        }
        assert "semantic_data_source" not in module_types
        assert "semantic_moe_jepa_evo_router" not in module_types
        assert "semantic_hash_router" not in module_types
        assert "semantic_moe_router" not in module_types
        if preset == "moe_jepa_evo":
            assert "mlp" in graph.variant_library
            mlp = graph.variant_library["mlp"]["default"]
            mlp_module_types = {
                getattr(node.neuron_def, "module_type", "")
                for node in mlp.nodes.values()
            }
            assert {"router_logits", "topk_route", "expert_dispatch", "expert_combine"} <= mlp_module_types

    deepseek_spec = build_model_spec_from_config(
        {"preset": "deepseek_v4", "vocab_size": 128, **_tiny_kwargs()},
        preview_defaults=True,
    )
    deepseek_graph = build_gpt_root_graph(name="deepseek_v4_router_score", model_spec=deepseek_spec)
    deepseek_mlp = deepseek_graph.variant_library["mlp"]["default"]
    topk_nodes = [
        node for node in deepseek_mlp.nodes.values()
        if getattr(node.neuron_def, "module_type", "") == "topk_route"
    ]
    assert topk_nodes
    assert topk_nodes[0].neuron_def.module_config["score_fn"] == "sqrt_softplus"


def test_ttt_llama_forward_smoke() -> None:
    spec = build_ttt_llama_spec(**_tiny_kwargs(), vocab_size=128, ttt_hidden_dim=24)
    graph = _cpu_graph(build_gpt_root_graph(name="ttt_smoke", model_spec=spec))
    attention_graph = graph.variant_library["attention"]["default"]
    assert any(node.neuron_def.module_type == "ttt_linear" for node in attention_graph.nodes.values())

    compiled = CompiledTorchGraph(graph)
    tokens = torch.randint(0, 128, (2, 8))
    targets = torch.randint(0, 128, (2, 8))
    loss = compiled(tokens, targets)[0]
    assert loss.ndim == 0


def test_jepa_trainer_freezes_and_updates_ema_targets() -> None:
    spec = build_llm_jepa_spec(**_tiny_kwargs(), vocab_size=128, ema_decay=0.9)
    graph = _cpu_graph(build_gpt_root_graph(name="jepa_train", model_spec=spec))

    compiled = CompiledTorchGraph(graph)
    TorchTrainer._prepare_ema_targets(compiled)
    model = compiled.node_modules["model"]
    online = model.node_modules["online_encoder"]
    target = model.node_modules["target_encoder"]
    assert all(not param.requires_grad for param in target.parameters())
    initial_target_param = next(target.parameters()).detach().clone()
    for online_param, target_param in zip(online.parameters(), target.parameters()):
        assert torch.equal(online_param, target_param)

    with torch.no_grad():
        next(online.parameters()).add_(0.5)
    TorchTrainer._ema_update_targets(compiled, 0.9)
    updated_target_param = next(target.parameters()).detach().clone()
    assert not torch.equal(updated_target_param, initial_target_param)
    assert not torch.equal(updated_target_param, next(online.parameters()).detach())

    trainer = TorchTrainer(
        graph,
        TorchTrainConfig(epochs=1, batch_size=2, learning_rate=1e-3, max_steps=1, device="cpu"),
    )
    tokens = torch.randint(0, 128, (4, 8))
    losses = trainer.train(tokens, tokens)
    assert len(losses) == 1


def test_jepa_block_masking_produces_contiguous_spans() -> None:
    torch.manual_seed(42)
    batch, seq_len = 8, 64
    tokens = torch.randint(0, 128, (batch, seq_len))

    block_stage = JEPAMaskStage(
        mask_ratio=0.5,
        mask_strategy="block",
        num_blocks=4,
        min_block_ratio=0.1,
        max_block_ratio=0.25,
    )
    masked_tokens, mask_float = block_stage(tokens)
    mask = mask_float.bool()

    assert mask.shape == tokens.shape
    assert mask.any(), "block mask should mask at least some tokens"
    assert not mask.all(), "block mask should leave some tokens unmasked"
    assert (masked_tokens[mask] == 0).all(), "masked positions should be replaced with mask_token_id"
    assert torch.equal(masked_tokens[~mask], tokens[~mask]), "unmasked positions should be unchanged"

    for row in range(batch):
        spans = []
        row_mask = mask[row]
        in_span = False
        start = 0
        for i in range(seq_len):
            if row_mask[i] and not in_span:
                in_span = True
                start = i
            elif not row_mask[i] and in_span:
                in_span = False
                spans.append((start, i))
        if in_span:
            spans.append((start, seq_len))
        min_len = max(1, int(0.1 * seq_len))
        for s, e in spans:
            assert (e - s) >= min_len, f"span [{s}:{e}) length {e - s} < min_block_len {min_len}"

    random_stage = JEPAMaskStage(mask_ratio=0.5, mask_strategy="random")
    _, random_mask = random_stage(tokens)
    diff = random_mask.bool()
    transitions = (diff[:, 1:] != diff[:, :-1]).float().sum(dim=1).mean()
    assert transitions > 5.0, "random masking should produce many transitions (scattered mask)"


def test_jepa_block_masking_config_wires_through_template() -> None:
    spec = build_llm_jepa_spec(
        **_tiny_kwargs(),
        vocab_size=128,
        jepa_mask_strategy="block",
        jepa_num_blocks=3,
        jepa_min_block_ratio=0.15,
        jepa_max_block_ratio=0.3,
    )
    assert spec.jepa_mask_strategy == "block"
    assert spec.jepa_num_blocks == 3

    graph = _cpu_graph(build_gpt_root_graph(name="jepa_block_cfg", model_spec=spec))
    model_subgraph = graph.nodes["model"].neuron_def.subgraph
    assert model_subgraph is not None
    mask_node = model_subgraph.nodes["mask"]
    cfg = mask_node.neuron_def.module_config
    assert cfg["mask_strategy"] == "block"
    assert cfg["num_blocks"] == 3
    assert cfg["min_block_ratio"] == 0.15
    assert cfg["max_block_ratio"] == 0.3


def test_hnet_spec_enforces_byte_vocab_and_raw_byte_chunking(tmp_path: Path, monkeypatch) -> None:
    spec = build_hnet_lm_spec(**_tiny_kwargs(), vocab_size=1024, byte_patch_size=2, byte_patch_stride=2)
    assert spec.vocab_size == 256

    monkeypatch.setattr(dataset_manager, "DATASETS_DIR", tmp_path)
    dataset_name = f"test_hnet_bytes_{uuid.uuid4().hex}"
    dataset_path = tmp_path / f"{dataset_name}.txt"
    dataset_path.write_bytes(b"abcdefghi")
    inputs, targets = load_dataset_bytes([dataset_name], seq_len=4)

    assert inputs == [[97, 98, 99, 100], [101, 102, 103, 104]]
    assert targets == [[98, 99, 100, 101], [102, 103, 104, 105]]


def test_hnet_trainer_runs_one_step() -> None:
    spec = build_hnet_lm_spec(**_tiny_kwargs(), byte_patch_size=2, byte_patch_stride=2)
    graph = _cpu_graph(build_gpt_root_graph(name="hnet_train", model_spec=spec))

    trainer = TorchTrainer(
        graph,
        TorchTrainConfig(epochs=1, batch_size=2, learning_rate=1e-3, max_steps=1, device="cpu"),
    )
    tokens = torch.randint(0, 256, (4, 8))
    targets = torch.randint(0, 256, (4, 8))
    losses = trainer.train(tokens, targets)
    assert len(losses) == 1


def test_universal_template_uses_single_shared_block_and_normalized_halting() -> None:
    spec = build_universal_llama_spec(**_tiny_kwargs(), vocab_size=128, max_recurrence_steps=3, halt_epsilon=0.01)
    graph = _cpu_graph(build_gpt_root_graph(name="universal_trace", model_spec=spec))
    model_subgraph = graph.nodes["model"].neuron_def.subgraph
    assert model_subgraph is not None
    assert sum(1 for node in model_subgraph.nodes.values() if node.neuron_def.module_type == "universal_transformer") == 1

    compiled = CompiledTorchGraph(graph)
    tokens = torch.randint(0, 128, (2, 8))
    targets = torch.randint(0, 128, (2, 8))
    outputs, trace = compiled.trace(tokens, targets)
    halt_weights = trace["model/universal"][1]
    assert outputs[0].ndim == 0
    assert halt_weights.shape == (2, 3)
    assert torch.allclose(halt_weights.sum(dim=1), torch.ones(2), atol=1e-4)

    trainer = TorchTrainer(
        graph,
        TorchTrainConfig(epochs=1, batch_size=2, learning_rate=1e-3, max_steps=1, device="cpu"),
    )
    losses = trainer.train(tokens, targets)
    assert len(losses) == 1


def test_dataset_source_role_wiring_covers_single_and_multi_input_templates() -> None:
    seq2seq_graph = apply_gpt_template(GPTTemplateRequest(name="seq2seq", config={"preset": "seq2seq", "num_layers": 1}))
    seq2seq_result = load_dataset_source_into_graph(seq2seq_graph, LoadDatasetRequest(dataset_names=["dummy"], seq_len=8))
    seq2seq_ports = [port.name for port in seq2seq_graph.nodes[seq2seq_result["dataset_source_node_id"]].neuron_def.output_ports]
    assert seq2seq_ports == ["enc_tokens", "dec_tokens", "targets"]

    jepa_graph = apply_gpt_template(GPTTemplateRequest(name="jepa", config={"preset": "llm_jepa", "num_layers": 1}))
    jepa_result = load_dataset_source_into_graph(jepa_graph, LoadDatasetRequest(dataset_names=["dummy"], seq_len=8))
    jepa_ports = [port.name for port in jepa_graph.nodes[jepa_result["dataset_source_node_id"]].neuron_def.output_ports]
    assert jepa_ports == ["tokens"]

    hybrid_graph = apply_gpt_template(
        GPTTemplateRequest(name="jsh", config={"preset": "jepa_semantic_hybrid", "num_layers": 1})
    )
    assert "dataset_source" in hybrid_graph.nodes
    assert "semantic_data_source" in hybrid_graph.nodes
    assert "tokens_in" not in hybrid_graph.nodes
    assert hybrid_graph.input_node_ids == ["dataset_source", "semantic_data_source"]
    hybrid_result = load_dataset_source_into_graph(
        hybrid_graph,
        LoadDatasetRequest(dataset_names=["dummy"], seq_len=8),
    )
    hybrid_ds_id = hybrid_result["dataset_source_node_id"]
    hybrid_ports = [
        port.name for port in hybrid_graph.nodes[hybrid_ds_id].neuron_def.output_ports
    ]
    assert "semantic_data_source" in hybrid_graph.nodes
    assert hybrid_ports == ["tokens", "targets"]
    assert hybrid_graph.input_node_ids == [hybrid_ds_id, "semantic_data_source"]
