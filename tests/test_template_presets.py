from __future__ import annotations

import ast
import inspect
import importlib.util
import os
import subprocess
import textwrap
import uuid
from pathlib import Path

import torch

from neuralfn.graph import Edge, NeuronGraph, NeuronInstance
from neuralfn.neuron import neuron_from_source, subgraph_neuron
from neuralfn.port import Port
from neuralfn.config import (
    MODERN_BASE_PRESETS,
    SHIPPED_GPT_TEMPLATE_BASE_PRESETS,
    SHIPPED_GPT_TEMPLATE_PRESETS,
    build_hnet_lm_spec,
    build_llm_jepa_spec,
    build_ttt_llama_spec,
    build_universal_llama_spec,
)
import neuralfn.torch_templates as torch_templates
from neuralfn.torch_backend import CompiledTorchGraph, JEPAMaskStage, TorchTrainConfig, TorchTrainer
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
        assert "--model-family" in argv
        assert argv[argv.index("--model-family") + 1] == "gpt"
        assert "--template-name" in argv
        assert argv[argv.index("--template-name") + 1] == preset


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
        expected_family = preset if preset in {"gpt2", "nanogpt"} else "gpt"
        assert argv[argv.index("--model-family") + 1] == expected_family
        assert argv[argv.index("--template-name") + 1] == preset


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


def test_root_graph_defaults_to_float32_amp() -> None:
    graph = build_gpt_root_graph(name="float32_default")
    assert graph.torch_config["amp_dtype"] == "float32"


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
