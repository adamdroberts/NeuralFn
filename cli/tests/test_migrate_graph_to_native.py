from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from neuralfn.graph import Edge, NeuronGraph, NeuronInstance
from neuralfn.neuron import neuron_from_source
from neuralfn.port import Port
from neuralfn.torch_templates import (
    build_gpt_root_graph,
    build_model_spec_from_config,
)


ROOT = Path(__file__).resolve().parents[2]
NFN = ROOT / "cli" / "nfn.py"


def _write_passthrough_graph(path: Path, *, custom: bool = False) -> None:
    port = Port("x", dtype="float")

    def definition(name: str):
        return neuron_from_source(
            f"def {name}(x):\n    return x\n",
            name,
            [port],
            [port],
        )

    graph = NeuronGraph(name="cli_native_migration")
    graph.add_node(NeuronInstance(definition("input"), instance_id="input"))
    graph.add_node(NeuronInstance(definition("output"), instance_id="output"))
    graph.input_node_ids = ["input"]
    graph.output_node_ids = ["output"]
    if custom:
        graph.add_node(NeuronInstance(definition("custom"), instance_id="custom"))
        graph.add_edge(Edge(id="a", src_node="input", src_port=0, dst_node="custom", dst_port=0))
        graph.add_edge(Edge(id="b", src_node="custom", src_port=0, dst_node="output", dst_port=0))
    else:
        graph.add_edge(Edge(id="a", src_node="input", src_port=0, dst_node="output", dst_port=0))
    path.write_text(json.dumps(graph.to_dict()), encoding="utf-8")


def _write_gpt2_diff_graph(path: Path) -> None:
    spec = build_model_spec_from_config(
        {
            "preset": "gpt2_diff",
            "num_layers": 1,
            "model_dim": 32,
            "num_heads": 4,
            "vocab_size": 50_257,
        },
        preview_defaults=True,
    )
    payload = build_gpt_root_graph(name="gpt2_diff", model_spec=spec).to_dict()
    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_migrate_help_is_available_from_lightweight_cli() -> None:
    completed = subprocess.run(
        [sys.executable, str(NFN), "migrate", "graph-to-native", "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "nfn migrate graph-to-native" in completed.stdout
    assert "--graph GRAPH" in completed.stdout
    assert "--weights WEIGHTS" in completed.stdout
    assert "--output-dir DIR" in completed.stdout


def test_graph_only_cli_dry_run_is_torch_free_and_writes_nothing(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.json"
    output_dir = tmp_path / "native"
    _write_passthrough_graph(graph_path)
    argv = [
        str(NFN),
        "migrate",
        "graph-to-native",
        "--graph",
        str(graph_path),
        "--output-dir",
        str(output_dir),
        "--dry-run",
    ]
    script = (
        "import runpy,sys;"
        f"sys.argv={argv!r};"
        "code=0;"
        "\ntry: runpy.run_path(sys.argv[0],run_name='__main__')"
        "\nexcept SystemExit as exc: code=int(exc.code or 0)"
        "\nprint('HEAVY_LOADED='+','.join(name for name in ('torch','numpy','networkx') if name in sys.modules),file=sys.stderr)"
        "\nraise SystemExit(code)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["dry_run"] is True
    assert payload["compatibility_report"]["compatible"] is True
    assert payload["manifest"]["schema"] == "neuralfn.native_execution_manifest"
    assert "HEAVY_LOADED=" in completed.stderr.splitlines()
    assert not output_dir.exists()


def test_incompatible_cli_preflight_returns_two_without_output(tmp_path: Path) -> None:
    graph_path = tmp_path / "custom.json"
    output_dir = tmp_path / "native"
    _write_passthrough_graph(graph_path, custom=True)

    completed = subprocess.run(
        [
            sys.executable,
            str(NFN),
            "migrate",
            "graph-to-native",
            "--graph",
            str(graph_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert payload["compatibility_report"]["compatible"] is False
    assert payload["compatibility_report"]["unsupported_node_paths"] == [
        "root/nodes/custom"
    ]
    assert not output_dir.exists()


def test_gpt2_diff_cli_migration_rejects_unconsumed_bundle_before_output(
    tmp_path: Path,
) -> None:
    graph_path = tmp_path / "gpt2-diff.json"
    metadata_path = tmp_path / "model_00000004.diff.json"
    output_dir = tmp_path / "native"
    _write_gpt2_diff_graph(graph_path)
    metadata_path.write_text(
        '{"schema":"neuralfn.native_gpt2_diff.training_checkpoint","version":2}',
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(NFN),
            "migrate",
            "graph-to-native",
            "--graph",
            str(graph_path),
            "--weights",
            str(metadata_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert (
        "Native gpt2_diff migration does not yet consume "
        "neuralfn.native_gpt2_diff.training_checkpoint version 2"
        in completed.stderr
    )
    assert not output_dir.exists()
