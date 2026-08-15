from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess

import pytest

from neuralfn.torch_templates import build_gpt_root_graph, build_model_spec_from_config
from neuralfn.native_graph_train import plan_native_graph_training


ROOT = Path(__file__).resolve().parents[1]
TRAINER_SOURCE = (
    ROOT / "neuralfn" / "csrc" / "native_gpt2" / "nfn_gpt2_native_train.cpp"
)
TILE_OPS = ROOT / "build" / "libnfn_native_train_tile_ops.so"
PREFLIGHT_PREFIX = "native GPT gpt2_diff packed differential preflight failed: "
DIFF_FORWARD_SYMBOL = (
    "nfn_native_tile_differential_packed_attention_forward_learned_lambda_bf16"
)
DIFF_BACKWARD_SYMBOL = (
    "nfn_native_tile_differential_packed_attention_backward_learned_lambda_bf16"
)
DIFF_RELEASE_SYMBOL = (
    "nfn_native_tile_differential_packed_attention_release_workspaces"
)
PACKED_ENV_KEYS = (
    "NFN_NATIVE_GPT_PACKED_QKV_ATTENTION",
    "NFN_NATIVE_GPT2_PACKED_QKV_ATTENTION",
)
BF16_HANDOFF_ENV_KEYS = (
    "NFN_NATIVE_GPT_BF16_QKV_GRAD_HANDOFF",
    "NFN_NATIVE_GPT2_BF16_QKV_GRAD_HANDOFF",
)


@pytest.fixture(scope="module")
def native_gpt_cli(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if shutil.which("c++") is None:
        pytest.skip("a C++ compiler is required for the native GPT preflight tests")
    output = tmp_path_factory.mktemp("native-gpt2-diff-cli") / "nfn_gpt_native_train"
    env = os.environ.copy()
    env["NFN_NATIVE_GPT_FORCE_REBUILD"] = "1"
    env["NFN_NATIVE_GPT_CXX_OPT_FLAGS"] = "-O0"
    completed = subprocess.run(
        ["bash", str(ROOT / "tools" / "build_native_gpt_cli.sh"), str(output)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return output


def _write_uint16_dataset(root: Path) -> Path:
    dataset = root / "tokens"
    dataset.mkdir()
    tokens = [index % 256 for index in range(2048)]
    payload = struct.pack("<" + "H" * len(tokens), *tokens)
    (dataset / "fineweb_train_000000.bin").write_bytes(payload)
    (dataset / "fineweb_val_000000.bin").write_bytes(payload)
    (dataset / "meta.json").write_text(
        json.dumps(
            {
                "data_format": "uint16_shards",
                "tokenizer_encoding": "gpt2",
                "tokenizer_vocab_size": 50257,
            }
        ),
        encoding="utf-8",
    )
    return dataset


def _write_graph(root: Path, preset: str) -> Path:
    model_spec = build_model_spec_from_config(
        {
            "preset": preset,
            "num_layers": 1,
            "model_dim": 768,
            "num_heads": 12,
            "vocab_size": 50257,
        },
        preview_defaults=True,
    )
    graph = build_gpt_root_graph(
        name=f"{preset}_native_fail_closed",
        model_spec=model_spec,
    )
    graph_path = root / f"{preset}.json"
    document = graph.to_dict()
    document["torch_config"]["template_spec"]["max_seq_len"] = 16
    position_config = (
        document["nodes"]["model"]["neuron_def"]["subgraph"]
        ["nodes"]["pos_embed"]["neuron_def"]["module_config"]
    )
    position_config["max_seq_len"] = 16
    graph_path.write_text(json.dumps(document), encoding="utf-8")
    return graph_path


def _materialize_proven_diff_graph(root: Path, graph: Path) -> tuple[Path, Path]:
    artifact_dir = root / f"{graph.stem}-native-proof"
    plan = plan_native_graph_training(
        graph,
        artifact_dir=artifact_dir,
        materialize=True,
    )
    assert plan.execution_ready
    assert plan.training_selector == "gpt2_diff"
    assert plan.graph_preflight_proof is not None
    assert plan.graph_preflight_proof.is_file()
    return plan.launch_graph, plan.graph_preflight_proof


def _write_proof_envelope(path: Path, contract: dict[str, object]) -> Path:
    contract_bytes = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    contract_sha256 = hashlib.sha256(contract_bytes).hexdigest()
    path.write_bytes(
        b'{"contract":'
        + contract_bytes
        + b',"contract_sha256":"'
        + contract_sha256.encode("ascii")
        + b'"}\n'
    )
    return path


def _clean_runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    for name in (*PACKED_ENV_KEYS, *BF16_HANDOFF_ENV_KEYS):
        env.pop(name, None)
    env["NFN_NATIVE_GPT_SETUP_PROGRESS"] = "0"
    env["NFN_NATIVE_GPT2_SETUP_PROGRESS"] = "0"
    return env


def _training_command(
    cli: Path,
    *,
    dataset: Path,
    graph: Path,
    proof: Path | None = None,
    output_dir: Path,
    tile_ops: Path,
    seq_len: int,
    template_name: str = "gpt2_diff",
) -> list[str]:
    command = [
        str(cli),
        "--backend",
        "tile-cuda",
        "--template-name",
        template_name,
        "--graph-file",
        str(graph),
        "--graph-fingerprint",
        hashlib.sha256(graph.read_bytes()).hexdigest(),
        "--dataset-alias",
        str(dataset),
        "--tile-ops-lib",
        str(tile_ops),
        "--output-dir",
        str(output_dir),
        "--batch-size",
        "1",
        "--train-seq-len",
        str(seq_len),
        "--train-batch-tokens",
        str(seq_len),
        "--num-layers",
        "1",
        "--max-steps",
        "1",
        "--eval-every-steps",
        "0",
        "--eval-batches",
        "0",
        "--progress-every-steps",
        "0",
    ]
    if proof is not None:
        command.extend(["--graph-preflight-proof", str(proof)])
    return command


def _assert_rejected_before_state_mutation(
    completed: subprocess.CompletedProcess[str],
    *,
    expected_error: str,
    output_dir: Path,
) -> dict[str, object]:
    assert completed.returncode == 2, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "native-transformer-lm-failed"
    assert payload["passed"] is False
    assert payload["error"] == expected_error
    assert payload["steps_completed"] == 0
    assert payload["train_microbatches_completed"] == 0
    assert payload["parameter_initialization_kernel_launches"] == 0
    assert payload["bf16_parameter_initialization_kernel_launches"] == 0
    assert payload["mixed_parameter_initialization_kernel_launches"] == 0
    assert payload["adamw_kernel_launches"] == 0
    assert payload["total_optimizer_steps_completed"] == 0
    checkpoint = payload["checkpoint"]
    assert checkpoint["requested"] is True
    assert checkpoint["checkpoint_written"] is False
    assert checkpoint["parameter_state_checkpoint_written"] is False
    assert checkpoint["optimizer_checkpoint_written"] is False
    assert not output_dir.exists()
    return payload


@pytest.mark.parametrize(
    ("seq_len", "env_name", "expected_error"),
    [
        (
            16,
            "NFN_NATIVE_GPT_PACKED_QKV_ATTENTION",
            PREFLIGHT_PREFIX
            + "packed QKV attention is disabled; "
            + "NFN_NATIVE_GPT_PACKED_QKV_ATTENTION and "
            + "NFN_NATIVE_GPT2_PACKED_QKV_ATTENTION must be unset or enabled",
        ),
        (
            8,
            None,
            PREFLIGHT_PREFIX
            + "train_seq_len must be at least 16 for packed QKV attention (got 8)",
        ),
        (
            16,
            "NFN_NATIVE_GPT_BF16_QKV_GRAD_HANDOFF",
            PREFLIGHT_PREFIX
            + "BF16 QKV gradient handoff is disabled; "
            + "NFN_NATIVE_GPT_BF16_QKV_GRAD_HANDOFF and "
            + "NFN_NATIVE_GPT2_BF16_QKV_GRAD_HANDOFF must be unset or enabled",
        ),
    ],
)
def test_graph_authored_gpt2_diff_configuration_rejects_before_state_mutation(
    native_gpt_cli: Path,
    tmp_path: Path,
    seq_len: int,
    env_name: str | None,
    expected_error: str,
) -> None:
    dataset = _write_uint16_dataset(tmp_path)
    authored_graph = _write_graph(tmp_path, "gpt2_diff")
    graph, proof = _materialize_proven_diff_graph(tmp_path, authored_graph)
    if seq_len < 16:
        # The planner correctly refuses to issue a proof for unsupported packed
        # geometry.  Rebind the otherwise reviewed contract to an exact seq8
        # snapshot so the raw C++ consumer's independent packed-size guard is
        # still exercised as defense in depth.
        graph_document = json.loads(graph.read_text(encoding="utf-8"))
        graph_document["torch_config"]["template_spec"]["max_seq_len"] = seq_len
        position_config = (
            graph_document["nodes"]["model"]["neuron_def"]["subgraph"]
            ["nodes"]["pos_embed"]["neuron_def"]["module_config"]
        )
        position_config["max_seq_len"] = seq_len
        rebound_graph = tmp_path / "gpt2-diff-seq8-source.json"
        rebound_graph.write_text(json.dumps(graph_document), encoding="utf-8")
        proof_document = json.loads(proof.read_text(encoding="utf-8"))
        rebound_contract = dict(proof_document["contract"])
        rebound_contract["geometry"] = dict(rebound_contract["geometry"])
        rebound_contract["geometry"]["max_seq_len"] = seq_len
        rebound_contract["source_graph_sha256"] = hashlib.sha256(
            rebound_graph.read_bytes()
        ).hexdigest()
        proof = _write_proof_envelope(
            tmp_path / "gpt2-diff-seq8-proof.json",
            rebound_contract,
        )
        graph = rebound_graph
    output_dir = tmp_path / "checkpoint-output"
    env = _clean_runtime_env()
    if env_name is not None:
        env[env_name] = "0"
    completed = subprocess.run(
        _training_command(
            native_gpt_cli,
            dataset=dataset,
            graph=graph,
            proof=proof,
            output_dir=output_dir,
            tile_ops=tmp_path / "missing-tile-ops.so",
            seq_len=seq_len,
        ),
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    payload = _assert_rejected_before_state_mutation(
        completed,
        expected_error=expected_error,
        output_dir=output_dir,
    )
    assert payload["graph_file"] == str(graph)
    assert payload["variant"]["differential_attention_enabled"] is True


def _build_tile_ops_without_differential_symbols(tmp_path: Path) -> Path:
    source = TRAINER_SOURCE.read_text(encoding="utf-8")
    required_initializer = source.split(
        "std::vector<std::string> required_symbols = {", 1
    )[1].split("\n    };", 1)[0]
    base_symbols = list(
        dict.fromkeys(
            re.findall(r'"(nfn_native_tile_[A-Za-z0-9_]+)"', required_initializer)
        )
    )
    assert DIFF_FORWARD_SYMBOL not in base_symbols
    assert DIFF_BACKWARD_SYMBOL not in base_symbols
    assert DIFF_RELEASE_SYMBOL not in base_symbols
    fake_source = tmp_path / "base_tile_ops.cpp"
    fake_source.write_text(
        "\n".join(f'extern "C" void {symbol}() {{}}' for symbol in base_symbols)
        + "\n",
        encoding="utf-8",
    )
    fake_library = tmp_path / "libbase_tile_ops.so"
    completed = subprocess.run(
        ["c++", "-std=c++20", "-shared", "-fPIC", str(fake_source), "-o", str(fake_library)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return fake_library


def test_graph_authored_gpt2_diff_requires_exact_tile_symbols_before_state_mutation(
    native_gpt_cli: Path,
    tmp_path: Path,
) -> None:
    dataset = _write_uint16_dataset(tmp_path)
    authored_graph = _write_graph(tmp_path, "gpt2_diff")
    graph, proof = _materialize_proven_diff_graph(tmp_path, authored_graph)
    output_dir = tmp_path / "checkpoint-output"
    fake_tile_ops = _build_tile_ops_without_differential_symbols(tmp_path)
    completed = subprocess.run(
        _training_command(
            native_gpt_cli,
            dataset=dataset,
            graph=graph,
            proof=proof,
            output_dir=output_dir,
            tile_ops=fake_tile_ops,
            seq_len=16,
        ),
        cwd=ROOT,
        env=_clean_runtime_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    _assert_rejected_before_state_mutation(
        completed,
        expected_error=(
            PREFLIGHT_PREFIX
            + "missing required Tile ABI symbols: "
            + DIFF_FORWARD_SYMBOL
            + ", "
            + DIFF_BACKWARD_SYMBOL
            + ", "
            + DIFF_RELEASE_SYMBOL
        ),
        output_dir=output_dir,
    )


def test_graph_authored_gpt2_diff_rejects_empty_verified_graph_before_tile_load(
    native_gpt_cli: Path,
    tmp_path: Path,
) -> None:
    dataset = _write_uint16_dataset(tmp_path)
    graph = tmp_path / "empty-gpt2-diff.json"
    graph.write_bytes(b"")
    output_dir = tmp_path / "checkpoint-output"
    completed = subprocess.run(
        _training_command(
            native_gpt_cli,
            dataset=dataset,
            graph=graph,
            output_dir=output_dir,
            tile_ops=tmp_path / "missing-tile-ops.so",
            seq_len=16,
        ),
        cwd=ROOT,
        env=_clean_runtime_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr.strip() == "gpt2_diff graph file must not be empty"
    assert not output_dir.exists()


def test_graph_authored_gpt2_diff_requires_canonical_planner_proof_before_plan(
    native_gpt_cli: Path,
    tmp_path: Path,
) -> None:
    dataset = _write_uint16_dataset(tmp_path)
    authored_graph = _write_graph(tmp_path, "gpt2_diff")
    graph, proof = _materialize_proven_diff_graph(tmp_path, authored_graph)
    missing_tile = tmp_path / "must-not-load-tile-ops.so"

    def run_case(
        name: str,
        selected_proof: Path | None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            _training_command(
                native_gpt_cli,
                dataset=dataset,
                graph=graph,
                proof=selected_proof,
                output_dir=tmp_path / f"{name}-output",
                tile_ops=missing_tile,
                seq_len=16,
            )
            + ["--print-plan"],
            cwd=ROOT,
            env=_clean_runtime_env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    missing = run_case("missing", None)
    assert missing.returncode == 2
    assert missing.stdout == ""
    assert missing.stderr.strip() == (
        PREFLIGHT_PREFIX
        + "--graph-preflight-proof is required for gpt2_diff graph execution "
        + "and plans"
    )

    proof_document = json.loads(proof.read_text(encoding="utf-8"))
    invalid_version_contract = dict(proof_document["contract"])
    invalid_version_contract["version"] = 0
    invalid_version = _write_proof_envelope(
        tmp_path / "invalid-version-proof.json",
        invalid_version_contract,
    )
    invalid_version_result = run_case("invalid-version", invalid_version)
    assert invalid_version_result.returncode == 2
    assert invalid_version_result.stdout == ""
    assert invalid_version_result.stderr.strip() == (
        PREFLIGHT_PREFIX
        + "invalid graph preflight proof: version must be a positive int64"
    )

    invalid_hash = tmp_path / "invalid-hash-proof.json"
    invalid_hash.write_text(
        proof.read_text(encoding="utf-8").replace(
            proof_document["contract_sha256"], "0" * 64
        ),
        encoding="utf-8",
    )
    invalid_hash_result = run_case("invalid-hash", invalid_hash)
    assert invalid_hash_result.returncode == 2
    assert invalid_hash_result.stdout == ""
    assert invalid_hash_result.stderr.strip() == (
        PREFLIGHT_PREFIX
        + "invalid graph preflight proof: contract_sha256 does not match the "
        + "raw contract bytes"
    )

    conflicting_geometry_contract = dict(proof_document["contract"])
    conflicting_geometry_contract["geometry"] = dict(
        conflicting_geometry_contract["geometry"]
    )
    conflicting_geometry_contract["geometry"]["max_seq_len"] = 32
    conflicting_geometry = _write_proof_envelope(
        tmp_path / "conflicting-geometry-proof.json",
        conflicting_geometry_contract,
    )
    conflicting_geometry_result = run_case(
        "conflicting-geometry", conflicting_geometry
    )
    assert conflicting_geometry_result.returncode == 2
    assert conflicting_geometry_result.stdout == ""
    assert conflicting_geometry_result.stderr.strip() == (
        PREFLIGHT_PREFIX
        + "invalid graph preflight proof: explicit runtime geometry conflicts "
        + "with the planner proof"
    )

    proof_symlink = tmp_path / "proof-symlink.json"
    proof_symlink.symlink_to(proof)
    symlink_result = run_case("symlink", proof_symlink)
    assert symlink_result.returncode == 2
    assert symlink_result.stdout == ""
    assert "failed to open regular non-symlink file" in symlink_result.stderr
    assert str(proof_symlink) in symlink_result.stderr

    for name in (
        "missing",
        "invalid-version",
        "invalid-hash",
        "conflicting-geometry",
        "symlink",
    ):
        assert not (tmp_path / f"{name}-output").exists()
    assert str(missing_tile) not in "".join(
        result.stderr
        for result in (
            missing,
            invalid_version_result,
            invalid_hash_result,
            conflicting_geometry_result,
            symlink_result,
        )
    )


def test_gpt2_diff_proof_rejects_dense_or_decoy_graph_and_nongelu_cli(
    native_gpt_cli: Path,
    tmp_path: Path,
) -> None:
    dataset = _write_uint16_dataset(tmp_path)
    authored_diff = _write_graph(tmp_path, "gpt2_diff")
    proven_graph, proof = _materialize_proven_diff_graph(
        tmp_path, authored_diff
    )
    dense_graph = _write_graph(tmp_path, "gpt2")

    decoy_document = json.loads(authored_diff.read_text(encoding="utf-8"))
    # Block resolution consumes the reviewed attention-family variant.  Keep
    # the template marker differential while changing that active operation to
    # dense SDPA, reproducing the exact marker/topology decoy the proof must
    # refuse.
    decoy_sdpa = decoy_document["variant_library"]["attention"]["default"][
        "nodes"
    ]["sdpa"]["neuron_def"]
    decoy_sdpa["name"] = "scaled_dot_product_attention"
    decoy_sdpa["module_type"] = "scaled_dot_product_attention"
    decoy_sdpa["module_config"] = {
        "is_causal": True,
        "backend": "sdpa",
        "dropout_p": 0.0,
    }
    decoy_graph = tmp_path / "gpt2-diff-marker-dense-active-topology.json"
    decoy_graph.write_text(json.dumps(decoy_document), encoding="utf-8")
    decoy_plan = plan_native_graph_training(decoy_graph)
    assert decoy_plan.training_selector == "gpt2_diff"
    assert decoy_plan.execution_ready is False
    assert decoy_plan.graph_preflight_proof is None
    assert any(
        "attention graph does not match its reviewed native adapter"
        in issue.message
        for issue in decoy_plan.training_issues
    )

    missing_tile = tmp_path / "must-not-load-tile-ops.so"
    for name, graph in (("dense", dense_graph), ("decoy", decoy_graph)):
        output_dir = tmp_path / f"{name}-output"
        completed = subprocess.run(
            _training_command(
                native_gpt_cli,
                dataset=dataset,
                graph=graph,
                proof=proof,
                output_dir=output_dir,
                tile_ops=missing_tile,
                seq_len=16,
            )
            + ["--print-plan"],
            cwd=ROOT,
            env=_clean_runtime_env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert completed.returncode == 2
        assert completed.stdout == ""
        assert completed.stderr.strip() == (
            PREFLIGHT_PREFIX
            + "invalid graph preflight proof: source_graph_sha256 does not "
            + "match the verified graph bytes"
        )
        assert not output_dir.exists()
        assert str(missing_tile) not in completed.stderr

    activation_output = tmp_path / "activation-output"
    activation = subprocess.run(
        _training_command(
            native_gpt_cli,
            dataset=dataset,
            graph=proven_graph,
            proof=proof,
            output_dir=activation_output,
            tile_ops=missing_tile,
            seq_len=16,
        )
        + ["--activation", "relu", "--print-plan"],
        cwd=ROOT,
        env=_clean_runtime_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert activation.returncode == 2
    assert activation.stdout == ""
    assert activation.stderr.strip() == (
        PREFLIGHT_PREFIX
        + "gpt2_diff requires GELU activation from the reviewed graph contract"
    )
    assert not activation_output.exists()

    wrong_selector_output = tmp_path / "wrong-selector-output"
    wrong_selector = subprocess.run(
        _training_command(
            native_gpt_cli,
            dataset=dataset,
            graph=dense_graph,
            proof=proof,
            output_dir=wrong_selector_output,
            tile_ops=missing_tile,
            seq_len=16,
            template_name="gpt2",
        )
        + ["--print-plan"],
        cwd=ROOT,
        env=_clean_runtime_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert wrong_selector.returncode == 2
    assert wrong_selector.stdout == ""
    assert wrong_selector.stderr.strip() == (
        "--graph-preflight-proof is only supported for gpt2_diff graph training"
    )
    assert not wrong_selector_output.exists()


def test_graph_authored_gpt2_diff_rejects_overflowing_geometry_before_tile_load(
    native_gpt_cli: Path,
    tmp_path: Path,
) -> None:
    dataset = _write_uint16_dataset(tmp_path)
    authored_graph = _write_graph(tmp_path, "gpt2_diff")
    graph, proof = _materialize_proven_diff_graph(tmp_path, authored_graph)
    document = json.loads(graph.read_text(encoding="utf-8"))
    template_spec = document["torch_config"]["template_spec"]
    template_spec["model_dim"] = 4_000_000_000
    template_spec["block_spec"]["num_heads"] = 2
    graph.write_text(json.dumps(document), encoding="utf-8")
    output_dir = tmp_path / "checkpoint-output"
    completed = subprocess.run(
        _training_command(
            native_gpt_cli,
            dataset=dataset,
            graph=graph,
            proof=proof,
            output_dir=output_dir,
            tile_ops=tmp_path / "missing-tile-ops.so",
            seq_len=16,
        ),
        cwd=ROOT,
        env=_clean_runtime_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr.strip() == (
        PREFLIGHT_PREFIX
        + "invalid graph preflight proof: source_graph_sha256 does not match "
        + "the verified graph bytes"
    )
    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("field_name", "invalid_token"),
    (
        (
            "model_dim",
            "99999999999999999999999999999999999999999999999999",
        ),
        (
            "mlp_multiplier",
            "1e999",
        ),
    ),
)
def test_graph_authored_gpt2_diff_rejects_unrepresentable_json_geometry_before_tile_load(
    native_gpt_cli: Path,
    tmp_path: Path,
    field_name: str,
    invalid_token: str,
) -> None:
    dataset = _write_uint16_dataset(tmp_path)
    authored_graph = _write_graph(tmp_path, "gpt2_diff")
    graph, proof = _materialize_proven_diff_graph(tmp_path, authored_graph)
    document = json.loads(graph.read_text(encoding="utf-8"))
    template_spec = document["torch_config"]["template_spec"]
    marker = "__INVALID_GRAPH_NUMBER__"
    if field_name == "mlp_multiplier":
        template_spec["block_spec"][field_name] = marker
    else:
        template_spec[field_name] = marker
    serialized = json.dumps(document)
    marker_token = json.dumps(marker)
    assert serialized.count(marker_token) == 1
    graph.write_text(
        serialized.replace(marker_token, invalid_token),
        encoding="utf-8",
    )
    output_dir = tmp_path / "checkpoint-output"
    missing_tile = tmp_path / "must-not-load-tile-ops.so"
    completed = subprocess.run(
        _training_command(
            native_gpt_cli,
            dataset=dataset,
            graph=graph,
            proof=proof,
            output_dir=output_dir,
            tile_ops=missing_tile,
            seq_len=16,
        ),
        cwd=ROOT,
        env=_clean_runtime_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr.strip() == (
        PREFLIGHT_PREFIX
        + "invalid graph preflight proof: source_graph_sha256 does not match "
        + "the verified graph bytes"
    )
    assert str(missing_tile) not in completed.stderr
    assert not output_dir.exists()


def test_graph_authored_dense_gpt_keeps_ordinary_seq8_fallback(
    native_gpt_cli: Path,
    tmp_path: Path,
) -> None:
    dataset = _write_uint16_dataset(tmp_path)
    graph = _write_graph(tmp_path, "gpt2")
    env = _clean_runtime_env()
    env["NFN_NATIVE_GPT_PACKED_QKV_ATTENTION"] = "0"
    completed = subprocess.run(
        _training_command(
            native_gpt_cli,
            dataset=dataset,
            graph=graph,
            output_dir=tmp_path / "unused-output",
            tile_ops=TILE_OPS,
            seq_len=8,
            template_name="gpt2",
        )
        + ["--print-plan"],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "native-transformer-lm-ready"
    assert payload["passed"] is True
    assert payload["error"] == ""
    assert payload["packed_qkv_attention_enabled"] is False
    assert payload["schedule"]["lr_schedule_total_steps"] == 0
    assert payload["schedule"]["lr_schedule_total_steps_explicit"] is False
    assert payload["schedule"]["effective_lr_schedule_total_steps"] == 1
    assert payload["optimizer"]["effective_lr_schedule_total_steps"] == 1
    kernel_abis = {
        stage["kernel_abi"] for stage in payload["training_step_plan"]["stages"]
    }
    assert "nfn_native_tile_scaled_dot_product_attention_float32" in kernel_abis
    assert (
        "nfn_native_tile_scaled_dot_product_attention_backward_to_qkv_from_merged_grad_float32"
        in kernel_abis
    )
    assert DIFF_FORWARD_SYMBOL not in kernel_abis
    assert DIFF_BACKWARD_SYMBOL not in kernel_abis


def test_unbound_gpt2_moa_selector_print_command_remains_compatible(
    native_gpt_cli: Path,
) -> None:
    completed = subprocess.run(
        [
            str(native_gpt_cli),
            "--backend",
            "tile-cuda",
            "--template-name",
            "gpt2_moa",
            "--print-command",
        ],
        cwd=ROOT,
        env=_clean_runtime_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--template-name gpt2_moa" in completed.stdout
    assert "graph-fingerprint" not in completed.stderr


@pytest.mark.parametrize(
    "value_arg",
    (
        "--lr-schedule=constant",
        "--learning-rate-schedule=constant",
        "--final-lr-fraction=0.25",
        "--learning-rate-decay-frac=0.25",
        "--learning-rate-decay-fraction=0.25",
    ),
)
def test_raw_native_gpt_parser_accepts_quality_value_equality_aliases(
    native_gpt_cli: Path,
    value_arg: str,
) -> None:
    completed = subprocess.run(
        [str(native_gpt_cli), value_arg, "--help"],
        cwd=ROOT,
        env=_clean_runtime_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Usage:" in completed.stdout


def test_graph_authored_gpt2_diff_plan_and_tile_check_require_exact_packed_abis(
    native_gpt_cli: Path,
    tmp_path: Path,
) -> None:
    if not TILE_OPS.is_file():
        pytest.skip("native Tile ops library is not built")
    authored_graph = _write_graph(tmp_path, "gpt2_diff")
    graph, proof = _materialize_proven_diff_graph(tmp_path, authored_graph)
    completed = subprocess.run(
        [
            str(native_gpt_cli),
            "--backend",
            "tile-cuda",
            "--template-name",
            "gpt2_diff",
            "--graph-file",
            str(graph),
            "--graph-fingerprint",
            hashlib.sha256(graph.read_bytes()).hexdigest(),
            "--graph-preflight-proof",
            str(proof),
            "--train-seq-len",
            "16",
            "--num-layers",
            "1",
            "--check-tile-ops",
            "--tile-ops-lib",
            str(TILE_OPS),
        ],
        cwd=ROOT,
        env=_clean_runtime_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["passed"] is True
    checked_symbols = {
        item["name"]: item["found"] for item in payload["tile_ops_check"]["symbols"]
    }
    assert checked_symbols[DIFF_FORWARD_SYMBOL] is True
    assert checked_symbols[DIFF_BACKWARD_SYMBOL] is True
    assert checked_symbols[DIFF_RELEASE_SYMBOL] is True
    kernel_abis = {
        stage["kernel_abi"] for stage in payload["training_step_plan"]["stages"]
    }
    assert DIFF_FORWARD_SYMBOL in kernel_abis
    assert DIFF_BACKWARD_SYMBOL in kernel_abis
    assert "nfn_native_tile_scaled_dot_product_attention_float32" not in kernel_abis


def test_gpt2_diff_preflight_is_statically_ordered_before_tile_load_and_runtime_fallback() -> None:
    source = TRAINER_SOURCE.read_text(encoding="utf-8")
    main_body = source.split("int main(int argc, char** argv)", 1)[1]
    assert main_body.index("verify_gpt2_diff_graph_preflight_proof(") < main_body.index(
        "custom_graph_template_metadata(cfg)"
    )
    proof_verifier = source.split(
        "bool verify_gpt2_diff_graph_preflight_proof(", 1
    )[1].split("fs::path native_gpt_moa_metadata_path", 1)[0]
    assert "read_regular_nofollow_file(" in proof_verifier
    assert '{"contract", "contract_sha256"}' in proof_verifier
    assert "contract_sha256 does not match the raw contract bytes" in proof_verifier
    assert "contract is not canonical compact sorted JSON" in proof_verifier
    assert 'proof.training_selector != "gpt2_diff"' in proof_verifier
    assert "proof.source_graph_sha256 != cfg->verified_graph_fingerprint" in proof_verifier
    assert "gpt2_diff_verified_proof_geometry" in source
    run_body = source.split("int run_transformer_lm_training_json(", 1)[1]
    assert run_body.index("gpt2_diff_packed_configuration_error(cfg)") < run_body.index(
        'run_setup_timed("setup.load_tile_ops"'
    )
    assert "kGpt2DiffPackedTileSymbols.begin()" in run_body
    assert (
        "variant.use_differential_attention && !packed_qkv_attention_enabled"
        in run_body
    )
    assert "internal packed-QKV invariant was not active during forward" in run_body
    assert "internal packed-QKV invariant was not active during backward" in run_body
    stage_plan = source.split("std::vector<StagePlan> build_gpt2_stage_plan", 1)[1].split(
        "void print_stage_plan_json", 1
    )[0]
    assert "? kGpt2DiffPackedForwardSymbol" in stage_plan
    assert "? kGpt2DiffPackedBackwardSymbol" in stage_plan
