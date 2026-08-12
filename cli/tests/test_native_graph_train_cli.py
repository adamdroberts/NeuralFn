from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from unittest import mock

import pytest

from neuralfn.torch_templates import build_gpt_root_graph, build_model_spec_from_config


ROOT = Path(__file__).resolve().parents[2]


def _load_cli():
    spec = importlib.util.spec_from_file_location(
        "nfn_native_graph_train_cli_test",
        ROOT / "cli" / "nfn.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_preset_graph(path: Path, preset: str) -> bytes:
    model_spec = build_model_spec_from_config(
        {
            "preset": preset,
            "model_dim": 32,
            "num_layers": 1,
            "num_heads": 4,
            "num_kv_heads": 4,
            "multiple_of": (
                None
                if preset in {"moe", "mixllama", "mixllama_fast"}
                else 16
            ),
            "vocab_size": 50257,
        },
        preview_defaults=True,
    )
    payload = build_gpt_root_graph(name=f"{preset}_cli", model_spec=model_spec).to_dict()
    source = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
    path.write_bytes(source)
    return source


def test_graph_print_command_uses_validated_selector_and_geometry(tmp_path: Path) -> None:
    cli = _load_cli()
    graph = tmp_path / "moa.json"
    _write_preset_graph(graph, "gpt2_moa")
    stdout = io.StringIO()

    with (
        mock.patch.object(
            cli,
            "_resolve_direct_native_train_cli",
            return_value="/tmp/nfn_gpt_native_train_linked",
        ),
        mock.patch.object(cli, "_resolve_direct_native_train_family_cli", return_value=None),
        mock.patch.object(cli.subprocess, "run") as run,
        mock.patch.object(cli.subprocess, "Popen") as popen,
        contextlib.redirect_stdout(stdout),
    ):
        return_code = cli._direct_native_train_cli_main(
            [
                "train",
                "--base-model",
                "llama",
                "--template-name",
                "nanogpt",
                "--graph-file",
                str(graph),
                "--num-layers",
                "99",
                "--train-seq-len",
                "4096",
                "--activation",
                "relu",
                "--native-cuda-print-command",
            ]
        )

    assert return_code == 0
    command = stdout.getvalue().strip()
    assert "--model-family gpt2" in command
    assert f"--graph-file {graph.resolve()}" in command
    assert "--template-name gpt2_moa" in command
    assert "--num-layers 1" in command
    assert "--train-seq-len 1024" in command
    assert "--native-cuda-activation moa" in command
    assert "nanogpt" not in command
    assert "4096" not in command
    run.assert_not_called()
    popen.assert_not_called()


def test_canonical_llama_graph_materializes_and_routes_family_trainer(tmp_path: Path) -> None:
    cli = _load_cli()
    graph = tmp_path / "llama.json"
    source_bytes = _write_preset_graph(graph, "llama")
    output_dir = tmp_path / "run"
    observed: dict[str, object] = {}
    stderr = io.StringIO()

    def fake_run(command, env, **kwargs):
        observed["command"] = list(command)
        observed["env"] = dict(env)
        observed["kwargs"] = dict(kwargs)
        return 0

    with (
        mock.patch.object(cli, "_resolve_direct_native_train_cli") as generic_resolver,
        mock.patch.object(
            cli,
            "_resolve_direct_native_train_family_cli",
            return_value="/tmp/nfn_llama_native_train",
        ) as family_resolver,
        mock.patch.object(cli, "_run_native_train_with_progress", side_effect=fake_run) as runner,
        mock.patch.object(cli.subprocess, "run") as subprocess_run,
        mock.patch.object(cli.subprocess, "Popen") as popen,
        contextlib.redirect_stderr(stderr),
    ):
        return_code = cli._direct_native_train_cli_main(
            [
                "train",
                "--base-model",
                "gpt2",
                "--graph-file",
                str(graph),
                "--dataset-alias",
                "/tmp/native-cache",
                "--output-dir",
                str(output_dir),
                "--max-steps",
                "1",
            ]
        )

    assert return_code == 0
    runner.assert_called_once()
    subprocess_run.assert_not_called()
    popen.assert_not_called()
    generic_resolver.assert_not_called()
    assert family_resolver.call_count >= 1
    command = observed["command"]
    assert isinstance(command, list)
    assert command[0] == "/tmp/nfn_llama_native_train"
    assert "--train-llama-dataset-loop" in command
    assert "--train-transformer-lm" not in command
    fingerprint = hashlib.sha256(source_bytes).hexdigest()
    assert command[command.index("--graph-fingerprint") + 1] == fingerprint
    artifact = output_dir / "native-ir"
    snapshot = artifact / "source-graph.json"
    assert snapshot.read_bytes() == source_bytes
    assert command[command.index("--graph-file") + 1] == str(snapshot.resolve())
    assert command[command.index("--template-name") + 1] == "llama"
    assert command[command.index("--model-dim") + 1] == "32"
    assert command[command.index("--num-heads") + 1] == "4"
    assert command[command.index("--num-kv-heads") + 1] == "4"
    assert "[nfn-native-graph]" in stderr.getvalue()


def test_llama_fast_graph_routes_with_canonical_native_identity(tmp_path: Path) -> None:
    cli = _load_cli()
    graph = tmp_path / "llama-fast.json"
    source_bytes = _write_preset_graph(graph, "llama_fast")
    output_dir = tmp_path / "run"
    observed: dict[str, object] = {}
    stderr = io.StringIO()

    def fake_run(command, env, **kwargs):
        observed["command"] = list(command)
        return 0

    with (
        mock.patch.object(
            cli,
            "_resolve_direct_native_train_family_cli",
            return_value="/tmp/nfn_llama_native_train",
        ),
        mock.patch.object(cli, "_run_native_train_with_progress", side_effect=fake_run),
        mock.patch.object(cli.subprocess, "run") as subprocess_run,
        contextlib.redirect_stderr(stderr),
    ):
        return_code = cli._direct_native_train_cli_main(
            [
                "train",
                "--base-model",
                "llama",
                "--preset",
                "llama_fast",
                "--graph-file",
                str(graph),
                "--dataset-alias",
                "/tmp/native-cache",
                "--output-dir",
                str(output_dir),
                "--max-steps",
                "1",
            ]
        )

    assert return_code == 0
    subprocess_run.assert_not_called()
    command = observed["command"]
    assert isinstance(command, list)
    assert command[0] == "/tmp/nfn_llama_native_train"
    assert command[command.index("--template-name") + 1] == "llama"
    fingerprint = hashlib.sha256(source_bytes).hexdigest()
    assert command[command.index("--graph-fingerprint") + 1] == fingerprint
    plan = json.loads(
        (output_dir / "native-ir" / "native-training-plan.json").read_text(
            encoding="utf-8"
        )
    )
    assert plan["training_selector"] == "llama_fast"
    provenance = plan["artifact_metadata"]["architecture_provenance"]
    assert provenance["source_preset"] == "llama_fast"
    assert provenance["source_runtime"] == "compile"
    assert provenance["native_template_name"] == "llama"
    assert "selector=llama_fast" in stderr.getvalue()


@pytest.mark.parametrize(
    ("preset", "selector", "native_template"),
    (
        ("moe", "moe", "mixllama"),
        ("mixllama", "mixllama", "mixllama"),
        ("mixllama_fast", "mixllama_fast", "mixllama-fast"),
    ),
)
def test_standard_moe_graph_routes_exact_family_action_and_geometry(
    tmp_path: Path,
    preset: str,
    selector: str,
    native_template: str,
) -> None:
    cli = _load_cli()
    graph = tmp_path / f"{preset}.json"
    source_bytes = _write_preset_graph(graph, preset)
    output_dir = tmp_path / f"run-{preset}"
    observed: dict[str, object] = {}

    def fake_run(command, env, **kwargs):
        observed["command"] = list(command)
        return 0

    with (
        mock.patch.object(
            cli,
            "_resolve_direct_native_train_family_cli",
            return_value="/tmp/nfn_mixllama_native_train",
        ),
        mock.patch.object(cli, "_run_native_train_with_progress", side_effect=fake_run),
        mock.patch.object(cli.subprocess, "run") as subprocess_run,
    ):
        return_code = cli._direct_native_train_cli_main(
            [
                "train",
                "--base-model",
                "gpt2",
                "--graph-file",
                str(graph),
                "--dataset-alias",
                "/tmp/native-cache",
                "--output-dir",
                str(output_dir),
                "--max-steps",
                "1",
            ]
        )

    assert return_code == 0
    subprocess_run.assert_not_called()
    command = observed["command"]
    assert isinstance(command, list)
    assert command[0] == "/tmp/nfn_mixllama_native_train"
    assert command.count("--train-moe-dataset-loop") == 1
    assert "--train-transformer-lm" not in command
    assert command[command.index("--template-name") + 1] == native_template
    assert command[command.index("--multiple-of") + 1] == "0"
    assert command[command.index("--router-aux-loss-coef") + 1] == "0.01"
    assert command[command.index("--layers-per-expert") + 1] == "1"
    fingerprint = hashlib.sha256(source_bytes).hexdigest()
    assert command[command.index("--graph-fingerprint") + 1] == fingerprint
    plan = json.loads(
        (output_dir / "native-ir" / "native-training-plan.json").read_text(
            encoding="utf-8"
        )
    )
    assert plan["training_selector"] == selector
    assert plan["artifact_metadata"]["architecture_provenance"]["checkpoint_identity"] == "mixllama"


def test_unreviewed_llama_neighbor_stops_before_resolver_or_subprocess(tmp_path: Path) -> None:
    cli = _load_cli()
    graph = tmp_path / "llama-fast-megakernel.json"
    _write_preset_graph(graph, "llama_fast_megakernel")
    stderr = io.StringIO()

    with (
        mock.patch.object(cli, "_resolve_direct_native_train_cli") as resolver,
        mock.patch.object(cli, "_resolve_direct_native_train_family_cli") as family_resolver,
        mock.patch.object(cli.subprocess, "run") as run,
        mock.patch.object(cli.subprocess, "Popen") as popen,
        contextlib.redirect_stderr(stderr),
    ):
        return_code = cli._direct_native_train_cli_main(
            [
                "train",
                "--base-model",
                "llama",
                "--graph-file",
                str(graph),
                "--dataset-alias",
                "/tmp/native-cache",
            ]
        )

    assert return_code == 2
    payload = json.loads(stderr.getvalue())
    assert payload["status"] == "native-graph-training-incompatible"
    assert payload["trainer_family"] == "llama"
    assert payload["execution_ready"] is False
    assert payload["issues"][0]["code"] == "architecture_persistence_unproven"
    assert payload["issues"][0]["path"].startswith("root/nodes/model")
    assert "diagnostic only" in payload["issues"][0]["message"]
    resolver.assert_not_called()
    family_resolver.assert_not_called()
    run.assert_not_called()
    popen.assert_not_called()


@pytest.mark.parametrize(
    "caller_proof",
    (
        ("--graph-preflight-proof", "/tmp/caller-decoy.json"),
        ("--graph-preflight-proof=/tmp/caller-decoy.json",),
    ),
)
def test_gpt2_diff_materializes_and_forwards_only_the_planner_proof(
    tmp_path: Path,
    caller_proof: tuple[str, ...],
) -> None:
    cli = _load_cli()
    graph = tmp_path / "gpt2-diff.json"
    source_bytes = _write_preset_graph(graph, "gpt2_diff")
    output_dir = tmp_path / "run"
    observed: dict[str, object] = {}
    stderr = io.StringIO()

    def fake_run(command, env, **kwargs):
        observed["command"] = list(command)
        return 0

    with (
        mock.patch.object(
            cli,
            "_resolve_direct_native_train_cli",
            return_value="/tmp/nfn_gpt_native_train_linked",
        ) as resolver,
        mock.patch.object(cli, "_resolve_direct_native_train_family_cli") as family_resolver,
        mock.patch.object(
            cli, "_run_native_train_with_progress", side_effect=fake_run
        ) as runner,
        mock.patch.object(cli.subprocess, "run") as subprocess_run,
        mock.patch.object(cli.subprocess, "Popen") as popen,
        contextlib.redirect_stderr(stderr),
    ):
        return_code = cli._direct_native_train_cli_main(
            [
                "train",
                "--base-model",
                "gpt2",
                "--graph-file",
                str(graph),
                "--dataset-alias",
                "/tmp/native-cache",
                "--output-dir",
                str(output_dir),
                "--max-steps",
                "1",
                *caller_proof,
            ]
        )

    assert return_code == 0
    runner.assert_called_once()
    subprocess_run.assert_not_called()
    popen.assert_not_called()
    assert resolver.call_count >= 1
    family_resolver.assert_not_called()
    command = observed["command"]
    assert isinstance(command, list)
    artifact = output_dir / "native-ir"
    proof = artifact / "native-training-proof.json"
    snapshot = artifact / "source-graph.json"
    assert proof.is_file()
    assert snapshot.read_bytes() == source_bytes
    assert command.count("--graph-preflight-proof") == 1
    assert command[command.index("--graph-preflight-proof") + 1] == str(
        proof.resolve()
    )
    assert "/tmp/caller-decoy.json" not in command
    assert command[command.index("--graph-file") + 1] == str(snapshot.resolve())
    assert command[command.index("--graph-fingerprint") + 1] == hashlib.sha256(
        source_bytes
    ).hexdigest()
    assert "selector=gpt2_diff" in stderr.getvalue()


@pytest.mark.parametrize(
    "malformed_tokens",
    (
        ("--graph-preflight-proof", "--native-cuda-dry-run"),
        ("--num-layers", "--native-cuda-print-command"),
        ("--graph-preflight-proof",),
    ),
)
def test_gpt2_diff_dangling_authoritative_value_never_swallows_inspection_or_launches(
    tmp_path: Path,
    malformed_tokens: tuple[str, ...],
) -> None:
    cli = _load_cli()
    graph = tmp_path / "gpt2-diff.json"
    _write_preset_graph(graph, "gpt2_diff")
    output_dir = tmp_path / "must-not-exist"
    stderr = io.StringIO()

    with (
        mock.patch.object(cli, "_resolve_direct_native_train_cli") as resolver,
        mock.patch.object(cli, "_resolve_direct_native_train_family_cli") as family_resolver,
        mock.patch.object(cli, "_run_native_train_with_progress") as runner,
        mock.patch.object(cli.subprocess, "run") as subprocess_run,
        mock.patch.object(cli.subprocess, "Popen") as popen,
        contextlib.redirect_stderr(stderr),
    ):
        return_code = cli._direct_native_train_cli_main(
            [
                "train",
                "--graph-file",
                str(graph),
                "--output-dir",
                str(output_dir),
                *malformed_tokens,
            ]
        )

    assert return_code == 2
    assert "requires a value" in stderr.getvalue()
    assert not output_dir.exists()
    resolver.assert_not_called()
    family_resolver.assert_not_called()
    runner.assert_not_called()
    subprocess_run.assert_not_called()
    popen.assert_not_called()


def test_graph_value_replacement_rejects_a_dangling_split_alias() -> None:
    cli = _load_cli()

    with pytest.raises(ValueError, match="--graph-preflight-proof requires a value"):
        cli._replace_value_argument(
            ["nfn_gpt_native_train", "--graph-preflight-proof", "--dry-run"],
            ("--graph-preflight-proof",),
            "--graph-preflight-proof",
            "/trusted/native-training-proof.json",
        )


@pytest.mark.parametrize(
    "inspection_flag",
    ("--native-cuda-dry-run", "--native-cuda-print-plan"),
)
def test_gpt2_diff_planner_inspection_is_nonmutating_and_never_invokes_cpp(
    tmp_path: Path,
    inspection_flag: str,
) -> None:
    cli = _load_cli()
    graph = tmp_path / "gpt2-diff.json"
    _write_preset_graph(graph, "gpt2_diff")
    output_dir = tmp_path / "inspection"
    stdout = io.StringIO()

    with (
        mock.patch.object(cli, "_resolve_direct_native_train_cli") as resolver,
        mock.patch.object(cli, "_resolve_direct_native_train_family_cli") as family_resolver,
        mock.patch.object(cli, "_run_native_train_with_progress") as runner,
        mock.patch.object(cli.subprocess, "run") as subprocess_run,
        mock.patch.object(cli.subprocess, "Popen") as popen,
        contextlib.redirect_stdout(stdout),
    ):
        return_code = cli._direct_native_train_cli_main(
            [
                "train",
                "--graph-file",
                str(graph),
                "--output-dir",
                str(output_dir),
                inspection_flag,
            ]
        )

    assert return_code == 0
    payload = json.loads(stdout.getvalue())
    assert payload["training_selector"] == "gpt2_diff"
    assert payload["execution_ready"] is True
    assert payload["graph_preflight_proof"] is None
    assert "--graph-preflight-proof" not in payload["trainer_arguments"]
    assert not output_dir.exists()
    resolver.assert_not_called()
    family_resolver.assert_not_called()
    runner.assert_not_called()
    subprocess_run.assert_not_called()
    popen.assert_not_called()


def test_gpt2_diff_print_command_reports_required_materialization_without_fake_argv(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    graph = tmp_path / "gpt2-diff.json"
    _write_preset_graph(graph, "gpt2_diff")
    output_dir = tmp_path / "print-command"
    stdout = io.StringIO()

    with (
        mock.patch.object(cli, "_resolve_direct_native_train_cli") as resolver,
        mock.patch.object(cli, "_run_native_train_with_progress") as runner,
        mock.patch.object(cli.subprocess, "run") as subprocess_run,
        contextlib.redirect_stdout(stdout),
    ):
        return_code = cli._direct_native_train_cli_main(
            [
                "train",
                "--graph-file",
                str(graph),
                "--output-dir",
                str(output_dir),
                "--native-cuda-print-command",
            ]
        )

    assert return_code == 0
    payload = json.loads(stdout.getvalue())
    assert payload["status"] == "native-graph-training-materialization-required"
    assert payload["training_selector"] == "gpt2_diff"
    assert payload["executable_command"] is None
    assert "native-training-proof.json" in payload["workflow"]
    assert not output_dir.exists()
    resolver.assert_not_called()
    runner.assert_not_called()
    subprocess_run.assert_not_called()


@pytest.mark.parametrize("mode_flag", ("--startup-only", "--native-cuda-check-tile-ops"))
def test_gpt2_diff_executable_preflight_modes_receive_materialized_proof(
    tmp_path: Path,
    mode_flag: str,
) -> None:
    cli = _load_cli()
    graph = tmp_path / "gpt2-diff.json"
    _write_preset_graph(graph, "gpt2_diff")
    output_dir = tmp_path / mode_flag.removeprefix("--")
    observed: dict[str, list[str]] = {}

    def fake_run(command, env, **kwargs):
        observed["command"] = list(command)
        return 0

    with (
        mock.patch.object(
            cli,
            "_resolve_direct_native_train_cli",
            return_value="/tmp/nfn_gpt_native_train_linked",
        ),
        mock.patch.object(cli, "_run_native_train_with_progress", side_effect=fake_run),
        mock.patch.object(cli.subprocess, "run") as subprocess_run,
    ):
        return_code = cli._direct_native_train_cli_main(
            [
                "train",
                "--graph-file",
                str(graph),
                "--dataset-alias",
                "/tmp/native-cache",
                "--output-dir",
                str(output_dir),
                mode_flag,
            ]
        )

    assert return_code == 0
    subprocess_run.assert_not_called()
    command = observed["command"]
    proof = output_dir / "native-ir" / "native-training-proof.json"
    assert proof.is_file()
    assert command.count("--graph-preflight-proof") == 1
    assert command[command.index("--graph-preflight-proof") + 1] == str(
        proof.resolve()
    )


def test_canonical_llama_graph_dry_run_cannot_enter_dataset_training(tmp_path: Path) -> None:
    cli = _load_cli()
    graph = tmp_path / "llama.json"
    _write_preset_graph(graph, "llama")
    completed = mock.Mock(returncode=0)

    with (
        mock.patch.object(
            cli,
            "_resolve_direct_native_train_family_cli",
            return_value="/tmp/nfn_llama_native_train",
        ),
        mock.patch.object(cli.subprocess, "run", return_value=completed) as run,
        mock.patch.object(cli, "_run_native_train_with_progress") as runner,
    ):
        return_code = cli._direct_native_train_cli_main(
            [
                "train",
                "--base-model",
                "llama",
                "--graph-file",
                str(graph),
                "--dataset-alias",
                "/definitely/not/a/dataset",
                "--native-cuda-dry-run",
            ]
        )

    assert return_code == 0
    runner.assert_not_called()
    command = list(run.call_args.args[0])
    assert "--dry-run" in command
    assert "--train-llama-dataset-loop" not in command


@pytest.mark.parametrize(
    "caller_action",
    ["--train-llama-dataset-loop", "--native-cuda-smoke-llama-loop"],
)
def test_graph_training_rejects_caller_selected_actions_before_resolution(
    tmp_path: Path,
    caller_action: str,
) -> None:
    cli = _load_cli()
    graph = tmp_path / "llama.json"
    _write_preset_graph(graph, "llama")
    stderr = io.StringIO()

    with (
        mock.patch.object(cli, "_resolve_direct_native_train_cli") as resolver,
        mock.patch.object(cli, "_resolve_direct_native_train_family_cli") as family_resolver,
        mock.patch.object(cli.subprocess, "run") as run,
        mock.patch.object(cli.subprocess, "Popen") as popen,
        contextlib.redirect_stderr(stderr),
    ):
        return_code = cli._direct_native_train_cli_main(
            [
                "train",
                "--base-model",
                "llama",
                "--graph-file",
                str(graph),
                caller_action,
            ]
        )

    assert return_code == 2
    assert caller_action in stderr.getvalue()
    resolver.assert_not_called()
    family_resolver.assert_not_called()
    run.assert_not_called()
    popen.assert_not_called()


def test_unsupported_node_graph_stops_before_subprocess_with_exact_path(tmp_path: Path) -> None:
    cli = _load_cli()
    graph = tmp_path / "unsupported.json"
    _write_preset_graph(graph, "gpt2")
    payload = json.loads(graph.read_text(encoding="utf-8"))
    payload["nodes"]["model"]["neuron_def"]["subgraph"]["nodes"]["token_embed"][
        "neuron_def"
    ]["module_type"] = "unregistered_future_op"
    graph.write_text(json.dumps(payload), encoding="utf-8")
    stderr = io.StringIO()

    with (
        mock.patch.object(cli.subprocess, "run") as run,
        mock.patch.object(cli.subprocess, "Popen") as popen,
        contextlib.redirect_stderr(stderr),
    ):
        return_code = cli._direct_native_train_cli_main(
            ["train", "--base-model", "gpt2", "--graph-file", str(graph)]
        )

    assert return_code == 2
    result = json.loads(stderr.getvalue())
    issue = result["issues"][0]
    assert issue["code"] == "unsupported_module"
    assert issue["operation"] == "unregistered_future_op"
    assert issue["path"].endswith("/nodes/token_embed")
    run.assert_not_called()
    popen.assert_not_called()


def test_actual_graph_train_snapshots_validated_bytes_and_routes_snapshot(tmp_path: Path) -> None:
    cli = _load_cli()
    graph = tmp_path / "gpt2.json"
    source_bytes = _write_preset_graph(graph, "gpt2")
    output_dir = tmp_path / "run"
    observed: dict[str, object] = {}
    stderr = io.StringIO()

    def fake_run(command, env, **kwargs):
        observed["command"] = list(command)
        observed["env"] = dict(env)
        observed["kwargs"] = dict(kwargs)
        return 0

    with (
        mock.patch.object(
            cli,
            "_resolve_direct_native_train_cli",
            return_value="/tmp/nfn_gpt_native_train_linked",
        ),
        mock.patch.object(cli, "_resolve_direct_native_train_family_cli", return_value=None),
        mock.patch.object(cli, "_run_native_train_with_progress", side_effect=fake_run) as runner,
        mock.patch.object(cli.subprocess, "run") as subprocess_run,
        mock.patch.object(cli.subprocess, "Popen") as popen,
        contextlib.redirect_stderr(stderr),
    ):
        return_code = cli._direct_native_train_cli_main(
            [
                "train",
                "--base-model",
                "gpt2",
                "--graph-file",
                str(graph),
                "--dataset-alias",
                "/tmp/native-cache",
                "--output-dir",
                str(output_dir),
                "--max-steps",
                "1",
            ]
        )

    assert return_code == 0
    runner.assert_called_once()
    subprocess_run.assert_not_called()
    popen.assert_not_called()
    artifact = output_dir / "native-ir"
    snapshot = artifact / "source-graph.json"
    assert snapshot.read_bytes() == source_bytes
    plan = json.loads((artifact / "native-training-plan.json").read_text(encoding="utf-8"))
    assert plan["execution_ready"] is True
    assert plan["trainer_consumes_native_ir"] is False
    assert plan["graph_preflight_enforced"] is True
    assert plan["training_selector"] == "gpt2"
    command = observed["command"]
    assert isinstance(command, list)
    assert command[command.index("--graph-file") + 1] == str(snapshot.resolve())
    assert command[command.index("--template-name") + 1] == "gpt2"
    assert command[command.index("--num-layers") + 1] == "1"
    assert "[nfn-native-graph]" in stderr.getvalue()
    assert str(artifact / "native-training-plan.json") in stderr.getvalue()


def test_graph_dry_run_preflights_without_materializing_training_output(tmp_path: Path) -> None:
    cli = _load_cli()
    graph = tmp_path / "gpt2.json"
    _write_preset_graph(graph, "gpt2")
    output_dir = tmp_path / "dry-run"
    observed: dict[str, object] = {}

    def fake_capture(command, env):
        observed["command"] = list(command)
        return 0

    with (
        mock.patch.object(
            cli,
            "_resolve_direct_native_train_cli",
            return_value="/tmp/nfn_gpt_native_train_linked",
        ),
        mock.patch.object(cli, "_resolve_direct_native_train_family_cli", return_value=None),
        mock.patch.object(cli, "_run_dense_gpt_compiled_cli_capture", side_effect=fake_capture),
    ):
        return_code = cli._direct_native_train_cli_main(
            [
                "train",
                "--base-model",
                "gpt2",
                "--graph-file",
                str(graph),
                "--output-dir",
                str(output_dir),
                "--native-cuda-dry-run",
            ]
        )

    assert return_code == 0
    assert not output_dir.exists()
    command = observed["command"]
    assert isinstance(command, list)
    assert command[command.index("--graph-file") + 1] == str(graph.resolve())
