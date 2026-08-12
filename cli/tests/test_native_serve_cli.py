from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
NFN = ROOT / "cli" / "nfn.py"


def _load_cli_module():
    spec = importlib.util.spec_from_file_location("nfn_native_serve_cli_test", NFN)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_serve_help_is_available_without_loading_editor_runtime() -> None:
    completed = subprocess.run(
        [sys.executable, str(NFN), "infer", "--serve", "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "nfn infer --serve" in completed.stdout
    assert "--checkpoint ARTIFACT" in completed.stdout
    assert "--host HOST" in completed.stdout
    assert "--queue-capacity N" in completed.stdout
    assert "--session-limit N" in completed.stdout
    assert "--kv-cache {off,auto,full,turboquant}" in completed.stdout
    assert "--turboquant-attention-backend {cpu,tile-cuda}" in completed.stdout
    assert "--tile-ops-lib PATH" in completed.stdout
    assert "--cuda-runtime-lib PATH_OR_SONAME" in completed.stdout
    assert "--cuda-device INDEX" in completed.stdout
    assert "--api-key-file PATH" in completed.stdout
    assert "--state-db PATH" in completed.stdout
    assert "--prefix-cache-capacity N" in completed.stdout
    assert "--allow-unauthenticated-remote" in completed.stdout


def test_native_serve_cli_plumbs_validated_config_without_starting_graph_runtime(
    monkeypatch,
) -> None:
    module = _load_cli_module()
    captured = []
    monkeypatch.setattr(
        "neuralfn.native_serve.run_native_inference_server",
        lambda config: captured.append(config),
    )

    result = module._native_serve_main(
        [
            "infer",
            "--checkpoint",
            "artifact-dir",
            "--serve",
            "--host",
            "127.0.0.1",
            "--port",
            "9001",
            "--served-model-name",
            "local-model",
            "--queue-capacity",
            "3",
            "--session-limit",
            "2",
            "--max-output-tokens",
            "64",
            "--kv-cache",
            "turboquant",
            "--turboquant-profile",
            "qjl-3.5",
            "--turboquant-attention-backend",
            "tile-cuda",
            "--tile-ops-lib",
            "libtile-strict.so",
            "--cuda-runtime-lib",
            "libcudart.so.13",
            "--cuda-device",
            "2",
            "--chat-template",
            "plain_roles",
            "--state-db",
            "serve-state.sqlite3",
        ]
    )

    assert result == 0
    assert len(captured) == 1
    config = captured[0]
    assert config.artifact == Path("artifact-dir")
    assert config.host == "127.0.0.1"
    assert config.port == 9001
    assert config.served_model_name == "local-model"
    assert config.queue_capacity == 3
    assert config.session_limit == 2
    assert config.max_output_tokens == 64
    assert config.kv_cache.mode == "turboquant"
    assert config.kv_cache.turboquant_profile == "qjl-3.5"
    assert config.kv_cache.turboquant_attention_backend == "tile-cuda"
    assert config.kv_cache.tile_ops_lib == "libtile-strict.so"
    assert config.kv_cache.cuda_runtime_lib == "libcudart.so.13"
    assert config.kv_cache.cuda_device == 2
    assert config.chat_template == "plain_roles"
    assert config.state_db == Path("serve-state.sqlite3")
    assert config.prefix_cache_capacity == 0


def test_native_serve_cli_rejects_prefix_cache_with_tile_turboquant(
    monkeypatch,
    capsys,
) -> None:
    module = _load_cli_module()
    monkeypatch.setattr(
        "neuralfn.native_serve.run_native_inference_server",
        lambda _config: (_ for _ in ()).throw(AssertionError("server started")),
    )

    result = module._native_serve_main(
        [
            "infer",
            "--checkpoint",
            "artifact-dir",
            "--serve",
            "--state-db",
            "serve-state.sqlite3",
            "--prefix-cache-capacity",
            "5",
            "--kv-cache",
            "turboquant",
            "--turboquant-attention-backend",
            "tile-cuda",
            "--tile-ops-lib",
            "libtile.so",
        ]
    )

    assert result == 1
    assert (
        "prefix_cache_capacity rejects Tile-CUDA TurboQuant attention"
        in capsys.readouterr().err
    )


def test_native_serve_cli_defaults_to_jointly_proven_cache_mode(monkeypatch) -> None:
    module = _load_cli_module()
    captured = []
    monkeypatch.setattr(
        "neuralfn.native_serve.run_native_inference_server",
        lambda config: captured.append(config),
    )

    result = module._native_serve_main(
        ["infer", "--checkpoint", "artifact-dir", "--serve"]
    )

    assert result == 0
    assert captured[0].kv_cache.mode == "auto"
    assert captured[0].session_limit == captured[0].queue_capacity + 1
    assert captured[0].prefix_cache_capacity == 0


def test_native_serve_cli_rejects_negative_prefix_cache_capacity(
    monkeypatch,
    capsys,
) -> None:
    module = _load_cli_module()
    monkeypatch.setattr(
        "neuralfn.native_serve.run_native_inference_server",
        lambda _config: (_ for _ in ()).throw(AssertionError("server started")),
    )

    result = module._native_serve_main(
        [
            "infer",
            "--checkpoint",
            "artifact-dir",
            "--serve",
            "--prefix-cache-capacity",
            "-1",
        ]
    )

    assert result == 1
    assert "prefix_cache_capacity must be a non-negative integer" in capsys.readouterr().err


def test_native_serve_accepts_native_checkpoint_alias(monkeypatch) -> None:
    module = _load_cli_module()
    captured = []
    monkeypatch.setattr(
        "neuralfn.native_serve.run_native_inference_server",
        lambda config: captured.append(config),
    )

    result = module._native_serve_main(
        ["infer", "--native-checkpoint", "artifact-dir", "--serve"]
    )

    assert result == 0
    assert captured[0].artifact == Path("artifact-dir")


def test_native_serve_dispatch_precedes_native_checkpoint_one_shot_detection(
    monkeypatch,
) -> None:
    module = _load_cli_module()
    calls = []
    monkeypatch.setattr(module, "_native_serve_main", lambda argv: calls.append(argv) or 17)
    monkeypatch.setattr(
        module,
        "_is_lightweight_native_gpt_infer",
        lambda _argv: (_ for _ in ()).throw(AssertionError("one-shot route ran first")),
    )

    result = module.main(["infer", "--checkpoint", "artifact", "--serve"])

    assert result == 17
    assert calls == [["infer", "--checkpoint", "artifact", "--serve"]]
