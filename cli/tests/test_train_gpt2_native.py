from __future__ import annotations

import os
import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from neuralfn.config import SHIPPED_GPT_TEMPLATE_PRESETS


ROOT = Path(__file__).resolve().parents[1]
NEURALFN_ROOT = ROOT.parent


class TrainGpt2NativeStartupTest(unittest.TestCase):
    def test_nfn_root_help_does_not_import_torch(self) -> None:
        for argv in (["--help"], ["--help-style", "verbose", "--help"], []):
            with self.subTest(argv=argv):
                code = textwrap.dedent(
                    f"""
                    from pathlib import Path
                    import runpy
                    import sys

                    root = Path({str(NEURALFN_ROOT)!r})
                    sys.argv = [str(root / "cli" / "nfn.py"), *{argv!r}]
                    try:
                        runpy.run_path(str(root / "cli" / "nfn.py"), run_name="__main__")
                    except SystemExit as exc:
                        exit_code = int(exc.code or 0)
                    else:
                        exit_code = 0
                    print("TORCH_LOADED", "torch" in sys.modules)
                    print("NFN_IMPL_LOADED", "nfn_impl" in sys.modules)
                    raise SystemExit(exit_code)
                    """
                )
                env = os.environ.copy()
                env.pop("PYTHONPATH", None)
                proc = subprocess.run(
                    [sys.executable, "-c", code],
                    cwd=NEURALFN_ROOT,
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )

                self.assertEqual(0, proc.returncode, proc.stderr)
                self.assertIn("Master NeuralFn CLI for train, infer, and eval.", proc.stdout)
                self.assertIn("TORCH_LOADED False", proc.stdout)
                self.assertIn("NFN_IMPL_LOADED False", proc.stdout)

    def test_nfn_console_entry_help_does_not_import_torch(self) -> None:
        code = textwrap.dedent(
            f"""
            from pathlib import Path
            import sys

            root = Path({str(NEURALFN_ROOT)!r})
            sys.path.insert(0, str(root / "cli"))
            sys.path.insert(0, str(root / "cli" / "scripts"))
            sys.path.insert(0, str(root))
            sys.argv = ["nfn", "--help"]

            from nfn import main

            exit_code = int(main() or 0)
            print("TORCH_LOADED", "torch" in sys.modules)
            print("NFN_IMPL_LOADED", "nfn_impl" in sys.modules)
            raise SystemExit(exit_code)
            """
        )
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=NEURALFN_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("Master NeuralFn CLI for train, infer, and eval.", proc.stdout)
        self.assertIn("TORCH_LOADED False", proc.stdout)
        self.assertIn("NFN_IMPL_LOADED False", proc.stdout)

    def test_nfn_subcommand_help_does_not_import_torch(self) -> None:
        cases = (
            (["train", "--help"], "Train NeuralFn models."),
            (["train", "--help-style", "verbose", "--help"], "Train NeuralFn models."),
            (["infer", "--help"], "Run inference from NeuralFn artifacts."),
            (["eval", "--help"], "Evaluate NeuralFn artifacts."),
            (["kernels", "--help"], "Inspect CUDA Tile kernel coverage and diagnostics."),
            (["kernels", "doctor", "--help"], "Inspect CUDA Tile kernel coverage and diagnostics."),
            (["kernels", "bench", "--help-style=long", "--help"], "Inspect CUDA Tile kernel coverage and diagnostics."),
            (["kernels", "examples", "--help"], "Inspect CUDA Tile kernel coverage and diagnostics."),
        )
        for argv, expected in cases:
            with self.subTest(argv=argv):
                code = textwrap.dedent(
                    f"""
                    from pathlib import Path
                    import runpy
                    import sys

                    root = Path({str(NEURALFN_ROOT)!r})
                    sys.argv = [str(root / "cli" / "nfn.py"), *{argv!r}]
                    try:
                        runpy.run_path(str(root / "cli" / "nfn.py"), run_name="__main__")
                    except SystemExit as exc:
                        exit_code = int(exc.code or 0)
                    else:
                        exit_code = 0
                    print("TORCH_LOADED", "torch" in sys.modules)
                    print("NFN_IMPL_LOADED", "nfn_impl" in sys.modules)
                    raise SystemExit(exit_code)
                    """
                )
                env = os.environ.copy()
                env.pop("PYTHONPATH", None)
                proc = subprocess.run(
                    [sys.executable, "-c", code],
                    cwd=NEURALFN_ROOT,
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )

                self.assertEqual(0, proc.returncode, proc.stderr)
                self.assertIn(expected, proc.stdout)
                if argv[0] == "train":
                    self.assertIn("--template-name NAME", proc.stdout)
                    self.assertIn("--graph-file PATH", proc.stdout)
                self.assertIn("TORCH_LOADED False", proc.stdout)
                self.assertIn("NFN_IMPL_LOADED False", proc.stdout)

    def test_nfn_kernels_list_does_not_import_torch(self) -> None:
        for argv in (["kernels"], ["kernels", "list"], ["kernels", "list", "--json"]):
            with self.subTest(argv=argv):
                code = textwrap.dedent(
                    f"""
                    from pathlib import Path
                    import runpy
                    import sys

                    root = Path({str(NEURALFN_ROOT)!r})
                    sys.argv = [str(root / "cli" / "nfn.py"), *{argv!r}]
                    try:
                        runpy.run_path(str(root / "cli" / "nfn.py"), run_name="__main__")
                    except SystemExit as exc:
                        exit_code = int(exc.code or 0)
                    else:
                        exit_code = 0
                    print("TORCH_LOADED", "torch" in sys.modules)
                    print("NFN_IMPL_LOADED", "nfn_impl" in sys.modules)
                    raise SystemExit(exit_code)
                    """
                )
                env = os.environ.copy()
                env.pop("PYTHONPATH", None)
                proc = subprocess.run(
                    [sys.executable, "-c", code],
                    cwd=NEURALFN_ROOT,
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )

                self.assertEqual(0, proc.returncode, proc.stderr)
                if "--json" in argv:
                    self.assertIn('"total_inventory"', proc.stdout)
                else:
                    self.assertIn("NeuralFn CUDA Tile kernel coverage:", proc.stdout)
                self.assertIn("TORCH_LOADED False", proc.stdout)
                self.assertIn("NFN_IMPL_LOADED False", proc.stdout)

    def test_native_cached_shard_dry_run_avoids_dataset_manager_imports(self) -> None:
        code = textwrap.dedent(
            f"""
            import json
            from pathlib import Path
            import runpy
            import struct
            import sys
            import tempfile

            root = Path({str(NEURALFN_ROOT)!r})
            sys.path.insert(0, str(root / "cli" / "scripts"))
            sys.path.insert(0, str(root))

            with tempfile.TemporaryDirectory() as tmpdir:
                dataset = Path(tmpdir) / "tiny"
                dataset.mkdir()
                token_bytes = struct.pack("<" + "H" * 256, *range(256))
                (dataset / "fineweb_train_000000.bin").write_bytes(token_bytes)
                (dataset / "fineweb_val_000000.bin").write_bytes(token_bytes)
                (dataset / "meta.json").write_text(json.dumps({{
                    "data_format": "uint16_shards",
                    "tokenizer_encoding": "gpt2",
                    "tokenizer_vocab_size": 50257,
                }}), encoding="utf-8")
                sys.argv = [
                    str(root / "cli" / "scripts" / "train_gpt2.py"),
                    "--dataset-alias",
                    str(dataset),
                    "--no-download-if-missing",
                    "--native-cuda-runner",
                    "subprocess",
                    "--native-cuda-dry-run",
                    "--native-cuda-print-command",
                ]
                try:
                    runpy.run_path(str(root / "cli" / "scripts" / "train_gpt2.py"), run_name="__main__")
                except SystemExit as exc:
                    exit_code = int(exc.code or 0)
                else:
                    exit_code = 0
                print("TORCH_LOADED", "torch" in sys.modules)
                print("DATASET_MANAGER_LOADED", "server.dataset_manager" in sys.modules)
                print("NUMPY_LOADED", "numpy" in sys.modules)
                print("TIKTOKEN_LOADED", "tiktoken" in sys.modules)
                print("TRAIN_GPT2_NATIVE_LOADED", "train_gpt2_native" in sys.modules)
                raise SystemExit(exit_code)
            """
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{NEURALFN_ROOT / 'cli' / 'scripts'}:{NEURALFN_ROOT}"
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=NEURALFN_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("Native CUDA runner: subprocess (requested=subprocess)", proc.stdout)
        self.assertIn("Estimated train rows:", proc.stdout)
        self.assertIn("TORCH_LOADED False", proc.stdout)
        self.assertIn("DATASET_MANAGER_LOADED False", proc.stdout)
        self.assertIn("NUMPY_LOADED False", proc.stdout)
        self.assertIn("TIKTOKEN_LOADED False", proc.stdout)

    def test_native_cached_shard_default_runner_uses_compiled_cli(self) -> None:
        code = textwrap.dedent(
            f"""
            import json
            from pathlib import Path
            import runpy
            import struct
            import sys
            import tempfile

            root = Path({str(NEURALFN_ROOT)!r})
            sys.path.insert(0, str(root / "cli" / "scripts"))
            sys.path.insert(0, str(root))

            with tempfile.TemporaryDirectory() as tmpdir:
                dataset = Path(tmpdir) / "tiny"
                dataset.mkdir()
                token_bytes = struct.pack("<" + "H" * 256, *range(256))
                (dataset / "fineweb_train_000000.bin").write_bytes(token_bytes)
                (dataset / "fineweb_val_000000.bin").write_bytes(token_bytes)
                sys.argv = [
                    str(root / "cli" / "scripts" / "train_gpt2.py"),
                    "--dataset-alias",
                    str(dataset),
                    "--no-download-if-missing",
                    "--native-cuda-dry-run",
                    "--native-cuda-print-command",
                ]
                try:
                    runpy.run_path(str(root / "cli" / "scripts" / "train_gpt2.py"), run_name="__main__")
                except SystemExit as exc:
                    exit_code = int(exc.code or 0)
                else:
                    exit_code = 0
                print("TORCH_LOADED", "torch" in sys.modules)
                print("DATASET_MANAGER_LOADED", "server.dataset_manager" in sys.modules)
                print("NUMPY_LOADED", "numpy" in sys.modules)
                print("TIKTOKEN_LOADED", "tiktoken" in sys.modules)
                print("TRAIN_GPT2_NATIVE_LOADED", "train_gpt2_native" in sys.modules)
                raise SystemExit(exit_code)
            """
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{NEURALFN_ROOT / 'cli' / 'scripts'}:{NEURALFN_ROOT}"
        env["NFN_NATIVE_GPT2_CLI"] = "/bin/echo"
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=NEURALFN_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("--dataset-alias", proc.stdout)
        self.assertRegex(proc.stdout, r"--dataset-alias /tmp/.*/tiny")
        self.assertNotIn("--target ", proc.stdout)
        self.assertNotIn("train_gpt2cu", proc.stdout)
        self.assertIn("--train-transformer-lm", proc.stdout)
        self.assertIn("TORCH_LOADED False", proc.stdout)
        self.assertIn("DATASET_MANAGER_LOADED False", proc.stdout)
        self.assertIn("NUMPY_LOADED False", proc.stdout)
        self.assertIn("TIKTOKEN_LOADED False", proc.stdout)
        self.assertIn("TRAIN_GPT2_NATIVE_LOADED False", proc.stdout)

    def test_native_dry_run_without_token_shards_defers_to_compiled_cli(self) -> None:
        code = textwrap.dedent(
            f"""
            from pathlib import Path
            import runpy
            import sys
            import tempfile

            root = Path({str(NEURALFN_ROOT)!r})
            sys.path.insert(0, str(root / "cli" / "scripts"))
            sys.path.insert(0, str(root))

            with tempfile.TemporaryDirectory() as tmpdir:
                dataset = Path(tmpdir) / "raw-only"
                dataset.mkdir()
                (dataset / "TinyStoriesV2-GPT4-train.txt").write_text("hello world", encoding="utf-8")
                (dataset / "TinyStoriesV2-GPT4-valid.txt").write_text("hello world", encoding="utf-8")
                sys.argv = [
                    str(root / "cli" / "scripts" / "train_gpt2.py"),
                    "--dataset-alias",
                    str(dataset),
                    "--no-download-if-missing",
                    "--native-cuda-dry-run",
                    "--native-cuda-print-command",
                ]
                try:
                    runpy.run_path(str(root / "cli" / "scripts" / "train_gpt2.py"), run_name="__main__")
                except SystemExit as exc:
                    exit_code = int(exc.code or 0)
                else:
                    exit_code = 0
                print("TRAIN_SHARD_EXISTS", (dataset / "fineweb_train_000000.bin").exists())
                print("TORCH_LOADED", "torch" in sys.modules)
                print("DATASET_MANAGER_LOADED", "server.dataset_manager" in sys.modules)
                print("NUMPY_LOADED", "numpy" in sys.modules)
                print("TIKTOKEN_LOADED", "tiktoken" in sys.modules)
                print("TRAIN_GPT2_NATIVE_LOADED", "train_gpt2_native" in sys.modules)
                raise SystemExit(exit_code)
            """
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{NEURALFN_ROOT / 'cli' / 'scripts'}:{NEURALFN_ROOT}"
        env["NFN_NATIVE_GPT2_CLI"] = "/bin/echo"
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=NEURALFN_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("--dataset-alias", proc.stdout)
        self.assertRegex(proc.stdout, r"--dataset-alias /tmp/.*/raw-only")
        self.assertNotIn("--target ", proc.stdout)
        self.assertNotIn("train_gpt2cu", proc.stdout)
        self.assertIn("--train-transformer-lm", proc.stdout)
        self.assertIn("TRAIN_SHARD_EXISTS False", proc.stdout)
        self.assertIn("TORCH_LOADED False", proc.stdout)
        self.assertIn("DATASET_MANAGER_LOADED False", proc.stdout)
        self.assertIn("NUMPY_LOADED False", proc.stdout)
        self.assertIn("TIKTOKEN_LOADED False", proc.stdout)
        self.assertIn("TRAIN_GPT2_NATIVE_LOADED False", proc.stdout)

    def test_native_dry_run_llm_kittens_backend_keeps_external_target(self) -> None:
        code = textwrap.dedent(
            f"""
            from pathlib import Path
            import runpy
            import sys

            root = Path({str(NEURALFN_ROOT)!r})
            sys.path.insert(0, str(root / "cli" / "scripts"))
            sys.path.insert(0, str(root))
            sys.argv = [
                str(root / "cli" / "scripts" / "train_gpt2.py"),
                "--dataset-alias",
                "/tmp/native-cache",
                "--native-cuda-kernel-backend",
                "llm-kittens",
                "--native-cuda-dry-run",
                "--native-cuda-print-command",
            ]
            try:
                runpy.run_path(str(root / "cli" / "scripts" / "train_gpt2.py"), run_name="__main__")
            except SystemExit as exc:
                exit_code = int(exc.code or 0)
            else:
                exit_code = 0
            print("TORCH_LOADED", "torch" in sys.modules)
            raise SystemExit(exit_code)
            """
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{NEURALFN_ROOT / 'cli' / 'scripts'}:{NEURALFN_ROOT}"
        env["NFN_NATIVE_GPT2_CLI"] = "/bin/echo"
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=NEURALFN_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("--backend llm-kittens", proc.stdout)
        self.assertIn("--target ", proc.stdout)
        self.assertIn("train_gpt2cu", proc.stdout)
        self.assertIn("TORCH_LOADED False", proc.stdout)

    def test_native_dry_run_does_not_import_torch(self) -> None:
        code = textwrap.dedent(
            f"""
            import json
            from pathlib import Path
            import os
            import runpy
            import sys
            import tempfile

            root = Path({str(NEURALFN_ROOT)!r})
            sys.path.insert(0, str(root / "cli" / "scripts"))
            sys.path.insert(0, str(root))

            with tempfile.TemporaryDirectory() as tmpdir:
                dataset = Path(tmpdir) / "tiny"
                dataset.mkdir()
                (dataset / "data.txt").write_text("hello world. " * 128, encoding="utf-8")
                (dataset / "val.txt").write_text("validation story. " * 64, encoding="utf-8")
                (dataset / "meta.json").write_text(json.dumps({{"data_format": "raw_text"}}), encoding="utf-8")
                sys.argv = [
                    str(root / "cli" / "scripts" / "train_gpt2.py"),
                    "--dataset-alias",
                    str(dataset),
                    "--no-download-if-missing",
                    "--native-cuda-runner",
                    "subprocess",
                    "--native-cuda-dry-run",
                    "--native-cuda-print-command",
                    "--eval-every-steps",
                    "1000",
                ]
                try:
                    runpy.run_path(str(root / "cli" / "scripts" / "train_gpt2.py"), run_name="__main__")
                except SystemExit as exc:
                    exit_code = int(exc.code or 0)
                else:
                    exit_code = 0
                print("TORCH_LOADED", "torch" in sys.modules)
                raise SystemExit(exit_code)
            """
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{NEURALFN_ROOT / 'cli' / 'scripts'}:{NEURALFN_ROOT}"
        env["NFN_NATIVE_GPT2_BINDING"] = "0"
        env["NFN_NATIVE_GPT2_CLI"] = str(NEURALFN_ROOT / "build" / "missing-test-native-cli")
        env["NFN_NATIVE_GPT2_LAUNCHER"] = str(NEURALFN_ROOT / "build" / "missing-test-launcher")
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=NEURALFN_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("Native CUDA validation eval: every 1000 optimizer steps", proc.stdout)
        self.assertIn("Native CUDA runner: subprocess (requested=subprocess)", proc.stdout)
        self.assertIn("train_gpt2cu -i", proc.stdout)
        self.assertIn("-v 1000", proc.stdout)
        self.assertIn("TORCH_LOADED False", proc.stdout)

    def test_nfn_train_gpt2_native_dry_run_does_not_import_torch(self) -> None:
        code = textwrap.dedent(
            f"""
            import json
            from pathlib import Path
            import runpy
            import sys
            import tempfile

            root = Path({str(NEURALFN_ROOT)!r})
            sys.path.insert(0, str(root / "cli"))
            sys.path.insert(0, str(root / "cli" / "scripts"))
            sys.path.insert(0, str(root))

            with tempfile.TemporaryDirectory() as tmpdir:
                dataset = Path(tmpdir) / "tiny"
                dataset.mkdir()
                (dataset / "data.txt").write_text("hello world. " * 128, encoding="utf-8")
                (dataset / "val.txt").write_text("validation story. " * 64, encoding="utf-8")
                (dataset / "meta.json").write_text(json.dumps({{"data_format": "raw_text"}}), encoding="utf-8")
                sys.argv = [
                    str(root / "cli" / "nfn.py"),
                    "train",
                    "--base-model",
                    "gpt2",
                    "--dataset-alias",
                    str(dataset),
                    "--no-download-if-missing",
                    "--no-tile-cuda-strict",
                    "--native-cuda-runner",
                    "subprocess",
                    "--native-cuda-dry-run",
                    "--native-cuda-print-command",
                    "--eval-every-steps",
                    "1000",
                ]
                try:
                    runpy.run_path(str(root / "cli" / "nfn.py"), run_name="__main__")
                except SystemExit as exc:
                    exit_code = int(exc.code or 0)
                else:
                    exit_code = 0
                print("TORCH_LOADED", "torch" in sys.modules)
                raise SystemExit(exit_code)
            """
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{NEURALFN_ROOT / 'cli'}:{NEURALFN_ROOT / 'cli' / 'scripts'}:{NEURALFN_ROOT}"
        env["NFN_NATIVE_GPT2_BINDING"] = "0"
        env["NFN_NATIVE_GPT2_CLI"] = str(NEURALFN_ROOT / "build" / "missing-test-native-cli")
        env["NFN_NATIVE_GPT2_LAUNCHER"] = str(NEURALFN_ROOT / "build" / "missing-test-launcher")
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=NEURALFN_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("Native CUDA validation eval: every 1000 optimizer steps", proc.stdout)
        self.assertIn("Native CUDA runner: subprocess (requested=subprocess)", proc.stdout)
        self.assertIn("train_gpt2cu -i", proc.stdout)
        self.assertIn("-v 1000", proc.stdout)
        self.assertIn("TORCH_LOADED False", proc.stdout)

    def test_nfn_train_gpt_alias_default_dispatches_directly_to_compiled_cli(self) -> None:
        code = textwrap.dedent(
            f"""
            from pathlib import Path
            import runpy
            import sys

            root = Path({str(NEURALFN_ROOT)!r})
            sys.argv = [
                str(root / "cli" / "nfn.py"),
                "train",
                "--base-model",
                "gpt",
                "--dataset-alias=/tmp/native-cache",
                "--output",
                "/tmp/gpt2.pt",
                "--native-cuda-dry-run",
                "--native-cuda-print-command",
                "--eval-every-steps=1000",
                "--native-cuda-activation=sd-prelu",
            ]
            try:
                runpy.run_path(str(root / "cli" / "nfn.py"), run_name="__main__")
            except SystemExit as exc:
                exit_code = int(exc.code or 0)
            else:
                exit_code = 0
            print("TORCH_LOADED", "torch" in sys.modules)
            print("NFN_IMPL_LOADED", "nfn_impl" in sys.modules)
            print("TRAIN_GPT2_NATIVE_LOADED", "train_gpt2_native" in sys.modules)
            print("DATASET_MANAGER_LOADED", "server.dataset_manager" in sys.modules)
            raise SystemExit(exit_code)
            """
        )
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env["NFN_NATIVE_GPT2_CLI"] = "/bin/echo"
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=NEURALFN_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("--dataset-alias /tmp/native-cache", proc.stdout)
        self.assertIn("--output-dir /tmp/gpt2", proc.stdout)
        self.assertIn("--eval-every-steps 1000", proc.stdout)
        self.assertIn("--native-cuda-activation sd-prelu", proc.stdout)
        self.assertIn("--dry-run", proc.stdout)
        self.assertIn("--print-command", proc.stdout)
        self.assertNotIn("--base-model", proc.stdout)
        self.assertIn("TORCH_LOADED False", proc.stdout)
        self.assertIn("NFN_IMPL_LOADED False", proc.stdout)
        self.assertIn("TRAIN_GPT2_NATIVE_LOADED False", proc.stdout)
        self.assertIn("DATASET_MANAGER_LOADED False", proc.stdout)

    def test_nfn_train_gpt2_direct_compiled_cli_preserves_template_and_graph_selectors(self) -> None:
        code = textwrap.dedent(
            f"""
            from pathlib import Path
            import runpy
            import sys

            root = Path({str(NEURALFN_ROOT)!r})
            sys.argv = [
                str(root / "cli" / "nfn.py"),
                "train",
                "--base-model",
                "gpt2",
                "--dataset-alias=/tmp/native-cache",
                "--template-name=semantic_router_moe",
                "--graph-file",
                "/tmp/custom-graph.json",
                "--native-cuda-dry-run",
                "--native-cuda-print-command",
            ]
            try:
                runpy.run_path(str(root / "cli" / "nfn.py"), run_name="__main__")
            except SystemExit as exc:
                exit_code = int(exc.code or 0)
            else:
                exit_code = 0
            print("TORCH_LOADED", "torch" in sys.modules)
            print("NFN_IMPL_LOADED", "nfn_impl" in sys.modules)
            print("TRAIN_GPT2_NATIVE_LOADED", "train_gpt2_native" in sys.modules)
            raise SystemExit(exit_code)
            """
        )
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env["NFN_NATIVE_GPT2_CLI"] = "/bin/echo"
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=NEURALFN_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("--dataset-alias /tmp/native-cache", proc.stdout)
        self.assertIn("--template-name semantic_router_moe", proc.stdout)
        self.assertIn("--graph-file /tmp/custom-graph.json", proc.stdout)
        self.assertIn("--train-transformer-lm", proc.stdout)
        self.assertNotIn("--base-model", proc.stdout)
        self.assertIn("TORCH_LOADED False", proc.stdout)
        self.assertIn("NFN_IMPL_LOADED False", proc.stdout)
        self.assertIn("TRAIN_GPT2_NATIVE_LOADED False", proc.stdout)

    def test_nfn_train_gpt2_direct_compiled_cli_accepts_every_template_selector(self) -> None:
        for preset in SHIPPED_GPT_TEMPLATE_PRESETS:
            with self.subTest(preset=preset):
                code = textwrap.dedent(
                    f"""
                    from pathlib import Path
                    import runpy
                    import sys

                    root = Path({str(NEURALFN_ROOT)!r})
                    sys.argv = [
                        str(root / "cli" / "nfn.py"),
                        "train",
                        "--base-model",
                        "gpt2",
                        "--dataset-alias=/tmp/native-cache",
                        "--preset",
                        {preset!r},
                        "--native-cuda-dry-run",
                        "--native-cuda-print-command",
                    ]
                    try:
                        runpy.run_path(str(root / "cli" / "nfn.py"), run_name="__main__")
                    except SystemExit as exc:
                        exit_code = int(exc.code or 0)
                    else:
                        exit_code = 0
                    print("TORCH_LOADED", "torch" in sys.modules)
                    print("NFN_IMPL_LOADED", "nfn_impl" in sys.modules)
                    print("TRAIN_GPT2_NATIVE_LOADED", "train_gpt2_native" in sys.modules)
                    raise SystemExit(exit_code)
                    """
                )
                env = os.environ.copy()
                env.pop("PYTHONPATH", None)
                env["NFN_NATIVE_GPT2_CLI"] = "/bin/echo"
                proc = subprocess.run(
                    [sys.executable, "-c", code],
                    cwd=NEURALFN_ROOT,
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )

                self.assertEqual(0, proc.returncode, proc.stderr)
                self.assertIn(f"--template-name {preset}", proc.stdout)
                self.assertIn("--train-transformer-lm", proc.stdout)
                self.assertNotIn("--base-model", proc.stdout)
                self.assertIn("TORCH_LOADED False", proc.stdout)
                self.assertIn("NFN_IMPL_LOADED False", proc.stdout)
                self.assertIn("TRAIN_GPT2_NATIVE_LOADED False", proc.stdout)

    def test_nfn_train_gpt2_can_dispatch_to_unified_native_train_cli(self) -> None:
        code = textwrap.dedent(
            f"""
            from pathlib import Path
            import runpy
            import sys

            root = Path({str(NEURALFN_ROOT)!r})
            sys.argv = [
                str(root / "cli" / "nfn.py"),
                "train",
                "--base-model",
                "gpt2",
                "--dataset-alias=/tmp/native-cache",
                "--native-cuda-dry-run",
                "--eval-every-steps=1000",
            ]
            try:
                runpy.run_path(str(root / "cli" / "nfn.py"), run_name="__main__")
            except SystemExit as exc:
                exit_code = int(exc.code or 0)
            else:
                exit_code = 0
            print("TORCH_LOADED", "torch" in sys.modules)
            print("NFN_IMPL_LOADED", "nfn_impl" in sys.modules)
            print("TRAIN_GPT2_NATIVE_LOADED", "train_gpt2_native" in sys.modules)
            raise SystemExit(exit_code)
            """
        )
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env["NFN_NATIVE_TRAIN_CLI"] = "/bin/echo"
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=NEURALFN_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("--dataset-alias /tmp/native-cache", proc.stdout)
        self.assertIn("--eval-every-steps 1000", proc.stdout)
        self.assertIn("--dry-run", proc.stdout)
        self.assertIn("--model-family gpt2", proc.stdout)
        self.assertNotIn("--base-model", proc.stdout)
        self.assertIn("TORCH_LOADED False", proc.stdout)
        self.assertIn("NFN_IMPL_LOADED False", proc.stdout)
        self.assertIn("TRAIN_GPT2_NATIVE_LOADED False", proc.stdout)

    def test_nfn_train_gpt3_defaults_to_2048_context_without_template_or_graph(self) -> None:
        code = textwrap.dedent(
            f"""
            from pathlib import Path
            import runpy
            import sys

            root = Path({str(NEURALFN_ROOT)!r})
            sys.argv = [
                str(root / "cli" / "nfn.py"),
                "train",
                "--base-model",
                "gpt3",
                "--dataset-alias=/tmp/native-cache",
                "--native-cuda-dry-run",
            ]
            try:
                runpy.run_path(str(root / "cli" / "nfn.py"), run_name="__main__")
            except SystemExit as exc:
                exit_code = int(exc.code or 0)
            else:
                exit_code = 0
            print("TORCH_LOADED", "torch" in sys.modules)
            print("NFN_IMPL_LOADED", "nfn_impl" in sys.modules)
            raise SystemExit(exit_code)
            """
        )
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env["NFN_NATIVE_TRAIN_CLI"] = "/bin/echo"
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=NEURALFN_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("--model-family gpt3", proc.stdout)
        self.assertIn("--train-seq-len 2048", proc.stdout)
        self.assertIn("TORCH_LOADED False", proc.stdout)
        self.assertIn("NFN_IMPL_LOADED False", proc.stdout)

    def test_nfn_train_gpt3_does_not_override_explicit_template_graph_or_seq_len(self) -> None:
        code = textwrap.dedent(
            f"""
            from pathlib import Path
            import runpy
            import sys

            root = Path({str(NEURALFN_ROOT)!r})
            sys.argv = [
                str(root / "cli" / "nfn.py"),
                "train",
                "--base-model",
                "gpt3",
                "--dataset-alias=/tmp/native-cache",
                "--template-name=gpt2_moa",
                "--graph-file=/tmp/custom-graph.json",
                "--train-seq-len=4096",
                "--native-cuda-dry-run",
            ]
            try:
                runpy.run_path(str(root / "cli" / "nfn.py"), run_name="__main__")
            except SystemExit as exc:
                exit_code = int(exc.code or 0)
            else:
                exit_code = 0
            raise SystemExit(exit_code)
            """
        )
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env["NFN_NATIVE_TRAIN_CLI"] = "/bin/echo"
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=NEURALFN_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("--model-family gpt3", proc.stdout)
        self.assertIn("--template-name gpt2_moa", proc.stdout)
        self.assertIn("--graph-file /tmp/custom-graph.json", proc.stdout)
        self.assertIn("--train-seq-len 4096", proc.stdout)
        self.assertNotIn("--train-seq-len 2048", proc.stdout)

    def test_nfn_train_gpt2_translates_kernel_backend_to_native_backend(self) -> None:
        code = textwrap.dedent(
            f"""
            from pathlib import Path
            import runpy
            import sys

            root = Path({str(NEURALFN_ROOT)!r})
            sys.argv = [
                str(root / "cli" / "nfn.py"),
                "train",
                "--base-model",
                "gpt2",
                "--dataset-alias=/tmp/native-cache",
                "--kernel-backend=tile-cuda",
                "--preset=gpt2_moa",
                "--graph=/tmp/custom-graph.json",
                "--native-cuda-print-plan",
                "--native-cuda-smoke-tile-ops",
                "--native-cuda-smoke-optimizer-step",
                "--native-cuda-smoke-lm-step",
                "--native-cuda-smoke-attention-step",
                "--native-cuda-smoke-mlp-step",
                "--native-cuda-smoke-norm-residual-step",
                "--native-cuda-smoke-transformer-block-step",
                "--native-cuda-smoke-transformer-lm-step",
                "--native-cuda-smoke-embedding-lm-step",
                "--train-embedding-lm",
                "--train-transformer-lm",
                "--eval-batches=1",
                "--eval-batch-size=1",
                "--native-cuda-lm-head-row-chunk-size=2048",
                "--native-cuda-cuda-runtime-lib=/opt/cuda/libcudart.so",
                "--native-cuda-dry-run",
            ]
            try:
                runpy.run_path(str(root / "cli" / "nfn.py"), run_name="__main__")
            except SystemExit as exc:
                exit_code = int(exc.code or 0)
            else:
                exit_code = 0
            print("TORCH_LOADED", "torch" in sys.modules)
            print("NFN_IMPL_LOADED", "nfn_impl" in sys.modules)
            raise SystemExit(exit_code)
            """
        )
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env["NFN_NATIVE_TRAIN_CLI"] = "/bin/echo"
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=NEURALFN_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("--backend tile-cuda", proc.stdout)
        self.assertIn("--template-name gpt2_moa", proc.stdout)
        self.assertIn("--graph-file /tmp/custom-graph.json", proc.stdout)
        self.assertIn("--native-cuda-activation moa", proc.stdout)
        self.assertIn("--print-plan", proc.stdout)
        self.assertIn("--smoke-tile-ops", proc.stdout)
        self.assertIn("--smoke-optimizer-step", proc.stdout)
        self.assertIn("--smoke-lm-step", proc.stdout)
        self.assertIn("--smoke-attention-step", proc.stdout)
        self.assertIn("--smoke-mlp-step", proc.stdout)
        self.assertIn("--smoke-norm-residual-step", proc.stdout)
        self.assertIn("--smoke-transformer-block-step", proc.stdout)
        self.assertIn("--smoke-transformer-lm-step", proc.stdout)
        self.assertIn("--smoke-embedding-lm-step", proc.stdout)
        self.assertIn("--train-embedding-lm", proc.stdout)
        self.assertIn("--train-transformer-lm", proc.stdout)
        self.assertIn("--eval-batches 1", proc.stdout)
        self.assertIn("--eval-batch-size 1", proc.stdout)
        self.assertIn("--lm-head-row-chunk-size 2048", proc.stdout)
        self.assertIn("--cuda-runtime-lib /opt/cuda/libcudart.so", proc.stdout)
        self.assertIn("--dry-run", proc.stdout)
        self.assertIn("TORCH_LOADED False", proc.stdout)
        self.assertIn("NFN_IMPL_LOADED False", proc.stdout)

    def test_nfn_train_non_native_model_rejects_before_torch_import(self) -> None:
        code = textwrap.dedent(
            f"""
            from pathlib import Path
            import os
            import runpy
            import sys
            import tempfile

            root = Path({str(NEURALFN_ROOT)!r})
            native_train = Path(tempfile.mkdtemp()) / "nfn_native_train"
            native_train.write_text(
                "#!/usr/bin/env bash\\n"
                "printf %s\\\\n \\\"No native C++ trainer is registered for model family 'llama'.\\\" >&2\\n"
                "printf 'Current native training coverage:\\\\n' >&2\\n"
                "printf '  gpt2: partial-native-trainer -> nfn_gpt2_native_train\\\\n' >&2\\n"
                "printf '  nanogpt: partial-native-trainer -> nfn_nanogpt_native_train\\\\n' >&2\\n"
                "exit 2\\n",
                encoding="utf-8",
            )
            native_train.chmod(0o755)
            os.environ["NFN_NATIVE_TRAIN_CLI"] = str(native_train)
            sys.argv = [
                str(root / "cli" / "nfn.py"),
                "train",
                "--base-model",
                "llama",
                "--tinystories",
                "--native-cuda-dry-run",
            ]
            try:
                runpy.run_path(str(root / "cli" / "nfn.py"), run_name="__main__")
            except SystemExit as exc:
                exit_code = int(exc.code or 0)
            else:
                exit_code = 0
            print("TORCH_LOADED", "torch" in sys.modules)
            print("NFN_IMPL_LOADED", "nfn_impl" in sys.modules)
            print("TRAIN_GPT2_NATIVE_LOADED", "train_gpt2_native" in sys.modules)
            print("NATIVE_TRAIN", native_train)
            raise SystemExit(exit_code)
            """
        )
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.pop("NFN_ALLOW_TORCH_TRAINING", None)
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=NEURALFN_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(2, proc.returncode)
        self.assertIn("No native C++ trainer is registered for model family 'llama'", proc.stderr)
        self.assertIn("gpt2: partial-native-trainer -> nfn_gpt2_native_train", proc.stderr)
        self.assertIn("nanogpt: partial-native-trainer -> nfn_nanogpt_native_train", proc.stderr)
        self.assertIn("TORCH_LOADED False", proc.stdout)
        self.assertIn("NFN_IMPL_LOADED False", proc.stdout)
        self.assertIn("TRAIN_GPT2_NATIVE_LOADED False", proc.stdout)

    def test_legacy_training_scripts_reject_before_torch_import(self) -> None:
        scripts = (
            ("train_llama_fast.py", "llama"),
            ("train_llama_megakernel.py", "llama"),
            ("train_nanogpt.py", "nanogpt"),
            ("train_mixllama_fast.py", "mixllama"),
            ("train_jepa_semantic.py", "jepa"),
            ("train_semantic_router_moe.py", "semantic-router-moe"),
            ("train_semantic_router_moe-overnight.py", "semantic-router-moe"),
            ("train_deepseek_v4.py", "deepseek-v4"),
        )
        for script_name, model_family in scripts:
            with self.subTest(script=script_name):
                code = textwrap.dedent(
                    f"""
                    from pathlib import Path
                    import os
                    import runpy
                    import sys
                    import tempfile

                    root = Path({str(NEURALFN_ROOT)!r})
                    script = root / "cli" / "scripts" / {script_name!r}
                    model_family = {model_family!r}
                    family_env = "NFN_NATIVE_" + "".join(ch if ch.isalnum() else "_" for ch in model_family.upper()).strip("_") + "_CLI"
                    family_cli = Path(tempfile.mkdtemp()) / ("nfn_" + "".join(ch if ch.isalnum() else "_" for ch in model_family.lower()).strip("_") + "_native_train")
                    family_cli.write_text(
                        "#!/usr/bin/env bash\\n"
                        "printf 'FAMILY_NATIVE_DIRECT\\\\n'\\n"
                        "printf '%s\\\\n' \\"$@\\"\\n"
                        "exit 2\\n",
                        encoding="utf-8",
                    )
                    family_cli.chmod(0o755)
                    os.environ[family_env] = str(family_cli)
                    sys.path.insert(0, str(root / "cli" / "scripts"))
                    sys.argv = [str(script), "--tinystories", "--native-cuda-dry-run"]
                    try:
                        runpy.run_path(str(script), run_name="__main__")
                    except SystemExit as exc:
                        exit_code = int(exc.code or 0)
                    else:
                        exit_code = 0
                    print("TORCH_LOADED", "torch" in sys.modules)
                    print("TRAIN_JEPA_LOADED", "train_jepa_semantic" in sys.modules)
                    raise SystemExit(exit_code)
                    """
                )
                env = os.environ.copy()
                env.pop("PYTHONPATH", None)
                env.pop("NFN_ALLOW_TORCH_TRAINING", None)
                proc = subprocess.run(
                    [sys.executable, "-c", code],
                    cwd=NEURALFN_ROOT,
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )

                self.assertEqual(2, proc.returncode)
                self.assertIn("FAMILY_NATIVE_DIRECT", proc.stdout)
                self.assertNotIn("--base-model", proc.stdout)
                self.assertIn("--tinystories", proc.stdout)
                self.assertIn("TORCH_LOADED False", proc.stdout)
                self.assertIn("TRAIN_JEPA_LOADED False", proc.stdout)

    def test_train_gpt2_evo_direct_script_prefers_family_native_preflight(self) -> None:
        code = textwrap.dedent(
            f"""
            from pathlib import Path
            import os
            import runpy
            import sys
            import tempfile

            root = Path({str(NEURALFN_ROOT)!r})
            script = root / "cli" / "scripts" / "train_gpt2_evo.py"
            native_evo = Path(tempfile.mkdtemp()) / "nfn_gpt2_evo_native_train"
            native_evo.write_text(
                "#!/usr/bin/env bash\\n"
                "printf 'EVO_NATIVE_DIRECT\\\\n'\\n"
                "printf '%s\\\\n' \\"$@\\"\\n"
                "exit 23\\n",
                encoding="utf-8",
            )
            native_evo.chmod(0o755)
            os.environ["NFN_NATIVE_GPT2_EVO_CLI"] = str(native_evo)
            sys.path.insert(0, str(root / "cli" / "scripts"))
            sys.argv = [str(script), "--tinystories", "--native-cuda-dry-run", "--eval-every-steps", "1000"]
            try:
                runpy.run_path(str(script), run_name="__main__")
            except SystemExit as exc:
                exit_code = int(exc.code or 0)
            else:
                exit_code = 0
            print("TORCH_LOADED", "torch" in sys.modules)
            print("TRAIN_JEPA_LOADED", "train_jepa_semantic" in sys.modules)
            raise SystemExit(exit_code)
            """
        )
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.pop("NFN_ALLOW_TORCH_TRAINING", None)
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=NEURALFN_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(23, proc.returncode)
        self.assertIn("EVO_NATIVE_DIRECT", proc.stdout)
        self.assertNotIn("--base-model", proc.stdout)
        self.assertIn("--tinystories", proc.stdout)
        self.assertIn("--native-cuda-dry-run", proc.stdout)
        self.assertIn("--eval-every-steps", proc.stdout)
        self.assertIn("1000", proc.stdout)
        self.assertIn("TORCH_LOADED False", proc.stdout)
        self.assertIn("TRAIN_JEPA_LOADED False", proc.stdout)

    def test_train_nanogpt_direct_script_defaults_to_native_token_lm(self) -> None:
        code = textwrap.dedent(
            f"""
            from pathlib import Path
            import os
            import runpy
            import sys
            import tempfile

            root = Path({str(NEURALFN_ROOT)!r})
            script = root / "cli" / "scripts" / "train_nanogpt.py"
            native_nanogpt = Path(tempfile.mkdtemp()) / "nfn_nanogpt_native_train"
            native_nanogpt.write_text(
                "#!/usr/bin/env bash\\n"
                "printf 'NANOGPT_NATIVE_DIRECT\\\\n'\\n"
                "printf '%s\\\\n' \\"$@\\"\\n"
                "exit 23\\n",
                encoding="utf-8",
            )
            native_nanogpt.chmod(0o755)
            os.environ["NFN_NATIVE_NANOGPT_CLI"] = str(native_nanogpt)
            sys.path.insert(0, str(root / "cli" / "scripts"))
            sys.argv = [str(script), "--tinystories", "--max-steps", "2", "--native-cuda-dry-run"]
            runpy.run_path(str(script), run_name="__main__")
            """
        )
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.pop("NFN_ALLOW_TORCH_TRAINING", None)
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=NEURALFN_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(23, proc.returncode)
        self.assertIn("NANOGPT_NATIVE_DIRECT", proc.stdout)
        self.assertNotIn("--base-model", proc.stdout)
        self.assertIn("--train-token-lm", proc.stdout)
        self.assertIn("--tinystories", proc.stdout)
        self.assertIn("--max-steps", proc.stdout)
        self.assertIn("--native-cuda-dry-run", proc.stdout)

    def test_nfn_nanogpt_train_defaults_to_native_token_lm(self) -> None:
        code = textwrap.dedent(
            f"""
            from pathlib import Path
            import os
            import runpy
            import sys
            import tempfile

            root = Path({str(NEURALFN_ROOT)!r})
            native_train = Path(tempfile.mkdtemp()) / "nfn_native_train"
            native_train.write_text(
                "#!/usr/bin/env bash\\n"
                "printf '%s\\\\n' \\"$@\\"\\n"
                "exit 23\\n",
                encoding="utf-8",
            )
            native_train.chmod(0o755)
            os.environ["NFN_NATIVE_TRAIN_CLI"] = str(native_train)
            sys.argv = [
                str(root / "cli" / "nfn.py"),
                "train",
                "--base-model",
                "nanogpt",
                "--tinystories",
                "--max-steps",
                "2",
                "--native-cuda-dry-run",
            ]
            runpy.run_path(str(root / "cli" / "nfn.py"), run_name="__main__")
            """
        )
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.pop("NFN_ALLOW_TORCH_TRAINING", None)
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=NEURALFN_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(23, proc.returncode)
        self.assertIn("--base-model", proc.stdout)
        self.assertIn("nanogpt", proc.stdout)
        self.assertIn("--train-token-lm", proc.stdout)
        self.assertIn("--tinystories", proc.stdout)
        self.assertIn("--max-steps", proc.stdout)
        self.assertIn("--dry-run", proc.stdout)

    def test_direct_nfn_script_dispatches_to_native_gpt2_without_pythonpath(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = Path(tmpdir) / "tiny"
            dataset.mkdir()
            (dataset / "data.txt").write_text("hello world. " * 128, encoding="utf-8")
            (dataset / "val.txt").write_text("validation story. " * 64, encoding="utf-8")
            (dataset / "meta.json").write_text(json.dumps({"data_format": "raw_text"}), encoding="utf-8")

            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            env["NFN_NATIVE_GPT2_BINDING"] = "0"
            env["NFN_NATIVE_GPT2_CLI"] = str(NEURALFN_ROOT / "build" / "missing-test-native-cli")
            env["NFN_NATIVE_GPT2_LAUNCHER"] = str(NEURALFN_ROOT / "build" / "missing-test-launcher")
            proc = subprocess.run(
                [
                    sys.executable,
                    str(NEURALFN_ROOT / "cli" / "nfn.py"),
                    "train",
                    "--base-model",
                    "gpt2",
                    "--dataset-alias",
                    str(dataset),
                    "--no-download-if-missing",
                    "--no-tile-cuda-strict",
                    "--native-cuda-runner",
                    "subprocess",
                    "--native-cuda-dry-run",
                    "--native-cuda-print-command",
                    "--eval-every-steps",
                    "1000",
                ],
                cwd=NEURALFN_ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertNotIn("ModuleNotFoundError", proc.stderr)
        self.assertIn("Native CUDA runner: subprocess (requested=subprocess)", proc.stdout)
        self.assertIn("Native CUDA validation eval: every 1000 optimizer steps", proc.stdout)
        self.assertIn("train_gpt2cu -i", proc.stdout)

    def test_direct_gpt2_script_dispatches_to_native_without_pythonpath(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = Path(tmpdir) / "tiny"
            dataset.mkdir()
            (dataset / "data.txt").write_text("hello world. " * 128, encoding="utf-8")
            (dataset / "val.txt").write_text("validation story. " * 64, encoding="utf-8")
            (dataset / "meta.json").write_text(json.dumps({"data_format": "raw_text"}), encoding="utf-8")

            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            env["NFN_NATIVE_GPT2_BINDING"] = "0"
            env["NFN_NATIVE_GPT2_CLI"] = str(NEURALFN_ROOT / "build" / "missing-test-native-cli")
            env["NFN_NATIVE_GPT2_LAUNCHER"] = str(NEURALFN_ROOT / "build" / "missing-test-launcher")
            proc = subprocess.run(
                [
                    sys.executable,
                    str(NEURALFN_ROOT / "cli" / "scripts" / "train_gpt2.py"),
                    "--dataset-alias",
                    str(dataset),
                    "--no-download-if-missing",
                    "--native-cuda-runner",
                    "subprocess",
                    "--native-cuda-dry-run",
                    "--native-cuda-print-command",
                    "--eval-every-steps",
                    "1000",
                ],
                cwd=NEURALFN_ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertNotIn("ModuleNotFoundError", proc.stderr)
        self.assertIn("Native CUDA runner: subprocess (requested=subprocess)", proc.stdout)
        self.assertIn("Native CUDA validation eval: every 1000 optimizer steps", proc.stdout)
        self.assertIn("train_gpt2cu -i", proc.stdout)

    def test_direct_gpt_script_uses_fast_path_without_compat_imports(self) -> None:
        code = textwrap.dedent(
            f"""
            from pathlib import Path
            import runpy
            import sys

            root = Path({str(NEURALFN_ROOT)!r})
            sys.argv = [
                str(root / "cli" / "scripts" / "train_gpt.py"),
                "--model-family",
                "gpt3",
                "--dataset-alias",
                "/tmp/native-cache",
                "--native-cuda-dry-run",
            ]
            try:
                runpy.run_path(str(root / "cli" / "scripts" / "train_gpt.py"), run_name="__main__")
            except SystemExit as exc:
                exit_code = int(exc.code or 0)
            else:
                exit_code = 0
            print("TORCH_LOADED", "torch" in sys.modules)
            print("TRAIN_GPT2_NATIVE_LOADED", "train_gpt2_native" in sys.modules)
            print("DATASET_MANAGER_LOADED", "server.dataset_manager" in sys.modules)
            print("NUMPY_LOADED", "numpy" in sys.modules)
            raise SystemExit(exit_code)
            """
        )
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env["NFN_NATIVE_GPT2_CLI"] = "/bin/echo"
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=NEURALFN_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("--model-family gpt3", proc.stdout)
        self.assertIn("--train-seq-len 2048", proc.stdout)
        self.assertIn("TORCH_LOADED False", proc.stdout)
        self.assertIn("TRAIN_GPT2_NATIVE_LOADED False", proc.stdout)
        self.assertIn("DATASET_MANAGER_LOADED False", proc.stdout)
        self.assertIn("NUMPY_LOADED False", proc.stdout)

    def test_train_gpt2_module_import_and_parser_do_not_import_torch(self) -> None:
        code = textwrap.dedent(
            f"""
            import importlib
            from pathlib import Path
            import sys

            root = Path({str(NEURALFN_ROOT)!r})
            sys.path.insert(0, str(root / "cli" / "scripts"))
            sys.path.insert(0, str(root))

            module = importlib.import_module("train_gpt2")
            parser = module.build_parser()
            args = parser.parse_args(["--native-cuda-dry-run"])
            module.resolve_mode_defaults(args)
            print("RUNTIME", args.runtime)
            print("RUNNER", args.native_cuda_runner)
            print("TORCH_LOADED", "torch" in sys.modules)
            print("DATASET_MANAGER_LOADED", "server.dataset_manager" in sys.modules)
            print("NUMPY_LOADED", "numpy" in sys.modules)
            """
        )
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=NEURALFN_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("RUNTIME native-cuda", proc.stdout)
        self.assertIn("RUNNER compiled-cli", proc.stdout)
        self.assertIn("TORCH_LOADED False", proc.stdout)
        self.assertIn("DATASET_MANAGER_LOADED False", proc.stdout)
        self.assertIn("NUMPY_LOADED False", proc.stdout)

    def test_train_gpt2_native_defaults_to_tinystories_not_parameter_golf(self) -> None:
        code = textwrap.dedent(
            f"""
            import importlib
            from pathlib import Path
            import sys

            root = Path({str(NEURALFN_ROOT)!r})
            sys.path.insert(0, str(root / "cli" / "scripts"))
            sys.path.insert(0, str(root))

            module = importlib.import_module("train_gpt2")
            parser = module.build_parser()
            args = parser.parse_args(["--native-cuda-dry-run"])
            print("DATASET_ALIAS", args.dataset_alias)
            print("DATASET_HF_PATH", args.dataset_hf_path)
            print("DATASET_TRAIN_FILE", args.dataset_train_file)
            print("DATASET_VAL_FILE", args.dataset_val_file)
            print("TORCH_LOADED", "torch" in sys.modules)
            """
        )
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=NEURALFN_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("DATASET_ALIAS roneneldan__TinyStories__TinyStoriesV2-GPT4", proc.stdout)
        self.assertIn("DATASET_HF_PATH roneneldan/TinyStories", proc.stdout)
        self.assertIn("DATASET_TRAIN_FILE TinyStoriesV2-GPT4-train.txt", proc.stdout)
        self.assertIn("DATASET_VAL_FILE TinyStoriesV2-GPT4-valid.txt", proc.stdout)
        self.assertNotIn("parameter-golf", proc.stdout)
        self.assertIn("TORCH_LOADED False", proc.stdout)

    def test_5090_shell_helpers_do_not_pass_wallclock_caps(self) -> None:
        for script_name in (
            "5090-llama-smoke.sh",
            "5090-llama-baseline.sh",
            "5090-llama-overnight.sh",
            "5090-mini-run.sh",
        ):
            with self.subTest(script_name=script_name):
                content = (NEURALFN_ROOT / "cli" / script_name).read_text(encoding="utf-8")
                self.assertNotIn("--max-wallclock-seconds", content)
                self.assertIn("--max-steps 20000", content)
                self.assertIn("--optimizer-profile adamw", content)

    def test_nfn_train_gpt2_rejects_torch_runtime_without_importing_torch(self) -> None:
        code = textwrap.dedent(
            f"""
            from pathlib import Path
            import runpy
            import sys

            root = Path({str(NEURALFN_ROOT)!r})
            sys.argv = [
                str(root / "cli" / "nfn.py"),
                "train",
                "--base-model",
                "gpt2",
                "--runtime",
                "torch",
                "--native-cuda-dry-run",
            ]
            try:
                runpy.run_path(str(root / "cli" / "nfn.py"), run_name="__main__")
            except SystemExit as exc:
                exit_code = int(exc.code or 0)
            else:
                exit_code = 0
            print("TORCH_LOADED", "torch" in sys.modules)
            print("NFN_IMPL_LOADED", "nfn_impl" in sys.modules)
            raise SystemExit(exit_code)
            """
        )
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=NEURALFN_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(2, proc.returncode)
        self.assertIn("invalid choice: 'torch'", proc.stderr)
        self.assertIn("TORCH_LOADED False", proc.stdout)
        self.assertIn("NFN_IMPL_LOADED False", proc.stdout)

    def test_native_gpt2_script_honors_nfn_datasets_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            datasets_dir = Path(tmpdir) / "datasets"
            dataset = datasets_dir / "alias"
            dataset.mkdir(parents=True)
            (dataset / "data.txt").write_text("hello world. " * 128, encoding="utf-8")
            (dataset / "val.txt").write_text("validation story. " * 64, encoding="utf-8")
            (dataset / "meta.json").write_text(json.dumps({"data_format": "raw_text"}), encoding="utf-8")

            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            env["NFN_DATASETS_DIR"] = str(datasets_dir)
            env["NFN_NATIVE_GPT2_BINDING"] = "0"
            env["NFN_NATIVE_GPT2_CLI"] = str(NEURALFN_ROOT / "build" / "missing-test-native-cli")
            env["NFN_NATIVE_GPT2_LAUNCHER"] = str(NEURALFN_ROOT / "build" / "missing-test-launcher")
            proc = subprocess.run(
                [
                    sys.executable,
                    str(NEURALFN_ROOT / "cli" / "scripts" / "train_gpt2_native.py"),
                    "--dataset-alias",
                    "alias",
                    "--no-download-if-missing",
                    "--native-cuda-dry-run",
                    "--native-cuda-print-command",
                ],
                cwd=NEURALFN_ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn(f"Using dataset: alias", proc.stdout)
        self.assertIn(f"Native CUDA dataset path: {dataset}", proc.stdout)
        self.assertNotIn(str(dataset / "fineweb_train_000000.bin"), proc.stdout)
        self.assertNotIn(str(Path.home() / ".cache" / "nfn" / "datasets"), proc.stdout)
        self.assertIn("Native CUDA runner: compiled-cli (requested=compiled-cli)", proc.stdout)
        self.assertIn("Native CUDA shard resolution: compiled C++ frontend (deferred dry-run)", proc.stdout)
        self.assertIn("--dataset-alias alias", proc.stdout)
        self.assertNotIn("--target ", proc.stdout)
        self.assertNotIn("train_gpt2cu", proc.stdout)

    def test_infer_gpt2_parser_and_help_do_not_import_torch(self) -> None:
        cases = (
            "import",
            "parse-evo-defaults",
            "help",
        )
        for case in cases:
            with self.subTest(case=case):
                code = textwrap.dedent(
                    f"""
                    import importlib
                    from pathlib import Path
                    import runpy
                    import sys

                    root = Path({str(NEURALFN_ROOT)!r})
                    sys.path.insert(0, str(root / "cli" / "scripts"))
                    sys.path.insert(0, str(root))

                    if {case!r} == "help":
                        sys.argv = [str(root / "cli" / "scripts" / "infer_gpt2.py"), "--evo", "--help"]
                        try:
                            runpy.run_path(str(root / "cli" / "scripts" / "infer_gpt2.py"), run_name="__main__")
                        except SystemExit as exc:
                            exit_code = int(exc.code or 0)
                        else:
                            exit_code = 0
                    else:
                        module = importlib.import_module("infer_gpt2")
                        exit_code = 0
                        if {case!r} == "parse-evo-defaults":
                            parser = module.build_parser()
                            args = parser.parse_args(["--evo"])
                            module.resolve_mode_defaults(args)
                            print("GRAPH", args.graph)
                            print("WEIGHTS", args.weights)
                    print("TORCH_LOADED", "torch" in sys.modules)
                    print("DATASET_MANAGER_LOADED", "server.dataset_manager" in sys.modules)
                    print("NUMPY_LOADED", "numpy" in sys.modules)
                    raise SystemExit(exit_code)
                    """
                )
                env = os.environ.copy()
                env.pop("PYTHONPATH", None)
                proc = subprocess.run(
                    [sys.executable, "-c", code],
                    cwd=NEURALFN_ROOT,
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )

                self.assertEqual(0, proc.returncode, proc.stderr)
                if case == "parse-evo-defaults":
                    self.assertIn("gpt2_evo.json", proc.stdout)
                    self.assertIn("gpt2_evo.pt", proc.stdout)
                self.assertIn("TORCH_LOADED False", proc.stdout)
                self.assertIn("DATASET_MANAGER_LOADED False", proc.stdout)
                self.assertIn("NUMPY_LOADED False", proc.stdout)

    def test_infer_gpt2_native_checkpoint_info_does_not_import_torch(self) -> None:
        code = textwrap.dedent(
            f"""
            from pathlib import Path
            import runpy
            import struct
            import sys
            import tempfile

            root = Path({str(NEURALFN_ROOT)!r})
            sys.path.insert(0, str(root / "cli" / "scripts"))
            sys.path.insert(0, str(root))
            from neuralfn.native_gpt2 import native_gpt2_parameter_count

            with tempfile.TemporaryDirectory() as tmpdir:
                checkpoint = Path(tmpdir) / "model_00000010.bin"
                header = [0] * 256
                header[:8] = [20240326, 5, 8, 16, 1, 1, 4, 16]
                nparams = native_gpt2_parameter_count(
                    max_seq_len=8,
                    padded_vocab_size=16,
                    num_layers=1,
                    channels=4,
                )
                checkpoint.write_bytes(struct.pack("<" + "i" * 256, *header) + b"\\0" * (nparams * 2))
                (Path(tmpdir) / "DONE_00000010").write_text("", encoding="utf-8")
                sys.argv = [
                    str(root / "cli" / "scripts" / "infer_gpt2.py"),
                    "--native-checkpoint",
                    str(checkpoint),
                    "--native-info",
                ]
                try:
                    runpy.run_path(str(root / "cli" / "scripts" / "infer_gpt2.py"), run_name="__main__")
                except SystemExit as exc:
                    exit_code = int(exc.code or 0)
                else:
                    exit_code = 0
                print("TORCH_LOADED", "torch" in sys.modules)
                print("DATASET_MANAGER_LOADED", "server.dataset_manager" in sys.modules)
                print("NUMPY_LOADED", "numpy" in sys.modules)
                raise SystemExit(exit_code)
            """
        )
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=NEURALFN_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("Native GPT-2 checkpoint detected", proc.stdout)
        self.assertIn("precision: bf16", proc.stdout)
        self.assertIn("checkpoint_step: 10", proc.stdout)
        self.assertIn("TORCH_LOADED False", proc.stdout)
        self.assertIn("DATASET_MANAGER_LOADED False", proc.stdout)
        self.assertIn("NUMPY_LOADED False", proc.stdout)

    def test_nfn_infer_native_checkpoint_is_recognized_without_nfn_impl(self) -> None:
        code = textwrap.dedent(
            f"""
            from pathlib import Path
            import runpy
            import struct
            import sys
            import tempfile

            root = Path({str(NEURALFN_ROOT)!r})
            sys.path.insert(0, str(root))
            from neuralfn.native_gpt2 import native_gpt2_parameter_count

            with tempfile.TemporaryDirectory() as tmpdir:
                checkpoint = Path(tmpdir) / "model_00000010.bin"
                header = [0] * 256
                header[:8] = [20240326, 5, 8, 16, 1, 1, 4, 16]
                nparams = native_gpt2_parameter_count(
                    max_seq_len=8,
                    padded_vocab_size=16,
                    num_layers=1,
                    channels=4,
                )
                checkpoint.write_bytes(struct.pack("<" + "i" * 256, *header) + b"\\0" * (nparams * 2))
                (Path(tmpdir) / "DONE_00000010").write_text("", encoding="utf-8")
                sys.argv = [
                    str(root / "cli" / "nfn.py"),
                    "infer",
                    "--checkpoint",
                    str(checkpoint),
                    "--native-info",
                ]
                try:
                    runpy.run_path(str(root / "cli" / "nfn.py"), run_name="__main__")
                except SystemExit as exc:
                    exit_code = int(exc.code or 0)
                else:
                    exit_code = 0
                print("TORCH_LOADED", "torch" in sys.modules)
                print("NFN_IMPL_LOADED", "nfn_impl" in sys.modules)
                raise SystemExit(exit_code)
            """
        )
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=NEURALFN_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("Native GPT-2 checkpoint detected", proc.stdout)
        self.assertIn("Native GPT-2 prompt inference is not wired yet", proc.stdout)
        self.assertIn("TORCH_LOADED False", proc.stdout)
        self.assertIn("NFN_IMPL_LOADED False", proc.stdout)


if __name__ == "__main__":
    unittest.main()
