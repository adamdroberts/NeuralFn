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
    def test_native_gpt_dataset_download_is_explicit_opt_in(self) -> None:
        code = textwrap.dedent(
            f"""
            from pathlib import Path
            import sys

            root = Path({str(NEURALFN_ROOT)!r})
            sys.path.insert(0, str(root / "cli" / "scripts"))
            sys.path.insert(0, str(root))

            from train_gpt_native import build_parser

            default_args = build_parser().parse_args([])
            explicit_args = build_parser().parse_args(["--download-if-missing"])
            print("DEFAULT_DOWNLOAD", default_args.download_if_missing)
            print("EXPLICIT_DOWNLOAD", explicit_args.download_if_missing)
            print("DATASET_MANAGER_LOADED", "server.dataset_manager" in sys.modules)
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
        self.assertIn("DEFAULT_DOWNLOAD False", proc.stdout)
        self.assertIn("EXPLICIT_DOWNLOAD True", proc.stdout)
        self.assertIn("DATASET_MANAGER_LOADED False", proc.stdout)
        self.assertIn("TORCH_LOADED False", proc.stdout)

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

    def test_nfn_console_entry_train_dispatches_native_without_python_harness(self) -> None:
        code = textwrap.dedent(
            f"""
            from pathlib import Path
            import sys

            root = Path({str(NEURALFN_ROOT)!r})
            sys.path.insert(0, str(root / "cli"))
            sys.path.insert(0, str(root / "cli" / "scripts"))
            sys.path.insert(0, str(root))
            sys.argv = [
                "nfn",
                "train",
                "--base-model",
                "gpt",
                "--dataset-alias=/tmp/native-cache",
                "--native-cuda-dry-run",
                "--eval-every-steps=1000",
            ]

            from nfn import main

            exit_code = int(main() or 0)
            print("TORCH_LOADED", "torch" in sys.modules)
            print("NFN_IMPL_LOADED", "nfn_impl" in sys.modules)
            print("TRAIN_GPT_NATIVE_LOADED", "train_gpt_native" in sys.modules)
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
        self.assertIn("--model-family gpt", proc.stdout)
        self.assertIn("--dataset-alias /tmp/native-cache", proc.stdout)
        self.assertIn("--eval-every-steps 1000", proc.stdout)
        self.assertIn("--dry-run", proc.stdout)
        self.assertIn("--train-transformer-lm", proc.stdout)
        self.assertIn("TORCH_LOADED False", proc.stdout)
        self.assertIn("NFN_IMPL_LOADED False", proc.stdout)
        self.assertIn("TRAIN_GPT_NATIVE_LOADED False", proc.stdout)

    def test_nfn_programmatic_train_dispatches_native_without_python_harness(self) -> None:
        code = textwrap.dedent(
            f"""
            from pathlib import Path
            import sys

            root = Path({str(NEURALFN_ROOT)!r})
            sys.path.insert(0, str(root / "cli"))
            sys.path.insert(0, str(root / "cli" / "scripts"))
            sys.path.insert(0, str(root))

            from nfn import main

            exit_code = int(main(
                [
                    "train",
                    "--base-model",
                    "gpt",
                    "--dataset-alias=/tmp/native-cache",
                    "--native-cuda-dry-run",
                    "--native-cuda-print-command",
                    "--eval-every-steps=1000",
                ],
                stdin_isatty=False,
                stdout_isatty=False,
            ) or 0)
            print("TORCH_LOADED", "torch" in sys.modules)
            print("NFN_IMPL_LOADED", "nfn_impl" in sys.modules)
            print("TRAIN_GPT_NATIVE_LOADED", "train_gpt_native" in sys.modules)
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
        self.assertIn("--model-family gpt", proc.stdout)
        self.assertIn("--dataset-alias /tmp/native-cache", proc.stdout)
        self.assertIn("--eval-every-steps 1000", proc.stdout)
        self.assertIn("--dry-run", proc.stdout)
        self.assertIn("--print-command", proc.stdout)
        self.assertIn("--train-transformer-lm", proc.stdout)
        self.assertIn("TORCH_LOADED False", proc.stdout)
        self.assertIn("NFN_IMPL_LOADED False", proc.stdout)
        self.assertIn("TRAIN_GPT_NATIVE_LOADED False", proc.stdout)

    def test_train_gpt2_evo_module_import_is_native_only(self) -> None:
        code = textwrap.dedent(
            f"""
            from pathlib import Path
            import builtins
            import sys

            blocked = {{"torch", "numpy", "server.dataset_manager", "neuralfn.torch_backend"}}
            real_import = builtins.__import__

            def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
                if name in blocked or any(name.startswith(prefix + ".") for prefix in blocked):
                    raise AssertionError(f"blocked import: {{name}}")
                return real_import(name, globals, locals, fromlist, level)

            root = Path({str(NEURALFN_ROOT)!r})
            sys.path.insert(0, str(root / "cli" / "scripts"))
            builtins.__import__ = guarded_import
            try:
                import train_gpt2_evo
            finally:
                builtins.__import__ = real_import

            print("MODE_NAME", train_gpt2_evo.MODE_NAME)
            print("TORCH_LOADED", "torch" in sys.modules)
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
        self.assertIn("MODE_NAME gpt2_evo", proc.stdout)
        self.assertIn("TORCH_LOADED False", proc.stdout)
        self.assertIn("NUMPY_LOADED False", proc.stdout)

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
                print("TRAIN_GPT_NATIVE_LOADED", "train_gpt_native" in sys.modules)
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
        self.assertIn("--backend tile-cuda", proc.stdout)
        self.assertNotIn("train_gpt2cu", proc.stdout)
        self.assertIn("TORCH_LOADED False", proc.stdout)
        self.assertIn("DATASET_MANAGER_LOADED False", proc.stdout)
        self.assertIn("NUMPY_LOADED False", proc.stdout)
        self.assertIn("TIKTOKEN_LOADED False", proc.stdout)

    def test_native_cached_shard_print_command_stays_metadata_only(self) -> None:
        code = textwrap.dedent(
            f"""
            import json
            import os
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
                native_cli = Path(tmpdir) / "nfn_gpt_native_train"
                native_cli.write_text(
                    "#!/usr/bin/env bash\\n"
                    "printf 'NATIVE_CLI_ENV CUDA_VISIBLE_DEVICES=%s CUDA_DEVICE_MAX_CONNECTIONS=%s CUDA_MODULE_LOADING=%s\\n' "
                    "\\"$CUDA_VISIBLE_DEVICES\\" \\"$CUDA_DEVICE_MAX_CONNECTIONS\\" \\"$CUDA_MODULE_LOADING\\"\\n"
                    "printf '%s\\n' \\"$@\\"\\n",
                    encoding="utf-8",
                )
                native_cli.chmod(0o755)
                os.environ["NFN_NATIVE_GPT2_CLI"] = str(native_cli)
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
                    "--native-cuda-no-checkpoint",
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
                print("TRAIN_GPT_NATIVE_LOADED", "train_gpt_native" in sys.modules)
                raise SystemExit(exit_code)
            """
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{NEURALFN_ROOT / 'cli' / 'scripts'}:{NEURALFN_ROOT}"
        env.pop("NFN_NATIVE_GPT2_CLI", None)
        env.pop("CUDA_MODULE_LOADING", None)
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
        self.assertNotIn("NATIVE_CLI_ENV", proc.stdout)
        self.assertIn("nfn_gpt_native_train", proc.stdout)
        self.assertIn("--dataset-alias", proc.stdout)
        self.assertRegex(proc.stdout, r"--dataset-alias /tmp/.*/tiny")
        self.assertNotIn("--target ", proc.stdout)
        self.assertNotIn("train_gpt2cu", proc.stdout)
        self.assertIn("--train-transformer-lm", proc.stdout)
        self.assertIn("--no-checkpoint", proc.stdout)
        self.assertIn("TORCH_LOADED False", proc.stdout)
        self.assertIn("DATASET_MANAGER_LOADED False", proc.stdout)
        self.assertIn("NUMPY_LOADED False", proc.stdout)
        self.assertIn("TIKTOKEN_LOADED False", proc.stdout)
        self.assertIn("TRAIN_GPT_NATIVE_LOADED False", proc.stdout)

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
                print("TRAIN_GPT_NATIVE_LOADED", "train_gpt_native" in sys.modules)
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
        self.assertIn("TRAIN_GPT_NATIVE_LOADED False", proc.stdout)

    def test_native_dry_run_llm_kittens_backend_is_rejected(self) -> None:
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

        self.assertEqual(2, proc.returncode)
        self.assertIn("native GPT kernel backend must be tile-cuda", proc.stderr)
        self.assertNotIn("train_gpt2cu", proc.stdout)
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
        self.assertIn("--eval-every-steps 1000", proc.stdout)
        self.assertIn("--backend tile-cuda", proc.stdout)
        self.assertNotIn("train_gpt2cu", proc.stdout)
        self.assertIn("TORCH_LOADED False", proc.stdout)

    def test_nfn_train_gpt_native_dry_run_does_not_import_torch(self) -> None:
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
        self.assertIn("--eval-every-steps 1000", proc.stdout)
        self.assertIn("--backend tile-cuda", proc.stdout)
        self.assertNotIn("train_gpt2cu", proc.stdout)
        self.assertIn("TORCH_LOADED False", proc.stdout)

    def test_train_gpt_native_direct_dry_run_prefers_linked_cli(self) -> None:
        code = textwrap.dedent(
            f"""
            from pathlib import Path
            import runpy
            import sys
            import tempfile

            root = Path({str(NEURALFN_ROOT)!r})
            sys.path.insert(0, str(root / "cli" / "scripts"))
            sys.path.insert(0, str(root))

            import neuralfn.native_gpt2 as native_gpt2

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                linked_cli = tmp / "nfn_gpt_native_train_linked"
                dynamic_cli = tmp / "nfn_gpt_native_train"
                linked_cli.write_text("#!/usr/bin/env bash\\nexit 0\\n", encoding="utf-8")
                dynamic_cli.write_text("#!/usr/bin/env bash\\nexit 0\\n", encoding="utf-8")
                linked_cli.chmod(0o755)
                dynamic_cli.chmod(0o755)
                native_gpt2.DEFAULT_NATIVE_GPT_CLI_LINKED = str(linked_cli)
                native_gpt2.DEFAULT_NATIVE_GPT2_CLI = str(dynamic_cli)

                sys.argv = [
                    str(root / "cli" / "scripts" / "train_gpt_native.py"),
                    "--dataset-alias",
                    str(tmp / "cached-shards"),
                    "--no-download-if-missing",
                    "--native-cuda-dry-run",
                    "--native-cuda-print-command",
                    "--native-cuda-no-checkpoint",
                ]
                try:
                    runpy.run_path(str(root / "cli" / "scripts" / "train_gpt_native.py"), run_name="__main__")
                except SystemExit as exc:
                    exit_code = int(exc.code or 0)
                else:
                    exit_code = 0
                print("TORCH_LOADED", "torch" in sys.modules)
                print("DATASET_MANAGER_LOADED", "server.dataset_manager" in sys.modules)
                raise SystemExit(exit_code)
            """
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{NEURALFN_ROOT / 'cli' / 'scripts'}:{NEURALFN_ROOT}"
        env.pop("NFN_NATIVE_GPT_CLI", None)
        env.pop("NFN_NATIVE_GPT2_CLI", None)
        env.pop("NFN_NATIVE_GPT_LINKED_CLI", None)
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
        self.assertIn("nfn_gpt_native_train_linked", proc.stdout)
        self.assertIn("--tile-ops-lib linked", proc.stdout)
        self.assertIn("--train-transformer-lm", proc.stdout)
        self.assertIn("--no-checkpoint", proc.stdout)
        self.assertIn("TORCH_LOADED False", proc.stdout)
        self.assertIn("DATASET_MANAGER_LOADED False", proc.stdout)

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
                "/tmp/gpt.pt",
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
            print("TRAIN_GPT_NATIVE_LOADED", "train_gpt_native" in sys.modules)
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
        self.assertIn("--output-dir /tmp/gpt", proc.stdout)
        self.assertIn("--eval-every-steps 1000", proc.stdout)
        self.assertIn("--native-cuda-activation sd-prelu", proc.stdout)
        self.assertIn("--dry-run", proc.stdout)
        self.assertIn("--print-command", proc.stdout)
        self.assertNotIn("--base-model", proc.stdout)
        self.assertIn("TORCH_LOADED False", proc.stdout)
        self.assertIn("NFN_IMPL_LOADED False", proc.stdout)
        self.assertIn("TRAIN_GPT_NATIVE_LOADED False", proc.stdout)
        self.assertIn("DATASET_MANAGER_LOADED False", proc.stdout)

    def test_train_gpt_print_command_dry_run_stops_before_native_spawn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            native_cli = Path(tmp) / "nfn_gpt_native_train"
            native_cli.write_text("#!/usr/bin/env bash\nexit 99\n", encoding="utf-8")
            native_cli.chmod(0o755)
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            env["NFN_NATIVE_GPT_CLI"] = str(native_cli)
            proc = subprocess.run(
                [
                    sys.executable,
                    str(NEURALFN_ROOT / "cli" / "scripts" / "train_gpt.py"),
                    "--tinystories",
                    "--native-cuda-dry-run",
                    "--native-cuda-print-command",
                    "--native-cuda-no-checkpoint",
                ],
                cwd=NEURALFN_ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("nfn_gpt_native_train", proc.stdout)
        self.assertIn("--dry-run", proc.stdout)
        self.assertIn("--print-command", proc.stdout)
        self.assertIn("--train-transformer-lm", proc.stdout)
        self.assertEqual("", proc.stderr)

    def test_nfn_train_print_command_dry_run_stops_before_native_spawn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            native_cli = Path(tmp) / "nfn_gpt_native_train"
            native_cli.write_text("#!/usr/bin/env bash\nexit 99\n", encoding="utf-8")
            native_cli.chmod(0o755)
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            env["NFN_NATIVE_GPT_CLI"] = str(native_cli)
            proc = subprocess.run(
                [
                    sys.executable,
                    str(NEURALFN_ROOT / "cli" / "nfn.py"),
                    "train",
                    "--tinystories",
                    "--native-cuda-dry-run",
                    "--native-cuda-print-command",
                    "--no-checkpoint",
                ],
                cwd=NEURALFN_ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("nfn_gpt_native_train", proc.stdout)
        self.assertIn("--dry-run", proc.stdout)
        self.assertIn("--print-command", proc.stdout)
        self.assertIn("--train-transformer-lm", proc.stdout)
        self.assertEqual("", proc.stderr)

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
            print("TRAIN_GPT_NATIVE_LOADED", "train_gpt_native" in sys.modules)
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
        self.assertIn("TRAIN_GPT_NATIVE_LOADED False", proc.stdout)

    def test_nfn_train_dense_gpt_default_template_is_generic_gpt(self) -> None:
        code = textwrap.dedent(
            f"""
            from pathlib import Path
            import runpy
            import sys

            root = Path({str(NEURALFN_ROOT)!r})
            sys.path.insert(0, str(root / "cli"))
            import nfn

            print("DEFAULT_TEMPLATE", nfn._native_template_name([]))
            sys.argv = [
                str(root / "cli" / "nfn.py"),
                "train",
                "--base-model",
                "gpt",
                "--dataset-alias=/tmp/native-cache",
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
            print("TRAIN_GPT_NATIVE_LOADED", "train_gpt_native" in sys.modules)
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
        self.assertIn("DEFAULT_TEMPLATE gpt", proc.stdout)
        self.assertIn("--model-family gpt", proc.stdout)
        self.assertIn("--train-transformer-lm", proc.stdout)
        self.assertIn("TORCH_LOADED False", proc.stdout)
        self.assertIn("NFN_IMPL_LOADED False", proc.stdout)
        self.assertIn("TRAIN_GPT_NATIVE_LOADED False", proc.stdout)

    def test_nfn_train_native_dispatch_sets_cuda_compute_defaults(self) -> None:
        code = textwrap.dedent(
            f"""
            from pathlib import Path
            import os
            import runpy
            import stat
            import sys
            import tempfile
            import textwrap

            root = Path({str(NEURALFN_ROOT)!r})
            with tempfile.TemporaryDirectory() as tmpdir:
                stub = Path(tmpdir) / "native-train-stub.py"
                stub.write_text(
                    textwrap.dedent('''
                    #!/usr/bin/env python3
                    import os
                    print("STUB_CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"))
                    print("STUB_CUDA_DEVICE_MAX_CONNECTIONS", os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS"))
                    ''').lstrip(),
                    encoding="utf-8",
                )
                stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
                os.environ["NFN_NATIVE_TRAIN_CLI"] = str(stub)
                sys.argv = [
                    str(root / "cli" / "nfn.py"),
                    "train",
                    "--base-model",
                    "gpt",
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
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env.pop("CUDA_DEVICE_MAX_CONNECTIONS", None)
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
        self.assertIn("STUB_CUDA_VISIBLE_DEVICES 0", proc.stdout)
        self.assertIn("STUB_CUDA_DEVICE_MAX_CONNECTIONS 1", proc.stdout)
        self.assertIn("TORCH_LOADED False", proc.stdout)
        self.assertIn("NFN_IMPL_LOADED False", proc.stdout)

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
                    print("TRAIN_GPT_NATIVE_LOADED", "train_gpt_native" in sys.modules)
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
                self.assertIn("TRAIN_GPT_NATIVE_LOADED False", proc.stdout)

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
            print("TRAIN_GPT_NATIVE_LOADED", "train_gpt_native" in sys.modules)
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
        self.assertIn("TRAIN_GPT_NATIVE_LOADED False", proc.stdout)

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
                "--native-cuda-no-checkpoint",
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
        self.assertIn("--no-checkpoint", proc.stdout)
        self.assertNotIn("--native-cuda-no-checkpoint", proc.stdout)
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
                "printf '  gpt2: implemented -> nfn_gpt_native_train\\\\n' >&2\\n"
                "printf '  nanogpt: partial-native-trainer -> nfn_gpt_native_train\\\\n' >&2\\n"
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
            print("TRAIN_GPT_NATIVE_LOADED", "train_gpt_native" in sys.modules)
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
        self.assertIn("gpt2: implemented -> nfn_gpt_native_train", proc.stderr)
        self.assertIn("nanogpt: partial-native-trainer -> nfn_gpt_native_train", proc.stderr)
        self.assertIn("TORCH_LOADED False", proc.stdout)
        self.assertIn("NFN_IMPL_LOADED False", proc.stdout)
        self.assertIn("TRAIN_GPT_NATIVE_LOADED False", proc.stdout)

    def test_nfn_train_ignores_legacy_torch_training_env(self) -> None:
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
                "printf 'NATIVE_REGISTRY_USED\\\\n'\\n"
                "printf '%s\\\\n' \\"$@\\"\\n"
                "exit 2\\n",
                encoding="utf-8",
            )
            native_train.chmod(0o755)
            os.environ["NFN_NATIVE_TRAIN_CLI"] = str(native_train)
            os.environ["NFN_ALLOW_TORCH_TRAINING"] = "1"
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
        self.assertIn("NATIVE_REGISTRY_USED", proc.stdout)
        self.assertIn("--base-model", proc.stdout)
        self.assertIn("llama", proc.stdout)
        self.assertIn("TORCH_LOADED False", proc.stdout)
        self.assertIn("NFN_IMPL_LOADED False", proc.stdout)

    def test_nfn_train_prefers_family_native_cli_for_every_native_family(self) -> None:
        cases = (
            ("gpt2-evo", ()),
            ("llama", ()),
            ("mixllama", ()),
            ("jepa", ()),
            ("semantic-router-moe", ()),
            ("deepseek-v4", ()),
            ("nanogpt", ("--train-token-lm",)),
        )
        for model_family, extra_args in cases:
            with self.subTest(model_family=model_family):
                code = textwrap.dedent(
                    f"""
                    from pathlib import Path
                    import os
                    import runpy
                    import sys
                    import tempfile

                    root = Path({str(NEURALFN_ROOT)!r})
                    model_family = {model_family!r}
                    extra_args = {list(extra_args)!r}
                    family_env = "NFN_NATIVE_" + "".join(ch if ch.isalnum() else "_" for ch in model_family.upper()).strip("_") + "_CLI"
                    family_cli = Path(tempfile.mkdtemp()) / ("nfn_" + "".join(ch if ch.isalnum() else "_" for ch in model_family.lower()).strip("_") + "_native_train")
                    family_cli.write_text(
                        "#!/usr/bin/env bash\\n"
                        "printf 'NFN_FAMILY_NATIVE_DIRECT\\\\n'\\n"
                        "printf '%s\\\\n' \\"$@\\"\\n"
                        "exit 21\\n",
                        encoding="utf-8",
                    )
                    family_cli.chmod(0o755)
                    os.environ[family_env] = str(family_cli)
                    sys.argv = [
                        str(root / "cli" / "nfn.py"),
                        "train",
                        "--base-model",
                        model_family,
                        "--tinystories",
                        "--native-cuda-dry-run",
                        *extra_args,
                    ]
                    try:
                        runpy.run_path(str(root / "cli" / "nfn.py"), run_name="__main__")
                    except SystemExit as exc:
                        exit_code = int(exc.code or 0)
                    else:
                        exit_code = 0
                    print("TORCH_LOADED", "torch" in sys.modules)
                    print("NFN_IMPL_LOADED", "nfn_impl" in sys.modules)
                    print("TRAIN_GPT_NATIVE_LOADED", "train_gpt_native" in sys.modules)
                    raise SystemExit(exit_code)
                    """
                )
                env = os.environ.copy()
                env.pop("PYTHONPATH", None)
                env.pop("NFN_NATIVE_TRAIN_CLI", None)
                proc = subprocess.run(
                    [sys.executable, "-c", code],
                    cwd=NEURALFN_ROOT,
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )

                self.assertEqual(21, proc.returncode)
                self.assertIn("NFN_FAMILY_NATIVE_DIRECT", proc.stdout)
                self.assertIn("--tinystories", proc.stdout)
                self.assertIn("--dry-run", proc.stdout)
                self.assertNotIn("--base-model", proc.stdout)
                if extra_args:
                    for arg in extra_args:
                        self.assertIn(arg, proc.stdout)
                self.assertIn("TORCH_LOADED False", proc.stdout)
                self.assertIn("NFN_IMPL_LOADED False", proc.stdout)
                self.assertIn("TRAIN_GPT_NATIVE_LOADED False", proc.stdout)

    def test_legacy_training_scripts_reject_before_torch_import(self) -> None:
        scripts = (
            ("train_llama_fast.py", "llama"),
            ("train_llama_megakernel.py", "llama"),
            ("train_nanogpt.py", "gpt"),
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

    def test_legacy_training_scripts_ignore_legacy_torch_training_env(self) -> None:
        code = textwrap.dedent(
            f"""
            from pathlib import Path
            import os
            import runpy
            import sys
            import tempfile

            root = Path({str(NEURALFN_ROOT)!r})
            script = root / "cli" / "scripts" / "train_llama_fast.py"
            family_cli = Path(tempfile.mkdtemp()) / "nfn_llama_native_train"
            family_cli.write_text(
                "#!/usr/bin/env bash\\n"
                "printf 'FAMILY_NATIVE_DIRECT\\\\n'\\n"
                "printf '%s\\\\n' \\"$@\\"\\n"
                "exit 2\\n",
                encoding="utf-8",
            )
            family_cli.chmod(0o755)
            os.environ["NFN_NATIVE_LLAMA_CLI"] = str(family_cli)
            os.environ["NFN_ALLOW_TORCH_TRAINING"] = "1"
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
                "printf 'EVO_NATIVE_ENV CUDA_VISIBLE_DEVICES=%s CUDA_DEVICE_MAX_CONNECTIONS=%s CUDA_MODULE_LOADING=%s\\\\n' "
                "\\"$CUDA_VISIBLE_DEVICES\\" \\"$CUDA_DEVICE_MAX_CONNECTIONS\\" \\"$CUDA_MODULE_LOADING\\"\\n"
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
        self.assertIn(
            "EVO_NATIVE_ENV CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1 CUDA_MODULE_LOADING=LAZY",
            proc.stdout,
        )
        self.assertNotIn("--base-model", proc.stdout)
        self.assertIn("--tinystories", proc.stdout)
        self.assertIn("--dry-run", proc.stdout)
        self.assertNotIn("--native-cuda-dry-run", proc.stdout)
        self.assertIn("--eval-every-steps", proc.stdout)
        self.assertIn("1000", proc.stdout)
        self.assertIn("TORCH_LOADED False", proc.stdout)
        self.assertIn("TRAIN_JEPA_LOADED False", proc.stdout)

    def test_train_gpt2_evo_direct_script_normalizes_native_cuda_preflight_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            native_evo = Path(tmpdir) / "nfn_gpt2_evo_native_train"
            native_evo.write_text(
                "#!/usr/bin/env bash\n"
                "printf 'EVO_NATIVE_DIRECT\\n'\n"
                "printf '%s\\n' \"$@\"\n"
                "exit 23\n",
                encoding="utf-8",
            )
            native_evo.chmod(0o755)
            env = os.environ.copy()
            env["NFN_NATIVE_GPT2_EVO_CLI"] = str(native_evo)
            env.pop("PYTHONPATH", None)
            env.pop("NFN_ALLOW_TORCH_TRAINING", None)
            proc = subprocess.run(
                [
                    sys.executable,
                    str(NEURALFN_ROOT / "cli" / "scripts" / "train_gpt2_evo.py"),
                    "--native-cuda-dry-run",
                    "--native-cuda-print-command",
                    "--native-cuda-print-plan",
                    "--native-cuda-smoke-evo-kernels",
                    "--native-cuda-tile-ops-lib",
                    "/tmp/libnfn_native_train_tile_ops.so",
                    "--native-cuda-cuda-runtime-lib=/tmp/libcudart.so",
                    "--native-cuda-no-checkpoint",
                    "--native-cuda-kernel-backend",
                    "tile-cuda",
                    "--native-cuda-output-dir=/tmp/nfn-out",
                    "--native-cuda-lm-head-row-chunk-size",
                    "32768",
                    "--native-cuda-checkpoint-every",
                    "0",
                    "--native-cuda-sample-every=0",
                    "--native-cuda-generate-tokens",
                    "32",
                    "--native-cuda-activation",
                    "moa",
                    "--native-cuda-moa-interval=25",
                    "--template",
                    "gpt2-moa",
                ],
                cwd=NEURALFN_ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertEqual(23, proc.returncode)
        self.assertIn("EVO_NATIVE_DIRECT", proc.stdout)
        self.assertIn("--dry-run", proc.stdout)
        self.assertNotIn("--native-cuda-dry-run", proc.stdout)
        self.assertIn("--print-command", proc.stdout)
        self.assertNotIn("--native-cuda-print-command", proc.stdout)
        self.assertIn("--print-plan", proc.stdout)
        self.assertNotIn("--native-cuda-print-plan", proc.stdout)
        self.assertIn("--smoke-evo-kernels", proc.stdout)
        self.assertNotIn("--native-cuda-smoke-evo-kernels", proc.stdout)
        self.assertIn("--tile-ops-lib", proc.stdout)
        self.assertIn("/tmp/libnfn_native_train_tile_ops.so", proc.stdout)
        self.assertIn("--cuda-runtime-lib=/tmp/libcudart.so", proc.stdout)
        self.assertNotIn("--native-cuda-cuda-runtime-lib", proc.stdout)
        self.assertIn("--no-checkpoint", proc.stdout)
        self.assertNotIn("--native-cuda-no-checkpoint", proc.stdout)
        self.assertIn("--backend", proc.stdout)
        self.assertIn("tile-cuda", proc.stdout)
        self.assertNotIn("--native-cuda-kernel-backend", proc.stdout)
        self.assertIn("--output-dir=/tmp/nfn-out", proc.stdout)
        self.assertNotIn("--native-cuda-output-dir", proc.stdout)
        self.assertIn("--lm-head-row-chunk-size", proc.stdout)
        self.assertIn("32768", proc.stdout)
        self.assertNotIn("--native-cuda-lm-head-row-chunk-size", proc.stdout)
        self.assertIn("--native-cuda-checkpoint-every", proc.stdout)
        self.assertIn("--native-cuda-sample-every=0", proc.stdout)
        self.assertIn("--native-cuda-generate-tokens", proc.stdout)
        self.assertIn("32", proc.stdout)
        self.assertIn("--native-cuda-activation", proc.stdout)
        self.assertIn("moa", proc.stdout)
        self.assertIn("--native-cuda-moa-interval=25", proc.stdout)
        self.assertIn("--template", proc.stdout)
        self.assertIn("gpt2-moa", proc.stdout)
        self.assertNotIn("--base-model", proc.stdout)

    def test_train_gpt2_evo_print_command_is_metadata_only_without_native_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            native_evo = Path(tmpdir) / "nfn_gpt2_evo_native_train"
            native_evo.write_text(
                "#!/usr/bin/env bash\n"
                "printf 'SHOULD_NOT_EXEC_NATIVE\\n'\n"
                "exit 23\n",
                encoding="utf-8",
            )
            native_evo.chmod(0o755)
            env = os.environ.copy()
            env["NFN_NATIVE_GPT2_EVO_CLI"] = str(native_evo)
            env.pop("PYTHONPATH", None)
            env.pop("NFN_ALLOW_TORCH_TRAINING", None)
            proc = subprocess.run(
                [
                    sys.executable,
                    str(NEURALFN_ROOT / "cli" / "scripts" / "train_gpt2_evo.py"),
                    "--tinystories",
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
        self.assertNotIn("SHOULD_NOT_EXEC_NATIVE", proc.stdout)
        self.assertIn(str(native_evo), proc.stdout)
        self.assertIn("--tinystories", proc.stdout)
        self.assertIn("--dry-run", proc.stdout)
        self.assertIn("--print-command", proc.stdout)
        self.assertNotIn("--native-cuda-dry-run", proc.stdout)
        self.assertNotIn("--native-cuda-print-command", proc.stdout)
        self.assertIn("--eval-every-steps 1000", proc.stdout)

    def test_train_nanogpt_direct_script_defaults_to_native_transformer_lm(self) -> None:
        code = textwrap.dedent(
            f"""
            from pathlib import Path
            import os
            import runpy
            import sys
            import tempfile

            root = Path({str(NEURALFN_ROOT)!r})
            script = root / "cli" / "scripts" / "train_nanogpt.py"
            native_gpt = Path(tempfile.mkdtemp()) / "nfn_gpt_native_train"
            native_gpt.write_text(
                "#!/usr/bin/env bash\\n"
                "printf 'NANOGPT_GPT_NATIVE_DIRECT\\\\n'\\n"
                "printf '%s\\\\n' \\"$@\\"\\n"
                "exit 23\\n",
                encoding="utf-8",
            )
            native_gpt.chmod(0o755)
            os.environ["NFN_NATIVE_GPT_CLI"] = str(native_gpt)
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
        self.assertIn("NANOGPT_GPT_NATIVE_DIRECT", proc.stdout)
        self.assertNotIn("--base-model", proc.stdout)
        self.assertIn("--template-name", proc.stdout)
        self.assertIn("nanogpt", proc.stdout)
        self.assertIn("--train-transformer-lm", proc.stdout)
        self.assertNotIn("--train-token-lm", proc.stdout)
        self.assertIn("--tinystories", proc.stdout)
        self.assertIn("--max-steps", proc.stdout)
        self.assertIn("--dry-run", proc.stdout)
        self.assertNotIn("--native-cuda-dry-run", proc.stdout)

    def test_nfn_nanogpt_train_defaults_to_native_transformer_lm(self) -> None:
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
                "nano_gpt",
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
        self.assertNotIn("--base-model", proc.stdout)
        self.assertIn("--model-family", proc.stdout)
        self.assertIn("nanogpt", proc.stdout)
        self.assertIn("--template-name", proc.stdout)
        self.assertIn("nanogpt", proc.stdout)
        self.assertIn("--train-transformer-lm", proc.stdout)
        self.assertNotIn("--train-token-lm", proc.stdout)
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
            env["NFN_NATIVE_GPT2_CLI"] = "/bin/echo"
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
        self.assertIn("--dataset-alias", proc.stdout)
        self.assertIn("--eval-every-steps 1000", proc.stdout)
        self.assertIn("--backend tile-cuda", proc.stdout)
        self.assertNotIn("train_gpt2cu", proc.stdout)

    def test_direct_gpt2_script_dispatches_to_native_without_pythonpath(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = Path(tmpdir) / "tiny"
            dataset.mkdir()
            (dataset / "data.txt").write_text("hello world. " * 128, encoding="utf-8")
            (dataset / "val.txt").write_text("validation story. " * 64, encoding="utf-8")
            (dataset / "meta.json").write_text(json.dumps({"data_format": "raw_text"}), encoding="utf-8")

            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            env["NFN_NATIVE_GPT2_CLI"] = "/bin/echo"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(NEURALFN_ROOT / "cli" / "scripts" / "train_gpt2.py"),
                    "--dataset-alias",
                    str(dataset),
                    "--no-download-if-missing",
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
        self.assertIn("--dataset-alias", proc.stdout)
        self.assertIn("--eval-every-steps 1000", proc.stdout)
        self.assertIn("--backend tile-cuda", proc.stdout)
        self.assertNotIn("train_gpt2cu", proc.stdout)

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
            print("TRAIN_GPT_NATIVE_LOADED", "train_gpt_native" in sys.modules)
            print("DATASET_MANAGER_LOADED", "server.dataset_manager" in sys.modules)
            print("NUMPY_LOADED", "numpy" in sys.modules)
            print("ARGV0", Path(sys.argv[0]).name)
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
        self.assertIn("TRAIN_GPT_NATIVE_LOADED False", proc.stdout)
        self.assertIn("DATASET_MANAGER_LOADED False", proc.stdout)
        self.assertIn("NUMPY_LOADED False", proc.stdout)
        self.assertIn("ARGV0 train_gpt.py", proc.stdout)

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

    def test_train_nanogpt_module_import_and_parser_do_not_import_torch(self) -> None:
        code = textwrap.dedent(
            f"""
            import importlib
            from pathlib import Path
            import sys

            root = Path({str(NEURALFN_ROOT)!r})
            sys.path.insert(0, str(root / "cli" / "scripts"))
            sys.path.insert(0, str(root))

            module = importlib.import_module("train_nanogpt")
            parser = module.build_parser()
            args = parser.parse_args(["--native-cuda-dry-run"])
            module.resolve_mode_defaults(args)
            print("RUNTIME", args.runtime)
            print("MODEL_FAMILY", args.model_family)
            print("TEMPLATE", args.template_name)
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
        self.assertIn("MODEL_FAMILY gpt", proc.stdout)
        self.assertIn("TEMPLATE nanogpt", proc.stdout)
        self.assertIn("TORCH_LOADED False", proc.stdout)
        self.assertIn("DATASET_MANAGER_LOADED False", proc.stdout)
        self.assertIn("NUMPY_LOADED False", proc.stdout)

    def test_train_deepseek_v4_module_import_and_parser_do_not_import_torch(self) -> None:
        code = textwrap.dedent(
            f"""
            import importlib
            from pathlib import Path
            import sys

            root = Path({str(NEURALFN_ROOT)!r})
            sys.path.insert(0, str(root / "cli" / "scripts"))
            sys.path.insert(0, str(root))

            module = importlib.import_module("train_deepseek_v4")
            parser = module.build_parser()
            args = parser.parse_args(["--native-cuda-dry-run"])
            module.resolve_mode_defaults(args)
            print("OUTPUT", args.output)
            print("MODE", module.mode_name())
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
        self.assertIn("OUTPUT", proc.stdout)
        self.assertIn("deepseek_v4.pt", proc.stdout)
        self.assertIn("MODE deepseek_v4", proc.stdout)
        self.assertIn("TORCH_LOADED False", proc.stdout)
        self.assertIn("DATASET_MANAGER_LOADED False", proc.stdout)
        self.assertIn("NUMPY_LOADED False", proc.stdout)

    def test_train_mixllama_module_import_and_parser_do_not_import_torch(self) -> None:
        code = textwrap.dedent(
            f"""
            import importlib
            from pathlib import Path
            import sys

            root = Path({str(NEURALFN_ROOT)!r})
            sys.path.insert(0, str(root / "cli" / "scripts"))
            sys.path.insert(0, str(root))

            module = importlib.import_module("train_mixllama_fast")
            parser = module.build_parser()
            args = parser.parse_args(["--native-cuda-dry-run"])
            module.resolve_mode_defaults(args)
            print("OUTPUT", args.output)
            print("MODE", module.mode_name())
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
        self.assertIn("OUTPUT", proc.stdout)
        self.assertIn("mixllama_fast.pt", proc.stdout)
        self.assertIn("MODE mixllama_fast", proc.stdout)
        self.assertIn("TORCH_LOADED False", proc.stdout)
        self.assertIn("DATASET_MANAGER_LOADED False", proc.stdout)
        self.assertIn("NUMPY_LOADED False", proc.stdout)

    def test_train_llama_fast_module_import_and_parser_do_not_import_torch(self) -> None:
        code = textwrap.dedent(
            f"""
            import importlib
            from pathlib import Path
            import sys

            root = Path({str(NEURALFN_ROOT)!r})
            sys.path.insert(0, str(root / "cli" / "scripts"))
            sys.path.insert(0, str(root))

            module = importlib.import_module("train_llama_fast")
            parser = module.build_parser()
            args = parser.parse_args(["--native-cuda-dry-run"])
            module.resolve_mode_defaults(args)
            print("OUTPUT", args.output)
            print("MODE", module.mode_name())
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
        self.assertIn("OUTPUT", proc.stdout)
        self.assertIn("llama_fast.pt", proc.stdout)
        self.assertIn("MODE llama_fast", proc.stdout)
        self.assertIn("TORCH_LOADED False", proc.stdout)
        self.assertIn("DATASET_MANAGER_LOADED False", proc.stdout)
        self.assertIn("NUMPY_LOADED False", proc.stdout)

    def test_train_llama_megakernel_module_import_and_parser_do_not_import_torch(self) -> None:
        code = textwrap.dedent(
            f"""
            import importlib
            from pathlib import Path
            import sys

            root = Path({str(NEURALFN_ROOT)!r})
            sys.path.insert(0, str(root / "cli" / "scripts"))
            sys.path.insert(0, str(root))

            module = importlib.import_module("train_llama_megakernel")
            parser = module.build_parser()
            args = parser.parse_args(["--native-cuda-dry-run"])
            module.resolve_mode_defaults(args)
            print("OUTPUT", args.output)
            print("MODE", module.mode_name())
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
        self.assertIn("OUTPUT", proc.stdout)
        self.assertIn("llama_megakernel.pt", proc.stdout)
        self.assertIn("MODE llama_megakernel", proc.stdout)
        self.assertIn("TORCH_LOADED False", proc.stdout)
        self.assertIn("DATASET_MANAGER_LOADED False", proc.stdout)
        self.assertIn("NUMPY_LOADED False", proc.stdout)

    def test_train_semantic_router_module_import_and_parser_do_not_import_torch(self) -> None:
        code = textwrap.dedent(
            f"""
            import importlib
            from pathlib import Path
            import sys

            root = Path({str(NEURALFN_ROOT)!r})
            sys.path.insert(0, str(root / "cli" / "scripts"))
            sys.path.insert(0, str(root))

            module = importlib.import_module("train_semantic_router_moe")
            parser = module.build_parser()
            args = parser.parse_args(["--native-cuda-dry-run"])
            module.resolve_mode_defaults(args)
            print("OUTPUT", args.output)
            print("MODE", module.mode_name())
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
        self.assertIn("OUTPUT", proc.stdout)
        self.assertIn("semantic_router_moe.pt", proc.stdout)
        self.assertIn("MODE semantic_router_moe", proc.stdout)
        self.assertIn("TORCH_LOADED False", proc.stdout)
        self.assertIn("DATASET_MANAGER_LOADED False", proc.stdout)
        self.assertIn("NUMPY_LOADED False", proc.stdout)

    def test_train_semantic_router_overnight_module_import_and_parser_do_not_import_torch(self) -> None:
        code = textwrap.dedent(
            f"""
            import importlib.util
            from pathlib import Path
            import sys

            root = Path({str(NEURALFN_ROOT)!r})
            sys.path.insert(0, str(root / "cli" / "scripts"))
            sys.path.insert(0, str(root))
            script = root / "cli" / "scripts" / "train_semantic_router_moe-overnight.py"

            spec = importlib.util.spec_from_file_location("train_semantic_router_moe_overnight", script)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            parser = module.build_parser()
            args = parser.parse_args(["--native-cuda-dry-run"])
            module.resolve_mode_defaults(args)
            print("OUTPUT", args.output)
            print("MODE", module.mode_name())
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
        self.assertIn("OUTPUT", proc.stdout)
        self.assertIn("semantic_router_moe_overnight.pt", proc.stdout)
        self.assertIn("MODE semantic_router_moe", proc.stdout)
        self.assertIn("TORCH_LOADED False", proc.stdout)
        self.assertIn("DATASET_MANAGER_LOADED False", proc.stdout)
        self.assertIn("NUMPY_LOADED False", proc.stdout)

    def test_train_jepa_module_import_and_parser_do_not_import_torch(self) -> None:
        code = textwrap.dedent(
            f"""
            import importlib
            from pathlib import Path
            import sys

            root = Path({str(NEURALFN_ROOT)!r})
            sys.path.insert(0, str(root / "cli" / "scripts"))
            sys.path.insert(0, str(root))

            module = importlib.import_module("train_jepa_semantic")
            parser = module.build_parser()
            args = parser.parse_args(["--tinystories"])
            print("DATASET_ALIAS", args.dataset_alias)
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
        self.assertIn("DATASET_ALIAS", proc.stdout)
        self.assertIn("TORCH_LOADED False", proc.stdout)

    def test_train_gpt_native_defaults_to_tinystories_not_parameter_golf(self) -> None:
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
            print("LM_HEAD_ROW_CHUNK_SIZE", args.lm_head_row_chunk_size)
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
        self.assertIn("LM_HEAD_ROW_CHUNK_SIZE 32768", proc.stdout)
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

    def test_sm120_gpt_helper_calls_native_cpp_directly(self) -> None:
        content = (NEURALFN_ROOT / "tools" / "train_gpt_sm120.sh").read_text(encoding="utf-8")

        self.assertIn("nfn_gpt_native_train", content)
        self.assertIn("TinyStories_train.bin", content)
        self.assertIn("TinyStories_val.bin", content)
        self.assertIn('CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"', content)
        self.assertIn("TRAIN_BATCH_TOKENS=\"$(env_or_alias3 NFN_NATIVE_GPT_TRAIN_BATCH_TOKENS", content)
        self.assertIn("NFN_SM120_TRAIN_BATCH_TOKENS 524288)", content)
        self.assertIn('--train-batch-tokens "${TRAIN_BATCH_TOKENS}"', content)
        self.assertIn("BATCH_SIZE=\"$(env_or_alias3 NFN_NATIVE_GPT_BATCH_SIZE", content)
        self.assertIn("NFN_SM120_BATCH_SIZE 64)", content)
        self.assertIn('--batch-size "${BATCH_SIZE}"', content)
        self.assertIn("TRAIN_SEQ_LEN=\"$(env_or_alias3 NFN_NATIVE_GPT_TRAIN_SEQ_LEN", content)
        self.assertIn("NFN_SM120_TRAIN_SEQ_LEN 1024)", content)
        self.assertIn('--train-seq-len "${TRAIN_SEQ_LEN}"', content)
        self.assertIn("LEARNING_RATE=\"$(env_or_alias3 NFN_NATIVE_GPT_LEARNING_RATE", content)
        self.assertIn("NFN_SM120_LEARNING_RATE 0.0006)", content)
        self.assertIn("WEIGHT_DECAY=\"$(env_or_alias3 NFN_NATIVE_GPT_WEIGHT_DECAY", content)
        self.assertIn("NFN_SM120_WEIGHT_DECAY 0.1)", content)
        self.assertIn("WARMUP_STEPS=\"$(env_or_alias3 NFN_NATIVE_GPT_WARMUP_STEPS", content)
        self.assertIn("NFN_SM120_WARMUP_STEPS 600)", content)
        self.assertIn('--learning-rate "${LEARNING_RATE}"', content)
        self.assertIn('--weight-decay "${WEIGHT_DECAY}"', content)
        self.assertIn('--warmup-steps "${WARMUP_STEPS}"', content)
        self.assertIn("MAX_STEPS=\"$(env_or_alias3 NFN_NATIVE_GPT_MAX_STEPS", content)
        self.assertIn("NFN_SM120_MAX_STEPS 20000)", content)
        self.assertIn('--max-steps "${MAX_STEPS}"', content)
        self.assertIn("EVAL_EVERY_STEPS=\"$(env_or_alias3 NFN_NATIVE_GPT_EVAL_EVERY_STEPS", content)
        self.assertIn("NFN_SM120_EVAL_EVERY_STEPS 1000)", content)
        self.assertIn('--eval-every-steps "${EVAL_EVERY_STEPS}"', content)
        self.assertIn("GENERATE_TOKENS=\"$(env_or_alias3 NFN_NATIVE_GPT_GENERATE_TOKENS", content)
        self.assertIn("NFN_SM120_GENERATE_TOKENS 144)", content)
        self.assertIn("CHECKPOINT_EVERY=\"$(env_or_alias3 NFN_NATIVE_GPT_CHECKPOINT_EVERY", content)
        self.assertIn("NFN_SM120_CHECKPOINT_EVERY 200)", content)
        self.assertIn('--native-cuda-generate-tokens "${GENERATE_TOKENS}"', content)
        self.assertIn('--native-cuda-checkpoint-every "${CHECKPOINT_EVERY}"', content)
        self.assertIn("--train-transformer-lm", content)
        self.assertNotIn("python", content)
        self.assertNotIn("train_gpt2cu", content)
        self.assertNotIn("--max-wallclock-seconds", content)
        self.assertNotIn("TorchTrainer", content)

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
                    str(NEURALFN_ROOT / "cli" / "scripts" / "train_gpt_native.py"),
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
        self.assertIn("Native CUDA shard resolution: compiled C++ frontend (deferred)", proc.stdout)
        self.assertIn("--dataset-alias alias", proc.stdout)
        self.assertNotIn("--target ", proc.stdout)
        self.assertNotIn("train_gpt2cu", proc.stdout)

    def test_train_gpt_native_metadata_action_defers_missing_dataset_to_cpp(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            native_cli = Path(tmpdir) / "nfn_gpt_native_train"
            native_cli.write_text(
                "#!/usr/bin/env bash\n"
                "printf 'NATIVE_CLI_ARGV'\n"
                "for arg in \"$@\"; do printf ' %q' \"$arg\"; done\n"
                "printf '\\n'\n",
                encoding="utf-8",
            )
            native_cli.chmod(0o755)
            datasets_dir = Path(tmpdir) / "datasets"

            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            env["NFN_DATASETS_DIR"] = str(datasets_dir)
            env["NFN_NATIVE_GPT_CLI"] = str(native_cli)
            env["NFN_NATIVE_GPT2_CLI"] = str(native_cli)
            env["NFN_NATIVE_GPT2_BINDING"] = "0"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(NEURALFN_ROOT / "cli" / "scripts" / "train_gpt_native.py"),
                    "--dataset-alias",
                    "missing_alias",
                    "--no-download-if-missing",
                    "--native-cuda-print-plan",
                ],
                cwd=NEURALFN_ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("Native CUDA shard resolution: compiled C++ frontend (deferred)", proc.stdout)
        self.assertIn("NATIVE_CLI_ARGV", proc.stdout)
        self.assertIn("--dataset-alias missing_alias", proc.stdout)
        self.assertIn("--print-plan", proc.stdout)
        self.assertNotIn("Dataset alias 'missing_alias' was not found", proc.stderr)

    def test_infer_gpt_parser_and_help_do_not_import_torch(self) -> None:
        cases = (
            "import",
            "parse-evo-defaults",
            "help",
        )
        for module_name in ("infer_gpt", "infer_gpt2"):
            script_name = f"{module_name}.py"
            for case in cases:
                with self.subTest(script=module_name, case=case):
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
                            sys.argv = [str(root / "cli" / "scripts" / {script_name!r}), "--evo", "--help"]
                            try:
                                runpy.run_path(str(root / "cli" / "scripts" / {script_name!r}), run_name="__main__")
                            except SystemExit as exc:
                                exit_code = int(exc.code or 0)
                            else:
                                exit_code = 0
                        else:
                            module = importlib.import_module({module_name!r})
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

    def test_infer_gpt2_native_checkpoint_sampler_uses_sdk_runner(self) -> None:
        source = (NEURALFN_ROOT / "cli" / "scripts" / "infer_gpt2.py").read_text(encoding="utf-8")
        function_body = source.split("def run_native_checkpoint_token_sampler", 1)[1].split(
            "\ndef render_native_checkpoint_sampler_text", 1
        )[0]

        self.assertIn("run_native_gpt_checkpoint_sampler", function_body)
        self.assertIn('runner="auto"', function_body)
        self.assertIn("temperature=", function_body)
        self.assertIn("top_k=", function_body)
        self.assertIn("repetition_penalty=", function_body)
        self.assertIn("seed=", function_body)
        self.assertNotIn("subprocess.run", function_body)
        self.assertNotIn("native_gpt_checkpoint_sampler_env", function_body)

    def test_infer_gpt_native_checkpoint_info_does_not_import_torch(self) -> None:
        for module_name in ("infer_gpt", "infer_gpt2"):
            script_name = f"{module_name}.py"
            with self.subTest(script=module_name):
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
                    from neuralfn.native_gpt import native_gpt_parameter_count

                    with tempfile.TemporaryDirectory() as tmpdir:
                        checkpoint = Path(tmpdir) / "model_00000010.bin"
                        header = [0] * 256
                        header[:8] = [20240326, 5, 8, 16, 1, 1, 4, 16]
                        nparams = native_gpt_parameter_count(
                            max_seq_len=8,
                            padded_vocab_size=16,
                            num_layers=1,
                            channels=4,
                        )
                        checkpoint.write_bytes(struct.pack("<" + "i" * 256, *header) + b"\\0" * (nparams * 2))
                        (Path(tmpdir) / "DONE_00000010").write_text("", encoding="utf-8")
                        sys.argv = [
                            str(root / "cli" / "scripts" / {script_name!r}),
                            "--native-checkpoint",
                            str(checkpoint),
                            "--native-info",
                        ]
                        try:
                            runpy.run_path(str(root / "cli" / "scripts" / {script_name!r}), run_name="__main__")
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
                self.assertIn("Native GPT checkpoint detected", proc.stdout)
                self.assertIn("checkpoint_step: 10", proc.stdout)
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
            from neuralfn.native_gpt import native_gpt_parameter_count

            with tempfile.TemporaryDirectory() as tmpdir:
                checkpoint = Path(tmpdir) / "model_00000010.bin"
                header = [0] * 256
                header[:8] = [20240326, 5, 8, 16, 1, 1, 4, 16]
                nparams = native_gpt_parameter_count(
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
        self.assertIn("Native GPT checkpoint detected", proc.stdout)
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
                    "--native-checkpoint",
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
        self.assertIn("Native GPT checkpoint detected", proc.stdout)
        self.assertIn("checkpoint_step: 10", proc.stdout)
        self.assertIn("TORCH_LOADED False", proc.stdout)
        self.assertIn("NFN_IMPL_LOADED False", proc.stdout)

    def test_nfn_infer_native_checkpoint_text_prompt_requires_explicit_tokenizer_opt_in(self) -> None:
        code = textwrap.dedent(
            f"""
            from pathlib import Path
            import os
            import runpy
            import struct
            import sys
            import tempfile

            root = Path({str(NEURALFN_ROOT)!r})
            sys.path.insert(0, str(root))
            from neuralfn.native_gpt2 import native_gpt2_parameter_count

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                checkpoint = tmp / "model_00000010.bin"
                native_cli = tmp / "nfn_gpt_native_train"
                native_cli.write_text(
                    "#!/usr/bin/env bash\\n"
                    "printf 'NATIVE_CLI_ARGV %s\\\\n' \\"$*\\"\\n"
                    "printf 'NATIVE_CLI_CUDA_VISIBLE_DEVICES %s\\\\n' \\"${{CUDA_VISIBLE_DEVICES:-}}\\"\\n"
                    "exit 2\\n",
                    encoding="utf-8",
                )
                native_cli.chmod(0o755)
                header = [0] * 256
                header[:8] = [20240326, 5, 8, 16, 1, 1, 4, 16]
                nparams = native_gpt2_parameter_count(
                    max_seq_len=8,
                    padded_vocab_size=16,
                    num_layers=1,
                    channels=4,
                )
                checkpoint.write_bytes(struct.pack("<" + "i" * 256, *header) + b"\\0" * (nparams * 2))
                os.environ["NFN_NATIVE_GPT_CLI"] = str(native_cli)
                sys.argv = [
                    str(root / "cli" / "nfn.py"),
                    "infer",
                    "--checkpoint",
                    str(checkpoint),
                    "--prompt",
                    "Once upon a time",
                    "--max-new-tokens",
                    "7",
                    "--temperature",
                    "0.5",
                    "--top-k",
                    "12",
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

        self.assertEqual(2, proc.returncode, proc.stderr)
        self.assertIn("Native GPT checkpoint detected", proc.stdout)
        self.assertNotIn("NATIVE_CLI_ARGV --sample-checkpoint", proc.stdout)
        self.assertIn("Native GPT .bin checkpoint prompt inference is token-id only by default", proc.stderr)
        self.assertIn("Pass --prompt-tokens", proc.stderr)
        self.assertIn("TORCH_LOADED False", proc.stdout)
        self.assertIn("NFN_IMPL_LOADED False", proc.stdout)

    def test_nfn_infer_native_checkpoint_prompt_tokens_dispatches_compiled_sampler(self) -> None:
        code = textwrap.dedent(
            f"""
            from pathlib import Path
            import os
            import runpy
            import struct
            import sys
            import tempfile

            root = Path({str(NEURALFN_ROOT)!r})
            sys.path.insert(0, str(root))
            from neuralfn.native_cuda_device import resolve_cuda_visible_devices_value
            from neuralfn.native_gpt2 import native_gpt2_parameter_count

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                checkpoint = tmp / "model_00000010.bin"
                native_cli = tmp / "nfn_gpt_native_train"
                native_cli.write_text(
                    "#!/usr/bin/env bash\\n"
                    "printf 'NATIVE_CLI_ARGV %s\\\\n' \\"$*\\"\\n"
                    "printf 'NATIVE_CLI_CUDA_VISIBLE_DEVICES %s\\\\n' \\"${{CUDA_VISIBLE_DEVICES:-}}\\"\\n"
                    "exit 2\\n",
                    encoding="utf-8",
                )
                native_cli.chmod(0o755)
                header = [0] * 256
                header[:8] = [20240326, 5, 8, 16, 1, 1, 4, 16]
                nparams = native_gpt2_parameter_count(
                    max_seq_len=8,
                    padded_vocab_size=16,
                    num_layers=1,
                    channels=4,
                )
                checkpoint.write_bytes(struct.pack("<" + "i" * 256, *header) + b"\\0" * (nparams * 2))
                os.environ["NFN_NATIVE_GPT_CLI"] = str(native_cli)
                sys.argv = [
                    str(root / "cli" / "nfn.py"),
                    "infer",
                    "--checkpoint",
                    str(checkpoint),
                    "--prompt-tokens",
                    "1,2,3",
                    "--max-new-tokens",
                    "7",
                ]
                try:
                    runpy.run_path(str(root / "cli" / "nfn.py"), run_name="__main__")
                except SystemExit as exc:
                    exit_code = int(exc.code or 0)
                else:
                    exit_code = 0
                print("TORCH_LOADED", "torch" in sys.modules)
                print("NFN_IMPL_LOADED", "nfn_impl" in sys.modules)
                print("EXPECTED_NATIVE_CUDA_VISIBLE_DEVICES", resolve_cuda_visible_devices_value("dedicated"))
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

        self.assertEqual(2, proc.returncode, proc.stderr)
        self.assertIn("Native GPT checkpoint detected", proc.stdout)
        self.assertIn("NATIVE_CLI_ARGV --sample-checkpoint", proc.stdout)
        self.assertIn("--prompt-tokens 1,2,3", proc.stdout)
        self.assertIn("--max-new-tokens 7", proc.stdout)
        expected_cuda = next(
            line.split(" ", 1)[1]
            for line in proc.stdout.splitlines()
            if line.startswith("EXPECTED_NATIVE_CUDA_VISIBLE_DEVICES ")
        )
        self.assertIn(f"NATIVE_CLI_CUDA_VISIBLE_DEVICES {expected_cuda}", proc.stdout)
        self.assertIn("TORCH_LOADED False", proc.stdout)
        self.assertIn("NFN_IMPL_LOADED False", proc.stdout)

    def test_nfn_infer_native_checkpoint_decodes_compiled_sampler_tokens(self) -> None:
        code = textwrap.dedent(
            f"""
            from pathlib import Path
            import os
            import runpy
            import struct
            import sys
            import tempfile

            root = Path({str(NEURALFN_ROOT)!r})
            sys.path.insert(0, str(root))
            from neuralfn.native_gpt2 import native_gpt2_parameter_count

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                checkpoint = tmp / "model_00000010.bin"
                native_cli = tmp / "nfn_gpt_native_train"
                native_cli.write_text(
                    "#!/usr/bin/env python3\\n"
                    "import json\\n"
                    "print(json.dumps({{'status': 'native-checkpoint-sampler', 'generated_tokens': [15496]}}))\\n",
                    encoding="utf-8",
                )
                native_cli.chmod(0o755)
                header = [0] * 256
                header[:8] = [20240326, 5, 8, 50257, 1, 1, 4, 50304]
                nparams = native_gpt2_parameter_count(
                    max_seq_len=8,
                    padded_vocab_size=50304,
                    num_layers=1,
                    channels=4,
                )
                checkpoint.write_bytes(struct.pack("<" + "i" * 256, *header) + b"\\0" * (nparams * 2))
                os.environ["NFN_NATIVE_GPT_CLI"] = str(native_cli)
                os.environ["NFN_NATIVE_GPT_ALLOW_PYTHON_TOKENIZER"] = "1"
                sys.argv = [
                    str(root / "cli" / "nfn.py"),
                    "infer",
                    "--checkpoint",
                    str(checkpoint),
                    "--prompt",
                    "Hello",
                    "--max-new-tokens",
                    "1",
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
        self.assertIn("Native GPT checkpoint detected", proc.stdout)
        self.assertIn('"generated_tokens": [15496]', proc.stdout)
        self.assertIn("Generated token ids: [15496]", proc.stdout)
        self.assertIn("Generated text:", proc.stdout)
        self.assertIn("Hello", proc.stdout)
        self.assertIn("TORCH_LOADED False", proc.stdout)
        self.assertIn("NFN_IMPL_LOADED False", proc.stdout)


if __name__ == "__main__":
    unittest.main()
