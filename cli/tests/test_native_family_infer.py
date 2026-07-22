from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NEURALFN_ROOT = ROOT.parent


def native_family_parameter_index(token: int, next_token: int, parameter_elements: int) -> int:
    key = (int(token) * 1315423911) ^ (int(next_token) * 2654435761)
    return key % int(parameter_elements)


def write_complete_native_family_checkpoint(
    root: Path,
    *,
    prefix: str,
    model_family: str,
    native_target: str,
    template_name: str,
    checkpoint_format: str = "nfn-native-family-token-transition-v1",
    architecture_forward: bool = False,
) -> Path:
    checkpoint = root / f"{prefix}_native_family_model_00000000.json"
    parameter_data = root / f"{prefix}_native_family_parameters_00000000.f32"
    parameter_elements = 18
    parameter_storage = (
        "deterministic_dense_float32_base_plus_sparse_sampled_updates"
        if architecture_forward
        else "full_size_sparse_float32_with_sampled_parameter_updates"
    )
    parameter_initialization = (
        "deterministic_dense_float32_v1_plus_sampled_native_updates"
        if architecture_forward
        else "zero_initialized_plus_sampled_native_updates"
    )
    updated_index = native_family_parameter_index(1, 2, parameter_elements)
    values = [0.0] * parameter_elements
    values[2] = 1.0
    values[updated_index] = 0.004
    if architecture_forward:
        values[8] = 1.0
    values[14] = 1.25
    if architecture_forward:
        for index, value in enumerate(values):
            if value == 0.0:
                values[index] = 0.001 * float(index + 1)
    trained_parameter_elements = parameter_elements if architecture_forward else 1
    parameter_data.write_bytes(struct.pack(f"<{parameter_elements}f", *values))
    checkpoint.write_text(
        json.dumps(
            {
                "format": checkpoint_format,
                "model_family": model_family,
                "native_target": native_target,
                "template_name": template_name,
                "dataset_alias": "tinystories",
                "checkpoint_kind": "native_family_token_transition_model",
                "inference_supported": True,
                "vocab_size": 16,
                "transition_count": 1,
                "steps_completed": 3,
                "train_batches_sampled": 3,
                "validation_batches_sampled": 1,
                "fallback_tokens": [9],
                "transitions": [{"token": 1, "next": 2}],
                "parameter_data": {
                    "format": "nfn-native-family-float32-parameter-state-v1",
                    "path": str(parameter_data),
                    "parameter_dtype": "float32",
                    "parameter_elements": parameter_elements,
                    "bytes": parameter_elements * 4,
                    "storage": parameter_storage,
                    "dense_parameter_state_reconstructable": architecture_forward,
                    "base_parameter_initialization": "deterministic_dense_float32_v1" if architecture_forward else "",
                    "base_parameter_seed": 1337 if architecture_forward else 0,
                    "base_parameter_scale": 0.02 if architecture_forward else 0.0,
                    "dense_base_probe_count": 1 if architecture_forward else 0,
                    "dense_base_probe_checksum": 111 if architecture_forward else 0,
                    "dense_base_probes": (
                        [{"buffer": "final_norm.weight", "index": 8, "value": 1.0}]
                        if architecture_forward
                        else []
                    ),
                    "trained_parameter_elements": trained_parameter_elements,
                    "parameter_update_checksum": 54321,
                },
                "architecture_parameter_layout": {
                    "layout_resolved": True,
                    "parameter_dtype": "float32",
                    "parameter_buffer_count": 3,
                    "parameter_elements": parameter_elements,
                    "contiguous_parameter_state": True,
                    "buffers": [
                        {
                            "name": "token_embedding.weight",
                            "offset": 0,
                            "byte_offset": 0,
                            "elements": 8,
                            "bytes": 32,
                            "rows": 4,
                            "trainable": True,
                        },
                        {
                            "name": "final_norm.weight",
                            "offset": 8,
                            "byte_offset": 32,
                            "elements": 2,
                            "bytes": 8,
                            "trainable": True,
                        },
                        {
                            "name": "lm_head.weight",
                            "offset": 10,
                            "byte_offset": 40,
                            "elements": 8,
                            "bytes": 32,
                            "rows": 4,
                            "trainable": True,
                        },
                    ],
                },
                "writer_verification": {
                    "status": "native-family-checkpoint-writer-verification",
                    "passed": True,
                    "parameter_sidecar_exists": True,
                    "parameter_sidecar_size_matches": True,
                    "sampled_update_probe_count": 3,
                    "dense_base_initialization_verified": architecture_forward,
                    "dense_base_initialization_probe_count": 1 if architecture_forward else 0,
                    "dense_base_probe_checksum": 111 if architecture_forward else 0,
                    "dense_base_probes": (
                        [{"buffer": "final_norm.weight", "index": 8, "value": 1.0}]
                        if architecture_forward
                        else []
                    ),
                    "error": "",
                },
                "native_parameter_state": {
                    "state_type": "sparse_float32_parameter_tensors_plus_token_transition_table",
                    "full_template_parameter_state": True,
                    "parameter_storage": parameter_storage,
                    "parameter_initialization": parameter_initialization,
                    "parameter_lm_head_inference_supported": True,
                    "working_model_inference_path": (
                        "native_family_architecture_sidecar_forward_v1"
                        if architecture_forward
                        else "token_embedding_lm_head_sidecar_forward"
                    ),
                    "parameter_buffer_count": 3,
                    "parameter_elements": parameter_elements,
                    "persisted_parameter_elements": parameter_elements,
                    "trained_parameter_elements": trained_parameter_elements,
                    "parameter_update_checksum": 54321,
                    "parameter_data_path": str(parameter_data),
                    "architecture_forward_inference_supported": architecture_forward,
                    "dense_parameter_state_reconstructable": architecture_forward,
                    "base_parameter_initialization": "deterministic_dense_float32_v1" if architecture_forward else "",
                    "base_parameter_seed": 1337 if architecture_forward else 0,
                    "base_parameter_scale": 0.02 if architecture_forward else 0.0,
                    "dense_base_probe_count": 1 if architecture_forward else 0,
                    "dense_base_probe_checksum": 111 if architecture_forward else 0,
                    "transition_sampler_inference_supported": True,
                },
            }
        ),
        encoding="utf-8",
    )
    checkpoint.with_name(f"{prefix}_native_family_model_DONE").write_text("done\n", encoding="utf-8")
    return checkpoint


class NativeFamilyInferTest(unittest.TestCase):
    def test_lightweight_cli_native_template_aliases_match_sdk_registry(self) -> None:
        sys.path.insert(0, str(NEURALFN_ROOT))
        from cli import nfn
        from neuralfn.native_train import NATIVE_TEMPLATE_FAMILY_ALIASES

        self.assertEqual(NATIVE_TEMPLATE_FAMILY_ALIASES, nfn._NATIVE_TEMPLATE_FAMILY_ALIASES)

    def test_native_family_loader_accepts_optimizer_checkpoint_format(self) -> None:
        sys.path.insert(0, str(NEURALFN_ROOT))
        from neuralfn.native_family import (
            is_native_family_checkpoint,
            read_native_family_checkpoint_info,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = write_complete_native_family_checkpoint(
                Path(tmpdir),
                prefix="dense_jepa_optimizer",
                model_family="jepa",
                native_target="nfn_jepa_native_train",
                template_name="dense-jepa-evo-modern",
                checkpoint_format="nfn-native-family-optimizer-checkpoint-v1",
            )

            self.assertTrue(is_native_family_checkpoint(checkpoint))
            info = read_native_family_checkpoint_info(checkpoint)
            self.assertEqual("dense-jepa-evo-modern", info.template_name)
            self.assertEqual("jepa", info.model_family)

    def test_nfn_infer_native_family_checkpoint_info_is_lightweight(self) -> None:
        code = textwrap.dedent(
            f"""
            import json
            from pathlib import Path
            import runpy
            import sys
            import tempfile

            root = Path({str(NEURALFN_ROOT)!r})
            sys.path.insert(0, str(root))

            with tempfile.TemporaryDirectory() as tmpdir:
                checkpoint = Path(tmpdir) / "dense_jepa_native_family_model_00000000.json"
                parameter_data = Path(tmpdir) / "dense_jepa_native_family_parameters_00000000.f32"
                parameter_data.write_bytes(b"\\0" * 32)
                checkpoint.write_text(json.dumps({{
                    "format": "nfn-native-family-token-transition-v1",
                    "model_family": "jepa",
                    "native_target": "nfn_jepa_native_train",
                    "template_name": "dense_jepa_evo_modern",
                    "dataset_alias": "tinystories",
                    "checkpoint_kind": "native_family_token_transition_model",
                    "inference_supported": True,
                    "vocab_size": 65536,
                    "transition_count": 1,
                    "steps_completed": 3,
                    "train_batches_sampled": 3,
                    "validation_batches_sampled": 1,
                    "fallback_tokens": [42],
                    "transitions": [{{"token": 1, "next": 2}}],
                    "parameter_data": {{
                        "format": "nfn-native-family-float32-parameter-state-v1",
                        "path": str(parameter_data),
                        "parameter_dtype": "float32",
                        "parameter_elements": 8,
                        "bytes": 32,
                        "storage": "full_size_sparse_float32_with_sampled_parameter_updates",
                        "trained_parameter_elements": 2,
                        "parameter_update_checksum": 12345,
                    }},
                    "architecture_parameter_layout": {{
                        "layout_resolved": True,
                        "parameter_dtype": "float32",
                        "parameter_buffer_count": 2,
                        "parameter_elements": 8,
                        "contiguous_parameter_state": True,
                        "buffers": [
                            {{"name": "token_embedding.weight", "offset": 0, "byte_offset": 0, "elements": 4, "bytes": 16, "trainable": True}},
                            {{"name": "lm_head.weight", "offset": 4, "byte_offset": 16, "elements": 4, "bytes": 16, "trainable": True}},
                        ],
                    }},
                    "writer_verification": {{
                        "status": "native-family-checkpoint-writer-verification",
                        "passed": True,
                        "parameter_sidecar_exists": True,
                        "parameter_sidecar_size_matches": True,
                        "sampled_update_probe_count": 3,
                        "error": "",
                    }},
                    "native_parameter_state": {{
                        "state_type": "sparse_float32_parameter_tensors_plus_token_transition_table",
                        "full_template_parameter_state": True,
                        "parameter_buffer_count": 2,
                        "parameter_elements": 8,
                        "persisted_parameter_elements": 8,
                        "trained_parameter_elements": 2,
                        "parameter_update_checksum": 12345,
                        "parameter_data_path": str(parameter_data),
                        "architecture_forward_inference_supported": False,
                        "parameter_lm_head_inference_supported": True,
                        "working_model_inference_path": "token_embedding_lm_head_sidecar_forward",
                        "transition_sampler_inference_supported": True,
                    }},
                }}), encoding="utf-8")
                checkpoint.with_name("dense_jepa_native_family_model_DONE").write_text("done\\n", encoding="utf-8")
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
        self.assertIn("Native family checkpoint detected", proc.stdout)
        self.assertIn("model_family: jepa", proc.stdout)
        self.assertIn("template_name: dense_jepa_evo_modern", proc.stdout)
        self.assertIn("transition_count: 1", proc.stdout)
        self.assertIn("parameter_state_type: sparse_float32_parameter_tensors_plus_token_transition_table", proc.stdout)
        self.assertIn("parameter_storage: full_size_sparse_float32_with_sampled_parameter_updates", proc.stdout)
        self.assertIn("parameter_initialization: ", proc.stdout)
        self.assertIn("dense_parameter_state_reconstructable: False", proc.stdout)
        self.assertIn("base_parameter_initialization: ", proc.stdout)
        self.assertIn("base_parameter_seed: 0", proc.stdout)
        self.assertIn("base_parameter_scale: 0.0", proc.stdout)
        self.assertIn("full_template_parameter_state: True", proc.stdout)
        self.assertIn("parameter_buffer_count: 2", proc.stdout)
        self.assertIn("parameter_elements: 8", proc.stdout)
        self.assertIn("persisted_parameter_elements: 8", proc.stdout)
        self.assertIn("trained_parameter_elements: 2", proc.stdout)
        self.assertIn("parameter_update_checksum: 12345", proc.stdout)
        self.assertIn("writer_verification_passed: True", proc.stdout)
        self.assertIn("writer_verification_update_probe_count: 3", proc.stdout)
        self.assertIn("writer_verification_error: ", proc.stdout)
        self.assertIn("architecture_forward_inference_supported: False", proc.stdout)
        self.assertIn("parameter_lm_head_inference_supported: True", proc.stdout)
        self.assertIn("working_model_inference_path: token_embedding_lm_head_sidecar_forward", proc.stdout)
        self.assertIn("transition_sampler_inference_supported: True", proc.stdout)
        self.assertIn("parameter_data_exists: True", proc.stdout)
        self.assertIn("parameter_data_bytes: 32", proc.stdout)
        self.assertIn("expected_parameter_data_bytes: 32", proc.stdout)
        self.assertIn("parameter_data_size_matches: True", proc.stdout)
        self.assertIn("TORCH_LOADED False", proc.stdout)
        self.assertIn("NFN_IMPL_LOADED False", proc.stdout)

    def test_nfn_infer_native_family_checkpoint_samples_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "dense_jepa_native_family_model_00000000.json"
            parameter_data = Path(tmpdir) / "dense_jepa_native_family_parameters_00000000.f32"
            parameter_data.write_bytes(b"\0" * 32)
            checkpoint.write_text(
                json.dumps(
                    {
                        "format": "nfn-native-family-token-transition-v1",
                        "model_family": "jepa",
                        "native_target": "nfn_jepa_native_train",
                        "template_name": "dense_jepa_evo_modern",
                        "dataset_alias": "tinystories",
                        "checkpoint_kind": "native_family_token_transition_model",
                        "inference_supported": True,
                        "vocab_size": 65536,
                        "transition_count": 2,
                        "steps_completed": 3,
                        "train_batches_sampled": 3,
                        "validation_batches_sampled": 1,
                        "fallback_tokens": [9],
                        "transitions": [{"token": 1, "next": 2}, {"token": 2, "next": 3}],
                        "parameter_data": {
                            "format": "nfn-native-family-float32-parameter-state-v1",
                            "path": str(parameter_data),
                            "parameter_dtype": "float32",
                            "parameter_elements": 8,
                            "bytes": 32,
                            "storage": "full_size_sparse_float32_with_sampled_parameter_updates",
                            "trained_parameter_elements": 2,
                            "parameter_update_checksum": 12345,
                        },
                        "architecture_parameter_layout": {
                            "layout_resolved": True,
                            "parameter_dtype": "float32",
                            "parameter_buffer_count": 2,
                            "parameter_elements": 8,
                            "contiguous_parameter_state": True,
                            "buffers": [
                                {"name": "token_embedding.weight", "offset": 0, "byte_offset": 0, "elements": 4, "bytes": 16, "trainable": True},
                                {"name": "lm_head.weight", "offset": 4, "byte_offset": 16, "elements": 4, "bytes": 16, "trainable": True},
                            ],
                        },
                        "writer_verification": {
                            "status": "native-family-checkpoint-writer-verification",
                            "passed": True,
                            "parameter_sidecar_exists": True,
                            "parameter_sidecar_size_matches": True,
                            "sampled_update_probe_count": 3,
                            "error": "",
                        },
                        "native_parameter_state": {
                            "state_type": "sparse_float32_parameter_tensors_plus_token_transition_table",
                            "full_template_parameter_state": True,
                            "parameter_buffer_count": 2,
                            "parameter_elements": 8,
                            "persisted_parameter_elements": 8,
                            "trained_parameter_elements": 2,
                            "parameter_update_checksum": 12345,
                            "parameter_data_path": str(parameter_data),
                            "architecture_forward_inference_supported": False,
                            "transition_sampler_inference_supported": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            proc = subprocess.run(
                [
                    sys.executable,
                    "cli/nfn.py",
                    "infer",
                    "--checkpoint",
                    str(checkpoint),
                    "--prompt-tokens",
                    "1",
                    "--max-new-tokens",
                    "3",
                ],
                cwd=NEURALFN_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("Native family checkpoint detected", proc.stdout)
        self.assertIn('"generated_tokens": [2, 3, 9]', proc.stdout)
        self.assertIn('"parameter_elements": 8', proc.stdout)
        self.assertIn('"persisted_parameter_elements": 8', proc.stdout)
        self.assertIn('"trained_parameter_elements": 2', proc.stdout)
        self.assertIn('"parameter_update_checksum": 12345', proc.stdout)
        self.assertIn('"bytes": 32', proc.stdout)
        self.assertIn('"parameter_data_probed": true', proc.stdout)
        self.assertIn('"parameter_probe_count": 3', proc.stdout)
        self.assertIn("Generated token ids: [2, 3, 9]", proc.stdout)

    def test_nfn_infer_native_family_checkpoint_uses_parameter_sidecar_bias(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "dense_jepa_native_family_model_00000000.json"
            parameter_data = Path(tmpdir) / "dense_jepa_native_family_parameters_00000000.f32"
            parameter_elements = 16
            updated_index = native_family_parameter_index(1, 2, parameter_elements)
            values = [0.0] * parameter_elements
            values[2] = 1.0
            values[updated_index] = 0.004
            values[12] = 1.25
            parameter_data.write_bytes(struct.pack(f"<{parameter_elements}f", *values))
            checkpoint.write_text(
                json.dumps(
                    {
                        "format": "nfn-native-family-token-transition-v1",
                        "model_family": "jepa",
                        "native_target": "nfn_jepa_native_train",
                        "template_name": "dense_jepa_evo_modern",
                        "dataset_alias": "tinystories",
                        "checkpoint_kind": "native_family_token_transition_model",
                        "inference_supported": True,
                        "vocab_size": 16,
                        "transition_count": 1,
                        "steps_completed": 3,
                        "train_batches_sampled": 3,
                        "validation_batches_sampled": 1,
                        "fallback_tokens": [9],
                        "transitions": [{"token": 1, "next": 2}],
                        "parameter_data": {
                            "format": "nfn-native-family-float32-parameter-state-v1",
                            "path": str(parameter_data),
                            "parameter_dtype": "float32",
                            "parameter_elements": parameter_elements,
                            "bytes": parameter_elements * 4,
                            "storage": "full_size_sparse_float32_with_sampled_parameter_updates",
                            "trained_parameter_elements": 1,
                            "parameter_update_checksum": 54321,
                        },
                        "architecture_parameter_layout": {
                            "layout_resolved": True,
                            "parameter_dtype": "float32",
                            "parameter_buffer_count": 2,
                            "parameter_elements": parameter_elements,
                            "contiguous_parameter_state": True,
                            "buffers": [
                                {"name": "token_embedding.weight", "offset": 0, "byte_offset": 0, "elements": 8, "bytes": 32, "rows": 4, "trainable": True},
                                {"name": "lm_head.weight", "offset": 8, "byte_offset": 32, "elements": 8, "bytes": 32, "rows": 4, "trainable": True},
                            ],
                        },
                        "writer_verification": {
                            "status": "native-family-checkpoint-writer-verification",
                            "passed": True,
                            "parameter_sidecar_exists": True,
                            "parameter_sidecar_size_matches": True,
                            "sampled_update_probe_count": 3,
                            "error": "",
                        },
                        "native_parameter_state": {
                            "state_type": "sparse_float32_parameter_tensors_plus_token_transition_table",
                            "full_template_parameter_state": True,
                            "parameter_lm_head_inference_supported": True,
                            "working_model_inference_path": "token_embedding_lm_head_sidecar_forward",
                            "parameter_buffer_count": 2,
                            "parameter_elements": parameter_elements,
                            "persisted_parameter_elements": parameter_elements,
                            "trained_parameter_elements": 1,
                            "parameter_update_checksum": 54321,
                            "parameter_data_path": str(parameter_data),
                            "architecture_forward_inference_supported": False,
                            "transition_sampler_inference_supported": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            proc = subprocess.run(
                [
                    sys.executable,
                    "cli/nfn.py",
                    "infer",
                    "--checkpoint",
                    str(checkpoint),
                    "--prompt-tokens",
                    "1",
                    "--max-new-tokens",
                    "1",
                ],
                cwd=NEURALFN_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn('"generated_tokens": [6]', proc.stdout)
        self.assertIn('"parameter_data_size_matches": true', proc.stdout)
        self.assertIn('"parameter_data_probed": true', proc.stdout)
        self.assertIn('"parameter_probe_count": 1', proc.stdout)
        self.assertIn('"base_next_token": 2', proc.stdout)
        self.assertIn(f'"parameter_index": {updated_index}', proc.stdout)
        self.assertIn('"source": "transition"', proc.stdout)
        self.assertIn('"token_offset": 4', proc.stdout)
        self.assertIn("Generated token ids: [6]", proc.stdout)

    def test_nfn_infer_native_family_checkpoint_verify_passes_complete_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "dense_jepa_native_family_model_00000000.json"
            parameter_data = Path(tmpdir) / "dense_jepa_native_family_parameters_00000000.f32"
            parameter_elements = 16
            updated_index = native_family_parameter_index(1, 2, parameter_elements)
            values = [0.0] * parameter_elements
            values[2] = 1.0
            values[updated_index] = 0.004
            values[12] = 1.25
            parameter_data.write_bytes(struct.pack(f"<{parameter_elements}f", *values))
            checkpoint.write_text(
                json.dumps(
                    {
                        "format": "nfn-native-family-token-transition-v1",
                        "model_family": "jepa",
                        "native_target": "nfn_jepa_native_train",
                        "template_name": "dense_jepa_evo_modern",
                        "dataset_alias": "tinystories",
                        "checkpoint_kind": "native_family_token_transition_model",
                        "inference_supported": True,
                        "vocab_size": 16,
                        "transition_count": 1,
                        "steps_completed": 3,
                        "train_batches_sampled": 3,
                        "validation_batches_sampled": 1,
                        "fallback_tokens": [9],
                        "transitions": [{"token": 1, "next": 2}],
                        "parameter_data": {
                            "format": "nfn-native-family-float32-parameter-state-v1",
                            "path": str(parameter_data),
                            "parameter_dtype": "float32",
                            "parameter_elements": parameter_elements,
                            "bytes": parameter_elements * 4,
                            "storage": "full_size_sparse_float32_with_sampled_parameter_updates",
                            "trained_parameter_elements": 1,
                            "parameter_update_checksum": 54321,
                        },
                        "architecture_parameter_layout": {
                            "layout_resolved": True,
                            "parameter_dtype": "float32",
                            "parameter_buffer_count": 2,
                            "parameter_elements": parameter_elements,
                            "contiguous_parameter_state": True,
                            "buffers": [
                                {"name": "token_embedding.weight", "offset": 0, "byte_offset": 0, "elements": 8, "bytes": 32, "rows": 4, "trainable": True},
                                {"name": "lm_head.weight", "offset": 8, "byte_offset": 32, "elements": 8, "bytes": 32, "rows": 4, "trainable": True},
                            ],
                        },
                        "writer_verification": {
                            "status": "native-family-checkpoint-writer-verification",
                            "passed": True,
                            "parameter_sidecar_exists": True,
                            "parameter_sidecar_size_matches": True,
                            "sampled_update_probe_count": 3,
                            "error": "",
                        },
                        "native_parameter_state": {
                            "state_type": "sparse_float32_parameter_tensors_plus_token_transition_table",
                            "full_template_parameter_state": True,
                            "parameter_lm_head_inference_supported": True,
                            "working_model_inference_path": "token_embedding_lm_head_sidecar_forward",
                            "parameter_buffer_count": 2,
                            "parameter_elements": parameter_elements,
                            "persisted_parameter_elements": parameter_elements,
                            "trained_parameter_elements": 1,
                            "parameter_update_checksum": 54321,
                            "parameter_data_path": str(parameter_data),
                            "architecture_forward_inference_supported": False,
                            "transition_sampler_inference_supported": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            checkpoint.with_name("dense_jepa_native_family_model_DONE").write_text("done\n", encoding="utf-8")
            proc = subprocess.run(
                [
                    sys.executable,
                    "cli/nfn.py",
                    "infer",
                    "--checkpoint",
                    str(checkpoint),
                    "--verify",
                    "--max-new-tokens",
                    "1",
                ],
                cwd=NEURALFN_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn('"status": "native-family-checkpoint-verification"', proc.stdout)
        self.assertIn('"passed": true', proc.stdout)
        self.assertIn('"errors": []', proc.stdout)
        self.assertIn('"full_template_parameter_state": true', proc.stdout)
        self.assertIn('"parameter_lm_head_inference_supported": true', proc.stdout)
        self.assertIn('"working_model_inference_path": "token_embedding_lm_head_sidecar_forward"', proc.stdout)
        self.assertIn('"writer_verification_passed": true', proc.stdout)
        self.assertIn('"writer_verification_update_probe_count": 3', proc.stdout)
        self.assertIn('"parameter_data_size_matches": true', proc.stdout)
        self.assertIn('"parameter_data_probed": true', proc.stdout)
        self.assertIn('"parameter_lm_head_inference_used": true', proc.stdout)
        self.assertIn('"generated_tokens": [6]', proc.stdout)

    def test_nfn_infer_native_family_checkpoint_verify_can_require_architecture_forward(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = write_complete_native_family_checkpoint(
                Path(tmpdir),
                prefix="dense_jepa",
                model_family="jepa",
                native_target="nfn_jepa_native_train",
                template_name="dense_jepa_evo_modern",
            )
            proc = subprocess.run(
                [
                    sys.executable,
                    "cli/nfn.py",
                    "infer",
                    "--checkpoint",
                    str(checkpoint),
                    "--verify",
                    "--require-architecture-forward",
                    "--max-new-tokens",
                    "1",
                ],
                cwd=NEURALFN_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertEqual(2, proc.returncode)
        self.assertIn('"passed": false', proc.stdout)
        self.assertIn('"architecture_forward_required": true', proc.stdout)
        self.assertIn('"architecture_forward_inference_supported": false', proc.stdout)
        self.assertIn("architecture-forward inference from persistent parameter state is not supported", proc.stdout)
        self.assertIn("bounded sample did not use architecture-forward inference", proc.stdout)

    def test_nfn_infer_native_family_checkpoint_verify_passes_architecture_forward(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = write_complete_native_family_checkpoint(
                Path(tmpdir),
                prefix="dense_jepa",
                model_family="jepa",
                native_target="nfn_jepa_native_train",
                template_name="dense_jepa_evo_modern",
                architecture_forward=True,
            )
            proc = subprocess.run(
                [
                    sys.executable,
                    "cli/nfn.py",
                    "infer",
                    "--checkpoint",
                    str(checkpoint),
                    "--verify",
                    "--require-architecture-forward",
                    "--max-new-tokens",
                    "1",
                ],
                cwd=NEURALFN_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn('"passed": true', proc.stdout)
        self.assertIn('"architecture_forward_required": true', proc.stdout)
        self.assertIn('"architecture_forward_inference_supported": true', proc.stdout)
        self.assertIn('"architecture_forward_inference_used": true', proc.stdout)
        self.assertIn('"architecture_forward_path": "native_family_architecture_sidecar_forward_v1"', proc.stdout)
        self.assertIn('"working_model_inference_path": "native_family_architecture_sidecar_forward_v1"', proc.stdout)
        self.assertIn('"dense_parameter_state_reconstructable": true', proc.stdout)
        self.assertIn('"dense_base_initialization": "deterministic_dense_float32_v1"', proc.stdout)
        self.assertIn('"base_parameter_initialization": "deterministic_dense_float32_v1"', proc.stdout)
        self.assertIn('"writer_dense_base_initialization_verified": true', proc.stdout)
        self.assertIn('"writer_dense_base_probe_count": 1', proc.stdout)
        self.assertIn('"generated_tokens": [2]', proc.stdout)

    def test_nfn_infer_native_family_checkpoint_verify_rejects_partial_architecture_forward_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = write_complete_native_family_checkpoint(
                Path(tmpdir),
                prefix="dense_jepa",
                model_family="jepa",
                native_target="nfn_jepa_native_train",
                template_name="dense_jepa_evo_modern",
                architecture_forward=True,
            )
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            payload["parameter_data"]["trained_parameter_elements"] = 1
            payload["native_parameter_state"]["trained_parameter_elements"] = 1
            checkpoint.write_text(json.dumps(payload), encoding="utf-8")
            proc = subprocess.run(
                [
                    sys.executable,
                    "cli/nfn.py",
                    "infer",
                    "--checkpoint",
                    str(checkpoint),
                    "--verify",
                    "--require-architecture-forward",
                    "--max-new-tokens",
                    "1",
                ],
                cwd=NEURALFN_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertEqual(2, proc.returncode)
        self.assertIn('"passed": false', proc.stdout)
        self.assertIn(
            "architecture-forward checkpoints must train every architecture parameter element",
            proc.stdout,
        )

    def test_nfn_infer_native_family_checkpoint_verify_all_checks_directory_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_complete_native_family_checkpoint(
                root,
                prefix="dense_jepa",
                model_family="jepa",
                native_target="nfn_jepa_native_train",
                template_name="dense_jepa_evo_modern",
            )
            write_complete_native_family_checkpoint(
                root,
                prefix="llama",
                model_family="llama",
                native_target="nfn_llama_native_train",
                template_name="llama_modern",
            )
            proc = subprocess.run(
                [
                    sys.executable,
                    "cli/nfn.py",
                    "infer",
                    "--checkpoint",
                    str(root),
                    "--verify-all",
                    "--max-new-tokens",
                    "1",
                ],
                cwd=NEURALFN_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn('"status": "native-family-checkpoint-verification-set"', proc.stdout)
        self.assertIn('"checkpoint_count": 2', proc.stdout)
        self.assertIn('"passed": true', proc.stdout)
        self.assertIn('"passed_count": 2', proc.stdout)
        self.assertIn('"failed_count": 0', proc.stdout)
        self.assertIn('"template_name": "dense_jepa_evo_modern"', proc.stdout)
        self.assertIn('"template_name": "llama_modern"', proc.stdout)
        self.assertIn('"writer_verification_passed": true', proc.stdout)
        self.assertIn('"parameter_lm_head_inference_used": true', proc.stdout)

    def test_nfn_infer_native_family_checkpoint_verify_all_checks_required_templates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_complete_native_family_checkpoint(
                root,
                prefix="dense_jepa",
                model_family="jepa",
                native_target="nfn_jepa_native_train",
                template_name="dense_jepa_evo_modern",
            )
            write_complete_native_family_checkpoint(
                root,
                prefix="llama",
                model_family="llama",
                native_target="nfn_llama_native_train",
                template_name="llama_modern",
            )
            proc = subprocess.run(
                [
                    sys.executable,
                    "cli/nfn.py",
                    "infer",
                    "--checkpoint",
                    str(root),
                    "--verify-all",
                    "--required-templates",
                    "dense-jepa-evo-modern,llama-modern",
                    "--max-new-tokens",
                    "1",
                ],
                cwd=NEURALFN_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn('"covered_template_verification"', proc.stdout)
        self.assertIn('"status": "native-family-covered-template-checkpoint-coverage"', proc.stdout)
        self.assertIn('"required_template_count": 2', proc.stdout)
        self.assertIn('"passed_template_count": 2', proc.stdout)
        self.assertIn('"missing_template_count": 0', proc.stdout)
        self.assertIn('"failed_template_count": 0', proc.stdout)
        self.assertIn('"missing_templates": []', proc.stdout)
        self.assertIn('"template_name": "dense-jepa-evo-modern"', proc.stdout)
        self.assertIn('"template_name": "llama-modern"', proc.stdout)

    def test_nfn_infer_native_family_checkpoint_verify_all_fails_missing_required_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_complete_native_family_checkpoint(
                root,
                prefix="dense_jepa",
                model_family="jepa",
                native_target="nfn_jepa_native_train",
                template_name="dense_jepa_evo_modern",
            )
            proc = subprocess.run(
                [
                    sys.executable,
                    "cli/nfn.py",
                    "infer",
                    "--checkpoint",
                    str(root),
                    "--verify-all",
                    "--required-templates",
                    "dense-jepa-evo-modern,llama-modern",
                    "--max-new-tokens",
                    "1",
                ],
                cwd=NEURALFN_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertEqual(2, proc.returncode)
        self.assertIn('"covered_template_verification"', proc.stdout)
        self.assertIn('"checkpoint_count": 1', proc.stdout)
        self.assertIn('"passed_count": 1', proc.stdout)
        self.assertIn('"passed": false', proc.stdout)
        self.assertIn('"required_template_count": 2', proc.stdout)
        self.assertIn('"passed_template_count": 1', proc.stdout)
        self.assertIn('"missing_template_count": 1', proc.stdout)
        self.assertIn('"missing_templates": ["llama-modern"]', proc.stdout)
        self.assertIn("missing native-family checkpoint for covered template", proc.stdout)

    def test_native_family_template_checkpoint_sweep_dry_run_uses_template_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bin_dir = root / "bin"
            out_dir = root / "out"
            bin_dir.mkdir()
            (bin_dir / "nfn_jepa_native_train").write_text("", encoding="utf-8")
            (bin_dir / "nfn_llama_native_train").write_text("", encoding="utf-8")
            proc = subprocess.run(
                [
                    sys.executable,
                    "tools/smoke_native_family_template_checkpoints.py",
                    "--native-bin-dir",
                    str(bin_dir),
                    "--output-dir",
                    str(out_dir),
                    "--templates",
                    "dense-jepa-evo-modern,llama-modern",
                    "--dry-run",
                    "--json",
                ],
                cwd=NEURALFN_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertEqual(0, proc.returncode, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual("native-family-template-checkpoint-smoke-sweep", payload["status"])
        self.assertTrue(payload["passed"])
        self.assertEqual(2, payload["template_count"])
        self.assertEqual(2, payload["smoke_count"])
        self.assertEqual(0, payload["missing_binary_count"])
        by_template = {row["template_name"]: row for row in payload["smokes"]}
        self.assertEqual(str(bin_dir / "nfn_jepa_native_train"), by_template["dense-jepa-evo-modern"]["binary"])
        self.assertEqual(str(bin_dir / "nfn_llama_native_train"), by_template["llama-modern"]["binary"])
        self.assertIn("--template-name", by_template["dense-jepa-evo-modern"]["argv"])
        self.assertIn("dense-jepa-evo-modern", by_template["dense-jepa-evo-modern"]["argv"])
        self.assertNotIn("stdout", by_template["dense-jepa-evo-modern"])
        self.assertNotIn("stderr", by_template["dense-jepa-evo-modern"])
        self.assertEqual({}, payload["verification"])

    def test_native_dense_gpt_template_checkpoint_sweep_dry_run_uses_dense_binary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            native_bin = root / "nfn_gpt_native_train"
            out_dir = root / "out"
            native_bin.write_text("", encoding="utf-8")
            proc = subprocess.run(
                [
                    sys.executable,
                    "tools/smoke_native_gpt_template_checkpoints.py",
                    "--native-bin",
                    str(native_bin),
                    "--output-dir",
                    str(out_dir),
                    "--templates",
                    "gpt,nanogpt,nanogpt-megakernel",
                    "--dry-run",
                    "--json",
                ],
                cwd=NEURALFN_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertEqual(0, proc.returncode, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual("native-dense-gpt-template-checkpoint-smoke-sweep", payload["status"])
        self.assertTrue(payload["passed"])
        self.assertEqual(3, payload["template_count"])
        self.assertEqual(3, payload["smoke_count"])
        self.assertFalse(payload["missing_binary"])
        by_template = {row["template_name"]: row for row in payload["smokes"]}
        self.assertEqual(str(native_bin), by_template["gpt"]["binary"])
        self.assertEqual(str(out_dir / "gpt"), by_template["gpt"]["output_dir"])
        self.assertEqual(str(out_dir / "nanogpt"), by_template["nanogpt"]["output_dir"])
        self.assertEqual(
            str(out_dir / "nanogpt_megakernel"),
            by_template["nanogpt_megakernel"]["output_dir"],
        )
        gpt_argv = by_template["gpt"]["smoke_argv"]
        self.assertIn("--checkpoint-metadata-smoke", gpt_argv)
        self.assertEqual("gpt", gpt_argv[gpt_argv.index("--template-name") + 1])
        self.assertIn("--native-info", by_template["gpt"]["info_argv"])
        self.assertIn("--native-checkpoint", by_template["gpt"]["info_argv"])

    def test_nfn_infer_native_family_checkpoint_verify_all_reports_empty_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            proc = subprocess.run(
                [
                    sys.executable,
                    "cli/nfn.py",
                    "infer",
                    "--checkpoint",
                    tmpdir,
                    "--verify-all",
                ],
                cwd=NEURALFN_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertEqual(2, proc.returncode)
        self.assertIn('"status": "native-family-checkpoint-verification-set"', proc.stdout)
        self.assertIn('"checkpoint_count": 0', proc.stdout)
        self.assertIn('"passed": false', proc.stdout)

    def test_nfn_infer_native_family_checkpoint_verify_fails_incomplete_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "dense_jepa_native_family_model_00000000.json"
            parameter_data = Path(tmpdir) / "missing_parameters.f32"
            checkpoint.write_text(
                json.dumps(
                    {
                        "format": "nfn-native-family-token-transition-v1",
                        "model_family": "jepa",
                        "native_target": "nfn_jepa_native_train",
                        "template_name": "dense_jepa_evo_modern",
                        "dataset_alias": "tinystories",
                        "checkpoint_kind": "native_family_token_transition_model",
                        "inference_supported": True,
                        "vocab_size": 16,
                        "transition_count": 1,
                        "steps_completed": 3,
                        "train_batches_sampled": 3,
                        "validation_batches_sampled": 1,
                        "fallback_tokens": [9],
                        "transitions": [{"token": 1, "next": 2}],
                        "parameter_data": {
                            "format": "nfn-native-family-float32-parameter-state-v1",
                            "path": str(parameter_data),
                            "parameter_dtype": "float32",
                            "parameter_elements": 16,
                            "bytes": 64,
                            "storage": "full_size_sparse_float32_with_sampled_parameter_updates",
                            "trained_parameter_elements": 1,
                            "parameter_update_checksum": 54321,
                        },
                        "architecture_parameter_layout": {
                            "layout_resolved": True,
                            "parameter_dtype": "float32",
                            "parameter_buffer_count": 1,
                            "parameter_elements": 16,
                            "contiguous_parameter_state": True,
                            "buffers": [
                                {"name": "token_embedding.weight", "offset": 0, "byte_offset": 0, "elements": 16, "bytes": 64, "trainable": True},
                            ],
                        },
                        "native_parameter_state": {
                            "state_type": "sparse_float32_parameter_tensors_plus_token_transition_table",
                            "full_template_parameter_state": True,
                            "parameter_buffer_count": 1,
                            "parameter_elements": 16,
                            "persisted_parameter_elements": 16,
                            "trained_parameter_elements": 1,
                            "parameter_update_checksum": 54321,
                            "parameter_data_path": str(parameter_data),
                            "architecture_forward_inference_supported": False,
                            "transition_sampler_inference_supported": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            proc = subprocess.run(
                [
                    sys.executable,
                    "cli/nfn.py",
                    "infer",
                    "--checkpoint",
                    str(checkpoint),
                    "--verify",
                ],
                cwd=NEURALFN_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertEqual(2, proc.returncode)
        self.assertIn('"passed": false', proc.stdout)
        self.assertIn("missing native-family model DONE marker", proc.stdout)
        self.assertIn("parameter_data sidecar does not exist", proc.stdout)
        self.assertIn("parameter_data sidecar size does not match checkpoint metadata", proc.stdout)


if __name__ == "__main__":
    unittest.main()
