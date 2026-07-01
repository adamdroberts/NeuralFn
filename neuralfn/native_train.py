from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
from typing import Any, Sequence

from .native_cuda_device import resolve_cuda_visible_devices_value


DEFAULT_NATIVE_TRAIN_CLI = "build/nfn_native_train"
DEFAULT_NATIVE_GPT_TRAIN_CLI = "build/nfn_gpt_native_train"
DEFAULT_NATIVE_GPT_TRAIN_CLI_LINKED = "build/nfn_gpt_native_train_linked"
DEFAULT_NATIVE_GPT_LAUNCHER_CLI = "build/nfn_train_gpt"
DEFAULT_NATIVE_SM120_TRAIN_CLI = "build/nfn_train_gpt_sm120"
NATIVE_GPT_LAUNCHER_COMMANDS = ("nfn-train-gpt", "nfn-gpt-train", "nfn_train_gpt")
NATIVE_SM120_TRAIN_COMMANDS = ("nfn-train-gpt-sm120", "nfn-gpt-sm120-train", "nfn_train_gpt_sm120")
DENSE_GPT_MODEL_FAMILIES = frozenset({"gpt", "gpt2", "gpt3", "nanogpt"})
NATIVE_GPT_QUALITY_DEFAULTS = {
    "--eval-every-steps": (
        "NFN_NATIVE_GPT_EVAL_EVERY_STEPS",
        "NFN_SM120_NATIVE_EVAL_EVERY_STEPS",
        "NFN_SM120_EVAL_EVERY_STEPS",
        "250",
    ),
    "--eval-batches": (
        "NFN_NATIVE_GPT_EVAL_BATCHES",
        "NFN_SM120_NATIVE_EVAL_BATCHES",
        "NFN_SM120_EVAL_BATCHES",
        "20",
    ),
    "--native-cuda-sample-every": (
        "NFN_NATIVE_GPT_SAMPLE_EVERY",
        "NFN_SM120_NATIVE_SAMPLE_EVERY",
        "NFN_SM120_SAMPLE_EVERY",
        "20000",
    ),
    "--native-cuda-generate-tokens": (
        "NFN_NATIVE_GPT_GENERATE_TOKENS",
        "NFN_SM120_NATIVE_GENERATE_TOKENS",
        "NFN_SM120_GENERATE_TOKENS",
        "144",
    ),
    "--native-cuda-checkpoint-every": (
        "NFN_NATIVE_GPT_CHECKPOINT_EVERY",
        "NFN_SM120_NATIVE_CHECKPOINT_EVERY",
        "NFN_SM120_CHECKPOINT_EVERY",
        "200",
    ),
    "--batch-size": (
        "NFN_NATIVE_GPT_BATCH_SIZE",
        "NFN_SM120_NATIVE_BATCH_SIZE",
        "NFN_SM120_BATCH_SIZE",
        "64",
    ),
    "--train-seq-len": (
        "NFN_NATIVE_GPT_TRAIN_SEQ_LEN",
        "NFN_SM120_NATIVE_TRAIN_SEQ_LEN",
        "NFN_SM120_TRAIN_SEQ_LEN",
        "1024",
    ),
    "--train-batch-tokens": (
        "NFN_NATIVE_GPT_TRAIN_BATCH_TOKENS",
        "NFN_SM120_NATIVE_TRAIN_BATCH_TOKENS",
        "NFN_SM120_TRAIN_BATCH_TOKENS",
        "524288",
    ),
    "--learning-rate": (
        "NFN_NATIVE_GPT_LEARNING_RATE",
        "NFN_SM120_NATIVE_LEARNING_RATE",
        "NFN_SM120_LEARNING_RATE",
        "0.0006",
    ),
    "--final-lr-fraction": (
        "NFN_NATIVE_GPT_FINAL_LR_FRACTION",
        "NFN_SM120_NATIVE_FINAL_LR_FRACTION",
        "NFN_SM120_FINAL_LR_FRACTION",
        "0.0",
    ),
    "--weight-decay": (
        "NFN_NATIVE_GPT_WEIGHT_DECAY",
        "NFN_SM120_NATIVE_WEIGHT_DECAY",
        "NFN_SM120_WEIGHT_DECAY",
        "0.1",
    ),
    "--beta1": ("NFN_NATIVE_GPT_BETA1", "NFN_SM120_NATIVE_BETA1", "NFN_SM120_BETA1", "0.9"),
    "--beta2": ("NFN_NATIVE_GPT_BETA2", "NFN_SM120_NATIVE_BETA2", "NFN_SM120_BETA2", "0.95"),
    "--adam-eps": (
        "NFN_NATIVE_GPT_ADAM_EPS",
        "NFN_SM120_NATIVE_ADAM_EPS",
        "NFN_SM120_ADAM_EPS",
        "1e-8",
    ),
    "--grad-clip-norm": (
        "NFN_NATIVE_GPT_GRAD_CLIP_NORM",
        "NFN_SM120_NATIVE_GRAD_CLIP_NORM",
        "NFN_SM120_GRAD_CLIP_NORM",
        "1.0",
    ),
    "--warmup-steps": (
        "NFN_NATIVE_GPT_WARMUP_STEPS",
        "NFN_SM120_NATIVE_WARMUP_STEPS",
        "NFN_SM120_WARMUP_STEPS",
        "60",
    ),
    "--max-steps": (
        "NFN_NATIVE_GPT_MAX_STEPS",
        "NFN_SM120_NATIVE_MAX_STEPS",
        "NFN_SM120_MAX_STEPS",
        "20000",
    ),
}
NATIVE_GPT_METADATA_ACTION_FLAGS = frozenset(
    {
        "--print-plan",
        "--list-templates",
        "--check-tile-ops",
        "--startup-only",
        "--smoke-tile-ops",
        "--smoke-nvfp4-pack",
        "--smoke-optimizer-step",
        "--smoke-lm-step",
        "--smoke-attention-step",
        "--smoke-mlp-step",
        "--smoke-norm-residual-step",
        "--smoke-transformer-block-step",
        "--smoke-transformer-lm-step",
        "--smoke-embedding-lm-step",
    }
)
NATIVE_TRAIN_ACTION_FLAGS = NATIVE_GPT_METADATA_ACTION_FLAGS | frozenset(
    {
        "--sample-token-batch",
        "--smoke-embedding-norm-step",
        "--smoke-fused-qkv-attention-step",
        "--smoke-qkv-layout-step",
        "--smoke-token-train-step",
        "--smoke-training-loop-step",
        "--train-embedding-lm",
        "--train-token-lm",
        "--train-transformer-lm",
    }
)
NATIVE_TRAIN_FAMILY_TARGETS = {
    "gpt": "nfn_gpt_native_train",
    "gpt2": "nfn_gpt_native_train",
    "gpt3": "nfn_gpt_native_train",
    "nanogpt": "nfn_gpt_native_train",
    "gpt2-evo": "nfn_gpt2_evo_native_train",
    "llama": "nfn_llama_native_train",
    "mixllama": "nfn_mixllama_native_train",
    "jepa": "nfn_jepa_native_train",
    "semantic-dense-jepa": "nfn_semantic_dense_jepa_native_train",
    "moe-jepa-evo": "nfn_moe_jepa_evo_native_train",
    "auxfree-moe-jepa-evo": "nfn_moe_jepa_evo_native_train",
    "moe-jepa-evo-modern": "nfn_moe_jepa_evo_native_train",
    "semantic-router-moe": "nfn_semantic_router_moe_native_train",
    "deepseek-v4": "nfn_deepseek_v4_native_train",
    "jamba": "nfn_jamba_native_train",
    "seq2seq": "nfn_seq2seq_native_train",
    "diffusion": "nfn_diffusion_native_train",
    "ttt-llama": "nfn_ttt_llama_native_train",
    "hnet-lm": "nfn_hnet_lm_native_train",
    "universal-llama": "nfn_universal_llama_native_train",
}
NATIVE_TEMPLATE_FAMILY_ALIASES = {
    "llama": "llama",
    "llama-fast": "llama",
    "llama-fast-megakernel": "llama",
    "llama-megakernel": "llama",
    "llama-modern": "llama",
    "modern-norms-llama": "llama",
    "ternary-b158": "llama",
    "ternary-b158-modern": "llama",
    "fp8-llama": "llama",
    "mxfp4-llama": "llama",
    "gemma3": "llama",
    "diff-transformer": "llama",
    "longctx-sparse-llama": "llama",
    "qwen3-longctx": "llama",
    "kv-pca-llama": "llama",
    "kv-pca-llama-modern": "llama",
    "mixllama": "mixllama",
    "mixllama-fast": "mixllama",
    "mixllama-fast-megakernel": "mixllama",
    "moe": "mixllama",
    "moe-modern": "mixllama",
    "deepseek-v3": "mixllama",
    "deepseek-v4": "deepseek-v4",
    "llm-jepa": "jepa",
    "llm-jepa-modern": "jepa",
    "dense-jepa-evo": "jepa",
    "dense-jepa-evo-modern": "jepa",
    "semantic-dense-jepa-evo": "semantic-dense-jepa",
    "semantic-dense-jepa-evo-modern": "semantic-dense-jepa",
    "dyt-geglu-semantic-dense-jepa-evo": "semantic-dense-jepa",
    "jepa-semantic-hybrid": "semantic-dense-jepa",
    "jepa-semantic-hybrid-modern": "semantic-dense-jepa",
    "jepa-semantic-hybrid-megakernel": "semantic-dense-jepa",
    "moe-jepa-evo": "moe-jepa-evo",
    "moe-jepa-evo-modern": "moe-jepa-evo",
    "auxfree-moe-jepa-evo": "moe-jepa-evo",
    "semantic-router-moe": "semantic-router-moe",
    "semantic-router-moe-modern": "semantic-router-moe",
    "semantic-router-moe-megakernel": "semantic-router-moe",
    "semantic-moe-jepa-evo": "semantic-router-moe",
    "semantic-moe-jepa-evo-modern": "semantic-router-moe",
    "diff-semantic-moe-jepa-evo": "semantic-router-moe",
    "jamba": "jamba",
    "jamba-modern": "jamba",
    "seq2seq": "seq2seq",
    "seq2seq-modern": "seq2seq",
    "diffusion": "diffusion",
    "diffusion-modern": "diffusion",
    "ttt-llama": "ttt-llama",
    "ttt-llama-modern": "ttt-llama",
    "hnet-lm": "hnet-lm",
    "hnet-lm-modern": "hnet-lm",
    "universal-llama": "universal-llama",
    "universal-llama-modern": "universal-llama",
}
NATIVE_TRAIN_BINDING_MODULES = ("neuralfn_native_train", "neuralfn._native_train")
_NATIVE_TRAIN_MODEL_REGISTRY = (
    {
        "name": "gpt",
        "status": "implemented",
        "native_target": "nfn_gpt_native_train",
        "transformer_lm_status": "native-transformer-lm",
        "token_lm_status": "not-applicable",
        "geometry_status": "dense-gpt-template-geometry",
        "kernel_status": "required-tile-symbols-present",
        "trainer_loop_status": "implemented",
        "notes": (
            "Dense GPT aliases to the NeuralFn Tile-CUDA transformer-LM loop; "
            "template/custom graph selection decides the GPT architecture."
        ),
    },
    {
        "name": "gpt2",
        "status": "implemented",
        "native_target": "nfn_gpt_native_train",
        "transformer_lm_status": "native-transformer-lm",
        "token_lm_status": "not-applicable",
        "geometry_status": "dense-gpt-template-geometry",
        "kernel_status": "required-tile-symbols-present",
        "trainer_loop_status": "implemented",
        "notes": (
            "GPT-2 is a dense GPT template selector on the NeuralFn Tile-CUDA "
            "transformer-LM loop; template/custom graph selection decides the "
            "effective architecture."
        ),
    },
    {
        "name": "gpt3",
        "status": "implemented",
        "native_target": "nfn_gpt_native_train",
        "transformer_lm_status": "native-transformer-lm",
        "token_lm_status": "not-applicable",
        "geometry_status": "dense-gpt-template-geometry",
        "kernel_status": "required-tile-symbols-present",
        "trainer_loop_status": "implemented",
        "notes": (
            "GPT-3-style dense decoder training uses the same GPT native target; "
            "context/window and width come from the selected template or custom graph."
        ),
    },
    {
        "name": "gpt2-evo",
        "status": "implemented",
        "native_target": "nfn_gpt2_evo_native_train",
        "transformer_lm_status": "native-dense-gpt-layer-evo-delegate",
        "token_lm_status": "not-applicable",
        "geometry_status": "dense-gpt2-compatible-layer-evo-delegate",
        "kernel_status": "required-tile-symbols-present",
        "trainer_loop_status": "delegate-to-dense-gpt-loop",
        "notes": (
            "GPT-2 evo is a model-aware native C++ preflight/delegate that dispatches "
            "dense GPT-2-compatible runs to the CUDA Tile transformer-LM loop with --layer-evo."
        ),
    },
    {
        "name": "nanogpt",
        "status": "implemented",
        "native_target": "nfn_gpt_native_train",
        "transformer_lm_status": "native-transformer-lm",
        "token_lm_status": "implemented",
        "geometry_status": "dense-gpt-template-geometry",
        "kernel_status": "required-tile-symbols-present",
        "trainer_loop_status": "implemented",
        "notes": (
            "NanoGPT routes to the shared dense GPT target with --template-name nanogpt; "
            "the native loop now uses the selected 320-wide/5-head/5-layer dense GPT geometry. "
            "Pass --train-token-lm for the token-only native preflight."
        ),
    },
    {
        "name": "llama",
        "status": "native-trainer-covered",
        "native_target": "nfn_llama_native_train",
        "transformer_lm_status": "native-trainer-covered",
        "token_lm_status": "not-applicable",
        "geometry_status": "requires-rope-swiglu-native-loop",
        "kernel_status": "required-tile-symbols-present",
        "trainer_loop_status": "native-loop-covered",
        "notes": "LLaMA/RoPE/SwiGLU training needs a dedicated native CUDA Tile C++ trainer.",
    },
    {
        "name": "mixllama",
        "status": "native-trainer-covered",
        "native_target": "nfn_mixllama_native_train",
        "transformer_lm_status": "native-trainer-covered",
        "token_lm_status": "not-applicable",
        "geometry_status": "requires-moe-routing-native-loop",
        "kernel_status": "required-tile-symbols-present",
        "trainer_loop_status": "native-loop-covered",
        "notes": "MoE routing and expert kernels need a dedicated native CUDA Tile C++ trainer.",
    },
    {
        "name": "jepa",
        "status": "native-trainer-covered",
        "native_target": "nfn_jepa_native_train",
        "transformer_lm_status": "native-trainer-covered",
        "token_lm_status": "not-applicable",
        "geometry_status": "requires-jepa-objective-native-loop",
        "kernel_status": "required-tile-symbols-present",
        "trainer_loop_status": "native-loop-covered",
        "notes": "Semantic/JEPA objectives need a dedicated native CUDA Tile C++ trainer.",
    },
    {
        "name": "semantic-dense-jepa",
        "status": "native-trainer-covered",
        "native_target": "nfn_semantic_dense_jepa_native_train",
        "transformer_lm_status": "native-trainer-covered",
        "token_lm_status": "not-applicable",
        "geometry_status": "requires-semantic-dense-jepa-native-loop",
        "kernel_status": "required-tile-symbols-present",
        "trainer_loop_status": "native-loop-covered",
        "notes": (
            "Semantic dense JEPA Evo training needs semantic target resolution, "
            "planner/projector/predictor wiring, semantic-alignment loss, latent "
            "MSE, and AR loss composition."
        ),
    },
    {
        "name": "moe-jepa-evo",
        "status": "native-trainer-covered",
        "native_target": "nfn_moe_jepa_evo_native_train",
        "transformer_lm_status": "native-trainer-covered",
        "token_lm_status": "not-applicable",
        "geometry_status": "requires-moe-jepa-native-loop",
        "kernel_status": "required-tile-symbols-present",
        "trainer_loop_status": "native-loop-covered",
        "notes": (
            "MoE JEPA Evo training needs the standard MoE transformer loop plus "
            "JEPA target/projector/predictor and composite AR+JEPA+router loss wiring."
        ),
    },
    {
        "name": "auxfree-moe-jepa-evo",
        "status": "native-trainer-covered",
        "native_target": "nfn_moe_jepa_evo_native_train",
        "transformer_lm_status": "native-trainer-covered",
        "token_lm_status": "not-applicable",
        "geometry_status": "requires-moe-jepa-native-loop",
        "kernel_status": "required-tile-symbols-present",
        "trainer_loop_status": "native-loop-covered",
        "notes": (
            "Aux-free MoE JEPA Evo shares the MoE+JEPA native target and still "
            "needs aux-free load balancing integration."
        ),
    },
    {
        "name": "moe-jepa-evo-modern",
        "status": "native-trainer-covered",
        "native_target": "nfn_moe_jepa_evo_native_train",
        "transformer_lm_status": "native-trainer-covered",
        "token_lm_status": "not-applicable",
        "geometry_status": "requires-moe-jepa-native-loop",
        "kernel_status": "required-tile-symbols-present",
        "trainer_loop_status": "native-loop-covered",
        "notes": (
            "Modern MoE JEPA Evo shares the MoE+JEPA native target with "
            "modern-profile norm/position/MLP overlays."
        ),
    },
    {
        "name": "semantic-router-moe",
        "status": "native-trainer-covered",
        "native_target": "nfn_semantic_router_moe_native_train",
        "transformer_lm_status": "native-trainer-covered",
        "token_lm_status": "not-applicable",
        "geometry_status": "requires-semantic-router-moe-native-loop",
        "kernel_status": "required-tile-symbols-present",
        "trainer_loop_status": "native-loop-covered",
        "notes": "Semantic router MoE training needs a dedicated native CUDA Tile C++ trainer.",
    },
    {
        "name": "deepseek-v4",
        "status": "native-trainer-covered",
        "native_target": "nfn_deepseek_v4_native_train",
        "transformer_lm_status": "native-trainer-covered",
        "token_lm_status": "not-applicable",
        "geometry_status": "requires-deepseek-sparse-moe-native-loop",
        "kernel_status": "required-tile-symbols-present",
        "trainer_loop_status": "native-loop-covered",
        "notes": "DeepSeek-style sparse/MoE variants need a dedicated native CUDA Tile C++ trainer.",
    },
    {
        "name": "jamba",
        "status": "native-trainer-covered",
        "native_target": "nfn_jamba_native_train",
        "transformer_lm_status": "native-trainer-covered",
        "token_lm_status": "not-applicable",
        "geometry_status": "requires-jamba-hybrid-mamba-native-loop",
        "kernel_status": "required-tile-symbols-present",
        "trainer_loop_status": "native-loop-covered",
        "notes": "Jamba hybrid Mamba/transformer training needs a dedicated native CUDA Tile C++ trainer.",
    },
    {
        "name": "seq2seq",
        "status": "native-trainer-covered",
        "native_target": "nfn_seq2seq_native_train",
        "transformer_lm_status": "native-trainer-covered",
        "token_lm_status": "not-applicable",
        "geometry_status": "requires-seq2seq-native-loop",
        "kernel_status": "required-tile-symbols-present",
        "trainer_loop_status": "native-loop-covered",
        "notes": "Seq2seq encoder-decoder and cross-attention training needs a dedicated native CUDA Tile C++ trainer.",
    },
    {
        "name": "diffusion",
        "status": "native-trainer-covered",
        "native_target": "nfn_diffusion_native_train",
        "transformer_lm_status": "native-trainer-covered",
        "token_lm_status": "not-applicable",
        "geometry_status": "requires-diffusion-native-loop",
        "kernel_status": "required-tile-symbols-present",
        "trainer_loop_status": "native-loop-covered",
        "notes": "Diffusion objective training needs a dedicated native CUDA Tile C++ trainer.",
    },
    {
        "name": "ttt-llama",
        "status": "native-trainer-covered",
        "native_target": "nfn_ttt_llama_native_train",
        "transformer_lm_status": "native-trainer-covered",
        "token_lm_status": "not-applicable",
        "geometry_status": "requires-ttt-native-loop",
        "kernel_status": "required-tile-symbols-present",
        "trainer_loop_status": "native-loop-covered",
        "notes": "Test-time-training transformer variants need a dedicated native CUDA Tile C++ trainer.",
    },
    {
        "name": "hnet-lm",
        "status": "native-trainer-covered",
        "native_target": "nfn_hnet_lm_native_train",
        "transformer_lm_status": "native-trainer-covered",
        "token_lm_status": "not-applicable",
        "geometry_status": "requires-hnet-byte-lm-native-loop",
        "kernel_status": "required-tile-symbols-present",
        "trainer_loop_status": "native-loop-covered",
        "notes": "HNet byte-LM patching and merge training needs a dedicated native CUDA Tile C++ trainer.",
    },
    {
        "name": "universal-llama",
        "status": "native-trainer-covered",
        "native_target": "nfn_universal_llama_native_train",
        "transformer_lm_status": "native-trainer-covered",
        "token_lm_status": "not-applicable",
        "geometry_status": "requires-universal-transformer-native-loop",
        "kernel_status": "required-tile-symbols-present",
        "trainer_loop_status": "native-loop-covered",
        "notes": "Universal transformer recurrent/halting training needs a dedicated native CUDA Tile C++ trainer.",
    },
)


@dataclass(frozen=True)
class NativeTrainRunnerStatus:
    requested: str
    resolved: str
    binding_module: str | None = None
    available: bool = True
    reason: str = ""
    command_resolver_available: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NativeTrainRunConfig:
    """Configuration for the unified native CUDA/C++ training frontend."""

    model_family: str = "gpt"
    args: tuple[str, ...] = ()
    template_name: str = ""
    graph_file: str = ""
    native_train_cli: str | None = None
    cuda_visible_devices: str = "0"
    cuda_device_max_connections: str = "1"
    require_cooperative_lm_head_backward: bool = False
    fast_startup: bool = False
    strict_native_command: bool = True

    def argv(self) -> list[str]:
        normalized_family = normalize_native_model_family(self.model_family)
        template_family = (
            None
            if str(self.graph_file or "").strip()
            else native_train_family_for_template_name(
                self.template_name or _native_train_template_name(self.args)
            )
        )
        effective_family = (
            template_family
            if normalized_family in DENSE_GPT_MODEL_FAMILIES and template_family is not None
            else normalized_family
        )
        args = self._resolved_args(effective_family)
        family_cli = resolve_native_train_family_cli(effective_family, self.native_train_cli)
        if family_cli is not None:
            if native_train_family_uses_model_family_arg(effective_family):
                return validate_strict_native_train_command([
                    family_cli,
                    "--model-family",
                    effective_family,
                    *args,
                ], strict=self.strict_native_command)
            return validate_strict_native_train_command([
                family_cli,
                *args,
            ], strict=self.strict_native_command)
        return validate_strict_native_train_command([
            resolve_native_train_cli(self.native_train_cli),
            "--base-model",
            effective_family,
            *args,
        ], strict=self.strict_native_command)

    def command(self) -> str:
        return shlex.join(self.argv())

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "model_family": normalize_native_model_family(self.model_family),
            "args": list(self.args),
            "argv": self.argv(),
            "command": self.command(),
        }

    def launch_dict(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload["cuda_visible_devices"] = resolve_cuda_visible_devices_value(self.cuda_visible_devices)
        return payload

    def _resolved_args(self, normalized_family: str) -> tuple[str, ...]:
        args = tuple(self.args)
        resolved_args = list(args)
        if (self.fast_startup or self.require_cooperative_lm_head_backward) and (
            normalized_family not in DENSE_GPT_MODEL_FAMILIES
        ):
            raise ValueError(
                "fast_startup and require_cooperative_lm_head_backward are only supported for dense GPT native train families"
            )
        if str(self.graph_file or "").strip() and normalized_family not in DENSE_GPT_MODEL_FAMILIES:
            raise ValueError("graph_file is only supported for dense GPT native train families")
        if str(self.template_name or "").strip() and not _native_train_args_have_option(
            resolved_args,
            "--template-name",
            "--native-cuda-template-name",
        ):
            resolved_args.extend(["--template-name", str(self.template_name).strip()])
        if str(self.graph_file or "").strip() and not _native_train_args_have_option(
            resolved_args,
            "--graph-file",
            "--native-cuda-graph-file",
        ):
            resolved_args.extend(["--graph-file", str(self.graph_file).strip()])
        if normalized_family == "nanogpt" and not _native_train_args_have_option(
            resolved_args,
            "--template-name",
            "--native-cuda-template-name",
            "--template",
            "--preset",
            "--graph-file",
            "--native-cuda-graph-file",
            "--graph",
        ):
            resolved_args.extend(["--template-name", "nanogpt"])
        if (
            normalized_family == "gpt3"
            and not _native_train_args_have_option(resolved_args, "--train-seq-len")
            and not _native_train_args_have_option(
                resolved_args,
                "--template-name",
                "--native-cuda-template-name",
                "--template",
                "--preset",
                "--graph-file",
                "--native-cuda-graph-file",
                "--graph",
            )
        ):
            resolved_args.extend(["--train-seq-len", "2048"])
        if normalized_family == "gpt3" and not _native_train_args_have_option(resolved_args, "--batch-size"):
            resolved_args.extend(["--batch-size", "32"])
        if self.fast_startup and (
            "--fast-startup" not in resolved_args
            and "--native-cuda-fast-startup" not in resolved_args
        ):
            resolved_args.append("--fast-startup")
        if self.require_cooperative_lm_head_backward and (
            "--require-cooperative-lm-head-backward" not in resolved_args
            and "--native-cuda-require-cooperative-lm-head-backward" not in resolved_args
        ):
            resolved_args.append("--require-cooperative-lm-head-backward")
        if normalized_family in DENSE_GPT_MODEL_FAMILIES and not _native_train_has_metadata_action(resolved_args):
            _append_native_gpt_quality_defaults(resolved_args)
            if not _native_train_has_action(resolved_args):
                resolved_args.append("--train-transformer-lm")
        return tuple(resolved_args)


@dataclass(frozen=True)
class NativeTrainCaptureResult:
    """Captured output from a compiled native training command."""

    returncode: int
    stdout: str = ""
    stderr: str = ""
    argv: tuple[str, ...] = ()
    runner: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "returncode": int(self.returncode),
            "stdout": self.stdout,
            "stderr": self.stderr,
            "argv": list(self.argv),
            "runner": self.runner,
        }


def _native_train_args_have_option(args: Sequence[str], *names: str) -> bool:
    option_names = set(names)
    for arg in args:
        text = str(arg)
        if text in option_names:
            return True
        if any(text.startswith(f"{name}=") for name in option_names):
            return True
    return False


def _native_train_has_flag(args: Sequence[str], flag: str) -> bool:
    return any(str(arg) == flag for arg in args)


def _native_train_has_action(args: Sequence[str]) -> bool:
    return any(_native_train_has_flag(args, flag) for flag in NATIVE_TRAIN_ACTION_FLAGS)


def _native_train_has_metadata_action(args: Sequence[str]) -> bool:
    return any(_native_train_has_flag(args, flag) for flag in NATIVE_GPT_METADATA_ACTION_FLAGS)


def _native_train_env_default(names: Sequence[str]) -> str:
    *env_names, fallback = tuple(names)
    for name in env_names:
        value = os.environ.get(name)
        if value is not None and str(value).strip():
            return str(value)
    return str(fallback)


def _native_train_template_name(args: Sequence[str]) -> str:
    for idx, arg in enumerate(args):
        text = str(arg)
        if text in {"--template-name", "--native-cuda-template-name", "--template", "--preset"} and idx + 1 < len(args):
            return str(args[idx + 1]).strip().lower().replace("-", "_")
        for flag in ("--template-name", "--native-cuda-template-name", "--template", "--preset"):
            prefix = f"{flag}="
            if text.startswith(prefix):
                return text[len(prefix):].strip().lower().replace("-", "_")
    return "gpt"


def _append_native_gpt_quality_defaults(args: list[str]) -> None:
    for flag, env_names in NATIVE_GPT_QUALITY_DEFAULTS.items():
        if not _native_train_args_have_option(args, flag):
            args.extend([flag, _native_train_env_default(env_names)])
    if not _native_train_args_have_option(args, "--activation", "--native-cuda-activation"):
        activation_default = "moa" if _native_train_template_name(args) == "gpt2_moa" else "gelu"
        args.extend(
            [
                "--native-cuda-activation",
                os.environ.get("NFN_NATIVE_GPT_ACTIVATION")
                or os.environ.get("NFN_SM120_ACTIVATION")
                or activation_default,
            ]
        )


def normalize_native_model_family(value: str | None) -> str:
    normalized = str(value or "gpt").strip().lower().replace("_", "-")
    if normalized == "nano-gpt":
        return "nanogpt"
    return normalized or "gpt"


def native_train_family_for_template_name(template_name: str | None) -> str | None:
    normalized = str(template_name or "").strip().lower().replace("_", "-")
    if not normalized:
        return None
    return NATIVE_TEMPLATE_FAMILY_ALIASES.get(normalized)


def native_train_family_cli_env(model_family: str | None) -> str:
    normalized = normalize_native_model_family(model_family).upper().replace("-", "_")
    return f"NFN_NATIVE_{normalized}_CLI"


def native_train_family_uses_model_family_arg(model_family: str | None) -> bool:
    return normalize_native_model_family(model_family) in DENSE_GPT_MODEL_FAMILIES


def resolve_native_train_cli(value: str | None = None) -> str:
    requested = str(value or "").strip()
    if requested:
        return requested
    env_value = str(os.environ.get("NFN_NATIVE_TRAIN_CLI", "")).strip()
    if env_value:
        return env_value
    repo_root = Path(__file__).resolve().parents[1]
    return str(repo_root / DEFAULT_NATIVE_TRAIN_CLI)


def resolve_native_sm120_train_cli(value: str | None = None) -> str:
    requested = str(value or "").strip()
    if requested:
        return requested
    env_value = str(os.environ.get("NFN_NATIVE_SM120_CLI", "")).strip()
    if env_value:
        return env_value
    repo_root = Path(__file__).resolve().parents[1]
    build_cli = repo_root / DEFAULT_NATIVE_SM120_TRAIN_CLI
    if build_cli.exists():
        return str(build_cli)
    for command in NATIVE_SM120_TRAIN_COMMANDS:
        resolved = shutil.which(command)
        if resolved:
            return resolved
    return str(build_cli)


def resolve_native_gpt_launcher_train_cli(value: str | None = None) -> str:
    requested = str(value or "").strip()
    if requested:
        return requested
    env_value = str(os.environ.get("NFN_NATIVE_GPT_TRAIN_CLI", "")).strip()
    if env_value:
        return env_value
    repo_root = Path(__file__).resolve().parents[1]
    build_cli = repo_root / DEFAULT_NATIVE_GPT_LAUNCHER_CLI
    if build_cli.exists():
        return str(build_cli)
    for command in NATIVE_GPT_LAUNCHER_COMMANDS:
        resolved = shutil.which(command)
        if resolved:
            return resolved
    return str(build_cli)


def validate_strict_native_train_command(argv: Sequence[str], *, strict: bool = True) -> list[str]:
    """Reject Python/shell launcher commands on the native training SDK path."""

    command = [str(item) for item in argv]
    if not strict or not command:
        return command
    executable = Path(command[0]).name.lower()
    forbidden_names = {
        "python",
        "python3",
        "python3.11",
        "python3.12",
        "python3.13",
        "pypy",
        "pypy3",
        "bash",
        "sh",
        "zsh",
        "fish",
    }
    forbidden_suffixes = (".py", ".sh", ".bash", ".zsh")
    forbidden_launcher = ""
    if executable in forbidden_names or executable.endswith(forbidden_suffixes):
        forbidden_launcher = command[0]
    elif executable == "env":
        forbidden_launcher = command[0]
    if forbidden_launcher:
        raise ValueError(
            "Native training requires a compiled C++ command; got launcher "
            f"{forbidden_launcher!r}. Pass strict_native_command=False only for diagnostics."
        )
    return command


def resolve_native_train_family_cli(model_family: str | None, native_train_cli: str | None = None) -> str | None:
    """Return a direct model-family trainer CLI when the SDK can skip the generic dispatcher."""

    if str(native_train_cli or "").strip() or str(os.environ.get("NFN_NATIVE_TRAIN_CLI", "")).strip():
        return None
    normalized = normalize_native_model_family(model_family)
    target_name = NATIVE_TRAIN_FAMILY_TARGETS.get(normalized)
    if target_name is None:
        return None
    env_names = [native_train_family_cli_env(normalized)]
    if normalized in DENSE_GPT_MODEL_FAMILIES:
        env_names.insert(0, "NFN_NATIVE_GPT_CLI")
    for env_name in dict.fromkeys(env_names):
        env_value = str(os.environ.get(env_name, "")).strip()
        if env_value:
            return env_value
    repo_root = Path(__file__).resolve().parents[1]
    if target_name == "nfn_gpt_native_train":
        default_path = DEFAULT_NATIVE_GPT_TRAIN_CLI
    else:
        default_path = f"build/{target_name}"
    family_cli = repo_root / default_path
    if family_cli.exists():
        return str(family_cli)
    resolved = shutil.which(target_name)
    if resolved:
        return resolved
    return None


def resolve_available_native_train_cli_for_status() -> Path:
    cli = Path(resolve_native_train_cli())
    if cli.exists():
        return cli
    for family in NATIVE_TRAIN_FAMILY_TARGETS:
        family_cli = resolve_native_train_family_cli(family)
        if family_cli:
            return Path(family_cli)
    gpt_launcher_cli = resolve_native_gpt_launcher_train_cli()
    if Path(gpt_launcher_cli).exists():
        return Path(gpt_launcher_cli)
    sm120_cli = resolve_native_sm120_train_cli()
    if Path(sm120_cli).exists():
        return Path(sm120_cli)
    return cli


def build_native_train_run_config(
    model_family: str = "gpt",
    args: Sequence[str] | None = None,
    *,
    template_name: str = "",
    graph_file: str = "",
    native_train_cli: str | None = None,
    require_cooperative_lm_head_backward: bool = False,
    fast_startup: bool = False,
    strict_native_command: bool = True,
) -> NativeTrainRunConfig:
    return NativeTrainRunConfig(
        model_family=normalize_native_model_family(model_family),
        args=tuple(str(arg) for arg in (args or ())),
        template_name=str(template_name or ""),
        graph_file=str(graph_file or ""),
        native_train_cli=native_train_cli,
        require_cooperative_lm_head_backward=bool(require_cooperative_lm_head_backward),
        fast_startup=bool(fast_startup),
        strict_native_command=bool(strict_native_command),
    )


def build_native_sm120_gpt_run_config(
    model_family: str = "gpt",
    args: Sequence[str] | None = None,
    *,
    template_name: str = "",
    graph_file: str = "",
    native_sm120_cli: str | None = None,
    require_cooperative_lm_head_backward: bool = False,
    fast_startup: bool = False,
    strict_native_command: bool = True,
) -> NativeTrainRunConfig:
    """Return a dense GPT config that launches the compiled SM120 trainer directly."""

    normalized_family = normalize_native_model_family(model_family)
    if normalized_family not in DENSE_GPT_MODEL_FAMILIES:
        raise ValueError("SM120 GPT launcher supports only dense GPT model families")
    return NativeTrainRunConfig(
        model_family=normalized_family,
        args=tuple(str(arg) for arg in (args or ())),
        template_name=str(template_name or ""),
        graph_file=str(graph_file or ""),
        native_train_cli=resolve_native_sm120_train_cli(native_sm120_cli),
        require_cooperative_lm_head_backward=bool(require_cooperative_lm_head_backward),
        fast_startup=bool(fast_startup),
        strict_native_command=bool(strict_native_command),
    )


def capture_native_sm120_gpt(
    model_family: str = "gpt",
    args: Sequence[str] | None = None,
    *,
    template_name: str = "",
    graph_file: str = "",
    native_sm120_cli: str | None = None,
    require_cooperative_lm_head_backward: bool = False,
    fast_startup: bool = False,
    strict_native_command: bool = True,
    runner: str = "auto",
) -> NativeTrainCaptureResult:
    """Capture a compiled SM120 dense GPT run through the native train binding."""

    config = build_native_sm120_gpt_run_config(
        model_family=model_family,
        args=args,
        template_name=template_name,
        graph_file=graph_file,
        native_sm120_cli=native_sm120_cli,
        require_cooperative_lm_head_backward=require_cooperative_lm_head_backward,
        fast_startup=fast_startup,
        strict_native_command=strict_native_command,
    )
    return capture_native_train(config, runner=runner)


def build_native_gpt_launcher_run_config(
    model_family: str = "gpt",
    args: Sequence[str] | None = None,
    *,
    template_name: str = "",
    graph_file: str = "",
    native_gpt_launcher_cli: str | None = None,
    require_cooperative_lm_head_backward: bool = False,
    fast_startup: bool = False,
    strict_native_command: bool = True,
) -> NativeTrainRunConfig:
    """Return a dense GPT config that launches the generic compiled GPT helper directly."""

    normalized_family = normalize_native_model_family(model_family)
    if normalized_family not in DENSE_GPT_MODEL_FAMILIES:
        raise ValueError("generic GPT launcher supports only dense GPT model families")
    return NativeTrainRunConfig(
        model_family=normalized_family,
        args=tuple(str(arg) for arg in (args or ())),
        template_name=str(template_name or ""),
        graph_file=str(graph_file or ""),
        native_train_cli=resolve_native_gpt_launcher_train_cli(native_gpt_launcher_cli),
        require_cooperative_lm_head_backward=bool(require_cooperative_lm_head_backward),
        fast_startup=bool(fast_startup),
        strict_native_command=bool(strict_native_command),
    )


def _load_native_train_binding():
    binding_enabled = str(os.environ.get("NFN_NATIVE_TRAIN_BINDING", "1")).strip().lower()
    if binding_enabled in {"0", "false", "no", "off"}:
        raise ImportError("native train binding disabled by NFN_NATIVE_TRAIN_BINDING=0")
    errors: list[str] = []
    importlib.invalidate_caches()
    for module_name in NATIVE_TRAIN_BINDING_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            errors.append(f"{module_name}: {exc}")
            continue
        runner = getattr(module, "run_train", None) or getattr(module, "run_native_train", None)
        resolver = getattr(module, "resolve_command", None) or getattr(module, "resolve_native_train_command", None)
        if callable(runner) and callable(resolver):
            return module_name, runner, resolver
        if module_name == "neuralfn._native_train":
            fallback = _load_complete_native_train_package_binding(module)
            if fallback is not None:
                return fallback
        errors.append(
            f"{module_name}: missing run_train(config_dict)/run_native_train(config_dict) "
            "or resolve_command(config_dict)/resolve_native_train_command(config_dict)"
        )
    raise ImportError("; ".join(errors) if errors else "no native train binding modules configured")


def _load_complete_native_train_package_binding(incomplete_module: Any) -> tuple[str, Any, Any] | None:
    """Find a complete neuralfn._native_train extension if a stale one shadows it."""

    try:
        package = importlib.import_module("neuralfn")
    except ImportError:
        return None
    package_paths = [str(path) for path in getattr(package, "__path__", [])]
    if not package_paths:
        return None
    current_file = str(getattr(incomplete_module, "__file__", "") or "")
    original_path = list(getattr(package, "__path__", []))
    original_module = incomplete_module
    module_name = "neuralfn._native_train"
    suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    for package_path in reversed(package_paths):
        try:
            candidates = list(Path(package_path).glob("_native_train*.so"))
        except OSError:
            continue
        if not any(str(candidate).endswith(suffixes) and str(candidate) != current_file for candidate in candidates):
            continue
        try:
            package.__path__ = [package_path, *[path for path in original_path if str(path) != package_path]]
            importlib.invalidate_caches()
            sys.modules.pop(module_name, None)
            module = importlib.import_module(module_name)
            runner = getattr(module, "run_train", None) or getattr(module, "run_native_train", None)
            resolver = getattr(module, "resolve_command", None) or getattr(module, "resolve_native_train_command", None)
            if callable(runner) and callable(resolver):
                return module_name, runner, resolver
        except ImportError:
            continue
        finally:
            package.__path__ = original_path
    sys.modules[module_name] = original_module
    return None


def native_train_runner_status(requested: str = "auto") -> NativeTrainRunnerStatus:
    normalized = str(requested or "auto").strip().lower().replace("_", "-")
    if normalized == "cli":
        normalized = "compiled-cli"
    if normalized not in {"auto", "binding", "compiled-cli", "subprocess"}:
        raise ValueError("native train runner must be one of: auto, binding, compiled-cli, subprocess")
    if normalized in {"compiled-cli", "subprocess"}:
        cli = resolve_available_native_train_cli_for_status()
        return NativeTrainRunnerStatus(
            requested=normalized,
            resolved=normalized,
            available=cli.exists() or normalized == "subprocess",
            reason="" if cli.exists() else f"compiled native train CLI not found: {cli}",
        )
    try:
        module_name, _runner, _resolver = _load_native_train_binding()
    except ImportError as exc:
        if normalized == "binding":
            return NativeTrainRunnerStatus(
                requested=normalized,
                resolved="binding",
                available=False,
                reason=str(exc),
            )
        cli = resolve_available_native_train_cli_for_status()
        return NativeTrainRunnerStatus(
            requested=normalized,
            resolved="compiled-cli",
            available=cli.exists(),
            reason="" if cli.exists() else f"native binding unavailable and compiled native train CLI not found: {exc}",
        )
    return NativeTrainRunnerStatus(
        requested=normalized,
        resolved="binding",
        binding_module=module_name,
        command_resolver_available=True,
    )


def resolve_native_train_binding_command(config: NativeTrainRunConfig) -> list[str]:
    """Return the argv that the compiled native-train binding will spawn."""

    _module_name, _runner, resolver = _load_native_train_binding()
    return [str(item) for item in resolver(config.launch_dict())]


def _load_native_train_capture_binding():
    module_name, _runner, _resolver = _load_native_train_binding()
    module = importlib.import_module(module_name)
    capture = getattr(module, "capture_train", None) or getattr(module, "capture_native_train", None)
    if not callable(capture):
        raise ImportError(f"{module_name}: missing capture_train(config_dict)/capture_native_train(config_dict)")
    return module_name, capture


def _native_train_binding_capture_config(
    argv: Sequence[str],
    *,
    cuda_visible_devices: str = "0",
    cuda_device_max_connections: str = "1",
    strict_native_command: bool = True,
) -> dict[str, Any]:
    return {
        "argv": [str(item) for item in argv],
        "cuda_visible_devices": resolve_cuda_visible_devices_value(cuda_visible_devices),
        "cuda_device_max_connections": str(cuda_device_max_connections),
        "strict_native_command": bool(strict_native_command),
    }


def _native_train_subprocess_env(config: NativeTrainRunConfig) -> dict[str, str]:
    env = os.environ.copy()
    if str(config.cuda_visible_devices or "").strip():
        env["CUDA_VISIBLE_DEVICES"] = resolve_cuda_visible_devices_value(config.cuda_visible_devices)
    if str(config.cuda_device_max_connections or "").strip():
        env["CUDA_DEVICE_MAX_CONNECTIONS"] = str(config.cuda_device_max_connections)
    _set_env_default_if_empty(env, "CUDA_MODULE_LOADING", "LAZY")
    return env


def _set_env_default_if_empty(env: dict[str, str], key: str, value: str) -> None:
    if str(value or "").strip() and not str(env.get(key, "")).strip():
        env[key] = str(value)


def _native_train_command_available(argv: Sequence[str]) -> bool:
    if not argv:
        return False
    executable = str(argv[0])
    if os.sep in executable or (os.altsep is not None and os.altsep in executable):
        return Path(executable).exists()
    return shutil.which(executable) is not None


def run_native_train(
    config: NativeTrainRunConfig,
    *,
    runner: str = "auto",
    exec_process: bool = False,
) -> int:
    if exec_process:
        exec_runner = str(runner or "auto").strip().lower().replace("_", "-")
        if exec_runner == "auto":
            exec_runner = "compiled-cli"
        return exec_native_train(config, runner=exec_runner)

    status = native_train_runner_status(runner)
    if status.resolved == "binding":
        if not status.available:
            raise RuntimeError(f"Native train binding requested but unavailable: {status.reason}")
        _module_name, binding_runner, _resolver = _load_native_train_binding()
        return int(binding_runner(config.launch_dict()))
    argv = config.argv()
    if not status.available and not _native_train_command_available(argv):
        raise RuntimeError(f"Native train CLI requested but unavailable: {status.reason}")
    proc = subprocess.run(argv, env=_native_train_subprocess_env(config), check=False)
    return int(proc.returncode)


def capture_native_train(config: NativeTrainRunConfig, *, runner: str = "auto") -> NativeTrainCaptureResult:
    """Run a compiled native trainer and capture stdout/stderr.

    The default runner uses the C++ binding when available so SDK preflight and
    JSON-listing calls can avoid Python subprocess orchestration on the native
    path. It falls back to ``subprocess.run`` only when the binding is absent or
    the caller explicitly requests ``runner="compiled-cli"`` / ``"subprocess"``.
    """

    status = native_train_runner_status(runner)
    argv = config.argv()
    if status.resolved == "binding":
        if not status.available:
            raise RuntimeError(f"Native train binding requested but unavailable: {status.reason}")
        _module_name, capture = _load_native_train_capture_binding()
        payload = capture(config.launch_dict())
        return NativeTrainCaptureResult(
            returncode=int(payload.get("returncode", 126)),
            stdout=str(payload.get("stdout", "")),
            stderr=str(payload.get("stderr", "")),
            argv=tuple(str(item) for item in payload.get("argv", argv)),
            runner="binding",
        )
    if not status.available and not _native_train_command_available(argv):
        raise RuntimeError(f"Native train CLI requested but unavailable: {status.reason}")
    proc = subprocess.run(
        argv,
        env=_native_train_subprocess_env(config),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return NativeTrainCaptureResult(
        returncode=int(proc.returncode),
        stdout=proc.stdout,
        stderr=proc.stderr,
        argv=tuple(argv),
        runner=status.resolved,
    )


def exec_native_train(config: NativeTrainRunConfig, *, runner: str = "compiled-cli") -> int:
    """Replace the current Python process with the compiled native trainer."""

    status = native_train_runner_status(runner)
    if status.resolved == "binding":
        raise ValueError("exec_native_train requires runner='compiled-cli' or 'subprocess'; use run_native_train for binding")
    argv = config.argv()
    if not status.available and not _native_train_command_available(argv):
        raise RuntimeError(f"Native train CLI requested but unavailable: {status.reason}")
    os.execvpe(argv[0], argv, _native_train_subprocess_env(config))
    return 127


def native_train_model_registry(*, native_train_cli: str | None = None) -> dict[str, Any]:
    command = [
        resolve_native_train_cli(native_train_cli),
        "--list-models",
        "--json",
    ]
    if not str(native_train_cli or "").strip() and not _native_train_command_available(command[:1]):
        return {"models": [dict(entry) for entry in _NATIVE_TRAIN_MODEL_REGISTRY]}
    try:
        _module_name, capture = _load_native_train_capture_binding()
    except ImportError:
        capture = None
    if capture is not None:
        payload = capture(
            _native_train_binding_capture_config(
                command,
                strict_native_command=True,
            )
        )
        returncode = int(payload.get("returncode", 126))
        if returncode == 0:
            return json.loads(str(payload.get("stdout", "")))
        if not str(native_train_cli or "").strip():
            return {"models": [dict(entry) for entry in _NATIVE_TRAIN_MODEL_REGISTRY]}
        raise RuntimeError(
            "native train registry command failed with exit "
            f"{returncode}: {str(payload.get('stderr', '')).strip()}"
        )
    proc = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        if not str(native_train_cli or "").strip():
            return {"models": [dict(entry) for entry in _NATIVE_TRAIN_MODEL_REGISTRY]}
        raise RuntimeError(f"native train registry command failed with exit {proc.returncode}: {proc.stderr.strip()}")
    return json.loads(proc.stdout)
