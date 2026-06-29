from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from native_training_guard import reject_torch_training_by_default


MODE_NAME = "gpt2_evo"
GRAPH_NAME = f"{MODE_NAME}_sdk"
GPT2_EVO_DEFAULTS = {
    "seed": 1337,
    "device": "cuda",
    "dataset_alias": "roneneldan__TinyStories__TinyStoriesV2-GPT4",
    "max_steps": 20_000,
    "train_seq_len": 1_024,
    "batch_size": 64,
    "train_batch_tokens": 524_288,
    "eval_batches": 20,
    "eval_batch_size": 64,
    "eval_every_steps": 1000,
    "train_log_every": 10,
    "max_wallclock_seconds": 0.0,
    "warmup_steps": 1000,
    "warmdown_fraction": 0.0,
    "vocab_size": 50_257,
    "num_layers": 12,
    "model_dim": 768,
    "num_heads": 12,
    "logit_softcap": 0.0,
    "optimizer_profile": "adamw",
    "learning_rate": 6e-4,
    "weight_decay": 0.1,
    "kernel_backend": "tile-cuda",
    "tile_cuda_activation_dtype": "nvfp4",
    "evo_layer_index": 6,
    "evo_layer_interval": 10,
    "evo_layer_population": 8,
    "evo_layer_mutation_scale": 0.02,
}


def main(argv: list[str] | None = None) -> int:
    """Dispatch GPT-2-evo training to the compiled native CUDA/C++ frontend."""

    original_argv = sys.argv
    if argv is not None:
        sys.argv = [str(Path(__file__).resolve()), *argv]
    try:
        reject_torch_training_by_default(
            "train_gpt2_evo.py",
            native_target="nfn train --base-model gpt2-evo",
            model_family="gpt2-evo",
            family_native_cli_env="NFN_NATIVE_GPT2_EVO_CLI",
            family_native_cli_name="nfn_gpt2_evo_native_train",
        )
    finally:
        if argv is not None:
            sys.argv = original_argv
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
