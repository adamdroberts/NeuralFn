from __future__ import annotations

import argparse
from datetime import datetime
import logging
import math
from pathlib import Path
import signal
import sys
import uuid

from native_training_guard import reject_torch_training_by_default

if __name__ == "__main__":
    reject_torch_training_by_default(
        "train_gpt2_evo.py",
        native_target="nfn train --base-model gpt2-evo",
        model_family="gpt2-evo",
        family_native_cli_env="NFN_NATIVE_GPT2_EVO_CLI",
        family_native_cli_name="nfn_gpt2_evo_native_train",
    )

import numpy as np
import torch

from cli_utils import artifact_path, create_argument_parser
from neuralfn import TorchTrainer
from neuralfn.config import build_gpt2_evo_spec
from neuralfn.torch_backend import resolve_amp_settings
from neuralfn.torch_templates import build_gpt_root_graph
from server.models import LoadDatasetRequest
from server.services.graph_ops import load_dataset_source_into_graph

from train_jepa_semantic import (
    DEFAULT_DATASET_ALIAS,
    add_all_train_rows_argument,
    apply_sanitized_template_spec,
    apply_cached_tokenizer_vocab_policy,
    add_dataset_download_arguments,
    add_dataset_selector_arguments,
    add_evolutionary_training_arguments,
    add_lr_schedule_arguments,
    add_max_steps_argument,
    add_max_wallclock_seconds_argument,
    add_pretraining_file_argument,
    add_raw_text_tokenizer_arguments,
    add_warmdown_fraction_argument,
    apply_raw_text_tokenizer_policy,
    apply_tinystories_dataset_defaults,
    dataset_download_kwargs_from_args,
    estimate_text_schedule,
    load_val_token_dataset,
    print_graph_summary,
    resolve_dataset_selector_args,
    resolve_effective_training_schedule,
    resolve_lr_schedule_defaults,
    resolve_or_download_dataset,
    resolve_pretraining_file_dataset,
    save_artifacts,
)
from train_llama_fast import (
    build_progress_logger,
    build_trainer_config,
    configure_console_logging,
    env_float,
    env_int,
    env_str,
    print_data_source_summary,
    print_resolved_summary,
    safe_evaluate_validation_loss,
)

MODE_NAME = "gpt2_evo"
GRAPH_NAME = f"{MODE_NAME}_sdk"
DEFAULT_ARTIFACT = artifact_path(f"{MODE_NAME}.pt")
INTERRUPTED_ARTIFACT = DEFAULT_ARTIFACT.with_name(f"{MODE_NAME}.interrupted.pt")

GPT2_EVO_DEFAULTS = {
    "seed": 1337,
    "device": "cuda",
    "run_id": str(uuid.uuid4()),
    "dataset_alias": DEFAULT_DATASET_ALIAS,
    "output": str(DEFAULT_ARTIFACT),
    "max_steps": 20_000,
    "train_seq_len": 1_024,
    "batch_size": 64,
    "train_batch_tokens": 524_288,
    "eval_batches": 20,
    "eval_batch_size": 64,
    "eval_every_steps": 250,
    "train_log_every": 10,
    "max_wallclock_seconds": 0.0,
    "warmup_steps": 60,
    "warmdown_fraction": 0.0,
    "vocab_size": 1_024,
    "num_layers": 12,
    "model_dim": 768,
    "num_heads": 12,
    "logit_softcap": 0.0,
    "optimizer_profile": "adamw",
    "learning_rate": 6e-4,
    "weight_decay": 0.1,
    "embed_lr": 0.02,
    "head_lr": 0.005,
    "tied_embed_lr": 0.01,
    "matrix_lr": 0.008,
    "scalar_lr": 0.004,
    "muon_momentum": 0.95,
    "muon_backend_steps": 5,
    "muon_momentum_warmup_start": 0.85,
    "muon_momentum_warmup_steps": 64,
    "beta1": 0.9,
    "beta2": 0.95,
    "adam_eps": 1e-8,
    "grad_clip_norm": 1.0,
    "kernel_backend": "tile-cuda",
    "tile_cuda_activation_dtype": "nvfp4",
    "amp_dtype": "bfloat16",
    "evo_layer_index": 6,
    "evo_layer_interval": 10,
    "evo_layer_population": 8,
    "evo_layer_mutation_scale": 0.02,
}

LOGGER = logging.getLogger("gpt2_evo_harness")


def log_stage(message: str) -> None:
    LOGGER.info(message)


def resolve_mode_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if not getattr(args, "output", ""):
        args.output = str(DEFAULT_ARTIFACT)
    resolve_lr_schedule_defaults(args)
    apply_raw_text_tokenizer_policy(
        args,
        preset_name="gpt2",
        default_vocab_size=int(GPT2_EVO_DEFAULTS["vocab_size"]),
    )
    return args


def build_parser() -> argparse.ArgumentParser:
    parser = create_argument_parser(
        description=(
            "Train a dense GPT-2 with one evolution-trained layer on the NeuralFn "
            "CUDA Tile harness."
        )
    )
    parser.add_argument("--run-id", default=env_str("RUN_ID", GPT2_EVO_DEFAULTS["run_id"]))
    parser.add_argument("--seed", type=int, default=env_int("SEED", GPT2_EVO_DEFAULTS["seed"]))
    parser.add_argument("--device", default=env_str("DEVICE", GPT2_EVO_DEFAULTS["device"]))
    add_dataset_selector_arguments(
        parser,
        default_alias=env_str("DATASET_ALIAS", GPT2_EVO_DEFAULTS["dataset_alias"]),
    )
    add_dataset_download_arguments(parser)
    add_pretraining_file_argument(parser)
    add_raw_text_tokenizer_arguments(parser)
    parser.add_argument("--output", default=env_str("OUTPUT", ""))

    parser.add_argument(
        "--kernel-backend",
        choices=("auto", "torch", "tile-cuda", "tile_cuda"),
        default=env_str("KERNEL_BACKEND", GPT2_EVO_DEFAULTS["kernel_backend"]),
        help="Kernel backend for the compiled graph. Defaults to tile-cuda.",
    )
    parser.add_argument(
        "--tile-cuda-strict",
        action=argparse.BooleanOptionalAction,
        default=str(env_str("TILE_CUDA_STRICT", "1")).strip().lower() not in {"0", "false", "no", "off"},
        help="Fail instead of falling back when CUDA Tile kernels or tensor contracts are unavailable.",
    )
    parser.add_argument(
        "--tile-cuda-report",
        default=env_str("TILE_CUDA_REPORT", ""),
        help="Optional path for a JSON CUDA Tile kernel coverage report.",
    )
    parser.add_argument(
        "--tile-cuda-activation-dtype",
        choices=("nvfp4", "float32", "none"),
        default=env_str("TILE_CUDA_ACTIVATION_DTYPE", GPT2_EVO_DEFAULTS["tile_cuda_activation_dtype"]),
        help=(
            "Optional Tile CUDA activation packing for supported projection and attention kernels. "
            "Defaults to nvfp4 for the RTX 5090 harness."
        ),
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("bfloat16", "bf16", "float16", "fp16", "float32", "fp32", "none"),
        default=env_str("AMP_DTYPE", GPT2_EVO_DEFAULTS["amp_dtype"]),
        help=(
            "PyTorch autocast dtype. Defaults to bfloat16 so GPT-2-scale output "
            "projection uses tensor-core GEMM instead of the fp32 SGEMM path."
        ),
    )

    parser.add_argument(
        "--evo-layer-index",
        type=int,
        default=env_int("EVO_LAYER_INDEX", GPT2_EVO_DEFAULTS["evo_layer_index"]),
        help="Transformer block index trained by evolution instead of gradients.",
    )
    parser.add_argument(
        "--evo-layer-interval",
        type=int,
        default=env_int("EVO_LAYER_INTERVAL", GPT2_EVO_DEFAULTS["evo_layer_interval"]),
        help="Run the evo-layer search every N optimizer steps.",
    )
    parser.add_argument(
        "--evo-layer-population",
        type=int,
        default=env_int("EVO_LAYER_POPULATION", GPT2_EVO_DEFAULTS["evo_layer_population"]),
        help="Population size per evo-layer search (current weights are candidate 0).",
    )
    parser.add_argument(
        "--evo-layer-mutation-scale",
        type=float,
        default=env_float("EVO_LAYER_MUTATION_SCALE", GPT2_EVO_DEFAULTS["evo_layer_mutation_scale"]),
        help="Stddev of the gaussian mutation applied to evo-layer candidates.",
    )
    parser.add_argument(
        "--evo-layer-seed",
        type=int,
        default=None,
        help="Optional RNG seed override for the evo-layer search (defaults to --seed).",
    )
    parser.add_argument(
        "--no-layer-evo",
        action="store_true",
        help="Disable the evo layer (pure-gradient ablation baseline).",
    )

    add_max_steps_argument(parser, default=env_int("ITERATIONS", GPT2_EVO_DEFAULTS["max_steps"]))
    parser.add_argument("--train-seq-len", type=int, default=env_int("TRAIN_SEQ_LEN", GPT2_EVO_DEFAULTS["train_seq_len"]))
    parser.add_argument("--batch-size", type=int, default=env_int("BATCH_SIZE", GPT2_EVO_DEFAULTS["batch_size"]))
    parser.add_argument(
        "--train-batch-tokens",
        type=int,
        default=env_int("TRAIN_BATCH_TOKENS", GPT2_EVO_DEFAULTS["train_batch_tokens"]),
    )
    parser.add_argument("--eval-batches", type=int, default=env_int("EVAL_BATCHES", GPT2_EVO_DEFAULTS["eval_batches"]))
    parser.add_argument("--eval-batch-size", type=int, default=env_int("EVAL_BATCH_SIZE", GPT2_EVO_DEFAULTS["eval_batch_size"]))
    parser.add_argument(
        "--eval-every-steps",
        type=int,
        default=env_int("EVAL_EVERY_STEPS", GPT2_EVO_DEFAULTS["eval_every_steps"]),
        help="Run live validation-loss evaluation every N optimizer steps. Use 0 to disable periodic eval.",
    )
    parser.add_argument("--train-log-every", type=int, default=env_int("TRAIN_LOG_EVERY", GPT2_EVO_DEFAULTS["train_log_every"]))
    add_max_wallclock_seconds_argument(
        parser,
        default=env_float("MAX_WALLCLOCK_SECONDS", GPT2_EVO_DEFAULTS["max_wallclock_seconds"]),
    )
    parser.add_argument("--warmup-steps", type=int, default=env_int("WARMUP_STEPS", GPT2_EVO_DEFAULTS["warmup_steps"]))
    add_warmdown_fraction_argument(
        parser,
        default=env_float("WARMDOWN_FRACTION", GPT2_EVO_DEFAULTS["warmdown_fraction"]),
    )
    add_all_train_rows_argument(parser)
    add_evolutionary_training_arguments(parser)

    parser.add_argument("--vocab-size", type=int, default=env_int("VOCAB_SIZE", GPT2_EVO_DEFAULTS["vocab_size"]))
    parser.add_argument("--num-layers", type=int, default=env_int("NUM_LAYERS", GPT2_EVO_DEFAULTS["num_layers"]))
    parser.add_argument("--model-dim", type=int, default=env_int("MODEL_DIM", GPT2_EVO_DEFAULTS["model_dim"]))
    parser.add_argument("--num-heads", type=int, default=env_int("NUM_HEADS", GPT2_EVO_DEFAULTS["num_heads"]))
    parser.add_argument("--logit-softcap", type=float, default=env_float("LOGIT_SOFTCAP", GPT2_EVO_DEFAULTS["logit_softcap"]))

    parser.add_argument("--optimizer-profile", default=env_str("OPTIMIZER_PROFILE", GPT2_EVO_DEFAULTS["optimizer_profile"]))
    parser.add_argument("--learning-rate", type=float, default=env_float("LEARNING_RATE", GPT2_EVO_DEFAULTS["learning_rate"]))
    add_lr_schedule_arguments(parser)
    parser.add_argument("--weight-decay", type=float, default=env_float("WEIGHT_DECAY", GPT2_EVO_DEFAULTS["weight_decay"]))
    parser.add_argument("--embed-lr", type=float, default=env_float("EMBED_LR", GPT2_EVO_DEFAULTS["embed_lr"]))
    parser.add_argument("--head-lr", type=float, default=env_float("HEAD_LR", GPT2_EVO_DEFAULTS["head_lr"]))
    parser.add_argument("--tied-embed-lr", type=float, default=env_float("TIED_EMBED_LR", GPT2_EVO_DEFAULTS["tied_embed_lr"]))
    parser.add_argument("--matrix-lr", type=float, default=env_float("MATRIX_LR", GPT2_EVO_DEFAULTS["matrix_lr"]))
    parser.add_argument("--scalar-lr", type=float, default=env_float("SCALAR_LR", GPT2_EVO_DEFAULTS["scalar_lr"]))
    parser.add_argument("--muon-momentum", type=float, default=env_float("MUON_MOMENTUM", GPT2_EVO_DEFAULTS["muon_momentum"]))
    parser.add_argument("--muon-backend-steps", type=int, default=env_int("MUON_BACKEND_STEPS", GPT2_EVO_DEFAULTS["muon_backend_steps"]))
    parser.add_argument(
        "--muon-momentum-warmup-start",
        type=float,
        default=env_float("MUON_MOMENTUM_WARMUP_START", GPT2_EVO_DEFAULTS["muon_momentum_warmup_start"]),
    )
    parser.add_argument(
        "--muon-momentum-warmup-steps",
        type=int,
        default=env_int("MUON_MOMENTUM_WARMUP_STEPS", GPT2_EVO_DEFAULTS["muon_momentum_warmup_steps"]),
    )
    parser.add_argument("--beta1", type=float, default=env_float("BETA1", GPT2_EVO_DEFAULTS["beta1"]))
    parser.add_argument("--beta2", type=float, default=env_float("BETA2", GPT2_EVO_DEFAULTS["beta2"]))
    parser.add_argument("--adam-eps", type=float, default=env_float("ADAM_EPS", GPT2_EVO_DEFAULTS["adam_eps"]))
    parser.add_argument("--grad-clip-norm", type=float, default=env_float("GRAD_CLIP_NORM", GPT2_EVO_DEFAULTS["grad_clip_norm"]))
    return parser


def build_graph(args: argparse.Namespace, dataset_name: str):
    evo_seed = args.evo_layer_seed if args.evo_layer_seed is not None else args.seed
    spec = build_gpt2_evo_spec(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        logit_softcap=args.logit_softcap,
        layer_evo_enabled=not bool(args.no_layer_evo),
        layer_evo_index=args.evo_layer_index,
        layer_evo_fraction=1.0 / max(int(args.evo_layer_interval), 1),
        layer_evo_population=args.evo_layer_population,
        layer_evo_mutation_scale=args.evo_layer_mutation_scale,
        layer_evo_seed=evo_seed,
    )
    graph = build_gpt_root_graph(name=GRAPH_NAME, model_spec=spec)
    activation_dtype = str(args.tile_cuda_activation_dtype).strip().lower()
    graph.torch_config = {
        **graph.torch_config,
        "device": args.device,
        "amp_dtype": args.amp_dtype,
        "tile_cuda_activation_dtype": "" if activation_dtype in {"", "none", "float32"} else activation_dtype,
    }
    load_dataset_source_into_graph(
        graph,
        LoadDatasetRequest(dataset_names=[dataset_name], seq_len=args.train_seq_len),
    )
    apply_sanitized_template_spec(
        graph,
        raw_text_encoding_name=str(getattr(args, "raw_text_encoding_name", "") or ""),
    )
    return graph, spec


def build_live_eval_fn(
    trainer: TorchTrainer,
    dataset_path: Path,
    *,
    device: str,
    seq_len: int,
    batch_size: int,
    eval_batches: int,
    encoding_name: str,
):
    loader = None

    def run_eval() -> float:
        nonlocal loader
        if eval_batches <= 0:
            return float("nan")
        compiled = trainer.active_compiled_graph or trainer.last_compiled_graph
        if compiled is None:
            raise RuntimeError("validation requested before the trainer compiled the graph")
        eval_device = torch.device(device)
        compiled.to(eval_device)
        amp_dtype, _amp_name, use_amp = resolve_amp_settings(
            (compiled.graph.torch_config or {}).get("amp_dtype", "float32")
        )
        use_amp = use_amp and eval_device.type == "cuda"
        if loader is None:
            val_dataset = load_val_token_dataset(dataset_path, seq_len=seq_len, encoding_name=encoding_name)
            loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        was_training = compiled.training
        compiled.eval()
        total_loss = 0.0
        total_rows = 0
        try:
            with torch.no_grad():
                for batch_idx, batch in enumerate(loader):
                    if batch_idx >= eval_batches:
                        break
                    if isinstance(batch, torch.Tensor):
                        flat_inputs = (batch.to(eval_device), batch.to(eval_device))
                    else:
                        values = tuple(item.to(eval_device) for item in batch)
                        flat_inputs = (values[0], values[0]) if len(values) == 1 else (values[0], values[1])
                    with torch.autocast(device_type=eval_device.type, dtype=amp_dtype, enabled=use_amp):
                        outputs = compiled(*flat_inputs)
                    loss = outputs[0]
                    batch_rows = int(flat_inputs[0].size(0))
                    total_loss += float(loss.item()) * batch_rows
                    total_rows += batch_rows
        finally:
            compiled.train(was_training)
        return total_loss / max(total_rows, 1)

    return run_eval


def wrap_on_step_with_layer_evo(base_on_step, *, eval_every_steps: int = 0, eval_fn=None):
    def on_step(step_info):
        base_on_step(step_info)
        step = int(step_info.get("step", 0))
        if (
            eval_fn is not None
            and eval_every_steps > 0
            and str(step_info.get("phase", "train")) == "train"
            and step > 0
            and step % int(eval_every_steps) == 0
        ):
            val_loss = safe_evaluate_validation_loss(eval_fn, logger=LOGGER)
            if math.isfinite(val_loss):
                LOGGER.info("Validation eval step %s: loss=%.6f", step, val_loss)
            else:
                LOGGER.info("Validation eval step %s: skipped", step)
        evo = step_info.get("layer_evo")
        if evo:
            LOGGER.info(
                "layer_evo step=%s layer=%s candidates=%s best_loss=%.6f mutation_scale=%s",
                step_info.get("step"),
                evo.get("layer_index"),
                evo.get("candidate_count"),
                float(evo.get("best_loss", float("nan"))),
                evo.get("mutation_scale"),
            )

    return on_step


def main() -> int:
    configure_console_logging()
    parser = build_parser()
    args = parser.parse_args()
    apply_tinystories_dataset_defaults(args)
    resolve_dataset_selector_args(args)
    resolve_pretraining_file_dataset(args)
    resolve_mode_defaults(args)
    interrupted_weights_path = INTERRUPTED_ARTIFACT
    interrupted_graph_path = interrupted_weights_path.with_suffix(".json")

    log_stage(f"Starting {MODE_NAME} harness run {args.run_id}")
    log_stage(f"CLI started at {datetime.now().isoformat(timespec='seconds')}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    log_stage(f"Seeds set to {args.seed}")

    if args.device != "cuda":
        print("This harness is configured to run on CUDA only.", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("CUDA device is not available in this environment.", file=sys.stderr)
        return 1
    log_stage("CUDA is available; resolving datasets and graph configuration")

    log_stage(f"Resolving dataset alias {args.dataset_alias}")
    dataset_name, dataset_path, dataset_meta = resolve_or_download_dataset(
        args.dataset_alias,
        raw_text_encoding_name=str(getattr(args, "raw_text_encoding_name", "gpt2")),
        **dataset_download_kwargs_from_args(args),
    )
    dataset_meta = apply_cached_tokenizer_vocab_policy(
        args,
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        dataset_meta=dataset_meta,
        default_vocab_size=int(GPT2_EVO_DEFAULTS["vocab_size"]),
    )
    log_stage("Estimating training schedule from cached dataset")
    derived = estimate_text_schedule(
        dataset_name,
        seq_len=args.train_seq_len,
        batch_size=args.batch_size,
        train_batch_tokens=args.train_batch_tokens,
        template_runtime="eager",
        device=args.device,
        all_train_rows=bool(args.all_train_rows),
        encoding_name=str(getattr(args, "raw_text_encoding_name", "gpt2")),
    )
    derived = {**derived}
    (
        derived,
        resolved_epochs,
        resolved_max_steps,
        resolved_lr_decay_iters,
        resolved_max_wallclock_seconds,
    ) = resolve_effective_training_schedule(
        args,
        derived,
    )
    trainer_cfg = build_trainer_config(
        args,
        resolved_epochs=resolved_epochs,
        max_steps=resolved_max_steps,
        lr_decay_iters=resolved_lr_decay_iters,
        max_wallclock_seconds=resolved_max_wallclock_seconds,
        drop_last=bool(derived["drop_last"]),
        respect_epoch_boundaries=bool(derived["respect_epoch_boundaries"]),
    )
    trainer_cfg.kernel_backend = str(args.kernel_backend)
    trainer_cfg.tile_cuda_strict = bool(args.tile_cuda_strict)
    trainer_cfg.tile_cuda_report_path = str(args.tile_cuda_report or "") or None

    log_stage(f"Building {MODE_NAME} graph")
    graph, spec = build_graph(args, dataset_name)
    print(f"Using dataset: {dataset_name}")
    print_graph_summary(graph)
    print_data_source_summary(dataset_name, dataset_path, dataset_meta, graph)
    resolved_training_summary = print_resolved_summary(args, spec, trainer_cfg, derived)
    if args.no_layer_evo:
        print("Evo layer: disabled (--no-layer-evo)")
    else:
        print(
            f"Evo layer: block_{args.evo_layer_index} "
            f"(every {args.evo_layer_interval} steps, population {args.evo_layer_population}, "
            f"mutation scale {args.evo_layer_mutation_scale})"
        )
    print(f"Kernel backend: {args.kernel_backend} (strict={bool(args.tile_cuda_strict)})")
    print(f"Tile CUDA activation dtype: {args.tile_cuda_activation_dtype} (AMP={args.amp_dtype})")
    if int(args.eval_every_steps) > 0:
        print(f"Periodic validation eval: every {int(args.eval_every_steps)} optimizer steps")

    log_stage("Initializing TorchTrainer")
    trainer = TorchTrainer(graph, trainer_cfg)
    base_on_step, on_epoch = build_progress_logger(
        train_log_every=args.train_log_every,
        resolved_epochs=resolved_epochs,
        max_steps=resolved_max_steps,
    )
    validation_eval_fn = build_live_eval_fn(
        trainer,
        dataset_path,
        device=args.device,
        seq_len=args.train_seq_len,
        batch_size=args.eval_batch_size,
        eval_batches=args.eval_batches,
        encoding_name=str(getattr(args, "raw_text_encoding_name", "gpt2")),
    )
    on_step = wrap_on_step_with_layer_evo(
        base_on_step,
        eval_every_steps=int(args.eval_every_steps),
        eval_fn=validation_eval_fn if int(args.eval_every_steps) > 0 else None,
    )
    interrupted = False
    force_abort = False
    previous_sigint = signal.getsignal(signal.SIGINT)

    def handle_sigint(signum, frame):
        nonlocal interrupted, force_abort
        if interrupted:
            force_abort = True
            raise KeyboardInterrupt
        interrupted = True
        log_stage("Interrupt received. Stopping after the current safe boundary.")
        trainer.stop()

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        log_stage(
            "Training about to start: "
            f"max_steps={resolved_max_steps}, warmup_steps={args.warmup_steps}, "
            f"train_log_every={args.train_log_every}, grad_accum_steps={derived['grad_accum_steps']}, "
            f"steps_per_epoch={derived['steps_per_epoch']}"
        )
        losses = trainer.train([], [], on_epoch=on_epoch, on_step=on_step)
        output_path = Path(args.output)
        graph_output_path = output_path.with_suffix(".json")

        if interrupted:
            log_stage("Training stopped after interrupt; saving interrupted artifacts")
            save_artifacts(
                graph,
                interrupted_weights_path,
                interrupted_graph_path,
                training_manifest=resolved_training_summary,
                dataset_name=dataset_name,
                dataset_path=dataset_path,
                dataset_meta=dataset_meta,
                raw_text_encoding_name=str(getattr(args, "raw_text_encoding_name", "gpt2")),
            )
            print("Interrupted by user.")
            print("Saved checkpoint:")
            print(f"  weights: {interrupted_weights_path}")
            print(f"  graph:   {interrupted_graph_path}")
            print("Losses:", [round(float(loss), 6) for loss in losses])
            return 130

        if not losses:
            print("Trainer returned no losses.", file=sys.stderr)
            return 1
        if not all(math.isfinite(float(loss)) for loss in losses):
            print("Encountered non-finite loss", file=sys.stderr)
            return 1

        log_stage("Training finished. Starting validation pass")
        val_loss = safe_evaluate_validation_loss(validation_eval_fn, logger=LOGGER)
        log_stage("Validation finished. Saving exported artifacts")
        save_artifacts(
            graph,
            output_path,
            graph_output_path,
            training_manifest=resolved_training_summary,
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            dataset_meta=dataset_meta,
            raw_text_encoding_name=str(getattr(args, "raw_text_encoding_name", "gpt2")),
        )
        log_stage("Artifacts saved.")
        print("Losses:", [round(float(loss), 6) for loss in losses])
        print(f"Final train loss: {float(losses[-1]):.6f}")
        if math.isfinite(val_loss):
            print(f"Validation loss: {val_loss:.6f}")
        else:
            print("Validation loss: skipped")
        print(f"Exported model: {output_path}")
        print(f"Exported graph: {graph_output_path}")
        print("Training completed successfully.")
        log_stage("Run completed successfully")
        return 0
    except KeyboardInterrupt:
        trainer.stop()
        log_stage("Saving interrupted artifacts after keyboard interrupt")
        save_artifacts(
            graph,
            interrupted_weights_path,
            interrupted_graph_path,
            training_manifest=resolved_training_summary,
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            dataset_meta=dataset_meta,
            raw_text_encoding_name=str(getattr(args, "raw_text_encoding_name", "gpt2")),
        )
        if force_abort:
            print("\nForced abort received.")
        else:
            print("\nInterrupted by user.")
        print("Saved checkpoint:")
        print(f"  weights: {interrupted_weights_path}")
        print(f"  graph:   {interrupted_graph_path}")
        return 130
    finally:
        signal.signal(signal.SIGINT, previous_sigint)


if __name__ == "__main__":
    raise SystemExit(main())
