from __future__ import annotations

import argparse
from datetime import datetime
import logging
import math
from pathlib import Path
import signal
import sys
import time
import uuid
from typing import Any

import numpy as np
import torch

from cli_utils import artifact_path, create_argument_parser
from neuralfn import TorchTrainer
from neuralfn.config import build_gpt2_megakernel_spec, build_gpt2_spec
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
    evaluate_model,
    print_data_source_summary,
    print_resolved_summary,
    safe_evaluate_validation_loss,
)

DEFAULT_ARTIFACT = artifact_path("gpt2.pt")
DEFAULT_GRAPH_ARTIFACT = DEFAULT_ARTIFACT.with_suffix(".json")
INTERRUPTED_ARTIFACT = DEFAULT_ARTIFACT.with_name("gpt2.interrupted.pt")
INTERRUPTED_GRAPH_ARTIFACT = INTERRUPTED_ARTIFACT.with_suffix(".json")

GPT2_DEFAULTS = {
    "seed": 1337,
    "device": "cuda",
    "run_id": str(uuid.uuid4()),
    "dataset_alias": DEFAULT_DATASET_ALIAS,
    "output": str(DEFAULT_ARTIFACT),
    "max_steps": 400,
    "train_seq_len": 192,
    "batch_size": 8,
    "train_batch_tokens": 24_576,
    "eval_batches": 8,
    "eval_batch_size": 8,
    "train_log_every": 1,
    "max_wallclock_seconds": 900.0,
    "warmup_steps": 8,
    "warmdown_fraction": 0.75,
    "vocab_size": 1_024,
    "num_layers": 5,
    "model_dim": 320,
    "num_heads": 5,
    "logit_softcap": 0.0,
    "optimizer_profile": "parameter_golf",
    "learning_rate": 3e-4,
    "weight_decay": 0.0,
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
}

LOGGER = logging.getLogger("gpt2_harness")


def log_stage(message: str) -> None:
    LOGGER.info(message)


def mode_name(*, megakernel: bool) -> str:
    return "gpt2_megakernel" if megakernel else "gpt2"


def default_output_path(*, megakernel: bool) -> Path:
    if megakernel:
        return DEFAULT_ARTIFACT.with_name("gpt2_megakernel.pt")
    return DEFAULT_ARTIFACT


def interrupted_output_path(*, megakernel: bool) -> Path:
    if megakernel:
        return DEFAULT_ARTIFACT.with_name("gpt2_megakernel.interrupted.pt")
    return INTERRUPTED_ARTIFACT


def graph_name(*, megakernel: bool) -> str:
    return f"{mode_name(megakernel=megakernel)}_sdk"


def resolve_mode_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if not getattr(args, "output", ""):
        args.output = str(default_output_path(megakernel=bool(args.megakernel)))
    resolve_lr_schedule_defaults(args)
    apply_raw_text_tokenizer_policy(
        args,
        preset_name=mode_name(megakernel=bool(args.megakernel)),
        default_vocab_size=int(GPT2_DEFAULTS["vocab_size"]),
    )
    return args


def build_parser() -> argparse.ArgumentParser:
    parser = create_argument_parser(description="Train gpt2 with the NeuralFn CUDA harness.")
    parser.add_argument("--megakernel", action="store_true", help="Use the gpt2_megakernel preset/runtime.")
    parser.add_argument("--run-id", default=env_str("RUN_ID", GPT2_DEFAULTS["run_id"]))
    parser.add_argument("--seed", type=int, default=env_int("SEED", GPT2_DEFAULTS["seed"]))
    parser.add_argument("--device", default=env_str("DEVICE", GPT2_DEFAULTS["device"]))
    add_dataset_selector_arguments(
        parser,
        default_alias=env_str("DATASET_ALIAS", GPT2_DEFAULTS["dataset_alias"]),
    )
    add_dataset_download_arguments(parser)
    add_pretraining_file_argument(parser)
    add_raw_text_tokenizer_arguments(parser)
    parser.add_argument("--output", default=env_str("OUTPUT", ""))

    add_max_steps_argument(parser, default=env_int("ITERATIONS", GPT2_DEFAULTS["max_steps"]))
    parser.add_argument("--train-seq-len", type=int, default=env_int("TRAIN_SEQ_LEN", GPT2_DEFAULTS["train_seq_len"]))
    parser.add_argument("--batch-size", type=int, default=env_int("BATCH_SIZE", GPT2_DEFAULTS["batch_size"]))
    parser.add_argument(
        "--train-batch-tokens",
        type=int,
        default=env_int("TRAIN_BATCH_TOKENS", GPT2_DEFAULTS["train_batch_tokens"]),
    )
    parser.add_argument("--eval-batches", type=int, default=env_int("EVAL_BATCHES", GPT2_DEFAULTS["eval_batches"]))
    parser.add_argument("--eval-batch-size", type=int, default=env_int("EVAL_BATCH_SIZE", GPT2_DEFAULTS["eval_batch_size"]))
    parser.add_argument("--train-log-every", type=int, default=env_int("TRAIN_LOG_EVERY", GPT2_DEFAULTS["train_log_every"]))
    add_max_wallclock_seconds_argument(
        parser,
        default=env_float("MAX_WALLCLOCK_SECONDS", GPT2_DEFAULTS["max_wallclock_seconds"]),
    )
    parser.add_argument("--warmup-steps", type=int, default=env_int("WARMUP_STEPS", GPT2_DEFAULTS["warmup_steps"]))
    add_warmdown_fraction_argument(
        parser,
        default=env_float("WARMDOWN_FRACTION", GPT2_DEFAULTS["warmdown_fraction"]),
    )
    add_all_train_rows_argument(parser)
    add_evolutionary_training_arguments(parser)

    parser.add_argument("--vocab-size", type=int, default=env_int("VOCAB_SIZE", GPT2_DEFAULTS["vocab_size"]))
    parser.add_argument("--num-layers", type=int, default=env_int("NUM_LAYERS", GPT2_DEFAULTS["num_layers"]))
    parser.add_argument("--model-dim", type=int, default=env_int("MODEL_DIM", GPT2_DEFAULTS["model_dim"]))
    parser.add_argument("--num-heads", type=int, default=env_int("NUM_HEADS", GPT2_DEFAULTS["num_heads"]))
    parser.add_argument("--logit-softcap", type=float, default=env_float("LOGIT_SOFTCAP", GPT2_DEFAULTS["logit_softcap"]))

    parser.add_argument("--optimizer-profile", default=env_str("OPTIMIZER_PROFILE", GPT2_DEFAULTS["optimizer_profile"]))
    parser.add_argument("--learning-rate", type=float, default=env_float("LEARNING_RATE", GPT2_DEFAULTS["learning_rate"]))
    add_lr_schedule_arguments(parser)
    parser.add_argument("--weight-decay", type=float, default=env_float("WEIGHT_DECAY", GPT2_DEFAULTS["weight_decay"]))
    parser.add_argument("--embed-lr", type=float, default=env_float("EMBED_LR", GPT2_DEFAULTS["embed_lr"]))
    parser.add_argument("--head-lr", type=float, default=env_float("HEAD_LR", GPT2_DEFAULTS["head_lr"]))
    parser.add_argument("--tied-embed-lr", type=float, default=env_float("TIED_EMBED_LR", GPT2_DEFAULTS["tied_embed_lr"]))
    parser.add_argument("--matrix-lr", type=float, default=env_float("MATRIX_LR", GPT2_DEFAULTS["matrix_lr"]))
    parser.add_argument("--scalar-lr", type=float, default=env_float("SCALAR_LR", GPT2_DEFAULTS["scalar_lr"]))
    parser.add_argument("--muon-momentum", type=float, default=env_float("MUON_MOMENTUM", GPT2_DEFAULTS["muon_momentum"]))
    parser.add_argument("--muon-backend-steps", type=int, default=env_int("MUON_BACKEND_STEPS", GPT2_DEFAULTS["muon_backend_steps"]))
    parser.add_argument(
        "--muon-momentum-warmup-start",
        type=float,
        default=env_float("MUON_MOMENTUM_WARMUP_START", GPT2_DEFAULTS["muon_momentum_warmup_start"]),
    )
    parser.add_argument(
        "--muon-momentum-warmup-steps",
        type=int,
        default=env_int("MUON_MOMENTUM_WARMUP_STEPS", GPT2_DEFAULTS["muon_momentum_warmup_steps"]),
    )
    parser.add_argument("--beta1", type=float, default=env_float("BETA1", GPT2_DEFAULTS["beta1"]))
    parser.add_argument("--beta2", type=float, default=env_float("BETA2", GPT2_DEFAULTS["beta2"]))
    parser.add_argument("--adam-eps", type=float, default=env_float("ADAM_EPS", GPT2_DEFAULTS["adam_eps"]))
    parser.add_argument("--grad-clip-norm", type=float, default=env_float("GRAD_CLIP_NORM", GPT2_DEFAULTS["grad_clip_norm"]))
    return parser


def build_graph(args: argparse.Namespace, dataset_name: str):
    builder = build_gpt2_megakernel_spec if args.megakernel else build_gpt2_spec
    spec = builder(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        logit_softcap=args.logit_softcap,
    )
    graph = build_gpt_root_graph(name=graph_name(megakernel=bool(args.megakernel)), model_spec=spec)
    graph.torch_config = {**graph.torch_config, "device": args.device, "amp_dtype": "float32"}
    load_dataset_source_into_graph(
        graph,
        LoadDatasetRequest(dataset_names=[dataset_name], seq_len=args.train_seq_len),
    )
    apply_sanitized_template_spec(
        graph,
        raw_text_encoding_name=str(getattr(args, "raw_text_encoding_name", "") or ""),
    )
    return graph, spec


def main() -> int:
    configure_console_logging()
    parser = build_parser()
    args = parser.parse_args()
    apply_tinystories_dataset_defaults(args)
    resolve_dataset_selector_args(args)
    resolve_pretraining_file_dataset(args)
    resolve_mode_defaults(args)
    run_label = mode_name(megakernel=bool(args.megakernel))
    interrupted_weights_path = interrupted_output_path(megakernel=bool(args.megakernel))
    interrupted_graph_path = interrupted_weights_path.with_suffix(".json")

    log_stage(f"Starting {run_label} harness run {args.run_id}")
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
        default_vocab_size=int(GPT2_DEFAULTS["vocab_size"]),
    )
    log_stage("Estimating training schedule from cached dataset")
    derived = estimate_text_schedule(
        dataset_name,
        seq_len=args.train_seq_len,
        batch_size=args.batch_size,
        train_batch_tokens=args.train_batch_tokens,
        template_runtime="megakernel" if args.megakernel else "eager",
        device=args.device,
        all_train_rows=bool(args.all_train_rows),
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

    log_stage(f"Building {run_label} graph")
    graph, spec = build_graph(args, dataset_name)
    print(f"Using dataset: {dataset_name}")
    print_graph_summary(graph)
    print_data_source_summary(dataset_name, dataset_path, dataset_meta, graph)
    resolved_training_summary = print_resolved_summary(args, spec, trainer_cfg, derived)

    log_stage("Initializing TorchTrainer")
    trainer = TorchTrainer(graph, trainer_cfg)
    on_step, on_epoch = build_progress_logger(
        train_log_every=args.train_log_every,
        resolved_epochs=resolved_epochs,
        max_steps=resolved_max_steps,
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

        log_stage("Training finished. Saving exported artifacts")
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
        log_stage("Artifacts saved. Starting validation pass")
        val_loss = safe_evaluate_validation_loss(
            lambda: evaluate_model(
                graph,
                dataset_path,
                device=args.device,
                seq_len=args.train_seq_len,
                batch_size=args.eval_batch_size,
                eval_batches=args.eval_batches,
                encoding_name=str(getattr(args, "raw_text_encoding_name", "gpt2")),
            ),
            logger=LOGGER,
        )
        log_stage("Validation finished.")
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
