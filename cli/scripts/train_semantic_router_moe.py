from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
import math
import os
from pathlib import Path
import signal
import sys
import time
import uuid
from typing import Any, Callable

from native_training_guard import reject_torch_training_by_default

if __name__ == "__main__":
    reject_torch_training_by_default("train_semantic_router_moe.py", native_target="nfn train --base-model semantic-router-moe", model_family="semantic-router-moe")

import numpy as np
import torch

from cli_utils import artifact_path, create_argument_parser
import neuralfn.semantic as semantic_module
from neuralfn import TorchTrainConfig, TorchTrainer
from neuralfn.config import build_semantic_router_moe_megakernel_spec, build_semantic_router_moe_spec, model_spec_to_dict
from neuralfn.torch_templates import build_gpt_root_graph

from train_jepa_semantic import (
    DEFAULT_DATASET_ALIAS,
    add_all_train_rows_argument,
    apply_sanitized_template_spec,
    apply_cached_tokenizer_vocab_policy,
    add_evolutionary_training_arguments,
    add_max_wallclock_seconds_argument,
    add_max_steps_argument,
    add_warmdown_fraction_argument,
    add_pretraining_file_argument,
    add_raw_text_tokenizer_arguments,
    add_dataset_download_arguments,
    add_dataset_selector_arguments,
    add_lr_schedule_arguments,
    apply_raw_text_tokenizer_policy,
    apply_tinystories_dataset_defaults,
    build_trainer_summary,
    estimate_schedule,
    dataset_download_kwargs_from_args,
    evaluate_model,
    format_routing_stats_suffix,
    print_data_source_summary,
    print_graph_summary,
    resolve_effective_training_schedule,
    resolve_lr_schedule_defaults,
    resolve_or_download_dataset,
    resolve_pretraining_file_dataset,
    resolve_dataset_selector_args,
    save_artifacts,
    safe_evaluate_validation_loss,
    sanitized_model_spec_dict,
)

DEFAULT_ARTIFACT = artifact_path("semantic_router_moe.pt")
DEFAULT_GRAPH_ARTIFACT = DEFAULT_ARTIFACT.with_suffix(".json")
INTERRUPTED_ARTIFACT = DEFAULT_ARTIFACT.with_name("semantic_router_moe.interrupted.pt")
INTERRUPTED_GRAPH_ARTIFACT = INTERRUPTED_ARTIFACT.with_suffix(".json")

ROUTER_DEFAULTS = {
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
    "train_log_every": 1,
    "val_loss_every": 250,
    "max_wallclock_seconds": 0.0,
    "warmup_steps": 60,
    "warmdown_fraction": 0.0,
    "vocab_size": 1_024,
    "num_layers": 4,
    "model_dim": 256,
    "num_heads": 4,
    "num_kv_heads": 4,
    "mlp_mult": 2.0,
    "multiple_of": 64,
    "experts": semantic_module.NUM_VOCAB_DIMS,
    "top_k": 2,
    "rope_base": 10_000.0,
    "qk_gain_init": 1.5,
    "logit_softcap": 30.0,
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
    "ar_loss_coef": 1.0,
    "semantic_align_loss_coef": 0.15,
}

LOGGER = logging.getLogger("semantic_router_moe_harness")


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def configure_console_logging() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def log_stage(message: str) -> None:
    LOGGER.info(message)


def format_elapsed(seconds: float) -> str:
    total = max(int(seconds), 0)
    mins, secs = divmod(total, 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{hours:d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def mode_name(*, megakernel: bool) -> str:
    return "semantic_router_moe_megakernel" if megakernel else "semantic_router_moe"


def default_output_path(*, megakernel: bool) -> Path:
    if megakernel:
        return DEFAULT_ARTIFACT.with_name("semantic_router_moe_megakernel.pt")
    return DEFAULT_ARTIFACT


def interrupted_output_path(*, megakernel: bool) -> Path:
    if megakernel:
        return DEFAULT_ARTIFACT.with_name("semantic_router_moe_megakernel.interrupted.pt")
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
        default_vocab_size=int(ROUTER_DEFAULTS["vocab_size"]),
    )
    return args


def build_parser() -> argparse.ArgumentParser:
    parser = create_argument_parser(description="Train semantic_router_moe with the NeuralFn CUDA harness.")
    parser.add_argument(
        "--megakernel",
        action="store_true",
        help="Use the semantic_router_moe_megakernel preset/runtime.",
    )
    parser.add_argument("--run-id", default=env_str("RUN_ID", ROUTER_DEFAULTS["run_id"]))
    parser.add_argument("--seed", type=int, default=env_int("SEED", ROUTER_DEFAULTS["seed"]))
    parser.add_argument("--device", default=env_str("DEVICE", ROUTER_DEFAULTS["device"]))
    add_dataset_selector_arguments(
        parser,
        default_alias=env_str("DATASET_ALIAS", ROUTER_DEFAULTS["dataset_alias"]),
    )
    add_dataset_download_arguments(parser)
    add_pretraining_file_argument(parser)
    add_raw_text_tokenizer_arguments(parser)
    parser.add_argument("--output", default=env_str("OUTPUT", ""))

    add_max_steps_argument(parser, default=env_int("ITERATIONS", ROUTER_DEFAULTS["max_steps"]))
    parser.add_argument("--train-seq-len", type=int, default=env_int("TRAIN_SEQ_LEN", ROUTER_DEFAULTS["train_seq_len"]))
    parser.add_argument("--batch-size", type=int, default=env_int("BATCH_SIZE", ROUTER_DEFAULTS["batch_size"]))
    parser.add_argument(
        "--train-batch-tokens",
        type=int,
        default=env_int("TRAIN_BATCH_TOKENS", ROUTER_DEFAULTS["train_batch_tokens"]),
    )
    parser.add_argument("--eval-batches", type=int, default=env_int("EVAL_BATCHES", ROUTER_DEFAULTS["eval_batches"]))
    parser.add_argument("--eval-batch-size", type=int, default=env_int("EVAL_BATCH_SIZE", ROUTER_DEFAULTS["eval_batch_size"]))
    parser.add_argument("--train-log-every", type=int, default=env_int("TRAIN_LOG_EVERY", ROUTER_DEFAULTS["train_log_every"]))
    parser.add_argument("--val-loss-every", type=int, default=env_int("VAL_LOSS_EVERY", ROUTER_DEFAULTS["val_loss_every"]))
    add_max_wallclock_seconds_argument(
        parser,
        default=env_float("MAX_WALLCLOCK_SECONDS", ROUTER_DEFAULTS["max_wallclock_seconds"]),
    )
    parser.add_argument("--warmup-steps", type=int, default=env_int("WARMUP_STEPS", ROUTER_DEFAULTS["warmup_steps"]))
    add_warmdown_fraction_argument(
        parser,
        default=env_float("WARMDOWN_FRACTION", ROUTER_DEFAULTS["warmdown_fraction"]),
    )
    add_all_train_rows_argument(parser)
    add_evolutionary_training_arguments(parser)

    parser.add_argument("--vocab-size", type=int, default=env_int("VOCAB_SIZE", ROUTER_DEFAULTS["vocab_size"]))
    parser.add_argument("--num-layers", type=int, default=env_int("NUM_LAYERS", ROUTER_DEFAULTS["num_layers"]))
    parser.add_argument("--model-dim", type=int, default=env_int("MODEL_DIM", ROUTER_DEFAULTS["model_dim"]))
    parser.add_argument("--num-heads", type=int, default=env_int("NUM_HEADS", ROUTER_DEFAULTS["num_heads"]))
    parser.add_argument("--num-kv-heads", type=int, default=env_int("NUM_KV_HEADS", ROUTER_DEFAULTS["num_kv_heads"]))
    parser.add_argument("--mlp-mult", type=float, default=env_float("MLP_MULT", ROUTER_DEFAULTS["mlp_mult"]))
    parser.add_argument("--multiple-of", type=int, default=env_int("MULTIPLE_OF", ROUTER_DEFAULTS["multiple_of"]))
    parser.add_argument("--experts", type=int, default=env_int("EXPERTS", ROUTER_DEFAULTS["experts"]))
    parser.add_argument("--top-k", type=int, default=env_int("TOP_K", ROUTER_DEFAULTS["top_k"]))
    parser.add_argument(
        "--experimental-semantic-router-vecs",
        action="store_true",
        help="Add semantic_router_vecs to the graph contract and route directly from the normalized semantic router vector.",
    )
    parser.add_argument("--rope-base", type=float, default=env_float("ROPE_BASE", ROUTER_DEFAULTS["rope_base"]))
    parser.add_argument("--qk-gain-init", type=float, default=env_float("QK_GAIN_INIT", ROUTER_DEFAULTS["qk_gain_init"]))
    parser.add_argument("--logit-softcap", type=float, default=env_float("LOGIT_SOFTCAP", ROUTER_DEFAULTS["logit_softcap"]))

    parser.add_argument("--optimizer-profile", default=env_str("OPTIMIZER_PROFILE", ROUTER_DEFAULTS["optimizer_profile"]))
    parser.add_argument("--learning-rate", type=float, default=env_float("LEARNING_RATE", ROUTER_DEFAULTS["learning_rate"]))
    add_lr_schedule_arguments(parser)
    parser.add_argument("--weight-decay", type=float, default=env_float("WEIGHT_DECAY", ROUTER_DEFAULTS["weight_decay"]))
    parser.add_argument("--embed-lr", type=float, default=env_float("EMBED_LR", ROUTER_DEFAULTS["embed_lr"]))
    parser.add_argument("--head-lr", type=float, default=env_float("HEAD_LR", ROUTER_DEFAULTS["head_lr"]))
    parser.add_argument("--tied-embed-lr", type=float, default=env_float("TIED_EMBED_LR", ROUTER_DEFAULTS["tied_embed_lr"]))
    parser.add_argument("--matrix-lr", type=float, default=env_float("MATRIX_LR", ROUTER_DEFAULTS["matrix_lr"]))
    parser.add_argument("--scalar-lr", type=float, default=env_float("SCALAR_LR", ROUTER_DEFAULTS["scalar_lr"]))
    parser.add_argument("--muon-momentum", type=float, default=env_float("MUON_MOMENTUM", ROUTER_DEFAULTS["muon_momentum"]))
    parser.add_argument("--muon-backend-steps", type=int, default=env_int("MUON_BACKEND_STEPS", ROUTER_DEFAULTS["muon_backend_steps"]))
    parser.add_argument(
        "--muon-momentum-warmup-start",
        type=float,
        default=env_float("MUON_MOMENTUM_WARMUP_START", ROUTER_DEFAULTS["muon_momentum_warmup_start"]),
    )
    parser.add_argument(
        "--muon-momentum-warmup-steps",
        type=int,
        default=env_int("MUON_MOMENTUM_WARMUP_STEPS", ROUTER_DEFAULTS["muon_momentum_warmup_steps"]),
    )
    parser.add_argument("--beta1", type=float, default=env_float("BETA1", ROUTER_DEFAULTS["beta1"]))
    parser.add_argument("--beta2", type=float, default=env_float("BETA2", ROUTER_DEFAULTS["beta2"]))
    parser.add_argument("--adam-eps", type=float, default=env_float("ADAM_EPS", ROUTER_DEFAULTS["adam_eps"]))
    parser.add_argument("--grad-clip-norm", type=float, default=env_float("GRAD_CLIP_NORM", ROUTER_DEFAULTS["grad_clip_norm"]))

    parser.add_argument("--ar-loss-coef", type=float, default=env_float("AR_LOSS_COEF", ROUTER_DEFAULTS["ar_loss_coef"]))
    parser.add_argument(
        "--semantic-align-loss-coef",
        type=float,
        default=env_float("SEMANTIC_ALIGN_LOSS_COEF", ROUTER_DEFAULTS["semantic_align_loss_coef"]),
    )
    return parser


def build_graph(args: argparse.Namespace, dataset_name: str):
    builder = build_semantic_router_moe_megakernel_spec if args.megakernel else build_semantic_router_moe_spec
    spec = builder(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_multiplier=args.mlp_mult,
        multiple_of=args.multiple_of,
        experts=args.experts,
        top_k=args.top_k,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        logit_softcap=args.logit_softcap,
        semantic_vocab_ref=semantic_module.DEFAULT_SEMANTIC_VOCAB_REF,
        experimental_semantic_router_vecs=bool(args.experimental_semantic_router_vecs),
        ar_loss_coef=args.ar_loss_coef,
        semantic_align_loss_coef=args.semantic_align_loss_coef,
    )
    graph = build_gpt_root_graph(name=graph_name(megakernel=bool(args.megakernel)), model_spec=spec)
    graph.torch_config = {**graph.torch_config, "device": args.device, "amp_dtype": "float32"}

    ds_node = graph.nodes["dataset_source"]
    ds_cfg = dict(ds_node.neuron_def.module_config or {})
    ds_node.neuron_def.module_config = {
        **ds_cfg,
        "dataset_names": [dataset_name],
        "seq_len": args.train_seq_len,
    }
    apply_sanitized_template_spec(
        graph,
        raw_text_encoding_name=str(getattr(args, "raw_text_encoding_name", "") or ""),
    )
    return graph, spec


def build_trainer_config(
    args: argparse.Namespace,
    *,
    resolved_epochs: int,
    drop_last: bool | None = None,
    max_steps: int | None = None,
    lr_decay_iters: int | None = None,
    max_wallclock_seconds: float | None = None,
    respect_epoch_boundaries: bool | None = None,
) -> TorchTrainConfig:
    resolved_drop_last = drop_last
    if resolved_drop_last is None and bool(getattr(args, "all_train_rows", False)):
        resolved_drop_last = False
    return TorchTrainConfig(
        epochs=resolved_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
        max_steps=args.max_steps if max_steps is None else max_steps,
        optimizer_profile=args.optimizer_profile,
        train_batch_tokens=args.train_batch_tokens,
        warmup_steps=args.warmup_steps,
        warmdown_fraction=args.warmdown_fraction,
        lr_decay_iters=args.lr_decay_iters if lr_decay_iters is None else lr_decay_iters,
        min_lr=args.min_lr,
        max_wallclock_seconds=(
            args.max_wallclock_seconds if max_wallclock_seconds is None else max_wallclock_seconds
        ),
        embed_lr=args.embed_lr,
        head_lr=args.head_lr,
        tied_embed_lr=args.tied_embed_lr,
        matrix_lr=args.matrix_lr,
        scalar_lr=args.scalar_lr,
        muon_momentum=args.muon_momentum,
        muon_backend_steps=args.muon_backend_steps,
        muon_momentum_warmup_start=args.muon_momentum_warmup_start,
        muon_momentum_warmup_steps=args.muon_momentum_warmup_steps,
        beta1=args.beta1,
        beta2=args.beta2,
        adam_eps=args.adam_eps,
        evolutionary=bool(getattr(args, "evolutionary", False)),
        evo_population_size=int(args.evo_population_size),
        evo_mutation_rate=float(args.evo_mutation_rate),
        evo_mutation_scale=float(args.evo_mutation_scale),
        evo_crossover_rate=float(args.evo_crossover_rate),
        evo_tournament_size=int(args.evo_tournament_size),
        evo_elite_count=int(args.evo_elite_count),
        evo_seed=int(args.seed) if args.evo_seed is None else int(args.evo_seed),
        grad_clip_norm=args.grad_clip_norm,
        drop_last=resolved_drop_last,
        respect_epoch_boundaries=(
            bool(getattr(args, "all_train_rows", False))
            if respect_epoch_boundaries is None
            else bool(respect_epoch_boundaries)
        ),
    )


def print_resolved_summary(
    args: argparse.Namespace,
    spec,
    trainer_cfg: TorchTrainConfig,
    derived: dict[str, int | bool],
) -> dict[str, Any]:
    summary = {
        "run_id": args.run_id,
        "seed": args.seed,
        "device": args.device,
        "dataset_alias": args.dataset_alias,
        "artifact_path": args.output,
        "graph_contract": ["tokens", "targets", "sem_targets"],
        "model_spec": sanitized_model_spec_dict(
            spec,
            raw_text_encoding_name=str(getattr(args, "raw_text_encoding_name", "") or ""),
        ),
        "trainer": build_trainer_summary(trainer_cfg),
        "derived_schedule": derived,
    }
    print("Resolved training configuration:")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def build_progress_logger(
    *,
    train_log_every: int,
    resolved_epochs: int,
    max_steps: int,
) -> tuple[Callable[[dict[str, Any]], None], Callable[[int, float], None]]:
    interval = max(int(train_log_every), 1)
    run_start = time.perf_counter()

    def on_step(info: dict[str, Any]) -> None:
        phase = str(info.get("phase", "train"))
        elapsed = format_elapsed(float(info.get("elapsed_seconds", time.perf_counter() - run_start)))
        routing_suffix = format_routing_stats_suffix(
            info.get("routing_stats"),
            semantic_labels=True,
        )
        actual_grad_accum_steps = int(info.get("actual_grad_accum_steps", info.get("grad_accum_steps", 0)))
        grad_accum_suffix = ""
        if actual_grad_accum_steps and actual_grad_accum_steps != int(info.get("grad_accum_steps", 0)):
            grad_accum_suffix = f" actual_grad_accum_steps={actual_grad_accum_steps}"
        if phase == "warmup":
            log_stage(
                "Warmup step "
                f"{int(info.get('step', 0))}/{int(info.get('warmup_steps', 0))} "
                f"loss={float(info.get('loss', float('nan'))):.6f} "
                f"elapsed={elapsed}{routing_suffix}"
            )
            return

        step = int(info.get("step", 0))
        should_log = (
            step == 1
            or step % interval == 0
            or step >= max_steps
            or int(info.get("epoch_step", 0)) >= int(info.get("steps_per_epoch", 0))
        )
        if not should_log:
            return
        lrs = [float(lr) for lr in info.get("learning_rates", [])]
        lr_preview = ", ".join(f"{lr:.4g}" for lr in lrs[:3])
        if len(lrs) > 3:
            lr_preview += ", ..."
        lr_text = f" lr=[{lr_preview}]" if lr_preview else ""
        log_stage(
            "Train step "
            f"{step}/{int(info.get('max_steps', max_steps))} "
            f"(epoch {int(info.get('epoch', 0))}/{int(info.get('max_epochs', resolved_epochs))}, "
            f"epoch-step {int(info.get('epoch_step', 0))}/{int(info.get('steps_per_epoch', 0))}) "
            f"loss={float(info.get('loss', float('nan'))):.6f} "
            f"elapsed={elapsed}{lr_text}{grad_accum_suffix}{routing_suffix}"
        )

    def on_epoch(epoch_idx: int, loss: float) -> None:
        elapsed = format_elapsed(time.perf_counter() - run_start)
        log_stage(
            f"Epoch {epoch_idx + 1}/{resolved_epochs} complete: avg_loss={float(loss):.6f} elapsed={elapsed}"
        )

    return on_step, on_epoch


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
        default_vocab_size=int(ROUTER_DEFAULTS["vocab_size"]),
    )
    log_stage("Estimating training schedule from cached dataset and semantic rows")
    derived = estimate_schedule(
        dataset_name,
        seq_len=args.train_seq_len,
        batch_size=args.batch_size,
        train_batch_tokens=args.train_batch_tokens,
        top_k=args.top_k,
        template_runtime="megakernel" if args.megakernel else "compile",
        device=args.device,
        all_train_rows=bool(args.all_train_rows),
    )
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
