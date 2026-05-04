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

import numpy as np
import torch

from cli_utils import artifact_path, create_argument_parser
from neuralfn import TorchTrainConfig, TorchTrainer
from neuralfn.config import build_mixllama_fast_megakernel_spec, build_mixllama_fast_spec, model_spec_to_dict
from neuralfn.torch_backend import CompiledTorchGraph
from neuralfn.torch_templates import build_gpt_root_graph
from server.models import LoadDatasetRequest
from server.services.graph_ops import load_dataset_source_into_graph

from train_jepa_semantic import (
    DEFAULT_DATASET_ALIAS,
    add_all_train_rows_argument,
    apply_cached_tokenizer_vocab_policy,
    apply_sanitized_template_spec,
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
    dataset_tokenizer_summary_lines,
    dataset_download_kwargs_from_args,
    estimate_text_schedule,
    format_routing_stats_suffix,
    load_val_token_dataset,
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

DEFAULT_ARTIFACT = artifact_path("mixllama_fast.pt")
DEFAULT_GRAPH_ARTIFACT = DEFAULT_ARTIFACT.with_suffix(".json")
INTERRUPTED_ARTIFACT = DEFAULT_ARTIFACT.with_name("mixllama_fast.interrupted.pt")
INTERRUPTED_GRAPH_ARTIFACT = INTERRUPTED_ARTIFACT.with_suffix(".json")

MIXLLAMA_DEFAULTS = {
    "seed": 1337,
    "device": "cuda",
    "run_id": str(uuid.uuid4()),
    "dataset_alias": DEFAULT_DATASET_ALIAS,
    "output": str(DEFAULT_ARTIFACT),
    "max_steps": 400,
    "train_seq_len": 128,
    "batch_size": 8,
    "train_batch_tokens": 8_192,
    "eval_batches": 8,
    "eval_batch_size": 8,
    "train_log_every": 1,
    "val_loss_every": 200,
    "max_wallclock_seconds": 900.0,
    "warmup_steps": 8,
    "warmdown_fraction": 0.75,
    "vocab_size": 1_024,
    "num_layers": 4,
    "model_dim": 256,
    "num_heads": 4,
    "num_kv_heads": 4,
    "mlp_mult": 2.0,
    "multiple_of": 64,
    "experts": 8,
    "top_k": 2,
    "rope_base": 10_000.0,
    "qk_gain_init": 1.5,
    "logit_softcap": 30.0,
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

LOGGER = logging.getLogger("mixllama_fast_harness")


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
    return "mixllama_fast_megakernel" if megakernel else "mixllama_fast"


def default_output_path(*, megakernel: bool) -> Path:
    if megakernel:
        return DEFAULT_ARTIFACT.with_name("mixllama_fast_megakernel.pt")
    return DEFAULT_ARTIFACT


def interrupted_output_path(*, megakernel: bool) -> Path:
    if megakernel:
        return DEFAULT_ARTIFACT.with_name("mixllama_fast_megakernel.interrupted.pt")
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
        default_vocab_size=int(MIXLLAMA_DEFAULTS["vocab_size"]),
    )
    return args


def build_parser() -> argparse.ArgumentParser:
    parser = create_argument_parser(description="Train mixllama_fast with the NeuralFn CUDA harness.")
    parser.add_argument("--megakernel", action="store_true", help="Use the mixllama_fast_megakernel preset/runtime.")
    parser.add_argument("--run-id", default=env_str("RUN_ID", MIXLLAMA_DEFAULTS["run_id"]))
    parser.add_argument("--seed", type=int, default=env_int("SEED", MIXLLAMA_DEFAULTS["seed"]))
    parser.add_argument("--device", default=env_str("DEVICE", MIXLLAMA_DEFAULTS["device"]))
    add_dataset_selector_arguments(
        parser,
        default_alias=env_str("DATASET_ALIAS", MIXLLAMA_DEFAULTS["dataset_alias"]),
    )
    add_dataset_download_arguments(parser)
    add_pretraining_file_argument(parser)
    add_raw_text_tokenizer_arguments(parser)
    parser.add_argument("--output", default=env_str("OUTPUT", ""))

    add_max_steps_argument(parser, default=env_int("ITERATIONS", MIXLLAMA_DEFAULTS["max_steps"]))
    parser.add_argument("--train-seq-len", type=int, default=env_int("TRAIN_SEQ_LEN", MIXLLAMA_DEFAULTS["train_seq_len"]))
    parser.add_argument("--batch-size", type=int, default=env_int("BATCH_SIZE", MIXLLAMA_DEFAULTS["batch_size"]))
    parser.add_argument(
        "--train-batch-tokens",
        type=int,
        default=env_int("TRAIN_BATCH_TOKENS", MIXLLAMA_DEFAULTS["train_batch_tokens"]),
    )
    parser.add_argument("--eval-batches", type=int, default=env_int("EVAL_BATCHES", MIXLLAMA_DEFAULTS["eval_batches"]))
    parser.add_argument("--eval-batch-size", type=int, default=env_int("EVAL_BATCH_SIZE", MIXLLAMA_DEFAULTS["eval_batch_size"]))
    parser.add_argument("--train-log-every", type=int, default=env_int("TRAIN_LOG_EVERY", MIXLLAMA_DEFAULTS["train_log_every"]))
    add_max_wallclock_seconds_argument(
        parser,
        default=env_float("MAX_WALLCLOCK_SECONDS", MIXLLAMA_DEFAULTS["max_wallclock_seconds"]),
    )
    parser.add_argument("--warmup-steps", type=int, default=env_int("WARMUP_STEPS", MIXLLAMA_DEFAULTS["warmup_steps"]))
    add_warmdown_fraction_argument(
        parser,
        default=env_float("WARMDOWN_FRACTION", MIXLLAMA_DEFAULTS["warmdown_fraction"]),
    )
    add_all_train_rows_argument(parser)
    add_evolutionary_training_arguments(parser)

    parser.add_argument("--vocab-size", type=int, default=env_int("VOCAB_SIZE", MIXLLAMA_DEFAULTS["vocab_size"]))
    parser.add_argument("--num-layers", type=int, default=env_int("NUM_LAYERS", MIXLLAMA_DEFAULTS["num_layers"]))
    parser.add_argument("--model-dim", type=int, default=env_int("MODEL_DIM", MIXLLAMA_DEFAULTS["model_dim"]))
    parser.add_argument("--num-heads", type=int, default=env_int("NUM_HEADS", MIXLLAMA_DEFAULTS["num_heads"]))
    parser.add_argument("--num-kv-heads", type=int, default=env_int("NUM_KV_HEADS", MIXLLAMA_DEFAULTS["num_kv_heads"]))
    parser.add_argument("--mlp-mult", type=float, default=env_float("MLP_MULT", MIXLLAMA_DEFAULTS["mlp_mult"]))
    parser.add_argument("--multiple-of", type=int, default=env_int("MULTIPLE_OF", MIXLLAMA_DEFAULTS["multiple_of"]))
    parser.add_argument("--experts", type=int, default=env_int("EXPERTS", MIXLLAMA_DEFAULTS["experts"]))
    parser.add_argument("--top-k", type=int, default=env_int("TOP_K", MIXLLAMA_DEFAULTS["top_k"]))
    parser.add_argument("--rope-base", type=float, default=env_float("ROPE_BASE", MIXLLAMA_DEFAULTS["rope_base"]))
    parser.add_argument("--qk-gain-init", type=float, default=env_float("QK_GAIN_INIT", MIXLLAMA_DEFAULTS["qk_gain_init"]))
    parser.add_argument("--logit-softcap", type=float, default=env_float("LOGIT_SOFTCAP", MIXLLAMA_DEFAULTS["logit_softcap"]))

    parser.add_argument("--optimizer-profile", default=env_str("OPTIMIZER_PROFILE", MIXLLAMA_DEFAULTS["optimizer_profile"]))
    parser.add_argument("--learning-rate", type=float, default=env_float("LEARNING_RATE", MIXLLAMA_DEFAULTS["learning_rate"]))
    add_lr_schedule_arguments(parser)
    parser.add_argument("--weight-decay", type=float, default=env_float("WEIGHT_DECAY", MIXLLAMA_DEFAULTS["weight_decay"]))
    parser.add_argument("--embed-lr", type=float, default=env_float("EMBED_LR", MIXLLAMA_DEFAULTS["embed_lr"]))
    parser.add_argument("--head-lr", type=float, default=env_float("HEAD_LR", MIXLLAMA_DEFAULTS["head_lr"]))
    parser.add_argument("--tied-embed-lr", type=float, default=env_float("TIED_EMBED_LR", MIXLLAMA_DEFAULTS["tied_embed_lr"]))
    parser.add_argument("--matrix-lr", type=float, default=env_float("MATRIX_LR", MIXLLAMA_DEFAULTS["matrix_lr"]))
    parser.add_argument("--scalar-lr", type=float, default=env_float("SCALAR_LR", MIXLLAMA_DEFAULTS["scalar_lr"]))
    parser.add_argument("--muon-momentum", type=float, default=env_float("MUON_MOMENTUM", MIXLLAMA_DEFAULTS["muon_momentum"]))
    parser.add_argument("--muon-backend-steps", type=int, default=env_int("MUON_BACKEND_STEPS", MIXLLAMA_DEFAULTS["muon_backend_steps"]))
    parser.add_argument(
        "--muon-momentum-warmup-start",
        type=float,
        default=env_float("MUON_MOMENTUM_WARMUP_START", MIXLLAMA_DEFAULTS["muon_momentum_warmup_start"]),
    )
    parser.add_argument(
        "--muon-momentum-warmup-steps",
        type=int,
        default=env_int("MUON_MOMENTUM_WARMUP_STEPS", MIXLLAMA_DEFAULTS["muon_momentum_warmup_steps"]),
    )
    parser.add_argument("--beta1", type=float, default=env_float("BETA1", MIXLLAMA_DEFAULTS["beta1"]))
    parser.add_argument("--beta2", type=float, default=env_float("BETA2", MIXLLAMA_DEFAULTS["beta2"]))
    parser.add_argument("--adam-eps", type=float, default=env_float("ADAM_EPS", MIXLLAMA_DEFAULTS["adam_eps"]))
    parser.add_argument("--grad-clip-norm", type=float, default=env_float("GRAD_CLIP_NORM", MIXLLAMA_DEFAULTS["grad_clip_norm"]))
    return parser


def build_graph(args: argparse.Namespace, dataset_name: str):
    builder = build_mixllama_fast_megakernel_spec if args.megakernel else build_mixllama_fast_spec
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
        grad_clip_norm=args.grad_clip_norm,
        drop_last=resolved_drop_last,
        respect_epoch_boundaries=(
            bool(getattr(args, "all_train_rows", False))
            if respect_epoch_boundaries is None
            else bool(respect_epoch_boundaries)
        ),
        evolutionary=bool(getattr(args, "evolutionary", False)),
        evo_population_size=int(args.evo_population_size),
        evo_mutation_rate=float(args.evo_mutation_rate),
        evo_mutation_scale=float(args.evo_mutation_scale),
        evo_crossover_rate=float(args.evo_crossover_rate),
        evo_tournament_size=int(args.evo_tournament_size),
        evo_elite_count=int(args.evo_elite_count),
        evo_seed=int(args.seed) if args.evo_seed is None else int(args.evo_seed),
    )


def print_data_source_summary(dataset_name: str, dataset_path: Path, dataset_meta: dict[str, object], graph) -> None:
    val_files = sorted(dataset_path.glob("fineweb_val_*.bin")) if dataset_path.is_dir() else []
    print("Text data source:")
    print(f"  - Local dataset name: {dataset_name}")
    print(f"  - Local storage path: {dataset_path}")
    if dataset_meta:
        print(f"  - Source: {dataset_meta.get('source')}")
        print(f"  - HF path: {dataset_meta.get('hf_path')}")
        print(f"  - Split: {dataset_meta.get('hf_split')}")
        print(f"  - Variant: {dataset_meta.get('variant')}")
        for line in dataset_tokenizer_summary_lines(dataset_meta):
            print(line)
        print(f"  - Validation shards: {dataset_meta.get('val_shards')}")
    if val_files:
        print(f"  - Validation shard path: {val_files[0]}")
    ds_cfg = dict(graph.nodes["dataset_source"].neuron_def.module_config or {})
    print("dataset_source config:", ds_cfg)


def evaluate_model(
    graph,
    dataset_path: Path,
    *,
    device: str,
    seq_len: int,
    batch_size: int,
    eval_batches: int,
    encoding_name: str = "gpt2",
) -> float:
    if eval_batches <= 0:
        return float("nan")

    val_dataset = load_val_token_dataset(dataset_path, seq_len=seq_len, encoding_name=encoding_name)
    loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    compiled = CompiledTorchGraph(graph)
    compiled.to(device)
    compiled.eval()

    amp_dtype = torch.float32
    use_amp = False
    total_loss = 0.0
    total_rows = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= eval_batches:
                break
            if isinstance(batch, torch.Tensor):
                flat_inputs = (batch.to(device), batch.to(device))
                batch_rows = int(batch.size(0))
            else:
                values = tuple(item.to(device) for item in batch)
                if len(values) == 1:
                    flat_inputs = (values[0], values[0])
                else:
                    flat_inputs = (values[0], values[1])
                batch_rows = int(flat_inputs[0].size(0))
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                outputs = compiled(*flat_inputs)
                loss = outputs[0]
            total_loss += float(loss.item()) * batch_rows
            total_rows += batch_rows
    return total_loss / max(total_rows, 1)


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
        "graph_contract": ["tokens", "targets"],
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
            semantic_labels=False,
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
        default_vocab_size=int(MIXLLAMA_DEFAULTS["vocab_size"]),
    )
    log_stage("Estimating training schedule from cached dataset")
    derived = estimate_text_schedule(
        dataset_name,
        seq_len=args.train_seq_len,
        batch_size=args.batch_size,
        train_batch_tokens=args.train_batch_tokens,
        template_runtime="megakernel" if args.megakernel else "compile",
        device=args.device,
        all_train_rows=bool(args.all_train_rows),
    )
    derived = {
        **derived,
    }
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
