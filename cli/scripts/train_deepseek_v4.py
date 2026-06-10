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
from neuralfn.config import build_deepseek_v4_spec
from neuralfn.torch_backend import CompiledTorchGraph
from neuralfn.torch_templates import build_gpt_root_graph
from server.models import LoadDatasetRequest
from server.services.graph_ops import load_dataset_source_into_graph

from train_jepa_semantic import (
    DEFAULT_DATASET_ALIAS,
    add_all_train_rows_argument,
    add_dataset_download_arguments,
    add_dataset_selector_arguments,
    add_evolutionary_training_arguments,
    add_lr_schedule_arguments,
    add_max_steps_argument,
    add_max_wallclock_seconds_argument,
    add_pretraining_file_argument,
    add_raw_text_tokenizer_arguments,
    add_warmdown_fraction_argument,
    apply_cached_tokenizer_vocab_policy,
    apply_raw_text_tokenizer_policy,
    apply_sanitized_template_spec,
    apply_tinystories_dataset_defaults,
    build_trainer_summary,
    dataset_download_kwargs_from_args,
    dataset_tokenizer_summary_lines,
    estimate_text_schedule,
    format_routing_stats_suffix,
    load_val_token_dataset,
    print_graph_summary,
    resolve_dataset_selector_args,
    resolve_effective_training_schedule,
    resolve_lr_schedule_defaults,
    resolve_or_download_dataset,
    resolve_pretraining_file_dataset,
    safe_evaluate_validation_loss,
    sanitized_model_spec_dict,
    save_artifacts,
)

DEFAULT_ARTIFACT = artifact_path("deepseek_v4.pt")

# GPT-2 BPE smoke defaults sized for a fast CUDA sanity run on the dict dataset.
DSV4_DEFAULTS = {
    "seed": 1337,
    "device": "cuda",
    "run_id": str(uuid.uuid4()),
    "dataset_alias": DEFAULT_DATASET_ALIAS,
    "output": str(DEFAULT_ARTIFACT),
    "max_steps": 10,
    "train_seq_len": 256,
    "batch_size": 4,
    "train_batch_tokens": 4_096,
    "eval_batches": 2,
    "eval_batch_size": 4,
    "train_log_every": 1,
    "val_loss_every": 200,
    "max_wallclock_seconds": 900.0,
    "warmup_steps": 2,
    "warmdown_fraction": 0.5,
    "vocab_size": 50_304,  # GPT-2 BPE (50257) padded to a multiple of 64
    "num_layers": 2,
    "model_dim": 128,
    "num_heads": 4,
    "num_kv_heads": 2,
    "mlp_mult": 2.0,
    "multiple_of": 64,
    "experts": 4,
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

LOGGER = logging.getLogger("deepseek_v4_harness")


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def configure_console_logging() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
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


def build_parser() -> argparse.ArgumentParser:
    parser = create_argument_parser(description="Train the deepseek_v4 capstone preset on the NeuralFn CUDA harness.")
    parser.add_argument("--run-id", default=env_str("RUN_ID", DSV4_DEFAULTS["run_id"]))
    parser.add_argument("--seed", type=int, default=env_int("SEED", DSV4_DEFAULTS["seed"]))
    parser.add_argument("--device", default=env_str("DEVICE", DSV4_DEFAULTS["device"]))
    add_dataset_selector_arguments(parser, default_alias=env_str("DATASET_ALIAS", DSV4_DEFAULTS["dataset_alias"]))
    add_dataset_download_arguments(parser)
    add_pretraining_file_argument(parser)
    add_raw_text_tokenizer_arguments(parser)
    parser.add_argument("--output", default=env_str("OUTPUT", ""))

    add_max_steps_argument(parser, default=env_int("ITERATIONS", DSV4_DEFAULTS["max_steps"]))
    parser.add_argument("--train-seq-len", type=int, default=env_int("TRAIN_SEQ_LEN", DSV4_DEFAULTS["train_seq_len"]))
    parser.add_argument("--batch-size", type=int, default=env_int("BATCH_SIZE", DSV4_DEFAULTS["batch_size"]))
    parser.add_argument("--train-batch-tokens", type=int, default=env_int("TRAIN_BATCH_TOKENS", DSV4_DEFAULTS["train_batch_tokens"]))
    parser.add_argument("--eval-batches", type=int, default=env_int("EVAL_BATCHES", DSV4_DEFAULTS["eval_batches"]))
    parser.add_argument("--eval-batch-size", type=int, default=env_int("EVAL_BATCH_SIZE", DSV4_DEFAULTS["eval_batch_size"]))
    parser.add_argument("--train-log-every", type=int, default=env_int("TRAIN_LOG_EVERY", DSV4_DEFAULTS["train_log_every"]))
    add_max_wallclock_seconds_argument(parser, default=env_float("MAX_WALLCLOCK_SECONDS", DSV4_DEFAULTS["max_wallclock_seconds"]))
    parser.add_argument("--warmup-steps", type=int, default=env_int("WARMUP_STEPS", DSV4_DEFAULTS["warmup_steps"]))
    add_warmdown_fraction_argument(parser, default=env_float("WARMDOWN_FRACTION", DSV4_DEFAULTS["warmdown_fraction"]))
    add_all_train_rows_argument(parser)
    add_evolutionary_training_arguments(parser)

    parser.add_argument("--vocab-size", type=int, default=env_int("VOCAB_SIZE", DSV4_DEFAULTS["vocab_size"]))
    parser.add_argument("--num-layers", type=int, default=env_int("NUM_LAYERS", DSV4_DEFAULTS["num_layers"]))
    parser.add_argument("--model-dim", type=int, default=env_int("MODEL_DIM", DSV4_DEFAULTS["model_dim"]))
    parser.add_argument("--num-heads", type=int, default=env_int("NUM_HEADS", DSV4_DEFAULTS["num_heads"]))
    parser.add_argument("--num-kv-heads", type=int, default=env_int("NUM_KV_HEADS", DSV4_DEFAULTS["num_kv_heads"]))
    parser.add_argument("--mlp-mult", type=float, default=env_float("MLP_MULT", DSV4_DEFAULTS["mlp_mult"]))
    parser.add_argument("--multiple-of", type=int, default=env_int("MULTIPLE_OF", DSV4_DEFAULTS["multiple_of"]))
    parser.add_argument("--experts", type=int, default=env_int("EXPERTS", DSV4_DEFAULTS["experts"]))
    parser.add_argument("--top-k", type=int, default=env_int("TOP_K", DSV4_DEFAULTS["top_k"]))
    parser.add_argument("--rope-base", type=float, default=env_float("ROPE_BASE", DSV4_DEFAULTS["rope_base"]))
    parser.add_argument("--qk-gain-init", type=float, default=env_float("QK_GAIN_INIT", DSV4_DEFAULTS["qk_gain_init"]))
    parser.add_argument("--logit-softcap", type=float, default=env_float("LOGIT_SOFTCAP", DSV4_DEFAULTS["logit_softcap"]))

    parser.add_argument("--optimizer-profile", default=env_str("OPTIMIZER_PROFILE", DSV4_DEFAULTS["optimizer_profile"]))
    parser.add_argument("--learning-rate", type=float, default=env_float("LEARNING_RATE", DSV4_DEFAULTS["learning_rate"]))
    add_lr_schedule_arguments(parser)
    parser.add_argument("--weight-decay", type=float, default=env_float("WEIGHT_DECAY", DSV4_DEFAULTS["weight_decay"]))
    parser.add_argument("--embed-lr", type=float, default=env_float("EMBED_LR", DSV4_DEFAULTS["embed_lr"]))
    parser.add_argument("--head-lr", type=float, default=env_float("HEAD_LR", DSV4_DEFAULTS["head_lr"]))
    parser.add_argument("--tied-embed-lr", type=float, default=env_float("TIED_EMBED_LR", DSV4_DEFAULTS["tied_embed_lr"]))
    parser.add_argument("--matrix-lr", type=float, default=env_float("MATRIX_LR", DSV4_DEFAULTS["matrix_lr"]))
    parser.add_argument("--scalar-lr", type=float, default=env_float("SCALAR_LR", DSV4_DEFAULTS["scalar_lr"]))
    parser.add_argument("--muon-momentum", type=float, default=env_float("MUON_MOMENTUM", DSV4_DEFAULTS["muon_momentum"]))
    parser.add_argument("--muon-backend-steps", type=int, default=env_int("MUON_BACKEND_STEPS", DSV4_DEFAULTS["muon_backend_steps"]))
    parser.add_argument("--muon-momentum-warmup-start", type=float, default=env_float("MUON_MOMENTUM_WARMUP_START", DSV4_DEFAULTS["muon_momentum_warmup_start"]))
    parser.add_argument("--muon-momentum-warmup-steps", type=int, default=env_int("MUON_MOMENTUM_WARMUP_STEPS", DSV4_DEFAULTS["muon_momentum_warmup_steps"]))
    parser.add_argument("--beta1", type=float, default=env_float("BETA1", DSV4_DEFAULTS["beta1"]))
    parser.add_argument("--beta2", type=float, default=env_float("BETA2", DSV4_DEFAULTS["beta2"]))
    parser.add_argument("--adam-eps", type=float, default=env_float("ADAM_EPS", DSV4_DEFAULTS["adam_eps"]))
    parser.add_argument("--grad-clip-norm", type=float, default=env_float("GRAD_CLIP_NORM", DSV4_DEFAULTS["grad_clip_norm"]))
    return parser


def build_graph(args: argparse.Namespace, dataset_name: str):
    spec = build_deepseek_v4_spec(
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
    graph = build_gpt_root_graph(name="deepseek_v4_sdk", model_spec=spec)
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


def build_trainer_config(args: argparse.Namespace, *, resolved_epochs: int, max_steps: int,
                         lr_decay_iters: int | None, max_wallclock_seconds: float,
                         drop_last: bool, respect_epoch_boundaries: bool) -> TorchTrainConfig:
    return TorchTrainConfig(
        epochs=resolved_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
        max_steps=max_steps,
        optimizer_profile=args.optimizer_profile,
        train_batch_tokens=args.train_batch_tokens,
        warmup_steps=args.warmup_steps,
        warmdown_fraction=args.warmdown_fraction,
        lr_decay_iters=lr_decay_iters,
        min_lr=args.min_lr,
        max_wallclock_seconds=max_wallclock_seconds,
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
        drop_last=drop_last,
        respect_epoch_boundaries=respect_epoch_boundaries,
        evolutionary=bool(getattr(args, "evolutionary", False)),
        evo_population_size=int(args.evo_population_size),
        evo_mutation_rate=float(args.evo_mutation_rate),
        evo_mutation_scale=float(args.evo_mutation_scale),
        evo_crossover_rate=float(args.evo_crossover_rate),
        evo_tournament_size=int(args.evo_tournament_size),
        evo_elite_count=int(args.evo_elite_count),
        evo_seed=int(args.seed) if args.evo_seed is None else int(args.evo_seed),
    )


def main() -> int:
    configure_console_logging()
    parser = build_parser()
    args = parser.parse_args()
    apply_tinystories_dataset_defaults(args)
    resolve_dataset_selector_args(args)
    resolve_pretraining_file_dataset(args)
    if not getattr(args, "output", ""):
        args.output = str(DEFAULT_ARTIFACT)
    resolve_lr_schedule_defaults(args)
    apply_raw_text_tokenizer_policy(args, preset_name="mixllama_fast",
                                    default_vocab_size=int(DSV4_DEFAULTS["vocab_size"]))

    log_stage(f"Starting deepseek_v4 harness run {args.run_id}")
    log_stage(f"CLI started at {datetime.now().isoformat(timespec='seconds')}")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device != "cuda":
        print("This harness is configured to run on CUDA only.", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("CUDA device is not available in this environment.", file=sys.stderr)
        return 1

    dataset_name, dataset_path, dataset_meta = resolve_or_download_dataset(
        args.dataset_alias,
        raw_text_encoding_name=str(getattr(args, "raw_text_encoding_name", "gpt2")),
        **dataset_download_kwargs_from_args(args),
    )
    dataset_meta = apply_cached_tokenizer_vocab_policy(
        args, dataset_name=dataset_name, dataset_path=dataset_path,
        dataset_meta=dataset_meta, default_vocab_size=int(DSV4_DEFAULTS["vocab_size"]),
    )
    derived = estimate_text_schedule(
        dataset_name, seq_len=args.train_seq_len, batch_size=args.batch_size,
        train_batch_tokens=args.train_batch_tokens, template_runtime="eager",
        device=args.device, all_train_rows=bool(args.all_train_rows),
    )
    (derived, resolved_epochs, resolved_max_steps, resolved_lr_decay_iters,
     resolved_max_wallclock_seconds) = resolve_effective_training_schedule(args, dict(derived))

    trainer_cfg = build_trainer_config(
        args, resolved_epochs=resolved_epochs, max_steps=resolved_max_steps,
        lr_decay_iters=resolved_lr_decay_iters, max_wallclock_seconds=resolved_max_wallclock_seconds,
        drop_last=bool(derived["drop_last"]), respect_epoch_boundaries=bool(derived["respect_epoch_boundaries"]),
    )

    log_stage("Building deepseek_v4 graph")
    graph, spec = build_graph(args, dataset_name)
    print(f"Using dataset: {dataset_name}")
    print_graph_summary(graph)

    trainer = TorchTrainer(graph, trainer_cfg)
    run_start = time.perf_counter()

    def on_step(info: dict[str, Any]) -> None:
        step = int(info.get("step", 0))
        log_stage(
            f"step {step}/{int(info.get('max_steps', resolved_max_steps))} "
            f"phase={info.get('phase','train')} loss={float(info.get('loss', float('nan'))):.6f} "
            f"elapsed={format_elapsed(time.perf_counter() - run_start)}"
            f"{format_routing_stats_suffix(info.get('routing_stats'), semantic_labels=False)}"
        )

    losses = trainer.train([], [], on_step=on_step)
    if not losses or not all(math.isfinite(float(loss)) for loss in losses):
        print("Encountered no/non-finite loss", file=sys.stderr)
        return 1

    output_path = Path(args.output)
    save_artifacts(
        graph, output_path, output_path.with_suffix(".json"),
        training_manifest={"run_id": args.run_id,
                           "model_spec": sanitized_model_spec_dict(spec, raw_text_encoding_name="gpt2"),
                           "trainer": build_trainer_summary(trainer_cfg)},
        dataset_name=dataset_name, dataset_path=dataset_path,
        dataset_meta=dataset_meta, raw_text_encoding_name="gpt2",
    )
    val_loss = safe_evaluate_validation_loss(
        lambda: _evaluate(graph, dataset_path, device=args.device, seq_len=args.train_seq_len,
                          batch_size=args.eval_batch_size, eval_batches=args.eval_batches),
        logger=LOGGER,
    )
    print("Losses:", [round(float(loss), 6) for loss in losses])
    print(f"Final train loss: {float(losses[-1]):.6f}")
    print(f"Validation loss: {val_loss:.6f}" if math.isfinite(val_loss) else "Validation loss: skipped")
    print(f"Exported model: {output_path}")
    log_stage("Run completed successfully")
    return 0


def _evaluate(graph, dataset_path: Path, *, device: str, seq_len: int,
              batch_size: int, eval_batches: int) -> float:
    if eval_batches <= 0:
        return float("nan")
    val_dataset = load_val_token_dataset(dataset_path, seq_len=seq_len, encoding_name="gpt2")
    loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    compiled = CompiledTorchGraph(graph)
    compiled.to(device)
    compiled.eval()
    total_loss, total_rows = 0.0, 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= eval_batches:
                break
            if isinstance(batch, torch.Tensor):
                flat = (batch.to(device), batch.to(device))
            else:
                vals = tuple(item.to(device) for item in batch)
                flat = (vals[0], vals[0]) if len(vals) == 1 else (vals[0], vals[1])
            outputs = compiled(*flat)
            rows = int(flat[0].size(0))
            total_loss += float(outputs[0].item()) * rows
            total_rows += rows
    return total_loss / max(total_rows, 1)


if __name__ == "__main__":
    raise SystemExit(main())
