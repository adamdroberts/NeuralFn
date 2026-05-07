from __future__ import annotations

import argparse
from datetime import datetime
import math
from pathlib import Path
import signal
import sys
import uuid

import numpy as np
import torch

from cli_utils import artifact_path, create_argument_parser
from neuralfn import TorchTrainConfig, TorchTrainer
from neuralfn.config import build_llama_fast_megakernel_spec, build_llama_megakernel_spec
from neuralfn.torch_templates import build_gpt_root_graph
from server.models import LoadDatasetRequest
from server.services.graph_ops import load_dataset_source_into_graph

from train_jepa_semantic import (
    DEFAULT_DATASET_ALIAS,
    add_all_train_rows_argument,
    apply_cached_tokenizer_vocab_policy,
    apply_sanitized_template_spec,
    add_evolutionary_training_arguments,
    add_warmdown_fraction_argument,
    add_max_wallclock_seconds_argument,
    add_max_steps_argument,
    add_pretraining_file_argument,
    add_raw_text_tokenizer_arguments,
    add_dataset_download_arguments,
    add_dataset_selector_arguments,
    add_lr_schedule_arguments,
    apply_raw_text_tokenizer_policy,
    apply_tinystories_dataset_defaults,
    dataset_download_kwargs_from_args,
    estimate_text_schedule,
    print_graph_summary,
    resolve_effective_training_schedule,
    resolve_lr_schedule_defaults,
    resolve_or_download_dataset,
    resolve_pretraining_file_dataset,
    resolve_dataset_selector_args,
    save_artifacts,
)
from train_llama_fast import (
    LLAMA_DEFAULTS,
    build_progress_logger,
    configure_console_logging,
    env_float,
    env_int,
    env_str,
    evaluate_model,
    print_data_source_summary,
    print_resolved_summary,
    safe_evaluate_validation_loss,
)


def mode_name(*, fast: bool) -> str:
    return "llama_fast_megakernel" if fast else "llama_megakernel"


def default_output_path(*, fast: bool) -> Path:
    return artifact_path(f"{mode_name(fast=fast)}.pt")


def interrupted_output_path(*, fast: bool) -> Path:
    return default_output_path(fast=fast).with_name(f"{mode_name(fast=fast)}.interrupted.pt")


def resolve_mode_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if not getattr(args, "output", ""):
        args.output = str(default_output_path(fast=bool(args.fast)))
    resolve_lr_schedule_defaults(args)
    apply_raw_text_tokenizer_policy(
        args,
        preset_name=mode_name(fast=bool(args.fast)),
        default_vocab_size=int(LLAMA_DEFAULTS["vocab_size"]),
    )
    return args


def build_parser() -> argparse.ArgumentParser:
    parser = create_argument_parser(description="Train llama_megakernel with the NeuralFn CUDA harness.")
    parser.add_argument("--fast", action="store_true", help="Use the llama_fast_megakernel preset.")
    parser.add_argument("--run-id", default=env_str("RUN_ID", str(uuid.uuid4())))
    parser.add_argument("--seed", type=int, default=env_int("SEED", LLAMA_DEFAULTS["seed"]))
    parser.add_argument("--device", default=env_str("DEVICE", LLAMA_DEFAULTS["device"]))
    add_dataset_selector_arguments(
        parser,
        default_alias=env_str("DATASET_ALIAS", DEFAULT_DATASET_ALIAS),
    )
    add_dataset_download_arguments(parser)
    add_pretraining_file_argument(parser)
    add_raw_text_tokenizer_arguments(parser)
    parser.add_argument("--output", default=env_str("OUTPUT", ""))

    add_max_steps_argument(parser, default=env_int("ITERATIONS", LLAMA_DEFAULTS["max_steps"]))
    parser.add_argument("--train-seq-len", type=int, default=env_int("TRAIN_SEQ_LEN", LLAMA_DEFAULTS["train_seq_len"]))
    parser.add_argument("--batch-size", type=int, default=env_int("BATCH_SIZE", LLAMA_DEFAULTS["batch_size"]))
    parser.add_argument(
        "--train-batch-tokens",
        type=int,
        default=env_int("TRAIN_BATCH_TOKENS", LLAMA_DEFAULTS["train_batch_tokens"]),
    )
    parser.add_argument("--eval-batches", type=int, default=env_int("EVAL_BATCHES", LLAMA_DEFAULTS["eval_batches"]))
    parser.add_argument("--eval-batch-size", type=int, default=env_int("EVAL_BATCH_SIZE", LLAMA_DEFAULTS["eval_batch_size"]))
    parser.add_argument("--train-log-every", type=int, default=env_int("TRAIN_LOG_EVERY", LLAMA_DEFAULTS["train_log_every"]))
    add_max_wallclock_seconds_argument(
        parser,
        default=env_float("MAX_WALLCLOCK_SECONDS", LLAMA_DEFAULTS["max_wallclock_seconds"]),
    )
    parser.add_argument("--warmup-steps", type=int, default=env_int("WARMUP_STEPS", LLAMA_DEFAULTS["warmup_steps"]))
    add_warmdown_fraction_argument(
        parser,
        default=env_float("WARMDOWN_FRACTION", LLAMA_DEFAULTS["warmdown_fraction"]),
    )
    add_all_train_rows_argument(parser)
    add_evolutionary_training_arguments(parser)

    parser.add_argument("--vocab-size", type=int, default=env_int("VOCAB_SIZE", LLAMA_DEFAULTS["vocab_size"]))
    parser.add_argument("--num-layers", type=int, default=env_int("NUM_LAYERS", LLAMA_DEFAULTS["num_layers"]))
    parser.add_argument("--model-dim", type=int, default=env_int("MODEL_DIM", LLAMA_DEFAULTS["model_dim"]))
    parser.add_argument("--num-heads", type=int, default=env_int("NUM_HEADS", LLAMA_DEFAULTS["num_heads"]))
    parser.add_argument("--num-kv-heads", type=int, default=env_int("NUM_KV_HEADS", LLAMA_DEFAULTS["num_kv_heads"]))
    parser.add_argument("--mlp-mult", type=float, default=env_float("MLP_MULT", LLAMA_DEFAULTS["mlp_mult"]))
    parser.add_argument("--multiple-of", type=int, default=env_int("MULTIPLE_OF", LLAMA_DEFAULTS["multiple_of"]))
    parser.add_argument("--rope-base", type=float, default=env_float("ROPE_BASE", LLAMA_DEFAULTS["rope_base"]))
    parser.add_argument("--qk-gain-init", type=float, default=env_float("QK_GAIN_INIT", LLAMA_DEFAULTS["qk_gain_init"]))
    parser.add_argument("--logit-softcap", type=float, default=env_float("LOGIT_SOFTCAP", LLAMA_DEFAULTS["logit_softcap"]))

    parser.add_argument("--optimizer-profile", default=env_str("OPTIMIZER_PROFILE", LLAMA_DEFAULTS["optimizer_profile"]))
    parser.add_argument("--learning-rate", type=float, default=env_float("LEARNING_RATE", LLAMA_DEFAULTS["learning_rate"]))
    add_lr_schedule_arguments(parser)
    parser.add_argument("--weight-decay", type=float, default=env_float("WEIGHT_DECAY", LLAMA_DEFAULTS["weight_decay"]))
    parser.add_argument("--embed-lr", type=float, default=env_float("EMBED_LR", LLAMA_DEFAULTS["embed_lr"]))
    parser.add_argument("--head-lr", type=float, default=env_float("HEAD_LR", LLAMA_DEFAULTS["head_lr"]))
    parser.add_argument("--tied-embed-lr", type=float, default=env_float("TIED_EMBED_LR", LLAMA_DEFAULTS["tied_embed_lr"]))
    parser.add_argument("--matrix-lr", type=float, default=env_float("MATRIX_LR", LLAMA_DEFAULTS["matrix_lr"]))
    parser.add_argument("--scalar-lr", type=float, default=env_float("SCALAR_LR", LLAMA_DEFAULTS["scalar_lr"]))
    parser.add_argument("--muon-momentum", type=float, default=env_float("MUON_MOMENTUM", LLAMA_DEFAULTS["muon_momentum"]))
    parser.add_argument("--muon-backend-steps", type=int, default=env_int("MUON_BACKEND_STEPS", LLAMA_DEFAULTS["muon_backend_steps"]))
    parser.add_argument(
        "--muon-momentum-warmup-start",
        type=float,
        default=env_float("MUON_MOMENTUM_WARMUP_START", LLAMA_DEFAULTS["muon_momentum_warmup_start"]),
    )
    parser.add_argument(
        "--muon-momentum-warmup-steps",
        type=int,
        default=env_int("MUON_MOMENTUM_WARMUP_STEPS", LLAMA_DEFAULTS["muon_momentum_warmup_steps"]),
    )
    parser.add_argument("--beta1", type=float, default=env_float("BETA1", LLAMA_DEFAULTS["beta1"]))
    parser.add_argument("--beta2", type=float, default=env_float("BETA2", LLAMA_DEFAULTS["beta2"]))
    parser.add_argument("--adam-eps", type=float, default=env_float("ADAM_EPS", LLAMA_DEFAULTS["adam_eps"]))
    parser.add_argument("--grad-clip-norm", type=float, default=env_float("GRAD_CLIP_NORM", LLAMA_DEFAULTS["grad_clip_norm"]))
    return parser


def build_graph(args: argparse.Namespace, dataset_name: str):
    builder = build_llama_fast_megakernel_spec if args.fast else build_llama_megakernel_spec
    spec = builder(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_multiplier=args.mlp_mult,
        multiple_of=args.multiple_of,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        logit_softcap=args.logit_softcap,
    )
    graph = build_gpt_root_graph(name=f"{mode_name(fast=bool(args.fast))}_sdk", model_spec=spec)
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


def main() -> int:
    configure_console_logging()
    parser = build_parser()
    args = parser.parse_args()
    apply_tinystories_dataset_defaults(args)
    resolve_dataset_selector_args(args)
    resolve_pretraining_file_dataset(args)
    resolve_mode_defaults(args)

    run_label = mode_name(fast=bool(args.fast))
    interrupted_weights_path = interrupted_output_path(fast=bool(args.fast))
    interrupted_graph_path = interrupted_weights_path.with_suffix(".json")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting {run_label} harness run {args.run_id}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] CLI started at {datetime.now().isoformat(timespec='seconds')}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Seeds set to {args.seed}")

    if args.device != "cuda":
        print("This harness is configured to run on CUDA only.", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("CUDA device is not available in this environment.", file=sys.stderr)
        return 1
    print(f"[{datetime.now().strftime('%H:%M:%S')}] CUDA is available; resolving datasets and graph configuration")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Resolving dataset alias {args.dataset_alias}")
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
        default_vocab_size=int(LLAMA_DEFAULTS["vocab_size"]),
    )
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Estimating training schedule from cached dataset")
    derived = estimate_text_schedule(
        dataset_name,
        seq_len=args.train_seq_len,
        batch_size=args.batch_size,
        train_batch_tokens=args.train_batch_tokens,
        template_runtime="megakernel",
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

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Building {run_label} graph")
    graph, spec = build_graph(args, dataset_name)
    print(f"Using dataset: {dataset_name}")
    print_graph_summary(graph)
    print_data_source_summary(dataset_name, dataset_path, dataset_meta, graph)
    resolved_training_summary = print_resolved_summary(args, spec, trainer_cfg, derived)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Initializing TorchTrainer")
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
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Interrupt received. Stopping after the current safe boundary.")
        trainer.stop()

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] Training about to start: "
            f"max_steps={resolved_max_steps}, warmup_steps={args.warmup_steps}, "
            f"train_log_every={args.train_log_every}, grad_accum_steps={derived['grad_accum_steps']}, "
            f"steps_per_epoch={derived['steps_per_epoch']}"
        )
        losses = trainer.train([], [], on_epoch=on_epoch, on_step=on_step)
        output_path = Path(args.output)
        graph_output_path = output_path.with_suffix(".json")

        if interrupted:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Training stopped after interrupt; saving interrupted artifacts")
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

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Training finished. Saving exported artifacts")
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
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Artifacts saved. Starting validation pass")
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
        )
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Validation finished.")
        print("Losses:", [round(float(loss), 6) for loss in losses])
        print(f"Final train loss: {float(losses[-1]):.6f}")
        if math.isfinite(val_loss):
            print(f"Validation loss: {val_loss:.6f}")
        else:
            print("Validation loss: skipped")
        print(f"Exported model: {output_path}")
        print(f"Exported graph: {graph_output_path}")
        print("Training completed successfully.")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Run completed successfully")
        return 0
    except KeyboardInterrupt:
        trainer.stop()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Saving interrupted artifacts after keyboard interrupt")
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
