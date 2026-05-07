from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import torch

from cli_utils import artifact_path, create_argument_parser
from neuralfn.torch_backend import CompiledTorchGraph

from infer_jepa_semantic import (
    add_raw_text_tokenizer_arguments,
    add_dataset_download_arguments,
    configure_console_logging,
    dataset_download_kwargs_from_args,
    decode_tokens,
    load_compiled_inference_graph,
    log_tokenizer_status,
    repetition_penalty_arg,
    resolve_autocast_dtype,
    resolve_inference_dataset_alias,
    resolve_inference_tokenizer_context,
    resolve_raw_text_encoding_name,
    resolve_prompt_tokens,
)
from infer_llama_fast import generate_sequence
from train_jepa_semantic import (
    DEFAULT_DATASET_ALIAS,
    add_dataset_selector_arguments,
    resolve_dataset_selector_args,
    resolve_or_download_dataset,
)

LOGGER = logging.getLogger("llama_megakernel_infer")


def log_stage(message: str) -> None:
    LOGGER.info(message)


def mode_name(*, fast: bool) -> str:
    return "llama_fast_megakernel" if fast else "llama_megakernel"


def default_weights_artifact(*, fast: bool) -> Path:
    return artifact_path(f"{mode_name(fast=fast)}.pt")


def default_graph_artifact(*, fast: bool) -> Path:
    return default_weights_artifact(fast=fast).with_suffix(".json")


def resolve_mode_defaults(args: argparse.Namespace) -> argparse.Namespace:
    graph_was_explicit = bool(getattr(args, "graph", ""))
    if not graph_was_explicit:
        args.graph = str(default_graph_artifact(fast=bool(args.fast)))
    if not getattr(args, "weights", "") and not graph_was_explicit:
        args.weights = str(default_weights_artifact(fast=bool(args.fast)))
    return args


def build_parser() -> argparse.ArgumentParser:
    parser = create_argument_parser(
        description="Run text generation with exported llama_megakernel artifacts on CUDA."
    )
    parser.add_argument("--fast", action="store_true", help="Use llama_fast_megakernel artifact defaults.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--graph", default="")
    parser.add_argument("--weights", default="")
    add_dataset_selector_arguments(parser, default_alias=DEFAULT_DATASET_ALIAS)
    add_dataset_download_arguments(parser)
    add_raw_text_tokenizer_arguments(parser)
    parser.add_argument("--prompt", default="")
    parser.add_argument(
        "--prompt-tokens",
        default="",
        help="Comma-separated token ids. Overrides --prompt when provided.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument(
        "--repetition-penalty",
        type=repetition_penalty_arg,
        default=1.0,
        help="Penalty applied to tokens already seen in the prompt or generated continuation. 1.0 disables it.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--stop-token", type=int, default=None)
    parser.add_argument(
        "--logits-node",
        default="auto",
        help="Trace key to read logits from. Defaults to auto-detecting model/softcap or model/lm_head.",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=0,
        help="Optional override for the traced context window. Default: dataset_source seq_len from the graph.",
    )
    return parser


def main() -> int:
    configure_console_logging()
    args = build_parser().parse_args()
    resolve_dataset_selector_args(args)
    resolve_mode_defaults(args)

    run_label = mode_name(fast=bool(args.fast))
    log_stage(f"Starting {run_label} inference")
    if args.device != "cuda":
        print("This inference script is configured to run on CUDA only.", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("CUDA device is not available in this environment.", file=sys.stderr)
        return 1

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    generator = torch.Generator(device=device.type)
    generator.manual_seed(args.seed)
    log_stage(f"Seed set to {args.seed}")

    graph_path = Path(args.graph).expanduser().resolve()
    if not graph_path.exists():
        print(f"Graph artifact not found: {graph_path}", file=sys.stderr)
        return 1

    try:
        log_stage(f"Loading graph from {graph_path}")
        graph, compiled, state_dict, resolved_weights_path = load_compiled_inference_graph(
            graph_path=graph_path,
            weights_path=Path(args.weights).expanduser().resolve() if getattr(args, "weights", "") else None,
            device=device,
        )
        log_stage(f"Loading weights from {resolved_weights_path}")
        raw_text_encoding_name = resolve_raw_text_encoding_name(
            graph,
            encoding_override=getattr(args, "raw_text_encoding_override", None),
        )
        dataset_alias = resolve_inference_dataset_alias(
            args,
            graph,
            default_alias=DEFAULT_DATASET_ALIAS,
            log=log_stage,
        )

        amp_dtype, amp_name = resolve_autocast_dtype(graph)
        log_stage(f"Compiled graph ready on {device.type} with autocast dtype {amp_name}")

        tokenizer, tokenizer_path, tokenizer_name, dataset_name, _dataset_path, _dataset_meta = resolve_inference_tokenizer_context(
            graph=graph,
            state_dict=state_dict,
            dataset_alias=dataset_alias,
            raw_text_encoding_name=raw_text_encoding_name,
            dataset_download_kwargs=dataset_download_kwargs_from_args(args),
            require_dataset=False,
        )
        log_tokenizer_status(log_stage, tokenizer, tokenizer_path, tokenizer_name)

        log_stage("Encoding prompt")
        prompt_ids = resolve_prompt_tokens(
            prompt=args.prompt,
            prompt_tokens=args.prompt_tokens,
            tokenizer=tokenizer,
        )
        if not prompt_ids:
            print("Prompt resolved to an empty token list.", file=sys.stderr)
            return 1

        dataset_cfg = (
            dict(graph.nodes.get("dataset_source").neuron_def.module_config or {})
            if "dataset_source" in graph.nodes
            else {}
        )
        context_window = int(args.context_window or dataset_cfg.get("seq_len") or len(prompt_ids))
        resolved_dataset_name = dataset_name or dataset_alias

        log_stage(
            "Inference configuration: "
            f"dataset_alias={resolved_dataset_name}, context_window={context_window}, "
            f"max_new_tokens={args.max_new_tokens}, repetition_penalty={args.repetition_penalty}"
        )
        if len(prompt_ids) > context_window:
            log_stage(
                f"Prompt length {len(prompt_ids)} exceeds context window {context_window}; "
                "generation will use the most recent tokens."
            )

        prompt_text = decode_tokens(tokenizer, prompt_ids) if tokenizer is not None else ""
        print(f"Prompt token ids: {prompt_ids}")
        if prompt_text:
            print(f"Prompt text: {prompt_text}")

        log_stage("Generation about to start")
        result = generate_sequence(
            compiled,
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            device=device,
            amp_dtype=amp_dtype,
            generator=generator,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            stop_token=args.stop_token,
            logits_node=args.logits_node,
            context_window=context_window,
            log_every=args.log_every,
            log=log_stage,
        )

        print(f"Generated token ids: {result['generated_token_ids']}")
        print(f"All token ids: {result['all_token_ids']}")
        if tokenizer is not None:
            print("Generated text:")
            print(result["generated_text"])
            print("Full text:")
            print(result["full_text"])
        else:
            print("Decoded text unavailable because no usable tokenizer is installed.")
            print(f"Tokenizer alias: {resolved_dataset_name}")

        log_stage("Inference run completed")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
