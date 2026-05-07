from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Any, Callable

import torch

from cli_utils import artifact_path, create_argument_parser
from neuralfn.torch_backend import CompiledTorchGraph

from infer_jepa_semantic import (
    add_raw_text_tokenizer_arguments,
    add_dataset_download_arguments,
    autocast_enabled_for,
    configure_console_logging,
    dataset_download_kwargs_from_args,
    decode_tokens,
    describe_token,
    find_logits_trace_key,
    load_compiled_inference_graph,
    log_tokenizer_status,
    resolve_autocast_dtype,
    resolve_inference_dataset_alias,
    resolve_inference_tokenizer_context,
    resolve_raw_text_encoding_name,
    resolve_prompt_tokens,
    repetition_penalty_arg,
    sample_next_token,
    top_p_arg,
)
from train_jepa_semantic import add_dataset_selector_arguments, resolve_dataset_selector_args

DEFAULT_DATASET_ALIAS = "willdepueoai__parameter-golf__sp1024__train1"
DEFAULT_WEIGHTS_ARTIFACT = artifact_path("llama_fast.pt")
DEFAULT_GRAPH_ARTIFACT = DEFAULT_WEIGHTS_ARTIFACT.with_suffix(".json")

LOGGER = logging.getLogger("llama_fast_infer")


def log_stage(message: str) -> None:
    LOGGER.info(message)


def build_parser() -> argparse.ArgumentParser:
    parser = create_argument_parser(description="Run text generation with exported llama_fast artifacts on CUDA.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--graph", default=str(DEFAULT_GRAPH_ARTIFACT))
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
    parser.add_argument("--top-p", type=top_p_arg, default=1.0)
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


def generate_sequence(
    compiled: CompiledTorchGraph,
    *,
    tokenizer,
    prompt_ids: list[int],
    device: torch.device,
    amp_dtype: torch.dtype,
    generator: torch.Generator,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    repetition_penalty: float,
    top_p: float = 1.0,
    stop_token: int | None,
    logits_node: str,
    context_window: int,
    log_every: int = 0,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    generated = list(prompt_ids)
    resolved_logits_key: str | None = None
    use_amp = autocast_enabled_for(device, amp_dtype)
    with torch.no_grad():
        for step_idx in range(max_new_tokens):
            context_ids = generated[-context_window:]
            tokens = torch.tensor([context_ids], dtype=torch.long, device=device)
            targets = torch.zeros_like(tokens)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                _outputs, trace = compiled.trace(tokens, targets)

            if resolved_logits_key is None:
                resolved_logits_key = find_logits_trace_key(trace, logits_node)
                if log is not None:
                    log(f"Using traced logits node {resolved_logits_key}")

            logits = trace[resolved_logits_key][0]
            next_token = sample_next_token(
                logits[:, -1, :],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                token_history=generated,
                repetition_penalty=repetition_penalty,
                generator=generator,
            )
            generated.append(next_token)

            should_log = (
                log is not None
                and log_every > 0
                and (
                    step_idx == 0
                    or (step_idx + 1) % max(log_every, 1) == 0
                    or step_idx + 1 >= max_new_tokens
                )
            )
            if should_log:
                token_piece = describe_token(tokenizer, next_token)
                log(
                    f"Generation step {step_idx + 1}/{max_new_tokens}: "
                    f"token={next_token} piece={token_piece!r}"
                )

            if stop_token is not None and next_token == stop_token:
                if log is not None:
                    log(f"Stop token {stop_token} reached; ending generation early")
                break

    generated_tail = generated[len(prompt_ids):]
    return {
        "generated_token_ids": generated_tail,
        "all_token_ids": generated,
        "generated_text": decode_tokens(tokenizer, generated_tail) if tokenizer is not None else "",
        "full_text": decode_tokens(tokenizer, generated) if tokenizer is not None else "",
        "resolved_logits_key": resolved_logits_key,
    }


def main() -> int:
    configure_console_logging()
    args = build_parser().parse_args()
    resolve_dataset_selector_args(args)

    log_stage("Starting llama_fast inference")
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

        dataset_cfg = dict(graph.nodes.get("dataset_source").neuron_def.module_config or {}) if "dataset_source" in graph.nodes else {}
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
            top_p=args.top_p,
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
            print("Decoded text unavailable because sentencepiece is not installed.")
            print(f"Tokenizer alias: {resolved_dataset_name}")

        log_stage("Inference run completed")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
