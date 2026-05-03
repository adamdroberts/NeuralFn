from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import torch

from cli_utils import artifact_path, create_argument_parser
from neuralfn.semantic import (
    EXPERT_TO_DIMENSION,
    SEMANTIC_IGNORE_INDEX,
    ConversationalVocabulary,
    semantic_vocab_ref_for_graph,
)
from neuralfn.torch_backend import CompiledTorchGraph

from infer_jepa_semantic import (
    add_raw_text_tokenizer_arguments,
    autocast_enabled_for,
    build_semantic_model_inputs,
    add_dataset_download_arguments,
    configure_console_logging,
    dataset_download_kwargs_from_args,
    decode_tokens,
    describe_token,
    find_logits_trace_key,
    graph_uses_semantic_router_vecs,
    log_tokenizer_status,
    load_compiled_inference_graph,
    parse_csv_ints,
    resolve_semantic_router_vecs,
    resolve_autocast_dtype,
    resolve_inference_dataset_alias,
    resolve_inference_artifact_defaults,
    resolve_inference_tokenizer_context,
    resolve_raw_text_encoding_name,
    resolve_prompt_text,
    resolve_prompt_tokens,
    repetition_penalty_arg,
    resolve_semantic_targets,
    sample_next_token,
)
from train_jepa_semantic import add_dataset_selector_arguments, resolve_dataset_selector_args

DEFAULT_DATASET_ALIAS = "willdepueoai__parameter-golf__sp1024__train1"
DEFAULT_WEIGHTS_ARTIFACT = artifact_path("semantic_router_moe.pt")
DEFAULT_GRAPH_ARTIFACT = DEFAULT_WEIGHTS_ARTIFACT.with_suffix(".json")

LOGGER = logging.getLogger("semantic_router_moe_infer")


def log_stage(message: str) -> None:
    LOGGER.info(message)


def mode_name(*, megakernel: bool) -> str:
    return "semantic_router_moe_megakernel" if megakernel else "semantic_router_moe"


def default_weights_artifact(*, megakernel: bool) -> Path:
    if megakernel:
        return DEFAULT_WEIGHTS_ARTIFACT.with_name("semantic_router_moe_megakernel.pt")
    return DEFAULT_WEIGHTS_ARTIFACT


def default_graph_artifact(*, megakernel: bool) -> Path:
    return default_weights_artifact(megakernel=megakernel).with_suffix(".json")


def resolve_mode_defaults(args: argparse.Namespace) -> argparse.Namespace:
    return resolve_inference_artifact_defaults(args, mode_name=mode_name(megakernel=bool(args.megakernel)))


def build_parser() -> argparse.ArgumentParser:
    parser = create_argument_parser(
        description="Run text generation with exported semantic_router_moe artifacts on CUDA."
    )
    parser.add_argument("--megakernel", action="store_true", help="Use the semantic_router_moe_megakernel artifacts.")
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
    parser.add_argument(
        "--sem-targets",
        default="",
        help=(
            "Optional comma-separated semantic target ids. Defaults to ignore "
            "sentinels so routing is selected automatically."
        ),
    )
    parser.add_argument(
        "--semantic-topics",
        default="",
        help="Optional comma-separated dimension=topic overrides, e.g. emotion_sentiment=love,domain=psychology.",
    )
    parser.add_argument(
        "--experimental-semantic-router-vecs",
        action="store_true",
        help="Generate the 0..1 semantic_router_vecs tensor. Required automatically when the loaded graph expects it.",
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


def describe_routing(trace: dict[str, tuple[torch.Tensor, ...]]) -> str | None:
    for key in ("model/semantic_hash_router", "semantic_hash_router", "model/hash_router"):
        routing = trace.get(key)
        if not routing or len(routing) < 2:
            continue
        weights = routing[0]
        indices = routing[1]
        if weights.ndim < 2 or indices.ndim < 2:
            continue
        expert_ids = [int(value) for value in indices[0].tolist()]
        dims = [EXPERT_TO_DIMENSION.get(expert_id, f"expert_{expert_id}") for expert_id in expert_ids]
        expert_weights = [round(float(value), 4) for value in weights[0].float().tolist()]
        return f"experts={expert_ids} dims={dims} weights={expert_weights}"
    return None


def main() -> int:
    configure_console_logging()
    args = build_parser().parse_args()
    resolve_dataset_selector_args(args)
    resolve_mode_defaults(args)

    run_label = mode_name(megakernel=bool(args.megakernel))
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
        use_amp = autocast_enabled_for(device, amp_dtype)
        log_stage(f"Compiled graph ready on {device.type} with autocast dtype {amp_name}")

        semantic_vocab_ref = semantic_vocab_ref_for_graph(graph)
        vocab = ConversationalVocabulary(semantic_vocab_ref)
        tokenizer, tokenizer_path, tokenizer_name, dataset_name, _dataset_path, _dataset_meta = resolve_inference_tokenizer_context(
            graph=graph,
            state_dict=state_dict,
            dataset_alias=dataset_alias,
            raw_text_encoding_name=raw_text_encoding_name,
            dataset_download_kwargs=dataset_download_kwargs_from_args(args),
            require_dataset=False,
        )
        log_tokenizer_status(log_stage, tokenizer, tokenizer_path, tokenizer_name)

        log_stage("Encoding prompt and preparing semantic targets")
        prompt_ids = resolve_prompt_tokens(
            prompt=args.prompt,
            prompt_tokens=args.prompt_tokens,
            tokenizer=tokenizer,
        )
        if not prompt_ids:
            print("Prompt resolved to an empty token list.", file=sys.stderr)
            return 1
        prompt_text = resolve_prompt_text(
            prompt=args.prompt,
            prompt_tokens=args.prompt_tokens,
            prompt_ids=prompt_ids,
            tokenizer=tokenizer,
        )

        dataset_cfg = dict(graph.nodes["dataset_source"].neuron_def.module_config or {})
        sem_cfg = dict(graph.nodes["semantic_data_source"].neuron_def.module_config or {})
        context_window = int(args.context_window or dataset_cfg.get("seq_len") or len(prompt_ids))
        semantic_dim = int(sem_cfg.get("seq_len", vocab.vector_dim))
        sem_targets, semantic_overrides = resolve_semantic_targets(
            args.sem_targets,
            args.semantic_topics,
            semantic_dim,
            device,
            vocab,
            sequence_text=prompt_text,
        )
        graph_requires_router_vecs = graph_uses_semantic_router_vecs(graph)
        semantic_router_vecs: torch.Tensor | None = None
        if graph_requires_router_vecs or args.experimental_semantic_router_vecs:
            semantic_router_vecs = resolve_semantic_router_vecs(
                sem_targets,
                vocab=vocab,
                device=device,
            )

        log_stage(
            "Inference configuration: "
            f"dataset_alias={dataset_name or dataset_alias}, context_window={context_window}, "
            f"semantic_dim={semantic_dim}, semantic_vocab_ref={semantic_vocab_ref}, "
            f"max_new_tokens={args.max_new_tokens}, repetition_penalty={args.repetition_penalty}"
        )
        if len(prompt_ids) > context_window:
            log_stage(
                f"Prompt length {len(prompt_ids)} exceeds context window {context_window}; "
                "generation will use the most recent tokens."
            )
        if semantic_router_vecs is not None:
            vec_message = (
                "Prepared semantic_router_vecs for graph-driven routing"
                if graph_requires_router_vecs
                else "Prepared semantic_router_vecs for inspection; the loaded graph does not consume them"
            )
            log_stage(f"{vec_message} (shape={tuple(semantic_router_vecs.shape)})")

        generated = list(prompt_ids)
        print(f"Prompt token ids: {prompt_ids}")
        if prompt_text:
            print(f"Prompt text: {prompt_text}")
        print(f"Semantic targets: {sem_targets[0].tolist()}")
        if semantic_router_vecs is not None:
            print(f"Semantic router vecs: {semantic_router_vecs[0].tolist()}")
        if semantic_overrides:
            print(f"Semantic topic overrides: {semantic_overrides}")

        log_stage("Generation about to start")
        resolved_logits_key: str | None = None
        with torch.no_grad():
            for step_idx in range(args.max_new_tokens):
                context_ids = generated[-context_window:]
                tokens = torch.tensor([context_ids], dtype=torch.long, device=device)
                targets = torch.zeros_like(tokens)
                model_inputs = build_semantic_model_inputs(
                    graph,
                    tokens,
                    targets,
                    sem_targets,
                    semantic_router_vecs=semantic_router_vecs,
                )
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    _outputs, trace = compiled.trace(*model_inputs)

                if resolved_logits_key is None:
                    resolved_logits_key = find_logits_trace_key(trace, args.logits_node)
                    log_stage(f"Using traced logits node {resolved_logits_key}")

                logits = trace[resolved_logits_key][0]
                next_token = sample_next_token(
                    logits[:, -1, :],
                    temperature=args.temperature,
                    top_k=args.top_k,
                    token_history=generated,
                    repetition_penalty=args.repetition_penalty,
                    generator=generator,
                )
                generated.append(next_token)

                should_log = (
                    step_idx == 0
                    or (step_idx + 1) % max(args.log_every, 1) == 0
                    or step_idx + 1 >= args.max_new_tokens
                )
                if should_log:
                    token_piece = describe_token(tokenizer, next_token)
                    routing_summary = describe_routing(trace)
                    routing_text = f" {routing_summary}" if routing_summary else ""
                    log_stage(
                        f"Generation step {step_idx + 1}/{args.max_new_tokens}: "
                        f"token={next_token} piece={token_piece!r}{routing_text}"
                    )

                if args.stop_token is not None and next_token == args.stop_token:
                    log_stage(f"Stop token {args.stop_token} reached; ending generation early")
                    break

        generated_tail = generated[len(prompt_ids) :]
        print(f"Generated token ids: {generated_tail}")
        print(f"All token ids: {generated}")
        if tokenizer is not None:
            generated_text = decode_tokens(tokenizer, generated_tail)
            full_text = decode_tokens(tokenizer, generated)
            print("Generated text:")
            print(generated_text)
            print("Full text:")
            print(full_text)
        else:
            print("Decoded text unavailable because sentencepiece is not installed.")
            print(f"Tokenizer alias: {dataset_name}")

        log_stage("Inference run completed")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
