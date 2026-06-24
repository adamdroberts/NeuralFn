from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from infer_gpt2 import (
    add_dataset_download_arguments,
    repetition_penalty_arg,
    add_dataset_selector_arguments,
)
from infer_llama_fast import (
    DEFAULT_DATASET_ALIAS,
    DEFAULT_GRAPH_ARTIFACT,
    DEFAULT_WEIGHTS_ARTIFACT,
    generate_sequence,
    log_stage,
)

GENERAL_PROMPT_SUITE = [
    "hello",
    "Once upon a time",
    "The capital of France is",
    "def hello_world():",
    "In a surprising discovery,",
]

SHAKESPEARE_PROMPT_SUITE = [
    "To be, or not to be,",
    "ROMEO:",
    "JULIET:",
    "My lord,",
    "What say you,",
]

PROMPT_SUITES = {
    "general": GENERAL_PROMPT_SUITE,
    "shakespeare": SHAKESPEARE_PROMPT_SUITE,
}

DEFAULT_REPORT_PATH = DEFAULT_WEIGHTS_ARTIFACT.with_suffix(".eval.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a repeatable llama_fast evaluation suite and save a JSON report "
            "with validation loss plus fixed-prompt generations."
        )
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--graph", default=str(DEFAULT_GRAPH_ARTIFACT))
    parser.add_argument("--weights", default="")
    add_dataset_selector_arguments(parser, default_alias=DEFAULT_DATASET_ALIAS)
    add_dataset_download_arguments(parser)
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
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
    parser.add_argument("--logits-node", default="auto")
    parser.add_argument("--context-window", type=int, default=0)
    parser.add_argument("--stop-token", type=int, default=None)
    parser.add_argument(
        "--prompt-suite",
        choices=("auto", *PROMPT_SUITES.keys()),
        default="auto",
        help="Prompt suite to use for generation checks. Defaults to auto-selecting from the dataset.",
    )
    return parser


def evaluate_validation_loss(
    compiled,
    dataset_path: Path,
    *,
    device,
    amp_dtype,
    seq_len: int,
    batch_size: int,
    eval_batches: int,
) -> float | None:
    import torch

    from infer_jepa_semantic import autocast_enabled_for
    from train_jepa_semantic import load_val_token_dataset

    if eval_batches <= 0:
        return None

    val_dataset = load_val_token_dataset(dataset_path, seq_len=seq_len)
    loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    total_rows = 0
    use_amp = autocast_enabled_for(device, amp_dtype)
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= eval_batches:
                break
            if isinstance(batch, torch.Tensor):
                flat_inputs = (batch.to(device), batch.to(device))
                batch_rows = int(batch.size(0))
            else:
                values = tuple(item.to(device) for item in batch)
                flat_inputs = (values[0], values[1] if len(values) > 1 else values[0])
                batch_rows = int(flat_inputs[0].size(0))
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = compiled(*flat_inputs)
                loss = outputs[0]
            total_loss += float(loss.item()) * batch_rows
            total_rows += batch_rows
    if total_rows <= 0:
        return None
    return total_loss / total_rows


def resolve_prompt_suite(
    *,
    dataset_name: str,
    dataset_meta: dict[str, object],
    requested_suite: str,
) -> tuple[str, list[str]]:
    if requested_suite != "auto":
        return requested_suite, list(PROMPT_SUITES[requested_suite])

    hf_path = str(dataset_meta.get("hf_path") or "").lower()
    alias = dataset_name.lower()
    if "shakespeare" in hf_path or "shakespeare" in alias or "shakespear" in alias:
        return "shakespeare", list(SHAKESPEARE_PROMPT_SUITE)
    return "general", list(GENERAL_PROMPT_SUITE)


def main() -> int:
    args = build_parser().parse_args()

    import torch

    from infer_jepa_semantic import (
        configure_console_logging,
        dataset_download_kwargs_from_args,
        load_compiled_inference_graph,
        log_tokenizer_status,
        resolve_inference_dataset_alias,
        resolve_inference_tokenizer_context,
        resolve_autocast_dtype,
        resolve_prompt_tokens,
        resolve_raw_text_encoding_name,
    )
    from train_jepa_semantic import resolve_dataset_selector_args

    configure_console_logging()
    resolve_dataset_selector_args(args)

    log_stage("Starting llama_fast evaluation")
    if args.device != "cuda":
        print("This evaluation script is configured to run on CUDA only.", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("CUDA device is not available in this environment.", file=sys.stderr)
        return 1

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    graph_path = Path(args.graph).expanduser().resolve()
    report_path = Path(args.report_path).expanduser().resolve()
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

        amp_dtype, amp_name = resolve_autocast_dtype(graph)
        log_stage(f"Compiled graph ready on {device.type} with autocast dtype {amp_name}")

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
        tokenizer, tokenizer_path, tokenizer_name, dataset_name, dataset_path, dataset_meta = resolve_inference_tokenizer_context(
            graph=graph,
            state_dict=state_dict,
            dataset_alias=dataset_alias,
            raw_text_encoding_name=raw_text_encoding_name,
            dataset_download_kwargs=dataset_download_kwargs_from_args(args),
            require_dataset=True,
        )
        if dataset_name is None or dataset_path is None or dataset_meta is None:
            raise RuntimeError("Evaluation requires a resolved dataset context.")
        log_tokenizer_status(log_stage, tokenizer, tokenizer_path, tokenizer_name)

        dataset_cfg = dict(graph.nodes.get("dataset_source").neuron_def.module_config or {}) if "dataset_source" in graph.nodes else {}
        context_window = int(args.context_window or dataset_cfg.get("seq_len") or 0)
        if context_window <= 0:
            print("Could not resolve a positive context window from the graph.", file=sys.stderr)
            return 1

        validation_loss = evaluate_validation_loss(
            compiled,
            dataset_path,
            device=device,
            amp_dtype=amp_dtype,
            seq_len=context_window,
            batch_size=args.eval_batch_size,
            eval_batches=args.eval_batches,
        )
        prompt_suite_name, prompt_suite = resolve_prompt_suite(
            dataset_name=dataset_name,
            dataset_meta=dataset_meta,
            requested_suite=args.prompt_suite,
        )
        log_stage(f"Using prompt suite {prompt_suite_name!r} for dataset {dataset_name}")

        prompt_results: list[dict[str, Any]] = []
        for prompt_idx, prompt in enumerate(prompt_suite):
            prompt_ids = resolve_prompt_tokens(prompt=prompt, prompt_tokens="", tokenizer=tokenizer)
            generator = torch.Generator(device=device.type)
            generator.manual_seed(args.seed + prompt_idx)
            log_stage(f"Evaluating prompt {prompt_idx + 1}/{len(prompt_suite)}: {prompt!r}")
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
                log_every=0,
                log=None,
            )
            prompt_results.append(
                {
                    "prompt": prompt,
                    "seed": args.seed + prompt_idx,
                    "prompt_token_ids": prompt_ids,
                    "generated_token_ids": result["generated_token_ids"],
                    "all_token_ids": result["all_token_ids"],
                    "generated_text": result["generated_text"],
                    "full_text": result["full_text"],
                    "resolved_logits_key": result["resolved_logits_key"],
                }
            )

        report = {
            "dataset_alias": dataset_name,
            "graph_path": str(graph_path),
            "weights_path": str(resolved_weights_path),
            "report_path": str(report_path),
            "context_window": context_window,
            "validation_loss": validation_loss,
            "prompt_suite": prompt_suite_name,
            "generation_config": {
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "repetition_penalty": args.repetition_penalty,
                "logits_node": args.logits_node,
                "stop_token": args.stop_token,
            },
            "prompts": prompt_results,
        }

        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        print(f"Validation loss: {validation_loss if validation_loss is not None else 'skipped'}")
        print(f"Wrote evaluation report: {report_path}")
        for item in prompt_results:
            print(f"Prompt: {item['prompt']}")
            print(item["full_text"])

        log_stage("Evaluation run completed")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
