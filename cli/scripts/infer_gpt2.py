from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import subprocess
import sys

from cli_utils import artifact_path, create_argument_parser


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for candidate in (SCRIPT_DIR, REPO_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

DEFAULT_DATASET_ALIAS = "willdepueoai__parameter-golf__sp1024__train1"
DEFAULT_WEIGHTS_ARTIFACT = artifact_path("gpt2.pt")
DEFAULT_GRAPH_ARTIFACT = DEFAULT_WEIGHTS_ARTIFACT.with_suffix(".json")
SUPPORTED_TOKENIZER_CHOICES = ("gpt2", "cl100k_base", "o200k_base", "sp1024", "sp2048", "sp4096", "sp8192")

LOGGER = logging.getLogger("gpt2_infer")


def log_stage(message: str) -> None:
    LOGGER.info(message)


def mode_name(*, megakernel: bool, evo: bool = False) -> str:
    if evo:
        if megakernel:
            raise ValueError("gpt2_evo inference uses eager artifacts; --evo cannot be combined with --megakernel")
        return "gpt2_evo"
    return "gpt2_megakernel" if megakernel else "gpt2"


def default_weights_artifact(*, megakernel: bool, evo: bool = False) -> Path:
    if evo:
        if megakernel:
            raise ValueError("gpt2_evo inference uses eager artifacts; --evo cannot be combined with --megakernel")
        return DEFAULT_WEIGHTS_ARTIFACT.with_name("gpt2_evo.pt")
    if megakernel:
        return DEFAULT_WEIGHTS_ARTIFACT.with_name("gpt2_megakernel.pt")
    return DEFAULT_WEIGHTS_ARTIFACT


def default_graph_artifact(*, megakernel: bool, evo: bool = False) -> Path:
    return default_weights_artifact(megakernel=megakernel, evo=evo).with_suffix(".json")


def resolve_mode_defaults(args: argparse.Namespace) -> argparse.Namespace:
    graph_was_explicit = bool(getattr(args, "graph", ""))
    resolved_mode = mode_name(megakernel=bool(args.megakernel), evo=bool(getattr(args, "evo", False)))
    if not graph_was_explicit:
        args.graph = str(artifact_path(f"{resolved_mode}.json"))
    if not getattr(args, "weights", "") and not graph_was_explicit:
        args.weights = str(artifact_path(f"{resolved_mode}.pt"))
    return args


def add_dataset_selector_arguments(parser: argparse.ArgumentParser, *, default_alias: str) -> None:
    parser.add_argument("--tinystories", action="store_true")
    parser.add_argument(
        "--dataset",
        choices=("golf1", "golf10", "shakespear", "shakespeare", "tinystories"),
        default=None,
    )
    parser.add_argument("--dataset-alias", default=default_alias)


def add_dataset_download_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--download-if-missing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dataset-hf-path", default=None)
    parser.add_argument("--dataset-variant", default=None)
    parser.add_argument("--dataset-train-shards", type=int, default=None)
    parser.add_argument("--dataset-repo-id", default=None)
    parser.add_argument("--dataset-remote-root-prefix", default=None)
    parser.add_argument("--dataset-train-file", default=None)
    parser.add_argument("--dataset-val-file", default=None)


def add_raw_text_tokenizer_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--tokenizer", choices=SUPPORTED_TOKENIZER_CHOICES, dest="raw_text_encoding_override", default=None)
    group.add_argument("--tokgpt2", action="store_const", const="gpt2", dest="raw_text_encoding_override")
    group.add_argument("--cl100k", action="store_const", const="cl100k_base", dest="raw_text_encoding_override")
    group.add_argument("--o200k", action="store_const", const="o200k_base", dest="raw_text_encoding_override")
    parser.add_argument("--tokenizer-hf-path", default=None)
    parser.add_argument("--tokenizer-repo-id", default=None)
    parser.add_argument("--tokenizer-remote-root-prefix", default=None)
    parser.add_argument("--tokenizer-repo-type", choices=("model", "dataset"), default=None)
    parser.set_defaults(tokenizer=None, raw_text_encoding_override=None)


def repetition_penalty_arg(raw: str) -> float:
    value = float(raw)
    if value < 1.0:
        raise argparse.ArgumentTypeError("--repetition-penalty must be greater than or equal to 1.0")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = create_argument_parser(description="Run text generation with exported GPT artifacts on CUDA.")
    parser.add_argument("--megakernel", action="store_true", help="Use the gpt2_megakernel artifacts.")
    parser.add_argument("--evo", action="store_true", help="Use the gpt2_evo eager artifacts.")
    parser.add_argument(
        "--native-checkpoint",
        default="",
        help="Path to a native llm.kittens/NeuralFn GPT model_*.bin checkpoint.",
    )
    parser.add_argument(
        "--native-info",
        action="store_true",
        help="Print native GPT checkpoint metadata without importing the graph-backed runtime.",
    )
    parser.add_argument(
        "--native-sampler-script",
        default=os.environ.get("NFN_NATIVE_GPT_SAMPLE_SCRIPT", ""),
        help=(
            "Deprecated; native .bin checkpoint prompts are tokenized locally and "
            "run through the compiled CUDA Tile sampler."
        ),
    )
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


def native_checkpoint_prompt_tokens(args: argparse.Namespace) -> str:
    from neuralfn.native_gpt import native_gpt_prompt_tokens

    return native_gpt_prompt_tokens(
        prompt=str(getattr(args, "prompt", "") or ""),
        prompt_tokens=str(getattr(args, "prompt_tokens", "") or ""),
        encoding_name=str(getattr(args, "raw_text_encoding_override", "") or "gpt2"),
    )


def native_checkpoint_token_sampler_argv(args: argparse.Namespace, checkpoint: str | Path) -> list[str]:
    from neuralfn.native_gpt import native_gpt_checkpoint_sampler_argv

    return native_gpt_checkpoint_sampler_argv(
        checkpoint,
        prompt_tokens=native_checkpoint_prompt_tokens(args),
        max_new_tokens=int(getattr(args, "max_new_tokens", 64)),
    )


def run_native_checkpoint_token_sampler(args: argparse.Namespace, checkpoint: str | Path) -> int:
    try:
        from neuralfn.native_gpt import run_native_gpt_checkpoint_sampler

        result = run_native_gpt_checkpoint_sampler(
            checkpoint,
            prompt_tokens=native_checkpoint_prompt_tokens(args),
            max_new_tokens=int(getattr(args, "max_new_tokens", 64)),
            temperature=float(getattr(args, "temperature", 0.8)),
            top_k=int(getattr(args, "top_k", 32)),
            repetition_penalty=float(getattr(args, "repetition_penalty", 1.0)),
            seed=int(getattr(args, "seed", 1337)),
            encoding_name=str(getattr(args, "raw_text_encoding_override", "") or "gpt2"),
            runner="auto",
        )
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except FileNotFoundError:
        print(
            "Native GPT prompt-token inference needs the compiled nfn_gpt_native_train binary. "
            "Build it with tools/build_native_gpt_cli.sh or set NFN_NATIVE_GPT_CLI.",
            file=sys.stderr,
        )
        return 2
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr)
    if result.returncode == 0:
        render_native_checkpoint_sampler_text(result.stdout)
    return int(result.returncode)


def render_native_checkpoint_sampler_text(stdout: str) -> None:
    from neuralfn.native_gpt import render_native_gpt_checkpoint_sampler_text

    rendered = render_native_gpt_checkpoint_sampler_text(stdout)
    if rendered:
        print(rendered)


def run_native_checkpoint_sampler(args: argparse.Namespace, checkpoint: str | Path) -> int:
    return run_native_checkpoint_token_sampler(args, checkpoint)


def handle_native_checkpoint_request(args: argparse.Namespace) -> int | None:
    native_checkpoint = str(getattr(args, "native_checkpoint", "") or "").strip()
    if not native_checkpoint and not bool(getattr(args, "native_info", False)):
        return None
    if not native_checkpoint:
        print("--native-info requires --native-checkpoint.", file=sys.stderr)
        return 2
    from neuralfn.native_gpt import read_native_gpt_checkpoint_info

    info = read_native_gpt_checkpoint_info(Path(native_checkpoint).expanduser())
    print("Native GPT checkpoint detected")
    print(f"  path: {info.path}")
    print(f"  precision: {info.precision} (version {info.version})")
    print(f"  shape: layers={info.num_layers} heads={info.num_heads} channels={info.channels} seq_len={info.max_seq_len}")
    print(f"  vocab: vocab_size={info.vocab_size} padded_vocab_size={info.padded_vocab_size}")
    if info.step is not None:
        marker = "present" if info.done_marker_exists else "missing"
        print(f"  checkpoint_step: {info.step} (DONE marker {marker})")
    if bool(getattr(args, "native_info", False)):
        return 0
    return run_native_checkpoint_sampler(args, native_checkpoint)


def main() -> int:
    args = build_parser().parse_args()
    native_result = handle_native_checkpoint_request(args)
    if native_result is not None:
        return native_result

    import torch

    from infer_jepa_semantic import (
        configure_console_logging,
        dataset_download_kwargs_from_args,
        decode_tokens,
        load_compiled_inference_graph,
        log_tokenizer_status,
        resolve_autocast_dtype,
        resolve_inference_dataset_alias,
        resolve_inference_tokenizer_context,
        resolve_raw_text_encoding_name,
        resolve_prompt_tokens,
    )
    from infer_llama_fast import generate_sequence
    from train_jepa_semantic import resolve_dataset_selector_args

    configure_console_logging()
    if args.evo and args.megakernel:
        print("--evo cannot be combined with --megakernel; gpt2_evo exports eager artifacts.", file=sys.stderr)
        return 2
    resolve_dataset_selector_args(args)
    resolve_mode_defaults(args)

    run_label = mode_name(megakernel=bool(args.megakernel), evo=bool(args.evo))
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
            repetition_penalty=args.repetition_penalty,
            stop_token=args.stop_token,
            logits_node=args.logits_node,
            context_window=context_window,
            log_every=args.log_every,
            log=log_stage,
        )

        print(f"Generated token ids: {result['generated_token_ids']}")
        if result["generated_text"]:
            print(f"Generated text: {result['generated_text']}")
        if result["full_text"]:
            print(f"Full text: {result['full_text']}")
        log_stage("Inference completed successfully")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
