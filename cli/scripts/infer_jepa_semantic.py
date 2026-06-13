from __future__ import annotations

import argparse
import base64
import logging
import pickle
from pathlib import Path
import sys
from typing import Any, Callable

import torch

from cli_utils import artifact_path, create_argument_parser
from neuralfn import load_graph
from neuralfn.inference import load_pt_checkpoint
from neuralfn.semantic import (
    EXPERT_TO_DIMENSION,
    SEMANTIC_IGNORE_INDEX,
    ConversationalVocabulary,
    build_semantic_targets_from_topics,
    extract_semantic_topics_from_text,
    resolve_semantic_topics,
    semantic_targets_to_router_vectors,
    semantic_vocab_ref_for_graph,
)
from neuralfn.torch_backend import CompiledTorchGraph, resolve_amp_settings
from server.dataset_manager import (
    is_sentencepiece_tokenizer_name,
    local_tiktoken_encoding_path,
    normalize_raw_text_encoding_name,
    raw_text_encoding_name_for_template_spec,
    raw_text_encoding_vocab_size,
    resolve_sentencepiece_encoding,
    resolve_sentencepiece_model_path,
    resolve_tiktoken_encoding,
    validate_cached_tokenizer_contract,
)
from train_jepa_semantic import (
    add_raw_text_tokenizer_arguments as add_shared_tokenizer_arguments,
    add_dataset_download_arguments,
    add_dataset_selector_arguments,
    dataset_download_kwargs_from_args,
    resolve_dataset_selector_args,
    resolve_or_download_dataset,
)

DEFAULT_DATASET_ALIAS = "willdepueoai__parameter-golf__sp1024__train1"
DEFAULT_WEIGHTS_ARTIFACT = artifact_path("jepa_semantic_hybrid.pt")
DEFAULT_GRAPH_ARTIFACT = DEFAULT_WEIGHTS_ARTIFACT.with_suffix(".json")

LOGGER = logging.getLogger("jepa_semantic_infer")
COMPILE_CHECKPOINT_MARKERS = (
    ".node_modules.attention.node_modules.q_proj.proj.weight",
    ".node_modules.attention.node_modules.k_proj.proj.weight",
    ".node_modules.attention.node_modules.v_proj.proj.weight",
    ".node_modules.attention.node_modules.out_proj.proj.weight",
)
MEGAKERNEL_CHECKPOINT_MARKER = ".node_modules.attention.node_modules.fused_attn."


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


def default_inference_weights_artifact(mode_name: str) -> Path:
    return artifact_path(f"{mode_name}.pt")


def default_inference_graph_artifact(mode_name: str) -> Path:
    return default_inference_weights_artifact(mode_name).with_suffix(".json")


def resolve_inference_artifact_defaults(args: argparse.Namespace, *, mode_name: str) -> argparse.Namespace:
    graph_was_explicit = bool(getattr(args, "graph", ""))
    if not graph_was_explicit:
        args.graph = str(default_inference_graph_artifact(mode_name))
    if not getattr(args, "weights", "") and not graph_was_explicit:
        args.weights = str(default_inference_weights_artifact(mode_name))
    return args


def artifact_metadata_for_graph(graph) -> dict[str, Any]:
    torch_config = dict(getattr(graph, "torch_config", {}) or {})
    return dict(torch_config.get("artifact_metadata", {}) or {})


def tokenizer_manifest_for_graph(graph) -> dict[str, Any] | None:
    torch_config = dict(getattr(graph, "torch_config", {}) or {})
    manifest = dict(torch_config.get("tokenizer_manifest", {}) or {})
    return manifest or None


def resolve_graph_weights_path(
    graph,
    *,
    graph_path: Path,
    weights_path: Path | None = None,
) -> Path:
    if weights_path is not None:
        return weights_path.expanduser().resolve()

    artifact_metadata = artifact_metadata_for_graph(graph)
    referenced = str(artifact_metadata.get("weights_file", "") or "").strip()
    if referenced:
        return (graph_path.parent / referenced).expanduser().resolve()
    return graph_path.with_suffix(".pt")


def infer_graph_template_runtime(graph) -> str | None:
    torch_config = dict(getattr(graph, "torch_config", {}) or {})
    template_spec = dict(torch_config.get("template_spec", {}) or {})
    template = dict(template_spec.get("template", {}) or {})
    runtime = str(template.get("runtime", "")).lower()
    if runtime in {"eager", "compile", "megakernel"}:
        return runtime
    return None


def _infer_checkpoint_runtime_from_state_dict(state_dict: dict[str, torch.Tensor]) -> str | None:
    keys = tuple(str(key) for key in state_dict.keys())
    if any(MEGAKERNEL_CHECKPOINT_MARKER in key for key in keys):
        return "megakernel"
    if any(marker in key for key in keys for marker in COMPILE_CHECKPOINT_MARKERS):
        return "compile"
    return None


def infer_checkpoint_runtime(
    state_dict: dict[str, torch.Tensor],
    checkpoint_metadata: dict[str, Any] | None = None,
) -> str | None:
    metadata = dict(checkpoint_metadata or {})
    runtime = str(metadata.get("template_runtime", "")).strip().lower()
    if runtime in {"eager", "compile", "megakernel"}:
        return runtime
    return _infer_checkpoint_runtime_from_state_dict(state_dict)


def validate_inference_artifact_runtime_compatibility(
    *,
    graph,
    state_dict: dict[str, torch.Tensor],
    checkpoint_metadata: dict[str, Any] | None = None,
    graph_path: Path,
    weights_path: Path,
) -> None:
    graph_runtime = infer_graph_template_runtime(graph)
    checkpoint_runtime = infer_checkpoint_runtime(state_dict, checkpoint_metadata)
    if graph_runtime is None or checkpoint_runtime is None or graph_runtime == checkpoint_runtime:
        return

    graph_name = str(getattr(graph, "name", graph_path.stem))
    suggested_graph = weights_path.with_suffix(".json")
    suggested_weights = graph_path.with_suffix(".pt")
    message_lines = [
        "Inference artifact runtime mismatch.",
        f"Graph runtime: {graph_runtime} ({graph_name}) from {graph_path}",
        f"Checkpoint runtime: {checkpoint_runtime} from {weights_path}",
        "Graph and weights must come from the same exported runtime variant.",
    ]
    if suggested_graph != graph_path:
        message_lines.append(f"Suggested graph: {suggested_graph}")
    if suggested_weights != weights_path:
        message_lines.append(f"Suggested weights: {suggested_weights}")
    raise RuntimeError("\n".join(message_lines))


def validate_inference_artifact_file_types(
    *,
    graph_path: Path,
    weights_path: Path,
) -> None:
    issues: list[str] = []
    suggestions: list[str] = []

    if graph_path.suffix.lower() == ".pt":
        issues.append(f"Graph artifact looks like a PyTorch checkpoint, not a graph export: {graph_path}")
        suggested_graph = graph_path.with_suffix(".json")
        if suggested_graph.exists():
            suggestions.append(f"Suggested graph: {suggested_graph}")

    if weights_path.suffix.lower() == ".json":
        if weights_path == graph_path:
            issues.append(f"Weights artifact points to the same JSON file as --graph: {weights_path}")
        else:
            issues.append(f"Weights artifact looks like a graph export, not a checkpoint: {weights_path}")
        suggested_weights = weights_path.with_suffix(".pt")
        if suggested_weights.exists():
            suggestions.append(f"Suggested weights: {suggested_weights}")

    if not issues:
        return

    message_lines = [
        "Inference artifact file type mismatch.",
        *issues,
        "Pass --graph the exported graph file (.json) and --weights the matching checkpoint (.pt).",
    ]
    message_lines.extend(suggestions)
    raise RuntimeError("\n".join(message_lines))


def infer_graph_dataset_alias(graph) -> str | None:
    dataset_node = graph.nodes.get("dataset_source")
    if dataset_node is None:
        return None
    module_config = dict(dataset_node.neuron_def.module_config or {})
    dataset_names = module_config.get("dataset_names")
    if not isinstance(dataset_names, list):
        return None
    for dataset_name in dataset_names:
        if isinstance(dataset_name, str) and dataset_name.strip():
            return dataset_name.strip()
    return None


def inference_dataset_selector_was_explicit(argv: list[str] | None = None) -> bool:
    tokens = list(sys.argv[1:] if argv is None else argv)
    return any(
        token == "--tinystories"
        or token == "--dataset"
        or token.startswith("--dataset=")
        or token == "--dataset-alias"
        or token.startswith("--dataset-alias=")
        for token in tokens
    )


def resolve_inference_dataset_alias(
    args: argparse.Namespace,
    graph,
    *,
    default_alias: str,
    log: Callable[[str], None] | None = None,
    argv: list[str] | None = None,
) -> str:
    current_alias = str(getattr(args, "dataset_alias", default_alias))
    if inference_dataset_selector_was_explicit(argv):
        return current_alias

    graph_alias = infer_graph_dataset_alias(graph)
    if not graph_alias or current_alias != default_alias or graph_alias == current_alias:
        return current_alias

    if log is not None:
        log(
            "No dataset selector override was provided; "
            f"using dataset alias from the exported graph: {graph_alias}"
        )
    args.dataset_alias = graph_alias
    return graph_alias


def load_compiled_inference_graph(
    *,
    graph_path: Path,
    weights_path: Path | None,
    device: torch.device,
) -> tuple[object, CompiledTorchGraph, dict[str, torch.Tensor], Path]:
    graph = load_graph(graph_path)
    resolved_weights_path = resolve_graph_weights_path(
        graph,
        graph_path=graph_path,
        weights_path=weights_path,
    )
    validate_inference_artifact_file_types(graph_path=graph_path, weights_path=resolved_weights_path)
    try:
        state_dict, checkpoint_metadata = load_pt_checkpoint(resolved_weights_path, map_location="cpu")
    except (pickle.UnpicklingError, EOFError, RuntimeError, TypeError, ValueError) as exc:
        suggested_weights = resolved_weights_path.with_suffix(".pt")
        message_lines = [
            f"Failed to load weights checkpoint from {resolved_weights_path}.",
            "Expected --weights to point to a PyTorch checkpoint export.",
        ]
        if resolved_weights_path.suffix.lower() == ".json":
            message_lines.append("The provided path ends in .json, which is usually the graph export.")
        if suggested_weights != resolved_weights_path and suggested_weights.exists():
            message_lines.append(f"Suggested weights: {suggested_weights}")
        message_lines.append(f"Original loader error: {exc}")
        raise RuntimeError("\n".join(message_lines)) from exc
    validate_inference_artifact_runtime_compatibility(
        graph=graph,
        state_dict=state_dict,
        checkpoint_metadata=checkpoint_metadata,
        graph_path=graph_path,
        weights_path=resolved_weights_path,
    )
    compiled = CompiledTorchGraph(graph)
    compiled.load_state_dict(state_dict)
    compiled.to(device)
    compiled.eval()
    return graph, compiled, state_dict, resolved_weights_path


def parse_csv_ints(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        values.append(int(stripped))
    return values


def mode_name(*, megakernel: bool) -> str:
    return "jepa_semantic_hybrid_megakernel" if megakernel else "jepa_semantic_hybrid"


def default_weights_artifact(*, megakernel: bool) -> Path:
    return default_inference_weights_artifact(mode_name(megakernel=megakernel))


def default_graph_artifact(*, megakernel: bool) -> Path:
    return default_inference_graph_artifact(mode_name(megakernel=megakernel))


def resolve_mode_defaults(args: argparse.Namespace) -> argparse.Namespace:
    return resolve_inference_artifact_defaults(args, mode_name=mode_name(megakernel=bool(args.megakernel)))


def graph_uses_semantic_router_vecs(graph) -> bool:
    template_spec = dict(graph.torch_config.get("template_spec", {}) or {})
    if bool(template_spec.get("experimental_semantic_router_vecs", False)):
        return True
    sem_node = graph.nodes.get("semantic_data_source")
    if sem_node is None:
        return False
    return any(port.name == "semantic_router_vecs" for port in sem_node.neuron_def.output_ports)


def add_raw_text_tokenizer_arguments(parser: argparse.ArgumentParser) -> None:
    add_shared_tokenizer_arguments(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = create_argument_parser(
        description="Run text generation with exported jepa_semantic_hybrid artifacts on CUDA."
    )
    parser.add_argument("--megakernel", action="store_true", help="Use the jepa_semantic_hybrid_megakernel artifacts.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-dtype", choices=("float32", "bfloat16", "float16"), default=None)
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


def load_sentencepiece_model(
    dataset_path: Path,
    dataset_meta: dict[str, object],
    *,
    raw_text_encoding_name: str = "gpt2",
):
    model_candidates: list[Path] = []
    tokenizer_files = dataset_meta.get("tokenizer_files")
    if isinstance(tokenizer_files, list):
        for filename in tokenizer_files:
            if isinstance(filename, str) and filename.endswith(".model"):
                model_candidates.append(dataset_path / "tokenizers" / filename)
    if dataset_path.is_dir():
        model_candidates.extend(sorted((dataset_path / "tokenizers").glob("*.model")))
    model_path = next((path for path in model_candidates if path.exists()), None)
    if model_path is not None:
        try:
            import sentencepiece as spm  # type: ignore
        except ImportError:
            return None, model_path, None
        processor = spm.SentencePieceProcessor()
        processor.load(str(model_path))
        return processor, model_path, None

    if dataset_meta.get("data_format") != "uint16_shards" and is_sentencepiece_tokenizer_name(raw_text_encoding_name):
        model_path = resolve_sentencepiece_model_path(raw_text_encoding_name)
        try:
            return resolve_sentencepiece_encoding(raw_text_encoding_name), model_path, None
        except RuntimeError:
            return None, model_path, None

    if dataset_meta.get("data_format") != "uint16_shards":
        encoding_path = local_tiktoken_encoding_path(raw_text_encoding_name)
        try:
            return resolve_tiktoken_encoding(raw_text_encoding_name), encoding_path, raw_text_encoding_name
        except Exception:
            return None, encoding_path, raw_text_encoding_name
    return None, None, None


def load_tokenizer_from_graph_manifest(
    graph,
    *,
    raw_text_encoding_name: str = "gpt2",
):
    manifest = tokenizer_manifest_for_graph(graph)
    if not manifest:
        return None, None, None

    backend = str(manifest.get("backend", "") or "").strip().lower()
    if backend == "sentencepiece":
        model_blob = str(manifest.get("model_proto_b64", "") or "")
        if not model_blob:
            return None, Path(str(manifest.get("model_file") or "graph-tokenizer.model")), None
        try:
            import sentencepiece as spm  # type: ignore
        except ImportError:
            return None, Path(str(manifest.get("model_file") or "graph-tokenizer.model")), None
        processor = spm.SentencePieceProcessor()
        model_bytes = base64.b64decode(model_blob.encode("ascii"))
        loader = getattr(processor, "LoadFromSerializedProto", None) or getattr(processor, "load_from_serialized_proto", None)
        if not callable(loader):
            raise RuntimeError("Installed sentencepiece runtime cannot load serialized tokenizer payloads.")
        loader(model_bytes)
        return processor, Path(str(manifest.get("model_file") or "graph-tokenizer.model")), None

    encoding_name = str(manifest.get("encoding_name") or raw_text_encoding_name or "gpt2").strip()
    if not encoding_name:
        return None, None, None
    encoding_path = local_tiktoken_encoding_path(encoding_name)
    try:
        return resolve_tiktoken_encoding(encoding_name), encoding_path, encoding_name
    except Exception:
        return None, encoding_path, encoding_name


def log_tokenizer_status(
    log: Callable[[str], None],
    tokenizer,
    tokenizer_path: Path | None,
    tokenizer_name: str | None = None,
) -> None:
    tokenizer_vocab_size = resolved_tokenizer_vocab_size(tokenizer)
    vocab_text = f" (vocab={tokenizer_vocab_size})" if tokenizer_vocab_size is not None else ""
    if tokenizer_path is not None and tokenizer_name is None:
        if tokenizer is None:
            log(
                f"Tokenizer model found at {tokenizer_path}, but sentencepiece is unavailable. "
                "Falling back to token-id mode."
            )
        else:
            log(f"Loaded tokenizer from {tokenizer_path}{vocab_text}")
        return
    if tokenizer_name is not None:
        if tokenizer is None:
            if tokenizer_path is not None:
                log(
                    f"Tiktoken encoding file found at {tokenizer_path}, but tiktoken is unavailable. "
                    "Falling back to token-id mode."
                )
            else:
                log(f"Requested {tokenizer_name} tiktoken, but tiktoken is unavailable. Falling back to token-id mode.")
        elif tokenizer_path is not None:
            log(f"Loaded {tokenizer_name} tiktoken from {tokenizer_path}{vocab_text}")
        else:
            log(f"Using {tokenizer_name} tiktoken for raw-text prompt encoding and decoding{vocab_text}.")
        return
    if tokenizer is not None:
        log(
            "No sentencepiece model found under the cached dataset alias; "
            f"using GPT-2 tiktoken for raw-text prompt encoding and decoding{vocab_text}."
        )
        return
    log("No text tokenizer found under the cached dataset alias; token-id mode only.")


def resolve_model_vocab_size(graph, state_dict: dict[str, torch.Tensor]) -> int:
    vocab_size = 0
    template_spec = dict(graph.torch_config.get("template_spec", {}))
    if template_spec.get("vocab_size") is not None:
        vocab_size = max(vocab_size, int(template_spec["vocab_size"]))

    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor) or value.ndim < 2:
            continue
        if key.endswith("token_embed.embedding.weight"):
            vocab_size = max(vocab_size, int(value.shape[0]))
        elif key.endswith("lm_head.proj.weight") or key.endswith("ar_head.proj.weight"):
            vocab_size = max(vocab_size, int(value.shape[0]))
    return vocab_size


def validate_inference_vocab_contract(
    *,
    dataset_name: str | None,
    dataset_path: Path | None,
    dataset_meta: dict[str, object] | None,
    graph,
    state_dict: dict[str, torch.Tensor],
    raw_text_encoding_name: str = "gpt2",
) -> dict[str, Any] | None:
    model_vocab_size = resolve_model_vocab_size(graph, state_dict)
    if dataset_name is not None and dataset_path is not None and dataset_meta is not None:
        contract = validate_cached_tokenizer_contract(
            dataset_name,
            dataset_path=dataset_path,
            dataset_meta=dict(dataset_meta),
            model_vocab_size=model_vocab_size if model_vocab_size > 0 else None,
        )
        if contract is not None or model_vocab_size <= 0:
            return contract

        expected_vocab_size = raw_text_encoding_vocab_size(raw_text_encoding_name)
        if model_vocab_size != expected_vocab_size:
            raise RuntimeError(
                f"Raw-text dataset {dataset_name!r} is using {raw_text_encoding_name} "
                f"(vocab size {expected_vocab_size}), but the loaded graph/checkpoint vocab size is {model_vocab_size}. "
                "Train/export a matching checkpoint or rerun inference with the corresponding raw-text tokenizer flag."
            )
        return {
            "dataset_name": dataset_name,
            "dataset_path": dataset_path,
            "dataset_meta": dataset_meta,
            "tokenizer_encoding": raw_text_encoding_name,
            "tokenizer_vocab_size": expected_vocab_size,
        }

    tokenizer_manifest = tokenizer_manifest_for_graph(graph)
    if tokenizer_manifest is None or model_vocab_size <= 0:
        return tokenizer_manifest

    expected_vocab_size = tokenizer_manifest.get("tokenizer_vocab_size")
    if expected_vocab_size is None:
        return tokenizer_manifest
    expected_vocab_size = int(expected_vocab_size)
    if model_vocab_size != expected_vocab_size:
        backend = str(tokenizer_manifest.get("backend", "tokenizer")).strip() or "tokenizer"
        raise RuntimeError(
            f"The exported graph declares a {backend} vocabulary size of {expected_vocab_size}, "
            f"but the loaded graph/checkpoint vocab size is {model_vocab_size}. "
            "Train/export a matching checkpoint or provide an explicit compatible weights artifact."
        )
    return tokenizer_manifest


def resolve_inference_tokenizer_context(
    *,
    graph,
    state_dict: dict[str, torch.Tensor],
    dataset_alias: str,
    raw_text_encoding_name: str,
    dataset_download_kwargs: dict[str, object],
    require_dataset: bool = False,
):
    tokenizer, tokenizer_path, tokenizer_name = load_tokenizer_from_graph_manifest(
        graph,
        raw_text_encoding_name=raw_text_encoding_name,
    )
    dataset_name: str | None = None
    dataset_path: Path | None = None
    dataset_meta: dict[str, object] | None = None
    if require_dataset or (tokenizer is None and tokenizer_path is None and tokenizer_name is None):
        dataset_name, dataset_path, dataset_meta = resolve_or_download_dataset(
            dataset_alias,
            raw_text_encoding_name=raw_text_encoding_name,
            **dataset_download_kwargs,
        )
        if tokenizer is None and tokenizer_path is None and tokenizer_name is None:
            tokenizer, tokenizer_path, tokenizer_name = load_sentencepiece_model(
                dataset_path,
                dataset_meta,
                raw_text_encoding_name=raw_text_encoding_name,
            )
    validate_inference_vocab_contract(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        dataset_meta=dataset_meta,
        graph=graph,
        state_dict=state_dict,
        raw_text_encoding_name=raw_text_encoding_name,
    )
    return tokenizer, tokenizer_path, tokenizer_name, dataset_name, dataset_path, dataset_meta


def resolve_raw_text_encoding_name(
    graph,
    *,
    encoding_override: str | None = None,
    prefer_cl100k: bool = False,
) -> str:
    resolved_override = normalize_raw_text_encoding_name(encoding_override)
    if resolved_override is not None:
        return resolved_override
    torch_config = dict(getattr(graph, "torch_config", {}) or {})
    template_spec = dict(torch_config.get("template_spec", {}) or {})
    resolved_template_name = normalize_raw_text_encoding_name(template_spec.get("raw_text_encoding_name"))
    if resolved_template_name is not None:
        return resolved_template_name
    manifest = tokenizer_manifest_for_graph(graph)
    if manifest is not None:
        resolved_manifest_name = normalize_raw_text_encoding_name(
            manifest.get("tokenizer_name") or manifest.get("encoding_name")
        )
        if resolved_manifest_name is not None:
            return resolved_manifest_name
    return raw_text_encoding_name_for_template_spec(
        template_spec,
        prefer_cl100k=prefer_cl100k,
    )


def resolve_prompt_tokens(
    *,
    prompt: str,
    prompt_tokens: str,
    tokenizer,
) -> list[int]:
    if prompt_tokens.strip():
        return parse_csv_ints(prompt_tokens)
    if tokenizer is None:
        raise RuntimeError(
            "Text prompt encoding requires a usable tokenizer. "
            "Install sentencepiece for tokenizer-backed datasets, or pass --prompt-tokens. "
            "Raw-text datasets also support GPT-2 tiktoken when it is installed."
        )
    if prompt:
        encode = getattr(tokenizer, "encode", None)
        if not callable(encode):
            raise RuntimeError("Resolved tokenizer does not expose an encode() method.")
        special_tokens = getattr(tokenizer, "special_tokens_set", set()) or set()
        if "<|endoftext|>" in special_tokens:
            return list(encode(prompt, allowed_special={"<|endoftext|>"}))
        try:
            return list(encode(prompt, out_type=int))
        except TypeError:
            return list(encode(prompt))
    bos_getter = getattr(tokenizer, "bos_id", None)
    if callable(bos_getter):
        bos_id = int(bos_getter())
        if bos_id >= 0:
            return [bos_id]
    eot_token = getattr(tokenizer, "eot_token", None)
    if eot_token is not None:
        eot_id = int(eot_token)
        if eot_id >= 0:
            return [eot_id]
    return [0]


def resolve_prompt_text(
    *,
    prompt: str,
    prompt_tokens: str,
    prompt_ids: list[int],
    tokenizer,
) -> str:
    if not prompt_tokens.strip() and prompt:
        return prompt
    if tokenizer is None or not prompt_ids:
        return ""
    try:
        return decode_tokens(tokenizer, prompt_ids)
    except ValueError:
        return ""


def parse_topic_overrides(raw: str) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for part in raw.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        if "=" not in stripped:
            raise ValueError(f"Expected semantic override in dimension=topic form, got {stripped!r}")
        dim_name, topic = stripped.split("=", 1)
        overrides[dim_name.strip()] = topic.strip()
    return overrides


def resolve_semantic_targets(
    raw: str,
    semantic_topics: str,
    semantic_dim: int,
    device: torch.device,
    vocab: ConversationalVocabulary,
    *,
    sequence_text: str = "",
) -> tuple[torch.Tensor, dict[str, str]]:
    if semantic_topics.strip():
        overrides = resolve_semantic_topics(parse_topic_overrides(semantic_topics), vocab=vocab)
        values = build_semantic_targets_from_topics(overrides, vocab=vocab).tolist()
    elif raw.strip():
        overrides = {}
        values = parse_csv_ints(raw)
    else:
        overrides = extract_semantic_topics_from_text(sequence_text, vocab=vocab) if sequence_text.strip() else {}
        values = (
            build_semantic_targets_from_topics(overrides, vocab=vocab).tolist()
            if overrides
            else [SEMANTIC_IGNORE_INDEX] * semantic_dim
        )
    if len(values) != semantic_dim:
        raise ValueError(f"Expected {semantic_dim} semantic target values, got {len(values)}")
    return torch.tensor([values], dtype=torch.long, device=device), overrides


def resolve_semantic_router_vecs(
    sem_targets: torch.Tensor,
    *,
    vocab: ConversationalVocabulary,
    device: torch.device,
) -> torch.Tensor:
    router_vecs = semantic_targets_to_router_vectors(sem_targets.detach().cpu().numpy(), vocab=vocab)
    return torch.tensor(router_vecs, dtype=torch.float32, device=device)


def build_semantic_model_inputs(
    graph,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    sem_targets: torch.Tensor,
    *,
    semantic_router_vecs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    if graph_uses_semantic_router_vecs(graph):
        if semantic_router_vecs is None:
            raise RuntimeError(
                "The loaded graph expects semantic_router_vecs, but no router vector tensor was provided."
            )
        return (tokens, targets, sem_targets, semantic_router_vecs)
    return (tokens, targets, sem_targets)


def resolve_autocast_settings(
    graph,
    *,
    amp_dtype_override: str | None = None,
) -> tuple[torch.dtype, str, bool]:
    return resolve_amp_settings(amp_dtype_override or graph.torch_config.get("amp_dtype", "float32"))


def resolve_autocast_dtype(
    graph,
    *,
    amp_dtype_override: str | None = None,
) -> tuple[torch.dtype, str]:
    amp_dtype, amp_name, _use_amp = resolve_autocast_settings(
        graph,
        amp_dtype_override=amp_dtype_override,
    )
    return amp_dtype, amp_name


def autocast_enabled_for(device: torch.device, amp_dtype: torch.dtype) -> bool:
    return device.type == "cuda" and amp_dtype in {torch.float16, torch.bfloat16}


def find_logits_trace_key(trace: dict[str, tuple[torch.Tensor, ...]], requested: str) -> str:
    if requested != "auto":
        if requested not in trace:
            raise KeyError(f"Trace key {requested!r} was not found in the compiled trace.")
        return requested

    preferred = (
        "model/softcap",
        "model/lm_head",
        "model/tied_lm_head",
        "softcap",
        "lm_head",
        "tied_lm_head",
    )
    for key in preferred:
        value = trace.get(key)
        if value and value[0].ndim == 3:
            return key

    for suffix in ("/softcap", "/lm_head", "/tied_lm_head"):
        for key, value in trace.items():
            if key.endswith(suffix) and value and value[0].ndim == 3:
                return key

    available = ", ".join(sorted(trace.keys()))
    raise KeyError(
        "Could not find a logits-like trace node. "
        f"Available trace keys: {available}"
    )


def top_k_filter(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0 or top_k >= logits.size(-1):
        return logits
    values, indices = torch.topk(logits, top_k, dim=-1)
    filtered = torch.full_like(logits, float("-inf"))
    return filtered.scatter(-1, indices, values)


def top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_remove = cumulative_probs > top_p
    sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
    sorted_remove[..., 0] = False
    remove_mask = torch.zeros_like(sorted_remove, dtype=torch.bool)
    remove_mask.scatter_(-1, sorted_indices, sorted_remove)
    return logits.masked_fill(remove_mask, float("-inf"))


def top_p_arg(raw: str) -> float:
    value = float(raw)
    if value <= 0.0 or value > 1.0:
        raise argparse.ArgumentTypeError("--top-p must be in the range (0, 1].")
    return value


def repetition_penalty_arg(raw: str) -> float:
    value = float(raw)
    if value < 1.0:
        raise argparse.ArgumentTypeError("--repetition-penalty must be greater than or equal to 1.0")
    return value


def apply_repetition_penalty(
    logits: torch.Tensor,
    *,
    token_history: list[int] | tuple[int, ...],
    repetition_penalty: float,
) -> torch.Tensor:
    if repetition_penalty <= 1.0 or not token_history:
        return logits
    penalized = logits.clone()
    vocab_size = penalized.size(-1)
    seen_tokens = sorted({int(token_id) for token_id in token_history if 0 <= int(token_id) < vocab_size})
    if not seen_tokens:
        return penalized
    seen_indices = torch.tensor(seen_tokens, device=penalized.device, dtype=torch.long)
    gather_shape = [1] * penalized.ndim
    gather_shape[-1] = seen_indices.numel()
    gather_indices = seen_indices.view(*gather_shape).expand(*penalized.shape[:-1], seen_indices.numel())
    seen_logits = penalized.gather(-1, gather_indices)
    adjusted_logits = torch.where(
        seen_logits < 0,
        seen_logits * repetition_penalty,
        seen_logits / repetition_penalty,
    )
    penalized.scatter_(-1, gather_indices, adjusted_logits)
    return penalized


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    token_history: list[int] | tuple[int, ...],
    repetition_penalty: float,
    top_p: float = 1.0,
    generator: torch.Generator,
) -> int:
    step_logits = apply_repetition_penalty(
        logits.float(),
        token_history=token_history,
        repetition_penalty=repetition_penalty,
    )
    if temperature <= 0.0:
        return int(torch.argmax(step_logits, dim=-1).item())
    step_logits = top_k_filter(step_logits, top_k)
    step_logits = top_p_filter(step_logits, top_p) / temperature
    probs = torch.softmax(step_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1, generator=generator)
    return int(next_token.item())


def decode_tokens(tokenizer, token_ids: list[int]) -> str:
    if tokenizer is None:
        return ""
    try:
        return str(tokenizer.decode(token_ids))
    except (IndexError, KeyError, ValueError) as exc:
        piece_size = None
        getter = getattr(tokenizer, "get_piece_size", None)
        if callable(getter):
            try:
                piece_size = int(getter())
            except Exception:
                piece_size = None
        if piece_size is None:
            n_vocab = getattr(tokenizer, "n_vocab", None)
            if n_vocab is not None:
                try:
                    piece_size = int(n_vocab)
                except Exception:
                    piece_size = None
        max_token_id = max(token_ids) if token_ids else -1
        size_text = f" for tokenizer vocab size {piece_size}" if piece_size is not None else ""
        raise ValueError(
            f"Tokenizer decode failed because token id {max_token_id} is out of range{size_text}."
        ) from exc


def resolved_tokenizer_vocab_size(tokenizer) -> int | None:
    if tokenizer is None:
        return None
    getter = getattr(tokenizer, "get_piece_size", None)
    if callable(getter):
        try:
            return int(getter())
        except Exception:
            pass
    n_vocab = getattr(tokenizer, "n_vocab", None)
    if n_vocab is None:
        return None
    try:
        return int(n_vocab)
    except Exception:
        return None


def describe_token(tokenizer, token_id: int) -> str:
    if tokenizer is None:
        return str(token_id)
    try:
        piece_getter = getattr(tokenizer, "id_to_piece", None)
        if callable(piece_getter):
            return str(piece_getter(int(token_id)))

        byte_getter = getattr(tokenizer, "decode_single_token_bytes", None)
        if callable(byte_getter):
            return byte_getter(int(token_id)).decode("utf-8", errors="backslashreplace")

        return str(tokenizer.decode([int(token_id)]))
    except Exception:
        return str(token_id)


def describe_routing(trace: dict[str, tuple[torch.Tensor, ...]]) -> str | None:
    routing = trace.get("model/hash_router")
    if not routing or len(routing) < 2:
        return None
    weights = routing[0]
    indices = routing[1]
    if weights.ndim < 2 or indices.ndim < 2:
        return None
    expert_ids = [int(value) for value in indices[0].tolist()]
    dims = [EXPERT_TO_DIMENSION.get(expert_id, f"expert_{expert_id}") for expert_id in expert_ids]
    expert_weights = [round(float(value), 4) for value in weights[0].float().tolist()]
    return f"experts={expert_ids} dims={dims} weights={expert_weights}"


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

        amp_dtype, amp_name, use_amp = resolve_autocast_settings(
            graph,
            amp_dtype_override=args.amp_dtype,
        )
        if args.amp_dtype:
            graph.torch_config = {**graph.torch_config, "amp_dtype": amp_name}
        amp_mode = amp_name if use_amp else "disabled (float32)"
        log_stage(f"Compiled graph ready on {device.type} with autocast {amp_mode}")

        tokenizer, tokenizer_path, tokenizer_name, dataset_name, dataset_path, dataset_meta = resolve_inference_tokenizer_context(
            graph=graph,
            state_dict=state_dict,
            dataset_alias=dataset_alias,
            raw_text_encoding_name=raw_text_encoding_name,
            dataset_download_kwargs=dataset_download_kwargs_from_args(args),
            require_dataset=False,
        )
        semantic_vocab_ref = semantic_vocab_ref_for_graph(graph)
        vocab = ConversationalVocabulary(semantic_vocab_ref)
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

        resolved_dataset_name = dataset_name or dataset_alias
        log_stage(
            "Inference configuration: "
            f"dataset_alias={resolved_dataset_name}, context_window={context_window}, "
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

        logits_trace_key = None
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
                if logits_trace_key is None:
                    logits_trace_key = find_logits_trace_key(trace, requested=args.logits_node)
                    log_stage(f"Resolved logits trace key: {logits_trace_key}")
                logits = trace[logits_trace_key][0][:, -1, :]
                next_token = sample_next_token(
                    logits,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    token_history=generated,
                    repetition_penalty=args.repetition_penalty,
                    generator=generator,
                )
                generated.append(next_token)
                if args.log_every > 0 and step_idx % args.log_every == 0:
                    token_desc = describe_token(tokenizer, next_token)
                    routing_summary = describe_routing(trace)
                    if routing_summary:
                        log_stage(
                            f"Generation step {step_idx + 1}/{args.max_new_tokens}: "
                            f"token={next_token} text={token_desc!r} {routing_summary}"
                        )
                    else:
                        log_stage(
                            f"Generation step {step_idx + 1}/{args.max_new_tokens}: "
                            f"token={next_token} text={token_desc!r}"
                        )
                if args.stop_token is not None and next_token == args.stop_token:
                    log_stage(f"Stop token {args.stop_token} emitted at generation step {step_idx + 1}.")
                    break

        generated_tail = generated[len(prompt_ids):]
        generated_text = decode_tokens(tokenizer, generated_tail) if tokenizer is not None else ""
        full_text = decode_tokens(tokenizer, generated) if tokenizer is not None else ""
        print(f"Semantic targets: {sem_targets[0].tolist()}")
        if semantic_router_vecs is not None:
            print(f"Semantic router vecs: {semantic_router_vecs[0].tolist()}")
        if semantic_overrides:
            print(f"Semantic topic overrides: {semantic_overrides}")
        print(f"Generated token ids: {generated_tail}")
        if generated_text:
            print("Generated text:")
            print(generated_text)
        if full_text:
            print("Full text:")
            print(full_text)
        log_stage("Inference run completed")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
