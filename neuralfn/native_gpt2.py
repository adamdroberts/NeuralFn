from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import struct
from typing import Any


DEFAULT_NATIVE_GPT2_EXECUTABLE = "/mnt/disk2/dev/open-source/llm.kittens/train_gpt2cu"
DEFAULT_NATIVE_GPT2_LAUNCHER = "build/nfn_gpt2_tile_train"
DEFAULT_NATIVE_GPT_CLI = "build/nfn_gpt_native_train"
DEFAULT_NATIVE_GPT2_CLI = DEFAULT_NATIVE_GPT_CLI
LEGACY_NATIVE_GPT2_CLI = "build/nfn_gpt2_native_train"
NATIVE_GPT2_BINDING_MODULES = ("neuralfn_native_gpt2", "neuralfn._native_gpt2")
NATIVE_GPT2_CHECKPOINT_MAGIC = 20240326
NATIVE_GPT2_CHECKPOINT_HEADER_INTS = 256
NATIVE_GPT2_CHECKPOINT_HEADER_BYTES = NATIVE_GPT2_CHECKPOINT_HEADER_INTS * 4
NATIVE_GPT2_CHECKPOINT_VERSIONS = {
    3: ("fp32", 4),
    5: ("bf16", 2),
}
RAW_TEXT_ENCODING_ALIASES = {
    "gpt2": "gpt2",
    "tokgpt2": "gpt2",
    "cl100k": "cl100k_base",
    "cl100k_base": "cl100k_base",
    "o200k": "o200k_base",
    "o200k_base": "o200k_base",
    "sp1024": "sp1024",
    "sp2048": "sp2048",
    "sp4096": "sp4096",
    "sp8192": "sp8192",
}
RAW_TEXT_ENCODING_VOCAB_SIZES = {
    "gpt2": 50257,
    "cl100k_base": 100277,
    "o200k_base": 200019,
    "sp1024": 1024,
    "sp2048": 2048,
    "sp4096": 4096,
    "sp8192": 8192,
}


@dataclass(frozen=True)
class NativeGpt2RunnerStatus:
    requested: str
    resolved: str
    binding_module: str | None = None
    available: bool = True
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NativeGpt2RunConfig:
    """Configuration for the native CUDA Tile/SM120 dense GPT trainer handoff."""

    executable: str
    train_data: str
    val_data: str
    output_dir: str
    model_descriptor: str
    eval_every_steps: int
    sample_every_steps: int
    generate_tokens: int
    checkpoint_every_steps: int
    batch_size: int
    seq_len: int
    train_batch_tokens: int
    learning_rate: float
    final_lr_fraction: float
    warmup_steps: int
    weight_decay: float
    max_steps: int
    eval_batches: int = 1
    eval_batch_size: int = 0
    lm_head_row_chunk_size: int = 4096
    activation: str = "gelu"
    moa_interval: int = 50
    kernel_backend: str = "tile-cuda"
    tile_ops_lib: str = ""
    smoke_tile_ops: bool = False
    smoke_optimizer_step: bool = False
    smoke_lm_step: bool = False
    smoke_attention_step: bool = False
    smoke_mlp_step: bool = False
    smoke_norm_residual_step: bool = False
    smoke_transformer_block_step: bool = False
    smoke_transformer_lm_step: bool = False
    smoke_embedding_lm_step: bool = False
    train_embedding_lm: bool = False
    train_transformer_lm: bool = True
    checkpoint_metadata_smoke: bool = False
    write_checkpoint: bool = True
    cuda_runtime_lib: str = ""
    hellaswag_eval: int = 0
    recompute: int = 0
    zero_stage: int = 1
    resume: int = 0
    cuda_device_max_connections: str = "1"
    dataset_alias: str | None = None
    template_name: str = "gpt"
    graph_file: str = ""
    model_family: str = "gpt"

    def argv(self) -> list[str]:
        args = [
            self.executable,
            "-i",
            self.train_data,
            "-j",
            self.val_data,
            "-o",
            self.output_dir,
            "-v",
            str(int(self.eval_every_steps)),
            "-s",
            str(int(self.sample_every_steps)),
            "-g",
            str(int(self.generate_tokens)),
            "-h",
            str(int(self.hellaswag_eval)),
            "-b",
            str(int(self.batch_size)),
            "-t",
            str(int(self.seq_len)),
            "-d",
            str(int(self.train_batch_tokens)),
            "-r",
            str(int(self.recompute)),
            "-z",
            str(int(self.zero_stage)),
            "-c",
            str(float(self.weight_decay)),
            "-l",
            str(float(self.learning_rate)),
            "-q",
            str(float(self.final_lr_fraction)),
            "-u",
            str(int(self.warmup_steps)),
            "-n",
            str(int(self.checkpoint_every_steps)),
            "-y",
            str(int(self.resume)),
            "-e",
            self.model_descriptor,
            "-af",
            self.activation,
            "-x",
            str(int(self.max_steps)),
        ]
        if self.activation == "moa":
            args.extend(["-ak", str(int(self.moa_interval))])
        return args

    def command(self) -> str:
        return shlex.join(self.argv())

    def launcher_argv(self, launcher: str | None = None) -> list[str]:
        launcher_path = resolve_native_gpt2_launcher(launcher)
        return [
            launcher_path,
            "--target",
            self.executable,
            "--",
            *self.argv()[1:],
        ]

    def launcher_command(self, launcher: str | None = None) -> str:
        return shlex.join(self.launcher_argv(launcher))

    def compiled_cli_argv(self, cli: str | None = None) -> list[str]:
        cli_path = resolve_native_gpt2_cli(cli)
        dataset_alias = str(self.dataset_alias or "").strip() or str(Path(self.train_data).parent)
        args = [
            cli_path,
            "--model-family",
            self.model_family,
            "--dataset-alias",
            dataset_alias,
            "--backend",
            self.kernel_backend,
            "--output-dir",
            self.output_dir,
            "--eval-every-steps",
            str(int(self.eval_every_steps)),
            "--eval-batches",
            str(int(self.eval_batches)),
            "--eval-batch-size",
            str(int(self.eval_batch_size)),
            "--lm-head-row-chunk-size",
            str(int(self.lm_head_row_chunk_size)),
            "--native-cuda-sample-every",
            str(int(self.sample_every_steps)),
            "--native-cuda-generate-tokens",
            str(int(self.generate_tokens)),
            "--native-cuda-checkpoint-every",
            str(int(self.checkpoint_every_steps)),
            "--batch-size",
            str(int(self.batch_size)),
            "--train-seq-len",
            str(int(self.seq_len)),
            "--train-batch-tokens",
            str(int(self.train_batch_tokens)),
            "--learning-rate",
            str(float(self.learning_rate)),
            "--final-lr-fraction",
            str(float(self.final_lr_fraction)),
            "--weight-decay",
            str(float(self.weight_decay)),
            "--warmup-steps",
            str(int(self.warmup_steps)),
            "--max-steps",
            str(int(self.max_steps)),
            "--num-layers",
            str(int(str(self.model_descriptor).removeprefix("d") or "12")),
            "--native-cuda-activation",
            self.activation,
        ]
        if str(self.kernel_backend).strip().lower().replace("_", "-") == "llm-kittens":
            args.extend(["--target", self.executable])
        if str(self.tile_ops_lib or "").strip():
            args.extend(["--tile-ops-lib", self.tile_ops_lib])
        if self.smoke_tile_ops:
            args.append("--smoke-tile-ops")
        if self.smoke_optimizer_step:
            args.append("--smoke-optimizer-step")
        if self.smoke_lm_step:
            args.append("--smoke-lm-step")
        if self.smoke_attention_step:
            args.append("--smoke-attention-step")
        if self.smoke_mlp_step:
            args.append("--smoke-mlp-step")
        if self.smoke_norm_residual_step:
            args.append("--smoke-norm-residual-step")
        if self.smoke_transformer_block_step:
            args.append("--smoke-transformer-block-step")
        if self.smoke_transformer_lm_step:
            args.append("--smoke-transformer-lm-step")
        if self.smoke_embedding_lm_step:
            args.append("--smoke-embedding-lm-step")
        if self.train_embedding_lm:
            args.append("--train-embedding-lm")
        if self.train_transformer_lm:
            args.append("--train-transformer-lm")
        if self.checkpoint_metadata_smoke:
            args.append("--checkpoint-metadata-smoke")
        if not self.write_checkpoint:
            args.append("--no-checkpoint")
        if str(self.cuda_runtime_lib or "").strip():
            args.extend(["--cuda-runtime-lib", self.cuda_runtime_lib])
        if self.activation == "moa":
            args.extend(["--native-cuda-moa-interval", str(int(self.moa_interval))])
        if str(self.template_name or "").strip():
            args.extend(["--template-name", str(self.template_name)])
        if str(self.graph_file or "").strip():
            args.extend(["--graph-file", str(self.graph_file)])
        return args

    def compiled_cli_command(self, cli: str | None = None) -> str:
        return shlex.join(self.compiled_cli_argv(cli))

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "argv": self.argv(),
            "command": self.command(),
            "launcher_argv": self.launcher_argv(),
            "launcher_command": self.launcher_command(),
            "compiled_cli_argv": self.compiled_cli_argv(),
            "compiled_cli_command": self.compiled_cli_command(),
        }


@dataclass(frozen=True)
class NativeGpt2CheckpointInfo:
    """Metadata read from a llm.kittens/NeuralFn native GPT checkpoint."""

    path: str
    version: int
    precision: str
    max_seq_len: int
    vocab_size: int
    num_layers: int
    num_heads: int
    channels: int
    padded_vocab_size: int
    parameter_count: int
    parameter_bytes: int
    expected_file_size: int
    actual_file_size: int
    size_matches: bool
    step: int | None = None
    done_marker: str | None = None
    done_marker_exists: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_native_gpt2_executable(value: str | None = None) -> str:
    requested = str(value or "").strip()
    if requested:
        return requested
    env_value = str(os.environ.get("NFN_NATIVE_GPT_TRAIN_BIN", "")).strip()
    if env_value:
        return env_value
    env_value = str(os.environ.get("NFN_NATIVE_GPT2_TRAIN_BIN", "")).strip()
    if env_value:
        return env_value
    default_path = Path(DEFAULT_NATIVE_GPT2_EXECUTABLE)
    return str(default_path if default_path.exists() else "train_gpt2cu")


def resolve_native_gpt2_launcher(value: str | None = None) -> str:
    requested = str(value or "").strip()
    if requested:
        return requested
    env_value = str(os.environ.get("NFN_NATIVE_GPT2_LAUNCHER", "")).strip()
    if env_value:
        return env_value
    repo_root = Path(__file__).resolve().parents[1]
    default_path = repo_root / DEFAULT_NATIVE_GPT2_LAUNCHER
    return str(default_path)


def resolve_native_gpt2_cli(value: str | None = None) -> str:
    requested = str(value or "").strip()
    if requested:
        return requested
    env_value = str(os.environ.get("NFN_NATIVE_GPT_CLI", "")).strip()
    if env_value:
        return env_value
    env_value = str(os.environ.get("NFN_NATIVE_GPT2_CLI", "")).strip()
    if env_value:
        return env_value
    repo_root = Path(__file__).resolve().parents[1]
    default_path = repo_root / DEFAULT_NATIVE_GPT2_CLI
    if default_path.exists():
        return str(default_path)
    legacy_path = repo_root / LEGACY_NATIVE_GPT2_CLI
    if legacy_path.exists():
        return str(legacy_path)
    return str(default_path)


def normalize_native_gpt2_encoding_name(encoding_name: str | None) -> str | None:
    normalized = str(encoding_name or "").strip().lower()
    if not normalized:
        return None
    resolved = RAW_TEXT_ENCODING_ALIASES.get(normalized)
    if resolved is None:
        allowed = ", ".join(sorted(RAW_TEXT_ENCODING_ALIASES))
        raise ValueError(
            f"Unsupported raw-text encoding override {encoding_name!r}. "
            f"Expected one of: {allowed}."
        )
    return resolved


def native_gpt2_encoding_vocab_size(encoding_name: str) -> int:
    normalized = normalize_native_gpt2_encoding_name(encoding_name)
    if normalized is None:
        raise ValueError("encoding_name must be non-empty")
    return RAW_TEXT_ENCODING_VOCAB_SIZES[normalized]


def _canonical_native_gpt2_model_family(model_family: str | None) -> str:
    normalized = str(model_family or "gpt").strip().lower().replace("_", "-")
    if normalized not in {"gpt", "gpt2", "gpt3"}:
        raise ValueError("native GPT model_family must be one of: gpt, gpt2, gpt3")
    return "gpt"


def native_gpt2_parameter_count(
    *,
    max_seq_len: int,
    padded_vocab_size: int,
    num_layers: int,
    channels: int,
) -> int:
    c = int(channels)
    l = int(num_layers)
    return int(
        int(padded_vocab_size) * c
        + int(max_seq_len) * c
        + l * c
        + l * c
        + l * 3 * c * c
        + l * 3 * c
        + l * c * c
        + l * c
        + l * c
        + l * c
        + l * 4 * c * c
        + l * 4 * c
        + l * c * 4 * c
        + l * c
        + c
        + c
    )


def _native_gpt2_checkpoint_step(path: Path) -> int | None:
    name = path.name
    if not (name.startswith("model_") and name.endswith(".bin")):
        return None
    raw = name.removeprefix("model_").removesuffix(".bin")
    if not raw.isdigit():
        return None
    return int(raw)


def read_native_gpt2_checkpoint_info(
    checkpoint_path: str | Path,
    *,
    validate_size: bool = True,
) -> NativeGpt2CheckpointInfo:
    path = Path(checkpoint_path).expanduser()
    with path.open("rb") as handle:
        header_bytes = handle.read(NATIVE_GPT2_CHECKPOINT_HEADER_BYTES)
    if len(header_bytes) != NATIVE_GPT2_CHECKPOINT_HEADER_BYTES:
        raise ValueError(f"Native GPT checkpoint header is truncated: {path}")
    header = struct.unpack("<" + "i" * NATIVE_GPT2_CHECKPOINT_HEADER_INTS, header_bytes)
    if int(header[0]) != NATIVE_GPT2_CHECKPOINT_MAGIC:
        raise ValueError(f"Not a native GPT checkpoint: {path}")
    version = int(header[1])
    if version not in NATIVE_GPT2_CHECKPOINT_VERSIONS:
        raise ValueError(f"Unsupported native GPT checkpoint version {version} in {path}")
    precision, bytes_per_param = NATIVE_GPT2_CHECKPOINT_VERSIONS[version]
    max_seq_len = int(header[2])
    vocab_size = int(header[3])
    num_layers = int(header[4])
    num_heads = int(header[5])
    channels = int(header[6])
    padded_vocab_size = int(header[7])
    parameter_count = native_gpt2_parameter_count(
        max_seq_len=max_seq_len,
        padded_vocab_size=padded_vocab_size,
        num_layers=num_layers,
        channels=channels,
    )
    parameter_bytes = int(parameter_count * bytes_per_param)
    expected_file_size = int(NATIVE_GPT2_CHECKPOINT_HEADER_BYTES + parameter_bytes)
    actual_file_size = int(path.stat().st_size) if path.exists() else 0
    size_matches = actual_file_size == expected_file_size
    if validate_size and not size_matches:
        raise ValueError(
            f"Bad native GPT checkpoint size for {path}: got {actual_file_size} bytes, "
            f"expected {expected_file_size} bytes"
        )
    step = _native_gpt2_checkpoint_step(path)
    done_marker = path.with_name(f"DONE_{step:08d}") if step is not None else None
    return NativeGpt2CheckpointInfo(
        path=str(path),
        version=version,
        precision=precision,
        max_seq_len=max_seq_len,
        vocab_size=vocab_size,
        num_layers=num_layers,
        num_heads=num_heads,
        channels=channels,
        padded_vocab_size=padded_vocab_size,
        parameter_count=parameter_count,
        parameter_bytes=parameter_bytes,
        expected_file_size=expected_file_size,
        actual_file_size=actual_file_size,
        size_matches=size_matches,
        step=step,
        done_marker=str(done_marker) if done_marker is not None else None,
        done_marker_exists=bool(done_marker is not None and done_marker.exists()),
    )


def is_native_gpt2_checkpoint(checkpoint_path: str | Path) -> bool:
    try:
        read_native_gpt2_checkpoint_info(checkpoint_path, validate_size=False)
    except (OSError, ValueError, struct.error):
        return False
    return True


def latest_native_gpt2_checkpoint(output_dir: str | Path) -> Path | None:
    root = Path(output_dir).expanduser()
    if not root.is_dir():
        return None
    steps: list[int] = []
    for marker in root.glob("DONE_*"):
        raw = marker.name.removeprefix("DONE_")
        if raw.isdigit():
            steps.append(int(raw))
    for step in sorted(steps, reverse=True):
        candidate = root / f"model_{step:08d}.bin"
        if candidate.exists() and is_native_gpt2_checkpoint(candidate):
            return candidate
    candidates = sorted(root.glob("model_*.bin"), key=lambda p: _native_gpt2_checkpoint_step(p) or -1, reverse=True)
    for candidate in candidates:
        if is_native_gpt2_checkpoint(candidate):
            return candidate
    return None


def _metadata_matches_native_encoding(dataset_meta: dict[str, Any], encoding_name: str) -> bool:
    normalized = normalize_native_gpt2_encoding_name(encoding_name) or "gpt2"
    expected_vocab = native_gpt2_encoding_vocab_size(normalized)
    if normalized.startswith("sp"):
        return (
            str(dataset_meta.get("tokenizer_name") or "").strip().lower() == normalized
            and int(dataset_meta.get("tokenizer_vocab_size") or 0) == expected_vocab
        )
    return (
        str(dataset_meta.get("tokenizer_encoding") or "").strip().lower() == normalized
        and int(dataset_meta.get("tokenizer_vocab_size") or 0) == expected_vocab
    )


def native_gpt2_activation(value: str | None) -> str:
    normalized = str(value or "gelu").strip().lower().replace("_", "-")
    aliases = {
        "gelu": "gelu",
        "relu": "relu",
        "silu": "silu",
        "swiglu": "swiglu",
        "geglu": "geglu",
        "relu2": "relu2",
        "prelu": "prelu",
        "sd-prelu": "sd-prelu",
        "sdprelu": "sd-prelu",
        "ensemble": "ensemble",
        "moa": "moa",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unsupported native GPT activation {value!r}; "
            f"expected one of {', '.join(sorted(set(aliases.values())))}."
        )
    return aliases[normalized]


def _normalize_native_gpt2_template_name(value: str | None) -> str:
    normalized = str(value or "gpt").strip().lower().replace("-", "_")
    return normalized or "gpt"


def _resolved_native_gpt2_template_name(value: str | None) -> str:
    normalized = _normalize_native_gpt2_template_name(value)
    return "gpt2" if normalized == "gpt" else normalized


def _native_gpt2_activation_for_template(template_name: str | None, activation: str | None) -> str:
    resolved = native_gpt2_activation(activation)
    if _resolved_native_gpt2_template_name(template_name) == "gpt2_moa" and resolved == "gelu":
        return "moa"
    return resolved


def native_gpt2_kernel_backend(value: str | None) -> str:
    normalized = str(value or "tile-cuda").strip().lower()
    if normalized not in {"llm-kittens", "tile-cuda"}:
        raise ValueError("native GPT kernel backend must be one of: llm-kittens, tile-cuda")
    return normalized


def resolve_native_gpt2_token_shards(
    dataset_name: str,
    *,
    dataset_path: Path,
    dataset_meta: dict[str, Any] | None = None,
    encoding_name: str = "gpt2",
    allow_train_as_val: bool = False,
) -> tuple[dict[str, Any], Path, Path]:
    encoding = normalize_native_gpt2_encoding_name(encoding_name) or "gpt2"
    meta = dict(dataset_meta or {})
    train_files = sorted(dataset_path.glob("fineweb_train_*.bin"))
    val_files = sorted(dataset_path.glob("fineweb_val_*.bin"))
    if (
        meta.get("data_format") != "uint16_shards"
        or not train_files
        or not _metadata_matches_native_encoding(meta, encoding)
    ):
        from server.dataset_manager import ensure_raw_text_token_cache

        meta = ensure_raw_text_token_cache(
            dataset_name,
            dataset_path=dataset_path,
            dataset_meta=dataset_meta,
            encoding_name=encoding,
        )
        train_files = sorted(dataset_path.glob("fineweb_train_*.bin"))
        val_files = sorted(dataset_path.glob("fineweb_val_*.bin"))
    if meta.get("data_format") != "uint16_shards" or not train_files:
        raise ValueError(
            "Native GPT training requires cached uint16 token shards. "
            f"Dataset {dataset_name!r} is not cached as uint16 for tokenizer {encoding!r}; "
            "use --tokgpt2 or a sentencepiece tokenizer with <=65536 tokens."
        )
    if not val_files:
        if not allow_train_as_val:
            raise ValueError(
                "Native GPT training requires validation token shards for live validation loss. "
                "Provide a validation file when caching the dataset, or pass "
                "--native-cuda-allow-train-val-fallback to reuse the train shard."
            )
        val_files = train_files
    return meta, train_files[0], val_files[0]


def build_native_gpt2_run_config(
    *,
    dataset_name: str,
    dataset_path: Path,
    dataset_meta: dict[str, Any] | None,
    encoding_name: str,
    executable: str | None,
    output_dir: Path,
    eval_every_steps: int,
    sample_every_steps: int,
    generate_tokens: int,
    checkpoint_every_steps: int,
    batch_size: int,
    seq_len: int,
    train_batch_tokens: int,
    learning_rate: float,
    min_lr: float | None,
    warmup_steps: int,
    weight_decay: float,
    max_steps: int,
    num_layers: int,
    activation: str,
    eval_batches: int = 1,
    eval_batch_size: int = 0,
    moa_interval: int = 50,
    kernel_backend: str = "tile-cuda",
    tile_ops_lib: str = "",
    smoke_tile_ops: bool = False,
    smoke_optimizer_step: bool = False,
    smoke_lm_step: bool = False,
    smoke_attention_step: bool = False,
    smoke_mlp_step: bool = False,
    smoke_norm_residual_step: bool = False,
    smoke_transformer_block_step: bool = False,
    smoke_transformer_lm_step: bool = False,
    smoke_embedding_lm_step: bool = False,
    train_embedding_lm: bool = False,
    train_transformer_lm: bool = True,
    checkpoint_metadata_smoke: bool = False,
    cuda_runtime_lib: str = "",
    lm_head_row_chunk_size: int = 4096,
    template_name: str = "gpt",
    graph_file: str = "",
    allow_train_as_val: bool = False,
    model_family: str = "gpt",
    write_checkpoint: bool = True,
) -> tuple[NativeGpt2RunConfig, dict[str, Any]]:
    meta, train_data, val_data = resolve_native_gpt2_token_shards(
        dataset_name,
        dataset_path=dataset_path,
        dataset_meta=dataset_meta,
        encoding_name=encoding_name,
        allow_train_as_val=allow_train_as_val,
    )
    lr = float(learning_rate)
    final_lr_fraction = 0.0 if min_lr is None else max(0.0, min(float(min_lr) / lr, 1.0))
    cfg = NativeGpt2RunConfig(
        executable=resolve_native_gpt2_executable(executable),
        train_data=str(train_data),
        val_data=str(val_data),
        output_dir=str(output_dir),
        model_family=_canonical_native_gpt2_model_family(model_family),
        model_descriptor=f"d{int(num_layers)}",
        eval_every_steps=max(1, int(eval_every_steps)),
        eval_batches=max(0, int(eval_batches)),
        eval_batch_size=max(0, int(eval_batch_size)),
        lm_head_row_chunk_size=max(1, int(lm_head_row_chunk_size)),
        sample_every_steps=max(1, int(sample_every_steps)),
        generate_tokens=max(1, int(generate_tokens)),
        checkpoint_every_steps=max(1, int(checkpoint_every_steps)),
        batch_size=int(batch_size),
        seq_len=int(seq_len),
        train_batch_tokens=int(train_batch_tokens),
        learning_rate=lr,
        final_lr_fraction=final_lr_fraction,
        warmup_steps=int(warmup_steps),
        weight_decay=float(weight_decay),
        max_steps=int(max_steps),
        activation=_native_gpt2_activation_for_template(template_name, activation),
        moa_interval=int(moa_interval),
        kernel_backend=native_gpt2_kernel_backend(kernel_backend),
        tile_ops_lib=str(tile_ops_lib or ""),
        smoke_tile_ops=bool(smoke_tile_ops),
        smoke_optimizer_step=bool(smoke_optimizer_step),
        smoke_lm_step=bool(smoke_lm_step),
        smoke_attention_step=bool(smoke_attention_step),
        smoke_mlp_step=bool(smoke_mlp_step),
        smoke_norm_residual_step=bool(smoke_norm_residual_step),
        smoke_transformer_block_step=bool(smoke_transformer_block_step),
        smoke_transformer_lm_step=bool(smoke_transformer_lm_step),
        smoke_embedding_lm_step=bool(smoke_embedding_lm_step),
        train_embedding_lm=bool(train_embedding_lm),
        train_transformer_lm=bool(train_transformer_lm),
        checkpoint_metadata_smoke=bool(checkpoint_metadata_smoke),
        cuda_runtime_lib=str(cuda_runtime_lib or ""),
        dataset_alias=str(dataset_path),
        template_name=_normalize_native_gpt2_template_name(template_name),
        graph_file=str(graph_file or ""),
        write_checkpoint=bool(write_checkpoint),
    )
    return cfg, meta


def build_native_gpt2_compiled_cli_run_config(
    *,
    dataset_alias: str,
    executable: str | None,
    output_dir: Path,
    eval_every_steps: int,
    sample_every_steps: int,
    generate_tokens: int,
    checkpoint_every_steps: int,
    batch_size: int,
    seq_len: int,
    train_batch_tokens: int,
    learning_rate: float,
    min_lr: float | None,
    warmup_steps: int,
    weight_decay: float,
    max_steps: int,
    num_layers: int,
    activation: str,
    eval_batches: int = 1,
    eval_batch_size: int = 0,
    moa_interval: int = 50,
    kernel_backend: str = "tile-cuda",
    tile_ops_lib: str = "",
    smoke_tile_ops: bool = False,
    smoke_optimizer_step: bool = False,
    smoke_lm_step: bool = False,
    smoke_attention_step: bool = False,
    smoke_mlp_step: bool = False,
    smoke_norm_residual_step: bool = False,
    smoke_transformer_block_step: bool = False,
    smoke_transformer_lm_step: bool = False,
    smoke_embedding_lm_step: bool = False,
    train_embedding_lm: bool = False,
    train_transformer_lm: bool = True,
    checkpoint_metadata_smoke: bool = False,
    cuda_runtime_lib: str = "",
    lm_head_row_chunk_size: int = 4096,
    template_name: str = "gpt",
    graph_file: str = "",
    model_family: str = "gpt",
    write_checkpoint: bool = True,
) -> NativeGpt2RunConfig:
    """Build a compiled-CLI handoff without Python-side token shard inspection."""

    lr = float(learning_rate)
    final_lr_fraction = 0.0 if min_lr is None else max(0.0, min(float(min_lr) / lr, 1.0))
    return NativeGpt2RunConfig(
        executable=resolve_native_gpt2_executable(executable),
        train_data="",
        val_data="",
        output_dir=str(output_dir),
        model_family=_canonical_native_gpt2_model_family(model_family),
        model_descriptor=f"d{int(num_layers)}",
        eval_every_steps=max(1, int(eval_every_steps)),
        eval_batches=max(0, int(eval_batches)),
        eval_batch_size=max(0, int(eval_batch_size)),
        lm_head_row_chunk_size=max(1, int(lm_head_row_chunk_size)),
        sample_every_steps=max(1, int(sample_every_steps)),
        generate_tokens=max(1, int(generate_tokens)),
        checkpoint_every_steps=max(1, int(checkpoint_every_steps)),
        batch_size=int(batch_size),
        seq_len=int(seq_len),
        train_batch_tokens=int(train_batch_tokens),
        learning_rate=lr,
        final_lr_fraction=final_lr_fraction,
        warmup_steps=int(warmup_steps),
        weight_decay=float(weight_decay),
        max_steps=int(max_steps),
        activation=_native_gpt2_activation_for_template(template_name, activation),
        moa_interval=int(moa_interval),
        kernel_backend=native_gpt2_kernel_backend(kernel_backend),
        tile_ops_lib=str(tile_ops_lib or ""),
        smoke_tile_ops=bool(smoke_tile_ops),
        smoke_optimizer_step=bool(smoke_optimizer_step),
        smoke_lm_step=bool(smoke_lm_step),
        smoke_attention_step=bool(smoke_attention_step),
        smoke_mlp_step=bool(smoke_mlp_step),
        smoke_norm_residual_step=bool(smoke_norm_residual_step),
        smoke_transformer_block_step=bool(smoke_transformer_block_step),
        smoke_transformer_lm_step=bool(smoke_transformer_lm_step),
        smoke_embedding_lm_step=bool(smoke_embedding_lm_step),
        train_embedding_lm=bool(train_embedding_lm),
        train_transformer_lm=bool(train_transformer_lm),
        checkpoint_metadata_smoke=bool(checkpoint_metadata_smoke),
        cuda_runtime_lib=str(cuda_runtime_lib or ""),
        dataset_alias=str(dataset_alias),
        template_name=_normalize_native_gpt2_template_name(template_name),
        graph_file=str(graph_file or ""),
        write_checkpoint=bool(write_checkpoint),
    )


def write_native_gpt2_run_config(
    config: NativeGpt2RunConfig,
    output_path: Path,
    *,
    runner: str = "auto",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **config.to_dict(),
        "runner": native_gpt2_runner_status(runner).to_dict(),
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _load_native_gpt2_binding():
    generic_binding_value = os.environ.get("NFN_NATIVE_GPT_BINDING")
    binding_env_name = "NFN_NATIVE_GPT_BINDING" if generic_binding_value is not None else "NFN_NATIVE_GPT2_BINDING"
    binding_enabled = str(
        generic_binding_value
        if generic_binding_value is not None
        else os.environ.get("NFN_NATIVE_GPT2_BINDING", "1")
    ).strip().lower()
    if binding_enabled in {"0", "false", "no", "off"}:
        raise ImportError(f"native GPT binding disabled by {binding_env_name}=0")
    errors: list[str] = []
    for module_name in NATIVE_GPT2_BINDING_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            errors.append(f"{module_name}: {exc}")
            continue
        runner = getattr(module, "run_gpt2", None) or getattr(module, "run_train", None)
        if callable(runner):
            return module_name, runner
        errors.append(f"{module_name}: missing run_gpt2(config_dict) or run_train(config_dict)")
    raise ImportError("; ".join(errors) if errors else "no native GPT binding modules configured")


def native_gpt2_runner_status(requested: str = "auto") -> NativeGpt2RunnerStatus:
    normalized = str(requested or "auto").strip().lower().replace("_", "-")
    if normalized not in {"auto", "binding", "compiled-cli", "launcher", "subprocess"}:
        raise ValueError("native GPT runner must be one of: auto, binding, compiled-cli, launcher, subprocess")
    if normalized == "subprocess":
        return NativeGpt2RunnerStatus(requested=normalized, resolved="subprocess")
    if normalized == "compiled-cli":
        cli = Path(resolve_native_gpt2_cli())
        return NativeGpt2RunnerStatus(
            requested=normalized,
            resolved="compiled-cli",
            available=cli.exists(),
            reason="" if cli.exists() else f"compiled native GPT CLI not found: {cli}",
        )
    if normalized == "launcher":
        launcher = Path(resolve_native_gpt2_launcher())
        return NativeGpt2RunnerStatus(
            requested=normalized,
            resolved="launcher",
            available=launcher.exists(),
            reason="" if launcher.exists() else f"compiled launcher not found: {launcher}",
        )
    try:
        module_name, _runner = _load_native_gpt2_binding()
    except ImportError as exc:
        if normalized == "binding":
            return NativeGpt2RunnerStatus(
                requested=normalized,
                resolved="binding",
                available=False,
                reason=str(exc),
            )
        cli = Path(resolve_native_gpt2_cli())
        if cli.exists():
            return NativeGpt2RunnerStatus(
                requested=normalized,
                resolved="compiled-cli",
                available=True,
                reason=f"native binding unavailable: {exc}",
            )
        launcher = Path(resolve_native_gpt2_launcher())
        if launcher.exists():
            return NativeGpt2RunnerStatus(
                requested=normalized,
                resolved="launcher",
                available=True,
                reason=f"native binding unavailable: {exc}",
            )
        return NativeGpt2RunnerStatus(
            requested=normalized,
            resolved="subprocess",
            available=True,
            reason=f"native binding unavailable and compiled native GPT CLI/launcher not found: {exc}",
        )
    return NativeGpt2RunnerStatus(
        requested=normalized,
        resolved="binding",
        binding_module=module_name,
    )


def run_native_gpt2(config: NativeGpt2RunConfig, *, runner: str = "auto") -> int:
    status = native_gpt2_runner_status(runner)
    if status.resolved == "binding":
        if not status.available:
            raise RuntimeError(f"Native GPT binding requested but unavailable: {status.reason}")
        _module_name, binding_runner = _load_native_gpt2_binding()
        return int(binding_runner(config.to_dict()))
    if status.resolved == "compiled-cli":
        if not status.available:
            raise RuntimeError(f"Native GPT compiled CLI requested but unavailable: {status.reason}")
        env = os.environ.copy()
        env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", config.cuda_device_max_connections)
        proc = subprocess.run(config.compiled_cli_argv(), env=env, check=False)
        return int(proc.returncode)
    if status.resolved == "launcher":
        if not status.available:
            raise RuntimeError(f"Native GPT launcher requested but unavailable: {status.reason}")
        env = os.environ.copy()
        env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", config.cuda_device_max_connections)
        proc = subprocess.run(config.launcher_argv(), env=env, check=False)
        return int(proc.returncode)

    env = os.environ.copy()
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", config.cuda_device_max_connections)
    proc = subprocess.run(config.argv(), env=env, check=False)
    return int(proc.returncode)
