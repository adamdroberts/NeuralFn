#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neuralfn.native_train import NATIVE_TRAIN_FAMILY_TARGETS  # noqa: E402
from tools.smoke_native_family_template_checkpoints import select_templates  # noqa: E402


LOOP_FLAGS = {
    "llama": "--train-llama-dataset-loop",
    "mixllama": "--train-moe-dataset-loop",
    "deepseek-v4": "--train-moe-dataset-loop",
    "moe-jepa-evo": "--train-moe-jepa-dataset-loop",
    "jepa": "--train-dense-jepa-dataset-loop",
    "semantic-dense-jepa": "--train-semantic-dense-jepa-dataset-loop",
    "semantic-router-moe": "--train-semantic-router-moe-dataset-loop",
    "jamba": "--train-jamba-dataset-loop",
    "seq2seq": "--train-seq2seq-dataset-loop",
    "diffusion": "--train-diffusion-dataset-loop",
    "ttt-llama": "--train-ttt-dataset-loop",
    "universal-llama": "--train-universal-dataset-loop",
    "hnet-lm": "--train-hnet-dataset-loop",
}

SMALL_TOKEN_DATASET = Path("/tmp/nfn-diffusion-token-shards")
SMALL_TOKEN_VOCAB_SIZE = 256
GPT2_TOKEN_VOCAB_SIZE = 50257


def resolve_dataset_argument(value: str) -> str:
    path = Path(value).expanduser()
    if path.exists() or "/" in value or value.startswith("."):
        return str(path.resolve())
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one real native GPU training step for each family template.")
    parser.add_argument("--native-bin-dir", default=str(ROOT / "build"))
    parser.add_argument("--tile-ops-lib", default=str(ROOT / "build/libnfn_native_train_tile_ops.so"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--templates", default="")
    parser.add_argument(
        "--dataset-alias",
        default="",
        help=(
            "Token dataset alias/path for non-HNet templates. Defaults to the "
            "local small uint16 shard when present, otherwise the native "
            "TinyStories/GPT-2 token cache."
        ),
    )
    parser.add_argument("--byte-dataset", default="")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    native_bin_dir = Path(args.native_bin_dir).expanduser().resolve()
    tile_ops_lib = Path(args.tile_ops_lib).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    templates = select_templates(args.templates)
    token_dataset_alias = args.dataset_alias
    token_vocab_size = GPT2_TOKEN_VOCAB_SIZE
    if not token_dataset_alias and SMALL_TOKEN_DATASET.exists():
        token_dataset_alias = str(SMALL_TOKEN_DATASET)
        token_vocab_size = SMALL_TOKEN_VOCAB_SIZE
    results: list[dict[str, Any]] = []

    for template_name, family in templates.items():
        target = NATIVE_TRAIN_FAMILY_TARGETS.get(family)
        loop_flag = LOOP_FLAGS.get(family)
        template_dir = output_dir / template_name
        template_dir.mkdir(parents=True, exist_ok=True)
        log_path = template_dir / "train.log"
        if not target or not loop_flag:
            result = {"template_name": template_name, "family": family, "passed": False,
                      "error": "missing native target or dataset-loop mapping"}
            results.append(result)
            if not args.keep_going:
                break
            continue
        command = [
            str(native_bin_dir / target), loop_flag,
            "--template-name", template_name,
            "--tile-ops-lib", str(tile_ops_lib),
            "--max-steps", "1", "--batch-size", "1",
            "--train-seq-len", "8", "--train-batch-tokens", "8",
            "--model-dim", "64", "--hidden-dim", "128",
            "--num-heads", "1", "--num-kv-heads", "1",
            "--vocab-size", str(token_vocab_size), "--padded-vocab-size", str(token_vocab_size),
            "--ttt-hidden-dim", "16", "--max-recurrence-steps", "2",
            "--byte-patch-size", "4", "--byte-patch-stride", "4",
            "--checkpoint-every-steps", "1", "--progress-every-steps", "1",
            "--output-dir", str(template_dir),
        ]
        if family == "hnet-lm":
            if not args.byte_dataset:
                result = {"template_name": template_name, "family": family, "passed": False,
                          "error": "--byte-dataset is required for HNet templates"}
                results.append(result)
                if not args.keep_going:
                    break
                continue
            command.extend(["--dataset-alias", resolve_dataset_argument(args.byte_dataset)])
        elif token_dataset_alias:
            command.extend(["--dataset-alias", resolve_dataset_argument(token_dataset_alias)])
        proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        log_path.write_text(proc.stdout, encoding="utf-8")
        parameter_files = sorted(template_dir.glob("*_native_family_parameters_*.f32"))
        model_files = sorted(template_dir.glob("*_native_family_model_*.json"))
        passed = proc.returncode == 0 and bool(parameter_files) and bool(model_files)
        result = {
            "template_name": template_name,
            "family": family,
            "target": target,
            "returncode": proc.returncode,
            "passed": passed,
            "parameter_checkpoint_count": len(parameter_files),
            "model_checkpoint_count": len(model_files),
            "log_path": str(log_path),
            "token_dataset_alias": token_dataset_alias or "default-native-dataset",
            "token_vocab_size": token_vocab_size,
        }
        if not passed:
            result["output_tail"] = proc.stdout[-4000:]
        results.append(result)
        if not passed and not args.keep_going:
            break

    passed_count = sum(bool(result["passed"]) for result in results)
    passed = len(results) == len(templates) and passed_count == len(templates)
    payload = {
        "status": "native-family-live-training-sweep",
        "template_count": len(templates),
        "result_count": len(results),
        "passed_count": passed_count,
        "passed": passed,
        "results": results,
    }
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(f"Native family live training sweep: {passed_count}/{len(templates)} passed")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
