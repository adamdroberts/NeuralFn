from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from neuralfn.native_embedding import (
    compile_embedding_datasets,
    read_embedding_checkpoint_header,
    stable_token_id,
    tokenize_embedding_text,
)


ROOT = Path(__file__).resolve().parents[1]


def test_stable_embedding_tokenizer_uses_uint32_safe_ids() -> None:
    assert stable_token_id("NeuralFn", 200_019) == stable_token_id("NeuralFn", 200_019)
    ids = tokenize_embedding_text("one two three", vocab_size=200_019, max_tokens=2)
    assert len(ids) == 2
    assert all(3 <= item < 200_019 for item in ids)


def test_compile_embedding_manifest_supports_multiple_objectives_and_weights(tmp_path: Path) -> None:
    raw = tmp_path / "raw.txt"
    raw.write_text("alpha topic\nbeta topic\n", encoding="utf-8")
    retrieval = tmp_path / "retrieval.jsonl"
    retrieval.write_text(
        json.dumps({"query": "alpha", "positive": "alpha document", "negatives": ["beta document"]}) + "\n",
        encoding="utf-8",
    )
    similarity = tmp_path / "similarity.csv"
    similarity.write_text("left,right,value\na,b,4\n", encoding="utf-8")
    classes = tmp_path / "classes.json"
    classes.write_text(json.dumps([{"body": "alpha", "topic": "a"}, {"body": "beta", "topic": "b"}]), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "datasets": [
                    {"name": "raw", "source": raw.name, "objective": "raw", "weight": 2},
                    {"name": "retrieval", "source": retrieval.name, "objective": "retrieval"},
                    {
                        "name": "similarity",
                        "source": similarity.name,
                        "objective": "similarity",
                        "columns": {"sentence1": "left", "sentence2": "right", "score": "value"},
                        "score_min": 0,
                        "score_max": 5,
                    },
                    {
                        "name": "class",
                        "source": classes.name,
                        "objective": "class",
                        "columns": {"text": "body", "label": "topic"},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "compiled.tsv"
    metadata = compile_embedding_datasets(manifest, output, vocab_size=100_003, max_tokens=16)

    assert metadata["records"] == 6
    assert metadata["objectives"] == {"raw": 2, "retrieval": 1, "similarity": 1, "class": 2}
    assert metadata["datasets"][0]["weight"] == 2.0
    assert output.read_text(encoding="utf-8").startswith("# nfn_embedding_indexed_v1\tvocab=100003")
    assert output.with_suffix(".tsv.json").is_file()


@pytest.fixture(scope="module")
def embedding_cli(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("native-embedding-build") / "nfn_embedding_native_train"
    proc = subprocess.run(
        ["bash", str(ROOT / "tools" / "build_native_embedding_cli.sh"), str(output)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return output


def test_native_embedding_pretrain_checkpoint_inference_and_exact_resume(tmp_path: Path, embedding_cli: Path) -> None:
    raw = tmp_path / "raw.txt"
    raw.write_text("alpha auction bidding\nbeta invoice payment\nalpha lot catalog\nbeta card refund\n", encoding="utf-8")
    compiled = tmp_path / "compiled.tsv"
    compile_embedding_datasets(
        {"datasets": [{"source": str(raw), "objective": "raw"}]},
        compiled,
        vocab_size=257,
        max_tokens=8,
    )
    output = tmp_path / "model"
    command = [
        str(embedding_cli),
        "--embedding-data", str(compiled),
        "--output-dir", str(output),
        "--embedding-vocab-size", "257",
        "--hidden-dim", "16",
        "--embedding-dim", "8",
        "--max-seq-len", "8",
        "--batch-size", "2",
        "--effective-batch-size", "4",
        "--max-steps", "2",
        "--checkpoint-every-steps", "1",
        "--progress-every-steps", "1",
    ]
    trained = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    assert trained.returncode == 0, trained.stderr
    payload = json.loads(trained.stdout)
    assert payload["status"] == "native-embedding-trained"
    assert payload["model_type"] == "text_embedding"
    checkpoint = output / "embedding_model.bin"
    assert checkpoint.is_file()
    assert (output / "embedding_optimizer.bin").is_file()
    assert (output / "DONE").is_file()
    assert read_embedding_checkpoint_header(checkpoint)["output_dim"] == 8

    inferred = subprocess.run(
        [str(embedding_cli), "--checkpoint", str(checkpoint), "--embed-text", "alpha auction"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert inferred.returncode == 0, inferred.stderr
    vector = json.loads(inferred.stdout)["embedding"]
    assert len(vector) == 8
    assert sum(value * value for value in vector) == pytest.approx(1.0, abs=1e-5)

    straight = tmp_path / "straight"
    straight_command = list(command)
    straight_command[straight_command.index(str(output))] = str(straight)
    straight_command[straight_command.index("2", straight_command.index("--max-steps"))] = "3"
    straight_run = subprocess.run(straight_command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    assert straight_run.returncode == 0, straight_run.stderr

    resumed = subprocess.run(
        [
            str(embedding_cli), "--embedding-data", str(compiled), "--output-dir", str(output),
            "--resume-from-checkpoint", str(output), "--embedding-stage", "resume",
            "--embedding-vocab-size", "257", "--hidden-dim", "16", "--embedding-dim", "8",
            "--max-seq-len", "8", "--batch-size", "2", "--effective-batch-size", "4",
            "--max-steps", "1", "--progress-every-steps", "0",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert resumed.returncode == 0, resumed.stderr
    assert json.loads(resumed.stdout)["steps_completed"] == 3
    assert (output / "embedding_model.bin").read_bytes() == (straight / "embedding_model.bin").read_bytes()


def test_native_embedding_lora_saves_adapter_and_merged_models(tmp_path: Path, embedding_cli: Path) -> None:
    raw = tmp_path / "raw.txt"
    raw.write_text("one topic\ntwo topic\n", encoding="utf-8")
    compiled = tmp_path / "compiled.tsv"
    compile_embedding_datasets({"datasets": [{"source": str(raw), "objective": "raw"}]}, compiled, vocab_size=67, max_tokens=4)
    base = tmp_path / "base"
    common = [
        "--embedding-data", str(compiled), "--embedding-vocab-size", "67", "--hidden-dim", "8",
        "--embedding-dim", "4", "--max-seq-len", "4", "--batch-size", "2",
        "--effective-batch-size", "2", "--max-steps", "1", "--progress-every-steps", "0",
    ]
    assert subprocess.run([str(embedding_cli), *common, "--output-dir", str(base)], check=False).returncode == 0
    tuned = tmp_path / "tuned"
    proc = subprocess.run(
        [str(embedding_cli), *common, "--output-dir", str(tuned), "--embedding-stage", "finetune", "--base-checkpoint", str(base), "--adapter-type", "lora", "--lora-rank", "2"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert read_embedding_checkpoint_header(tuned / "embedding_model.bin")["adapter_type"] == 0
    assert read_embedding_checkpoint_header(tuned / "embedding_adapter.bin")["adapter_type"] == 1

    quantized = tmp_path / "quantized"
    qproc = subprocess.run(
        [str(embedding_cli), *common, "--output-dir", str(quantized), "--embedding-stage", "finetune", "--base-checkpoint", str(base), "--adapter-type", "qlora", "--lora-rank", "2"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert qproc.returncode == 0, qproc.stderr
    assert read_embedding_checkpoint_header(quantized / "embedding_adapter.bin")["adapter_type"] == 2
    metadata = json.loads((quantized / "embedding_model.json").read_text(encoding="utf-8"))
    assert metadata["base_weight_quantization"] == "nf4-group64"


def test_native_retrieval_posttraining_ranks_positive_above_hard_negative(tmp_path: Path, embedding_cli: Path) -> None:
    records = tmp_path / "retrieval.jsonl"
    rows = [
        {"query": "red apple fruit", "positive": "fresh red apple fruit", "negatives": ["blue motor vehicle"]},
        {"query": "blue motor vehicle", "positive": "fast blue motor vehicle", "negatives": ["fresh red apple fruit"]},
    ]
    records.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    compiled = tmp_path / "compiled.tsv"
    compile_embedding_datasets(
        {"datasets": [{"source": str(records), "objective": "retrieval"}]},
        compiled,
        vocab_size=257,
        max_tokens=8,
    )
    output = tmp_path / "retrieval-model"
    proc = subprocess.run(
        [
            str(embedding_cli), "--embedding-data", str(compiled), "--output-dir", str(output),
            "--embedding-stage", "posttrain", "--embedding-vocab-size", "257", "--hidden-dim", "16",
            "--embedding-dim", "8", "--max-seq-len", "8", "--batch-size", "2",
            "--effective-batch-size", "2", "--max-steps", "80", "--learning-rate", "0.01",
            "--warmup-steps", "1", "--progress-every-steps", "0",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    def embed(text: str) -> list[float]:
        result = subprocess.run(
            [str(embedding_cli), "--checkpoint", str(output / "embedding_model.bin"), "--embed-text", text],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)["embedding"]

    query = embed("red apple fruit")
    positive = embed("fresh red apple fruit")
    negative = embed("blue motor vehicle")
    positive_score = sum(left * right for left, right in zip(query, positive))
    negative_score = sum(left * right for left, right in zip(query, negative))
    assert positive_score > negative_score
