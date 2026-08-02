from __future__ import annotations

import json
from pathlib import Path
import pickle
import struct
import subprocess
import zipfile

import pytest

from neuralfn.native_embedding import (
    compile_embedding_datasets,
    import_huggingface_embedding_model,
    prepare_embedding_training_command,
    read_embedding_checkpoint_header,
    stable_token_id,
    tokenize_embedding_text,
)
from neuralfn.native_embedding import _TorchStorageRef, _read_pytorch_zip_weights, _rebuild_torch_tensor


def _write_safetensors(path: Path, tensors: dict[str, tuple[list[int], list[float]]]) -> None:
    offset = 0
    header: dict[str, object] = {}
    chunks: list[bytes] = []
    for name, (shape, values) in tensors.items():
        chunk = struct.pack(f"<{len(values)}f", *values)
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [offset, offset + len(chunk)]}
        chunks.append(chunk)
        offset += len(chunk)
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(chunks))


def _matrix(rows: int, columns: int, start: float = 0.0) -> list[float]:
    return [start + (index % 17) * 0.001 for index in range(rows * columns)]


def _checkpoint_vectors(path: Path) -> list[list[float]]:
    payload = path.read_bytes()
    assert payload[:8] == b"NFNEMB2\0"
    offset = 8 + 16 * 4 + 4 * 4
    vectors: list[list[float]] = []
    while offset < len(payload):
        size = struct.unpack_from("<Q", payload, offset)[0]
        offset += 8
        values = list(struct.unpack_from(f"<{size}f", payload, offset)) if size else []
        offset += size * 4
        vectors.append(values)
    assert offset == len(payload)
    return vectors


def test_torch_free_pytorch_zip_state_dict_reader(tmp_path: Path) -> None:
    storage = _TorchStorageRef("0", "F32", 6)

    class FakeTensor:
        def __reduce__(self) -> object:
            return (_rebuild_torch_tensor, (storage, 1, [2, 2], [2, 1], False, None))

    tensor = FakeTensor()

    class StoragePickler(pickle.Pickler):
        def persistent_id(self, value: object) -> object:
            if value is storage:
                return ("storage", ("nfn-storage-type", "F32"), "0", "cpu", 6)
            return None

    import io

    stream = io.BytesIO()
    StoragePickler(stream, protocol=2).dump({"custom.weight": tensor})
    checkpoint = tmp_path / "pytorch_model.bin"
    with zipfile.ZipFile(checkpoint, "w") as archive:
        archive.writestr("archive/data.pkl", stream.getvalue())
        archive.writestr("archive/data/0", struct.pack("<6f", 0.0, 1.0, 2.0, 3.0, 4.0, 5.0))
    assert _read_pytorch_zip_weights(checkpoint) == {"custom.weight": ([2, 2], [1.0, 2.0, 3.0, 4.0])}


def _tiny_hf_bert(path: Path) -> None:
    path.mkdir()
    (path / "config.json").write_text(json.dumps({
        "model_type": "bert", "vocab_size": 11, "hidden_size": 4,
        "num_hidden_layers": 1, "num_attention_heads": 2,
        "intermediate_size": 8, "max_position_embeddings": 8,
    }), encoding="utf-8")
    (path / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\nauction\nbid\ninvoice\nrefund\n", encoding="utf-8")
    ones4, zeros4 = [1.0] * 4, [0.0] * 4
    prefix = "bert.encoder.layer.0"
    tensors = {
        "bert.embeddings.word_embeddings.weight": ([11, 4], _matrix(11, 4, 0.01)),
        "bert.embeddings.position_embeddings.weight": ([8, 4], _matrix(8, 4, 0.02)),
        "bert.embeddings.token_type_embeddings.weight": ([2, 4], _matrix(2, 4, 0.0)),
        "bert.embeddings.LayerNorm.weight": ([4], ones4),
        "bert.embeddings.LayerNorm.bias": ([4], zeros4),
        f"{prefix}.attention.output.LayerNorm.weight": ([4], ones4),
        f"{prefix}.attention.output.LayerNorm.bias": ([4], zeros4),
        f"{prefix}.output.LayerNorm.weight": ([4], ones4),
        f"{prefix}.output.LayerNorm.bias": ([4], zeros4),
    }
    for name, rows, columns in (
        ("attention.self.query", 4, 4), ("attention.self.key", 4, 4),
        ("attention.self.value", 4, 4), ("attention.output.dense", 4, 4),
        ("intermediate.dense", 8, 4), ("output.dense", 4, 8),
    ):
        tensors[f"{prefix}.{name}.weight"] = ([rows, columns], _matrix(rows, columns, 0.005))
        tensors[f"{prefix}.{name}.bias"] = ([rows], [0.0] * rows)
    _write_safetensors(path / "model.safetensors", tensors)


def _tiny_hf_gpt2(path: Path) -> None:
    path.mkdir()
    (path / "config.json").write_text(json.dumps({
        "model_type": "gpt2", "vocab_size": 11, "n_embd": 4,
        "n_layer": 1, "n_head": 2, "n_inner": 8, "n_positions": 8,
    }), encoding="utf-8")
    ones4, zeros4 = [1.0] * 4, [0.0] * 4
    prefix = "transformer.h.0"
    tensors = {
        "transformer.wte.weight": ([11, 4], _matrix(11, 4, 0.01)),
        "transformer.wpe.weight": ([8, 4], _matrix(8, 4, 0.02)),
        f"{prefix}.ln_1.weight": ([4], ones4), f"{prefix}.ln_1.bias": ([4], zeros4),
        f"{prefix}.ln_2.weight": ([4], ones4), f"{prefix}.ln_2.bias": ([4], zeros4),
        f"{prefix}.attn.c_attn.weight": ([4, 12], _matrix(4, 12, 0.003)),
        f"{prefix}.attn.c_attn.bias": ([12], [0.0] * 12),
        f"{prefix}.attn.c_proj.weight": ([4, 4], _matrix(4, 4, 0.004)),
        f"{prefix}.attn.c_proj.bias": ([4], zeros4),
        f"{prefix}.mlp.c_fc.weight": ([4, 8], _matrix(4, 8, 0.005)),
        f"{prefix}.mlp.c_fc.bias": ([8], [0.0] * 8),
        f"{prefix}.mlp.c_proj.weight": ([8, 4], _matrix(8, 4, 0.006)),
        f"{prefix}.mlp.c_proj.bias": ([4], zeros4),
        "transformer.ln_f.weight": ([4], ones4), "transformer.ln_f.bias": ([4], zeros4),
    }
    _write_safetensors(path / "model.safetensors", tensors)


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
    adapter_vectors = _checkpoint_vectors(tuned / "embedding_adapter.bin")
    assert len(adapter_vectors[11]) == 2 * 8  # layer-0 query LoRA A
    assert len(adapter_vectors[12]) == 8 * 2  # layer-0 query LoRA B

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


def test_pretraining_updates_transformer_attention_weights(tmp_path: Path, embedding_cli: Path) -> None:
    raw = tmp_path / "raw.txt"
    raw.write_text("one two three\nfour five six\n", encoding="utf-8")
    compiled = tmp_path / "compiled.tsv"
    compile_embedding_datasets({"datasets": [{"source": str(raw), "objective": "raw"}]}, compiled, vocab_size=31, max_tokens=4)
    common = [
        "--embedding-data", str(compiled), "--embedding-vocab-size", "31", "--hidden-dim", "8",
        "--num-layers", "2", "--num-heads", "2", "--intermediate-dim", "16",
        "--embedding-dim", "8", "--max-seq-len", "4", "--batch-size", "1",
        "--effective-batch-size", "1", "--progress-every-steps", "0", "--learning-rate", "0.01",
    ]
    one_step, two_steps = tmp_path / "one", tmp_path / "two"
    assert subprocess.run([str(embedding_cli), *common, "--max-steps", "1", "--output-dir", str(one_step)], check=False).returncode == 0
    assert subprocess.run([str(embedding_cli), *common, "--max-steps", "2", "--output-dir", str(two_steps)], check=False).returncode == 0
    first_vectors = _checkpoint_vectors(one_step / "embedding_model.bin")
    second_vectors = _checkpoint_vectors(two_steps / "embedding_model.bin")
    assert first_vectors[9] != second_vectors[9]  # layer-0 query matrix, not the projection head
    metadata = json.loads((two_steps / "embedding_model.json").read_text(encoding="utf-8"))
    assert metadata["encoder_core"] == "native_transformer_biencoder"
    assert metadata["num_layers"] == 2


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


def test_imports_hf_bert_transformer_and_uses_its_tokenizer(tmp_path: Path, embedding_cli: Path) -> None:
    hf_model = tmp_path / "hf-bert"
    _tiny_hf_bert(hf_model)
    imported = tmp_path / "imported" / "embedding_model.bin"
    metadata = import_huggingface_embedding_model(str(hf_model), imported, pooling="cls")
    assert metadata["architecture"] == "bert"
    assert metadata["tensor_count"] > 10
    assert read_embedding_checkpoint_header(imported) == {
        "version": 2,
        "vocab_size": 11,
        "hidden_dim": 4,
        "output_dim": 4,
        "max_tokens": 8,
        "step": 0,
        "adapter_type": 0,
        "num_layers": 1,
        "num_heads": 2,
        "intermediate_dim": 8,
        "mask_token_id": 4,
    }
    inferred = subprocess.run(
        [str(embedding_cli), "--checkpoint", str(imported), "--embed-token-ids", "2,5,6,3"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert inferred.returncode == 0, inferred.stderr
    assert len(json.loads(inferred.stdout)["embedding"]) == 4

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world\nauction bid\ninvoice refund\n", encoding="utf-8")
    output = tmp_path / "trained"
    command, compiled = prepare_embedding_training_command(
        [
            str(embedding_cli), "--embedding-dataset", str(corpus), "--output-dir", str(output),
            "--embedding-hf-model", str(hf_model), "--embedding-stage", "finetune",
            "--batch-size", "1", "--effective-batch-size", "1", "--max-steps", "1",
            "--progress-every-steps", "0",
        ],
        repo_root=ROOT,
    )
    assert compiled is not None and compiled["hf_import"]["hidden_dim"] == 4
    assert "--embedding-hf-model" not in command
    assert command[command.index("--num-layers") + 1] == "1"
    prepared_text = Path(compiled["path"]).read_text(encoding="utf-8")
    assert "2,5,6,3" in prepared_text
    trained = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    assert trained.returncode == 0, trained.stderr
    assert read_embedding_checkpoint_header(output)["num_layers"] == 1

    precompiled_command, precompiled_metadata = prepare_embedding_training_command(
        [
            str(embedding_cli), "--embedding-data", str(compiled["path"]),
            "--output-dir", str(tmp_path / "precompiled"), "--embedding-hf-model", str(hf_model),
        ],
        repo_root=ROOT,
    )
    assert "--embedding-data" in precompiled_command
    assert "--embedding-hf-model" not in precompiled_command
    assert "--base-checkpoint" in precompiled_command
    assert precompiled_metadata is not None and "hf_import" in precompiled_metadata


def test_imports_hf_gpt2_conv1d_weights_as_causal_encoder(tmp_path: Path, embedding_cli: Path) -> None:
    hf_model = tmp_path / "hf-gpt2"
    _tiny_hf_gpt2(hf_model)
    imported = tmp_path / "gpt-import" / "embedding_model.bin"
    metadata = import_huggingface_embedding_model(str(hf_model), imported, pooling="last")
    assert metadata["architecture"] == "gpt-derived"
    header = read_embedding_checkpoint_header(imported)
    assert (header["num_layers"], header["num_heads"], header["intermediate_dim"]) == (1, 2, 8)
    inferred = subprocess.run(
        [str(embedding_cli), "--checkpoint", str(imported), "--embed-token-ids", "5,6,7"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert inferred.returncode == 0, inferred.stderr
    payload = json.loads(inferred.stdout)
    assert payload["normalized"] is True
    assert len(payload["embedding"]) == 4
