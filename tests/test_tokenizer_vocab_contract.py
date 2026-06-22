from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from neuralfn.torch_backend import TorchTrainConfig, TorchTrainer
from neuralfn.torch_templates import build_gpt_root_graph, build_model_spec_from_config
from server import dataset_manager as dm
from server.models import ExecuteRequest, LoadDatasetRequest
from server.services.graph_ops import GraphOperationError, load_dataset_source_into_graph, trace_torch_graph

HARNESS_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "cli" / "scripts"


class _FakeO200kEncoding:
    n_vocab = 200019
    special_tokens_set = {"<|endoftext|>"}

    def encode(self, text: str, *, allowed_special=None):
        del allowed_special
        tokens: list[int] = []
        if text.startswith("<|endoftext|>"):
            tokens.append(199999)
            text = text[len("<|endoftext|>") :]
        tokens.extend(1000 + ord(ch) for ch in text)
        return tokens

    def decode(self, token_ids):
        pieces = []
        for token_id in token_ids:
            if int(token_id) == 13:
                pieces.append(".")
            elif int(token_id) == 326:
                pieces.append(" and")
            elif int(token_id) == 199999:
                pieces.append("<|endoftext|>")
            elif int(token_id) >= 1000:
                pieces.append(chr(int(token_id) - 1000))
            else:
                pieces.append(str(token_id))
        return "".join(pieces)

    def decode_single_token_bytes(self, token_id: int) -> bytes:
        return self.decode([token_id]).encode("utf-8")


@pytest.fixture(autouse=True)
def _offline_o200k(monkeypatch: pytest.MonkeyPatch) -> None:
    original = dm.resolve_tiktoken_encoding

    def fake_resolve(encoding_name: str):
        if str(encoding_name) == "o200k_base":
            return _FakeO200kEncoding()
        return original(encoding_name)

    monkeypatch.setattr(dm, "resolve_tiktoken_encoding", fake_resolve)


def _load_harness_module(module_name: str, file_name: str):
    module_path = HARNESS_SCRIPTS_DIR / file_name
    scripts_dir = str(HARNESS_SCRIPTS_DIR)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load harness module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _assert_call_includes(actual: dict[str, object], expected: dict[str, object]) -> None:
    assert {key: actual.get(key) for key in expected} == expected


def _write_tokenizer_backed_alias(
    root: Path,
    *,
    name: str,
    tokenizer_vocab_size: int,
    shard_values: list[int],
) -> Path:
    ds_dir = root / name
    tokenizers_dir = ds_dir / "tokenizers"
    tokenizers_dir.mkdir(parents=True, exist_ok=True)
    np.array(shard_values, dtype=np.uint16).tofile(ds_dir / "fineweb_train_000000.bin")
    vocab_lines = [f"tok_{idx}\t0" for idx in range(tokenizer_vocab_size)]
    (tokenizers_dir / "toy.vocab").write_text("\n".join(vocab_lines), encoding="utf-8")
    meta = {
        "source": "huggingface_cached_tokens",
        "variant": "sp-test",
        "tokenizer_name": "toy",
        "tokenizer_files": ["toy.vocab"],
        "data_format": "uint16_shards",
    }
    (ds_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return ds_dir


def _make_graph(*, vocab_size: int, preset: str = "mixllama_fast"):
    spec = build_model_spec_from_config(
        {"preset": preset, "vocab_size": vocab_size},
        preview_defaults=True,
    )
    graph = build_gpt_root_graph(name="vocab_contract_probe", model_spec=spec)
    graph.torch_config = {**graph.torch_config, "device": "cpu"}
    return graph


def _load_infer_jepa_module():
    return _load_harness_module("infer_jepa_semantic_test_module", "infer_jepa_semantic.py")


def _load_train_jepa_module():
    return _load_harness_module("train_jepa_semantic", "train_jepa_semantic.py")


def _load_infer_mixllama_module():
    _load_harness_module("train_jepa_semantic", "train_jepa_semantic.py")
    _load_harness_module("infer_jepa_semantic", "infer_jepa_semantic.py")
    return _load_harness_module("infer_mixllama_fast_test_module", "infer_mixllama_fast.py")


def _load_infer_llama_module():
    _load_harness_module("train_jepa_semantic", "train_jepa_semantic.py")
    _load_harness_module("infer_jepa_semantic", "infer_jepa_semantic.py")
    return _load_harness_module("infer_llama_fast_test_module", "infer_llama_fast.py")


def _load_infer_llama_megakernel_module():
    _load_harness_module("train_jepa_semantic", "train_jepa_semantic.py")
    _load_harness_module("infer_jepa_semantic", "infer_jepa_semantic.py")
    _load_harness_module("infer_llama_fast", "infer_llama_fast.py")
    return _load_harness_module("infer_llama_megakernel_test_module", "infer_llama_megakernel.py")


def _patch_cuda_generator_to_cpu(module, monkeypatch: pytest.MonkeyPatch) -> None:
    cpu_generator = torch.Generator
    monkeypatch.setattr(module.torch, "Generator", lambda *_, **__: cpu_generator(device="cpu"))


def test_validate_cached_tokenizer_contract_rejects_model_vocab_mismatch(tmp_path: Path) -> None:
    ds_name = "toy_alias"
    _write_tokenizer_backed_alias(
        tmp_path,
        name=ds_name,
        tokenizer_vocab_size=8,
        shard_values=[0, 1, 2, 3, 4, 5, 6, 7],
    )

    with pytest.raises(dm.DatasetTokenizerMismatchError, match="Model/checkpoint vocab size: 16"):
        dm.validate_cached_tokenizer_contract(
            ds_name,
            dataset_path=tmp_path / ds_name,
            dataset_meta=dm._load_dataset_meta(tmp_path / ds_name),
            model_vocab_size=16,
        )


def test_torch_trainer_fails_fast_for_tokenizer_backed_dataset_vocab_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ds_name = "toy_alias"
    _write_tokenizer_backed_alias(
        tmp_path,
        name=ds_name,
        tokenizer_vocab_size=8,
        shard_values=[0, 1, 2, 3, 4, 5, 6, 7],
    )
    monkeypatch.setattr(dm, "DATASETS_DIR", tmp_path)

    graph = _make_graph(vocab_size=16)
    load_dataset_source_into_graph(graph, LoadDatasetRequest(dataset_names=[ds_name], seq_len=4))
    trainer = TorchTrainer(graph, TorchTrainConfig(epochs=1, batch_size=1, device="cpu"))

    with pytest.raises(dm.DatasetTokenizerMismatchError, match="Model/checkpoint vocab size: 16"):
        trainer.train([], [])


def test_torch_trainer_fails_fast_for_tokenizer_backed_dataset_vocab_mismatch_dense(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ds_name = "toy_alias"
    _write_tokenizer_backed_alias(
        tmp_path,
        name=ds_name,
        tokenizer_vocab_size=8,
        shard_values=[0, 1, 2, 3, 4, 5, 6, 7],
    )
    monkeypatch.setattr(dm, "DATASETS_DIR", tmp_path)

    graph = _make_graph(vocab_size=16, preset="llama_fast")
    load_dataset_source_into_graph(graph, LoadDatasetRequest(dataset_names=[ds_name], seq_len=4))
    trainer = TorchTrainer(graph, TorchTrainConfig(epochs=1, batch_size=1, device="cpu"))

    with pytest.raises(dm.DatasetTokenizerMismatchError, match="Model/checkpoint vocab size: 16"):
        trainer.train([], [])


def test_trace_torch_graph_fails_fast_for_tokenizer_backed_dataset_vocab_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ds_name = "toy_alias"
    _write_tokenizer_backed_alias(
        tmp_path,
        name=ds_name,
        tokenizer_vocab_size=8,
        shard_values=[0, 1, 2, 3, 4, 5, 6, 7],
    )
    monkeypatch.setattr(dm, "DATASETS_DIR", tmp_path)

    graph = _make_graph(vocab_size=16)
    load_dataset_source_into_graph(graph, LoadDatasetRequest(dataset_names=[ds_name], seq_len=4))

    with pytest.raises(GraphOperationError, match="Model/checkpoint vocab size: 16"):
        trace_torch_graph(graph, ExecuteRequest())


def test_inference_decode_tokens_reports_out_of_range_cleanly() -> None:
    infer_module = _load_infer_jepa_module()

    class FakeTokenizer:
        def decode(self, _token_ids):
            raise IndexError("Out of range")

        def get_piece_size(self):
            return 8

    with pytest.raises(ValueError, match="token id 9 is out of range"):
        infer_module.decode_tokens(FakeTokenizer(), [9])


def test_inference_describe_token_decodes_tiktoken_piece() -> None:
    infer_module = _load_infer_jepa_module()
    encoding = dm.resolve_tiktoken_encoding("o200k_base")

    assert infer_module.describe_token(encoding, 13) == "."
    assert infer_module.describe_token(encoding, 326) == " and"


def test_inference_log_tokenizer_status_reports_tiktoken_vocab() -> None:
    infer_module = _load_infer_jepa_module()
    encoding = dm.resolve_tiktoken_encoding("o200k_base")
    messages: list[str] = []

    infer_module.log_tokenizer_status(
        messages.append,
        encoding,
        dm.local_tiktoken_encoding_path("o200k_base"),
        "o200k_base",
    )

    assert len(messages) == 1
    assert "o200k_base tiktoken" in messages[0]
    assert "vocab=" in messages[0]


def test_inference_preflight_rejects_model_vocab_mismatch(tmp_path: Path) -> None:
    infer_module = _load_infer_jepa_module()
    ds_name = "toy_alias"
    ds_dir = _write_tokenizer_backed_alias(
        tmp_path,
        name=ds_name,
        tokenizer_vocab_size=8,
        shard_values=[0, 1, 2, 3, 4, 5, 6, 7],
    )
    graph = _make_graph(vocab_size=8)
    state_dict = {
        "node_modules.model.node_modules.token_embed.embedding.weight": torch.zeros((16, 4)),
        "node_modules.model.node_modules.lm_head.proj.weight": torch.zeros((16, 4)),
    }

    with pytest.raises(dm.DatasetTokenizerMismatchError, match="Model/checkpoint vocab size: 16"):
        infer_module.validate_inference_vocab_contract(
            dataset_name=ds_name,
            dataset_path=ds_dir,
            dataset_meta=dm._load_dataset_meta(ds_dir),
            graph=graph,
            state_dict=state_dict,
        )


def test_inference_preflight_rejects_raw_text_o200k_vocab_mismatch(tmp_path: Path) -> None:
    infer_module = _load_infer_jepa_module()
    ds_name = "tiny_raw"
    ds_dir = tmp_path / ds_name
    ds_dir.mkdir(parents=True, exist_ok=True)
    (ds_dir / "data.txt").write_text("<|endoftext|> TinyStories", encoding="utf-8")
    dataset_meta = {
        "source": "huggingface",
        "hf_path": "roneneldan/TinyStories",
        "tokenizer_encoding": "o200k_base",
        "tokenizer_vocab_size": dm.raw_text_encoding_vocab_size("o200k_base"),
    }
    graph = _make_graph(vocab_size=1024, preset="llama_fast")
    state_dict = {
        "node_modules.model.node_modules.token_embed.embedding.weight": torch.zeros((1024, 4)),
        "node_modules.model.node_modules.lm_head.proj.weight": torch.zeros((1024, 4)),
    }

    with pytest.raises(RuntimeError, match="o200k_base"):
        infer_module.validate_inference_vocab_contract(
            dataset_name=ds_name,
            dataset_path=ds_dir,
            dataset_meta=dataset_meta,
            graph=graph,
            state_dict=state_dict,
            raw_text_encoding_name="o200k_base",
        )


def test_inference_preflight_accepts_matching_raw_text_o200k_vocab(tmp_path: Path) -> None:
    infer_module = _load_infer_jepa_module()
    ds_name = "tiny_raw"
    ds_dir = tmp_path / ds_name
    ds_dir.mkdir(parents=True, exist_ok=True)
    (ds_dir / "data.txt").write_text("<|endoftext|> TinyStories", encoding="utf-8")
    expected_vocab = dm.raw_text_encoding_vocab_size("o200k_base")
    dataset_meta = {
        "source": "huggingface",
        "hf_path": "roneneldan/TinyStories",
        "tokenizer_encoding": "o200k_base",
        "tokenizer_vocab_size": expected_vocab,
    }
    graph = _make_graph(vocab_size=expected_vocab, preset="llama_fast")
    state_dict = {
        "node_modules.model.node_modules.token_embed.embedding.weight": torch.zeros((expected_vocab, 4)),
        "node_modules.model.node_modules.lm_head.proj.weight": torch.zeros((expected_vocab, 4)),
    }

    contract = infer_module.validate_inference_vocab_contract(
        dataset_name=ds_name,
        dataset_path=ds_dir,
        dataset_meta=dataset_meta,
        graph=graph,
        state_dict=state_dict,
        raw_text_encoding_name="o200k_base",
    )

    assert contract is not None
    assert contract["tokenizer_encoding"] == "o200k_base"
    assert contract["tokenizer_vocab_size"] == expected_vocab


def test_train_summary_lines_report_raw_text_tiktoken_without_name() -> None:
    train_module = _load_train_jepa_module()

    lines = train_module.dataset_tokenizer_summary_lines(
        {
            "source": "huggingface",
            "hf_path": "roneneldan/TinyStories",
            "tokenizer_encoding": "o200k_base",
            "tokenizer_vocab_size": dm.raw_text_encoding_vocab_size("o200k_base"),
        }
    )

    assert "  - Tokenizer backend: tiktoken" in lines
    assert "  - Tokenizer encoding: o200k_base" in lines
    assert all(line != "  - Tokenizer: None" for line in lines)


def test_save_artifacts_records_weights_file_in_graph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_module = _load_train_jepa_module()
    graph = _make_graph(vocab_size=16, preset="llama_fast")
    weights_path = tmp_path / "toy_weights.pt"
    graph_path = tmp_path / "toy_graph.json"
    sentinel_state = "encoded-module-state"

    def apply_module_state(payload_graph) -> None:
        for node in payload_graph.nodes.values():
            neuron_def = node.neuron_def
            if neuron_def.kind == "module":
                neuron_def.module_state = sentinel_state
            elif neuron_def.kind == "subgraph" and neuron_def.subgraph is not None:
                apply_module_state(neuron_def.subgraph)

    def collect_module_states(payload: dict[str, object]) -> list[str]:
        states: list[str] = []
        for node in payload.get("nodes", {}).values():
            neuron_def = dict(node.get("neuron_def", {}) or {})
            states.append(str(neuron_def.get("module_state", "")))
            subgraph = neuron_def.get("subgraph")
            if isinstance(subgraph, dict):
                states.extend(collect_module_states(subgraph))
        return states

    apply_module_state(graph)

    monkeypatch.setattr(
        train_module,
        "export_to_pt",
        lambda _graph, path: Path(path).write_bytes(b"pt"),
    )

    train_module.save_artifacts(
        graph,
        weights_path,
        graph_path,
        training_manifest={"run_id": "unit-test", "trainer": {"max_steps": 10}},
        dataset_name="tiny_raw",
        dataset_meta={
            "source": "huggingface",
            "hf_path": "roneneldan/TinyStories",
            "tokenizer_encoding": "o200k_base",
            "tokenizer_vocab_size": dm.raw_text_encoding_vocab_size("o200k_base"),
        },
        raw_text_encoding_name="o200k_base",
    )

    payload = json.loads(graph_path.read_text(encoding="utf-8"))
    assert payload["torch_config"]["artifact_metadata"]["weights_file"] == "toy_weights.pt"
    assert payload["torch_config"]["artifact_metadata"]["stores_embedded_module_state"] is False
    assert payload["torch_config"]["tokenizer_manifest"]["backend"] == "tiktoken"
    assert payload["torch_config"]["tokenizer_manifest"]["encoding_name"] == "o200k_base"
    assert payload["torch_config"]["training_manifest"]["run_id"] == "unit-test"
    assert payload["torch_config"]["training_manifest"]["dataset"]["dataset_alias"] == "tiny_raw"
    assert sentinel_state not in collect_module_states(payload)


def test_infer_llama_preflight_rejects_model_vocab_mismatch(tmp_path: Path) -> None:
    infer_module = _load_infer_llama_module()
    shared_infer = sys.modules["infer_jepa_semantic"]
    ds_name = "toy_alias"
    ds_dir = _write_tokenizer_backed_alias(
        tmp_path,
        name=ds_name,
        tokenizer_vocab_size=8,
        shard_values=[0, 1, 2, 3, 4, 5, 6, 7],
    )
    graph = _make_graph(vocab_size=8, preset="llama_fast")
    state_dict = {
        "node_modules.model.node_modules.token_embed.embedding.weight": torch.zeros((16, 4)),
        "node_modules.model.node_modules.lm_head.proj.weight": torch.zeros((16, 4)),
    }

    with pytest.raises(dm.DatasetTokenizerMismatchError, match="Model/checkpoint vocab size: 16"):
        shared_infer.validate_inference_vocab_contract(
            dataset_name=ds_name,
            dataset_path=ds_dir,
            dataset_meta=dm._load_dataset_meta(ds_dir),
            graph=graph,
            state_dict=state_dict,
        )


def test_infer_llama_megakernel_preflight_rejects_model_vocab_mismatch(tmp_path: Path) -> None:
    infer_module = _load_infer_llama_megakernel_module()
    shared_infer = sys.modules["infer_jepa_semantic"]
    ds_name = "toy_alias"
    ds_dir = _write_tokenizer_backed_alias(
        tmp_path,
        name=ds_name,
        tokenizer_vocab_size=8,
        shard_values=[0, 1, 2, 3, 4, 5, 6, 7],
    )
    graph = _make_graph(vocab_size=8, preset="llama_fast_megakernel")
    state_dict = {
        "node_modules.model.node_modules.token_embed.embedding.weight": torch.zeros((16, 4)),
        "node_modules.model.node_modules.lm_head.proj.weight": torch.zeros((16, 4)),
    }

    with pytest.raises(dm.DatasetTokenizerMismatchError, match="Model/checkpoint vocab size: 16"):
        shared_infer.validate_inference_vocab_contract(
            dataset_name=ds_name,
            dataset_path=ds_dir,
            dataset_meta=dm._load_dataset_meta(ds_dir),
            graph=graph,
            state_dict=state_dict,
        )


def test_graph_trace_inputs_use_template_aware_raw_text_encoding() -> None:
    graph = _make_graph(vocab_size=dm.raw_text_encoding_vocab_size("o200k_base"), preset="llama_fast")
    load_dataset_source_into_graph(graph, LoadDatasetRequest(dataset_names=["tiny_raw"], seq_len=4))

    with patch("server.services.graph_ops.load_dataset_tokens", return_value=([[0, 1, 2, 3]], [[1, 2, 3, 4]])) as mocked:
        from server.services import graph_ops

        provided, sample_inputs = graph_ops._build_dataset_trace_inputs(
            graph,
            ["tiny_raw"],
            seq_len=4,
            preview_batch_size=1,
        )

    assert provided["dataset_source"][0].tolist() == [[0, 1, 2, 3]]
    assert sample_inputs["tokens"] == [0, 1, 2, 3]
    assert mocked.call_args.kwargs["encoding_name"] == "o200k_base"


def test_resolve_or_download_dataset_downloads_parseable_missing_alias(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    train_module = _load_train_jepa_module()
    alias = "willdepueoai__parameter-golf__sp1024__train2"
    calls: list[dict[str, object]] = []

    def fake_download(hf_path: str, **kwargs):
        calls.append({"hf_path": hf_path, **kwargs})
        ds_dir = tmp_path / alias
        (ds_dir / "tokenizers").mkdir(parents=True, exist_ok=True)
        (ds_dir / "meta.json").write_text(
            json.dumps({"variant": "sp1024", "train_shards": 2, "data_format": "uint16_shards"}, indent=2),
            encoding="utf-8",
        )
        return {"name": alias}

    monkeypatch.setattr(train_module, "DATASETS_DIR", tmp_path)
    monkeypatch.setattr(train_module, "download_hf_dataset", fake_download)

    dataset_name, dataset_path, dataset_meta = train_module.resolve_or_download_dataset(alias)

    assert dataset_name == alias
    assert dataset_path == tmp_path / alias
    assert dataset_meta["variant"] == "sp1024"
    assert len(calls) == 1
    _assert_call_includes(
        calls[0],
        {
            "hf_path": "willdepueoai/parameter-golf",
            "alias": alias,
            "variant": "sp1024",
            "train_shards": 2,
            "repo_id": "willdepueoai/parameter-golf",
            "remote_root_prefix": "datasets",
            "encoding_name": "gpt2",
        },
    )


def test_resolve_or_download_dataset_surfaces_validator_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    train_module = _load_train_jepa_module()
    alias = "willdepueoai__parameter-golf__sp1024__train2"
    fallback_calls: list[tuple[str, dict[str, object]]] = []

    def fake_download(_hf_path: str, **_kwargs):
        raise dm.DatasetTokenizerMismatchError("tokenizer mismatch from download")

    def fake_prepare_dataset_from_text(dataset_alias: str, *args, **kwargs):
        kwargs["args"] = args
        fallback_calls.append((dataset_alias, kwargs))
        raise dm.DatasetTokenizerMismatchError("tokenizer mismatch from download")

    monkeypatch.setattr(train_module, "DATASETS_DIR", tmp_path)
    monkeypatch.setattr(train_module, "download_hf_dataset", fake_download)
    monkeypatch.setattr(train_module, "_prepare_dataset_from_text", fake_prepare_dataset_from_text)

    with pytest.raises(dm.DatasetTokenizerMismatchError, match="tokenizer mismatch from download"):
        train_module.resolve_or_download_dataset(alias)
    assert fallback_calls


def test_resolve_or_download_dataset_requires_explicit_contract_for_unparseable_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_module = _load_train_jepa_module()
    monkeypatch.setattr(train_module, "DATASETS_DIR", tmp_path)

    with pytest.raises(ValueError, match="--dataset-hf-path, --dataset-variant, --dataset-train-shards"):
        train_module.resolve_or_download_dataset("custom_alias")


def test_resolve_or_download_dataset_prefers_explicit_contract_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    train_module = _load_train_jepa_module()
    alias = "willdepueoai__parameter-golf__sp1024__train2"
    calls: list[dict[str, object]] = []

    def fake_download(hf_path: str, **kwargs):
        calls.append({"hf_path": hf_path, **kwargs})
        ds_dir = tmp_path / alias
        ds_dir.mkdir(parents=True, exist_ok=True)
        (ds_dir / "meta.json").write_text(json.dumps({"data_format": "uint16_shards"}), encoding="utf-8")
        return {"name": alias}

    monkeypatch.setattr(train_module, "DATASETS_DIR", tmp_path)
    monkeypatch.setattr(train_module, "download_hf_dataset", fake_download)

    train_module.resolve_or_download_dataset(
        alias,
        dataset_hf_path="custom-owner/custom-repo",
        dataset_variant="sp2048",
        dataset_train_shards=5,
        dataset_repo_id="override/repo",
        dataset_remote_root_prefix="custom-prefix",
    )

    assert len(calls) == 1
    _assert_call_includes(
        calls[0],
        {
            "hf_path": "custom-owner/custom-repo",
            "alias": alias,
            "variant": "sp2048",
            "train_shards": 5,
            "repo_id": "override/repo",
            "remote_root_prefix": "custom-prefix",
            "encoding_name": "gpt2",
        },
    )


def test_resolve_or_download_dataset_accepts_explicit_raw_file_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_module = _load_train_jepa_module()
    alias = "roneneldan__TinyStories__TinyStoriesV2-GPT4"
    calls: list[dict[str, object]] = []

    def fake_download(hf_path: str, **kwargs):
        calls.append({"hf_path": hf_path, **kwargs})
        ds_dir = tmp_path / alias
        ds_dir.mkdir(parents=True, exist_ok=True)
        (ds_dir / "TinyStoriesV2-GPT4-train.txt").write_text("Once upon a time.\n", encoding="utf-8")
        (ds_dir / "TinyStoriesV2-GPT4-valid.txt").write_text("The end.\n", encoding="utf-8")
        (ds_dir / "meta.json").write_text(
            json.dumps(
                {
                    "hf_path": hf_path,
                    "train_file": kwargs.get("train_file"),
                    "val_file": kwargs.get("val_file"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return {"name": alias}

    monkeypatch.setattr(train_module, "DATASETS_DIR", tmp_path)
    monkeypatch.setattr(train_module, "download_hf_dataset", fake_download)

    dataset_name, dataset_path, dataset_meta = train_module.resolve_or_download_dataset(
        alias,
        dataset_hf_path="roneneldan/TinyStories",
        dataset_train_file="TinyStoriesV2-GPT4-train.txt",
        dataset_val_file="TinyStoriesV2-GPT4-valid.txt",
    )

    assert dataset_name == alias
    assert dataset_path == tmp_path / alias
    assert dataset_meta["train_file"] == "TinyStoriesV2-GPT4-train.txt"
    assert dataset_meta["val_file"] == "TinyStoriesV2-GPT4-valid.txt"
    assert len(calls) == 1
    _assert_call_includes(
        calls[0],
        {
            "hf_path": "roneneldan/TinyStories",
            "alias": alias,
            "repo_id": "roneneldan/TinyStories",
            "remote_root_prefix": "datasets",
            "train_file": "TinyStoriesV2-GPT4-train.txt",
            "val_file": "TinyStoriesV2-GPT4-valid.txt",
            "encoding_name": "gpt2",
        },
    )


def test_apply_tinystories_dataset_defaults_sets_raw_hf_contract() -> None:
    train_module = _load_train_jepa_module()
    parser = train_module.build_parser()
    args = parser.parse_args(["--tinystories"])

    train_module.apply_tinystories_dataset_defaults(args)
    train_module.resolve_dataset_selector_args(args)

    assert args.dataset_alias == train_module.TINYSTORIES_ALIAS
    assert args.dataset_hf_path == train_module.TINYSTORIES_HF_PATH
    assert args.dataset_train_file == train_module.TINYSTORIES_TRAIN_FILE
    assert args.dataset_val_file == train_module.TINYSTORIES_VAL_FILE


def test_resolve_dataset_selector_args_populates_tinystories_shortcut_contract() -> None:
    train_module = _load_train_jepa_module()
    parser = train_module.build_parser()
    args = parser.parse_args(["--dataset", "tinystories", "--dataset-alias", "custom_alias"])

    train_module.resolve_dataset_selector_args(args)

    assert args.dataset_alias == train_module.TINYSTORIES_ALIAS
    assert args.dataset_hf_path == train_module.TINYSTORIES_HF_PATH
    assert args.dataset_train_file == train_module.TINYSTORIES_TRAIN_FILE
    assert args.dataset_val_file == train_module.TINYSTORIES_VAL_FILE


@pytest.mark.parametrize(
    ("cli_args", "flag_name"),
    [
        (["--tinystories", "--dataset", "golf1"], "--dataset"),
        (["--tinystories", "--dataset-alias", "custom_alias"], "--dataset-alias"),
        (["--tinystories", "--dataset-hf-path", "custom/repo"], "--dataset-hf-path"),
        (["--tinystories", "--dataset-variant", "sp1024"], "--dataset-variant"),
        (["--tinystories", "--dataset-train-shards", "2"], "--dataset-train-shards"),
        (["--tinystories", "--dataset-repo-id", "custom/repo"], "--dataset-repo-id"),
        (["--tinystories", "--dataset-remote-root-prefix", "custom-prefix"], "--dataset-remote-root-prefix"),
        (["--tinystories", "--dataset-train-file", "train.txt"], "--dataset-train-file"),
        (["--tinystories", "--dataset-val-file", "val.txt"], "--dataset-val-file"),
    ],
)
def test_apply_tinystories_dataset_defaults_rejects_conflicting_dataset_flags(
    cli_args: list[str],
    flag_name: str,
) -> None:
    train_module = _load_train_jepa_module()
    parser = train_module.build_parser()
    args = parser.parse_args(cli_args)

    with pytest.raises(ValueError, match=flag_name):
        train_module.apply_tinystories_dataset_defaults(args)


@pytest.mark.parametrize(
    ("loader", "argv0"),
    [
        (_load_train_jepa_module, "train_jepa_semantic.py"),
    ],
)
def test_train_scripts_main_pass_tinystories_raw_file_contract_to_shared_resolver(
    loader,
    argv0: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = loader()
    sentinel = RuntimeError("after resolver")
    resolved_calls: list[dict[str, object]] = []

    def fake_resolve(alias: str, **kwargs):
        resolved_calls.append({"alias": alias, **kwargs})
        return alias, tmp_path / alias, {}

    estimate_name = "estimate_schedule" if argv0 in {"train_jepa_semantic.py"} else "estimate_text_schedule"

    def fake_estimate_schedule(*_args, **_kwargs):
        raise sentinel

    monkeypatch.setattr(module, "resolve_or_download_dataset", fake_resolve)
    monkeypatch.setattr(module, estimate_name, fake_estimate_schedule)
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(sys, "argv", [argv0, "--tinystories", "--download-if-missing"])

    with pytest.raises(RuntimeError, match="after resolver"):
        module.main()

    assert len(resolved_calls) == 1
    _assert_call_includes(
        resolved_calls[0],
        {
            "alias": "roneneldan__TinyStories__TinyStoriesV2-GPT4",
            "download_if_missing": True,
            "raw_text_encoding_name": "o200k_base",
            "dataset_hf_path": "roneneldan/TinyStories",
            "dataset_variant": None,
            "dataset_train_shards": None,
            "dataset_repo_id": None,
            "dataset_remote_root_prefix": None,
            "dataset_train_file": "TinyStoriesV2-GPT4-train.txt",
            "dataset_val_file": "TinyStoriesV2-GPT4-valid.txt",
        },
    )


@pytest.mark.parametrize(
    ("loader", "argv0"),
    [
        (_load_train_jepa_module, "train_jepa_semantic.py"),
    ],
)
def test_train_scripts_main_pass_dataset_tinystories_raw_file_contract_to_shared_resolver(
    loader,
    argv0: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = loader()
    sentinel = RuntimeError("after resolver")
    resolved_calls: list[dict[str, object]] = []

    def fake_resolve(alias: str, **kwargs):
        resolved_calls.append({"alias": alias, **kwargs})
        return alias, tmp_path / alias, {}

    estimate_name = "estimate_schedule" if argv0 in {"train_jepa_semantic.py"} else "estimate_text_schedule"

    def fake_estimate_schedule(*_args, **_kwargs):
        raise sentinel

    monkeypatch.setattr(module, "resolve_or_download_dataset", fake_resolve)
    monkeypatch.setattr(module, estimate_name, fake_estimate_schedule)
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(sys, "argv", [argv0, "--dataset", "tinystories", "--download-if-missing"])

    with pytest.raises(RuntimeError, match="after resolver"):
        module.main()

    assert len(resolved_calls) == 1
    _assert_call_includes(
        resolved_calls[0],
        {
            "alias": "roneneldan__TinyStories__TinyStoriesV2-GPT4",
            "download_if_missing": True,
            "raw_text_encoding_name": "o200k_base",
            "dataset_hf_path": "roneneldan/TinyStories",
            "dataset_variant": None,
            "dataset_train_shards": None,
            "dataset_repo_id": None,
            "dataset_remote_root_prefix": None,
            "dataset_train_file": "TinyStoriesV2-GPT4-train.txt",
            "dataset_val_file": "TinyStoriesV2-GPT4-valid.txt",
        },
    )


def test_infer_mixllama_main_resolves_dataset_before_tokenizer_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    infer_module = _load_infer_mixllama_module()
    shared_infer = sys.modules["infer_jepa_semantic"]
    graph_path = tmp_path / "graph.json"
    weights_path = tmp_path / "weights.pt"
    graph_path.write_text("{}", encoding="utf-8")
    weights_path.write_text("stub", encoding="utf-8")
    sentinel = RuntimeError("tokenizer reached")
    resolved_calls: list[dict[str, object]] = []

    def fake_resolve(alias: str, **kwargs):
        resolved_calls.append({"alias": alias, **kwargs})
        return alias, tmp_path / alias, {"tokenizer_files": []}

    def fake_load_sentencepiece_model(
        dataset_path: Path,
        dataset_meta: dict[str, object],
        *,
        raw_text_encoding_name: str = "gpt2",
    ):
        assert dataset_path == tmp_path / "custom_alias"
        assert dataset_meta == {"tokenizer_files": []}
        assert raw_text_encoding_name
        raise sentinel

    monkeypatch.setattr(shared_infer, "resolve_or_download_dataset", fake_resolve)
    monkeypatch.setattr(shared_infer, "load_sentencepiece_model", fake_load_sentencepiece_model)
    monkeypatch.setattr(
        infer_module,
        "load_compiled_inference_graph",
        lambda **_kwargs: (SimpleNamespace(torch_config={}, nodes={}), SimpleNamespace(), {}, weights_path),
    )
    monkeypatch.setattr(infer_module.torch.cuda, "is_available", lambda: True)
    _patch_cuda_generator_to_cpu(infer_module, monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "infer_mixllama_fast.py",
            "--dataset-alias",
            "custom_alias",
            "--graph",
            str(graph_path),
            "--weights",
            str(weights_path),
        ],
    )

    assert infer_module.main() == 1

    assert len(resolved_calls) == 1
    _assert_call_includes(
        resolved_calls[0],
        {
            "alias": "custom_alias",
            "download_if_missing": True,
            "raw_text_encoding_name": "gpt2",
            "dataset_hf_path": None,
            "dataset_variant": None,
            "dataset_train_shards": None,
            "dataset_repo_id": None,
            "dataset_remote_root_prefix": None,
            "dataset_train_file": None,
            "dataset_val_file": None,
        },
    )


def test_infer_llama_main_resolves_dataset_before_tokenizer_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    infer_module = _load_infer_llama_module()
    shared_infer = sys.modules["infer_jepa_semantic"]
    graph_path = tmp_path / "graph.json"
    weights_path = tmp_path / "weights.pt"
    graph_path.write_text("{}", encoding="utf-8")
    weights_path.write_text("stub", encoding="utf-8")
    sentinel = RuntimeError("tokenizer reached")
    resolved_calls: list[dict[str, object]] = []

    def fake_resolve(alias: str, **kwargs):
        resolved_calls.append({"alias": alias, **kwargs})
        return alias, tmp_path / alias, {"tokenizer_files": []}

    def fake_load_sentencepiece_model(
        dataset_path: Path,
        dataset_meta: dict[str, object],
        *,
        raw_text_encoding_name: str = "gpt2",
    ):
        assert dataset_path == tmp_path / "custom_alias"
        assert dataset_meta == {"tokenizer_files": []}
        assert raw_text_encoding_name
        raise sentinel

    monkeypatch.setattr(shared_infer, "resolve_or_download_dataset", fake_resolve)
    monkeypatch.setattr(shared_infer, "load_sentencepiece_model", fake_load_sentencepiece_model)
    monkeypatch.setattr(
        infer_module,
        "load_compiled_inference_graph",
        lambda **_kwargs: (SimpleNamespace(torch_config={}, nodes={}), SimpleNamespace(), {}, weights_path),
    )
    monkeypatch.setattr(infer_module.torch.cuda, "is_available", lambda: True)
    _patch_cuda_generator_to_cpu(infer_module, monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "infer_llama_fast.py",
            "--dataset-alias",
            "custom_alias",
            "--graph",
            str(graph_path),
            "--weights",
            str(weights_path),
        ],
    )

    assert infer_module.main() == 1

    assert len(resolved_calls) == 1
    _assert_call_includes(
        resolved_calls[0],
        {
            "alias": "custom_alias",
            "download_if_missing": True,
            "raw_text_encoding_name": "gpt2",
            "dataset_hf_path": None,
            "dataset_variant": None,
            "dataset_train_shards": None,
            "dataset_repo_id": None,
            "dataset_remote_root_prefix": None,
            "dataset_train_file": None,
            "dataset_val_file": None,
        },
    )


def test_infer_llama_megakernel_main_resolves_dataset_before_tokenizer_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    infer_module = _load_infer_llama_megakernel_module()
    shared_infer = sys.modules["infer_jepa_semantic"]
    graph_path = tmp_path / "graph.json"
    weights_path = tmp_path / "weights.pt"
    graph_path.write_text("{}", encoding="utf-8")
    weights_path.write_text("stub", encoding="utf-8")
    sentinel = RuntimeError("tokenizer reached")
    resolved_calls: list[dict[str, object]] = []

    def fake_resolve(alias: str, **kwargs):
        resolved_calls.append({"alias": alias, **kwargs})
        return alias, tmp_path / alias, {"tokenizer_files": []}

    def fake_load_sentencepiece_model(
        dataset_path: Path,
        dataset_meta: dict[str, object],
        *,
        raw_text_encoding_name: str = "gpt2",
    ):
        assert dataset_path == tmp_path / "custom_alias"
        assert dataset_meta == {"tokenizer_files": []}
        assert raw_text_encoding_name
        raise sentinel

    monkeypatch.setattr(shared_infer, "resolve_or_download_dataset", fake_resolve)
    monkeypatch.setattr(shared_infer, "load_sentencepiece_model", fake_load_sentencepiece_model)
    monkeypatch.setattr(
        infer_module,
        "load_compiled_inference_graph",
        lambda **_kwargs: (SimpleNamespace(torch_config={}, nodes={}), SimpleNamespace(), {}, weights_path),
    )
    monkeypatch.setattr(infer_module.torch.cuda, "is_available", lambda: True)
    _patch_cuda_generator_to_cpu(infer_module, monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "infer_llama_megakernel.py",
            "--dataset-alias",
            "custom_alias",
            "--graph",
            str(graph_path),
            "--weights",
            str(weights_path),
        ],
    )

    assert infer_module.main() == 1

    assert len(resolved_calls) == 1
    _assert_call_includes(
        resolved_calls[0],
        {
            "alias": "custom_alias",
            "download_if_missing": True,
            "raw_text_encoding_name": "gpt2",
            "dataset_hf_path": None,
            "dataset_variant": None,
            "dataset_train_shards": None,
            "dataset_repo_id": None,
            "dataset_remote_root_prefix": None,
            "dataset_train_file": None,
            "dataset_val_file": None,
        },
    )


def test_infer_llama_megakernel_fast_mode_selects_mode_specific_artifact_defaults() -> None:
    infer_module = _load_infer_llama_megakernel_module()

    default_args = infer_module.resolve_mode_defaults(infer_module.build_parser().parse_args([]))
    fast_args = infer_module.resolve_mode_defaults(infer_module.build_parser().parse_args(["--fast"]))

    assert Path(default_args.weights).name == "llama_megakernel.pt"
    assert Path(default_args.graph).name == "llama_megakernel.json"
    assert Path(fast_args.weights).name == "llama_fast_megakernel.pt"
    assert Path(fast_args.graph).name == "llama_fast_megakernel.json"


def test_estimate_text_schedule_uses_text_dataset_rows_only(monkeypatch: pytest.MonkeyPatch) -> None:
    train_module = _load_train_jepa_module()

    class FakeDataset:
        def __len__(self) -> int:
            return 97

    monkeypatch.setattr(train_module, "load_dataset_tensors", lambda *_args, **_kwargs: FakeDataset())
    monkeypatch.setattr(train_module, "load_semantic_tokens", lambda *_args, **_kwargs: torch.zeros((3, 2)))

    schedule = train_module.estimate_text_schedule(
        "custom_alias",
        seq_len=16,
        batch_size=4,
        train_batch_tokens=64,
    )

    assert {
        key: schedule[key]
        for key in (
            "train_rows",
            "microbatch_tokens",
            "effective_train_batch_tokens",
            "grad_accum_steps",
            "loader_batches",
            "steps_per_epoch",
        )
    } == {
        "train_rows": 97,
        "microbatch_tokens": 64,
        "effective_train_batch_tokens": 64,
        "grad_accum_steps": 1,
        "loader_batches": 25,
        "steps_per_epoch": 25,
    }
