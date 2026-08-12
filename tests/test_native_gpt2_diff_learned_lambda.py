from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import threading

import pytest

from neuralfn.torch_templates import build_gpt_root_graph, build_model_spec_from_config
from neuralfn.native_graph_train import plan_native_graph_training


ROOT = Path(__file__).resolve().parents[1]
TRAINER_SOURCE = ROOT / "neuralfn/csrc/native_gpt2/nfn_gpt2_native_train.cpp"
KERNEL_SOURCE = ROOT / "neuralfn/csrc/tile_cuda/kernels.cu"
TILE_WRAPPER_SOURCE = ROOT / "neuralfn/csrc/native_train/tile_ops.cu"
TOKEN_SHARDS_SOURCE = ROOT / "neuralfn/csrc/native_train/token_shards.cpp"
RUNTIME_ENV = "NFN_NATIVE_GPT2_DIFF_LEARNED_RUNTIME_TESTS"
DIFF_FORWARD_SYMBOL = (
    "nfn_native_tile_differential_packed_attention_forward_learned_lambda_bf16"
)
DIFF_BACKWARD_SYMBOL = (
    "nfn_native_tile_differential_packed_attention_backward_learned_lambda_bf16"
)
DIFF_RELEASE_SYMBOL = (
    "nfn_native_tile_differential_packed_attention_release_workspaces"
)
DIFF_RESUME_ERROR_PREFIX = "invalid gpt2_diff resume metadata: "
DIFF_HEADER_BYTES = 16 * 8
CUDA_ERROR_INVALID_VALUE = 1


def test_learned_lambda_is_a_real_per_layer_adamw_parameter() -> None:
    source = TRAINER_SOURCE.read_text(encoding="utf-8")

    assert "float* differential_lambdas = nullptr;" in source
    assert "float* differential_lambda_grads = nullptr;" in source
    assert "float* differential_lambda_avgs = nullptr;" in source
    assert "float* differential_lambda_avg_sqs = nullptr;" in source
    assert '"differential.lambda"' in source
    assert "variant.differential_lambda_init" in source
    assert (
        "differential_lambdas + (&block - blocks.data())" in source
    )
    assert (
        "differential_lambda_grads + (&block - blocks.data())" in source
    )
    assert "partial_count_for(trained_layers)" in source
    assert "dense_adamw_checkpoint_descriptor_count" in source

    descriptor_tail = source.split(
        "add(lnf_bias, accum_grad_lnf_bias", 1
    )[1].split("if (!error.empty())", 1)[0]
    assert "differential_lambdas" in descriptor_tail
    assert "differential_lambda_grads" in descriptor_tail
    assert "differential_lambda_avgs" in descriptor_tail
    assert "differential_lambda_avg_sqs" in descriptor_tail
    assert "trained_layers" in descriptor_tail
    assert "0.0f" in descriptor_tail


def test_learned_lambda_backward_recomputes_layer_local_branches_and_gradient() -> None:
    kernels = KERNEL_SOURCE.read_text(encoding="utf-8")
    wrappers = TILE_WRAPPER_SOURCE.read_text(encoding="utf-8")

    backward = kernels.split(
        "int launch_differential_packed_attention_backward_learned_lambda_bf16(",
        1,
    )[1].split("\n}\n\n}  // namespace neuralfn::tile_cuda", 1)[0]
    assert "differential_unpack_qkv_heads_float32_kernel" in backward
    assert backward.count("launch_scaled_dot_product_attention_float32(") == 2
    assert "differential_combine_rms_learned_lambda_backward_kernel" in backward
    assert "differential_reduce_lambda_partials_kernel<<<1, 256" in backward
    assert "grad_lambda" in backward
    assert "grad_lambda_partials[segment] = scratch[0]" in kernels
    assert "*grad_lambda += scratch[0]" in kernels
    assert "atomicAdd(grad_lambda" not in kernels
    assert "lambda_partial -= __bfloat162float(second[base + d]) * grad" in kernels
    assert backward.count(
        "attention_scale, true, false, false, 0, 0, 0, 0, true, stream"
    ) == 2
    assert "std::unique_lock<std::mutex>(slot->call_mutex)" in kernels
    assert "stream == cudaStreamPerThread" in kernels
    assert "cudaStreamSynchronize(stream)" in kernels
    assert "return cudaErrorMemoryAllocation;" in kernels
    assert "return allocation_status;" in kernels
    assert "return synchronize_status;" in kernels
    assert "return static_cast<int>(cudaErrorInvalidValue);" in backward
    assert DIFF_RELEASE_SYMBOL in wrappers
    assert DIFF_FORWARD_SYMBOL in wrappers
    assert DIFF_BACKWARD_SYMBOL in wrappers


def test_diff_persistence_is_additive_and_source_bound() -> None:
    source = TRAINER_SOURCE.read_text(encoding="utf-8")

    assert '"diff_parameters_" + name.substr(6, 8) + ".bin"' in source
    assert '"diff_optimizer_" + name.substr(6, 8) + ".bin"' in source
    assert 'name.substr(0, name.size() - 4) + ".diff.json"' in source
    assert "neuralfn.native_gpt2_diff.training_checkpoint" in source
    assert "neuralfn.native_graph_training_proof" in source
    assert '"--graph-preflight-proof"' in source
    assert "verify_gpt2_diff_graph_preflight_proof" in source
    assert '"graph_preflight_proof"' in source
    assert "trained_dense_v5_plus_diff_v1" in source
    assert '<< "  \\"version\\": 2' in source
    assert "continuation" in source
    assert "train_shards_sha256" in source
    assert "lr_schedule_total_steps" in source
    assert "neuralfn.gpt2_diff.numerics_profile.v1" in source
    assert "numerics_profile_sha256" in source
    assert "::environ" not in source
    assert "stable_diff_train_shards" in source
    assert "stable_diff_train_shard_fds" in source
    token_shards = TOKEN_SHARDS_SOURCE.read_text(encoding="utf-8")
    assert "::pread(" in token_shards
    assert "stable token shard changed before batch read" in token_shards
    assert "failed to read stable gpt2_diff train shard:" in source
    assert '"gpt2_diff graph file must not be empty"' in source
    assert "kMaxVerifiedGraphBytes = 16 * 1024 * 1024" in source
    assert '"checkpoint step must be in the range [1, 99999999]"' in source
    assert '"source_graph"' in source
    assert "cfg.verified_graph_fingerprint" in source
    assert "model SHA-256" not in source.split(
        "bool read_native_gpt_diff_resume_state", 1
    )[1].split("bool write_native_gpt_diff_metadata", 1)[0]
    assert " SHA-256 does not match the artifact bytes" in source
    assert "kDiffParameterMagic = 20260808" in source
    assert "kDiffOptimizerMagic = 20260809" in source

    assert (
        "optimizer_header[9] = dense_adamw_checkpoint_descriptor_count;" in source
    )
    assert (
        "header[9] != dense_adamw_checkpoint_descriptor_count" in source
    )
    assert "ExclusiveOutputFile out(checkpoint_path);" in source
    assert "CheckpointOutputTransaction output_transaction;" in source
    assert "fsync_checkpoint_directory(output_dir, &error)" in source
    assert "refusing to overwrite existing checkpoint target" in source
    assert "std::vector<float>().swap(resume_diff_state.dense_parameters)" in source
    assert "std::vector<float>().swap(resume_diff_state.dense_optimizer)" in source
    run_body = source.split("int run_transformer_lm_training_json(", 1)[1]
    assert run_body.index('run_setup_timed("setup.diff_resume_host_preflight"') < run_body.index(
        'run_setup_timed("setup.load_tile_ops"'
    )
    moa_resume_reader = source.split(
        "bool read_native_gpt_moa_resume_metadata(", 1
    )[1].split("bool write_native_gpt_moa_inference_metadata(", 1)[0]
    diff_resume_reader = source.split(
        "bool read_native_gpt_diff_resume_state(", 1
    )[1].split("bool write_native_gpt_diff_metadata(", 1)[0]
    assert "verified_graph_preflight_proof" not in moa_resume_reader
    assert "verified_graph_preflight_proof" in diff_resume_reader


def _configure_diff_abi(library: ctypes.CDLL) -> None:
    pointer = ctypes.c_void_p
    i64 = ctypes.c_int64
    forward = getattr(library, DIFF_FORWARD_SYMBOL)
    forward.argtypes = [
        pointer,
        pointer,
        i64,
        i64,
        i64,
        i64,
        pointer,
        ctypes.c_float,
        ctypes.c_float,
        pointer,
    ]
    forward.restype = ctypes.c_int
    backward = getattr(library, DIFF_BACKWARD_SYMBOL)
    backward.argtypes = [
        pointer,
        pointer,
        pointer,
        pointer,
        i64,
        i64,
        i64,
        i64,
        pointer,
        ctypes.c_float,
        ctypes.c_float,
        pointer,
        pointer,
    ]
    backward.restype = ctypes.c_int
    release = getattr(library, DIFF_RELEASE_SYMBOL)
    release.argtypes = []
    release.restype = ctypes.c_int


def _cuda_pointer(tensor: object) -> ctypes.c_void_p:
    return ctypes.c_void_p(int(tensor.data_ptr()))  # type: ignore[attr-defined]


@pytest.mark.skipif(
    os.environ.get(RUNTIME_ENV) != "1",
    reason=f"set {RUNTIME_ENV}=1 to run the CUDA learned-lambda oracle",
)
def test_compiled_learned_lambda_forward_backward_matches_torch_oracle_and_zero(
    runtime_tile_ops: Path,
) -> None:
    library = ctypes.CDLL(str(runtime_tile_ops))
    _configure_diff_abi(library)
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    forward = getattr(library, DIFF_FORWARD_SYMBOL)
    backward = getattr(library, DIFF_BACKWARD_SYMBOL)

    batch, heads, seq_len, head_dim = 1, 2, 16, 8
    model_dim = heads * head_dim
    torch.manual_seed(731)
    qkv = (
        torch.randn(
            batch * seq_len,
            3 * model_dim,
            device="cuda",
            dtype=torch.float32,
        )
        * 0.2
    ).to(torch.bfloat16)
    learned_lambda = torch.tensor([0.8], device="cuda", dtype=torch.float32)
    output = torch.empty(
        batch * seq_len, model_dim, device="cuda", dtype=torch.bfloat16
    )
    torch.cuda.synchronize()
    status = forward(
        _cuda_pointer(qkv),
        _cuda_pointer(output),
        batch,
        heads,
        seq_len,
        head_dim,
        _cuda_pointer(learned_lambda),
        0.2,
        1.0e-5,
        None,
    )
    assert status == 0
    torch.cuda.synchronize()

    qkv_reference = qkv.float().detach().requires_grad_(True)
    lambda_reference = torch.tensor(
        [0.8], device="cuda", dtype=torch.float32, requires_grad=True
    )
    qkv_float = qkv_reference.view(batch, seq_len, 3, heads, head_dim)
    q = qkv_float[:, :, 0].permute(0, 2, 1, 3)
    k = qkv_float[:, :, 1].permute(0, 2, 1, 3)
    v = qkv_float[:, :, 2].permute(0, 2, 1, 3)
    causal_mask = torch.triu(
        torch.full(
            (seq_len, seq_len),
            float("-inf"),
            device="cuda",
            dtype=torch.float32,
        ),
        diagonal=1,
    )

    def attention(query: object, key: object) -> object:
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(
            head_dim // 2
        )
        probabilities = torch.softmax(scores + causal_mask, dim=-1)
        return torch.matmul(probabilities, v)

    first = attention(q[..., : head_dim // 2], k[..., : head_dim // 2])
    second = attention(q[..., head_dim // 2 :], k[..., head_dim // 2 :])
    # The ABI rounds each branch to BF16 for the forward handoff, while its
    # attention backward consumes FP32 branch gradients.  Model that exact
    # identity straight-through contract instead of PyTorch's BF16-cast
    # backward, which would quantize the upstream gradient a second time.
    first_rounded = first.to(torch.bfloat16).float()
    second_rounded = second.to(torch.bfloat16).float()
    first_bf16 = first + (first_rounded - first).detach()
    second_bf16 = second + (second_rounded - second).detach()
    combined = first_bf16 - lambda_reference * second_bf16
    expected_output_heads = (
        combined
        * torch.rsqrt(combined.square().mean(dim=-1, keepdim=True) + 1.0e-5)
        * 0.2
    )
    expected_output = expected_output_heads.permute(0, 2, 1, 3).reshape(
        batch * seq_len, model_dim
    )
    torch.testing.assert_close(
        output.float(), expected_output.to(torch.bfloat16).float(), rtol=0.03, atol=0.02
    )

    grad_output = torch.randn_like(output, dtype=torch.float32) * 0.1
    grad_heads = grad_output.view(batch, seq_len, heads, head_dim).permute(
        0, 2, 1, 3
    )
    (expected_output_heads * grad_heads).sum().backward()
    assert qkv_reference.grad is not None
    assert lambda_reference.grad is not None
    expected_qkv_grad = qkv_reference.grad.to(torch.bfloat16).float()
    expected_lambda_grad = lambda_reference.grad[0]
    grad_qkv = torch.empty_like(qkv)
    grad_lambda = torch.zeros(1, device="cuda", dtype=torch.float32)
    status = backward(
        _cuda_pointer(qkv),
        _cuda_pointer(output),
        _cuda_pointer(grad_output),
        _cuda_pointer(grad_qkv),
        batch,
        heads,
        seq_len,
        head_dim,
        _cuda_pointer(learned_lambda),
        0.2,
        1.0e-5,
        _cuda_pointer(grad_lambda),
        None,
    )
    assert status == 0
    torch.cuda.synchronize()

    assert torch.isfinite(grad_qkv.float()).all()
    assert torch.count_nonzero(grad_qkv).item() > 0
    assert torch.isfinite(grad_lambda).all()
    assert grad_lambda.abs().item() > 0.0
    torch.testing.assert_close(
        grad_qkv.float(), expected_qkv_grad, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        grad_lambda[0], expected_lambda_grad, rtol=0.0, atol=0.0
    )

    grad_lambda.zero_()
    grad_output.zero_()
    status = backward(
        _cuda_pointer(qkv),
        _cuda_pointer(output),
        _cuda_pointer(grad_output),
        _cuda_pointer(grad_qkv),
        batch,
        heads,
        seq_len,
        head_dim,
        _cuda_pointer(learned_lambda),
        0.2,
        1.0e-5,
        _cuda_pointer(grad_lambda),
        None,
    )
    assert status == 0
    torch.cuda.synchronize()
    assert grad_lambda.item() == 0.0
    assert torch.count_nonzero(grad_qkv).item() == 0

    assert (
        forward(
            _cuda_pointer(qkv),
            _cuda_pointer(output),
            0,
            heads,
            seq_len,
            head_dim,
            _cuda_pointer(learned_lambda),
            0.2,
            1.0e-5,
            None,
        )
        == CUDA_ERROR_INVALID_VALUE
    )
    assert (
        forward(
            _cuda_pointer(qkv),
            _cuda_pointer(output),
            batch,
            heads,
            seq_len,
            head_dim,
            _cuda_pointer(learned_lambda),
            float("nan"),
            1.0e-5,
            None,
        )
        == CUDA_ERROR_INVALID_VALUE
    )
    assert (
        forward(
            _cuda_pointer(qkv),
            _cuda_pointer(output),
            1 << 62,
            heads,
            seq_len,
            head_dim,
            _cuda_pointer(learned_lambda),
            0.2,
            1.0e-5,
            None,
        )
        == CUDA_ERROR_INVALID_VALUE
    )
    assert (
        forward(
            _cuda_pointer(qkv),
            _cuda_pointer(output),
            batch,
            heads,
            seq_len,
            head_dim,
            _cuda_pointer(learned_lambda),
            0.2,
            0.0,
            None,
        )
        == CUDA_ERROR_INVALID_VALUE
    )
    # CUDA defines cudaStreamPerThread as the sentinel handle value 2.  The
    # cache rejects it because its meaning is host-thread-local.
    assert (
        forward(
            _cuda_pointer(qkv),
            _cuda_pointer(output),
            batch,
            heads,
            seq_len,
            head_dim,
            _cuda_pointer(learned_lambda),
            0.2,
            1.0e-5,
            ctypes.c_void_p(2),
        )
        == CUDA_ERROR_INVALID_VALUE
    )
    assert getattr(library, DIFF_RELEASE_SYMBOL)() == 0


@pytest.mark.skipif(
    os.environ.get(RUNTIME_ENV) != "1",
    reason=f"set {RUNTIME_ENV}=1 to run the CUDA workspace concurrency stress",
)
def test_learned_lambda_workspace_is_stream_owned_serialized_and_teardown_safe(
    runtime_tile_ops: Path,
) -> None:
    library = ctypes.CDLL(str(runtime_tile_ops))
    _configure_diff_abi(library)
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    forward = getattr(library, DIFF_FORWARD_SYMBOL)
    backward = getattr(library, DIFF_BACKWARD_SYMBOL)
    release = getattr(library, DIFF_RELEASE_SYMBOL)

    batch, heads, seq_len, head_dim = 1, 2, 16, 8
    model_dim = heads * head_dim
    torch.manual_seed(991)
    qkvs = [
        (torch.randn(batch * seq_len, 3 * model_dim, device="cuda") * 0.15).to(
            torch.bfloat16
        )
        for _ in range(2)
    ]
    grad_outputs = [
        torch.randn(batch * seq_len, model_dim, device="cuda") * 0.05
        for _ in range(2)
    ]
    lambdas = [
        torch.tensor([0.7 + 0.1 * index], device="cuda", dtype=torch.float32)
        for index in range(2)
    ]
    torch.cuda.synchronize()

    def allocate_results() -> tuple[list[object], list[object], list[object]]:
        return (
            [torch.empty(batch * seq_len, model_dim, device="cuda", dtype=torch.bfloat16) for _ in range(2)],
            [torch.empty_like(qkvs[index]) for index in range(2)],
            [torch.zeros(1, device="cuda", dtype=torch.float32) for _ in range(2)],
        )

    def launch(index: int, stream_pointer: ctypes.c_void_p | None, results: tuple[list[object], list[object], list[object]]) -> None:
        outputs, grad_qkvs, grad_lambdas = results
        assert forward(
            _cuda_pointer(qkvs[index]),
            _cuda_pointer(outputs[index]),
            batch,
            heads,
            seq_len,
            head_dim,
            _cuda_pointer(lambdas[index]),
            0.2,
            1.0e-5,
            stream_pointer,
        ) == 0
        assert backward(
            _cuda_pointer(qkvs[index]),
            _cuda_pointer(outputs[index]),
            _cuda_pointer(grad_outputs[index]),
            _cuda_pointer(grad_qkvs[index]),
            batch,
            heads,
            seq_len,
            head_dim,
            _cuda_pointer(lambdas[index]),
            0.2,
            1.0e-5,
            _cuda_pointer(grad_lambdas[index]),
            stream_pointer,
        ) == 0

    baseline = allocate_results()
    for index in range(2):
        launch(index, None, baseline)
    torch.cuda.synchronize()
    assert release() == 0

    distinct = allocate_results()
    distinct_streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for index, stream in enumerate(distinct_streams):
        launch(index, ctypes.c_void_p(stream.cuda_stream), distinct)
    for stream in distinct_streams:
        stream.synchronize()
    assert release() == 0
    for group_index, (expected_group, actual_group) in enumerate(
        zip(baseline, distinct, strict=True)
    ):
        for expected, actual in zip(expected_group, actual_group, strict=True):
            if group_index == 1:
                # The public backward returns BF16 QKV gradients.  The SDPA
                # reduction can differ by one BF16 ULP across stream schedules;
                # outputs and the fixed-order lambda reduction remain exact.
                torch.testing.assert_close(actual, expected, rtol=0.01, atol=2.0e-6)
            else:
                torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    shared = allocate_results()
    shared_stream = torch.cuda.Stream()
    shared_pointer = ctypes.c_void_p(shared_stream.cuda_stream)
    barrier = threading.Barrier(2)
    failures: list[BaseException] = []

    def threaded_launch(index: int) -> None:
        try:
            barrier.wait()
            launch(index, shared_pointer, shared)
        except BaseException as exc:  # pragma: no cover - surfaced below
            failures.append(exc)

    threads = [threading.Thread(target=threaded_launch, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
        assert not thread.is_alive()
    assert failures == []
    shared_stream.synchronize()
    assert release() == 0
    for group_index, (expected_group, actual_group) in enumerate(
        zip(baseline, shared, strict=True)
    ):
        for expected, actual in zip(expected_group, actual_group, strict=True):
            if group_index == 1:
                torch.testing.assert_close(actual, expected, rtol=0.01, atol=2.0e-6)
            else:
                torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    # A release leaves the ABI reusable and cannot free scratch from an active
    # composite call because the lease owns the slot mutex through final enqueue.
    after_release = allocate_results()
    launch(0, shared_pointer, after_release)
    assert release() == 0
    torch.testing.assert_close(after_release[0][0], baseline[0][0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        after_release[1][0], baseline[1][0], rtol=0.01, atol=2.0e-6
    )
    torch.testing.assert_close(after_release[2][0], baseline[2][0], rtol=0.0, atol=0.0)


def _write_uint16_dataset(root: Path) -> Path:
    dataset = root / "tokens"
    dataset.mkdir(parents=True)
    tokens = [index % 256 for index in range(4096)]
    payload = struct.pack("<" + "H" * len(tokens), *tokens)
    (dataset / "fineweb_train_000000.bin").write_bytes(payload)
    (dataset / "fineweb_val_000000.bin").write_bytes(payload)
    (dataset / "meta.json").write_text(
        json.dumps(
            {
                "data_format": "uint16_shards",
                "tokenizer_encoding": "gpt2",
                "tokenizer_vocab_size": 50257,
            }
        ),
        encoding="utf-8",
    )
    return dataset


def _write_graph(root: Path, preset: str, *, num_layers: int = 1) -> Path:
    model_spec = build_model_spec_from_config(
        {
            "preset": preset,
            "num_layers": num_layers,
            "model_dim": 768,
            "num_heads": 12,
            "vocab_size": 50257,
        },
        preview_defaults=True,
    )
    graph = build_gpt_root_graph(
        name=f"{preset}_learned_lambda_checkpoint",
        model_spec=model_spec,
    )
    path = root / f"{preset}.json"
    document = graph.to_dict()
    document["torch_config"]["template_spec"]["max_seq_len"] = 16
    position_config = (
        document["nodes"]["model"]["neuron_def"]["subgraph"]
        ["nodes"]["pos_embed"]["neuron_def"]["module_config"]
    )
    position_config["max_seq_len"] = 16
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _materialize_proven_diff_graph(root: Path, graph: Path) -> tuple[Path, Path]:
    artifact_dir = root / f"{graph.stem}-native-proof"
    plan = plan_native_graph_training(
        graph,
        artifact_dir=artifact_dir,
        materialize=True,
    )
    assert plan.execution_ready
    assert plan.training_selector == "gpt2_diff"
    assert plan.graph_preflight_proof is not None
    assert plan.graph_preflight_proof.is_file()
    return plan.launch_graph, plan.graph_preflight_proof


@pytest.fixture(scope="module")
def runtime_tile_ops(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if os.environ.get(RUNTIME_ENV) != "1":
        pytest.skip(f"set {RUNTIME_ENV}=1 to run learned-lambda checkpoint tests")
    output = (
        tmp_path_factory.mktemp("gpt2-diff-learned-tile-ops")
        / "libnfn_native_train_tile_ops.so"
    )
    env = os.environ.copy()
    env["NFN_NATIVE_BUILD_STRICT_TILE_OPS"] = "0"
    completed = subprocess.run(
        ["bash", str(ROOT / "tools/build_native_train_tile_ops.sh"), str(output)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert output.is_file()
    return output


@pytest.fixture(scope="module")
def runtime_cli(
    tmp_path_factory: pytest.TempPathFactory,
    runtime_tile_ops: Path,
) -> Path:
    if shutil.which("c++") is None:
        pytest.skip("a C++ compiler is required")
    output = tmp_path_factory.mktemp("gpt2-diff-learned-cli") / "nfn_gpt_native_train"
    strict_tile_ops = runtime_tile_ops.with_name(
        runtime_tile_ops.stem + "_strict" + runtime_tile_ops.suffix
    )
    strict_tile_ops.write_bytes(runtime_tile_ops.read_bytes())
    env = os.environ.copy()
    env["NFN_NATIVE_GPT_FORCE_REBUILD"] = "1"
    env["NFN_NATIVE_GPT_CXX_OPT_FLAGS"] = "-O0"
    env["NFN_NATIVE_TRAIN_TILE_OPS_LIB"] = str(runtime_tile_ops)
    env["NFN_NATIVE_STRICT_INFERENCE_TILE_OPS_LIB"] = str(strict_tile_ops)
    completed = subprocess.run(
        ["bash", str(ROOT / "tools/build_native_gpt_cli.sh"), str(output)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return output


def _training_command(
    cli: Path,
    *,
    dataset: Path,
    graph: Path,
    proof: Path | None = None,
    output_dir: Path,
    tile_ops: Path,
    template_name: str,
    max_steps: int,
    batch_size: int = 1,
    train_batch_tokens: int = 16,
    num_layers: int = 1,
    train_seed: int | None = None,
    lr_schedule: str = "constant",
    lr_schedule_total_steps: int | None = None,
    warmup_steps: int = 60,
    resume: Path | None = None,
    startup_only: bool = False,
) -> list[str]:
    command = [
        str(cli),
        "--backend",
        "tile-cuda",
        "--template-name",
        template_name,
        "--graph-file",
        str(graph),
        "--graph-fingerprint",
        hashlib.sha256(graph.read_bytes()).hexdigest(),
        "--dataset-alias",
        str(dataset),
        "--tile-ops-lib",
        str(tile_ops),
        "--output-dir",
        str(output_dir),
        "--batch-size",
        str(batch_size),
        "--train-seq-len",
        "16",
        "--train-batch-tokens",
        str(train_batch_tokens),
        "--num-layers",
        str(num_layers),
        "--max-steps",
        str(max_steps),
        "--lr-schedule",
        lr_schedule,
        "--warmup-steps",
        str(warmup_steps),
        "--eval-every-steps",
        "0",
        "--eval-batches",
        "0",
        "--progress-every-steps",
        "0",
        "--native-cuda-generate-tokens",
        "0",
        "--fast-startup",
        "--allow-basic-kernel-fallback",
    ]
    if proof is not None:
        command.extend(["--graph-preflight-proof", str(proof)])
    if train_seed is not None:
        command.extend(["--train-seed", str(train_seed)])
    if lr_schedule_total_steps is not None:
        command.extend(
            ["--lr-schedule-total-steps", str(lr_schedule_total_steps)]
        )
    if resume is not None:
        command.extend(["--resume-from-checkpoint", str(resume)])
    if startup_only:
        command.extend(["--startup-only", "--no-checkpoint"])
    return command


def _runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    env["NFN_NATIVE_GPT_SETUP_PROGRESS"] = "0"
    env["NFN_NATIVE_GPT2_SETUP_PROGRESS"] = "0"
    return env


def _run(
    command: list[str], *, env_overrides: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    env = _runtime_env()
    if env_overrides is not None:
        env.update(env_overrides)
    return subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=300,
    )


def _payload(completed: subprocess.CompletedProcess[str]) -> dict[str, object]:
    assert completed.stdout, completed.stderr
    return json.loads(completed.stdout)


def _assert_strict_resume_rejected_before_cuda(
    completed: subprocess.CompletedProcess[str],
    *,
    output_dir: Path,
    expected_error: str | None = None,
) -> dict[str, object]:
    assert completed.returncode == 2, completed.stderr
    payload = _payload(completed)
    if expected_error is None:
        assert str(payload["error"]).startswith(DIFF_RESUME_ERROR_PREFIX)
    else:
        assert payload["error"] == expected_error
    assert payload["loaded"] is False
    assert payload["cuda_runtime_loaded"] is False
    assert payload["steps_completed"] == 0
    assert payload["train_microbatches_completed"] == 0
    assert payload["adamw_kernel_launches"] == 0
    assert payload["float_allocation_cuda_malloc_count"] == 0
    assert payload["transformer_device_arena_cuda_malloc_count"] == 0
    assert payload["uint16_arena_cuda_malloc_count"] == 0
    assert payload["descriptor_arena_cuda_malloc_count"] == 0
    assert payload["descriptor_arena_copy_count"] == 0
    assert payload["float_arena_zero_fill_count"] == 0
    assert payload["adamw_state_zero_fill_count"] == 0
    assert payload["parameter_initialization_kernel_launches"] == 0
    assert payload["bf16_parameter_initialization_kernel_launches"] == 0
    assert payload["mixed_parameter_initialization_kernel_launches"] == 0
    assert payload["resume_checkpoint_h2d_copy_count"] == 0
    assert payload["resume_parameter_state_h2d_copy_count"] == 0
    assert payload["resume_optimizer_checkpoint_h2d_copy_count"] == 0
    assert not output_dir.exists()
    return payload


def _sidecar_payload(path: Path) -> bytes:
    raw = path.read_bytes()
    assert len(raw) > DIFF_HEADER_BYTES
    return raw[DIFF_HEADER_BYTES:]


def _replace_manifest_artifact_hash(directory: Path, artifact: str, path: Path) -> None:
    manifest_paths = list(directory.glob("model_*.diff.json"))
    assert len(manifest_paths) == 1
    manifest_path = manifest_paths[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[artifact]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.unlink()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_checkpoint_tamper_resume_exactness_and_dense_compatibility(
    runtime_cli: Path,
    runtime_tile_ops: Path,
    tmp_path: Path,
) -> None:
    num_layers = 2
    split_step = 2
    final_step = 4
    dataset = _write_uint16_dataset(tmp_path)
    authored_diff_graph = _write_graph(
        tmp_path, "gpt2_diff", num_layers=num_layers
    )
    diff_graph, diff_proof = _materialize_proven_diff_graph(
        tmp_path, authored_diff_graph
    )
    dense_graph = _write_graph(tmp_path, "gpt2", num_layers=num_layers)
    straight_dir = tmp_path / "straight"
    split_first_dir = tmp_path / "split-first"
    split_resume_dir = tmp_path / "split-resume"
    continuation_options = {
        "train_batch_tokens": 32,
        "num_layers": num_layers,
        "train_seed": 731,
        "lr_schedule": "cosine",
        "warmup_steps": 1,
        "proof": diff_proof,
    }
    dense_continuation_options = {
        key: value
        for key, value in continuation_options.items()
        if key != "proof"
    }

    straight = _run(
        _training_command(
            runtime_cli,
            dataset=dataset,
            graph=diff_graph,
            output_dir=straight_dir,
            tile_ops=runtime_tile_ops,
            template_name="gpt2_diff",
            max_steps=final_step,
            lr_schedule_total_steps=final_step,
            **continuation_options,
        )
    )
    assert straight.returncode == 0, straight.stderr
    first = _run(
        _training_command(
            runtime_cli,
            dataset=dataset,
            graph=diff_graph,
            output_dir=split_first_dir,
            tile_ops=runtime_tile_ops,
            template_name="gpt2_diff",
            max_steps=split_step,
            lr_schedule_total_steps=final_step,
            **continuation_options,
        )
    )
    assert first.returncode == 0, first.stderr

    first_checkpoint = split_first_dir / f"model_{split_step:08d}.bin"
    assert first_checkpoint.is_file()
    for expected in (
        f"parameters_{split_step:08d}.bin",
        f"optimizer_{split_step:08d}.bin",
        f"diff_parameters_{split_step:08d}.bin",
        f"diff_optimizer_{split_step:08d}.bin",
        f"model_{split_step:08d}.diff.json",
        f"DONE_{split_step:08d}",
    ):
        assert (split_first_dir / expected).exists()
    manifest = json.loads(
        (split_first_dir / f"model_{split_step:08d}.diff.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["schema"] == "neuralfn.native_gpt2_diff.training_checkpoint"
    assert manifest["version"] == 2
    assert manifest["checkpoint_kind"] == "trained_dense_v5_plus_diff_v1"
    proof_envelope = json.loads(diff_proof.read_text(encoding="utf-8"))
    assert manifest["graph_preflight_proof"] == {
        "schema": "neuralfn.native_graph_training_proof",
        "version": 1,
        "contract_sha256": proof_envelope["contract_sha256"],
    }
    assert manifest["lambda"] == {
        "count": num_layers,
        "dtype": "float32",
        "initial_value": 0.8,
        "output_scale": 0.2,
    }
    continuation = manifest["continuation"]
    assert continuation["optimizer_steps_completed"] == split_step
    assert continuation["train_microbatches_completed"] == split_step * 2
    assert continuation["microbatch_in_optimizer_step"] == 0
    assert continuation["grad_accum_steps"] == 2
    assert continuation["train_seed_explicit"] is True
    assert continuation["train_seed"] == 731
    assert continuation["lr_schedule"] == "cosine"
    assert continuation["lr_schedule_total_steps"] == final_step
    assert continuation["warmup_steps"] == 1
    assert len(continuation["train_shards_sha256"]) == 64
    assert continuation["bf16_block_weight_params"] is True
    assert continuation["bf16_block_dweight_staging"] is False
    assert continuation["dweight_first_microbatch_beta_zero"] is True
    assert len(continuation["numerics_profile_sha256"]) == 64

    first_lambdas = struct.unpack(
        f"<{num_layers}f",
        _sidecar_payload(
            split_first_dir / f"diff_parameters_{split_step:08d}.bin"
        ),
    )
    first_moments = struct.unpack(
        f"<{num_layers * 2}f",
        _sidecar_payload(
            split_first_dir / f"diff_optimizer_{split_step:08d}.bin"
        ),
    )
    assert all(math.isfinite(value) for value in first_lambdas)
    assert all(math.isfinite(value) for value in first_moments)
    assert all(value >= 0.0 for value in first_moments[num_layers:])
    straight_moments = struct.unpack(
        f"<{num_layers * 2}f",
        _sidecar_payload(
            straight_dir / f"diff_optimizer_{final_step:08d}.bin"
        ),
    )
    assert any(value != 0.0 for value in straight_moments[:num_layers])
    assert all(value > 0.0 for value in straight_moments[num_layers:])

    renamed_graph = tmp_path / "renamed-byte-identical-source.json"
    renamed_graph.write_bytes(diff_graph.read_bytes())

    resumed = _run(
        _training_command(
            runtime_cli,
            dataset=dataset,
            graph=renamed_graph,
            output_dir=split_resume_dir,
            tile_ops=runtime_tile_ops,
            template_name="gpt2_diff",
            max_steps=final_step - split_step,
            resume=first_checkpoint,
            # Omit the total horizon on the second leg: strict diff resume
            # inherits the persisted value while validating every other knob.
            **continuation_options,
        )
    )
    assert resumed.returncode == 0, resumed.stderr
    final_artifacts = (
        f"model_{final_step:08d}.bin",
        f"parameters_{final_step:08d}.bin",
        f"optimizer_{final_step:08d}.bin",
        f"diff_parameters_{final_step:08d}.bin",
        f"diff_optimizer_{final_step:08d}.bin",
    )
    for artifact in final_artifacts:
        assert (straight_dir / artifact).read_bytes() == (
            split_resume_dir / artifact
        ).read_bytes(), artifact
    resumed_payload = _payload(resumed)
    assert resumed_payload["resume_diff_metadata_validated"] is True
    assert resumed_payload["resume_diff_parameter_restored"] is True
    assert resumed_payload["resume_diff_optimizer_restored"] is True
    assert resumed_payload["resume_sampler_seek_applied"] is True
    assert resumed_payload["graph_preflight_proof_verified"] is True
    assert resumed_payload["graph_preflight_proof_schema"] == (
        "neuralfn.native_graph_training_proof"
    )
    assert resumed_payload["graph_preflight_proof_version"] == 1
    assert resumed_payload["graph_preflight_contract_sha256"] == (
        proof_envelope["contract_sha256"]
    )
    assert resumed_payload["lr_schedule_total_steps"] == 0
    assert resumed_payload["lr_schedule_total_steps_explicit"] is False
    assert resumed_payload["effective_lr_schedule_total_steps"] == final_step
    assert resumed_payload["optimizer"]["effective_lr_schedule_total_steps"] == final_step
    assert resumed_payload["training_sampler_start_batch"] == continuation[
        "sampler_start_batch"
    ]
    resumed_manifest_path = (
        split_resume_dir / f"model_{final_step:08d}.diff.json"
    )
    resumed_manifest = json.loads(
        resumed_manifest_path.read_text(encoding="utf-8")
    )
    assert resumed_manifest["continuation"]["lr_schedule_total_steps"] == final_step
    assert resumed_manifest["graph_preflight_proof"] == manifest[
        "graph_preflight_proof"
    ]
    artifact_keys = (
        "model",
        "dense_parameters",
        "dense_optimizer",
        "diff_parameters",
        "diff_optimizer",
    )
    for key, filename in zip(artifact_keys, final_artifacts, strict=True):
        artifact_path = split_resume_dir / filename
        artifact_metadata = resumed_manifest[key]
        assert artifact_metadata["path"] == filename
        assert artifact_metadata["nbytes"] == artifact_path.stat().st_size
        assert artifact_metadata["sha256"] == hashlib.sha256(
            artifact_path.read_bytes()
        ).hexdigest()

    dense_read = _run(
        _training_command(
            runtime_cli,
            dataset=dataset,
            graph=dense_graph,
            output_dir=tmp_path / "dense-read",
            tile_ops=runtime_tile_ops,
            template_name="gpt2",
            max_steps=0,
            resume=first_checkpoint,
            startup_only=True,
            lr_schedule_total_steps=final_step,
            **dense_continuation_options,
        )
    )
    assert dense_read.returncode == 0, dense_read.stderr
    assert _payload(dense_read)["resume_checkpoint_loaded"] is True

    missing_dir = tmp_path / "missing-diff"
    shutil.copytree(split_first_dir, missing_dir, copy_function=os.link)
    (missing_dir / f"model_{split_step:08d}.diff.json").unlink()
    missing_output = tmp_path / "missing-out"
    missing = _run(
        _training_command(
            runtime_cli,
            dataset=dataset,
            graph=diff_graph,
            output_dir=missing_output,
            tile_ops=runtime_tile_ops,
            template_name="gpt2_diff",
            max_steps=0,
            resume=missing_dir / f"model_{split_step:08d}.bin",
            startup_only=True,
            **continuation_options,
        )
    )
    _assert_strict_resume_rejected_before_cuda(
        missing, output_dir=missing_output
    )

    input_symlink_dir = tmp_path / "input-symlink-diff"
    shutil.copytree(split_first_dir, input_symlink_dir, copy_function=os.link)
    symlink_sidecar = (
        input_symlink_dir / f"diff_optimizer_{split_step:08d}.bin"
    )
    symlink_sidecar.unlink()
    symlink_sidecar.symlink_to(
        split_first_dir / f"diff_optimizer_{split_step:08d}.bin"
    )
    input_symlink_output = tmp_path / "input-symlink-out"
    input_symlink_rejected = _run(
        _training_command(
            runtime_cli,
            dataset=dataset,
            graph=diff_graph,
            output_dir=input_symlink_output,
            tile_ops=runtime_tile_ops,
            template_name="gpt2_diff",
            max_steps=0,
            resume=input_symlink_dir / f"model_{split_step:08d}.bin",
            startup_only=True,
            **continuation_options,
        )
    )
    input_symlink_payload = _assert_strict_resume_rejected_before_cuda(
        input_symlink_rejected, output_dir=input_symlink_output
    )
    assert "failed to open regular non-symlink file" in str(
        input_symlink_payload["error"]
    )

    duplicate_key_dir = tmp_path / "duplicate-key-diff"
    shutil.copytree(split_first_dir, duplicate_key_dir, copy_function=os.link)
    duplicate_manifest = (
        duplicate_key_dir / f"model_{split_step:08d}.diff.json"
    )
    duplicate_text = duplicate_manifest.read_text(encoding="utf-8")
    assert duplicate_text.startswith("{\n")
    duplicate_manifest.unlink()
    duplicate_manifest.write_text(
        duplicate_text.replace("{\n", "{\n  \"version\": 2,\n", 1),
        encoding="utf-8",
    )
    duplicate_output = tmp_path / "duplicate-key-out"
    duplicate_rejected = _run(
        _training_command(
            runtime_cli,
            dataset=dataset,
            graph=diff_graph,
            output_dir=duplicate_output,
            tile_ops=runtime_tile_ops,
            template_name="gpt2_diff",
            max_steps=0,
            resume=duplicate_key_dir / f"model_{split_step:08d}.bin",
            startup_only=True,
            **continuation_options,
        )
    )
    duplicate_payload = _assert_strict_resume_rejected_before_cuda(
        duplicate_rejected, output_dir=duplicate_output
    )
    assert "duplicate" in str(duplicate_payload["error"]).lower()

    proof_metadata_dir = tmp_path / "proof-metadata-tamper-diff"
    shutil.copytree(
        split_first_dir, proof_metadata_dir, copy_function=os.link
    )
    proof_metadata_manifest = (
        proof_metadata_dir / f"model_{split_step:08d}.diff.json"
    )
    proof_metadata_document = json.loads(
        proof_metadata_manifest.read_text(encoding="utf-8")
    )
    proof_metadata_document["graph_preflight_proof"]["contract_sha256"] = (
        "0" * 64
    )
    proof_metadata_manifest.unlink()
    proof_metadata_manifest.write_text(
        json.dumps(proof_metadata_document), encoding="utf-8"
    )
    proof_metadata_output = tmp_path / "proof-metadata-tamper-out"
    proof_metadata_rejected = _run(
        _training_command(
            runtime_cli,
            dataset=dataset,
            graph=diff_graph,
            output_dir=proof_metadata_output,
            tile_ops=runtime_tile_ops,
            template_name="gpt2_diff",
            max_steps=0,
            resume=proof_metadata_dir / f"model_{split_step:08d}.bin",
            startup_only=True,
            **continuation_options,
        )
    )
    _assert_strict_resume_rejected_before_cuda(
        proof_metadata_rejected,
        output_dir=proof_metadata_output,
        expected_error=(
            DIFF_RESUME_ERROR_PREFIX
            + "graph_preflight_proof does not match this resume"
        ),
    )

    nbytes_dir = tmp_path / "nbytes-tamper-diff"
    shutil.copytree(split_first_dir, nbytes_dir, copy_function=os.link)
    nbytes_manifest = nbytes_dir / f"model_{split_step:08d}.diff.json"
    nbytes_document = json.loads(nbytes_manifest.read_text(encoding="utf-8"))
    nbytes_document["diff_parameters"]["nbytes"] += 4
    nbytes_manifest.unlink()
    nbytes_manifest.write_text(json.dumps(nbytes_document), encoding="utf-8")
    nbytes_output = tmp_path / "nbytes-tamper-out"
    nbytes_rejected = _run(
        _training_command(
            runtime_cli,
            dataset=dataset,
            graph=diff_graph,
            output_dir=nbytes_output,
            tile_ops=runtime_tile_ops,
            template_name="gpt2_diff",
            max_steps=0,
            resume=nbytes_dir / f"model_{split_step:08d}.bin",
            startup_only=True,
            **continuation_options,
        )
    )
    _assert_strict_resume_rejected_before_cuda(
        nbytes_rejected,
        output_dir=nbytes_output,
        expected_error=(
            DIFF_RESUME_ERROR_PREFIX
            + "diff_parameters.nbytes does not match the expected layout"
        ),
    )

    tamper_dir = tmp_path / "tamper-diff"
    shutil.copytree(split_first_dir, tamper_dir, copy_function=os.link)
    tampered_path = tamper_dir / f"diff_parameters_{split_step:08d}.bin"
    tampered = bytearray(tampered_path.read_bytes())
    tampered[DIFF_HEADER_BYTES] ^= 0x01
    tampered_path.unlink()
    tampered_path.write_bytes(tampered)
    tamper_output = tmp_path / "tamper-out"
    rejected = _run(
        _training_command(
            runtime_cli,
            dataset=dataset,
            graph=diff_graph,
            output_dir=tamper_output,
            tile_ops=runtime_tile_ops,
            template_name="gpt2_diff",
            max_steps=0,
            resume=tamper_dir / f"model_{split_step:08d}.bin",
            startup_only=True,
            **continuation_options,
        )
    )
    _assert_strict_resume_rejected_before_cuda(
        rejected,
        output_dir=tamper_output,
        expected_error=(
            DIFF_RESUME_ERROR_PREFIX
            + "diff_parameters SHA-256 does not match the artifact bytes"
        ),
    )

    header_dir = tmp_path / "header-tamper-diff"
    shutil.copytree(split_first_dir, header_dir, copy_function=os.link)
    header_path = header_dir / f"diff_parameters_{split_step:08d}.bin"
    header_bytes = bytearray(header_path.read_bytes())
    struct.pack_into("<q", header_bytes, 6 * 8, 17)
    header_path.unlink()
    header_path.write_bytes(header_bytes)
    _replace_manifest_artifact_hash(header_dir, "diff_parameters", header_path)
    header_output = tmp_path / "header-tamper-out"
    header_rejected = _run(
        _training_command(
            runtime_cli,
            dataset=dataset,
            graph=diff_graph,
            output_dir=header_output,
            tile_ops=runtime_tile_ops,
            template_name="gpt2_diff",
            max_steps=0,
            resume=header_dir / f"model_{split_step:08d}.bin",
            startup_only=True,
            **continuation_options,
        )
    )
    _assert_strict_resume_rejected_before_cuda(
        header_rejected,
        output_dir=header_output,
        expected_error=(
            DIFF_RESUME_ERROR_PREFIX
            + f"diff_parameters_{split_step:08d}.bin header mismatch"
        ),
    )

    moment_dir = tmp_path / "moment-tamper-diff"
    shutil.copytree(split_first_dir, moment_dir, copy_function=os.link)
    moment_path = moment_dir / f"diff_optimizer_{split_step:08d}.bin"
    moment_bytes = bytearray(moment_path.read_bytes())
    struct.pack_into(
        "<f", moment_bytes, DIFF_HEADER_BYTES + num_layers * 4, -1.0
    )
    moment_path.unlink()
    moment_path.write_bytes(moment_bytes)
    _replace_manifest_artifact_hash(moment_dir, "diff_optimizer", moment_path)
    moment_output = tmp_path / "moment-tamper-out"
    moment_rejected = _run(
        _training_command(
            runtime_cli,
            dataset=dataset,
            graph=diff_graph,
            output_dir=moment_output,
            tile_ops=runtime_tile_ops,
            template_name="gpt2_diff",
            max_steps=0,
            resume=moment_dir / f"model_{split_step:08d}.bin",
            startup_only=True,
            **continuation_options,
        )
    )
    _assert_strict_resume_rejected_before_cuda(
        moment_rejected,
        output_dir=moment_output,
        expected_error=(
            DIFF_RESUME_ERROR_PREFIX
            + "diff optimizer second moments must be finite and non-negative"
        ),
    )

    dense_header_dir = tmp_path / "dense-header-tamper"
    shutil.copytree(split_first_dir, dense_header_dir, copy_function=os.link)
    dense_parameters_path = (
        dense_header_dir / f"parameters_{split_step:08d}.bin"
    )
    dense_parameter_bytes = bytearray(dense_parameters_path.read_bytes())
    struct.pack_into("<q", dense_parameter_bytes, 13 * 8, 1)
    dense_parameters_path.unlink()
    dense_parameters_path.write_bytes(dense_parameter_bytes)
    _replace_manifest_artifact_hash(
        dense_header_dir, "dense_parameters", dense_parameters_path
    )
    dense_header_output = tmp_path / "dense-header-out"
    dense_header_rejected = _run(
        _training_command(
            runtime_cli,
            dataset=dataset,
            graph=diff_graph,
            output_dir=dense_header_output,
            tile_ops=runtime_tile_ops,
            template_name="gpt2_diff",
            max_steps=0,
            resume=dense_header_dir / f"model_{split_step:08d}.bin",
            startup_only=True,
            **continuation_options,
        )
    )
    _assert_strict_resume_rejected_before_cuda(
        dense_header_rejected,
        output_dir=dense_header_output,
        expected_error=(
            DIFF_RESUME_ERROR_PREFIX
            + f"parameters_{split_step:08d}.bin header mismatch"
        ),
    )

    dense_nan_dir = tmp_path / "dense-nan-tamper"
    shutil.copytree(split_first_dir, dense_nan_dir, copy_function=os.link)
    dense_optimizer_path = dense_nan_dir / f"optimizer_{split_step:08d}.bin"
    dense_optimizer_bytes = bytearray(dense_optimizer_path.read_bytes())
    struct.pack_into("<f", dense_optimizer_bytes, 32 * 8, float("nan"))
    dense_optimizer_path.unlink()
    dense_optimizer_path.write_bytes(dense_optimizer_bytes)
    _replace_manifest_artifact_hash(
        dense_nan_dir, "dense_optimizer", dense_optimizer_path
    )
    dense_nan_output = tmp_path / "dense-nan-out"
    dense_nan_rejected = _run(
        _training_command(
            runtime_cli,
            dataset=dataset,
            graph=diff_graph,
            output_dir=dense_nan_output,
            tile_ops=runtime_tile_ops,
            template_name="gpt2_diff",
            max_steps=0,
            resume=dense_nan_dir / f"model_{split_step:08d}.bin",
            startup_only=True,
            **continuation_options,
        )
    )
    _assert_strict_resume_rejected_before_cuda(
        dense_nan_rejected,
        output_dir=dense_nan_output,
        expected_error=(
            DIFF_RESUME_ERROR_PREFIX
            + "dense optimizer first moments must be finite"
        ),
    )

    drift_cases = (
        (
            "seed-omitted",
            {},
            "continuation.train_seed_explicit does not match this resume",
        ),
        (
            "seed-changed",
            {"train_seed": 732},
            "continuation.train_seed does not match this resume",
        ),
        (
            "batch-changed",
            {"batch_size": 2, "train_batch_tokens": 64},
            "continuation.batch_size does not match this resume",
        ),
        (
            "horizon-changed",
            {"lr_schedule_total_steps": 5},
            "continuation.lr_schedule_total_steps does not match this resume",
        ),
    )
    for case_name, overrides, message in drift_cases:
        options = dict(continuation_options)
        if case_name == "seed-omitted":
            options["train_seed"] = None
        options.update(overrides)
        output_dir = tmp_path / f"{case_name}-out"
        drift = _run(
            _training_command(
                runtime_cli,
                dataset=dataset,
                graph=diff_graph,
                output_dir=output_dir,
                tile_ops=runtime_tile_ops,
                template_name="gpt2_diff",
                max_steps=0,
                resume=first_checkpoint,
                startup_only=True,
                **options,
            )
        )
        _assert_strict_resume_rejected_before_cuda(
            drift,
            output_dir=output_dir,
            expected_error=DIFF_RESUME_ERROR_PREFIX + message,
        )

    numerics_drift_cases = (
        (
            "bf16-weight-route",
            {"NFN_NATIVE_GPT2_BF16_BLOCK_WEIGHT_PARAMS": "0"},
            "continuation.bf16_block_weight_params does not match this resume",
        ),
        (
            "bf16-dweight-staging-route",
            {"NFN_NATIVE_GPT2_BF16_BLOCK_DWEIGHT_STAGING": "1"},
            "continuation.bf16_block_dweight_staging does not match this resume",
        ),
        (
            "first-microbatch-beta-route",
            {"NFN_NATIVE_GPT2_DWEIGHT_FIRST_MICROBATCH_BETA_ZERO": "0"},
            "continuation.dweight_first_microbatch_beta_zero does not match this resume",
        ),
        (
            "ce-reduction-profile-route",
            {"NFN_TILE_CUDA_CE_BF16_THREADS": "512"},
            "continuation.numerics_profile_sha256 does not match this resume",
        ),
        (
            "mlp-recompute-profile-route",
            {"NFN_NATIVE_GPT2_STORE_MLP_BLOCKS": "0"},
            "continuation.numerics_profile_sha256 does not match this resume",
        ),
        (
            "gemm-compute-profile-route",
            {"NFN_TILE_CUDA_LINEAR_BF16_GEMM_EX_FAST_16BF": "1"},
            "continuation.numerics_profile_sha256 does not match this resume",
        ),
        (
            "fused-bgrad-profile-route",
            {"NFN_NATIVE_GPT2_FUSE_BF16_BF16_DWEIGHT_BGRAD": "0"},
            "continuation.numerics_profile_sha256 does not match this resume",
        ),
        (
            "prob-correction-reduction-route",
            {"NFN_TILE_CUDA_LM_HEAD_PROB_ONLY_TARGET_CORRECTION_THREADS": "128"},
            "continuation.numerics_profile_sha256 does not match this resume",
        ),
        (
            "graph-body-serial-route",
            {"NFN_NATIVE_GPT2_LM_HEAD_GRAPH_BODY_SERIAL": "1"},
            "continuation.numerics_profile_sha256 does not match this resume",
        ),
        (
            "graph-body-conflicting-alias-route",
            {
                "NFN_TILE_CUDA_LM_HEAD_GRAPH_BODY_CUBLASLT": "0",
                "NFN_NATIVE_GPT_LM_HEAD_GRAPH_BODY_CUBLASLT": "1",
            },
            "continuation.numerics_profile_sha256 does not match this resume",
        ),
        (
            "ce-conflicting-alias-route",
            {
                "NFN_NATIVE_GPT_CE_BF16_EXP2": "0",
                "NFN_TILE_CUDA_CE_BF16_EXP2": "1",
            },
            "continuation.numerics_profile_sha256 does not match this resume",
        ),
        (
            "descriptor-cache-route",
            {"NFN_TILE_CUDA_CUBLASLT_DESCRIPTOR_CACHE": "0"},
            "continuation.numerics_profile_sha256 does not match this resume",
        ),
    )
    for case_name, env_overrides, message in numerics_drift_cases:
        output_dir = tmp_path / f"{case_name}-out"
        drift = _run(
            _training_command(
                runtime_cli,
                dataset=dataset,
                graph=diff_graph,
                output_dir=output_dir,
                tile_ops=runtime_tile_ops,
                template_name="gpt2_diff",
                max_steps=0,
                resume=first_checkpoint,
                startup_only=True,
                **continuation_options,
            ),
            env_overrides=env_overrides,
        )
        _assert_strict_resume_rejected_before_cuda(
            drift,
            output_dir=output_dir,
            expected_error=DIFF_RESUME_ERROR_PREFIX + message,
        )

    # Benign progress telemetry and an explicitly spelled alias that resolves
    # to the checkpoint's default numerical value must not perturb the
    # canonical effective numerics profile.
    compatible_output = tmp_path / "benign-env-compatible-out"
    compatible = _run(
        _training_command(
            runtime_cli,
            dataset=dataset,
            graph=diff_graph,
            output_dir=compatible_output,
            tile_ops=runtime_tile_ops,
            template_name="gpt2_diff",
            max_steps=0,
            resume=first_checkpoint,
            startup_only=True,
            **continuation_options,
        ),
        env_overrides={
            "NFN_NATIVE_GPT_SETUP_PROGRESS": "1",
            "NFN_NATIVE_GPT2_BF16_BLOCK_WEIGHT_PARAMS": "1",
            "NFN_NATIVE_GPT2_LM_HEAD_GRAPH_BODY_CUBLASLT": "0",
        },
    )
    assert compatible.returncode == 0, compatible.stderr
    compatible_payload = _payload(compatible)
    assert compatible_payload["loaded"] is True
    assert compatible_payload["resume_checkpoint_loaded"] is True
    assert compatible_payload["resume_diff_parameter_restored"] is True
    assert compatible_payload["resume_diff_optimizer_restored"] is True
    assert not compatible_output.exists()

    altered_dataset = tmp_path / "altered-tokens"
    shutil.copytree(dataset, altered_dataset)
    altered_shard = altered_dataset / "fineweb_train_000000.bin"
    altered_bytes = bytearray(altered_shard.read_bytes())
    altered_bytes[0] ^= 0x01
    altered_shard.write_bytes(altered_bytes)
    dataset_output = tmp_path / "dataset-drift-out"
    dataset_rejected = _run(
        _training_command(
            runtime_cli,
            dataset=altered_dataset,
            graph=diff_graph,
            output_dir=dataset_output,
            tile_ops=runtime_tile_ops,
            template_name="gpt2_diff",
            max_steps=0,
            resume=first_checkpoint,
            startup_only=True,
            **continuation_options,
        )
    )
    _assert_strict_resume_rejected_before_cuda(
        dataset_rejected,
        output_dir=dataset_output,
        expected_error=(
            DIFF_RESUME_ERROR_PREFIX
            + "continuation.train_shards_sha256 does not match this resume"
        ),
    )

    mismatched_graph = tmp_path / "same-graph-modified-bytes.json"
    mismatched_graph.write_bytes(diff_graph.read_bytes() + b"\n")
    graph_output = tmp_path / "graph-mismatch-out"
    graph_rejected = _run(
        _training_command(
            runtime_cli,
            dataset=dataset,
            graph=mismatched_graph,
            output_dir=graph_output,
            tile_ops=runtime_tile_ops,
            template_name="gpt2_diff",
            max_steps=0,
            resume=first_checkpoint,
            startup_only=True,
            **continuation_options,
        )
    )
    assert graph_rejected.returncode == 2
    assert graph_rejected.stdout == ""
    assert graph_rejected.stderr.strip() == (
        "native GPT gpt2_diff packed differential preflight failed: "
        "invalid graph preflight proof: source_graph_sha256 does not match "
        "the verified graph bytes"
    )
    assert not graph_output.exists()


def test_checkpoint_publication_refuses_preplanted_symlink_without_partial_files(
    runtime_cli: Path,
    runtime_tile_ops: Path,
    tmp_path: Path,
) -> None:
    dataset = _write_uint16_dataset(tmp_path)
    authored_graph = _write_graph(tmp_path, "gpt2_diff")
    graph, proof = _materialize_proven_diff_graph(tmp_path, authored_graph)
    output_dir = tmp_path / "symlink-output"
    output_dir.mkdir()
    outside = tmp_path / "outside-sentinel"
    outside.write_text("must-not-change", encoding="utf-8")
    attacked_target = output_dir / "model_00000001.diff.json"
    attacked_target.symlink_to(outside)

    completed = _run(
        _training_command(
            runtime_cli,
            dataset=dataset,
            graph=graph,
            proof=proof,
            output_dir=output_dir,
            tile_ops=runtime_tile_ops,
            template_name="gpt2_diff",
            max_steps=1,
            train_seed=911,
            lr_schedule="constant",
            warmup_steps=0,
        )
    )

    assert completed.returncode == 2, completed.stderr
    payload = _payload(completed)
    assert payload["error"] == (
        "refusing to overwrite existing checkpoint target: "
        + str(attacked_target)
    )
    assert outside.read_text(encoding="utf-8") == "must-not-change"
    assert attacked_target.is_symlink()
    assert sorted(path.name for path in output_dir.iterdir()) == [
        attacked_target.name
    ]


@pytest.mark.skipif(
    os.environ.get(RUNTIME_ENV) != "1",
    reason=f"set {RUNTIME_ENV}=1 to run the stable-shard mutation check",
)
def test_stable_diff_shard_mutation_fails_with_structured_cleanup(
    runtime_cli: Path,
    runtime_tile_ops: Path,
    tmp_path: Path,
) -> None:
    dataset = _write_uint16_dataset(tmp_path)
    authored_graph = _write_graph(tmp_path, "gpt2_diff")
    graph, proof = _materialize_proven_diff_graph(tmp_path, authored_graph)
    output_dir = tmp_path / "mutated-shard-output"
    command = _training_command(
        runtime_cli,
        dataset=dataset,
        graph=graph,
        proof=proof,
        output_dir=output_dir,
        tile_ops=runtime_tile_ops,
        template_name="gpt2_diff",
        max_steps=1,
        train_seed=317,
        lr_schedule="constant",
        warmup_steps=0,
    )
    env = _runtime_env()
    env["NFN_NATIVE_GPT_SETUP_PROGRESS"] = "1"
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    stderr_lines: list[str] = []
    marker_seen = False
    while True:
        line = process.stderr.readline()
        if line == "":
            break
        stderr_lines.append(line)
        if "setup done setup.diff_continuation_host_contract" in line:
            marker_seen = True
            break
    assert marker_seen, "trainer exited before retaining its verified shard descriptor"
    shard = dataset / "fineweb_train_000000.bin"
    with shard.open("r+b") as handle:
        first_byte = handle.read(1)
        assert first_byte
        handle.seek(0)
        handle.write(bytes([first_byte[0] ^ 0x01]))
        handle.flush()
        os.fsync(handle.fileno())
    stdout, remaining_stderr = process.communicate(timeout=300)
    completed = subprocess.CompletedProcess(
        command,
        process.returncode,
        stdout,
        "".join(stderr_lines) + remaining_stderr,
    )
    assert completed.returncode == 2, completed.stderr
    payload = _payload(completed)
    assert str(payload["error"]).startswith(
        "failed to read stable gpt2_diff train shard: "
        "stable token shard changed before batch read:"
    )
    assert payload["steps_completed"] == 0
    assert not output_dir.exists()
    assert payload["checkpoint"]["checkpoint_written"] is False
    assert payload["checkpoint"]["parameter_state_checkpoint_written"] is False
    assert payload["checkpoint"]["optimizer_checkpoint_written"] is False
    assert payload["checkpoint"]["diff_checkpoint_written"] is False
