from __future__ import annotations

import ctypes
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TILE_OPS_TEST_LIBRARY_ENV = "NFN_NATIVE_TILE_SPARSE_ATTENTION_ABI_LIB"

_AFFECTED_RAW_SPARSE_ENTRYPOINTS = {
    "nfn_native_tile_scaled_dot_product_attention_float32": (
        "neuralfn::tile_cuda::launch_scaled_dot_product_attention_float32("
    ),
    "nfn_native_tile_scaled_dot_product_attention_backward_float32": (
        "neuralfn::tile_cuda::launch_scaled_dot_product_attention_backward_float32("
    ),
    "nfn_native_tile_scaled_dot_product_attention_backward_from_merged_grad_float32": (
        "neuralfn::tile_cuda::launch_scaled_dot_product_attention_backward_from_merged_grad_float32("
    ),
}


def _function_body(source: str, name: str) -> str:
    start = source.index(f"int {name}(")
    opening_brace = source.index("{", start)
    depth = 0
    for index in range(opening_brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[opening_brace : index + 1]
    raise AssertionError(f"unterminated function body for {name}")


def test_raw_sparse_attention_limit_is_checked_before_every_affected_launch() -> None:
    source = (ROOT / "neuralfn/csrc/native_train/tile_ops.cu").read_text()
    validator = _function_body(
        source, "validate_raw_sparse_attention_key_sequence_length"
    )

    assert "kRawSparseAttentionMaxKeySequenceLength = 1024" in source
    assert (
        "use_sparse_rules && seq_k > kRawSparseAttentionMaxKeySequenceLength"
        in validator
    )
    assert "seq_k >= kRawSparseAttentionMaxKeySequenceLength" not in validator
    assert source.count("validate_raw_sparse_attention_key_sequence_length(") == 4

    for entrypoint, launch_call in _AFFECTED_RAW_SPARSE_ENTRYPOINTS.items():
        body = _function_body(source, entrypoint)
        validation = body.index("validate_raw_sparse_attention_key_sequence_length(")
        rejection = body.index("return validation_status;")
        launch = body.index(launch_call)
        assert validation < rejection < launch


def _configure_raw_attention_abi(library: ctypes.CDLL) -> None:
    pointer = ctypes.c_void_p
    i64 = ctypes.c_int64
    boolean = ctypes.c_bool

    library.nfn_native_tile_scaled_dot_product_attention_float32.argtypes = [
        pointer,
        pointer,
        pointer,
        pointer,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        ctypes.c_float,
        boolean,
        boolean,
        boolean,
        i64,
        i64,
        i64,
        i64,
        pointer,
    ]
    library.nfn_native_tile_scaled_dot_product_attention_float32.restype = ctypes.c_int

    backward_argtypes = [
        pointer,
        pointer,
        pointer,
        pointer,
        pointer,
        pointer,
        pointer,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        ctypes.c_float,
        boolean,
        boolean,
        boolean,
        i64,
        i64,
        i64,
        i64,
        pointer,
    ]
    for name in (
        "nfn_native_tile_scaled_dot_product_attention_backward_float32",
        "nfn_native_tile_scaled_dot_product_attention_backward_from_merged_grad_float32",
    ):
        function = getattr(library, name)
        function.argtypes = backward_argtypes
        function.restype = ctypes.c_int

    library.nfn_native_tile_attention_forward_stats_reset.argtypes = []
    library.nfn_native_tile_attention_forward_stats_reset.restype = None
    library.nfn_native_tile_attention_forward_scalar_launch_count.argtypes = []
    library.nfn_native_tile_attention_forward_scalar_launch_count.restype = ctypes.c_int64


def _call_raw_forward(
    library: ctypes.CDLL, *, seq_k: int, use_sparse_rules: bool
) -> int:
    return int(
        library.nfn_native_tile_scaled_dot_product_attention_float32(
            None,
            None,
            None,
            None,
            0,
            1,
            1,
            1,
            seq_k,
            1,
            1,
            1.0,
            True,
            False,
            use_sparse_rules,
            16,
            0,
            0,
            1,
            None,
        )
    )


def _call_raw_backward(library: ctypes.CDLL, name: str, *, seq_k: int) -> int:
    function = getattr(library, name)
    return int(
        function(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            1,
            1,
            1,
            1,
            seq_k,
            1,
            1,
            1.0,
            True,
            False,
            True,
            16,
            0,
            0,
            1,
            None,
        )
    )


def test_compiled_raw_sparse_attention_abi_rejects_only_above_1024() -> None:
    library_path_text = os.environ.get(TILE_OPS_TEST_LIBRARY_ENV, "").strip()
    if not library_path_text:
        pytest.skip(
            f"set {TILE_OPS_TEST_LIBRARY_ENV} to a freshly built Tile ops library"
        )
    library_path = Path(library_path_text)
    if not library_path.is_file():
        pytest.fail(f"Tile ops test library does not exist: {library_path}")

    library = ctypes.CDLL(str(library_path))
    _configure_raw_attention_abi(library)

    cuda_error_invalid_value = 1
    assert _call_raw_forward(
        library, seq_k=1025, use_sparse_rules=True
    ) == cuda_error_invalid_value
    for name in (
        "nfn_native_tile_scaled_dot_product_attention_backward_float32",
        "nfn_native_tile_scaled_dot_product_attention_backward_from_merged_grad_float32",
    ):
        assert _call_raw_backward(
            library, name, seq_k=1025
        ) == cuda_error_invalid_value

    library.nfn_native_tile_attention_forward_stats_reset()
    assert library.nfn_native_tile_attention_forward_scalar_launch_count() == 0
    assert _call_raw_forward(
        library, seq_k=1025, use_sparse_rules=True
    ) == cuda_error_invalid_value
    assert library.nfn_native_tile_attention_forward_scalar_launch_count() == 0

    # A zero-work launch avoids dereferencing the null fixture buffers. Reaching the
    # scalar-launch counter proves that the inclusive 1024-key boundary is retained.
    _call_raw_forward(library, seq_k=1024, use_sparse_rules=True)
    assert library.nfn_native_tile_attention_forward_scalar_launch_count() == 1
