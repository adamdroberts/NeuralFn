# TurboQuant Portable Reference

`neuralfn.turboquant` is the dependency-free correctness oracle and shared
table source for the exact-dense resident CPU KV-cache codec. It is based on the
[TurboQuant paper](https://arxiv.org/abs/2504.19874) and is intentionally
separate from the historical graph `kv_quant_pack` / `kv_quant_unpack` stages.
Those existing stages keep their current dtype, shape, and behavior.

The Native Execution compiler enables `turboquant_kv_cache` only for a bound
reviewed dense-v5 artifact whose topology and even head geometry pass the exact
resident gate. The reviewed preset set is `gpt2`, `gpt2_megakernel`,
`gpt2_moa`, `gpt2_zloss`, `gpt2_qknorm`, `gpt2_stable`, and `gpt2_softcap`;
QK-norm, softcap, and MoA selection contracts are independently checked. MoA
must be migrated from source-bound sibling metadata and uses the recorded
activation without candidate reprobes. The compiled CPU binding then
requires cache ABI v1 and the same deterministic tables.
Differential/modern variants, unbound MoA `.bin` files, non-dense state, and odd
head dimensions remain fail-closed. The trainer Tile sidecar exports an
additive CUDA attention feature ABI that directly reads the same CPU-v1 packed
records. Compatible reviewed-dense artifacts now advertise that feature
separately and resident sessions consume it only after explicit `tile-cuda`
configuration; CPU remains the default.

## CUDA sidecar agreement

`NfnNativeTileTurboQuantAttentionDescriptorV1` and these additive symbols live
in `tile_ops.h` without changing `nfn_native_tile_ops_abi_version() == 1`:

- `nfn_native_tile_turboquant_attention_abi_version()`;
- `nfn_native_tile_turboquant_attention_forward_v1()`;
- `nfn_native_tile_turboquant_attention_stats_reset()`; and
- `nfn_native_tile_turboquant_attention_launch_count()`.

The size-prefixed descriptor uses device pointers and explicit geometry/stride
fields. Historical keys and values retain the CPU-v1 byte layout; the kernel
decodes mixed-bit centroids in place and never constructs a sequence-by-head
dequantized cache. It supports separate query/KV head counts, includes one
lossless current K/V row in the same stable softmax, and processes long inputs
in deterministic 256-row online-softmax chunks. The maximum v1 total context is
16,384 rows (up to 16,383 compressed historical rows plus the current row).

Fail-honest live coverage rebuilds both the normal fast-math/TK sidecar and the
strict sidecar:

```bash
NFN_NATIVE_TURBOQUANT_CUDA_TEST=1 \
  python -m pytest tests/test_turboquant_cuda.py -q -rs
```

The RTX 5090 acceptance passes both MSE and QJL profiles, MHA and GQA,
dimensions 8/64/128, empty history, 1023/1024/1025 and 4K/16K boundaries,
bitwise repeated launches, launch-counter proof, and invalid descriptors
against the portable and native CPU oracles. This does not measure throughput,
VRAM, perplexity, or token agreement and does not change
`kernel_abi.turboquant_cache` from its CPU resident meaning. The resident
integration uses the separate `kernel_abi.turboquant_tile_attention` v1 gate,
requires the strict sidecar, and reports actual launch/transfer telemetry.

Run its end-to-end direct binding and public SDK lifecycle gate with:

```bash
NFN_NATIVE_TURBOQUANT_CUDA_TEST=1 \
  python -m pytest tests/test_native_resident_tile_turboquant.py -q -rs
```

The CPU still owns weights, projections, and row encoding; only packed
historical attention is GPU-resident. An explicit GPU request never falls back
to CPU.

Use `tools/bench_native_resident_turboquant.py` for the separate
transfer-inclusive full/CPU/Tile comparison. It runs timing and
quality/baseline-subtracted-device-memory in fresh workers, scores a bounded
teacher-forced tail through public `current_logits()`, records free-running and
teacher-forced greedy agreement, and rejects Tile runs without launch,
upload/transfer, zero-CPU-call, and positive sampled-VRAM evidence. Omitted
`--tokens-file` selects an explicitly labeled repeated-token synthetic corpus;
that mode validates mechanics only. The 2026-08-08 RTX 5090 1K/4K/16K
synthetic calibration matched Tile to the corresponding CPU profile exactly
but measured Tile slower on the tiny launch-dominated model. It makes no
speedup or quality-neutrality claim; see the detailed table in
[Resident Native Inference](native-inference.md#resident-turboquant-benchmark).

## Algorithms represented

The MSE path:

1. stores the input vector's L2 norm as float32;
2. normalizes and multiplies it by a deterministic seeded random orthogonal
   matrix generated from a Gaussian matrix with QR-style orthogonalization;
3. maps each rotated coordinate to a Lloyd-Max centroid for the exact
   sphere-coordinate density;
4. packs every mixed-width index into one contiguous little-endian bit stream;
   and
5. reconstructs a row with the shared codebooks and inverse rotation.

The QJL key path spends one bit per coordinate on the sign of a seeded Gaussian
projection of the normalized MSE residual and stores its float32 norm. A direct
key/query estimator adds the QJL correction without reconstructing a cache
matrix. Value rows always use the MSE reconstruction/weighted-accumulation path;
QJL is never applied to value accumulation.

## 3.5-bit profiles

`mse-3.5` assigns 4 bits to half of the fixed model-level outlier channels and
3 bits to the other half. `qjl-3.5` assigns 3/2 MSE bits to those channels and
uses the remaining one bit per channel for QJL residual signs. The caller must
supply exactly half of the channel indices as a fixed outlier set; the
reference does not silently invent a calibration policy. Head dimensions must
be even.

The 3.5 figure describes packed coordinate bits. Per-row norm/residual metadata
is counted separately by `TurboQuantEncodedVector.data_bytes`, so telemetry can
report actual rather than idealized storage.

## API

```python
from neuralfn import TurboQuantReferenceCodec

codec = TurboQuantReferenceCodec(
    128,
    profile="qjl-3.5",
    seed=7,
    outlier_indices=range(0, 128, 2),
)

encoded_key = codec.encode_key(key_row)
score = codec.key_inner_product(query_row, encoded_key)

encoded_value = codec.encode_value(value_row)
output = [0.0] * 128
codec.accumulate_value(output, attention_weight, encoded_value)
```

Other public helpers are:

- `deterministic_random_rotation(dimension, seed)`;
- `deterministic_qjl_projection(dimension, seed)`;
- `lloyd_max_centroids(dimension, bit_width)`;
- `pack_mixed_bit_indices(indices, bit_widths)`; and
- `unpack_mixed_bit_indices(payload, bit_widths)`.

Matrices/codebooks are built once per model/profile and shared across resident
sessions; they are not included in per-row byte accounting. The pure-Python
QR and quadrature steps establish deterministic tables. Row encoding, direct
key/query scoring, and weighted value accumulation run in C++ for resident
inference and never create a full dequantized cache matrix.

The table mapping passed to the private `_native_inference` extension is an
internal SDK/binding seam, not a public artifact input. The SDK always builds
the seed-0, even-channel-outlier tables above. C++ independently rejects
non-orthonormal rotations, any other width pattern, malformed or asymmetric
bounded codebooks, and degenerate QJL projection rows before creating a codec.
These structural checks do not claim an exact seed-0 fingerprint check for
arbitrary direct extension callers; use `NativeInferenceModel` instead of
constructing private binding table payloads.

## Precision and determinism

The seed stream uses SHA-256 plus a fixed Box-Muller transform, avoiding global
random state. Identical dimension, seed, outlier set, profile, and vector input
produce identical packed payloads. This establishes codec repeatability; it
does not promise equality with full-cache FP32 logits or tokens because the
codec is lossy.

The reference and compiled CPU implementation validate deterministic rotations, Lloyd-Max
symmetry/known limits, 3-bit byte straddling, mixed-width packing, norms,
payload byte counts, QJL seed-ensemble behavior, value-only MSE handling, and
lean imports. Byte-for-byte packed indices/signs and numerical key/value
operations agree between portable and native implementations for both
profiles. Resident tests cover deterministic greedy decode, exact actual versus
uncompressed byte telemetry, truncate/reset, session isolation, and explicit
lossy-cache reporting. The separate sidecar CUDA attention ABI also agrees as
described above. Live MSE/QJL resident tests prove dispatch and lifecycle with
zero CPU compressed-attention calls. Transfer-inclusive performance, VRAM,
perplexity delta, and greedy-token agreement benchmarks remain open.
