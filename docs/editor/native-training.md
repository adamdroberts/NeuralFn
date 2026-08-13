# Native CUDA Training in the Editor

The editor graph runtime is the union `scalar | torch | native-cuda`. Select
`native-cuda` from the graph settings panel to use the graph-authored native
training workflow.

## Start workflow

1. Keep the authored model graph on an exact reviewed native adapter topology.
   The dataset is selected separately in the Training panel and is not inserted
   as a `dataset_source` node.
2. Select one project-accessible cached dataset alias.
3. Press **Train**. The editor saves the current graph revision, calls
   `POST .../runs/preflight`, and waits for a compatible result.
4. Only an `execution_ready: true` result starts the normal run SSE request.

The preflight result is rendered before training starts. Every lowering or
adapter failure includes its stable graph path, error code, operation, and
message, for example:

```text
root/nodes/model/subgraph/nodes/token_embed [unsupported_module]: ...
```

The editor does not reinterpret an unsupported graph, fall back to Torch, or
silently choose a nearby native family.

## Current adapter boundary

The current execution-ready graph-file adapters are the exact reviewed GPT-2
profiles `gpt2`, `gpt2_megakernel`, `gpt2_moa`, `gpt2_qknorm`,
`gpt2_softcap`, `gpt2_stable`, and `gpt2_zloss`, plus exact canonical `llama`
and its exact compile-runtime alias `llama_fast`, plus exact standard-MoE
`moe`, `mixllama`, and `mixllama_fast`, plus trusted-planner proof-bound
`gpt2_diff` training (13 ready; 54 blocked).
The LLaMA adapter requires the reviewed RMSNorm/RoPE/MHA-or-GQA/dense-attention/
gate-first-SwiGLU topology and binds graph geometry and SHA-256 through training
and checkpoint discovery. Both profiles use the canonical native `llama` ABI
while retaining source-profile provenance. Other structurally lowerable presets remain
incompatible until their architecture-persistence adapter is proved. In
particular, `gpt2_diff` training requires the materialized
graph/fingerprint/proof triplet and the proof's unkeyed digest is local-handoff
integrity, not caller authenticity. Migration and resident inference do not yet
consume its graph-bound learned-lambda bundle; its exact low-level differential
path is packed-QKV-only.
That low-level path now emits version-2 strict continuation metadata binding all
five binaries, source graph, training-only shard identity, counters/sampler,
seed, accumulation shape, optimizer/LR horizon, BF16 routes, and a canonical
profile of supported effective numerics before Tile/CUDA/H2D. Editor completion
stores the strict validated `.diff.json`; this does not make migration or
resident differential attention/cache executable. The
compatibility response distinguishes structural Native IR support from
executable trainer support.

Each accepted run materializes an immutable per-run bundle under
`NEURALFN_ARTIFACTS_DIR/runs/<run-id>/native-ir/`, then invokes the same public
Native IR planner and native trainer registry/configuration used by the CLI.
The run record exposes its compatibility report, manifest/training-plan paths,
native command metadata, and the actual checkpoint path after completion.
For `gpt2_moa`, that path is the validated source-bound
`model_XXXXXXXX.moa.json`, not its sibling dense-v5 `.bin`; the metadata records
the final selected activation, canonical candidate set, positive interval,
model hash, graph hash, and DONE marker. Resume validates that sidecar and
restores its activation without a new candidate probe; missing or changed
metadata fails closed. This graph-bound contract does not remove direct
selector-only first-leg MoA training, whose ordinary dense-v5 output is not
accepted for exact resume.

Native editor training currently has these explicit limits:

- exactly one cached project dataset alias;
- pretraining only (no SFT/DPO/PPO/adapter path);
- no inline JSON training arrays;
- no cooperative cancellation in the compiled trainer ABI, so the editor
  disables Stop while a native run is active;
- progress is lifecycle/checkpoint based rather than per-step loss streaming.

These are fail-closed limits, not fallback triggers.

The dedicated Muse Glimmer trainer is currently a direct CLI/SDK native path,
not an editor-run-service promotion. Exact production Glimmer AR/SFT and
LoRA/QLoRA graphs can preflight to `nfn_muse_glimmer_native_train`, but the
trainer also requires an authenticated 627-tensor BF16 source, its SHA-256,
uint32 or structured-SFT dataset lineage, and (for SFT) the pinned ATEM hash.
The editor request schema does not yet carry that source/checkpoint contract,
so it must continue to reject a Glimmer start rather than launching with
invented defaults. Use the direct workflow documented in
[CLI Workflows](../cli.md#native-muse-glimmer-training).
