# Editor Native Training Service

Editor and MCP native runs use the authenticated project/session run service;
they are separate from the standalone native inference server.

## REST workflow

All paths are scoped as
`/api/projects/{project_id}/sessions/{session_id}/runs`.

| Endpoint | Behavior |
|---|---|
| `POST /preflight` | Compiles the saved graph through `plan_native_graph_training()` without persistent artifacts. Returns node-specific compatibility and adapter metadata. |
| `POST /start` | Starts a background run and returns the initial run snapshot as JSON. This is used by MCP. |
| `POST /` | Starts the same run and streams events over SSE. This is used by the editor. |
| `GET /active` | Returns live or persisted compatibility, artifact, and checkpoint metadata. |

`TrainRequest.runtime` accepts `scalar`, `torch`, or `native-cuda`. A native
request may override the saved runtime for that run. Unsupported runtimes are
rejected by Pydantic before dispatch.

## Fail-closed preparation

`server/services/native_training.py` is a thin integration layer over the
public SDK boundaries:

1. serialize the current in-memory graph as inert JSON;
2. dry-run `plan_native_graph_training()` and return all stable node paths;
3. require `execution_ready`, a registered trainer, and proved architecture
   persistence;
4. materialize a private per-run Native IR bundle and exact source snapshot;
5. compare the materialized graph fingerprint and selector with preflight;
6. build `NativeTrainRunConfig` from the planner-owned family, selector,
   geometry arguments, dataset alias, and run output directory;
7. call `run_native_train(..., runner="auto")`;
8. require a non-empty, contained native checkpoint before completing the run.

The execution-ready set is the seven dense GPT-2 profiles `gpt2`,
`gpt2_megakernel`, `gpt2_moa`, `gpt2_qknorm`, `gpt2_softcap`, `gpt2_stable`,
and `gpt2_zloss`, exact canonical
`llama` and its graph-equivalent compile-runtime alias `llama_fast`, plus exact
standard-MoE `moe`, `mixllama`, and `mixllama_fast`.
`NativeTrainRunConfig` independently re-runs LLaMA planning and
requires every plan-owned argument exactly once. The family binary verifies
the graph SHA-256 before dataset/CUDA setup and at checkpoint write, accepts
`--weight-decay`, and rejects unknown graph-mode arguments. Checkpoint
discovery accepts LLaMA only through validated inference-checkpoint v2 metadata
whose source digest exactly matches the preparation plan.
Both LLaMA selectors launch the canonical `llama` trainer/checkpoint ABI; run
metadata preserves the source selector, preset, runtime, and SHA-256.

MoA preparation passes the exact source digest, canonical
GELU/ReLU/SiLU/ReLU2 candidate order, and positive probe interval. Checkpoint
discovery accepts it only through `model_XXXXXXXX.moa.json` after validating
the named dense-v5 model, empty `DONE_XXXXXXXX`, final selected activation,
model/graph hashes, and geometry. It stores the metadata path, not the sibling
`.bin`, as the authoritative completed checkpoint. Resume validates the same
sidecar and restores its recorded activation without rerunning candidate
probes; missing or tampered metadata fails before training resumes. This is the
graph-bound server route. Direct selector-only first-leg MoA training remains a
separate dense-v5 workflow, and its unbound output cannot be resumed exactly.

`gpt2_diff` is execution-ready for graph training only after the server's
trusted planner materializes the exact source-bound
`native-training-proof.json`. The proof binds reviewed configuration/topology,
shape, source-SHA, and geometry identities; the trainer requires graph,
fingerprint, and proof together before plan/Tile/CUDA work. Its unkeyed digest
is local handoff integrity, not caller authenticity. Its low-level packed
trainer learns and persists one lambda plus optimizer
moments per layer in a graph-bound additive bundle. Its version-2 strict
continuation metadata binds the source, five binaries, resolver-ordered training
shards, optimizer/microbatch and sampler counters, seed, accumulation shape,
optimizer/LR horizon, BF16 routes, and a canonical profile of supported effective numerics before
Tile/CUDA/H2D; validation shards are excluded. Completion discovery returns the
strict `.diff.json` checkpoint only after validating its empty DONE marker,
five contained artifacts, exact sizes/digests/headers/finite state, source and
proof identities, and geometry. Each binary is hashed and inspected from its
own exact-size no-follow descriptor in bounded chunks; the service does not
retain or slice the full multi-gigabyte bundle in Python memory. Migration and
resident inference still do not
validate or execute that state. Graph training is 13 ready/54 blocked;
persistence and resident inference remain 12 ready/55 blocked.

Standard-MoE preparation independently re-plans the complete graph-owned
geometry, including floating multiplier, `multiple_of=None` as native `0`,
experts, top-k, router auxiliary-loss coefficient, runtime alias, and digest.
Normal execution selects `--train-moe-dataset-loop`. Checkpoint discovery
accepts only the strict standard-MoE metadata after its DONE marker, contained
float32 sidecar, ordered tensor table, whole-file/tensor hashes, semantics, and
source graph SHA match the prepared plan. Neighboring modern, megakernel,
DeepSeek, shared-expert, aux-free, and JEPA graphs remain closed.

There is no Torch or scalar fallback. The existing dense trainer currently
consumes the immutable validated graph snapshot plus canonical selector rather
than parsing Native IR itself, so metadata remains honest with
`trainer_consumes_native_ir: false`.

Muse Glimmer has a separately reviewed `nfn_muse_glimmer_native_train` direct
trainer and exact graph planner for AR, structured SFT, LoRA, and QLoRA. It is
not yet admitted by this editor/server run service: the REST request and
dataset-alias workflow do not carry the required authenticated BF16 source
checkpoint/SHA, ATEM lineage, or structured-SFT record contract. Server
preflight/start must fail closed until those fields and artifact ownership
rules are added; callers can use the direct CLI/SDK native training route
meanwhile.

## Persistence

`TrainingRun` stores `runtime`, `compatibility_report`, `artifact_metadata`,
and `checkpoint_path`. Migration `20260804_0003` adds those fields. The live
snapshot exposes the same keys while a run is active; `PersistenceWorker`
copies the final checkpoint path and artifact metadata to SQL when the native
thread finishes.

Artifacts are rooted at:

```text
NEURALFN_ARTIFACTS_DIR/
  runs/<run-id>/
    editor-graph.json
    native-ir/
      native-execution-manifest.json
      compatibility-report.json
      source-graph.json
      native-training-plan.json
    checkpoints/
      model*.bin
      model_00000000.moa.json
      DONE_00000000
      llama_native_family_model_00000000.json
      llama_native_family_parameters_00000000.f32
      llama_native_family_optimizer_00000000.bin
      llama_native_family_model_DONE
      mixllama_native_family_model_00000000.json
      mixllama_native_family_parameters_00000000.f32
      mixllama_native_family_optimizer_00000000.bin
      mixllama_native_family_model_DONE
```

Run IDs are constrained to one safe path component. Checkpoint discovery
rejects symlinks, empty files, paths outside the run checkpoint directory, and
unrecognized filenames. Ordinary dense runs store the `.bin` path; MoA stores
its validated `.moa.json` metadata path. Canonical LLaMA runs
store the metadata `.json` path only after the inspector verifies the contained
sidecar, tensor table, digest, DONE marker, and source-graph fingerprint.
Standard-MoE runs store their metadata JSON only after the corresponding strict
inspector proves the same bundle properties and exact graph identity.

The current compiled trainer ABI does not expose cooperative cancellation.
`stop_run` therefore returns `status: unsupported` for native runs rather than
pretending a still-running native process was stopped.
