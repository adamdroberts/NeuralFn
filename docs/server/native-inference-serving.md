# Standalone Native Inference Server

`neuralfn/native_serve.py` is a separate deployment surface from
`server/app.py`. It exists so model serving does not initialize editor state,
cookie authentication, the relational database, Redis, MCP, dataset services,
or the persistence worker.

## Startup order

`run_native_inference_server()` completes these steps synchronously before
calling `uvicorn.run()`:

1. validate host/auth policy and load private API-key material;
2. read and validate Native Execution Manifest v1, including both
   `capabilities.serve=true` and `model.text_generation=true`;
3. load the artifact-declared tiktoken codec;
4. resolve the artifact chat template or explicit operator fallback;
5. require a positive authoritative context limit;
6. load `NativeInferenceModel`, proving resident ABI and cache capabilities;
7. when structured output is jointly advertised, require artifact-selected
   presentation metadata and preflight exact vocabulary bytes plus complete
   printable-ASCII single-byte coverage;
8. when configured, securely open and migrate the versioned `--state-db`;
9. when `prefix_cache_capacity > 0`, require that store plus effective
   lossless-full or dense-CPU-TurboQuant session COW and construct the bounded
   prefix manager (Tile-CUDA is rejected);
10. create the isolated FastAPI application and bounded compute queue; and
11. invoke Uvicorn with exactly one worker.

Any failure before step 11 occurs before the listening socket opens. The
resident model loads immutable weights once. Lifespan shutdown stops and awaits
the background driver, awaits all tracked foreground Responses SSE drivers,
drains the compute queue, shuts down the prefix cache, and only then closes the
resident model and state store. Queue closure is a shared drain operation:
concurrent `close()` and `aclose()` callers join the same completion, submitted
work is not abandoned, and cancellation of `aclose()` is re-raised only after
the drain finishes. Calling close from the queue's own worker is rejected
because waiting for that worker would deadlock. The nested cleanup continues
through later stages when an earlier stage fails.

The two serving gates are deliberately independent. A future resident
embedding or encoder adapter must not become a chat model merely because it has
an in-process forward; a stale manifest that overclaims `serve` while omitting
or denying `model.text_generation` is rejected before model loading and never
appears in `/v1/models`.

The current resident-ready model ABIs are the seven reviewed dense-v5 profiles
(`gpt2`, megakernel, MoA, z-loss, QK-norm, stable, and softcap),
canonical `llama`/`llama_fast`, and exact standard-MoE
`moe`/`mixllama`/`mixllama_fast` with their strict graph-bound checkpoints.
Standard MoE uses the same one-model/many-session server lifecycle and lossless
`auto`/`full` cache as its SDK path. It does not advertise TurboQuant. LLaMA and
MoE still require separately supported tokenizer and chat-template metadata
before server startup can pass presentation validation.

MoA startup additionally requires the migrated manifest's source-bound
`checkpoint.moa` contract, originally produced from
`model_XXXXXXXX.moa.json` beside the named dense-v5 model and empty DONE
marker. The binding revalidates the source digest, canonical candidate set,
selected activation, and positive interval before using that activation for
CPU prefill/decode and cache work. CPU remains the default. When its artifact
also proves the separate Tile feature and startup explicitly supplies
`--turboquant-attention-backend tile-cuda --tile-ops-lib PATH`, only packed
historical attention moves to CUDA; weights, projections, and row encoding stay
on CPU. The same option applies to the other reviewed-dense artifacts, never to
LLaMA or standard MoE.

## Concurrency

`BoundedSingleWorkerQueue` owns a `ThreadPoolExecutor(max_workers=1)`. One slot
represents the running generation; `queue_capacity` additional slots represent
waiters. Admission uses a non-blocking semaphore and returns OpenAI-shaped
HTTP `429` rather than allowing an unbounded request backlog. A second
`session_limit` semaphore caps the total running-plus-queued request-session
reservations. It defaults to `queue_capacity + 1`; operators may set a smaller
positive `--session-limit` to impose a tighter resource boundary. The two 429
codes are `queue_saturated` and `session_limit_exceeded`, and `/health` reports
both limits, current reservations, and per-cause rejection counters.

Each accepted Chat or background request creates a distinct
`NativeInferenceSession`. A foreground Response also remains logically
isolated, but an enabled resident prefix cache may supply it as an exact COW
fork whose lease is retained through durable completion. The worker prefills
only the remaining rendered prompt suffix and decodes through the resident SDK.
Streaming callbacks are forwarded to the event loop only after native/Python
token state commitment. A disconnected foreground stream sets the session
cancellation signal; its detached driver remains tracked so shutdown awaits
lease disposal. The binding remains responsible for finer-grained checks
within a layer. A streamed background Response is different: its detached
driver owns generation, so subscriber disconnect only closes that replay and
does not cancel the job. Background generation deliberately remains cold and
never acquires or publishes a serving-prefix entry.

No batching or multi-worker model sharing is claimed.

A native failure is captured as a worker outcome and re-raised on the event
loop. This avoids poisoning the executor's next await on supported Python 3.13
runtimes. The request context closes the failed session, queue and session-limit
accounting are released in `finally`, and the next request creates a fresh
session against the same once-loaded model.

Foreground Chat Completions and Responses share the same bounded worker.
Background Responses are first committed as durable SQLite jobs and then
admitted to that worker when a slot is available. They do not create an
unbounded in-memory generation queue. App startup wakes the background driver
so jobs queued by an earlier process are resumed.

## Stateful Responses service

`neuralfn/native_responses.py` is the dependency-light contract layer for the
text Responses and Conversations subset plus one bounded constrained profile.
It validates request capabilities,
normalizes text message items, resolves API-key-scoped previous-response or
conversation history, renders and tokenizes the prompt, and records only JSON
items plus prompt token IDs. It never imports FastAPI, the editor server, Torch,
NumPy, or NetworkX.

`neuralfn/_native_prefix_cache.py` is a private serving implementation module,
not a supported top-level SDK export. `NativeServingRuntime` creates its
`NativePrefixCache` only when `NativeServeConfig.prefix_cache_capacity` is
positive. Capacity zero is the compatibility default and omits the
`prefix_cache` object from `/health`. A positive capacity requires an open
`NativeStateStore` and an effective model cache of either lossless `full` with
`session_prefix_cow`, or reviewed dense CPU `turboquant` with
`session_prefix_cow_cpu_turboquant`. Cache-off, unproved feature inventory, and
Tile-CUDA TurboQuant fail during startup.

The manager is a deterministic entry-count-bounded LRU. Stored response IDs
and `(conversation ID, revision)` pairs are scope-local aliases, not trust
proof: every acquisition compares the newly prepared prompt tokens with the
sealed entry, forks only the exact non-empty LCP, and verifies the child's token
history plus native cached-row count. Entries are pinned while native fork work
runs, so eviction/purge may retire an entry but cannot close it until the pin is
released. Native fork, stats, and close calls occur outside the manager lock.
The request lease owns the child or fresh session until persistence finishes.
Only stored `completed`/`incomplete` foreground Responses are admitted after
that transaction; failed, cancelled, non-stored, and background outcomes close
the lease. A `store: false` request can reuse a stored parent but cannot publish
an alias. Chat and background requests never acquire from this manager. The
combined acquire/execute/durable-finish lifecycle is private to the HTTP app;
the supported public `NativeResponsesService.execute()` plus `finish()` phases
remain cold and create no prefix-cache hit or admission.

Usage is deliberately conservative. Cached input tokens are bounded by the
exact token LCP and the child's native cached-row statistic. Cache writes are
the newly observed prepared-prompt rows, capped at the prompt suffix; decoded
output rows are excluded. Shared/private/detach and retained byte fields in
`/health.prefix_cache` are sums of per-session capacity observations, not
unique physical allocations. COW sharers may double-represent one allocation;
the returned `byte_accounting_scope` says so explicitly. Restart is cold
because only JSON/token history is durable.

Durable terminal publication and destructive mutation share a service-level
transition lock. Response, conversation, or conversation-item deletion purges
the complete API-key cache scope rather than only a named alias, because a
descendant can retain deleted content. Each purge advances a scope epoch even
when no entry exists; an older in-flight lease rechecks that epoch under the
publication lock and cannot republish after deletion. That transition lock and
epoch are process-local. Do not point multiple cache-enabled server processes
at one state database, and do not mutate that database concurrently through a
raw `NativeStateStore`; route semantic mutations through the owning
`NativeResponsesService`. The standalone launcher intentionally uses one
Uvicorn worker.

`neuralfn/native_constrained.py` compiles a strict flat root-object schema to a
printable-ASCII byte-prefix grammar. Generation reads the session's immutable
current logits, masks every globally invalid token, selects greedily among the
allowed one-byte tokens, and commits the exact selected prefix with `prefill`;
it never calls ordinary `decode`. Every commit is checked against readable
session token history, and a completed JSON object is parsed and independently
validated. The function-tool path reuses this engine for one forced function's
arguments, emits a typed `function_call`, and later accepts a separate typed
`function_call_output`. The service never executes the client function.

Constrained schema/argument generation is deliberately excluded from the
stream/background drivers: it must be stored, buffered, foreground, greedy,
and use the artifact-selected chat template. The separate client-result
continuation is ordinary text generation and may sample, but remains stored,
buffered, foreground, and uses disabled truncation. Function lineage is supported only through
`previous_response_id`; Conversations, local compaction, automatic
truncation, count-only requests, parallel tools, and streamed/background tool
items remain fail-closed. Validation occurs before response persistence,
generation admission, or session creation.

The service also owns OpenAI cursor-page semantics for response input items and
conversation items. Routes validate `after`, `limit`, and `order` before the
service reverses or slices the immutable stored order and derives
`first_id`/`last_id`/`has_more`. Response deletion returns the deleted-resource
JSON object after the terminal-state guard and durable delete succeed.

Compaction uses the same normalization and scope boundary. It stores the exact
normalized message context in SQLite, indexes an unguessable public reference
by digest, and emits the current `response.compaction`/`compaction` item shape.
The reference is durable across restart but local to this state database and
API-key scope; it is not portable OpenAI ciphertext and no summarization model
is claimed. When a caller later submits that reference as Response input, its
normal item history retains the submitted token and therefore depends on the
same private-file controls as the rest of the state database.
Stateful JSON bodies are limited to 1 MiB, and compaction applies the declared
model context limit before committing state.

`neuralfn/native_state.py` owns a separately versioned SQLite database rather
than reusing the editor's SQLAlchemy schema. Creation uses no-follow semantics
where available, forces mode `0600`, rejects symlink paths, enables foreign
keys, WAL, `synchronous=FULL`, and a busy timeout, and serializes writes behind
a process-local reentrant lock. Composite `(scope, id)` keys partition data by
the SHA-256 fingerprint of the accepted Bearer key; the unauthenticated
loopback scope uses a fixed anonymous fingerprint. Raw API keys and resident
KV/cache buffers are never stored.

The store has no automatic TTL. Deployment owners must define backup retention,
volume encryption, and secure deletion for the state file; deleting it also
invalidates all local compaction references.

The database separates response records, ordered response input/output items,
conversations, ordered conversation items, scope-bound response compactions,
a durable response-event ledger, and background job state. Schema version 2
added that ledger to a version-1 database without rewriting existing rows.
Schema version 3 is a semantic fence for durable typed function-call and
function-result items. Schema version 4 adds a non-negative, monotonic
`items_revision` to each conversation. Version-1/2/3 stores migrate in place;
existing conversation history is revision zero, and older binaries reject the
newer metadata version rather than ignoring its concurrency fence. Back up the
database before the first v4 open when rollback matters; restore that backup
before starting an older binary. A conversation-linked background job queued by
an older binary has no historical revision snapshot to migrate; the background
driver terminalizes it as failed with `conversation_snapshot_unavailable` rather
than generating against current items. A legacy previous-response-only job can
reconstruct only a currently completed/incomplete lineage and otherwise fails
with `response_lineage_unavailable`. The response-event ledger's primary key is
`(scope, response_id, sequence_number)`, its response foreign key cascades on
deletion, and a partial unique index permits only one semantic terminal per
response. A background request, its queued job, and
`response.created` for an originally streamed background request are inserted
before the HTTP response is returned. Queued jobs remain queued across restart.
On open, only jobs left `in_progress` are atomically marked `failed` with
`server_restarted`, avoiding
duplicate model execution. Queued cancellation is atomic; in-progress
cancellation is polled into the resident session's cancellation signal.

Conversation preparation uses `conversation_items_snapshot()` so ordered items
and revision come from one transaction. `finish_foreground_response()` commits
terminal response state, output rows, optional conversation rows, and the
expected-revision CAS atomically, and returns the post-commit revision used for
the new cache alias. `finish_background_job()` accepts the same optional
conversation ID/items/expected-revision inputs and performs the CAS inside its
existing response/job/event terminal transaction, but does not admit a resident
prefix. Blind item create/delete helpers increment the same revision. A stale
expected value raises `NativeStateConflictError` with
`code="conversation_conflict"`; no output or conversation append from that
attempt commits. The Responses layer then records a failed response. Buffered
HTTP returns 409, while a stream/background ledger ends in
`response.failed`. Independently, terminalization re-reads the complete stored
`previous_response_id` lineage under the same transition. Deleted or changed
ancestry raises `response_lineage_conflict`, fails the response, and prevents
both output publication and cache admission; reconstructed legacy lineage is
revalidated by the same path.

Responses SSE is generated from committed native token callbacks. Stable
response/item IDs are allocated before decoding. `BEGIN IMMEDIATE` append
transactions allocate contiguous sequence numbers across concurrent store
connections. Delta obfuscation is generated once and stored with each event so
default replay is stable; a retrieval may omit the field without mutating the
ledger. Final response/job state, the output item, done events, and the single
semantic terminal are committed in one transaction.

`POST /v1/responses` with `background: true`, `stream: true`, and `store: true`
tails this ledger rather than owning the generation session. A disconnect does
not cancel it. `GET /v1/responses/{id}?stream=true&starting_after=N` performs
authorization, scope, and original-mode validation before returning an SSE
response, then replays events strictly after `N` and tails until the terminal.
Only a background response originally created with streaming can use this
path. `include_obfuscation` defaults to true; false strips only the outgoing
field. Both the initial and resumed stream end with the semantic response
terminal instead of the Chat Completions `[DONE]` sentinel.

## Presentation boundary

Serving accepts only artifact-declared tiktoken encodings. `--chat-template
auto` accepts manifest `plain_roles` metadata or a literal `{messages}`
placeholder template. Operators can explicitly select `plain_roles` or a local
template file containing that placeholder. Arbitrary Jinja/Hugging Face chat
templates are rejected by this lean milestone rather than interpreted
partially.

Canonical LLaMA now proves the resident model ABI and raw-token SDK/CLI path,
but its native-family checkpoint migration does not make this presentation
boundary disappear. The migrated graph currently classifies tokenization as
SentencePiece and lacks the named tiktoken encoding/chat metadata this server
accepts, so startup rejects it before bind unless the artifact independently
supplies supported presentation metadata. Do not treat the registry's lean
serving ABI bit as end-to-end LLaMA Chat Completions coverage.

The same renderer and tokenizer determine prompt token usage and context-limit
validation. Generated token bytes are decoded incrementally for SSE, avoiding
replacement-character corruption when a UTF-8 code point spans tokens.
Structured/function capability is disabled when an operator selects
`plain_roles` or a template path explicitly, even if the resulting text looks
the same: only `--chat-template auto` proves that the renderer is the exact
artifact-selected presentation contract. Startup also scans exact token bytes
for the model vocabulary and refuses the constrained capability if any token
cannot be resolved or any required printable ASCII byte lacks a standalone
token.

## Dependencies and exclusions

The `[serve]` extra contains FastAPI, Pydantic, tiktoken, and Uvicorn. Importing
the serving module does not import Torch, NumPy, NetworkX, SQLAlchemy, or
`server.app`.

The SQLite service is opt-in through `--state-db`; without it, Responses and
Conversations remain unmounted and return `unsupported_resource`. With state
enabled, only the exact buffered flat-schema and single forced client-function
profiles described above are additive. General/parallel/hosted tools,
nested/array schemas, constrained streaming/background execution, Chat
Completions tools, reasoning modes, multimedia, Responses WebSocket mode,
Realtime, and legacy Completions are not implemented. They return explicit
OpenAI-shaped capability errors rather than being partially interpreted.

The serving prefix LRU is a second opt-in through
`--prefix-cache-capacity`; its default `0` preserves cold-per-request behavior.
It accelerates eligible foreground Responses lineage only and does not change
the REST resource shapes, durable source of truth, Chat Completions, background
jobs, or the exclusions above.

See [the REST contract](../rest-api/native-inference-serving.md) for request,
streaming, error, and authentication behavior.
