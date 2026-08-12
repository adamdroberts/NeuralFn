# Native Inference Serving API

NeuralFn's native inference server is a standalone, inference-only FastAPI
application. It is separate from the editor backend in `server/app.py`: it does
not initialize editor cookies, SQLAlchemy, Redis, the persistence worker, MCP,
Torch, NumPy, or NetworkX.

Install the lean serving dependencies and start one resident artifact:

```bash
pip install -e '.[serve]'
nfn infer \
  --checkpoint artifacts/model-native \
  --serve \
  --chat-template plain_roles \
  --state-db ./native-inference-state.sqlite3 \
  --prefix-cache-capacity 64
```

The default address is `http://127.0.0.1:8000`. The artifact is loaded exactly
once before Uvicorn is invoked, so a missing tokenizer/chat template, invalid
context limit, unproved resident ABI, unsupported cache request, missing
checkpoint, or unavailable in-process binding fails before the listening
socket opens.

Model-list admission requires both `capabilities.serve=true` and
`model.text_generation=true` in Native Execution Manifest v1. Embedding,
encoder-only, and malformed stale artifacts are rejected before resident model
loading and therefore never appear in `/v1/models`.

## Implemented resources

The response envelopes follow the official OpenAI
[Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)
[Models](https://platform.openai.com/docs/api-reference/models),
[Responses](https://developers.openai.com/api/reference/resources/responses/methods/create),
and [Conversations](https://developers.openai.com/api/reference/resources/conversations/methods/create)
resource shapes for the bounded fields implemented here.

| Method | Path | Behavior |
|---|---|---|
| `GET` | `/health` | Loaded model, backend, effective capabilities, and queue state |
| `GET` | `/v1/models` | OpenAI-shaped list containing the one served model |
| `GET` | `/v1/models/{model}` | Retrieve the served model or return `model_not_found` |
| `POST` | `/v1/chat/completions` | One text completion, buffered or streamed |
| `POST` | `/v1/responses` | Create a text response, or a jointly gated buffered constrained/function response |
| `GET` | `/v1/responses/{id}` | Retrieve JSON, or replay an originally streamed background response with `stream=true` and optional `starting_after` |
| `DELETE` | `/v1/responses/{id}` | Delete a terminal stored response and return `{id, object: "response", deleted: true}` |
| `GET` | `/v1/responses/{id}/input_items` | Cursor-list the direct input items stored for a response |
| `POST` | `/v1/responses/input_tokens` | Count rendered input tokens without generating |
| `POST` | `/v1/responses/compact` | Create a durable, lossless local context compaction |
| `POST` | `/v1/responses/{id}/cancel` | Cancel a queued or in-progress background response |
| `POST` | `/v1/conversations` | Create a conversation with optional initial items |
| `GET` | `/v1/conversations/{id}` | Retrieve a conversation |
| `POST` | `/v1/conversations/{id}` | Replace conversation metadata |
| `DELETE` | `/v1/conversations/{id}` | Delete a conversation and its items |
| `POST` | `/v1/conversations/{id}/items` | Add up to 20 text message items |
| `GET` | `/v1/conversations/{id}/items` | Cursor-list conversation items |
| `GET` | `/v1/conversations/{id}/items/{item}` | Retrieve one conversation item |
| `DELETE` | `/v1/conversations/{id}/items/{item}` | Delete one item and return the conversation |

Responses and Conversations routes are mounted only when `--state-db PATH` is
supplied. Without it they return `code: "unsupported_resource"`; Chat
Completions remains available without persistent state. The editor's `/api/**`
routes are never mounted. Legacy Completions, Responses WebSocket mode,
Realtime, embeddings, files, vector stores, hosted tools, and beta multi-agent
resources remain outside this server.

### Official Python SDK evidence

`tests/test_native_openai_sdk.py` is pinned to the official
`openai==2.44.0` client. It verifies typed Models, Chat Completions and chunks,
Responses and semantic streaming events, stored IDs and input resources,
lineage and local compaction, Conversations/items, background cancellation,
SQLite close/reopen persistence for response/conversation/queued-background
IDs, resumable background-stream retrieval with `starting_after` and
`include_obfuscation`, typed Pydantic structured parsing, one forced strict
function call with parsed arguments, the separate client-result continuation,
and the SDK's `400`/`401`/`404`/`409`/`429`/`500` exception classes. The test
module skips rather than silently claiming
coverage when that exact optional SDK version is unavailable. The audited
cached-client run passes 18 SDK tests; the combined SDK/ASGI serving command
passes 73 tests, while the constrained-engine plus ASGI slice passes 95.
`tests/test_native_openai_sdk_tcp.py` additionally runs that pinned client
through real loopback sockets. Its twelve default cases prove
Models, buffered Chat, Chat SSE framing and final usage, Pydantic structured
Responses parsing, ordered semantic Responses SSE, a forced Pydantic function
plus separate client-owned result continuation, invalid-Bearer
`AuthenticationError` mapping, typed `400`/`404` errors, synchronous and
`AsyncOpenAI` streaming, stored background-stream cursor resumption, and
foreground stream-close cooperative cancellation/session disposal plus
background stream-close continuation and graceful client/server/runtime/thread
shutdown. The optional resident case loads a freshly built resident binding and
strict Tile sidecar, serves a tiny synthetic dense-v5 artifact on an RTX 5090,
and requires a positive Tile-CUDA attention launch count. Three transport cases
also prove that the official client rejects an untrusted per-test TLS
certificate and succeeds with its explicit CA, that a real loopback HTTP proxy
receives `CONNECT` and tunnels to the HTTPS Uvicorn origin, and that a TLS edge
negotiates ALPN `h2` with HTTP/1 disabled before Models and Responses SSE cross
real HTTP/2 frames. That edge buffers the completed upstream Uvicorn HTTP/1.1
SSE body before emitting HTTP/2 DATA, so it does not prove incremental reverse-
proxy forwarding or direct Uvicorn HTTP/2 support. The current cached-client
run is `12 passed, 1 skipped` for the socket module and `30 passed, 1 skipped`
across both SDK modules; the optional resident case was separately proved live.

This evidence is intentionally narrower than full current-official
compatibility. Only the constrained schema and forced-function profiles below
are claimed; broader tools/schemas, multimodal input, non-stored background
work, direct/incremental production HTTP/2 proxy behavior, representative trained and all shipped
model/tokenizer artifacts, and portable OpenAI compaction still fail closed or remain
local as documented below. Lower-level ASGI tests separately cover incremental delivery and both
foreground disconnect cancellation and background disconnect continuation;
the synchronous SDK test transport buffers its in-process response before
parsing.

## Chat Completions

The bounded first milestone accepts these request fields:

- `model` (must match the configured served name)
- non-empty `messages` with `developer`, `system`, `user`, or `assistant` roles
- string content or arrays containing text parts only
- `temperature`, `top_p`, and `seed`
- `max_completion_tokens` or the legacy `max_tokens` alias
- `n`, which must equal `1`
- `stream` and `stream_options.include_usage`

All other fields fail with `code: "unsupported_feature"`; they are never
ignored. In particular, tools/tool calls, structured output, logprobs,
penalties, stop strings, image/audio/file content, storage, and reasoning modes
are unavailable. The server validates prompt plus reserved output tokens
against the artifact's declared context limit before queue admission.

Example:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "my-model",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_completion_tokens": 32,
    "temperature": 0
  }'
```

A non-streamed response uses the OpenAI `chat.completion` shape and includes
prompt/completion/total token usage. With `stream: true`, the server emits
actual committed-token `chat.completion.chunk` SSE records: an initial
assistant-role delta, text deltas, a finish-reason chunk, an optional empty
choices usage chunk, and finally:

```text
data: [DONE]
```

## Responses

The bounded Responses implementation accepts `model`, text `input`, string
`instructions`, `max_output_tokens`, `temperature`, `top_p`, `metadata`,
`store`, `stream`, `background`, `previous_response_id`, `conversation`, and
`truncation` (`disabled` or oldest-message `auto`). Jointly proven artifacts
also accept the exact `text.format`, `tools`, `tool_choice`, and
`parallel_tool_calls` profiles described below. Input may be a string or an
array of `message` items with developer, system, user, or assistant roles and
`input_text`/plain text content. It may also contain the scope-bound
`compaction` item returned by this server. `previous_response_id` and
`conversation` are mutually exclusive.

Stored lineage is scoped by the SHA-256 fingerprint of the authenticated Bearer
key. A response ID created with one key is a `response_not_found` error for
another key. Previous-response context replays durable input/output items; a
new request's `instructions` replace, rather than inherit, earlier response
instructions. Conversation context prepends its stored items and appends the
new response input/output after successful or incomplete generation.

The response input-item and conversation item list routes accept the OpenAI
SDK cursor parameters `after`, `limit` (1 through 100, default 20), and `order`
(`asc` or the default `desc`). Their list envelopes set `first_id`, `last_id`,
and `has_more` from the returned page. An unknown cursor is an
`invalid_cursor` error instead of silently restarting from the first page.
Deleting a terminal response returns the current deleted-response resource
shape with HTTP 200; it is not an empty HTTP 204 response.

`store` defaults to `true`. `store: false` permits an immediate buffered or
streamed result but leaves no retrievable ID. Background and conversation
requests require `store: true`. Input-token counting uses the same renderer,
tokenizer, lineage lookup, and truncation behavior as generation and returns:

```json
{"object":"response.input_tokens","input_tokens":42}
```

### Process-local resident prefix reuse

Serving prefix reuse is disabled by default. Set
`--prefix-cache-capacity N`, where `N > 0`, to retain at most `N` sealed native
sessions in a deterministic least-recently-used cache. The option requires
`--state-db`, because durable response and conversation history remains the
source of truth. Startup also requires one of these exact effective native
fork contracts:

- lossless `full` cache plus `session_prefix_cow`; or
- reviewed dense CPU `turboquant` plus
  `session_prefix_cow_cpu_turboquant`.

Cache-off and Tile-CUDA TurboQuant configurations are rejected before the
server binds. This does not add Tile device-state COW. Chat Completions and all
background Responses continue to create fresh sessions; background results do
not hit or enter the resident prefix cache.

A foreground request with `previous_response_id` looks up the stored parent
response alias. A conversation request looks up the exact `(conversation_id,
items_revision)` alias captured with its durable item snapshot. Either alias is
only a candidate: NeuralFn renders and tokenizes the complete current request,
computes an exact token longest common prefix against the sealed session, and
forks only that verified prefix. A zero-token LCP is a miss. The LRU pins an
entry while its native fork is in flight, performs deterministic unpinned
eviction, and never closes a session while holding the cache lock. Capacity is
an entry count, not a byte limit.

Only a stored foreground response that reaches `completed` or `incomplete` is
eligible for admission, and publication occurs after its durable terminal and
output-item transaction commits. Failed and cancelled responses close their
lease. `store: false` lineage may hit an existing parent but never publishes a
new entry. A process restart is always cold: SQLite preserves JSON and prompt
tokens, not native sessions or K/V allocations.

Responses usage is observation-based. `input_tokens_details.cached_tokens` is
the smaller of the exact forked LCP and the child session's native cached-row
count. `cache_write_tokens` counts only prepared-prompt rows newly written by
that session—the prompt suffix bounded by the observed native row delta—and
never includes decoded output rows. `/health` includes `prefix_cache` only
while this feature is enabled. It reports capacity/entries, response and
conversation aliases, active leases/forks, hits/misses/evictions/purges,
commits/rejections, cumulative token observations, and byte-capacity
observations. The accompanying `byte_accounting_scope` is authoritative: byte
fields sum per-session capacity observations, so COW-sharing sessions can
represent one allocation more than once. They are not unique physical process
bytes.

Cache aliases and entries are API-key scoped. Deleting any response,
conversation, or conversation item conservatively purges the entire matching
API-key scope, even when the deleted resource did not name the retained entry;
descendants can otherwise contain deleted history. Purge advances a scope
epoch so a lease acquired before deletion cannot republish stale state after
the deletion commits. This lock/epoch fence is process-local: do not share one
cache-enabled state database between server processes or mutate it concurrently
through a raw `NativeStateStore`. Use the owning `NativeResponsesService`; the
standalone launcher runs exactly one Uvicorn worker.

### Bounded strict JSON-schema output

This is a token-level constraint, not post-hoc JSON parsing. It is advertised
only when all of these agree:

- the Native Execution artifact declares
  `capabilities.structured_output=true` and
  `kernel_abi.structured_output` integer version 1, `status: "ready"`, profile
  `json-schema-ascii-byte-greedy-v1` and token selection
  `current_logits_exact_prefill`;
- the compiled binding exposes callable read-only `current_logits` and reports
  `current_logits_exact_prefill=true`;
- serving uses the exact artifact chat template (`--chat-template auto`), not
  an operator-selected fallback; and
- startup can resolve exact bytes for the model vocabulary and finds a
  standalone token for every printable ASCII byte used by the grammar.

The accepted `text.format` contains exactly `type`, `name`, `schema`, and
`strict`; `type` is `json_schema`, `strict` is `true`, and `name` matches
`[A-Za-z0-9_-]{1,64}`. The schema is one strict root object with 1 through 32
properties, every property appears exactly once in `required`, and
`additionalProperties` is `false`. A property is `string`, `integer`,
`number`, or `boolean`, optionally narrowed by a homogeneous finite enum of at
most 64 values; an enum must contain 1 through 64 unique values. Property names
are 1 through 64 printable-ASCII bytes, and string values plus emitted JSON are
printable ASCII. Generated properties always follow the insertion order of
`schema.properties` (the `required` array order does not reorder them). String
`title`/`description` annotations are preserved but do not alter the grammar.
The canonical schema is at most 32 KiB. The engine hard cap is 4096 output
byte-tokens, further bounded by the server's `--max-output-tokens` ceiling
(default 256) and the request's `max_output_tokens`.

Nested objects, arrays, null/unions, refs, defaults, patterns, numeric/string
bounds, semantic formats, and unknown keywords are rejected before queue
admission. A constrained request must set (or inherit the defaults for)
`store: true`, `stream: false`, `background: false`, `temperature: 0`,
`top_p: 1`, `parallel_tool_calls: false`, and no tools. The server renders a
canonical developer constraint, prefills once, repeatedly masks logits to the
grammar's allowed single-byte tokens, greedily selects the greatest allowed
logit with lowest-token-ID tie breaking, and commits that exact token through
prefix `prefill`. Ordinary `decode` is never called. Reaching the byte-token
limit before the object closes returns `status: incomplete` and
`incomplete_details.reason: max_output_tokens`; a completed object is parsed
and independently revalidated before it is returned.

With the official SDK, a complete typed Pydantic request is:

```python
from pydantic import BaseModel, ConfigDict

class WeatherAnswer(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    city: str
    temperature_c: int

response = client.responses.parse(
    model="my-model",
    input="Return the weather as strict JSON.",
    text_format=WeatherAnswer,
    max_output_tokens=256,
    temperature=0,
    top_p=1,
    store=True,
    stream=False,
    background=False,
)
answer: WeatherAnswer = response.output_parsed
```

Successful output is an ordinary assistant `message`/`output_text` item and
the SDK populates `response.output_parsed`.

### One forced client-executed function

Function mode requires the structured gate above plus
`capabilities.function_tools=true`, `kernel_abi.function_tools` integer version
1 with `status: "ready"`, profile `responses-forced-function-call-v1`, and
`structured_output_profile: "json-schema-ascii-byte-greedy-v1"`, plus the exact matching
`chat_template.tool_template` artifact record. The request supplies exactly
one flat function tool with `strict: true` and parameters in the same bounded
schema subset, selects it with
`tool_choice: {"type":"function","name":"the_name"}`, and sets
`parallel_tool_calls: false`. Automatic/required choice strings, multiple or
parallel functions, custom tools, and hosted web/file/code tools are rejected.

NeuralFn constrains only the JSON argument object. It returns one completed
`function_call` item with stable `fc_` and `call_` identifiers, the selected
name, and a JSON-string `arguments` field. It never imports or executes client
code. The client executes the function and submits a separate stored,
foreground request whose `previous_response_id` names that response and whose
sole input item is:

```json
{
  "type": "function_call_output",
  "call_id": "call_...",
  "output": "the client-owned string result"
}
```

The service reconstructs typed lineage, requires exactly one visible completed
unresolved call, rejects unknown/mismatched/duplicate/already-resolved IDs, and
then produces an ordinary text response. Tool calls/results cannot be used
with Conversations, local compaction, automatic truncation, streaming, or
background jobs. Function tools on `/v1/chat/completions` remain unsupported.

Complete client-owned flow:

```python
tool = {
    "type": "function",
    "name": "lookup_weather",
    "description": "Return the current weather.",
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": False,
    },
}

first = client.responses.create(
    model="my-model",
    input="Weather in London?",
    tools=[tool],
    tool_choice={"type": "function", "name": "lookup_weather"},
    parallel_tool_calls=False,
    temperature=0,
    top_p=1,
    store=True,
    stream=False,
    background=False,
    truncation="disabled",
)
call = first.output[0]

# The application, not NeuralFn, executes lookup_weather here.
client_result = '{"temperature_c":12}'

final = client.responses.create(
    model="my-model",
    previous_response_id=first.id,
    input=[{
        "type": "function_call_output",
        "call_id": call.call_id,
        "output": client_result,
    }],
    store=True,
    stream=False,
    background=False,
    truncation="disabled",
)
```

The continuation input must contain only that one `function_call_output`. Do
not repeat a non-empty tool declaration/selection or constrained format; omit
those fields, or use the explicit empty/plain forms `tools: []`,
`tool_choice: "none"`, and `text.format: {"type":"text"}`. It is ordinary
text generation, so ordinary `temperature`/`top_p` controls are accepted, but
it remains stored, buffered, foreground, and uses disabled truncation.

Responses streaming is distinct from Chat Completions streaming. Events carry
a stable response/item ID and monotonically increasing `sequence_number`.
The stream includes lifecycle, output-item/content-part, and output-text delta
events, then terminates with `response.completed`, `response.incomplete`, or
`response.failed`. It never emits `data: [DONE]`.

### Resuming a background stream

Durable replay is deliberately narrower than general Responses streaming. The
original request must set all three of `background: true`, `stream: true`, and
`store: true`. NeuralFn rejects `background: true` with `store: false`; unlike
the [hosted OpenAI background service](https://developers.openai.com/api/docs/guides/background#streaming-a-background-response),
this local server has no temporary non-stored background retention tier. A
foreground stream, or a background response that was originally created
without streaming, cannot later be opened as a stream.

The create request streams the durable event ledger immediately. If that HTTP
connection drops, only the subscriber is detached: the queued or running job
continues under the background driver. Reconnect with the response ID and the
last committed cursor:

```bash
curl --get http://127.0.0.1:8000/v1/responses/resp_123 \
  --data-urlencode stream=true \
  --data-urlencode starting_after=42
```

`starting_after` is an exclusive non-negative 64-bit sequence cursor and is
valid only when `stream=true`. Omitting it replays from `response.created`;
passing the terminal event's sequence returns a successful empty SSE suffix.
The request is authorized and the response's API-key scope plus original
background/stream flags are checked before SSE headers are returned. Only a
response created in the same API-key scope can be resumed.

Every output-text delta stores a random `obfuscation` value with the event.
Retrieve streams include that persisted field by default. Set
`include_obfuscation=false` to remove it from the outgoing replay without
modifying the stored event. The create body does not yet accept
`include_obfuscation`, so its initial stream always includes the padding field.
This is the compatible field/control surface, not a claim that NeuralFn
reproduces OpenAI's exact payload-size normalization algorithm.
`include` and `include[]` are accepted on retrieval for the official SDK query
shape, but they do not enable tools, hosted resources, reasoning, or non-empty
logprobs in this text-only server.

The ledger allocates contiguous sequence numbers transactionally and permits
one terminal event. Success commits the response snapshot, output item,
output/content done events, and `response.completed` or
`response.incomplete` together. Failure and cancellation similarly commit one
`response.failed` or `response.incomplete` terminal. A restart marks a job
that had reached `in_progress` as failed with `server_restarted` and appends
that terminal; a still-queued job remains eligible to run.

With `openai==2.44.0`, the corresponding typed retrieval is:

```python
events = client.responses.retrieve(
    response_id,
    stream=True,
    starting_after=cursor,
    include_obfuscation=False,
)
for event in events:
    cursor = event.sequence_number
```

Outside the two profiles above, tools/tool-result items, hosted tools,
structured JSON/JSON-schema output, reasoning modes, and image/audio/file
content return `unsupported_feature` before queue admission or model
execution. Empty `tools`, `tool_choice: "none"`, and plain
`text.format.type: "text"` remain accepted as explicit no-tool, plain-text
configuration.

### Local response compaction

`POST /v1/responses/compact` follows the current `response.compaction` envelope
for `model`, text `input`, string `instructions`, and
`previous_response_id`. Its `output` contains retained user messages followed
by one `compaction` item. Passing those output items, optionally followed by
new text messages, as a later Responses `input` restores the exact normalized
context without duplicating the retained user messages.

This bounded native implementation is lossless and local: it does not run a
summarization model. The value carried in the OpenAI `encrypted_content` field
is an unguessable reference to context in the configured state database. The
compaction registry indexes its digest; if the token is later submitted as a
Response input, that ordinary input-item JSON retains the token. It survives
restart and is isolated by API-key fingerprint, but it is not portable OpenAI
ciphertext and cannot be used with another state database or API-key scope.
Prompt-cache controls and `service_tier` fail with `unsupported_feature`
rather than being ignored.

Stateful JSON request bodies are capped at 1 MiB, and local compaction refuses
context that already exceeds the artifact's declared model window. This bounds
parsing and persistence; because the local representation restores the full
text context, it is not a mechanism for reducing prompt-token pressure.

## Conversations and durable background work

The state database is versioned SQLite in WAL mode. Schema version 2 added the
API-key-scoped `response_events` ledger and its one-terminal constraint.
Schema version 3 made typed `function_call` and `function_call_output` items a
durable lineage contract. Schema version 4 adds the monotonic
`conversations.items_revision` used to take one transactionally consistent
item snapshot and later compare-and-swap its completion. Opening a version-1,
version-2, or version-3 database adds any missing ledger/revision structure and
advances the metadata in place; existing conversation history becomes revision
zero. After that migration, an older NeuralFn binary refuses the version-4
database. Take a backup before first opening it if rollback to an older binary
must remain possible, and restore that backup before rollback. NeuralFn creates the file
with mode `0600`, refuses symlink database paths, and partitions every response,
conversation, item, compaction, event, and job by API-key fingerprint. Only JSON
records and prompt token IDs are durable; resident KV/cache memory is never
serialized.

Migration cannot manufacture the revision observed by a conversation-linked
background response that an older binary already queued. When claimed after
upgrade, that legacy job is terminalized as failed with
`error.code: "conversation_snapshot_unavailable"` rather than being generated
against current conversation items. A legacy queued job with only
`previous_response_id` may reconstruct a snapshot only from a currently
completed/incomplete lineage; missing or nonterminal ancestry fails with
`response_lineage_unavailable`.

Conversation-linked generation captures items and revision in one snapshot.
A successful or incomplete foreground completion commits the response terminal
state, output rows, conversation input/output rows, and revision increment in
one transaction before cache publication. Background completion performs the
same revision CAS inside its response/job/event transaction but deliberately
stays resident-cache cold. If an item append/delete or another completion wins
first, the stale branch commits none of those output/conversation rows and is
terminalized as failed with `error.code: "conversation_conflict"`. A buffered
request returns HTTP 409 with `param: "conversation"`; a foreground stream or
background response that has already begun emits/stores `response.failed`.
Every `previous_response_id` branch independently re-reads its complete stored
lineage under the terminalization transition. Deleting an ancestor, or changing
its status/ancestry, similarly fails the stale branch with
`response_lineage_conflict` and `param: "previous_response_id"`; no output or
cache entry is published. Legacy reconstructed lineage is subject to the same
finish-time revalidation.

There is no automatic TTL or compaction-record deletion endpoint in this
bounded service. Operators own backup retention, secure state-file disposal,
and storage-volume encryption; deleting the state database invalidates every
stored response, conversation, background job, and local compaction reference.

`background: true` persists a queued response and returns immediately. The
single resident compute worker drains durable jobs, including jobs queued before
a restart. A job already marked in-progress when the database is reopened is
failed once with `error.code: "server_restarted"` rather than run twice. The
cancel endpoint atomically removes a queued job or signals an in-progress
session; terminal responses reject cancellation. Active responses must be
cancelled before deletion.

## Authentication and remote binding

Loopback binding is unauthenticated by default. Configure one Bearer key using
`NFN_INFER_API_KEY`, or one or more rotation keys in a private file:

```bash
chmod 600 /secure/nfn.keys
nfn infer --checkpoint artifacts/model-native --serve \
  --api-key-file /secure/nfn.keys
```

When a key is configured, every route, including `/health`, requires
`Authorization: Bearer ...`. Key comparison is constant-time. The key file is
rejected if group or other users can access it.

Binding to anything other than a loopback address fails before model load
unless a key is configured. `--allow-unauthenticated-remote` is an explicit
security override; it should be used only behind a trusted external access
control layer.

## Queue and errors

One thread executes resident model work. `--queue-capacity N` bounds the number
of waiting generations in addition to the running request. Admission is
non-blocking; saturation returns HTTP `429` with `code: "queue_saturated"`.
`--session-limit N` independently bounds all admitted request-session
reservations, including the running request and queued work. It defaults to
`queue_capacity + 1`; reaching it returns HTTP `429` with
`code: "session_limit_exceeded"`. `/health` exposes both limits, live session
reservations, and queue/session rejection counters.
The runtime creates an isolated native session per Chat or background request.
With prefix reuse enabled, a foreground Response instead receives an
independent COW fork of an exact cached prefix or a fresh leased session. Its
lease survives through durable terminalization and is then admitted or closed
under the rules above. Worker exceptions are returned to the event-loop side
before being re-raised, so a failed/poisoned request releases its slot and the
next admitted request can proceed; the resident model is not reloaded and no
subprocess fallback is used. It does not claim batching.

During lifespan shutdown the app stops and awaits the background driver, then
awaits every tracked foreground Responses SSE driver, including one whose HTTP
subscriber disconnected. It drains the single-worker queue before shutting
down the prefix cache, then closes the resident model and state database. This
ordering prevents a late driver from publishing or using a session after model
teardown.

All HTTP failures use:

```json
{
  "error": {
    "message": "Human-readable message",
    "type": "invalid_request_error",
    "param": "messages",
    "code": "context_length_exceeded"
  }
}
```

Validation/capability errors are `400`, missing model/resources are `404`,
invalid Bearer credentials are `401`, durable lineage/revision conflicts are
`409`, queue or session-limit saturation is `429`, and native generation
failures are normalized to `500` without exposing internal paths.

## Current artifact boundary

The compiled resident engine currently supports the seven reviewed dense GPT
preset topologies (`gpt2`, megakernel, MoA, z-loss, QK-norm, stable, and softcap)
with a real
bf16-v5 CPU forward with full-prefix recomputation (`off`), a preallocated
lossless K/V cache (`auto`/`full`), or the proved packed native CPU
TurboQuant cache (`turboquant`) for the supported dense adapters. An ordinary
compatible `nfn migrate graph-to-native --weights model_*.bin` artifact is
fingerprinted, bound to resident ABI v1, and stamped with lean serving
capability; MoA instead uses the metadata workflow below. The artifact must
also declare a supported tiktoken encoding and positive context limit and
provide a supported chat template (or the operator must select an explicit
fallback). Generic `.pt` bundles, graph-only artifacts,
differential/modern dense variants, bare MoA `.bin` files, and unimplemented families remain
rejected. Canonical LLaMA separately proves the resident/lean-serving model ABI
with lossless `auto`/`full` and recompute `off`, but its migration metadata uses
SentencePiece classification and does not supply the supported tiktoken codec,
chat template, or authoritative context presentation required here. It
therefore is not end-to-end text-serving proof and fails before bind unless an
artifact independently satisfies those presentation gates. Reviewed-dense
artifacts may opt into the separately gated hybrid Tile-CUDA TurboQuant
attention backend at process startup; CPU remains the default, while all
non-dense TurboQuant serving remains outside this proof. That Tile backend
cannot be combined with `--prefix-cache-capacity`: its device K/V ownership has
not passed the required session-fork/COW gate.

`gpt2_moa` is admitted only after migration through
`model_XXXXXXXX.moa.json`, with its named dense-v5 sibling and empty
`DONE_XXXXXXXX`. The metadata must bind the exact source graph/model, canonical
GELU/ReLU/SiLU/ReLU2 candidates, selected activation, and positive interval.
The resident engine uses that fixed activation through prefill/decode and
lossless or packed TurboQuant cache paths. With explicit strict-sidecar startup
configuration, only its compressed historical attention may use the hybrid
Tile-CUDA backend; ordinary model compute and encoding remain CPU-resident.
