"""Bounded OpenAI Python SDK contract checks for native serving.

These tests intentionally pin the SDK version they audited.  The main NeuralFn
test environment does not currently install ``openai``, so collection skips this
module instead of claiming SDK coverage that was not run.  Re-audit the shapes
and update ``_AUDITED_OPENAI_VERSION`` when intentionally upgrading the client.

The transport buffers ASGI responses in-process.  It verifies official SDK
deserialization and SSE event parsing, while the lower-level native serving
tests remain responsible for incremental delivery and disconnect semantics.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
import json
from pathlib import Path

import anyio
import httpx
import pytest
from pydantic import BaseModel, ConfigDict


_AUDITED_OPENAI_VERSION = "2.44.0"
openai = pytest.importorskip(
    "openai",
    minversion=_AUDITED_OPENAI_VERSION,
    reason="OpenAI SDK compatibility tests require the optional openai package",
)
if openai.__version__ != _AUDITED_OPENAI_VERSION:
    pytest.skip(
        "OpenAI SDK compatibility was audited only against "
        f"openai=={_AUDITED_OPENAI_VERSION}; found {openai.__version__}",
        allow_module_level=True,
    )

from openai import (  # noqa: E402
    AuthenticationError,
    BadRequestError,
    ConflictError,
    InternalServerError,
    NotFoundError,
    OpenAI,
    RateLimitError,
)
from openai.types import Model  # noqa: E402
from openai.types.chat import ChatCompletion, ChatCompletionChunk  # noqa: E402
from openai.types.conversations import (  # noqa: E402
    Conversation,
    ConversationDeletedResource,
)
from openai.types.responses import (  # noqa: E402
    CompactedResponse,
    InputTokenCountResponse,
    ParsedResponse,
    ParsedResponseFunctionToolCall,
    ParsedResponseOutputMessage,
    ParsedResponseOutputText,
    Response,
    ResponseOutputMessage,
    ResponseOutputText,
)

from test_native_serve import (  # noqa: E402
    BearerAuth,
    FailOnceModel,
    _constrained_stateful_runtime,
    _runtime,
    _stateful_runtime,
    create_native_inference_app,
    _with_client,
)


class StrictAnswer(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    city: str
    temperature_c: int


class LookupWeatherArguments(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    city: str


class _SyncASGITransport(httpx.BaseTransport):
    """Bridge the synchronous OpenAI client to an ASGI app without a socket."""

    def __init__(self, app) -> None:
        self._app = app

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        body = request.read()

        async def send() -> httpx.Response:
            transport = httpx.ASGITransport(
                app=self._app,
                raise_app_exceptions=False,
            )
            try:
                async_request = httpx.Request(
                    request.method,
                    request.url,
                    headers=request.headers,
                    content=body,
                )
                response = await transport.handle_async_request(async_request)
                content = await response.aread()
                return httpx.Response(
                    response.status_code,
                    headers=response.headers,
                    content=content,
                    request=request,
                )
            finally:
                await transport.aclose()

        return asyncio.run(send())


def _sdk_client(app, *, api_key: str = "sdk-test-key") -> tuple[OpenAI, httpx.Client]:
    http_client = httpx.Client(transport=_SyncASGITransport(app))
    client = OpenAI(
        api_key=api_key,
        base_url="http://testserver/v1",
        http_client=http_client,
        max_retries=0,
    )
    return client, http_client


def _close_app(client: OpenAI, http_client: httpx.Client, app) -> None:
    client.close()
    http_client.close()
    app.state.generation_queue.close()
    app.state.native_runtime.close()


@pytest.fixture
def text_sdk_client() -> Iterator[tuple[OpenAI, object]]:
    app = create_native_inference_app(_runtime(), queue_capacity=1)
    client, http_client = _sdk_client(app)
    try:
        yield client, app
    finally:
        _close_app(client, http_client, app)


@pytest.fixture
def stateful_sdk_client(tmp_path: Path) -> Iterator[tuple[OpenAI, object]]:
    app = create_native_inference_app(
        _stateful_runtime(tmp_path / "openai-sdk-state.sqlite"),
        auth=BearerAuth(("sdk-test-key",)),
        queue_capacity=1,
    )
    client, http_client = _sdk_client(app)
    try:
        yield client, app
    finally:
        _close_app(client, http_client, app)


@pytest.fixture
def constrained_sdk_client(tmp_path: Path) -> Iterator[tuple[OpenAI, object, object]]:
    runtime = _constrained_stateful_runtime(
        tmp_path / "openai-sdk-constrained-state.sqlite",
    )
    app = create_native_inference_app(
        runtime,
        auth=BearerAuth(("sdk-test-key",)),
        queue_capacity=1,
    )
    client, http_client = _sdk_client(app)
    try:
        yield client, app, runtime.model
    finally:
        _close_app(client, http_client, app)


def test_sdk_parses_models_chat_and_chat_stream(text_sdk_client) -> None:
    client, _app = text_sdk_client

    models = client.models.list()
    assert isinstance(models.data[0], Model)
    assert models.data[0].id == "nfn-test"
    assert isinstance(client.models.retrieve("nfn-test"), Model)

    completion = client.chat.completions.create(
        model="nfn-test",
        messages=[{"role": "user", "content": "Hello"}],
        max_completion_tokens=2,
        temperature=0,
    )
    assert isinstance(completion, ChatCompletion)
    assert completion.object == "chat.completion"
    assert completion.choices[0].message.content == "Hello!"
    assert completion.usage is not None
    assert completion.usage.total_tokens == 5

    chunks = list(
        client.chat.completions.create(
            model="nfn-test",
            messages=[{"role": "user", "content": "Hello"}],
            max_completion_tokens=2,
            temperature=0,
            stream=True,
            stream_options={"include_usage": True},
        )
    )
    assert chunks
    assert all(isinstance(chunk, ChatCompletionChunk) for chunk in chunks)
    assert "".join(
        chunk.choices[0].delta.content or ""
        for chunk in chunks
        if chunk.choices
    ) == "Hello!"
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.total_tokens == 5


def test_sdk_chat_keeps_function_tools_and_structured_output_fail_closed(
    text_sdk_client,
) -> None:
    client, app = text_sdk_client
    model = app.state.native_runtime.model

    with pytest.raises(BadRequestError) as tools_error:
        client.chat.completions.create(
            model="nfn-test",
            messages=[{"role": "user", "content": "Weather in London?"}],
            tools=[
                openai.pydantic_function_tool(
                    LookupWeatherArguments,
                    name="lookup_weather",
                )
            ],
        )
    assert tools_error.value.status_code == 400
    assert tools_error.value.code == "unsupported_feature"
    assert tools_error.value.param == "tools"

    with pytest.raises(BadRequestError) as structured_error:
        client.chat.completions.parse(
            model="nfn-test",
            messages=[{"role": "user", "content": "Return the weather."}],
            response_format=StrictAnswer,
        )
    assert structured_error.value.status_code == 400
    assert structured_error.value.code == "unsupported_feature"
    assert structured_error.value.param == "response_format"
    assert model.session_creates == 0


def test_sdk_parses_buffered_strict_structured_response(
    constrained_sdk_client,
) -> None:
    client, _app, model = constrained_sdk_client
    raw_output = '{"city":"London","temperature_c":12}'
    model.queue_constrained(raw_output)

    response = client.responses.parse(
        model="nfn-test",
        input="Return the weather as strict JSON.",
        text_format=StrictAnswer,
        max_output_tokens=64,
        temperature=0,
        top_p=1,
        store=True,
    )

    assert isinstance(response, ParsedResponse)
    assert response.status == "completed"
    assert response.temperature == 0
    assert response.top_p == 1
    assert response.text is not None
    assert response.text.format is not None
    assert response.text.format.type == "json_schema"
    assert response.text.format.strict is True
    assert response.text.format.name == "StrictAnswer"
    schema = response.text.format.schema_
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["properties"]["city"]["type"] == "string"
    assert schema["properties"]["temperature_c"]["type"] == "integer"

    assert len(response.output) == 1
    message = response.output[0]
    assert isinstance(message, ParsedResponseOutputMessage)
    assert message.type == "message"
    assert message.status == "completed"
    assert len(message.content) == 1
    content = message.content[0]
    assert isinstance(content, ParsedResponseOutputText)
    assert content.type == "output_text"
    assert content.text == raw_output
    assert json.loads(content.text) == {
        "city": "London",
        "temperature_c": 12,
    }
    assert isinstance(content.parsed, StrictAnswer)
    assert content.parsed.city == "London"
    assert type(content.parsed.temperature_c) is int
    assert response.output_parsed is content.parsed
    assert response.output_text == raw_output
    assert model.session_creates == 1
    assert model.current_logits_calls > 0
    assert model.decode_calls == 0


def test_sdk_parses_forced_function_call_and_client_result_continuation(
    constrained_sdk_client,
) -> None:
    client, _app, model = constrained_sdk_client
    raw_arguments = '{"city":"London"}'
    model.queue_constrained(raw_arguments)

    response = client.responses.parse(
        model="nfn-test",
        input="Look up the weather in London.",
        tools=[
            openai.pydantic_function_tool(
                LookupWeatherArguments,
                name="lookup_weather",
            )
        ],
        tool_choice={"type": "function", "name": "lookup_weather"},
        parallel_tool_calls=False,
        max_output_tokens=64,
        temperature=0,
        top_p=1,
        store=True,
    )

    assert isinstance(response, ParsedResponse)
    assert response.status == "completed"
    assert response.parallel_tool_calls is False
    assert len(response.tools) == 1
    response_tool = response.tools[0]
    assert response_tool.type == "function"
    assert response_tool.name == "lookup_weather"
    assert response_tool.strict is True
    assert response_tool.parameters is not None
    assert response_tool.parameters["type"] == "object"
    assert response_tool.parameters["additionalProperties"] is False
    assert response_tool.parameters["properties"]["city"]["type"] == "string"
    assert response.tool_choice.type == "function"
    assert response.tool_choice.name == "lookup_weather"

    assert len(response.output) == 1
    call = response.output[0]
    assert isinstance(call, ParsedResponseFunctionToolCall)
    assert call.type == "function_call"
    assert call.status == "completed"
    assert call.name == "lookup_weather"
    assert isinstance(call.call_id, str) and call.call_id
    assert call.arguments == raw_arguments
    assert isinstance(call.parsed_arguments, LookupWeatherArguments)
    assert call.parsed_arguments.city == "London"

    def lookup_weather(arguments: LookupWeatherArguments) -> dict[str, int]:
        assert arguments.city == "London"
        model.function_executions += 1
        return {"temperature_c": 12}

    function_result = lookup_weather(call.parsed_arguments)
    assert model.function_executions == 1

    final_text = "Weather: 12 C"
    model.queue_text(final_text)
    final_response = client.responses.create(
        model="nfn-test",
        previous_response_id=response.id,
        input=[
            {
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps(function_result, separators=(",", ":")),
            }
        ],
        max_output_tokens=64,
        temperature=0,
        top_p=1,
        store=True,
    )

    assert isinstance(final_response, Response)
    assert final_response.status == "completed"
    assert final_response.previous_response_id == response.id
    assert final_response.tool_choice == "none"
    assert final_response.tools == []
    assert len(final_response.output) == 1
    final_message = final_response.output[0]
    assert isinstance(final_message, ResponseOutputMessage)
    assert final_message.type == "message"
    assert final_message.status == "completed"
    assert len(final_message.content) == 1
    final_content = final_message.content[0]
    assert isinstance(final_content, ResponseOutputText)
    assert final_content.type == "output_text"
    assert final_content.text == final_text
    assert final_response.output_text == final_text
    assert model.function_executions == 1
    assert model.session_creates == 2
    assert model.decode_calls == 1


@pytest.mark.parametrize(
    ("request_overrides", "expected_param"),
    (
        pytest.param({"stream": True}, "stream", id="stream"),
        pytest.param({"background": True}, "background", id="background"),
        pytest.param({"temperature": 0.1}, "temperature", id="temperature"),
        pytest.param({"top_p": 0.9}, "top_p", id="top-p"),
    ),
)
def test_sdk_strict_structured_response_rejects_unsupported_execution_modes(
    constrained_sdk_client,
    request_overrides: dict[str, object],
    expected_param: str,
) -> None:
    client, _app, model = constrained_sdk_client
    request: dict[str, object] = {
        "model": "nfn-test",
        "input": "Return the weather as strict JSON.",
        "text_format": StrictAnswer,
        "max_output_tokens": 64,
        "temperature": 0,
        "top_p": 1,
        "store": True,
    }
    request.update(request_overrides)

    with pytest.raises(BadRequestError) as error:
        client.responses.parse(**request)
    assert error.value.status_code == 400
    assert error.value.code == "unsupported_feature"
    assert error.value.param == expected_param
    assert model.session_creates == 0


@pytest.mark.parametrize(
    ("request_overrides", "expected_param"),
    (
        pytest.param(
            {"tool_choice": "auto"},
            "tool_choice",
            id="automatic-tool-choice",
        ),
        pytest.param(
            {"parallel_tool_calls": True},
            "parallel_tool_calls",
            id="parallel-tool-calls",
        ),
        pytest.param(
            {"truncation": "auto"},
            "truncation",
            id="automatic-truncation",
        ),
    ),
)
def test_sdk_forced_function_call_rejects_unsupported_selection_modes(
    constrained_sdk_client,
    request_overrides: dict[str, object],
    expected_param: str,
) -> None:
    client, _app, model = constrained_sdk_client
    request: dict[str, object] = {
        "model": "nfn-test",
        "input": "Look up the weather in London.",
        "tools": [
            openai.pydantic_function_tool(
                LookupWeatherArguments,
                name="lookup_weather",
            )
        ],
        "tool_choice": {"type": "function", "name": "lookup_weather"},
        "parallel_tool_calls": False,
        "max_output_tokens": 64,
        "temperature": 0,
        "top_p": 1,
        "store": True,
    }
    request.update(request_overrides)

    with pytest.raises(BadRequestError) as error:
        client.responses.parse(**request)
    assert error.value.status_code == 400
    assert error.value.code == "unsupported_feature"
    assert error.value.param == expected_param
    assert model.session_creates == 0


def test_sdk_function_call_output_rejects_wrong_call_id_before_generation(
    constrained_sdk_client,
) -> None:
    client, _app, model = constrained_sdk_client
    model.queue_constrained('{"city":"London"}')
    response = client.responses.parse(
        model="nfn-test",
        input="Look up the weather in London.",
        tools=[
            openai.pydantic_function_tool(
                LookupWeatherArguments,
                name="lookup_weather",
            )
        ],
        tool_choice={"type": "function", "name": "lookup_weather"},
        parallel_tool_calls=False,
        max_output_tokens=64,
        temperature=0,
        top_p=1,
        store=True,
    )
    assert len(response.output) == 1
    call = response.output[0]
    assert isinstance(call, ParsedResponseFunctionToolCall)
    assert call.call_id
    assert model.session_creates == 1

    with pytest.raises(BadRequestError) as error:
        client.responses.create(
            model="nfn-test",
            previous_response_id=response.id,
            input=[
                {
                    "type": "function_call_output",
                    "call_id": "call_not_visible",
                    "output": '{"temperature_c":12}',
                }
            ],
            max_output_tokens=64,
            temperature=0,
            top_p=1,
            store=True,
        )
    assert error.value.status_code == 400
    assert error.value.code == "function_call_not_found"
    assert error.value.param == "input.0.call_id"
    assert model.session_creates == 1


def test_sdk_parses_stateful_responses_streaming_and_compaction(
    stateful_sdk_client,
) -> None:
    client, _app = stateful_sdk_client

    response = client.responses.create(
        model="nfn-test",
        input="Hello",
        max_output_tokens=2,
        temperature=0,
        store=True,
    )
    assert isinstance(response, Response)
    assert response.status == "completed"
    assert response.output_text == "Hello!"

    retrieved = client.responses.retrieve(response.id)
    assert isinstance(retrieved, Response)
    assert retrieved.id == response.id

    input_items = client.responses.input_items.list(
        response.id,
        include=["message.output_text.logprobs"],
        limit=10,
        order="asc",
    )
    assert input_items.has_more is False
    assert [item.type for item in input_items.data] == ["message"]

    token_count = client.responses.input_tokens.count(
        model="nfn-test",
        input="Hello",
    )
    assert isinstance(token_count, InputTokenCountResponse)
    assert token_count.object == "response.input_tokens"
    assert token_count.input_tokens == 3

    events = list(
        client.responses.create(
            model="nfn-test",
            input="Hello",
            max_output_tokens=2,
            temperature=0,
            stream=True,
            store=True,
        )
    )
    assert events[0].type == "response.created"
    assert events[-1].type == "response.completed"
    assert [event.sequence_number for event in events] == list(range(len(events)))

    follow_up = client.responses.create(
        model="nfn-test",
        input="Again",
        max_output_tokens=2,
        previous_response_id=response.id,
        store=True,
    )
    assert isinstance(follow_up, Response)
    assert follow_up.previous_response_id == response.id

    compacted = client.responses.compact(model="nfn-test", input="Hello")
    assert isinstance(compacted, CompactedResponse)
    assert compacted.object == "response.compaction"
    compaction_items = [item for item in compacted.output if item.type == "compaction"]
    assert len(compaction_items) == 1
    assert compaction_items[0].encrypted_content.startswith("nfncmp_")

    # openai==2.44.0 deliberately discards the documented deletion envelope.
    assert client.responses.delete(response.id) is None


def test_sdk_parses_conversations_and_background_cancel(stateful_sdk_client) -> None:
    client, _app = stateful_sdk_client

    conversation = client.conversations.create(
        items=[{"type": "message", "role": "user", "content": "one"}],
        metadata={"stage": "created"},
    )
    assert isinstance(conversation, Conversation)
    assert conversation.object == "conversation"

    updated = client.conversations.update(
        conversation.id,
        metadata={"stage": "updated"},
    )
    assert isinstance(updated, Conversation)
    assert updated.metadata == {"stage": "updated"}

    created_items = client.conversations.items.create(
        conversation.id,
        items=[{"type": "message", "role": "user", "content": "two"}],
    )
    listed_items = client.conversations.items.list(
        conversation.id,
        limit=10,
        order="asc",
    )
    assert len(created_items.data) == 1
    assert len(listed_items.data) == 2

    item = client.conversations.items.retrieve(
        created_items.data[0].id,
        conversation_id=conversation.id,
    )
    after_item_delete = client.conversations.items.delete(
        item.id,
        conversation_id=conversation.id,
    )
    assert isinstance(after_item_delete, Conversation)

    background = client.responses.create(
        model="nfn-test",
        input="later",
        background=True,
        store=True,
    )
    assert isinstance(background, Response)
    assert background.status == "queued"
    cancelled = client.responses.cancel(background.id)
    assert isinstance(cancelled, Response)
    assert cancelled.status == "cancelled"

    deleted = client.conversations.delete(conversation.id)
    assert isinstance(deleted, ConversationDeletedResource)
    assert deleted.deleted is True


def test_sdk_stored_ids_survive_sqlite_restart(tmp_path: Path) -> None:
    state_path = tmp_path / "openai-sdk-restart.sqlite"
    first_app = create_native_inference_app(
        _stateful_runtime(state_path),
        auth=BearerAuth(("sdk-test-key",)),
        queue_capacity=1,
    )
    first_client, first_http_client = _sdk_client(first_app)
    try:
        response = first_client.responses.create(
            model="nfn-test",
            input="persist me",
            max_output_tokens=2,
            store=True,
        )
        conversation = first_client.conversations.create(
            items=[{"type": "message", "role": "user", "content": "remember me"}],
        )
        background = first_client.responses.create(
            model="nfn-test",
            input="queued across restart",
            background=True,
            store=True,
        )
        response_id = response.id
        conversation_id = conversation.id
        background_id = background.id
    finally:
        _close_app(first_client, first_http_client, first_app)

    second_app = create_native_inference_app(
        _stateful_runtime(state_path),
        auth=BearerAuth(("sdk-test-key",)),
        queue_capacity=1,
    )
    second_client, second_http_client = _sdk_client(second_app)
    try:
        restored_response = second_client.responses.retrieve(response_id)
        restored_conversation = second_client.conversations.retrieve(conversation_id)
        restored_background = second_client.responses.retrieve(background_id)

        assert isinstance(restored_response, Response)
        assert restored_response.id == response_id
        assert restored_response.status == "completed"
        assert isinstance(restored_conversation, Conversation)
        assert restored_conversation.id == conversation_id
        assert isinstance(restored_background, Response)
        assert restored_background.status == "queued"

        cancelled = second_client.responses.cancel(background_id)
        assert isinstance(cancelled, Response)
        assert cancelled.status == "cancelled"
    finally:
        _close_app(second_client, second_http_client, second_app)


def test_sdk_background_stream_create_and_resume_across_sqlite_restart(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "openai-sdk-stream-restart.sqlite"
    first_app = create_native_inference_app(
        _stateful_runtime(state_path),
        auth=BearerAuth(("sdk-test-key",)),
        queue_capacity=1,
    )
    captured: dict[str, int | str] = {}

    async def create_stream(client: httpx.AsyncClient) -> None:
        streamed = await client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer sdk-test-key"},
            json={
                "model": "nfn-test",
                "input": "durable SDK stream",
                "max_output_tokens": 2,
                "temperature": 0,
                "background": True,
                "stream": True,
                "store": True,
            },
        )
        assert streamed.status_code == 200, streamed.text
        events = [
            json.loads(line[6:])
            for line in streamed.text.splitlines()
            if line.startswith("data: ")
        ]
        assert events[0]["type"] == "response.created"
        assert events[-1]["type"] == "response.completed"
        captured["response_id"] = events[0]["response"]["id"]
        captured["cursor"] = next(
            event["sequence_number"]
            for event in events
            if event["type"] == "response.output_text.delta"
        )
        captured["event_count"] = len(events)

    anyio.run(_with_client, first_app, create_stream)

    second_app = create_native_inference_app(
        _stateful_runtime(state_path),
        auth=BearerAuth(("sdk-test-key",)),
        queue_capacity=1,
    )
    client, http_client = _sdk_client(second_app)
    try:
        resumed = list(
            client.responses.retrieve(
                str(captured["response_id"]),
                stream=True,
                starting_after=int(captured["cursor"]),
                include=["message.output_text.logprobs"],
                include_obfuscation=False,
            )
        )
        assert resumed
        assert all(
            event.sequence_number > int(captured["cursor"]) for event in resumed
        )
        assert [event.sequence_number for event in resumed] == list(
            range(int(captured["cursor"]) + 1, int(captured["event_count"]))
        )
        assert resumed[-1].type == "response.completed"
    finally:
        _close_app(client, http_client, second_app)


def test_sdk_maps_openai_status_error_classes(stateful_sdk_client) -> None:
    client, app = stateful_sdk_client

    with pytest.raises(BadRequestError) as bad_request:
        client.responses.create(
            model="nfn-test",
            input="Hello",
            extra_body={"unknown_option": True},
        )
    assert bad_request.value.status_code == 400
    assert bad_request.value.code == "unsupported_feature"

    with pytest.raises(NotFoundError) as not_found:
        client.responses.retrieve("resp_missing")
    assert not_found.value.status_code == 404
    assert not_found.value.code == "response_not_found"

    bad_http_client = httpx.Client(transport=_SyncASGITransport(app))
    bad_client = OpenAI(
        api_key="wrong-key",
        base_url="http://testserver/v1",
        http_client=bad_http_client,
        max_retries=0,
    )
    try:
        with pytest.raises(AuthenticationError) as unauthorized:
            bad_client.models.list()
        assert unauthorized.value.status_code == 401
        assert unauthorized.value.code == "invalid_api_key"
    finally:
        bad_client.close()
        bad_http_client.close()

    active = client.responses.create(
        model="nfn-test",
        input="later",
        background=True,
        store=True,
    )
    with pytest.raises(ConflictError) as conflict:
        client.responses.delete(active.id)
    assert conflict.value.status_code == 409
    assert conflict.value.code == "response_active"
    client.responses.cancel(active.id)

    reserved_tickets = []
    while True:
        ticket, rejection = app.state.generation_queue.admit()
        if ticket is None:
            break
        assert rejection is None
        reserved_tickets.append(ticket)
    assert reserved_tickets
    assert rejection == "queue_saturated"
    try:
        with pytest.raises(RateLimitError) as rate_limited:
            client.chat.completions.create(
                model="nfn-test",
                messages=[{"role": "user", "content": "Hello"}],
            )
        assert rate_limited.value.status_code == 429
        assert rate_limited.value.code == "queue_saturated"
    finally:
        for ticket in reserved_tickets:
            asyncio.run(ticket.run(lambda: None))


def test_sdk_maps_generation_failures_to_internal_server_error() -> None:
    app = create_native_inference_app(_runtime(FailOnceModel()), queue_capacity=1)
    client, http_client = _sdk_client(app)
    try:
        with pytest.raises(InternalServerError) as server_error:
            client.chat.completions.create(
                model="nfn-test",
                messages=[{"role": "user", "content": "Hello"}],
            )
        assert server_error.value.status_code == 500
        assert server_error.value.code == "generation_failed"
    finally:
        _close_app(client, http_client, app)
