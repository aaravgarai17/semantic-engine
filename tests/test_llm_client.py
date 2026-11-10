"""The real Anthropic client, exercised end to end against a fake HTTP server.

Every other test stubs the model out, which is why a dependency bump once
broke client construction without a single test failing. This one builds the
real SDK client and sends a real request through httpx, so an incompatible
anthropic/httpx pair fails CI instead of the first live query.
"""

import json

import httpx

from app.llm import ClaudeLLM


def _fake_messages_api(request: httpx.Request) -> httpx.Response:
    assert request.url.path == "/v1/messages"
    body = json.loads(request.content)
    assert body["messages"][0]["content"] == "What is ERR_4032?"
    return httpx.Response(200, json={
        "id": "msg_test", "type": "message", "role": "assistant",
        "model": body["model"], "stop_reason": "end_turn", "stop_sequence": None,
        "content": [{"type": "text", "text": "A DNS timeout [1]."}],
        "usage": {"input_tokens": 42, "output_tokens": 7},
    })


def test_real_client_constructs_and_round_trips():
    from anthropic import Anthropic

    llm = ClaudeLLM(api_key="test-key")
    llm._client = Anthropic(
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(_fake_messages_api)),
    )

    text, tokens_in, tokens_out = llm.complete("system", "What is ERR_4032?")

    assert text == "A DNS timeout [1]."
    assert (tokens_in, tokens_out) == (42, 7)
