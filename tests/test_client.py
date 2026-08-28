"""Unit tests for LLMClient.

The OpenAI SDK owns its own httpx client, so we test by injecting an
`httpx.MockTransport` (the SDK's supported testing path) rather than
patching HTTP globally.
"""

import httpx
import pytest

from interp_utils.llm import Completion, CompletionError, LLMClient, Usage


def _chat_body(text: str, *, prompt=5, completion=7) -> dict:
    return {
        "id": "x",
        "object": "chat.completion",
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
    }


def _client(handler) -> LLMClient:
    """Client whose transport is driven by `handler(request) -> Response`.

    Retries are disabled so mocked error responses surface immediately
    and one request maps to exactly one handler call.
    """
    transport = httpx.MockTransport(handler)
    return LLMClient(
        "openrouter",
        api_key="test-key",
        http_client=httpx.AsyncClient(transport=transport),
        max_retries=0,
    )


def test_unknown_provider_rejected():
    with pytest.raises(ValueError, match="Unknown provider"):
        LLMClient("nope", api_key="k")


def test_missing_key_rejected():
    with pytest.raises(ValueError, match="No API key"):
        LLMClient("openrouter", api_key="")


async def test_complete_parses_text_and_usage():
    client = _client(lambda req: httpx.Response(200, json=_chat_body("hello")))
    result = await client.complete(
        [{"role": "user", "content": "hi"}], model="test-model"
    )
    assert isinstance(result, Completion)
    assert result.text == "hello"
    assert result.finish_reason == "stop"
    assert result.usage == Usage(prompt_tokens=5, completion_tokens=7)


async def test_complete_sends_expected_request():
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        seen["auth"] = req.headers.get("authorization")
        return httpx.Response(200, json=_chat_body("ok"))

    client = _client(handler)
    await client.complete([{"role": "user", "content": "hi"}], model="m")
    assert seen["url"].endswith("/chat/completions")
    assert seen["auth"] == "Bearer test-key"


async def test_usage_accumulates_per_model():
    client = _client(lambda req: httpx.Response(200, json=_chat_body("hi")))
    await client.complete([{"role": "user", "content": "a"}], model="m")
    await client.complete([{"role": "user", "content": "b"}], model="m")
    assert client.usage["m"].total_tokens == 24


async def test_complete_safe_returns_error_record():
    client = _client(lambda req: httpx.Response(500, json={"error": "boom"}))
    result = await client.complete_safe(
        [{"role": "user", "content": "x"}], model="m"
    )
    assert isinstance(result, CompletionError)
    assert result.model == "m"
    assert result.error_type


async def test_complete_many_preserves_order_and_isolates_failures():
    def handler(req: httpx.Request) -> httpx.Response:
        # Fail exactly the item whose content is "1", by inspecting the
        # request body — robust to concurrent, out-of-order execution.
        if b'"1"' in req.content:
            return httpx.Response(500, json={"error": "boom"})
        return httpx.Response(200, json=_chat_body(req.content.decode()))

    client = _client(handler)
    reqs = [[{"role": "user", "content": str(i)}] for i in range(3)]
    results = await client.complete_many(reqs, model="m")

    assert len(results) == 3
    # Order preserved: index 1 is the failure, 0 and 2 succeed.
    assert isinstance(results[0], Completion)
    assert isinstance(results[1], CompletionError)
    assert isinstance(results[2], Completion)


def test_usage_addition():
    a = Usage(prompt_tokens=1, completion_tokens=2)
    b = Usage(prompt_tokens=3, completion_tokens=4)
    assert (a + b) == Usage(prompt_tokens=4, completion_tokens=6)
    assert (a + b).total_tokens == 10
