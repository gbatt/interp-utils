"""Live smoke tests. Gated behind INTERP_UTILS_LIVE=1 — they hit real
APIs and spend tokens. The default suite skips them.

    INTERP_UTILS_LIVE=1 uv run pytest tests/test_live.py
"""

import os

import pytest

from interp_utils.llm import Completion, LLMClient
from interp_utils.settings import get_settings

pytestmark = pytest.mark.skipif(
    os.environ.get("INTERP_UTILS_LIVE") != "1",
    reason="set INTERP_UTILS_LIVE=1 to run live API tests",
)

# A cheap, widely-available model on OpenRouter for smoke testing.
SMOKE_MODEL = "openai/gpt-4o-mini"
# A cheap reasoning model that returns a separate reasoning trace.
REASONING_MODEL = "openai/gpt-oss-20b"
# A non-reasoning model on Nebius that serves /v1/completions + logprobs.
NEBIUS_TEXT_MODEL = "google/gemma-3-27b-it"


async def test_openrouter_roundtrip():
    if not get_settings().openrouter_api_key:
        pytest.skip("no OPENROUTER_API_KEY")
    client = LLMClient("openrouter", max_concurrency=2)
    try:
        result = await client.complete(
            [{"role": "user", "content": "Reply with exactly: pong"}],
            model=SMOKE_MODEL,
            max_tokens=10,
            temperature=0.0,
        )
        assert isinstance(result, Completion)
        assert "pong" in result.text.lower()
        assert result.usage.total_tokens > 0
        assert client.usage[SMOKE_MODEL].total_tokens > 0
    finally:
        await client.aclose()


async def test_openrouter_captures_reasoning():
    if not get_settings().openrouter_api_key:
        pytest.skip("no OPENROUTER_API_KEY")
    client = LLMClient("openrouter", max_concurrency=2)
    try:
        result = await client.complete(
            [{"role": "user", "content": "What is 17 * 23? Think briefly."}],
            model=REASONING_MODEL,
            max_tokens=400,
        )
        assert isinstance(result, Completion)
        # The reasoning model returns its CoT in a separate field, not
        # inline; the final answer is in text.
        assert result.reasoning is not None
        assert len(result.reasoning) > 0
        assert "391" in result.text
    finally:
        await client.aclose()


async def test_openrouter_batch_concurrency():
    if not get_settings().openrouter_api_key:
        pytest.skip("no OPENROUTER_API_KEY")
    client = LLMClient("openrouter", max_concurrency=4)
    try:
        reqs = [
            [{"role": "user", "content": f"Say the number {i}."}]
            for i in range(6)
        ]
        results = await client.complete_many(
            reqs, model=SMOKE_MODEL, max_tokens=10, temperature=0.0
        )
        assert len(results) == 6
        assert all(isinstance(r, Completion) for r in results)
    finally:
        await client.aclose()


async def test_nebius_text_continuation():
    if not get_settings().nebius_api_key:
        pytest.skip("no NEBIUS_API_KEY")
    client = LLMClient("nebius", max_concurrency=2)
    try:
        result = await client.complete_text(
            "The capital of France is",
            model=NEBIUS_TEXT_MODEL,
            max_tokens=10,
            temperature=0.0,
        )
        assert isinstance(result, Completion)
        assert result.kind == "text"
        assert "Paris" in result.text
        assert result.usage.completion_tokens > 0
    finally:
        await client.aclose()


async def test_nebius_echo_logprobs():
    # echo=True + max_tokens=0 returns the prompt tokens' own logprobs —
    # the shape resampling scoring slices by character offset. The leading
    # <bos> token has no preceding context, so its logprob/top_logprobs
    # are None. Nebius caps logprobs at 20.
    if not get_settings().nebius_api_key:
        pytest.skip("no NEBIUS_API_KEY")
    client = LLMClient("nebius", max_concurrency=2)
    try:
        result = await client.complete_text(
            "The capital of France is",
            model=NEBIUS_TEXT_MODEL,
            max_tokens=0,
            echo=True,
            logprobs=5,
        )
        assert isinstance(result, Completion)
        # echo returns the prompt itself in `text`.
        assert result.text == "The capital of France is"
        lps = result.logprobs
        assert lps is not None
        assert len(lps) > 1
        # Same per-token shape as chat, with offsets for slicing.
        assert all(t.text_offset is not None for t in lps)
        # First (bos) token has no preceding context.
        assert lps[0].logprob is None
        assert lps[0].top is None
        # A later token carries a real logprob and alternatives.
        assert lps[-1].logprob is not None
        assert lps[-1].top
    finally:
        await client.aclose()
