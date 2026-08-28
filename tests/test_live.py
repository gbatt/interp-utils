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
