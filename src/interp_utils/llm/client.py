"""Provider-agnostic async completion client over the OpenAI SDK.

Both OpenRouter and Nebius expose OpenAI-compatible APIs, so one client
with a per-provider base URL covers them. The client adds: bounded
concurrency, explicit failure records for batch calls, and per-model
token-usage accounting.
"""

from __future__ import annotations

from typing import cast

import anyio
import asyncer
import httpx
from openai import AsyncOpenAI, OpenAIError
from openai.types.chat import ChatCompletionMessageParam

from interp_utils.settings import get_settings

from .models import (
    Completion,
    CompletionError,
    CompletionResult,
    Message,
    TokenLogprob,
    Usage,
)

# Provider presets: base URL + which settings field holds the key.
PROVIDERS: dict[str, dict[str, str]] = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_attr": "openrouter_api_key",
    },
    "nebius": {
        "base_url": "https://api.studio.nebius.com/v1",
        "key_attr": "nebius_api_key",
    },
}


class LLMClient:
    """One async client bound to a single provider.

    Concurrency is bounded by a CapacityLimiter shared across all calls
    made through this client, so batch helpers cannot exceed it.
    """

    def __init__(
        self,
        provider: str,
        *,
        max_concurrency: int = 8,
        api_key: str | None = None,
        http_client: httpx.AsyncClient | None = None,
        max_retries: int = 2,
    ) -> None:
        if provider not in PROVIDERS:
            raise ValueError(
                f"Unknown provider {provider!r}; "
                f"expected one of {sorted(PROVIDERS)}"
            )
        preset = PROVIDERS[provider]
        if api_key is None:
            api_key = getattr(get_settings(), preset["key_attr"])
        if not api_key:
            raise ValueError(
                f"No API key for provider {provider!r}. Set "
                f"{preset['key_attr'].upper()} in the environment or .env."
            )
        self.provider = provider
        self._client = AsyncOpenAI(
            base_url=preset["base_url"],
            api_key=api_key,
            # The SDK aliases httpx as `httpx2` internally, so the type
            # checker sees a spurious mismatch on an identical class.
            http_client=http_client,  # ty: ignore[invalid-argument-type]
            max_retries=max_retries,
        )
        self._limiter = anyio.CapacityLimiter(max_concurrency)
        self._usage: dict[str, Usage] = {}

    @property
    def usage(self) -> dict[str, Usage]:
        """Accumulated token usage per model, this client's lifetime."""
        return dict(self._usage)

    def _record(self, model: str, usage: Usage) -> None:
        self._usage[model] = self._usage.get(model, Usage()) + usage

    async def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        top_logprobs: int | None = None,
        extra_body: dict | None = None,
    ) -> Completion:
        """Single chat completion. Raises on API error.

        `top_logprobs` requests per-token logprobs with that many
        alternatives. `extra_body` passes provider-specific fields
        (e.g. OpenRouter provider routing, Nebius prefill options).
        """
        async with self._limiter:
            resp = await self._client.chat.completions.create(
                model=model,
                messages=cast("list[ChatCompletionMessageParam]", messages),
                temperature=temperature,
                max_tokens=max_tokens,
                logprobs=top_logprobs is not None,
                top_logprobs=top_logprobs,
                extra_body=extra_body,
            )
        choice = resp.choices[0]
        usage = Usage()
        if resp.usage is not None:
            usage = Usage(
                prompt_tokens=resp.usage.prompt_tokens,
                completion_tokens=resp.usage.completion_tokens,
            )
        self._record(model, usage)
        return Completion(
            text=choice.message.content or "",
            model=resp.model or model,
            finish_reason=choice.finish_reason,
            usage=usage,
            logprobs=_parse_logprobs(choice),
        )

    async def complete_safe(
        self, messages: list[Message], *, model: str, **kwargs
    ) -> CompletionResult:
        """Like `complete`, but returns a CompletionError instead of
        raising — for use inside batches where one failure must not sink
        the rest."""
        try:
            return await self.complete(messages, model=model, **kwargs)
        except OpenAIError as exc:
            return CompletionError(
                error_type=type(exc).__name__,
                message=str(exc),
                model=model,
            )

    async def complete_many(
        self,
        requests: list[list[Message]],
        *,
        model: str,
        **kwargs,
    ) -> list[CompletionResult]:
        """Run many completions concurrently (bounded by the client's
        limiter). Order is preserved; failures are CompletionError
        records in place, never dropped."""
        async with asyncer.create_task_group() as tg:
            pending = [
                tg.soonify(self.complete_safe)(msgs, model=model, **kwargs)
                for msgs in requests
            ]
        return [p.value for p in pending]

    async def aclose(self) -> None:
        await self._client.close()


def _parse_logprobs(choice) -> list[TokenLogprob] | None:
    lp = getattr(choice, "logprobs", None)
    if lp is None or getattr(lp, "content", None) is None:
        return None
    out: list[TokenLogprob] = []
    for tok in lp.content:
        top = {
            alt.token: alt.logprob
            for alt in (getattr(tok, "top_logprobs", None) or [])
        }
        out.append(TokenLogprob(token=tok.token, logprob=tok.logprob, top=top))
    return out
