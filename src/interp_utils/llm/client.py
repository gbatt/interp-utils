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
from openai.types import CompletionChoice
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion import Choice

from interp_utils.settings import get_settings

from .models import (
    Completion,
    CompletionError,
    CompletionResult,
    Message,
    TokenLogprob,
    Usage,
)

# Default generation cap for raw text completions. With `max_tokens=None`
# the provider generates until the context is full — a real cost footgun
# for reasoning models prefilled with an open <think> block. This bounds
# it at the full-CoT budget the reference resampling pipelines use (16k);
# override per call for longer generations, or set 0 for echo-only
# logprob scoring.
DEFAULT_TEXT_MAX_TOKENS = 16384

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
            kind="chat",
            text=choice.message.content or "",
            model=model,
            reasoning=_parse_reasoning(choice),
            finish_reason=choice.finish_reason,
            usage=usage,
            logprobs=_parse_logprobs(choice),
        )

    async def complete_text(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int | None = DEFAULT_TEXT_MAX_TOKENS,
        logprobs: int | None = None,
        echo: bool = False,
        stop: str | list[str] | None = None,
        extra_body: dict | None = None,
    ) -> Completion:
        """Single raw text completion (/v1/completions). Raises on API
        error.

        The prefix for fixed-prefix CoT resampling is baked directly into
        `prompt` (e.g. "...Assistant:\\n<think>\\n{prefix}"); the returned
        `text` is the continuation. `logprobs=N` returns per-token
        logprobs with N alternatives; `echo=True` with `max_tokens=0`
        returns the prompt tokens' own logprobs for sentence-likelihood
        scoring. `extra_body` passes provider-specific params (e.g. Nebius
        logprob-shape quirks).

        `max_tokens` defaults to a finite cap (see DEFAULT_TEXT_MAX_TOKENS)
        rather than None, since None lets the provider generate until the
        context fills. Pass an explicit value (including 0) to override.
        """
        async with self._limiter:
            resp = await self._client.completions.create(
                model=model,
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                logprobs=logprobs,
                echo=echo,
                stop=stop,
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
            kind="text",
            text=choice.text,
            model=model,
            finish_reason=choice.finish_reason,
            usage=usage,
            logprobs=_parse_text_logprobs(choice),
        )

    @staticmethod
    def _error(exc: OpenAIError, model: str) -> CompletionError:
        return CompletionError(
            error_type=type(exc).__name__, message=str(exc), model=model
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
            return self._error(exc, model)

    async def complete_text_safe(
        self, prompt: str, *, model: str, **kwargs
    ) -> CompletionResult:
        """Non-raising variant of `complete_text` for use in batches."""
        try:
            return await self.complete_text(prompt, model=model, **kwargs)
        except OpenAIError as exc:
            return self._error(exc, model)

    async def _gather(self, safe_method, items, model, kwargs):
        """Run `safe_method(item, model=model, **kwargs)` over items
        concurrently (bounded by the limiter). Order preserved; the safe
        methods turn failures into CompletionError records in place."""
        async with asyncer.create_task_group() as tg:
            pending = [
                tg.soonify(safe_method)(item, model=model, **kwargs)
                for item in items
            ]
        return [p.value for p in pending]

    async def complete_many(
        self,
        requests: list[list[Message]],
        *,
        model: str,
        **kwargs,
    ) -> list[CompletionResult]:
        """Run many chat completions concurrently. Order is preserved;
        failures are CompletionError records in place, never dropped."""
        return await self._gather(self.complete_safe, requests, model, kwargs)

    async def complete_text_many(
        self,
        prompts: list[str],
        *,
        model: str,
        **kwargs,
    ) -> list[CompletionResult]:
        """Run many text completions concurrently. Order is preserved;
        failures are CompletionError records in place, never dropped."""
        return await self._gather(
            self.complete_text_safe, prompts, model, kwargs
        )

    async def aclose(self) -> None:
        await self._client.close()


# Provider field names carrying a separate reasoning/CoT trace, in
# priority order. Not part of the OpenAI SDK message type, so they arrive
# as pydantic model extras.
_REASONING_KEYS = ("reasoning", "reasoning_content")


def _parse_reasoning(choice: Choice) -> str | None:
    extra = choice.message.model_extra or {}
    for key in _REASONING_KEYS:
        value = extra.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _parse_logprobs(choice: Choice) -> list[TokenLogprob] | None:
    lp = choice.logprobs
    if lp is None or lp.content is None:
        return None
    out: list[TokenLogprob] = []
    for tok in lp.content:
        top = {alt.token: alt.logprob for alt in tok.top_logprobs}
        out.append(TokenLogprob(token=tok.token, logprob=tok.logprob, top=top))
    return out


def _parse_text_logprobs(
    choice: CompletionChoice,
) -> list[TokenLogprob] | None:
    lp = choice.logprobs
    if lp is None or not lp.tokens:
        return None
    token_logprobs = lp.token_logprobs or []
    top_logprobs = lp.top_logprobs or []
    text_offset = lp.text_offset or []
    out: list[TokenLogprob] = []
    for i, token in enumerate(lp.tokens):
        top = top_logprobs[i] if i < len(top_logprobs) else None
        out.append(
            TokenLogprob(
                token=token,
                logprob=token_logprobs[i] if i < len(token_logprobs) else None,
                top=dict(top) if top is not None else None,
                text_offset=text_offset[i] if i < len(text_offset) else None,
            )
        )
    return out
