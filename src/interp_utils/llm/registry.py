"""Model registry: logical name -> serving spec, so research code says
`get_model("gemma-3-27b")` and never thinks about provider / model id /
backend pin again.

The specs encode *serving facts* (where a model is available and what it
can do), never research decisions about which model to use. Capability
facts are empirical and drift, so each carries `verified_on` — None means
reported (e.g. from a reference codebase), not checked by us.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .client import LLMClient
from .models import Completion, CompletionResult, Message


@dataclass(frozen=True)
class ModelSpec:
    name: str  # logical name (the registry key)
    provider: str  # which LLMClient: "openrouter" | "nebius"
    model_id: str  # the provider's id string
    # How the model emits its CoT (parsing hint): "harmony" (gpt-oss
    # channel markers), "think_tags" (inline <think>...</think>), "field"
    # (a separate reasoning API field on chat), or "none".
    reasoning_format: Literal["harmony", "think_tags", "field", "none"] = "none"
    # Supports echo/prompt-token logprob scoring on the text path.
    echo: bool = False
    # Generation-logprob coverage: "none" (provider returns no logprobs),
    # "content" (only the final content channel — reasoning tokens are
    # excluded), "full" (all generated tokens, incl. reasoning). None =
    # coverage unknown / not verified.
    logprobs: Literal["none", "content", "full"] | None = None
    # OpenRouter backend order to pin (e.g. to force a backend that
    # returns full-stream logprobs); injected into extra_body.
    backend_pin: tuple[str, ...] | None = None
    # Date every capability field was empirically confirmed BY US. The
    # bar is strict: if `logprobs` claims coverage, verified means we
    # checked that logprobs actually come back for reasoning tokens too
    # (when the model reasons). None = reported or only partially checked.
    verified_on: str | None = None
    notes: str = ""


# Seed registry. Extend as models are brought into play; promote entries
# to verified_on by actually checking them.
MODELS: dict[str, ModelSpec] = {
    # --- verified by us (2026-08-28) ---
    "gemma-3-27b": ModelSpec(
        name="gemma-3-27b",
        provider="nebius",
        model_id="google/gemma-3-27b-it",
        reasoning_format="none",
        echo=True,
        logprobs="full",  # non-reasoning: all generated tokens covered
        verified_on="2026-08-28",
        notes="echo-logprobs (max top-20) + text continuation verified.",
    ),
    "gpt-oss-120b": ModelSpec(
        name="gpt-oss-120b",
        provider="nebius",
        model_id="openai/gpt-oss-120b",
        reasoning_format="harmony",
        echo=True,
        logprobs="full",  # generation logprobs cover reasoning tokens
        verified_on="2026-08-28",
        notes=(
            "text /v1/completions: prefill, echo, and generation logprobs "
            "over reasoning (harmony) tokens all verified. Harmony control "
            "tokens flow as raw text on the completions path."
        ),
    ),
    "deepseek-v4-flash": ModelSpec(
        name="deepseek-v4-flash",
        provider="nebius",
        model_id="deepseek-ai/DeepSeek-V4-Flash-0731",
        reasoning_format="field",
        echo=True,
        logprobs="full",  # chat logprobs cover the reasoning stream
        verified_on="2026-08-28",
        notes=(
            "chat: reasoning in separate field, but chat logprobs cover "
            "the full reasoning stream (verified). echo works but requires "
            "max_tokens>=1 (Nebius rejects max_tokens=0 for this model). "
            "text /v1/completions prefill works."
        ),
    ),
    "gpt-oss-20b": ModelSpec(
        name="gpt-oss-20b",
        provider="openrouter",
        model_id="openai/gpt-oss-20b",
        reasoning_format="harmony",
        echo=False,  # OpenRouter has no echo/prompt-token scoring
        logprobs="full",  # generation logprobs cover the harmony reasoning
        backend_pin=("darkbloom/fp8",),
        verified_on="2026-08-28",
        notes=(
            "chat generation logprobs over the full harmony stream (incl. "
            "reasoning tokens) verified with the 'darkbloom/fp8' backend "
            "pin. Other backends differ: 'novita/fp4' gives content-only. "
            "The pin forces darkbloom/fp8 (allow_fallbacks off) for "
            "reproducible coverage. echo/prompt-token scoring is not "
            "available on OpenRouter. Not on a public Nebius endpoint."
        ),
    ),
    # OpenRouter <think>-style reasoning models: chat delivers the CoT in
    # the separate `reasoning` field and logprobs cover CONTENT ONLY. The
    # text/completions path is NOT served on OR for these, so
    # reasoning-token logprobs are not accessible via OpenRouter at all.
    "inkling-small": ModelSpec(
        name="inkling-small",
        provider="openrouter",
        model_id="thinkingmachines/inkling-small",
        reasoning_format="field",
        echo=False,
        logprobs="content",
        backend_pin=("together",),
        verified_on="2026-08-28",
        notes=(
            "chat content-only logprobs via 'together' (reasoning in the "
            "separate field, excluded from logprobs). /v1/completions not "
            "served on OR (422)."
        ),
    ),
    "qwen3.5-9b": ModelSpec(
        name="qwen3.5-9b",
        provider="openrouter",
        model_id="qwen/qwen3.5-9b",
        reasoning_format="field",
        echo=False,
        logprobs="content",
        backend_pin=("venice/fp8",),
        verified_on="2026-08-28",
        notes=(
            "chat content-only logprobs only via 'venice/fp8' (and even "
            "there minimal — final-answer tokens); 'together' and "
            "'parasail/bf16' return no logprobs. /v1/completions not served "
            "on OR."
        ),
    ),
    "qwen3.5-27b": ModelSpec(
        name="qwen3.5-27b",
        provider="openrouter",
        model_id="qwen/qwen3.5-27b",
        reasoning_format="field",
        echo=False,
        logprobs="content",
        backend_pin=("alibaba",),
        verified_on="2026-08-28",
        notes=(
            "chat content-only logprobs via 'alibaba' or 'novita/bf16' "
            "('phala' returns none). /v1/completions not served on OR."
        ),
    ),
    "qwen3.6-27b": ModelSpec(
        name="qwen3.6-27b",
        provider="openrouter",
        model_id="qwen/qwen3.6-27b",
        reasoning_format="field",
        echo=False,
        logprobs="content",
        backend_pin=("alibaba",),
        verified_on="2026-08-28",
        notes=(
            "chat content-only logprobs via 'alibaba' ('coreweave/fp8' "
            "returns none). /v1/completions not served on OR."
        ),
    ),
    "gemma-4-31b": ModelSpec(
        name="gemma-4-31b",
        provider="openrouter",
        model_id="google/gemma-4-31b-it",
        reasoning_format="field",
        echo=False,
        logprobs=None,
        verified_on="2026-09-03",
        notes=(
            "chat: reasoning in the separate field, final answer in content; "
            "reasoning effort set per-call via extra_body ('minimal' accepted). "
            "Serving + shape verified; slow route (verbose reasoning even at "
            "minimal). logprobs not checked."
        ),
    ),
}


class Model:
    """A registry model bound to a client. Auto-fills the provider model
    id and injects any backend pin into extra_body. Sampling (temperature,
    top_p, etc.) is a per-call choice, not a model property."""

    def __init__(self, spec: ModelSpec, client: LLMClient) -> None:
        self.spec = spec
        self.client = client

    def _prep(self, kwargs: dict) -> dict:
        out = dict(kwargs)
        if self.spec.backend_pin:
            extra = dict(out.get("extra_body") or {})
            extra.setdefault(
                "provider",
                {
                    "order": list(self.spec.backend_pin),
                    "allow_fallbacks": False,
                },
            )
            out["extra_body"] = extra
        return out

    async def complete(self, messages: list[Message], **kwargs) -> Completion:
        return await self.client.complete(
            messages, model=self.spec.model_id, **self._prep(kwargs)
        )

    async def complete_text(self, prompt: str, **kwargs) -> Completion:
        return await self.client.complete_text(
            prompt, model=self.spec.model_id, **self._prep(kwargs)
        )

    async def complete_many(
        self, requests: list[list[Message]], **kwargs
    ) -> list[CompletionResult]:
        return await self.client.complete_many(
            requests, model=self.spec.model_id, **self._prep(kwargs)
        )

    async def complete_text_many(
        self, prompts: list[str], **kwargs
    ) -> list[CompletionResult]:
        return await self.client.complete_text_many(
            prompts, model=self.spec.model_id, **self._prep(kwargs)
        )


def get_model(
    name: str,
    *,
    client: LLMClient | None = None,
    max_concurrency: int = 8,
) -> Model:
    """Resolve a registry model to a ready-to-call `Model`.

    Creates a client for the spec's provider, or reuse one you pass (e.g.
    to share a usage ledger across models on the same provider — the
    passed client's provider must match the spec's).
    """
    if name not in MODELS:
        raise KeyError(f"Unknown model {name!r}; known: {sorted(MODELS)}")
    spec = MODELS[name]
    if client is None:
        client = LLMClient(spec.provider, max_concurrency=max_concurrency)
    elif client.provider != spec.provider:
        raise ValueError(
            f"Model {name!r} is served by {spec.provider!r}, but the "
            f"passed client is for {client.provider!r}."
        )
    return Model(spec, client)
