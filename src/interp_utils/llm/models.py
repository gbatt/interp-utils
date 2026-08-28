"""Data models for completion requests and results."""

from typing import Any, Literal

from pydantic import BaseModel

Message = dict[str, Any]
"""Chat message: {"role": ..., "content": ...}. Kept as a dict to stay
transparent to provider-specific fields."""


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


class TokenLogprob(BaseModel):
    """Logprob for one token, with top alternatives if requested. Shared
    by chat (`complete`) and text (`complete_text`) so downstream code is
    backend-agnostic.

    `logprob` and `top` are None for a position the provider gives neither
    (e.g. the first echoed prompt token, which has no preceding context).
    `text_offset` — the token's character offset in the prompt+completion
    string — is populated only on the text path (load-bearing for slicing
    a target sentence out of an echoed prompt); it is None for chat.
    """

    token: str
    logprob: float | None
    top: dict[str, float] | None = None
    text_offset: int | None = None


class Completion(BaseModel):
    """One completion result, from either endpoint. `kind` records which:
    `"chat"` (from `complete`) or `"text"` (from `complete_text`). The two
    differ only in `reasoning` — the chat path may carry a separate CoT
    trace; the text path's CoT is inline in `text` (e.g. <think>...), so
    `reasoning` is None there."""

    kind: Literal["chat", "text"]
    text: str
    # The requested model — the stable key to group results by. The
    # provider-resolved variant (e.g. OpenRouter routing) is deliberately
    # not stored here to avoid requested/served mismatches in analysis.
    model: str
    # Separate reasoning/CoT trace, when the provider returns one in a
    # dedicated field (OpenRouter: `reasoning`, Nebius/DeepSeek:
    # `reasoning_content`). None on the text path and for non-reasoning
    # models, or when CoT is emitted inline in `text`.
    reasoning: str | None = None
    finish_reason: str | None = None
    usage: Usage = Usage()
    logprobs: list[TokenLogprob] | None = None


class CompletionError(BaseModel):
    """Explicit failure record for a batch item. Never silently dropped:
    sample loss that correlates with content is a bias."""

    kind: Literal["error"] = "error"
    error_type: str
    message: str
    model: str


CompletionResult = Completion | CompletionError
