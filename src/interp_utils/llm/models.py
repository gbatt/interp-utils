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
    """Logprob for one sampled token, with top alternatives if requested."""

    token: str
    logprob: float
    top: dict[str, float] = {}


class Completion(BaseModel):
    kind: Literal["completion"] = "completion"
    text: str
    model: str
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
