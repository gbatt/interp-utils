"""Unit tests for the model registry and Model wrapper (fully mocked)."""

import json

import httpx
import pytest

from interp_utils import Model, get_model
from interp_utils.llm import LLMClient, ModelSpec
from interp_utils.llm.registry import MODELS


def _mock_client(provider: str, handler) -> LLMClient:
    return LLMClient(
        provider,
        api_key="test-key",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )


def _text_body(text: str = "ok") -> dict:
    return {
        "id": "x",
        "object": "text_completion",
        "model": "served-id",
        "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


def _spy_model(spec: ModelSpec, seen: dict) -> Model:
    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json=_text_body())

    return Model(spec, _mock_client(spec.provider, handler))


def test_unknown_model_rejected():
    with pytest.raises(KeyError, match="Unknown model"):
        get_model(
            "no-such-model", client=_mock_client("nebius", lambda r: None)
        )


def test_seed_models_are_self_consistent():
    for name, spec in MODELS.items():
        assert spec.name == name
        assert spec.provider in ("openrouter", "nebius")
        assert spec.model_id


def test_verified_entries_have_known_logprob_coverage():
    # The strict semantic: a verified spec has confirmed its logprob
    # coverage (never left unknown). Unverified entries may be None.
    for spec in MODELS.values():
        if spec.verified_on is not None:
            assert spec.logprobs in ("none", "content", "full"), (
                f"{spec.name} is verified but logprobs coverage is unknown"
            )


def test_client_provider_mismatch_rejected():
    or_client = _mock_client("openrouter", lambda r: None)
    # gemma-3-27b is a nebius model.
    with pytest.raises(ValueError, match="served by 'nebius'"):
        get_model("gemma-3-27b", client=or_client)


async def test_model_id_is_sent_not_logical_name():
    spec = MODELS["gemma-3-27b"]
    seen: dict = {}
    m = _spy_model(spec, seen)
    await m.complete_text("hi")
    assert seen["body"]["model"] == "google/gemma-3-27b-it"


async def test_caller_sampling_passes_through():
    # Sampling is a per-call choice; the wrapper just forwards it.
    spec = ModelSpec(name="x", provider="nebius", model_id="id")
    seen: dict = {}
    m = _spy_model(spec, seen)
    await m.complete_text("hi", temperature=0.3, top_p=0.9)
    assert seen["body"]["temperature"] == 0.3
    assert seen["body"]["top_p"] == 0.9


async def test_backend_pin_injected_into_extra_body():
    spec = ModelSpec(
        name="x",
        provider="openrouter",
        model_id="id",
        backend_pin=("darkbloom/fp8",),
    )
    seen: dict = {}
    m = _spy_model(spec, seen)
    await m.complete_text("hi")
    provider = seen["body"]["provider"]
    assert provider["order"] == ["darkbloom/fp8"]
    assert provider["allow_fallbacks"] is False
