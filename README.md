# interp-utils

Utils for LLM experimentation.

- **`interp_utils.llm`** — one async client over OpenAI-compatible
  providers (OpenRouter, Nebius): bounded concurrency, explicit failure
  records, per-model usage accounting, logprobs, and
  continue-from-prefill for resampling workflows.

## Setup

```bash
uv sync
cp .env.example .env  # fill in API keys
uv run pytest         # offline test suite
```

See `AGENTS.md` for commands, architecture, and conventions.
