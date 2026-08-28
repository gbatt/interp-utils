# interp-utils

Async utilities for black-box LLM experimentation: provider clients,
resampling, judging. Generic infrastructure only — no experiment-specific
code belongs here.

## Commands

All through uv:

- `uv run pytest` — test suite (fully mocked; no network)
- `INTERP_UTILS_LIVE=1 uv run pytest tests/test_live.py` — live smoke
  tests (needs API keys in `.env`; spends real tokens)
- `uv run ruff check` / `uv run ruff format` — lint / format
- `uv run ty check` — type check
- `uv run prek run --all-files` — all pre-commit hooks

## Architecture

- `src/interp_utils/settings.py` — pydantic-settings; API keys from env
  or `.env` (see `.env.example`)
- `src/interp_utils/llm/` — provider-agnostic async completion client.
  One `LLMClient` over the OpenAI SDK; providers (OpenRouter, Nebius)
  are base-URL + key presets. Bounded concurrency via anyio
  `CapacityLimiter`; per-model token-usage accounting on the client.
- Planned (do not build speculatively): `judge/` (rubric autorater with
  N-vote aggregation and calibration export), `resample/` (fix prefix →
  resample N → judge → aggregate), `cot/` (trace parsing/chunking).

## Conventions

- Async-first. Sync wrappers only at the notebook boundary, if at all.
- Batch APIs return explicit failure records (`CompletionError`), never
  silently dropped items. Sample loss that correlates with content is a
  bias; keep it visible.
- Deterministic calls (temperature-0 judging) may be cached when a cache
  exists; stochastic sampling calls are never cached.
- Live-network tests are gated behind `INTERP_UTILS_LIVE=1`; the default
  suite must pass offline.
- Keep the dependency surface small, but prefer a well-maintained
  library over reimplementing nontrivial behavior here.

## Public-repo discipline

This repo is public. No secrets, no personal context, no
project-specific research details in code, comments, or docs — keep
everything here generic infrastructure. This applies to this file too.
