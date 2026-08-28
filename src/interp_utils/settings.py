"""Environment-backed settings. Keys come from env vars or a local .env."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openrouter_api_key: str | None = None
    nebius_api_key: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
