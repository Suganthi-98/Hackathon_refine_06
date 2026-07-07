"""
AI layer configuration — provider-agnostic.

Supports two providers, selected by AI_PROVIDER:

  "anthropic"  (original)   — uses ANTHROPIC_API_KEY + Anthropic SDK
  "bosch"      (default)    — uses Bosch LLM Farm OpenAI-compatible endpoint

.env keys
---------
AI_PROVIDER             "bosch" | "anthropic"      default: bosch

# Bosch LLM Farm  (used when AI_PROVIDER=bosch)
# Credential names match the environment variables distributed by Bosch:
#
#   set AOAI_FARM_API_KEY=<your key>
#   set AOAI_FARM_DOMAIN=https://aoai-farm.bosch-temp.com
#   set AOAI_MODEL=askbosch-prod-farm-openai-gpt-4o-mini-2024-07-18
#   set AOAI_API_VERSION=2024-08-01-preview
#   set AOAI_FARM_SUBSCRIPTION_ID=personal-ovu1cob-prod
#   set PX_PROXY_URL=http://127.0.0.1:3128
#
AOAI_FARM_API_KEY           required for bosch
AOAI_FARM_DOMAIN            default: https://aoai-farm.bosch-temp.com
AOAI_MODEL                  default: askbosch-prod-farm-openai-gpt-4o-mini-2024-07-18
AOAI_API_VERSION            default: 2024-08-01-preview
AOAI_FARM_SUBSCRIPTION_ID   subscription routing header (optional)
PX_PROXY_URL                HTTP proxy for egress (optional, e.g. http://127.0.0.1:3128)

# Anthropic (used when AI_PROVIDER=anthropic)
ANTHROPIC_API_KEY           required for anthropic

# Shared inference settings (apply to both providers)
AI_TEMPERATURE              default: 0.2
AI_TIMEOUT                  default: 30.0  (seconds — LLM Farm can be slow)
AI_MAX_TOKENS               default: 1024

# Feature flags
AI_ADVISOR_ENABLED          default: true
AI_CACHE_ENABLED            default: true

Usage
-----
    from app.ai.config import ai_settings
    from app.ai.client import build_client

    client = build_client(ai_settings)   # returns BoschClient or ClaudeClient
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings
from pathlib import Path
import os


class AISettings(BaseSettings):
    """
    Provider-agnostic AI infrastructure settings.

    Kept separate from app.core.config.Settings so the AI layer can be
    swapped or disabled without touching business-logic configuration.
    """

    # ─── Provider selector ───────────────────────────────────────────────────
    ai_provider: Literal["bosch", "anthropic"] = "bosch"

    # ─── Bosch LLM Farm ──────────────────────────────────────────────────────
    # Env-var names match Bosch's own credential distribution exactly.
    aoai_farm_api_key: str = Field(
        "", env=("AOAI_FARM_API_KEY", "BOSCH_API_KEY")
    )
    aoai_farm_domain: str = Field(
        "https://aoai-farm.bosch-temp.com", env=("AOAI_FARM_DOMAIN", "BOSCH_ENDPOINT")
    )
    aoai_model: str = Field(
        "askbosch-prod-farm-openai-gpt-4o-mini-2024-07-18",
        env=("AOAI_MODEL", "BOSCH_DEPLOYMENT", "AI_MODEL"),
    )
    aoai_api_version: str = Field(
        "2024-08-01-preview", env=("AOAI_API_VERSION", "BOSCH_API_VERSION")
    )
    aoai_farm_subscription_id: str = Field(
        "", env=("AOAI_FARM_SUBSCRIPTION_ID", "BOSCH_FARM_SUBSCRIPTION_ID")
    )   # subscription routing header (optional)
    px_proxy_url: Optional[str] = Field(None, env=("PX_PROXY_URL",))    # e.g. "http://127.0.0.1:3128"

    # ─── Anthropic (kept for fallback / local dev) ───────────────────────────
    anthropic_api_key: str = ""

    # ─── Shared inference ────────────────────────────────────────────────────
    ai_temperature: float = 0.2
    ai_timeout: float = 30.0     # LLM Farm response times can exceed 8 s
    ai_max_tokens: int = 1024

    # ─── Feature flags ───────────────────────────────────────────────────────
    ai_advisor_enabled: bool = True
    ai_cache_enabled: bool = True

    def __init__(self, _env_file: Optional[str] = None, **values):
        # If an env file path was provided, parse it for legacy BOSCH_* keys
        # and map them to the current AOAI_* field names before validation.
        if _env_file:
            try:
                text = Path(_env_file).read_text()
            except Exception:
                text = None

            if text:
                for line in text.splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip().upper()
                    v = v.strip()

                    if k == "BOSCH_API_KEY":
                        values.setdefault("aoai_farm_api_key", v)
                    elif k == "BOSCH_ENDPOINT":
                        values.setdefault("aoai_farm_domain", v)
                    elif k == "BOSCH_DEPLOYMENT":
                        values.setdefault("aoai_model", v)
                    elif k == "BOSCH_API_VERSION":
                        values.setdefault("aoai_api_version", v)
                    elif k == "BOSCH_FARM_SUBSCRIPTION_ID":
                        values.setdefault("aoai_farm_subscription_id", v)
                    elif k == "AI_MODEL":
                        values.setdefault("aoai_model", v)
                    elif k == "PX_PROXY_URL":
                        values.setdefault("px_proxy_url", v)
                    elif k == "AI_PROVIDER":
                        values.setdefault("ai_provider", v)
                    elif k == "AI_ADVISOR_ENABLED":
                        values.setdefault("ai_advisor_enabled", v)
                    elif k == "AI_CACHE_ENABLED":
                        values.setdefault("ai_cache_enabled", v)

        super().__init__(**values)

    @property
    def bosch_chat_url(self) -> str:
        """
        Full Chat Completions URL for the configured Bosch LLM Farm deployment.

        Pattern:
            {AOAI_FARM_DOMAIN}/api/openai/deployments/
            {AOAI_MODEL}/chat/completions?api-version={AOAI_API_VERSION}

        Example:
            https://aoai-farm.bosch-temp.com/api/openai/deployments/
            askbosch-prod-farm-openai-gpt-4o-mini-2024-07-18/
            chat/completions?api-version=2024-08-01-preview
        """
        return (
            f"{self.aoai_farm_domain}/api/openai/deployments"
            f"/{self.aoai_model}/chat/completions"
            f"?api-version={self.aoai_api_version}"
        )

    # --- Backwards-compatible aliases for older env names / callers ---
    @property
    def bosch_api_key(self) -> str:
        return self.aoai_farm_api_key

    @property
    def bosch_endpoint(self) -> str:
        return self.aoai_farm_domain

    @property
    def bosch_deployment(self) -> str:
        return self.aoai_model

    @property
    def bosch_api_version(self) -> str:
        return self.aoai_api_version

    @property
    def bosch_farm_subscription_id(self) -> str:
        return self.aoai_farm_subscription_id

    @property
    def ai_model(self) -> str:
        return self.aoai_model

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


# Module-level singleton — import this everywhere instead of constructing
# a new instance per request.
ai_settings = AISettings()
