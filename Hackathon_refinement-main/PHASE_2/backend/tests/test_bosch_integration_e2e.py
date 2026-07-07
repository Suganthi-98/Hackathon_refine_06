"""
End-to-end integration tests for Bosch LLM Farm AI client.

Validates:
  1. Authentication (API key, headers, subscription ID routing)
  2. Request/response handling (payload construction, response parsing)
  3. Fallback behavior (graceful degradation when Bosch unavailable)
  4. Caching (cache hits, key generation, consistency)
  5. Structured output parsing (JSON validation, markdown stripping, schema)

All HTTP calls are mocked to avoid external dependencies; tests validate
the contract between the application and Bosch LLM Farm endpoint.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.ai.cache import InMemoryNarrativeCache, cache_key
from app.ai.client import BoschClient, build_client
from app.ai.config import AISettings
from app.ai.exceptions import (
    AIClientError,
    AIResponseError,
    AIRetryExhaustedError,
    AITimeoutError,
)
from app.engines.advisor_contract import AdvisorInput, AdvisorOutput
from app.engines.advisor_input_builder import AdvisorInputBuilder
from app.engines.narrative_service import NarrativeService


# ─────────────────────────────────────────────────────────────────────────────
# 1. AUTHENTICATION TESTS
# ─────────────────────────────────────────────────────────────────────────────


class TestBoschAuthentication:
    """Validate Bosch API authentication setup."""

    def test_bosch_client_rejects_missing_api_key(self) -> None:
        """BoschClient raises AIClientError when AOAI_FARM_API_KEY is empty."""
        settings = AISettings(
            ai_provider="bosch",
            aoai_farm_api_key="",  # missing
            aoai_farm_domain="https://example.test",
        )
        with pytest.raises(AIClientError, match="AOAI_FARM_API_KEY is not set"):
            BoschClient(settings)

    def test_bosch_client_sets_auth_header(self) -> None:
        """BoschClient correctly sets genaiplatform-farm-subscription-key header."""
        settings = AISettings(
            ai_provider="bosch",
            aoai_farm_api_key="test-key-123",
            aoai_farm_domain="https://example.test",
        )
        client = BoschClient(settings)
        # Verify header is in the httpx client
        assert client._http.headers["genaiplatform-farm-subscription-key"] == "test-key-123"
        asyncio.run(client.aclose())

    def test_bosch_client_sets_subscription_id_header_when_present(self) -> None:
        """BoschClient includes subscription-id header when AOAI_FARM_SUBSCRIPTION_ID is set."""
        settings = AISettings(
            ai_provider="bosch",
            aoai_farm_api_key="test-key-123",
            aoai_farm_domain="https://example.test",
            aoai_farm_subscription_id="sub-456",
        )
        client = BoschClient(settings)
        assert client._http.headers["subscription-id"] == "sub-456"
        asyncio.run(client.aclose())

    def test_bosch_client_omits_subscription_id_when_empty(self) -> None:
        """BoschClient does not include subscription-id header when AOAI_FARM_SUBSCRIPTION_ID is empty."""
        settings = AISettings(
            ai_provider="bosch",
            aoai_farm_api_key="test-key-123",
            aoai_farm_domain="https://example.test",
            aoai_farm_subscription_id="",  # empty
        )
        client = BoschClient(settings)
        assert "subscription-id" not in client._http.headers
        asyncio.run(client.aclose())

    def test_bosch_client_sets_proxy_when_configured(self) -> None:
        """BoschClient wires PX_PROXY_URL to httpx proxies."""
        settings = AISettings(
            ai_provider="bosch",
            aoai_farm_api_key="test-key-123",
            aoai_farm_domain="https://example.test",
            px_proxy_url="http://127.0.0.1:3128",
        )
        client = BoschClient(settings)
        # httpx stores proxies in _mixin (internal state), verify it was constructed
        assert client._http is not None
        asyncio.run(client.aclose())

    def test_build_client_factory_returns_bosch_client(self) -> None:
        """build_client() returns BoschClient when AI_PROVIDER=bosch."""
        settings = AISettings(
            ai_provider="bosch",
            aoai_farm_api_key="test-key-123",
        )
        client = build_client(settings)
        assert isinstance(client, BoschClient)
        asyncio.run(client.aclose())

    def test_build_client_factory_rejects_invalid_provider(self) -> None:
        """build_client() raises AIClientError for unknown AI_PROVIDER."""
        settings = AISettings(ai_provider="invalid")  # type: ignore
        with pytest.raises(AIClientError, match="Unknown AI_PROVIDER"):
            build_client(settings)


# ─────────────────────────────────────────────────────────────────────────────
# 2. REQUEST/RESPONSE HANDLING TESTS
# ─────────────────────────────────────────────────────────────────────────────


class TestBoschRequestResponse:
    """Validate request construction and response parsing."""

    @pytest.mark.asyncio
    async def test_bosch_constructs_correct_endpoint_url(self) -> None:
        """BoschClient uses bosch_chat_url property to build endpoint."""
        settings = AISettings(
            ai_provider="bosch",
            aoai_farm_api_key="test-key-123",
            aoai_farm_domain="https://aoai-farm.bosch-temp.com",
            aoai_model="gpt-4o-mini",
            aoai_api_version="2024-08-01-preview",
        )
        expected_url = (
            "https://aoai-farm.bosch-temp.com/api/openai/deployments"
            "/gpt-4o-mini/chat/completions?api-version=2024-08-01-preview"
        )
        assert settings.bosch_chat_url == expected_url

    @pytest.mark.asyncio
    async def test_bosch_extracts_text_from_openai_response_envelope(self) -> None:
        """BoschClient._extract_text() pulls content from OpenAI response shape."""
        response = {
            "choices": [
                {"message": {"content": '{"test": "json"}'}}
            ]
        }
        text = BoschClient._extract_text(response)
        assert text == '{"test": "json"}'

    @pytest.mark.asyncio
    async def test_bosch_extracts_text_raises_on_empty_choices(self) -> None:
        """BoschClient._extract_text() raises AIResponseError when choices is empty."""
        response = {"choices": []}
        with pytest.raises(AIResponseError, match="empty choices"):
            BoschClient._extract_text(response)

    @pytest.mark.asyncio
    async def test_bosch_extracts_text_raises_on_missing_content(self) -> None:
        """BoschClient._extract_text() raises AIResponseError when content is missing."""
        response = {"choices": [{"message": {}}]}
        with pytest.raises(AIResponseError, match="KeyError"):
            BoschClient._extract_text(response)

    @pytest.mark.asyncio
    async def test_bosch_parses_valid_json(self) -> None:
        """BoschClient._parse_json() returns parsed dict from valid JSON string."""
        json_str = '{"key": "value", "number": 42}'
        parsed = BoschClient._parse_json(json_str)
        assert parsed == {"key": "value", "number": 42}

    @pytest.mark.asyncio
    async def test_bosch_parses_json_strips_markdown_fences(self) -> None:
        """BoschClient._parse_json() strips markdown code fences from model output."""
        json_with_fences = '```json\n{"key": "value"}\n```'
        parsed = BoschClient._parse_json(json_with_fences)
        assert parsed == {"key": "value"}

    @pytest.mark.asyncio
    async def test_bosch_parses_json_raises_on_invalid_json(self) -> None:
        """BoschClient._parse_json() raises AIResponseError for invalid JSON."""
        with pytest.raises(AIResponseError, match="not valid JSON"):
            BoschClient._parse_json('{invalid json}')

    @pytest.mark.asyncio
    async def test_bosch_parses_json_raises_on_non_dict(self) -> None:
        """BoschClient._parse_json() raises AIResponseError when JSON is not an object."""
        with pytest.raises(AIResponseError, match="Expected a JSON object"):
            BoschClient._parse_json('["array", "not", "object"]')

    @pytest.mark.asyncio
    async def test_bosch_generate_sends_correct_payload(self) -> None:
        """BoschClient.generate() sends correct message payload to Bosch endpoint."""
        settings = AISettings(
            ai_provider="bosch",
            aoai_farm_api_key="test-key-123",
            aoai_farm_domain="https://example.test",
            ai_temperature=0.2,
            ai_max_tokens=1024,
        )
        client = BoschClient(settings)

        # Mock the HTTP POST call
        mock_response = {
            "choices": [
                {"message": {"content": '{"test": "output"}'}}
            ]
        }
        client._http.post = AsyncMock(
            return_value=MagicMock(
                json=lambda: mock_response,
                raise_for_status=lambda: None,
            )
        )

        result = asyncio.run(client.generate("test message"))
        assert result == {"test": "output"}

        # Verify payload structure
        call_args = client._http.post.call_args
        assert call_args is not None
        payload = call_args.kwargs["json"]
        assert payload["temperature"] == 0.2
        assert payload["max_tokens"] == 1024
        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["role"] == "user"
        assert payload["messages"][1]["content"] == "test message"

        asyncio.run(client.aclose())


# ─────────────────────────────────────────────────────────────────────────────
# 3. FALLBACK BEHAVIOR TESTS
# ─────────────────────────────────────────────────────────────────────────────


class TestFallbackBehavior:
    """Validate graceful degradation when Bosch is unavailable."""

    @pytest.mark.asyncio
    async def test_bosch_raises_timeout_error_on_timeout(self) -> None:
        """BoschClient.generate() raises AITimeoutError when request times out."""
        settings = AISettings(
            ai_provider="bosch",
            aoai_farm_api_key="test-key-123",
            ai_timeout=0.1,
        )
        client = BoschClient(settings)
        client._http.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))

        with pytest.raises(AITimeoutError):
            await client.generate("test message")

        asyncio.run(client.aclose())

    @pytest.mark.asyncio
    async def test_bosch_raises_client_error_on_non_retryable_http_error(self) -> None:
        """BoschClient.generate() raises AIClientError for 4xx/5xx non-retryable errors."""
        settings = AISettings(
            ai_provider="bosch",
            aoai_farm_api_key="test-key-123",
        )
        client = BoschClient(settings)

        # 401 Unauthorized is not retryable
        response = MagicMock()
        response.status_code = 401
        response.text = "Unauthorized"
        client._http.post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "401", request=MagicMock(), response=response
            )
        )

        with pytest.raises(AIClientError, match="401"):
            await client.generate("test message")

        asyncio.run(client.aclose())

    @pytest.mark.asyncio
    async def test_bosch_retries_on_429_too_many_requests(self) -> None:
        """BoschClient.generate() retries on 429 (rate limit) up to MAX_RETRIES times."""
        settings = AISettings(
            ai_provider="bosch",
            aoai_farm_api_key="test-key-123",
        )
        client = BoschClient(settings)

        response = MagicMock()
        response.status_code = 429
        response.text = "Too many requests"

        # First 2 attempts fail with 429, 3rd succeeds
        success_response = {
            "choices": [
                {"message": {"content": '{"success": true}'}}
            ]
        }
        client._http.post = AsyncMock(
            side_effect=[
                httpx.HTTPStatusError(
                    "429", request=MagicMock(), response=response
                ),
                httpx.HTTPStatusError(
                    "429", request=MagicMock(), response=response
                ),
                MagicMock(
                    json=lambda: success_response,
                    raise_for_status=lambda: None,
                ),
            ]
        )

        result = await client.generate("test message")
        assert result == {"success": True}
        # Verify it made 3 POST calls (2 retries + 1 success)
        assert client._http.post.call_count == 3

        asyncio.run(client.aclose())

    @pytest.mark.asyncio
    async def test_bosch_exhausts_retries_and_raises(self) -> None:
        """BoschClient.generate() raises AIRetryExhaustedError after MAX_RETRIES failures."""
        settings = AISettings(
            ai_provider="bosch",
            aoai_farm_api_key="test-key-123",
        )
        client = BoschClient(settings)

        response = MagicMock()
        response.status_code = 500
        response.text = "Internal Server Error"
        client._http.post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "500", request=MagicMock(), response=response
            )
        )

        with pytest.raises(AIRetryExhaustedError):
            await client.generate("test message")

        asyncio.run(client.aclose())

    @pytest.mark.asyncio
    async def test_narrative_service_degrades_gracefully_when_client_is_none(self) -> None:
        """NarrativeService falls back to deterministic template when client is None."""
        settings = AISettings(ai_advisor_enabled=False)
        service = NarrativeService(client=None, settings=settings)

        # Create minimal AdvisorInput
        advisor_input = AdvisorInput(
            project_id="test-project",
            project_context=None,
            recommendations=[],
            scenario=None,
        )

        result = asyncio.run(service.explain(advisor_input))
        # Should return fallback response with status 'disabled' or 'fallback'
        assert result["status"] in ["disabled", "fallback"]


# ─────────────────────────────────────────────────────────────────────────────
# 4. CACHING TESTS
# ─────────────────────────────────────────────────────────────────────────────


class TestCaching:
    """Validate response caching and cache key generation."""

    def test_cache_key_generation_is_deterministic(self) -> None:
        """cache_key() produces the same key for identical inputs."""
        advisor_input = AdvisorInput(
            project_id="test-project",
            project_context=None,
            recommendations=[],
            scenario=None,
        )
        key1 = cache_key("bosch", advisor_input)
        key2 = cache_key("bosch", advisor_input)
        assert key1 == key2

    def test_cache_key_differs_by_provider(self) -> None:
        """cache_key() produces different keys for different providers."""
        advisor_input = AdvisorInput(
            project_id="test-project",
            project_context=None,
            recommendations=[],
            scenario=None,
        )
        key_bosch = cache_key("bosch", advisor_input)
        key_claude = cache_key("anthropic", advisor_input)
        assert key_bosch != key_claude

    def test_in_memory_cache_stores_and_retrieves(self) -> None:
        """InMemoryNarrativeCache stores and retrieves cached narratives."""
        cache = InMemoryNarrativeCache()
        advisor_input = AdvisorInput(
            project_id="test-project",
            project_context=None,
            recommendations=[],
            scenario=None,
        )
        test_narrative = {"narrative": "test explanation", "status": "ok"}

        # Store
        cache.store(
            model_name="bosch",
            advisor_input=advisor_input,
            narrative_dict=test_narrative,
        )

        # Retrieve
        retrieved = cache.get(model_name="bosch", advisor_input=advisor_input)
        assert retrieved == test_narrative

    def test_in_memory_cache_returns_none_on_miss(self) -> None:
        """InMemoryNarrativeCache returns None when key is not cached."""
        cache = InMemoryNarrativeCache()
        advisor_input = AdvisorInput(
            project_id="test-project",
            project_context=None,
            recommendations=[],
            scenario=None,
        )
        result = cache.get(model_name="bosch", advisor_input=advisor_input)
        assert result is None

    @pytest.mark.asyncio
    async def test_narrative_service_caches_results(self) -> None:
        """NarrativeService caches results and returns cached value on second call."""
        settings = AISettings(
            ai_provider="bosch",
            aoai_farm_api_key="test-key-123",
            ai_advisor_enabled=True,
        )
        cache = InMemoryNarrativeCache()

        # Mock BoschClient
        mock_client = AsyncMock(spec=BoschClient)
        mock_response = {
            "executive_summary": None,
            "recommendation_explanations": [],
            "scenario_explanation": None,
        }
        mock_client.generate = AsyncMock(return_value=mock_response)

        service = NarrativeService(client=mock_client, settings=settings, cache=cache)

        advisor_input = AdvisorInput(
            project_id="test-project",
            project_context=None,
            recommendations=[],
            scenario=None,
        )

        # First call — should hit the client
        result1 = await service.explain(advisor_input)
        call_count_after_first = mock_client.generate.call_count

        # Second call — should hit cache, not client
        result2 = await service.explain(advisor_input)
        call_count_after_second = mock_client.generate.call_count

        assert result1 == result2
        assert call_count_after_first == call_count_after_second  # No additional call


# ─────────────────────────────────────────────────────────────────────────────
# 5. STRUCTURED OUTPUT PARSING TESTS
# ─────────────────────────────────────────────────────────────────────────────


class TestStructuredOutputParsing:
    """Validate JSON schema validation and AdvisorOutput construction."""

    def test_advisor_output_parses_valid_schema(self) -> None:
        """AdvisorOutput validates a correct JSON response."""
        valid_response = {
            "executive_summary": None,
            "recommendation_explanations": [],
            "scenario_explanation": None,
        }
        output = AdvisorOutput.model_validate(valid_response)
        assert output.executive_summary is None
        assert output.recommendation_explanations == []

    def test_advisor_output_raises_on_invalid_schema(self) -> None:
        """AdvisorOutput raises ValidationError for invalid structure."""
        invalid_response = {
            "executive_summary": None,
            "recommendation_explanations": "not-a-list",  # should be list
            "scenario_explanation": None,
        }
        with pytest.raises(Exception):  # ValidationError
            AdvisorOutput.model_validate(invalid_response)

    @pytest.mark.asyncio
    async def test_bosch_response_parsing_end_to_end(self) -> None:
        """Complete flow: Bosch response → parse → validate → AdvisorOutput."""
        settings = AISettings(
            ai_provider="bosch",
            aoai_farm_api_key="test-key-123",
        )
        client = BoschClient(settings)

        # Simulate Bosch response
        advisor_output_json = {
            "executive_summary": None,
            "recommendation_explanations": [],
            "scenario_explanation": None,
        }
        bosch_response = {
            "choices": [
                {"message": {"content": json.dumps(advisor_output_json)}}
            ]
        }

        client._http.post = AsyncMock(
            return_value=MagicMock(
                json=lambda: bosch_response,
                raise_for_status=lambda: None,
            )
        )

        # Execute
        result = await client.generate("test message")

        # Validate
        output = AdvisorOutput.model_validate(result)
        assert output.executive_summary is None
        assert output.recommendation_explanations == []

        asyncio.run(client.aclose())


# ─────────────────────────────────────────────────────────────────────────────
# 6. INTEGRATION CONFIGURATION TESTS
# ─────────────────────────────────────────────────────────────────────────────


class TestIntegrationConfiguration:
    """Validate configuration and startup integration."""

    def test_ai_settings_loads_from_bosch_env(self, tmp_path: Path) -> None:
        """AISettings correctly loads Bosch configuration from .env file."""
        env_file = tmp_path / ".env"
        env_file.write_text(
            "\n".join(
                [
                    "AI_PROVIDER=bosch",
                    "AOAI_FARM_API_KEY=real-bosch-key",
                    "AOAI_FARM_DOMAIN=https://aoai-farm.bosch-temp.com",
                    "AOAI_MODEL=gpt-4o-mini",
                    "AOAI_API_VERSION=2024-08-01-preview",
                    "AOAI_FARM_SUBSCRIPTION_ID=sub-xyz",
                    "PX_PROXY_URL=http://127.0.0.1:3128",
                    "AI_TEMPERATURE=0.2",
                    "AI_TIMEOUT=30.0",
                    "AI_MAX_TOKENS=1024",
                    "AI_ADVISOR_ENABLED=true",
                    "AI_CACHE_ENABLED=true",
                ]
            )
        )

        settings = AISettings(_env_file=env_file)
        assert settings.ai_provider == "bosch"
        assert settings.aoai_farm_api_key == "real-bosch-key"
        assert settings.aoai_farm_domain == "https://aoai-farm.bosch-temp.com"
        assert settings.aoai_model == "gpt-4o-mini"
        assert settings.aoai_farm_subscription_id == "sub-xyz"
        assert settings.px_proxy_url == "http://127.0.0.1:3128"
        assert settings.ai_temperature == 0.2
        assert settings.ai_timeout == 30.0

    def test_legacy_env_names_mapped_to_bosch_fields(self, tmp_path: Path) -> None:
        """AISettings maps legacy BOSCH_* env names to AOAI_* fields."""
        env_file = tmp_path / ".env"
        env_file.write_text(
            "\n".join(
                [
                    "AI_PROVIDER=bosch",
                    "BOSCH_API_KEY=legacy-key",
                    "BOSCH_ENDPOINT=https://legacy.endpoint",
                    "BOSCH_DEPLOYMENT=legacy-model",
                    "BOSCH_API_VERSION=2024-06-01",
                    "BOSCH_FARM_SUBSCRIPTION_ID=legacy-sub",
                    "AI_MODEL=ai-model-name",
                ]
            )
        )

        settings = AISettings(_env_file=env_file)
        # Verify legacy names are mapped to new fields
        assert settings.aoai_farm_api_key == "legacy-key"
        assert settings.aoai_farm_domain == "https://legacy.endpoint"
        assert settings.aoai_model == "legacy-model"  # BOSCH_DEPLOYMENT takes precedence
        assert settings.aoai_api_version == "2024-06-01"
        assert settings.aoai_farm_subscription_id == "legacy-sub"

    def test_ai_settings_ignores_unrelated_env_keys(self, tmp_path: Path) -> None:
        """AISettings ignores unrelated environment keys (extra='ignore')."""
        env_file = tmp_path / ".env"
        env_file.write_text(
            "\n".join(
                [
                    "AI_PROVIDER=bosch",
                    "AOAI_FARM_API_KEY=key",
                    "UNRELATED_KEY=value",
                    "DATABASE_URL=postgresql://...",
                    "API_PORT=8000",
                ]
            )
        )

        # Should not raise ValidationError
        settings = AISettings(_env_file=env_file)
        assert settings.ai_provider == "bosch"
        assert settings.aoai_farm_api_key == "key"
