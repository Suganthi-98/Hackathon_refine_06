from pathlib import Path

from app.ai.config import AISettings
from app.core.config import Settings


def test_core_settings_ignores_unrelated_ai_environment(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "AI_PROVIDER=bosch",
                "BOSCH_API_KEY=test-key",
                "BOSCH_ENDPOINT=https://example.test",
                "BOSCH_DEPLOYMENT=test-deployment",
                "BOSCH_API_VERSION=2024-06-01",
                "BOSCH_FARM_SUBSCRIPTION_ID=sub-123",
                "AI_MODEL=gpt-4o-mini",
                "AI_ADVISOR_ENABLED=true",
                "AI_CACHE_ENABLED=true",
                "PX_PROXY_URL=https://proxy.test",
            ]
        )
    )

    settings = Settings(_env_file=env_file)

    assert settings.api_host == "0.0.0.0"
    assert settings.debug is False


def test_ai_settings_loads_bosch_and_proxy_values(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "AI_PROVIDER=bosch",
                "BOSCH_API_KEY=test-key",
                "BOSCH_ENDPOINT=https://example.test",
                "BOSCH_DEPLOYMENT=test-deployment",
                "BOSCH_API_VERSION=2024-06-01",
                "BOSCH_FARM_SUBSCRIPTION_ID=sub-123",
                "PX_PROXY_URL=https://proxy.test",
                "AI_MODEL=gpt-4o-mini",
                "AI_ADVISOR_ENABLED=true",
                "AI_CACHE_ENABLED=true",
            ]
        )
    )

    ai_settings = AISettings(_env_file=env_file)

    assert ai_settings.ai_provider == "bosch"
    assert ai_settings.bosch_api_key == "test-key"
    assert ai_settings.bosch_farm_subscription_id == "sub-123"
    assert ai_settings.px_proxy_url == "https://proxy.test"
