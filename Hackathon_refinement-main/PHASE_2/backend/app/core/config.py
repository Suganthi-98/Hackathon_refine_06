"""
Sprint Whisperer Backend Configuration
All simulation parameters and constants in one place.
Never hardcode configuration values inline.
"""

from pathlib import Path
from typing import Dict

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from .env file."""

    # ─── API Settings ────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False

    # ─── Monte Carlo Configuration ───────────────────────────────────────────
    mc_iterations: int = 10_000
    effort_variance_min: float = 0.80
    effort_variance_mode: float = 1.00
    effort_variance_max: float = 1.35
    velocity_std_dev: float = 0.15
    velocity_min_clamp: float = 0.30
    velocity_max_clamp: float = 1.50

    # ─── Blocker Configuration ───────────────────────────────────────────────
    blocker_velocity_impact: Dict[str, float] = {
        "Critical": 0.40,
        "High": 0.20,
        "Medium": 0.10,
        "Low": 0.05,
    }
    blocker_max_velocity_reduction: float = 0.70

    # ─── Spillover Configuration ─────────────────────────────────────────────
    spillover_capacity_compression_factor: float = 0.85
    spillover_forecast_weight: float = 0.30
    spillover_item_cap_pct: float = 0.15

    # ─── Forecast Configuration ──────────────────────────────────────────────
    min_velocity_factor: float = 0.70
    
    # ─── Utilization Configuration ───────────────────────────────────────────
    utilization_penalty_growth_cap: float = 1.50

    # ─── Risk Scoring ────────────────────────────────────────────────────────
    risk_weights: Dict[str, float] = {
        "schedule": 0.35,
        "resource": 0.25,
        "dependency": 0.25,
        "scope": 0.15,
    }
    risk_critical_threshold: int = 75
    risk_high_threshold: int = 50
    risk_medium_threshold: int = 25

    # ─── Resource Utilization ────────────────────────────────────────────────
    underutilization_threshold: float = 0.60
    overload_threshold: float = 1.00
    sprint_overload_warning_threshold: float = 0.90
    sprint_overload_critical_threshold: float = 1.10

    # ─── Session Management ──────────────────────────────────────────────────
    session_timeout_minutes: int = 30

    # ─── File Upload ─────────────────────────────────────────────────────────
    max_file_size_mb: int = 10
    allowed_extensions: list = [".xlsx"]  # Fixed list, not from .env

    # ─── Demo Mode ───────────────────────────────────────────────────────────
    demo_workbook_path: str = str(
        Path(__file__).resolve().parents[3] / "INPUT" / "TIO2_Sprint_Intelligence_v5_final.xlsx"
    )
    frontend_origin: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )


# Global settings instance
settings = Settings()
