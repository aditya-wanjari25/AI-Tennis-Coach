"""Tests for the configuration module.

Validates that pydantic-settings correctly loads, validates, and
protects sensitive configuration values.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tennis_coach.config import Settings


def test_settings_loads_with_required_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings loads successfully when ANTHROPIC_API_KEY is provided."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.anthropic_api_key.get_secret_value() == "sk-ant-test-key"


def test_settings_uses_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Optional fields fall back to declared defaults."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.anthropic_model == "claude-sonnet-4-5"
    assert settings.pose_min_detection_confidence == 0.5
    assert settings.log_level == "INFO"


def test_settings_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing ANTHROPIC_API_KEY produces a validation error."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)  # type: ignore[call-arg]
    assert "anthropic_api_key" in str(exc_info.value).lower()


def test_settings_rejects_invalid_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Confidence values outside [0.0, 1.0] are rejected."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
    monkeypatch.setenv("POSE_MIN_DETECTION_CONFIDENCE", "1.5")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_secretstr_hides_key_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    """SecretStr prevents the API key from leaking via repr/str."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-super-secret")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert "sk-ant-super-secret" not in repr(settings)
    assert "sk-ant-super-secret" not in str(settings)
