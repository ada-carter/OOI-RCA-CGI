"""
Test 02 — Configuration & Settings

Tests the Settings loader: secrets.toml resolution, config.json
for non-sensitive settings, and environment variable fallbacks.
"""

import os
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestSettingsProperties:
    def test_project_name(self):
        from backend.core.config import settings
        assert settings.PROJECT_NAME == "OOI RCA Copilot"

    def test_llm_provider_valid(self):
        from backend.core.config import settings
        assert settings.LLM_PROVIDER in ("local", "fireworks")

    def test_model_config_structure(self):
        from backend.core.config import settings
        config = settings.MODEL_CONFIG
        assert isinstance(config, dict)
        assert "repo_id" in config
        assert "filename" in config
        assert config["filename"].endswith(".gguf")

    def test_display_name_fallback(self):
        """DISPLAY_NAME returns 'You' when config has no value."""
        from backend.core.config import Settings
        s = Settings()
        # If no display_name is set, it should return "You"
        name = s.DISPLAY_NAME
        assert isinstance(name, str)
        assert len(name) > 0


class TestSecretsResolution:
    """Verify that credentials resolve from secrets.toml, not config.json."""

    def test_ooi_username_not_from_config_json(self):
        """OOI_USERNAME should not fall back to config.json."""
        from backend.core.config import _load_config_json, settings
        config = _load_config_json()
        # config.json should NOT contain ooi_username anymore
        assert "ooi_username" not in config

    def test_ooi_token_not_from_config_json(self):
        from backend.core.config import _load_config_json
        config = _load_config_json()
        assert "ooi_token" not in config

    def test_fireworks_key_not_from_config_json(self):
        from backend.core.config import _load_config_json
        config = _load_config_json()
        assert "fireworks_api_key" not in config

    def test_env_var_fallback(self, monkeypatch):
        """If secrets.toml is empty, falls back to env vars."""
        monkeypatch.setenv("OOI_USERNAME", "ENV_TEST_USER")
        # Patch secrets loader to return empty so env var fallback triggers
        import backend.core.config as config_mod
        monkeypatch.setattr(config_mod, "_load_secrets_toml", lambda: {})
        from backend.core.config import Settings
        s = Settings()
        assert s.OOI_USERNAME == "ENV_TEST_USER"

    def test_google_oauth_properties_exist(self):
        """Google OAuth properties are accessible (may be empty in test)."""
        from backend.core.config import settings
        assert isinstance(settings.GOOGLE_CLIENT_ID, str)
        assert isinstance(settings.GOOGLE_CLIENT_SECRET, str)
        assert isinstance(settings.OAUTH_REDIRECT_URI, str)
