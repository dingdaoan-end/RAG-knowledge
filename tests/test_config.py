import pytest


def test_settings_read_environment_and_hide_key(monkeypatch):
    from config import Settings

    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.setenv("API_KEY", "test-secret")
    monkeypatch.setenv("BASE_URL", "https://example.test/v1")
    settings = Settings.from_env()
    assert settings.model_name == "test-model"
    assert settings.api_key == "test-secret"
    assert settings.base_url == "https://example.test/v1"
    assert "test-secret" not in repr(settings)


def test_official_key_alias_and_optional_base_url(monkeypatch):
    from config import Settings

    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-official-key")
    monkeypatch.delenv("BASE_URL", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)
    settings = Settings.from_env()
    assert settings.api_key == "test-official-key"
    assert settings.base_url is None
    assert settings.model_name


def test_blank_model_is_rejected(monkeypatch):
    from config import Settings

    monkeypatch.setenv("MODEL_NAME", "  ")
    with pytest.raises(ValueError):
        Settings.from_env()
