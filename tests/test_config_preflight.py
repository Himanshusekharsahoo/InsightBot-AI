"""Tests for InsightBot AI Configuration Preflight Validator.

Verifies static configuration inspection, credential presence checks,
environment-specific constraints, database URL validation, and secret sanitization.
All tests are strictly deterministic, offline, and database-safe.
"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from insightbot.config.preflight import (
	check_configuration,
	run_preflight_cli,
	sanitize_url,
)
from insightbot.config.settings import Settings


def test_sanitize_url_masks_password():
	"""Ensure database URL passwords are fully masked."""
	raw = "postgresql://myuser:super_secret_pw_99@10.0.0.1:5432/insightbot"
	sanitized = sanitize_url(raw)
	assert "super_secret_pw_99" not in sanitized
	assert "postgresql://myuser:***@10.0.0.1:5432/insightbot" in sanitized

	raw_no_pw = "postgresql://localhost:5432/insightbot"
	assert sanitize_url(raw_no_pw) == "postgresql://localhost:5432/insightbot"


def test_valid_development_configuration_mock(monkeypatch):
	"""Verify that development configuration with mock provider passes cleanly."""
	monkeypatch.setenv("INSIGHTBOT_ENV", "development")
	monkeypatch.setenv("INSIGHTBOT_LLM_PROVIDER", "mock")
	monkeypatch.setenv("INSIGHTBOT_EMBEDDING_PROVIDER", "mock")
	monkeypatch.setenv("INSIGHTBOT_VECTOR_STORE_MODE", "in_memory")
	monkeypatch.delenv("OPENAI_API_KEY", raising=False)
	monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

	report = check_configuration()
	assert report.is_valid is True
	assert len(report.errors) == 0
	assert report.environment == "development"
	text = report.format_text()
	assert "Preflight Status: PASS" in text


def test_valid_development_configuration_real_with_keys(monkeypatch):
	"""Verify development configuration with valid Google Gemini provider and credentials."""
	monkeypatch.setenv("INSIGHTBOT_ENV", "development")
	monkeypatch.setenv("INSIGHTBOT_LLM_PROVIDER", "google")
	monkeypatch.setenv("INSIGHTBOT_EMBEDDING_PROVIDER", "google")
	monkeypatch.setenv("INSIGHTBOT_EMBEDDING_DIM", "768")
	monkeypatch.setenv("GOOGLE_API_KEY", "dummy-google-test-key-valid")
	monkeypatch.setenv("INSIGHTBOT_DATABASE_URL", "postgresql://postgres:pass@localhost:5432/insightbot")

	report = check_configuration()
	assert report.is_valid is True
	assert len(report.errors) == 0
	assert report.summary["llm_provider"] == "google"
	assert report.summary["embedding_dim"] == 768


def test_missing_credentials_for_openai_provider(monkeypatch):
	"""Verify actionable error when OpenAI is selected without OPENAI_API_KEY."""
	monkeypatch.setenv("INSIGHTBOT_ENV", "development")
	monkeypatch.setenv("INSIGHTBOT_LLM_PROVIDER", "openai")
	monkeypatch.setenv("INSIGHTBOT_EMBEDDING_PROVIDER", "mock")
	monkeypatch.delenv("OPENAI_API_KEY", raising=False)

	report = check_configuration()
	assert report.is_valid is False
	assert any("OPENAI_API_KEY" in err for err in report.errors)
	assert any("Action: Set OPENAI_API_KEY" in err for err in report.errors)


def test_missing_credentials_for_google_provider(monkeypatch):
	"""Verify actionable error when Google Gemini is selected without GOOGLE_API_KEY."""
	monkeypatch.setenv("INSIGHTBOT_ENV", "development")
	monkeypatch.setenv("INSIGHTBOT_LLM_PROVIDER", "google")
	monkeypatch.setenv("INSIGHTBOT_EMBEDDING_PROVIDER", "mock")
	monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

	report = check_configuration()
	assert report.is_valid is False
	assert any("GOOGLE_API_KEY" in err for err in report.errors)
	assert any("Action: Set GOOGLE_API_KEY" in err for err in report.errors)


def test_missing_credentials_for_embedding_provider(monkeypatch):
	"""Verify actionable error when real embedding provider is active without credentials."""
	monkeypatch.setenv("INSIGHTBOT_ENV", "development")
	monkeypatch.setenv("INSIGHTBOT_LLM_PROVIDER", "mock")
	monkeypatch.setenv("INSIGHTBOT_EMBEDDING_PROVIDER", "openai")
	monkeypatch.delenv("OPENAI_API_KEY", raising=False)

	report = check_configuration()
	assert report.is_valid is False
	assert any("OpenAI embedding provider is active" in err for err in report.errors)


def test_unsupported_provider_values(monkeypatch):
	"""Verify unsupported provider values are rejected by schema validation."""
	monkeypatch.setenv("INSIGHTBOT_LLM_PROVIDER", "unsupported_ai_vendor")

	with pytest.raises(ValidationError) as exc_info:
		Settings()
	assert "Unsupported LLM provider 'unsupported_ai_vendor'" in str(exc_info.value)


def test_database_url_validation():
	"""Verify PostgreSQL scheme requirement without external network connections."""
	# Valid postgres schemes
	s1 = Settings(database_url="postgresql://user:pass@127.0.0.1:5432/insightbot")
	assert s1.database_url == "postgresql://user:pass@127.0.0.1:5432/insightbot"

	s2 = Settings(database_url="postgres://user:pass@localhost:5432/insightbot")
	assert s2.database_url == "postgres://user:pass@localhost:5432/insightbot"

	# Non-postgres scheme rejected
	with pytest.raises(ValidationError) as exc_mysql:
		Settings(database_url="mysql://user:pass@localhost:3306/insightbot")
	assert "Invalid database URL scheme 'mysql'" in str(exc_mysql.value)

	with pytest.raises(ValidationError) as exc_sqlite:
		Settings(database_url="sqlite:///insightbot.db")
	assert "Invalid database URL scheme 'sqlite'" in str(exc_sqlite.value)


def test_production_environment_security_enforcement(monkeypatch):
	"""Verify production mode strictly enforces API keys and prohibits mock/in_memory."""
	monkeypatch.setenv("INSIGHTBOT_ENV", "production")
	monkeypatch.setenv("INSIGHTBOT_LLM_PROVIDER", "mock")
	monkeypatch.setenv("INSIGHTBOT_EMBEDDING_PROVIDER", "mock")
	monkeypatch.setenv("INSIGHTBOT_API_KEY", "short_key")  # < 16 chars

	with pytest.raises(ValidationError) as exc_info:
		Settings()
	err_msg = str(exc_info.value)
	assert "at least 16 characters" in err_msg or "mock providers are strictly forbidden" in err_msg


def test_no_secret_values_in_error_or_report_messages(monkeypatch):
	"""Verify that secret passwords and API keys are never exposed in reports or errors."""
	secret_pw = "super_classified_db_password_XYZ123"
	secret_key = "secret_agent_api_token_ABC456"

	monkeypatch.setenv("INSIGHTBOT_ENV", "development")
	monkeypatch.setenv("INSIGHTBOT_LLM_PROVIDER", "mock")
	monkeypatch.setenv("INSIGHTBOT_EMBEDDING_PROVIDER", "mock")
	monkeypatch.setenv("INSIGHTBOT_DATABASE_URL", f"postgresql://bot_user:{secret_pw}@db.prod.internal:5432/insightbot")
	monkeypatch.setenv("INSIGHTBOT_API_KEY", secret_key)

	report = check_configuration()
	text_output = report.format_text()
	dict_output = json.dumps(report.to_dict())

	# Assert secret password NEVER appears in cleartext
	assert secret_pw not in text_output
	assert secret_pw not in dict_output
	assert ":***@" in text_output

	# Assert secret API key NEVER appears in cleartext
	assert secret_key not in text_output
	assert secret_key not in dict_output


def test_env_example_consistency():
	"""Verify .env.example defines safe development defaults and supported settings."""
	env_path = Path(".env.example")
	assert env_path.exists(), ".env.example file must exist in project root"

	content = env_path.read_text(encoding="utf-8")
	# Must default to development for safe local developer onboarding
	assert "INSIGHTBOT_ENV=development" in content
	# Must include PostgreSQL database URL
	assert "INSIGHTBOT_DATABASE_URL=postgresql://" in content
	# Must include supported provider options
	assert "INSIGHTBOT_LLM_PROVIDER=google" in content
	assert "INSIGHTBOT_EMBEDDING_PROVIDER=google" in content
	assert "INSIGHTBOT_EMBEDDING_DIM=768" in content
	# Must include local development CORS
	assert "http://localhost:3000" in content
	# Must NOT contain real sensitive API keys
	assert "AIzaSy" not in content
	assert "sk-proj-" not in content


def test_preflight_cli_json_and_exit_codes(monkeypatch):
	"""Verify CLI flags --json, --quiet, and appropriate exit codes."""
	monkeypatch.setenv("INSIGHTBOT_ENV", "development")
	monkeypatch.setenv("INSIGHTBOT_LLM_PROVIDER", "mock")
	monkeypatch.setenv("INSIGHTBOT_EMBEDDING_PROVIDER", "mock")
	monkeypatch.setenv("INSIGHTBOT_VECTOR_STORE_MODE", "in_memory")

	# Valid configuration returns 0
	code_valid = run_preflight_cli(["--quiet"])
	assert code_valid == 0

	# Invalid configuration (missing real provider key) returns 1
	monkeypatch.setenv("INSIGHTBOT_LLM_PROVIDER", "google")
	monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
	code_invalid = run_preflight_cli(["--quiet"])
	assert code_invalid == 1
