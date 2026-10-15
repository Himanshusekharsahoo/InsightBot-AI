"""InsightBot AI Configuration Preflight Validator.

Provides fast, static, deterministic configuration validation before expensive application
startup, network requests, or database initialization.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Optional

try:
	from dotenv import load_dotenv

	load_dotenv()
except ImportError:
	pass

from insightbot.config.settings import Settings


def sanitize_url(url: str) -> str:
	"""Mask sensitive credentials in database connection URLs."""
	if not url:
		return ""
	try:
		parsed = urllib.parse.urlsplit(url)
		if parsed.password:
			netloc = parsed.netloc.replace(f":{parsed.password}@", ":***@")
			return urllib.parse.urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))
	except Exception:
		pass
	return re.sub(r":([^/@]+)@", ":***@", url)


@dataclass
class PreflightReport:
	"""Structured result of a configuration preflight check."""

	is_valid: bool
	environment: str
	errors: list[str] = field(default_factory=list)
	warnings: list[str] = field(default_factory=list)
	info: list[str] = field(default_factory=list)
	summary: dict[str, Any] = field(default_factory=dict)

	def format_text(self) -> str:
		"""Render a human-readable, non-sensitive CLI report."""
		lines: list[str] = []
		lines.append("=" * 64)
		lines.append("  InsightBot AI — Configuration Preflight")
		lines.append("=" * 64)
		lines.append(f"  Environment:       {self.environment}")
		if "database_url" in self.summary:
			lines.append(f"  Database Target:   {self.summary['database_url']}")
		if "vector_store_mode" in self.summary:
			lines.append(f"  Storage Mode:      {self.summary['vector_store_mode']}")
		if "llm_provider" in self.summary:
			lines.append(f"  LLM Provider:      {self.summary['llm_provider']} ({self.summary.get('llm_model', 'default')})")
		if "embedding_provider" in self.summary:
			lines.append(
				f"  Embedding:         {self.summary['embedding_provider']} "
				f"({self.summary.get('embedding_model', 'default')}, dim={self.summary.get('embedding_dim', 'auto')})"
			)
		lines.append("-" * 64)

		if self.errors:
			lines.append("  CONFIGURATION ERRORS (Action Required):")
			for i, err in enumerate(self.errors, 1):
				lines.append(f"    {i}. [ERROR] {err}")
			lines.append("")

		if self.warnings:
			lines.append("  CONFIGURATION WARNINGS:")
			for i, warn in enumerate(self.warnings, 1):
				lines.append(f"    {i}. [WARN]  {warn}")
			lines.append("")

		if self.info:
			lines.append("  INFORMATIONAL NOTES:")
			for inf in self.info:
				lines.append(f"    * {inf}")
			lines.append("")

		lines.append("-" * 64)
		status_str = "PASS — Configuration is valid." if self.is_valid else "FAIL — Configuration errors detected."
		lines.append(f"  Preflight Status: {status_str}")
		lines.append("=" * 64)
		return "\n".join(lines)

	def to_dict(self) -> dict[str, Any]:
		"""Serialize report to a dictionary safe for JSON export."""
		return {
			"is_valid": self.is_valid,
			"environment": self.environment,
			"errors": self.errors,
			"warnings": self.warnings,
			"info": self.info,
			"summary": self.summary,
		}


def check_configuration(settings: Optional[Settings] = None) -> PreflightReport:
	"""Perform static, non-network configuration validation.

	Validates:
	1. Pydantic model integrity and field constraints.
	2. Provider selection and required credentials without external API calls.
	3. Database URL syntax and storage mode without connecting to the database.
	4. Security invariants based on environment (development vs production).
	5. CORS origin rules and chunk size constraints.
	"""
	errors: list[str] = []
	warnings: list[str] = []
	info: list[str] = []
	summary: dict[str, Any] = {}

	# 1. Attempt to load / validate Settings
	cfg: Optional[Settings] = settings
	if cfg is None:
		try:
			cfg = Settings()
		except Exception as e:
			raw_err = str(e)
			# Sanitize any potential secret values in error text
			sanitized_err = re.sub(r":([^/@]+)@", ":***@", raw_err)
			errors.append(f"Settings schema validation failed: {sanitized_err}")
			return PreflightReport(
				is_valid=False,
				environment=os.getenv("INSIGHTBOT_ENV", "development"),
				errors=errors,
				warnings=warnings,
				info=info,
				summary=summary,
			)

	env = cfg.environment
	summary["environment"] = env
	summary["database_url"] = sanitize_url(cfg.database_url)
	summary["vector_store_mode"] = cfg.vector_store_mode
	summary["llm_provider"] = cfg.llm_provider
	summary["llm_model"] = cfg.llm_model
	summary["embedding_provider"] = cfg.embedding_provider
	summary["embedding_model"] = cfg.embedding_model
	summary["embedding_dim"] = cfg.embedding_dimension

	# 2. Environment & Security Checks
	if env in ("production", "prod", "staging", "stage"):
		info.append(f"Running in strict {env} security mode.")
		if not (cfg.api_key and len(cfg.api_key.strip()) >= 16) and not any(
			len(k.strip()) >= 16 for k in cfg.tenant_api_keys.values()
		):
			errors.append(
				f"In {env} mode, INSIGHTBOT_API_KEY or INSIGHTBOT_TENANT_API_KEYS must be set to a strong secret (>= 16 characters)."
			)
		if cfg.llm_provider == "mock" or cfg.embedding_provider == "mock":
			errors.append(f"In {env} mode, mock AI providers ('mock') are strictly forbidden.")
		if cfg.vector_store_mode == "in_memory":
			errors.append(f"In {env} mode, ephemeral 'in_memory' vector storage is forbidden.")
		if "*" in cfg.cors_origins:
			errors.append(f"In {env} mode, wildcard '*' CORS origin is strictly forbidden.")
	else:
		info.append("Running in local development mode.")
		if not cfg.api_key and not cfg.tenant_api_keys:
			info.append("Single-tenant local mode active (unauthenticated local API access enabled).")
		dev_origins = ("localhost:3000", "127.0.0.1:3000")
		if not any(any(d in o for d in dev_origins) for o in cfg.cors_origins):
			warnings.append(
				"Frontend development server (http://localhost:3000) is not listed in INSIGHTBOT_CORS_ORIGINS. "
				"Next.js frontend requests might be blocked by CORS."
			)

	# 3. Database URL & Storage Mode Validation
	if cfg.vector_store_mode == "in_memory":
		info.append("Vector store is configured for in-memory mode. Data will NOT persist across restarts.")
	else:
		try:
			parsed_db = urllib.parse.urlsplit(cfg.database_url)
			if parsed_db.scheme not in ("postgresql", "postgres"):
				errors.append(
					f"Invalid database URL scheme '{parsed_db.scheme}'. "
					"InsightBot requires PostgreSQL ('postgresql://' or 'postgres://'). "
					"For offline testing without a database, set INSIGHTBOT_VECTOR_STORE_MODE=in_memory."
				)
			elif not parsed_db.hostname:
				errors.append("Database URL is missing a hostname (e.g. 'localhost' or '127.0.0.1').")
		except Exception as e:
			errors.append(f"Malformed database connection URL: {e}")

	# 4. LLM Provider & Credential Validation
	llm_prov = cfg.llm_provider.lower()
	if llm_prov in ("google", "gemini"):
		google_key = os.getenv("GOOGLE_API_KEY")
		if not google_key or not google_key.strip():
			errors.append(
				"Google Gemini LLM provider is active ('INSIGHTBOT_LLM_PROVIDER=google'), "
				"but 'GOOGLE_API_KEY' is not set in your environment or .env file. "
				"Action: Set GOOGLE_API_KEY or switch to 'INSIGHTBOT_LLM_PROVIDER=openai' (with OPENAI_API_KEY) "
				"or 'INSIGHTBOT_LLM_PROVIDER=mock' for offline testing."
			)
	elif llm_prov == "openai":
		openai_key = os.getenv("OPENAI_API_KEY")
		if not openai_key or not openai_key.strip():
			errors.append(
				"OpenAI LLM provider is active ('INSIGHTBOT_LLM_PROVIDER=openai'), "
				"but 'OPENAI_API_KEY' is not set in your environment or .env file. "
				"Action: Set OPENAI_API_KEY or switch to 'INSIGHTBOT_LLM_PROVIDER=google' (with GOOGLE_API_KEY) "
				"or 'INSIGHTBOT_LLM_PROVIDER=mock' for offline testing."
			)
	elif llm_prov == "mock":
		if env not in ("production", "prod", "staging", "stage"):
			info.append("Mock LLM provider is active. Synthetic grounded responses will be generated without external API calls.")
	else:
		errors.append(f"Unsupported LLM provider '{cfg.llm_provider}'. Supported: google, openai, mock.")

	# 5. Embedding Provider & Credential Validation
	emb_prov = cfg.embedding_provider.lower()
	if emb_prov in ("google", "gemini"):
		google_key = os.getenv("GOOGLE_API_KEY")
		if not google_key or not google_key.strip():
			errors.append(
				"Google Gemini embedding provider is active ('INSIGHTBOT_EMBEDDING_PROVIDER=google'), "
				"but 'GOOGLE_API_KEY' is not set in your environment or .env file. "
				"Action: Set GOOGLE_API_KEY or configure 'INSIGHTBOT_EMBEDDING_PROVIDER=mock' for offline testing."
			)
		if cfg.embedding_dimension != 768:
			warnings.append(
				f"Google Gemini embeddings ('gemini-embedding-001') produce 768-dimensional vectors, "
				f"but INSIGHTBOT_EMBEDDING_DIM is set to {cfg.embedding_dimension}."
			)
	elif emb_prov == "openai":
		openai_key = os.getenv("OPENAI_API_KEY")
		if not openai_key or not openai_key.strip():
			errors.append(
				"OpenAI embedding provider is active ('INSIGHTBOT_EMBEDDING_PROVIDER=openai'), "
				"but 'OPENAI_API_KEY' is not set in your environment or .env file. "
				"Action: Set OPENAI_API_KEY or configure 'INSIGHTBOT_EMBEDDING_PROVIDER=mock' for offline testing."
			)
		if cfg.embedding_model == "text-embedding-3-small" and cfg.embedding_dimension not in (512, 1536):
			warnings.append(
				f"OpenAI text-embedding-3-small standard dimensions are 1536 or 512, "
				f"but INSIGHTBOT_EMBEDDING_DIM is set to {cfg.embedding_dimension}."
			)
	elif emb_prov == "mock":
		if env not in ("production", "prod", "staging", "stage"):
			info.append("Mock embedding provider is active. Deterministic pseudo-vectors will be used offline.")
	else:
		errors.append(f"Unsupported embedding provider '{cfg.embedding_provider}'. Supported: google, openai, mock.")

	# 6. Text Chunking Invariants
	if cfg.chunk_overlap >= cfg.chunk_size:
		errors.append(f"chunk_overlap ({cfg.chunk_overlap}) must be strictly less than chunk_size ({cfg.chunk_size}).")

	is_valid = len(errors) == 0
	return PreflightReport(
		is_valid=is_valid,
		environment=env,
		errors=errors,
		warnings=warnings,
		info=info,
		summary=summary,
	)


def run_preflight_cli(argv: Optional[list[str]] = None) -> int:
	"""CLI entry point for configuration preflight.

	Returns:
		0 if configuration is valid.
		1 if configuration errors were found.
	"""
	args = argv if argv is not None else sys.argv[1:]
	is_json = "--json" in args
	is_quiet = "--quiet" in args or "-q" in args

	report = check_configuration()

	if is_json:
		print(json.dumps(report.to_dict(), indent=2))
	elif not is_quiet:
		print(report.format_text())
	elif not report.is_valid:
		for err in report.errors:
			print(f"[ERROR] {err}", file=sys.stderr)

	return 0 if report.is_valid else 1


if __name__ == "__main__":
	sys.exit(run_preflight_cli())
