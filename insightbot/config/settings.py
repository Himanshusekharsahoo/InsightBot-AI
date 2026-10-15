import os
import secrets
from functools import lru_cache
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from browser_use.config import CONFIG, Config, FlatEnvConfig, InsightBotConfig, OldConfig, load_insightbot_config


def _default_provider(env_var: str) -> str:
	explicit = os.getenv(env_var)
	if explicit and explicit.strip():
		return explicit.strip().lower()
	if os.getenv('GOOGLE_API_KEY') and not os.getenv('OPENAI_API_KEY'):
		return 'google'
	return 'openai'


def _default_llm_model() -> str:
	explicit = os.getenv('INSIGHTBOT_LLM_MODEL')
	if explicit and explicit.strip():
		return explicit.strip()
	prov = _default_provider('INSIGHTBOT_LLM_PROVIDER')
	if prov in ('google', 'gemini'):
		return 'gemini-flash-latest'
	elif prov == 'mock':
		return 'mock-llm'
	return 'gpt-4o-mini'


def _default_embedding_model() -> str:
	explicit = os.getenv('INSIGHTBOT_EMBEDDING_MODEL')
	if explicit and explicit.strip():
		return explicit.strip()
	prov = _default_provider('INSIGHTBOT_EMBEDDING_PROVIDER')
	if prov in ('google', 'gemini'):
		return 'gemini-embedding-001'
	elif prov == 'mock':
		return 'mock-embedding'
	return 'text-embedding-3-small'


def _default_embedding_dimension() -> int:
	explicit = os.getenv('INSIGHTBOT_EMBEDDING_DIM') or os.getenv('INSIGHTBOT_EMBEDDING_DIMENSION')
	if explicit and explicit.strip():
		try:
			return int(explicit.strip())
		except ValueError:
			pass
	prov = _default_provider('INSIGHTBOT_EMBEDDING_PROVIDER')
	if prov in ('google', 'gemini'):
		return 768
	return 1536


class Settings(BaseModel):
	"""Settings for InsightBot AI RAG, Database, and Knowledge services."""

	model_config = ConfigDict(validate_default=True)

	# Database settings
	database_url: str = Field(
		default_factory=lambda: os.getenv(
			'INSIGHTBOT_DATABASE_URL', os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/insightbot')
		),
		description='PostgreSQL connection URL',
	)
	vector_store_mode: str = Field(
		default_factory=lambda: os.getenv('INSIGHTBOT_VECTOR_STORE_MODE', 'auto'),
		description='Vector store mode: auto, native, array, in_memory',
	)
	min_similarity_score: float = Field(
		default_factory=lambda: float(os.getenv('INSIGHTBOT_MIN_SIMILARITY_SCORE', '0.2')),
		description='Minimum similarity threshold for grounded evidence retrieval',
	)

	# Embedding settings
	embedding_provider: str = Field(
		default_factory=lambda: _default_provider('INSIGHTBOT_EMBEDDING_PROVIDER'),
		description='Embedding provider: openai, google, mock',
	)
	embedding_model: str = Field(
		default_factory=_default_embedding_model,
		description='Embedding model name',
	)
	embedding_dimension: int = Field(
		default_factory=_default_embedding_dimension,
		description='Embedding vector dimensions',
	)

	# Text chunking settings
	chunk_size: int = Field(
		default_factory=lambda: int(os.getenv('INSIGHTBOT_CHUNK_SIZE', '1000')),
		description='Character length per chunk',
	)
	chunk_overlap: int = Field(
		default_factory=lambda: int(os.getenv('INSIGHTBOT_CHUNK_OVERLAP', '200')),
		description='Overlap character count between chunks',
	)

	# Upload and ingestion limits
	max_upload_size_bytes: int = Field(
		default_factory=lambda: int(os.getenv('INSIGHTBOT_MAX_UPLOAD_SIZE', str(20 * 1024 * 1024))),
		description='Maximum allowed document upload size in bytes (20MB default)',
	)
	max_chunks_per_document: int = Field(
		default_factory=lambda: int(os.getenv('INSIGHTBOT_MAX_CHUNKS_PER_DOC', '5000')),
		description='Maximum allowed chunks per document during ingestion',
	)

	# LLM generation settings
	llm_provider: str = Field(
		default_factory=lambda: _default_provider('INSIGHTBOT_LLM_PROVIDER'),
		description='LLM provider for synthesis',
	)
	llm_model: str = Field(
		default_factory=_default_llm_model,
		description='LLM model name for synthesis',
	)

	# AI API timeout & concurrency limits
	ai_request_timeout: float = Field(
		default_factory=lambda: float(os.getenv('INSIGHTBOT_AI_TIMEOUT', '30.0')),
		description='Timeout in seconds for external AI API calls',
	)
	ai_max_concurrent_requests: int = Field(
		default_factory=lambda: int(os.getenv('INSIGHTBOT_AI_MAX_CONCURRENT_REQUESTS', '10')),
		description='Maximum concurrent in-flight AI synthesis requests per process',
	)

	# Browser Automation Engine settings
	browser_enabled: bool = Field(
		default_factory=lambda: os.getenv('INSIGHTBOT_BROWSER_ENABLED', 'false').lower() in ('1', 'true', 'yes'),
		description='Enable autonomous browser agent for web research',
	)
	browser_headless: bool = Field(
		default_factory=lambda: os.getenv('INSIGHTBOT_BROWSER_HEADLESS', 'true').lower() in ('1', 'true', 'yes'),
		description='Run browser in headless mode without UI',
	)
	browser_use_cloud: bool = Field(
		default_factory=lambda: os.getenv('INSIGHTBOT_BROWSER_USE_CLOUD', 'false').lower() in ('1', 'true', 'yes'),
		description='Provision remote browser via Browser Use Cloud (requires BROWSER_USE_API_KEY)',
	)

	# API security / telemetry
	api_key: Optional[str] = Field(
		default_factory=lambda: os.getenv('INSIGHTBOT_API_KEY'),
		description='InsightBot API key',
	)

	# Multi-tenancy authorization
	multi_tenancy_enabled: bool = Field(
		default_factory=lambda: os.getenv('INSIGHTBOT_MULTI_TENANCY_ENABLED', 'false').lower() in ('1', 'true', 'yes'),
		description='Enable strict multi-tenant authorization boundaries',
	)
	default_organization_id: str = Field(
		default_factory=lambda: os.getenv('INSIGHTBOT_DEFAULT_ORGANIZATION_ID', 'default'),
		description='Default organization ID for single-tenant mode or admin key',
	)
	tenant_api_keys: dict[str, str] = Field(
		default_factory=lambda: Settings._parse_tenant_keys(os.getenv('INSIGHTBOT_TENANT_API_KEYS', '')),
		description='Mapping of organization_id to authorized API key',
	)

	@staticmethod
	def _parse_tenant_keys(raw: str) -> dict[str, str]:
		raw = raw.strip()
		if not raw:
			return {}
		if raw.startswith('{'):
			import json

			try:
				parsed = json.loads(raw)
				if isinstance(parsed, dict):
					return {str(k).strip(): str(v).strip() for k, v in parsed.items()}
			except Exception:
				pass
		mapping: dict[str, str] = {}
		for pair in raw.split(','):
			pair = pair.strip()
			if ':' in pair:
				k, v = pair.split(':', 1)
				if k.strip() and v.strip():
					mapping[k.strip()] = v.strip()
		return mapping

	def resolve_tenant_key(self, api_key: Any) -> Optional[tuple[str, bool]]:
		"""Resolve provided API key to (organization_id, is_admin).

		Returns:
			tuple[str, bool]: (organization_id, is_admin) if valid, None if unauthorized.
		"""
		if not api_key or not isinstance(api_key, str):
			return None

		cleaned_key = api_key.strip()
		if not cleaned_key or len(cleaned_key) > 512:
			return None

		# 1. Match primary global / admin API key (constant-time comparison)
		if self.api_key and secrets.compare_digest(self.api_key, cleaned_key):
			return (self.default_organization_id, True)

		# 2. Match tenant-specific mapped API keys (constant-time comparison)
		for org_id, key in self.tenant_api_keys.items():
			if secrets.compare_digest(key, cleaned_key):
				return (org_id, False)

		# 3. Structured tenant key: ik_<org_id>_<secret> matching configured tenant
		if cleaned_key.startswith('ik_'):
			parts = cleaned_key.split('_', 2)
			if len(parts) >= 2 and parts[1]:
				org_id = parts[1]
				expected_key = self.tenant_api_keys.get(org_id)
				if expected_key and secrets.compare_digest(expected_key, cleaned_key):
					return (org_id, False)

		return None

	# Environment & deployment
	environment: str = Field(
		default_factory=lambda: os.getenv('INSIGHTBOT_ENV', os.getenv('ENVIRONMENT', 'development')).lower(),
		description='Deployment environment: development, staging, production',
	)
	cors_origins: list[str] = Field(
		default_factory=lambda: [
			o.strip()
			for o in os.getenv(
				'INSIGHTBOT_CORS_ORIGINS',
				'http://localhost:3000,http://localhost:8000,http://127.0.0.1:3000,http://127.0.0.1:8000',
			).split(',')
			if o.strip()
		],
		description='Allowed CORS origins',
	)

	@classmethod
	def _validate_non_empty(cls, v: str, name: str) -> str:
		if not v or not v.strip():
			raise ValueError(f"'{name}' cannot be empty")
		return v.strip()

	@field_validator('database_url')
	@classmethod
	def validate_database_url(cls, v: str) -> str:
		if not v or not v.strip():
			raise ValueError("database_url cannot be empty")
		val = v.strip()
		import urllib.parse
		try:
			parsed = urllib.parse.urlsplit(val)
		except Exception as e:
			raise ValueError(f"Malformed database URL: {e}") from e

		if parsed.scheme not in ('postgresql', 'postgres'):
			raise ValueError(
				f"Invalid database URL scheme '{parsed.scheme}'. Must be 'postgresql' or 'postgres' "
				f"(e.g., 'postgresql://user:password@localhost:5432/dbname')."
			)
		if not parsed.hostname and not parsed.path:
			raise ValueError("Invalid database URL: missing host or database name.")
		return val

	@field_validator('embedding_provider')
	@classmethod
	def validate_embedding_provider(cls, v: str) -> str:
		val = v.strip().lower()
		if val not in ('openai', 'google', 'gemini', 'mock'):
			raise ValueError(f"Unsupported embedding provider '{v}'. Must be one of: openai, google, mock")
		return val

	@field_validator('llm_provider')
	@classmethod
	def validate_llm_provider(cls, v: str) -> str:
		val = v.strip().lower()
		if val not in ('openai', 'google', 'gemini', 'mock'):
			raise ValueError(f"Unsupported LLM provider '{v}'. Must be one of: openai, google, mock")
		return val

	@field_validator('vector_store_mode')
	@classmethod
	def validate_vector_store_mode(cls, v: str) -> str:
		val = v.strip().lower()
		if val not in ('auto', 'native', 'native_pgvector', 'array', 'pg_array', 'in_memory'):
			raise ValueError(f"Unsupported vector_store_mode '{v}'. Must be one of: auto, native, array, in_memory")
		return val

	@field_validator('ai_request_timeout')
	@classmethod
	def validate_timeout(cls, v: float) -> float:
		if v <= 0:
			raise ValueError(f'ai_request_timeout must be greater than 0, got {v}')
		return float(v)

	@field_validator('embedding_dimension')
	@classmethod
	def validate_dimension(cls, v: int) -> int:
		if v <= 0:
			raise ValueError(f'embedding_dimension must be positive integer, got {v}')
		return int(v)

	@field_validator('chunk_size')
	@classmethod
	def validate_chunk_size(cls, v: int) -> int:
		if v <= 0:
			raise ValueError(f'chunk_size must be positive integer, got {v}')
		return int(v)

	@field_validator('chunk_overlap')
	@classmethod
	def validate_chunk_overlap(cls, v: int) -> int:
		if v < 0:
			raise ValueError(f'chunk_overlap cannot be negative, got {v}')
		return int(v)

	@field_validator('max_upload_size_bytes')
	@classmethod
	def validate_max_upload_size(cls, v: int) -> int:
		if v <= 0:
			raise ValueError(f'max_upload_size_bytes must be positive, got {v}')
		return int(v)

	@field_validator('min_similarity_score')
	@classmethod
	def validate_min_similarity(cls, v: float) -> float:
		if not (0.0 <= v <= 1.0):
			raise ValueError(f'min_similarity_score must be between 0.0 and 1.0, got {v}')
		return float(v)

	@field_validator('embedding_model', 'llm_model')
	@classmethod
	def validate_models(cls, v: str) -> str:
		if not v or not v.strip():
			raise ValueError('Model name cannot be empty')
		return v.strip()

	@model_validator(mode='after')
	def validate_chunk_constraints(self) -> 'Settings':
		if self.chunk_overlap >= self.chunk_size:
			raise ValueError(f'chunk_overlap ({self.chunk_overlap}) must be strictly less than chunk_size ({self.chunk_size})')

		# Deployment safeguards for production and staging
		if self.environment in ('production', 'prod', 'staging', 'stage'):
			has_valid_global_key = bool(self.api_key and len(self.api_key.strip()) >= 16)
			has_valid_tenant_keys = bool(self.tenant_api_keys) and all(
				len(k.strip()) >= 16 for k in self.tenant_api_keys.values()
			)
			if not (has_valid_global_key or has_valid_tenant_keys):
				raise ValueError(
					f"In {self.environment} ('INSIGHTBOT_ENV={self.environment}'), 'INSIGHTBOT_API_KEY' or "
					"'INSIGHTBOT_TENANT_API_KEYS' must be configured with a strong secret key (at least 16 characters)."
				)
			if self.llm_provider == 'mock' or self.embedding_provider == 'mock':
				raise ValueError(
					f"In {self.environment} ('INSIGHTBOT_ENV={self.environment}'), mock providers are strictly forbidden."
				)
			if self.vector_store_mode == 'in_memory':
				raise ValueError(
					f"In {self.environment} ('INSIGHTBOT_ENV={self.environment}'), 'in_memory' vector storage is forbidden."
				)
			if '*' in self.cors_origins:
				raise ValueError(
					f"In {self.environment} ('INSIGHTBOT_ENV={self.environment}'), wildcard '*' CORS origins are forbidden. "
					"Specify explicit origins in 'INSIGHTBOT_CORS_ORIGINS'."
				)
		return self


@lru_cache()
def get_settings() -> Settings:
	"""Retrieve cached Settings instance."""
	return Settings()


__all__ = [
	'InsightBotConfig',
	'Config',
	'OldConfig',
	'FlatEnvConfig',
	'CONFIG',
	'load_insightbot_config',
	'Settings',
	'get_settings',
]
