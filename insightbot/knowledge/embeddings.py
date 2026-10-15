"""Embedding Provider Abstraction and Implementations for InsightBot AI.

Provides BaseEmbeddingModel interface, OpenAIEmbeddingModel, GoogleEmbeddingModel,
and MockEmbeddingModel for offline unit testing.
"""

import asyncio
import hashlib
import logging
import math
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Optional

from insightbot.config.settings import get_settings
from insightbot.security.policy import SensitiveDataSanitizer

logger = logging.getLogger('insightbot.knowledge.embeddings')


class EmbeddingProviderError(Exception):
	"""Base exception for embedding provider errors."""

	pass


class EmbeddingConfigurationError(EmbeddingProviderError):
	"""Raised when provider configuration or credentials are missing."""

	pass


class BaseEmbeddingModel(ABC):
	"""Abstract interface for text embedding providers."""

	@property
	@abstractmethod
	def dimension(self) -> int:
		"""Dimension of the vector produced by this model."""
		pass

	@property
	@abstractmethod
	def model_name(self) -> str:
		"""Identifier of the embedding model."""
		pass

	@abstractmethod
	async def embed_text(self, text: str) -> list[float]:
		"""Generate a vector embedding for a single text string."""
		pass

	@abstractmethod
	async def embed_batch(self, texts: list[str]) -> list[list[float]]:
		"""Generate vector embeddings for a batch of text strings."""
		pass


class OpenAIEmbeddingModel(BaseEmbeddingModel):
	"""OpenAI-compatible text embedding provider using official openai async client."""

	_DIMENSIONS = {
		'text-embedding-3-small': 1536,
		'text-embedding-3-large': 3072,
		'text-embedding-ada-002': 1536,
	}

	def __init__(
		self,
		model_name: Optional[str] = None,
		dimension: Optional[int] = None,
		api_key: Optional[str] = None,
		base_url: Optional[str] = None,
		timeout: Optional[float] = None,
		max_batch_size: int = 100,
	) -> None:
		settings = get_settings()
		self._model_name = model_name or settings.embedding_model
		self._api_key = api_key or os.getenv('OPENAI_API_KEY')
		self._base_url = base_url or os.getenv('OPENAI_BASE_URL')
		self._timeout = timeout or settings.ai_request_timeout
		self.max_batch_size = max_batch_size

		if not self._api_key:
			raise EmbeddingConfigurationError(
				"OpenAI API key is missing. Set the 'OPENAI_API_KEY' environment variable or specify a configured provider."
			)

		self._dimension = dimension or self._DIMENSIONS.get(self._model_name, settings.embedding_dimension)
		self._client: Any = None

	def _get_client(self) -> Any:
		if self._client is None:
			try:
				from openai import AsyncOpenAI

				self._client = AsyncOpenAI(
					api_key=self._api_key,
					base_url=self._base_url,
					timeout=self._timeout,
				)
			except ImportError as e:
				raise EmbeddingProviderError(
					"The 'openai' package is required for OpenAI embeddings. Run 'uv add openai'."
				) from e
		return self._client

	@property
	def dimension(self) -> int:
		return self._dimension

	@property
	def model_name(self) -> str:
		return self._model_name

	async def embed_text(self, text: str) -> list[float]:
		batch = await self.embed_batch([text])
		if not batch:
			raise EmbeddingProviderError('Provider returned empty response for single embedding request.')
		return batch[0]

	async def embed_batch(self, texts: list[str]) -> list[list[float]]:
		if not texts:
			return []

		client = self._get_client()
		all_embeddings: list[list[float]] = []

		# Process in batches to stay within API batch limits
		for i in range(0, len(texts), self.max_batch_size):
			chunk = texts[i : i + self.max_batch_size]
			# OpenAI cannot embed empty strings
			cleaned_chunk = [t if t.strip() else ' ' for t in chunk]
			create_kwargs: dict[str, Any] = {
				'input': cleaned_chunk,
				'model': self._model_name,
				'timeout': self._timeout,
			}
			# Pass dimensions if supported by text-embedding-3 models
			if self._model_name.startswith('text-embedding-3'):
				create_kwargs['dimensions'] = self._dimension

			try:
				response = await client.embeddings.create(**create_kwargs)
			except Exception as e:
				err_cls = e.__class__.__name__
				err_str = str(e)
				if (
					'Authentication' in err_cls
					or 'incorrect api key' in err_str.lower()
					or 'invalid api key' in err_str.lower()
					or '401' in err_str
				):
					raise EmbeddingConfigurationError('OpenAI authentication failed. Verify OPENAI_API_KEY.') from e
				elif 'RateLimit' in err_cls or '429' in err_str:
					raise EmbeddingProviderError('OpenAI embedding rate limit exceeded.') from e
				elif 'Timeout' in err_cls or isinstance(e, asyncio.TimeoutError):
					raise EmbeddingProviderError(f'OpenAI embedding request timed out after {self._timeout}s.') from e
				elif 'Connection' in err_cls:
					raise EmbeddingProviderError('Failed to connect to OpenAI embedding API.') from e
				sanitized = SensitiveDataSanitizer.sanitize(str(e))
				raise EmbeddingProviderError(
					f"OpenAI embedding request failed for model '{self._model_name}': {sanitized}"
				) from e

			# Validate and sort by index
			sorted_data = sorted(response.data, key=lambda x: getattr(x, 'index', 0))
			for item in sorted_data:
				emb = getattr(item, 'embedding', None)
				if not isinstance(emb, list):
					raise EmbeddingProviderError(f"OpenAI returned non-list embedding object for model '{self._model_name}'")
				if len(emb) != self._dimension:
					raise EmbeddingProviderError(
						f"OpenAI embedding dimension mismatch: expected {self._dimension}, got {len(emb)} for model '{self._model_name}'"
					)
				validated_vec: list[float] = []
				for val in emb:
					if not isinstance(val, (float, int)) or math.isnan(val) or math.isinf(val):
						raise EmbeddingProviderError(f'OpenAI returned invalid non-finite numeric vector value: {val}')
					validated_vec.append(float(val))
				all_embeddings.append(validated_vec)

		if len(all_embeddings) != len(texts):
			raise EmbeddingProviderError(
				f'OpenAI batch embedding count mismatch: expected {len(texts)}, got {len(all_embeddings)}'
			)

		return all_embeddings


class GoogleEmbeddingModel(BaseEmbeddingModel):
	"""Google Gemini embedding provider using google-genai."""

	_DIMENSIONS = {
		'text-embedding-004': 768,
		'gemini-embedding-001': 768,
		'gemini-embedding-2': 768,
		'embedding-001': 768,
		'gemini-embedding-exp-0827': 768,
	}

	def __init__(
		self,
		model_name: Optional[str] = None,
		dimension: Optional[int] = None,
		api_key: Optional[str] = None,
		timeout: Optional[float] = None,
	) -> None:
		settings = get_settings()
		self._model_name = model_name or os.getenv('INSIGHTBOT_EMBEDDING_MODEL') or 'gemini-embedding-001'
		self._api_key = api_key or os.getenv('GOOGLE_API_KEY')
		self._timeout = timeout or settings.ai_request_timeout
		if not self._api_key:
			raise EmbeddingConfigurationError("Google API key is missing. Set 'GOOGLE_API_KEY' environment variable.")
		self._dimension = dimension or self._DIMENSIONS.get(self._model_name, 768)
		self._client: Any = None

	def _get_client(self) -> Any:
		if self._client is None:
			try:
				from google import genai

				self._client = genai.Client(api_key=self._api_key)
			except ImportError as e:
				raise EmbeddingProviderError('google-genai package is required. Run uv add google-genai.') from e
		return self._client

	@property
	def dimension(self) -> int:
		return self._dimension

	@property
	def model_name(self) -> str:
		return self._model_name

	async def embed_text(self, text: str) -> list[float]:
		res = await self.embed_batch([text])
		return res[0]

	async def embed_batch(self, texts: list[str]) -> list[list[float]]:
		if not texts:
			return []
		client = self._get_client()
		try:
			embeddings: list[list[float]] = []
			for t in texts:
				cleaned_text = t if t.strip() else ' '
				call_kwargs: dict[str, Any] = {
					'model': self._model_name,
					'contents': cleaned_text,
				}
				if self._dimension:
					call_kwargs['config'] = {'output_dimensionality': self._dimension}

				resp: Any = client.models.embed_content(**call_kwargs)
				vals = None
				emb_obj = getattr(resp, 'embedding', None)
				if emb_obj is not None:
					vals = getattr(emb_obj, 'values', None)
				elif hasattr(resp, 'embeddings') and resp.embeddings:
					vals = getattr(resp.embeddings[0], 'values', None)

				if vals is None or not vals:
					raise EmbeddingProviderError(
						f"Google provider returned empty or malformed embedding vector for model '{self._model_name}'"
					)
				emb_list = list(vals)
				if len(emb_list) != self._dimension:
					raise EmbeddingProviderError(
						f"Google embedding dimension mismatch: expected {self._dimension}, got {len(emb_list)} for model '{self._model_name}'"
					)

				validated_vec: list[float] = []
				for val in emb_list:
					if not isinstance(val, (float, int)) or math.isnan(val) or math.isinf(val):
						raise EmbeddingProviderError(f'Google returned non-finite numeric vector value: {val}')
					validated_vec.append(float(val))
				embeddings.append(validated_vec)
			return embeddings
		except Exception as e:
			if isinstance(e, EmbeddingProviderError):
				raise
			err_str = str(e)
			err_cls = e.__class__.__name__
			sanitized = SensitiveDataSanitizer.sanitize(err_str)
			if (
				'API_KEY_INVALID' in err_str
				or 'PermissionDenied' in err_cls
				or '401' in err_str
				or '403' in err_str
				or 'Unauthenticated' in err_str
			):
				raise EmbeddingConfigurationError(f'Google authentication failed: {sanitized}') from e
			elif 'ResourceExhausted' in err_cls or '429' in err_str or 'RateLimit' in err_cls:
				raise EmbeddingProviderError('Google embedding rate limit exceeded.') from e
			elif 'Timeout' in err_cls or isinstance(e, asyncio.TimeoutError):
				raise EmbeddingProviderError(f'Google embedding request timed out after {self._timeout}s.') from e
			raise EmbeddingProviderError(f'Google embedding failed: {sanitized}') from e


_STOPWORDS = {
	'the',
	'a',
	'an',
	'is',
	'are',
	'was',
	'were',
	'of',
	'in',
	'to',
	'for',
	'with',
	'on',
	'at',
	'by',
	'from',
	'what',
	'which',
	'who',
	'when',
	'where',
	'why',
	'how',
	'does',
	'do',
	'did',
	'as',
	'it',
	'its',
	'this',
	'that',
	'and',
	'or',
	'be',
	'been',
	'being',
	'have',
	'has',
	'had',
}

_CONCEPT_MAP: dict[str, list[str]] = {
	'database': ['database', 'postgresql', 'postgres', 'db', 'sql', 'persistence', 'pgvector'],
	'postgresql': ['database', 'postgresql', 'postgres', 'db', 'persistence'],
	'postgres': ['database', 'postgresql', 'postgres', 'db', 'persistence'],
	'deployment': ['deployment', 'deploy', 'architecture', 'infrastructure', 'docker', 'resilient'],
	'deploy': ['deployment', 'deploy', 'architecture', 'infrastructure'],
	'architecture': ['architecture', 'deployment', 'system', 'subsystem', 'overview'],
	'persistence': ['persistence', 'database', 'postgresql', 'store', 'storage'],
	'vector': ['vector', 'pgvector', 'similarity', 'cosine', 'embedding', 'hnsw', 'search'],
	'knowledge': ['knowledge', 'document', 'documents', 'ingest', 'ingestion', 'retrieval'],
	'agent': ['agent', 'browser', 'research', 'knowledgeagent', 'assistant'],
}


def _stem(word: str) -> str:
	for suffix in ('ing', 'tion', 'ment', 'ed', 'es', 's'):
		if word.endswith(suffix) and len(word) > len(suffix) + 3:
			return word[: -len(suffix)]
	return word


def _token_vector(token: str, dimension: int) -> list[float]:
	vec: list[float] = []
	seed = token.encode('utf-8')
	for i in range(dimension):
		h = hashlib.md5(seed + f':{i}'.encode('utf-8')).hexdigest()
		val = (int(h[:8], 16) / 0xFFFFFFFF) * 2.0 - 1.0
		vec.append(val)
	norm = math.sqrt(sum(x * x for x in vec))
	return [x / norm for x in vec] if norm > 0 else vec


class MockEmbeddingModel(BaseEmbeddingModel):
	"""TEST/DEVELOPMENT PROVIDER ONLY: Deterministic pseudo-embedding provider for unit tests and offline testing.

	Generates normalized vectors from token-level hashing and concept mapping
	so semantically related texts score high similarities while unrelated texts score near zero.
	NOTE: Does not provide true semantic embeddings; use real embedding models (OpenAI/Google) in production.
	"""

	is_mock: bool = True

	def __init__(self, dimension: int = 1536, model_name: str = 'mock-embedding') -> None:
		self._dimension = dimension
		self._model_name = model_name

	@property
	def dimension(self) -> int:
		return self._dimension

	@property
	def model_name(self) -> str:
		return self._model_name

	async def embed_text(self, text: str) -> list[float]:
		return self._generate_vector(text)

	async def embed_batch(self, texts: list[str]) -> list[list[float]]:
		return [self._generate_vector(t) for t in texts]

	def _generate_vector(self, text: str) -> list[float]:
		words = re.findall(r'[a-zA-Z0-9]+', text.lower())
		if not words:
			return [0.0] * self._dimension

		accum: list[float] = [0.0] * self._dimension
		expanded: list[tuple[str, float]] = []

		for raw_w in words:
			if raw_w in _STOPWORDS:
				continue
			w = _stem(raw_w)
			expanded.append((w, 1.0))
			for concept, related in _CONCEPT_MAP.items():
				if raw_w == concept or raw_w in related or w == concept:
					for r in related:
						expanded.append((f'_concept_{r}', 0.6))

		if not expanded:
			return _token_vector('stopword_neutral', self._dimension)

		for token_str, weight in expanded:
			tv = _token_vector(token_str, self._dimension)
			for i in range(self._dimension):
				accum[i] += weight * tv[i]

		norm = math.sqrt(sum(x * x for x in accum))
		if norm > 0:
			return [x / norm for x in accum]
		return accum


def get_embedding_provider(
	provider: Optional[str] = None,
	model: Optional[str] = None,
	dimension: Optional[int] = None,
	api_key: Optional[str] = None,
	base_url: Optional[str] = None,
	timeout: Optional[float] = None,
) -> BaseEmbeddingModel:
	"""Factory returning configured BaseEmbeddingModel implementation."""
	settings = get_settings()
	prov = (provider or settings.embedding_provider).lower()
	model_name = model or settings.embedding_model
	dim = dimension or settings.embedding_dimension

	if prov == 'mock':
		logger.info('Using MockEmbeddingModel: deterministic pseudo-embeddings for offline testing and development.')
		return MockEmbeddingModel(dimension=dim, model_name=model_name)
	elif prov == 'openai':
		return OpenAIEmbeddingModel(
			model_name=model_name,
			dimension=dim,
			api_key=api_key,
			base_url=base_url,
			timeout=timeout,
		)
	elif prov in ('google', 'gemini'):
		g_model = model or (None if settings.embedding_model.startswith('text-embedding-') else settings.embedding_model)
		g_dim = dimension or (768 if settings.embedding_dimension == 1536 else settings.embedding_dimension)
		return GoogleEmbeddingModel(
			model_name=g_model,
			dimension=g_dim,
			api_key=api_key,
			timeout=timeout,
		)

	else:
		raise EmbeddingConfigurationError(f"Unknown embedding provider '{prov}'. Supported: openai, google, mock.")


__all__ = [
	'EmbeddingProviderError',
	'EmbeddingConfigurationError',
	'BaseEmbeddingModel',
	'OpenAIEmbeddingModel',
	'GoogleEmbeddingModel',
	'MockEmbeddingModel',
	'get_embedding_provider',
]
