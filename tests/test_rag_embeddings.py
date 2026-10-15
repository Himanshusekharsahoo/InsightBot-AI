"""Unit tests for embedding providers, validation, and error handling."""

import math
from unittest.mock import AsyncMock, MagicMock

import pytest

from insightbot.knowledge.embeddings import (
	EmbeddingConfigurationError,
	EmbeddingProviderError,
	GoogleEmbeddingModel,
	MockEmbeddingModel,
	OpenAIEmbeddingModel,
	get_embedding_provider,
)


@pytest.mark.asyncio
async def test_mock_embedding_provider():
	model = MockEmbeddingModel(dimension=1536)
	assert model.dimension == 1536
	assert model.model_name == 'mock-embedding'
	assert getattr(model, 'is_mock', False) is True

	# Single embedding
	vec = await model.embed_text('InsightBot private knowledge')
	assert len(vec) == 1536

	# Unit vector check
	norm = math.sqrt(sum(x * x for x in vec))
	assert pytest.approx(norm, 1e-5) == 1.0

	# Batch embedding
	texts = ['Text one', 'Text two', 'Text three']
	batch = await model.embed_batch(texts)
	assert len(batch) == 3
	for b in batch:
		assert len(b) == 1536


@pytest.mark.asyncio
async def test_mock_embedding_deterministic():
	model = MockEmbeddingModel(dimension=512)
	v1 = await model.embed_text('deterministic content')
	v2 = await model.embed_text('deterministic content')
	assert v1 == v2

	v3 = await model.embed_text('other content')
	assert v1 != v3


def test_openai_embedding_missing_credentials(monkeypatch):
	monkeypatch.delenv('OPENAI_API_KEY', raising=False)
	with pytest.raises(EmbeddingConfigurationError) as exc_info:
		OpenAIEmbeddingModel(api_key='')
	assert 'API key is missing' in str(exc_info.value)


def test_google_embedding_missing_credentials(monkeypatch):
	monkeypatch.delenv('GOOGLE_API_KEY', raising=False)
	with pytest.raises(EmbeddingConfigurationError) as exc_info:
		GoogleEmbeddingModel(api_key='')
	assert 'Google API key is missing' in str(exc_info.value)


def test_get_embedding_provider_factory():
	prov = get_embedding_provider(provider='mock', dimension=768)
	assert isinstance(prov, MockEmbeddingModel)
	assert prov.dimension == 768

	with pytest.raises(EmbeddingConfigurationError):
		get_embedding_provider(provider='unknown_provider')


@pytest.mark.asyncio
async def test_openai_embedding_mocked_success():
	"""Verify OpenAIEmbeddingModel parses valid responses and respects dimension."""
	model = OpenAIEmbeddingModel(
		model_name='text-embedding-3-small',
		dimension=4,
		api_key='sk-test-mocked-key-1234567890',
	)

	# Mock AsyncOpenAI client
	mock_client = MagicMock()
	item1 = MagicMock()
	item1.index = 0
	item1.embedding = [0.1, 0.2, 0.3, 0.4]

	item2 = MagicMock()
	item2.index = 1
	item2.embedding = [0.5, 0.6, 0.7, 0.8]

	mock_response = MagicMock()
	mock_response.data = [item2, item1]  # Out of order to verify index sorting

	mock_client.embeddings.create = AsyncMock(return_value=mock_response)
	model._client = mock_client

	vectors = await model.embed_batch(['First sentence', 'Second sentence'])
	assert len(vectors) == 2
	assert vectors[0] == [0.1, 0.2, 0.3, 0.4]
	assert vectors[1] == [0.5, 0.6, 0.7, 0.8]

	# Verify create called with dimensions parameter
	mock_client.embeddings.create.assert_called_once()
	call_kwargs = mock_client.embeddings.create.call_args.kwargs
	assert call_kwargs['model'] == 'text-embedding-3-small'
	assert call_kwargs['dimensions'] == 4
	assert call_kwargs['input'] == ['First sentence', 'Second sentence']


@pytest.mark.asyncio
async def test_openai_embedding_dimension_mismatch_rejection():
	"""Verify OpenAIEmbeddingModel rejects responses whose dimensions mismatch."""
	model = OpenAIEmbeddingModel(
		model_name='text-embedding-3-small',
		dimension=4,
		api_key='sk-test-mocked-key-1234567890',
	)

	mock_client = MagicMock()
	item = MagicMock()
	item.index = 0
	item.embedding = [0.1, 0.2]  # Only 2 dims instead of 4

	mock_response = MagicMock()
	mock_response.data = [item]
	mock_client.embeddings.create = AsyncMock(return_value=mock_response)
	model._client = mock_client

	with pytest.raises(EmbeddingProviderError) as exc_info:
		await model.embed_text('Sample text')
	assert 'dimension mismatch' in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_openai_embedding_non_finite_rejection():
	"""Verify OpenAIEmbeddingModel rejects NaN or Inf values."""
	model = OpenAIEmbeddingModel(
		model_name='text-embedding-3-small',
		dimension=3,
		api_key='sk-test-mocked-key-1234567890',
	)

	mock_client = MagicMock()
	item = MagicMock()
	item.index = 0
	item.embedding = [0.1, float('nan'), 0.3]

	mock_response = MagicMock()
	mock_response.data = [item]
	mock_client.embeddings.create = AsyncMock(return_value=mock_response)
	model._client = mock_client

	with pytest.raises(EmbeddingProviderError) as exc_info:
		await model.embed_text('Sample text')
	assert 'non-finite' in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_openai_embedding_rate_limit_and_secret_redaction():
	"""Verify rate limit errors and ensure API keys are never leaked in errors."""
	secret_key = 'sk-super-secret-real-api-key-999999'
	model = OpenAIEmbeddingModel(
		model_name='text-embedding-3-small',
		api_key=secret_key,
	)

	class FakeRateLimitError(Exception):
		pass

	mock_client = MagicMock()
	mock_client.embeddings.create = AsyncMock(
		side_effect=FakeRateLimitError(f'Rate limit exceeded for authorization Bearer {secret_key}')
	)
	model._client = mock_client

	with pytest.raises(EmbeddingProviderError) as exc_info:
		await model.embed_text('Rate limit test')

	err_msg = str(exc_info.value)
	# Verify the sensitive key was sanitized
	assert secret_key not in err_msg
	assert '[REDACTED]' in err_msg or 'RateLimit' in err_msg or 'rate limit' in err_msg.lower()


@pytest.mark.asyncio
async def test_google_embedding_mocked_success():
	"""Verify GoogleEmbeddingModel parses valid content responses."""
	model = GoogleEmbeddingModel(
		model_name='text-embedding-004',
		dimension=4,
		api_key='AIzaSyFakeGoogleApiKey123456789012345',
	)

	mock_client = MagicMock()
	mock_resp = MagicMock()
	mock_emb = MagicMock()
	mock_emb.values = [0.1, 0.2, 0.3, 0.4]
	mock_resp.embedding = mock_emb

	mock_client.models.embed_content = MagicMock(return_value=mock_resp)
	model._client = mock_client

	vec = await model.embed_text('Google embedding test')
	assert vec == [0.1, 0.2, 0.3, 0.4]


@pytest.mark.asyncio
async def test_google_embedding_dimension_mismatch_rejection():
	"""Verify GoogleEmbeddingModel rejects dimension mismatches."""
	model = GoogleEmbeddingModel(
		model_name='text-embedding-004',
		dimension=4,
		api_key='AIzaSyFakeGoogleApiKey123456789012345',
	)
	mock_client = MagicMock()
	mock_resp = MagicMock()
	mock_emb = MagicMock()
	mock_emb.values = [0.1, 0.2]  # 2 dims instead of 4
	mock_resp.embedding = mock_emb
	mock_client.models.embed_content = MagicMock(return_value=mock_resp)
	model._client = mock_client

	with pytest.raises(EmbeddingProviderError) as exc_info:
		await model.embed_text('test text')
	assert 'dimension mismatch' in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_google_embedding_non_finite_rejection():
	"""Verify GoogleEmbeddingModel rejects NaN/Inf values."""
	model = GoogleEmbeddingModel(
		model_name='text-embedding-004',
		dimension=3,
		api_key='AIzaSyFakeGoogleApiKey123456789012345',
	)
	mock_client = MagicMock()
	mock_resp = MagicMock()
	mock_emb = MagicMock()
	mock_emb.values = [0.1, float('nan'), 0.3]
	mock_resp.embedding = mock_emb
	mock_client.models.embed_content = MagicMock(return_value=mock_resp)
	model._client = mock_client

	with pytest.raises(EmbeddingProviderError) as exc_info:
		await model.embed_text('test text')
	assert 'non-finite' in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_google_embedding_auth_failure_actionable_error():
	"""Verify GoogleEmbeddingModel handles authentication errors with actionable message."""
	secret_key = 'AIzaSySecretFakeApiKey1234567890'
	model = GoogleEmbeddingModel(
		model_name='text-embedding-004',
		api_key=secret_key,
	)
	mock_client = MagicMock()
	mock_client.models.embed_content = MagicMock(side_effect=Exception('API_KEY_INVALID: Google API key is invalid or expired.'))
	model._client = mock_client

	with pytest.raises(EmbeddingConfigurationError) as exc_info:
		await model.embed_text('auth test')
	err_msg = str(exc_info.value)
	assert 'authentication failed' in err_msg.lower()
	assert secret_key not in err_msg


@pytest.mark.asyncio
async def test_google_embedding_rate_limit_and_timeout():
	"""Verify Google rate limit and timeout handling."""
	model = GoogleEmbeddingModel(
		model_name='text-embedding-004',
		api_key='AIzaSyFakeGoogleApiKey123456789012345',
		timeout=5.0,
	)
	mock_client = MagicMock()
	mock_client.models.embed_content = MagicMock(side_effect=Exception('ResourceExhausted: 429 Resource has been exhausted'))
	model._client = mock_client

	with pytest.raises(EmbeddingProviderError) as exc_info:
		await model.embed_text('rate limit test')
	assert 'rate limit' in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_openai_embedding_auth_failure_actionable_error():
	"""Verify OpenAIEmbeddingModel raises actionable error on auth failure."""
	secret_key = 'sk-super-secret-key-1234567890'
	model = OpenAIEmbeddingModel(
		model_name='text-embedding-3-small',
		api_key=secret_key,
	)

	class FakeAuthenticationError(Exception):
		pass

	mock_client = MagicMock()
	mock_client.embeddings.create = AsyncMock(side_effect=FakeAuthenticationError(f'Incorrect API key provided: {secret_key}'))
	model._client = mock_client

	with pytest.raises(EmbeddingConfigurationError) as exc_info:
		await model.embed_text('auth test')
	assert 'authentication failed' in str(exc_info.value).lower()
	assert secret_key not in str(exc_info.value)
