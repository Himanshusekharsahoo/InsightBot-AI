"""Tests for AI provider resilience, retry/backoff, and research status distinction."""

from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

from insightbot.api.routes import create_app, get_research_engine, get_store
from insightbot.db.in_memory import InMemoryVectorStore
from insightbot.knowledge.embeddings import MockEmbeddingModel
from insightbot.knowledge.store import KnowledgeStore
from insightbot.orchestrator.llm import GoogleLLMProvider, LLMProviderError
from insightbot.orchestrator.research_engine import (
	NormalizedEvidence,
	ResearchCitation,
	ResearchEngine,
	ResearchEngineResult,
)


@pytest.mark.asyncio
async def test_google_llm_provider_retries_transient_503():
	"""Verify GoogleLLMProvider retries on transient 503 and returns text upon recovery."""
	provider = GoogleLLMProvider(api_key='AIzaSyTestMockKeyForRetry123456789')

	mock_client = MagicMock()
	mock_response = MagicMock()
	mock_response.text = 'Recovered after 503 spike'

	# Fail on attempt 0 with 503, succeed on attempt 1
	mock_client.models.generate_content.side_effect = [
		Exception('503 UNAVAILABLE: Model is overloaded'),
		mock_response,
	]
	provider._client = mock_client

	with patch('asyncio.sleep') as mock_sleep:
		result = await provider.generate_response('Test prompt')
		assert result == 'Recovered after 503 spike'
		assert mock_client.models.generate_content.call_count == 2
		mock_sleep.assert_called_once_with(2.0)


@pytest.mark.asyncio
async def test_google_llm_provider_retries_transient_429():
	"""Verify GoogleLLMProvider retries on transient 429 rate limits."""
	provider = GoogleLLMProvider(api_key='AIzaSyTestMockKeyForRetry123456789')

	mock_client = MagicMock()
	mock_response = MagicMock()
	mock_response.text = 'Recovered after 429 rate limit'

	# Fail on attempt 0 with 429, succeed on attempt 1
	mock_client.models.generate_content.side_effect = [
		Exception('429 RESOURCE_EXHAUSTED: Rate limit exceeded'),
		mock_response,
	]
	provider._client = mock_client

	with patch('asyncio.sleep') as mock_sleep:
		result = await provider.generate_response('Test prompt')
		assert result == 'Recovered after 429 rate limit'
		assert mock_client.models.generate_content.call_count == 2
		mock_sleep.assert_called_once_with(2.0)


@pytest.mark.asyncio
async def test_google_llm_provider_exhausted_retries_raises_sanitized():
	"""Verify GoogleLLMProvider terminates after max_retries and sanitizes error text."""
	provider = GoogleLLMProvider(api_key='AIzaSyTestMockKeyForRetry123456789')

	mock_client = MagicMock()
	mock_client.models.generate_content.side_effect = Exception('503 UNAVAILABLE with key AIzaSySuperSecretKey123456')
	provider._client = mock_client

	with patch('asyncio.sleep'):
		with pytest.raises(LLMProviderError) as exc_info:
			await provider.generate_response('Test prompt')

		# Called 3 times (attempt 0, 1, 2)
		assert mock_client.models.generate_content.call_count == 3
		err_msg = str(exc_info.value)
		assert 'AIzaSySuperSecretKey123456' not in err_msg
		assert '[REDACTED]' in err_msg


@pytest.mark.asyncio
async def test_research_engine_handles_llm_failure_without_fabrication():
	"""Verify ResearchEngine marks result ungrounded with empty citations when LLM fails."""
	engine = ResearchEngine()

	mock_llm = MagicMock()

	async def failing_generate(*args, **kwargs):
		raise LLMProviderError('Google completion failed: 503 UNAVAILABLE [REDACTED]')

	mock_llm.generate_response = failing_generate
	engine._llm_provider = mock_llm

	# Seed with simulated internal evidence
	simulated_ev = [
		NormalizedEvidence(
			source_type='internal_knowledge',
			source_uri='falcon_telemetry.txt',
			content='Falcon operates at 26.5 GHz',
			document_id='doc-1',
			chunk_id='chk-1',
		)
	]

	async def mock_retrieve(*args, **kwargs):
		return simulated_ev

	engine.retrieve_internal = mock_retrieve

	result = await engine.conduct_research(query='What frequency?', include_web=False)

	# Must not be grounded, citations must be empty, must state error
	assert result.is_grounded is False
	assert result.sufficient_evidence is False
	assert len(result.citations) == 0
	assert 'AI provider error:' in result.summary
	assert '503 UNAVAILABLE' in result.summary


def test_api_research_endpoint_distinguishes_provider_error_vs_grounded(tmp_path):
	"""Verify /research route returns 'provider_unavailable' on upstream failure and 'completed' on grounded."""
	app = create_app()
	mem_store = InMemoryVectorStore()
	mock_embed = MockEmbeddingModel(dimension=64)
	ks = KnowledgeStore(storage_dir=tmp_path, vector_store=mem_store, embedding_provider=mock_embed)
	app.dependency_overrides[get_store] = lambda: ks

	test_client = TestClient(app)

	class FailingResearchEngine:
		async def conduct_research(self, *args, **kwargs):
			return ResearchEngineResult(
				query='Test',
				summary='AI provider error: Google completion failed: 503 UNAVAILABLE.',
				internal_evidence=[
					NormalizedEvidence(
						source_type='internal_knowledge',
						source_uri='falcon.txt',
						content='Data',
					)
				],
				citations=[],
				is_grounded=False,
				sufficient_evidence=False,
				metadata={'error': 'AI provider error: 503'},
			)

	class GroundedResearchEngine:
		async def conduct_research(self, *args, **kwargs):
			return ResearchEngineResult(
				query='Test',
				summary='Grounded answer citing falcon.txt.',
				internal_evidence=[
					NormalizedEvidence(
						source_type='internal_knowledge',
						source_uri='falcon.txt',
						content='Data',
					)
				],
				citations=[
					ResearchCitation(
						source_type='internal_knowledge',
						source_uri='falcon.txt',
						snippet='Data',
					)
				],
				is_grounded=True,
				sufficient_evidence=True,
			)

	try:
		app.dependency_overrides[get_research_engine] = lambda: FailingResearchEngine()
		res = test_client.post('/research', json={'task': 'Frequency check', 'sync': True})
		assert res.status_code == 200
		data = res.json()
		assert data['status'] == 'provider_unavailable'
		assert data['sources'] == []  # Must NOT attribute sources when provider failed!
		assert 'AI provider error' in data['summary']

		app.dependency_overrides[get_research_engine] = lambda: GroundedResearchEngine()
		res2 = test_client.post('/research', json={'task': 'Frequency check', 'sync': True})
		assert res2.status_code == 200
		data2 = res2.json()
		assert data2['status'] == 'completed'
		assert data2['sources'] == ['falcon.txt']
	finally:
		app.dependency_overrides.clear()
