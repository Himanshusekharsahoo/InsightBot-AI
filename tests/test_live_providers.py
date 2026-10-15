"""Live External AI Provider Integration Tests.

These tests make real network calls to Google Gemini and OpenAI APIs.
They are separated from offline CI and strictly skipped unless:
    INSIGHTBOT_ENABLE_LIVE_TESTS=1
is explicitly set in the execution environment.
"""

import os

import pytest

from insightbot.knowledge.embeddings import (
	EmbeddingConfigurationError,
	EmbeddingProviderError,
	GoogleEmbeddingModel,
	OpenAIEmbeddingModel,
)
from insightbot.orchestrator.llm import (
	GoogleLLMProvider,
	LLMConfigurationError,
	LLMProviderError,
	OpenAILLMProvider,
)


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_google_gemini_integration():
	"""Live verification of real Google Gemini LLM and embedding generation."""
	api_key = os.getenv('GOOGLE_API_KEY')
	if not api_key:
		pytest.skip('GOOGLE_API_KEY not configured in environment.')

	# 1. Embeddings verification (768 dimensions)
	embed_model = GoogleEmbeddingModel(
		model_name=os.getenv('INSIGHTBOT_EMBEDDING_MODEL') or 'gemini-embedding-001',
		dimension=768,
		api_key=api_key,
	)
	try:
		vec = await embed_model.embed_text('InsightBot production readiness test.')
		assert isinstance(vec, list)
		assert len(vec) == 768, f'Expected 768 dimensions, got {len(vec)}'
		assert all(isinstance(x, float) for x in vec)
	except Exception as e:
		if '503' in str(e) or 'UNAVAILABLE' in str(e):
			pytest.skip(f'Google embedding endpoint temporarily unavailable (503 spike): {e}')
		raise

	# 2. LLM synthesis verification
	llm = GoogleLLMProvider(
		model_name=os.getenv('INSIGHTBOT_LLM_MODEL') or 'gemini-flash-latest',
		api_key=api_key,
	)
	try:
		resp = await llm.generate_response('Respond with exact word: READY')
		assert 'READY' in resp.upper()
	except Exception as e:
		if '503' in str(e) or 'UNAVAILABLE' in str(e):
			pytest.skip(f'Google completion endpoint temporarily unavailable (503 spike): {e}')
		raise


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_openai_integration():
	"""Live verification of real OpenAI LLM and embedding generation.

	Distinguishes credential absence, quota exhaustion (BLOCKED), and successful completion.
	"""
	api_key = os.getenv('OPENAI_API_KEY')
	if not api_key:
		pytest.skip('OPENAI_API_KEY not configured in environment.')

	# 1. Embeddings verification
	embed_model = OpenAIEmbeddingModel(
		model_name='text-embedding-3-small',
		dimension=1536,
		api_key=api_key,
	)
	try:
		vec = await embed_model.embed_text('InsightBot production readiness test.')
		assert isinstance(vec, list)
		assert len(vec) == 1536
	except (EmbeddingProviderError, EmbeddingConfigurationError) as e:
		err_str = str(e)
		if 'insufficient_quota' in err_str or 'credit_balance_exhausted' in err_str or 'rate limit' in err_str.lower():
			pytest.skip('OpenAI live embedding BLOCKED: account quota exhausted (429 RateLimitError).')
		raise

	# 2. LLM synthesis verification
	llm = OpenAILLMProvider(
		model_name='gpt-4o-mini',
		api_key=api_key,
	)
	try:
		resp = await llm.generate_response('Respond with exact word: READY')
		assert 'READY' in resp.upper()
	except (LLMProviderError, LLMConfigurationError) as e:
		err_str = str(e)
		if 'insufficient_quota' in err_str or 'credit_balance_exhausted' in err_str or 'rate limit' in err_str.lower():
			pytest.skip('OpenAI live LLM BLOCKED: account quota exhausted (429 RateLimitError).')
		raise
