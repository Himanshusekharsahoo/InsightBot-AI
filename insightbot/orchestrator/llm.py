"""LLM Provider Abstraction and Implementations for InsightBot AI.

Provides BaseLLMProvider interface, OpenAILLMProvider, GoogleLLMProvider,
and MockLLMProvider for deterministic offline testing.
"""

import asyncio
import logging
import math
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Union

from insightbot.config.settings import get_settings
from insightbot.security.policy import SensitiveDataSanitizer

logger = logging.getLogger('insightbot.orchestrator.llm')


class LLMProviderError(Exception):
	"""Base exception for LLM provider errors."""

	pass


class LLMConfigurationError(LLMProviderError):
	"""Raised when provider configuration or credentials are missing."""

	pass


class BaseLLMProvider(ABC):
	"""Abstract interface for LLM synthesis providers."""

	@property
	@abstractmethod
	def provider_name(self) -> str:
		"""Identifier of the LLM provider."""
		pass

	@property
	@abstractmethod
	def model_name(self) -> str:
		"""Identifier of the model."""
		pass

	@abstractmethod
	async def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
		"""Generate a text response given prompt and optional system instructions."""
		pass


class OpenAILLMProvider(BaseLLMProvider):
	"""Production-grade OpenAI LLM provider using official AsyncOpenAI client."""

	def __init__(
		self,
		model_name: Optional[str] = None,
		api_key: Optional[str] = None,
		base_url: Optional[str] = None,
		timeout: Optional[float] = None,
		temperature: float = 0.0,
	) -> None:
		settings = get_settings()
		self._model_name = model_name or settings.llm_model
		self._api_key = api_key or os.getenv('OPENAI_API_KEY')
		self._base_url = base_url or os.getenv('OPENAI_BASE_URL')
		self._timeout = timeout or settings.ai_request_timeout
		self.temperature = temperature
		self._client: Any = None

		if not self._api_key:
			raise LLMConfigurationError(
				"OpenAI API key is missing. Set the 'OPENAI_API_KEY' environment variable or configure INSIGHTBOT_LLM_PROVIDER=mock for offline development."
			)

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
				raise LLMProviderError("The 'openai' package is required for OpenAI synthesis. Run 'uv add openai'.") from e
		return self._client

	@property
	def provider_name(self) -> str:
		return 'openai'

	@property
	def model_name(self) -> str:
		return self._model_name

	async def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
		if not prompt or not prompt.strip():
			return ''

		client = self._get_client()
		messages: list[dict[str, str]] = []
		if system_prompt:
			messages.append({'role': 'system', 'content': system_prompt})
		messages.append({'role': 'user', 'content': prompt})

		try:
			response = await client.chat.completions.create(
				model=self._model_name,
				messages=messages,
				temperature=self.temperature,
				timeout=self._timeout,
			)
			if not response.choices:
				raise LLMProviderError(f"OpenAI returned empty choices array for model '{self._model_name}'")
			content = response.choices[0].message.content
			if content is None:
				raise LLMProviderError(f"OpenAI returned null content for model '{self._model_name}'")
			return content.strip()

		except Exception as e:
			err_cls = e.__class__.__name__
			err_str = str(e)
			if (
				'Authentication' in err_cls
				or 'incorrect api key' in err_str.lower()
				or 'invalid api key' in err_str.lower()
				or '401' in err_str
			):
				raise LLMConfigurationError('OpenAI authentication failed. Verify OPENAI_API_KEY.') from e
			elif 'RateLimit' in err_cls or '429' in err_str:
				raise LLMProviderError('OpenAI rate limit exceeded. Retry after backoff.') from e
			elif 'Timeout' in err_cls or isinstance(e, asyncio.TimeoutError):
				raise LLMProviderError(f'OpenAI completion request timed out after {self._timeout}s.') from e
			elif 'Connection' in err_cls:
				raise LLMProviderError('Failed to connect to OpenAI API endpoint.') from e
			sanitized = SensitiveDataSanitizer.sanitize(str(e))
			raise LLMProviderError(f"OpenAI completion request failed for model '{self._model_name}': {sanitized}") from e


class GoogleLLMProvider(BaseLLMProvider):
	"""Google Gemini LLM provider using google-genai client."""

	def __init__(
		self,
		model_name: Optional[str] = None,
		api_key: Optional[str] = None,
		timeout: Optional[float] = None,
		temperature: float = 0.0,
	) -> None:
		settings = get_settings()
		self._model_name = model_name or os.getenv('INSIGHTBOT_LLM_MODEL') or 'gemini-flash-latest'
		self._api_key = api_key or os.getenv('GOOGLE_API_KEY')

		self._timeout = timeout or settings.ai_request_timeout
		self.temperature = temperature
		self._client: Any = None

		if not self._api_key:
			raise LLMConfigurationError("Google API key is missing. Set 'GOOGLE_API_KEY' environment variable.")

	def _get_client(self) -> Any:
		if self._client is None:
			try:
				from google import genai

				self._client = genai.Client(api_key=self._api_key)
			except ImportError as e:
				raise LLMProviderError("The 'google-genai' package is required.") from e
		return self._client

	@property
	def provider_name(self) -> str:
		return 'google'

	@property
	def model_name(self) -> str:
		return self._model_name

	async def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
		if not prompt or not prompt.strip():
			return ''
		client = self._get_client()
		max_retries = 2
		for attempt in range(max_retries + 1):
			try:
				full_prompt = f'System: {system_prompt}\n\nUser: {prompt}' if system_prompt else prompt
				response = await asyncio.to_thread(
					client.models.generate_content,
					model=self._model_name,
					contents=full_prompt,
				)
				text = getattr(response, 'text', None)
				if not text:
					raise LLMProviderError(f"Google provider returned empty response for model '{self._model_name}'")
				return str(text).strip()
			except Exception as e:
				if isinstance(e, LLMProviderError):
					raise
				err_str = str(e)
				err_cls = e.__class__.__name__
				is_hard_quota = 'quota' in err_str.lower() or 'quotafailure' in err_str.lower()
				is_transient = (
					not is_hard_quota
					and (
						'503' in err_str
						or 'UNAVAILABLE' in err_str
						or '429' in err_str
						or 'ResourceExhausted' in err_cls
						or 'RESOURCE_EXHAUSTED' in err_str
					)
				)
				if is_transient and attempt < max_retries:
					sleep_secs = 2.0 * (attempt + 1)
					logger.warning(
						f'Google completion received transient upstream error ({err_cls}), '
						f'retrying ({attempt + 1}/{max_retries}) after {sleep_secs}s backoff...'
					)
					await asyncio.sleep(sleep_secs)
					continue

				sanitized = SensitiveDataSanitizer.sanitize(err_str)
				if (
					'API_KEY_INVALID' in err_str
					or 'PermissionDenied' in err_cls
					or '401' in err_str
					or '403' in err_str
					or 'Unauthenticated' in err_str
				):
					raise LLMConfigurationError(f'Google authentication failed: {sanitized}') from e
				elif is_hard_quota or 'ResourceExhausted' in err_cls or '429' in err_str or 'RateLimit' in err_cls:
					raise LLMProviderError(f'Google quota or rate limit exceeded: {sanitized}') from e
				elif 'Timeout' in err_cls or isinstance(e, asyncio.TimeoutError):
					raise LLMProviderError(f'Google completion timed out after {self._timeout}s.') from e
				raise LLMProviderError(f'Google completion failed: {sanitized}') from e
		return ''


class MockLLMProvider(BaseLLMProvider):
	"""TEST/DEVELOPMENT PROVIDER ONLY: Deterministic pseudo-LLM responder for offline testing.

	Generates deterministic grounded answers citing retrieved source chunks without making
	external network calls.
	"""

	is_mock: bool = True

	def __init__(
		self,
		model_name: str = 'mock-llm',
		custom_responder: Optional[Callable[[str, Optional[str]], Union[str, Any]]] = None,
	) -> None:
		self._model_name = model_name
		self._custom_responder = custom_responder

	@property
	def provider_name(self) -> str:
		return 'mock'

	@property
	def model_name(self) -> str:
		return self._model_name

	async def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
		if self._custom_responder is not None:
			res = self._custom_responder(prompt, system_prompt)
			if asyncio.iscoroutine(res):
				return str(await res)
			return str(res)

		# Check if prompt contains untrusted evidence
		if 'UNTRUSTED_REFERENCE_EVIDENCE' not in prompt and 'Evidence' not in prompt:
			return 'The internal knowledge base does not contain enough information to answer this question.'

		# Isolate untrusted reference evidence if present so prior conversation turns are never cited as evidence
		evidence_section = prompt
		if '<UNTRUSTED_REFERENCE_EVIDENCE>' in prompt:
			parts = prompt.split('<UNTRUSTED_REFERENCE_EVIDENCE>', 1)
			if len(parts) > 1 and '</UNTRUSTED_REFERENCE_EVIDENCE>' in parts[1]:
				evidence_section = parts[1].split('</UNTRUSTED_REFERENCE_EVIDENCE>', 1)[0]
			else:
				evidence_section = parts[1]

		# Extract chunks and URLs strictly from untrusted reference evidence
		doc_ids = re.findall(r'(?:Document ID:|doc_id[=:])\s*([^\s\n\r,\]]+)', evidence_section)
		chunk_ids = re.findall(r'(?:Chunk ID:|chunk_id[=:])\s*([^\s\n\r,\]]+)', evidence_section)
		filenames = re.findall(r'(?:Source File:|Source\s+\d+:|source_uri[=:])\s*([^\s\n\r,\]]+)', evidence_section)
		urls = re.findall(r'(?:Source URL:|Evidence ID:[^\n]+\nSource URL:)\s*([^\s\n\r,\]]+)', evidence_section)

		if not chunk_ids and not doc_ids and not urls:
			return 'The internal knowledge base does not contain enough information to answer this question.'

		# Verify relevance: If user question keywords have zero overlap with evidence content, return insufficient information
		question_match = re.search(r'User Question:\s*([^\n\r]+)', prompt)
		if question_match:
			q_text = question_match.group(1).lower()
			stop_words = {
				'what', 'is', 'the', 'of', 'in', 'and', 'to', 'a', 'an', 'for', 'that', 'this',
				'how', 'when', 'where', 'which', 'who', 'why', 'are', 'was', 'were', 'does', 'did',
			}
			q_words = {w for w in re.findall(r'\b[a-z]{3,}\b', q_text) if w not in stop_words}
			from insightbot.knowledge.embeddings import _CONCEPT_MAP, _stem

			expanded_q_words = set(q_words)
			for w in q_words:
				stemmed = _stem(w)
				expanded_q_words.add(stemmed)
				for concept, related in _CONCEPT_MAP.items():
					if w == concept or stemmed == concept or w in related:
						expanded_q_words.update(related)

			ev_words = set(re.findall(r'\b[a-z0-9_\-\.]{2,}\b', evidence_section.lower()))
			ev_stems = {_stem(w) for w in ev_words}
			all_ev_terms = ev_words | ev_stems
			if expanded_q_words and not (expanded_q_words & all_ev_terms):
				return 'The internal knowledge base does not contain enough information to answer this question.'
		else:
			question_match = None
			expanded_q_words = set()

		# Parse blocks and cite only chunks that contain relevant factual concepts
		blocks = re.split(r'\n+---\n+', evidence_section)
		candidate_blocks = []
		max_overlap = 0

		for block in blocks:
			c_match = re.search(r'(?:Chunk ID:|chunk_id[=:])\s*([^\s\n\r,\]]+)', block)
			d_match = re.search(r'(?:Document ID:|doc_id[=:])\s*([^\s\n\r,\]]+)', block)
			f_match = re.search(r'(?:Source File:|Source\s+\d+:|source_uri[=:])\s*([^\s\n\r,\]]+)', block)

			if not c_match and not d_match:
				continue

			c_id = c_match.group(1) if c_match else 'chk-unknown'
			d_id = d_match.group(1) if d_match else 'doc-unknown'
			f_name = f_match.group(1) if f_match else 'source.txt'

			overlap = 1
			if question_match and expanded_q_words:
				block_words = set(re.findall(r'\b[a-z0-9_\-\.]{2,}\b', block.lower()))
				block_stems = {_stem(w) for w in block_words}
				all_block_terms = block_words | block_stems
				overlap = len(expanded_q_words & all_block_terms)

			if overlap > 0:
				max_overlap = max(max_overlap, overlap)
				candidate_blocks.append((overlap, c_id, d_id, f_name, block))

		if question_match and expanded_q_words:
			min_required = 2 if len(q_words) >= 2 else 1
			if max_overlap < min_required:
				return 'The internal knowledge base does not contain enough information to answer this question.'

		citations = []
		relevant_snippets = []

		# Filter out weak distractor chunks that fall below 60% of max overlap
		threshold = max(1, math.ceil(max_overlap * 0.6))
		for overlap, c_id, d_id, f_name, block in candidate_blocks:
			if overlap < threshold:
				continue

			citations.append(f'[Document: {f_name}, doc_id: {d_id}, chunk_id: {c_id}]')
			content_match = re.search(
				r'Content:\s*\n?(.*?)(?:\n\n---|\n\nUser Question:|</UNTRUSTED_REFERENCE_EVIDENCE>|\Z)',
				block,
				re.DOTALL,
			)
			if content_match:
				relevant_snippets.append(content_match.group(1).strip())

		for u in urls[:3]:
			citations.append(f'[Web: {u}]')

		if not citations:
			return 'The internal knowledge base does not contain enough information to answer this question.'

		citation_str = ' '.join(citations[:4])
		evidence_snippet = f' {" ".join(relevant_snippets)}' if relevant_snippets else ''
		source_desc = 'verified internal knowledge' if (chunk_ids or doc_ids) else 'verified web research'
		return f'Based on {source_desc}: {citation_str}\n\nEvidence confirms the requested facts.{evidence_snippet}'


def get_llm_provider(
	provider: Optional[str] = None,
	model: Optional[str] = None,
	api_key: Optional[str] = None,
	base_url: Optional[str] = None,
	timeout: Optional[float] = None,
) -> BaseLLMProvider:
	"""Factory returning configured BaseLLMProvider implementation."""
	settings = get_settings()
	prov = (provider or settings.llm_provider).lower()
	model_name = model or settings.llm_model

	if prov == 'mock':
		logger.info('Using MockLLMProvider: deterministic offline responder for development and testing.')
		return MockLLMProvider(model_name=model_name)
	elif prov == 'openai':
		return OpenAILLMProvider(
			model_name=model_name,
			api_key=api_key,
			base_url=base_url,
			timeout=timeout,
		)
	elif prov in ('google', 'gemini'):
		g_model = model or (None if settings.llm_model.startswith('gpt-') else settings.llm_model)
		return GoogleLLMProvider(
			model_name=g_model,
			api_key=api_key,
			timeout=timeout,
		)

	else:
		raise LLMConfigurationError(f"Unknown LLM provider '{prov}'. Supported: openai, google, mock.")


__all__ = [
	'LLMProviderError',
	'LLMConfigurationError',
	'BaseLLMProvider',
	'OpenAILLMProvider',
	'GoogleLLMProvider',
	'MockLLMProvider',
	'get_llm_provider',
]
