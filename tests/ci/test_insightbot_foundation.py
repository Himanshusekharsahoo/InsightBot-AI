"""Tests for InsightBot AI Foundation & Browser Engine Integration."""

from insightbot import (
	Browser,
	BrowserResearchAgent,
	BrowserSession,
	InsightBotAgent,
	KnowledgeStore,
	__version__,
)
from insightbot.core import (
	InsightBotTask,
	ResearchRequest,
	ResearchResult,
	WebEvidence,
)
from insightbot.engine.browser import (
	BrowserResearchAgent as EngineBrowserResearchAgent,
)
from insightbot.engine.browser import (
	BrowserSession as EngineBrowserSession,
)
from insightbot.engine.browser import (
	extract_evidence_from_history,
)
from insightbot.security import SecurityPolicy, SensitiveDataSanitizer


def test_insightbot_metadata():
	"""Verify InsightBot version and base imports."""
	assert __version__ == '1.0.0'
	assert issubclass(BrowserResearchAgent, EngineBrowserResearchAgent)
	assert BrowserSession is EngineBrowserSession
	assert Browser is BrowserSession


def test_core_models():
	"""Verify Pydantic models for research requests and results."""
	request = ResearchRequest(query='Quantum computing state of the art', max_steps=15)
	assert request.query == 'Quantum computing state of the art'
	assert request.max_steps == 15

	evidence = WebEvidence(
		url='https://arxiv.org/abs/1234',
		title='Quantum Paper',
		snippet='Achieved quantum supremacy in sample testing.',
		relevance_score=0.98,
	)
	assert evidence.url == 'https://arxiv.org/abs/1234'
	assert evidence.relevance_score == 0.98

	result = ResearchResult(
		query=request.query,
		summary='Quantum supremacy demonstrated in 2026.',
		evidence=[evidence],
		sources=[evidence.url],
		is_successful=True,
		duration_seconds=12.4,
	)
	assert result.query == request.query
	assert len(result.evidence) == 1
	assert result.sources == [evidence.url]
	assert result.is_successful is True

	task = InsightBotTask(task_id='task-001', instruction='Investigate new battery tech', result=result)
	assert task.task_id == 'task-001'
	assert task.result.summary == result.summary


def test_security_domain_filtering():
	"""Verify domain filtering and security policies."""
	policy = SecurityPolicy(
		allowed_domains=['*.nature.com', 'arxiv.org', 'wikipedia.org'],
		prohibited_domains=['blocked.nature.com', 'malicious.com'],
	)
	domain_filter = policy.create_domain_filter()

	# Whitelisted domains
	assert domain_filter.is_domain_allowed('https://www.nature.com/articles/123') is True
	assert domain_filter.is_domain_allowed('https://arxiv.org/abs/456') is True
	assert domain_filter.is_domain_allowed('https://en.wikipedia.org/wiki/AI') is True

	# Blacklisted / unlisted domains
	assert domain_filter.is_domain_allowed('https://blocked.nature.com/bad') is False
	assert domain_filter.is_domain_allowed('https://malicious.com') is False
	assert domain_filter.is_domain_allowed('https://untrusted-site.com') is False


def test_sensitive_data_sanitizer():
	"""Verify credentials and secret redaction."""
	raw_text = 'Connected with api_key=secret_token_12345 to host https://internal.corp'
	sanitized = SensitiveDataSanitizer.sanitize(raw_text, sensitive_terms=['secret_token_12345'])
	assert 'secret_token_12345' not in sanitized
	assert '[REDACTED]' in sanitized


def test_knowledge_store(tmp_path):
	"""Verify local team knowledge store indexing and querying."""
	store = KnowledgeStore(storage_dir=tmp_path)
	assert len(store.get_all()) == 0

	doc = store.store_document(
		doc_id='doc-101',
		title='Deep Learning Breakthroughs',
		source_url='https://nature.com/article/ai',
		content='Novel transformers achieve state of the art on reasoning tasks.',
		metadata={'category': 'ai'},
	)
	assert doc.doc_id == 'doc-101'
	assert len(store.get_all()) == 1

	# Search matches
	matches = store.search('transformers')
	assert len(matches) == 1
	assert matches[0].doc_id == 'doc-101'

	# Non-matching search
	empty = store.search('nonexistent topic')
	assert len(empty) == 0


def test_evidence_extraction_helper():
	"""Verify helper extracting evidence from mock agent history."""

	class MockHistory:
		def urls(self):
			return ['https://example.com/research', 'https://example.com/details']

		def extracted_content(self):
			return ['Key discovery point 1', 'Key discovery point 2']

		def screenshot_paths(self):
			return ['/tmp/screen1.png', '/tmp/screen2.png']

		def final_result(self):
			return 'Overall synthesized research summary'

		def is_successful(self):
			return True

		def errors(self):
			return []

	result = extract_evidence_from_history(
		query='Find new discoveries',
		history=MockHistory(),
		duration_seconds=5.2,
	)

	assert result.query == 'Find new discoveries'
	assert result.summary == 'Overall synthesized research summary'
	assert len(result.evidence) == 2
	assert result.evidence[0].url == 'https://example.com/research'
	assert result.evidence[0].snippet == 'Key discovery point 1'
	assert result.is_successful is True
	assert result.duration_seconds == 5.2


from unittest.mock import MagicMock


def test_browser_research_agent_init():
	"""Verify BrowserResearchAgent initializes with InsightBot logging name."""
	mock_llm = MagicMock()
	agent = BrowserResearchAgent(task='Search for renewable energy advancements', llm=mock_llm)
	assert agent.logger.name == 'insightbot.engine.browser.BrowserResearchAgent'
	assert agent.task == 'Search for renewable energy advancements'


def test_orchestrator_agent_init(tmp_path):
	"""Verify InsightBotAgent orchestrator initialization and sub-engine linkage."""
	mock_llm = MagicMock()
	ks = KnowledgeStore(storage_dir=tmp_path)
	orchestrator = InsightBotAgent(
		task='Investigate quantum semiconductors',
		llm=mock_llm,
		knowledge_store=ks,
	)
	assert orchestrator.task == 'Investigate quantum semiconductors'
	assert orchestrator.knowledge_store is ks
	assert isinstance(orchestrator._sub_engine, BrowserResearchAgent)
