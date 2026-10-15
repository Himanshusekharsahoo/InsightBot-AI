"""Focused test suite for InsightBot AI Phase 14D:
Research Job Orchestration, Live Progress & Browser Integration.

Verifies:
1. Persistent research job lifecycle and state-transition validation.
2. Immediate job acknowledgement (queued) and background execution.
3. Synchronous execution fallback (sync=True).
4. Server-Sent Events (SSE) live progress updates and clean terminal completion.
5. Job cancellation and active task termination.
6. Multi-tenant isolation for job status, artifacts, and event streams.
7. Configured browser-agent integration via DeterministicFakeBrowserAgent.
8. Explicit unavailable status when browser agent is unconfigured (non-fabrication).
9. Process restart reconciliation for orphaned in-progress/queued jobs.
10. Schema version 4 migration and PostgreSQL baseline data preservation.
"""

import json
import os

import asyncpg
import pytest
from fastapi.testclient import TestClient

from insightbot.api.routes import app, get_agent, get_job_manager, get_research_engine, get_store
from insightbot.config.settings import get_settings
from insightbot.db.in_memory import InMemoryVectorStore
from insightbot.db.models import (
	RESEARCH_STATUS_CANCELLED,
	RESEARCH_STATUS_COMPLETED,
	RESEARCH_STATUS_IN_PROGRESS,
	RESEARCH_STATUS_INTERRUPTED,
	RESEARCH_STATUS_QUEUED,
	ResearchSession,
)
from insightbot.db.vector_store import PgVectorStore
from insightbot.engine.browser.agent import DeterministicFakeBrowserAgent
from insightbot.knowledge.embeddings import MockEmbeddingModel
from insightbot.knowledge.store import KnowledgeStore
from insightbot.orchestrator.job_manager import ResearchJobManager
from insightbot.orchestrator.knowledge_agent import KnowledgeAgent
from insightbot.orchestrator.llm import MockLLMProvider
from insightbot.orchestrator.research_engine import ResearchEngine


@pytest.fixture
def test_environment(tmp_path, monkeypatch):
	"""Configures multi-tenant test environment with in-memory store and fresh JobManager."""
	settings = get_settings()
	monkeypatch.setattr(settings, 'environment', 'development')
	monkeypatch.setattr(settings, 'api_key', 'admin-master-key')
	monkeypatch.setattr(
		settings,
		'tenant_api_keys',
		{
			'org_alpha': 'key-alpha-12345',
			'org_beta': 'key-beta-67890',
		},
	)
	monkeypatch.setattr(settings, 'default_organization_id', 'org_alpha')

	mem_store = InMemoryVectorStore()
	mock_embed = MockEmbeddingModel(dimension=64)
	store = KnowledgeStore(storage_dir=tmp_path, vector_store=mem_store, embedding_provider=mock_embed)

	mock_llm = MockLLMProvider()
	agent = KnowledgeAgent(knowledge_store=store, llm_provider=mock_llm)
	fake_browser = DeterministicFakeBrowserAgent()
	engine = ResearchEngine(knowledge_store=store, llm_provider=mock_llm, browser_agent=fake_browser)
	job_mgr = ResearchJobManager()

	app.dependency_overrides[get_store] = lambda: store
	app.dependency_overrides[get_agent] = lambda: agent
	app.dependency_overrides[get_research_engine] = lambda: engine
	app.dependency_overrides[get_job_manager] = lambda: job_mgr

	client = TestClient(app)
	yield client, store, engine, job_mgr, fake_browser

	app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_research_session_lifecycle_and_invalid_state_transitions():
	"""Verify valid lifecycle progression and rejection of invalid state transitions from terminal states."""
	store = InMemoryVectorStore()

	session = ResearchSession(
		id='res-lifecycle-001',
		organization_id='org_test',
		title='State Transition Test',
		query='Evaluate state machines',
		status=RESEARCH_STATUS_QUEUED,
	)
	await store.create_research_session(session)

	# 1. Valid: queued -> in_progress
	updated1 = await store.update_research_session(
		session_id='res-lifecycle-001',
		organization_id='org_test',
		status=RESEARCH_STATUS_IN_PROGRESS,
		progress_stage='started',
		progress_percent=10,
	)
	assert updated1 is not None
	assert updated1.status == RESEARCH_STATUS_IN_PROGRESS
	assert updated1.progress_percent == 10

	# 2. Valid: in_progress -> completed
	updated2 = await store.update_research_session(
		session_id='res-lifecycle-001',
		organization_id='org_test',
		status=RESEARCH_STATUS_COMPLETED,
		progress_stage='completed',
		progress_percent=100,
	)
	assert updated2 is not None
	assert updated2.status == RESEARCH_STATUS_COMPLETED

	# 3. Invalid: completed -> in_progress (Terminal state transition must raise ValueError)
	with pytest.raises(ValueError, match='Invalid state transition'):
		await store.update_research_session(
			session_id='res-lifecycle-001',
			organization_id='org_test',
			status=RESEARCH_STATUS_IN_PROGRESS,
		)


def test_async_research_job_acknowledgement_and_status_polling(test_environment):
	"""Verify POST /research returns immediately with status 'queued', and background execution completes."""
	client, store, _, _, _ = test_environment
	headers = {'X-API-Key': 'key-alpha-12345'}

	# 1. Initiate async research (sync=False default)
	post_resp = client.post(
		'/research',
		json={'task': 'Research distributed ledger consensus algorithms', 'max_steps': 5, 'sync': False},
		headers=headers,
	)
	assert post_resp.status_code == 200
	data = post_resp.json()
	assert data['status'] == RESEARCH_STATUS_QUEUED
	research_id = data.get('research_id')
	assert research_id is not None
	assert research_id.startswith('res-')

	# 2. Check initial status via GET /research/{id}/status
	status_resp = client.get(f'/research/{research_id}/status', headers=headers)
	assert status_resp.status_code == 200
	status_data = status_resp.json()
	assert status_data['research_id'] == research_id
	assert status_data['status'] in (RESEARCH_STATUS_QUEUED, RESEARCH_STATUS_IN_PROGRESS, RESEARCH_STATUS_COMPLETED)

	# 3. Poll until terminal state reached (background task executed by event loop)
	terminal = False
	for _ in range(30):
		poll_resp = client.get(f'/research/{research_id}/status', headers=headers)
		if poll_resp.status_code == 200:
			poll_data = poll_resp.json()
			if poll_data['is_terminal']:
				terminal = True
				assert poll_data['status'] in (RESEARCH_STATUS_COMPLETED, 'completed_unverified')
				assert poll_data['progress_percent'] == 100
				break
		import time

		time.sleep(0.05)

	assert terminal is True, 'Research job did not reach terminal state within timeout'

	# 4. Verify full session artifact is retrievable via GET /research/{id}
	get_resp = client.get(f'/research/{research_id}', headers=headers)
	assert get_resp.status_code == 200
	session_data = get_resp.json()
	assert session_data['id'] == research_id
	assert session_data['status'] == RESEARCH_STATUS_COMPLETED
	assert len(session_data['summary']) > 0


def test_sync_research_execution_backward_compatibility(test_environment):
	"""Verify POST /research with sync=True runs synchronously and returns completed response."""
	client, _, _, _, _ = test_environment
	headers = {'X-API-Key': 'key-alpha-12345'}

	post_resp = client.post(
		'/research',
		json={'task': 'Evaluate zero-knowledge proof scaling', 'max_steps': 5, 'sync': True},
		headers=headers,
	)
	assert post_resp.status_code == 200
	data = post_resp.json()
	assert data['status'] == RESEARCH_STATUS_COMPLETED
	assert data['research_id'] is not None
	assert len(data['summary']) > 0


def test_research_sse_event_streaming(test_environment):
	"""Verify Server-Sent Events (SSE) stream emits properly formatted progress events and terminates."""
	client, store, _, job_mgr, _ = test_environment
	headers = {'X-API-Key': 'key-alpha-12345'}

	# 1. Create a queued research session
	post_resp = client.post(
		'/research',
		json={'task': 'Live streaming progress test query', 'sync': False},
		headers=headers,
	)
	res_id = post_resp.json()['research_id']

	# 2. Connect to SSE stream
	with client.stream('GET', f'/research/{res_id}/events', headers=headers) as stream:
		assert stream.status_code == 200
		assert 'text/event-stream' in stream.headers.get('content-type', '')

		events = []
		for line in stream.iter_lines():
			if line.startswith('data: '):
				event_json = json.loads(line[6:])
				events.append(event_json)
				if event_json.get('is_terminal'):
					break

		assert len(events) >= 1
		# Verify event payload structure
		first_event = events[0]
		assert first_event['research_id'] == res_id
		assert 'status' in first_event
		assert 'stage' in first_event
		assert 'percent' in first_event
		assert 'message' in first_event

		# Verify terminal event was emitted
		last_event = events[-1]
		assert last_event['is_terminal'] is True
		assert last_event['percent'] == 100


@pytest.mark.asyncio
async def test_research_job_cancellation(test_environment):
	"""Verify that an in-progress job can be cancelled and transitions to 'cancelled' status."""
	_, store, engine, job_mgr, fake_browser = test_environment
	headers = {'X-API-Key': 'key-alpha-12345'}

	# Configure delay on fake browser agent so job stays in progress
	fake_browser.delay_seconds = 2.0

	from httpx import ASGITransport, AsyncClient

	async with AsyncClient(transport=ASGITransport(app=app), base_url='http://testserver') as ac:
		post_resp = await ac.post(
			'/research',
			json={'task': 'Job cancellation test', 'sync': False},
			headers=headers,
		)
		res_id = post_resp.json()['research_id']

		# Immediately call cancel endpoint while task is actively running
		cancel_resp = await ac.post(f'/research/{res_id}/cancel', headers=headers)
		assert cancel_resp.status_code == 200
		cancel_data = cancel_resp.json()
		assert cancel_data['cancelled'] is True
		assert cancel_data['status'] == RESEARCH_STATUS_CANCELLED

		# Verify status is cancelled
		status_resp = await ac.get(f'/research/{res_id}/status', headers=headers)
		assert status_resp.status_code == 200
		assert status_resp.json()['status'] == RESEARCH_STATUS_CANCELLED
		assert status_resp.json()['is_terminal'] is True


def test_research_job_tenant_isolation(test_environment):
	"""Verify strict tenant boundaries across status polling, retrieval, cancellation, and SSE streams."""
	client, _, _, _, _ = test_environment
	headers_alpha = {'X-API-Key': 'key-alpha-12345'}
	headers_beta = {'X-API-Key': 'key-beta-67890'}

	# Org Alpha creates a research job
	post_resp = client.post(
		'/research',
		json={'task': 'Alpha private security report', 'sync': False},
		headers=headers_alpha,
	)
	alpha_res_id = post_resp.json()['research_id']

	# Org Beta must receive 404 across all access paths
	assert client.get(f'/research/{alpha_res_id}/status', headers=headers_beta).status_code == 404
	assert client.get(f'/research/{alpha_res_id}', headers=headers_beta).status_code == 404
	assert client.get(f'/research/{alpha_res_id}/export?format=markdown', headers=headers_beta).status_code == 404
	assert client.post(f'/research/{alpha_res_id}/cancel', headers=headers_beta).status_code == 404
	assert client.get(f'/research/{alpha_res_id}/events', headers=headers_beta).status_code == 404


@pytest.mark.asyncio
async def test_process_restart_reconciliation():
	"""Verify that orphaned 'queued' or 'in_progress' jobs are cleanly marked as failed on application restart."""
	store = InMemoryVectorStore()
	job_mgr = ResearchJobManager()

	# Create orphaned sessions
	s1 = ResearchSession(
		id='res-orphan-queued',
		organization_id='org_rec',
		title='Orphaned Queued',
		query='Queued before restart',
		status=RESEARCH_STATUS_QUEUED,
	)
	s2 = ResearchSession(
		id='res-orphan-running',
		organization_id='org_rec',
		title='Orphaned In Progress',
		query='Running before restart',
		status=RESEARCH_STATUS_IN_PROGRESS,
	)
	s3 = ResearchSession(
		id='res-completed',
		organization_id='org_rec',
		title='Already Completed',
		query='Finished before restart',
		status=RESEARCH_STATUS_COMPLETED,
	)

	await store.create_research_session(s1)
	await store.create_research_session(s2)
	await store.create_research_session(s3)

	# Simulate application startup reconciliation
	reconciled = await job_mgr.reconcile_restarted_jobs(store=KnowledgeStore(vector_store=store))
	assert reconciled == 2

	# Verify s1 and s2 are now interrupted with restart explanation
	f1 = await store.get_research_session('res-orphan-queued')
	assert f1 is not None
	assert f1.status == RESEARCH_STATUS_INTERRUPTED
	assert 'interrupted' in (f1.error or '').lower()

	f2 = await store.get_research_session('res-orphan-running')
	assert f2 is not None
	assert f2.status == RESEARCH_STATUS_INTERRUPTED

	# Verify s3 was unchanged
	f3 = await store.get_research_session('res-completed')
	assert f3 is not None
	assert f3.status == RESEARCH_STATUS_COMPLETED


@pytest.mark.asyncio
async def test_deterministic_fake_browser_agent_integration():
	"""Verify browser agent integration harvests normalized web evidence and verifies citations."""
	mem_store = InMemoryVectorStore()
	mock_llm = MockLLMProvider()
	fake_browser = DeterministicFakeBrowserAgent(
		findings=[
			{
				'url': 'https://spec.org/browser-agent',
				'title': 'Autonomous Browser Agent Standards',
				'snippet': 'Browser agents navigate via Chrome DevTools Protocol to harvest research data.',
				'relevance_score': 0.98,
			}
		]
	)

	engine = ResearchEngine(
		knowledge_store=KnowledgeStore(
			vector_store=mem_store,
			embedding_provider=MockEmbeddingModel(dimension=64),
		),
		llm_provider=mock_llm,
		browser_agent=fake_browser,
	)

	result = await engine.conduct_research(
		query='Explain autonomous browser agent navigation standards',
		include_web=True,
	)

	assert fake_browser.call_count == 1
	assert len(result.web_evidence) == 1
	assert result.web_evidence[0].source_uri == 'https://spec.org/browser-agent'
	assert 'Chrome DevTools Protocol' in result.web_evidence[0].content
	assert result.metadata['web_status']['status'] == 'completed'


@pytest.mark.asyncio
async def test_missing_browser_configuration_returns_explicit_unavailable():
	"""Verify that research engine with no browser agent attached records explicit unavailable status."""
	mem_store = InMemoryVectorStore()
	mock_llm = MockLLMProvider()

	# Explicitly attach no browser agent
	engine = ResearchEngine(
		knowledge_store=KnowledgeStore(
			vector_store=mem_store,
			embedding_provider=MockEmbeddingModel(dimension=64),
		),
		llm_provider=mock_llm,
		browser_agent=None,
	)

	result = await engine.conduct_research(
		query='What web facts are available?',
		include_web=True,
	)

	web_status = result.metadata.get('web_status', {})
	assert web_status.get('status') == 'unavailable'
	assert web_status.get('reason') == 'browser_agent_not_attached'
	assert len(result.web_evidence) == 0


@pytest.mark.asyncio
@pytest.mark.integration
async def test_postgres_schema_v4_and_baseline_data_preservation():
	"""Verify PostgreSQL schema version 4 migration and baseline table row preservation."""
	pg_url = os.getenv('INSIGHTBOT_DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/insightbot')

	try:
		conn = await asyncpg.connect(pg_url, timeout=2)
		await conn.close()
	except Exception:
		pytest.skip(f'PostgreSQL service not reachable at {pg_url}')

	store = PgVectorStore(database_url=pg_url, embedding_dim=64)
	await store.initialize()

	try:
		# Verify schema version 4
		conn = await asyncpg.connect(pg_url)
		version = await conn.fetchval('SELECT MAX(version) FROM schema_migrations;')
		assert version == 4, f'Expected schema version 4, got {version}'

		# Verify baseline tables exist and original document records (4 docs, 3 chunks) are untouched
		doc_count = await conn.fetchval('SELECT COUNT(*) FROM documents;')
		chunk_count = await conn.fetchval('SELECT COUNT(*) FROM document_chunks;')
		assert doc_count == 4, f'Expected 4 documents preserved, got {doc_count}'
		assert chunk_count == 3, f'Expected 3 chunks preserved, got {chunk_count}'

		# Test creating and updating a research session in PostgreSQL with schema v4 columns
		test_session = ResearchSession(
			id='res-pg-v4-test-01',
			organization_id='org_test_v4',
			title='PostgreSQL Schema V4 Verification',
			query='Test schema version 4 progress columns',
			status=RESEARCH_STATUS_QUEUED,
			progress_stage='queued',
			progress_percent=0,
			progress_message='Queued in PostgreSQL',
		)
		await store.create_research_session(test_session)

		# Fetch and verify
		fetched = await store.get_research_session('res-pg-v4-test-01', organization_id='org_test_v4')
		assert fetched is not None
		assert fetched.status == RESEARCH_STATUS_QUEUED
		assert fetched.progress_stage == 'queued'
		assert fetched.progress_percent == 0

		# Update progress
		updated = await store.update_research_session(
			session_id='res-pg-v4-test-01',
			organization_id='org_test_v4',
			status=RESEARCH_STATUS_IN_PROGRESS,
			progress_stage='synthesizing',
			progress_percent=75,
			progress_message='Synthesizing report in PG',
		)
		assert updated is not None
		assert updated.status == RESEARCH_STATUS_IN_PROGRESS
		assert updated.progress_percent == 75
		assert updated.progress_stage == 'synthesizing'

	finally:
		# Clean up test session to maintain baseline cleanliness
		await store.delete_research_session('res-pg-v4-test-01', organization_id='org_test_v4')
		await store.close()
		await conn.close()
