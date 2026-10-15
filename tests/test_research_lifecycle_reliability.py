"""Comprehensive deterministic test suite for InsightBot AI Phase 43-44:
Reliable Research Job Lifecycle, Failure Recovery & Restart Persistence.

Verifies:
1. Complete state machine transitions, including RESEARCH_STATUS_INTERRUPTED.
2. Rejection of transitions out of all terminal states.
3. Protection against progress regressing or mutating already-terminal records.
4. Durable session creation before queuing acknowledgement.
5. Deterministic, idempotent startup recovery of orphaned jobs to 'interrupted'.
6. Preservation of completed, failed, cancelled, and interrupted records during recovery.
7. Cooperative cancellation of queued and in-progress jobs.
8. Safe resolution of cancellation racing with completion.
9. Duplicate cancellation and terminal state idempotency.
10. Atomic report persistence: failure of report persistence marks job as failed (no false success).
11. Multi-tenant isolation for job status and cancellation.
12. Terminal status reporting for polling termination (is_terminal=True).
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from insightbot.api.routes import app, get_agent, get_job_manager, get_research_engine, get_store
from insightbot.config.settings import get_settings
from insightbot.db.in_memory import InMemoryVectorStore
from insightbot.db.models import (
	RESEARCH_STATUS_CANCELLED,
	RESEARCH_STATUS_COMPLETED,
	RESEARCH_STATUS_COMPLETED_UNVERIFIED,
	RESEARCH_STATUS_FAILED,
	RESEARCH_STATUS_IN_PROGRESS,
	RESEARCH_STATUS_INTERRUPTED,
	RESEARCH_STATUS_QUEUED,
	TERMINAL_RESEARCH_STATUSES,
	VALID_RESEARCH_TRANSITIONS,
	ResearchSession,
)
from insightbot.engine.browser.agent import DeterministicFakeBrowserAgent
from insightbot.knowledge.embeddings import MockEmbeddingModel
from insightbot.knowledge.store import KnowledgeStore
from insightbot.orchestrator.job_manager import ResearchJobManager
from insightbot.orchestrator.knowledge_agent import KnowledgeAgent
from insightbot.orchestrator.llm import MockLLMProvider
from insightbot.orchestrator.research_engine import (
	ResearchCitation,
	ResearchEngine,
	ResearchEngineResult,
)


@pytest.fixture
def lifecycle_env(tmp_path, monkeypatch):
	"""Configures isolated test environment for lifecycle and recovery testing."""
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
async def test_state_machine_valid_and_invalid_transitions():
	"""Verify all valid lifecycle transitions and strict rejection of transitions from terminal states."""
	store = InMemoryVectorStore()

	# 1. Verify terminal statuses set
	assert RESEARCH_STATUS_INTERRUPTED in TERMINAL_RESEARCH_STATUSES
	assert RESEARCH_STATUS_CANCELLED in TERMINAL_RESEARCH_STATUSES
	assert RESEARCH_STATUS_FAILED in TERMINAL_RESEARCH_STATUSES
	assert RESEARCH_STATUS_COMPLETED in TERMINAL_RESEARCH_STATUSES

	# 2. Verify all terminal states have empty outgoing transitions
	for term_st in TERMINAL_RESEARCH_STATUSES:
		assert VALID_RESEARCH_TRANSITIONS[term_st] == set(), f"Terminal status '{term_st}' must have no valid outgoing transitions."

	# 3. Queued allows transition to interrupted
	assert RESEARCH_STATUS_INTERRUPTED in VALID_RESEARCH_TRANSITIONS[RESEARCH_STATUS_QUEUED]
	# In_progress allows transition to interrupted
	assert RESEARCH_STATUS_INTERRUPTED in VALID_RESEARCH_TRANSITIONS[RESEARCH_STATUS_IN_PROGRESS]

	# 4. Create a test session in queued status
	session = ResearchSession(
		id='res-sm-01',
		organization_id='org_test',
		title='State Machine Test',
		query='Test transitions',
		status=RESEARCH_STATUS_QUEUED,
	)
	await store.create_research_session(session)

	# Transition: queued -> in_progress (valid)
	s_prog = await store.update_research_session('res-sm-01', 'org_test', status=RESEARCH_STATUS_IN_PROGRESS)
	assert s_prog is not None
	assert s_prog.status == RESEARCH_STATUS_IN_PROGRESS

	# Transition: in_progress -> interrupted (valid)
	s_int = await store.update_research_session('res-sm-01', 'org_test', status=RESEARCH_STATUS_INTERRUPTED)
	assert s_int is not None
	assert s_int.status == RESEARCH_STATUS_INTERRUPTED

	# Transition: interrupted -> in_progress (INVALID - terminal state cannot transition)
	with pytest.raises(ValueError, match='Invalid state transition'):
		await store.update_research_session('res-sm-01', 'org_test', status=RESEARCH_STATUS_IN_PROGRESS)

	# Transition: interrupted -> completed (INVALID)
	with pytest.raises(ValueError, match='Invalid state transition'):
		await store.update_research_session('res-sm-01', 'org_test', status=RESEARCH_STATUS_COMPLETED)


@pytest.mark.asyncio
async def test_terminal_session_progress_protection():
	"""Verify that late progress updates on an already-terminal session do not corrupt terminal progress."""
	store = InMemoryVectorStore()

	session = ResearchSession(
		id='res-term-prot',
		organization_id='org_test',
		title='Terminal Protection Test',
		query='Guard against regressions',
		status=RESEARCH_STATUS_COMPLETED,
		progress_stage='completed',
		progress_percent=100,
		progress_message='All work finished',
	)
	await store.create_research_session(session)

	# A late straggler callback tries to update progress to 40% on completed session
	updated = await store.update_research_session(
		session_id='res-term-prot',
		organization_id='org_test',
		progress_stage='synthesizing',
		progress_percent=40,
		progress_message='Synthesizing chunks...',
	)

	assert updated is not None
	assert updated.status == RESEARCH_STATUS_COMPLETED
	assert updated.progress_percent == 100
	assert updated.progress_stage == 'completed'
	assert updated.progress_message == 'All work finished'


@pytest.mark.asyncio
async def test_in_progress_monotonic_progress_percent():
	"""Verify progress percentage is clamped between 0-100 and does not regress during in_progress."""
	store = InMemoryVectorStore()

	session = ResearchSession(
		id='res-monotonic',
		organization_id='org_test',
		query='Monotonic progress',
		status=RESEARCH_STATUS_IN_PROGRESS,
		progress_percent=50,
	)
	await store.create_research_session(session)

	# Attempt to regress progress to 20%
	u1 = await store.update_research_session(
		'res-monotonic',
		'org_test',
		progress_percent=20,
	)
	assert u1 is not None
	assert u1.progress_percent == 50  # Must not regress below 50

	# Advance progress to 85%
	u2 = await store.update_research_session(
		'res-monotonic',
		'org_test',
		progress_percent=85,
	)
	assert u2 is not None
	assert u2.progress_percent == 85

	# Attempt to set beyond 100%
	u3 = await store.update_research_session(
		'res-monotonic',
		'org_test',
		progress_percent=150,
	)
	assert u3 is not None
	assert u3.progress_percent == 100


def test_durable_session_creation_before_queuing_acknowledgement(lifecycle_env):
	"""Verify POST /research persists the session in the database before acknowledging queued status."""
	client, store, _, _, _ = lifecycle_env
	headers = {'X-API-Key': 'key-alpha-12345'}

	resp = client.post(
		'/research',
		json={'task': 'Persist before acknowledge test', 'max_steps': 5, 'sync': False},
		headers=headers,
	)
	assert resp.status_code == 200
	data = resp.json()
	research_id = data['research_id']
	assert data['status'] == RESEARCH_STATUS_QUEUED

	# Query store directly to verify session was persisted durably
	loop = asyncio.get_event_loop()
	session = loop.run_until_complete(store.get_research_session(research_id, organization_id='org_alpha'))
	assert session is not None
	assert session.id == research_id
	assert session.query == 'Persist before acknowledge test'
	assert session.organization_id == 'org_alpha'


@pytest.mark.asyncio
async def test_startup_recovery_interrupted_and_idempotency():
	"""Verify startup recovery marks orphaned queued and running jobs as interrupted, and is idempotent."""
	store = InMemoryVectorStore()
	k_store = KnowledgeStore(vector_store=store)
	job_mgr = ResearchJobManager()

	# 1. Create a set of jobs across various states
	s_queued = ResearchSession(
		id='res-rec-queued',
		organization_id='org_rec',
		query='Queued task orphaned on restart',
		status=RESEARCH_STATUS_QUEUED,
		progress_percent=0,
	)
	s_running = ResearchSession(
		id='res-rec-running',
		organization_id='org_rec',
		query='Running task orphaned on restart',
		status=RESEARCH_STATUS_IN_PROGRESS,
		progress_percent=45,
	)
	s_completed = ResearchSession(
		id='res-rec-completed',
		organization_id='org_rec',
		query='Completed task before restart',
		status=RESEARCH_STATUS_COMPLETED,
		progress_percent=100,
		summary='Existing verified report',
	)
	s_cancelled = ResearchSession(
		id='res-rec-cancelled',
		organization_id='org_rec',
		query='Cancelled task before restart',
		status=RESEARCH_STATUS_CANCELLED,
		progress_percent=100,
	)

	await store.create_research_session(s_queued)
	await store.create_research_session(s_running)
	await store.create_research_session(s_completed)
	await store.create_research_session(s_cancelled)

	# 2. Run first recovery pass
	count1 = await job_mgr.reconcile_restarted_jobs(k_store)
	assert count1 == 2, 'Should reconcile exactly the 2 active/queued jobs'

	# Verify s_queued is now interrupted
	q_rec = await store.get_research_session('res-rec-queued')
	assert q_rec is not None
	assert q_rec.status == RESEARCH_STATUS_INTERRUPTED
	assert q_rec.progress_stage == 'interrupted'
	assert 'interrupted' in (q_rec.error or '').lower()

	# Verify s_running is now interrupted
	r_rec = await store.get_research_session('res-rec-running')
	assert r_rec is not None
	assert r_rec.status == RESEARCH_STATUS_INTERRUPTED
	assert r_rec.progress_stage == 'interrupted'

	# Verify completed and cancelled jobs were untouched
	c_rec = await store.get_research_session('res-rec-completed')
	assert c_rec is not None
	assert c_rec.status == RESEARCH_STATUS_COMPLETED
	assert c_rec.summary == 'Existing verified report'

	cn_rec = await store.get_research_session('res-rec-cancelled')
	assert cn_rec is not None
	assert cn_rec.status == RESEARCH_STATUS_CANCELLED

	# 3. Run second recovery pass (idempotency verification)
	count2 = await job_mgr.reconcile_restarted_jobs(k_store)
	assert count2 == 0, 'Second recovery pass must be a no-op (0 jobs reconciled)'


@pytest.mark.asyncio
async def test_job_cancellation_cooperative_and_duplicate():
	"""Verify cooperative job cancellation, tenant validation, and duplicate cancellation safety."""
	store = InMemoryVectorStore()
	k_store = KnowledgeStore(vector_store=store)
	job_mgr = ResearchJobManager()

	# Session owned by org_alpha
	session = ResearchSession(
		id='res-cancel-01',
		organization_id='org_alpha',
		query='Cancel test query',
		status=RESEARCH_STATUS_IN_PROGRESS,
	)
	await store.create_research_session(session)

	# 1. Unauthorized tenant cannot cancel
	unauth_res = await job_mgr.cancel_job('res-cancel-01', 'org_beta', k_store)
	assert unauth_res is False
	s_after_unauth = await store.get_research_session('res-cancel-01')
	assert s_after_unauth is not None
	assert s_after_unauth.status == RESEARCH_STATUS_IN_PROGRESS

	# 2. Authorized tenant cancels
	auth_res = await job_mgr.cancel_job('res-cancel-01', 'org_alpha', k_store)
	assert auth_res is True
	s_after_cancel = await store.get_research_session('res-cancel-01')
	assert s_after_cancel is not None
	assert s_after_cancel.status == RESEARCH_STATUS_CANCELLED
	assert s_after_cancel.progress_stage == 'cancelled'

	# 3. Duplicate cancellation on already terminal session safely returns False
	dup_res = await job_mgr.cancel_job('res-cancel-01', 'org_alpha', k_store)
	assert dup_res is False


@pytest.mark.asyncio
async def test_cancellation_racing_with_completion():
	"""Verify that when a cancellation races with completion, completed state is preserved without error."""
	store = InMemoryVectorStore()
	k_store = KnowledgeStore(vector_store=store)
	job_mgr = ResearchJobManager()

	# Session marked as completed
	session = ResearchSession(
		id='res-race-01',
		organization_id='org_alpha',
		query='Race condition query',
		status=RESEARCH_STATUS_COMPLETED,
		summary='Report generated before cancellation landed',
	)
	await store.create_research_session(session)

	# Late cancellation request arrives
	res = await job_mgr.cancel_job('res-race-01', 'org_alpha', k_store)
	assert res is False

	# Completed session report must not be overwritten
	curr = await store.get_research_session('res-race-01')
	assert curr is not None
	assert curr.status == RESEARCH_STATUS_COMPLETED
	assert curr.summary == 'Report generated before cancellation landed'


@pytest.mark.asyncio
async def test_atomic_persistence_failure_results_in_failed_not_completed():
	"""Verify that if saving completion report artifacts fails, the job transitions to failed (no false success)."""
	class FailingReportVectorStore(InMemoryVectorStore):
		async def update_research_session(self, session_id: str, organization_id: str, **kwargs):
			if kwargs.get('status') in (RESEARCH_STATUS_COMPLETED, RESEARCH_STATUS_COMPLETED_UNVERIFIED):
				raise RuntimeError('Disk full: Failed to write research report')
			return await super().update_research_session(session_id, organization_id, **kwargs)

	store_impl = FailingReportVectorStore()
	test_session = ResearchSession(
		id='res-atomic-fail',
		organization_id='org_alpha',
		query='Atomic persistence test',
		status=RESEARCH_STATUS_QUEUED,
	)
	await store_impl.create_research_session(test_session)

	k_store = KnowledgeStore(vector_store=store_impl)
	job_mgr = ResearchJobManager()

	# Mock engine to return successful grounded result
	mock_engine = MagicMock()
	mock_result = ResearchEngineResult(
		query='Atomic test',
		summary='Grounded synthesis',
		citations=[ResearchCitation(claim='Test', source_uri='doc.txt', source_type='internal')],
		is_grounded=True,
		sufficient_evidence=True,
	)
	mock_engine.conduct_research = AsyncMock(return_value=mock_result)

	events_received = []

	async def listener():
		async for ev in job_mgr.subscribe('res-atomic-fail'):
			events_received.append(ev)

	listen_task = asyncio.create_task(listener())
	await asyncio.sleep(0.01)

	# Execute job directly
	await job_mgr._execute_job(
		research_id='res-atomic-fail',
		organization_id='org_alpha',
		workspace_id=None,
		query='Atomic test',
		max_steps=5,
		engine=mock_engine,
		store=k_store,
	)

	await asyncio.sleep(0.05)
	listen_task.cancel()
	try:
		await listen_task
	except asyncio.CancelledError:
		pass

	# Verify terminal event emitted was FAILED, never COMPLETED!
	assert any(ev.get('status') == RESEARCH_STATUS_FAILED for ev in events_received)
	assert not any(ev.get('status') == RESEARCH_STATUS_COMPLETED for ev in events_received)

	# Verify persisted state is FAILED
	persisted = await store_impl.get_research_session('res-atomic-fail')
	assert persisted is not None
	assert persisted.status == RESEARCH_STATUS_FAILED
	assert 'persistence failed' in (persisted.error or '').lower()


def test_api_status_terminal_flag_and_cancellation_contract(lifecycle_env):
	"""Verify /research/{id}/status reports is_terminal=True for interrupted and cancelled jobs."""
	client, store, _, _, _ = lifecycle_env
	headers = {'X-API-Key': 'key-alpha-12345'}

	# 1. Create interrupted session in store
	s_int = ResearchSession(
		id='res-api-int',
		organization_id='org_alpha',
		query='Interrupted query',
		status=RESEARCH_STATUS_INTERRUPTED,
		progress_stage='interrupted',
		progress_percent=100,
		error='Interrupted by server restart',
	)
	loop = asyncio.get_event_loop()
	loop.run_until_complete(store.create_research_session(s_int))

	# 2. Check /research/{id}/status
	resp = client.get('/research/res-api-int/status', headers=headers)
	assert resp.status_code == 200
	data = resp.json()
	assert data['status'] == RESEARCH_STATUS_INTERRUPTED
	assert data['is_terminal'] is True
	assert data['error'] == 'Interrupted by server restart'

	# 3. Attempt to cancel an already interrupted session
	cancel_resp = client.post('/research/res-api-int/cancel', headers=headers)
	assert cancel_resp.status_code == 200
	cancel_data = cancel_resp.json()
	assert cancel_data['cancelled'] is False
	assert cancel_data['status'] == RESEARCH_STATUS_INTERRUPTED
	assert 'already in terminal state' in cancel_data['message']
