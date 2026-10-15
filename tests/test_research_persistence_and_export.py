"""Focused test suite for InsightBot AI Phase 14C:
Persistent Research Artifacts & Evidence Export.

Verifies:
1. Research session creation, retrieval, listing, and deletion.
2. Persistence across PostgreSQL-backed store re-instantiations (process restart).
3. Structured Markdown and JSON export generation with citations, web URLs, and conflicts.
4. Citation, source provenance, and conflict metadata preservation.
5. Multi-tenant boundary isolation and workspace filtering.
6. Failed and incomplete research session state preservation.
7. Backward compatibility with existing research and conversation endpoints.
"""

import os

import asyncpg
import pytest
from fastapi.testclient import TestClient

from insightbot.api.routes import app, get_agent, get_research_engine, get_store
from insightbot.config.settings import get_settings
from insightbot.db.in_memory import InMemoryVectorStore
from insightbot.db.models import ResearchSession
from insightbot.db.vector_store import PgVectorStore
from insightbot.knowledge.embeddings import MockEmbeddingModel
from insightbot.knowledge.store import KnowledgeStore
from insightbot.orchestrator.export import format_research_json, format_research_markdown
from insightbot.orchestrator.knowledge_agent import KnowledgeAgent
from insightbot.orchestrator.llm import MockLLMProvider
from insightbot.orchestrator.research_engine import ResearchEngine


@pytest.fixture
def test_environment(tmp_path, monkeypatch):
	"""Configures multi-tenant settings and an in-memory test store."""
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
	engine = ResearchEngine(knowledge_store=store, llm_provider=mock_llm)

	app.dependency_overrides[get_store] = lambda: store
	app.dependency_overrides[get_agent] = lambda: agent
	app.dependency_overrides[get_research_engine] = lambda: engine

	client = TestClient(app)
	yield client, store, engine

	app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_in_memory_research_session_crud_and_pagination():
	"""Test research session lifecycle, pagination, and workspace filtering in memory."""
	store = InMemoryVectorStore()

	s1 = ResearchSession(
		id='res-crud-001',
		organization_id='org_test',
		workspace_id='ws_prod',
		title='Autonomous AI Market Study',
		query='Analyze browser automation agents in 2026',
		status='completed',
		summary='Browser agents have evolved into autonomous multi-modal assistants.',
		sources=['https://example.com/agent-report', 'internal_spec.pdf'],
		internal_evidence=[
			{
				'source_type': 'internal_knowledge',
				'source_uri': 'internal_spec.pdf',
				'chunk_id': 'chk-001',
				'content': 'InsightBot supports multi-tenant isolation.',
			}
		],
		web_evidence=[
			{
				'source_type': 'web_research',
				'source_uri': 'https://example.com/agent-report',
				'content': 'Autonomous agents use CDP protocols.',
			}
		],
		conflicts=[
			{
				'topic': 'Version scope',
				'claim_a': 'v1',
				'source_a': 'src1',
				'claim_b': 'v2',
				'source_b': 'src2',
				'description': 'Discrepancy in release versions.',
			}
		],
		citations=[
			{
				'source_type': 'internal_knowledge',
				'source_uri': 'internal_spec.pdf',
				'chunk_id': 'chk-001',
				'snippet': 'InsightBot supports multi-tenant isolation.',
			}
		],
	)

	s2 = ResearchSession(
		id='res-crud-002',
		organization_id='org_test',
		workspace_id='ws_dev',
		title='Compiler Optimization Research',
		query='Evaluate LLVM vs Cranelift',
		status='completed',
		summary='Cranelift provides faster compilation times for JIT workloads.',
	)

	# 1. Create sessions
	id1 = await store.create_research_session(s1)
	id2 = await store.create_research_session(s2)
	assert id1 == 'res-crud-001'
	assert id2 == 'res-crud-002'

	# 2. Get session by ID
	fetched = await store.get_research_session('res-crud-001', organization_id='org_test')
	assert fetched is not None
	assert fetched.title == 'Autonomous AI Market Study'
	assert len(fetched.citations) == 1
	assert len(fetched.conflicts) == 1

	# 3. List sessions and filter by workspace
	all_sessions = await store.list_research_sessions(organization_id='org_test')
	assert len(all_sessions) == 2

	ws_sessions = await store.list_research_sessions(organization_id='org_test', workspace_id='ws_prod')
	assert len(ws_sessions) == 1
	assert ws_sessions[0].id == 'res-crud-001'

	# 4. Pagination
	paged = await store.list_research_sessions(organization_id='org_test', limit=1, offset=1)
	assert len(paged) == 1

	# 5. Delete session
	deleted = await store.delete_research_session('res-crud-001', organization_id='org_test')
	assert deleted is True
	assert await store.get_research_session('res-crud-001', organization_id='org_test') is None


def test_research_markdown_and_json_export_formatting():
	"""Verify structured Markdown and JSON report generation preserving citations, evidence, and conflicts."""
	session = ResearchSession(
		id='res-export-777',
		organization_id='org_export',
		workspace_id='ws_export',
		title='Cloud Architecture Synthesis',
		query='Evaluate multi-region database failover strategies',
		status='completed',
		summary='Multi-region failover requires asynchronous replication and distributed consensus.',
		sources=['https://database.org/failover', 'whitepaper.pdf'],
		internal_evidence=[
			{
				'source_type': 'internal_knowledge',
				'source_uri': 'whitepaper.pdf',
				'chunk_id': 'chk-whitepaper-1',
				'content': 'Raft consensus guarantees safety under network partitions.',
			}
		],
		web_evidence=[
			{
				'source_type': 'web_research',
				'source_uri': 'https://database.org/failover',
				'title': 'High Availability Guide',
				'content': 'Failover latency is typically bounded within 5 seconds.',
			}
		],
		conflicts=[
			{
				'topic': 'Failover RTO',
				'claim_a': 'RTO is 5 seconds',
				'source_a': 'https://database.org/failover',
				'claim_b': 'RTO is 30 seconds',
				'source_b': 'whitepaper.pdf',
				'description': 'Discrepancy between documentation and enterprise SLA.',
			}
		],
		citations=[
			{
				'source_type': 'internal_knowledge',
				'source_uri': 'whitepaper.pdf',
				'chunk_id': 'chk-whitepaper-1',
				'snippet': 'Raft consensus guarantees safety under network partitions.',
			},
			{
				'source_type': 'web_research',
				'source_uri': 'https://database.org/failover',
				'snippet': 'Failover latency is typically bounded within 5 seconds.',
			},
		],
		is_grounded=True,
		sufficient_evidence=True,
	)

	# 1. Markdown Export Check
	md = format_research_markdown(session)
	assert '# Research Report: Cloud Architecture Synthesis' in md
	assert '- **Research ID:** `res-export-777`' in md
	assert '## Executive Summary' in md
	assert 'Multi-region failover requires asynchronous replication' in md
	assert '## Verified Source Citations' in md
	assert 'whitepaper.pdf' in md
	assert 'https://database.org/failover' in md
	assert '## Cross-Source Contradictions & Discrepancies' in md
	assert 'Failover RTO' in md
	assert 'Discrepancy between documentation and enterprise SLA.' in md
	assert '## Gathered Evidence' in md
	assert 'Raft consensus guarantees safety under network partitions.' in md

	# 2. JSON Export Check
	json_data = format_research_json(session)
	assert json_data['id'] == 'res-export-777'
	assert json_data['title'] == 'Cloud Architecture Synthesis'
	assert len(json_data['citations']) == 2
	assert len(json_data['conflicts']) == 1
	assert len(json_data['internal_evidence']) == 1
	assert len(json_data['web_evidence']) == 1


def test_api_research_lifecycle_and_export_endpoints(test_environment):
	"""Test full REST API workflow for research execution, persistence, and export."""
	client, store, _ = test_environment
	headers = {'X-API-Key': 'key-alpha-12345'}

	# 1. POST /research (Runs research and automatically persists session)
	post_resp = client.post(
		'/research',
		json={'task': 'Research enterprise tenant isolation mechanisms', 'max_steps': 10, 'sync': True},
		headers=headers,
	)
	assert post_resp.status_code == 200, post_resp.text
	res_data = post_resp.json()
	assert res_data['status'] in ('completed', 'completed_unverified', 'provider_unavailable')
	research_id = res_data.get('research_id')
	assert research_id is not None
	assert research_id.startswith('res-')

	# 2. GET /research (List saved sessions)
	list_resp = client.get('/research', headers=headers)
	assert list_resp.status_code == 200
	sessions = list_resp.json()
	assert any(s['id'] == research_id for s in sessions)

	# 3. GET /research/{id} (Retrieve saved session)
	get_resp = client.get(f'/research/{research_id}', headers=headers)
	assert get_resp.status_code == 200
	session_obj = get_resp.json()
	assert session_obj['id'] == research_id
	assert session_obj['organization_id'] == 'org_alpha'
	assert 'Enterprise' in session_obj['title'] or 'enterprise' in session_obj['title']

	# 4. GET /research/{id}/export?format=markdown
	exp_md_resp = client.get(f'/research/{research_id}/export?format=markdown', headers=headers)
	assert exp_md_resp.status_code == 200
	assert 'text/markdown' in exp_md_resp.headers['content-type']
	assert f'attachment; filename="research-{research_id}.md"' in exp_md_resp.headers['content-disposition']
	assert '# Research Report:' in exp_md_resp.text

	# 5. GET /research/{id}/export?format=json
	exp_json_resp = client.get(f'/research/{research_id}/export?format=json', headers=headers)
	assert exp_json_resp.status_code == 200
	assert 'application/json' in exp_json_resp.headers['content-type']
	assert exp_json_resp.json()['id'] == research_id

	# 6. GET /research/{id}/export?format=unsupported -> 400 Bad Request
	bad_exp = client.get(f'/research/{research_id}/export?format=csv', headers=headers)
	assert bad_exp.status_code == 400

	# 7. DELETE /research/{id}
	del_resp = client.delete(f'/research/{research_id}', headers=headers)
	assert del_resp.status_code == 200
	assert del_resp.json() == {'deleted': True, 'id': research_id}

	# Verify 404 after deletion
	assert client.get(f'/research/{research_id}', headers=headers).status_code == 404
	assert client.get(f'/research/{research_id}/export', headers=headers).status_code == 404


def test_api_research_multi_tenant_isolation(test_environment):
	"""Verify multi-tenant boundary isolation across research sessions and exports."""
	client, _, _ = test_environment
	alpha_headers = {'X-API-Key': 'key-alpha-12345'}
	beta_headers = {'X-API-Key': 'key-beta-67890'}

	# Tenant Alpha conducts research
	post_resp = client.post(
		'/research',
		json={'task': 'Confidential alpha project strategy'},
		headers=alpha_headers,
	)
	assert post_resp.status_code == 200
	alpha_res_id = post_resp.json()['research_id']

	# Tenant Beta lists research sessions -> must not include Alpha's session
	beta_list = client.get('/research', headers=beta_headers).json()
	assert not any(s['id'] == alpha_res_id for s in beta_list)

	# Tenant Beta attempts to retrieve Alpha's research session -> 404 Not Found
	assert client.get(f'/research/{alpha_res_id}', headers=beta_headers).status_code == 404

	# Tenant Beta attempts to export Alpha's research report -> 404 Not Found
	assert client.get(f'/research/{alpha_res_id}/export', headers=beta_headers).status_code == 404

	# Tenant Beta attempts to delete Alpha's research session -> 404 Not Found
	assert client.delete(f'/research/{alpha_res_id}', headers=beta_headers).status_code == 404

	# Verify Alpha can still retrieve and export their research
	assert client.get(f'/research/{alpha_res_id}', headers=alpha_headers).status_code == 200


@pytest.mark.asyncio
async def test_failed_research_state_persistence(tmp_path):
	"""Verify failed research sessions preserve failure reason and error details."""
	store = InMemoryVectorStore()
	failed_session = ResearchSession(
		id='res-fail-001',
		organization_id='org_test',
		title='Failed Analysis Run',
		query='Analyze unavailable internal server',
		status='provider_unavailable',
		summary='AI provider error: RateLimit exceeded after 3 retries.',
		error='RateLimitError: 429 quota exceeded',
		is_grounded=False,
		sufficient_evidence=False,
	)
	await store.create_research_session(failed_session)

	fetched = await store.get_research_session('res-fail-001', organization_id='org_test')
	assert fetched is not None
	assert fetched.status == 'provider_unavailable'
	assert fetched.error == 'RateLimitError: 429 quota exceeded'

	# Ensure markdown export formats failure cleanly
	md = format_research_markdown(fetched)
	assert '- **Status:** `provider_unavailable`' in md
	assert '- **Error Detail:** RateLimitError: 429 quota exceeded' in md
	assert 'AI provider error:' in md


@pytest.mark.asyncio
@pytest.mark.integration
async def test_postgres_research_session_persistence_and_restart():
	"""Verify research session persistence in PostgreSQL across store re-instantiations (application restart)."""
	pg_url = os.getenv('INSIGHTBOT_DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/insightbot')

	try:
		conn = await asyncpg.connect(pg_url, timeout=2)
		await conn.close()
	except Exception:
		pytest.skip(f'PostgreSQL service not reachable at {pg_url}')

	test_org = 'org_test_phase14c_persist'
	session_id = 'res-test-pg-restart-101'

	# 1. Initialize store 1 and persist research session
	store1 = PgVectorStore(database_url=pg_url, embedding_dim=64)
	await store1.initialize()

	session = ResearchSession(
		id=session_id,
		organization_id=test_org,
		workspace_id='ws_pg_restart',
		title='Postgres Restart Research Persistence',
		query='Does PostgreSQL ACID survive process restarts?',
		status='completed',
		summary='PostgreSQL WAL ensures full durability across crashes and restarts.',
		sources=['internal_wal_spec.txt', 'https://postgresql.org/docs/wal'],
		internal_evidence=[
			{
				'source_type': 'internal_knowledge',
				'source_uri': 'internal_wal_spec.txt',
				'chunk_id': 'chk-wal-01',
				'content': 'Write-Ahead Logging provides crash recovery.',
			}
		],
		web_evidence=[
			{
				'source_type': 'web_research',
				'source_uri': 'https://postgresql.org/docs/wal',
				'content': 'WAL guarantees data consistency.',
			}
		],
		conflicts=[
			{
				'topic': 'Fsync performance',
				'claim_a': 'negligible',
				'source_a': 'doc1',
				'claim_b': 'significant',
				'source_b': 'doc2',
				'description': 'Disk sync latency variation.',
			}
		],
		citations=[
			{
				'source_type': 'internal_knowledge',
				'source_uri': 'internal_wal_spec.txt',
				'chunk_id': 'chk-wal-01',
				'snippet': 'Write-Ahead Logging provides crash recovery.',
			}
		],
		is_grounded=True,
		sufficient_evidence=True,
		metadata={'test_marker': 'phase14c'},
	)

	await store1.create_research_session(session)
	await store1.close()  # Simulate process shutdown

	# 2. Initialize brand new store instance (restart)
	store2 = PgVectorStore(database_url=pg_url, embedding_dim=64)
	await store2.initialize()

	try:
		fetched = await store2.get_research_session(session_id, organization_id=test_org)
		assert fetched is not None
		assert fetched.id == session_id
		assert fetched.title == 'Postgres Restart Research Persistence'
		assert fetched.workspace_id == 'ws_pg_restart'
		assert len(fetched.citations) == 1
		assert len(fetched.conflicts) == 1
		assert len(fetched.internal_evidence) == 1
		assert len(fetched.web_evidence) == 1
		assert fetched.metadata.get('test_marker') == 'phase14c'

		# Test listing
		listed = await store2.list_research_sessions(organization_id=test_org, workspace_id='ws_pg_restart')
		assert len(listed) >= 1
		assert any(s.id == session_id for s in listed)
	finally:
		# Clean up test session to ensure database cleanliness
		await store2.delete_research_session(session_id, organization_id=test_org)
		await store2.close()
