"""Tests for frontend API extensions: /auth/me, document chunks inspection,
hybrid search query flag, and research status enrichment.
"""

import pytest
from fastapi.testclient import TestClient

from insightbot.api.routes import app, get_agent, get_research_engine, get_store
from insightbot.config.settings import get_settings
from insightbot.db.in_memory import InMemoryVectorStore
from insightbot.knowledge.embeddings import MockEmbeddingModel
from insightbot.knowledge.store import KnowledgeStore
from insightbot.orchestrator.knowledge_agent import KnowledgeAgent
from insightbot.orchestrator.research_engine import (
	ConflictItem,
	NormalizedEvidence,
	ResearchCitation,
	ResearchEngine,
	ResearchEngineResult,
)


@pytest.fixture
def multi_tenant_client(tmp_path, monkeypatch):
	settings = get_settings()
	monkeypatch.setattr(settings, 'environment', 'development')
	monkeypatch.setattr(settings, 'api_key', 'admin-master-secret-key-1234')
	monkeypatch.setattr(
		settings,
		'tenant_api_keys',
		{
			'tenant_alpha': 'alpha-secret-token-12345',
			'tenant_beta': 'beta-secret-token-12345',
		},
	)
	monkeypatch.setattr(settings, 'default_organization_id', 'default_org')

	mem_store = InMemoryVectorStore()
	mock_embed = MockEmbeddingModel(dimension=64)
	ks = KnowledgeStore(storage_dir=tmp_path, vector_store=mem_store, embedding_provider=mock_embed)

	async def mock_llm(q, ctx):
		return 'Grounded response for testing.'

	agent = KnowledgeAgent(knowledge_store=ks, llm=mock_llm)
	engine = ResearchEngine(knowledge_store=ks)

	app.dependency_overrides[get_store] = lambda: ks
	app.dependency_overrides[get_agent] = lambda: agent
	app.dependency_overrides[get_research_engine] = lambda: engine

	client = TestClient(app)
	yield client, ks, engine

	app.dependency_overrides.clear()


def test_auth_me_endpoint_resolves_tenant_context(multi_tenant_client):
	client, _, _ = multi_tenant_client

	# 1. Tenant Alpha
	res = client.get('/auth/me', headers={'Authorization': 'Bearer alpha-secret-token-12345'})
	assert res.status_code == 200
	data = res.json()
	assert data['organization_id'] == 'tenant_alpha'
	assert data['is_admin'] is False

	# 2. Tenant Beta with workspace header
	res = client.get(
		'/auth/me',
		headers={
			'Authorization': 'Bearer beta-secret-token-12345',
			'X-Workspace-Id': 'ws-finance',
		},
	)
	assert res.status_code == 200
	data = res.json()
	assert data['organization_id'] == 'tenant_beta'
	assert data['workspace_id'] == 'ws-finance'
	assert data['is_admin'] is False

	# 3. Admin Key
	res = client.get('/auth/me', headers={'Authorization': 'Bearer admin-master-secret-key-1234'})
	assert res.status_code == 200
	data = res.json()
	assert data['organization_id'] == 'default_org'
	assert data['is_admin'] is True


def test_document_chunks_inspection_and_isolation(multi_tenant_client):
	client, _, _ = multi_tenant_client

	# 1. Ingest document for Tenant Alpha
	content_a = b'# Architecture\nInsightBot uses PostgreSQL with pgvector for private RAG.'
	res_a = client.post(
		'/knowledge/documents',
		headers={'Authorization': 'Bearer alpha-secret-token-12345'},
		files={'file': ('alpha_arch.md', content_a, 'text/markdown')},
	)
	assert res_a.status_code == 201
	doc_a_id = res_a.json()['id']

	# 2. Ingest document for Tenant Beta
	content_b = b'# Security\nTenant Beta internal encryption keys and security standards.'
	res_b = client.post(
		'/knowledge/documents',
		headers={'Authorization': 'Bearer beta-secret-token-12345'},
		files={'file': ('beta_sec.md', content_b, 'text/markdown')},
	)
	assert res_b.status_code == 201
	doc_b_id = res_b.json()['id']

	# 3. Tenant Alpha inspects chunks of their own document
	chunks_res_a = client.get(
		f'/knowledge/documents/{doc_a_id}/chunks',
		headers={'Authorization': 'Bearer alpha-secret-token-12345'},
	)
	assert chunks_res_a.status_code == 200
	chunks_a = chunks_res_a.json()
	assert len(chunks_a) >= 1
	assert chunks_a[0]['document_id'] == doc_a_id
	assert chunks_a[0]['organization_id'] == 'tenant_alpha'
	assert 'PostgreSQL' in chunks_a[0]['content']

	# 4. Tenant Alpha attempts to inspect chunks of Tenant Beta's document -> 404 Not Found
	cross_res = client.get(
		f'/knowledge/documents/{doc_b_id}/chunks',
		headers={'Authorization': 'Bearer alpha-secret-token-12345'},
	)
	assert cross_res.status_code == 404
	assert 'not found' in cross_res.json()['detail'].lower()

	# 5. Non-existent document -> 404 Not Found
	non_existent = client.get(
		'/knowledge/documents/doc-non-existent-999/chunks',
		headers={'Authorization': 'Bearer alpha-secret-token-12345'},
	)
	assert non_existent.status_code == 404


def test_hybrid_search_query_parameter(multi_tenant_client):
	client, _, _ = multi_tenant_client

	# Ingest test document
	content = b'Deterministic chunking and hybrid lexical and semantic search for teams.'
	res = client.post(
		'/knowledge/documents',
		headers={'Authorization': 'Bearer alpha-secret-token-12345'},
		files={'file': ('search_doc.txt', content, 'text/plain')},
	)
	assert res.status_code == 201

	# Search with hybrid=True
	search_res = client.post(
		'/knowledge/search',
		headers={'Authorization': 'Bearer alpha-secret-token-12345'},
		json={'query': 'deterministic chunking', 'top_k': 3, 'hybrid': True},
	)
	assert search_res.status_code == 200
	items = search_res.json()
	assert len(items) >= 1
	assert 'chunking' in items[0]['content'].lower()


def test_research_api_returns_web_status_citations_and_conflicts(multi_tenant_client):
	client, _, engine = multi_tenant_client

	# Mock conduct_research to return detailed result
	async def mock_conduct(*args, **kwargs):
		return ResearchEngineResult(
			query='Verify database requirements',
			summary='PostgreSQL 17 is used in production with array fallback.',
			internal_evidence=[
				NormalizedEvidence(
					source_type='internal_knowledge',
					source_uri='db_spec.md',
					content='PostgreSQL 17 is required in production.',
					document_id='doc-test-1',
					chunk_id='chk-test-1',
				)
			],
			web_evidence=[],
			conflicts=[
				ConflictItem(
					topic='Database Version',
					claim_a='PostgreSQL 17',
					source_a='db_spec.md',
					claim_b='PostgreSQL 15',
					source_b='legacy_readme.md',
					description='Different versions cited across documents',
				)
			],
			citations=[
				ResearchCitation(
					source_type='internal_knowledge',
					source_uri='db_spec.md',
					document_id='doc-test-1',
					chunk_id='chk-test-1',
					snippet='PostgreSQL 17 is required in production.',
				)
			],
			is_grounded=True,
			sufficient_evidence=True,
			metadata={
				'web_status': {'status': 'skipped', 'reason': 'browser_agent_not_attached'},
			},
		)

	engine.conduct_research = mock_conduct

	res = client.post(
		'/research',
		headers={'Authorization': 'Bearer alpha-secret-token-12345'},
		json={'task': 'Verify database requirements', 'sync': True},
	)
	assert res.status_code == 200
	data = res.json()
	assert data['status'] == 'completed'
	assert data['web_status'] == {'status': 'skipped', 'reason': 'browser_agent_not_attached'}
	assert len(data['conflicts']) == 1
	assert data['conflicts'][0]['topic'] == 'Database Version'
	assert len(data['citations']) == 1
	assert data['citations'][0]['chunk_id'] == 'chk-test-1'
