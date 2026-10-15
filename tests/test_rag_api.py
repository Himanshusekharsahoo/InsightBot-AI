"""Unit tests for FastAPI endpoints: health, documents, search, and ask."""

import base64

import pytest
from fastapi.testclient import TestClient

from insightbot.api.routes import app, get_agent, get_store
from insightbot.db.in_memory import InMemoryVectorStore
from insightbot.knowledge.embeddings import MockEmbeddingModel
from insightbot.knowledge.store import KnowledgeStore
from insightbot.orchestrator.knowledge_agent import KnowledgeAgent


@pytest.fixture
def test_client(tmp_path):
	mem_store = InMemoryVectorStore()
	mock_embed = MockEmbeddingModel(dimension=64)
	ks = KnowledgeStore(storage_dir=tmp_path, vector_store=mem_store, embedding_provider=mock_embed)

	async def mock_llm(q, ctx):
		return 'Grounded answer from test evidence.'

	agent = KnowledgeAgent(knowledge_store=ks, llm=mock_llm)

	app.dependency_overrides[get_store] = lambda: ks
	app.dependency_overrides[get_agent] = lambda: agent

	client = TestClient(app)
	yield client

	app.dependency_overrides.clear()


def test_api_health(test_client):
	res = test_client.get('/health')
	assert res.status_code == 200
	data = res.json()
	assert data['status'] == 'ok'
	assert data['service'] == 'InsightBot AI'
	assert data['storage_mode'] == 'in_memory'
	assert data['database'] == 'connected'


def test_api_document_lifecycle(test_client):
	# 1. Ingest via multipart file upload
	file_content = b'InsightBot deployment architecture with PostgreSQL.'
	res = test_client.post(
		'/knowledge/documents',
		files={'file': ('architecture.txt', file_content, 'text/plain')},
	)
	assert res.status_code == 201
	doc = res.json()
	doc_id = doc['id']
	assert doc['filename'] == 'architecture.txt'
	assert doc['processing_status'] == 'completed'

	# 2. List documents
	list_res = test_client.get('/knowledge/documents')
	assert list_res.status_code == 200
	docs = list_res.json()
	assert len(docs) == 1
	assert docs[0]['id'] == doc_id

	# 3. Get single document
	get_res = test_client.get(f'/knowledge/documents/{doc_id}')
	assert get_res.status_code == 200
	assert get_res.json()['id'] == doc_id

	# 4. Semantic search
	search_res = test_client.post(
		'/knowledge/search',
		json={'query': 'PostgreSQL deployment', 'top_k': 3},
	)
	assert search_res.status_code == 200
	search_items = search_res.json()
	assert len(search_items) >= 1
	assert search_items[0]['document_id'] == doc_id

	# 5. Grounded ask
	ask_res = test_client.post(
		'/knowledge/ask',
		json={'question': 'What database is used?', 'top_k': 3},
	)
	assert ask_res.status_code == 200
	answer_data = ask_res.json()
	assert answer_data['is_grounded'] is True
	assert len(answer_data['sources']) >= 1

	# 6. Delete document
	del_res = test_client.delete(f'/knowledge/documents/{doc_id}')
	assert del_res.status_code == 200
	assert del_res.json()['deleted'] is True

	# 7. Confirm deleted
	missing_res = test_client.get(f'/knowledge/documents/{doc_id}')
	assert missing_res.status_code == 404


def test_api_json_base64_upload(test_client):
	raw_text = 'Content uploaded as base64 string.'
	b64_content = base64.b64encode(raw_text.encode('utf-8')).decode('ascii')

	res = test_client.post(
		'/knowledge/documents',
		json={
			'filename': 'notes.md',
			'content_base64': b64_content,
			'metadata': {'tags': ['test']},
		},
	)
	assert res.status_code == 201
	assert res.json()['source_type'] == 'markdown'


def test_api_upload_path_traversal_rejection(test_client):
	res = test_client.post(
		'/knowledge/documents',
		files={'file': ('../../etc/passwd', b'root:x:0:0', 'text/plain')},
	)
	# Should be rejected because .passwd has no valid extension or is invalid
	assert res.status_code == 400


def test_api_upload_oversized_rejection(test_client, monkeypatch):
	from insightbot.config.settings import get_settings

	monkeypatch.setattr(get_settings(), 'max_upload_size_bytes', 50)
	res = test_client.post(
		'/knowledge/documents',
		files={'file': ('oversized.txt', b'A' * 100, 'text/plain')},
	)
	assert res.status_code == 400
	assert 'exceeds maximum allowed limit' in res.json()['detail']


def test_api_health_degraded(test_client):
	class FailingStore:
		async def initialize(self):
			raise ConnectionError('Database offline')

	app.dependency_overrides[get_store] = lambda: FailingStore()
	res = test_client.get('/health')
	assert res.status_code == 200
	data = res.json()
	assert data['status'] == 'degraded'
	assert data['database'] == 'unavailable'


def test_api_unsupported_file_extension_rejection(test_client):
	res = test_client.post(
		'/knowledge/documents',
		files={'file': ('script.py', b'print("hello")', 'text/x-python')},
	)
	assert res.status_code == 400
	assert 'Unsupported file extension' in res.json()['detail']


def test_api_error_sanitization_no_credentials_leak(test_client):
	class FailingIngestStore:
		async def initialize(self):
			pass

		async def ingest_document(self, **kwargs):
			raise ConnectionError('postgresql://admin:super_secret_pw@db.internal:5432/secrets failed')

	app.dependency_overrides[get_store] = lambda: FailingIngestStore()
	res = test_client.post(
		'/knowledge/documents',
		files={'file': ('test.txt', b'some content', 'text/plain')},
	)
	assert res.status_code == 500
	detail = res.json()['detail']
	assert 'super_secret_pw' not in detail
	assert 'postgresql://' not in detail
	assert 'ConnectionError' in detail


def test_api_research_endpoint(test_client):
	from insightbot.api.routes import get_research_engine
	from insightbot.orchestrator.research_engine import (
		NormalizedEvidence,
		ResearchCitation,
		ResearchEngineResult,
	)

	class MockResearchEngine:
		async def conduct_research(self, query: str, max_web_steps: int = 25, include_web: bool = True):
			return ResearchEngineResult(
				query=query,
				summary='Grounded research summary on architecture.',
				internal_evidence=[
					NormalizedEvidence(
						source_type='internal_knowledge',
						source_uri='architecture.txt',
						content='PostgreSQL deployment',
					)
				],
				citations=[
					ResearchCitation(
						citation_id='cit-1',
						source_type='internal_knowledge',
						source_uri='architecture.txt',
						snippet='PostgreSQL deployment',
					)
				],
				is_grounded=True,
				metadata={'doc_id': 'doc-res-123'},
			)

	app.dependency_overrides[get_research_engine] = lambda: MockResearchEngine()
	res = test_client.post(
		'/research',
		json={'task': 'Research PostgreSQL deployment', 'max_steps': 10, 'sync': True},
	)
	assert res.status_code == 200
	data = res.json()
	assert data['status'] == 'completed'
	assert data['task'] == 'Research PostgreSQL deployment'
	assert 'architecture' in data['summary']
	assert data['sources'] == ['architecture.txt']
	assert data['evidence_count'] == 1
	assert data['doc_id'] == 'doc-res-123'


def test_api_auth_enforcement_when_api_key_configured(test_client, monkeypatch):
	"""Verify API enforces X-API-Key or Bearer token when configured, while keeping /health open."""
	from insightbot.config import settings

	monkeypatch.setenv('INSIGHTBOT_API_KEY', 'prod_secret_token_7788')
	settings.get_settings.cache_clear()

	try:
		# 1. Health check is always open
		health_res = test_client.get('/health')
		assert health_res.status_code == 200

		# 2. Protected endpoint without key is rejected
		unauth_res = test_client.get('/knowledge/documents')
		assert unauth_res.status_code == 401
		assert 'Unauthorized' in unauth_res.json()['detail']

		# 3. Protected endpoint with invalid key is rejected
		bad_res = test_client.get('/knowledge/documents', headers={'X-API-Key': 'wrong_key'})
		assert bad_res.status_code == 401

		# 4. Protected endpoint with valid X-API-Key succeeds
		valid_header_res = test_client.get('/knowledge/documents', headers={'X-API-Key': 'prod_secret_token_7788'})
		assert valid_header_res.status_code == 200

		# 5. Protected endpoint with valid Bearer token succeeds
		valid_bearer_res = test_client.get('/knowledge/documents', headers={'Authorization': 'Bearer prod_secret_token_7788'})
		assert valid_bearer_res.status_code == 200
	finally:
		settings.get_settings.cache_clear()


def test_api_json_payload_oversized_rejection(test_client, monkeypatch):
	"""Verify JSON upload exceeding max_upload_size_bytes is rejected with 400."""
	from insightbot.config import settings

	# Set small limit of 50 bytes
	monkeypatch.setenv('INSIGHTBOT_MAX_UPLOAD_SIZE', '50')
	settings.get_settings.cache_clear()

	try:
		oversized_bytes = b'A' * 100
		b64_content = base64.b64encode(oversized_bytes).decode('ascii')
		res = test_client.post(
			'/knowledge/documents',
			json={'filename': 'large.txt', 'content_base64': b64_content},
		)
		assert res.status_code == 400
		assert 'exceeds maximum allowed limit' in res.json()['detail']
	finally:
		settings.get_settings.cache_clear()
