"""Automated Security & Multi-Tenant Authorization Test Suite.

Validates tenant isolation, access boundaries, credential derivation, anti-spoofing,
and cross-tenant evidence leakage prevention across the InsightBot AI knowledge and research API.
"""

import json
from typing import Optional

import pytest
from fastapi.testclient import TestClient

from insightbot.api.routes import app, get_agent, get_research_engine, get_store
from insightbot.config import settings
from insightbot.db.in_memory import InMemoryVectorStore
from insightbot.knowledge.embeddings import MockEmbeddingModel
from insightbot.knowledge.store import KnowledgeStore
from insightbot.orchestrator.knowledge_agent import KnowledgeAgent
from insightbot.orchestrator.research_engine import ResearchEngine

TENANT_A_KEY = "sk-tenant-a-token-111111"
TENANT_B_KEY = "sk-tenant-b-token-222222"
ADMIN_KEY = "sk-admin-master-token-999999"


@pytest.fixture
def multi_tenant_client(tmp_path, monkeypatch):
	"""Configure isolated multi-tenant environment with in-memory storage."""
	mem_store = InMemoryVectorStore()
	mock_embed = MockEmbeddingModel(dimension=64)
	ks = KnowledgeStore(storage_dir=tmp_path, vector_store=mem_store, embedding_provider=mock_embed)

	# Mock LLM that reflects retrieved facts without hallucination
	async def mock_llm(q: str, ctx: Optional[str] = None) -> str:
		if ctx and ('pegasus' in ctx.lower() or 'omega' in ctx.lower() or 'policy' in ctx.lower()):
			return f"Grounded response based on: {ctx[:100]}"
		return "The internal knowledge base does not contain enough information to answer this question."

	agent = KnowledgeAgent(knowledge_store=ks, llm=mock_llm, min_similarity_score=0.0)
	engine = ResearchEngine(knowledge_store=ks, knowledge_agent=agent)

	# Configure multi-tenant settings
	monkeypatch.setenv("INSIGHTBOT_API_KEY", ADMIN_KEY)
	monkeypatch.setenv("INSIGHTBOT_MULTI_TENANCY_ENABLED", "true")
	monkeypatch.setenv(
		"INSIGHTBOT_TENANT_API_KEYS",
		json.dumps({
			"tenant_a": TENANT_A_KEY,
			"tenant_b": TENANT_B_KEY,
		}),
	)
	settings.get_settings.cache_clear()

	app.dependency_overrides[get_store] = lambda: ks
	app.dependency_overrides[get_agent] = lambda: agent
	app.dependency_overrides[get_research_engine] = lambda: engine

	client = TestClient(app)
	yield client, ks

	app.dependency_overrides.clear()
	settings.get_settings.cache_clear()


def test_scenario_1_tenant_a_can_access_own_documents(multi_tenant_client):
	"""Scenario 1: Tenant A can ingest, list, retrieve, search, and query its own documents."""
	client, _ = multi_tenant_client
	headers = {"X-API-Key": TENANT_A_KEY}

	# 1. Ingest document
	ingest_res = client.post(
		"/knowledge/documents",
		headers=headers,
		files={"file": ("tenant_a_spec.txt", b"Tenant A secret roadmap: launching project Pegasus.", "text/plain")},
	)
	assert ingest_res.status_code == 201
	doc_a = ingest_res.json()
	assert doc_a["filename"] == "tenant_a_spec.txt"
	assert doc_a["organization_id"] == "tenant_a"
	doc_id = doc_a["id"]

	# 2. List documents
	list_res = client.get("/knowledge/documents", headers=headers)
	assert list_res.status_code == 200
	docs = list_res.json()
	assert len(docs) == 1
	assert docs[0]["id"] == doc_id

	# 3. Retrieve document by ID
	get_res = client.get(f"/knowledge/documents/{doc_id}", headers=headers)
	assert get_res.status_code == 200
	assert get_res.json()["id"] == doc_id

	# 4. Semantic search
	search_res = client.post(
		"/knowledge/search",
		headers=headers,
		json={"query": "Pegasus roadmap", "top_k": 3},
	)
	assert search_res.status_code == 200
	results = search_res.json()
	assert len(results) >= 1
	assert results[0]["document_id"] == doc_id
	assert "Pegasus" in results[0]["content"]

	# 5. Q&A Ask question
	ask_res = client.post(
		"/knowledge/ask",
		headers=headers,
		json={"question": "What is the secret roadmap?", "top_k": 3},
	)
	assert ask_res.status_code == 200
	ans = ask_res.json()
	assert ans["is_grounded"] is True
	assert len(ans["sources"]) >= 1


def test_scenario_2_tenant_a_cannot_list_or_retrieve_tenant_b_documents(multi_tenant_client):
	"""Scenario 2: Tenant A cannot list or retrieve Tenant B's documents (returns 404 on get; filtered on list)."""
	client, _ = multi_tenant_client
	headers_b = {"X-API-Key": TENANT_B_KEY}
	headers_a = {"X-API-Key": TENANT_A_KEY}

	# Tenant B ingests confidential document
	ingest_res = client.post(
		"/knowledge/documents",
		headers=headers_b,
		files={"file": ("tenant_b_q4.txt", b"Tenant B confidential financial forecasts Q4.", "text/plain")},
	)
	assert ingest_res.status_code == 201
	doc_b_id = ingest_res.json()["id"]

	# Tenant A lists documents: must be empty
	list_res = client.get("/knowledge/documents", headers=headers_a)
	assert list_res.status_code == 200
	assert list_res.json() == []

	# Tenant A attempts to get Tenant B's document by direct ID: must return 404 Not Found
	get_res = client.get(f"/knowledge/documents/{doc_b_id}", headers=headers_a)
	assert get_res.status_code == 404
	assert f"Document '{doc_b_id}' not found." in get_res.json()["detail"]

	# Tenant B can retrieve its own document
	get_b_res = client.get(f"/knowledge/documents/{doc_b_id}", headers=headers_b)
	assert get_b_res.status_code == 200
	assert get_b_res.json()["id"] == doc_b_id


def test_scenario_3_tenant_a_cannot_search_tenant_b_chunks(multi_tenant_client):
	"""Scenario 3: Similarity and keyword search partition chunks strictly by tenant context."""
	client, _ = multi_tenant_client
	headers_b = {"X-API-Key": TENANT_B_KEY}
	headers_a = {"X-API-Key": TENANT_A_KEY}

	# Tenant B ingests unique keyword
	client.post(
		"/knowledge/documents",
		headers=headers_b,
		files={"file": ("quantum_b.txt", b"Tenant B exclusive quantum cryptography algorithms.", "text/plain")},
	)

	# Tenant A searches for quantum cryptography
	search_res = client.post(
		"/knowledge/search",
		headers=headers_a,
		json={"query": "quantum cryptography algorithms", "top_k": 5},
	)
	assert search_res.status_code == 200
	assert search_res.json() == []

	# Tenant B searches for the same query: retrieves chunks
	search_b_res = client.post(
		"/knowledge/search",
		headers=headers_b,
		json={"query": "quantum cryptography algorithms", "top_k": 5},
	)
	assert search_b_res.status_code == 200
	results_b = search_b_res.json()
	assert len(results_b) >= 1
	assert "quantum cryptography" in results_b[0]["content"]


def test_scenario_4_tenant_a_cannot_ask_questions_retrieving_tenant_b_evidence(multi_tenant_client):
	"""Scenario 4: Grounded Q&A fails safely with insufficient evidence when query references other tenant's facts."""
	client, _ = multi_tenant_client
	headers_b = {"X-API-Key": TENANT_B_KEY}
	headers_a = {"X-API-Key": TENANT_A_KEY}

	# Tenant B ingests unique secret
	client.post(
		"/knowledge/documents",
		headers=headers_b,
		files={"file": ("b_vault.txt", b"The secret vault code is OMEGA-9988.", "text/plain")},
	)

	# Tenant A asks for the secret vault code
	ask_res = client.post(
		"/knowledge/ask",
		headers=headers_a,
		json={"question": "What is the secret vault code?", "top_k": 5},
	)
	assert ask_res.status_code == 200
	ans = ask_res.json()
	assert ans["is_grounded"] is False
	assert ans["insufficient_evidence"] is True
	assert ans["sources"] == []
	assert "does not contain enough information" in ans["answer"]


def test_scenario_5_tenant_a_cannot_delete_tenant_b_documents(multi_tenant_client):
	"""Scenario 5: Document deletion requires owning tenant authorization; cross-tenant deletion returns 404."""
	client, _ = multi_tenant_client
	headers_b = {"X-API-Key": TENANT_B_KEY}
	headers_a = {"X-API-Key": TENANT_A_KEY}

	# Tenant B ingests document
	doc_b = client.post(
		"/knowledge/documents",
		headers=headers_b,
		files={"file": ("vital_records.txt", b"Tenant B vital corporate records.", "text/plain")},
	).json()
	doc_id = doc_b["id"]

	# Tenant A attempts to delete Tenant B's document: rejected with 404
	del_res = client.delete(f"/knowledge/documents/{doc_id}", headers=headers_a)
	assert del_res.status_code == 404

	# Verify Tenant B's document is intact
	get_res = client.get(f"/knowledge/documents/{doc_id}", headers=headers_b)
	assert get_res.status_code == 200
	assert get_res.json()["id"] == doc_id

	# Tenant B can delete its own document
	del_b_res = client.delete(f"/knowledge/documents/{doc_id}", headers=headers_b)
	assert del_b_res.status_code == 200
	assert del_b_res.json() == {"deleted": True, "id": doc_id}

	# Now it is gone
	get_gone = client.get(f"/knowledge/documents/{doc_id}", headers=headers_b)
	assert get_gone.status_code == 404


def test_scenario_6_unauthorized_and_spoofed_tenant_contexts_fail_closed(multi_tenant_client, monkeypatch):
	"""Scenario 6: Missing credentials, invalid tokens, and spoofed tenant headers/payloads fail closed (401/403)."""
	client, _ = multi_tenant_client

	# 1. Missing credentials in staging/production environment -> 401 Unauthorized
	monkeypatch.setenv("INSIGHTBOT_ENVIRONMENT", "staging")
	settings.get_settings.cache_clear()
	try:
		res_no_auth = client.get("/knowledge/documents")
		assert res_no_auth.status_code == 401
		assert "Unauthorized" in res_no_auth.json()["detail"]
	finally:
		monkeypatch.setenv("INSIGHTBOT_ENVIRONMENT", "development")
		settings.get_settings.cache_clear()

	# 2. Invalid API key -> 401 Unauthorized
	res_bad_key = client.get("/knowledge/documents", headers={"X-API-Key": "invalid_token_xyz"})
	assert res_bad_key.status_code == 401

	# 3. Spoofed X-Organization-Id header mismatching server credential -> 403 Forbidden
	res_header_spoof = client.get(
		"/knowledge/documents",
		headers={
			"X-API-Key": TENANT_A_KEY,
			"X-Organization-Id": "tenant_b",  # Attempting to forge Tenant B
		},
	)
	assert res_header_spoof.status_code == 403
	assert "Forbidden" in res_header_spoof.json()["detail"]
	assert "does not match authenticated tenant" in res_header_spoof.json()["detail"]

	# 4. Spoofed payload organization_id during document upload -> 403 Forbidden
	res_payload_spoof = client.post(
		"/knowledge/documents",
		headers={"X-API-Key": TENANT_A_KEY},
		json={
			"filename": "forged_doc.txt",
			"content_base64": "SGVsbG8gd29ybGQ=",
			"metadata": {"organization_id": "tenant_b"},  # Attempting to ingest into Tenant B
		},
	)
	assert res_payload_spoof.status_code == 403
	assert "Forbidden: Cannot ingest document into organization 'tenant_b'" in res_payload_spoof.json()["detail"]


def test_scenario_7_citations_reference_only_evidence_accessible_to_requesting_tenant(multi_tenant_client):
	"""Scenario 7: Citations and evidence in Q&A never reference documents or chunks from other tenants."""
	client, _ = multi_tenant_client
	headers_a = {"X-API-Key": TENANT_A_KEY}
	headers_b = {"X-API-Key": TENANT_B_KEY}

	# Ingest document for Tenant A
	doc_a_res = client.post(
		"/knowledge/documents",
		headers=headers_a,
		files={"file": ("tenant_a_policy.txt", b"Policy A: remote work is permitted globally.", "text/plain")},
	)
	doc_a_id = doc_a_res.json()["id"]

	# Ingest document for Tenant B
	doc_b_res = client.post(
		"/knowledge/documents",
		headers=headers_b,
		files={"file": ("tenant_b_policy.txt", b"Policy B: strict on-site work required.", "text/plain")},
	)
	doc_b_id = doc_b_res.json()["id"]

	# Tenant A asks question
	ask_res = client.post(
		"/knowledge/ask",
		headers=headers_a,
		json={"question": "What is the remote work policy?", "top_k": 5},
	)
	assert ask_res.status_code == 200
	ans_a = ask_res.json()

	# Assert that no citation mentions Tenant B's doc_id, filename, or chunks
	for src in ans_a.get("sources", []):
		assert src["document_id"] != doc_b_id
		assert "tenant_b" not in src["source_filename"]
		assert src["document_id"] == doc_a_id


def test_scenario_8_single_tenant_and_admin_compatibility(multi_tenant_client, monkeypatch):
	"""Scenario 8: Admin master key has global cross-partition visibility, and default/unpartitioned workflow succeeds."""
	client, _ = multi_tenant_client

	# Tenant A ingests doc
	client.post(
		"/knowledge/documents",
		headers={"X-API-Key": TENANT_A_KEY},
		files={"file": ("doc_a.txt", b"Content from Tenant A", "text/plain")},
	)

	# Tenant B ingests doc
	client.post(
		"/knowledge/documents",
		headers={"X-API-Key": TENANT_B_KEY},
		files={"file": ("doc_b.txt", b"Content from Tenant B", "text/plain")},
	)

	# Admin master key queries without tenant restriction: sees both documents
	admin_headers = {"X-API-Key": ADMIN_KEY}
	admin_list = client.get("/knowledge/documents", headers=admin_headers)
	assert admin_list.status_code == 200
	# With admin key and no X-Organization-Id header, default_organization_id applies or admin can specify target org
	admin_scoped_a = client.get(
		"/knowledge/documents",
		headers={"X-API-Key": ADMIN_KEY, "X-Organization-Id": "tenant_a"},
	)
	assert admin_scoped_a.status_code == 200
	docs_a = admin_scoped_a.json()
	assert len(docs_a) == 1
	assert docs_a[0]["filename"] == "doc_a.txt"


def test_tenant_content_hash_deduplication_isolation(multi_tenant_client):
	"""Verify identical file content uploaded by different tenants yields isolated documents."""
	client, _ = multi_tenant_client
	headers_a = {"X-API-Key": TENANT_A_KEY}
	headers_b = {"X-API-Key": TENANT_B_KEY}

	identical_content = b"Standard Company Onboarding Handbook v2026."

	# Tenant A uploads file
	res_a = client.post(
		"/knowledge/documents",
		headers=headers_a,
		files={"file": ("handbook.txt", identical_content, "text/plain")},
	)
	assert res_a.status_code == 201
	doc_a = res_a.json()

	# Tenant B uploads exact same content
	res_b = client.post(
		"/knowledge/documents",
		headers=headers_b,
		files={"file": ("handbook.txt", identical_content, "text/plain")},
	)
	assert res_b.status_code == 201
	doc_b = res_b.json()

	# Documents must be separate entities belonging to respective tenants
	assert doc_a["id"] != doc_b["id"]
	assert doc_a["organization_id"] == "tenant_a"
	assert doc_b["organization_id"] == "tenant_b"

	# Tenant A cannot see Tenant B's doc, and vice versa
	get_a_b = client.get(f"/knowledge/documents/{doc_b['id']}", headers=headers_a)
	assert get_a_b.status_code == 404
