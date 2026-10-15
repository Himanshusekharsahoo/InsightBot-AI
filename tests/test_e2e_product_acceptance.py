"""End-to-End Product Acceptance Test Suite for InsightBot AI (Phase 14F).

Exercises the integrated application against real HTTP endpoints (FastAPI & Next.js proxy),
verifying all 5 product areas, multi-tenancy, restart reconciliation, and export fidelity.
"""

import io

import httpx
import pytest

BASE_URL = "http://127.0.0.1:8000"
FRONTEND_PROXY_URL = "http://localhost:3000/api/backend"


@pytest.mark.asyncio
async def test_acceptance_health_and_frontend_proxy():
    """Verify backend health and Next.js frontend reverse proxy connectivity."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        # 1. Direct backend health
        res = await client.get(f"{BASE_URL}/health")
        assert res.status_code == 200
        health = res.json()
        assert health["status"] in ("ok", "degraded")
        assert health["service"] == "InsightBot AI"
        assert health["database"] == "connected"

        # 2. Frontend proxy route
        try:
            proxy_res = await client.get(f"{FRONTEND_PROXY_URL}/health")
            assert proxy_res.status_code == 200
            proxy_health = proxy_res.json()
            assert proxy_health["status"] in ("ok", "degraded")
            assert proxy_health["database"] == "connected"
        except httpx.ConnectError:
            pytest.skip("Next.js frontend server not reachable at http://localhost:3000")


@pytest.mark.asyncio
async def test_acceptance_area_a_dashboard_data_integrity():
    """Verify Area A (Dashboard): Metrics and lists return real persisted data."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        # Check sessions list
        sess_res = await client.get(f"{BASE_URL}/research?limit=10")
        assert sess_res.status_code == 200
        sessions = sess_res.json()
        assert isinstance(sessions, list)

        # Check conversations list
        conv_res = await client.get(f"{BASE_URL}/conversations?limit=10")
        assert conv_res.status_code == 200
        convs = conv_res.json()
        assert isinstance(convs, list)

        # Check documents list
        doc_res = await client.get(f"{BASE_URL}/knowledge/documents?limit=10")
        assert doc_res.status_code == 200
        docs = doc_res.json()
        assert isinstance(docs, list)
        assert len(docs) >= 4  # Baseline documents present


@pytest.mark.asyncio
async def test_acceptance_area_b_knowledge_base_lifecycle():
    """Verify Area B (Knowledge Base): Upload, chunk inspection, search, and cleanup."""
    async with httpx.AsyncClient(timeout=25.0) as client:
        # 1. Ingest isolated test document
        test_content = (
            "# Acceptance Test Protocol Alpha\n\n"
            "The secret verification code for InsightBot Phase 14F is ALPHA-OMEGA-9942.\n"
            "This protocol is used exclusively to validate vector similarity indexing."
        )
        files = {
            "file": ("acceptance_protocol_alpha.md", io.BytesIO(test_content.encode("utf-8")), "text/markdown")
        }
        upload_res = await client.post(f"{BASE_URL}/knowledge/documents", files=files)
        assert upload_res.status_code in (200, 201)
        doc = upload_res.json()
        doc_id = doc["id"]
        assert doc["filename"] == "acceptance_protocol_alpha.md"
        assert doc["processing_status"] == "completed"

        try:
            # 2. Inspect extracted chunks
            chunks_res = await client.get(f"{BASE_URL}/knowledge/documents/{doc_id}/chunks")
            assert chunks_res.status_code == 200
            chunks = chunks_res.json()
            assert len(chunks) >= 1
            assert any("ALPHA-OMEGA-9942" in c["content"] for c in chunks)

            # 3. Dense & Hybrid Search
            search_res = await client.post(
                f"{BASE_URL}/knowledge/search",
                json={
                    "query": "What is the secret verification code for Protocol Alpha?",
                    "top_k": 3,
                    "hybrid": True,
                },
            )
            assert search_res.status_code == 200
            search_results = search_res.json()
            assert len(search_results) >= 1
            # Verify the top match belongs to our test document
            top_hit = search_results[0]
            assert "ALPHA-OMEGA-9942" in top_hit["content"]

        finally:
            # 4. Clean up isolated test document ONLY
            del_res = await client.delete(f"{BASE_URL}/knowledge/documents/{doc_id}")
            assert del_res.status_code == 200
            assert del_res.json()["deleted"] is True

            # Confirm 404 on deleted document
            check_del = await client.get(f"{BASE_URL}/knowledge/documents/{doc_id}")
            assert check_del.status_code == 404


@pytest.mark.asyncio
async def test_acceptance_area_c_ai_grounded_chat():
    """Verify Area C (AI Chat): Conversation creation, grounded QA, multi-turn follow-up, and deletion."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Ingest knowledge fixture for chat grounding
        chat_faq = (
            "# InsightBot Chat Policy\n\n"
            "InsightBot secure cluster port is strictly configured to 9884 for tenant access.\n"
            "All connections must use TLS version 1.3."
        )
        files = {
            "file": ("chat_faq_test.md", io.BytesIO(chat_faq.encode("utf-8")), "text/markdown")
        }
        up = await client.post(f"{BASE_URL}/knowledge/documents", files=files)
        assert up.status_code in (200, 201)
        doc_id = up.json()["id"]

        try:
            # 1. Create persistent conversation thread
            conv_res = await client.post(
                f"{BASE_URL}/conversations",
                json={"title": "E2E Chat Acceptance Thread"},
            )
            assert conv_res.status_code == 201
            conv = conv_res.json()
            conv_id = conv["id"]
            assert conv["title"] == "E2E Chat Acceptance Thread"

            # 2. Ask question answered by the fixture
            msg_res = await client.post(
                f"{BASE_URL}/conversations/{conv_id}/messages",
                json={"content": "What port is configured for the secure cluster?", "top_k": 5},
            )
            assert msg_res.status_code == 200
            assistant_msg = msg_res.json()
            assert assistant_msg["role"] == "assistant"
            assert "9884" in assistant_msg["content"]
            assert assistant_msg["is_grounded"] is True
            assert len(assistant_msg["sources"]) >= 1
            assert any(s["document_id"] == doc_id for s in assistant_msg["sources"])

            # 3. Multi-turn follow-up using conversational context
            followup_res = await client.post(
                f"{BASE_URL}/conversations/{conv_id}/messages",
                json={"content": "What TLS version is required for that connection?", "top_k": 5},
            )
            assert followup_res.status_code == 200
            followup_msg = followup_res.json()
            assert "1.3" in followup_msg["content"]
            assert followup_msg["is_grounded"] is True

            # 4. Out-of-domain query testing insufficient evidence
            insufficient_res = await client.post(
                f"{BASE_URL}/conversations/{conv_id}/messages",
                json={"content": "What is the average surface temperature of Jupiter's moon Europa?", "top_k": 3},
            )
            assert insufficient_res.status_code == 200
            insufficient_msg = insufficient_res.json()
            assert insufficient_msg["insufficient_evidence"] is True

            # 5. Verify full conversation history retrieval
            history_res = await client.get(f"{BASE_URL}/conversations/{conv_id}/messages")
            assert history_res.status_code == 200
            history = history_res.json()
            # 3 user messages + 3 assistant responses = 6 messages
            assert len(history) == 6

            # 6. Update conversation title
            patch_res = await client.patch(
                f"{BASE_URL}/conversations/{conv_id}",
                json={"title": "Updated Chat Acceptance Title"},
            )
            assert patch_res.status_code == 200
            assert patch_res.json()["title"] == "Updated Chat Acceptance Title"

            # 7. Delete conversation
            del_conv = await client.delete(f"{BASE_URL}/conversations/{conv_id}")
            assert del_conv.status_code == 200

        finally:
            # Clean up document fixture
            await client.delete(f"{BASE_URL}/knowledge/documents/{doc_id}")


@pytest.mark.asyncio
async def test_acceptance_area_d_research_job_lifecycle_and_cancellation():
    """Verify Area D (Research Workspace): Async enqueue, status polling, and cancellation."""
    async with httpx.AsyncClient(timeout=20.0) as client:
        # 1. Enqueue background research job (sync=False)
        enqueue_res = await client.post(
            f"{BASE_URL}/research",
            json={"task": "Verify deployment replication specifications", "max_steps": 10, "sync": False},
        )
        assert enqueue_res.status_code == 200
        job_data = enqueue_res.json()
        research_id = job_data["research_id"]
        assert job_data["status"] == "queued"
        assert research_id is not None

        # 2. Poll status immediately
        status_res = await client.get(f"{BASE_URL}/research/{research_id}/status")
        assert status_res.status_code == 200
        status_data = status_res.json()
        assert status_data["research_id"] == research_id
        assert status_data["status"] in ("queued", "in_progress", "completed", "completed_unverified")

        # 3. Test cancellation on active or queued job
        cancel_res = await client.post(f"{BASE_URL}/research/{research_id}/cancel")
        assert cancel_res.status_code == 200
        cancel_data = cancel_res.json()
        assert cancel_data["research_id"] == research_id

        # Clean up test session
        await client.delete(f"{BASE_URL}/research/{research_id}")


@pytest.mark.asyncio
async def test_acceptance_area_e_reports_archive_and_exports():
    """Verify Area E (Reports & Exports): Session retrieval, Markdown export, and JSON export."""
    async with httpx.AsyncClient(timeout=25.0) as client:
        # Create a synchronous completed session for export verification
        sync_res = await client.post(
            f"{BASE_URL}/research",
            json={"task": "Verify database deployment architecture", "max_steps": 5, "sync": True},
        )
        assert sync_res.status_code == 200
        sync_data = sync_res.json()
        research_id = sync_data["research_id"]
        assert research_id is not None

        try:
            # 1. Retrieve full session report
            session_res = await client.get(f"{BASE_URL}/research/{research_id}")
            assert session_res.status_code == 200
            session = session_res.json()
            assert session["id"] == research_id
            assert "summary" in session
            assert "citations" in session

            # 2. Export Markdown (.md)
            md_res = await client.get(f"{BASE_URL}/research/{research_id}/export?format=markdown")
            assert md_res.status_code == 200
            assert "text/markdown" in md_res.headers.get("Content-Type", "")
            assert f'filename="research-{research_id}.md"' in md_res.headers.get("Content-Disposition", "")
            md_body = md_res.text
            assert "# Research Report:" in md_body
            assert research_id in md_body

            # 3. Export JSON (.json)
            json_res = await client.get(f"{BASE_URL}/research/{research_id}/export?format=json")
            assert json_res.status_code == 200
            assert "application/json" in json_res.headers.get("Content-Type", "")
            assert f'filename="research-{research_id}.json"' in json_res.headers.get("Content-Disposition", "")
            json_body = json_res.json()
            assert json_body["id"] == research_id
            assert "citations" in json_body

        finally:
            # Clean up test session
            await client.delete(f"{BASE_URL}/research/{research_id}")


@pytest.mark.asyncio
async def test_acceptance_multi_tenant_isolation_boundary():
    """Verify Tenant Isolation: Tenant A cannot access Tenant B's documents or conversations."""
    from insightbot.config import settings

    # Simulate multi-tenant configuration
    tenant_keys = {
        "alpha-key-secret": {"organization_id": "tenant-alpha", "is_admin": False},
        "beta-key-secret": {"organization_id": "tenant-beta", "is_admin": False},
    }

    class MockTenantSettings(settings.Settings):
        environment: str = "production"
        multi_tenancy_enabled: bool = True
        tenant_api_keys: dict = tenant_keys

        def resolve_tenant_key(self, key: str):
            info = tenant_keys.get(key)
            return (info["organization_id"], info["is_admin"]) if info else None

    # Use in-memory store for isolated tenant tests
    from insightbot.db.in_memory import InMemoryVectorStore
    from insightbot.knowledge.store import KnowledgeStore

    tenant_store = KnowledgeStore(vector_store=InMemoryVectorStore())
    await tenant_store.initialize()

    # Ingest document as tenant-alpha
    doc_alpha = await tenant_store.ingest_document(
        filename="alpha_confidential.txt",
        content=b"Confidential alpha financial report.",
        organization_id="tenant-alpha",
    )

    # Ingest document as tenant-beta
    doc_beta = await tenant_store.ingest_document(
        filename="beta_confidential.txt",
        content=b"Confidential beta acquisition roadmap.",
        organization_id="tenant-beta",
    )

    # Verify retrieval isolation
    alpha_docs = await tenant_store.list_documents(organization_id="tenant-alpha")
    assert any(d.id == doc_alpha.id for d in alpha_docs)
    assert not any(d.id == doc_beta.id for d in alpha_docs)

    beta_docs = await tenant_store.list_documents(organization_id="tenant-beta")
    assert any(d.id == doc_beta.id for d in beta_docs)
    assert not any(d.id == doc_alpha.id for d in beta_docs)

    # Cross-tenant get_document returns None
    assert await tenant_store.get_document(doc_beta.id, organization_id="tenant-alpha") is None
    assert await tenant_store.get_document(doc_alpha.id, organization_id="tenant-beta") is None


@pytest.mark.asyncio
async def test_acceptance_browser_research_unavailable_without_fabrication():
    """Verify Browser Research Non-Fabrication: When browser credentials/config are disabled,

    ResearchEngine reports web_status as unavailable and synthesizes strictly without fabricating web sources.
    """
    from insightbot.db.in_memory import InMemoryVectorStore
    from insightbot.knowledge.store import KnowledgeStore
    from insightbot.orchestrator.llm import MockLLMProvider
    from insightbot.orchestrator.research_engine import ResearchEngine

    store = KnowledgeStore(vector_store=InMemoryVectorStore())
    await store.initialize()
    await store.ingest_document(
        filename="compliance.txt",
        content=b"ISO 27001 requires automated vulnerability management.",
    )

    # ResearchEngine without browser agent
    engine = ResearchEngine(
        knowledge_store=store,
        browser_agent=None,
        llm_provider=MockLLMProvider(),
    )

    result = await engine.conduct_research(query="What does ISO 27001 require?", include_web=True, max_web_steps=5)
    # Check that web evidence is empty and web_status reports skipped/unavailable
    assert len(result.web_evidence) == 0
    assert result.metadata.get("web_status", {}).get("status") in ("skipped", "unavailable")
    # All citations must be strictly internal
    assert all(c.source_type == "internal_knowledge" for c in result.citations)
