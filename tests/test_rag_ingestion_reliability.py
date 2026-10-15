"""Unit and integration tests for document ingestion reliability and resilience.

Validates:
1. Supported file formats (PDF, DOCX, TXT, MD) and metadata preservation.
2. Unsupported file extensions rejection with client-facing error.
3. Empty (0-byte) and whitespace-only files rejection (no orphaned completed entries).
4. Corrupted and malformed file handling without credential/path leakage.
5. Embedding failure rollback leaving zero orphaned documents or chunks.
6. Pre-existing document failure preservation (non-destructive cleanup).
7. Duplicate content hash deduplication returning existing document ID.
8. Maximum chunk limit per document enforcement.
9. FastAPI route level status codes (400 validation error vs 500 server error).
"""

import base64
import io

import docx
import pytest
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

from insightbot.api.routes import app, get_agent, get_store
from insightbot.db.in_memory import InMemoryVectorStore
from insightbot.db.models import Document
from insightbot.knowledge.embeddings import MockEmbeddingModel
from insightbot.knowledge.parsers import DocumentParsingError
from insightbot.knowledge.store import KnowledgeStore
from insightbot.knowledge.validation import SecurityValidationError
from insightbot.orchestrator.knowledge_agent import KnowledgeAgent


def _create_sample_pdf(pages_text: list[str]) -> bytes:
	"""Generate in-memory synthetic PDF bytes."""
	buffer = io.BytesIO()
	canv = canvas.Canvas(buffer)
	for text in pages_text:
		canv.drawString(100, 750, text)
		canv.showPage()
	canv.save()
	return buffer.getvalue()


def _create_sample_docx(paragraphs: list[str]) -> bytes:
	"""Generate in-memory synthetic DOCX bytes."""
	doc = docx.Document()
	for p in paragraphs:
		doc.add_paragraph(p)
	buffer = io.BytesIO()
	doc.save(buffer)
	return buffer.getvalue()


class FailingEmbeddingModel(MockEmbeddingModel):
	"""Embedding model that simulates network/inference failures."""

	async def embed_batch(self, texts: list[str]) -> list[list[float]]:
		raise RuntimeError("Simulated embedding inference timeout with key sk-proj-abc1234567890def1234567890")


@pytest.fixture
def mem_store():
	return InMemoryVectorStore(embedding_dim=64)


@pytest.fixture
def mock_embed():
	return MockEmbeddingModel(dimension=64)


@pytest.fixture
def knowledge_store(tmp_path, mem_store, mock_embed):
	return KnowledgeStore(storage_dir=tmp_path, vector_store=mem_store, embedding_provider=mock_embed)


@pytest.fixture
def test_client(tmp_path, knowledge_store):
	async def mock_llm(q, ctx):
		return "Grounded answer from evidence."

	agent = KnowledgeAgent(knowledge_store=knowledge_store, llm=mock_llm)

	app.dependency_overrides[get_store] = lambda: knowledge_store
	app.dependency_overrides[get_agent] = lambda: agent

	client = TestClient(app)
	yield client
	app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_ingest_all_supported_formats(knowledge_store, mem_store):
	"""Verify all supported formats (TXT, MD, PDF, DOCX) ingest successfully with metadata."""
	# 1. TXT
	txt_content = b"Plain text architecture overview of InsightBot AI."
	txt_doc = await knowledge_store.ingest_document(
		filename="overview.txt",
		content=txt_content,
		metadata={"department": "Platform", "version": "1.0"},
	)
	assert txt_doc.processing_status == "completed"
	assert txt_doc.source_type == "txt"
	assert txt_doc.metadata.get("department") == "Platform"

	# 2. Markdown
	md_content = b"# Architecture\n\nInsightBot RAG supports deterministic chunking."
	md_doc = await knowledge_store.ingest_document(
		filename="architecture.md",
		content=md_content,
		metadata={"category": "spec"},
	)
	assert md_doc.processing_status == "completed"
	assert md_doc.source_type == "markdown"
	assert md_doc.metadata.get("category") == "spec"

	# 3. PDF
	pdf_bytes = _create_sample_pdf(["Page 1: Deployment topologies.", "Page 2: Database replication."])
	pdf_doc = await knowledge_store.ingest_document(
		filename="deployment.pdf",
		content=pdf_bytes,
		metadata={"confidentiality": "internal"},
	)
	assert pdf_doc.processing_status == "completed"
	assert pdf_doc.source_type == "pdf"
	assert pdf_doc.metadata.get("page_count") == 2

	# 4. DOCX
	docx_bytes = _create_sample_docx(["Specification paragraph 1.", "Specification paragraph 2."])
	docx_doc = await knowledge_store.ingest_document(
		filename="spec.docx",
		content=docx_bytes,
		metadata={"reviewed": True},
	)
	assert docx_doc.processing_status == "completed"
	assert docx_doc.source_type == "docx"

	# Confirm 4 documents persisted in vector store
	docs = await knowledge_store.list_documents()
	assert len(docs) == 4


@pytest.mark.asyncio
async def test_unsupported_extensions_rejected(knowledge_store):
	"""Verify unsupported file extensions are rejected with clear error."""
	unsupported_files = [
		("script.exe", b"MZ\x90\x00executable content"),
		("payload.json", b'{"key": "value"}'),
		("data.csv", b"col1,col2\nval1,val2"),
		("run.sh", b"#!/bin/bash\necho hello"),
	]

	for filename, content in unsupported_files:
		with pytest.raises(SecurityValidationError) as exc_info:
			await knowledge_store.ingest_document(filename=filename, content=content)
		assert "Unsupported file extension" in str(exc_info.value)


@pytest.mark.asyncio
async def test_empty_and_whitespace_content_rejected(knowledge_store, mem_store):
	"""Verify 0-byte and whitespace-only files do not become completed entries."""
	# 0-byte content
	with pytest.raises(SecurityValidationError) as exc_0byte:
		await knowledge_store.ingest_document(filename="empty.txt", content=b"")
	assert "empty" in str(exc_0byte.value).lower()

	# Whitespace-only content
	with pytest.raises(ValueError) as exc_ws:
		await knowledge_store.ingest_document(filename="spaces.txt", content=b"    \n\t  \r\n   ")
	assert "empty" in str(exc_ws.value).lower() or "whitespace" in str(exc_ws.value).lower()

	# Verify zero documents and zero chunks in storage
	docs = await knowledge_store.list_documents()
	assert len(docs) == 0
	assert len(mem_store._chunks) == 0


@pytest.mark.asyncio
async def test_corrupted_document_handling_and_no_path_leakage(knowledge_store, mem_store):
	"""Verify corrupted PDF/DOCX raise actionable errors without leaking paths or credentials."""
	corrupted_pdf = b"%PDF-1.4\ncorrupted junk bytes\x00\xffnot a valid structure"
	with pytest.raises(DocumentParsingError) as exc_pdf:
		await knowledge_store.ingest_document(filename="corrupted.pdf", content=corrupted_pdf)
	err_msg = str(exc_pdf.value)
	assert "Failed to parse PDF document" in err_msg
	assert "C:\\" not in err_msg
	assert "/home" not in err_msg
	assert "secret" not in err_msg

	corrupted_docx = b"PK\x03\x04corrupted docx content not a valid zip"
	with pytest.raises(DocumentParsingError) as exc_docx:
		await knowledge_store.ingest_document(filename="corrupted.docx", content=corrupted_docx)
	err_docx = str(exc_docx.value)
	assert "Failed to parse DOCX document" in err_docx
	assert "C:\\" not in err_docx
	assert "/home" not in err_docx

	# Ensure nothing was saved
	docs = await knowledge_store.list_documents()
	assert len(docs) == 0
	assert len(mem_store._chunks) == 0


@pytest.mark.asyncio
async def test_embedding_failure_leaves_zero_orphaned_records(tmp_path, mem_store):
	"""Verify that embedding failures clean up both chunks and document records for new uploads."""
	failing_embed = FailingEmbeddingModel(dimension=64)
	ks = KnowledgeStore(storage_dir=tmp_path, vector_store=mem_store, embedding_provider=failing_embed)

	valid_content = b"Valid document content that will fail during embedding batch call."
	with pytest.raises(RuntimeError) as exc_info:
		await ks.ingest_document(filename="valid.txt", content=valid_content)

	assert "Simulated embedding inference" in str(exc_info.value)
	# Verify zero orphaned documents or chunks left behind
	docs = await ks.list_documents()
	assert len(docs) == 0
	assert len(mem_store._chunks) == 0


@pytest.mark.asyncio
async def test_preexisting_document_failure_preservation(tmp_path, mem_store):
	"""Verify that re-ingesting a pre-existing document on failure marks it failed without deleting."""
	from insightbot.knowledge.chunker import compute_content_hash

	content = b"Content for document that previously failed and is retried."
	chash = compute_content_hash(content)

	# Pre-insert existing document in failed state
	pre_existing = Document(
		id="doc-preexisting-456",
		filename="retry.txt",
		source_type="txt",
		mime_type="text/plain",
		content_hash=chash,
		processing_status="failed",
		metadata={"previous_attempt": True},
	)
	await mem_store.upsert_document(pre_existing)

	# Ingest with failing embedding provider
	failing_embed = FailingEmbeddingModel(dimension=64)
	ks = KnowledgeStore(storage_dir=tmp_path, vector_store=mem_store, embedding_provider=failing_embed)

	with pytest.raises(RuntimeError):
		await ks.ingest_document(filename="retry.txt", content=content)

	# Document record should still exist with status failed and sanitized message
	doc_record = await mem_store.get_document("doc-preexisting-456")
	assert doc_record is not None
	assert doc_record.processing_status == "failed"
	assert doc_record.error_message is not None
	assert "sk-proj-abc1234567890def1234567890" not in doc_record.error_message  # sanitized
	assert "[REDACTED]" in doc_record.error_message
	assert len(mem_store._chunks) == 0


@pytest.mark.asyncio
async def test_duplicate_upload_deduplication(knowledge_store, mem_store):
	"""Verify identical document re-upload returns existing document and updates metadata."""
	content = b"InsightBot content hash deduplication guarantee."
	doc1 = await knowledge_store.ingest_document(
		filename="contract.txt",
		content=content,
		metadata={"stage": "initial"},
	)
	assert doc1.processing_status == "completed"

	# Re-upload same content with updated metadata
	doc2 = await knowledge_store.ingest_document(
		filename="contract_copy.txt",
		content=content,
		metadata={"stage": "approved", "reviewer": "Bob"},
	)
	assert doc2.id == doc1.id
	assert doc2.metadata.get("stage") == "approved"
	assert doc2.metadata.get("reviewer") == "Bob"

	# Verify only 1 document in storage
	docs = await knowledge_store.list_documents()
	assert len(docs) == 1


@pytest.mark.asyncio
async def test_chunk_limit_exceeded_rejected(knowledge_store, mem_store, monkeypatch):
	"""Verify document exceeding max_chunks_per_document is rejected without orphaned records."""
	monkeypatch.setattr(knowledge_store.settings, "max_chunks_per_document", 2)

	long_text = ("Section content paragraph with descriptive information.\n\n" * 150).encode("utf-8")
	with pytest.raises(ValueError) as exc_info:
		await knowledge_store.ingest_document(filename="long.txt", content=long_text)

	assert "exceeding the maximum limit of 2" in str(exc_info.value)
	docs = await knowledge_store.list_documents()
	assert len(docs) == 0
	assert len(mem_store._chunks) == 0


def test_api_upload_validation_and_error_codes(test_client):
	"""Test HTTP API layer for document uploads: 201 Created vs 400 Bad Request."""
	# 1. Valid upload -> 201
	res_ok = test_client.post(
		"/knowledge/documents",
		files={"file": ("guide.txt", b"Valid documentation content.", "text/plain")},
	)
	assert res_ok.status_code == 201
	data = res_ok.json()
	assert data["filename"] == "guide.txt"
	assert data["processing_status"] == "completed"

	# 2. Empty file upload -> 400
	res_empty = test_client.post(
		"/knowledge/documents",
		files={"file": ("empty.txt", b"", "text/plain")},
	)
	assert res_empty.status_code == 400
	assert "empty" in res_empty.json()["detail"].lower()

	# 3. Unsupported extension -> 400
	res_unsupported = test_client.post(
		"/knowledge/documents",
		files={"file": ("program.exe", b"MZ\x00\x00binary", "application/octet-stream")},
	)
	assert res_unsupported.status_code == 400
	assert "unsupported" in res_unsupported.json()["detail"].lower()

	# 4. Corrupted file -> 400
	res_corrupt = test_client.post(
		"/knowledge/documents",
		files={"file": ("corrupt.pdf", b"NOT A VALID PDF HEADER", "application/pdf")},
	)
	assert res_corrupt.status_code == 400
	assert "failed to parse pdf" in res_corrupt.json()["detail"].lower()

	# 5. JSON base64 empty string -> 400
	res_json_empty = test_client.post(
		"/knowledge/documents",
		json={"filename": "empty.txt", "content_base64": base64.b64encode(b"").decode()},
	)
	assert res_json_empty.status_code == 400
	assert "invalid json or base64" in res_json_empty.json()["detail"].lower()

	# 6. JSON base64 whitespace-only decoded bytes -> 400
	res_json_ws = test_client.post(
		"/knowledge/documents",
		json={"filename": "spaces.txt", "content_base64": base64.b64encode(b"   \n  ").decode()},
	)
	assert res_json_ws.status_code == 400
	assert "empty" in res_json_ws.json()["detail"].lower() or "whitespace" in res_json_ws.json()["detail"].lower()
