"""Unit tests for KnowledgeStore ingestion, duplicate prevention, and semantic search."""

import pytest

from insightbot.db.in_memory import InMemoryVectorStore
from insightbot.knowledge.embeddings import MockEmbeddingModel
from insightbot.knowledge.store import KnowledgeStore
from insightbot.knowledge.validation import SecurityValidationError


@pytest.mark.asyncio
async def test_knowledge_store_ingest_and_search(tmp_path):
	mem_store = InMemoryVectorStore()
	mock_embed = MockEmbeddingModel(dimension=64)

	store = KnowledgeStore(
		storage_dir=tmp_path,
		vector_store=mem_store,
		embedding_provider=mock_embed,
	)

	# Ingest sample text document
	content = (
		'InsightBot AI deployment uses PostgreSQL 17 as the primary persistent database. '
		'Vector search is accelerated using the pgvector extension.\n\n'
		'Document ingestion supports PDF, DOCX, TXT, and Markdown.'
	).encode('utf-8')

	doc = await store.ingest_document(
		filename='deployment.txt',
		content=content,
		metadata={'category': 'infrastructure'},
	)

	assert doc.filename == 'deployment.txt'
	assert doc.source_type == 'txt'
	assert doc.processing_status == 'completed'

	# Verify stored chunks exist
	chunks = await mem_store.get_chunks(doc.id)
	assert len(chunks) >= 1
	for chk in chunks:
		assert chk.embedding is not None
		assert len(chk.embedding) == 64

	# Search
	results = await store.search('PostgreSQL database', top_k=3)
	assert len(results) >= 1
	assert results[0].document_id == doc.id
	assert 'PostgreSQL' in results[0].content

	# Duplicate ingestion detection (same content)
	dup_doc = await store.ingest_document(
		filename='deployment_copy.txt',
		content=content,
	)
	assert dup_doc.id == doc.id  # Returns existing document without duplicating

	# List documents
	docs = await store.list_documents()
	assert len(docs) == 1

	# Delete document
	deleted = await store.delete_document(doc.id)
	assert deleted is True
	assert len(await store.list_documents()) == 0
	assert len(await mem_store.get_chunks(doc.id)) == 0


@pytest.mark.asyncio
async def test_knowledge_store_empty_document_error(tmp_path):
	"""Verify 0-byte and whitespace-only documents are rejected and not persisted."""
	mem_store = InMemoryVectorStore()
	mock_embed = MockEmbeddingModel(dimension=64)

	store = KnowledgeStore(
		storage_dir=tmp_path,
		vector_store=mem_store,
		embedding_provider=mock_embed,
	)

	# 0-byte document is rejected by security validation
	with pytest.raises(SecurityValidationError) as exc_info:
		await store.ingest_document(filename='empty.txt', content=b'')
	assert 'empty' in str(exc_info.value).lower()

	# Whitespace-only document is rejected by parser/chunker validation
	with pytest.raises(ValueError) as exc_ws:
		await store.ingest_document(filename='whitespace.txt', content=b'   \n\t  \r\n ')
	assert 'empty' in str(exc_ws.value).lower() or 'whitespace' in str(exc_ws.value).lower()

	# Verify zero documents and zero chunks in storage
	assert len(await store.list_documents()) == 0
	assert len(mem_store._chunks) == 0


@pytest.mark.asyncio
async def test_knowledge_store_search_interfaces_and_closed_handling(tmp_path):
	"""Verify search_async, empty query handling, and closed store safety."""
	mem_store = InMemoryVectorStore()
	mock_embed = MockEmbeddingModel(dimension=64)

	store = KnowledgeStore(
		storage_dir=tmp_path,
		vector_store=mem_store,
		embedding_provider=mock_embed,
	)

	content = b'Persistent database configuration with PostgreSQL 17.'
	doc = await store.ingest_document(
		filename='config.txt',
		content=content,
		metadata={'author': 'Platform Team', 'doc_type': 'specification'},
	)

	# Verify metadata preservation on document and chunk
	assert doc.metadata['doc_type'] == 'specification'
	assert doc.metadata['author'] == 'Platform Team'
	chunks = await mem_store.get_chunks(doc.id)
	assert len(chunks) == 1
	assert chunks[0].metadata['doc_type'] == 'specification'
	assert chunks[0].metadata['source_filename'] == 'config.txt'

	# 1. search_async direct interface
	async_results = await store.search_async('PostgreSQL database', top_k=2)
	assert len(async_results) >= 1
	assert async_results[0].document_id == doc.id

	# 2. Empty query handling
	empty_res = await store.search_async('   ')
	assert empty_res == []
	assert store.search('   ') == []

	# 3. Synchronous compatibility method
	store.store_document(
		doc_id='legacy-doc-1',
		title='Legacy Architecture Guide',
		source_url='https://internal/docs/arch',
		content='Legacy system overview with architecture details.',
	)
	legacy_matches = store.search_legacy('Architecture')
	assert len(legacy_matches) == 1
	assert legacy_matches[0].doc_id == 'legacy-doc-1'

	# 4. Closed store safety
	assert store.is_closed is False
	await store.close()
	assert store.is_closed is True
	# Calling search safely re-initializes connection pool
	reopened = await store.search_async('PostgreSQL database', top_k=1)
	assert len(reopened) >= 1
	assert store.is_closed is False
	await store.close()
