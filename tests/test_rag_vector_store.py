"""Unit tests for vector storage abstraction, document persistence, and semantic search."""

import pytest

from insightbot.db.in_memory import InMemoryVectorStore
from insightbot.db.models import Document, DocumentChunk
from insightbot.db.vector_store import _cosine_similarity


def test_cosine_similarity():
	assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
	assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
	assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)
	assert _cosine_similarity([], []) == 0.0


@pytest.mark.asyncio
async def test_in_memory_vector_store_crud():
	store = InMemoryVectorStore()
	await store.initialize()

	# Create document
	doc = Document(
		id='doc-001',
		filename='architecture.pdf',
		source_type='pdf',
		mime_type='application/pdf',
		content_hash='abc123hash',
		processing_status='completed',
		metadata={'author': 'Team'},
	)
	doc_id = await store.upsert_document(doc)
	assert doc_id == 'doc-001'

	# Get document
	fetched = await store.get_document('doc-001')
	assert fetched is not None
	assert fetched.filename == 'architecture.pdf'
	assert fetched.metadata['author'] == 'Team'

	# Get by content hash
	by_hash = await store.get_document_by_content_hash('abc123hash')
	assert by_hash is not None
	assert by_hash.id == 'doc-001'

	# Insert chunks
	chunks = [
		DocumentChunk(
			id='chk-001',
			document_id='doc-001',
			chunk_index=0,
			content='Chunk 1 about PostgreSQL',
			embedding=[1.0, 0.0, 0.0],
			metadata={'page_number': 1},
		),
		DocumentChunk(
			id='chk-002',
			document_id='doc-001',
			chunk_index=1,
			content='Chunk 2 about pgvector',
			embedding=[0.0, 1.0, 0.0],
			metadata={'page_number': 2},
		),
	]
	inserted_ids = await store.insert_chunks(chunks)
	assert len(inserted_ids) == 2

	# Get chunks
	doc_chunks = await store.get_chunks('doc-001')
	assert len(doc_chunks) == 2
	assert doc_chunks[0].id == 'chk-001'
	assert doc_chunks[1].id == 'chk-002'

	# Similarity search
	query_vec = [1.0, 0.0, 0.0]
	results = await store.similarity_search(query_vec, top_k=2)
	assert len(results) == 2
	assert results[0].chunk_id == 'chk-001'
	assert pytest.approx(results[0].score, 1e-4) == 1.0
	assert results[1].chunk_id == 'chk-002'
	assert pytest.approx(results[1].score, 1e-4) == 0.0

	# Filtered search
	filtered = await store.similarity_search(query_vec, top_k=2, filter_metadata={'document_id': 'nonexistent'})
	assert len(filtered) == 0

	# List documents
	doc_list = await store.list_documents()
	assert len(doc_list) == 1

	# Cascading delete
	deleted = await store.delete_document('doc-001')
	assert deleted is True

	# Verify chunks also removed
	assert await store.get_document('doc-001') is None
	assert len(await store.get_chunks('doc-001')) == 0
	assert len(await store.list_documents()) == 0

	await store.close()
