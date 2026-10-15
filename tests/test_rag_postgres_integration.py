"""Integration tests for real PostgreSQL storage and vector search."""

import os

import asyncpg
import pytest

from insightbot.db.models import Document, DocumentChunk
from insightbot.db.vector_store import PgVectorStore
from insightbot.knowledge.embeddings import MockEmbeddingModel

PG_URL = os.getenv('INSIGHTBOT_DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/insightbot')


async def _is_postgres_online() -> bool:
	try:
		conn = await asyncpg.connect(PG_URL, timeout=2.0)
		await conn.close()
		return True
	except Exception:
		return False


@pytest.mark.asyncio
@pytest.mark.integration
async def test_postgres_real_integration():
	if not await _is_postgres_online():
		pytest.skip(f'PostgreSQL service is not reachable at {PG_URL}')

	store = PgVectorStore(database_url=PG_URL, embedding_dim=64)
	await store.initialize()

	# Verify storage mode is reported accurately (pg_array when extension is not installed)
	assert store.storage_mode == 'pg_array'

	# Create real document
	doc = Document(
		id='doc-integration-001',
		filename='integration_test.txt',
		source_type='txt',
		mime_type='text/plain',
		content_hash='hash_integration_test_123',
		processing_status='completed',
		metadata={'environment': 'integration'},
	)
	await store.upsert_document(doc)

	# Fetch document
	fetched = await store.get_document('doc-integration-001')
	assert fetched is not None
	assert fetched.filename == 'integration_test.txt'

	# Insert chunks with mock embedding
	embedder = MockEmbeddingModel(dimension=64)
	chunk_text = 'PostgreSQL pgvector integration test chunk.'
	emb = await embedder.embed_text(chunk_text)

	chunk = DocumentChunk(
		id='chk-integration-001',
		document_id='doc-integration-001',
		chunk_index=0,
		content=chunk_text,
		metadata={'source_filename': 'integration_test.txt'},
		embedding=emb,
	)
	await store.insert_chunks([chunk])

	# Search
	results = await store.similarity_search(query_embedding=emb, top_k=1)
	assert len(results) >= 1
	assert results[0].document_id == 'doc-integration-001'
	assert pytest.approx(results[0].score, 1e-4) == 1.0

	# Cascading delete
	await store.delete_document('doc-integration-001')
	assert await store.get_document('doc-integration-001') is None
	assert len(await store.get_chunks('doc-integration-001')) == 0

	await store.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_postgres_dimension_mismatch_rejection():
	if not await _is_postgres_online():
		pytest.skip(f'PostgreSQL service is not reachable at {PG_URL}')

	store = PgVectorStore(database_url=PG_URL, embedding_dim=64)
	await store.initialize()

	doc = Document(
		id='doc-dim-test',
		filename='dim_test.txt',
		source_type='txt',
		mime_type='text/plain',
	)
	await store.upsert_document(doc)

	# Chunk with mismatched dimension (128 instead of 64)
	wrong_chunk = DocumentChunk(
		id='chk-wrong-dim',
		document_id='doc-dim-test',
		chunk_index=0,
		content='Wrong dimension content',
		embedding=[0.1] * 128,
	)

	with pytest.raises(ValueError, match='Embedding dimension mismatch'):
		await store.insert_chunks([wrong_chunk])

	# Query vector with mismatched dimension
	with pytest.raises(ValueError, match='Query embedding dimension mismatch'):
		await store.similarity_search(query_embedding=[0.1] * 128, top_k=1)

	await store.delete_document('doc-dim-test')
	await store.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_postgres_require_native_pgvector_diagnostic():
	if not await _is_postgres_online():
		pytest.skip(f'PostgreSQL service is not reachable at {PG_URL}')

	from insightbot.db.base import PgVectorUnavailableError

	# When user explicitly requires native pgvector mode
	store = PgVectorStore(database_url=PG_URL, embedding_dim=64, vector_mode='native')

	with pytest.raises(PgVectorUnavailableError) as exc_info:
		await store.initialize()

	assert 'pgvector' in str(exc_info.value).lower()
	assert 'vector.control' in str(exc_info.value) or 'strictly required' in str(exc_info.value)

	await store.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_postgres_metadata_filter_and_keyword_search():
	"""Verify metadata filtering and keyword search in PostgreSQL."""
	if not await _is_postgres_online():
		pytest.skip(f'PostgreSQL service is not reachable at {PG_URL}')

	store = PgVectorStore(database_url=PG_URL, embedding_dim=64)
	await store.initialize()

	doc = Document(
		id='doc-meta-test-01',
		filename='meta_doc.txt',
		source_type='txt',
		mime_type='text/plain',
		content_hash='hash_meta_test_abc',
		processing_status='completed',
		metadata={'doc_type': 'special_manual', 'tier': 'enterprise'},
	)
	await store.upsert_document(doc)

	chunk = DocumentChunk(
		id='chk-meta-test-01',
		document_id='doc-meta-test-01',
		chunk_index=0,
		content='PostgreSQL cluster running with mTLS authentication enabled for security.',
		metadata={'source_filename': 'meta_doc.txt', 'tier': 'enterprise'},
		embedding=[0.05] * 64,
	)
	await store.insert_chunks([chunk])

	# 1. Similarity search with document_id filter
	filtered_res = await store.similarity_search(
		query_embedding=[0.05] * 64,
		top_k=2,
		filter_metadata={'document_id': 'doc-meta-test-01'},
	)
	assert len(filtered_res) == 1
	assert filtered_res[0].chunk_id == 'chk-meta-test-01'

	# 2. Similarity search with non-matching filter
	non_match_res = await store.similarity_search(
		query_embedding=[0.05] * 64,
		top_k=2,
		filter_metadata={'document_id': 'nonexistent-doc'},
	)
	assert len(non_match_res) == 0

	# 3. Keyword search
	kw_res = await store.keyword_search(query='mTLS authentication', top_k=2)
	assert len(kw_res) >= 1
	assert any(r.chunk_id == 'chk-meta-test-01' for r in kw_res)

	await store.delete_document('doc-meta-test-01')
	await store.close()
