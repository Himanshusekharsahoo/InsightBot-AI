"""End-to-End Pipeline Verification for InsightBot AI RAG.

Tests document ingestion, chunking, embedding, PostgreSQL storage,
semantic similarity search, grounded KnowledgeAgent answering, deduplication,
and cascading deletion using real PostgreSQL 17 persistence.
"""

import os
from pathlib import Path

import asyncpg
import pytest

from insightbot.db.vector_store import PgVectorStore
from insightbot.knowledge.embeddings import MockEmbeddingModel
from insightbot.knowledge.store import KnowledgeStore
from insightbot.orchestrator.knowledge_agent import KnowledgeAgent

PG_URL = os.getenv('INSIGHTBOT_DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/insightbot')
FIXTURES_DIR = Path(__file__).parent / 'fixtures' / 'knowledge'


async def _is_postgres_online() -> bool:
	try:
		conn = await asyncpg.connect(PG_URL, timeout=2.0)
		await conn.close()
		return True
	except Exception:
		return False


@pytest.mark.asyncio
@pytest.mark.integration
async def test_end_to_end_rag_pipeline_with_postgres(tmp_path):
	"""Complete end-to-end RAG verification against real PostgreSQL persistence."""
	if not await _is_postgres_online():
		pytest.skip(f'PostgreSQL service is not reachable at {PG_URL}')

	# 1. Setup real PostgreSQL vector store with mock embedder for deterministic testing
	embed_dim = 64
	pg_store = PgVectorStore(database_url=PG_URL, embedding_dim=embed_dim)
	embedder = MockEmbeddingModel(dimension=embed_dim)

	store = KnowledgeStore(
		storage_dir=tmp_path / 'legacy',
		vector_store=pg_store,
		embedding_provider=embedder,
	)
	await store.initialize()

	# Verify storage mode is honest
	assert pg_store.storage_mode in ('native_pgvector', 'pg_array')
	is_native = pg_store.storage_mode == 'native_pgvector'

	# 2. Ingest deterministic isolated test document
	fixture_file = FIXTURES_DIR / 'insightbot_overview.txt'
	assert fixture_file.exists(), f'Fixture file missing: {fixture_file}'
	import uuid

	run_id = uuid.uuid4().hex[:8]
	base_content = fixture_file.read_text(encoding='utf-8')
	content = f'{base_content}\n\nUnique test session tag: {run_id}'.encode('utf-8')

	doc = None
	try:
		doc = await store.ingest_document(
			filename='insightbot_overview.txt',
			content=content,
			metadata={'author': 'Engineering Team', 'doc_type': 'specification'},
		)

		assert doc.id.startswith('doc-')
		assert doc.filename == 'insightbot_overview.txt'
		assert doc.processing_status == 'completed'
		assert doc.metadata['doc_type'] == 'specification'
		assert doc.metadata['author'] == 'Engineering Team'

		# 3. Verify chunks and chunk metadata persisted in PostgreSQL
		chunks = await pg_store.get_chunks(doc.id)
		assert len(chunks) >= 1
		for chk in chunks:
			assert chk.document_id == doc.id
			assert chk.embedding is not None
			assert len(chk.embedding) == embed_dim
			assert chk.metadata['doc_type'] == 'specification'
			assert chk.metadata['source_filename'] == 'insightbot_overview.txt'
			assert chk.metadata['page_number'] >= 1
			assert 'InsightBot AI' in chk.content or 'PostgreSQL' in chk.content

		# 4. Retrieve the expected document by filtering on its actual document ID
		search_query = 'PostgreSQL 17 serves as the primary persistence layer'
		results = await store.search_async(
			query=search_query,
			top_k=3,
			filter_metadata={'document_id': doc.id},
		)
		assert len(results) >= 1
		assert results[0].document_id == doc.id
		# 5. Verify that expected fact appears in retrieved content
		assert any('PostgreSQL' in r.content for r in results)

		# 6. Verify KnowledgeAgent grounded answering with verifiable citations
		async def mock_grounded_llm(question: str, evidence_context: str) -> str:
			assert 'PostgreSQL 17' in evidence_context
			return 'The primary database specified is PostgreSQL 17.'

		agent = KnowledgeAgent(knowledge_store=store, llm=mock_grounded_llm)
		answer = await agent.answer(
			'What is the primary persistent database?',
			top_k=3,
			filter_metadata={'document_id': doc.id},
		)

		assert answer.is_grounded is True
		assert answer.sufficient_context is True
		assert answer.insufficient_evidence is False
		assert 'PostgreSQL 17' in answer.answer
		assert len(answer.sources) >= 1
		assert answer.sources[0].source_filename == 'insightbot_overview.txt'
		assert answer.sources[0].document_id == doc.id
		assert answer.sources[0].chunk_id == chunks[0].id
		assert answer.sources[0].score >= 0.2

		# Verify insufficient evidence for an unrelated question with standard threshold
		unrelated_answer = await agent.answer('What is the planetary mass of Jupiter?')
		assert unrelated_answer.insufficient_evidence is True
		assert unrelated_answer.sufficient_context is False
		assert len(unrelated_answer.sources) == 0

		# 7. Deduplication: Ingest identical content
		dup_doc = await store.ingest_document(
			filename='insightbot_overview_copy.txt',
			content=content,
		)
		assert dup_doc.id == doc.id
		# Chunks count in db should remain unchanged (no duplicate chunks)
		all_chunks = await pg_store.get_chunks(doc.id)
		assert len(all_chunks) == len(chunks)

		# 8. List documents from PostgreSQL
		doc_list = await store.list_documents()
		matching_docs = [d for d in doc_list if d.id == doc.id]
		assert len(matching_docs) == 1

		# 9. Cascading deletion
		deleted = await store.delete_document(doc.id)
		assert deleted is True

		# Verify document and chunks are gone from PostgreSQL
		assert await pg_store.get_document(doc.id) is None
		assert len(await pg_store.get_chunks(doc.id)) == 0
		doc = None  # Successfully deleted

	finally:
		if doc is not None:
			try:
				await store.delete_document(doc.id)
			except Exception:
				pass
		await store.close()
