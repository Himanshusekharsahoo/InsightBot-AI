"""Retrieval Quality & Pipeline Unit Tests.

Covers:
- Exact factual retrieval
- Paraphrased questions
- Unrelated questions and negative score rejection
- Empty queries
- Duplicate document deduplication
- Missing evidence
- Metadata filtering (document_id, doc_type, custom tags)
- Hybrid dense vector + lexical keyword retrieval
"""

import pytest

from insightbot.db.in_memory import InMemoryVectorStore
from insightbot.knowledge.embeddings import MockEmbeddingModel
from insightbot.knowledge.store import KnowledgeStore


@pytest.mark.asyncio
async def test_exact_factual_retrieval(tmp_path):
	"""Test exact factual query retrieval matching stored technical concepts."""
	mem_store = InMemoryVectorStore()
	mock_embed = MockEmbeddingModel(dimension=64)
	store = KnowledgeStore(storage_dir=tmp_path, vector_store=mem_store, embedding_provider=mock_embed)

	doc = await store.ingest_document(
		filename='database_tech.txt',
		content=b'The storage subsystem persistence engine is built on PostgreSQL 17 with vector similarity search.',
	)

	results = await store.search_async('database persistence engine', top_k=1)
	assert len(results) == 1
	assert results[0].document_id == doc.id
	assert results[0].score >= 0.2
	assert 'PostgreSQL 17' in results[0].content


@pytest.mark.asyncio
async def test_paraphrased_query_retrieval(tmp_path):
	"""Test semantic retrieval of paraphrased questions using concept expansion."""
	mem_store = InMemoryVectorStore()
	mock_embed = MockEmbeddingModel(dimension=64)
	store = KnowledgeStore(storage_dir=tmp_path, vector_store=mem_store, embedding_provider=mock_embed)

	await store.ingest_document(
		filename='infra.txt',
		content=b'Deployment architecture and infrastructure guidelines for resilient docker containers.',
	)

	# Paraphrased query: "how to deploy systems"
	results = await store.search_async('how to deploy systems resiliently', top_k=1)
	assert len(results) >= 1
	assert results[0].score >= 0.2
	assert 'infrastructure' in results[0].content.lower()


@pytest.mark.asyncio
async def test_unrelated_query_has_low_score_and_filtered(tmp_path):
	"""Test unrelated query yields low or negligible similarity score without negative threshold hacks."""
	mem_store = InMemoryVectorStore()
	mock_embed = MockEmbeddingModel(dimension=64)
	store = KnowledgeStore(storage_dir=tmp_path, vector_store=mem_store, embedding_provider=mock_embed)

	await store.ingest_document(
		filename='database_guide.txt',
		content=b'PostgreSQL 17 persistence and vector search configuration.',
	)

	results = await store.search_async('photosynthesis in tropical plants', top_k=5)
	# Scores for unrelated text should be well below positive relevance threshold or near zero
	for r in results:
		assert r.score < 0.2


@pytest.mark.asyncio
async def test_empty_query_returns_empty_results(tmp_path):
	mem_store = InMemoryVectorStore()
	mock_embed = MockEmbeddingModel(dimension=64)
	store = KnowledgeStore(storage_dir=tmp_path, vector_store=mem_store, embedding_provider=mock_embed)
	await store.ingest_document(filename='doc.txt', content=b'Some content here.')

	assert await store.search_async('') == []
	assert await store.search_async('   \n\t  ') == []


@pytest.mark.asyncio
async def test_duplicate_document_content_hash_deduplication(tmp_path):
	"""Test duplicate document uploads are deduplicated by SHA256 content hash."""
	mem_store = InMemoryVectorStore()
	mock_embed = MockEmbeddingModel(dimension=64)
	store = KnowledgeStore(storage_dir=tmp_path, vector_store=mem_store, embedding_provider=mock_embed)

	content = b'Unique technical document content with specific sha256 hash.'
	doc1 = await store.ingest_document(filename='doc1.txt', content=content, metadata={'version': 1})
	assert doc1.processing_status == 'completed'

	# Ingest identical content with different filename
	doc2 = await store.ingest_document(filename='doc2.txt', content=content, metadata={'version': 2})
	# Same document ID reused
	assert doc2.id == doc1.id
	assert doc2.content_hash == doc1.content_hash
	# Total documents in store remains 1
	all_docs = await store.list_documents()
	assert len(all_docs) == 1


@pytest.mark.asyncio
async def test_metadata_filtering(tmp_path):
	"""Test metadata criteria filtering across stored chunks."""
	mem_store = InMemoryVectorStore()
	mock_embed = MockEmbeddingModel(dimension=64)
	store = KnowledgeStore(storage_dir=tmp_path, vector_store=mem_store, embedding_provider=mock_embed)

	doc_a = await store.ingest_document(
		filename='guide_dev.md',
		content=b'Database configuration for development environment.',
		metadata={'doc_type': 'dev_guide', 'author': 'team_a'},
	)
	doc_b = await store.ingest_document(
		filename='guide_prod.md',
		content=b'Database configuration for production environment.',
		metadata={'doc_type': 'prod_guide', 'author': 'team_b'},
	)

	# Filter by doc_type = dev_guide
	dev_results = await store.search_async(
		query='database configuration',
		top_k=5,
		filter_metadata={'doc_type': 'dev_guide'},
	)
	assert len(dev_results) == 1
	assert dev_results[0].document_id == doc_a.id

	# Filter by author = team_b
	prod_results = await store.search_async(
		query='database configuration',
		top_k=5,
		filter_metadata={'author': 'team_b'},
	)
	assert len(prod_results) == 1
	assert prod_results[0].document_id == doc_b.id


@pytest.mark.asyncio
async def test_hybrid_dense_and_lexical_retrieval(tmp_path):
	"""Test hybrid search combines vector similarity with exact lexical keyword matching."""
	mem_store = InMemoryVectorStore()
	mock_embed = MockEmbeddingModel(dimension=64)
	store = KnowledgeStore(storage_dir=tmp_path, vector_store=mem_store, embedding_provider=mock_embed)

	# Ingest chunk with exact identifier 'CVE-2026-9999'
	await store.ingest_document(
		filename='security_advisory.txt',
		content=b'Security bulletin addressing vulnerability identifier CVE-2026-9999 in core parser.',
	)
	await store.ingest_document(
		filename='general_security.txt',
		content=b'General security guidelines and authorization best practices for team members.',
	)

	# Pure vector search
	vector_results = await store.search_async(query='CVE-2026-9999', top_k=2, hybrid=False)
	# Hybrid search
	hybrid_results = await store.search_async(query='CVE-2026-9999', top_k=2, hybrid=True)

	assert len(hybrid_results) >= 1
	# Top result in hybrid search should be the advisory with the exact lexical token
	assert 'CVE-2026-9999' in hybrid_results[0].content
	# Hybrid score should include the lexical boost
	assert hybrid_results[0].score >= vector_results[0].score


@pytest.mark.asyncio
async def test_hybrid_retrieval_preserves_weak_vector_strong_lexical_matches(tmp_path):
	"""Verify hybrid candidate selection does not discard chunks with weak dense vectors but exact keyword matches."""
	from insightbot.knowledge.embeddings import BaseEmbeddingModel

	class OrthogonalEmbeddingModel(BaseEmbeddingModel):
		"""Embeddings where target chunk has 0.0 vector similarity to query."""

		@property
		def dimension(self) -> int:
			return 4

		@property
		def model_name(self) -> str:
			return 'orthogonal'

		async def embed_text(self, text: str) -> list[float]:
			# Query unit vector (0.25 dot product with unrelated docs, 0.0 with target doc)
			return [0.0, 0.9682458365518543, 0.25, 0.0]

		async def embed_batch(self, texts: list[str]) -> list[list[float]]:
			results = []
			for t in texts:
				if 'unrelated' in t.lower():
					# Moderate similarity (0.25) to query
					results.append([0.0, 0.0, 1.0, 0.0])
				else:
					# Target doc has 0.0 cosine similarity to query
					results.append([1.0, 0.0, 0.0, 0.0])
			return results

	mem_store = InMemoryVectorStore(embedding_dim=4)
	embed = OrthogonalEmbeddingModel()
	store = KnowledgeStore(storage_dir=tmp_path, vector_store=mem_store, embedding_provider=embed)

	# Ingest target doc with exact keyword but 0.0 vector similarity to query
	target_doc = await store.ingest_document(
		filename='target.txt',
		content=b'Important configuration file containing UNIQUE_SECRET_TOKEN_42 for deployment.',
	)
	# Ingest other documents with moderate vector similarity (0.25) but no lexical overlap
	for i in range(5):
		await store.ingest_document(
			filename=f'other_{i}.txt',
			content=f'Unrelated content number {i} discussing general concepts.'.encode('utf-8'),
		)

	# Pure vector search (top_k=2): target doc has 0.0 score, so top 2 are unrelated docs
	vector_results = await store.search_async(query='UNIQUE_SECRET_TOKEN_42', top_k=2, hybrid=False)
	assert len(vector_results) == 2
	assert target_doc.id not in [r.document_id for r in vector_results]

	# Hybrid search: keyword search retrieves target_doc despite 0.0 vector score, and lexical boost makes it #1
	hybrid_results = await store.search_async(query='UNIQUE_SECRET_TOKEN_42', top_k=1, hybrid=True)
	assert len(hybrid_results) == 1
	assert hybrid_results[0].document_id == target_doc.id
	assert 'UNIQUE_SECRET_TOKEN_42' in hybrid_results[0].content
	assert hybrid_results[0].score > 0.2
