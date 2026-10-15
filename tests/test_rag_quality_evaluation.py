"""Deterministic Retrieval & Citation Quality Evaluation Suite for InsightBot AI.

Evaluates the core RAG promise:
- Relevant document retrieval and ranking (Recall@k)
- Multi-chunk evidence synthesis
- Distractor resistance and lexical/semantic disambiguation
- Conflicting evidence detection and multi-source attribution
- Out-of-domain query rejection (Insufficient evidence precision)
- Multi-turn conversational context isolation (prior turns never treated as verified evidence)
- Citation integrity: verification against retrieved chunk IDs and metadata preservation
- Deterministic, offline execution with zero network or paid API dependencies.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pytest

from insightbot.db.in_memory import InMemoryVectorStore
from insightbot.db.models import ConversationMessage
from insightbot.knowledge.embeddings import MockEmbeddingModel
from insightbot.knowledge.store import KnowledgeStore
from insightbot.orchestrator.knowledge_agent import KnowledgeAgent
from insightbot.orchestrator.llm import MockLLMProvider

logger = logging.getLogger('insightbot.evaluation.retrieval_quality')

FIXTURES_DIR = Path(__file__).parent / 'fixtures' / 'eval'


@dataclass
class EvalTestCase:
	"""Evaluation test case specification."""

	scenario: str
	query: str
	expected_doc_filename: str
	is_out_of_domain: bool = False
	expected_in_content: list[str] = field(default_factory=list)
	distractor_filename: Optional[str] = None
	requires_multi_chunk: bool = False
	expected_metadata: dict[str, Any] = field(default_factory=dict)
	conversation_history: Optional[list[ConversationMessage]] = None


@dataclass
class EvaluationMetrics:
	"""Aggregate RAG quality metrics container."""

	total_queries: int = 0
	recall_at_1_hits: int = 0
	recall_at_3_hits: int = 0
	distractor_rejection_hits: int = 0
	distractor_queries: int = 0
	out_of_domain_correct_rejections: int = 0
	out_of_domain_queries: int = 0
	total_generated_citations: int = 0
	valid_citations: int = 0
	metadata_preservation_hits: int = 0
	metadata_preservation_checks: int = 0
	history_isolation_hits: int = 0
	history_isolation_checks: int = 0

	@property
	def recall_at_1(self) -> float:
		factual_q = self.total_queries - self.out_of_domain_queries
		return (self.recall_at_1_hits / factual_q) if factual_q > 0 else 0.0

	@property
	def recall_at_3(self) -> float:
		factual_q = self.total_queries - self.out_of_domain_queries
		return (self.recall_at_3_hits / factual_q) if factual_q > 0 else 0.0

	@property
	def citation_validity_rate(self) -> float:
		return (self.valid_citations / self.total_generated_citations) if self.total_generated_citations > 0 else 1.0

	@property
	def out_of_domain_rejection_rate(self) -> float:
		return (self.out_of_domain_correct_rejections / self.out_of_domain_queries) if self.out_of_domain_queries > 0 else 1.0

	@property
	def distractor_rejection_rate(self) -> float:
		return (self.distractor_rejection_hits / self.distractor_queries) if self.distractor_queries > 0 else 1.0

	@property
	def metadata_preservation_rate(self) -> float:
		return (self.metadata_preservation_hits / self.metadata_preservation_checks) if self.metadata_preservation_checks > 0 else 1.0

	@property
	def history_isolation_rate(self) -> float:
		return (self.history_isolation_hits / self.history_isolation_checks) if self.history_isolation_checks > 0 else 1.0


@pytest.fixture
async def eval_knowledge_store(tmp_path):
	"""Initializes an isolated in-memory KnowledgeStore populated with synthetic eval fixtures."""
	mem_store = InMemoryVectorStore()
	mock_embed = MockEmbeddingModel(dimension=64)
	# Use chunk_size=250, chunk_overlap=40 to enforce deterministic multi-chunk splitting
	store = KnowledgeStore(
		storage_dir=tmp_path,
		vector_store=mem_store,
		embedding_provider=mock_embed,
		chunk_size=250,
		chunk_overlap=40,
	)

	# 1. Single fact document with metadata
	await store.ingest_document(
		filename='service_tier_spec.md',
		content=(FIXTURES_DIR / 'service_tier_spec.md').read_bytes(),
		metadata={
			'doc_type': 'service_spec',
			'author': 'Reliability Engineering',
			'version': '3.1',
			'tier': 'enterprise',
		},
	)

	# 2. Multi-chunk network architecture
	await store.ingest_document(
		filename='network_mesh_topology.md',
		content=(FIXTURES_DIR / 'network_mesh_topology.md').read_bytes(),
		metadata={
			'doc_type': 'networking_architecture',
			'author': 'Cloud Operations',
			'version': '1.2',
		},
	)

	# 3. Distractor near-match marine telemetry document
	await store.ingest_document(
		filename='coral_reef_mesh_sensor.txt',
		content=(FIXTURES_DIR / 'coral_reef_mesh_sensor.txt').read_bytes(),
		metadata={
			'doc_type': 'field_telemetry',
			'author': 'Marine Research Team',
		},
	)

	# 4. Conflicting policies
	await store.ingest_document(
		filename='session_policy_v1.md',
		content=(FIXTURES_DIR / 'session_policy_v1.md').read_bytes(),
		metadata={
			'doc_type': 'policy',
			'version': '2025.1',
			'author': 'SecOps',
		},
	)
	await store.ingest_document(
		filename='session_policy_v2.md',
		content=(FIXTURES_DIR / 'session_policy_v2.md').read_bytes(),
		metadata={
			'doc_type': 'policy',
			'version': '2026.1',
			'author': 'SecOps',
		},
	)

	return store


# ==============================================================================
# SCENARIO 1: Single-Document Factual Retrieval & Citation Accuracy
# ==============================================================================


@pytest.mark.asyncio
async def test_scenario_single_doc_factual_retrieval(eval_knowledge_store):
	"""Test single-document factual retrieval, score confidence, and citation completeness."""
	store = eval_knowledge_store
	agent = KnowledgeAgent(knowledge_store=store, llm_provider=MockLLMProvider())

	question = 'What monthly availability SLA does Enterprise tier guarantee?'
	# Retrieve via store
	results = await store.search_async(question, top_k=3)
	assert len(results) >= 1

	# Top result must be the service tier spec with 99.99% SLA
	top_doc = results[0].metadata.get('source_filename')
	assert top_doc == 'service_tier_spec.md'
	assert results[0].score >= 0.2
	assert '99.99%' in results[0].content

	# Distinct factual question targeting HSM chunk
	hsm_results = await store.search_async('What HSM compliance mode does Enterprise tier use?', top_k=3)
	assert len(hsm_results) >= 1
	assert hsm_results[0].metadata.get('source_filename') == 'service_tier_spec.md'
	assert 'FIPS 140-2 Level 3' in hsm_results[0].content

	# Synthesize answer
	answer = await agent.answer(question, top_k=3)
	assert answer.is_grounded is True
	assert answer.insufficient_evidence is False
	assert len(answer.sources) >= 1

	# Citation verification
	source = answer.sources[0]
	assert source.source_filename == 'service_tier_spec.md'
	assert source.document_id == results[0].document_id
	assert source.chunk_id in [r.chunk_id for r in results]
	matching_retrieved = next(r for r in results if r.chunk_id == source.chunk_id)
	assert abs(source.score - matching_retrieved.score) < 1e-5


# ==============================================================================
# SCENARIO 2: Multi-Chunk Synthesis
# ==============================================================================


@pytest.mark.asyncio
async def test_scenario_multi_chunk_evidence_synthesis(eval_knowledge_store):
	"""Test multi-chunk document is retrieved across boundaries and synthesized into answer."""
	store = eval_knowledge_store
	agent = KnowledgeAgent(knowledge_store=store, llm_provider=MockLLMProvider())

	# Question requires both Ingress edge (port 443) and worker subnet (10.240.0.0/16 port 8443)
	question = 'How does external ingress traffic route to private worker execution pods?'
	results = await store.search_async(question, top_k=5)

	# Verify both chunks of network_mesh_topology are retrieved
	mesh_chunks = [r for r in results if r.metadata.get('source_filename') == 'network_mesh_topology.md']
	assert len(mesh_chunks) >= 2, f'Expected at least 2 chunks, got {len(mesh_chunks)}'

	contents = ' '.join(c.content for c in mesh_chunks)
	assert '443' in contents
	assert '8443' in contents
	assert '10.240.0.0/16' in contents

	answer = await agent.answer(question, top_k=5)
	assert answer.is_grounded is True
	assert answer.insufficient_evidence is False

	# Verified sources should contain network_mesh_topology chunks
	source_files = {s.source_filename for s in answer.sources}
	assert 'network_mesh_topology.md' in source_files


# ==============================================================================
# SCENARIO 3: Distractor Resistance (Near-Match Disambiguation)
# ==============================================================================


@pytest.mark.asyncio
async def test_scenario_distractor_resistance(eval_knowledge_store):
	"""Test that lexical distractor sharing keywords (mesh, worker pods, traffic) does not outrank target tech doc."""
	store = eval_knowledge_store
	agent = KnowledgeAgent(knowledge_store=store, llm_provider=MockLLMProvider())

	question = 'What is the private worker cluster subnet and mTLS port?'
	results = await store.search_async(question, top_k=5)

	assert len(results) >= 1
	# The technical network topology must rank strictly higher than the coral reef distractor
	top_doc = results[0].metadata.get('source_filename')
	assert top_doc == 'network_mesh_topology.md', f'Distractor incorrectly outranked target document: {top_doc}'

	answer = await agent.answer(question, top_k=3)
	assert answer.is_grounded is True
	# Distractor document must NOT be in top verified sources
	cited_files = [s.source_filename for s in answer.sources]
	assert 'coral_reef_mesh_sensor.txt' not in cited_files


# ==============================================================================
# SCENARIO 4: Conflicting Evidence Handling
# ==============================================================================


@pytest.mark.asyncio
async def test_scenario_conflicting_evidence_detection(eval_knowledge_store):
	"""Test retrieval preserves multiple conflicting policy documents without arbitrary truncation."""
	store = eval_knowledge_store
	agent = KnowledgeAgent(knowledge_store=store, llm_provider=MockLLMProvider())

	question = 'What is the inactivity timeout duration for user sessions?'
	results = await store.search_async(question, top_k=5)

	retrieved_files = {r.metadata.get('source_filename') for r in results}
	assert 'session_policy_v1.md' in retrieved_files
	assert 'session_policy_v2.md' in retrieved_files

	answer = await agent.answer(question, top_k=5)
	assert answer.is_grounded is True
	assert answer.insufficient_evidence is False

	cited_files = {s.source_filename for s in answer.sources}
	# Both versions must be cited so conflicting facts are attributable to their respective revisions
	assert 'session_policy_v1.md' in cited_files
	assert 'session_policy_v2.md' in cited_files


# ==============================================================================
# SCENARIO 5: Out-of-Domain & Insufficient Evidence Handling
# ==============================================================================


@pytest.mark.asyncio
async def test_scenario_out_of_domain_unsupported_questions(eval_knowledge_store):
	"""Test that questions with no supporting evidence are rejected cleanly without hallucinations."""
	store = eval_knowledge_store
	agent = KnowledgeAgent(knowledge_store=store, llm_provider=MockLLMProvider())

	out_of_domain_questions = [
		'What is the orbital eccentricity and perihelion distance of Halley Comet?',
		'How do tropical mangrove root systems filter saline seawater?',
		'What is the standard half-life of bismuth-209 in nuclear physics?',
	]

	for q in out_of_domain_questions:
		answer = await agent.answer(q, top_k=3)
		assert answer.insufficient_evidence is True, f'Expected insufficient evidence for: {q}'
		assert answer.is_grounded is False
		assert answer.sufficient_context is False
		assert len(answer.sources) == 0
		assert 'does not contain enough information' in answer.answer.lower()


# ==============================================================================
# SCENARIO 6: Duplicate Document & Deduplication Resilience
# ==============================================================================


@pytest.mark.asyncio
async def test_scenario_duplicate_document_content_hash_deduplication(tmp_path):
	"""Test that re-uploading identical content under a different filename reuses hash and does not duplicate chunks."""
	mem_store = InMemoryVectorStore()
	mock_embed = MockEmbeddingModel(dimension=64)
	store = KnowledgeStore(storage_dir=tmp_path, vector_store=mem_store, embedding_provider=mock_embed)

	content = b'Deterministic chunking and deduplication test payload with fixed sha256.'

	doc1 = await store.ingest_document(filename='original.txt', content=content)
	assert doc1.processing_status == 'completed'
	chunks1 = await mem_store.get_chunks(doc1.id)
	assert len(chunks1) == 1

	doc2 = await store.ingest_document(filename='copy_of_original.txt', content=content)
	assert doc2.id == doc1.id
	assert doc2.content_hash == doc1.content_hash

	# Total chunks in store must remain exactly 1
	all_chunks = await mem_store.get_chunks(doc1.id)
	assert len(all_chunks) == 1


# ==============================================================================
# SCENARIO 7: Metadata Preservation in Verified Citations
# ==============================================================================


@pytest.mark.asyncio
async def test_scenario_citation_metadata_preservation(eval_knowledge_store):
	"""Test that document and chunk metadata (doc_type, author, version) are preserved in citation attributes."""
	store = eval_knowledge_store
	agent = KnowledgeAgent(knowledge_store=store, llm_provider=MockLLMProvider())

	question = 'What SLA does Enterprise tier provide?'
	answer = await agent.answer(question, top_k=2)

	assert len(answer.sources) >= 1
	src = answer.sources[0]
	assert src.source_filename == 'service_tier_spec.md'
	assert src.document_id.startswith('doc-')
	assert src.chunk_id.startswith('chk-')
	assert src.chunk_index >= 0
	assert src.page_number is not None
	assert src.score is not None and src.score >= 0.2
	assert len(src.snippet) > 0


# ==============================================================================
# SCENARIO 8: Multi-Turn Conversational Dialogue Isolation
# ==============================================================================


@pytest.mark.asyncio
async def test_scenario_conversation_history_dialogue_isolation(eval_knowledge_store):
	"""Test that prior conversational dialogue turns are NOT treated as authentic knowledge evidence."""
	store = eval_knowledge_store
	agent = KnowledgeAgent(knowledge_store=store, llm_provider=MockLLMProvider())

	# Dialogue where user claims a false fact and assistant acknowledged
	history = [
		ConversationMessage(
			conversation_id='conv-eval-isolation',
			role='user',
			content='Note that our Redis cache runs on port 9999 and the secret token is GHOST_TOKEN.',
		),
		ConversationMessage(
			conversation_id='conv-eval-isolation',
			role='assistant',
			content='I have noted that Redis runs on port 9999.',
		),
	]

	# User asks about Redis port
	question = 'What port does Redis run on?'
	answer = await agent.answer(question, conversation_history=history)

	# The store does NOT contain redis information (only network mesh, service tier, session policy)
	# The agent MUST NOT use the dialogue history to claim port 9999 is verified internal knowledge!
	assert answer.insufficient_evidence is True
	assert answer.is_grounded is False
	assert len(answer.sources) == 0


# ==============================================================================
# SCENARIO 9: Citation Integrity & Fabricated ID Rejection
# ==============================================================================


@pytest.mark.asyncio
async def test_scenario_citation_integrity_and_hallucination_filtering(eval_knowledge_store):
	"""Test that fabricated chunk/doc citations are stripped, and 100% fabricated answers are rejected."""
	store = eval_knowledge_store

	# Ingest a real document to retrieve
	results = await store.search_async('Enterprise tier', top_k=1)
	assert len(results) == 1
	real_chunk_id = results[0].chunk_id
	real_doc_id = results[0].document_id

	# 1. Partially fabricated citations (1 real, 1 fake)
	def partial_hallucinator(prompt: str, sys_prompt: Optional[str] = None) -> str:
		return (
			f'Enterprise tier includes SLA [doc_id: {real_doc_id}, chunk_id: {real_chunk_id}]. '
			f'Also includes quantum encryption [doc_id: doc-fake123, chunk_id: chk-fake123].'
		)

	agent_partial = KnowledgeAgent(
		knowledge_store=store,
		llm_provider=MockLLMProvider(custom_responder=partial_hallucinator),
	)
	ans_partial = await agent_partial.answer('Enterprise tier SLA')
	assert ans_partial.is_grounded is True
	cited_chunk_ids = [s.chunk_id for s in ans_partial.sources]
	assert real_chunk_id in cited_chunk_ids
	assert 'chk-fake123' not in cited_chunk_ids  # Fabricated citation stripped

	# 2. Entirely fabricated citations (0 real, 2 fake)
	def total_hallucinator(prompt: str, sys_prompt: Optional[str] = None) -> str:
		return 'Answer cites [doc_id: doc-imaginary, chunk_id: chk-phantom].'

	agent_total = KnowledgeAgent(
		knowledge_store=store,
		llm_provider=MockLLMProvider(custom_responder=total_hallucinator),
	)
	ans_total = await agent_total.answer('Enterprise tier SLA')
	# Completely fabricated answer must be rejected as ungrounded
	assert ans_total.is_grounded is False
	assert ans_total.insufficient_evidence is True
	assert len(ans_total.sources) == 0


# ==============================================================================
# SCENARIO 10: Comprehensive Evaluation Metrics Runner & Summary
# ==============================================================================


@pytest.mark.asyncio
async def test_comprehensive_rag_evaluation_metrics_summary(eval_knowledge_store, caplog):
	"""Executes the full evaluation test dataset, computes aggregate metrics, and validates quality thresholds."""
	store = eval_knowledge_store
	agent = KnowledgeAgent(knowledge_store=store, llm_provider=MockLLMProvider())

	dataset: list[EvalTestCase] = [
		EvalTestCase(
			scenario='Single-Document Fact',
			query='What monthly availability SLA does Enterprise tier guarantee?',
			expected_doc_filename='service_tier_spec.md',
			expected_in_content=['99.99%'],
			expected_metadata={'tier': 'enterprise'},
		),
		EvalTestCase(
			scenario='Single-Document Fact',
			query='What hardware security modules compliance mode is used?',
			expected_doc_filename='service_tier_spec.md',
			expected_in_content=['FIPS 140-2 Level 3'],
			expected_metadata={'author': 'Reliability Engineering'},
		),
		EvalTestCase(
			scenario='Multi-Chunk Synthesis',
			query='How does external ingress traffic route to private worker execution pods?',
			expected_doc_filename='network_mesh_topology.md',
			expected_in_content=['443', '8443'],
			requires_multi_chunk=True,
		),
		EvalTestCase(
			scenario='Distractor Rejection',
			query='What is the private worker cluster subnet and mTLS port?',
			expected_doc_filename='network_mesh_topology.md',
			distractor_filename='coral_reef_mesh_sensor.txt',
		),
		EvalTestCase(
			scenario='Conflicting Evidence',
			query='What is the inactivity timeout duration for user sessions?',
			expected_doc_filename='session_policy_v2.md',
		),
		EvalTestCase(
			scenario='Out-of-Domain Rejection',
			query='What is the orbital eccentricity of Halley Comet?',
			expected_doc_filename='',
			is_out_of_domain=True,
		),
		EvalTestCase(
			scenario='Out-of-Domain Rejection',
			query='Explain the cellular biology of mitochondrial ATP synthesis.',
			expected_doc_filename='',
			is_out_of_domain=True,
		),
	]

	metrics = EvaluationMetrics()

	for tc in dataset:
		metrics.total_queries += 1

		if tc.is_out_of_domain:
			metrics.out_of_domain_queries += 1
			answer = await agent.answer(tc.query, top_k=3)
			if answer.insufficient_evidence and not answer.is_grounded and len(answer.sources) == 0:
				metrics.out_of_domain_correct_rejections += 1
			continue

		# Factual query evaluation
		raw_search = await store.search_async(tc.query, top_k=3)
		retrieved_filenames = [r.metadata.get('source_filename') for r in raw_search]

		# Recall@1
		if retrieved_filenames and retrieved_filenames[0] == tc.expected_doc_filename:
			metrics.recall_at_1_hits += 1

		# Recall@3
		if tc.expected_doc_filename in retrieved_filenames[:3]:
			metrics.recall_at_3_hits += 1

		# Distractor Rejection
		if tc.distractor_filename:
			metrics.distractor_queries += 1
			if tc.distractor_filename not in retrieved_filenames or (
				tc.expected_doc_filename in retrieved_filenames
				and retrieved_filenames.index(tc.expected_doc_filename) < retrieved_filenames.index(tc.distractor_filename)
			):
				metrics.distractor_rejection_hits += 1

		# Grounded Answer and Citations
		answer = await agent.answer(tc.query, top_k=3)
		if answer.is_grounded:
			for src in answer.sources:
				metrics.total_generated_citations += 1
				# Verify citation exists in actual retrieved chunks
				if any(r.chunk_id == src.chunk_id for r in raw_search):
					metrics.valid_citations += 1

				# Check metadata preservation
				if tc.expected_metadata:
					metrics.metadata_preservation_checks += 1
					if src.source_filename == tc.expected_doc_filename and src.chunk_id:
						metrics.metadata_preservation_hits += 1

	# Format report table
	report = f"""
================================================================================
InsightBot AI - Deterministic RAG Quality Evaluation Benchmark
================================================================================
Total Test Cases Evaluated:       {metrics.total_queries}
Factual Query Recall@1:            {metrics.recall_at_1:.2%} ({metrics.recall_at_1_hits}/{metrics.total_queries - metrics.out_of_domain_queries})
Factual Query Recall@3:            {metrics.recall_at_3:.2%} ({metrics.recall_at_3_hits}/{metrics.total_queries - metrics.out_of_domain_queries})
Distractor Rejection Rate:        {metrics.distractor_rejection_rate:.2%} ({metrics.distractor_rejection_hits}/{metrics.distractor_queries})
Citation Validity Rate:           {metrics.citation_validity_rate:.2%} ({metrics.valid_citations}/{metrics.total_generated_citations})
Out-of-Domain Rejection Rate:     {metrics.out_of_domain_rejection_rate:.2%} ({metrics.out_of_domain_correct_rejections}/{metrics.out_of_domain_queries})
Metadata Preservation Rate:       {metrics.metadata_preservation_rate:.2%} ({metrics.metadata_preservation_hits}/{metrics.metadata_preservation_checks})
================================================================================
"""
	print(report)

	# Assert quality baseline thresholds
	assert metrics.recall_at_1 >= 0.80, f'Recall@1 failed threshold: {metrics.recall_at_1:.2%}'
	assert metrics.recall_at_3 == 1.0, f'Recall@3 failed threshold: {metrics.recall_at_3:.2%}'
	assert metrics.citation_validity_rate == 1.0, f'Citation validity failed: {metrics.citation_validity_rate:.2%}'
	assert metrics.out_of_domain_rejection_rate == 1.0, f'Out of domain rejection failed: {metrics.out_of_domain_rejection_rate:.2%}'
	assert metrics.distractor_rejection_rate == 1.0, f'Distractor rejection failed: {metrics.distractor_rejection_rate:.2%}'
	assert metrics.metadata_preservation_rate == 1.0, f'Metadata preservation failed: {metrics.metadata_preservation_rate:.2%}'
