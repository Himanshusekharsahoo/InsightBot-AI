"""Private Knowledge & RAG store for InsightBot AI teams.

Provides hybrid storage supporting PostgreSQL + pgvector persistence,
deterministic chunking, vector embeddings, and semantic similarity search,
while maintaining backward compatibility with the legacy research document index.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

from insightbot.config.settings import get_settings
from insightbot.db.base import BaseVectorStore
from insightbot.db.models import (
	Conversation,
	ConversationMessage,
	ConversationWithMessages,
	Document,
	DocumentChunk,
	KnowledgeSearchResult,
	ResearchSession,
)
from insightbot.db.vector_store import PgVectorStore
from insightbot.knowledge.chunker import DeterministicChunker, compute_content_hash
from insightbot.knowledge.embeddings import BaseEmbeddingModel, get_embedding_provider
from insightbot.knowledge.parsers import DocumentParserFactory
from insightbot.knowledge.validation import validate_upload
from insightbot.security.policy import SensitiveDataSanitizer

logger = logging.getLogger('insightbot.knowledge.store')


@dataclass
class ResearchDocument:
	"""A structured knowledge item gathered by the research sub-engine (legacy compatibility)."""

	doc_id: str
	title: str
	source_url: str
	content: str
	metadata: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class SearchResultFuture(list):
	"""Dual-purpose search outcome.

	Behaves as a list of legacy ResearchDocument items when used synchronously,
	and as an awaitable coroutine returning list[KnowledgeSearchResult] when awaited.
	"""

	def __init__(self, sync_items: list[Any], async_factory: Any) -> None:
		super().__init__(sync_items)
		self._async_factory = async_factory

	def __await__(self):
		return self._async_factory().__await__()


class KnowledgeStore:
	"""Private team knowledge repository and semantic vector search store."""

	def __init__(
		self,
		storage_dir: Union[str, Path, None] = None,
		vector_store: Optional[BaseVectorStore] = None,
		embedding_provider: Optional[BaseEmbeddingModel] = None,
		chunk_size: Optional[int] = None,
		chunk_overlap: Optional[int] = None,
	) -> None:
		settings = get_settings()

		# File-based legacy index directory
		if storage_dir is None:
			storage_dir = Path.home() / '.config' / 'insightbot' / 'knowledge'
		self.storage_dir = Path(storage_dir)
		self.storage_dir.mkdir(parents=True, exist_ok=True)
		self._index_file = self.storage_dir / 'index.json'
		self._documents: dict[str, ResearchDocument] = {}
		self._load_legacy_index()

		# Core RAG dependencies
		self._settings = settings
		self._vector_store = vector_store
		self._embedding_provider = embedding_provider
		self._chunker = DeterministicChunker(
			chunk_size=chunk_size or settings.chunk_size,
			chunk_overlap=chunk_overlap or settings.chunk_overlap,
		)
		self._initialized: bool = False
		self._closed: bool = False

	@property
	def settings(self):
		"""Retrieve settings instance."""
		return self._settings or get_settings()

	def _load_legacy_index(self) -> None:
		if self._index_file.exists():
			try:
				data = json.loads(self._index_file.read_text(encoding='utf-8'))
				for item in data:
					doc = ResearchDocument(**item)
					self._documents[doc.doc_id] = doc
			except Exception:
				self._documents = {}

	def _save_legacy_index(self) -> None:
		try:
			data = [doc.__dict__ for doc in self._documents.values()]
			self._index_file.write_text(json.dumps(data, indent=2), encoding='utf-8')
		except Exception:
			pass

	@property
	def vector_store(self) -> BaseVectorStore:
		"""Lazy initialization of configured vector store."""
		if self._vector_store is None:
			settings = get_settings()
			vmode = getattr(settings, 'vector_store_mode', 'auto').lower()
			if vmode == 'in_memory':
				from insightbot.db.in_memory import InMemoryVectorStore

				self._vector_store = InMemoryVectorStore(embedding_dim=settings.embedding_dimension)
			else:
				self._vector_store = PgVectorStore()
		return self._vector_store

	@property
	def embedding_provider(self) -> BaseEmbeddingModel:
		"""Lazy initialization of configured embedding provider."""
		if self._embedding_provider is None:
			self._embedding_provider = get_embedding_provider()
		return self._embedding_provider

	@property
	def is_closed(self) -> bool:
		"""Return True if the store connection pool has been closed."""
		return self._closed

	async def initialize(self) -> None:
		"""Initialize underlying database connections and schemas."""
		if not self._initialized or self._closed:
			await self.vector_store.initialize()
			self._initialized = True
			self._closed = False

	async def close(self) -> None:
		"""Release database connection pool resources safely."""
		self._closed = True
		if self._vector_store is not None:
			try:
				await self._vector_store.close()
			except Exception as e:
				logger.warning(f'Error closing vector store: {e}')
			self._initialized = False

	# -------------------------------------------------------------------------
	# Phase 1 RAG Document Ingestion & Retrieval API
	# -------------------------------------------------------------------------

	async def ingest_document(
		self,
		filename: str,
		content: bytes,
		metadata: Optional[dict[str, Any]] = None,
		source_id: Optional[str] = None,
		organization_id: Optional[str] = None,
		workspace_id: Optional[str] = None,
	) -> Document:
		"""Ingest, parse, chunk, embed, and persist a document.

		Performs content hash deduplication, transaction tracking, and safe error handling.
		"""
		await self.initialize()

		# 1. Validate upload filename and size constraints
		safe_filename = validate_upload(
			filename,
			content_length=len(content),
			max_size_bytes=self.settings.max_upload_size_bytes,
		)
		if not content or not content.strip():
			raise ValueError(f"Document '{safe_filename}' is empty or contains only whitespace.")

		content_hash = compute_content_hash(content)

		# 2. Duplicate detection: if document with same hash already exists in this tenant and is completed
		existing = None
		if hasattr(self.vector_store, 'get_document_by_content_hash'):
			existing = await self.vector_store.get_document_by_content_hash(content_hash, organization_id=organization_id)
			if existing and existing.processing_status == 'completed':
				logger.info(f'Document with identical content hash {content_hash} already ingested: {existing.id}')
				if 'doc_type' not in existing.metadata:
					existing.metadata['doc_type'] = existing.source_type
				if metadata:
					existing.metadata.update(metadata)
					await self.vector_store.upsert_document(existing)
				return existing

		# 3. Parse document
		parsed = DocumentParserFactory.parse_file(safe_filename, content)
		if not parsed.text or not parsed.text.strip():
			raise ValueError(f"Document '{safe_filename}' contains no extractable text.")

		# 4. Create initial document record with 'processing' status
		doc_metadata = {
			'source_filename': safe_filename,
			'doc_type': (metadata or {}).get('doc_type', parsed.source_type),
			**parsed.metadata,
			**(metadata or {}),
		}
		doc_kwargs: dict[str, Any] = {
			'filename': safe_filename,
			'source_id': source_id,
			'source_type': parsed.source_type,
			'mime_type': parsed.mime_type,
			'content_hash': content_hash,
			'processing_status': 'processing',
			'metadata': doc_metadata,
			'organization_id': organization_id,
			'workspace_id': workspace_id,
		}
		if existing is not None:
			doc_kwargs['id'] = existing.id
			# Clean up any leftover chunks from previous attempt
			await self.vector_store.delete_chunks(existing.id)

		doc = Document(**doc_kwargs)
		await self.vector_store.upsert_document(doc)

		# 5. Split document into deterministic chunks
		try:
			chunks = self._chunker.chunk_document(
				document_id=doc.id,
				parsed_doc=parsed,
				document_metadata=doc_metadata,
				organization_id=organization_id,
				workspace_id=workspace_id,
			)

			max_chunks = getattr(self.settings, 'max_chunks_per_document', 5000)
			if len(chunks) > max_chunks:
				raise ValueError(
					f"Document '{safe_filename}' generated {len(chunks)} chunks, exceeding the maximum limit of {max_chunks}."
				)

			if not chunks:
				raise ValueError(f"Document '{safe_filename}' yielded 0 chunks.")

			# 6. Generate embeddings in batch
			chunk_texts = [c.content for c in chunks]
			embeddings = await self.embedding_provider.embed_batch(chunk_texts)
			for chk, emb in zip(chunks, embeddings):
				chk.embedding = emb

			# 7. Persist chunks in vector store
			await self.vector_store.insert_chunks(chunks)

			# 8. Mark document as completed
			doc.processing_status = 'completed'
			doc.error_message = None
			await self.vector_store.upsert_document(doc)

			logger.info(f"Successfully ingested document '{safe_filename}' (id={doc.id}, chunks={len(chunks)})")
			return doc

		except Exception as e:
			logger.error(f"Ingestion failed for document '{safe_filename}': {e}", exc_info=True)
			try:
				await self.vector_store.delete_chunks(doc.id)
			except Exception as cleanup_err:
				logger.warning(f"Error cleaning up chunks for failed document '{doc.id}': {cleanup_err}")

			sanitized_err = SensitiveDataSanitizer.sanitize(str(e))
			if existing is None:
				# Brand new document that failed: delete document row to prevent orphaned partial records
				try:
					await self.vector_store.delete_document(doc.id, organization_id=organization_id)
				except Exception as del_err:
					logger.warning(f"Error deleting failed new document record '{doc.id}': {del_err}")
			else:
				# Pre-existing document record: do not destroy it; preserve and mark failed
				doc.processing_status = 'failed'
				doc.error_message = sanitized_err
				try:
					await self.vector_store.upsert_document(doc)
				except Exception as upsert_err:
					logger.warning(f"Error updating failed status for document '{doc.id}': {upsert_err}")
			raise

	async def ingest_file(
		self,
		file_path: Union[str, Path],
		metadata: Optional[dict[str, Any]] = None,
		organization_id: Optional[str] = None,
		workspace_id: Optional[str] = None,
	) -> Document:
		"""Ingest a local filesystem file."""
		path = Path(file_path)
		if not path.is_file():
			raise FileNotFoundError(f'File not found: {path}')
		content = path.read_bytes()
		return await self.ingest_document(
			filename=path.name,
			content=content,
			metadata={**(metadata or {}), 'file_path': str(path.resolve())},
			organization_id=organization_id,
			workspace_id=workspace_id,
		)

	async def list_documents(self, limit: int = 100, offset: int = 0, organization_id: Optional[str] = None) -> list[Document]:
		"""List ingested documents ordered by creation time descending, optionally scoped by organization."""
		await self.initialize()
		return await self.vector_store.list_documents(limit=limit, offset=offset, organization_id=organization_id)

	async def get_document(self, document_id: str, organization_id: Optional[str] = None) -> Optional[Document]:
		"""Retrieve document metadata by document ID, optionally scoped by organization."""
		await self.initialize()
		return await self.vector_store.get_document(document_id, organization_id=organization_id)

	async def get_chunks(self, document_id: str, organization_id: Optional[str] = None) -> list[DocumentChunk]:
		"""Retrieve all chunks belonging to a document ordered by chunk_index, optionally scoped by organization."""
		await self.initialize()
		return await self.vector_store.get_chunks(document_id, organization_id=organization_id)

	async def delete_document(self, document_id: str, organization_id: Optional[str] = None) -> bool:
		"""Delete a document and all its chunks, optionally scoped by organization."""
		await self.initialize()
		return await self.vector_store.delete_document(document_id, organization_id=organization_id)

	# -------------------------------------------------------------------------
	# Phase 14B Persistent Conversation History API
	# -------------------------------------------------------------------------

	async def create_conversation(self, conversation: Conversation) -> str:
		"""Create a new persistent conversation thread."""
		await self.initialize()
		return await self.vector_store.create_conversation(conversation)

	async def get_conversation(self, conversation_id: str, organization_id: Optional[str] = None) -> Optional[Conversation]:
		"""Retrieve conversation metadata by ID, optionally scoped to organization."""
		await self.initialize()
		return await self.vector_store.get_conversation(conversation_id, organization_id=organization_id)

	async def list_conversations(
		self,
		limit: int = 50,
		offset: int = 0,
		organization_id: Optional[str] = None,
		workspace_id: Optional[str] = None,
	) -> list[Conversation]:
		"""List conversation threads ordered by updated_at descending, optionally scoped."""
		await self.initialize()
		return await self.vector_store.list_conversations(
			limit=limit, offset=offset, organization_id=organization_id, workspace_id=workspace_id
		)

	async def update_conversation(
		self,
		conversation_id: str,
		title: Optional[str] = None,
		metadata: Optional[dict[str, Any]] = None,
		organization_id: Optional[str] = None,
	) -> Optional[Conversation]:
		"""Update conversation title or metadata, optionally scoped to organization."""
		await self.initialize()
		return await self.vector_store.update_conversation(
			conversation_id=conversation_id,
			title=title,
			metadata=metadata,
			organization_id=organization_id,
		)

	async def delete_conversation(self, conversation_id: str, organization_id: Optional[str] = None) -> bool:
		"""Delete a conversation and cascade delete all its messages."""
		await self.initialize()
		return await self.vector_store.delete_conversation(conversation_id, organization_id=organization_id)

	async def add_message(self, message: ConversationMessage) -> str:
		"""Persist a message in a conversation thread."""
		await self.initialize()
		return await self.vector_store.add_message(message)

	async def get_messages(self, conversation_id: str, limit: Optional[int] = None) -> list[ConversationMessage]:
		"""Retrieve chronological messages for a conversation thread."""
		await self.initialize()
		return await self.vector_store.get_messages(conversation_id, limit=limit)

	async def get_conversation_with_messages(
		self,
		conversation_id: str,
		organization_id: Optional[str] = None,
		message_limit: Optional[int] = None,
	) -> Optional[ConversationWithMessages]:
		"""Retrieve conversation metadata and its chronological message history."""
		conv = await self.get_conversation(conversation_id, organization_id=organization_id)
		if not conv:
			return None
		messages = await self.get_messages(conversation_id, limit=message_limit)
		return ConversationWithMessages(
			id=conv.id,
			organization_id=conv.organization_id,
			workspace_id=conv.workspace_id,
			title=conv.title,
			metadata=conv.metadata,
			created_at=conv.created_at,
			updated_at=conv.updated_at,
			messages=messages,
		)

	# -------------------------------------------------------------------------
	# Phase 14C Persistent Research Artifacts API
	# -------------------------------------------------------------------------

	async def create_research_session(self, session: ResearchSession) -> str:
		"""Persist a research session and return its ID."""
		await self.initialize()
		return await self.vector_store.create_research_session(session)

	async def get_research_session(self, session_id: str, organization_id: Optional[str] = None) -> Optional[ResearchSession]:
		"""Retrieve a research session by ID, optionally scoped to organization."""
		await self.initialize()
		return await self.vector_store.get_research_session(session_id, organization_id=organization_id)

	async def list_research_sessions(
		self,
		limit: int = 50,
		offset: int = 0,
		organization_id: Optional[str] = None,
		workspace_id: Optional[str] = None,
	) -> list[ResearchSession]:
		"""List research sessions ordered by created_at descending, optionally scoped."""
		await self.initialize()
		return await self.vector_store.list_research_sessions(
			limit=limit, offset=offset, organization_id=organization_id, workspace_id=workspace_id
		)

	async def delete_research_session(self, session_id: str, organization_id: Optional[str] = None) -> bool:
		"""Delete a research session, optionally scoped to organization."""
		await self.initialize()
		return await self.vector_store.delete_research_session(session_id, organization_id=organization_id)

	async def update_research_session(
		self,
		session_id: str,
		organization_id: str,
		status: Optional[str] = None,
		progress_stage: Optional[str] = None,
		progress_percent: Optional[int] = None,
		progress_message: Optional[str] = None,
		summary: Optional[str] = None,
		sources: Optional[list[str]] = None,
		internal_evidence: Optional[list[dict[str, Any]]] = None,
		web_evidence: Optional[list[dict[str, Any]]] = None,
		conflicts: Optional[list[dict[str, Any]]] = None,
		citations: Optional[list[dict[str, Any]]] = None,
		is_grounded: Optional[bool] = None,
		sufficient_evidence: Optional[bool] = None,
		error: Optional[str] = None,
		metadata: Optional[dict[str, Any]] = None,
		started_at: Optional[str] = None,
		completed_at: Optional[str] = None,
	) -> Optional[ResearchSession]:
		"""Update status, progress, or report content for a research session with state validation."""
		await self.initialize()
		return await self.vector_store.update_research_session(
			session_id=session_id,
			organization_id=organization_id,
			status=status,
			progress_stage=progress_stage,
			progress_percent=progress_percent,
			progress_message=progress_message,
			summary=summary,
			sources=sources,
			internal_evidence=internal_evidence,
			web_evidence=web_evidence,
			conflicts=conflicts,
			citations=citations,
			is_grounded=is_grounded,
			sufficient_evidence=sufficient_evidence,
			error=error,
			metadata=metadata,
			started_at=started_at,
			completed_at=completed_at,
		)

	async def search_async(
		self,
		query: str,
		top_k: int = 5,
		filter_metadata: Optional[dict[str, Any]] = None,
		hybrid: bool = False,
		organization_id: Optional[str] = None,
	) -> list[KnowledgeSearchResult]:
		"""Direct asynchronous semantic search against stored document chunks.

		Args:
			query: User search query
			top_k: Number of chunks to retrieve
			filter_metadata: Optional metadata criteria (e.g. document_id, doc_type)
			hybrid: When True, combines dense semantic vector retrieval with lexical keyword matching
			organization_id: Optional organization tenant boundary
		"""
		if not query or not query.strip():
			return []
		await self.initialize()

		if not hybrid:
			query_embedding = await self.embedding_provider.embed_text(query)
			return await self.vector_store.similarity_search(
				query_embedding=query_embedding,
				top_k=top_k,
				filter_metadata=filter_metadata,
				organization_id=organization_id,
			)

		# Hybrid retrieval: Fetch BOTH dense vector candidates and lexical keyword candidates
		query_embedding = await self.embedding_provider.embed_text(query)
		fetch_limit = max(top_k * 3, 10)

		# 1. Dense vector candidates
		vector_results = await self.vector_store.similarity_search(
			query_embedding=query_embedding,
			top_k=fetch_limit,
			filter_metadata=filter_metadata,
			organization_id=organization_id,
		)

		# 2. Lexical keyword candidates
		keyword_results = await self.vector_store.keyword_search(
			query=query,
			top_k=fetch_limit,
			filter_metadata=filter_metadata,
			organization_id=organization_id,
		)

		query_words = set(re.findall(r'[a-zA-Z0-9_\-]+', query.lower()))

		# Candidate map by chunk_id
		all_candidates: dict[str, KnowledgeSearchResult] = {}
		vector_scores: dict[str, float] = {}

		for r in vector_results:
			all_candidates[r.chunk_id] = r
			vector_scores[r.chunk_id] = r.score

		for r in keyword_results:
			if r.chunk_id not in all_candidates:
				all_candidates[r.chunk_id] = r

		# Rescore each candidate: 70% dense vector similarity + 30% lexical overlap
		scored_candidates: list[KnowledgeSearchResult] = []
		for cid, r in all_candidates.items():
			chunk_words = set(re.findall(r'[a-zA-Z0-9_\-]+', r.content.lower()))
			lexical_ratio = (len(query_words & chunk_words) / len(query_words)) if query_words else 0.0
			dense_score = vector_scores.get(cid, 0.0)
			combined_score = round(0.70 * dense_score + 0.30 * lexical_ratio, 4)
			scored_r = r.model_copy()
			scored_r.score = combined_score
			scored_candidates.append(scored_r)

		scored_candidates.sort(key=lambda x: x.score, reverse=True)
		return scored_candidates[:top_k]

	def search_legacy(self, query: str, top_k: int = 5) -> list[ResearchDocument]:
		"""Synchronous legacy keyword search across stored ResearchDocuments."""
		if not query or not query.strip():
			return []
		q = query.lower()
		sync_matches: list[ResearchDocument] = []
		for doc in self._documents.values():
			if q in doc.title.lower() or q in doc.content.lower():
				sync_matches.append(doc)
				if len(sync_matches) >= top_k:
					break
		return sync_matches

	def search(
		self,
		query: str,
		top_k: int = 5,
		filter_metadata: Optional[dict[str, Any]] = None,
		hybrid: bool = False,
		organization_id: Optional[str] = None,
	) -> SearchResultFuture:
		"""Search documents.

		Supports both synchronous legacy keyword search on ResearchDocuments and
		asynchronous semantic vector search on DocumentChunks via `await store.search(...)`.
		"""
		sync_matches = self.search_legacy(query=query, top_k=top_k)

		# 2. Define async vector search coroutine delegating to search_async
		async def _async_search() -> list[KnowledgeSearchResult]:
			return await self.search_async(
				query=query,
				top_k=top_k,
				filter_metadata=filter_metadata,
				hybrid=hybrid,
				organization_id=organization_id,
			)

		return SearchResultFuture(sync_matches, _async_search)

	# -------------------------------------------------------------------------
	# Legacy Compatibility Methods
	# -------------------------------------------------------------------------

	def store_document(
		self,
		doc_id: str,
		title: str,
		source_url: str,
		content: str,
		metadata: Optional[dict[str, Any]] = None,
	) -> ResearchDocument:
		"""Save a research document to the legacy team knowledge base."""
		doc = ResearchDocument(
			doc_id=doc_id,
			title=title,
			source_url=source_url,
			content=content,
			metadata=metadata or {},
		)
		self._documents[doc_id] = doc
		self._save_legacy_index()
		return doc

	def get_all(self) -> list[ResearchDocument]:
		"""Return all stored legacy research documents."""
		return list(self._documents.values())


__all__ = [
	'ResearchDocument',
	'SearchResultFuture',
	'KnowledgeStore',
]
