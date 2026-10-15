"""InsightBot AI Database & Persistence Interfaces.

Defines persistence abstractions for research records and vector store backends.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel, Field

from insightbot.core.models import ResearchResult
from insightbot.db.models import (
	Conversation,
	ConversationMessage,
	Document,
	DocumentChunk,
	KnowledgeSearchResult,
	ResearchSession,
)


class ResearchRecord(BaseModel):
	"""Database record representation of a completed research session."""

	record_id: str = Field(..., description='Unique record identifier')
	user_id: Optional[str] = Field(default=None, description='Requesting team member or service account')
	query: str = Field(..., description='Search query')
	result: ResearchResult = Field(..., description='Structured research result')
	created_at: str = Field(..., description='Timestamp of record creation')


class BaseResearchRepository(ABC):
	"""Abstract repository interface for research result persistence."""

	@abstractmethod
	async def save_record(self, record: ResearchRecord) -> str:
		"""Persist a research record and return its ID."""
		pass

	@abstractmethod
	async def get_record(self, record_id: str) -> Optional[ResearchRecord]:
		"""Retrieve a research record by ID."""
		pass

	@abstractmethod
	async def list_records(self, limit: int = 50) -> list[ResearchRecord]:
		"""List latest research records."""
		pass


class PgVectorUnavailableError(RuntimeError):
	"""Raised when native PostgreSQL vector extension (pgvector) is strictly required but unavailable."""

	pass


STORAGE_MODE_NATIVE_PGVECTOR = 'native_pgvector'
STORAGE_MODE_PG_ARRAY = 'pg_array'
STORAGE_MODE_IN_MEMORY = 'in_memory'


class BaseVectorStore(ABC):
	"""Abstract storage interface for documents, chunks, and vector similarity search.

	Decouples the knowledge pipeline from specific vector engines (PostgreSQL+pgvector,
	in-memory testing, or future backends).
	"""

	@property
	@abstractmethod
	def storage_mode(self) -> str:
		"""Return the current storage mode identifier (native_pgvector, pg_array, in_memory)."""
		pass

	@abstractmethod
	async def initialize(self) -> None:
		"""Initialize tables, extensions, or storage indexes."""
		pass

	@abstractmethod
	async def upsert_document(self, document: Document) -> str:
		"""Store or update a document metadata record."""
		pass

	@abstractmethod
	async def insert_chunks(self, chunks: list[DocumentChunk]) -> list[str]:
		"""Insert document chunks with their vector embeddings."""
		pass

	@abstractmethod
	async def delete_document(self, document_id: str, organization_id: Optional[str] = None) -> bool:
		"""Delete a document and all its associated chunks (cascade), optionally scoped to organization."""
		pass

	@abstractmethod
	async def delete_chunks(self, document_id: str) -> int:
		"""Delete all chunks for a document."""
		pass

	@abstractmethod
	async def get_document(self, document_id: str, organization_id: Optional[str] = None) -> Optional[Document]:
		"""Retrieve a document by its ID, optionally scoped to organization."""
		pass

	@abstractmethod
	async def get_document_by_content_hash(self, content_hash: str, organization_id: Optional[str] = None) -> Optional[Document]:
		"""Retrieve a document by its content hash to prevent duplicate ingestion, optionally scoped to organization."""
		pass

	@abstractmethod
	async def list_documents(self, limit: int = 100, offset: int = 0, organization_id: Optional[str] = None) -> list[Document]:
		"""List stored documents ordered by creation time descending, optionally scoped to organization."""
		pass

	@abstractmethod
	async def get_chunks(self, document_id: str, organization_id: Optional[str] = None) -> list[DocumentChunk]:
		"""Retrieve all chunks belonging to a document ordered by chunk_index, optionally scoped to organization."""
		pass

	@abstractmethod
	async def similarity_search(
		self,
		query_embedding: list[float],
		top_k: int = 5,
		filter_metadata: Optional[dict[str, Any]] = None,
		organization_id: Optional[str] = None,
	) -> list[KnowledgeSearchResult]:
		"""Perform semantic similarity search against stored chunk embeddings, optionally scoped to organization."""
		pass

	@abstractmethod
	async def keyword_search(
		self,
		query: str,
		top_k: int = 5,
		filter_metadata: Optional[dict[str, Any]] = None,
		organization_id: Optional[str] = None,
	) -> list[KnowledgeSearchResult]:
		"""Perform lexical keyword search across stored document chunks, optionally scoped to organization."""
		pass

	@abstractmethod
	async def create_conversation(self, conversation: Conversation) -> str:
		"""Persist a new conversation thread and return its ID."""
		pass

	@abstractmethod
	async def get_conversation(self, conversation_id: str, organization_id: Optional[str] = None) -> Optional[Conversation]:
		"""Retrieve conversation metadata by ID, optionally scoped to organization."""
		pass

	@abstractmethod
	async def list_conversations(
		self,
		limit: int = 50,
		offset: int = 0,
		organization_id: Optional[str] = None,
		workspace_id: Optional[str] = None,
	) -> list[Conversation]:
		"""List conversation threads ordered by updated_at descending, optionally scoped."""
		pass

	@abstractmethod
	async def update_conversation(
		self,
		conversation_id: str,
		title: Optional[str] = None,
		metadata: Optional[dict[str, Any]] = None,
		organization_id: Optional[str] = None,
	) -> Optional[Conversation]:
		"""Update conversation title or metadata, optionally scoped to organization."""
		pass

	@abstractmethod
	async def delete_conversation(self, conversation_id: str, organization_id: Optional[str] = None) -> bool:
		"""Delete conversation and cascade delete its messages, optionally scoped to organization."""
		pass

	@abstractmethod
	async def add_message(self, message: ConversationMessage) -> str:
		"""Persist a message in a conversation thread."""
		pass

	@abstractmethod
	async def get_messages(self, conversation_id: str, limit: Optional[int] = None) -> list[ConversationMessage]:
		"""Retrieve chronological messages for a conversation thread."""
		pass

	@abstractmethod
	async def create_research_session(self, session: ResearchSession) -> str:
		"""Persist a research session and return its ID."""
		pass

	@abstractmethod
	async def get_research_session(self, session_id: str, organization_id: Optional[str] = None) -> Optional[ResearchSession]:
		"""Retrieve a research session by ID, optionally scoped to organization."""
		pass

	@abstractmethod
	async def list_research_sessions(
		self,
		limit: int = 50,
		offset: int = 0,
		organization_id: Optional[str] = None,
		workspace_id: Optional[str] = None,
	) -> list[ResearchSession]:
		"""List research sessions ordered by created_at descending, optionally scoped."""
		pass

	@abstractmethod
	async def delete_research_session(self, session_id: str, organization_id: Optional[str] = None) -> bool:
		"""Delete a research session, optionally scoped to organization."""
		pass

	@abstractmethod
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
		pass

	@abstractmethod
	async def close(self) -> None:
		"""Close underlying connections or release pool resources."""
		pass


__all__ = [
	'ResearchRecord',
	'BaseResearchRepository',
	'BaseVectorStore',
	'PgVectorUnavailableError',
	'STORAGE_MODE_NATIVE_PGVECTOR',
	'STORAGE_MODE_PG_ARRAY',
	'STORAGE_MODE_IN_MEMORY',
]
