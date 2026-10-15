"""In-memory Vector Store for fast isolated unit testing.

Implements BaseVectorStore without external database dependencies.
"""

import re
from typing import Any, Optional

from insightbot.db.base import STORAGE_MODE_IN_MEMORY, BaseVectorStore
from insightbot.db.models import (
	RESEARCH_STATUS_IN_PROGRESS,
	TERMINAL_RESEARCH_STATUSES,
	VALID_RESEARCH_TRANSITIONS,
	Conversation,
	ConversationMessage,
	Document,
	DocumentChunk,
	KnowledgeSearchResult,
	ResearchSession,
)
from insightbot.db.vector_store import _cosine_similarity


class InMemoryVectorStore(BaseVectorStore):
	"""Pure in-memory vector store implementing BaseVectorStore for tests."""

	def __init__(self, embedding_dim: Optional[int] = None) -> None:
		self._documents: dict[str, Document] = {}
		self._chunks: dict[str, DocumentChunk] = {}
		self._conversations: dict[str, Conversation] = {}
		self._messages: dict[str, list[ConversationMessage]] = {}
		self._research_sessions: dict[str, ResearchSession] = {}
		self._initialized: bool = False
		self.embedding_dim = embedding_dim

	@property
	def storage_mode(self) -> str:
		"""Return in-memory storage mode identifier."""
		return STORAGE_MODE_IN_MEMORY

	async def initialize(self) -> None:
		self._initialized = True

	async def close(self) -> None:
		self._initialized = False

	async def upsert_document(self, document: Document) -> str:
		self._documents[document.id] = document.model_copy(deep=True)
		return document.id

	async def insert_chunks(self, chunks: list[DocumentChunk]) -> list[str]:
		inserted = []
		for chunk in chunks:
			if self.embedding_dim is not None and chunk.embedding is not None:
				if len(chunk.embedding) != self.embedding_dim:
					raise ValueError(
						f"Embedding dimension mismatch for chunk '{chunk.id}': "
						f'got {len(chunk.embedding)}, expected {self.embedding_dim}'
					)
			self._chunks[chunk.id] = chunk.model_copy(deep=True)
			inserted.append(chunk.id)
		return inserted

	async def delete_document(self, document_id: str, organization_id: Optional[str] = None) -> bool:
		if document_id in self._documents:
			doc = self._documents[document_id]
			if organization_id and doc.organization_id != organization_id:
				return False
			del self._documents[document_id]
			# Cascade delete associated chunks
			chunk_ids_to_del = [cid for cid, c in self._chunks.items() if c.document_id == document_id]
			for cid in chunk_ids_to_del:
				del self._chunks[cid]
			return True
		return False

	async def delete_chunks(self, document_id: str) -> int:
		chunk_ids_to_del = [cid for cid, c in self._chunks.items() if c.document_id == document_id]
		for cid in chunk_ids_to_del:
			del self._chunks[cid]
		return len(chunk_ids_to_del)

	async def get_document(self, document_id: str, organization_id: Optional[str] = None) -> Optional[Document]:
		doc = self._documents.get(document_id)
		if doc and organization_id and doc.organization_id != organization_id:
			return None
		return doc.model_copy(deep=True) if doc else None

	async def get_document_by_content_hash(self, content_hash: str, organization_id: Optional[str] = None) -> Optional[Document]:
		for doc in self._documents.values():
			if doc.content_hash == content_hash:
				if organization_id and doc.organization_id != organization_id:
					continue
				return doc.model_copy(deep=True)
		return None

	async def list_documents(self, limit: int = 100, offset: int = 0, organization_id: Optional[str] = None) -> list[Document]:
		docs = list(self._documents.values())
		if organization_id:
			docs = [d for d in docs if d.organization_id == organization_id]
		docs.sort(key=lambda d: d.created_at, reverse=True)
		return [d.model_copy(deep=True) for d in docs[offset : offset + limit]]

	async def get_chunks(self, document_id: str, organization_id: Optional[str] = None) -> list[DocumentChunk]:
		matching = [c for c in self._chunks.values() if c.document_id == document_id]
		if organization_id:
			matching = [c for c in matching if c.organization_id == organization_id]
		matching.sort(key=lambda c: c.chunk_index)
		return [c.model_copy(deep=True) for c in matching]

	async def similarity_search(
		self,
		query_embedding: list[float],
		top_k: int = 5,
		filter_metadata: Optional[dict[str, Any]] = None,
		organization_id: Optional[str] = None,
	) -> list[KnowledgeSearchResult]:
		if not query_embedding:
			return []

		if self.embedding_dim is not None and len(query_embedding) != self.embedding_dim:
			raise ValueError(f'Query embedding dimension mismatch: got {len(query_embedding)}, expected {self.embedding_dim}.')

		org_filter = organization_id or (filter_metadata.get('organization_id') if filter_metadata else None)
		candidates = list(self._chunks.values())
		if org_filter:
			candidates = [c for c in candidates if c.organization_id == org_filter]

		if filter_metadata:
			filtered: list[DocumentChunk] = []
			for c in candidates:
				matches = True
				for k, v in filter_metadata.items():
					if k == 'organization_id':
						continue
					elif k == 'document_id':
						if c.document_id != v:
							matches = False
							break
					elif c.metadata.get(k) != v:
						matches = False
						break
				if matches:
					filtered.append(c)
			candidates = filtered

		scored: list[tuple[float, DocumentChunk]] = []
		for c in candidates:
			if c.embedding:
				sim = _cosine_similarity(query_embedding, c.embedding)
				scored.append((sim, c))

		scored.sort(key=lambda x: x[0], reverse=True)
		top_items = scored[:top_k]

		return [
			KnowledgeSearchResult(
				document_id=c.document_id,
				chunk_id=c.id,
				content=c.content,
				score=score,
				chunk_index=c.chunk_index,
				metadata=c.metadata,
			)
			for score, c in top_items
		]

	async def keyword_search(
		self,
		query: str,
		top_k: int = 5,
		filter_metadata: Optional[dict[str, Any]] = None,
		organization_id: Optional[str] = None,
	) -> list[KnowledgeSearchResult]:
		if not query or not query.strip():
			return []

		org_filter = organization_id or (filter_metadata.get('organization_id') if filter_metadata else None)
		candidates = list(self._chunks.values())
		if org_filter:
			candidates = [c for c in candidates if c.organization_id == org_filter]

		if filter_metadata:
			doc_filter = filter_metadata.get('document_id')
			if doc_filter:
				candidates = [c for c in candidates if c.document_id == doc_filter]

		query_words = set(re.findall(r'[a-zA-Z0-9_\-]+', query.lower()))
		if not query_words:
			return []

		candidates = list(self._chunks.values())
		if filter_metadata:
			filtered: list[DocumentChunk] = []
			for c in candidates:
				matches = True
				for k, v in filter_metadata.items():
					if k == 'document_id':
						if c.document_id != v:
							matches = False
							break
					elif c.metadata.get(k) != v:
						matches = False
						break
				if matches:
					filtered.append(c)
			candidates = filtered

		scored: list[tuple[float, DocumentChunk]] = []
		for c in candidates:
			chunk_words = set(re.findall(r'[a-zA-Z0-9_\-]+', c.content.lower()))
			overlap = len(query_words & chunk_words)
			if overlap > 0:
				score = overlap / len(query_words)
				scored.append((score, c))

		scored.sort(key=lambda x: x[0], reverse=True)
		top_items = scored[:top_k]

		return [
			KnowledgeSearchResult(
				document_id=c.document_id,
				chunk_id=c.id,
				content=c.content,
				score=score,
				chunk_index=c.chunk_index,
				metadata=c.metadata,
			)
			for score, c in top_items
		]

	async def create_conversation(self, conversation: Conversation) -> str:
		"""Persist a new conversation thread."""
		self._conversations[conversation.id] = conversation.model_copy(deep=True)
		if conversation.id not in self._messages:
			self._messages[conversation.id] = []
		return conversation.id

	async def get_conversation(self, conversation_id: str, organization_id: Optional[str] = None) -> Optional[Conversation]:
		"""Retrieve conversation metadata by ID, optionally scoped to organization."""
		conv = self._conversations.get(conversation_id)
		if conv and organization_id and conv.organization_id != organization_id:
			return None
		return conv.model_copy(deep=True) if conv else None

	async def list_conversations(
		self,
		limit: int = 50,
		offset: int = 0,
		organization_id: Optional[str] = None,
		workspace_id: Optional[str] = None,
	) -> list[Conversation]:
		"""List conversation threads ordered by updated_at descending, optionally scoped."""
		convs = list(self._conversations.values())
		if organization_id:
			convs = [c for c in convs if c.organization_id == organization_id]
		if workspace_id:
			convs = [c for c in convs if c.workspace_id == workspace_id]
		convs.sort(key=lambda c: c.updated_at, reverse=True)
		return [c.model_copy(deep=True) for c in convs[offset : offset + limit]]

	async def update_conversation(
		self,
		conversation_id: str,
		title: Optional[str] = None,
		metadata: Optional[dict[str, Any]] = None,
		organization_id: Optional[str] = None,
	) -> Optional[Conversation]:
		"""Update conversation title or metadata, optionally scoped to organization."""
		conv = await self.get_conversation(conversation_id, organization_id=organization_id)
		if not conv:
			return None
		from datetime import datetime, timezone

		if title is not None:
			conv.title = title
		if metadata is not None:
			conv.metadata.update(metadata)
		conv.updated_at = datetime.now(timezone.utc).isoformat()
		self._conversations[conversation_id] = conv
		return conv.model_copy(deep=True)

	async def delete_conversation(self, conversation_id: str, organization_id: Optional[str] = None) -> bool:
		"""Delete conversation and cascade delete its messages, optionally scoped to organization."""
		conv = self._conversations.get(conversation_id)
		if not conv:
			return False
		if organization_id and conv.organization_id != organization_id:
			return False
		del self._conversations[conversation_id]
		self._messages.pop(conversation_id, None)
		return True

	async def add_message(self, message: ConversationMessage) -> str:
		"""Persist a message in a conversation thread."""
		from datetime import datetime, timezone

		if message.conversation_id not in self._messages:
			self._messages[message.conversation_id] = []
		self._messages[message.conversation_id].append(message.model_copy(deep=True))
		if message.conversation_id in self._conversations:
			self._conversations[message.conversation_id].updated_at = datetime.now(timezone.utc).isoformat()
		return message.id

	async def get_messages(self, conversation_id: str, limit: Optional[int] = None) -> list[ConversationMessage]:
		"""Retrieve chronological messages for a conversation thread."""
		msgs = [m.model_copy(deep=True) for m in self._messages.get(conversation_id, [])]
		msgs.sort(key=lambda m: m.created_at)
		if limit is not None:
			msgs = msgs[-limit:]
		return msgs

	async def create_research_session(self, session: ResearchSession) -> str:
		"""Persist a research session and return its ID."""
		self._research_sessions[session.id] = session.model_copy(deep=True)
		return session.id

	async def get_research_session(self, session_id: str, organization_id: Optional[str] = None) -> Optional[ResearchSession]:
		"""Retrieve a research session by ID, optionally scoped to organization."""
		session = self._research_sessions.get(session_id)
		if not session:
			return None
		if organization_id and session.organization_id != organization_id:
			return None
		return session.model_copy(deep=True)

	async def list_research_sessions(
		self,
		limit: int = 50,
		offset: int = 0,
		organization_id: Optional[str] = None,
		workspace_id: Optional[str] = None,
	) -> list[ResearchSession]:
		"""List research sessions ordered by created_at descending, optionally scoped."""
		sessions = list(self._research_sessions.values())
		if organization_id:
			sessions = [s for s in sessions if s.organization_id == organization_id]
		if workspace_id:
			sessions = [s for s in sessions if s.workspace_id == workspace_id]
		sessions.sort(key=lambda s: s.created_at, reverse=True)
		return [s.model_copy(deep=True) for s in sessions[offset : offset + limit]]

	async def delete_research_session(self, session_id: str, organization_id: Optional[str] = None) -> bool:
		"""Delete a research session, optionally scoped to organization."""
		session = self._research_sessions.get(session_id)
		if not session:
			return False
		if organization_id and session.organization_id != organization_id:
			return False
		del self._research_sessions[session_id]
		return True

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
		session = self._research_sessions.get(session_id)
		if not session or session.organization_id != organization_id:
			return None

		# State transition validation
		if status is not None and status != session.status:
			allowed_next = VALID_RESEARCH_TRANSITIONS.get(session.status, set())
			if status not in allowed_next:
				raise ValueError(
					f"Invalid state transition: Cannot transition research session '{session_id}' "
					f"from terminal status '{session.status}' to '{status}'"
				)
			session.status = status

		from datetime import datetime, timezone

		now_iso = datetime.now(timezone.utc).isoformat()

		is_terminal = session.status in TERMINAL_RESEARCH_STATUSES

		# Late progress updates cannot mutate already-terminal sessions
		if not is_terminal:
			if progress_stage is not None:
				session.progress_stage = progress_stage
			if progress_percent is not None:
				clamped = max(0, min(100, progress_percent))
				if session.status == RESEARCH_STATUS_IN_PROGRESS:
					session.progress_percent = max(session.progress_percent, clamped)
				else:
					session.progress_percent = clamped
			if progress_message is not None:
				session.progress_message = progress_message
		elif status is not None:
			# Transitioning to terminal in this update
			if progress_stage is not None:
				session.progress_stage = progress_stage
			if progress_percent is not None:
				session.progress_percent = max(0, min(100, progress_percent))
			if progress_message is not None:
				session.progress_message = progress_message
		if summary is not None:
			session.summary = summary
		if sources is not None:
			session.sources = sources
		if internal_evidence is not None:
			session.internal_evidence = internal_evidence
		if web_evidence is not None:
			session.web_evidence = web_evidence
		if conflicts is not None:
			session.conflicts = conflicts
		if citations is not None:
			session.citations = citations
		if is_grounded is not None:
			session.is_grounded = is_grounded
		if sufficient_evidence is not None:
			session.sufficient_evidence = sufficient_evidence
		if error is not None:
			session.error = error
		if metadata is not None:
			session.metadata.update(metadata)
		if started_at is not None:
			session.started_at = started_at
		if completed_at is not None:
			session.completed_at = completed_at

		session.updated_at = now_iso
		self._research_sessions[session_id] = session
		return session.model_copy(deep=True)


__all__ = ['InMemoryVectorStore']
