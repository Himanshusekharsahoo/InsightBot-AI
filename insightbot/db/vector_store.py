"""PostgreSQL and pgvector-backed Vector Store implementation.

Implements BaseVectorStore using asyncpg and pgvector for high-performance
asynchronous persistence and semantic similarity search.
"""

import asyncio
import json
import logging
import math
import re
from datetime import datetime
from typing import Any, Optional

import asyncpg
from pgvector.asyncpg import register_vector

from insightbot.config.settings import get_settings
from insightbot.db.base import (
	STORAGE_MODE_NATIVE_PGVECTOR,
	STORAGE_MODE_PG_ARRAY,
	BaseVectorStore,
)
from insightbot.db.migrations import apply_migrations, check_pgvector_available
from insightbot.db.models import (
	TERMINAL_RESEARCH_STATUSES,
	VALID_RESEARCH_TRANSITIONS,
	Conversation,
	ConversationMessage,
	Document,
	DocumentChunk,
	KnowledgeSearchResult,
	ResearchSession,
	SourceAttribution,
)

logger = logging.getLogger('insightbot.db.vector_store')


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
	"""Compute cosine similarity between two float vectors."""
	if not vec_a or not vec_b or len(vec_a) != len(vec_b):
		return 0.0
	dot = sum(a * b for a, b in zip(vec_a, vec_b))
	norm_a = math.sqrt(sum(a * a for a, b in zip(vec_a, vec_b)))
	norm_b = math.sqrt(sum(b * b for a, b in zip(vec_b, vec_b)))
	if norm_a == 0.0 or norm_b == 0.0:
		return 0.0
	return dot / (norm_a * norm_b)


class PgVectorStore(BaseVectorStore):
	"""PostgreSQL and pgvector-backed Vector Store."""

	def __init__(
		self,
		database_url: Optional[str] = None,
		embedding_dim: Optional[int] = None,
		vector_mode: Optional[str] = None,
		min_pool_size: int = 2,
		max_pool_size: int = 10,
	) -> None:
		settings = get_settings()
		self.database_url = database_url or settings.database_url
		self.embedding_dim = embedding_dim or settings.embedding_dimension
		self.vector_mode = (vector_mode or getattr(settings, 'vector_store_mode', 'auto')).lower()
		if self.vector_mode in ('native', 'native_pgvector'):
			self.require_native_pgvector = True
			self.force_array_mode = False
		elif self.vector_mode in ('array', 'pg_array'):
			self.require_native_pgvector = False
			self.force_array_mode = True
		else:
			self.require_native_pgvector = False
			self.force_array_mode = False

		self.min_pool_size = min_pool_size
		self.max_pool_size = max_pool_size
		self._pool: Optional[asyncpg.Pool] = None
		self._has_pgvector: bool = False

	@property
	def storage_mode(self) -> str:
		"""Return the current active storage mode (native_pgvector or pg_array)."""
		if self._has_pgvector:
			return STORAGE_MODE_NATIVE_PGVECTOR
		return STORAGE_MODE_PG_ARRAY

	async def initialize(self) -> None:
		"""Initialize connection pool, run migrations, and register pgvector types."""
		if self._pool is not None:
			try:
				current_loop = asyncio.get_running_loop()
				pool_loop = getattr(self._pool, '_loop', None)
				if pool_loop is not None and (pool_loop.is_closed() or pool_loop is not current_loop):
					try:
						self._pool.terminate()
					except Exception:
						pass
					self._pool = None
				else:
					return
			except RuntimeError:
				return

		try:
			# Custom connection init callback to register vector type if available
			async def _init_connection(conn: asyncpg.Connection) -> None:
				if not self.force_array_mode:
					has_vector = await check_pgvector_available(conn)
					if has_vector:
						try:
							await register_vector(conn)
						except Exception as e:
							logger.debug(f'pgvector registration note: {e}')

			self._pool = await asyncpg.create_pool(
				dsn=self.database_url,
				min_size=self.min_pool_size,
				max_size=self.max_pool_size,
				init=_init_connection,
				timeout=10.0,
			)
		except Exception as e:
			raise ConnectionError(
				f"Failed to connect to PostgreSQL at '{self.database_url}'. "
				f'Ensure PostgreSQL is running and credentials are valid. Error: {e}'
			) from e

		# Apply migrations to ensure tables and indexes exist
		async with self._pool.acquire() as conn:
			if self.force_array_mode:
				await apply_migrations(conn, embedding_dim=self.embedding_dim, require_native_pgvector=False)
				self._has_pgvector = False
			else:
				self._has_pgvector = await apply_migrations(
					conn,
					embedding_dim=self.embedding_dim,
					require_native_pgvector=self.require_native_pgvector,
				)

	async def _ensure_pool(self) -> asyncpg.Pool:
		"""Ensure connection pool is initialized and attached to the current active event loop."""
		if self._pool is not None:
			try:
				current_loop = asyncio.get_running_loop()
				pool_loop = getattr(self._pool, '_loop', None)
				if pool_loop is not None and (pool_loop.is_closed() or pool_loop is not current_loop):
					try:
						self._pool.terminate()
					except Exception:
						pass
					self._pool = None
			except RuntimeError:
				pass
		if self._pool is None:
			await self.initialize()
		assert self._pool is not None
		return self._pool

	async def _get_conn(self) -> Any:
		"""Acquire connection from pool, ensuring initialized."""
		pool = await self._ensure_pool()
		return await pool.acquire()

	async def close(self) -> None:
		"""Close underlying connection pool."""
		if self._pool is not None:
			await self._pool.close()
			self._pool = None

	async def upsert_document(self, document: Document) -> str:
		"""Store or update a document metadata record."""
		pool = await self._ensure_pool()

		query = """
			INSERT INTO documents (
				id, source_id, filename, source_type, mime_type,
				content_hash, processing_status, error_message, metadata,
				organization_id, workspace_id, updated_at
			)
			VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, CURRENT_TIMESTAMP)
			ON CONFLICT (id) DO UPDATE SET
				source_id = EXCLUDED.source_id,
				filename = EXCLUDED.filename,
				source_type = EXCLUDED.source_type,
				mime_type = EXCLUDED.mime_type,
				content_hash = EXCLUDED.content_hash,
				processing_status = EXCLUDED.processing_status,
				error_message = EXCLUDED.error_message,
				metadata = EXCLUDED.metadata,
				updated_at = CURRENT_TIMESTAMP
			RETURNING id;
		"""
		async with pool.acquire() as conn:
			row = await conn.fetchrow(
				query,
				document.id,
				document.source_id,
				document.filename,
				document.source_type,
				document.mime_type,
				document.content_hash,
				document.processing_status,
				document.error_message,
				json.dumps(document.metadata),
				document.organization_id,
				document.workspace_id,
			)
			return str(row['id'])

	async def update_document_status(
		self,
		document_id: str,
		status: str,
		error_message: Optional[str] = None,
	) -> None:
		"""Update processing status and error message for a document."""
		pool = await self._ensure_pool()

		query = """
			UPDATE documents
			SET processing_status = $2, error_message = $3, updated_at = CURRENT_TIMESTAMP
			WHERE id = $1;
		"""
		async with pool.acquire() as conn:
			await conn.execute(query, document_id, status, error_message)

	async def get_document_by_content_hash(self, content_hash: str, organization_id: Optional[str] = None) -> Optional[Document]:
		"""Look up an existing document by its content hash, optionally scoped by organization."""
		pool = await self._ensure_pool()

		if organization_id:
			query = """
				SELECT id, source_id, filename, source_type, mime_type, content_hash,
				       processing_status, error_message, metadata, organization_id,
				       workspace_id, created_at, updated_at
				FROM documents
				WHERE content_hash = $1 AND organization_id = $2
				ORDER BY created_at DESC
				LIMIT 1;
			"""
			async with pool.acquire() as conn:
				row = await conn.fetchrow(query, content_hash, organization_id)
				if not row:
					return None
				return self._row_to_document(row)
		else:
			query = """
				SELECT id, source_id, filename, source_type, mime_type, content_hash,
				       processing_status, error_message, metadata, organization_id,
				       workspace_id, created_at, updated_at
				FROM documents
				WHERE content_hash = $1
				ORDER BY created_at DESC
				LIMIT 1;
			"""
			async with pool.acquire() as conn:
				row = await conn.fetchrow(query, content_hash)
				if not row:
					return None
				return self._row_to_document(row)

	async def insert_chunks(self, chunks: list[DocumentChunk]) -> list[str]:
		"""Insert document chunks with their vector embeddings in a single transaction."""
		if not chunks:
			return []

		for chunk in chunks:
			if chunk.embedding is not None and len(chunk.embedding) != self.embedding_dim:
				raise ValueError(
					f"Embedding dimension mismatch for chunk '{chunk.id}': "
					f'chunk vector length is {len(chunk.embedding)}, but vector store is configured for dimension {self.embedding_dim}.'
				)

		pool = await self._ensure_pool()

		inserted_ids: list[str] = []
		async with pool.acquire() as conn:
			async with conn.transaction():
				for chunk in chunks:
					metadata_json = json.dumps(chunk.metadata)
					# If embedding vector exists, adapt based on pgvector support
					if self._has_pgvector and chunk.embedding is not None:
						query = """
							INSERT INTO document_chunks (
								id, document_id, chunk_index, content, metadata,
								embedding, organization_id, workspace_id
							)
							VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
							ON CONFLICT (id) DO UPDATE SET
								content = EXCLUDED.content,
								metadata = EXCLUDED.metadata,
								embedding = EXCLUDED.embedding
							RETURNING id;
						"""
						row = await conn.fetchrow(
							query,
							chunk.id,
							chunk.document_id,
							chunk.chunk_index,
							chunk.content,
							metadata_json,
							chunk.embedding,
							chunk.organization_id,
							chunk.workspace_id,
						)
					else:
						# Array fallback or no embedding
						query = """
							INSERT INTO document_chunks (
								id, document_id, chunk_index, content, metadata,
								embedding, organization_id, workspace_id
							)
							VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
							ON CONFLICT (id) DO UPDATE SET
								content = EXCLUDED.content,
								metadata = EXCLUDED.metadata,
								embedding = EXCLUDED.embedding
							RETURNING id;
						"""
						row = await conn.fetchrow(
							query,
							chunk.id,
							chunk.document_id,
							chunk.chunk_index,
							chunk.content,
							metadata_json,
							chunk.embedding,
							chunk.organization_id,
							chunk.workspace_id,
						)
					inserted_ids.append(str(row['id']))

		return inserted_ids

	async def delete_document(self, document_id: str, organization_id: Optional[str] = None) -> bool:
		"""Delete a document and all its associated chunks via foreign-key cascade, optionally scoped by organization."""
		pool = await self._ensure_pool()

		if organization_id:
			query = 'DELETE FROM documents WHERE id = $1 AND organization_id = $2 RETURNING id;'
			async with pool.acquire() as conn:
				row = await conn.fetchrow(query, document_id, organization_id)
				return row is not None
		else:
			query = 'DELETE FROM documents WHERE id = $1 RETURNING id;'
			async with pool.acquire() as conn:
				row = await conn.fetchrow(query, document_id)
				return row is not None

	async def delete_chunks(self, document_id: str) -> int:
		"""Delete all chunks for a document."""
		pool = await self._ensure_pool()

		query = 'DELETE FROM document_chunks WHERE document_id = $1;'
		async with pool.acquire() as conn:
			res = await conn.execute(query, document_id)
			# res is like 'DELETE 5'
			parts = res.split()
			return int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0

	async def get_document(self, document_id: str, organization_id: Optional[str] = None) -> Optional[Document]:
		"""Retrieve a document by its ID, optionally scoped by organization."""
		pool = await self._ensure_pool()

		if organization_id:
			org_clause = '(organization_id = $2 OR organization_id IS NULL)' if organization_id == 'default' else 'organization_id = $2'
			query = f"""
				SELECT id, source_id, filename, source_type, mime_type, content_hash,
				       processing_status, error_message, metadata, organization_id,
				       workspace_id, created_at, updated_at
				FROM documents
				WHERE id = $1 AND {org_clause};
			"""
			async with pool.acquire() as conn:
				row = await conn.fetchrow(query, document_id, organization_id)
				if not row:
					return None
				return self._row_to_document(row)
		else:
			query = """
				SELECT id, source_id, filename, source_type, mime_type, content_hash,
				       processing_status, error_message, metadata, organization_id,
				       workspace_id, created_at, updated_at
				FROM documents
				WHERE id = $1;
			"""
			async with pool.acquire() as conn:
				row = await conn.fetchrow(query, document_id)
				if not row:
					return None
				return self._row_to_document(row)

	async def list_documents(self, limit: int = 100, offset: int = 0, organization_id: Optional[str] = None) -> list[Document]:
		"""List stored documents ordered by creation time descending, optionally scoped by organization."""
		pool = await self._ensure_pool()

		if organization_id:
			org_clause = '(organization_id = $1 OR organization_id IS NULL)' if organization_id == 'default' else 'organization_id = $1'
			query = f"""
				SELECT id, source_id, filename, source_type, mime_type, content_hash,
				       processing_status, error_message, metadata, organization_id,
				       workspace_id, created_at, updated_at
				FROM documents
				WHERE {org_clause}
				ORDER BY created_at DESC
				LIMIT $2 OFFSET $3;
			"""
			async with pool.acquire() as conn:
				rows = await conn.fetch(query, organization_id, limit, offset)
				return [self._row_to_document(r) for r in rows]
		else:
			query = """
				SELECT id, source_id, filename, source_type, mime_type, content_hash,
				       processing_status, error_message, metadata, organization_id,
				       workspace_id, created_at, updated_at
				FROM documents
				ORDER BY created_at DESC
				LIMIT $1 OFFSET $2;
			"""
			async with pool.acquire() as conn:
				rows = await conn.fetch(query, limit, offset)
				return [self._row_to_document(r) for r in rows]

	async def get_chunks(self, document_id: str, organization_id: Optional[str] = None) -> list[DocumentChunk]:
		"""Retrieve all chunks belonging to a document ordered by chunk_index, optionally scoped by organization."""
		pool = await self._ensure_pool()

		if organization_id:
			org_clause = '(organization_id = $2 OR organization_id IS NULL)' if organization_id == 'default' else 'organization_id = $2'
			query = f"""
				SELECT id, document_id, chunk_index, content, metadata, embedding,
				       organization_id, workspace_id, created_at
				FROM document_chunks
				WHERE document_id = $1 AND {org_clause}
				ORDER BY chunk_index ASC;
			"""
			async with pool.acquire() as conn:
				rows = await conn.fetch(query, document_id, organization_id)
				return [self._row_to_chunk(r) for r in rows]
		else:
			query = """
				SELECT id, document_id, chunk_index, content, metadata, embedding,
				       organization_id, workspace_id, created_at
				FROM document_chunks
				WHERE document_id = $1
				ORDER BY chunk_index ASC;
			"""
			async with pool.acquire() as conn:
				rows = await conn.fetch(query, document_id)
				return [self._row_to_chunk(r) for r in rows]

	async def create_conversation(self, conversation: Conversation) -> str:
		"""Persist a new conversation thread."""
		pool = await self._ensure_pool()
		query = """
			INSERT INTO conversations (
				id, organization_id, workspace_id, title, metadata, created_at, updated_at
			)
			VALUES ($1, $2, $3, $4, $5, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
			ON CONFLICT (id) DO UPDATE SET
				title = EXCLUDED.title,
				metadata = EXCLUDED.metadata,
				updated_at = CURRENT_TIMESTAMP
			RETURNING id;
		"""
		async with pool.acquire() as conn:
			row = await conn.fetchrow(
				query,
				conversation.id,
				conversation.organization_id,
				conversation.workspace_id,
				conversation.title,
				json.dumps(conversation.metadata),
			)
			return str(row['id'])

	async def get_conversation(self, conversation_id: str, organization_id: Optional[str] = None) -> Optional[Conversation]:
		"""Retrieve conversation metadata by ID, optionally scoped to organization."""
		pool = await self._ensure_pool()
		if organization_id:
			query = """
				SELECT id, organization_id, workspace_id, title, metadata, created_at, updated_at
				FROM conversations
				WHERE id = $1 AND organization_id = $2;
			"""
			async with pool.acquire() as conn:
				row = await conn.fetchrow(query, conversation_id, organization_id)
				return self._row_to_conversation(row) if row else None
		else:
			query = """
				SELECT id, organization_id, workspace_id, title, metadata, created_at, updated_at
				FROM conversations
				WHERE id = $1;
			"""
			async with pool.acquire() as conn:
				row = await conn.fetchrow(query, conversation_id)
				return self._row_to_conversation(row) if row else None

	async def list_conversations(
		self,
		limit: int = 50,
		offset: int = 0,
		organization_id: Optional[str] = None,
		workspace_id: Optional[str] = None,
	) -> list[Conversation]:
		"""List conversation threads ordered by updated_at descending, optionally scoped."""
		pool = await self._ensure_pool()
		if organization_id and workspace_id:
			query = """
				SELECT id, organization_id, workspace_id, title, metadata, created_at, updated_at
				FROM conversations
				WHERE organization_id = $1 AND workspace_id = $2
				ORDER BY updated_at DESC
				LIMIT $3 OFFSET $4;
			"""
			async with pool.acquire() as conn:
				rows = await conn.fetch(query, organization_id, workspace_id, limit, offset)
				return [self._row_to_conversation(r) for r in rows]
		elif organization_id:
			query = """
				SELECT id, organization_id, workspace_id, title, metadata, created_at, updated_at
				FROM conversations
				WHERE organization_id = $1
				ORDER BY updated_at DESC
				LIMIT $2 OFFSET $3;
			"""
			async with pool.acquire() as conn:
				rows = await conn.fetch(query, organization_id, limit, offset)
				return [self._row_to_conversation(r) for r in rows]
		else:
			query = """
				SELECT id, organization_id, workspace_id, title, metadata, created_at, updated_at
				FROM conversations
				ORDER BY updated_at DESC
				LIMIT $1 OFFSET $2;
			"""
			async with pool.acquire() as conn:
				rows = await conn.fetch(query, limit, offset)
				return [self._row_to_conversation(r) for r in rows]

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
		new_title = title if title is not None else conv.title
		new_meta = {**conv.metadata, **(metadata or {})} if metadata is not None else conv.metadata
		pool = await self._ensure_pool()
		query = """
			UPDATE conversations
			SET title = $2, metadata = $3, updated_at = CURRENT_TIMESTAMP
			WHERE id = $1
			RETURNING id, organization_id, workspace_id, title, metadata, created_at, updated_at;
		"""
		async with pool.acquire() as conn:
			row = await conn.fetchrow(query, conversation_id, new_title, json.dumps(new_meta))
			return self._row_to_conversation(row) if row else None

	async def delete_conversation(self, conversation_id: str, organization_id: Optional[str] = None) -> bool:
		"""Delete conversation and cascade delete its messages, optionally scoped to organization."""
		pool = await self._ensure_pool()
		if organization_id:
			query = 'DELETE FROM conversations WHERE id = $1 AND organization_id = $2 RETURNING id;'
			async with pool.acquire() as conn:
				row = await conn.fetchrow(query, conversation_id, organization_id)
				return row is not None
		else:
			query = 'DELETE FROM conversations WHERE id = $1 RETURNING id;'
			async with pool.acquire() as conn:
				row = await conn.fetchrow(query, conversation_id)
				return row is not None

	async def add_message(self, message: ConversationMessage) -> str:
		"""Persist a message in a conversation thread and touch conversation updated_at."""
		pool = await self._ensure_pool()
		sources_json = json.dumps([s.model_dump() for s in message.sources])
		meta_json = json.dumps(message.metadata)
		insert_query = """
			INSERT INTO conversation_messages (
				id, conversation_id, role, content, sources, is_grounded,
				insufficient_evidence, metadata, created_at
			)
			VALUES ($1, $2, $3, $4, $5, $6, $7, $8, CURRENT_TIMESTAMP)
			RETURNING id;
		"""
		touch_query = 'UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = $1;'
		async with pool.acquire() as conn:
			async with conn.transaction():
				row = await conn.fetchrow(
					insert_query,
					message.id,
					message.conversation_id,
					message.role,
					message.content,
					sources_json,
					message.is_grounded,
					message.insufficient_evidence,
					meta_json,
				)
				await conn.execute(touch_query, message.conversation_id)
				return str(row['id'])

	async def get_messages(self, conversation_id: str, limit: Optional[int] = None) -> list[ConversationMessage]:
		"""Retrieve chronological messages for a conversation thread."""
		pool = await self._ensure_pool()
		if limit is not None:
			query = """
				SELECT id, conversation_id, role, content, sources, is_grounded,
				       insufficient_evidence, metadata, created_at
				FROM (
					SELECT id, conversation_id, role, content, sources, is_grounded,
					       insufficient_evidence, metadata, created_at
					FROM conversation_messages
					WHERE conversation_id = $1
					ORDER BY created_at DESC
					LIMIT $2
				) sub
				ORDER BY created_at ASC;
			"""
			async with pool.acquire() as conn:
				rows = await conn.fetch(query, conversation_id, limit)
				return [self._row_to_message(r) for r in rows]
		else:
			query = """
				SELECT id, conversation_id, role, content, sources, is_grounded,
				       insufficient_evidence, metadata, created_at
				FROM conversation_messages
				WHERE conversation_id = $1
				ORDER BY created_at ASC;
			"""
			async with pool.acquire() as conn:
				rows = await conn.fetch(query, conversation_id)
				return [self._row_to_message(r) for r in rows]

	async def create_research_session(self, session: ResearchSession) -> str:
		"""Persist a research session and return its ID."""
		pool = await self._ensure_pool()
		query = """
			INSERT INTO research_sessions (
				id, organization_id, workspace_id, title, query, status,
				progress_stage, progress_percent, progress_message,
				summary, sources, internal_evidence, web_evidence, conflicts, citations,
				is_grounded, sufficient_evidence, error, metadata, created_at, started_at, completed_at, updated_at
			) VALUES (
				$1, $2, $3, $4, $5, $6,
				$7, $8, $9,
				$10, $11::jsonb, $12::jsonb, $13::jsonb, $14::jsonb, $15::jsonb,
				$16, $17, $18, $19::jsonb, $20, $21, $22, $23
			)
			ON CONFLICT (id) DO UPDATE SET
				title = EXCLUDED.title,
				status = EXCLUDED.status,
				progress_stage = EXCLUDED.progress_stage,
				progress_percent = EXCLUDED.progress_percent,
				progress_message = EXCLUDED.progress_message,
				summary = EXCLUDED.summary,
				sources = EXCLUDED.sources,
				internal_evidence = EXCLUDED.internal_evidence,
				web_evidence = EXCLUDED.web_evidence,
				conflicts = EXCLUDED.conflicts,
				citations = EXCLUDED.citations,
				is_grounded = EXCLUDED.is_grounded,
				sufficient_evidence = EXCLUDED.sufficient_evidence,
				error = EXCLUDED.error,
				metadata = EXCLUDED.metadata,
				started_at = EXCLUDED.started_at,
				completed_at = EXCLUDED.completed_at,
				updated_at = CURRENT_TIMESTAMP
			RETURNING id;
		"""
		created_dt = datetime.fromisoformat(session.created_at) if isinstance(session.created_at, str) else session.created_at
		started_dt = (
			datetime.fromisoformat(session.started_at)
			if session.started_at and isinstance(session.started_at, str)
			else session.started_at
		)
		completed_dt = (
			datetime.fromisoformat(session.completed_at)
			if session.completed_at and isinstance(session.completed_at, str)
			else session.completed_at
		)
		updated_dt = (
			datetime.fromisoformat(session.updated_at)
			if session.updated_at and isinstance(session.updated_at, str)
			else (session.updated_at or created_dt)
		)
		async with pool.acquire() as conn:
			row = await conn.fetchrow(
				query,
				session.id,
				session.organization_id,
				session.workspace_id,
				session.title,
				session.query,
				session.status,
				session.progress_stage,
				session.progress_percent,
				session.progress_message,
				session.summary,
				json.dumps(session.sources),
				json.dumps(session.internal_evidence),
				json.dumps(session.web_evidence),
				json.dumps(session.conflicts),
				json.dumps(session.citations),
				session.is_grounded,
				session.sufficient_evidence,
				session.error,
				json.dumps(session.metadata),
				created_dt,
				started_dt,
				completed_dt,
				updated_dt,
			)
			return str(row['id'])

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
		pool = await self._ensure_pool()
		async with pool.acquire() as conn:
			current_row = await conn.fetchrow(
				'SELECT status FROM research_sessions WHERE id = $1 AND organization_id = $2;',
				session_id,
				organization_id,
			)
			if not current_row:
				return None

			current_status = current_row['status']
			if status is not None and status != current_status:
				allowed_next = VALID_RESEARCH_TRANSITIONS.get(current_status, set())
				if status not in allowed_next:
					raise ValueError(
						f"Invalid state transition: Cannot transition research session '{session_id}' "
						f"from terminal status '{current_status}' to '{status}'"
					)

			updates = ['updated_at = CURRENT_TIMESTAMP']
			args: list[Any] = [session_id, organization_id]
			idx = 3

			if status is not None:
				updates.append(f'status = ${idx}')
				args.append(status)
				idx += 1

			# Late progress updates cannot mutate already-terminal sessions
			if not (current_status in TERMINAL_RESEARCH_STATUSES and status is None):
				if progress_stage is not None:
					updates.append(f'progress_stage = ${idx}')
					args.append(progress_stage)
					idx += 1
				if progress_percent is not None:
					clamped_pct = max(0, min(100, progress_percent))
					if current_status == 'in_progress' and status is None:
						updates.append(f'progress_percent = GREATEST(progress_percent, ${idx})')
					else:
						updates.append(f'progress_percent = ${idx}')
					args.append(clamped_pct)
					idx += 1
				if progress_message is not None:
					updates.append(f'progress_message = ${idx}')
					args.append(progress_message)
					idx += 1
			if summary is not None:
				updates.append(f'summary = ${idx}')
				args.append(summary)
				idx += 1
			if sources is not None:
				updates.append(f'sources = ${idx}::jsonb')
				args.append(json.dumps(sources))
				idx += 1
			if internal_evidence is not None:
				updates.append(f'internal_evidence = ${idx}::jsonb')
				args.append(json.dumps(internal_evidence))
				idx += 1
			if web_evidence is not None:
				updates.append(f'web_evidence = ${idx}::jsonb')
				args.append(json.dumps(web_evidence))
				idx += 1
			if conflicts is not None:
				updates.append(f'conflicts = ${idx}::jsonb')
				args.append(json.dumps(conflicts))
				idx += 1
			if citations is not None:
				updates.append(f'citations = ${idx}::jsonb')
				args.append(json.dumps(citations))
				idx += 1
			if is_grounded is not None:
				updates.append(f'is_grounded = ${idx}')
				args.append(is_grounded)
				idx += 1
			if sufficient_evidence is not None:
				updates.append(f'sufficient_evidence = ${idx}')
				args.append(sufficient_evidence)
				idx += 1
			if error is not None:
				updates.append(f'error = ${idx}')
				args.append(error)
				idx += 1
			if metadata is not None:
				updates.append(f'metadata = ${idx}::jsonb')
				args.append(json.dumps(metadata))
				idx += 1
			if started_at is not None:
				started_dt = datetime.fromisoformat(started_at) if isinstance(started_at, str) else started_at
				updates.append(f'started_at = ${idx}')
				args.append(started_dt)
				idx += 1
			if completed_at is not None:
				completed_dt = datetime.fromisoformat(completed_at) if isinstance(completed_at, str) else completed_at
				updates.append(f'completed_at = ${idx}')
				args.append(completed_dt)
				idx += 1

			set_clause = ', '.join(updates)
			update_sql = f'UPDATE research_sessions SET {set_clause} WHERE id = $1 AND organization_id = $2 RETURNING *;'
			updated_row = await conn.fetchrow(update_sql, *args)
			return self._row_to_research_session(updated_row) if updated_row else None

	async def get_research_session(self, session_id: str, organization_id: Optional[str] = None) -> Optional[ResearchSession]:
		"""Retrieve a research session by ID, optionally scoped to organization."""
		pool = await self._ensure_pool()
		async with pool.acquire() as conn:
			if organization_id:
				row = await conn.fetchrow(
					'SELECT * FROM research_sessions WHERE id = $1 AND organization_id = $2;',
					session_id,
					organization_id,
				)
			else:
				row = await conn.fetchrow('SELECT * FROM research_sessions WHERE id = $1;', session_id)
			return self._row_to_research_session(row) if row else None

	async def list_research_sessions(
		self,
		limit: int = 50,
		offset: int = 0,
		organization_id: Optional[str] = None,
		workspace_id: Optional[str] = None,
	) -> list[ResearchSession]:
		"""List research sessions ordered by created_at descending, optionally scoped."""
		pool = await self._ensure_pool()
		clauses = []
		args: list[Any] = []
		idx = 1
		if organization_id:
			clauses.append(f'organization_id = ${idx}')
			args.append(organization_id)
			idx += 1
		if workspace_id:
			clauses.append(f'workspace_id = ${idx}')
			args.append(workspace_id)
			idx += 1
		where_str = f'WHERE {" AND ".join(clauses)}' if clauses else ''
		sql = f'SELECT * FROM research_sessions {where_str} ORDER BY created_at DESC LIMIT ${idx} OFFSET ${idx + 1};'
		args.extend([limit, offset])
		async with pool.acquire() as conn:
			rows = await conn.fetch(sql, *args)
			return [self._row_to_research_session(r) for r in rows]

	async def delete_research_session(self, session_id: str, organization_id: Optional[str] = None) -> bool:
		"""Delete a research session, optionally scoped to organization."""
		pool = await self._ensure_pool()
		async with pool.acquire() as conn:
			if organization_id:
				row = await conn.fetchrow(
					'DELETE FROM research_sessions WHERE id = $1 AND organization_id = $2 RETURNING id;',
					session_id,
					organization_id,
				)
			else:
				row = await conn.fetchrow('DELETE FROM research_sessions WHERE id = $1 RETURNING id;', session_id)
			return row is not None

	async def similarity_search(
		self,
		query_embedding: list[float],
		top_k: int = 5,
		filter_metadata: Optional[dict[str, Any]] = None,
		organization_id: Optional[str] = None,
	) -> list[KnowledgeSearchResult]:
		"""Perform semantic similarity search against stored chunk embeddings, optionally scoped by organization."""
		if not query_embedding:
			return []

		if len(query_embedding) != self.embedding_dim:
			raise ValueError(f'Query embedding dimension mismatch: got {len(query_embedding)}, expected {self.embedding_dim}.')

		pool = await self._ensure_pool()

		doc_id_filter = filter_metadata.get('document_id') if filter_metadata else None
		org_filter = organization_id or (filter_metadata.get('organization_id') if filter_metadata else None)

		async with pool.acquire() as conn:
			if self._has_pgvector:
				if doc_id_filter and org_filter:
					query = """
						SELECT document_id, id AS chunk_id, content, chunk_index, metadata,
						       1 - (embedding <=> $1) AS score
						FROM document_chunks
						WHERE embedding IS NOT NULL AND document_id = $2 AND organization_id = $3
						ORDER BY embedding <=> $1 ASC
						LIMIT $4;
					"""
					rows = await conn.fetch(query, query_embedding, doc_id_filter, org_filter, top_k)
				elif doc_id_filter:
					query = """
						SELECT document_id, id AS chunk_id, content, chunk_index, metadata,
						       1 - (embedding <=> $1) AS score
						FROM document_chunks
						WHERE embedding IS NOT NULL AND document_id = $2
						ORDER BY embedding <=> $1 ASC
						LIMIT $3;
					"""
					rows = await conn.fetch(query, query_embedding, doc_id_filter, top_k)
				elif org_filter:
					org_clause = '(organization_id = $2 OR organization_id IS NULL)' if org_filter == 'default' else 'organization_id = $2'
					query = f"""
						SELECT document_id, id AS chunk_id, content, chunk_index, metadata,
						       1 - (embedding <=> $1) AS score
						FROM document_chunks
						WHERE embedding IS NOT NULL AND {org_clause}
						ORDER BY embedding <=> $1 ASC
						LIMIT $3;
					"""
					rows = await conn.fetch(query, query_embedding, org_filter, top_k)
				else:
					query = """
						SELECT document_id, id AS chunk_id, content, chunk_index, metadata,
						       1 - (embedding <=> $1) AS score
						FROM document_chunks
						WHERE embedding IS NOT NULL
						ORDER BY embedding <=> $1 ASC
						LIMIT $2;
					"""
					rows = await conn.fetch(query, query_embedding, top_k)

				results = []
				for r in rows:
					raw_meta = r['metadata']
					meta = json.loads(raw_meta) if isinstance(raw_meta, str) else dict(raw_meta or {})
					results.append(
						KnowledgeSearchResult(
							document_id=r['document_id'],
							chunk_id=r['chunk_id'],
							content=r['content'],
							score=float(r['score']),
							chunk_index=r['chunk_index'],
							metadata=meta,
						)
					)
				return results
			else:
				# Software cosine similarity fallback when pgvector extension is not present
				if doc_id_filter and org_filter:
					query = """
						SELECT document_id, id AS chunk_id, content, chunk_index, metadata, embedding
						FROM document_chunks
						WHERE embedding IS NOT NULL AND document_id = $1 AND organization_id = $2;
					"""
					rows = await conn.fetch(query, doc_id_filter, org_filter)
				elif doc_id_filter:
					query = """
						SELECT document_id, id AS chunk_id, content, chunk_index, metadata, embedding
						FROM document_chunks
						WHERE embedding IS NOT NULL AND document_id = $1;
					"""
					rows = await conn.fetch(query, doc_id_filter)
				elif org_filter:
					org_clause = '(organization_id = $1 OR organization_id IS NULL)' if org_filter == 'default' else 'organization_id = $1'
					query = f"""
						SELECT document_id, id AS chunk_id, content, chunk_index, metadata, embedding
						FROM document_chunks
						WHERE embedding IS NOT NULL AND {org_clause};
					"""
					rows = await conn.fetch(query, org_filter)
				else:
					query = """
						SELECT document_id, id AS chunk_id, content, chunk_index, metadata, embedding
						FROM document_chunks
						WHERE embedding IS NOT NULL;
					"""
					rows = await conn.fetch(query)

				scored: list[tuple[float, Any]] = []
				for r in rows:
					emb = r['embedding']
					if emb:
						emb_list = list(emb)
						sim = _cosine_similarity(query_embedding, emb_list)
						scored.append((sim, r))

				scored.sort(key=lambda x: x[0], reverse=True)
				top_rows = scored[:top_k]

				results = []
				for sim, r in top_rows:
					raw_meta = r['metadata']
					meta = json.loads(raw_meta) if isinstance(raw_meta, str) else dict(raw_meta or {})
					results.append(
						KnowledgeSearchResult(
							document_id=r['document_id'],
							chunk_id=r['chunk_id'],
							content=r['content'],
							score=sim,
							chunk_index=r['chunk_index'],
							metadata=meta,
						)
					)
				return results

	async def keyword_search(
		self,
		query: str,
		top_k: int = 5,
		filter_metadata: Optional[dict[str, Any]] = None,
		organization_id: Optional[str] = None,
	) -> list[KnowledgeSearchResult]:
		"""Perform lexical keyword search across stored document chunks in PostgreSQL, optionally scoped by organization."""
		if not query or not query.strip():
			return []

		pool = await self._ensure_pool()

		words = [w for w in re.findall(r'[a-zA-Z0-9_\-]+', query.lower()) if len(w) > 1]
		if not words:
			words = query.strip().split()
		if not words:
			return []

		doc_id_filter = filter_metadata.get('document_id') if filter_metadata else None
		org_filter = organization_id or (filter_metadata.get('organization_id') if filter_metadata else None)

		async with pool.acquire() as conn:
			like_patterns = [f'%{w}%' for w in words]
			if doc_id_filter and org_filter:
				query_sql = """
					SELECT document_id, id AS chunk_id, content, chunk_index, metadata
					FROM document_chunks
					WHERE document_id = $1 AND organization_id = $2 AND (content ILIKE ANY($3))
					LIMIT $4;
				"""
				rows = await conn.fetch(query_sql, doc_id_filter, org_filter, like_patterns, top_k * 3)
			elif doc_id_filter:
				query_sql = """
					SELECT document_id, id AS chunk_id, content, chunk_index, metadata
					FROM document_chunks
					WHERE document_id = $1 AND (content ILIKE ANY($2))
					LIMIT $3;
				"""
				rows = await conn.fetch(query_sql, doc_id_filter, like_patterns, top_k * 3)
			elif org_filter:
				query_sql = """
					SELECT document_id, id AS chunk_id, content, chunk_index, metadata
					FROM document_chunks
					WHERE organization_id = $1 AND (content ILIKE ANY($2))
					LIMIT $3;
				"""
				rows = await conn.fetch(query_sql, org_filter, like_patterns, top_k * 3)
			else:
				query_sql = """
					SELECT document_id, id AS chunk_id, content, chunk_index, metadata
					FROM document_chunks
					WHERE content ILIKE ANY($1)
					LIMIT $2;
				"""
				rows = await conn.fetch(query_sql, like_patterns, top_k * 3)

		query_words = set(words)
		scored: list[tuple[float, Any]] = []
		for r in rows:
			chunk_words = set(re.findall(r'[a-zA-Z0-9_\-]+', r['content'].lower()))
			overlap = len(query_words & chunk_words)
			score = overlap / len(query_words) if query_words else 1.0
			scored.append((score, r))

		scored.sort(key=lambda x: x[0], reverse=True)
		top_rows = scored[:top_k]

		results = []
		for score, r in top_rows:
			raw_meta = r['metadata']
			meta = json.loads(raw_meta) if isinstance(raw_meta, str) else dict(raw_meta or {})
			results.append(
				KnowledgeSearchResult(
					document_id=r['document_id'],
					chunk_id=r['chunk_id'],
					content=r['content'],
					score=score,
					chunk_index=r['chunk_index'],
					metadata=meta,
				)
			)
		return results

	@staticmethod
	def _row_to_document(row: asyncpg.Record) -> Document:
		raw_meta = row['metadata']
		meta = json.loads(raw_meta) if isinstance(raw_meta, str) else dict(raw_meta or {})
		meta.setdefault('doc_type', row['source_type'])
		meta.setdefault('source_filename', row['filename'])
		created = row['created_at'].isoformat() if isinstance(row['created_at'], datetime) else str(row['created_at'])
		updated = row['updated_at'].isoformat() if isinstance(row['updated_at'], datetime) else str(row['updated_at'])
		return Document(
			id=row['id'],
			source_id=row['source_id'],
			filename=row['filename'],
			source_type=row['source_type'],
			mime_type=row['mime_type'],
			content_hash=row['content_hash'],
			processing_status=row['processing_status'],
			error_message=row['error_message'],
			metadata=meta,
			organization_id=row['organization_id'],
			workspace_id=row['workspace_id'],
			created_at=created,
			updated_at=updated,
		)

	@staticmethod
	def _row_to_chunk(row: asyncpg.Record) -> DocumentChunk:
		raw_meta = row['metadata']
		meta = json.loads(raw_meta) if isinstance(raw_meta, str) else dict(raw_meta or {})
		meta.setdefault('doc_type', meta.get('source_type', 'txt'))
		emb = list(row['embedding']) if row['embedding'] is not None else None
		created = row['created_at'].isoformat() if isinstance(row['created_at'], datetime) else str(row['created_at'])
		return DocumentChunk(
			id=row['id'],
			document_id=row['document_id'],
			chunk_index=row['chunk_index'],
			content=row['content'],
			metadata=meta,
			embedding=emb,
			organization_id=row['organization_id'],
			workspace_id=row['workspace_id'],
			created_at=created,
		)

	@staticmethod
	def _row_to_conversation(row: asyncpg.Record) -> Conversation:
		raw_meta = row['metadata']
		meta = json.loads(raw_meta) if isinstance(raw_meta, str) else dict(raw_meta or {})
		created = row['created_at'].isoformat() if isinstance(row['created_at'], datetime) else str(row['created_at'])
		updated = row['updated_at'].isoformat() if isinstance(row['updated_at'], datetime) else str(row['updated_at'])
		return Conversation(
			id=row['id'],
			organization_id=row['organization_id'],
			workspace_id=row['workspace_id'],
			title=row['title'],
			metadata=meta,
			created_at=created,
			updated_at=updated,
		)

	@staticmethod
	def _row_to_message(row: asyncpg.Record) -> ConversationMessage:
		raw_meta = row['metadata']
		meta = json.loads(raw_meta) if isinstance(raw_meta, str) else dict(raw_meta or {})
		raw_sources = row['sources']
		sources_data = json.loads(raw_sources) if isinstance(raw_sources, str) else (raw_sources or [])
		sources = [SourceAttribution(**s) if isinstance(s, dict) else s for s in sources_data]
		created = row['created_at'].isoformat() if isinstance(row['created_at'], datetime) else str(row['created_at'])
		return ConversationMessage(
			id=row['id'],
			conversation_id=row['conversation_id'],
			role=row['role'],
			content=row['content'],
			sources=sources,
			is_grounded=row['is_grounded'],
			insufficient_evidence=row['insufficient_evidence'],
			metadata=meta,
			created_at=created,
		)

	@staticmethod
	def _row_to_research_session(row: asyncpg.Record) -> ResearchSession:
		raw_meta = row['metadata']
		meta = json.loads(raw_meta) if isinstance(raw_meta, str) else dict(raw_meta or {})
		raw_sources = row['sources']
		sources = json.loads(raw_sources) if isinstance(raw_sources, str) else (raw_sources or [])
		raw_internal = row['internal_evidence']
		internal_ev = json.loads(raw_internal) if isinstance(raw_internal, str) else (raw_internal or [])
		raw_web = row['web_evidence']
		web_ev = json.loads(raw_web) if isinstance(raw_web, str) else (raw_web or [])
		raw_conflicts = row['conflicts']
		conflicts = json.loads(raw_conflicts) if isinstance(raw_conflicts, str) else (raw_conflicts or [])
		raw_citations = row['citations']
		citations = json.loads(raw_citations) if isinstance(raw_citations, str) else (raw_citations or [])
		created = row['created_at'].isoformat() if isinstance(row['created_at'], datetime) else str(row['created_at'])
		started = (
			row['started_at'].isoformat()
			if ('started_at' in row and row['started_at'] and isinstance(row['started_at'], datetime))
			else (str(row['started_at']) if ('started_at' in row and row['started_at']) else None)
		)
		completed = (
			row['completed_at'].isoformat()
			if ('completed_at' in row and row['completed_at'] and isinstance(row['completed_at'], datetime))
			else (str(row['completed_at']) if ('completed_at' in row and row['completed_at']) else None)
		)
		updated = (
			row['updated_at'].isoformat()
			if ('updated_at' in row and row['updated_at'] and isinstance(row['updated_at'], datetime))
			else (str(row['updated_at']) if ('updated_at' in row and row['updated_at']) else created)
		)
		return ResearchSession(
			id=row['id'],
			organization_id=row['organization_id'],
			workspace_id=row['workspace_id'],
			title=row['title'],
			query=row['query'],
			status=row['status'],
			progress_stage=row['progress_stage'] if 'progress_stage' in row else None,
			progress_percent=row['progress_percent']
			if ('progress_percent' in row and row['progress_percent'] is not None)
			else 0,
			progress_message=row['progress_message'] if 'progress_message' in row else None,
			summary=row['summary'],
			sources=sources,
			internal_evidence=internal_ev,
			web_evidence=web_ev,
			conflicts=conflicts,
			citations=citations,
			is_grounded=row['is_grounded'],
			sufficient_evidence=row['sufficient_evidence'],
			error=row['error'],
			metadata=meta,
			created_at=created,
			started_at=started,
			completed_at=completed,
			updated_at=updated,
		)


__all__ = [
	'PgVectorStore',
	'_cosine_similarity',
]
