"""InsightBot AI Database Migrations & Schema Initialization.

Provides idempotent schema creation, migration management, and extension verification
for PostgreSQL + pgvector.
"""

import logging

import asyncpg

from insightbot.db.base import PgVectorUnavailableError

logger = logging.getLogger('insightbot.db.migrations')

SCHEMA_VERSION = 4


from typing import Any


async def check_pgvector_available(conn: Any) -> bool:
	"""Check if PostgreSQL vector extension is available in pg_available_extensions."""
	try:
		row = await conn.fetchrow("SELECT 1 FROM pg_available_extensions WHERE name = 'vector'")
		return row is not None
	except Exception as e:
		logger.warning(f'Error checking pgvector availability: {e}')
		return False


async def apply_migrations(conn: Any, embedding_dim: int = 1536, require_native_pgvector: bool = False) -> bool:
	"""Apply database migrations idempotently.

	Returns True if pgvector extension is active, False otherwise.
	Raises PgVectorUnavailableError if require_native_pgvector is True but pgvector is unavailable.
	"""
	async with conn.transaction():
		# 1. Create schema_migrations table
		await conn.execute("""
			CREATE TABLE IF NOT EXISTS schema_migrations (
				version INT PRIMARY KEY,
				applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
				description TEXT
			);
		""")

		# 2. Check and enable vector extension if available
		vector_available = await check_pgvector_available(conn)
		has_vector_extension = False

		if vector_available:
			try:
				await conn.execute('CREATE EXTENSION IF NOT EXISTS vector;')
				has_vector_extension = True
				logger.info('Successfully enabled PostgreSQL vector extension.')
			except Exception as e:
				if require_native_pgvector:
					raise PgVectorUnavailableError(
						f"PostgreSQL 'vector' extension is in catalog but CREATE EXTENSION failed: {e}"
					) from e
				logger.warning(
					f'pgvector extension available in catalog but failed to CREATE EXTENSION: {e}. Falling back to array storage.'
				)
		else:
			if require_native_pgvector:
				raise PgVectorUnavailableError(
					"Native PostgreSQL 'vector' extension (pgvector) is strictly required by configuration, "
					"but 'vector.control' was not found in the PostgreSQL extensions catalog. "
					'To resolve: install pgvector for your PostgreSQL version or configure '
					"INSIGHTBOT_VECTOR_STORE_MODE='auto' (or 'array') for development compatibility mode."
				)
			logger.warning(
				"PostgreSQL 'vector' extension is NOT installed in PostgreSQL extensions catalog. "
				'Vector embeddings will be stored as float8[] arrays. '
				'To enable hardware-accelerated vector indexing, install pgvector.'
			)

		# 3. Create knowledge_sources table
		await conn.execute("""
			CREATE TABLE IF NOT EXISTS knowledge_sources (
				id VARCHAR(64) PRIMARY KEY,
				name VARCHAR(255) NOT NULL,
				source_type VARCHAR(64) NOT NULL DEFAULT 'local_upload',
				description TEXT,
				config JSONB DEFAULT '{}'::jsonb,
				metadata JSONB DEFAULT '{}'::jsonb,
				organization_id VARCHAR(64),
				workspace_id VARCHAR(64),
				created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
				updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
			);
		""")

		# 4. Create documents table
		await conn.execute("""
			CREATE TABLE IF NOT EXISTS documents (
				id VARCHAR(64) PRIMARY KEY,
				source_id VARCHAR(64) REFERENCES knowledge_sources(id) ON DELETE SET NULL,
				filename VARCHAR(512) NOT NULL,
				source_type VARCHAR(64) NOT NULL,
				mime_type VARCHAR(128) NOT NULL,
				content_hash VARCHAR(64),
				processing_status VARCHAR(32) NOT NULL DEFAULT 'pending',
				error_message TEXT,
				metadata JSONB DEFAULT '{}'::jsonb,
				organization_id VARCHAR(64),
				workspace_id VARCHAR(64),
				created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
				updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
			);
		""")

		await conn.execute('CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash);')
		await conn.execute('CREATE INDEX IF NOT EXISTS idx_documents_source_id ON documents(source_id);')
		await conn.execute('CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(processing_status);')
		await conn.execute('CREATE INDEX IF NOT EXISTS idx_documents_org_id ON documents(organization_id);')

		# 5. Create document_chunks table
		if has_vector_extension:
			await conn.execute(f"""
				CREATE TABLE IF NOT EXISTS document_chunks (
					id VARCHAR(64) PRIMARY KEY,
					document_id VARCHAR(64) NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
					chunk_index INT NOT NULL,
					content TEXT NOT NULL,
					metadata JSONB DEFAULT '{{}}'::jsonb,
					embedding vector({embedding_dim}),
					organization_id VARCHAR(64),
					workspace_id VARCHAR(64),
					created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
				);
			""")
			# HNSW index for fast approximate nearest neighbor search
			try:
				await conn.execute("""
					CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw 
					ON document_chunks USING hnsw (embedding vector_cosine_ops);
				""")
			except Exception as e:
				logger.debug(f'HNSW index creation note (table may be empty or unindexed yet): {e}')
		else:
			await conn.execute("""
				CREATE TABLE IF NOT EXISTS document_chunks (
					id VARCHAR(64) PRIMARY KEY,
					document_id VARCHAR(64) NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
					chunk_index INT NOT NULL,
					content TEXT NOT NULL,
					metadata JSONB DEFAULT '{}'::jsonb,
					embedding float8[],
					organization_id VARCHAR(64),
					workspace_id VARCHAR(64),
					created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
				);
			""")

		await conn.execute('CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON document_chunks(document_id);')
		await conn.execute('CREATE INDEX IF NOT EXISTS idx_chunks_doc_chunk_idx ON document_chunks(document_id, chunk_index);')
		await conn.execute('CREATE INDEX IF NOT EXISTS idx_chunks_org_id ON document_chunks(organization_id);')

		# 6. Create conversations table
		await conn.execute("""
			CREATE TABLE IF NOT EXISTS conversations (
				id VARCHAR(64) PRIMARY KEY,
				organization_id VARCHAR(64) NOT NULL,
				workspace_id VARCHAR(64),
				title VARCHAR(255) NOT NULL DEFAULT 'New Conversation',
				metadata JSONB DEFAULT '{}'::jsonb,
				created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
				updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
			);
		""")
		await conn.execute('CREATE INDEX IF NOT EXISTS idx_conversations_org_id ON conversations(organization_id);')
		await conn.execute('CREATE INDEX IF NOT EXISTS idx_conversations_updated_at ON conversations(updated_at DESC);')

		# 7. Create conversation_messages table
		await conn.execute("""
			CREATE TABLE IF NOT EXISTS conversation_messages (
				id VARCHAR(64) PRIMARY KEY,
				conversation_id VARCHAR(64) NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
				role VARCHAR(32) NOT NULL,
				content TEXT NOT NULL,
				sources JSONB DEFAULT '[]'::jsonb,
				is_grounded BOOLEAN NOT NULL DEFAULT TRUE,
				insufficient_evidence BOOLEAN NOT NULL DEFAULT FALSE,
				metadata JSONB DEFAULT '{}'::jsonb,
				created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
			);
		""")
		await conn.execute('CREATE INDEX IF NOT EXISTS idx_conv_messages_conv_id ON conversation_messages(conversation_id);')
		await conn.execute('CREATE INDEX IF NOT EXISTS idx_conv_messages_created_at ON conversation_messages(created_at ASC);')

		# 8. Create research_sessions table
		await conn.execute("""
			CREATE TABLE IF NOT EXISTS research_sessions (
				id VARCHAR(64) PRIMARY KEY,
				organization_id VARCHAR(64) NOT NULL,
				workspace_id VARCHAR(64),
				title VARCHAR(255) NOT NULL,
				query TEXT NOT NULL,
				status VARCHAR(32) NOT NULL DEFAULT 'completed',
				progress_stage VARCHAR(64),
				progress_percent INT DEFAULT 0,
				progress_message TEXT,
				summary TEXT NOT NULL DEFAULT '',
				sources JSONB DEFAULT '[]'::jsonb,
				internal_evidence JSONB DEFAULT '[]'::jsonb,
				web_evidence JSONB DEFAULT '[]'::jsonb,
				conflicts JSONB DEFAULT '[]'::jsonb,
				citations JSONB DEFAULT '[]'::jsonb,
				is_grounded BOOLEAN NOT NULL DEFAULT TRUE,
				sufficient_evidence BOOLEAN NOT NULL DEFAULT TRUE,
				error TEXT,
				metadata JSONB DEFAULT '{}'::jsonb,
				created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
				started_at TIMESTAMPTZ,
				completed_at TIMESTAMPTZ,
				updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
			);
		""")
		await conn.execute('CREATE INDEX IF NOT EXISTS idx_research_sessions_org_id ON research_sessions(organization_id);')
		await conn.execute('CREATE INDEX IF NOT EXISTS idx_research_sessions_workspace_id ON research_sessions(workspace_id);')
		await conn.execute('CREATE INDEX IF NOT EXISTS idx_research_sessions_status ON research_sessions(status);')
		await conn.execute('CREATE INDEX IF NOT EXISTS idx_research_sessions_created_at ON research_sessions(created_at DESC);')

		# 9. Schema Version 4: Ensure progress columns exist on pre-existing research_sessions tables
		await conn.execute("""
			ALTER TABLE research_sessions ADD COLUMN IF NOT EXISTS progress_stage VARCHAR(64);
			ALTER TABLE research_sessions ADD COLUMN IF NOT EXISTS progress_percent INT DEFAULT 0;
			ALTER TABLE research_sessions ADD COLUMN IF NOT EXISTS progress_message TEXT;
			ALTER TABLE research_sessions ADD COLUMN IF NOT EXISTS started_at TIMESTAMPTZ;
			ALTER TABLE research_sessions ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP;
		""")

		# 10. Record migrations
		await conn.execute(
			"""
			INSERT INTO schema_migrations (version, description)
			VALUES ($1, $2)
			ON CONFLICT (version) DO NOTHING;
		""",
			1,
			f'Phase 1 schema with vector_dim={embedding_dim}, pgvector={has_vector_extension}',
		)
		await conn.execute(
			"""
			INSERT INTO schema_migrations (version, description)
			VALUES ($1, $2)
			ON CONFLICT (version) DO NOTHING;
		""",
			2,
			'Phase 14B conversations and conversation_messages schema',
		)
		await conn.execute(
			"""
			INSERT INTO schema_migrations (version, description)
			VALUES ($1, $2)
			ON CONFLICT (version) DO NOTHING;
		""",
			3,
			'Phase 14C research_sessions schema',
		)
		await conn.execute(
			"""
			INSERT INTO schema_migrations (version, description)
			VALUES ($1, $2)
			ON CONFLICT (version) DO NOTHING;
		""",
			4,
			'Phase 14D research job orchestration, progress stages, and timestamps',
		)

	return has_vector_extension


async def init_database(
	database_url: str,
	embedding_dim: int = 1536,
	require_native_pgvector: bool = False,
) -> bool:
	"""Connect to PostgreSQL and apply all migrations."""
	conn = await asyncpg.connect(database_url)
	try:
		return await apply_migrations(
			conn,
			embedding_dim=embedding_dim,
			require_native_pgvector=require_native_pgvector,
		)
	finally:
		await conn.close()
