"""InsightBot AI Knowledge & Database Models (Pydantic v2).

Defines core schemas for documents, chunks, knowledge sources, and search results.
Designed for clean future multi-tenant expansion with organization_id and workspace_id.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


def _utc_now_iso() -> str:
	return datetime.now(timezone.utc).isoformat()


def _generate_id(prefix: str = 'id') -> str:
	return f'{prefix}-{uuid.uuid4().hex[:12]}'


class KnowledgeSource(BaseModel):
	"""Source integration or container of documents (e.g., local upload, team folder)."""

	model_config = ConfigDict(extra='ignore')

	id: str = Field(default_factory=lambda: _generate_id('src'), description='Unique source identifier')
	name: str = Field(..., min_length=1, description='Human-readable source name')
	source_type: str = Field(default='local_upload', description='Type of source, e.g. local_upload, team_vault')
	description: Optional[str] = Field(default=None, description='Optional description of the source')
	config: dict[str, Any] = Field(default_factory=dict, description='Configuration parameters for the source')
	metadata: dict[str, Any] = Field(default_factory=dict, description='Arbitrary metadata dictionary')
	organization_id: Optional[str] = Field(default=None, description='Future multi-tenancy organization ID')
	workspace_id: Optional[str] = Field(default=None, description='Future multi-tenancy workspace ID')
	created_at: str = Field(default_factory=_utc_now_iso, description='ISO timestamp of creation')
	updated_at: str = Field(default_factory=_utc_now_iso, description='ISO timestamp of last update')


class Document(BaseModel):
	"""A parsed and ingested knowledge document."""

	model_config = ConfigDict(extra='ignore')

	id: str = Field(default_factory=lambda: _generate_id('doc'), description='Unique document identifier')
	source_id: Optional[str] = Field(default=None, description='Source container identifier if applicable')
	filename: str = Field(..., min_length=1, description='Original filename of the document')
	source_type: str = Field(..., description='Document format/type: pdf, docx, txt, markdown')
	mime_type: str = Field(..., description='MIME type of the source document')
	content_hash: Optional[str] = Field(default=None, description='SHA256 hash of source document content')
	processing_status: str = Field(default='pending', description='Status: pending, processing, completed, failed')
	error_message: Optional[str] = Field(default=None, description='Error message if processing failed')
	metadata: dict[str, Any] = Field(default_factory=dict, description='Document metadata such as author, title, page_count')
	organization_id: Optional[str] = Field(default=None, description='Future multi-tenancy organization ID')
	workspace_id: Optional[str] = Field(default=None, description='Future multi-tenancy workspace ID')
	created_at: str = Field(default_factory=_utc_now_iso, description='ISO timestamp of ingestion')
	updated_at: str = Field(default_factory=_utc_now_iso, description='ISO timestamp of last update')


class DocumentChunk(BaseModel):
	"""A deterministic chunk extracted from an ingested document."""

	model_config = ConfigDict(extra='ignore')

	id: str = Field(default_factory=lambda: _generate_id('chk'), description='Unique chunk identifier')
	document_id: str = Field(..., description='Parent document identifier')
	content: str = Field(..., min_length=1, description='Textual content of the chunk')
	chunk_index: int = Field(..., ge=0, description='Sequential 0-indexed position within the document')
	metadata: dict[str, Any] = Field(
		default_factory=dict,
		description='Chunk metadata: source_filename, section_heading, page_number, start_char, end_char',
	)
	embedding: Optional[list[float]] = Field(default=None, description='Vector embedding representation')
	organization_id: Optional[str] = Field(default=None, description='Future multi-tenancy organization ID')
	workspace_id: Optional[str] = Field(default=None, description='Future multi-tenancy workspace ID')
	created_at: str = Field(default_factory=_utc_now_iso, description='ISO timestamp of creation')


class KnowledgeSearchResult(BaseModel):
	"""Result item returned by semantic similarity vector search."""

	model_config = ConfigDict(extra='ignore')

	document_id: str = Field(..., description='Associated document ID')
	chunk_id: str = Field(..., description='Matching chunk ID')
	content: str = Field(..., description='Chunk content text')
	score: float = Field(..., description='Cosine similarity score (higher is more similar)')
	chunk_index: int = Field(default=0, ge=0, description='Chunk sequential index')
	metadata: dict[str, Any] = Field(default_factory=dict, description='Preserved chunk metadata')


class SourceAttribution(BaseModel):
	"""Source citation tracking supporting a grounded answer."""

	model_config = ConfigDict(extra='ignore')

	source_filename: str = Field(..., description='Filename of supporting document')
	document_id: str = Field(..., description='Supporting document ID')
	chunk_id: str = Field(..., description='Supporting chunk ID')
	chunk_index: int = Field(default=0, description='Supporting chunk index')
	page_number: Optional[int] = Field(default=None, description='Page number if applicable')
	section_heading: Optional[str] = Field(default=None, description='Section heading if applicable')
	snippet: str = Field(..., description='Relevant text excerpt supporting the answer')
	score: float = Field(default=1.0, description='Similarity or relevance score')


class KnowledgeAnswer(BaseModel):
	"""Structured grounded answer produced by KnowledgeAgent."""

	model_config = ConfigDict(extra='ignore')

	question: str = Field(..., description='Original user question')
	answer: str = Field(..., description='Synthesized grounded answer')
	sources: list[SourceAttribution] = Field(default_factory=list, description='Attributed evidence sources')
	is_grounded: bool = Field(default=True, description='Whether the answer is grounded in retrieved documents')
	sufficient_context: bool = Field(
		default=True,
		description='Whether sufficient context was found in the private knowledge base',
	)
	insufficient_evidence: bool = Field(
		default=False,
		description='True if evidence was insufficient to answer the question',
	)
	created_at: str = Field(default_factory=_utc_now_iso, description='ISO timestamp')


class TenantContext(BaseModel):
	"""Server-side derived tenant identity and authorization context."""

	model_config = ConfigDict(extra='ignore')

	organization_id: str = Field(..., min_length=1, description='Authenticated organization / tenant ID')
	workspace_id: Optional[str] = Field(default=None, description='Optional workspace ID within organization')
	api_key_id: Optional[str] = Field(default=None, description='Identifier or fingerprint of API key')
	is_admin: bool = Field(default=False, description='Whether context has administrative privilege')


class Conversation(BaseModel):
	"""A persistent multi-turn research conversation thread."""

	model_config = ConfigDict(extra='ignore')

	id: str = Field(default_factory=lambda: _generate_id('conv'), description='Unique conversation identifier')
	organization_id: str = Field(..., min_length=1, description='Owning organization / tenant ID')
	workspace_id: Optional[str] = Field(default=None, description='Optional workspace ID within organization')
	title: str = Field(default='New Conversation', min_length=1, max_length=255, description='Conversation title')
	metadata: dict[str, Any] = Field(default_factory=dict, description='Arbitrary thread metadata')
	created_at: str = Field(default_factory=_utc_now_iso, description='ISO timestamp of creation')
	updated_at: str = Field(default_factory=_utc_now_iso, description='ISO timestamp of last update')


class ConversationMessage(BaseModel):
	"""A message within a persistent conversation thread."""

	model_config = ConfigDict(extra='ignore')

	id: str = Field(default_factory=lambda: _generate_id('msg'), description='Unique message identifier')
	conversation_id: str = Field(..., min_length=1, description='Parent conversation ID')
	role: str = Field(..., description="Role of sender: 'user', 'assistant', or 'system'")
	content: str = Field(..., min_length=1, description='Message body text')
	sources: list[SourceAttribution] = Field(default_factory=list, description='Attributed sources for assistant response')
	is_grounded: bool = Field(default=True, description='Whether response is strictly grounded in evidence')
	insufficient_evidence: bool = Field(default=False, description='Whether response had insufficient evidence')
	metadata: dict[str, Any] = Field(default_factory=dict, description='Additional structured metadata')
	created_at: str = Field(default_factory=_utc_now_iso, description='ISO timestamp of message creation')


class ConversationWithMessages(Conversation):
	"""Conversation model enriched with its message history."""

	messages: list[ConversationMessage] = Field(default_factory=list, description='Chronological message list')


RESEARCH_STATUS_QUEUED = 'queued'
RESEARCH_STATUS_IN_PROGRESS = 'in_progress'
RESEARCH_STATUS_COMPLETED = 'completed'
RESEARCH_STATUS_COMPLETED_UNVERIFIED = 'completed_unverified'
RESEARCH_STATUS_FAILED = 'failed'
RESEARCH_STATUS_PROVIDER_UNAVAILABLE = 'provider_unavailable'
RESEARCH_STATUS_CANCELLED = 'cancelled'
RESEARCH_STATUS_INTERRUPTED = 'interrupted'

TERMINAL_RESEARCH_STATUSES: set[str] = {
	RESEARCH_STATUS_COMPLETED,
	RESEARCH_STATUS_COMPLETED_UNVERIFIED,
	RESEARCH_STATUS_FAILED,
	RESEARCH_STATUS_PROVIDER_UNAVAILABLE,
	RESEARCH_STATUS_CANCELLED,
	RESEARCH_STATUS_INTERRUPTED,
}

VALID_RESEARCH_TRANSITIONS: dict[str, set[str]] = {
	RESEARCH_STATUS_QUEUED: {
		RESEARCH_STATUS_IN_PROGRESS,
		RESEARCH_STATUS_FAILED,
		RESEARCH_STATUS_CANCELLED,
		RESEARCH_STATUS_PROVIDER_UNAVAILABLE,
		RESEARCH_STATUS_INTERRUPTED,
	},
	RESEARCH_STATUS_IN_PROGRESS: {
		RESEARCH_STATUS_IN_PROGRESS,
		RESEARCH_STATUS_COMPLETED,
		RESEARCH_STATUS_COMPLETED_UNVERIFIED,
		RESEARCH_STATUS_FAILED,
		RESEARCH_STATUS_PROVIDER_UNAVAILABLE,
		RESEARCH_STATUS_CANCELLED,
		RESEARCH_STATUS_INTERRUPTED,
	},
	RESEARCH_STATUS_COMPLETED: set(),
	RESEARCH_STATUS_COMPLETED_UNVERIFIED: set(),
	RESEARCH_STATUS_FAILED: set(),
	RESEARCH_STATUS_PROVIDER_UNAVAILABLE: set(),
	RESEARCH_STATUS_CANCELLED: set(),
	RESEARCH_STATUS_INTERRUPTED: set(),
}


class ResearchSession(BaseModel):
	"""A persistent autonomous research session and evidence record."""

	model_config = ConfigDict(extra='ignore')

	id: str = Field(default_factory=lambda: _generate_id('res'), description='Unique research session identifier')
	organization_id: str = Field(..., min_length=1, description='Owning organization / tenant ID')
	workspace_id: Optional[str] = Field(default=None, description='Optional workspace ID within organization')
	title: str = Field(default='Untitled Research', min_length=1, max_length=255, description='Research title')
	query: str = Field(..., min_length=1, description='Original research task or query')
	status: str = Field(
		default='completed',
		description="Session status: 'queued', 'in_progress', 'completed', 'completed_unverified', 'failed', 'provider_unavailable', 'cancelled', 'interrupted'",
	)
	progress_stage: Optional[str] = Field(default=None, description='Current progress stage')
	progress_percent: int = Field(default=0, ge=0, le=100, description='Progress percentage (0-100)')
	progress_message: Optional[str] = Field(default=None, description='Progress milestone description')
	summary: str = Field(default='', description='Synthesized report summary or answer')
	sources: list[str] = Field(default_factory=list, description='List of source URIs or URLs')
	internal_evidence: list[dict[str, Any]] = Field(default_factory=list, description='Normalized internal evidence items')
	web_evidence: list[dict[str, Any]] = Field(default_factory=list, description='Normalized web evidence items')
	conflicts: list[dict[str, Any]] = Field(default_factory=list, description='Detected cross-source conflict items')
	citations: list[dict[str, Any]] = Field(default_factory=list, description='Verified citation attributions')
	is_grounded: bool = Field(default=True, description='Whether synthesis is strictly grounded')
	sufficient_evidence: bool = Field(default=True, description='Whether sufficient evidence was found')
	error: Optional[str] = Field(default=None, description='Error message if research failed')
	metadata: dict[str, Any] = Field(default_factory=dict, description='Additional execution metadata')
	created_at: str = Field(default_factory=_utc_now_iso, description='ISO timestamp of session creation')
	started_at: Optional[str] = Field(default=None, description='ISO timestamp of execution start')
	completed_at: Optional[str] = Field(default=None, description='ISO timestamp of completion')
	updated_at: str = Field(default_factory=_utc_now_iso, description='ISO timestamp of last update')


# Backward compatibility alias
ResearchRecord = ResearchSession


class ResearchJobStatusResponse(BaseModel):
	"""Response payload for polling research job status and progress."""

	model_config = ConfigDict(extra='ignore')

	research_id: str = Field(..., description='Unique research session ID')
	organization_id: str = Field(..., description='Owning tenant organization ID')
	workspace_id: Optional[str] = Field(default=None, description='Workspace ID within organization')
	status: str = Field(..., description='Current lifecycle status')
	progress_stage: Optional[str] = Field(default=None, description='Current execution progress stage')
	progress_percent: int = Field(default=0, ge=0, le=100, description='Progress percentage (0-100)')
	progress_message: Optional[str] = Field(default=None, description='Progress milestone message')
	created_at: str = Field(..., description='ISO creation timestamp')
	started_at: Optional[str] = Field(default=None, description='ISO execution start timestamp')
	completed_at: Optional[str] = Field(default=None, description='ISO completion timestamp')
	updated_at: str = Field(..., description='ISO update timestamp')
	error: Optional[str] = Field(default=None, description='Error detail if failed')
	is_terminal: bool = Field(default=False, description='Whether job reached a terminal state')


__all__ = [
	'KnowledgeSource',
	'Document',
	'DocumentChunk',
	'KnowledgeSearchResult',
	'SourceAttribution',
	'KnowledgeAnswer',
	'TenantContext',
	'Conversation',
	'ConversationMessage',
	'ConversationWithMessages',
	'ResearchSession',
	'ResearchRecord',
	'ResearchJobStatusResponse',
	'RESEARCH_STATUS_QUEUED',
	'RESEARCH_STATUS_IN_PROGRESS',
	'RESEARCH_STATUS_COMPLETED',
	'RESEARCH_STATUS_COMPLETED_UNVERIFIED',
	'RESEARCH_STATUS_FAILED',
	'RESEARCH_STATUS_PROVIDER_UNAVAILABLE',
	'RESEARCH_STATUS_CANCELLED',
	'RESEARCH_STATUS_INTERRUPTED',
	'TERMINAL_RESEARCH_STATUSES',
	'VALID_RESEARCH_TRANSITIONS',
]
