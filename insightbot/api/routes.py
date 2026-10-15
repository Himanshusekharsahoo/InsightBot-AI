"""InsightBot AI REST API module.

Provides structured FastAPI endpoints for knowledge ingestion, semantic search,
grounded question-answering, and system health status.
"""

import asyncio
import base64
import json
import logging
from contextlib import asynccontextmanager
from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from insightbot.config.settings import get_settings
from insightbot.db.models import (
	RESEARCH_STATUS_CANCELLED,
	RESEARCH_STATUS_COMPLETED,
	RESEARCH_STATUS_COMPLETED_UNVERIFIED,
	RESEARCH_STATUS_FAILED,
	RESEARCH_STATUS_INTERRUPTED,
	RESEARCH_STATUS_PROVIDER_UNAVAILABLE,
	RESEARCH_STATUS_QUEUED,
	TERMINAL_RESEARCH_STATUSES,
	Conversation,
	ConversationMessage,
	ConversationWithMessages,
	Document,
	DocumentChunk,
	KnowledgeAnswer,
	KnowledgeSearchResult,
	ResearchJobStatusResponse,
	ResearchSession,
	TenantContext,
)
from insightbot.knowledge.parsers import DocumentParsingError, UnsupportedDocumentError
from insightbot.knowledge.store import KnowledgeStore
from insightbot.knowledge.validation import SecurityValidationError
from insightbot.orchestrator.export import format_research_json, format_research_markdown
from insightbot.orchestrator.job_manager import ResearchJobManager, get_job_manager
from insightbot.orchestrator.knowledge_agent import KnowledgeAgent
from insightbot.orchestrator.research_engine import ResearchEngine
from insightbot.security.policy import (
	TENANT_ID_PATTERN,
	SensitiveDataSanitizer,
	get_ai_semaphore,
	validate_target_domains,
)

logger = logging.getLogger('insightbot.api')

api_key_header = APIKeyHeader(name='X-API-Key', auto_error=False)


async def get_current_tenant(
	request: Request,
	header_key: Optional[str] = Depends(api_key_header),
) -> TenantContext:
	"""Derive authenticated TenantContext strictly from server-side credential verification."""
	settings = get_settings()
	auth_header = request.headers.get('Authorization', '')
	bearer_token = auth_header[7:].strip() if auth_header.startswith('Bearer ') else None
	provided_key = header_key or bearer_token

	# 1. Staging or production: fail closed immediately if no credential provided
	if settings.environment in ('production', 'prod', 'staging', 'stage'):
		if not provided_key:
			raise HTTPException(
				status_code=status.HTTP_401_UNAUTHORIZED,
				detail='Unauthorized: Missing API key.',
				headers={'WWW-Authenticate': 'Bearer'},
			)

	# 2. Resolve credentials against server-side configuration
	if provided_key:
		resolved = settings.resolve_tenant_key(provided_key)
		if resolved is None:
			raise HTTPException(
				status_code=status.HTTP_401_UNAUTHORIZED,
				detail='Unauthorized: Invalid API key.',
				headers={'WWW-Authenticate': 'Bearer'},
			)
		org_id, is_admin = resolved
	else:
		# Development or test fallback
		if settings.api_key or settings.tenant_api_keys:
			raise HTTPException(
				status_code=status.HTTP_401_UNAUTHORIZED,
				detail='Unauthorized: Missing API key.',
				headers={'WWW-Authenticate': 'Bearer'},
			)
		org_id, is_admin = (settings.default_organization_id, False)

	# 3. Client header check: never trust client header to switch or forge another tenant
	client_org_header = request.headers.get('X-Organization-Id')
	if client_org_header:
		if not TENANT_ID_PATTERN.match(client_org_header):
			raise HTTPException(
				status_code=status.HTTP_400_BAD_REQUEST,
				detail="Invalid X-Organization-Id header format. Must be alphanumeric, underscores, or dashes (1-64 chars).",
			)
		if client_org_header != org_id and not is_admin:
			raise HTTPException(
				status_code=status.HTTP_403_FORBIDDEN,
				detail=f"Forbidden: Client header organization_id '{client_org_header}' does not match authenticated tenant '{org_id}'.",
			)

	workspace_id = request.headers.get('X-Workspace-Id')
	if workspace_id and not TENANT_ID_PATTERN.match(workspace_id):
		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST,
			detail="Invalid X-Workspace-Id header format. Must be alphanumeric, underscores, or dashes (1-64 chars).",
		)

	effective_org = client_org_header if (is_admin and client_org_header) else org_id

	return TenantContext(
		organization_id=effective_org,
		workspace_id=workspace_id,
		is_admin=is_admin,
	)


async def verify_api_key(request: Request, header_key: Optional[str] = Depends(api_key_header)) -> None:
	"""Legacy compatibility auth dependency delegating to get_current_tenant."""
	await get_current_tenant(request, header_key)


# -----------------------------------------------------------------------------
# Request & Response Schemas
# -----------------------------------------------------------------------------


class ResearchAPIRequest(BaseModel):
	"""API payload for initiating a research task."""

	task: str = Field(..., min_length=1, description='Research question or task for InsightBot')
	max_steps: int = Field(default=25, ge=1, le=100, description='Maximum research steps')
	target_domains: Optional[list[str]] = Field(default=None, description='Optional domain whitelist')
	sync: bool = Field(default=False, description='If True, execute synchronously and wait for completion')


class ResearchAPIResponse(BaseModel):
	"""API response containing research findings."""

	status: str = Field(default='completed')
	task: str
	summary: str
	sources: list[str] = Field(default_factory=list)
	evidence_count: int = 0
	doc_id: Optional[str] = None
	research_id: Optional[str] = Field(default=None, description='Persisted research session identifier')
	web_status: Optional[dict[str, Any]] = Field(default=None, description='Web research execution status')
	conflicts: list[dict[str, Any]] = Field(default_factory=list, description='Detected cross-source conflicts')
	citations: list[dict[str, Any]] = Field(default_factory=list, description='Verified source citations')
	progress_stage: Optional[str] = Field(default=None, description='Current execution progress stage')
	progress_percent: int = Field(default=0, description='Progress percentage (0-100)')


# Compatibility aliases
ResearchRequest = ResearchAPIRequest
ResearchResponse = ResearchAPIResponse


class KnowledgeSearchRequest(BaseModel):
	"""API payload for semantic knowledge search."""

	query: str = Field(..., min_length=1, description='Search query string')
	top_k: int = Field(default=5, ge=1, le=50, description='Number of results to retrieve')
	document_id: Optional[str] = Field(default=None, description='Optional document filter')
	hybrid: bool = Field(default=False, description='Enable hybrid dense vector + lexical keyword search')


class KnowledgeAskRequest(BaseModel):
	"""API payload for asking a question answered strictly from private knowledge."""

	question: str = Field(..., min_length=1, description='Question to answer using private knowledge base')
	top_k: int = Field(default=5, ge=1, le=20, description='Number of evidence chunks to retrieve')


class DocumentUploadPayload(BaseModel):
	"""JSON payload for document ingestion with base64-encoded content."""

	filename: str = Field(..., min_length=1, description='Original filename with extension')
	content_base64: str = Field(..., min_length=1, description='Base64-encoded file content')
	metadata: dict[str, Any] = Field(default_factory=dict, description='Optional document metadata')


class HealthResponse(BaseModel):
	"""Health check status response."""

	status: str = 'ok'
	service: str = 'InsightBot AI'
	version: str = '1.0.0'
	database: str = 'unknown'
	storage_mode: str = 'unknown'
	pgvector_enabled: bool = False


class ConversationCreateRequest(BaseModel):
	"""Payload for creating a new conversation thread."""

	title: Optional[str] = Field(
		default='New Conversation',
		min_length=1,
		max_length=255,
		description='Initial conversation title',
	)
	workspace_id: Optional[str] = Field(
		default=None,
		pattern=r'^[a-zA-Z0-9_\-]{1,64}$',
		description='Optional workspace ID within organization',
	)
	metadata: dict[str, Any] = Field(default_factory=dict, description='Arbitrary conversation metadata')


class ConversationUpdateRequest(BaseModel):
	"""Payload for updating conversation title or metadata."""

	title: Optional[str] = Field(default=None, min_length=1, max_length=255, description='Updated conversation title')
	metadata: Optional[dict[str, Any]] = Field(default=None, description='Updated metadata dictionary')


class ConversationMessageCreateRequest(BaseModel):
	"""Payload for posting a user message to a conversation thread."""

	content: str = Field(..., min_length=1, description='User message content')
	top_k: int = Field(default=5, ge=1, le=20, description='Number of candidate evidence chunks to retrieve')
	similarity_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description='Minimum similarity threshold for retrieval')
	max_history_turns: int = Field(default=6, ge=0, le=20, description='Maximum recent turns of conversation history to supply')


# -----------------------------------------------------------------------------
# Dependency Providers
# -----------------------------------------------------------------------------

_knowledge_store: Optional[KnowledgeStore] = None
_knowledge_agent: Optional[KnowledgeAgent] = None


def get_store() -> KnowledgeStore:
	global _knowledge_store
	if _knowledge_store is None:
		_knowledge_store = KnowledgeStore()
	return _knowledge_store


def get_agent() -> KnowledgeAgent:
	global _knowledge_agent
	if _knowledge_agent is None:
		_knowledge_agent = KnowledgeAgent(knowledge_store=get_store())
	return _knowledge_agent


_browser_agent: Optional[Any] = None


def get_browser_agent() -> Optional[Any]:
	global _browser_agent
	if _browser_agent is None:
		from insightbot.engine.browser.agent import create_browser_agent

		_browser_agent = create_browser_agent()
	return _browser_agent


_research_engine: Optional[ResearchEngine] = None


def get_research_engine() -> ResearchEngine:
	global _research_engine
	if _research_engine is None:
		_research_engine = ResearchEngine(
			knowledge_store=get_store(),
			browser_agent=get_browser_agent(),
		)
	return _research_engine


StoreDep = Annotated[KnowledgeStore, Depends(get_store)]
AgentDep = Annotated[KnowledgeAgent, Depends(get_agent)]
ResearchEngineDep = Annotated[ResearchEngine, Depends(get_research_engine)]
JobManagerDep = Annotated[ResearchJobManager, Depends(get_job_manager)]
TenantDep = Annotated[TenantContext, Depends(get_current_tenant)]
AuthDep = Annotated[None, Depends(verify_api_key)]

# -----------------------------------------------------------------------------
# Router Definition
# -----------------------------------------------------------------------------


router = APIRouter()


@router.get('/health', tags=['System'])
async def health_check(store: StoreDep) -> HealthResponse:
	"""Check system status, database connectivity, and vector storage mode."""
	db_status = 'unavailable'
	storage_mode = 'unknown'
	has_vector = False
	overall_status = 'ok'
	try:
		await store.initialize()
		db_status = 'connected'
		storage_mode = getattr(store.vector_store, 'storage_mode', 'unknown')
		has_vector = (storage_mode == 'native_pgvector') or getattr(store.vector_store, '_has_pgvector', False)
	except Exception as e:
		logger.warning(f'Health check database check warning: {e}')
		db_status = 'unavailable'
		overall_status = 'degraded'

	return HealthResponse(
		status=overall_status,
		service='InsightBot AI',
		version='1.0.0',
		database=db_status,
		storage_mode=storage_mode,
		pgvector_enabled=has_vector,
	)


@router.get('/auth/me', tags=['System'], summary='Get current authenticated tenant and authorization context')
async def get_auth_context(tenant: TenantDep) -> TenantContext:
	"""Return the authenticated organization, workspace, and admin status."""
	return tenant


def validate_ingestion_metadata(meta: Any) -> dict[str, Any]:
	"""Validate that ingestion metadata is a bounded, correctly-typed JSON object."""
	if meta is None:
		return {}
	if not isinstance(meta, dict):
		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST,
			detail='Metadata must be a JSON object (key-value dictionary).',
		)
	if len(meta) > 50:
		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST,
			detail='Metadata exceeds maximum allowed number of keys (max 50).',
		)

	def _validate_node(node: Any, depth: int = 1) -> None:
		if depth > 4:
			raise HTTPException(
				status_code=status.HTTP_400_BAD_REQUEST,
				detail='Metadata nesting depth exceeds maximum allowed limit (4 levels).',
			)
		if isinstance(node, dict):
			if len(node) > 50:
				raise HTTPException(
					status_code=status.HTTP_400_BAD_REQUEST,
					detail='Metadata dictionary contains too many keys (max 50).',
				)
			for k, v in node.items():
				if not isinstance(k, str) or len(k) > 128:
					raise HTTPException(
						status_code=status.HTTP_400_BAD_REQUEST,
						detail='Metadata keys must be strings up to 128 characters.',
					)
				_validate_node(v, depth + 1)
		elif isinstance(node, list):
			if len(node) > 100:
				raise HTTPException(
					status_code=status.HTTP_400_BAD_REQUEST,
					detail='Metadata list contains too many items (max 100).',
				)
			for item in node:
				_validate_node(item, depth + 1)
		elif isinstance(node, str):
			if len(node) > 2048:
				raise HTTPException(
					status_code=status.HTTP_400_BAD_REQUEST,
					detail='Metadata string value exceeds maximum length (2048 characters).',
				)
		elif isinstance(node, (int, float, bool)) or node is None:
			pass
		else:
			raise HTTPException(
				status_code=status.HTTP_400_BAD_REQUEST,
				detail=f'Unsupported data type in metadata: {type(node).__name__}',
			)

	_validate_node(meta)
	return meta


@router.post(
	'/knowledge/documents',
	status_code=status.HTTP_201_CREATED,
	tags=['Knowledge'],
	summary='Ingest a new document (PDF, DOCX, TXT, Markdown)',
)
async def ingest_document(
	request: Request,
	store: StoreDep,
	tenant: TenantDep,
) -> Document:
	"""Upload and ingest a document into the private knowledge base."""
	content_type = request.headers.get('content-type', '')
	filename: str
	content: bytes
	meta: dict[str, Any] = {}
	max_size = get_settings().max_upload_size_bytes

	if 'multipart/form-data' in content_type:
		from starlette.datastructures import UploadFile as StarletteUploadFile

		form = await request.form()
		uploaded_file = form.get('file')
		if not isinstance(uploaded_file, StarletteUploadFile):
			raise HTTPException(
				status_code=status.HTTP_400_BAD_REQUEST,
				detail="Multipart form field 'file' is required.",
			)
		filename = uploaded_file.filename or 'uploaded_file.txt'

		if uploaded_file.size is not None and uploaded_file.size <= 0:
			raise HTTPException(
				status_code=status.HTTP_400_BAD_REQUEST,
				detail=f"Uploaded file '{filename}' is empty.",
			)

		if uploaded_file.size is not None and uploaded_file.size > max_size:
			raise HTTPException(
				status_code=status.HTTP_400_BAD_REQUEST,
				detail=f'File size ({uploaded_file.size} bytes) exceeds maximum allowed limit ({max_size} bytes).',
			)

		content = await uploaded_file.read(max_size + 1)
		if len(content) == 0:
			raise HTTPException(
				status_code=status.HTTP_400_BAD_REQUEST,
				detail=f"Uploaded file '{filename}' is empty.",
			)
		if len(content) > max_size:
			raise HTTPException(
				status_code=status.HTTP_400_BAD_REQUEST,
				detail=f'File size ({len(content)} bytes) exceeds maximum allowed limit ({max_size} bytes).',
			)
		meta_field = form.get('metadata')
		if meta_field:
			if not isinstance(meta_field, str):
				raise HTTPException(
					status_code=status.HTTP_400_BAD_REQUEST,
					detail="Multipart form field 'metadata' must be a valid JSON string.",
				)
			try:
				import json

				parsed_meta = json.loads(meta_field)
			except Exception as json_err:
				raise HTTPException(
					status_code=status.HTTP_400_BAD_REQUEST,
					detail=f'Invalid JSON in metadata field: {json_err}',
				) from json_err
			meta = validate_ingestion_metadata(parsed_meta)
		else:
			meta = {}
	elif 'application/json' in content_type:
		try:
			body = await request.json()
			payload = DocumentUploadPayload(**body)
			filename = payload.filename
			content = base64.b64decode(payload.content_base64)
			if len(content) == 0:
				raise HTTPException(
					status_code=status.HTTP_400_BAD_REQUEST,
					detail=f"Uploaded file '{filename}' is empty.",
				)
			if len(content) > max_size:
				raise HTTPException(
					status_code=status.HTTP_400_BAD_REQUEST,
					detail=f'File size ({len(content)} bytes) exceeds maximum allowed limit ({max_size} bytes).',
				)
			meta = validate_ingestion_metadata(payload.metadata)
		except HTTPException:
			raise
		except Exception as e:
			raise HTTPException(
				status_code=status.HTTP_400_BAD_REQUEST,
				detail=f'Invalid JSON or base64 content: {e}',
			) from e
	else:
		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST,
			detail="Content-Type must be 'multipart/form-data' or 'application/json'.",
		)

	# Anti-spoofing: verify client does not attempt to ingest into another tenant's space
	req_org = meta.get('organization_id') if isinstance(meta, dict) else None
	if req_org and req_org != tenant.organization_id and not tenant.is_admin:
		raise HTTPException(
			status_code=status.HTTP_403_FORBIDDEN,
			detail=f"Forbidden: Cannot ingest document into organization '{req_org}' with tenant context '{tenant.organization_id}'.",
		)

	try:
		doc = await store.ingest_document(
			filename=filename,
			content=content,
			metadata=meta,
			organization_id=tenant.organization_id,
			workspace_id=tenant.workspace_id,
		)
		return doc
	except (SecurityValidationError, DocumentParsingError, UnsupportedDocumentError, ValueError) as e:
		sanitized_msg = SensitiveDataSanitizer.sanitize(str(e))
		raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=sanitized_msg) from e
	except Exception as e:
		logger.error(f'Failed to ingest document: {e}', exc_info=True)
		clean_msg = f'Document ingestion failed: {type(e).__name__}'
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail=clean_msg,
		) from e


@router.get(
	'/knowledge/documents',
	tags=['Knowledge'],
	summary='List all ingested documents',
)
async def list_documents(
	store: StoreDep,
	tenant: TenantDep,
	limit: Annotated[int, Query(ge=1, le=500)] = 50,
	offset: Annotated[int, Query(ge=0)] = 0,
) -> list[Document]:
	"""List ingested documents ordered by creation time descending."""
	try:
		return await store.list_documents(
			limit=limit,
			offset=offset,
			organization_id=tenant.organization_id,
		)
	except Exception as e:
		logger.error(f'Failed to list documents: {e}', exc_info=True)
		clean_msg = f'Database query failed: {type(e).__name__}'
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail=clean_msg,
		) from e


@router.get(
	'/knowledge/documents/{document_id}',
	tags=['Knowledge'],
	summary='Get a document by ID',
)
async def get_document(
	document_id: str,
	store: StoreDep,
	tenant: TenantDep,
) -> Document:
	"""Retrieve metadata for a single document."""
	doc = await store.get_document(document_id, organization_id=tenant.organization_id)
	if not doc:
		raise HTTPException(
			status_code=status.HTTP_404_NOT_FOUND,
			detail=f"Document '{document_id}' not found.",
		)
	return doc


@router.get(
	'/knowledge/documents/{document_id}/chunks',
	tags=['Knowledge'],
	summary='Get all chunks for a document',
)
async def get_document_chunks(
	document_id: str,
	store: StoreDep,
	tenant: TenantDep,
) -> list[DocumentChunk]:
	"""Retrieve all parsed and embedded chunks for a document."""
	doc = await store.get_document(document_id, organization_id=tenant.organization_id)
	if not doc:
		raise HTTPException(
			status_code=status.HTTP_404_NOT_FOUND,
			detail=f"Document '{document_id}' not found.",
		)
	return await store.get_chunks(document_id, organization_id=tenant.organization_id)


@router.delete(
	'/knowledge/documents/{document_id}',
	tags=['Knowledge'],
	summary='Delete a document and all its chunks',
)
async def delete_document(
	document_id: str,
	store: StoreDep,
	tenant: TenantDep,
) -> dict[str, Any]:
	"""Delete a document and all associated chunks."""
	deleted = await store.delete_document(document_id, organization_id=tenant.organization_id)
	if not deleted:
		raise HTTPException(
			status_code=status.HTTP_404_NOT_FOUND,
			detail=f"Document '{document_id}' not found.",
		)
	return {'deleted': True, 'id': document_id}


@router.post(
	'/knowledge/search',
	tags=['Knowledge'],
	summary='Perform semantic similarity search',
)
async def semantic_search(
	payload: KnowledgeSearchRequest,
	store: StoreDep,
	tenant: TenantDep,
) -> list[KnowledgeSearchResult]:
	"""Search internal knowledge using semantic vector embeddings."""
	filter_meta = {'document_id': payload.document_id} if payload.document_id else None
	try:
		return await store.search_async(
			query=payload.query,
			top_k=payload.top_k,
			filter_metadata=filter_meta,
			hybrid=payload.hybrid,
			organization_id=tenant.organization_id,
		)
	except ValueError as e:
		raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
	except Exception as e:
		logger.error(f'Semantic search failed: {e}', exc_info=True)
		clean_msg = f'Search failed: {type(e).__name__}'
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail=clean_msg,
		) from e


@router.post(
	'/knowledge/ask',
	tags=['Knowledge'],
	summary='Ask a question with grounded citations',
)
async def ask_question(
	payload: KnowledgeAskRequest,
	agent: AgentDep,
	tenant: TenantDep,
) -> KnowledgeAnswer:
	"""Synthesize a grounded answer using retrieved private documents."""
	async with get_ai_semaphore():
		try:
			return await agent.answer(
				question=payload.question,
				top_k=payload.top_k,
				organization_id=tenant.organization_id,
			)
		except ValueError as e:
			raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
		except Exception as e:
			logger.error(f'KnowledgeAgent Q&A failed: {e}', exc_info=True)
			clean_msg = f'Answering failed: {type(e).__name__}'
			raise HTTPException(
				status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
				detail=clean_msg,
			) from e


@router.post(
	'/research',
	tags=['Research'],
	summary='Conduct end-to-end multi-source research',
)
async def conduct_research(
	payload: ResearchAPIRequest,
	engine: ResearchEngineDep,
	store: StoreDep,
	tenant: TenantDep,
	job_manager: JobManagerDep,
	sync: Annotated[Optional[bool], Query(description='Optional query flag to force synchronous execution')] = None,
) -> ResearchAPIResponse:
	"""Conduct unified research across private documents and autonomous web discovery."""
	if payload.target_domains:
		try:
			validate_target_domains(payload.target_domains)
		except ValueError as ve:
			raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve)) from ve

	session_title = payload.task.strip().split('\n')[0][:80]
	from datetime import datetime, timezone

	now_iso = datetime.now(timezone.utc).isoformat()
	effective_sync = payload.sync if sync is None else sync

	if not effective_sync:
		# Asynchronous background execution (Default for Phase 14D)
		session = ResearchSession(
			organization_id=tenant.organization_id,
			workspace_id=tenant.workspace_id,
			title=session_title,
			query=payload.task,
			status=RESEARCH_STATUS_QUEUED,
			progress_stage='queued',
			progress_percent=0,
			progress_message='Research job accepted and queued for background processing.',
			created_at=now_iso,
			updated_at=now_iso,
		)
		saved_id = await store.create_research_session(session)
		await job_manager.enqueue_job(
			session=session,
			task_query=payload.task,
			max_steps=payload.max_steps,
			engine=engine,
			store=store,
			target_domains=payload.target_domains,
		)
		return ResearchAPIResponse(
			status=RESEARCH_STATUS_QUEUED,
			task=payload.task,
			summary='Research job accepted and queued for background processing.',
			research_id=saved_id,
			progress_stage='queued',
			progress_percent=0,
		)

	# Synchronous execution (for backwards compatibility when sync=True)
	try:
		kwargs: dict[str, Any] = {
			'query': payload.task,
			'max_web_steps': payload.max_steps,
			'include_web': True,
		}
		import inspect

		sig = inspect.signature(engine.conduct_research)
		if 'organization_id' in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
			kwargs['organization_id'] = tenant.organization_id
		if 'target_domains' in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
			kwargs['target_domains'] = payload.target_domains

		async with get_ai_semaphore():
			result = await engine.conduct_research(**kwargs)
		sources = [c.source_uri for c in result.citations if c.source_uri]
		if not sources and result.is_grounded and result.sufficient_evidence:
			sources = [ev.source_uri for ev in result.internal_evidence + result.web_evidence if ev.source_uri]

		if result.metadata.get('error') or result.summary.startswith('AI provider error:'):
			status_str = RESEARCH_STATUS_PROVIDER_UNAVAILABLE
		elif result.is_grounded:
			status_str = RESEARCH_STATUS_COMPLETED
		else:
			status_str = RESEARCH_STATUS_COMPLETED_UNVERIFIED

		completed_iso = datetime.now(timezone.utc).isoformat()
		session = ResearchSession(
			organization_id=tenant.organization_id,
			workspace_id=tenant.workspace_id,
			title=session_title,
			query=payload.task,
			status=status_str,
			progress_stage='completed',
			progress_percent=100,
			progress_message='Research complete' if status_str == RESEARCH_STATUS_COMPLETED else status_str,
			summary=result.answer or result.summary,
			sources=sources,
			internal_evidence=[e.model_dump() for e in result.internal_evidence],
			web_evidence=[e.model_dump() for e in result.web_evidence],
			conflicts=[c.model_dump() for c in result.conflicts],
			citations=[c.model_dump() for c in result.citations],
			is_grounded=result.is_grounded,
			sufficient_evidence=result.sufficient_evidence,
			error=result.metadata.get('error'),
			metadata=result.metadata,
			created_at=now_iso,
			completed_at=completed_iso,
			updated_at=completed_iso,
		)
		saved_id = await store.create_research_session(session)

		return ResearchAPIResponse(
			status=status_str,
			task=payload.task,
			summary=result.answer or result.summary,
			sources=sources,
			evidence_count=result.evidence_count,
			doc_id=result.doc_id,
			research_id=saved_id,
			web_status=result.metadata.get('web_status'),
			conflicts=[c.model_dump() for c in result.conflicts],
			citations=[c.model_dump() for c in result.citations],
			progress_stage='completed',
			progress_percent=100,
		)
	except ValueError as e:
		raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
	except Exception as e:
		logger.error(f'ResearchEngine execution failed: {e}', exc_info=True)
		clean_msg = f'Research failed: {type(e).__name__}'
		try:
			from datetime import datetime, timezone

			failed_session = ResearchSession(
				organization_id=tenant.organization_id,
				workspace_id=tenant.workspace_id,
				title=session_title,
				query=payload.task,
				status=RESEARCH_STATUS_FAILED,
				progress_stage='failed',
				progress_percent=100,
				progress_message=f'Research failed: {e}',
				summary=f'Research failed: {e}',
				error=str(e),
				is_grounded=False,
				sufficient_evidence=False,
				created_at=now_iso,
				completed_at=datetime.now(timezone.utc).isoformat(),
				updated_at=datetime.now(timezone.utc).isoformat(),
			)
			await store.create_research_session(failed_session)
		except Exception as save_err:
			logger.warning(f'Failed to record failed research session: {save_err}')
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail=clean_msg,
		) from e


@router.get(
	'/research/{research_id}/status',
	tags=['Research'],
	summary='Get status and live progress for a research job',
)
async def get_research_job_status(
	research_id: str,
	store: StoreDep,
	tenant: TenantDep,
) -> ResearchJobStatusResponse:
	"""Retrieve the current lifecycle status and progress for a research job."""
	session = await store.get_research_session(research_id, organization_id=tenant.organization_id)
	if not session:
		raise HTTPException(
			status_code=status.HTTP_404_NOT_FOUND,
			detail=f"Research job '{research_id}' not found.",
		)
	is_terminal = session.status in TERMINAL_RESEARCH_STATUSES
	return ResearchJobStatusResponse(
		research_id=session.id,
		organization_id=session.organization_id,
		workspace_id=session.workspace_id,
		status=session.status,
		progress_stage=session.progress_stage,
		progress_percent=session.progress_percent,
		progress_message=session.progress_message,
		created_at=session.created_at,
		started_at=session.started_at,
		completed_at=session.completed_at,
		updated_at=session.updated_at,
		error=session.error,
		is_terminal=is_terminal,
	)


@router.post(
	'/research/{research_id}/cancel',
	tags=['Research'],
	summary='Cancel a queued or in-progress research job',
)
async def cancel_research_job(
	research_id: str,
	store: StoreDep,
	tenant: TenantDep,
	job_manager: JobManagerDep,
) -> dict[str, Any]:
	"""Cancel a queued or in-progress research job."""
	session = await store.get_research_session(research_id, organization_id=tenant.organization_id)
	if not session:
		raise HTTPException(
			status_code=status.HTTP_404_NOT_FOUND,
			detail=f"Research job '{research_id}' not found.",
		)
	if session.status in TERMINAL_RESEARCH_STATUSES:
		return {
			'cancelled': False,
			'research_id': research_id,
			'status': session.status,
			'message': f"Job is already in terminal state '{session.status}'",
		}
	cancelled = await job_manager.cancel_job(
		research_id=research_id,
		organization_id=tenant.organization_id,
		store=store,
	)
	return {
		'cancelled': cancelled,
		'research_id': research_id,
		'status': RESEARCH_STATUS_CANCELLED,
	}


@router.get(
	'/research/{research_id}/events',
	tags=['Research'],
	summary='Subscribe to live Server-Sent Events (SSE) for research job progress',
)
async def research_job_events(
	request: Request,
	research_id: str,
	store: StoreDep,
	tenant: TenantDep,
	job_manager: JobManagerDep,
) -> StreamingResponse:
	"""Stream real-time research execution progress events via Server-Sent Events (SSE)."""
	session = await store.get_research_session(research_id, organization_id=tenant.organization_id)
	if not session:
		raise HTTPException(
			status_code=status.HTTP_404_NOT_FOUND,
			detail=f"Research job '{research_id}' not found.",
		)

	async def event_generator():
		# Initial state event
		is_terminal = session.status in TERMINAL_RESEARCH_STATUSES
		initial_event = {
			'research_id': session.id,
			'status': session.status,
			'stage': session.progress_stage or session.status,
			'percent': session.progress_percent,
			'message': session.progress_message or f'Research job is {session.status}',
			'timestamp': session.updated_at,
			'is_terminal': is_terminal,
			'error': session.error,
		}
		yield f'event: progress\ndata: {json.dumps(initial_event)}\n\n'

		if is_terminal:
			return

		try:
			async for event in job_manager.subscribe(research_id):
				if await request.is_disconnected():
					break
				yield f'event: progress\ndata: {json.dumps(event)}\n\n'
				if event.get('is_terminal'):
					break
		except asyncio.CancelledError:
			pass

	return StreamingResponse(
		event_generator(),
		media_type='text/event-stream',
		headers={
			'Cache-Control': 'no-cache',
			'Connection': 'keep-alive',
			'X-Accel-Buffering': 'no',
		},
	)


@router.get(
	'/research',
	tags=['Research'],
	summary='List saved research sessions for authenticated tenant',
)
async def list_research_sessions(
	store: StoreDep,
	tenant: TenantDep,
	workspace_id: Annotated[Optional[str], Query(description='Filter by workspace ID')] = None,
	limit: Annotated[int, Query(ge=1, le=100, description='Max sessions to return')] = 50,
	offset: Annotated[int, Query(ge=0, description='Offset for pagination')] = 0,
) -> list[ResearchSession]:
	"""List saved research sessions scoped to authenticated tenant with pagination and optional workspace filtering."""
	effective_workspace = tenant.workspace_id or workspace_id
	return await store.list_research_sessions(
		limit=limit,
		offset=offset,
		organization_id=tenant.organization_id,
		workspace_id=effective_workspace,
	)


@router.get(
	'/research/{research_id}',
	tags=['Research'],
	summary='Retrieve saved research session by ID',
)
async def get_research_session(
	research_id: str,
	store: StoreDep,
	tenant: TenantDep,
) -> ResearchSession:
	"""Retrieve a full saved research session and its evidence by ID."""
	session = await store.get_research_session(research_id, organization_id=tenant.organization_id)
	if not session:
		raise HTTPException(
			status_code=status.HTTP_404_NOT_FOUND,
			detail=f"Research session '{research_id}' not found.",
		)
	return session


@router.delete(
	'/research/{research_id}',
	tags=['Research'],
	summary='Delete a saved research session',
)
async def delete_research_session(
	research_id: str,
	store: StoreDep,
	tenant: TenantDep,
) -> dict[str, Any]:
	"""Delete a saved research session."""
	deleted = await store.delete_research_session(research_id, organization_id=tenant.organization_id)
	if not deleted:
		raise HTTPException(
			status_code=status.HTTP_404_NOT_FOUND,
			detail=f"Research session '{research_id}' not found.",
		)
	return {'deleted': True, 'id': research_id}


@router.get(
	'/research/{research_id}/export',
	tags=['Research'],
	summary='Export a saved research report as Markdown or JSON',
)
async def export_research_session(
	research_id: str,
	request: Request,
	store: StoreDep,
	tenant: TenantDep,
	format: Annotated[Optional[str], Query(description="Export format: 'markdown' or 'json'")] = None,
) -> Response:
	"""Export a saved research report in Markdown or JSON format."""
	if not TENANT_ID_PATTERN.match(research_id):
		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST,
			detail="Invalid research ID format. Must be alphanumeric, underscores, or dashes (1-64 chars).",
		)

	session = await store.get_research_session(research_id, organization_id=tenant.organization_id)
	if not session:
		raise HTTPException(
			status_code=status.HTTP_404_NOT_FOUND,
			detail=f"Research session '{research_id}' not found.",
		)

	# Resolve format
	requested_format = format.lower() if format else None
	accept_header = request.headers.get('Accept', '')

	if requested_format:
		if requested_format not in ('markdown', 'json'):
			raise HTTPException(
				status_code=status.HTTP_400_BAD_REQUEST,
				detail=f"Unsupported export format '{format}'. Supported formats: 'markdown', 'json'.",
			)
		export_type = requested_format
	elif 'text/markdown' in accept_header:
		export_type = 'markdown'
	elif 'application/json' in accept_header:
		export_type = 'json'
	else:
		export_type = 'markdown'

	if export_type == 'json':
		return JSONResponse(
			content=format_research_json(session),
			media_type='application/json',
			headers={'Content-Disposition': f'attachment; filename="research-{research_id}.json"'},
		)
	else:
		md_text = format_research_markdown(session)
		return Response(
			content=md_text,
			media_type='text/markdown; charset=utf-8',
			headers={'Content-Disposition': f'attachment; filename="research-{research_id}.md"'},
		)


# -----------------------------------------------------------------------------
# Conversation Endpoints
# -----------------------------------------------------------------------------


@router.post(
	'/conversations',
	tags=['Conversations'],
	summary='Create a new conversation thread',
	status_code=status.HTTP_201_CREATED,
)
async def create_conversation(
	payload: ConversationCreateRequest,
	store: StoreDep,
	tenant: TenantDep,
) -> Conversation:
	"""Create a new multi-turn conversation thread scoped to the authenticated tenant."""
	effective_workspace = tenant.workspace_id or payload.workspace_id
	conv = Conversation(
		organization_id=tenant.organization_id,
		workspace_id=effective_workspace,
		title=payload.title or 'New Conversation',
		metadata=payload.metadata,
	)
	await store.create_conversation(conv)
	return conv


@router.get(
	'/conversations',
	tags=['Conversations'],
	summary='List conversations for authenticated tenant',
)
async def list_conversations(
	store: StoreDep,
	tenant: TenantDep,
	workspace_id: Annotated[Optional[str], Query(description='Filter by workspace ID')] = None,
	limit: Annotated[int, Query(ge=1, le=100, description='Max conversations to return')] = 50,
	offset: Annotated[int, Query(ge=0, description='Offset for pagination')] = 0,
) -> list[Conversation]:
	"""List conversations scoped to authenticated tenant with pagination and optional workspace filtering."""
	if workspace_id and not TENANT_ID_PATTERN.match(workspace_id):
		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST,
			detail="Invalid workspace_id query parameter format.",
		)
	effective_workspace = tenant.workspace_id or workspace_id
	return await store.list_conversations(
		organization_id=tenant.organization_id,
		workspace_id=effective_workspace,
		limit=limit,
		offset=offset,
	)


@router.get(
	'/conversations/{conversation_id}',
	tags=['Conversations'],
	summary='Retrieve conversation with chronological messages',
)
async def get_conversation(
	conversation_id: str,
	store: StoreDep,
	tenant: TenantDep,
) -> ConversationWithMessages:
	"""Retrieve conversation details and complete ordered message history."""
	conv = await store.get_conversation_with_messages(
		conversation_id=conversation_id,
		organization_id=tenant.organization_id,
	)
	if not conv:
		raise HTTPException(
			status_code=status.HTTP_404_NOT_FOUND,
			detail=f"Conversation '{conversation_id}' not found.",
		)
	return conv


@router.patch(
	'/conversations/{conversation_id}',
	tags=['Conversations'],
	summary='Update conversation title or metadata',
)
async def update_conversation(
	conversation_id: str,
	payload: ConversationUpdateRequest,
	store: StoreDep,
	tenant: TenantDep,
) -> Conversation:
	"""Update conversation title or metadata."""
	updated = await store.update_conversation(
		conversation_id=conversation_id,
		organization_id=tenant.organization_id,
		title=payload.title,
		metadata=payload.metadata,
	)
	if not updated:
		raise HTTPException(
			status_code=status.HTTP_404_NOT_FOUND,
			detail=f"Conversation '{conversation_id}' not found.",
		)
	return updated


@router.delete(
	'/conversations/{conversation_id}',
	tags=['Conversations'],
	summary='Delete a conversation thread and its messages',
)
async def delete_conversation(
	conversation_id: str,
	store: StoreDep,
	tenant: TenantDep,
) -> dict[str, Any]:
	"""Delete a conversation thread and all its messages."""
	deleted = await store.delete_conversation(
		conversation_id=conversation_id,
		organization_id=tenant.organization_id,
	)
	if not deleted:
		raise HTTPException(
			status_code=status.HTTP_404_NOT_FOUND,
			detail=f"Conversation '{conversation_id}' not found.",
		)
	return {'deleted': True, 'id': conversation_id}


@router.get(
	'/conversations/{conversation_id}/messages',
	tags=['Conversations'],
	summary='Get ordered messages for a conversation',
)
async def get_conversation_messages(
	conversation_id: str,
	store: StoreDep,
	tenant: TenantDep,
) -> list[ConversationMessage]:
	"""Retrieve ordered messages for a specific conversation."""
	conv = await store.get_conversation(conversation_id, organization_id=tenant.organization_id)
	if not conv:
		raise HTTPException(
			status_code=status.HTTP_404_NOT_FOUND,
			detail=f"Conversation '{conversation_id}' not found.",
		)
	return await store.get_messages(conversation_id)


@router.post(
	'/conversations/{conversation_id}/messages',
	tags=['Conversations'],
	summary='Submit user message and receive grounded assistant response',
)
async def post_conversation_message(
	conversation_id: str,
	payload: ConversationMessageCreateRequest,
	store: StoreDep,
	agent: AgentDep,
	tenant: TenantDep,
) -> ConversationMessage:
	"""Submit a user query to a conversation thread, generate a grounded multi-turn response, and persist both turns."""
	conv = await store.get_conversation(conversation_id, organization_id=tenant.organization_id)
	if not conv:
		raise HTTPException(
			status_code=status.HTTP_404_NOT_FOUND,
			detail=f"Conversation '{conversation_id}' not found.",
		)

	# Fetch existing messages for conversational context
	existing_messages = await store.get_messages(conversation_id)

	# 1. Persist user turn
	user_msg = ConversationMessage(
		conversation_id=conversation_id,
		role='user',
		content=payload.content,
	)
	await store.add_message(user_msg)

	# 2. Generate grounded assistant answer with conversational context
	async with get_ai_semaphore():
		try:
			answer = await agent.answer(
				question=payload.content,
				top_k=payload.top_k,
				organization_id=tenant.organization_id,
				conversation_history=existing_messages,
				max_history_turns=payload.max_history_turns,
			)
		except Exception as e:
			logger.error(f'Grounded chat answer generation failed: {e}', exc_info=True)
			clean_err = f'Chat synthesis failed: {type(e).__name__}'
			raise HTTPException(
				status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
				detail=clean_err,
			) from e

	# 3. Persist assistant turn
	assistant_msg = ConversationMessage(
		conversation_id=conversation_id,
		role='assistant',
		content=answer.answer,
		sources=answer.sources,
		is_grounded=answer.is_grounded,
		insufficient_evidence=answer.insufficient_evidence,
		metadata={
			'sufficient_context': answer.sufficient_context,
		},
	)
	await store.add_message(assistant_msg)

	# 4. If conversation still has default title, auto-generate title from first user query
	if conv.title == 'New Conversation':
		derived_title = payload.content.strip().split('\n')[0][:50]
		await store.update_conversation(
			conversation_id=conversation_id,
			organization_id=tenant.organization_id,
			title=derived_title,
		)

	return assistant_msg


@asynccontextmanager
async def lifespan(app: FastAPI):
	"""Manage application startup, database initialization, job reconciliation, and shutdown."""
	logger.info('Initializing InsightBot AI application resources...')
	from insightbot.config.preflight import check_configuration

	preflight = check_configuration()
	if not preflight.is_valid:
		for err in preflight.errors:
			logger.error(f'Configuration error: {err}')
		if preflight.environment in ('production', 'prod', 'staging', 'stage'):
			raise RuntimeError(f'Application startup aborted due to configuration errors: {"; ".join(preflight.errors)}')
	for warn in preflight.warnings:
		logger.warning(f'Configuration warning: {warn}')

	store = get_store()
	try:
		await store.initialize()
	except Exception as e:
		logger.warning(f'Database initialization during startup encountered warning: {e}')

	job_manager = get_job_manager()
	try:
		reconciled = await job_manager.reconcile_restarted_jobs(store)
		if reconciled > 0:
			logger.info(f'Reconciled {reconciled} orphaned research jobs after application restart.')
	except Exception as e:
		logger.warning(f'Orphaned research job reconciliation encountered warning: {e}')

	yield

	logger.info('Shutting down InsightBot AI application resources...')
	try:
		await job_manager.shutdown()
	except Exception as e:
		logger.warning(f'Job manager shutdown error: {e}')
	try:
		await store.close()
	except Exception as e:
		logger.warning(f'Store connection closure warning: {e}')


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
	"""Defensive HTTP security headers for all API responses."""

	async def dispatch(self, request: Request, call_next: Any) -> Response:
		response = await call_next(request)
		response.headers.setdefault('X-Content-Type-Options', 'nosniff')
		response.headers.setdefault('X-Frame-Options', 'DENY')
		response.headers.setdefault('Referrer-Policy', 'strict-origin-when-cross-origin')
		response.headers.setdefault(
			'Content-Security-Policy',
			"default-src 'self'; frame-ancestors 'none';",
		)
		response.headers.setdefault('Permissions-Policy', 'geolocation=(), microphone=(), camera=()')
		return response


def create_app() -> FastAPI:
	"""Create configured FastAPI application."""
	settings = get_settings()
	app = FastAPI(
		title='InsightBot AI',
		description='Private AI Research and Knowledge Agent REST API',
		version='1.0.0',
		lifespan=lifespan,
	)
	app.add_middleware(SecurityHeadersMiddleware)
	app.add_middleware(
		CORSMiddleware,
		allow_origins=settings.cors_origins,
		allow_credentials=True,
		allow_methods=['*'],
		allow_headers=['*'],
	)
	app.include_router(router)
	return app


app = create_app()

__all__ = [
	'ResearchAPIRequest',
	'ResearchAPIResponse',
	'ResearchRequest',
	'ResearchResponse',
	'ResearchJobStatusResponse',
	'RESEARCH_STATUS_INTERRUPTED',
	'KnowledgeSearchRequest',
	'KnowledgeAskRequest',
	'DocumentUploadPayload',
	'DocumentChunk',
	'HealthResponse',
	'ConversationCreateRequest',
	'ConversationUpdateRequest',
	'ConversationMessageCreateRequest',
	'Conversation',
	'ConversationMessage',
	'ConversationWithMessages',
	'ResearchSession',
	'format_research_markdown',
	'format_research_json',
	'get_store',
	'get_agent',
	'get_browser_agent',
	'get_research_engine',
	'get_job_manager',
	'get_current_tenant',
	'verify_api_key',
	'TenantDep',
	'JobManagerDep',
	'AuthDep',
	'router',
	'create_app',
	'app',
]
