"""InsightBot AI Core Data Models (Pydantic v2)."""

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class WebEvidence(BaseModel):
	"""Structured evidence extracted from web pages during research."""

	model_config = ConfigDict(extra='ignore')

	url: str = Field(..., description='Source URL where evidence was found')
	title: Optional[str] = Field(default=None, description='Page title')
	snippet: str = Field(..., description='Relevant extracted text or fact')
	relevance_score: float = Field(default=1.0, ge=0.0, le=1.0, description='Confidence or relevance score')
	timestamp: str = Field(
		default_factory=lambda: datetime.now(timezone.utc).isoformat(),
		description='ISO timestamp when evidence was captured',
	)
	screenshot_path: Optional[str] = Field(default=None, description='Local or remote path to screenshot artifact')
	metadata: dict[str, Any] = Field(default_factory=dict, description='Additional metadata')


class ResearchRequest(BaseModel):
	"""Input specification for a research task."""

	model_config = ConfigDict(extra='ignore')

	query: str = Field(..., min_length=1, description='Primary research query or goal')
	target_domains: Optional[list[str]] = Field(default=None, description='Allowed domains to restrict search')
	max_steps: int = Field(default=30, ge=1, le=200, description='Maximum navigation steps for browser sub-engine')
	extract_schema: Optional[dict[str, Any]] = Field(default=None, description='Optional JSON schema for structured extraction')
	headless: bool = Field(default=False, description='Run browser headlessly')
	context: Optional[dict[str, Any]] = Field(default_factory=dict, description='Contextual information or team metadata')


class ResearchResult(BaseModel):
	"""Structured outcome produced by BrowserResearchAgent and InsightBotAgent."""

	model_config = ConfigDict(extra='ignore')

	query: str = Field(..., description='Original research query')
	summary: str = Field(..., description='Synthesized findings and answer')
	evidence: list[WebEvidence] = Field(default_factory=list, description='Gathered web evidence items')
	sources: list[str] = Field(default_factory=list, description='List of unique source URLs consulted')
	is_successful: bool = Field(default=True, description='Whether the research goal was achieved')
	error: Optional[str] = Field(default=None, description='Error message if failed')
	duration_seconds: float = Field(default=0.0, description='Total duration of research execution')
	created_at: str = Field(
		default_factory=lambda: datetime.now(timezone.utc).isoformat(),
		description='ISO timestamp of result completion',
	)
	metadata: dict[str, Any] = Field(default_factory=dict, description='Execution metadata')


class InsightBotTask(BaseModel):
	"""Task definition dispatched to the InsightBot orchestrator."""

	model_config = ConfigDict(extra='ignore')

	task_id: str = Field(..., description='Unique task identifier')
	instruction: str = Field(..., description='Task instructions for the agent')
	created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
	status: str = Field(default='pending', description='Task status: pending, running, completed, failed')
	result: Optional[ResearchResult] = Field(default=None, description='Completed research result')
