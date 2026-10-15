"""BrowserResearchAgent: Internal browser research sub-engine for InsightBot AI.

Demotes the autonomous browser navigation loop into an internal, specialized
research worker that navigates websites, extracts DOM information, and returns
structured research evidence to the InsightBot orchestrator and application.
"""

import time
from typing import Any, Optional, Union

from browser_use.agent.service import Agent as _BaseAgent
from browser_use.agent.views import ActionResult, AgentHistoryList, AgentOutput
from insightbot.core.models import ResearchRequest, ResearchResult
from insightbot.engine.browser.extraction import extract_evidence_from_history


class BrowserResearchAgent(_BaseAgent):
	"""Internal browser research engine.

	Executes web navigation, DOM inspection, reading, and extraction tasks
	to harvest structured WebEvidence on behalf of the higher-level InsightBot
	knowledge orchestrator.
	"""

	def __init__(
		self,
		task: Optional[str] = None,
		request: Optional[Union[ResearchRequest, str]] = None,
		*args: Any,
		**kwargs: Any,
	) -> None:
		if request is not None:
			if isinstance(request, ResearchRequest):
				query_str = request.query
			else:
				query_str = str(request)
			task_str = task or query_str
		else:
			task_str = task or ''

		super().__init__(task=task_str, *args, **kwargs)
		self.logger.name = 'insightbot.engine.browser.BrowserResearchAgent'
		self._current_request: Optional[ResearchRequest] = request if isinstance(request, ResearchRequest) else None

	async def conduct_research(
		self,
		request: Optional[Union[ResearchRequest, str]] = None,
		max_steps: Optional[int] = None,
		target_domains: Optional[list[str]] = None,
		**kwargs: Any,
	) -> ResearchResult:
		"""Execute web research workflow and return structured ResearchResult with WebEvidence.

		Research Request
		       ↓
		BrowserResearchAgent
		       ↓
		Browser Engine
		       ↓
		Navigate / Search / Read / Extract
		       ↓
		Web Evidence
		       ↓
		Structured Research Result
		"""
		target_query: str
		steps: int = 30

		if request is not None:
			if isinstance(request, ResearchRequest):
				self._current_request = request
				target_query = request.query
				steps = request.max_steps
			else:
				target_query = str(request)
				steps = max_steps or 30
			self.task = target_query
		elif self._current_request is not None:
			target_query = self._current_request.query
			steps = max_steps or self._current_request.max_steps
		else:
			target_query = self.task
			steps = max_steps or 30

		start_time = time.time()
		history: AgentHistoryList = await self.run(max_steps=steps)
		duration = time.time() - start_time

		return extract_evidence_from_history(
			query=target_query,
			history=history,
			duration_seconds=duration,
		)


# Expose as Agent within the engine namespace for backwards compatibility
Agent = BrowserResearchAgent


class DeterministicFakeBrowserAgent:
	"""Deterministic fake browser agent for testing and offline development without external network calls."""

	def __init__(
		self,
		findings: Optional[list[dict[str, Any]]] = None,
		fail_with: Optional[Exception] = None,
		delay_seconds: float = 0.0,
	) -> None:
		self.findings = (
			findings
			if findings is not None
			else [
				{
					'url': 'https://example.com/autonomous-research',
					'title': 'Autonomous AI Browser Agent Spec',
					'snippet': 'InsightBot AI executes verified browser research with strict citation boundaries.',
					'relevance_score': 0.95,
				}
			]
		)
		self.fail_with = fail_with
		self.delay_seconds = delay_seconds
		self.call_count = 0
		self.last_query: Optional[str] = None

	async def conduct_research(
		self,
		request: Optional[Union[ResearchRequest, str]] = None,
		max_steps: Optional[int] = None,
		**kwargs: Any,
	) -> Any:
		"""Return deterministic simulated web evidence."""
		self.call_count += 1
		if isinstance(request, ResearchRequest):
			self.last_query = request.query
		else:
			self.last_query = str(request) if request is not None else ''

		if self.delay_seconds > 0:
			import asyncio

			await asyncio.sleep(self.delay_seconds)

		if self.fail_with:
			raise self.fail_with

		class _FakeResult:
			def __init__(self, evidence: list[dict[str, Any]]) -> None:
				self.evidence = evidence

		return _FakeResult(self.findings)


def create_browser_agent(
	task: Optional[str] = None,
	headless: Optional[bool] = None,
	use_cloud: Optional[bool] = None,
	allowed_domains: Optional[list[str]] = None,
	prohibited_domains: Optional[list[str]] = None,
	target_domains: Optional[list[str]] = None,
) -> Optional[Any]:
	"""Factory function to instantiate BrowserResearchAgent when configured with SSRF boundaries."""
	import logging

	from insightbot.config.settings import get_settings
	from insightbot.security.policy import DEFAULT_PROHIBITED_DOMAINS, validate_target_domains

	agent_logger = logging.getLogger('insightbot.engine.browser')
	settings = get_settings()

	if not getattr(settings, 'browser_enabled', False):
		agent_logger.info("Browser research is disabled via configuration ('INSIGHTBOT_BROWSER_ENABLED=false').")
		return None

	import os

	api_key = os.getenv('BROWSER_USE_API_KEY')
	if not api_key:
		agent_logger.info('BROWSER_USE_API_KEY not found in environment; browser research agent unavailable.')
		return None

	effective_allowed: Optional[list[str]] = None
	raw_allowed = allowed_domains or target_domains
	if raw_allowed:
		effective_allowed = validate_target_domains(raw_allowed)

	try:
		from browser_use import Browser, ChatBrowserUse

		is_headless = bool(headless if headless is not None else getattr(settings, 'browser_headless', True))
		is_cloud = bool(use_cloud if use_cloud is not None else getattr(settings, 'browser_use_cloud', False))

		safe_prohibited = list(DEFAULT_PROHIBITED_DOMAINS)
		if prohibited_domains:
			for p in prohibited_domains:
				if p not in safe_prohibited:
					safe_prohibited.append(p)

		browser = (
			Browser(
				use_cloud=True,
				allowed_domains=effective_allowed,
				prohibited_domains=safe_prohibited,
			)
			if is_cloud
			else Browser(
				headless=is_headless,
				allowed_domains=effective_allowed,
				prohibited_domains=safe_prohibited,
			)
		)
		llm = ChatBrowserUse()
		return BrowserResearchAgent(task=task or '', browser=browser, llm=llm)
	except Exception as e:
		agent_logger.warning(f'Could not instantiate BrowserResearchAgent: {e}')
		return None


__all__ = [
	'BrowserResearchAgent',
	'Agent',
	'DeterministicFakeBrowserAgent',
	'create_browser_agent',
	'ActionResult',
	'AgentOutput',
	'AgentHistoryList',
]
