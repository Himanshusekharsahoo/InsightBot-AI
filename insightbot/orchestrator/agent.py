"""InsightBotAgent: High-Level Private AI Research & Knowledge Agent for Teams.

Coordinates team research workflows by planning research strategies,
dispatching web investigations to the internal BrowserResearchAgent sub-engine,
synthesizing facts, and persisting insights into the private KnowledgeStore.
"""

import logging
from typing import Any, Optional, Union

from insightbot.core.models import ResearchRequest, ResearchResult
from insightbot.engine.browser.agent import BrowserResearchAgent
from insightbot.knowledge.store import KnowledgeStore

logger = logging.getLogger('insightbot.orchestrator')


class InsightBotAgent:
	"""Top-level orchestrator for team research and knowledge management."""

	def __init__(
		self,
		task: Optional[str] = None,
		request: Optional[Union[ResearchRequest, str]] = None,
		llm: Any = None,
		browser_profile: Any = None,
		knowledge_store: Optional[KnowledgeStore] = None,
		**kwargs: Any,
	) -> None:
		if request is not None:
			task_str = request.query if isinstance(request, ResearchRequest) else str(request)
		else:
			task_str = task or ''

		self.task = task_str
		self.llm = llm
		self.browser_profile = browser_profile
		self.knowledge_store = knowledge_store or KnowledgeStore()
		self.kwargs = kwargs
		self._sub_engine = BrowserResearchAgent(
			task=self.task,
			llm=llm,
			browser_profile=browser_profile,
			**kwargs,
		)

	async def conduct_research(
		self,
		request: Optional[Union[ResearchRequest, str]] = None,
		max_steps: int = 30,
	) -> ResearchResult:
		"""Conduct structured research mission and persist findings into KnowledgeStore."""
		req_obj: ResearchRequest
		if isinstance(request, ResearchRequest):
			req_obj = request
		elif isinstance(request, str):
			req_obj = ResearchRequest(query=request, max_steps=max_steps)
		else:
			req_obj = ResearchRequest(query=self.task, max_steps=max_steps)

		logger.info(f'InsightBot orchestrator initiating research mission: {req_obj.query}')
		research_result: ResearchResult = await self._sub_engine.conduct_research(request=req_obj)

		# Persist synthesized findings to local knowledge base
		if research_result.summary:
			doc_id = f'research-{len(self.knowledge_store.get_all()) + 1}'
			primary_url = research_result.sources[0] if research_result.sources else 'web'
			self.knowledge_store.store_document(
				doc_id=doc_id,
				title=req_obj.query[:80],
				source_url=primary_url,
				content=research_result.summary,
				metadata={
					'query': req_obj.query,
					'evidence_count': len(research_result.evidence),
					'sources': research_result.sources,
				},
			)
			logger.info(f'InsightBot indexed research findings into KnowledgeStore (id={doc_id})')

		return research_result

	async def run(self, max_steps: int = 50) -> Any:
		"""Execute the orchestrated research workflow, returning history for compatibility."""
		logger.info(f'InsightBot orchestrator running research mission: {self.task}')
		result = await self._sub_engine.run(max_steps=max_steps)

		# If result has extracted content, persist to knowledge base
		final_content = result.final_result() if hasattr(result, 'final_result') else str(result)
		if final_content:
			doc_id = f'research-{len(self.knowledge_store.get_all()) + 1}'
			urls = result.urls() if hasattr(result, 'urls') and result.urls() else ['web']
			primary_url = str(urls[-1] or 'web') if urls else 'web'
			self.knowledge_store.store_document(
				doc_id=doc_id,
				title=self.task[:80],
				source_url=primary_url,
				content=final_content,
				metadata={'task': self.task},
			)
			logger.info(f'InsightBot indexed research findings into KnowledgeStore (id={doc_id})')

		return result

	@property
	def history(self) -> Any:
		return self._sub_engine.history


__all__ = ['InsightBotAgent']
