"""InsightBot AI Async Research Worker.

Processes queued research requests asynchronously using the internal
BrowserResearchAgent sub-engine and indexes findings into the team KnowledgeStore.
"""

import asyncio
import logging
from typing import Optional

from insightbot.core.models import ResearchRequest, ResearchResult
from insightbot.engine.browser.agent import BrowserResearchAgent
from insightbot.knowledge.store import KnowledgeStore

logger = logging.getLogger('insightbot.workers.research')


class ResearchWorker:
	"""Worker that executes background web research jobs."""

	def __init__(self, knowledge_store: Optional[KnowledgeStore] = None) -> None:
		self.knowledge_store = knowledge_store or KnowledgeStore()
		self._queue: asyncio.Queue[tuple[ResearchRequest, asyncio.Future[ResearchResult]]] = asyncio.Queue()
		self._running = False
		self._task: Optional[asyncio.Task[None]] = None

	async def enqueue(self, request: ResearchRequest) -> asyncio.Future[ResearchResult]:
		"""Submit a research request to the background worker queue."""
		loop = asyncio.get_running_loop()
		future: asyncio.Future[ResearchResult] = loop.create_future()
		await self._queue.put((request, future))
		return future

	async def start(self) -> None:
		"""Start worker processing loop."""
		self._running = True
		self._task = asyncio.create_task(self._run_loop())

	async def stop(self) -> None:
		"""Stop worker processing loop."""
		self._running = False
		if self._task:
			self._task.cancel()
			try:
				await self._task
			except asyncio.CancelledError:
				pass

	async def _run_loop(self) -> None:
		while self._running:
			try:
				request, future = await self._queue.get()
				try:
					agent = BrowserResearchAgent(
						task=request.query,
						use_vision='auto',
					)
					result = await agent.conduct_research(request=request)
					# Index into knowledge store
					if result.summary:
						doc_id = f'worker-research-{len(self.knowledge_store.get_all()) + 1}'
						primary_url = result.sources[0] if result.sources else 'web'
						self.knowledge_store.store_document(
							doc_id=doc_id,
							title=request.query[:80],
							source_url=primary_url,
							content=result.summary,
							metadata={'request': request.model_dump()},
						)
					future.set_result(result)
				except Exception as e:
					logger.error(f'Worker failed executing research task: {e}', exc_info=True)
					future.set_exception(e)
				finally:
					self._queue.task_done()
			except asyncio.CancelledError:
				break
			except Exception as e:
				logger.error(f'Worker unexpected error: {e}', exc_info=True)
