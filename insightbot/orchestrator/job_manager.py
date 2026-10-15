"""InsightBot AI Research Job Manager & SSE Event Broadcaster.

Orchestrates background research job execution, maintains live event broadcaster queues
for Server-Sent Events (SSE), manages job cancellation, prevents concurrent duplicate execution,
and handles clean startup reconciliation and shutdown.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Optional

from insightbot.db.models import (
	RESEARCH_STATUS_CANCELLED,
	RESEARCH_STATUS_COMPLETED,
	RESEARCH_STATUS_COMPLETED_UNVERIFIED,
	RESEARCH_STATUS_FAILED,
	RESEARCH_STATUS_IN_PROGRESS,
	RESEARCH_STATUS_INTERRUPTED,
	RESEARCH_STATUS_PROVIDER_UNAVAILABLE,
	RESEARCH_STATUS_QUEUED,
	TERMINAL_RESEARCH_STATUSES,
	ResearchSession,
)
from insightbot.knowledge.store import KnowledgeStore
from insightbot.orchestrator.research_engine import ResearchEngine
from insightbot.security.policy import SensitiveDataSanitizer, get_ai_semaphore

logger = logging.getLogger('insightbot.orchestrator.job_manager')


def _utc_now_iso() -> str:
	return datetime.now(timezone.utc).isoformat()


class ResearchJobManager:
	"""Orchestrates asynchronous research job execution and live SSE event broadcasting."""

	def __init__(self) -> None:
		self._active_tasks: dict[str, asyncio.Task[None]] = {}
		self._subscribers: dict[str, list[asyncio.Queue[Optional[dict[str, Any]]]]] = {}
		self._running_jobs: set[str] = set()
		self._lock = asyncio.Lock()

	async def enqueue_job(
		self,
		session: ResearchSession,
		task_query: str,
		max_steps: int,
		engine: ResearchEngine,
		store: KnowledgeStore,
		target_domains: Optional[list[str]] = None,
	) -> str:
		"""Enqueue and initiate background research execution for a registered session."""
		async with self._lock:
			if session.id in self._running_jobs:
				logger.warning(f"Research job '{session.id}' is already running; skipping duplicate execution.")
				return session.id

			self._running_jobs.add(session.id)
			task = asyncio.create_task(
				self._execute_job(
					research_id=session.id,
					organization_id=session.organization_id,
					workspace_id=session.workspace_id,
					query=task_query,
					max_steps=max_steps,
					engine=engine,
					store=store,
					target_domains=target_domains,
				),
				name=f'research-job-{session.id}',
			)
			self._active_tasks[session.id] = task

			def _on_done(t: asyncio.Task[None]) -> None:
				self._active_tasks.pop(session.id, None)
				self._running_jobs.discard(session.id)
				if not t.cancelled() and t.exception():
					logger.error(f"Research background task '{session.id}' encountered unhandled error: {t.exception()}")

			task.add_done_callback(_on_done)
			return session.id

	async def emit_progress(
		self,
		research_id: str,
		stage: str,
		percent: int,
		message: str,
		status: str = RESEARCH_STATUS_IN_PROGRESS,
		is_terminal: bool = False,
		error: Optional[str] = None,
	) -> None:
		"""Broadcast a structured progress event to all active SSE subscribers."""
		event_payload = {
			'research_id': research_id,
			'status': status,
			'stage': stage,
			'percent': max(0, min(100, percent)),
			'message': message,
			'timestamp': _utc_now_iso(),
			'is_terminal': is_terminal,
			'error': error,
		}

		queues = self._subscribers.get(research_id, [])
		for q in list(queues):
			try:
				if q.full():
					# Drop oldest unconsumed event to maintain bounded memory
					try:
						q.get_nowait()
					except Exception:
						pass
				q.put_nowait(event_payload)
			except Exception as e:
				logger.debug(f"Failed to deliver SSE event to queue for '{research_id}': {e}")

		if is_terminal and research_id in self._subscribers:
			# Send termination sentinel to subscribers
			for q in list(self._subscribers.get(research_id, [])):
				try:
					q.put_nowait(None)
				except Exception:
					pass

	async def subscribe(self, research_id: str) -> AsyncGenerator[dict[str, Any], None]:
		"""Subscribe to live progress events for a research job as an async generator."""
		queue: asyncio.Queue[Optional[dict[str, Any]]] = asyncio.Queue(maxsize=100)
		if research_id not in self._subscribers:
			self._subscribers[research_id] = []
		self._subscribers[research_id].append(queue)

		try:
			while True:
				event = await queue.get()
				if event is None:
					break
				yield event
				if event.get('is_terminal'):
					break
		finally:
			if research_id in self._subscribers and queue in self._subscribers[research_id]:
				self._subscribers[research_id].remove(queue)
				if not self._subscribers[research_id]:
					self._subscribers.pop(research_id, None)

	async def cancel_job(
		self,
		research_id: str,
		organization_id: str,
		store: KnowledgeStore,
	) -> bool:
		"""Cancel an active or queued research job cooperatively."""
		async with self._lock:
			session = await store.get_research_session(research_id, organization_id=organization_id)
			if not session:
				logger.warning(f"Attempted to cancel nonexistent or cross-tenant research job '{research_id}'")
				return False

			if session.status in TERMINAL_RESEARCH_STATUSES:
				logger.info(f"Research job '{research_id}' is already in terminal state '{session.status}'; cancel ignored.")
				return False

			task = self._active_tasks.get(research_id)
			if task and not task.done():
				task.cancel()

			completed_iso = _utc_now_iso()
			try:
				await store.update_research_session(
					session_id=research_id,
					organization_id=organization_id,
					status=RESEARCH_STATUS_CANCELLED,
					progress_stage='cancelled',
					progress_percent=100,
					progress_message='Research job cancelled by user',
					completed_at=completed_iso,
				)
			except Exception as e:
				logger.warning(f"Error persisting cancelled state for '{research_id}': {e}")
				return False

			await self.emit_progress(
				research_id=research_id,
				stage='cancelled',
				percent=100,
				message='Research job cancelled by user',
				status=RESEARCH_STATUS_CANCELLED,
				is_terminal=True,
			)
			self._running_jobs.discard(research_id)
			return True

	async def reconcile_restarted_jobs(
		self,
		store: KnowledgeStore,
		mark_status: str = RESEARCH_STATUS_INTERRUPTED,
	) -> int:
		"""Reconcile orphaned queued and in_progress research jobs after application restart."""
		reconciled_count = 0
		try:
			# List saved sessions across the store
			sessions = await store.list_research_sessions(limit=500, offset=0)
			orphaned = [s for s in sessions if s.status in (RESEARCH_STATUS_QUEUED, RESEARCH_STATUS_IN_PROGRESS)]
			for s in orphaned:
				now_iso = _utc_now_iso()
				logger.info(f"Reconciling orphaned research job '{s.id}' (was '{s.status}'). Marking as {mark_status}.")
				try:
					await store.update_research_session(
						session_id=s.id,
						organization_id=s.organization_id,
						status=mark_status,
						progress_stage=mark_status,
						progress_percent=100,
						progress_message='Research job was interrupted by server restart or shutdown.',
						error='Job interrupted by application shutdown/restart.',
						completed_at=now_iso,
					)
					reconciled_count += 1
				except Exception as err:
					logger.warning(f"Failed to reconcile session '{s.id}': {err}")
		except Exception as e:
			logger.warning(f'Error during startup research job reconciliation: {e}')
		return reconciled_count

	async def shutdown(self, timeout: float = 5.0) -> None:
		"""Cleanly terminate all in-flight research tasks and close subscriber queues."""
		logger.info(f'Shutting down ResearchJobManager ({len(self._active_tasks)} active tasks)...')
		tasks = list(self._active_tasks.values())
		for t in tasks:
			t.cancel()

		if tasks:
			try:
				await asyncio.wait_for(
					asyncio.gather(*tasks, return_exceptions=True),
					timeout=timeout,
				)
			except asyncio.TimeoutError:
				logger.warning(f'Some research tasks did not complete cancellation within {timeout}s.')

		# Close all remaining subscriber queues
		for research_id, queues in list(self._subscribers.items()):
			for q in queues:
				try:
					q.put_nowait(None)
				except Exception:
					pass
		self._subscribers.clear()
		self._active_tasks.clear()
		self._running_jobs.clear()
		logger.info('ResearchJobManager shutdown complete.')

	async def _execute_job(
		self,
		research_id: str,
		organization_id: str,
		workspace_id: Optional[str],
		query: str,
		max_steps: int,
		engine: ResearchEngine,
		store: KnowledgeStore,
		target_domains: Optional[list[str]] = None,
	) -> None:
		"""Internal execution loop coordinating status updates, progress emissions, and persistence."""
		start_iso = _utc_now_iso()
		try:
			# Check pre-execution cancellation
			current_task = asyncio.current_task()
			if current_task and current_task.cancelled():
				raise asyncio.CancelledError()

			# 1. Transition to in_progress
			await store.update_research_session(
				session_id=research_id,
				organization_id=organization_id,
				status=RESEARCH_STATUS_IN_PROGRESS,
				progress_stage='started',
				progress_percent=5,
				progress_message='Starting research workflow...',
				started_at=start_iso,
			)
			await self.emit_progress(
				research_id=research_id,
				stage='started',
				percent=5,
				message='Starting research workflow...',
			)

			# 2. Define real-time progress callback
			async def on_progress(stage: str, percent: int, message: str) -> None:
				if current_task and current_task.cancelled():
					return
				try:
					await store.update_research_session(
						session_id=research_id,
						organization_id=organization_id,
						progress_stage=stage,
						progress_percent=percent,
						progress_message=message,
					)
				except Exception as p_err:
					logger.debug(f"Progress DB update warning for '{research_id}': {p_err}")
				await self.emit_progress(
					research_id=research_id,
					stage=stage,
					percent=percent,
					message=message,
				)

			# 3. Conduct research with bounded concurrency
			if current_task and current_task.cancelled():
				raise asyncio.CancelledError()

			async with get_ai_semaphore():
				import inspect

				sig = inspect.signature(engine.conduct_research)
				research_kwargs: dict[str, Any] = {
					'query': query,
					'max_web_steps': max_steps,
					'include_web': True,
					'organization_id': organization_id,
					'on_progress': on_progress,
				}
				if 'target_domains' in sig.parameters or any(
					p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
				):
					research_kwargs['target_domains'] = target_domains

				result = await engine.conduct_research(**research_kwargs)

			# Check if task was cancelled during research synthesis
			if current_task and current_task.cancelled():
				raise asyncio.CancelledError()

			# Guard against racing cancellation or terminal state updates
			latest_session = await store.get_research_session(research_id, organization_id=organization_id)
			if latest_session and latest_session.status in TERMINAL_RESEARCH_STATUSES:
				logger.info(
					f"Research job '{research_id}' is already in terminal status '{latest_session.status}'; suppressing completion overwrite."
				)
				return

			sources = [c.source_uri for c in result.citations if c.source_uri]
			if not sources and result.is_grounded and result.sufficient_evidence:
				sources = [ev.source_uri for ev in result.internal_evidence + result.web_evidence if ev.source_uri]

			if result.metadata.get('error') or result.summary.startswith('AI provider error:'):
				status_str = RESEARCH_STATUS_PROVIDER_UNAVAILABLE
				err_msg = result.metadata.get('error') or result.summary
			elif result.is_grounded:
				status_str = RESEARCH_STATUS_COMPLETED
				err_msg = None
			else:
				status_str = RESEARCH_STATUS_COMPLETED_UNVERIFIED
				err_msg = None

			completed_iso = _utc_now_iso()

			# 4. Persist completed artifacts atomically before broadcasting success
			try:
				await store.update_research_session(
					session_id=research_id,
					organization_id=organization_id,
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
					error=err_msg,
					metadata=result.metadata,
					completed_at=completed_iso,
				)
			except Exception as persist_err:
				logger.error(
					f"Failed to persist research completion artifacts for '{research_id}': {persist_err}",
					exc_info=True,
				)
				sanitized_err = SensitiveDataSanitizer.sanitize(f'Report persistence failed: {persist_err}')
				try:
					await store.update_research_session(
						session_id=research_id,
						organization_id=organization_id,
						status=RESEARCH_STATUS_FAILED,
						progress_stage='failed',
						progress_percent=100,
						progress_message=f'Research failed: {sanitized_err}',
						error=sanitized_err,
						completed_at=completed_iso,
					)
				except Exception:
					pass
				await self.emit_progress(
					research_id=research_id,
					stage='failed',
					percent=100,
					message=f'Research failed: {sanitized_err}',
					status=RESEARCH_STATUS_FAILED,
					is_terminal=True,
					error=sanitized_err,
				)
				return

			await self.emit_progress(
				research_id=research_id,
				stage='completed',
				percent=100,
				message='Research workflow finished successfully'
				if status_str == RESEARCH_STATUS_COMPLETED
				else f'Research finished with status: {status_str}',
				status=status_str,
				is_terminal=True,
				error=err_msg,
			)

		except asyncio.CancelledError:
			logger.info(f"Research job '{research_id}' received cancellation request.")
			completed_iso = _utc_now_iso()
			try:
				curr = await store.get_research_session(research_id, organization_id=organization_id)
				if not curr or curr.status != RESEARCH_STATUS_CANCELLED:
					await store.update_research_session(
						session_id=research_id,
						organization_id=organization_id,
						status=RESEARCH_STATUS_CANCELLED,
						progress_stage='cancelled',
						progress_percent=100,
						progress_message='Research job cancelled by user',
						completed_at=completed_iso,
					)
			except Exception as c_err:
				logger.warning(f"Error persisting cancelled status for '{research_id}': {c_err}")

			await self.emit_progress(
				research_id=research_id,
				stage='cancelled',
				percent=100,
				message='Research job cancelled',
				status=RESEARCH_STATUS_CANCELLED,
				is_terminal=True,
			)
			raise

		except Exception as e:
			try:
				curr = await store.get_research_session(research_id, organization_id=organization_id)
				if curr and curr.status in TERMINAL_RESEARCH_STATUSES:
					logger.info(
						f"Research job '{research_id}' was already in terminal status '{curr.status}'; suppressing failure overwrite."
					)
					return
			except Exception:
				pass

			logger.error(f"Research job '{research_id}' execution failed: {e}", exc_info=True)
			sanitized_err = SensitiveDataSanitizer.sanitize(str(e))
			completed_iso = _utc_now_iso()
			try:
				await store.update_research_session(
					session_id=research_id,
					organization_id=organization_id,
					status=RESEARCH_STATUS_FAILED,
					progress_stage='failed',
					progress_percent=100,
					progress_message=f'Research failed: {sanitized_err}',
					error=sanitized_err,
					completed_at=completed_iso,
				)
			except Exception as f_err:
				logger.warning(f"Error persisting failed status for '{research_id}': {f_err}")

			await self.emit_progress(
				research_id=research_id,
				stage='failed',
				percent=100,
				message=f'Research failed: {sanitized_err}',
				status=RESEARCH_STATUS_FAILED,
				is_terminal=True,
				error=sanitized_err,
			)


# Global singleton instance
_default_job_manager: Optional[ResearchJobManager] = None


def get_job_manager() -> ResearchJobManager:
	"""Retrieve the global ResearchJobManager singleton."""
	global _default_job_manager
	if _default_job_manager is None:
		_default_job_manager = ResearchJobManager()
	return _default_job_manager


__all__ = [
	'ResearchJobManager',
	'get_job_manager',
]
