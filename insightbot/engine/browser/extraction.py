"""InsightBot Browser Engine Web Extraction & Evidence Processing."""

from typing import Any

from insightbot.core.models import ResearchResult, WebEvidence


def extract_evidence_from_history(
	query: str,
	history: Any,
	duration_seconds: float = 0.0,
) -> ResearchResult:
	"""Convert an execution history into structured ResearchResult with WebEvidence.

	Extracts visited URLs, page titles, text extracts, and screenshots from the
	Browser-Use AgentHistoryList, organizing them into a structured result.
	"""
	evidence_items: list[WebEvidence] = []
	visited_urls: list[str] = []

	if hasattr(history, 'urls'):
		visited_urls = list(dict.fromkeys([u for u in history.urls() if u and u != 'about:blank']))

	# Collect extracted content
	extracted_contents: list[str] = []
	if hasattr(history, 'extracted_content'):
		extracted_contents = [c for c in history.extracted_content() if c]

	screenshot_paths: list[str] = []
	if hasattr(history, 'screenshot_paths'):
		screenshot_paths = [s for s in history.screenshot_paths() if s]

	# Build evidence entries from extracts and visited URLs
	if extracted_contents:
		for idx, content in enumerate(extracted_contents):
			source_url = visited_urls[min(idx, len(visited_urls) - 1)] if visited_urls else 'web'
			screenshot = screenshot_paths[min(idx, len(screenshot_paths) - 1)] if screenshot_paths else None
			evidence_items.append(
				WebEvidence(
					url=source_url,
					snippet=content,
					screenshot_path=screenshot,
					relevance_score=1.0,
				)
			)
	elif visited_urls:
		for idx, url in enumerate(visited_urls):
			screenshot = screenshot_paths[min(idx, len(screenshot_paths) - 1)] if screenshot_paths else None
			evidence_items.append(
				WebEvidence(
					url=url,
					snippet=f'Visited and inspected page: {url}',
					screenshot_path=screenshot,
					relevance_score=0.8,
				)
			)

	# Get final synthesized result
	summary = ''
	if hasattr(history, 'final_result') and history.final_result():
		summary = str(history.final_result())
	elif extracted_contents:
		summary = '\n\n'.join(extracted_contents)
	else:
		summary = f'Completed browsing for: {query}. Inspected {len(visited_urls)} page(s).'

	is_successful = True
	if hasattr(history, 'is_successful') and history.is_successful() is not None:
		is_successful = bool(history.is_successful())

	error = None
	if hasattr(history, 'errors') and any(history.errors()):
		errs = [str(e) for e in history.errors() if e]
		if errs:
			error = '; '.join(errs)

	return ResearchResult(
		query=query,
		summary=summary,
		evidence=evidence_items,
		sources=visited_urls,
		is_successful=is_successful,
		error=error,
		duration_seconds=duration_seconds,
	)
