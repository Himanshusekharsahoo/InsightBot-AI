"""InsightBot AI Core Exceptions."""


class InsightBotError(Exception):
	"""Base exception for all InsightBot errors."""

	pass


class BrowserEngineError(InsightBotError):
	"""Raised when browser automation engine encounters an error."""

	pass


class SecurityPolicyViolationError(InsightBotError):
	"""Raised when an operation violates security policy (e.g. domain restriction)."""

	pass


class KnowledgeStoreError(InsightBotError):
	"""Raised when knowledge indexing or retrieval fails."""

	pass


class ResearchExecutionError(InsightBotError):
	"""Raised when a research task fails to complete."""

	pass
