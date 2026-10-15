"""InsightBot Browser Engine Subsystem.

Provides internal browser automation, web navigation, DOM inspection,
and evidence extraction services for the InsightBot AI platform.
"""

from insightbot.engine.browser.agent import (
	ActionResult,
	Agent,
	AgentHistoryList,
	AgentOutput,
	BrowserResearchAgent,
)
from insightbot.engine.browser.dom import (
	DOMElementNode,
	DomService,
	DOMState,
	MarkdownChunk,
	extract_clean_markdown,
)
from insightbot.engine.browser.extraction import extract_evidence_from_history
from insightbot.engine.browser.profile import BrowserProfile, ProxySettings, ViewportSize
from insightbot.engine.browser.session import Browser, BrowserSession, SessionManager
from insightbot.engine.browser.tools import ActionModel, Controller, Tools

__all__ = [
	'BrowserResearchAgent',
	'Agent',
	'BrowserSession',
	'Browser',
	'BrowserProfile',
	'ProxySettings',
	'ViewportSize',
	'SessionManager',
	'Tools',
	'Controller',
	'ActionModel',
	'ActionResult',
	'AgentHistoryList',
	'AgentOutput',
	'DomService',
	'extract_clean_markdown',
	'DOMState',
	'DOMElementNode',
	'MarkdownChunk',
	'extract_evidence_from_history',
]
