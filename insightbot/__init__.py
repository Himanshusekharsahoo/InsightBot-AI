"""InsightBot AI: Private AI Research & Knowledge Agent for Teams.

System Architecture:
- InsightBotAgent: Top-level research orchestrator and knowledge coordinator
- BrowserResearchAgent: Internal browser research and extraction sub-engine
- BrowserSession: Low-level browser session management and CDP interface
- KnowledgeStore: Private team knowledge repository and local RAG
- InsightBotConfig: Centralized configuration and environment loader
- SecurityPolicy: Enterprise policy controls and domain filtering
"""

__version__ = '1.0.0'

from insightbot.config import CONFIG, InsightBotConfig
from insightbot.core import (
	BrowserEngineError,
	InsightBotError,
	InsightBotTask,
	KnowledgeStoreError,
	ResearchRequest,
	ResearchResult,
	SecurityPolicyViolationError,
	WebEvidence,
)
from insightbot.engine.browser import (
	ActionResult,
	AgentHistoryList,
	Browser,
	BrowserProfile,
	BrowserResearchAgent,
	BrowserSession,
	DomService,
	Tools,
)
from insightbot.knowledge import KnowledgeStore, ResearchDocument
from insightbot.orchestrator import InsightBotAgent, KnowledgeAgent
from insightbot.orchestrator.research_engine import (
	ConflictItem,
	NormalizedEvidence,
	ResearchCitation,
	ResearchEngine,
	ResearchEngineResult,
)
from insightbot.security import DomainFilter, SecurityPolicy

# Aliases for backwards compatibility within the platform
Agent = InsightBotAgent

__all__ = [
	'InsightBotAgent',
	'KnowledgeAgent',
	'BrowserResearchAgent',
	'ResearchEngine',
	'NormalizedEvidence',
	'ConflictItem',
	'ResearchCitation',
	'ResearchEngineResult',
	'Agent',
	'KnowledgeStore',
	'ResearchDocument',
	'InsightBotConfig',
	'CONFIG',
	'BrowserSession',
	'Browser',
	'BrowserProfile',
	'Tools',
	'DomService',
	'ActionResult',
	'AgentHistoryList',
	'WebEvidence',
	'ResearchRequest',
	'ResearchResult',
	'InsightBotTask',
	'SecurityPolicy',
	'DomainFilter',
	'InsightBotError',
	'BrowserEngineError',
	'KnowledgeStoreError',
	'SecurityPolicyViolationError',
	'__version__',
]
