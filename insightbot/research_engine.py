"""InsightBot AI: Research Engine.

Provides unified research synthesis fusing private knowledge base retrieval
and autonomous web exploration with strict fact-grounding and citation verification.
"""

from insightbot.orchestrator.research_engine import (
	BrowserResearchAgentProtocol,
	ConflictItem,
	NormalizedEvidence,
	ResearchCitation,
	ResearchEngine,
	ResearchEngineResult,
)

__all__ = [
	'ResearchEngine',
	'NormalizedEvidence',
	'ConflictItem',
	'ResearchCitation',
	'ResearchEngineResult',
	'BrowserResearchAgentProtocol',
]
