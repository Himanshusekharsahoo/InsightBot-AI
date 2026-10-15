"""InsightBot Browser Engine DOM Extraction & Representation."""

from browser_use.dom.markdown_extractor import extract_clean_markdown
from browser_use.dom.service import DomService
from browser_use.dom.views import (
	DOMInteractedElement,
	EnhancedDOMTreeNode,
	MarkdownChunk,
	SerializedDOMState,
)

# Application aliases
DOMState = SerializedDOMState
DOMElementNode = EnhancedDOMTreeNode

__all__ = [
	'DomService',
	'extract_clean_markdown',
	'SerializedDOMState',
	'DOMState',
	'DOMInteractedElement',
	'EnhancedDOMTreeNode',
	'DOMElementNode',
	'MarkdownChunk',
]
