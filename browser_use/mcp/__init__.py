"""MCP (Model Context Protocol) support for InsightBot AI.

This module provides integration with MCP servers and clients for browser automation and research.
"""

from browser_use.mcp.client import MCPClient
from browser_use.mcp.controller import MCPToolWrapper

__all__ = ['MCPClient', 'MCPToolWrapper', 'BrowserUseServer', 'InsightBotServer']  # type: ignore


def __getattr__(name):
	"""Lazy import to avoid importing server module when only client is needed."""
	if name in ('BrowserUseServer', 'InsightBotServer'):
		from browser_use.mcp.server import BrowserUseServer, InsightBotServer

		return InsightBotServer if name == 'InsightBotServer' else BrowserUseServer
	raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
