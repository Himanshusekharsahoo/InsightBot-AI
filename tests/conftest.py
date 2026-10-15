"""Root pytest configuration for InsightBot AI.

Provides global fixtures and hooks to separate offline CI tests from live provider tests.
"""

import os

import pytest


def pytest_runtest_setup(item: pytest.Item) -> None:
	"""Enforce explicit opt-in for live tests requiring external network access."""
	if 'live' in item.keywords:
		enable_live = os.getenv('INSIGHTBOT_ENABLE_LIVE_TESTS', '').lower() in ('1', 'true', 'yes')
		if not enable_live:
			pytest.skip(
				'Live external AI test skipped. Explicit opt-in required: set INSIGHTBOT_ENABLE_LIVE_TESTS=1.'
			)
