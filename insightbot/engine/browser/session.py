"""InsightBot Browser Engine Session Management.

Exposes BrowserSession and lifecycle control for web automation.
"""

from browser_use.browser import BrowserSession as _BaseBrowserSession
from browser_use.browser.profile import BrowserProfile
from browser_use.browser.session_manager import SessionManager

# Primary application-facing class
BrowserSession = _BaseBrowserSession
Browser = _BaseBrowserSession

__all__ = [
	'BrowserSession',
	'Browser',
	'BrowserProfile',
	'SessionManager',
]
