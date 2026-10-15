"""InsightBot AI Security Subsystem.

Enforces domain filtering, data sanitization, air-gapped constraints,
and enterprise safety policies during browser research sessions.
"""

import fnmatch
import ipaddress
import re
from typing import Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field

DEFAULT_PROHIBITED_DOMAINS: list[str] = [
	'localhost',
	'*.localhost',
	'127.0.0.1',
	'::1',
	'0.0.0.0',
	'169.254.169.254',
	'metadata.google.internal',
	'*.metadata.google.internal',
	'*.internal',
	'*.local',
	'*.lan',
	'*.home.arpa',
]

_DELIMITER_PATTERN = re.compile(
	r'(?:[<＜\uFE64]|&lt;)\s*(/?)\s*(UNTRUSTED_REFERENCE_EVIDENCE|CONVERSATION_HISTORY)\b[^>＞\uFE65&]*(?:[>＞\uFE65]|&gt;)',
	re.IGNORECASE,
)


def sanitize_evidence_delimiters(content: str) -> str:
	"""Neutralize prompt injection delimiter tags inside untrusted text chunks."""
	if not content:
		return ''
	return _DELIMITER_PATTERN.sub(r'[\1\2]', str(content))


def is_ssrf_prohibited(target: str) -> bool:
	"""Check if a host, IP, or URL targets SSRF, loopback, link-local, or private network boundaries."""
	if not target:
		return True

	target = str(target).strip()
	if not target:
		return True

	# If a scheme is present, restrict strictly to http and https
	if '://' in target:
		try:
			parsed = urlparse(target)
			if parsed.scheme.lower() not in ('http', 'https'):
				return True
			host = parsed.hostname or ''
			if not host:
				return True
		except Exception:
			return True
	elif target.lower().startswith(('file:', 'gopher:', 'ftp:', 'data:', 'javascript:')):
		return True
	else:
		host = target.split('/')[0].split(':')[0]

	host = host.strip('[]').lower()
	if not host:
		return True

	# Direct match or wildcard suffix against default prohibited list
	for pattern in DEFAULT_PROHIBITED_DOMAINS:
		p = pattern.lower().strip('[]')
		if DomainFilter._matches(p, host):
			return True

	# Check standard IP addresses (IPv4 and IPv6)
	try:
		ip = ipaddress.ip_address(host)
		if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
			return True
		# Check for IPv4-mapped IPv6 address (e.g. ::ffff:127.0.0.1)
		if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
			mapped = ip.ipv4_mapped
			if (
				mapped.is_private
				or mapped.is_loopback
				or mapped.is_link_local
				or mapped.is_reserved
				or mapped.is_multicast
				or mapped.is_unspecified
			):
				return True
	except ValueError:
		pass

	# Check alternative IPv4 representations (octal, hex, decimal dword, short-form) via socket.inet_aton
	try:
		import socket

		raw_bytes = socket.inet_aton(host)
		# socket.inet_aton succeeded: host is a valid IPv4 representation
		ip4 = ipaddress.IPv4Address(raw_bytes)
		if (
			ip4.is_private
			or ip4.is_loopback
			or ip4.is_link_local
			or ip4.is_reserved
			or ip4.is_multicast
			or ip4.is_unspecified
		):
			return True
	except (OSError, ValueError):
		pass

	return False


def validate_target_domains(domains: Optional[list[str]]) -> list[str]:
	"""Validate and clean caller-specified target domains against SSRF prohibitions.

	Raises ValueError if any target domain resolves to an SSRF or prohibited destination.
	"""
	if not domains:
		return []

	validated: list[str] = []
	for d in domains:
		cleaned = str(d).strip().lower()
		if not cleaned:
			continue
		if is_ssrf_prohibited(cleaned):
			raise ValueError(f"Prohibited navigation target '{d}' is not permitted for security reasons.")
		validated.append(cleaned)
	return validated


class DomainFilter:
	"""Enforces whitelist and blacklist domain policies."""

	def __init__(
		self,
		allowed_domains: Optional[list[str]] = None,
		prohibited_domains: Optional[list[str]] = None,
	) -> None:
		self.allowed_domains = allowed_domains or []
		combined_prohibited = list(DEFAULT_PROHIBITED_DOMAINS)
		if prohibited_domains:
			for p in prohibited_domains:
				if p not in combined_prohibited:
					combined_prohibited.append(p)
		self.prohibited_domains = combined_prohibited

	def is_domain_allowed(self, target: str) -> bool:
		"""Check if a host or URL is permitted by this policy."""
		if not target:
			return False

		# Extract hostname if given a full URL
		if '://' in target:
			try:
				parsed = urlparse(target)
				host = parsed.hostname or target
			except Exception:
				host = target
		else:
			host = target.split('/')[0].split(':')[0]

		host = host.lower()

		# Check SSRF protections
		if is_ssrf_prohibited(host):
			return False

		# Check prohibited domains first
		for pattern in self.prohibited_domains:
			pattern = pattern.lower()
			if self._matches(pattern, host):
				return False

		# If allowed_domains is empty, everything not prohibited is allowed
		if not self.allowed_domains:
			return True

		# Check allowed domains
		for pattern in self.allowed_domains:
			pattern = pattern.lower()
			if self._matches(pattern, host):
				return True

		return False

	@staticmethod
	def _matches(pattern: str, host: str) -> bool:
		if pattern == host:
			return True
		if pattern.startswith('*.'):
			suffix = pattern[2:]
			if host == suffix or host.endswith('.' + suffix):
				return True
		elif host.endswith('.' + pattern):
			return True
		return fnmatch.fnmatch(host, pattern)


class SecurityPolicy(BaseModel):
	"""Enterprise and team browser research security policy."""

	allowed_domains: list[str] = Field(default_factory=list, description='Allowed domain patterns')
	prohibited_domains: list[str] = Field(default_factory=list, description='Prohibited domain patterns')
	air_gapped: bool = Field(default=False, description='Disallow external telemetry and phone-home checks')
	mask_sensitive_data: bool = Field(default=True, description='Mask sensitive credentials in traces')
	allow_file_downloads: bool = Field(default=True, description='Allow downloading documents during research')

	def create_domain_filter(self) -> DomainFilter:
		return DomainFilter(
			allowed_domains=self.allowed_domains,
			prohibited_domains=self.prohibited_domains,
		)


class SensitiveDataSanitizer:
	"""Redacts sensitive values like passwords and API keys from logs and evidence."""

	REDACTED_TEXT = '[REDACTED]'

	_API_KEY_PATTERNS = [
		re.compile(r'sk-(?:proj-)?[a-zA-Z0-9_\-]{16,}'),
		re.compile(r'AIza[a-zA-Z0-9_\-]{20,}'),
		re.compile(r'Bearer\s+[a-zA-Z0-9_\-\.]{16,}', re.IGNORECASE),
		re.compile(r'((?:Authorization|Proxy-Authorization)\s*:\s*(?:Bearer|Basic)\s+)[^\s\r\n,]+', re.IGNORECASE),
		re.compile(
			r'("?(?:api[_-]?key|access[_-]?token|refresh[_-]?token|authorization|secret(?:[_-]?key)?|password|client[_-]?secret)"?\s*[:=]\s*"?[^"\s,]{6,}"?)',
			re.IGNORECASE,
		),
	]

	_DB_URL_PATTERN = re.compile(r'(postgres(?:ql)?:\/\/[^:]+:)([^@]+)(@)', re.IGNORECASE)

	@classmethod
	def redact_api_key(cls, api_key: Optional[str]) -> str:
		"""Redact an API key showing only prefix and suffix if long enough."""
		if not api_key:
			return ''
		if len(api_key) <= 8:
			return cls.REDACTED_TEXT
		return f'{api_key[:3]}...{api_key[-4:]}'

	@classmethod
	def sanitize(cls, text: str, sensitive_terms: Optional[list[str]] = None) -> str:
		if not text:
			return text
		sanitized = text
		if sensitive_terms:
			for term in sensitive_terms:
				if term and len(term) > 2:
					sanitized = sanitized.replace(term, cls.REDACTED_TEXT)

		# Sanitize database URLs
		sanitized = cls._DB_URL_PATTERN.sub(rf'\g<1>{cls.REDACTED_TEXT}\g<3>', sanitized)

		# Also apply common credential pattern scrubbing
		for pattern in cls._API_KEY_PATTERNS:
			if pattern.groups == 1:
				sanitized = pattern.sub(rf'\g<1>{cls.REDACTED_TEXT}', sanitized)
			else:
				sanitized = pattern.sub(cls.REDACTED_TEXT, sanitized)
		return sanitized


import asyncio

TENANT_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_\-]{1,64}$')

_ai_semaphore: Optional[asyncio.Semaphore] = None


def get_ai_semaphore() -> asyncio.Semaphore:
	"""Process-local concurrency semaphore limiting simultaneous heavy AI synthesis jobs."""
	global _ai_semaphore
	if _ai_semaphore is None:
		from insightbot.config.settings import get_settings

		_ai_semaphore = asyncio.Semaphore(get_settings().ai_max_concurrent_requests)
	return _ai_semaphore
