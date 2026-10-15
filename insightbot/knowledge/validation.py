"""Upload security and filename validation for InsightBot AI.

Protects against path traversal, arbitrary file writes, unapproved extensions,
and oversized payloads.
"""

import re
from pathlib import Path

ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.txt', '.md', '.markdown'}
DEFAULT_MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


class SecurityValidationError(ValueError):
	"""Raised when file validation fails for security reasons."""

	pass


def sanitize_filename(filename: str) -> str:
	"""Sanitize client-provided filename to prevent path traversal.

	Strips directory components and replaces unsafe characters.
	"""
	if not filename or not filename.strip():
		raise SecurityValidationError('Filename cannot be empty.')

	# Detect explicit path traversal attempts
	if '..' in filename or '/' in filename or '\\' in filename:
		pass  # Will be stripped to basename, but let's check basename

	base_name = Path(filename).name.strip()
	if not base_name or base_name in ('.', '..'):
		raise SecurityValidationError('Invalid filename with directory traversal indicators.')

	# Keep only alphanumeric, hyphen, underscore, and dot
	sanitized = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', base_name)
	if not sanitized or sanitized.startswith('.'):
		raise SecurityValidationError('Filename cannot start with a dot or be empty after sanitization.')

	return sanitized


def validate_upload(
	filename: str,
	content_length: int,
	max_size_bytes: int = DEFAULT_MAX_FILE_SIZE,
	allowed_extensions: set[str] | None = None,
) -> str:
	"""Validate upload for security constraints: filename safety, extension, and file size.

	Returns the safe sanitized filename.
	"""
	if content_length <= 0:
		raise SecurityValidationError(f"File content for '{filename}' is empty.")

	if content_length > max_size_bytes:
		raise SecurityValidationError(
			f'File size ({content_length} bytes) exceeds maximum allowed limit ({max_size_bytes} bytes).'
		)

	safe_name = sanitize_filename(filename)
	ext = Path(safe_name).suffix.lower()

	allowed = allowed_extensions or ALLOWED_EXTENSIONS
	if ext not in allowed:
		raise SecurityValidationError(f"Unsupported file extension '{ext}'. Allowed extensions: {', '.join(sorted(allowed))}")

	return safe_name


__all__ = [
	'ALLOWED_EXTENSIONS',
	'DEFAULT_MAX_FILE_SIZE',
	'SecurityValidationError',
	'sanitize_filename',
	'validate_upload',
]
