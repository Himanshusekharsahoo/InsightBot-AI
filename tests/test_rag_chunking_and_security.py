"""Unit tests for chunking, text normalization, content hashing, and upload security."""

import pytest

from insightbot.knowledge.chunker import (
	DeterministicChunker,
	compute_content_hash,
	normalize_text,
)
from insightbot.knowledge.parsers import DocumentPage, ParsedDocument
from insightbot.knowledge.validation import (
	SecurityValidationError,
	sanitize_filename,
	validate_upload,
)


def test_normalize_text():
	raw = 'Line 1   with spaces   \r\n\r\n\r\n\r\nLine 2 with \t tabs.  \rLine 3'
	normalized = normalize_text(raw)
	assert '\r' not in normalized
	assert '\n\n\n' not in normalized
	assert 'Line 1 with spaces' in normalized
	assert 'Line 2 with tabs.' in normalized
	assert 'Line 3' in normalized


def test_compute_content_hash():
	hash1 = compute_content_hash(b'InsightBot AI test content')
	hash2 = compute_content_hash(b'InsightBot AI test content')
	hash3 = compute_content_hash(b'Different content')

	assert len(hash1) == 64  # sha256 hex string
	assert hash1 == hash2
	assert hash1 != hash3


def test_chunker_validation():
	with pytest.raises(ValueError):
		DeterministicChunker(chunk_size=500, chunk_overlap=500)  # overlap == size

	with pytest.raises(ValueError):
		DeterministicChunker(chunk_size=500, chunk_overlap=600)  # overlap > size

	with pytest.raises(ValueError):
		DeterministicChunker(chunk_size=-10, chunk_overlap=0)

	with pytest.raises(ValueError):
		DeterministicChunker(chunk_size=500, chunk_overlap=-5)


def test_deterministic_chunking():
	chunker = DeterministicChunker(chunk_size=100, chunk_overlap=20)
	text = (
		'Paragraph one contains initial documentation facts. '
		'It describes the vector database deployment and system requirements.\n\n'
		'Paragraph two contains secondary architectural guidelines. '
		'This ensures high availability and horizontal scaling.'
	)

	parsed = ParsedDocument(
		filename='architecture.txt',
		source_type='txt',
		mime_type='text/plain',
		text=text,
		pages=[DocumentPage(page_number=1, text=text)],
	)

	chunks = chunker.chunk_document(document_id='doc-test-1', parsed_doc=parsed)
	assert len(chunks) > 1

	for idx, chk in enumerate(chunks):
		assert chk.document_id == 'doc-test-1'
		assert chk.chunk_index == idx
		assert len(chk.content) > 0
		assert chk.metadata['source_filename'] == 'architecture.txt'
		assert chk.metadata['doc_type'] == 'txt'
		assert chk.metadata['page_number'] == 1


def test_multi_page_chunking():
	chunker = DeterministicChunker(chunk_size=100, chunk_overlap=20)
	parsed = ParsedDocument(
		filename='multipage.pdf',
		source_type='pdf',
		mime_type='application/pdf',
		text='Page 1 text.\n\nPage 2 text.',
		pages=[
			DocumentPage(page_number=1, text='Page 1 text detailing components and modules.'),
			DocumentPage(page_number=2, text='Page 2 text explaining database configuration.'),
		],
	)

	chunks = chunker.chunk_document(document_id='doc-multi', parsed_doc=parsed)
	assert len(chunks) == 2
	assert chunks[0].metadata['page_number'] == 1
	assert chunks[1].metadata['page_number'] == 2
	assert chunks[0].metadata['doc_type'] == 'pdf'
	assert chunks[1].metadata['doc_type'] == 'pdf'
	assert 'Page 1' in chunks[0].content
	assert 'Page 2' in chunks[1].content


def test_upload_security_filename_sanitization():
	assert sanitize_filename('safe_report.pdf') == 'safe_report.pdf'
	# Directory traversal stripped
	assert sanitize_filename('../../etc/passwd.txt') == 'passwd.txt'
	assert sanitize_filename('C:\\Windows\\System32\\calc.exe.txt') == 'calc.exe.txt'

	# Unsafe characters replaced
	assert sanitize_filename('my report (final)!.pdf') == 'my_report__final__.pdf'

	with pytest.raises(SecurityValidationError):
		sanitize_filename('')

	with pytest.raises(SecurityValidationError):
		sanitize_filename('   ')


def test_upload_security_validation():
	# Valid file
	safe_name = validate_upload(
		filename='document.pdf',
		content_length=1024,
		max_size_bytes=10000,
	)
	assert safe_name == 'document.pdf'

	# Oversized file
	with pytest.raises(SecurityValidationError):
		validate_upload(
			filename='huge.pdf',
			content_length=50000,
			max_size_bytes=10000,
		)

	# Unsupported extension
	with pytest.raises(SecurityValidationError):
		validate_upload(
			filename='malicious.exe',
			content_length=100,
		)


def test_sensitive_data_sanitizer_secrets_and_db_urls():
	"""Verify SensitiveDataSanitizer scrubs database credentials, tokens, and authorization headers."""
	from insightbot.security.policy import SensitiveDataSanitizer

	# 1. Database connection URL with credentials
	raw_db_log = 'Connecting to postgresql://postgres:MySuperSecretPass123@db.prod.internal:5432/insightbot_db'
	sanitized_db = SensitiveDataSanitizer.sanitize(raw_db_log)
	assert 'MySuperSecretPass123' not in sanitized_db
	assert 'Connecting to postgresql://postgres:[REDACTED]@db.prod.internal:5432/insightbot_db' == sanitized_db

	# 2. Authorization header
	raw_auth_hdr = 'Header sent: Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.token.sig'
	sanitized_auth = SensitiveDataSanitizer.sanitize(raw_auth_hdr)
	assert 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.token.sig' not in sanitized_auth
	assert '[REDACTED]' in sanitized_auth

	# 3. OpenAI key with sk-proj-
	raw_openai = 'Failed request with key sk-proj-abc1234567890def1234567890'
	sanitized_openai = SensitiveDataSanitizer.sanitize(raw_openai)
	assert 'sk-proj-abc1234567890def1234567890' not in sanitized_openai
	assert '[REDACTED]' in sanitized_openai

	# 4. Google API key
	raw_google = 'Google client init with key AIzaSyFakeGoogleApiKey1234567890123'
	sanitized_google = SensitiveDataSanitizer.sanitize(raw_google)
	assert 'AIzaSyFakeGoogleApiKey1234567890123' not in sanitized_google
	assert '[REDACTED]' in sanitized_google

	# 5. Non-sensitive text preserved
	plain = 'Normal system operational log at timestamp 2026-10-11T12:00:00Z'
	assert SensitiveDataSanitizer.sanitize(plain) == plain


def test_settings_configuration_validation():
	"""Verify Settings model rejects invalid timeouts, providers, dimensions, and chunk constraints."""
	from insightbot.config.settings import Settings

	# 1. Negative or zero timeout
	with pytest.raises(ValueError):
		Settings(ai_request_timeout=0.0)
	with pytest.raises(ValueError):
		Settings(ai_request_timeout=-5.0)

	# 2. Unsupported embedding provider
	with pytest.raises(ValueError):
		Settings(embedding_provider='unsupported_provider')

	# 3. Unsupported LLM provider
	with pytest.raises(ValueError):
		Settings(llm_provider='unsupported_llm')

	# 4. Incompatible dimension
	with pytest.raises(ValueError):
		Settings(embedding_dimension=0)
	with pytest.raises(ValueError):
		Settings(embedding_dimension=-100)

	# 5. Incompatible chunk overlap >= chunk_size
	with pytest.raises(ValueError):
		Settings(chunk_size=500, chunk_overlap=500)
	with pytest.raises(ValueError):
		Settings(chunk_size=500, chunk_overlap=600)

	# 6. Invalid min_similarity_score
	with pytest.raises(ValueError):
		Settings(min_similarity_score=-0.1)
	with pytest.raises(ValueError):
		Settings(min_similarity_score=1.5)


def test_settings_production_safeguards(monkeypatch):
	"""Verify Settings enforces strict constraints in production, prod, and staging."""
	from insightbot.config.settings import Settings

	monkeypatch.delenv('INSIGHTBOT_LLM_PROVIDER', raising=False)
	monkeypatch.delenv('INSIGHTBOT_EMBEDDING_PROVIDER', raising=False)

	valid_key = 'super_secret_production_key_123456'

	for env in ('production', 'prod', 'staging', 'stage'):
		# 1. Missing or weak API key
		with pytest.raises(ValueError, match='INSIGHTBOT_API_KEY'):
			Settings(environment=env, api_key=None)
		with pytest.raises(ValueError, match='INSIGHTBOT_API_KEY'):
			Settings(environment=env, api_key='short_key')

		# 2. Mock provider forbidden
		with pytest.raises(ValueError, match='mock providers are strictly forbidden'):
			Settings(environment=env, api_key=valid_key, llm_provider='mock')
		with pytest.raises(ValueError, match='mock providers are strictly forbidden'):
			Settings(environment=env, api_key=valid_key, embedding_provider='mock')

		# 3. in_memory store forbidden
		with pytest.raises(ValueError, match='in_memory.*forbidden'):
			Settings(environment=env, api_key=valid_key, vector_store_mode='in_memory')

		# 4. Wildcard CORS forbidden
		with pytest.raises(ValueError, match='wildcard.*CORS.*forbidden'):
			Settings(environment=env, api_key=valid_key, cors_origins=['*'])

		# 5. Valid deployment configuration passes
		cfg = Settings(
			environment=env,
			api_key=valid_key,
			llm_provider='google',
			embedding_provider='google',
			embedding_dimension=768,
			vector_store_mode='auto',
			cors_origins=['https://app.insightbot.internal'],
		)
		assert cfg.environment == env
		assert cfg.api_key == valid_key
		assert cfg.cors_origins == ['https://app.insightbot.internal']


def test_settings_development_and_test_safeguards_relaxed():
	"""Verify Settings permits mock providers and in-memory store in dev/test environments."""
	from insightbot.config.settings import Settings

	for env in ('development', 'dev', 'test', 'local'):
		dev_cfg = Settings(
			environment=env,
			api_key=None,
			llm_provider='mock',
			embedding_provider='mock',
			vector_store_mode='in_memory',
			cors_origins=['*'],
		)
		assert dev_cfg.environment == env
		assert dev_cfg.api_key is None
		assert dev_cfg.llm_provider == 'mock'
		assert dev_cfg.vector_store_mode == 'in_memory'
		assert '*' in dev_cfg.cors_origins
