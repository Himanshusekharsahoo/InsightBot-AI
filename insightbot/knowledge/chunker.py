"""Deterministic text chunking and normalization for InsightBot AI RAG.

Provides reproducible text chunking with overlap, page tracking,
whitespace normalization, and content hashing.
"""

import hashlib
import re
from typing import Any, Optional

from insightbot.db.models import DocumentChunk
from insightbot.knowledge.parsers import ParsedDocument


def normalize_text(text: str) -> str:
	"""Normalize text whitespace without destroying meaningful paragraph boundaries.

	- Converts CRLF/CR to LF
	- Trims trailing spaces on each line
	- Collapses 3+ consecutive newlines to double newline
	- Collapses horizontal whitespace (spaces/tabs) to a single space
	"""
	if not text:
		return ''

	# Normalize line breaks
	text = text.replace('\r\n', '\n').replace('\r', '\n')

	# Split lines, collapse horizontal spaces per line, trim ends
	normalized_lines = []
	for line in text.split('\n'):
		line = re.sub(r'[ \t]+', ' ', line).strip()
		normalized_lines.append(line)

	rejoined = '\n'.join(normalized_lines)
	# Collapse 3 or more newlines into 2 (paragraph break)
	rejoined = re.sub(r'\n{3,}', '\n\n', rejoined)
	return rejoined.strip()


def compute_content_hash(content: bytes | str) -> str:
	"""Compute SHA-256 hash of document content for deduplication."""
	if isinstance(content, str):
		data = content.encode('utf-8')
	else:
		data = content
	return hashlib.sha256(data).hexdigest()


class DeterministicChunker:
	"""Splits documents into deterministic overlapping chunks with page tracking."""

	def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
		if chunk_overlap >= chunk_size:
			raise ValueError(f'chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})')
		if chunk_size <= 0:
			raise ValueError('chunk_size must be positive')
		if chunk_overlap < 0:
			raise ValueError('chunk_overlap cannot be negative')

		self.chunk_size = chunk_size
		self.chunk_overlap = chunk_overlap

	def chunk_document(
		self,
		document_id: str,
		parsed_doc: ParsedDocument,
		document_metadata: Optional[dict[str, Any]] = None,
		organization_id: Optional[str] = None,
		workspace_id: Optional[str] = None,
	) -> list[DocumentChunk]:
		"""Chunk a parsed document into DocumentChunk objects with deterministic indexing.

		If pages are present, chunks are associated with their corresponding page number.
		Preserves document metadata, filename, doc_type, and page number.
		"""
		chunks: list[DocumentChunk] = []
		chunk_index = 0
		doc_meta = dict(document_metadata or {})
		inherited_doc_type = doc_meta.get('doc_type', parsed_doc.metadata.get('doc_type', parsed_doc.source_type))

		# If we have multiple pages with text, chunk page by page to preserve exact page numbers
		if parsed_doc.pages and len(parsed_doc.pages) > 1:
			for page in parsed_doc.pages:
				norm_page_text = normalize_text(page.text)
				if not norm_page_text:
					continue

				page_chunks = self._split_text(norm_page_text)
				for text_segment, start_char, end_char in page_chunks:
					if not text_segment.strip():
						continue
					merged_chunk_meta = {
						'source_filename': parsed_doc.filename,
						'source_type': parsed_doc.source_type,
						'doc_type': inherited_doc_type,
						'page_number': page.page_number,
						'start_char': start_char,
						'end_char': end_char,
						**parsed_doc.metadata,
						**doc_meta,
					}
					chk = DocumentChunk(
						document_id=document_id,
						chunk_index=chunk_index,
						content=text_segment,
						metadata=merged_chunk_meta,
						organization_id=organization_id,
						workspace_id=workspace_id,
					)
					chunks.append(chk)
					chunk_index += 1
		else:
			# Single-page, plain text, or markdown
			norm_full_text = normalize_text(parsed_doc.text)
			if norm_full_text:
				text_chunks = self._split_text(norm_full_text)
				for text_segment, start_char, end_char in text_chunks:
					if not text_segment.strip():
						continue
					merged_chunk_meta = {
						'source_filename': parsed_doc.filename,
						'source_type': parsed_doc.source_type,
						'doc_type': inherited_doc_type,
						'page_number': 1,
						'start_char': start_char,
						'end_char': end_char,
						**parsed_doc.metadata,
						**doc_meta,
					}
					chk = DocumentChunk(
						document_id=document_id,
						chunk_index=chunk_index,
						content=text_segment,
						metadata=merged_chunk_meta,
						organization_id=organization_id,
						workspace_id=workspace_id,
					)
					chunks.append(chk)
					chunk_index += 1

		return chunks

	def _split_text(self, text: str) -> list[tuple[str, int, int]]:
		"""Split text into (segment, start_char, end_char) with step = chunk_size - chunk_overlap.

		Prefers splitting on paragraph or sentence boundaries when possible.
		"""
		step = self.chunk_size - self.chunk_overlap
		total_len = len(text)
		segments: list[tuple[str, int, int]] = []

		if total_len <= self.chunk_size:
			return [(text, 0, total_len)]

		start = 0
		while start < total_len:
			end = min(start + self.chunk_size, total_len)

			# If we haven't reached the end, try to find a natural boundary near end
			if end < total_len:
				# Look for paragraph break
				last_break = text.rfind('\n\n', start + self.chunk_overlap, end)
				if last_break != -1:
					end = last_break + 2
				else:
					# Look for sentence break
					last_period = max(
						text.rfind('. ', start + self.chunk_overlap, end),
						text.rfind('.\n', start + self.chunk_overlap, end),
					)
					if last_period != -1:
						end = last_period + 1
					else:
						# Look for space
						last_space = text.rfind(' ', start + self.chunk_overlap, end)
						if last_space != -1:
							end = last_space + 1

			chunk_content = text[start:end].strip()
			if chunk_content:
				segments.append((chunk_content, start, end))

			if end >= total_len:
				break

			# Advance start pointer
			next_start = end - self.chunk_overlap
			if next_start <= start:
				next_start = start + step
			start = next_start

		return segments


__all__ = [
	'normalize_text',
	'compute_content_hash',
	'DeterministicChunker',
]
