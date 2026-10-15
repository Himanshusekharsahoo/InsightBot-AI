"""Unit tests for document parsers (PDF, DOCX, TXT, Markdown)."""

import io

import docx
import pytest

from insightbot.knowledge.parsers import (
	DocumentParserFactory,
	DocumentParsingError,
	DocxParser,
	MarkdownParser,
	PDFParser,
	TextParser,
	UnsupportedDocumentError,
)


def _create_sample_pdf(pages_text: list[str]) -> bytes:
	"""Helper to create an in-memory PDF."""
	from reportlab.pdfgen import canvas

	buffer = io.BytesIO()
	canv = canvas.Canvas(buffer)
	for text in pages_text:
		canv.drawString(100, 750, text)
		canv.showPage()
	canv.save()
	return buffer.getvalue()


def _create_sample_docx(paragraphs: list[str]) -> bytes:
	"""Helper to create an in-memory DOCX."""
	doc = docx.Document()
	for p in paragraphs:
		doc.add_paragraph(p)
	buffer = io.BytesIO()
	doc.save(buffer)
	return buffer.getvalue()


def test_txt_parser():
	parser = TextParser()
	content = 'InsightBot AI is a private knowledge platform.\n\nSecond paragraph here.'.encode('utf-8')
	doc = parser.parse('guide.txt', content)
	assert doc.source_type == 'txt'
	assert doc.mime_type == 'text/plain'
	assert 'InsightBot AI' in doc.text
	assert len(doc.pages) == 1
	assert doc.pages[0].page_number == 1


def test_txt_parser_empty():
	parser = TextParser()
	doc = parser.parse('empty.txt', b'')
	assert doc.text == ''
	assert doc.metadata.get('empty') is True


def test_markdown_parser():
	parser = MarkdownParser()
	content = '# Heading 1\n\nInsightBot RAG supports Markdown.\n\n- Point 1\n- Point 2'.encode('utf-8')
	doc = parser.parse('doc.md', content)
	assert doc.source_type == 'markdown'
	assert doc.mime_type == 'text/markdown'
	assert 'Heading 1' in doc.text
	assert 'Point 1' in doc.text


def test_pdf_parser():
	pdf_bytes = _create_sample_pdf(['Page 1 content about architecture.', 'Page 2 content about PostgreSQL.'])
	parser = PDFParser()
	doc = parser.parse('architecture.pdf', pdf_bytes)
	assert doc.source_type == 'pdf'
	assert doc.mime_type == 'application/pdf'
	assert len(doc.pages) == 2
	assert doc.pages[0].page_number == 1
	assert 'Page 1' in doc.pages[0].text
	assert doc.pages[1].page_number == 2
	assert 'Page 2' in doc.pages[1].text


def test_pdf_parser_empty_and_corrupt():
	parser = PDFParser()
	# Empty bytes
	empty_doc = parser.parse('empty.pdf', b'')
	assert empty_doc.text == ''
	assert empty_doc.metadata.get('empty') is True

	# Corrupt bytes
	with pytest.raises(DocumentParsingError):
		parser.parse('corrupt.pdf', b'NOT A VALID PDF FILE HEADER')


def test_docx_parser():
	docx_bytes = _create_sample_docx(['First paragraph of specifications.', 'Second paragraph with details.'])
	parser = DocxParser()
	doc = parser.parse('specs.docx', docx_bytes)
	assert doc.source_type == 'docx'
	assert doc.mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
	assert 'First paragraph' in doc.text
	assert 'Second paragraph' in doc.text


def test_docx_parser_empty_and_corrupt():
	parser = DocxParser()
	empty_doc = parser.parse('empty.docx', b'')
	assert empty_doc.text == ''

	with pytest.raises(DocumentParsingError):
		parser.parse('corrupt.docx', b'NOT A DOCX FILE')


def test_parser_factory_supported_and_unsupported():
	assert isinstance(DocumentParserFactory.get_parser('file.pdf'), PDFParser)
	assert isinstance(DocumentParserFactory.get_parser('file.docx'), DocxParser)
	assert isinstance(DocumentParserFactory.get_parser('file.txt'), TextParser)
	assert isinstance(DocumentParserFactory.get_parser('file.md'), MarkdownParser)
	assert isinstance(DocumentParserFactory.get_parser('file.markdown'), MarkdownParser)

	with pytest.raises(UnsupportedDocumentError):
		DocumentParserFactory.get_parser('file.exe')

	with pytest.raises(UnsupportedDocumentError):
		DocumentParserFactory.get_parser('image.png')
