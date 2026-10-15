"""InsightBot AI Knowledge CLI Handlers.

Implements CLI commands:
- insightbot knowledge ingest <path>
- insightbot knowledge list
- insightbot knowledge search <query>
- insightbot knowledge ask <question>
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

from insightbot.knowledge.parsers import DocumentParserFactory
from insightbot.knowledge.store import KnowledgeStore
from insightbot.orchestrator.knowledge_agent import KnowledgeAgent

logger = logging.getLogger('insightbot.cli.knowledge')


def handle_knowledge_cli(args: Any) -> int:
	"""Dispatch knowledge CLI subcommands."""
	cmd = getattr(args, 'knowledge_command', None)

	if not cmd:
		print('Usage: insightbot knowledge {ingest|list|search|ask} ...')
		return 1

	if cmd == 'ingest':
		return handle_ingest(args)
	elif cmd == 'list':
		return handle_list(args)
	elif cmd == 'search':
		return handle_search(args)
	elif cmd == 'ask':
		return handle_ask(args)
	else:
		print(f'Unknown knowledge subcommand: {cmd}', file=sys.stderr)
		return 1


def handle_ingest(args: Any) -> int:
	"""Ingest individual file or directory recursively."""
	target_path = Path(args.path)
	if not target_path.exists():
		print(f"Error: Path '{args.path}' does not exist.", file=sys.stderr)
		return 1

	store = KnowledgeStore()
	files_to_ingest: list[Path] = []

	if target_path.is_file():
		files_to_ingest.append(target_path)
	elif target_path.is_dir():
		allowed_exts = set(DocumentParserFactory._PARSERS.keys())
		for p in sorted(target_path.rglob('*')):
			if p.is_file() and p.suffix.lower() in allowed_exts:
				files_to_ingest.append(p)
		if not files_to_ingest:
			print(f"No supported document files found in '{target_path}'.", file=sys.stderr)
			return 1
	else:
		print(f"Error: '{target_path}' is neither a file nor a directory.", file=sys.stderr)
		return 1

	print(f"Ingesting {len(files_to_ingest)} document(s) from '{target_path}'...")

	async def _run_ingest():
		success_count = 0
		fail_count = 0

		try:
			for file_path in files_to_ingest:
				try:
					content = file_path.read_bytes()
					doc = await store.ingest_document(
						filename=file_path.name,
						content=content,
						metadata={'source_path': str(file_path.resolve())},
					)
					print(f'  [OK] {file_path.name} (id={doc.id}, type={doc.source_type}, status={doc.processing_status})')
					success_count += 1
				except Exception as e:
					print(f'  [FAILED] {file_path.name}: {e}', file=sys.stderr)
					fail_count += 1

			print(f'\nIngestion summary: {success_count} succeeded, {fail_count} failed.')
			return 0 if fail_count == 0 and success_count > 0 else 1
		finally:
			await store.close()

	try:
		return asyncio.run(_run_ingest())
	except Exception as e:
		print(f'Error during ingestion: {e}', file=sys.stderr)
		return 1


def handle_list(args: Any) -> int:
	"""List all ingested documents in the KnowledgeStore."""
	store = KnowledgeStore()

	async def _run_list():
		try:
			docs = await store.list_documents(limit=100)
			if docs:
				print(f'[KnowledgeStore] ({len(docs)} documents):')
				for d in docs:
					print(f'  [{d.id}] {d.filename} ({d.source_type}) - status: {d.processing_status} [{d.created_at[:19]}]')
			else:
				# Fallback check legacy
				legacy_docs = store.get_all()
				if legacy_docs:
					print(f'[KnowledgeStore (Legacy)] ({len(legacy_docs)} documents):')
					for ld in legacy_docs:
						print(f'  [{ld.doc_id}] {ld.title} ({ld.source_url})')
				else:
					print('Knowledge store is empty. Ingest documents using: insightbot knowledge ingest <path>')
			return 0
		except Exception as e:
			print(f'Error accessing database: {e}', file=sys.stderr)
			return 1
		finally:
			await store.close()

	return asyncio.run(_run_list())


def handle_search(args: Any) -> int:
	"""Perform semantic search on stored document chunks."""
	query = getattr(args, 'query', '')
	top_k = getattr(args, 'top_k', 5)
	if not query or not query.strip():
		print('Error: Search query cannot be empty.', file=sys.stderr)
		return 1

	store = KnowledgeStore()

	async def _run_search():
		try:
			results = await store.search_async(query=query, top_k=top_k)
			if not results:
				print(f"No semantic matches found for query: '{query}'")
				return 0

			print(f"[Semantic Matches] for '{query}' ({len(results)} found):")
			for r in results:
				fname = r.metadata.get('source_filename', 'unknown')
				page = f', Page {r.metadata.get("page_number")}' if r.metadata.get('page_number') else ''
				score_str = f'score: {r.score:.3f}'
				print(f'  * [{score_str}] {fname} (doc={r.document_id}, chunk={r.chunk_id}{page})')
				snippet = r.content[:160].replace('\n', ' ')
				print(f'    "{snippet}..."')
			return 0
		except Exception as e:
			print(f'Search failed: {e}', file=sys.stderr)
			return 1
		finally:
			await store.close()

	return asyncio.run(_run_search())


def handle_ask(args: Any) -> int:
	"""Answer a question grounded in internal knowledge."""
	question = getattr(args, 'question', '')
	top_k = getattr(args, 'top_k', 5)
	if not question or not question.strip():
		print('Error: Question cannot be empty.', file=sys.stderr)
		return 1

	store = KnowledgeStore()
	agent = KnowledgeAgent(knowledge_store=store)

	async def _run_ask():
		try:
			answer_obj = await agent.answer(question=question, top_k=top_k)
			print(f'\n[Question]: {question}\n')
			print(f'[Answer]:\n{answer_obj.answer}\n')

			if answer_obj.sources:
				print('[Citations / Evidence Sources]:')
				for idx, s in enumerate(answer_obj.sources, start=1):
					page_str = f', page {s.page_number}' if s.page_number else ''
					print(
						f'  [{idx}] {s.source_filename} (doc_id={s.document_id}, chunk_id={s.chunk_id}{page_str}, score={s.score:.2f})'
					)
					print(f'      Snippet: {s.snippet}')
			else:
				print('[Citations]: No supporting sources found.')

			if answer_obj.insufficient_evidence:
				print('\n[Notice]: Insufficient internal evidence was found to fully answer this question.')

			return 0
		except Exception as e:
			print(f'Question-answering failed: {e}', file=sys.stderr)
			return 1
		finally:
			await store.close()

	return asyncio.run(_run_ask())
