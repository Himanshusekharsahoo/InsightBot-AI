"""Unit tests for InsightBot CLI knowledge commands."""

import argparse

from insightbot.cli.knowledge import handle_ask, handle_ingest, handle_knowledge_cli, handle_list, handle_search
from insightbot.db.in_memory import InMemoryVectorStore
from insightbot.knowledge.embeddings import MockEmbeddingModel
from insightbot.knowledge.store import KnowledgeStore
from insightbot.orchestrator.knowledge_agent import KnowledgeAgent
from insightbot.orchestrator.llm import MockLLMProvider


def test_cli_ingest_and_list(tmp_path, monkeypatch):
	# Create sample docs directory
	docs_dir = tmp_path / 'sample_docs'
	docs_dir.mkdir()
	(docs_dir / 'sample1.txt').write_text('InsightBot knowledge document 1.', encoding='utf-8')
	(docs_dir / 'sample2.md').write_text('# Knowledge 2\nDetails on database.', encoding='utf-8')

	# In-memory mock store
	mem_store = InMemoryVectorStore()
	mock_embed = MockEmbeddingModel(dimension=32)
	ks = KnowledgeStore(storage_dir=tmp_path / 'legacy', vector_store=mem_store, embedding_provider=mock_embed)

	monkeypatch.setattr('insightbot.cli.knowledge.KnowledgeStore', lambda: ks)
	monkeypatch.setattr(
		'insightbot.cli.knowledge.KnowledgeAgent',
		lambda knowledge_store: KnowledgeAgent(knowledge_store=knowledge_store, llm_provider=MockLLMProvider()),
	)

	# 1. Test Ingest
	args_ingest = argparse.Namespace(path=str(docs_dir))
	code = handle_ingest(args_ingest)
	assert code == 0

	# 2. Test List
	args_list = argparse.Namespace()
	code_list = handle_list(args_list)
	assert code_list == 0

	# 3. Test Search
	args_search = argparse.Namespace(query='database', top_k=3)
	code_search = handle_search(args_search)
	assert code_search == 0

	# 4. Test Ask
	args_ask = argparse.Namespace(question='What database is used?', top_k=3)
	code_ask = handle_ask(args_ask)
	assert code_ask == 0


def test_cli_ingest_nonexistent_path():
	args = argparse.Namespace(path='nonexistent_folder_xyz_123')
	code = handle_ingest(args)
	assert code == 1


def test_cli_dispatch_unknown_subcommand():
	args = argparse.Namespace(knowledge_command='unknown_subcmd')
	code = handle_knowledge_cli(args)
	assert code == 1
