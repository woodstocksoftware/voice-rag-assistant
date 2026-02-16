"""
Tests for src.rag.knowledge_base.KnowledgeBase.

ChromaDB is used with a real (temp-directory) persistent client so we get
real vector operations. The Anthropic Claude API is mocked.
"""

from unittest.mock import MagicMock, patch

import chromadb
from chromadb.utils import embedding_functions
import pytest

from tests.conftest import SAMPLE_DOCUMENTS, SAMPLE_METADATAS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_kb(chroma_tmp_dir, mock_anthropic_client, collection_name="test_kb"):
    """
    Build a KnowledgeBase that uses a temp ChromaDB directory and a mocked
    Claude client, without triggering the real Anthropic() constructor.
    """
    # Create real ChromaDB client BEFORE patching so we get a real client, not a mock
    real_client = chromadb.PersistentClient(path=chroma_tmp_dir)

    with patch("src.rag.knowledge_base.chromadb.PersistentClient", return_value=real_client), \
         patch("src.rag.knowledge_base.Anthropic", return_value=mock_anthropic_client):
        from src.rag.knowledge_base import KnowledgeBase
        kb = KnowledgeBase(collection_name=collection_name)

    # After construction, kb.chroma_client, kb.collection, kb.claude are all
    # wired up (collection is real ChromaDB, claude is mock).
    return kb


def _make_claude_response(text: str):
    """Build a minimal mock response matching Anthropic's shape."""
    block = MagicMock()
    block.text = text
    resp = MagicMock()
    resp.content = [block]
    return resp


# ---------------------------------------------------------------------------
# Tests — initialisation
# ---------------------------------------------------------------------------

class TestKnowledgeBaseInit:
    def test_empty_collection_has_zero_count(self, chroma_tmp_dir, mock_anthropic):
        kb = _make_kb(chroma_tmp_dir, mock_anthropic, "init_test")
        assert kb.count() == 0

    def test_collection_name_matches(self, chroma_tmp_dir, mock_anthropic):
        kb = _make_kb(chroma_tmp_dir, mock_anthropic, "custom_name")
        assert kb.collection.name == "custom_name"


# ---------------------------------------------------------------------------
# Tests — add_documents
# ---------------------------------------------------------------------------

class TestAddDocuments:
    def test_add_single_document(self, chroma_tmp_dir, mock_anthropic):
        kb = _make_kb(chroma_tmp_dir, mock_anthropic)
        kb.add_documents(["Test document about pool hours."])
        assert kb.count() == 1

    def test_add_multiple_documents(self, chroma_tmp_dir, mock_anthropic):
        kb = _make_kb(chroma_tmp_dir, mock_anthropic)
        kb.add_documents(SAMPLE_DOCUMENTS, SAMPLE_METADATAS)
        assert kb.count() == len(SAMPLE_DOCUMENTS)

    def test_add_documents_generates_default_metadatas(self, chroma_tmp_dir, mock_anthropic):
        kb = _make_kb(chroma_tmp_dir, mock_anthropic)
        kb.add_documents(["Doc one", "Doc two"])
        # Retrieve to check metadata
        results = kb.collection.get(ids=["doc_0", "doc_1"])
        assert results["metadatas"][0]["source"] == "doc_0"
        assert results["metadatas"][1]["source"] == "doc_1"

    def test_add_documents_with_custom_metadatas(self, chroma_tmp_dir, mock_anthropic):
        kb = _make_kb(chroma_tmp_dir, mock_anthropic)
        kb.add_documents(
            ["Custom doc"],
            metadatas=[{"source": "custom-source"}],
        )
        results = kb.collection.get(ids=["doc_0"])
        assert results["metadatas"][0]["source"] == "custom-source"

    def test_add_documents_increments_ids(self, chroma_tmp_dir, mock_anthropic):
        kb = _make_kb(chroma_tmp_dir, mock_anthropic)
        kb.add_documents(["First batch"])
        kb.add_documents(["Second batch"])
        assert kb.count() == 2
        # Second doc should have id "doc_1" because count was 1 when added
        results = kb.collection.get(ids=["doc_1"])
        assert results["documents"][0] == "Second batch"


# ---------------------------------------------------------------------------
# Tests — count
# ---------------------------------------------------------------------------

class TestCount:
    def test_count_empty(self, chroma_tmp_dir, mock_anthropic):
        kb = _make_kb(chroma_tmp_dir, mock_anthropic)
        assert kb.count() == 0

    def test_count_after_adds(self, chroma_tmp_dir, mock_anthropic):
        kb = _make_kb(chroma_tmp_dir, mock_anthropic)
        kb.add_documents(SAMPLE_DOCUMENTS, SAMPLE_METADATAS)
        assert kb.count() == 5


# ---------------------------------------------------------------------------
# Tests — query (Claude mocked)
# ---------------------------------------------------------------------------

class TestQuery:
    def test_query_returns_answer_and_sources(self, chroma_tmp_dir, mock_anthropic):
        kb = _make_kb(chroma_tmp_dir, mock_anthropic)
        kb.add_documents(SAMPLE_DOCUMENTS, SAMPLE_METADATAS)

        mock_anthropic.messages.create.return_value = _make_claude_response(
            "Check-in is at 3 PM."
        )

        result = kb.query("What time is check-in?")

        assert "answer" in result
        assert "sources" in result
        assert result["answer"] == "Check-in is at 3 PM."
        assert len(result["sources"]) > 0

    def test_query_calls_claude_with_context(self, chroma_tmp_dir, mock_anthropic):
        kb = _make_kb(chroma_tmp_dir, mock_anthropic)
        kb.add_documents(SAMPLE_DOCUMENTS, SAMPLE_METADATAS)

        mock_anthropic.messages.create.return_value = _make_claude_response("Answer.")

        kb.query("What time is check-in?")

        # Verify Claude was called
        mock_anthropic.messages.create.assert_called_once()
        call_kwargs = mock_anthropic.messages.create.call_args
        # Should include model, max_tokens, system, messages
        assert call_kwargs.kwargs["model"] == "claude-sonnet-4-20250514"
        assert call_kwargs.kwargs["max_tokens"] == 500
        assert "system" in call_kwargs.kwargs
        # The user message should contain "check-in"
        user_msg = call_kwargs.kwargs["messages"][0]["content"]
        assert "check-in" in user_msg.lower()

    def test_query_returns_relevant_sources(self, chroma_tmp_dir, mock_anthropic):
        kb = _make_kb(chroma_tmp_dir, mock_anthropic)
        kb.add_documents(SAMPLE_DOCUMENTS, SAMPLE_METADATAS)

        mock_anthropic.messages.create.return_value = _make_claude_response(
            "The pool is on the 5th floor."
        )

        result = kb.query("Where is the swimming pool?")

        # "pool info" should be one of the top sources
        assert "pool info" in result["sources"]

    def test_query_respects_n_results(self, chroma_tmp_dir, mock_anthropic):
        kb = _make_kb(chroma_tmp_dir, mock_anthropic)
        kb.add_documents(SAMPLE_DOCUMENTS, SAMPLE_METADATAS)

        mock_anthropic.messages.create.return_value = _make_claude_response("Answer.")

        result = kb.query("pool", n_results=2)
        assert len(result["sources"]) <= 2

    def test_query_empty_collection_returns_fallback(self, chroma_tmp_dir, mock_anthropic):
        """When there are no documents, the KB should return a fallback message
        without calling Claude."""
        kb = _make_kb(chroma_tmp_dir, mock_anthropic)

        result = kb.query("anything")

        assert "don't have any information" in result["answer"].lower()
        assert result["sources"] == []
        # Claude should NOT have been called
        mock_anthropic.messages.create.assert_not_called()

    def test_query_with_single_document(self, chroma_tmp_dir, mock_anthropic):
        kb = _make_kb(chroma_tmp_dir, mock_anthropic)
        kb.add_documents(["The restaurant is open from 7 AM to 11 PM."])

        mock_anthropic.messages.create.return_value = _make_claude_response(
            "The restaurant opens at 7 AM."
        )

        result = kb.query("When does the restaurant open?", n_results=3)

        assert result["answer"] == "The restaurant opens at 7 AM."
        # Only 1 source available even though we asked for 3
        assert len(result["sources"]) == 1


# ---------------------------------------------------------------------------
# Tests — seed data pattern (matches __main__ block)
# ---------------------------------------------------------------------------

class TestSeedData:
    def test_sample_data_can_be_added_and_queried(self, chroma_tmp_dir, mock_anthropic):
        """Reproduce the __main__ block: seed 5 docs then query."""
        kb = _make_kb(chroma_tmp_dir, mock_anthropic)
        kb.add_documents(SAMPLE_DOCUMENTS, SAMPLE_METADATAS)
        assert kb.count() == 5

        mock_anthropic.messages.create.return_value = _make_claude_response(
            "Check-in is at 3 PM and check-out is at 11 AM."
        )

        result = kb.query("What time can I check in?")
        assert "3 PM" in result["answer"]
