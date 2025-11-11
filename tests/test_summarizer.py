"""Unit tests for core/summarizer.py: summary generation logic (mocked)."""

from unittest.mock import MagicMock

import pytest

from core.summarizer import Summarizer


@pytest.fixture
def summarizer():
    return Summarizer(model="mock-model", temperature=0.1)


def test_summarize_paper_returns_string(summarizer):
    summarizer._llm_summarize = MagicMock(return_value="This is a summary.")
    summary = summarizer.summarize_paper(
        title="Test Title",
        abstract="Test abstract",
        intro="Test intro",
        topic="Test topic",
        metadata={"title": "Test Title"},
        full_text="Full text here.",
    )
    assert isinstance(summary, str)
    # Check that the summary contains the title and abstract (formatted output)
    assert "Test Title" in summary
    assert "Test abstract" in summary
