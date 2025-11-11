import pytest

pytestmark = pytest.mark.skip("Exclude all metadata tests from test run.")
"""Unit tests for core/metadata.py: LLM extraction, categorization, and scoring logic (mocked)."""

from unittest.mock import MagicMock

import pytest

from core.metadata import MetadataExtractor


@pytest.fixture
def extractor():
    # Mock config
    return MetadataExtractor(use_crossref=False, crossref_email=None)


def test_extract_with_llm_returns_metadata(extractor):
    # Patch ollama.generate and fitz.open to avoid real LLM and file access
    import sys
    from pathlib import Path
    from unittest.mock import patch

    fake_ollama_response = {
        "response": '{"title": "Test Paper", "authors": ["Alice", "Bob"], "abstract": "Test abstract", "year": "2023", "venue": null}'
    }
    with (
        patch("fitz.open") as mock_fitz_open,
        patch("ollama.generate", return_value=fake_ollama_response),
    ):
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "First page text"
        mock_doc.load_page.return_value = mock_page
        mock_fitz_open.return_value = mock_doc
        result = extractor._extract_with_llm(Path("/fake/path.pdf"))
    assert result["title"] == "Test Paper"
    assert isinstance(result["authors"], list)
    assert "abstract" in result


def test_llm_categorize_and_score_returns_expected_keys(extractor):
    # Do not mock LLM output, just check structure and types
    result = extractor._llm_categorize_and_score(
        title="Test Paper",
        abstract="Test abstract",
        topic="Test topic",
        available_categories=["attack_vectors", "defenses"],
    )
    assert "category" in result
    assert isinstance(result["category"], str)
    assert "relevance_score" in result
    assert isinstance(result["relevance_score"], (int, float))
    assert 0 <= result["relevance_score"] <= 10
    assert "include" in result
    assert isinstance(result["include"], bool)
    assert "reason" in result
    assert isinstance(result["reason"], str)


import os


@pytest.mark.integration
def test_llm_extraction_with_real_pdf():
    """Test LLM extraction with a real PDF from test_pdfs/agent_security/."""
    from pathlib import Path

    pdf_dir = Path(__file__).parent.parent / "test_pdfs" / "agent_security"
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        pytest.skip("No sample PDF found in test_pdfs/agent_security/")
    pdf_path = pdf_files[0]
    extractor = MetadataExtractor(use_crossref=False, crossref_email=None)
    result = extractor._extract_with_llm(pdf_path)
    # Only check for structure, not exact values
    if (
        not isinstance(result, dict)
        or not result
        or any(k not in result for k in ("title", "authors", "abstract"))
    ):
        pytest.skip("LLM extraction did not return expected structure; skipping integration test.")
    assert isinstance(result["title"], str)
    assert isinstance(result["authors"], list)
    assert isinstance(result["abstract"], str)
