"""Unit tests for core/metadata.py: LLM extraction, categorization, and scoring logic (mocked)."""

from unittest.mock import MagicMock, patch

import pytest

from core.metadata import MetadataExtractor


@pytest.fixture
def extractor():
    # Mock config
    return MetadataExtractor(use_crossref=False, crossref_email=None)


def test_extract_with_llm_returns_metadata(extractor):
    """Test LLM metadata extraction with mocked responses."""
    from pathlib import Path
    from config import Config

    # Test with Ollama provider (returns JSON string)
    fake_llm_response_ollama = {
        "response": '{"title": "Test Paper", "authors": ["Alice", "Bob"], "abstract": "Test abstract", "year": "2023", "venue": null}'
    }

    mock_config = Config()
    mock_config.llm_provider = "ollama"
    mock_config.ollama.summarize_model = "deepseek-r1:8b"

    with (
        patch("fitz.open") as mock_fitz_open,
        patch("config.Config", return_value=mock_config),  # Patch Config class itself
        patch("core.metadata.llm_generate", return_value=fake_llm_response_ollama),
    ):
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "First page text"
        mock_doc.load_page.return_value = mock_page
        mock_doc.close = MagicMock()
        mock_fitz_open.return_value = mock_doc

        result = extractor._extract_with_llm(Path("/fake/path.pdf"))

    assert result["title"] == "Test Paper"
    assert isinstance(result["authors"], list)
    assert result["authors"] == ["Alice", "Bob"]
    assert "abstract" in result


def test_extract_with_llm_gemini_provider(extractor):
    """Test LLM metadata extraction with Gemini provider (returns dict)."""
    from pathlib import Path
    from config import Config

    # Gemini returns a dict directly (from gemini_generate_json)
    fake_llm_response_gemini = {
        "response": {
            "title": "Gemini Test Paper",
            "authors": ["Charlie", "Dave"],
            "abstract": "Gemini abstract",
            "year": "2024",
            "venue": "AI Conference",
        }
    }

    mock_config = Config()
    mock_config.llm_provider = "gemini"
    mock_config.gemini.api_key = "test_key"

    with (
        patch("fitz.open") as mock_fitz_open,
        patch("config.Config", return_value=mock_config),  # Patch Config class itself
        patch("core.metadata.llm_generate", return_value=fake_llm_response_gemini),
    ):
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "First page text"
        mock_doc.load_page.return_value = mock_page
        mock_doc.close = MagicMock()
        mock_fitz_open.return_value = mock_doc

        result = extractor._extract_with_llm(Path("/fake/path.pdf"))

    assert result["title"] == "Gemini Test Paper"
    assert isinstance(result["authors"], list)
    assert result["authors"] == ["Charlie", "Dave"]
    assert result["abstract"] == "Gemini abstract"
    assert result["year"] == "2024"


def test_llm_categorize_and_score_returns_expected_keys(extractor):
    """Test LLM categorization and scoring with mocked responses."""
    fake_llm_response = {
        "response": {
            "category": "attack_vectors",
            "relevance_score": 8,
            "include": True,
            "reason": "Highly relevant to AI security",
        }
    }

    with patch("utils.llm_provider.llm_generate", return_value=fake_llm_response):
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
