"""Unit tests for text processing utilities."""

import pytest
from utils.text import (
    normalize_text,
    extract_abstract,
    extract_introduction,
    clean_title,
    truncate_text,
    create_bibtex_key,
)


def test_normalize_text():
    """Test text normalization."""
    text = "This   has   extra    spaces\n\nand newlines"
    normalized = normalize_text(text)

    assert "  " not in normalized  # No double spaces
    assert "\n\n" not in normalized


def test_extract_abstract():
    """Test abstract extraction."""
    text = """
    Title of Paper
    
    Abstract
    This is the abstract of the paper. It contains important information
    about the research.
    
    Introduction
    This is the introduction section.
    """

    abstract = extract_abstract(text)
    assert abstract is not None
    assert "abstract" in abstract.lower()
    assert "introduction" not in abstract.lower()


def test_extract_abstract_none():
    """Test abstract extraction returns None for invalid text."""
    text = "Short text"
    abstract = extract_abstract(text)
    assert abstract is None


def test_extract_introduction():
    """Test introduction extraction."""
    text = """
    Abstract
    This paper presents a new method.
    
    Introduction
    This is the introduction section with lots of content and sufficient length to meet the minimum requirements.
    It discusses the background and motivation in great detail with enough words to pass the length check.
    We provide comprehensive context about the problem domain and explain why this research is important.
    
    Related Work
    Previous studies have shown...
    """

    intro = extract_introduction(text)
    assert intro is not None
    assert "introduction" in intro.lower() or "background" in intro.lower()


def test_clean_title():
    """Test title cleaning."""
    title1 = "Machine Learning for\nHealthcare."
    title2 = "Deep   Learning   Models"

    clean1 = clean_title(title1)
    clean2 = clean_title(title2)

    assert "\n" not in clean1
    assert not clean1.endswith(".")
    assert "  " not in clean2


def test_truncate_text():
    """Test text truncation."""
    text = "This is a very long text " * 100

    truncated = truncate_text(text, max_chars=50)
    assert len(truncated) <= 53  # 50 + "..."
    assert truncated.endswith("...")


def test_truncate_text_no_truncation():
    """Test truncation doesn't truncate short text."""
    text = "Short text"
    truncated = truncate_text(text, max_chars=100)
    assert truncated == text
    assert not truncated.endswith("...")


def test_create_bibtex_key():
    """Test BibTeX key creation."""
    authors = ["John Smith", "Jane Doe"]
    year = "2023"
    title = "Machine Learning for Healthcare Applications"

    key = create_bibtex_key(authors, year, title)

    assert "smith" in key.lower()
    assert "2023" in key
    assert len(key) > 0
    assert " " not in key  # No spaces


def test_create_bibtex_key_no_authors():
    """Test BibTeX key with no authors."""
    authors = []
    year = "2023"
    title = "Some Paper Title"

    key = create_bibtex_key(authors, year, title)

    assert "unknown" in key.lower()
    assert "2023" in key


def test_create_bibtex_key_short_title():
    """Test BibTeX key with short title words."""
    authors = ["Smith"]
    year = "2023"
    title = "On AI"  # All short words

    key = create_bibtex_key(authors, year, title)

    assert "smith" in key.lower()
    assert "2023" in key
    # Should still create a valid key
    assert len(key) > 0
