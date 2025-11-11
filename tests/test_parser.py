"""Unit tests for PDFParser using a real PDF from test_pdfs/."""

from pathlib import Path

import pytest

from core.parser import PDFParser


@pytest.fixture
def sample_pdf_path():
    # Use a real PDF from test_pdfs/agent_security/ (replace with an actual file name)
    pdf_dir = Path(__file__).parent.parent / "test_pdfs" / "agent_security"
    # Find the first PDF in the directory
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        pytest.skip("No sample PDF found in test_pdfs/agent_security/")
    return pdf_files[0]


def test_extract_text(sample_pdf_path):
    parser = PDFParser()
    text, text_hash = parser.extract_text(sample_pdf_path)
    assert isinstance(text, str)
    assert len(text) > 0
    assert isinstance(text_hash, str)
    assert len(text_hash) > 0


def test_extract_sections(sample_pdf_path):
    parser = PDFParser()
    text, _ = parser.extract_text(sample_pdf_path)
    sections = parser.extract_sections(text)
    assert isinstance(sections, dict)
    assert "abstract" in sections
    assert "introduction" in sections


def test_get_page_count(sample_pdf_path):
    parser = PDFParser()
    page_count = parser.get_page_count(sample_pdf_path)
    assert isinstance(page_count, int)
    assert page_count > 0
