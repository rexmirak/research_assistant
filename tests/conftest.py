"""Test utilities and helpers."""

import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp = Path(tempfile.mkdtemp())
    yield temp
    shutil.rmtree(temp)


@pytest.fixture
def sample_pdf_dir(temp_dir):
    """Create sample directory structure with PDFs."""
    # Create category directories
    cat_a = temp_dir / "CategoryA"
    cat_b = temp_dir / "CategoryB"
    cat_a.mkdir()
    cat_b.mkdir()

    # Note: Actual PDF files would be needed for full integration tests
    # This is a placeholder structure

    return temp_dir
