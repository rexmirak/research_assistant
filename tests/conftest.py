"""Test utilities and helpers."""

import shutil
import sys
import tempfile
from pathlib import Path

import pytest

# Add project root to Python path so we can import cli, config, etc.
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


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
