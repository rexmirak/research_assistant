"""Unit tests for hash utilities."""

import tempfile
from pathlib import Path

import pytest

from utils.hash import file_hash, stable_paper_id, text_hash


def test_file_hash():
    """Test file hashing."""
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("test content")
        temp_path = Path(f.name)

    try:
        hash1 = file_hash(temp_path)
        hash2 = file_hash(temp_path)

        # Same file should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 length
    finally:
        temp_path.unlink()


def test_text_hash():
    """Test text hashing."""
    text1 = "This is a test"
    text2 = "This is a test"
    text3 = "This is different"

    hash1 = text_hash(text1)
    hash2 = text_hash(text2)
    hash3 = text_hash(text3)

    assert hash1 == hash2  # Same text
    assert hash1 != hash3  # Different text
    assert len(hash1) == 64  # SHA256


def test_text_hash_normalization():
    """Test text hash normalizes whitespace."""
    text1 = "Hello   World"
    text2 = "Hello World"
    text3 = "hello world"  # Different case

    hash1 = text_hash(text1)
    hash2 = text_hash(text2)
    hash3 = text_hash(text3)

    assert hash1 == hash2  # Whitespace normalized
    assert hash1 == hash3  # Case normalized


def test_stable_paper_id():
    """Test stable paper ID generation."""
    title = "Machine Learning for Healthcare"
    authors = ["John Doe", "Jane Smith"]
    year = "2023"

    id1 = stable_paper_id(title, authors, year)
    id2 = stable_paper_id(title, authors, year)

    assert id1 == id2  # Same metadata produces same ID
    assert len(id1) == 12  # First 12 chars of hash

    # Different metadata produces different ID
    id3 = stable_paper_id(title, authors, "2024")
    assert id1 != id3


def test_stable_paper_id_author_order():
    """Test paper ID is stable regardless of author order."""
    title = "Test Paper"
    authors1 = ["Alice", "Bob"]
    authors2 = ["Bob", "Alice"]

    id1 = stable_paper_id(title, authors1, "2023")
    id2 = stable_paper_id(title, authors2, "2023")

    assert id1 == id2  # Author order shouldn't matter
